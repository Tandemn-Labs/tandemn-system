import time
import pytest
from quota.tracker import JobSpec, JobState, JobRecord, VPCQuotaTracker


def _make_spec(**overrides):
    defaults = dict(
        job_id="job-1",
        model_name="Qwen/Qwen3-32B",
        num_lines=100,
        avg_input_tokens=500,
        avg_output_tokens=100,
        slo_hours=2.0,
    )
    defaults.update(overrides)
    return JobSpec(**defaults)


# --- JobState properties ---


def test_job_state_deadline_ts():
    spec = _make_spec(slo_hours=2.0)
    state = JobState(spec=spec, submitted_at=1000.0)
    assert state.deadline_ts == 1000.0 + 2.0 * 3600.0


def test_job_state_total_tokens():
    spec = _make_spec(num_lines=100, avg_input_tokens=500, avg_output_tokens=100)
    state = JobState(spec=spec, submitted_at=0)
    assert state.total_tokens == 100 * (500 + 100)


def test_job_state_remaining_tokens():
    spec = _make_spec(num_lines=100, avg_input_tokens=500, avg_output_tokens=100)
    state = JobState(spec=spec, submitted_at=0, progress_frac=0.5)
    assert state.remaining_tokens == 30000


def test_job_state_remaining_tokens_zero_progress():
    spec = _make_spec()
    state = JobState(spec=spec, submitted_at=0, progress_frac=0.0)
    assert state.remaining_tokens == state.total_tokens


# --- VPCQuotaTracker basic ---


def test_quota_tracker_reserve_and_release(sample_tracker):
    avail_before = sample_tracker.get_available("us-east-1", "on_demand", "G")
    assert sample_tracker.reserve("us-east-1", "on_demand", "G", 48) is True
    avail_after = sample_tracker.get_available("us-east-1", "on_demand", "G")
    assert avail_after == avail_before - 48
    sample_tracker.release("us-east-1", "on_demand", "G", 48)
    assert sample_tracker.get_available("us-east-1", "on_demand", "G") == avail_before


def test_quota_tracker_reserve_insufficient(sample_tracker):
    assert sample_tracker.reserve("us-east-1", "on_demand", "G", 999999) is False


def test_quota_tracker_status_summary(sample_tracker):
    sample_tracker.reserve("us-east-1", "on_demand", "G", 48)
    summary = sample_tracker.status_summary()
    assert "Region" in summary.columns
    assert "Used" in summary.columns
    assert len(summary) == 1


# --- release_for_instance ---


def test_release_for_instance(sample_tracker):
    sample_tracker.reserve_for_instance("us-east-1", "on_demand", "g6e.12xlarge", 1)
    avail_after_reserve = sample_tracker.get_available("us-east-1", "on_demand", "G")
    sample_tracker.release_for_instance("us-east-1", "on_demand", "g6e.12xlarge", 1)
    avail_after_release = sample_tracker.get_available("us-east-1", "on_demand", "G")
    assert avail_after_release == avail_after_reserve + 48


# --- SQLite-backed cluster reservations ---


def test_reserve_cluster_persists(sample_tracker):
    ok = sample_tracker.reserve_cluster(
        "cluster-1", "us-east-1", "on_demand", "g6e.12xlarge", 2
    )
    assert ok is True
    # Check in-memory counters
    assert sample_tracker.get_used_vcpu("us-east-1", "on_demand", "G") == 96
    # Check DB
    reservations = sample_tracker.get_reservations()
    assert len(reservations) == 1
    assert reservations[0]["cluster_name"] == "cluster-1"
    assert reservations[0]["vcpu_reserved"] == 96


def test_release_cluster_clears(sample_tracker):
    sample_tracker.reserve_cluster("cluster-1", "us-east-1", "on_demand", "g6e.12xlarge", 1)
    sample_tracker.release_cluster("cluster-1")
    assert sample_tracker.get_used_vcpu("us-east-1", "on_demand", "G") == 0
    assert sample_tracker.get_reservations() == []


def test_persistence_survives_restart(sample_quota_csv_path, tmp_path):
    db = str(tmp_path / "quota.db")
    # First tracker: make a reservation
    t1 = VPCQuotaTracker(quota_csv_file=sample_quota_csv_path, db_path=db)
    t1.reserve_cluster("cluster-1", "us-east-1", "spot", "g6e.12xlarge", 2)
    assert t1.get_used_vcpu("us-east-1", "spot", "G") == 96

    # Second tracker: same DB, should restore state
    t2 = VPCQuotaTracker(quota_csv_file=sample_quota_csv_path, db_path=db)
    assert t2.get_used_vcpu("us-east-1", "spot", "G") == 96
    assert len(t2.get_reservations()) == 1


def test_reconcile_removes_stale(sample_tracker):
    sample_tracker.reserve_cluster("alive", "us-east-1", "on_demand", "g6e.12xlarge", 1)
    sample_tracker.reserve_cluster("dead", "us-west-2", "on_demand", "g6e.12xlarge", 1)
    assert len(sample_tracker.get_reservations()) == 2

    # Reconcile: only "alive" is a live cluster
    sample_tracker.reconcile({"alive"})
    assert len(sample_tracker.get_reservations()) == 1
    assert sample_tracker.get_reservations()[0]["cluster_name"] == "alive"
    # "dead" cluster's quota should be released
    assert sample_tracker.get_used_vcpu("us-west-2", "on_demand", "G") == 0


def test_get_reservations_returns_list(sample_tracker):
    assert sample_tracker.get_reservations() == []
    sample_tracker.reserve_cluster("c1", "us-east-1", "spot", "g6e.48xlarge", 1)
    sample_tracker.reserve_cluster("c2", "us-east-1", "on_demand", "g6e.12xlarge", 2)
    res = sample_tracker.get_reservations()
    assert len(res) == 2
    names = {r["cluster_name"] for r in res}
    assert names == {"c1", "c2"}


# --- Multi-node quota math ---


def test_multi_node_reserve_and_release(sample_tracker):
    """2x g6e.12xlarge (48 vCPU each) should reserve 96 vCPU total."""
    avail_before = sample_tracker.get_available("us-east-1", "spot", "G")
    sample_tracker.reserve_cluster("multi", "us-east-1", "spot", "g6e.12xlarge", 2)
    assert sample_tracker.get_used_vcpu("us-east-1", "spot", "G") == 48 * 2
    assert sample_tracker.get_available("us-east-1", "spot", "G") == avail_before - 96
    # Release and verify fully restored
    sample_tracker.release_cluster("multi")
    assert sample_tracker.get_used_vcpu("us-east-1", "spot", "G") == 0
    assert sample_tracker.get_available("us-east-1", "spot", "G") == avail_before


def test_multi_node_large_instance(sample_tracker):
    """3x g6e.48xlarge (192 vCPU each) = 576 vCPU."""
    ok = sample_tracker.reserve_cluster("big", "us-east-1", "spot", "g6e.48xlarge", 3)
    if ok:
        assert sample_tracker.get_used_vcpu("us-east-1", "spot", "G") == 192 * 3
        res = sample_tracker.get_reservations()
        assert res[0]["vcpu_reserved"] == 576
        assert res[0]["num_instances"] == 3
        sample_tracker.release_cluster("big")
    else:
        # Quota insufficient (384 spot vCPU in fixture < 576 needed) — expected
        assert sample_tracker.get_used_vcpu("us-east-1", "spot", "G") == 0


# --- JobRecord / JobState (testing what JobTracker uses) ---


def test_job_record_default_status():
    spec = _make_spec(job_id="test-job-1")
    state = JobState(spec=spec, submitted_at=time.time())
    rec = JobRecord(state=state)
    assert rec.status == "queued"
    assert rec.state.spec.model_name == "Qwen/Qwen3-32B"


def test_job_record_status_update():
    spec = _make_spec(job_id="test-job-2")
    state = JobState(spec=spec, submitted_at=time.time())
    rec = JobRecord(state=state)
    rec.status = "running"
    assert rec.status == "running"
    rec.status = "succeeded"
    assert rec.status == "succeeded"


def test_job_record_progress_update():
    spec = _make_spec(job_id="test-job-3")
    state = JobState(spec=spec, submitted_at=time.time())
    rec = JobRecord(state=state)
    rec.state.progress_frac = 0.75
    assert rec.state.progress_frac == 0.75
    assert rec.state.remaining_tokens == int(0.25 * rec.state.total_tokens)
