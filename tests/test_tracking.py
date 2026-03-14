import pytest
from tracking.tracking import JobSpec, JobState, JobRecord, VPCQuotaTracker


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


# --- VPCQuotaTracker ---


def test_quota_tracker_reserve_and_release(sample_quota_csv_path):
    tracker = VPCQuotaTracker(quota_csv_file=sample_quota_csv_path)
    avail_before = tracker.get_available("us-east-1", "on_demand", "G")
    assert tracker.reserve("us-east-1", "on_demand", "G", 48) is True
    avail_after = tracker.get_available("us-east-1", "on_demand", "G")
    assert avail_after == avail_before - 48
    tracker.release("us-east-1", "on_demand", "G", 48)
    assert tracker.get_available("us-east-1", "on_demand", "G") == avail_before


def test_quota_tracker_reserve_insufficient(sample_quota_csv_path):
    tracker = VPCQuotaTracker(quota_csv_file=sample_quota_csv_path)
    assert tracker.reserve("us-east-1", "on_demand", "G", 999999) is False


def test_quota_tracker_status_summary(sample_quota_csv_path):
    tracker = VPCQuotaTracker(quota_csv_file=sample_quota_csv_path)
    tracker.reserve("us-east-1", "on_demand", "G", 48)
    summary = tracker.status_summary()
    assert "Region" in summary.columns
    assert "Used" in summary.columns
    assert len(summary) == 1
