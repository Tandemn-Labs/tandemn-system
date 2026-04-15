from types import SimpleNamespace
from threading import Lock

import pytest

import orca_server.dashboard as dashboard
from orca_server.job_manager import JobRecord
from quota.tracker import JobSpec, JobState


class FakeClusterManager:
    def __init__(self, states_by_job=None):
        self._states_by_job = states_by_job or {}

    def get_replica_states(self, job_id):
        return dict(self._states_by_job.get(job_id, {}))


class FakeTracker:
    def __init__(self, jobs):
        self.lock = Lock()
        self.jobs = jobs


def _make_job_record(
    *,
    job_id="job-1",
    status="loading_model",
    submitted_at=1000.0,
    num_replicas=1,
    instance_type="g5.xlarge",
):
    spec = JobSpec(
        job_id=job_id,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        num_lines=1000,
        avg_input_tokens=512,
        avg_output_tokens=256,
        slo_hours=1.0,
    )
    state = JobState(
        spec=spec,
        submitted_at=submitted_at,
        instance_types=instance_type,
    )
    return JobRecord(state=state, status=status, num_replicas=num_replicas)


def _build_payload(monkeypatch, rec, states, *, now, price):
    tracker = FakeTracker({rec.state.spec.job_id: rec})
    app_state = SimpleNamespace(
        metrics_collector=None,
        cluster_manager=FakeClusterManager({rec.state.spec.job_id: states}),
        redis_available=False,
    )
    monkeypatch.setattr("orca_server.job_manager.get_job_tracker", lambda: tracker)
    monkeypatch.setattr(dashboard, "_get_cached_price", lambda *_args: price)
    monkeypatch.setattr(dashboard.time, "time", lambda: now)

    dashboard._peak_cost.clear()
    dashboard._event_log.clear()
    dashboard._prev_job_status.clear()
    dashboard._prev_chunk_progress.clear()
    dashboard._prev_replica_phases.clear()

    return dashboard._build_dashboard_payload(app_state)


def test_cost_accrual_uses_launch_time_before_and_after_running_since(monkeypatch):
    rec = _make_job_record(status="loading_model")
    states = {
        "job-1-r0": {
            "phase": "provisioned",
            "launched_at": 1500.0,
            "instance_type": "g5.xlarge",
            "num_instances": 2,
        }
    }

    first = _build_payload(monkeypatch, rec, states, now=2000.0, price=2.0)
    first_cost = first["cost"]["job-1"]["accrued_usd"]
    assert first_cost == pytest.approx(0.5556, abs=1e-4)

    rec.status = "generating"
    states["job-1-r0"]["phase"] = "running"
    states["job-1-r0"]["running_since"] = 1900.0

    second = _build_payload(monkeypatch, rec, states, now=2000.0, price=2.0)
    assert second["cost"]["job-1"]["accrued_usd"] == pytest.approx(first_cost, abs=1e-4)
    assert second["cost"]["job-1"]["num_running_replicas"] == 1


def test_cost_accrual_fallback_multiplies_by_replica_count(monkeypatch):
    rec = _make_job_record(status="launching", submitted_at=1800.0, num_replicas=3)

    payload = _build_payload(monkeypatch, rec, {}, now=2100.0, price=1.5)

    assert payload["cost"]["job-1"]["accrued_usd"] == pytest.approx(0.375, abs=1e-4)
    assert payload["cost"]["job-1"]["num_running_replicas"] == 3
