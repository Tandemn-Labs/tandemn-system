"""Unit tests for orca_server.metrics_db."""
import json
import time

import pytest

from orca_server.metrics_db import MetricsDB
from orca_server.monitoring import MetricsSnapshot


@pytest.fixture
def db(tmp_path):
    return MetricsDB(db_path=str(tmp_path / "test_metrics.sqlite"))


# Sample metrics.csv covering the key fields the DB ingests
SAMPLE_METRICS_CSV = """\
model_name,Qwen/Qwen2.5-72B-Instruct
model_architecture,Qwen2ForCausalLM
params_billion,72.7
instance_type,g6e.48xlarge
gpu_name,L40S
gpu_model,NVIDIA L40S
tp_size,8
pp_size,1
num_nodes,1
gpus_per_node,8
gpu_mem_gb,48.0
precision,bfloat16
runtime_stack,vllm
total_runtime_sec,308.4
model_load_time_sec,42.1
generation_time_sec,266.3
job_start_timestamp,1741996200.0
job_end_timestamp,1741996508.4
num_requests_total,500
num_requests_completed,500
avg_input_tokens,256.0
max_input_tokens,512
min_input_tokens,64
p50_input_tokens,240.0
p90_input_tokens,480.0
p99_input_tokens,510.0
avg_output_tokens,128.0
max_output_tokens,256
min_output_tokens,32
p50_output_tokens,120.0
p90_output_tokens,240.0
p99_output_tokens,254.0
total_input_tokens,128000
total_output_tokens,64000
total_tokens,192000
max_num_seqs,256
max_model_len,8192
gpu_memory_utilization,0.90
dtype,bfloat16
kv_cache_dtype,auto
num_preemptions,3
throughput_requests_per_sec,1.62
output_tokens_per_sec,207.5
total_tokens_per_sec,622.5
"""


# ---------------------------------------------------------------------------
# TestPushRun
# ---------------------------------------------------------------------------

class TestPushRun:
    def test_push_run_returns_positive_int(self, db, tmp_path):
        csv_file = tmp_path / "metrics.csv"
        csv_file.write_text(SAMPLE_METRICS_CSV)
        run_id = db.push_run(
            "job-001",
            str(csv_file),
            actual_region="us-east-1",
            actual_market="spot",
            solver="roofline",
            job_dirname="qwen72b/test/success-roofline-L40S-tp8-pp1-20260315",
        )
        assert isinstance(run_id, int)
        assert run_id > 0

    def test_all_scalar_fields_stored(self, db, tmp_path):
        csv_file = tmp_path / "metrics.csv"
        csv_file.write_text(SAMPLE_METRICS_CSV)
        run_id = db.push_run(
            "job-002",
            str(csv_file),
            actual_region="us-east-1",
            actual_market="spot",
            solver="roofline",
            job_dirname="test-run",
        )
        run = db.get_run(run_id)
        assert run is not None
        assert run["model_name"] == "Qwen/Qwen2.5-72B-Instruct"
        assert run["instance_type"] == "g6e.48xlarge"
        assert run["tp_size"] == 8
        assert run["total_runtime_sec"] == pytest.approx(308.4)
        assert run["num_requests_total"] == 500
        assert run["total_tokens"] == 192000
        assert run["actual_region"] == "us-east-1"
        assert run["solver"] == "roofline"

    def test_context_fields_stored(self, db, tmp_path):
        csv_file = tmp_path / "metrics.csv"
        csv_file.write_text("model_name,TestModel\n")
        run_id = db.push_run(
            "job-ctx",
            str(csv_file),
            actual_region="us-west-2",
            actual_market="on_demand",
            solver="user_specified",
            job_dirname="some/job/dir",
        )
        run = db.get_run(run_id)
        assert run["actual_region"] == "us-west-2"
        assert run["actual_market"] == "on_demand"
        assert run["solver"] == "user_specified"
        assert run["job_dirname"] == "some/job/dir"

    def test_duplicate_job_id_replaced(self, db, tmp_path):
        csv_file = tmp_path / "metrics.csv"
        csv_file.write_text(SAMPLE_METRICS_CSV)
        db.push_run(
            "job-dup",
            str(csv_file),
            actual_region="us-east-1",
            actual_market="spot",
            solver="roofline",
            job_dirname="run-a",
        )
        id2 = db.push_run(
            "job-dup",
            str(csv_file),
            actual_region="us-west-2",
            actual_market="on_demand",
            solver="user_specified",
            job_dirname="run-b",
        )
        run = db.get_run(id2)
        assert run["actual_region"] == "us-west-2"
        assert run["solver"] == "user_specified"

    def test_latency_from_snapshot(self, db, tmp_path):
        csv_file = tmp_path / "metrics.csv"
        csv_file.write_text(SAMPLE_METRICS_CSV)
        snap = MetricsSnapshot("job-lat", time.time())
        snap.ttft_ms_p50 = 45.2
        snap.ttft_ms_p95 = 180.0
        snap.ttft_ms_p99 = 230.1
        snap.e2e_ms_p50 = 500.0
        run_id = db.push_run(
            "job-lat",
            str(csv_file),
            actual_region="us-east-1",
            actual_market="spot",
            solver="roofline",
            job_dirname="test",
            last_snapshot=snap,
        )
        run = db.get_run(run_id)
        assert run["ttft_ms_p50"] == pytest.approx(45.2)
        assert run["ttft_ms_p99"] == pytest.approx(230.1)
        assert run["e2e_ms_p50"] == pytest.approx(500.0)

    def test_missing_csv_does_not_crash(self, db, tmp_path):
        run_id = db.push_run(
            "job-nocsv",
            str(tmp_path / "nonexistent.csv"),
            actual_region="us-east-1",
            actual_market="spot",
            solver="roofline",
            job_dirname="test",
        )
        assert run_id > 0
        run = db.get_run(run_id)
        assert run["job_id"] == "job-nocsv"


# ---------------------------------------------------------------------------
# TestTimeseries
# ---------------------------------------------------------------------------

class TestTimeseries:
    def test_append_and_retrieve(self, db):
        snapshots = [
            {
                "job_id": "job-ts",
                "timestamp": 1000.0 + i,
                "num_requests_running": i * 10,
                "gpu_cache_usage_perc": 0.5,
            }
            for i in range(5)
        ]
        db.append_timeseries("job-ts", snapshots)
        result = db.get_timeseries("job-ts")
        assert len(result) == 5
        assert result[0]["timestamp"] == pytest.approx(1000.0)
        assert result[-1]["timestamp"] == pytest.approx(1004.0)

    def test_time_range_filter(self, db):
        snapshots = [
            {"job_id": "job-ts2", "timestamp": float(i), "num_requests_running": i}
            for i in range(10)
        ]
        db.append_timeseries("job-ts2", snapshots)
        result = db.get_timeseries("job-ts2", start=3.0, end=6.0)
        timestamps = [r["timestamp"] for r in result]
        assert all(3.0 <= t <= 6.0 for t in timestamps)
        assert len(result) == 4

    def test_ordered_by_timestamp(self, db):
        # Insert in reverse order
        snapshots = [
            {"job_id": "job-ts3", "timestamp": float(9 - i), "num_requests_running": i}
            for i in range(10)
        ]
        db.append_timeseries("job-ts3", snapshots)
        result = db.get_timeseries("job-ts3")
        timestamps = [r["timestamp"] for r in result]
        assert timestamps == sorted(timestamps)

    def test_empty_job_returns_empty_list(self, db):
        assert db.get_timeseries("nonexistent-job") == []


# ---------------------------------------------------------------------------
# TestSchedulerAggregates
# ---------------------------------------------------------------------------

class TestSchedulerAggregates:
    def test_aggregates_computed_from_timeseries(self, db, tmp_path):
        snapshots = [
            {
                "job_id": "job-agg",
                "timestamp": float(i),
                "num_requests_running": float(i * 10),
                "num_requests_waiting": float(i),
                "num_requests_swapped": 0.0,
                "gpu_cache_usage_perc": i * 0.1,
            }
            for i in range(1, 6)  # running: 10, 20, 30, 40, 50
        ]
        db.append_timeseries("job-agg", snapshots)

        csv_file = tmp_path / "metrics.csv"
        csv_file.write_text("total_runtime_sec,60.0\ntotal_tokens,1000\n")
        run_id = db.push_run(
            "job-agg",
            str(csv_file),
            actual_region="us-east-1",
            actual_market="spot",
            solver="roofline",
            job_dirname="test",
        )
        run = db.get_run(run_id)
        assert run["running_avg"] == pytest.approx(30.0)   # avg(10,20,30,40,50)
        assert run["running_max"] == pytest.approx(50.0)
        assert run["scheduler_samples"] == 5
        assert run["kv_cache_util_pct_avg"] == pytest.approx(30.0)  # avg(10,20,30,40,50)%

    def test_kv_pct_converted_from_fraction(self, db, tmp_path):
        """gpu_cache_usage_perc is 0-1 in snapshots; stored as 0-100 pct."""
        snapshots = [
            {"job_id": "job-kv", "timestamp": float(i), "gpu_cache_usage_perc": 0.5}
            for i in range(3)
        ]
        db.append_timeseries("job-kv", snapshots)
        csv_file = tmp_path / "metrics.csv"
        csv_file.write_text("")
        run_id = db.push_run(
            "job-kv",
            str(csv_file),
            actual_region="us-east-1",
            actual_market="spot",
            solver="roofline",
            job_dirname="test",
        )
        run = db.get_run(run_id)
        assert run["kv_cache_util_pct_avg"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# TestListRuns
# ---------------------------------------------------------------------------

class TestListRuns:
    def test_list_runs_empty(self, db):
        assert db.list_runs() == []

    def test_list_runs_filter_model(self, db, tmp_path):
        csv1 = tmp_path / "m1.csv"
        csv1.write_text("model_name,Qwen/Qwen2.5-72B\n")
        csv2 = tmp_path / "m2.csv"
        csv2.write_text("model_name,meta-llama/Llama-3-70B\n")
        db.push_run("job-q", str(csv1), actual_region="us-east-1", actual_market="spot", solver="roofline", job_dirname="a")
        db.push_run("job-l", str(csv2), actual_region="us-east-1", actual_market="spot", solver="roofline", job_dirname="b")

        runs = db.list_runs(model="Qwen")
        assert len(runs) == 1
        assert runs[0]["model_name"] == "Qwen/Qwen2.5-72B"

    def test_list_runs_filter_gpu(self, db, tmp_path):
        csv1 = tmp_path / "g1.csv"
        csv1.write_text("model_name,ModelA\ngpu_name,L40S\n")
        csv2 = tmp_path / "g2.csv"
        csv2.write_text("model_name,ModelB\ngpu_name,A100\n")
        db.push_run("job-g1", str(csv1), actual_region="us-east-1", actual_market="spot", solver="roofline", job_dirname="c")
        db.push_run("job-g2", str(csv2), actual_region="us-east-1", actual_market="spot", solver="roofline", job_dirname="d")

        runs = db.list_runs(gpu="L40S")
        assert len(runs) == 1
        assert runs[0]["gpu_name"] == "L40S"

    def test_list_runs_limit(self, db, tmp_path):
        for i in range(5):
            csv_file = tmp_path / f"m{i}.csv"
            csv_file.write_text(f"model_name,Model{i}\n")
            db.push_run(f"job-{i}", str(csv_file), actual_region="us-east-1", actual_market="spot", solver="roofline", job_dirname=f"run-{i}")
        runs = db.list_runs(limit=3)
        assert len(runs) == 3


# ---------------------------------------------------------------------------
# TestGetRun
# ---------------------------------------------------------------------------

class TestGetRun:
    def test_get_run_none_for_unknown(self, db):
        assert db.get_run(999) is None

    def test_get_run_returns_correct_row(self, db, tmp_path):
        csv_file = tmp_path / "m.csv"
        csv_file.write_text("model_name,TestModel\n")
        run_id = db.push_run(
            "job-get",
            str(csv_file),
            actual_region="us-east-1",
            actual_market="spot",
            solver="roofline",
            job_dirname="test",
        )
        run = db.get_run(run_id)
        assert run is not None
        assert run["job_id"] == "job-get"
        assert run["id"] == run_id
