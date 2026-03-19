"""Persistent metrics storage: SQLite with schema mirroring profiling results.json."""
from __future__ import annotations

import csv
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def _get_price_per_hour(
    instance_type: str, region: Optional[str], use_spot: bool
) -> Optional[float]:
    """Fetch hourly price via SkyPilot's catalog (uses its cached/refreshed data)."""
    try:
        from sky import catalog
        return catalog.get_hourly_cost(
            instance_type=instance_type,
            use_spot=use_spot,
            region=region,
            zone=None,
            clouds="aws",
        )
    except Exception as e:
        logger.debug("[MetricsDB] Could not fetch price for %s: %s", instance_type, e)
        return None

# Alias: CSV key → schema column name
FIELD_ALIASES: dict[str, str] = {
    "throughput_tokens_per_sec": "total_tokens_per_sec",
    "throughput_output_tokens_per_sec": "output_tokens_per_sec",
    "throughput_input_tokens_per_sec": "input_tokens_per_sec",
    "tensor_parallel_size": "tp_size",
    "pipeline_parallel_size": "pp_size",
}

# ISO timestamp fields that need datetime → unix conversion
ISO_TIMESTAMP_FIELDS: set[str] = {"job_start_timestamp", "job_end_timestamp"}

INT_FIELDS: set[str] = {
    "tp_size", "pp_size", "num_nodes", "gpus_per_node", "gpu_count_total",
    "num_requests_total", "num_requests_completed",
    "max_input_tokens", "min_input_tokens",
    "max_output_tokens", "min_output_tokens",
    "total_input_tokens", "total_output_tokens", "total_tokens",
    "max_num_seqs", "max_model_len", "num_preemptions",
    "vocab_size", "num_experts_active",
    "is_moe", "model_fits_single_gpu",
    "is_lmcache", "is_continuous_batching", "spec_decode", "cuda_graphs",
    "crosses_node_boundary",
    "gpu_samples", "scheduler_samples",
}

FLOAT_FIELDS: set[str] = {
    "params_billion", "gpu_mem_gb",
    "total_runtime_sec", "model_load_time_sec", "generation_time_sec",
    "job_start_timestamp", "job_end_timestamp",
    "avg_input_tokens", "p50_input_tokens", "p90_input_tokens", "p99_input_tokens",
    "avg_output_tokens", "p50_output_tokens", "p90_output_tokens", "p99_output_tokens",
    "gpu_memory_utilization",
    "throughput_requests_per_sec", "input_tokens_per_sec",
    "output_tokens_per_sec", "total_tokens_per_sec", "tokens_per_sec_per_gpu",
    "ttft_ms_p50", "ttft_ms_p95", "ttft_ms_p99",
    "tpot_ms_p50", "tpot_ms_p95", "tpot_ms_p99",
    "e2e_ms_p50", "e2e_ms_p95", "e2e_ms_p99",
    "tpot_client_ms_p50", "tpot_client_ms_p95", "tpot_client_ms_p99",
    "ttft_server_ms_p50", "ttft_server_ms_p95", "ttft_server_ms_p99",
    "e2e_server_ms_p50", "e2e_server_ms_p95", "e2e_server_ms_p99",
    "price_per_hour", "price_per_gpu_hour_usd",
    "cost_for_run_usd", "tokens_per_dollar", "cost_per_1m_tokens_total",
    "avg_sm_util_pct", "max_sm_util_pct",
    "avg_mem_bw_util_pct", "max_mem_bw_util_pct",
    "avg_mem_util_pct", "max_mem_util_pct",
    "running_avg", "running_max", "waiting_avg", "waiting_max",
    "swapped_avg", "swapped_max",
    "kv_cache_util_pct_avg", "kv_cache_util_pct_max",
    "model_size_gb", "vram_headroom_gb", "params_per_gpu",
    "attention_heads_per_kv_head", "kv_heads_per_tp",
    "bandwidth_per_param", "flops_per_param",
    "cost_per_1m_tokens_prefill_usd", "cost_per_1m_tokens_decode_usd",
    "gpu_bandwidth_gbps", "gpu_tflops_fp16",
    "queue_time_ms_p50", "queue_time_ms_p95", "queue_time_ms_p99",
    "prefill_time_ms_p50", "prefill_time_ms_p95", "prefill_time_ms_p99",
    "decode_time_ms_p50", "decode_time_ms_p95", "decode_time_ms_p99",
    "prefix_cache_hit_rate",
    "inference_time_ms_p50", "inference_time_ms_p95", "inference_time_ms_p99",
}

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL UNIQUE,

    -- Identity / config
    model_name TEXT, model_architecture TEXT, params_billion REAL,
    instance_type TEXT, gpu_name TEXT, gpu_model TEXT,
    tp_size INTEGER, pp_size INTEGER, num_nodes INTEGER,
    gpus_per_node INTEGER, gpu_count_total INTEGER,
    gpu_mem_gb REAL,
    actual_region TEXT, actual_market TEXT, solver TEXT, job_dirname TEXT,
    precision TEXT, runtime_stack TEXT, interconnect TEXT,
    gpu_bandwidth_gbps REAL, gpu_tflops_fp16 REAL, gpu_generation TEXT,

    -- Timing
    total_runtime_sec REAL, model_load_time_sec REAL, generation_time_sec REAL,
    job_start_timestamp REAL, job_end_timestamp REAL,

    -- Workload
    num_requests_total INTEGER, num_requests_completed INTEGER,
    avg_input_tokens REAL, max_input_tokens INTEGER, min_input_tokens INTEGER,
    p50_input_tokens REAL, p90_input_tokens REAL, p99_input_tokens REAL,
    avg_output_tokens REAL, max_output_tokens INTEGER, min_output_tokens INTEGER,
    p50_output_tokens REAL, p90_output_tokens REAL, p99_output_tokens REAL,
    total_input_tokens INTEGER, total_output_tokens INTEGER, total_tokens INTEGER,
    max_num_seqs INTEGER, max_model_len INTEGER,
    gpu_memory_utilization REAL, dtype TEXT, kv_cache_dtype TEXT,
    num_preemptions INTEGER,

    -- Throughput
    throughput_requests_per_sec REAL,
    input_tokens_per_sec REAL,
    output_tokens_per_sec REAL,
    total_tokens_per_sec REAL,
    tokens_per_sec_per_gpu REAL,

    -- Latency percentiles (client-side SSE timing)
    ttft_ms_p50 REAL, ttft_ms_p95 REAL, ttft_ms_p99 REAL,
    tpot_ms_p50 REAL, tpot_ms_p95 REAL, tpot_ms_p99 REAL,
    e2e_ms_p50 REAL,  e2e_ms_p95 REAL,  e2e_ms_p99 REAL,
    -- Client-side TPOT (per-request: (e2e-ttft)/(output_tokens-1))
    tpot_client_ms_p50 REAL, tpot_client_ms_p95 REAL, tpot_client_ms_p99 REAL,
    -- Server-side latency from Prometheus histogram deltas
    ttft_server_ms_p50 REAL, ttft_server_ms_p95 REAL, ttft_server_ms_p99 REAL,
    e2e_server_ms_p50 REAL, e2e_server_ms_p95 REAL, e2e_server_ms_p99 REAL,

    -- Cost
    price_per_hour REAL, price_per_gpu_hour_usd REAL,
    cost_for_run_usd REAL,
    tokens_per_dollar REAL,
    cost_per_1m_tokens_total REAL,

    -- GPU utilization aggregates (NULL until Phase 2 pynvml)
    avg_sm_util_pct REAL, max_sm_util_pct REAL,
    avg_mem_bw_util_pct REAL, max_mem_bw_util_pct REAL,
    avg_mem_util_pct REAL, max_mem_util_pct REAL,
    gpu_samples INTEGER,

    -- Scheduler aggregates (computed from timeseries)
    running_avg REAL, running_max REAL,
    waiting_avg REAL, waiting_max REAL,
    swapped_avg REAL, swapped_max REAL,
    kv_cache_util_pct_avg REAL, kv_cache_util_pct_max REAL,
    scheduler_samples INTEGER,

    -- Model properties
    vocab_size INTEGER, is_moe INTEGER,
    num_experts_active INTEGER,
    model_size_gb REAL,
    model_fits_single_gpu INTEGER,
    vram_headroom_gb REAL,
    params_per_gpu REAL,
    attention_heads_per_kv_head REAL,
    kv_heads_per_tp REAL,
    model_config_json TEXT,

    -- Derived (hardware efficiency)
    bandwidth_per_param REAL,
    flops_per_param REAL,
    crosses_node_boundary INTEGER,

    -- Derived (prefill/decode cost)
    cost_per_1m_tokens_prefill_usd REAL,
    cost_per_1m_tokens_decode_usd REAL,

    -- Feature flags
    is_lmcache INTEGER DEFAULT 0,
    is_continuous_batching INTEGER DEFAULT 1,
    spec_decode INTEGER DEFAULT 0,
    cuda_graphs INTEGER DEFAULT 1,
    kv_offload_target TEXT,

    -- Raw metrics dump
    raw_metrics TEXT,

    created_at REAL NOT NULL
)
"""

_CREATE_TIMESERIES = """
CREATE TABLE IF NOT EXISTS timeseries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id     TEXT NOT NULL,
    timestamp  REAL NOT NULL,
    metrics    TEXT NOT NULL,
    replica_id TEXT
)
"""

_CREATE_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_ts_job_time ON timeseries (job_id, timestamp)"
)

_CREATE_INDEX_REPLICA = (
    "CREATE INDEX IF NOT EXISTS idx_ts_job_replica_time ON timeseries (job_id, replica_id, timestamp)"
)

_CREATE_REPLICA_SUMMARIES = """
CREATE TABLE IF NOT EXISTS replica_summaries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id     TEXT NOT NULL,
    replica_id TEXT NOT NULL,
    metrics    TEXT NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE(job_id, replica_id)
)
"""

_CREATE_INDEX_REPLICA_SUMMARY = (
    "CREATE INDEX IF NOT EXISTS idx_rs_job ON replica_summaries (job_id)"
)


class MetricsDB:
    def __init__(self, db_path: str = "temp/metrics_db.sqlite") -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # Columns added after initial schema — ALTER TABLE for existing DBs
    _MIGRATE_COLUMNS = [
        ("tpot_client_ms_p50", "REAL"), ("tpot_client_ms_p95", "REAL"), ("tpot_client_ms_p99", "REAL"),
        ("ttft_server_ms_p50", "REAL"), ("ttft_server_ms_p95", "REAL"), ("ttft_server_ms_p99", "REAL"),
        ("e2e_server_ms_p50", "REAL"), ("e2e_server_ms_p95", "REAL"), ("e2e_server_ms_p99", "REAL"),
        ("params_per_gpu", "REAL"), ("attention_heads_per_kv_head", "REAL"), ("kv_heads_per_tp", "REAL"),
        ("model_config_json", "TEXT"), ("bandwidth_per_param", "REAL"), ("flops_per_param", "REAL"),
        ("crosses_node_boundary", "INTEGER"), ("cost_per_1m_tokens_prefill_usd", "REAL"),
        ("cost_per_1m_tokens_decode_usd", "REAL"), ("kv_offload_target", "TEXT"),
        ("queue_time_ms_p50", "REAL"), ("queue_time_ms_p95", "REAL"), ("queue_time_ms_p99", "REAL"),
        ("prefill_time_ms_p50", "REAL"), ("prefill_time_ms_p95", "REAL"), ("prefill_time_ms_p99", "REAL"),
        ("decode_time_ms_p50", "REAL"), ("decode_time_ms_p95", "REAL"), ("decode_time_ms_p99", "REAL"),
        ("prefix_cache_hit_rate", "REAL"),
        ("inference_time_ms_p50", "REAL"), ("inference_time_ms_p95", "REAL"), ("inference_time_ms_p99", "REAL"),
    ]

    def _init_schema(self) -> None:
        with self._get_conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_RUNS)
            conn.execute(_CREATE_TIMESERIES)
            conn.execute(_CREATE_INDEX)
            # Migrate: add columns that may be missing in existing DBs
            existing = {r[1] for r in conn.execute("PRAGMA table_info(runs)").fetchall()}
            for col, dtype in self._MIGRATE_COLUMNS:
                if col not in existing:
                    conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {dtype}")
            # Migrate timeseries: add replica_id if missing (must run before replica index)
            ts_cols = {r[1] for r in conn.execute("PRAGMA table_info(timeseries)").fetchall()}
            if "replica_id" not in ts_cols:
                conn.execute("ALTER TABLE timeseries ADD COLUMN replica_id TEXT")
            conn.execute(_CREATE_INDEX_REPLICA)
            conn.execute(_CREATE_REPLICA_SUMMARIES)
            conn.execute(_CREATE_INDEX_REPLICA_SUMMARY)
            conn.commit()

    # ------------------------------------------------------------------
    # Timeseries (flushed every 60s during a running job)
    # ------------------------------------------------------------------

    def append_timeseries(self, job_id: str, snapshots: list[dict]) -> None:
        rows = [
            (job_id, s.get("timestamp", time.time()), json.dumps(s), s.get("replica_id"))
            for s in snapshots
        ]
        try:
            with self._get_conn() as conn:
                conn.executemany(
                    "INSERT INTO timeseries (job_id, timestamp, metrics, replica_id) VALUES (?, ?, ?, ?)",
                    rows,
                )
                conn.commit()
        except sqlite3.OperationalError as e:
            if "no such table" not in str(e):
                raise
            self._init_schema()
            with self._get_conn() as conn:
                conn.executemany(
                    "INSERT INTO timeseries (job_id, timestamp, metrics, replica_id) VALUES (?, ?, ?, ?)",
                    rows,
                )
                conn.commit()

    # ------------------------------------------------------------------
    # Per-replica summaries (POSTed by each replica after all chunks done)
    # ------------------------------------------------------------------

    def push_replica_summary(self, job_id: str, replica_id: str, metrics: dict) -> None:
        """Store a per-replica build_metrics summary dict."""
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO replica_summaries (job_id, replica_id, metrics, created_at) "
                "VALUES (?, ?, ?, ?)",
                (job_id, replica_id, json.dumps(metrics), time.time()),
            )
            conn.commit()

    def get_replica_summaries(self, job_id: str) -> list[dict]:
        """Retrieve all per-replica summaries for a job."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT replica_id, metrics FROM replica_summaries WHERE job_id = ? ORDER BY created_at",
                (job_id,),
            ).fetchall()
        result = []
        for r in rows:
            try:
                d = json.loads(r["metrics"])
                d["replica_id"] = r["replica_id"]
                result.append(d)
            except Exception:
                pass
        return result

    def aggregate_replica_summaries(self, job_id: str) -> dict | None:
        """Aggregate per-replica summaries into a single job-level summary.

        Sum: tokens, requests, throughput.  Avg: latency, GPU util.
        """
        summaries = self.get_replica_summaries(job_id)
        if not summaries:
            return None

        agg: dict = {}
        # Fields to sum across replicas
        sum_keys = [
            "num_requests_total", "num_requests_completed", "num_requests_skipped",
            "total_input_tokens", "total_output_tokens", "total_tokens",
            "num_preemptions", "gpu_samples", "scheduler_samples",
        ]
        # Fields to average across replicas
        avg_keys = [
            "throughput_requests_per_sec", "throughput_tokens_per_sec",
            "throughput_output_tokens_per_sec", "throughput_input_tokens_per_sec",
            "ttft_ms_p50", "ttft_ms_p95", "ttft_ms_p99",
            "tpot_ms_p50", "tpot_ms_p95", "tpot_ms_p99",
            "e2e_ms_p50", "e2e_ms_p95", "e2e_ms_p99",
            "tpot_client_ms_p50", "tpot_client_ms_p95", "tpot_client_ms_p99",
            "ttft_server_ms_p50", "ttft_server_ms_p95", "ttft_server_ms_p99",
            "e2e_server_ms_p50", "e2e_server_ms_p95", "e2e_server_ms_p99",
            "queue_time_ms_p50", "queue_time_ms_p95", "queue_time_ms_p99",
            "prefill_time_ms_p50", "prefill_time_ms_p95", "prefill_time_ms_p99",
            "decode_time_ms_p50", "decode_time_ms_p95", "decode_time_ms_p99",
            "inference_time_ms_p50", "inference_time_ms_p95", "inference_time_ms_p99",
            "avg_sm_util_pct", "max_sm_util_pct",
            "avg_mem_bw_util_pct", "max_mem_bw_util_pct",
            "avg_mem_util_pct", "max_mem_util_pct",
            "running_avg", "running_max", "waiting_avg", "waiting_max",
            "kv_cache_util_pct_avg", "kv_cache_util_pct_max",
            "prefix_cache_hit_rate",
        ]
        # Copy-first fields (take from first replica)
        copy_keys = [
            "model_name", "quantization", "cloud_provider", "instance_type",
            "gpu_name", "engine", "tensor_parallel_size", "pipeline_parallel_size",
            "max_model_len", "max_num_seqs", "gpu_memory_utilization",
            "dtype", "kv_cache_dtype", "model_architecture", "params_billion",
            "is_moe", "num_experts_active", "vocab_size",
            "gpu_mem_gb", "gpu_tflops_fp16", "gpu_bandwidth_gbps",
            "gpu_model", "gpu_generation", "interconnect",
            "num_nodes", "precision", "runtime_stack",
            "model_size_gb", "model_config_json",
        ]

        for k in sum_keys:
            vals = [s.get(k) for s in summaries if s.get(k) is not None]
            agg[k] = sum(vals) if vals else None

        for k in avg_keys:
            vals = [s.get(k) for s in summaries if s.get(k) is not None]
            agg[k] = sum(vals) / len(vals) if vals else None

        for k in copy_keys:
            for s in summaries:
                if s.get(k) is not None:
                    agg[k] = s[k]
                    break

        # Timing: use max generation_time, model_load_time across replicas
        for k in ("generation_time_sec", "model_load_time_sec", "total_runtime_sec"):
            vals = [s.get(k) for s in summaries if s.get(k) is not None]
            agg[k] = max(vals) if vals else None

        # Throughput: sum across replicas (each replica's throughput adds up)
        for k in ("throughput_requests_per_sec", "throughput_tokens_per_sec",
                   "throughput_output_tokens_per_sec", "throughput_input_tokens_per_sec"):
            vals = [s.get(k) for s in summaries if s.get(k) is not None]
            agg[k] = sum(vals) if vals else None

        # Timestamps: earliest start, latest end
        starts = [s.get("job_start_timestamp") for s in summaries if s.get("job_start_timestamp")]
        ends = [s.get("job_end_timestamp") for s in summaries if s.get("job_end_timestamp")]
        if starts:
            agg["job_start_timestamp"] = min(starts)
        if ends:
            agg["job_end_timestamp"] = max(ends)

        # Cost: aggregate from first replica (same instance type)
        for k in ("price_per_hour", "cost_for_run_usd", "tokens_per_dollar"):
            vals = [s.get(k) for s in summaries if s.get(k) is not None]
            if vals:
                agg[k] = sum(vals) if k == "cost_for_run_usd" else vals[0]

        agg["num_replicas"] = len(summaries)
        return agg

    # ------------------------------------------------------------------
    # Runs (written once at job completion)
    # ------------------------------------------------------------------

    def push_run(
        self,
        job_id: str,
        metrics_csv_path: str,
        *,
        actual_region: str,
        actual_market: str,
        solver: str,
        job_dirname: str,
        last_snapshot=None,
    ) -> int:
        row = self._parse_metrics_csv(metrics_csv_path)
        row["job_id"] = job_id
        row["actual_region"] = actual_region
        row["actual_market"] = actual_market
        row["solver"] = solver
        row["job_dirname"] = job_dirname
        row["created_at"] = time.time()

        # Latency percentiles from final MetricsSnapshot (fallback only —
        # metrics.csv has more accurate client-side values when available)
        if last_snapshot is not None:
            for f in [
                "ttft_ms_p50", "ttft_ms_p95", "ttft_ms_p99",
                "tpot_ms_p50", "tpot_ms_p95", "tpot_ms_p99",
                "e2e_ms_p50", "e2e_ms_p95", "e2e_ms_p99",
                "queue_time_ms_p50", "queue_time_ms_p95", "queue_time_ms_p99",
                "prefill_time_ms_p50", "prefill_time_ms_p95", "prefill_time_ms_p99",
                "decode_time_ms_p50", "decode_time_ms_p95", "decode_time_ms_p99",
                "inference_time_ms_p50", "inference_time_ms_p95", "inference_time_ms_p99",
                "prefix_cache_hit_rate",
            ]:
                if row.get(f) is None:
                    val = getattr(last_snapshot, f, None)
                    if val is not None:
                        row[f] = val

        # Store raw CSV dump before adding derived fields
        row["raw_metrics"] = json.dumps({k: v for k, v in row.items() if k != "raw_metrics"})

        derived = self._compute_derived(row, job_id)
        row.update(derived)

        # Filter to known schema columns only — unknown CSV fields already in raw_metrics
        schema_cols = self._get_schema_columns()
        insert_row = {k: v for k, v in row.items() if k in schema_cols}

        cols = list(insert_row.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)
        values = [insert_row[c] for c in cols]

        with self._get_conn() as conn:
            cur = conn.execute(
                f"INSERT OR REPLACE INTO runs ({col_names}) VALUES ({placeholders})",
                values,
            )
            conn.commit()
            return cur.lastrowid

    def _get_schema_columns(self) -> set[str]:
        """Return the set of column names in the runs table."""
        with self._get_conn() as conn:
            rows = conn.execute("PRAGMA table_info(runs)").fetchall()
        return {r["name"] for r in rows}

    def _parse_metrics_csv(self, path: str) -> dict:
        """Parse key,value CSV → typed dict.

        Handles:
        - Header row (key=="metric") is skipped
        - Field aliases (e.g. throughput_tokens_per_sec → total_tokens_per_sec)
        - ISO 8601 timestamps → unix float
        - Unknown keys stored in _extras for raw_metrics, not inserted into runs
        """
        result: dict = {}
        try:
            with open(path) as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 2:
                        continue
                    key = row[0].strip()
                    val = row[1].strip()
                    # Skip header row
                    if not key or key == "metric":
                        continue
                    # Apply field aliases
                    key = FIELD_ALIASES.get(key, key)
                    # ISO timestamp fields
                    if key in ISO_TIMESTAMP_FIELDS:
                        try:
                            from datetime import datetime, timezone
                            dt = datetime.fromisoformat(val)
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            result[key] = dt.timestamp()
                        except (ValueError, TypeError):
                            result[key] = None
                    elif key in INT_FIELDS:
                        try:
                            result[key] = int(float(val))
                        except (ValueError, TypeError):
                            result[key] = None
                    elif key in FLOAT_FIELDS:
                        try:
                            result[key] = float(val)
                        except (ValueError, TypeError):
                            result[key] = None
                    else:
                        result[key] = val if val else None
        except FileNotFoundError:
            logger.warning("[MetricsDB] metrics.csv not found at %s", path)
        return result

    def _compute_derived(self, row: dict, job_id: str) -> dict:
        """Compute scheduler aggregates, cost, and other derived fields."""
        from orca_server.config import AWS_INSTANCES

        derived: dict = {}

        # Scheduler aggregates from timeseries
        try:
            ts = self.get_timeseries(job_id)
            if ts:
                running_vals = [
                    t.get("num_requests_running", 0)
                    for t in ts if t.get("num_requests_running") is not None
                ]
                waiting_vals = [
                    t.get("num_requests_waiting", 0)
                    for t in ts if t.get("num_requests_waiting") is not None
                ]
                swapped_vals = [
                    t.get("num_requests_swapped", 0)
                    for t in ts if t.get("num_requests_swapped") is not None
                ]
                kv_vals = [
                    t.get("gpu_cache_usage_perc", 0) * 100
                    for t in ts if t.get("gpu_cache_usage_perc") is not None
                ]
                derived["scheduler_samples"] = len(ts)
                if running_vals:
                    derived["running_avg"] = sum(running_vals) / len(running_vals)
                    derived["running_max"] = float(max(running_vals))
                if waiting_vals:
                    derived["waiting_avg"] = sum(waiting_vals) / len(waiting_vals)
                    derived["waiting_max"] = float(max(waiting_vals))
                if swapped_vals:
                    derived["swapped_avg"] = sum(swapped_vals) / len(swapped_vals)
                    derived["swapped_max"] = float(max(swapped_vals))
                if kv_vals:
                    derived["kv_cache_util_pct_avg"] = sum(kv_vals) / len(kv_vals)
                    derived["kv_cache_util_pct_max"] = float(max(kv_vals))
        except Exception as e:
            logger.warning("[MetricsDB] Could not compute scheduler aggregates: %s", e)

        # gpu_count_total
        instance_type = row.get("instance_type")
        num_nodes = row.get("num_nodes") or 1
        gpu_count_total = None
        if instance_type and instance_type in AWS_INSTANCES:
            gpus_per_node = AWS_INSTANCES[instance_type][1]
            gpu_count_total = gpus_per_node * num_nodes
            derived["gpu_count_total"] = gpu_count_total
            derived["gpus_per_node"] = gpus_per_node

        # tokens_per_sec_per_gpu
        total_tokens_per_sec = row.get("total_tokens_per_sec")
        if total_tokens_per_sec and gpu_count_total:
            derived["tokens_per_sec_per_gpu"] = total_tokens_per_sec / gpu_count_total

        # Cost — use SkyPilot catalog for real per-region prices
        actual_region = row.get("actual_region")
        use_spot = (row.get("actual_market", "spot") == "spot")
        price_per_hour = _get_price_per_hour(instance_type, actual_region, use_spot) if instance_type else None
        if price_per_hour:
            derived["price_per_hour"] = price_per_hour
            if gpu_count_total:
                derived["price_per_gpu_hour_usd"] = price_per_hour / gpu_count_total
            total_runtime_sec = row.get("total_runtime_sec")
            if total_runtime_sec:
                cost = price_per_hour * total_runtime_sec / 3600
                derived["cost_for_run_usd"] = cost
                total_tokens = row.get("total_tokens")
                if total_tokens and total_tokens > 0 and cost > 0:
                    derived["tokens_per_dollar"] = total_tokens / cost
                    derived["cost_per_1m_tokens_total"] = cost / (total_tokens / 1_000_000)

        return derived

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_runs(
        self, *, model: Optional[str] = None, gpu: Optional[str] = None,
        limit: int = 50, offset: int = 0
    ) -> list[dict]:
        query = "SELECT * FROM runs WHERE 1=1"
        params: list = []
        if model:
            query += " AND model_name LIKE ?"
            params.append(f"%{model}%")
        if gpu:
            query += " AND gpu_name = ?"
            params.append(gpu)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: int) -> Optional[dict]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def get_timeseries(
        self,
        job_id: str,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> list[dict]:
        query = "SELECT timestamp, metrics FROM timeseries WHERE job_id = ?"
        params: list = [job_id]
        if start is not None:
            query += " AND timestamp >= ?"
            params.append(start)
        if end is not None:
            query += " AND timestamp <= ?"
            params.append(end)
        query += " ORDER BY timestamp ASC"
        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
        result = []
        for r in rows:
            try:
                d = json.loads(r["metrics"])
                d["timestamp"] = r["timestamp"]
                result.append(d)
            except Exception:
                pass
        return result


_metrics_db: Optional[MetricsDB] = None


def get_metrics_db() -> MetricsDB:
    global _metrics_db
    if _metrics_db is None:
        _metrics_db = MetricsDB()
    return _metrics_db
