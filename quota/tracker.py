from dataclasses import dataclass, field
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal
from threading import Lock
import pandas as pd

import logging

logger = logging.getLogger(__name__)


@dataclass
class VPCQuotaTracker:
    quota_csv_file: str = "quota/aws_gpu_quota_by_region.csv"
    db_path: str = "temp/quota.db"
    quota_df: pd.DataFrame = field(init=False)
    # Key: (region, market, family_type) → vcpu_in_use
    used_vcpu: Dict[Tuple[str, str, str], int] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)

    def __post_init__(self):
        self.reload_quota()
        self._init_db()
    
    def reload_quota(self):
        """Reload the quota from CSV (called after Refresh)"""
        logger.info(f"Reloading Quota from {self.quota_csv_file}")
        self.quota_df = pd.read_csv(self.quota_csv_file)
        logger.info(f"[QuotaTracker] Loaded {len(self.quota_df)} instance types")
    
    def get_baseline_quota(self, region:str, market:str, family_type:str):
        """Get the AWS Quota limit for family type"""
        col = f"{region}_{market}"
        family_rows = self.quota_df[self.quota_df["Family_Type"] == family_type]
        if col not in family_rows:
            return 0
        return family_rows[col].iloc[0]

    def get_used_vcpu(self, region:str, market:str, family_type:str):
        """Get the vCPU in use for the given region, market, and family type"""
        return self.used_vcpu.get((region, market, family_type), 0)
    
    def get_available(self,region:str, market:str, family_type:str):
        """combine the baseline quota and the used vCPU to get the available vCPU"""
        baseline_quota = self.get_baseline_quota(region, market, family_type)
        used_vcpu = self.get_used_vcpu(region, market, family_type)
        return baseline_quota - used_vcpu
    
    def reserve(self, region:str, market:str, family_type:str, vcpu:int):
        """Reserve vCPU, returns True if successful, False otherwise"""
        with self.lock:
            available = self.get_available(region, market, family_type)

            if vcpu > available:
                logger.warning(f"[QuotaTracker] Not enough quota for {region}, {market}, {family_type}")
                return False
            self.used_vcpu[(region, market, family_type)] = self.used_vcpu.get((region, market, family_type), 0) + vcpu
            logger.info(f"[QuotaTracker] Reserved {vcpu} vCPU for {region}, {market}, {family_type}")
            return True
    
    def release(self, region:str, market:str, family_type:str, vcpu:int):
        """Release vCPU quota"""
        with self.lock:
            old = self.get_used_vcpu(region, market, family_type)
            self.used_vcpu[(region, market, family_type)] = max(0, old - vcpu)
            logger.info(f"[QuotaTracker] Released {vcpu} vCPU for {region}, {market}, {family_type}")
    
    def get_family_for_instance(self, instance_type:str):
        """Get the family for the given instance example - g6e.xlarge -> G Family"""
        row = self.quota_df[self.quota_df["Instance_Type"] == instance_type]
        return row["Family_Type"].iloc[0]
    
    def reserve_for_instance(self, region:str, market:str, instance_type:str, num_instances:int):
        """Convenience: reserve by instance type (auto-looks up family + vCPU)."""
        row = self.quota_df[self.quota_df["Instance_Type"] == instance_type]
        if row.empty:
            raise ValueError(f"Instance type {instance_type} not found in quota CSV")
        family_type = row["Family_Type"].iloc[0]
        vcpu = row["vCPU"].iloc[0]
        return self.reserve(region, market, family_type, vcpu * num_instances)

    def release_for_instance(self, region: str, market: str, instance_type: str, num_instances: int):
        """Convenience: release by instance type (auto-looks up family + vCPU)."""
        row = self.quota_df[self.quota_df["Instance_Type"] == instance_type]
        if row.empty:
            return
        family_type = row["Family_Type"].iloc[0]
        vcpu = row["vCPU"].iloc[0]
        self.release(region, market, family_type, vcpu * num_instances)

    def status_summary(self) -> pd.DataFrame:
        """Get a human-readable summary of quota usage."""
        rows = []
        for (region, market, family), used in self.used_vcpu.items():
            baseline = self.get_baseline_quota(region, market, family)
            rows.append({
                "Region": region,
                "Market": market,
                "Family": family,
                "Baseline": baseline,
                "Used": used,
                "Available": baseline - used,
                "Usage %": f"{(used/baseline*100):.1f}%" if baseline > 0 else "N/A"
            })
        return pd.DataFrame(rows)

    # ---- SQLite-backed reservation persistence ----

    def _init_db(self):
        """Create reservations table and replay persisted state into memory."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reservations (
                    cluster_name TEXT PRIMARY KEY,
                    region TEXT NOT NULL,
                    market TEXT NOT NULL,
                    instance_type TEXT NOT NULL,
                    num_instances INTEGER NOT NULL,
                    family_type TEXT NOT NULL,
                    vcpu_reserved INTEGER NOT NULL,
                    reserved_at REAL NOT NULL
                )
            """)
            # Replay persisted reservations into in-memory counters
            rows = conn.execute("SELECT region, market, family_type, vcpu_reserved FROM reservations").fetchall()
            for region, market, family_type, vcpu in rows:
                key = (region, market, family_type)
                self.used_vcpu[key] = self.used_vcpu.get(key, 0) + vcpu
            if rows:
                logger.info(f"[QuotaTracker] Restored {len(rows)} reservations from {self.db_path}")

    def reserve_cluster(self, cluster_name: str, region: str, market: str,
                        instance_type: str, num_instances: int) -> bool:
        """Reserve quota for a cluster and persist to SQLite. Returns True on success."""
        row = self.quota_df[self.quota_df["Instance_Type"] == instance_type]
        if row.empty:
            logger.warning(f"[QuotaTracker] Instance type {instance_type} not found in CSV")
            return False
        family_type = row["Family_Type"].iloc[0]
        vcpu = int(row["vCPU"].iloc[0]) * num_instances

        if not self.reserve(region, market, family_type, vcpu):
            return False

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO reservations VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (cluster_name, region, market, instance_type, num_instances,
                 family_type, vcpu, time.time()),
            )
        logger.info(f"[QuotaTracker] Reserved cluster {cluster_name}: {instance_type} x{num_instances} in {region}/{market}")
        return True

    def release_cluster(self, cluster_name: str):
        """Release quota for a cluster and remove from SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT region, market, family_type, vcpu_reserved FROM reservations WHERE cluster_name = ?",
                (cluster_name,),
            ).fetchone()
            if row is None:
                return
            region, market, family_type, vcpu = row
            self.release(region, market, family_type, vcpu)
            conn.execute("DELETE FROM reservations WHERE cluster_name = ?", (cluster_name,))
        logger.info(f"[QuotaTracker] Released cluster {cluster_name}")

    def reconcile(self, live_clusters: set):
        """Remove reservations for clusters that no longer exist."""
        with sqlite3.connect(self.db_path) as conn:
            all_names = [r[0] for r in conn.execute("SELECT cluster_name FROM reservations").fetchall()]
        stale = [name for name in all_names if name not in live_clusters]
        for name in stale:
            logger.info(f"[QuotaTracker] Reconciling stale reservation: {name}")
            self.release_cluster(name)
        if stale:
            logger.info(f"[QuotaTracker] Reconciled {len(stale)} stale reservations")

    def get_reservations(self) -> list:
        """Return all active reservations as a list of dicts."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM reservations ORDER BY reserved_at").fetchall()
            return [dict(r) for r in rows]


@dataclass
class JobSpec:
    job_id: str
    model_name: str
    num_lines: int
    avg_input_tokens: int
    avg_output_tokens: int
    slo_hours: float
    job_type: str = "batch"
    region: str = "us-east-1"  
    market: str = "spot" 

@dataclass
class JobState:
    spec: JobSpec
    submitted_at: float # TODO: We should prob standardize this
    progress_frac: float = 0.0
    gpu_base: Optional[str] = None    # added
    tp: Optional[int] = None
    pp: Optional[int] = None
    replicas: Optional[int] = None
    allocated_gpus: Optional[int] = None  #added
    vcpu_needed: Optional[int] = None   # #added
    instance_types: Optional[str] = None  #added
    num_instances: Optional[int] = None            #added
    instance_ids: Optional[List[str]] = None  #added
    allocations: Optional[List[Tuple[str, int]]] = None  #added

    @property
    def deadline_ts(self) -> float:
        return self.submitted_at + self.spec.slo_hours * 3600.0

    @property
    def total_tokens(self) -> int:
        return self.spec.num_lines * (self.spec.avg_input_tokens + self.spec.avg_output_tokens)

    @property
    def remaining_tokens(self) -> int:
        return int((1.0 - self.progress_frac) * self.total_tokens)

@dataclass
class JobRecord:
    state: JobState
    status: Literal[
        "queued", "launching", "running", "succeeded", "failed", "cancelled"
    ] = "queued"
    endpoint_url: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)
    head_ip: Optional[str] = None
    output_s3_path: Optional[str] = None  # S3 path for output file download