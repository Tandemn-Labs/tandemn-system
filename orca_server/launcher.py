"""
SkyPilot cluster launch orchestration for vLLM batch and online jobs.
"""

import asyncio
import logging
import subprocess
import threading
import time
from threading import Event, Lock, Thread

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import sky
import yaml

from orca_server import config as _cfg
from orca_server.config import (
    HF_TOKEN,
    VLLM_PORT,
    YAML_OUTPUT,
)
from orca_server.koi_contract import ReasonCode, TERMINAL_PHASES
from orca_server.job_manager import (
    close_job_logger,
    download_output_from_s3,
    generate_job_dirname,
    get_cluster_manager,
    get_job_tracker,
    prefix_job_dirname,
    setup_job_logger,
    sky_down_with_retry,
)
from models.requests import BatchedRequest, OnlineServingRequest
from models.resources import MagicOutput
from quota.region_selector import (
    get_cached_quotas,
    get_instance_family,
    get_ordered_regions,
)
from orca_server.job_templates import (
    get_vllm_config_template,
    replace_run_vllm,
    replace_run_vllm_online,
)
from utils.utils import split_uri, update_template, update_yaml_file

# EFA-capable instance families (NVSwitch/NVLink, multi-NIC).
# network_tier=best enables EFA, DLAMI, NCCL auto-config on AWS.
EFA_PREFIXES = ("p3.", "p3dn.", "p4d.", "p4de.", "p5.", "p5e.")
_KOI_LAUNCH_HEARTBEAT_INTERVAL_SECONDS = 45.0
_koi_launch_heartbeats = {}
_koi_launch_heartbeat_lock = Lock()


def _requested_market(
    request: BatchedRequest, config: Optional[MagicOutput] = None
) -> Optional[str]:
    """Resolve the intended market for a launch attempt.

    planned_market is the exact market selected by Koi for this config.
    preferred_market is the caller's broader preference.
    prefer_spot is the legacy boolean fallback.
    """
    if config and getattr(config, "planned_market", None) in {"spot", "on_demand"}:
        return config.planned_market
    if getattr(request, "planned_market", None) in {"spot", "on_demand"}:
        return request.planned_market
    if getattr(request, "preferred_market", None) in {"spot", "on_demand"}:
        return request.preferred_market
    prefer_spot = getattr(request, "prefer_spot", None)
    if prefer_spot is None:
        return None
    return "spot" if prefer_spot else "on_demand"


def _needs_efa(instance_type: str) -> bool:
    """Check if instance type supports EFA high-performance networking."""
    return any(instance_type.startswith(p) for p in EFA_PREFIXES)


def _validate_parallelism_topology(config: MagicOutput) -> None:
    """Validate that TP stays intra-node and PP fits across the configured nodes."""
    instance_info = _cfg.AWS_INSTANCES.get(config.instance_type)
    if instance_info is None:
        logger.warning(
            "[Topology] Unknown instance type %s; skipping TP/PP validation",
            config.instance_type,
        )
        return

    _gpu_name, gpus_per_node, _vcpus, _vram_gb = instance_info
    if config.tp_size > gpus_per_node:
        raise ValueError(
            f"TP={config.tp_size} requires at least {config.tp_size} GPUs per node, "
            f"but {config.instance_type} only has {gpus_per_node}."
        )

    stages_per_node = max(1, gpus_per_node // config.tp_size)
    min_nodes = (config.pp_size + stages_per_node - 1) // stages_per_node
    if config.num_nodes < min_nodes:
        raise ValueError(
            f"PP={config.pp_size} requires {min_nodes} nodes but config has {config.num_nodes}. "
            f"Instance {config.instance_type} with TP={config.tp_size} fits "
            f"{stages_per_node} PP stages per node ({gpus_per_node} GPUs/node)."
        )


def _post_koi_webhook(
    path: str,
    payload: dict,
    event: str,
    *,
    dedup_key: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> None:
    """Deliver a webhook to Koi through the durable outbox.

    Primary path (production): enqueue into OutboxDB. The publisher thread
    drains it with retry/backoff, so Koi restarts and network blips don't
    lose events. Passing `dedup_key` collapses naturally-duplicated events
    at the source (e.g. watchdog + monitor_replica both detecting the same
    replica death → one row via INSERT OR IGNORE on the PK).

    Fallback path (dev / rollback): if the outbox singleton is absent
    (ORCA_OUTBOX_DB_PATH="" opt-out or startup hasn't initialized it yet),
    fall back to a direct best-effort POST.
    """
    try:
        from orca_server.config import KOI_SERVICE_URL

        if not KOI_SERVICE_URL:
            return

        # Primary: durable outbox.
        from orca_server.outbox import get_outbox

        outbox = get_outbox()
        if outbox is not None:
            job_id = (
                payload.get("replica_id")
                or payload.get("group_id")
                or payload.get("job_id")
                or "unknown"
            )
            event_type = event.replace("-", "_")
            outbox.enqueue(
                path,
                event_type,
                payload,
                job_id=job_id,
                dedup_key=dedup_key,
                correlation_id=correlation_id,
            )
            return

        # Fallback: outbox disabled, direct fire-and-forget.
        import requests as _req

        _req.post(f"{KOI_SERVICE_URL}{path}", json=payload, timeout=5)
    except Exception as exc:
        ident = (
            payload.get("replica_id")
            or payload.get("job_id")
            or payload.get("group_id")
            or "unknown"
        )
        logger.warning("[Koi] Failed to notify %s for %s: %s", event, ident, exc)


def _notify_koi_config_attempted(
    parent_job_id,
    koi_webhook_info,
    instance_type,
    region,
    market,
    launched,
    attempt_index,
    time_to_launch=0,
    failure_reason="",
):
    """Fire-and-forget: tell Koi about one allocation attempt."""
    from orca_server.config import INSTANCE_TO_GPU

    _post_koi_webhook(
        "/job/config-attempted",
        {
            "job_id": parent_job_id,
            "decision_id": koi_webhook_info.get("decision_id")
            if koi_webhook_info
            else None,
            "instance_type": instance_type,
            "gpu_type": INSTANCE_TO_GPU.get(instance_type, "unknown"),
            "region": region,
            "market": market,
            "launched": launched,
            "failure_reason": failure_reason,
            "time_to_launch": time_to_launch,
            "attempt_index": attempt_index,
        },
        "config-attempted",
    )


def _notify_koi_launch_heartbeat(
    *,
    replica_id,
    parent_job_id,
    koi_webhook_info,
    instance_type,
    tp,
    pp,
    attempt_index,
    phase,
    message,
    region="unknown",
    market="unknown",
):
    """Fire-and-forget: tell Koi a chunked launch is still alive."""
    from orca_server.config import INSTANCE_TO_GPU

    if not koi_webhook_info:
        return
    _post_koi_webhook(
        "/job/launch-heartbeat",
        {
            "job_id": replica_id,
            "decision_id": koi_webhook_info.get("decision_id"),
            "group_id": parent_job_id,
            "gpu_type": INSTANCE_TO_GPU.get(instance_type, "unknown"),
            "instance_type": instance_type,
            "tp": tp,
            "pp": pp,
            "region": region,
            "market": market,
            "attempt_index": attempt_index,
            "phase": phase,
            "message": message,
            "timestamp": time.time(),
        },
        "launch-heartbeat",
    )


def _run_koi_launch_heartbeat(replica_id: str) -> None:
    while True:
        with _koi_launch_heartbeat_lock:
            entry = _koi_launch_heartbeats.get(replica_id)
            if not entry:
                return
            stop_event = entry["stop_event"]
            payload = dict(entry["payload"])
        _notify_koi_launch_heartbeat(**payload)
        if stop_event.wait(_KOI_LAUNCH_HEARTBEAT_INTERVAL_SECONDS):
            return


def _start_koi_launch_heartbeat(
    *,
    replica_id,
    parent_job_id,
    koi_webhook_info,
    instance_type,
    tp,
    pp,
    attempt_index=0,
    phase="searching_capacity",
    message="Searching for capacity",
    region="unknown",
    market="unknown",
):
    """Start a lightweight launch heartbeat loop for one chunked replica."""
    from orca_server.config import KOI_SERVICE_URL

    if not KOI_SERVICE_URL or not koi_webhook_info:
        return False

    _stop_koi_launch_heartbeat(replica_id)
    stop_event = Event()
    thread = Thread(
        target=_run_koi_launch_heartbeat,
        args=(replica_id,),
        daemon=True,
        name=f"koi-heartbeat-{replica_id[:12]}",
    )
    payload = {
        "replica_id": replica_id,
        "parent_job_id": parent_job_id,
        "koi_webhook_info": koi_webhook_info,
        "instance_type": instance_type,
        "tp": tp,
        "pp": pp,
        "attempt_index": attempt_index,
        "phase": phase,
        "message": message,
        "region": region,
        "market": market,
    }
    with _koi_launch_heartbeat_lock:
        _koi_launch_heartbeats[replica_id] = {
            "stop_event": stop_event,
            "thread": thread,
            "payload": payload,
        }
    thread.start()
    return True


def _update_koi_launch_heartbeat(replica_id: str, **kwargs) -> bool:
    with _koi_launch_heartbeat_lock:
        entry = _koi_launch_heartbeats.get(replica_id)
        if not entry:
            return False
        for key, value in kwargs.items():
            if value is not None:
                entry["payload"][key] = value
    return True


def _stop_koi_launch_heartbeat(replica_id: str) -> bool:
    with _koi_launch_heartbeat_lock:
        entry = _koi_launch_heartbeats.pop(replica_id, None)
    if not entry:
        return False
    entry["stop_event"].set()
    thread = entry.get("thread")
    if thread and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=1)
    return True


async def sp_launch_vllm_batch_with_fallback(
    request: BatchedRequest,
    configs: List[MagicOutput],
    solver: str = "roofline",
    early_messages: list = None,
    quota_tracker=None,
    persist: bool = False,
    timeout_seconds: float = 300.0,  # 5-minute total timeout
) -> Tuple[bool, MagicOutput]:
    """Launch vLLM batch job with fallback to alternative instance types.

    Tries each config in order. If all fail or 5 minutes elapse, returns failure.
    Configs can come from roofline solver OR Koi alternatives.
    """
    if early_messages is None:
        early_messages = []

    try:
        return await asyncio.wait_for(
            _launch_with_fallback_inner(
                request,
                configs,
                solver,
                early_messages,
                quota_tracker,
                persist,
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.error(
            f"[Launch] Timeout after {timeout_seconds:.0f}s, tried {len(configs)} configs"
        )
        return (False, configs[0])


async def _launch_with_fallback_inner(
    request: BatchedRequest,
    configs: List[MagicOutput],
    solver: str,
    early_messages: list,
    quota_tracker,
    persist: bool,
) -> Tuple[bool, MagicOutput]:
    """Inner loop — tries each config in order until one succeeds."""
    for i, config in enumerate(configs):
        msg = f"[Launch] Trying config {i + 1}/{len(configs)}: {config.instance_type} TP={config.tp_size} PP={config.pp_size}"
        logger.info(msg)
        early_messages.append(("INFO", msg))

        try:
            await sp_launch_vllm_batch(
                request,
                config,
                solver,
                early_messages=early_messages,
                quota_tracker=quota_tracker,
                persist=persist,
            )
            logger.info(f"[Launch] Success with config {i + 1}: {config.instance_type}")
            return (True, config)

        except Exception as e:
            logger.warning(f"[Launch] Config {i + 1} failed: {e}")
            if i < len(configs) - 1:
                logger.info(f"[Launch] Trying alternative {i + 2}/{len(configs)}...")
                continue
            else:
                logger.error(f"[Launch] All {len(configs)} configs failed")
                return (False, configs[0])


async def sp_launch_vllm_batch(
    request: BatchedRequest,
    config: MagicOutput,
    solver: str = "roofline",
    early_messages: list = None,
    quota_tracker=None,
    persist: bool = False,
):
    # Generate informative job directory name
    job_dirname = generate_job_dirname(
        request, solver, config.tp_size, config.pp_size, config.instance_type
    )

    # Create output dir and job logger early
    output_dir = Path(f"outputs/{job_dirname}")
    output_dir.mkdir(parents=True, exist_ok=True)
    job_logger = setup_job_logger(config.decision_id, str(output_dir / "job.log"))

    # Flush early messages collected before logger existed
    if early_messages:
        for level, msg in early_messages:
            getattr(job_logger, level.lower(), job_logger.info)(msg)

    s3_base, _ = split_uri(request.input_file)
    # S3 output path: s3://bucket/base/job_dirname/output.jsonl
    s3_output_dir = f"{s3_base}/{job_dirname}"

    # Verify model exists in S3; fall back to HuggingFace if unavailable
    if request.s3_model_path:
        s3_model_path = request.s3_model_path.rstrip("/") + "/"
        s3_check = subprocess.run(
            ["aws", "s3", "ls", s3_model_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if s3_check.returncode != 0 or not s3_check.stdout.strip():
            job_logger.warning(
                f"[S3] Model not found at {s3_model_path}. "
                f"Falling back to HuggingFace download."
            )
            request = request.model_copy(update={"s3_model_path": None})
        else:
            job_logger.info(f"[S3] Verified model exists at {s3_model_path}")

    hf_token = request.hf_token or HF_TOKEN or ""
    _validate_parallelism_topology(config)
    num_nodes = config.num_nodes

    # Select per-config template or fall back to generic
    template_path = get_vllm_config_template(
        model_name=request.model_name,
        instance_type=config.instance_type,
        tp=config.tp_size,
        pp=config.pp_size,
        logger=job_logger,
    )

    # Get quota-aware ordered regions
    instance_family = get_instance_family(config.instance_type)
    quotas = get_cached_quotas(instance_family)
    ordered_regions = get_ordered_regions(
        instance_type=config.instance_type,
        num_nodes=num_nodes,
        quotas=quotas,
        prefer_spot=getattr(request, "prefer_spot", True),
    )

    # Build resources with any_of for fallback regions
    use_efa = _needs_efa(config.instance_type)
    if ordered_regions:
        any_of_resources = []
        for candidate in ordered_regions[:5]:
            res = {
                "region": candidate.region,
                "instance_type": config.instance_type,
                "use_spot": candidate.use_spot,
                "disk_size": "300GB",
                "ports": VLLM_PORT,
            }
            if use_efa:
                res["network_tier"] = "best"
            any_of_resources.append(res)
        job_logger.info(
            f"[RegionSelector] Trying regions: {[(c.region, 'spot' if c.use_spot else 'on-demand') for c in ordered_regions[:5]]}"
        )
        resources_config = {"any_of": any_of_resources}
    else:
        resources_config = {
            "infra": "aws",
            "instance_type": config.instance_type,
            "disk_size": "300GB",
            "ports": VLLM_PORT,
        }
        if use_efa:
            resources_config["network_tier"] = "best"

    # For per-config templates, substitute all placeholders
    if "vllm_configs" in template_path:
        # Build substitution dict (same as generic template)
        replace_dict = replace_run_vllm(request, config, job_dirname, logger=job_logger)

        template_content = Path(template_path).read_text()
        for key, value in replace_dict.items():
            template_content = template_content.replace("{" + key + "}", str(value))

        # Write to temp file and parse as yaml
        yaml_data = yaml.safe_load(template_content)

        # Preserve image_id from template if specified (e.g., custom AMI for A100)
        template_image_id = yaml_data.get("resources", {}).get("image_id")
        template_region = yaml_data.get("resources", {}).get("region")

        # If template has a specific image_id, use its region and don't do quota-based fallback
        if template_image_id:
            job_logger.info(
                f"[Template] Using custom AMI: {template_image_id} in {template_region}"
            )
            resources_config = {
                "cloud": "aws",
                "accelerators": yaml_data["resources"].get("accelerators", "A100:8"),
                "disk_size": yaml_data["resources"].get("disk_size", "300GB"),
                "ports": yaml_data["resources"].get("ports", VLLM_PORT),
                "image_id": template_image_id,
                "region": template_region,
            }

        # Update dynamic fields
        yaml_data["name"] = config.decision_id
        yaml_data["num_nodes"] = num_nodes
        yaml_data["resources"] = resources_config
        yaml_data["file_mounts"]["/data"]["source"] = s3_base
        yaml_data["envs"]["HF_TOKEN"] = hf_token
        yaml_data["envs"]["TD_SERVER_URL"] = _cfg.TD_SERVER_URL
        yaml_data["envs"]["JOB_ID"] = config.decision_id
        yaml_data["envs"]["ORCA_API_KEY"] = _cfg.ORCA_API_KEY

        # Add S3 model weight mount if requested
        if request.s3_model_path:
            s3_src = request.s3_model_path
            s3_src = s3_src.rstrip("/")
            model_mount_path = f"/models/{request.model_name}"
            yaml_data["file_mounts"][model_mount_path] = {
                "source": s3_src,
                "mode": "COPY",
            }

        # Write final yaml
        Path(YAML_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
        with open(YAML_OUTPUT, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    else:
        # Generic template - use old substitution method
        replace_run_dict = replace_run_vllm(
            request, config, job_dirname, logger=job_logger
        )
        run_string = update_template("templates/vllm_run", replace_run_dict)

        replace_yaml = {
            "name": config.decision_id,
            "num_nodes": num_nodes,
            "resources": resources_config,
            "run": run_string,
            "file_mounts./data.source": s3_base,
            "envs.HF_TOKEN": hf_token,
            "envs.TD_SERVER_URL": _cfg.TD_SERVER_URL,
            "envs.JOB_ID": config.decision_id,
            "envs.ORCA_API_KEY": _cfg.ORCA_API_KEY,
            "envs.VLLM_USE_V1": "1"
            if _cfg.supports_vllm_v1(config.instance_type)
            else "0",
        }

        # Add S3 model weight mount if requested
        if request.s3_model_path:
            s3_src = request.s3_model_path
            s3_src = s3_src.rstrip("/")
            model_mount_path = f"/models/{request.model_name}"
            replace_yaml[f"file_mounts.{model_mount_path}.source"] = s3_src
            replace_yaml[f"file_mounts.{model_mount_path}.mode"] = "COPY"

        update_yaml_file("templates/vllm.yaml", replace_yaml, YAML_OUTPUT)

    cm = get_cluster_manager()

    # Register job in tracker
    jt = get_job_tracker()
    job_state = jt.build_job_state_batched(request, config)
    jt.add(job_state)
    jt.update_status(config.decision_id, "launching")
    # Mark as chunked if this is a chunked deployment (enables watchdog heartbeat monitoring)
    if getattr(request, "chunks", None):
        jt.set_chunked_info(config.decision_id, len(request.chunks), 1)

    # Construct S3 output path for later download
    output_s3_path = f"{s3_output_dir}/{request.output_file}"
    job_logger.info(f"[Job] Output will be saved to: {s3_output_dir}/")

    def monitor_and_download(job_id, *, actual_region, actual_market, solver):
        """Background thread: stream logs, then download output when done."""
        try:
            sky.tail_logs(cluster_name=config.decision_id, job_id=job_id, follow=True)
            job_logger.info(
                f"[Job] {config.decision_id} completed. Downloading output..."
            )

            # Download output from S3 to local dir (base name, no prefix yet)
            local_path = download_output_from_s3(
                output_s3_path, job_dirname, logger=job_logger
            )

            # Determine success: both output file and metrics.csv must exist
            base_dir = Path(f"outputs/{job_dirname}")
            is_success = (
                local_path is not None
                and base_dir.exists()
                and (base_dir / "metrics.csv").exists()
            )

            # Update job tracker
            get_job_tracker().update_progress(config.decision_id, 1.0)
            get_job_tracker().update_status(
                config.decision_id, "succeeded" if is_success else "failed"
            )

            if is_success:
                from orca_server.monitoring import get_metrics_collector
                from orca_server.metrics_db import get_metrics_db

                last_snap = get_metrics_collector().get_latest(config.decision_id)
                db = get_metrics_db()
                try:
                    db.push_run(
                        job_id=config.decision_id,
                        metrics_csv_path=str(base_dir / "metrics.csv"),
                        actual_region=actual_region,
                        actual_market=actual_market,
                        solver=solver,
                        job_dirname=job_dirname,
                        last_snapshot=last_snap,
                    )
                except Exception as db_err:
                    job_logger.warning(f"[MetricsDB] Push failed: {db_err}")

                # Export timeseries to local experiment directory
                try:
                    import csv as _csv

                    ts_data = db.get_timeseries(config.decision_id)
                    if ts_data:
                        ts_path = base_dir / "timeseries.csv"
                        all_keys = list(ts_data[0].keys())
                        with open(ts_path, "w", newline="") as tf:
                            writer = _csv.DictWriter(
                                tf, fieldnames=all_keys, extrasaction="ignore"
                            )
                            writer.writeheader()
                            for row in ts_data:
                                writer.writerow(row)
                        job_logger.info(
                            f"[Job] Saved timeseries.csv ({len(ts_data)} samples) to {ts_path}"
                        )

                        # Generate timeseries PDF
                        try:
                            from orca_server.plot_timeseries import (
                                plot_timeseries as _plot_ts,
                            )

                            pdf_path = base_dir / "timeseries.pdf"
                            _metrics_csv = str(base_dir / "metrics.csv")
                            _metrics_arg = (
                                _metrics_csv
                                if (base_dir / "metrics.csv").exists()
                                else None
                            )
                            _plot_ts(
                                str(ts_path),
                                str(pdf_path),
                                metrics_csv_path=_metrics_arg,
                            )
                            job_logger.info(
                                f"[Job] Generated timeseries.pdf at {pdf_path}"
                            )
                        except Exception as pe:
                            job_logger.warning(f"[Job] Timeseries plot failed: {pe}")
                except Exception as te:
                    job_logger.warning(f"[Job] Timeseries export failed: {te}")

            # Rename dir with success-/failed- prefix
            status = "success" if is_success else "failed"
            prefixed_dirname = prefix_job_dirname(job_dirname, status)
            target_dir = Path(f"outputs/{prefixed_dirname}")
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            job_logger.info(f"[Job] {status.upper()}: outputs/{prefixed_dirname}")
            close_job_logger(job_logger)
            base_dir.rename(target_dir)

        except Exception as e:
            job_logger.error(f"[Job] Error in monitor thread: {e}")
            get_job_tracker().update_status(config.decision_id, "failed")
            # Ensure a failed- dir exists even if everything blew up
            failed_dirname = prefix_job_dirname(job_dirname, "failed")
            failed_dir = Path(f"outputs/{failed_dirname}")
            base_dir = Path(f"outputs/{job_dirname}")
            job_logger.info(f"[Job] FAILED: outputs/{failed_dirname}")
            close_job_logger(job_logger)
            if base_dir.exists() and not failed_dir.exists():
                failed_dir.parent.mkdir(parents=True, exist_ok=True)
                base_dir.rename(failed_dir)
            elif not failed_dir.exists():
                failed_dir.mkdir(parents=True, exist_ok=True)
        finally:
            from orca_server.monitoring import get_metrics_collector

            get_metrics_collector().stop_collecting(config.decision_id)
            if quota_tracker is not None:
                quota_tracker.release_cluster(config.decision_id)
            cm.unregister(config.decision_id)
            cm.unregister_thread(config.decision_id)
            if not persist:
                sky_down_with_retry(config.decision_id)
            else:
                job_logger.info(
                    f"[Teardown] --persist: keeping cluster {config.decision_id} alive"
                )

    try:
        job_logger.info(f"[SkyPilot] Launching cluster {config.decision_id}...")
        task = sky.Task.from_yaml(YAML_OUTPUT)
        result_id = sky.launch(task, cluster_name=config.decision_id, down=True)
        job_id, handle = sky.stream_and_get(result_id, follow=True)
        job_logger.info(f"[SkyPilot] Launch complete. job_id={job_id}")

        # Extract actual region/market from SkyPilot handle and reserve quota
        actual_region = handle.launched_resources.region
        actual_market = "spot" if handle.launched_resources.use_spot else "on_demand"
        if quota_tracker is not None:
            quota_tracker.reserve_cluster(
                cluster_name=config.decision_id,
                region=actual_region,
                market=actual_market,
                instance_type=config.instance_type,
                num_instances=num_nodes,
            )
        cm.register(
            config.decision_id,
            config.decision_id,
            region=actual_region,
            market=actual_market,
            instance_type=config.instance_type,
            num_instances=num_nodes,
        )
        job_logger.info(
            f"[Quota] Reserved {config.instance_type} x{num_nodes} in {actual_region}/{actual_market}"
        )

        # Update job tracker: running + head IP
        jt.update_status(config.decision_id, "running")
        head_ip = getattr(handle, "head_ip", None)
        if head_ip:
            jt.set_head_ip(config.decision_id, head_ip)
            from orca_server.monitoring import get_metrics_collector

            endpoint_url = f"http://{head_ip}:{VLLM_PORT}"
            jt.set_endpoint_url(config.decision_id, endpoint_url)
            get_metrics_collector().start_collecting(config.decision_id, endpoint_url)

        # Stream logs in background and download when done
        t = threading.Thread(
            target=monitor_and_download,
            args=(job_id,),
            kwargs=dict(
                actual_region=actual_region, actual_market=actual_market, solver=solver
            ),
            daemon=False,
            name=f"orca-monitor-{config.decision_id[:12]}",
        )
        cm.register_thread(config.decision_id, t)
        if persist:
            cm.mark_persist(config.decision_id)
        t.start()

    except Exception as e:
        job_logger.error(
            f"[SkyPilot] Failed to launch cluster {config.decision_id}: {e}"
        )
        jt.update_status(config.decision_id, "failed")
        close_job_logger(job_logger)
        raise Exception(f"Failed to launch cluster {config.decision_id}: {e}")


async def launch_chunked_replicas(
    request: BatchedRequest,
    configs: List[MagicOutput],
    num_replicas: int,
    solver: str = "roofline",
    early_messages: list = None,
    quota_tracker=None,
    persist: bool = False,
) -> bool:
    """Launch N independent SkyPilot clusters for a chunked job.

    Each replica is an independent cluster that pulls chunks from the Redis queue
    via HTTP endpoints on the control plane.

    Each replica tries configs in order (fallback). Different replicas may end up
    on different instance types if the primary has no capacity.
    """
    if not _cfg.TD_SERVER_URL:
        raise ValueError(
            "TD_SERVER_URL is not set. Chunked replicas need the control plane URL "
            "to pull chunks. Start the server with --url or set the "
            "TD_SERVER_URL environment variable."
        )

    if early_messages is None:
        early_messages = []

    primary = configs[0]

    from orca_server.job_manager import (
        setup_job_logger,
        generate_job_dirname,
    )
    from pathlib import Path

    job_dirname = generate_job_dirname(
        request, solver, primary.tp_size, primary.pp_size, primary.instance_type
    )
    output_dir = Path(f"outputs/{job_dirname}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "replicas").mkdir(exist_ok=True)
    job_logger = setup_job_logger(primary.decision_id, str(output_dir / "job.log"))

    if early_messages:
        for level, msg in early_messages:
            getattr(job_logger, level.lower(), job_logger.info)(msg)

    jt = get_job_tracker()
    job_state = jt.build_job_state_batched(request, primary)
    jt.add(job_state)
    jt.update_status(primary.decision_id, "launching")

    # Store job_dirname on the record for assembly
    rec = jt.get(primary.decision_id)
    if rec:
        rec._job_dirname = job_dirname

    cm = get_cluster_manager()
    parent_job_id = primary.decision_id

    # Koi webhook info — job-level fields shared by all replicas.
    # Config-specific fields (gpu_type, instance_type, tp, pp) are added
    # by _launch_chunked_replica based on whichever config actually succeeds.
    total_tokens = (request.num_lines or 0) * (
        (request.avg_input_tokens or 0) + (request.avg_output_tokens or 0)
    )
    koi_webhook_info = {
        "decision_id": request.koi_decision_id,
        "group_id": parent_job_id,
        "slo_deadline_hours": request.slo_deadline_hours or 8.0,
        "total_tokens": total_tokens,
        "deploy_timestamp": time.time(),
        "predicted_tps": getattr(request, "koi_predicted_tps", 0),
    }

    # Accumulator for /job/launch-failed webhook (shared across replica threads)
    _attempt_log = []
    _attempt_log_lock = threading.Lock()
    _launch_start_time = time.time()
    _launch_failed_fired = threading.Event()

    # Launch all replicas in background threads so the server stays responsive.
    # Each replica launch takes 5-15 min (SkyPilot provision + setup).
    # If we did this inline, the event loop would be blocked and chunk pull
    # requests from already-running replicas would timeout.

    # Pre-register all replicas as "launching"
    for i in range(num_replicas):
        rid = f"{parent_job_id}-r{i}"
        cm.set_replica_state(
            parent_job_id, rid, phase="launching", launched_at=time.time()
        )
        cm.register_for_job(parent_job_id, rid)

    def _launch_replica_thread(i):
        """Try each config in order until one succeeds (per-replica fallback)."""
        import asyncio as _aio

        replica_id = f"{parent_job_id}-r{i}"

        for j, cfg in enumerate(configs):
            replica_config = cfg.model_copy(
                update={
                    "decision_id": replica_id,
                    "replicas": 1,
                    "num_instances": cfg.num_nodes,
                }
            )
            cm.set_replica_state(
                parent_job_id,
                replica_id,
                phase="launching",
                launched_at=time.time(),
                instance_type=cfg.instance_type,
                tp=cfg.tp_size,
                pp=cfg.pp_size,
                num_instances=cfg.num_nodes,
                config_index=j,
            )
            job_logger.info(
                f"[Chunked] Replica {i} trying config {j + 1}/{len(configs)}: "
                f"{cfg.instance_type} TP={cfg.tp_size} PP={cfg.pp_size}"
            )
            try:
                loop = _aio.new_event_loop()
                loop.run_until_complete(
                    _launch_chunked_replica(
                        request,
                        replica_config,
                        replica_id,
                        parent_job_id=parent_job_id,
                        job_dirname=job_dirname,
                        job_logger=job_logger,
                        quota_tracker=quota_tracker,
                        persist=persist,
                        koi_webhook_info=koi_webhook_info,
                        config_index=j,
                    )
                )
                loop.close()
                return  # success — stop trying alternatives
            except Exception as e:
                requested_market = _requested_market(request, cfg) or "unknown"
                _stop_koi_launch_heartbeat(replica_id)
                # Notify Koi: this config attempt failed
                _notify_koi_config_attempted(
                    parent_job_id,
                    koi_webhook_info,
                    cfg.instance_type,
                    region="unknown",
                    market=requested_market,
                    launched=False,
                    attempt_index=j,
                    failure_reason=str(e)[:500],
                )
                with _attempt_log_lock:
                    _attempt_log.append(
                        {
                            "gpu_type": _cfg.INSTANCE_TO_GPU.get(
                                cfg.instance_type, "unknown"
                            ),
                            "instance_type": cfg.instance_type,
                            "region": "unknown",
                            "market": requested_market,
                            "reason": str(e)[:500],
                        }
                    )
                if j < len(configs) - 1:
                    job_logger.warning(
                        f"[Chunked] Replica {i} config {j + 1} failed: {e}. Trying next..."
                    )
                    # Clean up partial cluster state before retrying
                    try:
                        sky_down_with_retry(replica_id)
                    except Exception:
                        pass
                else:
                    job_logger.error(
                        f"[Chunked] Replica {i} all {len(configs)} configs failed: {e}"
                    )
                    cm.set_replica_state(parent_job_id, replica_id, phase="failed")
                    # Check if ALL replicas failed at launch → fire /job/launch-failed
                    states = cm.get_replica_states(parent_job_id)
                    if states and all(
                        s.get("phase") in TERMINAL_PHASES for s in states.values()
                    ):
                        any_ok = any(
                            s.get("phase") == "completed" for s in states.values()
                        )
                        if not any_ok and not _launch_failed_fired.is_set():
                            _launch_failed_fired.set()
                            jt_tmp = get_job_tracker()
                            rec_tmp = jt_tmp.get(parent_job_id)
                            if rec_tmp and rec_tmp.status in (
                                "launching",
                                "loading_model",
                            ):
                                jt_tmp.update_status(parent_job_id, "failed")
                                job_logger.warning(
                                    "[Chunked] All replicas failed at launch — marking job as failed"
                                )
                            with _attempt_log_lock:
                                attempts = list(_attempt_log)
                            _post_koi_webhook(
                                "/job/launch-failed",
                                {
                                    "job_id": parent_job_id,
                                    "decision_id": request.koi_decision_id,
                                    "configs_tried": attempts,
                                    "failure_reasons": [a["reason"] for a in attempts],
                                    "total_time_seconds": round(
                                        time.time() - _launch_start_time, 2
                                    ),
                                },
                                "launch-failed",
                                dedup_key=f"launch_failed:{parent_job_id}",
                            )

    for i in range(num_replicas):
        t = threading.Thread(
            target=_launch_replica_thread,
            args=(i,),
            daemon=False,
            name=f"orca-launch-r{i}",
        )
        t.start()

    # Track parent job (individual replicas register themselves on launch)
    cm.register(
        parent_job_id,
        parent_job_id,
        instance_type=primary.instance_type,
        num_instances=num_replicas * primary.num_nodes,
    )
    return True


async def _launch_chunked_replica(
    request: BatchedRequest,
    config: MagicOutput,
    replica_id: str,
    parent_job_id: str,
    job_dirname: str,
    job_logger=None,
    quota_tracker=None,
    persist: bool = False,
    config_index: int = 0,
    koi_webhook_info: dict = None,
):
    """Launch a single replica cluster for chunked batch inference."""
    _launch_start = time.time()
    from pathlib import Path

    hf_token = request.hf_token or HF_TOKEN or ""
    _validate_parallelism_topology(config)
    num_nodes = config.num_nodes

    # Get quota-aware ordered regions
    instance_family = get_instance_family(config.instance_type)
    quotas = get_cached_quotas(instance_family)
    requested_market = _requested_market(request, config)
    prefer_spot = (
        requested_market == "spot"
        if requested_market is not None
        else getattr(request, "prefer_spot", True)
    )
    ordered_regions = get_ordered_regions(
        instance_type=config.instance_type,
        num_nodes=num_nodes,
        quotas=quotas,
        prefer_spot=prefer_spot,
        target_market=requested_market,
    )

    use_efa = _needs_efa(config.instance_type)
    if ordered_regions:
        any_of_resources = []
        for candidate in ordered_regions[:5]:
            res = {
                "region": candidate.region,
                "instance_type": config.instance_type,
                "use_spot": candidate.use_spot,
                "disk_size": "300GB",
                "ports": VLLM_PORT,
            }
            if use_efa:
                res["network_tier"] = "best"
            any_of_resources.append(res)
        resources_config = {"any_of": any_of_resources}
    else:
        if requested_market is not None:
            raise RuntimeError(
                f"No {requested_market} quota candidates for {config.instance_type}"
            )
        resources_config = {
            "infra": "aws",
            "instance_type": config.instance_type,
            "disk_size": "300GB",
            "ports": VLLM_PORT,
        }
        if use_efa:
            resources_config["network_tier"] = "best"

    # Build YAML using the chunked runner template
    replace_run_dict = replace_run_vllm(request, config, job_dirname, logger=job_logger)
    run_string = update_template("templates/vllm_run_chunked", replace_run_dict)

    yaml_output = f"temp/output_{replica_id}.yaml"
    replace_yaml = {
        "name": replica_id,
        "num_nodes": num_nodes,
        "resources": resources_config,
        "run": run_string,
        "envs.HF_TOKEN": hf_token,
        "envs.TD_SERVER_URL": _cfg.TD_SERVER_URL,
        "envs.JOB_ID": parent_job_id,
        "envs.REPLICA_ID": replica_id,
        "envs.ORCA_API_KEY": _cfg.ORCA_API_KEY,
        "envs.AVG_INPUT_TOKENS": str(request.avg_input_tokens or 2000),
        "envs.AVG_OUTPUT_TOKENS": str(request.avg_output_tokens or 1024),
        "envs.VLLM_USE_V1": "1" if _cfg.supports_vllm_v1(config.instance_type) else "0",
    }

    # Add S3 model weight mount if requested
    if request.s3_model_path:
        s3_src = request.s3_model_path
        s3_src = s3_src.rstrip("/")
        model_mount_path = f"/models/{request.model_name}"
        replace_yaml[f"file_mounts.{model_mount_path}.source"] = s3_src
        replace_yaml[f"file_mounts.{model_mount_path}.mode"] = "COPY"

    update_yaml_file("templates/vllm_chunked.yaml", replace_yaml, yaml_output)

    _start_koi_launch_heartbeat(
        replica_id=replica_id,
        parent_job_id=parent_job_id,
        koi_webhook_info=koi_webhook_info,
        instance_type=config.instance_type,
        tp=config.tp_size,
        pp=config.pp_size,
        attempt_index=config_index,
        phase="searching_capacity",
        message="Searching for capacity",
    )

    try:
        task = sky.Task.from_yaml(yaml_output)
        result_id = sky.launch(task, cluster_name=replica_id, down=True)
        _update_koi_launch_heartbeat(
            replica_id,
            phase="provisioning",
            message="Provisioning cluster",
        )
        job_id_sky, handle = sky.stream_and_get(result_id, follow=True)

        actual_region = handle.launched_resources.region
        actual_market = "spot" if handle.launched_resources.use_spot else "on_demand"
        _update_koi_launch_heartbeat(
            replica_id,
            phase="bootstrapping",
            message="Cluster provisioned, bootstrapping runtime",
            region=actual_region,
            market=actual_market,
        )
        if quota_tracker is not None:
            quota_tracker.reserve_cluster(
                cluster_name=replica_id,
                region=actual_region,
                market=actual_market,
                instance_type=config.instance_type,
                num_instances=num_nodes,
            )

        cm = get_cluster_manager()
        current_state = cm.get_replica_states(parent_job_id).get(replica_id, {})
        cm.register(
            replica_id,
            parent_job_id,
            region=actual_region,
            market=actual_market,
            instance_type=config.instance_type,
            num_instances=num_nodes,
        )
        cm.set_replica_state(
            parent_job_id,
            replica_id,
            phase="provisioned",
            region=actual_region,
            market=actual_market,
            instance_type=config.instance_type,
            launched_at=current_state.get("launched_at", _launch_start),
            num_instances=num_nodes,
            koi_webhook_info=koi_webhook_info,
            tp=config.tp_size,
            pp=config.pp_size,
            config_index=config_index,
        )

        # Notify Koi: this config attempt succeeded
        _notify_koi_config_attempted(
            parent_job_id,
            koi_webhook_info,
            config.instance_type,
            region=actual_region,
            market=actual_market,
            launched=True,
            attempt_index=config_index,
            time_to_launch=round(time.time() - _launch_start, 2),
        )

        _update_koi_launch_heartbeat(
            replica_id,
            phase="waiting_model_ready",
            message="Replica provisioned, waiting for model_ready",
            region=actual_region,
            market=actual_market,
        )

        # Notify Koi: replica is provisioned (pre-model_ready visibility)
        from orca_server.config import INSTANCE_TO_GPU

        _post_koi_webhook(
            "/job/launching",
            {
                "job_id": replica_id,
                "decision_id": koi_webhook_info.get("decision_id")
                if koi_webhook_info
                else None,
                "group_id": parent_job_id,
                "gpu_type": INSTANCE_TO_GPU.get(config.instance_type, "unknown"),
                "instance_type": config.instance_type,
                "tp": config.tp_size,
                "pp": config.pp_size,
                "region": actual_region,
                "market": actual_market,
                "attempt_index": config_index,
            },
            "launching",
            dedup_key=f"job_launching:{replica_id}:{config_index}",
        )
    except Exception:
        _stop_koi_launch_heartbeat(replica_id)
        raise

    # NOTE: Koi webhook (/job/started) is now fired from server.py when the
    # in-cluster runner reports "model_ready" phase, NOT here. This ensures
    # Koi only starts monitoring after vLLM is actually serving.

    def monitor_replica(sky_job_id):
        """Background thread: stream replica logs, tear down when done.

        Phase stays 'provisioned' until the sidecar ingest endpoint receives
        first metrics — that's the real proof vLLM is up and serving.
        The ingest endpoint (server.py) transitions to 'running' with running_since.

        Guarantee-fire invariant: on ANY exit path where the replica is not
        in a terminal phase and no /job/replica-failed has been enqueued,
        the finally block emits one with reason_code=MONITOR_THREAD_EXITED.
        Dedup via replica_failed:{replica_id} collapses any race with the
        inline emits below or with watchdog detection.
        """
        # Tracks whether we already enqueued a failure on one of the
        # explicit paths below. The guarantee-fire fallback in the finally
        # block checks this to avoid a bogus double-send (the outbox would
        # dedup it anyway, but clean code > relying on the safety net).
        replica_failed_enqueued = {"flag": False}

        def _emit_replica_failed(reason_code: ReasonCode, detail: str) -> None:
            state = cm.get_replica_states(parent_job_id).get(replica_id, {})
            info = state.get("koi_webhook_info") or {}
            _post_koi_webhook(
                "/job/replica-failed",
                {
                    "job_id": replica_id,
                    "replica_id": replica_id,
                    "group_id": parent_job_id,
                    "decision_id": info.get("decision_id"),
                    "instance_type": state.get("instance_type", "unknown"),
                    "region": state.get("region", "unknown"),
                    "market": state.get("market", "unknown"),
                    "status": "failed",
                    "reason": detail[:200],
                    "reason_code": reason_code.value,
                    "reason_detail": detail[:500],
                },
                "replica-failed",
                dedup_key=f"replica_failed:{replica_id}",
            )
            replica_failed_enqueued["flag"] = True

        try:
            sky.tail_logs(cluster_name=replica_id, job_id=sky_job_id, follow=True)
            # Verify this is a real completion — not a killed instance
            try:
                from orca_server.chunk_manager import get_chunk_manager as _gcm

                _progress = _gcm().get_progress(parent_job_id)
            except Exception:
                _progress = None
            if _progress and not _progress.get("all_done", False):
                # Chunks still pending → replica was killed mid-job, not completed
                if job_logger:
                    job_logger.warning(
                        f"[Chunked] Replica {replica_id} exited but chunks still pending — treating as failure"
                    )
                cm.set_replica_state(parent_job_id, replica_id, phase="failed")
                _emit_replica_failed(
                    ReasonCode.CLEAN_EXIT_PENDING_CHUNKS,
                    "Clean exit with pending chunks (likely killed)",
                )
            else:
                if job_logger:
                    job_logger.info(f"[Chunked] Replica {replica_id} completed")
                cm.set_replica_state(parent_job_id, replica_id, phase="completed")
        except Exception as e:
            # sky.tail_logs raises when the log stream is cancelled by another
            # process (e.g. assembly, SkyPilot internal).  This is NOT a replica
            # failure — the vLLM process may still be running or already done.
            current = cm.get_replica_states(parent_job_id).get(replica_id, {})
            cur_phase = current.get("phase", "")
            err_str = str(e)
            is_log_cancel = (
                "cancelled" in err_str.lower() or "cancel" in err_str.lower()
            )
            jt_tmp = get_job_tracker()
            rec_tmp = jt_tmp.get(parent_job_id)
            job_done = rec_tmp and rec_tmp.status in ("succeeded", "failed")

            if cur_phase in TERMINAL_PHASES:
                if job_logger:
                    job_logger.info(
                        f"[Chunked] Replica {replica_id} log stream ended (phase={cur_phase})"
                    )
            elif is_log_cancel or job_done:
                # Log cancellation or job already finished — not a real failure
                if job_logger:
                    job_logger.info(
                        f"[Chunked] Replica {replica_id} log stream cancelled (job {'done' if job_done else 'active'}): {e}"
                    )
            else:
                if job_logger:
                    job_logger.error(f"[Chunked] Replica {replica_id} error: {e}")
                cm.set_replica_state(parent_job_id, replica_id, phase="failed")
                _emit_replica_failed(ReasonCode.LOG_STREAM_ERROR, str(e))
        finally:
            _stop_koi_launch_heartbeat(replica_id)
            if quota_tracker is not None:
                quota_tracker.release_cluster(replica_id)
            cm.unregister(replica_id)
            cm.unregister_thread(replica_id)
            if not persist:
                sky_down_with_retry(replica_id)

            # GUARANTEED-FIRE FALLBACK: if the thread is exiting, no
            # /job/replica-failed has been enqueued, AND the replica isn't
            # in a terminal phase, something fell through the explicit
            # branches above (new exit path, programming bug, etc.). Emit
            # a fallback event so Koi doesn't hang waiting for a replica
            # that's effectively gone. Dedup_key collapses any race with
            # watchdog detection.
            try:
                if not replica_failed_enqueued["flag"]:
                    state = cm.get_replica_states(parent_job_id).get(replica_id, {})
                    phase = state.get("phase", "")
                    if phase not in TERMINAL_PHASES:
                        _emit_replica_failed(
                            ReasonCode.MONITOR_THREAD_EXITED,
                            (
                                "Monitor thread exited without an explicit "
                                f"failure emit and phase was {phase!r}"
                            ),
                        )
            except Exception as fb_exc:
                logger.error(
                    "[Chunked] Fallback replica-failed emission failed for %s: %s",
                    replica_id,
                    fb_exc,
                )

            # Check if ALL replicas are terminal — if so, mark the job failed
            # (prevents orphaned jobs stuck in "launching" when setup fails)
            jt = get_job_tracker()
            rec = jt.get(parent_job_id)
            if rec and rec.status in ("launching", "loading_model"):
                states = cm.get_replica_states(parent_job_id)
                if states and all(
                    s.get("phase") in TERMINAL_PHASES for s in states.values()
                ):
                    any_success = any(
                        s.get("phase") == "completed" for s in states.values()
                    )
                    if not any_success:
                        jt.update_status(parent_job_id, "failed")
                        if job_logger:
                            job_logger.warning(
                                "[Chunked] All replicas failed — marking job as failed"
                            )

    t = threading.Thread(
        target=monitor_replica,
        args=(job_id_sky,),
        daemon=False,
        name=f"orca-replica-{replica_id[:12]}",
    )
    cm.register_thread(replica_id, t)
    t.start()


async def sp_launch_vllm_online(request: OnlineServingRequest, config: MagicOutput):
    """Launch persistent vllm online deployment"""
    _validate_parallelism_topology(config)
    replace_run_dict = replace_run_vllm_online(request, config)
    run_string = update_template("templates/vllm_run_online", replace_run_dict)

    replace_yaml = {
        "name": config.decision_id,
        "num_nodes": config.num_nodes,
        "resources.instance_type": config.instance_type,
        "resources.ports": str(VLLM_PORT),
        "run": run_string,
    }
    update_yaml_file("templates/vllm_online.yaml", replace_yaml, YAML_OUTPUT)
    task = sky.Task.from_yaml(YAML_OUTPUT)
    result_id = sky.launch(
        task, cluster_name=config.decision_id, down=False
    )  # do not LET IT DIE!

    # return the public IP of the deployment
    job_id, handle = sky.stream_and_get(result_id, follow=True)
    sky.tail_logs(cluster_name=config.decision_id, job_id=job_id, follow=True)

    public_ip = handle.head_ip

    endpoint_url = f"http://{public_ip}:{VLLM_PORT}"

    logger.info(f"vLLM server launched at {endpoint_url}")

    url = f"http://{public_ip}:{VLLM_PORT}/v1/models"
    response = requests.get(url, timeout=5)
    if (
        response.status_code == 200
    ):  # do sth here for a valid API up and sth else otherwise
        logger.info(f"vLLM server API is up at {endpoint_url}")
        return endpoint_url
    else:
        raise Exception(f"vLLM server API is not up at {endpoint_url}")
