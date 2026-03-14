"""
SkyPilot cluster launch orchestration for vLLM batch and online jobs.
"""

import subprocess
import threading
from pathlib import Path
from typing import List, Tuple

import requests
import sky
import yaml

from orca_server.config import (
    HF_TOKEN,
    S3_MODEL_BUCKET,
    S3_MODEL_PREFIX,
    VLLM_PORT,
    YAML_OUTPUT,
)
from orca_server.job_manager import (
    close_job_logger,
    download_output_from_s3,
    generate_job_dirname,
    get_cluster_manager,
    get_job_tracker,
    prefix_job_dirname,
    setup_job_logger,
)
from models.requests import BatchedRequest, OnlineServingRequest
from models.resources import MagicOutput
from quota.region_selector import get_cached_quotas, get_instance_family, get_ordered_regions
from orca_server.job_templates import get_vllm_config_template, replace_run_vllm, replace_run_vllm_online
from utils.utils import split_uri, update_template, update_yaml_file


async def sp_launch_vllm_batch_with_fallback(
    request: BatchedRequest,
    configs: List[MagicOutput],
    solver: str = "roofline",
    early_messages: list = None,
    quota_tracker=None,
) -> Tuple[bool, MagicOutput]:
    """Launch vLLM batch job with fallback to alternative instance types."""
    if early_messages is None:
        early_messages = []

    for i, config in enumerate(configs):
        msg = f"[Launch] Trying config {i + 1}/{len(configs)}: {config.instance_type} TP={config.tp_size} PP={config.pp_size}"
        early_messages.append(("INFO", msg))

        try:
            await sp_launch_vllm_batch(
                request, config, solver, early_messages=early_messages,
                quota_tracker=quota_tracker,
            )
            print(f"[Launch] Success with config {i + 1}: {config.instance_type}")
            return (True, config)

        except Exception as e:
            print(f"[Launch] Config {i + 1} failed: {e}")
            if i < len(configs) - 1:
                print("[Launch] Trying next instance type...")
                continue
            else:
                print(f"[Launch] All {len(configs)} configs failed")
                return (False, configs[0])


async def sp_launch_vllm_batch(
    request: BatchedRequest,
    config: MagicOutput,
    solver: str = "roofline",
    early_messages: list = None,
    quota_tracker=None,
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
    if request.s3_models:
        s3_model_path = (
            f"s3://{S3_MODEL_BUCKET}/{S3_MODEL_PREFIX}/{request.model_name}/"
        )
        s3_check = subprocess.run(
            ["aws", "s3", "ls", s3_model_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if s3_check.returncode != 0 or not s3_check.stdout.strip():
            job_logger.warning(
                f"[S3] Model '{request.model_name}' not found at {s3_model_path}. "
                f"Falling back to HuggingFace download."
            )
            request = request.model_copy(update={"s3_models": False})
        else:
            job_logger.info(f"[S3] Verified model exists at {s3_model_path}")

    hf_token = request.hf_token or HF_TOKEN or ""
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
        prefer_spot=True,
    )

    # Build resources with any_of for fallback regions
    if ordered_regions:
        any_of_resources = []
        for candidate in ordered_regions[:5]:
            any_of_resources.append(
                {
                    "region": candidate.region,
                    "instance_type": config.instance_type,
                    "use_spot": candidate.use_spot,
                    "disk_size": "300GB",
                    "ports": VLLM_PORT,
                }
            )
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

        # Add S3 model weight mount if requested
        if request.s3_models:
            model_mount_path = f"/models/{request.model_name}"
            yaml_data["file_mounts"][model_mount_path] = {
                "source": f"s3://{S3_MODEL_BUCKET}/{S3_MODEL_PREFIX}/{request.model_name}",
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
        }

        # Add S3 model weight mount if requested
        if request.s3_models:
            model_mount_path = f"/models/{request.model_name}"
            replace_yaml[f"file_mounts.{model_mount_path}.source"] = (
                f"s3://{S3_MODEL_BUCKET}/{S3_MODEL_PREFIX}/{request.model_name}"
            )
            replace_yaml[f"file_mounts.{model_mount_path}.mode"] = "COPY"

        update_yaml_file("templates/vllm.yaml", replace_yaml, YAML_OUTPUT)

    cm = get_cluster_manager()

    # Register job in tracker
    jt = get_job_tracker()
    job_state = jt.build_job_state_batched(request, config)
    jt.add(job_state)
    jt.update_status(config.decision_id, "launching")

    # Construct S3 output path for later download
    output_s3_path = f"{s3_output_dir}/{request.output_file}"
    job_logger.info(f"[Job] Output will be saved to: {s3_output_dir}/")

    def monitor_and_download(job_id):
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
            get_job_tracker().update_status(config.decision_id, "succeeded" if is_success else "failed")

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
            if quota_tracker is not None:
                quota_tracker.release_cluster(config.decision_id)
            cm.unregister(config.decision_id)

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
        cm.register(config.decision_id, config.decision_id,
                     region=actual_region, market=actual_market,
                     instance_type=config.instance_type, num_instances=num_nodes)
        job_logger.info(f"[Quota] Reserved {config.instance_type} x{num_nodes} in {actual_region}/{actual_market}")

        # Update job tracker: running + head IP
        jt.update_status(config.decision_id, "running")
        head_ip = getattr(handle, 'head_ip', None)
        if head_ip:
            jt.set_head_ip(config.decision_id, head_ip)

        # Stream logs in background and download when done
        threading.Thread(target=monitor_and_download, args=(job_id,), daemon=True).start()

    except Exception as e:
        job_logger.error(
            f"[SkyPilot] Failed to launch cluster {config.decision_id}: {e}"
        )
        jt.update_status(config.decision_id, "failed")
        close_job_logger(job_logger)
        raise Exception(f"Failed to launch cluster {config.decision_id}: {e}")


async def sp_launch_vllm_online(request: OnlineServingRequest, config: MagicOutput):
    """Launch persistent vllm online deployment"""
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

    print(f"vLLM server launched at {endpoint_url}")

    url = f"http://{public_ip}:{VLLM_PORT}/v1/models"
    response = requests.get(url, timeout=5)
    if (
        response.status_code == 200
    ):  # do sth here for a valid API up and sth else otherwise
        print(f"vLLM server API is up at {endpoint_url}")
        return endpoint_url
    else:
        raise Exception(f"vLLM server API is not up at {endpoint_url}")
