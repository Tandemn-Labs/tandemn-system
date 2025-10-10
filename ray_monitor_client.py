"""
This is basically the sidecar that goes along 
with the ray cluster, and it will monitor 
1 - Health of each chain.
2 - Model:Head Node Mapping
3 - Head_Node:Chain GPUs mapping
4 - Observability Metrics from vLLM Prometheus Server
The idea is that it polls the Ray GCS(Global Control System) for the node/worker status
"""

import ray
from ray.utils.state import list_nodes, list_actors
import requests
from dataclasses import dataclass
from enum import Enum
import time
# I am not really sure about this function, 
# will see
from prometheus_client.parser import text_string_to_metric_families

# this is for all the workers in the ray cluster
class WorkerHealthState(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    OVERLOADED = "OVERLOADED"
    UNHEALTHY = "UNHEALTHY"
    DEAD = "DEAD"

@dataclass
class VLLMWorkerMetrics:
    """all the possible metrics
    coming from the vLLM Server
    refer @https://nm-vllm.readthedocs.io/en/latest/serving/metrics.html"""

    num_requests_running: int
    num_requests_waiting: int
    num_requests_swapped: int
    gpu_cache_usage_perc: float
    cpu_cache_usage_perc: float
    time_to_first_token_seconds: float
    time_per_output_token_seconds: float
    e2e_request_latency_seconds: float
    prompt_tokens_total: int
    generation_tokens_total: int
    num_preemptions_total: int
    request_success_total: int
    is_chain_reachable: bool
    last_update: float


def scrape_vllm_metrics(worker_endpoint: str):
    """Scrape Metrics coming from Prometheus of the vLLM Server"""
    try: 
        response = requests.get(f"{worker_endpoint}/metrics", timeout=5)
        response.raise_for_status() # raise exception if the request is not successful

        # instantiate the metrics
        metrics=VLLMWorkerMetrics()
        metrics.is_chain_reachable=True
        metrics.last_update=time.time()

        for family in text_string_to_metric_families(response.text):
            if family.name=="vllm_is_chain_reachable":
                metrics.is_chain_reachable=True
            elif family.name=="vllm_num_requests_running":
                metrics.num_requests_running=family.samples[0].value
            elif family.name=="vllm_num_requests_waiting":
                metrics.num_requests_waiting=family.samples[0].value
            elif family.name=="vllm_num_requests_swapped":
                metrics.num_requests_swapped=family.samples[0].value
            elif family.name=="vllm_gpu_cache_usage_perc":
                metrics.gpu_cache_usage_perc=family.samples[0].value
            elif family.name=="vllm_cpu_cache_usage_perc":
                metrics.cpu_cache_usage_perc=family.samples[0].value
            elif family.name=="vllm_time_to_first_token_seconds":
                metrics.time_to_first_token_seconds=family.samples[0].value
            elif family.name=="vllm_time_per_output_token_seconds":
                metrics.time_per_output_token_seconds=family.samples[0].value
            elif family.name=="vllm_e2e_request_latency_seconds":
                metrics.e2e_request_latency_seconds=family.samples[0].value
            elif family.name=="vllm_prompt_tokens_total":
                metrics.prompt_tokens_total=family.samples[0].value
            elif family.name=="vllm_generation_tokens_total":
                metrics.generation_tokens_total=family.samples[0].value
            elif family.name=="vllm_num_preemptions_total":
                metrics.num_preemptions_total=family.samples[0].value
            elif family.name=="vllm_request_success_total":
                metrics.request_success_total=family.samples[0].value
            elif family.name=="vllm_last_update":
                metrics.last_update=family.samples[0].value
            
    except Exception as e:
        print(f"Error scraping vLLM metrics: {e}")
        metrics.is_chain_reachable=False
        metrics.last_update=time.time()
        
    return metrics




    # Parse the metrics


