#!/bin/bash

# endpoint="/submit/batch"
endpoint="/test/placement"

HF_TOKEN="HF_TOKEN_HERE" # not neccessary if the model does not require it.
OPENROUTER_API_KEY="OPENROUTER_API_KEY_HERE"

# MODEL_NAME="meta-llama/Meta-Llama-3-70B-Instruct"
MODEL_NAME="Qwen/Qwen3-32B"
# MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
# MODEL_NAME="Qwen/Qwen3-235B-A22B"
# MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

# Placement solver: "roofline", "llm", or "user_specified"
# placement_solver="llm"
placement_solver="roofline"
# placement_solver="user_specified"
gpu_type="L40S" # [A100, L40S, H100]
tp_size=8 # tensor parallelism: 1, 2, 4, 8
pp_size=2 # pipeline parallelism: 1, 2, 3, 4

llm_advisor_tier="paid" # only used if placement_solver is "llm"

# input_file="s3://tandemn-orca/batch/input.jsonl"
# input_file="s3://tandemn-orca/workload/sharegpt-numreq_200-avginputlen_956-avgoutputlen_50.jsonl"
input_file="s3://tandemn-orca/workload/sharegpt-numreq_200-avginputlen_1604-avgoutputlen_100.jsonl"
# input_file="s3://tandemn-orca/workload/sharegpt-numreq_200-avginputlen_2926-avgoutputlen_100.jsonl"

curl --request POST \
    --url http://localhost:26336${endpoint} \
    --header 'content-type: application/json' \
    --data '{
        "placement_solver": "'"${placement_solver}"'",
        "llm_advisor_tier": "'"${llm_advisor_tier}"'",
        "user_id": "test-user",
        "input_file": "'"${input_file}"'",
        "output_file": "output.jsonl",
        "avg_output_tokens": 100,
        "max_output_tokens": 256,
        "description": "Qwen batch inference test",
        "task_type": "chat_completion",
        "task_priority": "high",
        "model_name": "'"${MODEL_NAME}"'",
        "engine": "vllm",
        "slo_mode": "batch",
        "slo_deadline_hours": 1,
        "placement": "aws:us-east-1:auto",
        "hf_token": "'"${HF_TOKEN}"'",
        "openrouter_api_key": "'"${OPENROUTER_API_KEY}"'",
        "gpu_type": "'"${gpu_type:-}"'",
        "tp_size": '"${tp_size:-null}"',
        "pp_size": '"${pp_size:-null}"',
        "s3_models": true
    }'

# --- Example: user_specified (bypass solver, use A100 TP=8 PP=1) ---
# curl --request POST \
#     --url http://localhost:26336/submit/batch \
#     --header 'content-type: application/json' \
#     --data '{
#         "placement_solver": "user_specified",
#         "user_id": "test-user",
#         "input_file": "s3://tandemn-orca/workload/sharegpt-numreq_200-avginputlen_2926-avgoutputlen_100.jsonl",
#         "output_file": "output.jsonl",
#         "avg_output_tokens": 100,
#         "max_output_tokens": 256,
#         "description": "Qwen 72B on A100 user_specified",
#         "task_type": "chat_completion",
#         "task_priority": "high",
#         "model_name": "Qwen/Qwen2.5-72B-Instruct",
#         "engine": "vllm",
#         "slo_mode": "batch",
#         "slo_deadline_hours": 1,
#         "placement": "aws:us-east-1:auto",
#         "gpu_type": "A100",
#         "tp_size": 8,
#         "pp_size": 1
#     }'
