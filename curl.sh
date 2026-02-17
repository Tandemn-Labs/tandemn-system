#!/bin/bash

endpoint="/submit/batch"
# endpoint="/test/placement"

HF_TOKEN="HF_TOKEN_HERE"
OPENROUTER_API_KEY="OPENROUTER_API_KEY_HERE"

# MODEL_NAME="meta-llama/Meta-Llama-3-70B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
# MODEL_NAME="Qwen/Qwen3-235B-A22B"
# MODEL_NAME="Qwen/Qwen2.5-A14B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-32B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

        # "llm_advisor_tier": "paid",
        # "llm_advisor_tier": "free",

        # "placement_solver": "llm",
        # "placement_solver": "roofline",

curl --request POST \
    --url http://localhost:26336${endpoint} \
    --header 'content-type: application/json' \
    --data '{
        "placement_solver": "llm",
        "llm_advisor_tier": "paid",
        "user_id": "test-user",
        "input_file": "s3://tandemn-orca/batch/input.jsonl",
        "output_file": "output.jsonl",
        "num_lines": 100,
        "avg_input_tokens": 50,
        "avg_output_tokens": 100,
        "max_input_tokens": 200,
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
        "openrouter_api_key": "'"${OPENROUTER_API_KEY}"'"
    }'
