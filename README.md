# Orca

Orca is Tandemn's end-to-end system for processing taking users' requests for inference (batched, online). Orca takes a user request, makes a decision on the GPUs needed, and then launches the required instances.

## Installation

```bash
# Clone the repo with its submodule
git clone --recurse-submodules https://github.com/Tandemn-Labs/orca.git

# If you already cloned without --recurse-submodules, initialize the submodule manually
git submodule update --init --recursive
```

Do this in whatever Python virtual environment you prefer

```bash
# Install dependencies
pip install -r requirements.txt

# If you use uv, do this instead
uv pip install -r requirements.txt
```

### Configuring cloud

We use SkyPilot to launch AWS instances. Make sure you have AWS credentials. Check that you have a `~/.aws/credentials` file. We support only AWS for now, and am working to add more cloud providers.
(Also check out: https://github.com/tandemn-labs/tandemn-tuna)

## Using the roofline placement solver

Other files you need:

- AWS Quota CSV file in `quota/aws_gpu_quota_by_region.csv`
- Profiling files in `~/.perfdb/[GPU name]/files`

These default files are included in the repo.

1. Start the server

```
python -m server
```

2. Upload a JSONL file of input prompts to S3. We use the OpenAI batch format (Check: https://developers.openai.com/api/docs/guides/batch)

3. Now send a BatchedRequest object to the default endpoint `http://localhost:26336/submit/batch`

```
curl --request POST \
  --url http://localhost:26336/test/placement \
  --header 'content-type: application/json' \
  --data '{
  "user_id": "test-user",
  "input_file": "s3://tandemn-orca/workload/sharegpt-numreq_200-avginputlen_956-avgoutputlen_50.jsonl",
  "output_file": "output.jsonl",
  "avg_output_tokens": 100,
  "max_output_tokens": 256,
  "description": "Qwen batch inference test",
  "task_type": "chat_completion",
  "task_priority": "high",
  "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
  "engine": "vllm",
  "slo_mode": "batch",
  "slo_deadline_hours": 1,
  "placement": "aws:us-east-1:auto",
  "placement_solver": "roofline",
  "s3_models": true
}'
```
