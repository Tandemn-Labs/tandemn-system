# Orca

Orca is Tandemn's end-to-end system for processing taking users' requests for inference (batched, online). Orca takes a user request, makes a decision on the GPUs needed, and then launches the required instances.

## Installation

Do this in whatever Python virtual environment you prefer

```
# Install dependencies
pip install -r requirements.txt

# Ensure you have a ~/.aws/credentials file

# Create a temp directory for generated YAMLS
mkdir temp
```

We use SkyPilot to launch AWS instances. Make sure you have AWS credentials. Check that you have a `~/.aws/credentials` file.

### Using the existing placement solver

There is a simple placement solver `./placement/aws_magic.py`. Using this solver requires a few additional steps.

Ensure you have a `.env` file with an OpenRouter key.

```
TD_OPENROUTER_KEY=[yourverynicekey]
```

Other files you need:

- AWS Quota CSV file in `~/.quotas/aws_gpu_quota_by_region.csv`
- Profiling files in `~/.perfdb/[GPU name]/files`

### Running things

1. Start the server

```
python -m server
```

2. Upload a file to S3.

3. Now send a BatchedRequest object to the default endpoint `http://localhost:26336/submit/batch`

```
curl --request POST \
  --url http://localhost:26336/submit/batch \
  --header 'content-type: application/json' \
  --data '{
  "user_id": "user-123",
  "input_file": "s3://tandemn-user-data/test.jsonl",
  "output_file": "output.txt",
  "num_lines": 1000,
  "avg_input_tokens": 150,
  "avg_output_tokens": 200,
  "description": "Batch summarize support tickets",
  "task_type": "summarization",
  "task_priority": "high",
  "model_name": "meta-llama/Meta-Llama-3-70B-Instruct",
  "engine": "vllm",
  "quantization_bits": "8",
  "is_speculative_decode": true,
  "is_PD_disaggregation": false,
  "slo_mode": "batch",
  "slo_deadline_hours": 4,
  "placement": "aws:us-east-1:g5-12xlarge"
}'
```

## Files

`server.py` runs a HTTP endpoint that takes in user requests.

The placement logic sits in the `./placement/` directory. Look at the `magic.py` abstract class for the function you need to implement `decide()`, returning a `MagicOutput`. The architecture essentially allows for a drop-in replacement of the placement logic/algorithm.

### Directories

- `./models/` - The data models for objects we use in the code, including `BatchedRequest`
- `./placement/` - The code for the placement logic (i.e. Takes user request and decide what GPUs to use, what engine configs etc)
  - `./placement/magic.py`
- `./storage/` - Storage logic, mostly S3 for now
- `./templates/` - Template files for SkyPilot launches. Certain fields gets replaced at runtime
  - `.yaml` files are for SkyPilot launch
  - Files with no extension are templates for commands that are used to replace the `run` field in the YAMLs
- `./tracking/` - Some tracking utilities. Unused for now.
- `./utils/` - Some common utils
