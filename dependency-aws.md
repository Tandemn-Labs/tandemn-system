# Running Large Language Models on AWS GPU Instances

Technical notes on deploying vLLM with 70B+ parameter models on AWS p4d/p4de instances (A100 GPUs).

## Environment Summary

| Component | Version | Notes |
|-----------|---------|-------|
| Instance | p4d.24xlarge | 8x A100 40GB, NVSwitch |
| NVIDIA Driver | 580.105.08 | Custom AMI required |
| CUDA | 12.8 | Bundled with driver |
| vLLM | 0.10.0 | Latest stable |
| PyTorch | 2.7.1+cu126 | Installed by vLLM |
| transformers | 5.x | Installed by vLLM |

## Issues and Solutions

### 1. CUDA Driver Compatibility

**Problem**: AWS default AMIs ship with driver 535.x, which only supports CUDA 12.2. vLLM 0.10.0 ships with `cu126` wheels requiring CUDA 12.6+.

**Symptom**:
```
CUDA error 802: system not yet initialized
```

**Solution**: Build a custom AMI with driver 580+ or use vLLM 0.7.x with `cu121` wheels.

```bash
# Custom AMI: ami-04f8546cd7cc1dcd9 (us-east-1)
# Driver 580.105.08 | CUDA 12.8 | PyTorch 2.7.1+cu128
```

### 2. Transformers 5.x Breaking Change

**Problem**: `transformers` 4.47+ removed the `all_special_tokens_extended` property. vLLM's tokenizer initialization still references this attribute.

**Symptom**:
```
AttributeError: 'Qwen2Tokenizer' has no attribute 'all_special_tokens_extended'
```

**Solution**: Monkey-patch the attribute before vLLM imports transformers.

```python
# vllm_compat_patch.py
from transformers import PreTrainedTokenizerBase

if not hasattr(PreTrainedTokenizerBase, 'all_special_tokens_extended'):
    @property
    def all_special_tokens_extended(self):
        return list(set(self.all_special_tokens))
    PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended
```

### 3. sitecustomize.py Not Loading in Virtualenv

**Problem**: Python only auto-imports `sitecustomize.py` from system site-packages, not from virtualenv site-packages.

**Solution**: Import the compatibility patch directly in the entry point script before any vLLM imports.

```python
# batch_runner.py (top of file)
try:
    import vllm_compat_patch
except ImportError:
    pass
```

### 4. CUDA Fork Initialization Error

**Problem**: Running `nvidia-smi` or `torch.cuda.device_count()` in setup scripts initializes CUDA. vLLM then fails when spawning worker processes via fork.

**Symptom**:
```
RuntimeError: Cannot re-initialize CUDA in forked subprocess
```

**Solution**: Avoid any CUDA initialization before vLLM starts. Remove diagnostic commands from setup scripts or run them in subshells that exit before the main process.

### 5. vLLM DisabledTqdm Bug

**Problem**: vLLM 0.10.0 with newer `huggingface_hub` passes `disable=True` twice to tqdm, causing a TypeError.

**Solution**: Patch `weight_utils.py` during setup:

```bash
sed -i 's/super().__init__(\*args, \*\*kwargs, disable=True)/kwargs.pop("disable", None); super().__init__(*args, **kwargs, disable=True)/' \
  "$SITE_PACKAGES/vllm/model_executor/model_loader/weight_utils.py"
```

### 6. NVSwitch and Fabric Manager

**Problem**: p4d/p4de instances use NVSwitch for GPU-to-GPU communication. The Fabric Manager service must be running for NVLink to function.

**Symptom**: GPU topology shows `PHB` or `SYS` instead of `NV12` between GPUs.

**Solution**: Verify and start Fabric Manager in setup:

```bash
if systemctl list-units --type=service --all | grep -q nvidia-fabricmanager; then
    sudo systemctl start nvidia-fabricmanager
fi
```

### 7. Python Template Syntax Conflict

**Problem**: Bash default value syntax `${VAR:-default}` conflicts with Python's `str.format()`.

**Symptom**:
```
KeyError: 'LD_LIBRARY_PATH'
```

**Solution**: Escape braces in YAML templates:

```yaml
# Wrong
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}

# Correct (when processed by Python str.format)
export LD_LIBRARY_PATH=${{LD_LIBRARY_PATH:-}}
```

## Model-Specific Notes

### Qwen2.5-72B-Instruct

- Requires the `all_special_tokens_extended` patch (Qwen tokenizer is affected)
- TP=8 on single p4d node, ~17GB per GPU
- Recommended: `max_model_len=32768`, `gpu_memory_utilization=0.90`

### Llama 3 70B

- Less sensitive to transformers version changes
- Same TP=8 configuration works

## Template Architecture

```
templates/
├── vllm.yaml                    # Generic template (L40S, etc.)
├── vllm_batch_runner.py         # Batch processing script
├── vllm_compat_patch.py         # Transformers compatibility patch
└── vllm_configs/
    ├── vllm_A100.yaml           # A100-specific (custom AMI)
    └── vllm_{model}-{gpu}-tp{tp}-pp{pp}.yaml  # Per-config overrides
```

Template selection priority:
1. Exact match: `vllm_{model}-{gpu}-tp{tp}-pp{pp}.yaml`
2. GPU-specific: `vllm_{gpu}.yaml`
3. Generic: `vllm.yaml`

## Performance Baseline

Qwen2.5-72B-Instruct on p4d.24xlarge (8x A100):

| Metric | Value |
|--------|-------|
| Model load time | ~128s |
| Throughput | 1.74 req/s |
| Token throughput | ~5,050 tok/s |
| Memory per GPU | ~17GB weights + ~13GB KV cache |

---

## Appendix: Detailed Error Logs and Fixes

### Issue 1: CUDA Driver Version Mismatch

**Error Log:**
```
(RayWorkerWrapper pid=12345) RuntimeError: CUDA error: CUDA driver version is
insufficient for CUDA runtime version
CUDA kernel errors might be asynchronously reported at some other API call,
so the stacktrace below might be incorrect.

torch.cuda.CudaError: CUDA error 802: system not yet initialized
```

**Environment:**
- AWS default AMI: Driver 535.183.01
- vLLM 0.10.0: Requires CUDA 12.6+ (ships with cu126 wheels)
- Driver 535 only supports up to CUDA 12.2

**Root Cause:**
vLLM 0.10.0 PyPI wheels are built against CUDA 12.6 (`torch==2.7.1+cu126`). AWS Deep Learning AMIs ship with driver 535.x which maxes out at CUDA 12.2. The runtime/driver version mismatch causes initialization failure.

**Fix:**
Option A: Build custom AMI with driver 580+
```bash
# Install driver 580.105.08 + CUDA 12.8
# AMI: ami-04f8546cd7cc1dcd9 (us-east-1)
```

Option B: Use older vLLM with cu121
```bash
uv pip install "vllm==0.7.3" --extra-index-url https://download.pytorch.org/whl/cu121
```

---

### Issue 2: Transformers all_special_tokens_extended Removal

**Error Log:**
```
Traceback (most recent call last):
  File "/home/ubuntu/batch_runner.py", line 89, in <module>
    from vllm import LLM, SamplingParams
  File ".venv/lib/python3.12/site-packages/vllm/__init__.py", line 18, in <module>
    ...
  File ".venv/lib/python3.12/site-packages/vllm/transformers_utils/tokenizer.py", line 245
    all_special = tokenizer.all_special_tokens_extended
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'Qwen2Tokenizer' has no attribute 'all_special_tokens_extended'
```

**Environment:**
- transformers 5.2.0 (installed by vLLM 0.10.0)
- vLLM 0.10.0
- Qwen2.5-72B-Instruct model

**Root Cause:**
The `all_special_tokens_extended` property was deprecated in transformers 4.45 and removed in 4.47+. vLLM's tokenizer initialization code still references this attribute. Qwen tokenizers are particularly affected.

**Fix:**
Create `vllm_compat_patch.py`:
```python
"""Compatibility patch for vLLM with transformers 4.47+"""
from transformers import PreTrainedTokenizerBase

if not hasattr(PreTrainedTokenizerBase, 'all_special_tokens_extended'):
    @property
    def _all_special_tokens_extended(self):
        all_tokens = set(self.all_special_tokens)
        for token in self.additional_special_tokens:
            all_tokens.add(token)
        return list(all_tokens)

    PreTrainedTokenizerBase.all_special_tokens_extended = _all_special_tokens_extended
    print("Patched transformers: added all_special_tokens_extended")
```

---

### Issue 3: sitecustomize.py Ignored in Virtualenv

**Error Log:**
Same as Issue 2 - patch exists but not applied.

**Debug Output:**
```bash
$ ls .venv/lib/python3.12/site-packages/sitecustomize.py
.venv/lib/python3.12/site-packages/sitecustomize.py  # File exists

$ python -c "import sys; print(sys.path)"
# sitecustomize.py never executed - no "Patched transformers" message
```

**Root Cause:**
Python's `sitecustomize.py` auto-import mechanism only works from the **system** site-packages directory (`/usr/lib/python3.12/site-packages/`), not from virtualenv site-packages. Installing the patch via:
```bash
echo "import vllm_compat_patch" > "$SITE_PACKAGES/sitecustomize.py"
```
Places it in `.venv/lib/python3.12/site-packages/sitecustomize.py`, which Python ignores.

**Fix:**
Import patch directly at top of entry point script:
```python
# batch_runner.py - FIRST import
try:
    import vllm_compat_patch
except ImportError:
    pass  # Patch not available locally
```

---

### Issue 4: CUDA Re-initialization in Forked Process

**Error Log:**
```
(RayWorkerWrapper pid=13366) RuntimeError: Cannot re-initialize CUDA in forked subprocess.
To use CUDA with multiprocessing, you must use the 'spawn' start method

torch.multiprocessing.spawn.ProcessRaisedException:
-- Process 0 terminated with the following error:
RuntimeError: CUDA has been already initialized, you may want to use spawn mode.
```

**Environment:**
- Setup script ran `nvidia-smi` for diagnostics
- Setup script ran `python -c "import torch; torch.cuda.device_count()"`
- vLLM uses multiprocessing with fork (not spawn)

**Root Cause:**
Any CUDA operation before vLLM starts (nvidia-smi, torch.cuda calls) initializes the CUDA context in the parent process. When vLLM forks worker processes, CUDA cannot be re-initialized in the forked children.

**Problematic Code:**
```bash
# setup section - BAD
nvidia-smi  # Initializes CUDA
python3 -c "import torch; print(torch.cuda.device_count())"  # Initializes CUDA
```

**Fix:**
Remove all CUDA-touching commands from setup, or run them in subshells:
```bash
# Option 1: Remove entirely
# Option 2: Run in subshell that exits
(nvidia-smi || true)  # Subshell exits, CUDA context destroyed
```

---

### Issue 5: DisabledTqdm Duplicate Keyword Argument

**Error Log:**
```
Traceback (most recent call last):
  File ".venv/lib/python3.12/site-packages/vllm/model_executor/model_loader/weight_utils.py"
    super().__init__(*args, **kwargs, disable=True)
TypeError: __init__() got multiple values for keyword argument 'disable'
```

**Environment:**
- vLLM 0.10.0
- huggingface_hub 0.30+ (newer versions pass disable in kwargs)

**Root Cause:**
vLLM's `DisabledTqdm` class passes `disable=True` explicitly, but newer huggingface_hub already includes `disable` in kwargs, causing duplicate keyword argument error.

**Original Code:**
```python
class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)  # BUG: disable may be in kwargs
```

**Fix:**
```bash
WEIGHT_UTILS="$SITE_PACKAGES/vllm/model_executor/model_loader/weight_utils.py"
sed -i 's/super().__init__(\*args, \*\*kwargs, disable=True)/kwargs.pop("disable", None); super().__init__(*args, **kwargs, disable=True)/' "$WEIGHT_UTILS"
```

---

### Issue 6: NVSwitch Fabric Manager Not Running

**Error Log:**
```
$ nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
GPU0     X      PHB     PHB     PHB     SYS     SYS     SYS     SYS
GPU1    PHB      X      PHB     PHB     SYS     SYS     SYS     SYS
...
# Should show NV12 for NVSwitch, not PHB/SYS
```

**Runtime Symptom:**
- Slow all-reduce operations
- Tensor parallel communication bottleneck
- Poor multi-GPU scaling

**Root Cause:**
p4d/p4de instances use NVSwitch (HGX A100) for GPU interconnect. The nvidia-fabricmanager service must be running for NVLink through NVSwitch to work. Without it, GPUs fall back to PCIe.

**Fix:**
```bash
if systemctl list-units --type=service --all | grep -q nvidia-fabricmanager; then
    if ! systemctl is-active --quiet nvidia-fabricmanager; then
        sudo systemctl start nvidia-fabricmanager
        sleep 2
    fi
fi

# Verify
nvidia-smi topo -m  # Should show NV12 between all GPUs
nvidia-smi nvlink -s  # Should show active links
```

---

### Issue 7: Python str.format() Conflicts with Bash Syntax

**Error Log:**
```
Traceback (most recent call last):
  File "server.py", line 655
    template_content = template_content.replace("{" + key + "}", str(value))
KeyError: 'LD_LIBRARY_PATH'
```

Or:
```
KeyError: 'VAR'
```

**Problematic Template:**
```yaml
run: |
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/usr/local/lib}
  export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
```

**Root Cause:**
Python's `str.format()` and string replacement interprets `${VAR:-default}` as a template placeholder `{VAR:-default}`, attempting to substitute a key named `VAR:-default` or `LD_LIBRARY_PATH`.

**Fix:**
Escape braces in YAML templates:
```yaml
run: |
  export LD_LIBRARY_PATH=${{LD_LIBRARY_PATH:-/usr/local/lib}}
  export CUDA_HOME=${{CUDA_HOME:-/usr/local/cuda}}
```

After Python processing, `{{` becomes `{` and `}}` becomes `}`.

---

### Issue 8: Custom AMI Not Being Used

**Error Log:**
```
(setup pid=5146) + nvidia-smi
Driver Version: 535.183.01   # WRONG - should be 580.105.08
CUDA Version: 12.2           # WRONG - should be 12.8
```

**Template Configuration:**
```yaml
resources:
  cloud: aws
  accelerators: A100:8
  image_id: ami-04f8546cd7cc1dcd9  # Custom AMI specified
  region: us-east-1
```

**Root Cause:**
Server code was overwriting the entire `resources` block, discarding the template's `image_id`:
```python
# server.py - BAD
yaml_data["resources"] = resources_config  # Overwrites everything including image_id
```

**Fix:**
Preserve image_id from template when specified:
```python
template_image_id = yaml_data.get("resources", {}).get("image_id")
template_region = yaml_data.get("resources", {}).get("region")

if template_image_id:
    resources_config = {
        "cloud": "aws",
        "accelerators": yaml_data["resources"].get("accelerators"),
        "image_id": template_image_id,  # Preserve custom AMI
        "region": template_region,
        ...
    }
```

---

### Issue 9: Template Placeholders Not Substituted

**Error Log:**
```bash
# Generated YAML contains literal placeholders
python ~/batch_runner.py \
  -i {input_file} \           # Not substituted!
  -o {output_file} \          # Not substituted!
  --model {model} \           # Not substituted!
```

**Root Cause:**
Per-config templates (in `vllm_configs/`) were only substituting a subset of placeholders, missing `input_file`, `output_file`, `model`, etc.

**Fix:**
Use the full `replace_run_vllm()` substitution for all templates:
```python
if "vllm_configs" in template_path:
    replace_dict = replace_run_vllm(request, config, job_dirname)  # Full substitution

    template_content = Path(template_path).read_text()
    for key, value in replace_dict.items():
        template_content = template_content.replace("{" + key + "}", str(value))
```

---

### Issue 10: Metrics Missing instance_type and gpu_name

**Error Log:**
```csv
# metrics.csv
...,instance_type,gpu_name,...
...,unknown,unknown,...
```

**Template Configuration:**
```yaml
# Missing arguments in batch_runner.py call
python ~/batch_runner.py \
  -i {input_file} \
  -o {output_file} \
  --model Qwen/Qwen2.5-72B-Instruct \
  -tp 8 \
  -pp 1 \
  # Missing: --cloud, --instance-type, --gpu-name
  --hf-token "$HF_TOKEN"
```

**Root Cause:**
Specific per-config templates (e.g., `vllm_qwen2.5-72b-A100-tp8-pp1.yaml`) were missing the infrastructure tracking arguments that the generic template had.

**Fix:**
Add missing arguments to all templates:
```yaml
python ~/batch_runner.py \
  ...
  --cloud {cloud} \
  --instance-type {instance_type} \
  --gpu-name {gpu_name} \
  --engine {engine} \
  --quantization {quantization} \
  --dtype {dtype} \
  --kv-cache-dtype {kv_cache_dtype} \
  --hf-token "$HF_TOKEN"
```

---

### Issue 11: Unbound Variable in Bash Template

**Error Log:**
```
(setup pid=4803) bash: VLLM_VERSION: unbound variable
ERROR: Job 1's setup failed. Failed workers: (pid=4803, returncode=1).
```

**Problematic Template:**
```bash
set -euxo pipefail  # -u causes unbound variable error

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "Installed vLLM version: $VLLM_VERSION"  # VLLM_VERSION never defined!

if [[ "$VLLM_VERSION" == 0.10.* ]]; then  # Also fails here
```

**Root Cause:**
Template referenced `$VLLM_VERSION` but never defined it. Combined with `set -u` (treat unbound variables as error), this causes immediate failure.

**Fix:**
Either define the variable or remove the reference. Since we're using a fixed vLLM version (0.10.0), simplify by removing version-conditional logic:

```bash
# Instead of version checks, just apply patches directly
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

# Patch 1: Fix DisabledTqdm bug
WEIGHT_UTILS="$SITE_PACKAGES/vllm/model_executor/model_loader/weight_utils.py"
sed -i 's/super().__init__(\*args, \*\*kwargs, disable=True)/kwargs.pop("disable", None); super().__init__(*args, **kwargs, disable=True)/' "$WEIGHT_UTILS"
```

---

## Debugging Checklist

When deploying on new GPU instances:

1. **Check driver version**: `nvidia-smi` → Driver 550+ for vLLM 0.10.0
2. **Check CUDA version**: Must match vLLM wheel (cu126 = CUDA 12.6+)
3. **Check Fabric Manager** (NVSwitch instances): `systemctl status nvidia-fabricmanager`
4. **Check GPU topology**: `nvidia-smi topo -m` → Should show NV12 for NVSwitch
5. **Verify patch applied**: Look for "Patched transformers" in logs
6. **Check template substitution**: Verify no literal `{placeholder}` in generated YAML
