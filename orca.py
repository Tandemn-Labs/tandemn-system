import os
import re
import glob
import math
import pandas as pd
from pathlib import Path
import torch
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from huggingface_hub import InferenceClient       
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM
import time
from openai import OpenAI
import pprint
from termcolor import colored
from tabulate import tabulate
perfdb_dir = "./perf_db"
quota_csv = "./temp/aws_gpu_quota_by_region.csv"

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # Don't wrap columns to the next line

OPENROUTER_API_KEY = "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"


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
    submitted_at: float
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

def normalize_gpu_name(s_name):
    if s_name is None:
        return ""
    s_name = str(s_name).strip()
    s_name = re.sub(r"(?i)^nvidia\s+", "", s_name)          # drop NVIDIA prefix
    s_name = re.sub(r"^\s*\d+(\.\d+)?\s*x\s*", "", s_name)  # drop leading '4x '
    return s_name.strip()

def get_num_params_from_text(model_name):
    """
    This is just a hack for now, in the future it should be 
    a function that gets it from Huggingface model card.
    """
    if not model_name:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)[bB](?:illion)?", model_name)
    return float(m.group(1)) if m else None

def load_all_perfdb_files(roofline_dir):
    """
    Loads a list of all the dicts
    [{
        "path": "./perf_db/L40S/perfdb_l40s_llama_70b.csv",
        "gpu_base": "L40S",
        "models": set([...]),
        "model_size_b": 70.0 or None,
        "df": normalized dataframe with columns:
              model_name, gpu_base, tp, pp, tokens_per_sec, mem_per_gpu_gb
    }, ...]
    """
    perf_db_dir = Path(roofline_dir)
    folders = [p for p in perf_db_dir.rglob("*") if p.is_dir()]
    perfdb_files = [p for p in perf_db_dir.rglob("*") if p.is_file()]
    all_results = []
    for path in perfdb_files:
        gpu_base = path.parent.name
        df_raw = pd.read_csv(path)
        df = pd.DataFrame(
            {
                "model_name": df_raw["Model Name"],
                "gpu_base": gpu_base,
                "tp": df_raw["TP"].astype(int),
                "pp": df_raw["PP"].astype(int),
                "max_input_length": df_raw["Max Input Length"].astype(int),
                "max_output_length": df_raw["Max Output Length"].astype(int),
                "tokens_per_sec": df_raw["Total Tokens Per Second"].astype(float),
                "mem_per_gpu_gb": df_raw["Mem Per GPU GB"].astype(float),
            }
        )
        models = set(df["model_name"].unique())
        model_size_b = get_num_params_from_text(df["model_name"].iloc[0])
        print("Model size b:", model_size_b)
        all_results.append({
            "path": path,
            "gpu_base": gpu_base,
            "models": models,
            "model_size_b": model_size_b,
            "df": df
        })
    return all_results

def select_perf_files_closest_to_model_size(perf_files, model_size_b, k=1):
    """
    Selects the k perf files closest to the model size.
    """
    print("Perf files that are closest to the model size: ", len(perf_files))
    print("Model size that is closest to the model you submitted", model_size_b)
    print("K:", k)
    return sorted(perf_files, key=lambda x: abs(x["model_size_b"] - model_size_b))[:k]

def load_quota_csv(quota_csv):
    """
    Loads the quota csv into a dataframe.
    """
    df = pd.read_csv(quota_csv)
    df["gpu_base"] = df["GPU_Type"].apply(normalize_gpu_name)
    df["gpu_count"] = df["GPU_Type"].str.extract(r"(\d+)\s*x\s*")[0].fillna(1.0).astype(float) # get the 4 in 4XA100
    print("Detected the GPU Types in your Quota: ", df["gpu_base"].unique())
    print("Detected the GPU Count: ", df["gpu_count"].unique())
    return df


##### Enumerate Candidates ###########

def find_best_perf_match_based_on_input_output_length(df, job_avg_input, job_avg_output):
    df["input_dist"] = abs(df["max_input_length"] - job_avg_input)
    df["output_dist"] = abs(df["max_output_length"] - job_avg_output)
    df["total_dist"] = df["input_dist"] + df["output_dist"]
    
    df["under_input"] = (df["max_input_length"] < job_avg_input).astype(int)
    df["under_output"] = (df["max_output_length"] < job_avg_output).astype(int)
    df["under_penalty"] = df["under_input"] + df["under_output"]

    df = df.sort_values(
        ["under_penalty", "total_dist", "tokens_per_sec"],
        ascending=[True, True, False],
    )
    return df

def get_vcpu_count_from_gpu(quota_df, gpu_base, gpus_needed):
    """
    Given the GPU base, (example L40s), and the GPUs needed (eg 12, based on TP=4, PP=3),
    returns the vCPU Count for that given Instance Type (g4dn.12xlarge)
    """
    instances = quota_df[quota_df["gpu_base"] == gpu_base].copy()
    if instances.empty:
        return []
    packings = []
    for _, inst in instances.iterrows():
        gpu_per = inst["gpu_count"]
        # if gpu_per > gpus_needed:
        #     continue
        
        num_inst = math.ceil(gpus_needed / gpu_per)
        vcpu_needed = int(num_inst * inst["vCPU"])
        packings.append((vcpu_needed, inst["Instance_Type"], num_inst))
    
    packings.sort(key=lambda x: x[0])
    return packings

def enumerate_candidates(perf_file, quota_df, region, market, avg_input_tokens, avg_output_tokens, num_lines,
                         slo_hours, guard_frac=0.1):
    gpu_base = perf_file["gpu_base"]
    df = perf_file["df"]
    
    col = f"{region}_{market}"
    vcpu_quota = float(quota_df[quota_df["gpu_base"] == gpu_base][col].max())
    print("Detected the vCPU Quota for the GPU Type: ", gpu_base, "in the region: ", region, "and market: ", market, "is: ", vcpu_quota)
    if vcpu_quota <= 0:
        return []
    
    effective_slo_hours = slo_hours * (1.0 - guard_frac)
    job_tokens = num_lines * (avg_input_tokens + avg_output_tokens)
    candidates = []
    
    for tp in [1, 2, 4, 8]:
        for pp in [1, 2, 3, 4]:
            # select the closest match with respect to 
            hit = df[(df["tp"] == tp) & (df["pp"] == pp)]

            if hit.empty:
                print(
                    colored(
                        f"Configuration not found for TP: {tp}, PP: {pp}",
                        "red",
                        attrs=["bold"]
                    )
                )
            else:
                print(
                    colored(
                        f"Detected the perf file for TP: {tp}, PP: {pp}:\n", 
                        "cyan", attrs=["bold"]
                    )
                )
                # Tabulate and color dataframe output in cyan
                if not hit.empty:
                    headers = list(hit.columns)
                    data = [[colored(str(val), "cyan", attrs=["bold"]) if not isinstance(val, float)
                             else colored(f"{val:.4f}", "cyan", attrs=["bold"]) for val in row]
                            for _, row in hit.iterrows()]
                    print(tabulate(data, headers=headers, tablefmt="psql", stralign="left"))
            if hit.empty:
                continue
            hit=hit.copy()

            hit = find_best_perf_match_based_on_input_output_length(hit, avg_input_tokens, avg_output_tokens)
            # Pretty print the matched dataframe, and color only those rows

            # Function to color a DataFrame row
            def color_row(row, color="green"):
                row_strs = []
                for col, val in row.items():
                    if isinstance(val, float):
                        val_str = f"{val:.4f}"
                    else:
                        val_str = str(val)
                    row_strs.append(colored(val_str, color, attrs=["bold"]))
                return row_strs

            if not hit.empty:
                # Only show top 3 rows
                headers = list(hit.columns)
                top_hit = hit.head(3)
                data = [color_row(row) for _, row in top_hit.iterrows()]
                print(tabulate(data, headers=headers, tablefmt="psql", stralign="left"))
            tokens_per_sec = float(hit["tokens_per_sec"].iloc[0])

            for replicas in range(1, 50):
                gpus_needed = replicas * tp * pp
                
                # Packings sorted by vCPU (cheapest first)
                packings = get_vcpu_count_from_gpu(quota_df, gpu_base, gpus_needed)
                
                found_any = False
                for vcpu, inst_type, num_inst in packings:
                    # EARLY EXIT 1: if cheapest packing exceeds quota, skip rest
                    if vcpu > vcpu_quota:
                        break  # All remaining packings are more expensive
                    
                    runtime_hours = job_tokens / (tokens_per_sec * replicas) / 3600.0
                    if runtime_hours <= effective_slo_hours:
                        candidates.append({
                            "gpu_base": gpu_base,
                            "tp": tp, "pp": pp,
                            "replicas": replicas,
                            "gpus_needed": gpus_needed,
                            "vcpu_needed": vcpu,
                            "vcpu_remaining": vcpu_quota - vcpu,
                            "instance_type": inst_type,
                            "runtime_hours": runtime_hours,
                            "gpu_time": gpus_needed * runtime_hours,
                        })
                        # break #? is this needed?
                
                # EARLY EXIT 2: if even cheapest packing exceeded quota,
                # more replicas will only get worse
                if packings and packings[0][0] > vcpu_quota:
                    break
    
    return candidates


def enumerate_gpus_and_candidates(user_model_size_b, num_lines, avg_input_tokens, avg_output_tokens, slo_hours, region="us-east-1", market="spot",k_closest=1):
    # Step 1: Find perf files closest to user's model size
    closest_perf_files = select_perf_files_closest_to_model_size(
        perf_files, user_model_size_b, k=k_closest
    )
    print(f"[Stage A] Found {len(closest_perf_files)} GPU type(s) with perf data:")
    for pf in closest_perf_files:
        print(f"  - {pf['gpu_base']} ({pf['model_size_b']}B model)")
    # Step 2: Enumerate candidates for EACH GPU type
    all_candidates = []
    for pf in closest_perf_files:
        gpu_type = pf["gpu_base"]
        cs = enumerate_candidates(pf, quota_df, region, market, avg_input_tokens, avg_output_tokens, num_lines, slo_hours)
        print(f"[Stage A] {gpu_type}: {len(cs)} candidates")
        all_candidates.extend(cs)

    # Step 3: Math Advisor's preferences. 
    print(f"[Stage A] Total: {len(all_candidates)} candidates across all GPU types")
    return all_candidates


## Adding the LLM Advisors and the CPMI Stuff ###

### CPMI Model ###
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def load_c_pmi_model(model_name: str = "Qwen/Qwen3-0.6B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def _avg_nll(text: str, tokenizer, model) -> float:
    with torch.no_grad():
        input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs[0]
    return loss.item()

def c_pmi_score(
    context: str,
    hypothesis: str,
    tokenizer,
    model,
    sep: str = " <|endoftext|> ",
) -> float:
    lpx = -_avg_nll(context + sep + hypothesis, tokenizer, model)
    lpx_context = -_avg_nll(context, tokenizer, model)
    lpx_hyp = -_avg_nll(hypothesis, tokenizer, model)
    pmi = lpx - lpx_context - lpx_hyp
    return pmi


def c_pmi_rank_plans(
    context: str,
    plan_labels,
    tokenizer,
    model,
    temperature: float = 1.0,
):
    hypotheses = [
        f"In this situation, the best plan is {label}."
        for label in plan_labels
    ]
    scores = [
        c_pmi_score(context, hyp, tokenizer, model)
        for hyp in hypotheses
    ]
    max_s = max(scores)
    exps = [math.exp((s - max_s) / max(temperature, 1e-6)) for s in scores]
    Z = sum(exps)
    probs = [e / Z for e in exps]
    label_to_prob = {label: prob for label, prob in zip(plan_labels, probs)}
    best_label = max(label_to_prob.items(), key=lambda x: x[1])[0]
    return best_label, label_to_prob

### Ranker with LLM ###
def llm_choose_config_from_candidates(
    job: JobState,
    candidates: List[Dict[str, Any]],
    model_id: str,
    openrouter_api_key: str,
    advisor_name: str,
    top_k: int = 3,
    temperature: float = 0.7,
) -> Optional[Dict[str, Any]]:
    """
    Ask an HF LLM (Phi or other) to choose among top_k analytic candidates.
    Returns the chosen config dict or None.
    """
    if not candidates:
        return None

    # candidates_sorted = sorted(candidates, key=lambda c: c["gpu_time"])
    candidates_top = candidates[: min(top_k, len(candidates))]

    labels = ["A", "B", "C", "D", "E"]
    labeled = list(zip(labels, candidates_top))

    # client = InferenceClient(model=model_id, token=hf_token)
    # OpenRouter client setup
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    prompt_lines = []
    prompt_lines.append(
        f"You are an expert GPU scheduler ({advisor_name}) choosing tensor/pipeline "
        "parallelism for an LLM job.\n"
    )
    prompt_lines.append("Goals, in order:\n")
    prompt_lines.append("1. The job must finish within its SLO (deadline).\n")
    prompt_lines.append("2. Minimize total GPU-hours used.\n")
    prompt_lines.append("3. Prefer simpler configs (fewer TP/PP/replicas) when close.\n\n")

    prompt_lines.append("Job:\n")
    prompt_lines.append(f"- Model: {job.spec.model_name}\n")
    prompt_lines.append(f"- Lines (requests): {job.spec.num_lines}\n")
    prompt_lines.append(f"- Avg input tokens: {job.spec.avg_input_tokens}\n")
    prompt_lines.append(f"- Avg output tokens: {job.spec.avg_output_tokens}\n")
    prompt_lines.append(f"- SLO: {job.spec.slo_hours} hours\n")
    prompt_lines.append(f"- Total tokens (approx): {job.total_tokens}\n\n")

    prompt_lines.append("Candidate configs:\n")
    for label, cfg in labeled:
        prompt_lines.append(
            f"Plan {label}:\n"
            f"- tp: {cfg['tp']}\n"
            f"- pp: {cfg['pp']}\n"
            f"- replicas: {cfg['replicas']}\n"
            f"- total GPUs: {cfg['gpus_needed']}\n"
            f"- predicted runtime: {cfg['runtime_hours']:.2f} hours\n"
            f"- GPU-hours: {cfg['gpu_time']:.2f}\n\n"
        )

    prompt_lines.append(
        "Which plan best satisfies the goals? Respond with exactly one line:\n"
        "Best plan: A\n"
    )

    prompt = "".join(prompt_lines)

    # resp = client.text_generation(
    #     prompt,
    #     max_new_tokens=64,
    #     temperature=temperature,
    #     do_sample=False,
    # )
    # text = resp.strip()
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=64,
        temperature=temperature,
    )
    text = response.choices[0].message.content.strip()

    chosen_label = None
    for label, _ in labeled:
        if f"Best plan: {label}" in text or f"best plan: {label}" in text:
            chosen_label = label
            break
        if f"Plan {label}" in text or f"plan {label}" in text:
            chosen_label = label
            break

    if chosen_label is None:
        # fallback
        for label, _ in labeled:
            if f" {label}" in text:
                chosen_label = label
                break

    for label, cfg in labeled:
        if label == chosen_label:
            return cfg

    return None

def choose_and_apply_llm_plan(job_state, plans, plan_labels, cpmi_model, cpmi_tokenizer):
        """
        Given a job_state, list of plans [(label, cfg)], and plan_labels,
        builds context, queries C-PMI, chooses best config, and updates job_state in place.
        Returns (best_label, probs, chosen_cfg).
        """
        context_lines = []
        context_lines.append("We are choosing a GPU parallelism configuration for an LLM job.\n")
        context_lines.append("Goals:\n")
        context_lines.append("1. Meet the SLO (deadline).\n")
        context_lines.append("2. Minimize total GPU-hours.\n")
        context_lines.append("3. Prefer simpler configs when close.\n")
        context_lines.append("4. In VPC, also prefer lower vCPU usage when close.\n\n")

        context_lines.append("Job:\n")
        context_lines.append(f"- Model: {job_state.spec.model_name}\n")
        context_lines.append(f"- Lines: {job_state.spec.num_lines}\n")
        context_lines.append(f"- Avg input tokens: {job_state.spec.avg_input_tokens}\n")
        context_lines.append(f"- Avg output tokens: {job_state.spec.avg_output_tokens}\n")
        context_lines.append(f"- SLO: {job_state.spec.slo_hours} hours\n")
        context_lines.append(f"- Total tokens: {job_state.total_tokens}\n\n")

        context_lines.append("Advisor proposals:\n")
        for name, cfg in plans:
            context_lines.append(
                f"{name} proposes:\n"
                f"  - gpu_base: {cfg.get('gpu_base')}\n"
                f"  - tp: {cfg['tp']}\n"
                f"  - pp: {cfg['pp']}\n"
                f"  - replicas: {cfg['replicas']}\n"
                f"  - total GPUs: {cfg['gpus_needed']}\n"
                f"  - instance_type: {cfg.get('instance_type')}\n"
                f"  - vcpu_needed: {cfg.get('vcpu_needed')}\n"
                f"  - predicted runtime: {cfg['runtime_hours']:.2f} hours\n"
                f"  - GPU-hours: {cfg['gpu_time']:.2f}\n\n"
            )

        context_str = "".join(context_lines)
        print(context_str)

        best_label, probs = c_pmi_rank_plans(
            context=context_str,
            plan_labels=plan_labels,
            tokenizer=c_pmi_tokenizer,
            model=c_pmi_model,
        )

        print("[C-PMI] probs:", probs)
        print("[C-PMI] winner:", best_label)

        # Choose config by label
        chosen_cfg = next(cfg for name, cfg in plans if name == best_label)

        # Update JobState (minimal fields for testing)
        job_state.gpu_base = chosen_cfg.get("gpu_base")
        job_state.tp = chosen_cfg["tp"]
        job_state.pp = chosen_cfg["pp"]
        job_state.replicas = chosen_cfg["replicas"]
        job_state.allocated_gpus = chosen_cfg["gpus_needed"]
        job_state.vcpu_needed = chosen_cfg.get("vcpu_needed", 0)
        job_state.instance_type = chosen_cfg.get("instance_type")
        job_state.num_instances = chosen_cfg.get("num_instances", 0)

        print("[FINAL] chosen:", chosen_cfg)
        print("[FINAL] job_state:", job_state)

        return best_label, probs, chosen_cfg
    

# # ============== QUICK TEST ==============

if __name__ == "__main__":
    perf_files = load_all_perfdb_files(perfdb_dir)
    closest_perf_file = select_perf_files_closest_to_model_size(perf_files, 60.0, 1)
    quota_df = load_quota_csv(quota_csv)

    print("\n" + "="*60)
    print("TESTING get_vcpu_count_from_gpu()")
    print("="*60)
    
    # Test 1: Get vCPU packings for 12 GPUs of L40S
    test_packings = get_vcpu_count_from_gpu(quota_df, "L40S", 12)
    print(f"\nNeed 12 L40S GPUs, possible packings:")
    for vcpu, inst, num in test_packings[:]:  # show top 5
        print(f"  {vcpu:4d} vCPU  {num:2d}x {inst}")
    
    
    print("\n" + "="*60)
    print("TESTING enumerate_candidates()")
    print("="*60)
    
    # Test 2: Enumerate candidates for a simple job
    if perf_files:
        test_num_lines = 20000
        test_avg_input_tokens = 4096
        test_avg_output_tokens = 2048 # perfect?
        test_slo = 10 #hours
        job_spec = JobSpec(
            job_id="test_job",
            model_name="llama-3.3-60b",
            num_lines=test_num_lines,   
            avg_input_tokens=test_avg_input_tokens,
            avg_output_tokens=test_avg_output_tokens,
            slo_hours=test_slo,
            region="us-east-1",
            market="on_demand"
        )
        job_state = JobState(
            spec=job_spec,
            submitted_at=time.time()
        )
        print(f"\nJob: {test_avg_input_tokens:,} input tokens, {test_avg_output_tokens:,} output tokens, SLO={test_slo}h")
        
        # NEW: Use the orchestrator (handles ALL GPU types)
        candidates = enumerate_gpus_and_candidates(
            user_model_size_b=60.0,  # need to automatically extract this in the future
            num_lines=test_num_lines,
            avg_input_tokens=test_avg_input_tokens,
            avg_output_tokens=test_avg_output_tokens,
            slo_hours=test_slo,
            region="us-east-1",
            market="on_demand"
        )
        candidates_sorted = sorted(candidates, key=lambda c: (c["replicas"], c["vcpu_needed"], c["runtime_hours"]))
        top_candidates = candidates_sorted[:20]
        for i, c in enumerate(top_candidates):
            print(f"  {i+1}. {c['gpu_base']} tp={c['tp']} pp={c['pp']} "
                    f"r={c['replicas']} gpus={c['gpus_needed']} "
                    f"runtime={c['runtime_hours']:.2f}h gpu-h={c['gpu_time']:.1f}")

        # Math baseline (like notebook): minimal GPU-hours
        math_cfg = min(top_candidates, key=lambda c: (c["replicas"], c["vcpu_needed"], c["runtime_hours"]))
        print(f"[Math] tp={math_cfg['tp']} pp={math_cfg['pp']} r={math_cfg['replicas']} gpu-h={math_cfg['gpu_time']:.2f}")
        mimo_cfg = llm_choose_config_from_candidates(
            job=job_state,
            candidates=top_candidates,
            model_id="xiaomi/mimo-v2-flash:free",
            openrouter_api_key=OPENROUTER_API_KEY,
            advisor_name="XiaomiMimoAdvisor",
            top_k=20)
        if mimo_cfg:
            print(f"[XiaomiMimo Advisor] Picks: tp={mimo_cfg['tp']} pp={mimo_cfg['pp']} "
                    f"r={mimo_cfg['replicas']}")
        devstral_cfg = llm_choose_config_from_candidates(
            job=job_state,
            candidates=top_candidates,
            model_id="mistralai/devstral-2512:free",
            openrouter_api_key=OPENROUTER_API_KEY,
            advisor_name="DevstralAdvisor",
            top_k=20)
        if devstral_cfg:
            print(f"[Devstral Advisor] Picks: tp={devstral_cfg['tp']} pp={devstral_cfg['pp']} "
                    f"r={devstral_cfg['replicas']}")    
        plans = [("Math", math_cfg), ("XiaomiMimo", mimo_cfg), ("Devstral", devstral_cfg)]
        plan_labels = [name for name, _ in plans]
        c_pmi_model, c_pmi_tokenizer = load_c_pmi_model()
        # Use the new function
        best_label, probs, chosen_cfg = choose_and_apply_llm_plan(job_state, plans, plan_labels, c_pmi_model, c_pmi_tokenizer)
        print("No perf files found!")