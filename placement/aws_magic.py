import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from models.resources import MagicOutput
import pandas as pd

from openai import OpenAI
from pydantic import BaseModel
from tabulate import tabulate
from termcolor import colored
import torch
from models.requests import BatchedRequest, OnlineServingRequest
from placement.magic import VPCMagic
from utils.utils import (
    get_num_params_from_text,
    get_vcpu_count_from_gpu,
    load_all_perfdb_files,
    load_aws_quota_csv,
    select_perf_files_closest_to_model_size,
    sort_perf_entries_io_length,
)
from transformers import AutoTokenizer, AutoModelForCausalLM


class AWSPlacementCandidate(BaseModel):
    """Configuration candidate for GPU allocation"""

    gpu_base: str
    tp: int
    pp: int
    replicas: int
    num_inst: int
    vcpu_needed: int
    instance_type: str
    runtime_hours: float
    gpu_time: float


class AWSAllocation(VPCMagic):
    """Implementation of VPCMagic for AWS"""

    def __init__(
        self,
        openrouter_key,
        perfdb_dir="./perfdb",
        aws_quota_csv="./quotas/aws_gpu_quota_by_region.csv",
        k_nearest_model_size=1,
    ):
        self.perfdb_dir = perfdb_dir
        self.aws_quota_csv = aws_quota_csv
        self.quota_df = load_aws_quota_csv(self.aws_quota_csv)
        self.k_nearest_model_size = k_nearest_model_size
        self.openrouter_key = openrouter_key

    def decide(
        self, request: Union[BatchedRequest, OnlineServingRequest]
    ) -> MagicOutput:
        """Main decision function called by server"""

        if isinstance(request, BatchedRequest):
            candidate = self.process_batch(request)
            return MagicOutput(
                decision_id=f"mo-{uuid.uuid4()}",
                engine="vllm",
                instance_type=candidate.instance_type,
                num_inst=candidate.num_inst,
                tp_size=candidate.tp,
                pp_size=candidate.pp,
                replicas=candidate.replicas,
            )

    def process_batch(
        self, req: BatchedRequest, region="us-east-1", market="on_demand"
    ) -> AWSPlacementCandidate:
        # Load inputs: Performance DB + AWS Quota
        model_size = get_num_params_from_text(req.model_name)
        perf_files = load_all_perfdb_files(self.perfdb_dir)
        closest_perf_files = select_perf_files_closest_to_model_size(
            perf_files, model_size, self.k_nearest_model_size
        )

        print(f"[Stage A] Found {len(closest_perf_files)} GPU type(s) with perf data:")
        for pf in closest_perf_files:
            print(f"  - {pf['gpu_base']} ({pf['model_size_b']}B model)")

        # Generate candidates for each perf file
        # Look at enumerate_candidates() for details
        all_candidates = []
        for pf in closest_perf_files:
            gpu_type = pf["gpu_base"]
            cs = AWSAllocation.enumerate_candidates(
                pf,
                self.quota_df,
                region,
                market,
                req.avg_input_tokens,
                req.avg_output_tokens,
                req.num_lines,
                req.slo_deadline_hours,
            )
            print(f"[Stage A] {gpu_type}: {len(cs)} candidates")
            all_candidates.extend(cs)
        print(f"[Stage A] Total: {len(all_candidates)} candidates across all GPU types")

        # Now evaluate the candidates, math and LLMs
        plans = []

        # [Evaluate candidate]: Math
        math_cfg = min(
            all_candidates,
            key=lambda c: (c.replicas, c.vcpu_needed, c.runtime_hours),
        )
        print(
            f"[Math] tp={math_cfg.tp} pp={math_cfg.pp} r={math_cfg.replicas} gpu-h={math_cfg.gpu_time:.2f}"
        )
        plans.append(("Math", math_cfg))

        # [Evaluate candidate]: LLM (Mimo)
        mimo_cfg = AWSAllocation.llm_choose_config_from_candidates(
            candidates=all_candidates,
            model_id="z-ai/glm-4.5-air:free",
            openrouter_api_key=self.openrouter_key,
            advisor_name="XiaomiMimoAdvisor",
            req=req,
            top_k=20,
        )
        if mimo_cfg:
            print(
                f"[XiaomiMimo Advisor] Picks: tp={mimo_cfg.tp} pp={mimo_cfg.pp} r={mimo_cfg.replicas}"
            )
            plans.append(("XiaomiMimo", mimo_cfg))

        # [Evaluate candidate]: LLM (Devstral)
        devstral_cfg = AWSAllocation.llm_choose_config_from_candidates(
            candidates=all_candidates,
            model_id="nvidia/nemotron-3-nano-30b-a3b:free",
            openrouter_api_key=self.openrouter_key,
            advisor_name="DevstralAdvisor",
            req=req,
            top_k=20,
        )
        if devstral_cfg:
            print(
                f"[Devstral Advisor] Picks: tp={devstral_cfg.tp} pp={devstral_cfg.pp} r={devstral_cfg.replicas}"
            )
            plans.append(("Devstral", devstral_cfg))

        def load_c_pmi_model(model_name: str = "Qwen/Qwen3-0.6B"):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16
            )
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(DEVICE)
            model.eval()
            return model, tokenizer

        c_pmi_model, c_pmi_tokenizer = load_c_pmi_model()

        plan_labels = [name for name, _ in plans]
        _, _, chosen_cfg = AWSAllocation.choose_and_apply_llm_plan(
            req, plans, plan_labels, c_pmi_model, c_pmi_tokenizer
        )
        return chosen_cfg

    @staticmethod
    def enumerate_candidates(
        perf_file: Dict[str, Any],
        quota_df: pd.DataFrame,
        region: str,
        market: str,
        avg_input_tokens: int,
        avg_output_tokens: int,
        num_lines: int,
        slo_hours: float,
        guard_frac: float = 0.1,
    ) -> List[AWSPlacementCandidate]:
        """
        Extract vCPU quota for specific GPU, Region, Spot / On-Demand
        Reads given ONE perf file
        For each TP + PP combination:
        - Check perf file for this particular combination, sort based on i/o
          length and how close it is to our particular case, choose first one as
          the assumed value for TPS
        - Now for each replica:
            - Compute total devices and number of what instances needed
            - Keep all candidates where vCPU is within total quota for that fam
        """
        gpu_base = perf_file["gpu_base"]
        df = perf_file["df"]

        col = f"{region}_{market}"
        vcpu_quota = float(
            quota_df[quota_df["gpu_base"] == gpu_base][col].max()
        )  # TODO: Why is the quota a float?
        print(
            "Detected the vCPU Quota for the GPU Type: ",
            gpu_base,
            "in the region: ",
            region,
            "and market: ",
            market,
            "is: ",
            vcpu_quota,
        )
        if vcpu_quota <= 0:
            return []

        effective_slo_hours = slo_hours * (1.0 - guard_frac)
        job_tokens = num_lines * (avg_input_tokens + avg_output_tokens)
        candidates = []

        for tp in [1, 2, 4, 8]:
            for pp in [1, 2, 3, 4]:
                # select the closest match with respect to
                hit = df[(df["tp"] == tp) & (df["pp"] == pp)]

                # Printing each hit for each valid (TP + PP) found
                if hit.empty:
                    print(
                        colored(
                            f"Configuration not found for TP: {tp}, PP: {pp}",
                            "red",
                            attrs=["bold"],
                        )
                    )
                else:
                    # Tabulate and color dataframe output in cyan
                    print(
                        colored(
                            f"Detected the entries for TP: {tp}, PP: {pp}:\n",
                            "cyan",
                            attrs=["bold"],
                        )
                    )

                    headers = list(hit.columns)
                    data = [
                        [
                            colored(str(val), "cyan", attrs=["bold"])
                            if not isinstance(val, float)
                            else colored(f"{val:.4f}", "cyan", attrs=["bold"])
                            for val in row
                        ]
                        for _, row in hit.iterrows()
                    ]
                    print(
                        tabulate(
                            data, headers=headers, tablefmt="psql", stralign="left"
                        )
                    )

                if hit.empty:
                    continue

                # Actually take the hit
                hit = hit.copy()
                hit = sort_perf_entries_io_length(
                    hit, avg_input_tokens, avg_output_tokens
                )
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

                # Only show top 3 rows
                headers = list(hit.columns)
                top_hit = hit.head(3)
                data = [color_row(row) for _, row in top_hit.iterrows()]
                print(tabulate(data, headers=headers, tablefmt="psql", stralign="left"))
                tokens_per_sec = float(hit["tokens_per_sec"].iloc[0])

                for replicas in range(1, 50):
                    gpus_needed = replicas * tp * pp

                    # Packings sorted by vCPU (cheapest first)
                    packings = get_vcpu_count_from_gpu(
                        quota_df, gpu_base, tp, pp, replicas
                    )

                    for vcpu, inst_type, num_inst in packings:
                        # EARLY EXIT 1: if cheapest packing exceeds quota, skip rest
                        if vcpu > vcpu_quota:
                            break  # All remaining packings are more expensive

                        runtime_hours = (
                            job_tokens / (tokens_per_sec * replicas)
                        ) / 3600.0
                        if runtime_hours <= effective_slo_hours:
                            candidates.append(
                                AWSPlacementCandidate(
                                    gpu_base=gpu_base,
                                    tp=tp,
                                    pp=pp,
                                    replicas=replicas,
                                    num_inst=num_inst,
                                    vcpu_needed=vcpu,
                                    instance_type=inst_type,
                                    runtime_hours=runtime_hours,
                                    gpu_time=gpus_needed * runtime_hours,
                                )
                            )

        return candidates

    @staticmethod
    def llm_choose_config_from_candidates(
        candidates: List[AWSPlacementCandidate],
        model_id: str,
        openrouter_api_key: str,
        advisor_name: str,
        req: BatchedRequest,
        top_k: int = 3,
        temperature: float = 0.7,
    ) -> Optional[AWSPlacementCandidate]:
        """
        Ask an HF LLM (Phi or other) to choose among top_k analytic candidates.
        Returns the chosen config or None.
        """

        # Process candidates and label them
        # TODO: limited to the 26 characters of the alphabet lol
        if not candidates:
            return None
        candidates_top = candidates[: min(top_k, len(candidates))]
        labels = [chr(ord("A") + i) for i in range(len(candidates_top))]
        labeled = list(zip(labels, candidates_top))

        # Making the prompt
        prompt_lines = []
        prompt_lines.append(
            f"You are an expert GPU scheduler ({advisor_name}) choosing tensor/pipeline "
            "parallelism for an LLM job.\n"
        )
        prompt_lines.append("Goals, in order:\n")
        prompt_lines.append("1. The job must finish within its SLO (deadline).\n")
        prompt_lines.append("2. Minimize total GPU-hours used.\n")
        prompt_lines.append(
            "3. Prefer simpler configs (fewer TP/PP/replicas) when close.\n\n"
        )

        prompt_lines.append("Job:\n")
        prompt_lines.append(f"- Model: {req.model_name}\n")
        prompt_lines.append(f"- Lines (requests): {req.num_lines}\n")
        prompt_lines.append(f"- Avg input tokens: {req.avg_input_tokens}\n")
        prompt_lines.append(f"- Avg output tokens: {req.avg_output_tokens}\n")
        prompt_lines.append(f"- SLO: {req.slo_deadline_hours} hours\n")
        prompt_lines.append(
            f"- Total tokens (approx): {req.num_lines * (req.avg_input_tokens + req.avg_output_tokens)}\n\n"
        )

        prompt_lines.append("Candidate configs:\n")
        for label, cfg in labeled:
            total_gpus = cfg.tp * cfg.pp * cfg.replicas
            prompt_lines.append(
                f"Plan {label}:\n"
                f"- tp: {cfg.tp}\n"
                f"- pp: {cfg.pp}\n"
                f"- replicas: {cfg.replicas}\n"
                f"- total GPUs: {total_gpus}\n"
                f"- predicted runtime: {cfg.runtime_hours:.2f} hours\n"
                f"- GPU-hours: {cfg.gpu_time:.2f}\n\n"
            )

        prompt_lines.append(
            "Which plan best satisfies the goals? Respond with exactly one line:\n"
            "Best plan: A\n"
        )
        prompt = "".join(prompt_lines)

        # OpenRouter client setup and calling
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )

        # TODO: Temp since I don't want to run OpenRouter calls
        # response = client.chat.completions.create(
        #     model=model_id,
        #     messages=[{"role": "user", "content": prompt}],
        #     max_tokens=64,
        #     temperature=temperature,
        # )
        # text = response.choices[0].message.content.strip()
        text = ""
        print(advisor_name, f": {text}")

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

        for label, cfg in labeled:  # TODO: Do offset from 'A' instead
            if label == chosen_label:
                return cfg

        return None

    @staticmethod
    def _avg_nll(text: str, tokenizer, model) -> float:
        with torch.no_grad():
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs[0]
        return loss.item()

    @staticmethod
    def c_pmi_score(
        context: str,
        hypothesis: str,
        tokenizer,
        model,
        sep: str = " <|endoftext|> ",
    ) -> float:
        _avg_nll = AWSAllocation._avg_nll
        lpx = -_avg_nll(context + sep + hypothesis, tokenizer, model)
        lpx_context = -_avg_nll(context, tokenizer, model)
        lpx_hyp = -_avg_nll(hypothesis, tokenizer, model)
        pmi = lpx - lpx_context - lpx_hyp
        return pmi

    @staticmethod
    def c_pmi_rank_plans(
        context: str,
        plan_labels,
        tokenizer,
        model,
        temperature: float = 1.0,
    ):
        hypotheses = [
            f"In this situation, the best plan is {label}." for label in plan_labels
        ]
        scores = [
            AWSAllocation.c_pmi_score(context, hyp, tokenizer, model)
            for hyp in hypotheses
        ]
        max_s = max(scores)
        exps = [math.exp((s - max_s) / max(temperature, 1e-6)) for s in scores]
        Z = sum(exps)
        probs = [e / Z for e in exps]
        label_to_prob = {label: prob for label, prob in zip(plan_labels, probs)}
        best_label = max(label_to_prob.items(), key=lambda x: x[1])[0]
        return best_label, label_to_prob

    @staticmethod
    def choose_and_apply_llm_plan(
        req: BatchedRequest,
        plans: List[Tuple[str, AWSPlacementCandidate]],
        plan_labels: List[str],
        cpmi_model: Any,
        cpmi_tokenizer: Any,
    ) -> Tuple[str, Dict[str, float], AWSPlacementCandidate]:
        """
        Given a job_state, list of plans [(label, cfg)], and plan_labels,
        builds context, queries C-PMI, chooses best config, and updates job_state in place.
        Returns (best_label, probs, chosen_cfg).
        """
        context_lines = []
        context_lines.append(
            "We are choosing a GPU parallelism configuration for an LLM job.\n"
        )
        context_lines.append("Goals:\n")
        context_lines.append("1. Meet the SLO (deadline).\n")
        context_lines.append("2. Minimize total GPU-hours.\n")
        context_lines.append("3. Prefer simpler configs when close.\n")
        context_lines.append("4. In VPC, also prefer lower vCPU usage when close.\n\n")

        context_lines.append("Job:\n")
        context_lines.append(f"- Model: {req.model_name}\n")
        context_lines.append(f"- Lines: {req.num_lines}\n")
        context_lines.append(f"- Avg input tokens: {req.avg_input_tokens}\n")
        context_lines.append(f"- Avg output tokens: {req.avg_output_tokens}\n")
        context_lines.append(f"- SLO: {req.slo_deadline_hours} hours\n")
        context_lines.append(
            f"- Total tokens: {req.num_lines * (req.avg_input_tokens + req.avg_output_tokens)}\n\n"
        )

        context_lines.append("Advisor proposals:\n")

        print(plans)
        for name, cfg in plans:
            gpu_base = cfg.gpu_base
            tp = cfg.tp
            pp = cfg.pp
            replicas = cfg.replicas
            total_gpus = tp * pp * replicas
            instance_type = cfg.instance_type
            vcpu_needed = cfg.vcpu_needed
            runtime_hours = cfg.runtime_hours
            gpu_time = cfg.gpu_time
            context_lines.append(
                f"{name} proposes:\n"
                f"  - gpu_base: {gpu_base}\n"
                f"  - tp: {tp}\n"
                f"  - pp: {pp}\n"
                f"  - replicas: {replicas}\n"
                f"  - total GPUs: {total_gpus}\n"
                f"  - instance_type: {instance_type}\n"
                f"  - vcpu_needed: {vcpu_needed}\n"
                f"  - predicted runtime: {runtime_hours:.2f} hours\n"
                f"  - GPU-hours: {gpu_time:.2f}\n\n"
            )

        context_str = "".join(context_lines)
        print(context_str)

        best_label, probs = AWSAllocation.c_pmi_rank_plans(
            context=context_str,
            plan_labels=plan_labels,
            tokenizer=cpmi_tokenizer,
            model=cpmi_model,
        )

        print("[C-PMI] probs:", probs)
        print("[C-PMI] winner:", best_label)

        # Choose config by label
        chosen_cfg = next(cfg for name, cfg in plans if name == best_label)

        print("[FINAL] chosen:", chosen_cfg)

        return best_label, probs, chosen_cfg
