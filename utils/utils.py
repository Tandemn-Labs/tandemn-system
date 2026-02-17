import math
import re
import yaml
import pandas as pd
from pathlib import Path
from typing import Any, Dict


def update_yaml_file(
    yaml_path: str, replace_dict: Dict[str, Any], output_path: str
) -> Dict[str, Any]:
    """
    Read a YAML file and update specified fields at runtime.

    Args:
        yaml_path: Path to the YAML file
        replace_dict: Dict of keys to replace in the YAML and the values

    Returns:
        The updated YAML data as a dictionary
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Loop through dict of items to replace
    for key, value in replace_dict.items():
        # Handles nested keys, like "resources.instance_type"
        if "." in key:
            keys = key.split(".")
            current = data
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            data[key] = value

    # Write the updated YAML back (create parent directory if needed)
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return data


def update_template(template_file: str, replace_dict: Dict[str, str]) -> str:
    """
    Read a template file and replace placeholders with values from a dictionary.

    Args:
        template_file: Path to the template file containing format placeholders
        replace_dict: Dictionary mapping placeholder names to replacement values.
                      Keys should match the placeholder names in the template
                      (e.g., {"name": "value"} for {name} in template)

    Returns:
        The formatted string with all placeholders replaced by their
        corresponding values from replace_dict
    """
    tmpl = Path(template_file).read_text(encoding="utf-8")
    script = tmpl.format(**replace_dict)

    return script


def split_uri(uri: str) -> tuple[str, str]:
    """
    Split a cloud bucket URI into bucket (with scheme) and object path.

    Args:
        uri: A URI string like "s3://tandemn/test/input.txt"

    Returns:
        A tuple of (bucket_uri, object_path) where:
        - bucket_uri: The scheme and bucket name (e.g., "s3://tandemn")
        - object_path: The path within the bucket (e.g., "test/input.txt")

    Examples:
        >>> split_uri("s3://tandemn/test/input.txt")
        ('s3://tandemn', 'test/input.txt')
        >>> split_uri("gs://my-bucket/folder/file.json")
        ('gs://my-bucket', 'folder/file.json')
    """
    # Find the scheme separator "://"
    scheme_end = uri.find("://")
    if scheme_end == -1:
        raise ValueError("Invalid URI format: missing scheme (expected '://')")

    # Find the first slash after the bucket name
    # Start searching after "://" + bucket name
    bucket_end = uri.find("/", scheme_end + 3)
    if bucket_end == -1:
        # No path after bucket, return bucket and empty string
        return uri, ""

    bucket_uri = uri[:bucket_end]
    object_path = uri[bucket_end + 1 :]  # +1 to skip the slash

    return bucket_uri, object_path


#################### Performance DB & AWS vCPU Quota processing: ####################


def get_num_params_from_text(model_name):
    """
    This is just a hack for now, in the future it should be
    a function that gets it from Huggingface model card.
    """
    if not model_name:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)[bB](?:illion)?", model_name)
    return float(m.group(1)) if m else None


def load_all_perfdb_files(perfdb_dir):
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
    perfdb_dir = Path(perfdb_dir)
    perfdb_files = [p for p in perfdb_dir.rglob("*") if p.is_file()]
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
        models = set(
            df["model_name"].unique()
        )  # TODO: Do we have multiple models per file?
        model_size_b = get_num_params_from_text(df["model_name"].iloc[0])
        print("Model size b:", model_size_b)
        all_results.append(
            {
                "path": path,
                "gpu_base": gpu_base,
                "models": models,
                "model_size_b": model_size_b,
                "df": df,
            }
        )
    return all_results


def select_perf_files_closest_to_model_size(perf_files, model_size_b, k=1):
    """
    Selects the k perf files closest to the model size.
    """
    print("Perf files that are closest to the model size: ", len(perf_files))
    print("Model size that is closest to the model you submitted", model_size_b)
    print("K:", k)
    return sorted(perf_files, key=lambda x: abs(x["model_size_b"] - model_size_b))[:k]


def load_aws_quota_csv(
    quota_csv,
):  # TODO: Why not clean the CSV instead of parsing it in this function?
    """
    Loads the quota csv into a dataframe.
    """

    def normalize_gpu_name(s_name):
        if s_name is None:
            return ""
        s_name = str(s_name).strip()
        s_name = re.sub(r"^\s*\d+(\.\d+)?\s*x\s*", "", s_name)  # drop leading '4x '
        return s_name.strip()

    df = pd.read_csv(quota_csv)
    df["gpu_base"] = df["GPU_Type"].apply(normalize_gpu_name)
    df["gpu_count"] = (
        df["GPU_Type"].str.extract(r"(\d+)\s*x\s*")[0].fillna(1.0).astype(float)
    )  # get the 4 in 4XA100
    print("Detected the GPU Types in your Quota: ", df["gpu_base"].unique())
    print("Detected the GPU Count: ", df["gpu_count"].unique())
    return df


def sort_perf_entries_io_length(df, job_avg_input, job_avg_output):
    """
    Given a set of rows, sort based on under penalty, total distance, and
    tokens / second (in that order)
    - Under penalty: How much perf's max io length < job_avg_io
    - Total distance: Sum of abs of difference
    Note: The same object gets sorted in-place
    """
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


def get_vcpu_count_from_gpu(
    quota_df, gpu_base, gpus_needed, prefer_single_instance=False
):
    """
    Given the GPU base and parallelism configuration (tp, pp, replicas),
    filters instances where the GPU count matches the TP number.
    Returns list: [(vCPU count, Instance Type, Num of instances needed), ...]

    If prefer_single_instance=True, prioritize instances where num_inst=1.
    This is important for TP>1 because tensor parallelism requires high-bandwidth
    intra-node communication (NVLink/PCIe). PP can work inter-node since it only
    passes activations between pipeline stages.
    """
    # Filter by GPU base and GPU count matching TP
    instances = quota_df[
        (quota_df["gpu_base"] == gpu_base) & (quota_df["gpu_count"] == tp)
    ].copy()
    if instances.empty:
        return []
    packings = []
    for _, inst in instances.iterrows():
        # Number of instances needed = replicas * pp
        # (each instance has tp GPUs)
        num_inst = replicas * pp
        vcpu_needed = int(num_inst * inst["vCPU"])
        packings.append((vcpu_needed, inst["Instance_Type"], num_inst))

    if prefer_single_instance:
        # Sort by: (1) prefer single instance, (2) then by vCPU cost
        packings.sort(key=lambda x: (x[2] > 1, x[0]))
    else:
        packings.sort(key=lambda x: x[0])
    return packings
