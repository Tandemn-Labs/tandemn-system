import pytest
from utils.utils import (
    split_uri,
    get_num_params_from_text,
    load_aws_quota_csv,
    load_all_perfdb_files,
    select_perf_files_closest_to_model_size,
    sort_perf_entries_io_length,
    update_yaml_file,
    update_template,
)


# --- split_uri ---


def test_split_uri_s3():
    assert split_uri("s3://tandemn/test/input.txt") == ("s3://tandemn", "test/input.txt")


def test_split_uri_deep_path():
    assert split_uri("s3://b/a/b/c/d.txt") == ("s3://b", "a/b/c/d.txt")


def test_split_uri_no_path():
    assert split_uri("s3://bucket") == ("s3://bucket", "")


def test_split_uri_gs():
    assert split_uri("gs://my-bucket/folder/file.json") == (
        "gs://my-bucket",
        "folder/file.json",
    )


def test_split_uri_invalid_raises():
    with pytest.raises(ValueError, match="missing scheme"):
        split_uri("no-scheme/path")


# --- get_num_params_from_text ---


def test_get_num_params_70b():
    assert get_num_params_from_text("Meta-Llama-3-70B-Instruct") == 70.0


def test_get_num_params_8b():
    assert get_num_params_from_text("Llama-3-8B") == 8.0


def test_get_num_params_decimal():
    assert get_num_params_from_text("Qwen2.5-32B") == 32.0


def test_get_num_params_no_match():
    assert get_num_params_from_text("gpt-4") is None


def test_get_num_params_none():
    assert get_num_params_from_text(None) is None


def test_get_num_params_empty():
    assert get_num_params_from_text("") is None


# --- load_aws_quota_csv ---


def test_load_aws_quota_csv(sample_quota_csv_path):
    df = load_aws_quota_csv(sample_quota_csv_path)
    assert "gpu_base" in df.columns
    assert "gpu_count" in df.columns
    # Check normalization: "4x L40S" → "L40S"
    assert "L40S" in df["gpu_base"].values
    # Check count extraction: "4x L40S" → 4.0
    row = df[df["Instance_Type"] == "g6e.12xlarge"].iloc[0]
    assert row["gpu_count"] == 4.0


# --- load_all_perfdb_files ---


def test_load_all_perfdb_files(sample_perfdb_dir):
    results = load_all_perfdb_files(sample_perfdb_dir)
    assert len(results) == 1
    entry = results[0]
    assert entry["gpu_base"] == "L40S"
    assert entry["model_size_b"] == 70.0
    assert "df" in entry
    assert len(entry["df"]) == 2


# --- select_perf_files_closest_to_model_size ---


def test_select_perf_closest(sample_perfdb_dir):
    results = load_all_perfdb_files(sample_perfdb_dir)
    # Add a fake entry for diversity
    import copy

    fake = copy.deepcopy(results[0])
    fake["model_size_b"] = 8.0
    all_files = results + [fake]
    selected = select_perf_files_closest_to_model_size(all_files, model_size_b=65.0, k=1)
    assert len(selected) == 1
    assert selected[0]["model_size_b"] == 70.0


# --- sort_perf_entries_io_length ---


def test_sort_perf_entries_prefers_no_under_penalty(sample_perfdb_dir):
    import pandas as pd

    df = pd.DataFrame(
        {
            "max_input_length": [2048, 8192],
            "max_output_length": [512, 1024],
            "tokens_per_sec": [100.0, 200.0],
        }
    )
    sorted_df = sort_perf_entries_io_length(df, job_avg_input=4096, job_avg_output=800)
    # Row with 8192/1024 has no under_penalty; row with 2048/512 has under_penalty
    assert sorted_df.iloc[0]["max_input_length"] == 8192


# --- update_yaml_file ---


def test_update_yaml_flat(tmp_path):
    yaml_in = tmp_path / "in.yaml"
    yaml_out = tmp_path / "out.yaml"
    yaml_in.write_text("name: old\ncount: 1\n")
    result = update_yaml_file(str(yaml_in), {"name": "new"}, str(yaml_out))
    assert result["name"] == "new"
    assert result["count"] == 1


def test_update_yaml_nested(tmp_path):
    yaml_in = tmp_path / "in.yaml"
    yaml_out = tmp_path / "out.yaml"
    yaml_in.write_text("resources:\n  instance_type: old\n  disk: 50\n")
    result = update_yaml_file(
        str(yaml_in), {"resources.instance_type": "g6e.12xlarge"}, str(yaml_out)
    )
    assert result["resources"]["instance_type"] == "g6e.12xlarge"
    assert result["resources"]["disk"] == 50


# --- update_template ---


def test_update_template(tmp_path):
    tmpl = tmp_path / "tmpl.txt"
    tmpl.write_text("Hello {name}, you have {count} items.")
    result = update_template(str(tmpl), {"name": "Alice", "count": "3"})
    assert result == "Hello Alice, you have 3 items."
