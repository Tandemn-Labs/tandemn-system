"""Smoke tests: every new module imports cleanly."""


def test_config_importable():
    from config import AWS_INSTANCES, VLLM_PORT
    assert len(AWS_INSTANCES) > 0
    assert VLLM_PORT == 8001


def test_input_parser_importable():
    from input_parser import estimate_tokens, extract_prompt_text, parse_input_file_stats
    assert callable(estimate_tokens)
    assert callable(extract_prompt_text)
    assert callable(parse_input_file_stats)


def test_job_manager_importable():
    from job_manager import (
        setup_job_logger, close_job_logger,
        get_cluster_manager, get_job_tracker,
        JobTracker, ClusterManager,
    )
    assert callable(setup_job_logger)


def test_templates_importable():
    from templates import (
        get_vllm_config_template, replace_run_vllm,
        replace_run_vllm_online, real_magic,
    )
    assert callable(real_magic)


def test_launcher_importable():
    from launcher import (
        sp_launch_vllm_batch_with_fallback,
        sp_launch_vllm_batch,
        sp_launch_vllm_online,
    )
    assert callable(sp_launch_vllm_batch)


def test_server_app_importable():
    from server import app
    assert app is not None
