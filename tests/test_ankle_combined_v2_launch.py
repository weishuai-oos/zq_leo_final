"""GPU-free launch checks for the Combined to CombinedV2 continuation job."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_ankle_combined_v2.slurm"
OPTIONS = (
    "T800_PROJECT_ROOT", "T800_PYTHON", "T800_GPUS", "NUM_ENVS", "SEED",
    "MAX_ITERATIONS", "ADDITIONAL_ITERATIONS", "SAVE_INTERVAL", "RUN_NAME",
    "CHECKPOINT", "CHECKPOINT_PATH", "SLURM_GPUS_ON_NODE", "FINETUNE_LR_SCALE",
    "FINETUNE_WARMUP_ITERATIONS", "FINETUNE_ORIGIN_ITERATION",
    "T800_TEST_FAIL_STAGE", "T800_TEST_CALLS",
)


def run(tmp_path, *args, **overrides):
    env = os.environ.copy()
    for key in OPTIONS:
        env.pop(key, None)
    env.update(overrides)
    return subprocess.run(
        ["bash", str(SCRIPT), *args], cwd=tmp_path, env=env,
        text=True, capture_output=True, timeout=20,
    )


def fake_python(tmp_path):
    executable = tmp_path / "fake python"
    calls = tmp_path / "calls.jsonl"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['T800_TEST_CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps(args) + '\\n')\n"
        "stage = 'cuda' if args[0] == '-' else 'smoke' if 'scripts/check_config.py' in args else 'train'\n"
        "if os.environ.get('T800_TEST_FAIL_STAGE') == stage:\n"
        "    raise SystemExit({'cuda': 19, 'smoke': 23, 'train': 41}[stage])\n"
    )
    executable.chmod(0o755)
    return executable, calls


def test_default_preview_resumes_combined_100000_for_40000_updates(tmp_path):
    result = run(tmp_path, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "Task: T800_base-AnkleCombinedV2" in result.stdout
    assert "4096 global / 2048 per GPU" in result.stdout
    assert "--nproc_per_node=2" in result.stdout
    assert "ckpt_100000.pt" in result.stdout
    assert "--additional_iterations 40000" in result.stdout
    assert "--max_iterations" not in result.stdout
    assert "--save_interval 500" in result.stdout
    assert "--finetune_lr_scale 0.5" in result.stdout
    assert "--finetune_warmup_iterations 1000" in result.stdout


def test_preview_checks_inputs_without_starting_python(tmp_path):
    executable, calls = fake_python(tmp_path)
    result = run(
        tmp_path, "--dry-run", T800_PYTHON=str(executable), T800_TEST_CALLS=str(calls),
        NUM_ENVS="4097",
    )
    assert result.returncode == 2
    assert "divisible" in result.stderr
    assert not calls.exists()


@pytest.mark.parametrize("name,value", [
    ("ADDITIONAL_ITERATIONS", "0"), ("SAVE_INTERVAL", "0500"),
    ("FINETUNE_LR_SCALE", "0.0"), ("FINETUNE_LR_SCALE", "1.1"),
    ("FINETUNE_WARMUP_ITERATIONS", "-1"), ("FINETUNE_ORIGIN_ITERATION", "bad"),
])
def test_invalid_budget_or_finetune_parameters_are_rejected(tmp_path, name, value):
    result = run(tmp_path, "--dry-run", **{name: value})
    assert result.returncode == 2
    assert name in result.stderr


def test_ambiguous_checkpoint_or_absolute_budget_is_rejected(tmp_path):
    result = run(tmp_path, "--dry-run", CHECKPOINT_PATH="a.pt", CHECKPOINT="b.pt")
    assert result.returncode == 2 and "different files" in result.stderr
    result = run(tmp_path, "--dry-run", MAX_ITERATIONS="130000")
    assert result.returncode == 2 and "ADDITIONAL_ITERATIONS" in result.stderr


def test_missing_checkpoint_fails_before_any_python(tmp_path):
    executable, calls = fake_python(tmp_path)
    result = run(
        tmp_path, T800_PYTHON=str(executable), T800_TEST_CALLS=str(calls),
        CHECKPOINT_PATH="missing.pt",
    )
    assert result.returncode == 1 and "CHECKPOINT_PATH" in result.stderr
    assert not calls.exists()


def test_custom_checkpoint_and_origin_reach_smoke_and_torchrun_as_single_arguments(tmp_path):
    executable, calls = fake_python(tmp_path)
    checkpoint = tmp_path / "checkpoint with spaces.pt"
    checkpoint.touch()
    result = run(
        tmp_path, T800_PYTHON=str(executable), T800_TEST_CALLS=str(calls),
        CHECKPOINT_PATH=checkpoint.name, T800_GPUS="1", ADDITIONAL_ITERATIONS="5000",
        FINETUNE_LR_SCALE="0.25", FINETUNE_WARMUP_ITERATIONS="0",
        FINETUNE_ORIGIN_ITERATION="100000",
    )
    assert result.returncode == 0, result.stderr
    preflight, smoke, training = [json.loads(line) for line in calls.read_text().splitlines()]
    assert preflight == ["-", "1"]
    for command in (smoke, training):
        assert command[command.index("--task") + 1] == "T800_base-AnkleCombinedV2"
        assert command[command.index("--checkpoint") + 1] == str(checkpoint)
    assert "--smoke" in smoke and "scripts/check_config.py" in smoke
    assert smoke[smoke.index("--finetune-lr-scale") + 1] == "0.25"
    assert smoke[smoke.index("--finetune-warmup-iterations") + 1] == "0"
    assert smoke[smoke.index("--finetune-origin-iteration") + 1] == "100000"
    assert "--nproc_per_node=1" in training and "--max_restarts=0" in training
    assert training[training.index("--finetune_lr_scale") + 1] == "0.25"
    assert training[training.index("--additional_iterations") + 1] == "5000"
    assert training[training.index("--finetune_origin_iteration") + 1] == "100000"
    assert training[training.index("--finetune_warmup_iterations") + 1] == "0"


@pytest.mark.parametrize("stage,exit_code,call_count", [
    ("cuda", 19, 1), ("smoke", 23, 2), ("train", 41, 3),
])
def test_failures_exit_the_batch_instead_of_waiting_with_allocated_gpus(tmp_path, stage, exit_code, call_count):
    executable, calls = fake_python(tmp_path)
    result = run(
        tmp_path, T800_PYTHON=str(executable), T800_TEST_CALLS=str(calls),
        T800_TEST_FAIL_STAGE=stage,
    )
    assert result.returncode == exit_code
    assert len(calls.read_text().splitlines()) == call_count
