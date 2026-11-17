"""Bounded tests for the four ankle continuation launch surfaces."""

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_ankle.slurm"
SUBMIT = ROOT / "scripts/submit_ankle.sh"


def run(script, *args, **overrides):
    env = os.environ.copy()
    env.update(overrides)
    return subprocess.run(["bash", str(script), *args], cwd=ROOT, env=env, text=True, capture_output=True, timeout=20)


def test_dry_run_prints_exactly_four_resume_commands():
    result = run(SUBMIT, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Checkpoint:") == 4
    for task in ("AnkleSmooth", "AnkleDamping", "AnkleTargetLimit", "AnklePose"):
        assert f"T800_base-{task}" in result.stdout
    assert "ckpt_85500.pt" in result.stdout
    assert "--max_iterations 115500" in result.stdout
    assert "--checkpoint" in result.stdout


def test_invalid_array_index_fails_before_training():
    result = run(SCRIPT, "--dry-run", SLURM_ARRAY_TASK_ID="4")
    assert result.returncode == 2
    assert "SLURM_ARRAY_TASK_ID" in result.stderr


def test_invalid_environment_count_fails_before_training():
    result = run(SCRIPT, "--dry-run", NUM_ENVS="4097")
    assert result.returncode == 2
    assert "divisible" in result.stderr


def test_zero_learning_rate_scale_is_rejected():
    result = run(SCRIPT, "--dry-run", FINETUNE_LR_SCALE="0.0")
    assert result.returncode == 2
    assert "FINETUNE_LR_SCALE" in result.stderr


def test_damping_dry_run_exposes_variant_specific_task():
    result = run(SCRIPT, "--dry-run", SLURM_ARRAY_TASK_ID="1")
    assert result.returncode == 0, result.stderr
    assert "T800_base-AnkleDamping" in result.stdout
    assert "--task T800_base-AnkleDamping" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "logs/cusrl/T800_base-AnkleDamping/<timestamp>:" in result.stdout
