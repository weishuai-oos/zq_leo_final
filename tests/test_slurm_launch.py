"""Check launch boundaries without submitting jobs or starting Isaac/GPU work."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train.slurm"


def launch(tmp_path, overrides=None, *, dry_run=True):
    env = os.environ.copy()
    for key in (
        "T800_PROJECT_ROOT", "T800_PYTHON", "T800_GPUS", "NUM_ENVS", "SEED", "MAX_ITERATIONS",
        "SAVE_INTERVAL", "RUN_NAME", "CHECKPOINT", "SLURM_GPUS_ON_NODE",
    ):
        env.pop(key, None)
    env.update(overrides or {})
    return subprocess.run(
        ["bash", str(SCRIPT), *(["--dry-run"] if dry_run else [])],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20,
    )


def fake_python(tmp_path):
    executable = tmp_path / "fake python"
    calls = tmp_path / "calls.jsonl"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['T800_TEST_CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1] == '-' and os.environ.get('T800_TEST_CUDA_FAIL'):\n"
        "    raise SystemExit(19)\n"
    )
    executable.chmod(0o755)
    return executable, calls


def test_preview_uses_new_project_from_any_working_directory(tmp_path):
    result = launch(tmp_path)
    assert result.returncode == 0, result.stderr
    assert f"Project: {ROOT}" in result.stdout
    assert "4096 global / 2048 per GPU" in result.stdout
    assert "--nproc_per_node=2" in result.stdout
    assert "--checkpoint" not in result.stdout
    assert "none (from scratch)" in result.stdout


@pytest.mark.parametrize("overrides", [{"NUM_ENVS": "4097"}, {"T800_GPUS": "0"}, {"MAX_ITERATIONS": "-1"}])
def test_invalid_resource_parameters_fail_before_python(tmp_path, overrides):
    executable, calls = fake_python(tmp_path)
    result = launch(tmp_path, {"T800_PYTHON": str(executable), "T800_TEST_CALLS": str(calls), **overrides})
    assert result.returncode == 2
    assert not calls.exists()


def test_cuda_failure_prevents_smoke_and_training(tmp_path):
    executable, calls = fake_python(tmp_path)
    result = launch(tmp_path, {
        "T800_PYTHON": str(executable), "T800_TEST_CALLS": str(calls), "T800_TEST_CUDA_FAIL": "1",
    }, dry_run=False)
    assert result.returncode == 19
    assert [json.loads(line) for line in calls.read_text().splitlines()] == [["-", "2"]]


def test_smoke_precedes_torchrun_and_checkpoint_path_stays_one_argument(tmp_path):
    executable, calls = fake_python(tmp_path)
    checkpoint = tmp_path / "run with spaces" / "ckpt_1000.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"test checkpoint path only")
    result = launch(tmp_path, {
        "T800_PYTHON": str(executable), "T800_TEST_CALLS": str(calls),
        "SLURM_GPUS_ON_NODE": "1", "CHECKPOINT": str(checkpoint),
    }, dry_run=False)
    assert result.returncode == 0, result.stderr
    preflight, smoke, training = [json.loads(line) for line in calls.read_text().splitlines()]
    assert preflight == ["-", "1"]
    assert smoke == ["-u", "scripts/check_config.py", "--headless", "--device", "cuda:0", "--smoke"]
    assert "--nproc_per_node=1" in training
    assert training[training.index("--checkpoint") + 1] == str(checkpoint)
