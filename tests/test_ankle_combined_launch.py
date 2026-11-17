"""Launch-boundary tests for the standalone combined ankle training job."""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_ankle_combined.slurm"


def run(tmp_path, *args, **overrides):
    env = os.environ.copy()
    for key in ("T800_PROJECT_ROOT", "T800_PYTHON", "T800_GPUS", "NUM_ENVS", "SEED", "MAX_ITERATIONS", "SAVE_INTERVAL", "RUN_NAME", "CHECKPOINT", "SLURM_GPUS_ON_NODE"):
        env.pop(key, None)
    env.update(overrides)
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=20)


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


def test_dry_run_contains_standalone_task_and_budget(tmp_path):
    result = run(tmp_path, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "Task: T800_base-AnkleCombined" in result.stdout
    assert "4096 global / 2048 per GPU" in result.stdout
    assert "--nproc_per_node=2" in result.stdout
    assert "--task T800_base-AnkleCombined" in result.stdout
    assert "--max_iterations 100000" in result.stdout
    assert "--save_interval 500" in result.stdout
    assert "--checkpoint" not in result.stdout


def test_checkpoint_is_rejected_even_for_preview(tmp_path):
    result = run(tmp_path, "--dry-run", CHECKPOINT="/tmp/old.pt")
    assert result.returncode == 2
    assert "must start from scratch" in result.stderr


def test_invalid_environment_count_fails_before_python(tmp_path):
    executable, calls = fake_python(tmp_path)
    result = run(tmp_path, "--dry-run", T800_PYTHON=str(executable), T800_TEST_CALLS=str(calls), NUM_ENVS="4097")
    assert result.returncode == 2
    assert "divisible" in result.stderr
    assert not calls.exists()


def test_smoke_precedes_torchrun_and_uses_combined_task(tmp_path):
    executable, calls = fake_python(tmp_path)
    result = run(tmp_path, T800_PYTHON=str(executable), T800_TEST_CALLS=str(calls), SLURM_GPUS_ON_NODE="1")
    assert result.returncode == 0, result.stderr
    preflight, smoke, training = [json.loads(line) for line in calls.read_text().splitlines()]
    assert preflight == ["-", "1"]
    assert smoke == ["-u", "scripts/check_config.py", "--task", "T800_base-AnkleCombined", "--headless", "--device", "cuda:0", "--num-envs", "2", "--smoke"]
    assert "--nproc_per_node=1" in training
    assert "--task" in training and training[training.index("--task") + 1] == "T800_base-AnkleCombined"
    assert "--checkpoint" not in training
