"""Exercise stage-two launch boundaries without starting Isaac, CUDA or Slurm."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_impact.slurm"


def launch(tmp_path, overrides=None, *, dry_run=True, script=SCRIPT):
    env = os.environ.copy()
    for key in (
        "T800_PROJECT_ROOT", "T800_PYTHON", "T800_GPUS", "NUM_ENVS", "SEED", "MAX_ITERATIONS",
        "ADDITIONAL_ITERATIONS", "SAVE_INTERVAL", "RUN_NAME", "CHECKPOINT", "SLURM_GPUS_ON_NODE",
        "T800_TEST_CUDA_FAIL", "T800_TEST_SMOKE_FAIL", "SLURM_JOB_ID",
        "IMPACT_SINGLE_HIT_MIGRATION", "TASK",
    ):
        env.pop(key, None)
    env.update(overrides or {})
    return subprocess.run(
        ["bash", str(script), *(["--dry-run"] if dry_run else [])],
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
        "if 'scripts/check_config.py' in sys.argv and os.environ.get('T800_TEST_SMOKE_FAIL'):\n"
        "    raise SystemExit(23)\n"
    )
    executable.chmod(0o755)
    checkpoint = tmp_path / "run with spaces" / "ckpt_26500.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"path fixture only")
    return executable, calls, checkpoint


def test_checkpoint_is_required_even_for_preview(tmp_path):
    result = launch(tmp_path)
    assert result.returncode == 2
    assert "CHECKPOINT is required" in result.stderr


def test_preview_from_arbitrary_directory_uses_additional_budget(tmp_path):
    executable, calls, checkpoint = fake_python(tmp_path)
    result = launch(tmp_path, {"CHECKPOINT": str(checkpoint), "T800_PYTHON": str(executable)})
    assert result.returncode == 0, result.stderr
    assert f"Project: {ROOT}" in result.stdout
    assert "4096 global / 2048 per GPU" in result.stdout
    assert "--task T800_base-Impact" in result.stdout
    assert "--additional_iterations 40000" in result.stdout
    assert "--max_iterations" not in result.stdout
    assert not calls.exists()


def test_preview_can_select_the_independent_impact_v2_task(tmp_path):
    executable, calls, checkpoint = fake_python(tmp_path)
    result = launch(tmp_path, {
        "TASK": "T800_base-ImpactV2", "CHECKPOINT": str(checkpoint), "T800_PYTHON": str(executable),
    })
    assert result.returncode == 0, result.stderr
    assert "--task T800_base-ImpactV2" in result.stdout
    assert "T800_base-ImpactV2" in result.stdout
    assert not calls.exists()


@pytest.mark.parametrize("source_iteration", [140000, 154500])
def test_plus_preview_keeps_explicit_source_and_independent_run_name(tmp_path, source_iteration):
    executable, calls, checkpoint = fake_python(tmp_path)
    selected = checkpoint.with_name(f"ckpt_{source_iteration}.pt")
    checkpoint.rename(selected)
    name = f"ImpactV2Plus_from_{source_iteration}_seed42"
    result = launch(tmp_path, {
        "TASK": "T800_base-ImpactV2Plus", "CHECKPOINT": str(selected),
        "RUN_NAME": name, "T800_PYTHON": str(executable),
    })
    assert result.returncode == 0, result.stderr
    assert "--task T800_base-ImpactV2Plus" in result.stdout
    assert "--additional_iterations 40000" in result.stdout
    assert f"--run_name {name}" in result.stdout
    assert str(selected).replace(" ", "\\ ") in result.stdout
    assert not calls.exists()


def test_slurm_spooled_script_uses_configured_working_directory(tmp_path):
    executable, calls, checkpoint = fake_python(tmp_path)
    spooled_script = tmp_path / "slurm_script"
    spooled_script.write_text(SCRIPT.read_text())
    result = launch(ROOT, {
        "SLURM_JOB_ID": "12345", "CHECKPOINT": str(checkpoint),
        "T800_PYTHON": str(executable),
    }, script=spooled_script)
    assert result.returncode == 0, result.stderr
    assert f"Project: {ROOT}" in result.stdout
    assert not calls.exists()


@pytest.mark.parametrize("overrides", [
    {"NUM_ENVS": "4097"}, {"T800_GPUS": "0"}, {"ADDITIONAL_ITERATIONS": "-1"},
    {"MAX_ITERATIONS": "0"}, {"MAX_ITERATIONS": "50000", "ADDITIONAL_ITERATIONS": "25000"},
    {"IMPACT_SINGLE_HIT_MIGRATION": "true"},
])
def test_invalid_launch_fails_before_python(tmp_path, overrides):
    executable, calls, checkpoint = fake_python(tmp_path)
    result = launch(tmp_path, {"CHECKPOINT": str(checkpoint), "T800_PYTHON": str(executable), **overrides})
    assert result.returncode == 2
    assert not calls.exists()


@pytest.mark.parametrize(("failure", "exit_code", "num_calls"), [
    ("T800_TEST_CUDA_FAIL", 19, 1), ("T800_TEST_SMOKE_FAIL", 23, 2),
])
def test_failed_preflight_or_smoke_prevents_training(tmp_path, failure, exit_code, num_calls):
    executable, calls, checkpoint = fake_python(tmp_path)
    result = launch(tmp_path, {
        "CHECKPOINT": str(checkpoint), "T800_PYTHON": str(executable),
        "T800_TEST_CALLS": str(calls), failure: "1",
    }, dry_run=False)
    assert result.returncode == exit_code
    assert len(calls.read_text().splitlines()) == num_calls


def test_impact_smoke_precedes_torchrun_and_preserves_spaced_checkpoint(tmp_path):
    executable, calls, checkpoint = fake_python(tmp_path)
    result = launch(tmp_path, {
        "CHECKPOINT": str(checkpoint), "T800_PYTHON": str(executable),
        "T800_TEST_CALLS": str(calls), "SLURM_GPUS_ON_NODE": "1", "MAX_ITERATIONS": "60000",
    }, dry_run=False)
    assert result.returncode == 0, result.stderr
    preflight, smoke, training = [json.loads(line) for line in calls.read_text().splitlines()]
    assert preflight == ["-", "1"]
    assert smoke == [
        "-u", "scripts/check_config.py", "--task", "T800_base-Impact", "--checkpoint", str(checkpoint),
        "--headless", "--device", "cuda:0", "--num_envs", "2", "--smoke",
    ]
    assert "--nproc_per_node=1" in training
    assert training[training.index("--checkpoint") + 1] == str(checkpoint)
    assert training[training.index("--max_iterations") + 1] == "60000"
    assert "--additional_iterations" not in training


def test_project_directory_with_spaces_is_supported(tmp_path):
    executable, calls, checkpoint = fake_python(tmp_path)
    project = tmp_path / "checkout with spaces"
    (project / "scripts/cusrl").mkdir(parents=True)
    (project / "scripts/cusrl/train.py").touch()
    (project / "scripts/check_config.py").touch()
    result = launch(tmp_path, {
        "T800_PROJECT_ROOT": str(project), "CHECKPOINT": str(checkpoint),
        "T800_PYTHON": str(executable), "T800_TEST_CALLS": str(calls),
    }, dry_run=False)
    assert result.returncode == 0, result.stderr
    assert f"Project: {project}" in result.stdout
    assert len(calls.read_text().splitlines()) == 3


def test_single_hit_migration_reaches_both_smoke_and_training(tmp_path):
    executable, calls, checkpoint = fake_python(tmp_path)
    result = launch(tmp_path, {
        "CHECKPOINT": str(checkpoint), "T800_PYTHON": str(executable),
        "T800_TEST_CALLS": str(calls), "IMPACT_SINGLE_HIT_MIGRATION": "1",
    }, dry_run=False)
    assert result.returncode == 0, result.stderr
    _, smoke, training = [json.loads(line) for line in calls.read_text().splitlines()]
    assert "--impact-single-hit-migration" in smoke
    assert "--impact-single-hit-migration" in training


def test_impact_v2_rejects_legacy_single_hit_migration_switch(tmp_path):
    executable, calls, checkpoint = fake_python(tmp_path)
    result = launch(tmp_path, {
        "TASK": "T800_base-ImpactV2", "CHECKPOINT": str(checkpoint),
        "T800_PYTHON": str(executable), "IMPACT_SINGLE_HIT_MIGRATION": "1",
    })
    assert result.returncode == 2
    assert "IMPACT_SINGLE_HIT_MIGRATION" in result.stderr or "single-hit" in result.stderr
    assert not calls.exists()


def test_relative_checkpoint_is_resolved_before_changing_directory(tmp_path):
    executable, calls, checkpoint = fake_python(tmp_path)
    result = launch(tmp_path, {
        "CHECKPOINT": str(checkpoint.relative_to(tmp_path)), "T800_PYTHON": str(executable),
        "T800_TEST_CALLS": str(calls),
    }, dry_run=False)
    assert result.returncode == 0, result.stderr
    training = json.loads(calls.read_text().splitlines()[-1])
    assert training[training.index("--checkpoint") + 1] == str(checkpoint)


def test_headless_bootstrap_keeps_logical_slots_and_removes_physical_head_config(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("_impact_bootstrap_test", ROOT / "scripts/cusrl/bootstrap.py")
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    head_assets = ModuleType("t800_lab.tasks.t800_impact.head_assets")

    def make_headless_urdf(source, destination):
        assert source == "original.urdf"
        destination.write_text("temporary headless URDF fixture")
        return str(destination)

    head_assets.make_headless_urdf = make_headless_urdf
    monkeypatch.setitem(sys.modules, head_assets.__name__, head_assets)
    head_names = ["J27_HEAD_PITCH", "J28_HEAD_YAW"]
    names = ["BODY_JOINT", *head_names]
    cfg = SimpleNamespace(
        impact=SimpleNamespace(head_mode="headless"),
        actions=SimpleNamespace(joint_pos=SimpleNamespace(joint_names=list(names))),
        scene=SimpleNamespace(
            motion_reference=SimpleNamespace(robot_model_path="original.urdf"),
            robot=SimpleNamespace(
                spawn=SimpleNamespace(asset_path="original.urdf", usd_dir=None),
                actuators={"body": object(), "head": object()},
                init_state=SimpleNamespace(
                    joint_pos={"BODY_JOINT": 1, "J27_HEAD_PITCH": 0.05, "J28_HEAD_YAW": 0.62},
                    joint_vel={".*": 0, "J2[78]_HEAD_.*": 0},
                ),
            ),
            impact_ground_LINK_HEAD_YAW=object(), impact_ground_LINK_HEAD_PITCH=object(),
            impact_ground_LINK_TORSO_YAW=object(),
        ),
    )
    with bootstrap.configure_usd_cache(cfg) as cache:
        assert Path(cfg.scene.robot.spawn.asset_path).is_file()
        assert cfg.scene.robot.spawn.usd_dir == cache
        assert cfg.scene.motion_reference.robot_model_path == cfg.scene.robot.spawn.asset_path
        assert cfg.actions.joint_pos.joint_names == names
        assert list(cfg.scene.robot.actuators) == ["body"]
        assert cfg.scene.robot.init_state.joint_pos == {"BODY_JOINT": 1}
        assert cfg.scene.robot.init_state.joint_vel == {".*": 0}
        assert cfg.scene.impact_ground_LINK_HEAD_YAW is None
        assert cfg.scene.impact_ground_LINK_HEAD_PITCH is None
        assert cfg.scene.impact_ground_LINK_TORSO_YAW is not None
    assert not Path(cache).exists()


def train_cli_namespace(argv):
    """Execute only the real argument parser, without importing Isaac/AppLauncher."""
    import argparse
    import math

    task_spec = importlib.util.spec_from_file_location(
        "_impact_cli_task_names", ROOT / "source/t800_lab/t800_lab/task_names.py"
    )
    task_names = importlib.util.module_from_spec(task_spec)
    task_spec.loader.exec_module(task_names)

    source = ast.parse((ROOT / "scripts/cusrl/train.py").read_text())
    nodes = []
    for node in source.body:
        if isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names):
            break
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            continue
        nodes.append(node)

    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--headless", action="store_true")

    namespace = {
        "argparse": argparse, "math": math, "bootstrap": task_names,
        "AppLauncher": AppLauncher, "__doc__": "test parser",
    }
    previous = sys.argv
    try:
        sys.argv = ["train.py", *argv]
        exec(compile(ast.Module(nodes, type_ignores=[]), "train.py", "exec"), namespace)
    finally:
        sys.argv = previous
    return namespace["args_cli"]


def test_stage_one_cli_defaults_still_train_from_scratch():
    args = train_cli_namespace([])
    assert args.task == args.run_name == "T800_base"
    assert args.checkpoint is None
    assert args.max_iterations is None


def test_impact_cli_requires_checkpoint_and_defaults_name_to_task():
    with pytest.raises(SystemExit):
        train_cli_namespace(["--task", "T800_base-Impact"])
    args = train_cli_namespace(["--task", "T800_base-Impact", "--checkpoint", "checkpoint with spaces.pt"])
    assert args.run_name == "T800_base-Impact"
    assert args.checkpoint == "checkpoint with spaces.pt"


def test_budget_flags_are_exclusive_and_additional_flag_is_stage_two_only():
    with pytest.raises(SystemExit):
        train_cli_namespace(["--additional_iterations", "50"])
    with pytest.raises(SystemExit):
        train_cli_namespace([
            "--task", "T800_base-Impact", "--checkpoint", "x.pt",
            "--max_iterations", "50000", "--additional_iterations", "50",
        ])


def test_single_hit_migration_is_an_explicit_impact_training_option():
    with pytest.raises(SystemExit):
        train_cli_namespace(["--impact-single-hit-migration"])
    args = train_cli_namespace([
        "--task", "T800_base-Impact", "--checkpoint", "x.pt", "--impact-single-hit-migration",
    ])
    assert args.impact_single_hit_migration
