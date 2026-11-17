"""Use this checkout without importing or installing the original leo_lab."""

from importlib import import_module
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "t800_lab"))

from t800_lab.task_names import (  # noqa: E402
    ALL_TASKS as ALL_TASKS,
    ANKLE_TASKS as ANKLE_TASKS,
    COMBINED_ANKLE_TASK as COMBINED_ANKLE_TASK,
    COMBINED_V2_ANKLE_TASK as COMBINED_V2_ANKLE_TASK,
    DYNAMICS_DR_TASKS as DYNAMICS_DR_TASKS,
    IMPACT_TASKS as IMPACT_TASKS,
    IMPACT_V2_TASK as IMPACT_V2_TASK,
    IMPACT_V2_PLUS_TASK as IMPACT_V2_PLUS_TASK,
    TRAIN_TASKS as TRAIN_TASKS,
    impact_parent_task as impact_parent_task,
)


def load_task_config(task: str, key: str):
    """Load one explicitly registered config, after AppLauncher starts Kit."""
    import gymnasium as gym
    import t800_lab.tasks  # noqa: F401

    if task not in ALL_TASKS:
        raise ValueError(f"Unsupported T800 task: {task}")
    entry = gym.spec(task).kwargs[key]
    module, symbol = entry.split(":", 1)
    return getattr(import_module(module), symbol)()


def configure_usd_cache(env_cfg) -> TemporaryDirectory:
    """Isolate URDF conversion per process, including distributed ranks.

    Keep the returned context alive until the environment closes. Explicitly
    setting usd_dir also avoids defaults patched in external IsaacLab installs.
    """
    # The importer can finish an asynchronous USD write during shutdown. A
    # disposable cache cleanup error must not mask the original training error.
    cache = TemporaryDirectory(prefix="t800_usd_", ignore_cleanup_errors=True)
    env_cfg.scene.robot.spawn.usd_dir = cache.name
    impact = getattr(env_cfg, "impact", None)
    if impact is not None and impact.head_mode == "headless":
        module = (
            "t800_lab.tasks.t800_impact_v2_plus.env_cfg"
            if impact_parent_task(getattr(env_cfg, "run_name", "")) == IMPACT_V2_PLUS_TASK
            else "t800_lab.tasks.t800_impact_v2.env_cfg"
            if impact_parent_task(getattr(env_cfg, "run_name", "")) == IMPACT_V2_TASK
            else "t800_lab.tasks.t800_impact.head_assets"
        )
        make_headless_urdf = import_module(module).make_headless_urdf

        env_cfg.scene.robot.spawn.asset_path = make_headless_urdf(
            env_cfg.scene.robot.spawn.asset_path, Path(cache.name) / "t800_headless.urdf"
        )
        env_cfg.scene.motion_reference.robot_model_path = env_cfg.scene.robot.spawn.asset_path
        env_cfg.scene.robot.actuators.pop("head", None)
        head_names = ("J27_HEAD_PITCH", "J28_HEAD_YAW")
        body_joint_names = [name for name in env_cfg.actions.joint_pos.joint_names if name not in head_names]
        for field in ("joint_pos", "joint_vel"):
            values = getattr(env_cfg.scene.robot.init_state, field, None)
            if isinstance(values, dict):
                for pattern in list(values):
                    if any(re.fullmatch(pattern, name) for name in head_names) and not any(
                        re.fullmatch(pattern, name) for name in body_joint_names
                    ):
                        del values[pattern]
        for name in dir(env_cfg.scene):
            if name.startswith("impact_ground_LINK_HEAD"):
                setattr(env_cfg.scene, name, None)
    return cache
