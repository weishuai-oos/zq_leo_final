"""Independent regression against the effective historical task configuration.

The fixtures were extracted from the original final task, including every
inherited post_init override, before flattening. The CPU model records explicit
configuration only; real IsaacLab defaults are checked by check_config.py.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from _config_model import ConfigModel, normalize

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "source/t800_lab/t800_lab"


def historical_expected(mode):
    old = json.loads((ROOT / f"tests/fixtures/old_{mode}_config.json").read_text())
    old.pop("ANKLE_STABLE_LEVEL")
    old.pop("monitors")  # Empty LeoLab wrapper field; new env uses IsaacLab directly.
    old["run_name"] = "T800_base"
    for group in ("amp_policy", "amp_reference"):
        for term in ("base_lin_vel", "base_ang_vel"):
            assert old["observations"][group].pop(term) is None
    generator = old["scene"]["terrain"]["terrain_generator"]
    generator["num_cols"] = 50
    for terrain in generator["sub_terrains"].values():
        terrain["proportion"] *= 0.8
        terrain["proportion"] = round(terrain["proportion"], 10)
    generator["sub_terrains"] = {"fight_mat_flat": {"proportion": 0.20}, **generator["sub_terrains"]}
    return old


@pytest.fixture(scope="module")
def model():
    return ConfigModel(ROOT, "t800_lab").load("t800_lab.tasks.t800_base.env_cfg")


@pytest.fixture(scope="module")
def ankle_model():
    model = ConfigModel(ROOT, "t800_lab")
    base_is_config = model._is_config
    model._is_config = lambda name: base_is_config(name) or name == "t800_lab.tasks.t800_ankle.env_cfg"
    return model.load("t800_lab.tasks.t800_ankle.env_cfg")


@pytest.mark.parametrize(("mode", "class_name"), [("train", "T800BaseEnvCfg"), ("play", "T800BaseEnvCfg_PLAY")])
def test_effective_config_matches_historical_final_task(model, mode, class_name):
    assert normalize(getattr(model, class_name)()) == historical_expected(mode)


def test_action_and_observation_order_and_dimensions(model):
    cfg = model.T800BaseEnvCfg()
    actions = cfg.actions.joint_pos
    assert len(actions.joint_names) == 25
    assert actions.preserve_order and actions.use_default_offset
    for group_name, terms, dimension in (
        ("policy", ["base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions"], 84),
        (
            "critic",
            [
                "base_lin_vel",
                "base_ang_vel",
                "projected_gravity",
                "velocity_commands",
                "joint_pos",
                "joint_vel",
                "actions",
            ],
            87,
        ),
        ("amp_policy", ["velocity_commands", "projected_gravity", "joint_pos_rel", "joint_vel"], 96),
        ("amp_reference", ["velocity_commands", "projected_gravity", "joint_pos_rel", "joint_vel"], 96),
    ):
        group = getattr(cfg.observations, group_name)
        obs = {k: v for k, v in vars(group).items() if hasattr(v, "func")}
        assert list(obs) == terms
        actual_dimension = 0
        for name, term in obs.items():
            size = (
                len(term.params["asset_cfg"].joint_names)
                if name.startswith("joint_")
                else (25 if name == "actions" else 3)
            )
            actual_dimension += size * getattr(term, "history_length", 1)
        assert actual_dimension == dimension
        assert group.concatenate_terms is False
    assert cfg.sim.dt * cfg.decimation == pytest.approx(0.02)
    assert cfg.scene.contact_forces.update_period == pytest.approx(0.005)
    assert cfg.scene.motion_reference.update_period == pytest.approx(0.02)
    for name in ("J04_ANKLE_PITCH_L", "J10_ANKLE_PITCH_R"):
        assert cfg.scene.robot.init_state.joint_pos[name] == -0.14


def test_terrain_mix_has_exact_column_counts(model):
    cfg = model.T800BaseEnvCfg().scene.terrain.terrain_generator
    assert cfg.num_rows == 10 and cfg.num_cols == 50
    assert cfg.curriculum and cfg.size == (8.0, 8.0)
    proportions = [v.proportion for v in cfg.sub_terrains.values()]
    assert proportions == pytest.approx([0.20, 0.08, 0.12, 0.12, 0.08, 0.40])
    cumulative = np.cumsum(proportions) / sum(proportions)
    # IsaacLab TerrainGenerator._generate_curriculum_terrains column selection.
    selected = [int(np.where(column / cfg.num_cols + 0.001 < cumulative)[0][0]) for column in range(cfg.num_cols)]
    assert np.bincount(selected).tolist() == [10, 4, 6, 6, 4, 20]


def test_combined_ankle_config_is_flat_and_does_not_mutate_base(model, ankle_model):
    base_before = model.T800BaseEnvCfg()
    combined = ankle_model.T800AnkleCombinedEnvCfg()
    base_after = model.T800BaseEnvCfg()
    assert combined.ankle_variant == "AnkleCombined"
    assert combined.run_name == "T800_base-AnkleCombined"
    assert not hasattr(combined, "ankle_finetune_reward_weights")
    assert combined.rewards.ankle_target_rate_l2.weight == -0.04
    assert combined.rewards.ankle_target_acceleration_l2.weight == -0.008
    assert combined.rewards.ankle_support_joint_pose_l2.weight == -0.02
    pose = combined.rewards.ankle_support_joint_pose_l2
    assert pose.params["threshold"] == 1.0
    assert pose.params["normalizer_rad"] == 0.1
    assert pose.params["asset_cfg"].joint_names == [
        "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R"
    ]
    assert set(combined.scene.robot.actuators["feet"].stiffness.values()) == {40.0}
    assert set(combined.scene.robot.actuators["feet"].damping.values()) == {1.2}
    assert set(base_before.scene.robot.actuators["feet"].damping.values()) == {0.3}
    assert set(base_after.scene.robot.actuators["feet"].damping.values()) == {0.3}
    play = ankle_model.T800AnkleCombinedEnvCfg_PLAY()
    assert play.rewards.ankle_support_joint_pose_l2.weight == -0.02
    assert set(play.scene.robot.actuators["feet"].damping.values()) == {1.2}
    assert play.events.push_robot is None and play.events.base_external_force_torque is None


def test_motion_and_urdf_resources_are_self_contained(model):
    cfg = model.T800BaseEnvCfg()
    motion = cfg.scene.motion_reference.motion_buffers["boxing_guard"]
    directory = Path(motion.path)
    selection = yaml.safe_load(Path(motion.filtered_motion_selection_filepath).read_text())
    assert len(list(directory.glob("*.npz"))) == 14
    assert len(selection["selected_files"]) == len(selection["motion_weights"]) == 13
    assert sum(selection["motion_weights"]) == pytest.approx(1.0)
    for filename in selection["selected_files"] + ["标准姿势_t800_controller_retargeted.npz"]:
        path = (directory / filename).resolve()
        assert path.is_relative_to(PACKAGE) and path.is_file()
        with np.load(path) as data:
            names = data["joint_names"].tolist()
            assert names == cfg.actions.joint_pos.joint_names
            assert data["joint_pos"].shape[1] == 25
            assert np.isfinite(data["joint_pos"]).all()
    urdf = Path(cfg.scene.robot.spawn.asset_path).resolve()
    assert urdf.is_relative_to(PACKAGE) and urdf.is_file()
    robot = ET.parse(urdf).getroot()
    joint_names = {joint.attrib["name"] for joint in robot.findall("joint")}
    assert set(cfg.actions.joint_pos.joint_names) <= joint_names
    for mesh in robot.iter("mesh"):
        filename = mesh.attrib["filename"]
        assert "://" not in filename
        resolved = (urdf.parent / filename).resolve()
        assert resolved.is_relative_to(PACKAGE) and resolved.is_file()


def test_all_local_import_targets_exist():
    missing = []
    for path in PACKAGE.rglob("*.py"):
        parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names if alias.name.startswith("t800_lab")]
            elif isinstance(node, ast.ImportFrom):
                target = (
                    importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                    if node.level
                    else node.module
                )
                targets = [target] if target and target.startswith("t800_lab") else []
                if node.level and not node.module:
                    targets = [f"{target}.{alias.name}" for alias in node.names]
            else:
                continue
            for target in targets:
                base = PACKAGE.parent.joinpath(*target.split("."))
                if not base.with_suffix(".py").is_file() and not (base / "__init__.py").is_file():
                    missing.append(f"{path.relative_to(ROOT)} imports {target}")
    assert not missing, "\n".join(missing)


def test_registration_has_base_impact_and_combined_train_and_play():
    from types import SimpleNamespace
    import runpy

    calls = []
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"gymnasium": SimpleNamespace(register=lambda **kw: calls.append(kw["id"]))}):
        runpy.run_path(str(PACKAGE / "tasks/__init__.py"))
    ankle = [f"T800_base-{name}" for name in ("AnkleSmooth", "AnkleDamping", "AnkleTargetLimit", "AnklePose")]
    train = [
        "T800_base", "T800_base-Impact", "T800_base-ImpactV2", "T800_base-ImpactV2Plus", *ankle,
        "T800_base-ImpactV2-From154500-DR", "T800_base-ImpactV2Plus-From173000-DR",
        "T800_base-AnkleCombined", "T800_base-AnkleCombinedV2",
    ]
    assert sorted(calls) == sorted([name for task in train for name in (task, task + "-Play")])


def test_combined_v2_preserves_policy_contract_and_original_combined(ankle_model):
    original = normalize(ankle_model.T800AnkleCombinedEnvCfg())
    cfg = ankle_model.T800AnkleCombinedV2EnvCfg()
    play = ankle_model.T800AnkleCombinedV2EnvCfg_PLAY()
    assert normalize(ankle_model.T800AnkleCombinedEnvCfg()) == original
    for name in ("actions", "observations", "commands", "terminations", "curriculum"):
        assert normalize(getattr(cfg, name)) == original[name]
    assert cfg.sim.dt == 0.005 and cfg.decimation == 4
    assert cfg.run_name == "T800_base-AnkleCombinedV2"
    assert set(cfg.scene.robot.actuators["feet"].stiffness.values()) == {40.0}
    assert set(cfg.scene.robot.actuators["feet"].damping.values()) == {2.0}
    assert cfg.events.collider_offsets.params["contact_offset_distribution_params"] == (0.002, 0.002)
    assert cfg.events.collider_offsets.params["rest_offset_distribution_params"] is None
    assert cfg.rewards.ankle_target_rate_l2.weight == -0.04
    assert cfg.rewards.ankle_target_acceleration_l2.weight == -0.008
    assert cfg.rewards.ankle_support_joint_pose_l2.weight == -0.02
    assert cfg.rewards.ankle_physics_acceleration_l2.weight == -0.02
    assert cfg.rewards.ankle_support_foot_angular_velocity_l2.weight == -0.2
    assert normalize(play.rewards) == normalize(cfg.rewards)
    assert play.ankle_substep_stability == cfg.ankle_substep_stability
    assert play.events.push_robot is None and play.events.base_external_force_torque is None
