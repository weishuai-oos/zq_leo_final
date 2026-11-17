"""Numerical fall/reward contracts plus explicit config construction without Kit."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from _config_model import ConfigModel


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/t800_lab/t800_lab/tasks/t800_impact"


def load_cpu_module(name):
    qualified = "_impact_environment_test_" + name
    spec = importlib.util.spec_from_file_location(qualified, TASK / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


mdp = load_cpu_module("mdp")


def test_impact_v2_package_has_no_runtime_imports_from_retired_impact_or_ankle_tasks():
    """The second-stage task remains loadable after old task packages are removed."""
    package = ROOT / "source/t800_lab/t800_lab/tasks/t800_impact_v2"
    assert package.is_dir()
    source_files = sorted(package.glob("*.py"))
    assert source_files
    for path in source_files:
        tree = ast.parse(path.read_text(), filename=str(path))
        imports = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        imports.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert all("t800_impact" not in name and "t800_ankle" not in name for name in imports), path


def test_impact_v2_configuration_declares_the_v2_ankle_contract():
    """Static contract guards against silently falling back to base ankle damping."""
    cfg = ROOT / "source/t800_lab/t800_lab/tasks/t800_impact_v2/env_cfg.py"
    text = cfg.read_text()
    for required in (
        "T800ImpactV2EnvCfg",
        "T800_base-ImpactV2",
        "2.0",
        "0.002",
        "ankle_physics_acceleration_l2",
        "ankle_support_foot_angular_velocity_l2",
    ):
        assert required in text


@pytest.mark.parametrize("task_package", ["t800_impact_v2", "t800_impact_v2_plus"])
def test_impact_v2_constructs_without_old_tasks_and_matches_actual_source_contract(monkeypatch, task_package):
    """Execute post_init, including source hashing, without Kit or retired tasks."""
    import importlib
    from types import ModuleType
    from unittest.mock import patch

    from _config_model import Node, Symbol

    wrappers = ROOT / "source/t800_lab/t800_lab/utils/wrappers/cusrl"
    monkeypatch.setattr(sys, "path", [str(ROOT / "source/t800_lab"), *sys.path])

    class Entity(Node):
        name = joint_names = body_names = None
        preserve_order = False

        def __init__(self, name, **kwargs):
            super().__init__(name=name, **kwargs)

    class Term(Node):
        params = {}

    class CallableSymbol(Symbol):
        def __init__(self, path):
            super().__init__(path)
            self.__module__, _, self.__qualname__ = path.rpartition(".")
            if path == "isaaclab.envs.mdp.randomize_rigid_body_collider_offsets":
                self.__module__ = "isaaclab.envs.mdp.events"

        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return CallableSymbol(self.path + "." + name)

    def real_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    class Model(ConfigModel):
        def _is_config(self, name):
            return name in {
                "t800_lab.tasks.t800_impact_v2.env_cfg", "t800_lab.tasks.t800_impact_v2_plus.env_cfg",
            } or super()._is_config(name)

        def _import(self, namespace, statement, module_name):
            if isinstance(statement, ast.Import) and all(
                alias.name.split(".")[0] in {"xml", "hashlib", "json", "tempfile"} for alias in statement.names
            ):
                for alias in statement.names:
                    namespace[alias.asname or alias.name.split(".")[0]] = importlib.import_module(
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )
                return
            super()._import(namespace, statement, module_name)

        def _external(self, name, attr=None):
            full = name if attr is None else name + "." + attr
            assert not full.startswith(("t800_lab.tasks.t800_ankle", "t800_lab.tasks.t800_impact."))
            if full == "isaaclab.managers.SceneEntityCfg":
                return Entity
            if full.startswith("isaaclab.managers.") and full.endswith(
                ("RewardTermCfg", "ObservationTermCfg", "EventTermCfg", "TerminationTermCfg")
            ):
                return Term
            if full in {"t800_lab.tasks.t800_impact_v2.mdp", "t800_lab.tasks.t800_impact_v2_plus.mdp"}:
                return real_module(full, self.package_root.joinpath(*full.split(".")[1:]).with_suffix(".py"))
            if full in {"t800_lab.tasks.t800_impact_v2.impact.ImpactCfg", "t800_lab.tasks.t800_impact_v2_plus.impact.ImpactCfg"}:
                return real_module(name, self.package_root.joinpath(*name.split(".")[1:]).with_suffix(".py")).ImpactCfg
            if full == "t800_lab.tasks.t800_impact_v2_plus.recovery.RecoveryConfig":
                return real_module(name, self.package_root / "tasks/t800_impact_v2_plus/recovery.py").RecoveryConfig
            if full in {
                "t800_lab.tasks.t800_impact_v2.env.ImpactJointPositionActionCfg",
                "t800_lab.tasks.t800_impact_v2_plus.env.ImpactJointPositionActionCfg",
            }:
                return lambda **kwargs: Node(class_type=CallableSymbol(full.removesuffix("Cfg")), **kwargs)
            value = super()._external(name, attr)
            if full.endswith(".ankle_stable_mdp"):
                for function in vars(value).values():
                    if getattr(function, "__module__", None) == "_cpu_ankle_stable_mdp":
                        function.__module__ = full
            return CallableSymbol(value.path) if isinstance(value, Symbol) else value

    with patch.dict(sys.modules):
        for name, path in (
            ("t800_lab.utils", wrappers.parent.parent),
            ("t800_lab.utils.wrappers", wrappers.parent),
            ("t800_lab.utils.wrappers.cusrl", wrappers),
        ):
            module = ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
        base_adapter = ModuleType("t800_lab.utils.wrappers.cusrl.vecenv_wrapper")
        base_adapter.CusrlT800Adapter = object
        sys.modules[base_adapter.__name__] = base_adapter
        plus_package = ModuleType("t800_lab.tasks.t800_impact_v2_plus")
        plus_package.__path__ = [str(ROOT / "source/t800_lab/t800_lab/tasks/t800_impact_v2_plus")]
        sys.modules[plus_package.__name__] = plus_package
        model = Model(ROOT, "t800_lab")
        module = model.load(f"t800_lab.tasks.{task_package}.env_cfg")
        cfg = getattr(module, "T800ImpactV2PlusEnvCfg" if task_package.endswith("_plus") else "T800ImpactV2EnvCfg")()
        assert cfg.impact_v2_source_play_signature == "bd85bb8e6735e14844b9074f68cffc504cea28421fc0ea63b4f160e6a9ffeb9d"
        assert set(cfg.scene.robot.actuators["feet"].damping.values()) == {2.0}
        assert [item.proportion for item in cfg.scene.terrain.terrain_generator.sub_terrains.values()] == [0.8, 0.2]
        pose = cfg.rewards.ankle_support_joint_pose_l2
        assert pose.weight == -0.02
        expected_params = {"term_func", "term_params", "asset_cfg", "sensor_cfg"}
        if task_package.endswith("_plus"):
            expected_params.add("recovery_factor")
        assert set(pose.params) == expected_params
        assert pose.params["sensor_cfg"].body_names == ["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"]
        if task_package.endswith("_plus"):
            from t800_lab.utils.wrappers.cusrl.impact import impact_v2_configuration_signature
            parent = model.load("t800_lab.tasks.t800_impact_v2.env_cfg").T800ImpactV2EnvCfg()
            assert cfg.impact_v2_parent_configuration_signature == impact_v2_configuration_signature(parent)
            # Runtime configuration is simulator independent and must match
            # real ckpt_154500. Full callable identities additionally require
            # the real-Isaac checkpoint smoke (ConfigModel stubs externals).
            assert cfg.impact_v2_parent_runtime_signature == "2d9eaa9afb7eb1ae0074b6928132f30c1897cd000f908b4f866c01039f787aec"
            assert cfg.recovery.max_s == 2.5
            for name in ("standard_arm_pose_l1", "ankle_target_rate_l2", "ankle_physics_acceleration_l2"):
                assert getattr(cfg.rewards, name).weight == getattr(parent.rewards, name).weight


class ImpactConfigModel(ConfigModel):
    def _is_config(self, name):
        return super()._is_config(name) or name in {
            "t800_lab.tasks.t800_impact.env_cfg", "t800_lab.tasks.t800_impact.terrain_cfg",
        }

    def _external(self, name, attr=None):
        full = name if attr is None else name + "." + attr
        if name == "t800_lab.tasks.t800_impact.impact_config":
            return getattr(load_cpu_module("impact_config"), attr)
        if full == "t800_lab.tasks.t800_impact.mdp":
            return mdp
        return super()._external(name, attr)

    def _import(self, namespace, statement, module_name):
        if isinstance(statement, ast.Import) and statement.names[0].name == "xml.etree.ElementTree":
            import xml.etree.ElementTree

            namespace[statement.names[0].asname] = xml.etree.ElementTree
            return
        super()._import(namespace, statement, module_name)


@pytest.fixture(scope="module")
def configurations():
    model = ImpactConfigModel(ROOT, "t800_lab")
    return model.load("t800_lab.tasks.t800_impact.env_cfg"), model.load("t800_lab.tasks.t800_base.env_cfg")


def test_impact_config_preserves_order_offsets_scales_and_stage_one(configurations):
    impact_module, base_module = configurations
    base = base_module.T800BaseEnvCfg()
    cfg = impact_module.T800ImpactEnvCfg()
    assert cfg.actions.joint_pos.joint_names == base.actions.joint_pos.joint_names
    assert cfg.actions.joint_pos.scale == base.actions.joint_pos.scale
    assert cfg.scene.robot.init_state.joint_pos == base.scene.robot.init_state.joint_pos
    assert cfg.sim.dt == 0.005 and cfg.decimation == 4
    for group in ("policy", "critic", "amp_policy", "amp_reference"):
        old = getattr(base.observations, group)
        new = getattr(cfg.observations, group)
        assert [name for name, value in vars(old).items() if hasattr(value, "func")] == [
            name for name, value in vars(new).items() if hasattr(value, "func")]
    assert cfg.observations.policy.joint_pos.params["noise_range"] == (-0.01, 0.01)
    assert cfg.observations.policy.joint_vel.params["noise_range"] == (-1.5, 1.5)
    assert not hasattr(cfg.observations.policy.joint_pos, "noise")
    assert cfg.events.push_robot is None
    assert cfg.events.base_external_force_torque is None
    assert base_module.T800BaseEnvCfg().events.push_robot is not None
    assert base_module.T800BaseEnvCfg().rewards.standard_head_pose_l1.weight == -0.2


def test_terrain_mix_has_exact_80_20_and_keeps_horizontal_shape(configurations):
    module, _ = configurations
    cfg = module.T800ImpactEnvCfg()
    terrain = cfg.scene.terrain.terrain_generator
    assert list(terrain.sub_terrains) == ["fight_mat_flat", "fight_mat_horizontal_obstacles"]
    assert [item.proportion for item in terrain.sub_terrains.values()] == [0.8, 0.2]
    assert terrain.num_cols == 10 and terrain.curriculum
    obstacle = terrain.sub_terrains["fight_mat_horizontal_obstacles"]
    assert obstacle.terrain_kind == "horizontal"
    assert obstacle.obstacle_count_range == (0, 20)
    assert obstacle.obstacle_exposure_range == (0.03, 0.12)


def test_soft_rewards_resolve_scene_entities_at_top_level(configurations):
    module, _ = configurations
    cfg = module.T800ImpactEnvCfg()
    for name, value in vars(cfg.rewards).items():
        if value is not None and getattr(value, "func", None) is mdp.recovery_scaled:
            assert "asset_cfg" not in value.params["term_params"], name
    arm = cfg.rewards.standard_arm_pose_l1
    assert len(arm.params["asset_cfg"].joint_names) == 10
    assert cfg.rewards.feet_air_time is None
    assert cfg.rewards.standard_head_pose_l1 is None
    assert cfg.rewards.joint_deviation_head is None
    assert cfg.rewards.termination_penalty.weight == -200.0
    assert cfg.rewards.ankle_actual_soft_limit_hinge_l2.weight == -0.2


def test_head_reset_event_has_mandatory_env_ids_for_isaac_manager(configurations):
    import inspect

    module, _ = configurations
    term = module.T800ImpactEnvCfg().events.head_faults
    args = inspect.signature(term.func).parameters
    assert list(args) == ["env", "env_ids"]
    assert args["env_ids"].default is inspect.Parameter.empty


def test_ground_sensor_filters_use_one_body_each_and_all_collider_links(configurations):
    import xml.etree.ElementTree as ET

    module, _ = configurations
    cfg = module.T800ImpactEnvCfg()
    bodies = {link.attrib["name"] for link in ET.parse(cfg.scene.robot.spawn.asset_path).getroot().findall("link")
              if link.find("collision") is not None}
    sensors = {name[len(mdp.GROUND_SENSOR_PREFIX):]: sensor for name, sensor in vars(cfg.scene).items()
               if name.startswith(mdp.GROUND_SENSOR_PREFIX)}
    assert set(sensors) == bodies
    for body, sensor in sensors.items():
        assert sensor.prim_path == "{ENV_REGEX_NS}/Robot/" + body
        assert sensor.filter_prim_paths_expr == ["/World/ground/terrain/mesh"]
        assert sensor.update_period == 0.005 and sensor.history_length == 4
        assert sensor.track_pose == (body in mdp.FOOT_BODIES)


def sensor(force=0.0, *, old_force=0.0, rotation=(1.0, 0.0, 0.0, 0.0), point_z=-0.064):
    history = torch.zeros(1, 4, 1, 1, 3)
    history[:, 0, 0, 0, 2] = force
    history[:, 1, 0, 0, 2] = old_force
    return SimpleNamespace(data=SimpleNamespace(
        force_matrix_w_history=history,
        force_matrix_w=history[:, 0],
        quat_w=torch.tensor(rotation).reshape(1, 1, 4),
        pos_w=torch.zeros(1, 1, 3),
        contact_pos_w=torch.tensor([0.0, 0.0, point_z]).reshape(1, 1, 1, 3),
    ))


def environment(**sensors):
    return SimpleNamespace(num_envs=1, device="cpu", scene=SimpleNamespace(sensors={
        mdp.GROUND_SENSOR_PREFIX + name: value for name, value in sensors.items()}))


def test_fall_uses_filtered_ground_history_and_does_not_infer_from_total_force():
    env = environment(LINK_BASE=sensor(old_force=5.0))
    assert mdp.ground_fall(env).item()
    env.scene.sensors[mdp.GROUND_SENSOR_PREFIX + "LINK_BASE"].data.force_matrix_w_history.zero_()
    # A large unfiltered contact (self, cage or opponent) is not a ground fall.
    env.scene.sensors["contact_forces"] = SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.ones(1, 1, 3) * 5000))
    assert not mdp.ground_fall(env).item()


@pytest.mark.parametrize(("rotation", "z", "expected"), [
    ((1.0, 0.0, 0.0, 0.0), -0.064, False),
    ((1.0, 0.0, 0.0, 0.0), -0.04, True),
    ((0.7071068, 0.7071068, 0.0, 0.0), -0.064, True),
])
def test_foot_side_detection(rotation, z, expected):
    env = environment(LINK_ANKLE_ROLL_L=sensor(5.0, rotation=rotation, point_z=z))
    assert mdp.ground_fall(env).item() is expected


def test_prior_sole_contact_is_not_paired_with_airborne_foot_rotation():
    env = environment(LINK_ANKLE_ROLL_L=sensor(0.0, old_force=5.0, rotation=(0.7071068, 0.7071068, 0.0, 0.0)))
    assert not mdp.ground_fall(env).item()
    env._impact_foot_ground_flags = torch.tensor([True])
    assert mdp.ground_fall(env).item()  # A previously sampled invalid substep is retained.


def test_soft_reward_uses_resolved_entity_and_keeps_body_action_penalty_head_independent():
    entity = object()
    runtime = SimpleNamespace(recovery_scale=lambda: torch.tensor([0.35]))
    env = SimpleNamespace(num_envs=1, device="cpu", impact_runtime=runtime,
        action_manager=SimpleNamespace(action=torch.ones(1, 25), prev_action=torch.zeros(1, 25)))

    def original(env, asset_cfg, multiplier):
        assert asset_cfg is entity
        return torch.tensor([multiplier])

    assert mdp.recovery_scaled(env, original, {"multiplier": 2.0}, asset_cfg=entity).item() == pytest.approx(0.7)
    expected = mdp.body_action_rate(env)
    env.action_manager.action[:, 23:] = float("nan")
    assert torch.equal(mdp.body_action_rate(env), expected)
    snapshot = runtime.last_step_style_scale.clone()
    runtime.recovery_scale = lambda: torch.ones(1)
    assert torch.equal(snapshot, torch.tensor([0.35]))


def test_amp_recovery_scale_is_per_transition_and_base_reward_stays_unchanged():
    from contextlib import nullcontext
    import math
    import runpy

    hook_class = runpy.run_path(str(TASK.parent / "t800_base/agent_cfg.py"))["ScaledAdversarialMotionPrior"]
    hook = object.__new__(hook_class)
    hook._sample_demonstration = lambda n: torch.zeros(n, 96)
    hook.transition_rms = SimpleNamespace(update=lambda value: None, normalize=lambda value: value)
    hook.discriminator = lambda observation: torch.zeros(observation.shape[0], 1)
    hook.reward_scale = 0.03
    hook.agent = SimpleNamespace(autocast=nullcontext, record=lambda **values: None)
    baseline = {"amp_obs": torch.zeros(2, 96), "reward": torch.zeros(2, 1)}
    gated = {"amp_obs": torch.zeros(2, 96), "reward": torch.zeros(2, 1),
             "impact_style_scale": torch.tensor([[1.0], [0.35]])}
    hook.post_step(baseline)
    hook.post_step(gated)
    assert torch.allclose(baseline["reward"], torch.full((2, 1), math.log(2) * 0.03))
    assert torch.equal(baseline["reward"][:1], gated["reward"][:1])
    assert gated["reward"][1].item() == pytest.approx(baseline["reward"][1].item() * 0.35)
    assert "impact_style_scale" not in gated
