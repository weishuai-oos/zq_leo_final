"""CPU contract checks for head telemetry isolation and logical action mapping."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
IMPACT = ROOT / "source/t800_lab/t800_lab/tasks/t800_impact"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


head = _load("impact_head_cpu", IMPACT / "head_mdp.py")


class Robot:
    def __init__(self, num_envs=3, *, headless=False, reverse=False):
        self.joint_names = list(head.LOGICAL_JOINT_NAMES[:-2] if headless else head.LOGICAL_JOINT_NAMES)
        if reverse:
            self.joint_names.reverse()
        self.num_joints = len(self.joint_names)
        indices = torch.tensor([head.LOGICAL_JOINT_NAMES.index(name) for name in self.joint_names])
        default = (indices.float() / 100).repeat(num_envs, 1)
        for name, value in zip(head.HEAD_JOINT_NAMES, head.HEAD_REFERENCE_POSITIONS):
            if name in self.joint_names:
                default[:, self.joint_names.index(name)] = value
        self.data = SimpleNamespace(
            default_joint_pos=default, default_joint_vel=torch.zeros_like(default),
            joint_pos=default + 0.1, joint_vel=default + 0.2,
            joint_stiffness=torch.full_like(default, 50.0), joint_damping=torch.full_like(default, 0.3),
            joint_effort_limits=torch.full_like(default, 52.0),
        )
        self.actuators = {"all": SimpleNamespace(
            joint_names=self.joint_names, stiffness=self.data.joint_stiffness.clone(),
            damping=self.data.joint_damping.clone(), effort_limit=self.data.joint_effort_limits.clone(),
        )}
        self.targets = None

    def _write(self, field, values, joint_ids, env_ids):
        getattr(self.data, field)[env_ids[:, None], joint_ids] = values

    def write_joint_stiffness_to_sim(self, values, joint_ids, env_ids):
        self._write("joint_stiffness", values, joint_ids, env_ids)

    def write_joint_damping_to_sim(self, values, joint_ids, env_ids):
        self._write("joint_damping", values, joint_ids, env_ids)

    def write_joint_effort_limit_to_sim(self, values, joint_ids, env_ids):
        self._write("joint_effort_limits", values, joint_ids, env_ids)

    def find_joints(self, patterns, preserve_order=False):
        names = []
        if preserve_order:
            for pattern in patterns:
                matched = [name for name in self.joint_names if re.fullmatch(pattern, name)]
                if not matched:
                    raise ValueError(pattern)
                names.extend(name for name in matched if name not in names)
        else:
            names = [name for name in self.joint_names if any(re.fullmatch(pattern, name) for pattern in patterns)]
        return [self.joint_names.index(name) for name in names], names

    def set_joint_position_target(self, values, joint_ids):
        self.targets = (values.clone(), joint_ids)


def _env(**kwargs):
    robot = Robot(**kwargs)
    return SimpleNamespace(
        scene={"robot": robot}, num_envs=robot.data.joint_pos.shape[0], device="cpu", physics_dt=0.005,
        cfg=SimpleNamespace(
            impact=SimpleNamespace(head_mode="full", mode="train", head_weak_probability=0.1,
                                   head_passive_probability=0.15, head_fault_delay_s=(2.0, 8.0),
                                   head_fault_warmup_updates=0),
            observations=SimpleNamespace(policy=SimpleNamespace(enable_corruption=True)),
        ),
        action_manager=SimpleNamespace(action=torch.arange(25, dtype=torch.float32).repeat(robot.data.joint_pos.shape[0], 1)),
    )


def _invalidate_head(robot):
    for field in ("joint_pos", "joint_vel"):
        getattr(robot.data, field)[:, robot.joint_names.index(head.HEAD_JOINT_NAMES[0])] = torch.nan
        getattr(robot.data, field)[:, robot.joint_names.index(head.HEAD_JOINT_NAMES[1])] = torch.inf


def test_order_is_exactly_the_stage_one_checkpoint_order():
    tree = ast.parse((ROOT / "source/t800_lab/t800_lab/assets/t800.py").read_text())
    existing = next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "T800_DFS_JOINT_NAMES" for target in node.targets))
    assert tuple(existing) == head.LOGICAL_JOINT_NAMES
    assert len(head.AMP_JOINT_NAMES) == 21
    assert head.AMP_JOINT_NAMES[-2:] == head.HEAD_JOINT_NAMES
    defaults_tree = ast.parse((ROOT / "source/t800_lab/t800_lab/tasks/t800_base/constants.py").read_text())
    nominal = next(ast.literal_eval(node.value) for node in defaults_tree.body
                   if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "T800_BOXING_GUARD_NOMINAL_JOINT_POS" for target in node.targets))
    assert tuple(nominal[name] for name in head.HEAD_JOINT_NAMES) == head.HEAD_REFERENCE_POSITIONS


@pytest.mark.parametrize("headless", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_logical_observations_keep_width_order_and_body_values(headless, reverse):
    env = _env(headless=headless, reverse=reverse)
    asset_cfg = SimpleNamespace(name="robot")
    for fn, width in ((head.joint_pos_masked, 25), (head.joint_pos_rel_masked, 25),
                      (head.joint_vel_masked, 25), (head.amp_joint_pos_rel_masked, 21),
                      (head.amp_joint_vel_masked, 21)):
        result = fn(env, asset_cfg=asset_cfg)
        assert result.shape == (3, width)
        assert torch.isfinite(result).all()
    q = head.joint_pos_masked(env)
    assert torch.allclose(q[:, :23], (torch.arange(23) / 100 + .1).repeat(3, 1))
    assert torch.allclose(q[:, -2:], torch.tensor(head.HEAD_REFERENCE_POSITIONS).repeat(3, 1))
    assert torch.allclose(head.joint_pos_rel_masked(env)[:, :23], torch.full((3, 23), .1))


def test_nan_inf_head_telemetry_is_excluded_after_noise_without_mutation():
    env = _env(reverse=True)
    robot = env.scene["robot"]
    _invalidate_head(robot)
    env.action_manager.action[:, 23:] = torch.tensor([torch.nan, -torch.inf])
    q = head.joint_pos_masked(env, noise_range=(-0.01, 0.01))
    dq = head.joint_vel_masked(env, noise_range=(-1.5, 1.5))
    action = head.last_action_masked(env)
    for result in (q, dq, action, head.joint_pos_rel_masked(env), head.amp_joint_pos_rel_masked(env), head.amp_joint_vel_masked(env)):
        assert torch.isfinite(result).all()
    assert torch.equal(q[:, -2:], head.head_reference_positions(env))
    assert torch.equal(dq[:, -2:], torch.zeros(3, 2))
    assert torch.equal(action[:, -2:], torch.zeros(3, 2))
    assert torch.isnan(env.action_manager.action[:, 23]).all()
    assert torch.isnan(robot.data.joint_pos[:, robot.joint_names.index(head.HEAD_JOINT_NAMES[0])]).all()
    env.cfg.observations.policy.enable_corruption = False
    assert torch.equal(head.joint_pos_masked(env, noise_range=(-1., 1.)), head.joint_pos_masked(env))


@pytest.mark.parametrize("headless", [False, True])
def test_reference_and_policy_amp_match_with_independent_joint_orders(headless):
    env = _env(headless=headless, reverse=True)
    robot = env.scene["robot"]
    reference = Robot(headless=headless)
    reference.reference_frame = SimpleNamespace(
        joint_pos=reference.data.joint_pos[:, None].clone(),
        joint_vel=reference.data.joint_vel[:, None].clone(),
        joint_pos_mask=torch.ones_like(reference.data.joint_pos[:, None]),
        joint_vel_mask=torch.ones_like(reference.data.joint_vel[:, None]),
    )
    env.scene["motion_reference"] = reference
    if not headless:
        _invalidate_head(robot)
        for field in ("joint_pos", "joint_vel", "joint_pos_mask", "joint_vel_mask"):
            getattr(reference.reference_frame, field)[:, :, -2:] = torch.nan
    assert torch.allclose(head.reference_amp_joint_pos_rel_masked(env), head.amp_joint_pos_rel_masked(env))
    assert torch.allclose(head.reference_amp_joint_vel_masked(env), head.amp_joint_vel_masked(env))


def test_export_masks_are_scriptable_and_remove_head_before_normalization():
    obs = torch.randn(2, 84)
    obs[:, [32, 33, 57, 58, 82, 83]] = torch.nan
    mask = torch.jit.script(head.HeadObservationMask((0.1, 0.2)))
    result = mask(obs)
    assert result.shape == (2, 84)
    assert torch.isfinite(result).all()
    assert torch.equal(result[:, 32:34], torch.tensor([[.1, .2], [.1, .2]]))
    body = [i for i in range(84) if i not in (32, 33, 57, 58, 82, 83)]
    assert torch.equal(result[:, body], obs[:, body])
    action = torch.randn(2, 25)
    action[:, -2:] = torch.inf
    out = torch.jit.script(head.HeadActionMask())(action)
    assert torch.equal(out[:, :23], action[:, :23])
    assert torch.equal(out[:, -2:], torch.zeros(2, 2))


def test_passive_fault_is_delayed_and_reset_restores_full_physical_and_model_gains():
    env = _env()
    cfg = env.cfg.impact
    cfg.head_passive_probability, cfg.head_weak_probability = 1., 0.
    cfg.head_fault_delay_s = (.01, .01)
    head.reset_head_faults(env)
    robot = env.scene["robot"]
    assert torch.equal(robot.data.joint_stiffness[:, -2:], torch.full((3, 2), 50.))
    head.update_head_faults(env, .005)
    assert not env._impact_head_faults.active.any()
    head.update_head_faults(env, .005)
    assert env._impact_head_faults.active.all()
    for field in ("joint_stiffness", "joint_damping", "joint_effort_limits"):
        assert torch.equal(getattr(robot.data, field)[:, -2:], torch.zeros(3, 2))
        assert (getattr(robot.data, field)[:, :23] > 0.).all()
    assert torch.equal(robot.actuators["all"].effort_limit[:, -2:], torch.zeros(3, 2))
    cfg.head_passive_probability = 0.
    head.reset_head_faults(env, torch.tensor([1]))
    assert torch.equal(robot.data.joint_stiffness[1, -2:], torch.full((2,), 50.))
    assert torch.equal(robot.actuators["all"].stiffness[1, -2:], torch.full((2,), 50.))
    assert torch.equal(robot.data.joint_stiffness[[0, 2], -2:], torch.zeros(2, 2))
    head.reset_head_faults(env)
    assert torch.equal(robot.data.joint_effort_limits[:, -2:], torch.full((3, 2), 52.))
    assert not env._impact_head_faults.active.any()


def test_repeated_weak_resets_do_not_accumulate_damage_and_headless_is_valid():
    env = _env()
    env.cfg.impact.head_passive_probability = 0.
    env.cfg.impact.head_weak_probability = 1.
    env.cfg.impact.head_fault_delay_s = (0., 0.)
    for _ in range(8):
        head.reset_head_faults(env)
        robot = env.scene["robot"]
        assert ((robot.data.joint_stiffness[:, -2:] >= 5.) & (robot.data.joint_stiffness[:, -2:] <= 20.)).all()
        assert ((robot.data.joint_effort_limits[:, -2:] >= 5.2) & (robot.data.joint_effort_limits[:, -2:] <= 20.8)).all()
    missing = _env(headless=True)
    head.reset_head_faults(missing)
    head.update_head_faults(missing, .005)
    assert not missing._impact_head_faults.ids


def test_fault_curriculum_preserves_fixed_head_adaptation_and_evaluation_bypasses_warmup():
    env = _env()
    cfg = env.cfg.impact
    cfg.head_passive_probability, cfg.head_weak_probability = 1., 0.
    cfg.head_fault_delay_s = (0., 0.)
    cfg.head_fault_warmup_updates = 500
    env.impact_runtime = SimpleNamespace(completed_updates=499)
    head.reset_head_faults(env)
    assert torch.equal(env._impact_head_faults.kind, torch.zeros(3, dtype=torch.long))
    assert not env._impact_head_faults.active.any()
    env.impact_runtime.completed_updates = 500
    head.reset_head_faults(env)
    assert env._impact_head_faults.active.all()
    env.impact_runtime.completed_updates = 0
    cfg.mode = "common"
    head.reset_head_faults(env)
    assert env._impact_head_faults.active.all()


def _actions_with_fake_isaac(monkeypatch):
    """Exercise our override against the local JointAction initialization contract."""
    class Base:
        def __init__(self, cfg, env):
            self.cfg, self._env, self.device, self.num_envs = cfg, env, env.device, env.num_envs
            self._asset = env.scene[cfg.asset_name]
            self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, cfg.preserve_order)
            self._num_joints = len(self._joint_ids)
            self._raw_actions = torch.zeros(self.num_envs, self.action_dim)
            self._processed_actions = torch.zeros_like(self.raw_actions)
            self._scale = torch.ones(self.num_envs, self.action_dim) if isinstance(cfg.scale, dict) else cfg.scale
            if isinstance(cfg.scale, dict):
                for pattern, value in cfg.scale.items():
                    for i, name in enumerate(self._joint_names):
                        if re.fullmatch(pattern, name):
                            self._scale[:, i] = value
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        @property
        def processed_actions(self):
            return self._processed_actions

        def process_actions(self, actions):
            self._raw_actions[:] = actions
            self._processed_actions = self._raw_actions * self._scale + self._offset

        def apply_actions(self):
            self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

        def reset(self, env_ids=None):
            self._raw_actions[env_ids] = 0.

    modules = {
        "isaaclab": ModuleType("isaaclab"), "isaaclab.envs": ModuleType("isaaclab.envs"),
        "isaaclab.envs.mdp": ModuleType("isaaclab.envs.mdp"), "isaaclab.envs.mdp.actions": ModuleType("isaaclab.envs.mdp.actions"),
        "isaaclab.envs.mdp.actions.actions_cfg": SimpleNamespace(JointPositionActionCfg=type("Cfg", (), {})),
        "isaaclab.envs.mdp.actions.joint_actions": SimpleNamespace(JointPositionAction=Base),
        "isaaclab.utils": SimpleNamespace(configclass=lambda cls: cls),
        "impact_head_test": ModuleType("impact_head_test"), "impact_head_test.head_mdp": head,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return _load("impact_head_test.actions", IMPACT / "actions.py")


@pytest.mark.parametrize("headless", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_actions_preserve_raw_draws_body_targets_and_physical_step_hook(monkeypatch, headless, reverse):
    module = _actions_with_fake_isaac(monkeypatch)
    env = _env(headless=headless, reverse=reverse)
    calls = []
    env.impact_runtime = SimpleNamespace(physics_step=lambda: calls.append(True))
    cfg = SimpleNamespace(asset_name="robot", joint_names=list(head.LOGICAL_JOINT_NAMES), preserve_order=True,
                          scale={name: .1 + i / 100 for i, name in enumerate(head.LOGICAL_JOINT_NAMES)},
                          offset=0., clip=None, use_default_offset=True)
    term = module.ImpactJointPositionAction(cfg, env)
    sampled = torch.randn(3, 25)
    sampled[:, -2:] = torch.tensor([torch.nan, torch.inf])
    snapshot = sampled.clone()
    term.process_actions(sampled)
    assert term.action_dim == 25
    torch.testing.assert_close(sampled, snapshot, equal_nan=True)
    torch.testing.assert_close(term.raw_actions, snapshot, equal_nan=True)
    robot = env.scene["robot"]
    assert term.processed_actions.shape == (3, robot.num_joints)
    for column, name in enumerate(term._joint_names):
        actual = robot.joint_names.index(name)
        expected = robot.data.default_joint_pos[:, actual]
        if name not in head.HEAD_JOINT_NAMES:
            logical = head.LOGICAL_JOINT_NAMES.index(name)
            expected = expected + sampled[:, logical] * cfg.scale[name]
        torch.testing.assert_close(term.processed_actions[:, column], expected)
    term.apply_actions()
    term.apply_actions()
    assert len(calls) == 2
    assert torch.isfinite(robot.targets[0]).all()
    term.reset(torch.tensor([1]))
    assert torch.equal(term.raw_actions[1], torch.zeros(25))


def test_headless_urdf_removes_graph_and_retains_absolute_body_meshes(tmp_path):
    module = _load("impact_head_assets_cpu", IMPACT / "head_assets.py")
    source = ROOT / "source/t800_lab/t800_lab/assets/resources/t800/urdf/serial_t800.urdf"
    before = source.read_bytes()
    target = Path(module.make_headless_urdf(source, tmp_path / "headless.urdf"))
    full, missing = ET.fromstring(before), ET.parse(target).getroot()
    assert source.read_bytes() == before
    assert len(full.findall("link")) - len(missing.findall("link")) == 2
    moving = [joint for joint in missing.findall("joint") if joint.attrib["type"] != "fixed"]
    assert len(moving) == 23
    names = {link.attrib["name"] for link in missing.findall("link")}
    assert not module.HEAD_LINK_NAMES.intersection(names)
    for joint in missing.findall("joint"):
        for tag in ("parent", "child"):
            assert joint.find(tag).attrib["link"] in names
    for mesh in missing.iter("mesh"):
        assert Path(mesh.attrib["filename"]).is_absolute()
        assert Path(mesh.attrib["filename"]).exists()
    with pytest.raises(ValueError, match="differ"):
        module.make_headless_urdf(source, source)
