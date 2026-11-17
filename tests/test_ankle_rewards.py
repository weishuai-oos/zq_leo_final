from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
MDP_PATH = ROOT / "source/t800_lab/t800_lab/tasks/t800_base/ankle_stable_mdp.py"


def _load_mdp():
    spec = importlib.util.spec_from_file_location("ankle_stable_mdp_cpu", MDP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeAsset:
    joint_names = ["J11_ANKLE_ROLL_R", "J04_ANKLE_PITCH_L", "J10_ANKLE_PITCH_R", "J05_ANKLE_ROLL_L"]

    def __init__(self, q, qdes, vel, limits):
        self.find_joints_calls = 0
        self.data = SimpleNamespace(joint_pos=q, joint_pos_target=qdes, joint_vel=vel, soft_joint_pos_limits=limits)

    def find_joints(self, names, preserve_order=True):
        self.find_joints_calls += 1
        assert preserve_order
        return [self.joint_names.index(name) for name in names], None


class FakeEnv:
    def __init__(self, asset):
        self.scene = {"robot": asset}
        self.episode_length_buf = torch.zeros(asset.data.joint_pos.shape[0], dtype=torch.long)
        self.common_step_counter = 0

    def step(self, q, qdes, vel):
        self.scene["robot"].data.joint_pos = q
        self.scene["robot"].data.joint_pos_target = qdes
        self.scene["robot"].data.joint_vel = vel
        self.common_step_counter += 1
        self.episode_length_buf += 1


def _env(num_envs=2):
    q = torch.zeros(num_envs, 4)
    qdes = torch.zeros_like(q)
    vel = torch.zeros_like(q)
    limits = torch.tensor([[-1.0, 1.0]] * 4).repeat(num_envs, 1, 1)
    return FakeEnv(FakeAsset(q, qdes, vel, limits))


def test_quiet_target_and_actual_trajectory_is_zero_and_finite():
    m = _load_mdp()
    env = _env()
    assert torch.equal(m.ankle_target_rate_l2(env), torch.zeros(2))
    env.step(torch.zeros(2, 4), torch.zeros(2, 4), torch.zeros(2, 4))
    for fn in (
        m.ankle_target_rate_l2,
        m.ankle_target_acceleration_l2,
        m.ankle_overspeed_hinge_l2,
        m.ankle_acceleration_l2,
    ):
        assert torch.allclose(fn(env), torch.zeros(2))
        assert torch.isfinite(fn(env)).all()


def test_constant_velocity_target_is_first_order_only_and_jitter_is_penalized():
    m = _load_mdp()
    env = _env(1)
    env.step(torch.zeros(1, 4), torch.ones(1, 4) * 0.02, torch.zeros(1, 4))
    m.ankle_target_rate_l2(env)  # reward manager samples the reset/first frame
    env.step(torch.zeros(1, 4), torch.ones(1, 4) * 0.04, torch.zeros(1, 4))
    assert m.ankle_target_rate_l2(env).item() > 0
    assert math.isclose(m.ankle_target_acceleration_l2(env).item(), 0.0, abs_tol=1e-8)
    env.step(torch.zeros(1, 4), torch.ones(1, 4) * 0.0, torch.zeros(1, 4))
    assert m.ankle_target_acceleration_l2(env).item() > 0


def test_overspeed_uses_pitch_roll_physical_thresholds_and_actual_acceleration():
    m = _load_mdp()
    env = _env(1)
    env.step(torch.zeros(1, 4), torch.zeros(1, 4), torch.tensor([[9.0, 6.0, 9.0, 6.0]]))
    assert m.ankle_overspeed_hinge_l2(env).item() > 0
    env.step(torch.zeros(1, 4), torch.zeros(1, 4), torch.zeros(1, 4))
    assert m.ankle_acceleration_l2(env).item() > 0


def test_name_resolution_survives_scrambled_simulator_order():
    m = _load_mdp()
    env = _env(1)
    # Canonical pitch target is 0.02 rad even though sim order is scrambled.
    env.step(torch.zeros(1, 4), torch.tensor([[0.0, 0.02, 0.0, 0.02]]), torch.zeros(1, 4))
    m.ankle_target_rate_l2(env)
    assert math.isclose(m.ankle_target_rate_l2(env).item(), 0.0, abs_tol=1e-8)
    env.step(torch.zeros(1, 4), torch.tensor([[0.0, 0.04, 0.0, 0.04]]), torch.zeros(1, 4))
    assert math.isclose(m.ankle_target_rate_l2(env).item(), 0.5 * (0.02 / 0.02 / 4.0) ** 2, rel_tol=1e-6)
    assert env.scene["robot"].find_joints_calls == 1
    env.step(torch.zeros(1, 4), torch.tensor([[0.0, 0.06, 0.0, 0.06]]), torch.zeros(1, 4))
    m.ankle_target_rate_l2(env)
    assert env.scene["robot"].find_joints_calls == 1


def test_reset_and_subset_reset_do_not_leak_history_or_create_spikes():
    m = _load_mdp()
    env = _env(3)
    env.step(torch.zeros(3, 4), torch.ones(3, 4), torch.zeros(3, 4))
    env.episode_length_buf[1] = 0
    env.scene["robot"].data.joint_pos_target[1] = 0.0
    env.common_step_counter += 1
    assert math.isclose(m.ankle_target_acceleration_l2(env)[1].item(), 0.0, abs_tol=1e-8)
    m.reset_ankle_stable_history(env, [0])
    env.scene["robot"].data.joint_pos_target[0] = 1.0
    env.common_step_counter += 1
    assert math.isclose(m.ankle_target_acceleration_l2(env)[0].item(), 0.0, abs_tol=1e-8)
    assert torch.isfinite(m.ankle_actual_soft_limit_hinge_l2(env)).all()


def test_target_inside_but_actual_outside_soft_limits_has_only_actual_limit_penalty():
    m = _load_mdp()
    env = _env(1)
    env.scene["robot"].data.joint_pos[:] = 2.0
    env.step(torch.ones(1, 4) * 2.0, torch.zeros(1, 4), torch.zeros(1, 4))
    assert math.isclose(m.ankle_target_soft_limit_hinge_l2(env).item(), 0.0, abs_tol=1e-8)
    assert m.ankle_actual_soft_limit_hinge_l2(env).item() > 0


def test_reset_detects_first_post_reset_progress_one_and_masks_second_differences():
    m = _load_mdp()
    env = _env(1)
    zero = torch.zeros(1, 4)
    env.step(zero, zero, zero)
    m.ankle_target_acceleration_l2(env)
    env.step(zero, torch.ones(1, 4) * 0.02, zero)
    m.ankle_target_acceleration_l2(env)
    env.step(zero, torch.ones(1, 4) * 0.06, zero)
    assert m.ankle_target_acceleration_l2(env).item() > 0

    # IsaacLab has already incremented this reset env to one when rewards run.
    env.episode_length_buf[:] = 1
    env.common_step_counter += 1
    env.scene["robot"].data.joint_pos_target[:] = 3.0
    assert m.ankle_target_acceleration_l2(env).item() == 0.0
    env.step(zero, torch.ones(1, 4) * 4.0, zero)
    assert m.ankle_target_acceleration_l2(env).item() == 0.0
    env.step(zero, torch.ones(1, 4) * 5.0, zero)
    assert m.ankle_target_acceleration_l2(env).item() == 0.0


def test_nonfinite_physics_is_finite_and_penalized_not_silently_zeroed():
    m = _load_mdp()
    env = _env(1)
    env.step(torch.zeros(1, 4), torch.full((1, 4), float("nan")), torch.full((1, 4), float("inf")))
    values = [
        m.ankle_target_rate_l2(env),
        m.ankle_target_acceleration_l2(env),
        m.ankle_overspeed_hinge_l2(env),
        m.ankle_acceleration_l2(env),
        m.ankle_target_soft_limit_hinge_l2(env),
        m.ankle_actual_soft_limit_hinge_l2(env),
    ]
    assert all(torch.isfinite(value).all() for value in values)
    assert m.ankle_overspeed_hinge_l2(env).item() == 1_000_000.0
