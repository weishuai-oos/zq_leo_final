"""Physics-step stability costs must survive cancellation, contact changes and reset."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "source/t800_lab/t800_lab/tasks/t800_ankle/stability.py"
SETTINGS = {
    "contact_force_enter_n": 50.0,
    "contact_force_exit_n": 10.0,
    "contact_gap_tolerance_s": 0.02,
    "support_settle_time_s": 0.03,
    "acceleration_deadband_rad_s2": 100.0,
    "acceleration_scale_rad_s2": 100.0,
    "foot_roll_deadband_rad_s": 0.5,
    "foot_pitch_deadband_rad_s": 1.0,
    "foot_roll_scale_rad_s": 1.0,
    "foot_pitch_scale_rad_s": 2.0,
    "normalized_error_cap": 20.0,
}


@pytest.fixture(scope="module", params=(MODULE_PATH, ROOT / "source/t800_lab/t800_lab/tasks/t800_impact_v2/mdp.py"))
def stability_module(request):
    spec = importlib.util.spec_from_file_location("_ankle_substep_cpu", request.param)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Frames:
    """Feed independent synthetic physical states through the public accumulator API."""

    def __init__(self, module, num_envs=1):
        self.count = num_envs
        self.tracker = module.AnkleStabilityAccumulator(num_envs, "cpu", dict(SETTINGS), 0.005, 4)
        self.counter = 0
        self.velocity = torch.zeros(num_envs, 4)
        self.omega = torch.zeros(num_envs, 2, 3)
        self.quat = torch.zeros(num_envs, 2, 4)
        self.quat[..., 0] = 1.0
        self.force = torch.zeros(num_envs, 2, 3)

    def begin(self, velocity=None):
        if velocity is not None:
            self.velocity = torch.full((self.count, 4), float(velocity))
        self.tracker.begin(self.velocity, self.counter)

    def frame(self, velocity=None, *, counter=None, dt=0.005):
        if velocity is not None:
            self.velocity = torch.full((self.count, 4), float(velocity))
        number = self.counter + 1 if counter is None else counter
        self.tracker.observe(self.velocity, self.omega, self.quat, self.force, number, dt)
        self.counter = number

    def control(self, velocities=None, *, read=True):
        self.begin()
        for velocity in velocities if velocities is not None else (None,) * 4:
            self.frame(velocity)
        return self.tracker.values() if read else None

    def settle(self):
        self.force[..., 2] = 100.0
        self.control(read=False)
        self.control(read=False)


def test_substep_changes_do_not_cancel_when_control_endpoints_match(stability_module):
    frames = Frames(stability_module)
    acceleration, rotation = frames.control((1.0, 0.0, 1.0, 0.0))
    # Four |delta v|=1 changes at 5ms each: |a|=200, excess/scale=1.
    # The 20ms endpoint difference is zero, but the physical mean-square is one.
    assert frames.velocity.sum().item() == 0.0
    assert acceleration.item() == pytest.approx(1.0)
    assert rotation.item() == 0.0
    assert frames.tracker.sample_count == 4


def test_begin_copies_velocity_snapshot_instead_of_aliasing_simulator_buffer(stability_module):
    frames = Frames(stability_module)
    frames.begin()
    frames.velocity.fill_(1.0)
    for _ in range(4):
        frames.frame()
    acceleration, _ = frames.tracker.values()
    # Only the first substep changes by 1 rad/s: (1 + 0 + 0 + 0) / 4.
    assert acceleration.item() == pytest.approx(0.25)


def test_duplicate_physics_counter_does_not_change_cost_or_support_clock(stability_module):
    frames = Frames(stability_module)
    frames.force[..., 2] = 100.0
    frames.begin()
    frames.frame(1.0)
    age = frames.tracker.support_age.clone()
    frames.frame(999.0, counter=1)
    assert frames.tracker.sample_count == 1
    assert torch.equal(frames.tracker.support_age, age)
    for velocity in (0.0, 1.0, 0.0):
        frames.frame(velocity)
    assert frames.tracker.values()[0].item() == pytest.approx(1.0)


def test_missing_samples_and_counter_gaps_are_rejected(stability_module):
    frames = Frames(stability_module)
    frames.begin()
    frames.frame()
    with pytest.raises((ValueError, RuntimeError)):
        frames.tracker.values()
    with pytest.raises((ValueError, RuntimeError)):
        frames.frame(counter=3)


def test_extra_substep_and_wrong_physics_dt_are_rejected(stability_module):
    frames = Frames(stability_module)
    frames.control()
    with pytest.raises((ValueError, RuntimeError)):
        frames.frame()
    other = Frames(stability_module)
    other.begin()
    with pytest.raises((ValueError, RuntimeError)):
        other.frame(dt=0.02)


def test_reading_costs_has_no_sampling_or_contact_side_effects(stability_module):
    frames = Frames(stability_module)
    frames.settle()
    frames.omega[..., 0] = 1.5
    expected = tuple(value.clone() for value in frames.control())
    age = frames.tracker.support_age.clone()
    for _ in range(3):
        result = frames.tracker.values()
        assert all(torch.equal(actual, wanted) for actual, wanted in zip(result, expected))
        assert frames.tracker.sample_count == 4
        assert torch.equal(frames.tracker.support_age, age)


def test_collection_continues_when_no_reward_function_reads_previous_step(stability_module):
    frames = Frames(stability_module)
    frames.force[..., 2] = 100.0
    frames.omega[..., 0] = 1.5
    frames.control(read=False)  # Mirrors zero-weight warmup: no reward read occurs.
    acceleration, rotation = frames.control()
    assert acceleration.item() == 0.0
    assert rotation.item() > 0.0
    assert frames.tracker.support_age[0, 0].item() == pytest.approx(0.04)


def test_partial_reset_preserves_other_environment_and_completed_step_cost(stability_module):
    frames = Frames(stability_module, num_envs=2)
    frames.settle()
    frames.omega[..., 0] = 1.5
    before = tuple(value.clone() for value in frames.control())
    age = frames.tracker.support_age.clone()
    frames.tracker.reset(torch.tensor([0]))
    assert frames.tracker.sample_count == 4
    assert torch.count_nonzero(frames.tracker.support_age[0]) == 0
    assert torch.equal(frames.tracker.support_age[1], age[1])
    assert all(torch.equal(actual, wanted) for actual, wanted in zip(frames.tracker.values(), before))
    _, rotation = frames.control()
    assert rotation[0].item() == 0.0  # Reset foot has only 20ms support again.
    assert rotation[1].item() == pytest.approx(0.5)


def test_new_control_step_after_reset_uses_reset_velocity_not_pre_reset_velocity(stability_module):
    frames = Frames(stability_module)
    frames.velocity.fill_(100.0)
    assert frames.control()[0].item() == 0.0
    frames.tracker.reset(torch.tensor([0]))
    frames.velocity.zero_()  # Simulator state has jumped during reset, without elapsed physics.
    assert frames.control()[0].item() == 0.0


def test_contact_entry_hysteresis_and_touchdown_grace(stability_module):
    frames = Frames(stability_module)
    frames.omega[..., 0] = 1.5
    frames.force[..., 2] = 49.0
    frames.control(read=False)
    _, rotation = frames.control()
    assert rotation.item() == 0.0
    assert torch.count_nonzero(frames.tracker.support_age) == 0
    frames.force[..., 2] = 51.0
    assert frames.control()[1].item() == 0.0  # First 20ms are below 30ms grace.
    frames.force[..., 2] = 20.0  # Below enter threshold, above exit threshold.
    assert frames.control()[1].item() > 0.0
    assert frames.tracker.support_age[0, 0].item() == pytest.approx(0.04)


def test_short_contact_gap_preserves_age_without_counting_air_time(stability_module):
    frames = Frames(stability_module)
    frames.settle()
    frames.omega[..., 0] = 1.5
    age = frames.tracker.support_age.clone()
    frames.force[..., 2] = 0.0
    assert frames.control()[1].item() == 0.0
    assert torch.equal(frames.tracker.support_age, age)  # 20ms gap retains but does not advance age.
    frames.force[..., 2] = 100.0
    assert frames.control()[1].item() == pytest.approx(0.5)
    assert frames.tracker.support_age[0, 0].item() == pytest.approx(0.06)


def test_long_contact_gap_clears_age_and_restarts_touchdown_grace(stability_module):
    frames = Frames(stability_module)
    frames.settle()
    frames.omega[..., 0] = 1.5
    frames.force[..., 2] = 0.0
    frames.control(read=False)
    frames.control(read=False)  # 40ms exceeds the 20ms gap tolerance.
    assert torch.count_nonzero(frames.tracker.support_age) == 0
    frames.force[..., 2] = 100.0
    assert frames.control()[1].item() == 0.0


def test_swing_foot_rotation_does_not_penalize_supported_other_foot(stability_module):
    frames = Frames(stability_module)
    frames.force[:, 0, 2] = 100.0
    frames.control(read=False)
    frames.control(read=False)
    frames.omega[:, 1, :2] = 100.0  # Right foot is in the air.
    assert frames.control()[1].item() == 0.0
    frames.omega[:, 0, 0] = 1.5
    # Left foot: roll cost 1 and pitch cost 0; average two axes and two feet.
    assert frames.control()[1].item() == pytest.approx(0.25)


def test_roll_cost_is_stronger_for_equal_excess_and_yaw_is_free(stability_module):
    frames = Frames(stability_module)
    frames.settle()
    frames.omega[..., 0] = 1.5  # 1rad/s above the roll deadband; normalized excess 1.
    roll = frames.control()[1].item()
    frames.omega.zero_()
    frames.omega[..., 1] = 2.0  # 1rad/s above the pitch deadband; normalized excess 0.5.
    pitch = frames.control()[1].item()
    frames.omega.zero_()
    frames.omega[..., 2] = 100.0
    yaw = frames.control()[1].item()
    assert roll == pytest.approx(0.5)
    assert pitch == pytest.approx(0.125)
    assert roll == pytest.approx(4 * pitch)
    assert yaw == 0.0


def test_angular_cost_uses_foot_axes_on_a_rotated_support_surface(stability_module):
    frames = Frames(stability_module)
    frames.settle()
    frames.quat[:] = torch.tensor([math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0])
    frames.omega[..., 0] = 5.0  # A 90deg Y rotation maps local foot yaw onto world X.
    assert frames.control()[1].item() == pytest.approx(0.0, abs=1e-6)
    frames.omega.zero_()
    frames.omega[..., 2] = 1.5  # World Z is now local -roll, so it must be penalized.
    assert frames.control()[1].item() == pytest.approx(0.5, rel=1e-5)


def test_large_and_nonfinite_motion_is_capped_finite_and_not_silently_zero(stability_module):
    frames = Frames(stability_module)
    frames.settle()
    frames.omega[..., :2] = 1e10
    acceleration, rotation = frames.control((1e10, -1e10, 1e10, -1e10))
    assert acceleration.item() == pytest.approx(400.0)
    assert rotation.item() == pytest.approx(400.0)
    frames.omega[..., :2] = float("inf")
    acceleration, rotation = frames.control((float("nan"),) * 4)
    assert torch.isfinite(acceleration).all() and torch.isfinite(rotation).all()
    assert acceleration.item() > 0.0 and rotation.item() > 0.0
    assert acceleration.item() <= 400.0 and rotation.item() <= 400.0
