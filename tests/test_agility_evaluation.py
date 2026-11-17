"""CPU evidence for command timing, first-fall censoring and response outcomes."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "agility_evaluation", Path(__file__).resolve().parents[1] / "scripts/cusrl/evaluate_agility.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_schedules_cover_exactly_15_seconds_and_edges_have_no_one_frame_delay():
    scenarios = MODULE.build_scenarios()
    assert len(scenarios) == 16
    for scenario in scenarios:
        commands, segments = MODULE.compile_schedule(scenario, 0.02)
        assert commands.shape == (750, 3)
        assert segments.shape == (750,)
        assert not commands[:100].any()
        assert commands[100].any()  # First post-2s step uses new command now.
    fast, _ = MODULE.compile_schedule(next(item for item in scenarios if item.name == "fast_fb"), 0.02)
    assert fast[124, 0] == pytest.approx(0.8)
    assert fast[125, 0] == pytest.approx(-0.8)
    assert not fast[500:].any()


def test_first_fall_terminal_frame_is_valid_but_revival_is_excluded():
    falls = np.array([[False, False], [True, False], [False, False], [True, False]])
    valid = MODULE.first_fall_valid_mask(falls)
    assert valid[:, 0].tolist() == [True, True, False, False]
    assert valid[:, 1].all()
    velocity = np.array([[0.0, 0, 0], [1.0, 0, 0], [1000.0, 0, 0], [1000.0, 0, 0]])
    summary = MODULE.summarize_trial(velocity, np.zeros_like(velocity), np.zeros(4), np.ones(4), valid[:, 0], falls[:, 0], 0.02)
    assert summary["tracking_rmse_xy"] == pytest.approx(np.sqrt(0.5))
    assert summary["first_fall_s"] == pytest.approx(0.04)


def test_causal_filter_has_no_future_leak_and_no_post_fall_output():
    values = np.zeros((12, 3))
    values[5:] = 1.0
    valid = np.arange(12) < 10
    smooth = MODULE.causal_mean(values, valid, 5)
    assert not smooth[:5].any()
    assert smooth[5, 0] == pytest.approx(0.2)
    assert smooth[9, 0] == 1
    assert np.isnan(smooth[10:]).all()


def test_reach_requires_full_hold_and_failures_do_not_become_zero_latency():
    valid = np.ones(20, dtype=bool)
    no_fall = np.zeros(20, dtype=bool)
    condition = np.zeros(20, dtype=bool)
    condition[2:7] = True  # 5 samples span only 80ms; insufficient.
    failure = MODULE.sustained_reach(condition, valid, no_fall, 0.02)
    assert failure["status"] == "not_reached"
    assert failure["latency_s"] is None
    condition[7] = True
    success = MODULE.sustained_reach(condition, valid, no_fall, 0.02)
    assert success["latency_s"] == pytest.approx(0.06)
    assert success["confirmed_s"] == pytest.approx(0.16)
    falls = no_fall.copy()
    falls[4] = True
    censored = MODULE.first_fall_valid_mask(falls)
    failure = MODULE.sustained_reach(condition, censored, falls, 0.02)
    assert failure["status"] == "fell_before_reach"
    assert failure["latency_s"] is None
    unexposed = MODULE.sustained_reach(condition, np.zeros(20, dtype=bool), no_fall, 0.02)
    assert unexposed["status"] == "not_exposed"


def test_stop_time_and_distance_use_pre_command_position_and_confirmation():
    # +1m/s command changes to zero after 1 second. Speed becomes zero 0.2s
    # after the stop, then remains zero for the mandatory 0.1-second hold.
    dt = 0.02
    commands = np.zeros((100, 3))
    commands[:50, 0] = 1
    velocity = np.zeros_like(commands)
    velocity[:59, 0] = 1
    positions = np.zeros_like(commands)
    positions[:, 0] = np.cumsum(velocity[:, 0]) * dt
    records = MODULE.transition_measurements(
        commands, velocity, positions, np.ones(100, dtype=bool), np.zeros(100, dtype=bool), np.zeros(3), dt, 0,
    )
    stop = records[0]["stop"]
    assert stop["status"] == "reached"
    assert stop["latency_s"] == pytest.approx(0.2)
    assert stop["confirmed_s"] == pytest.approx(0.3)
    assert stop["stop_distance_xy_m"] == pytest.approx(0.18)
    assert stop["stop_path_xy_m"] == pytest.approx(0.18)


def test_direct_reversal_zero_cross_and_signed_targets():
    commands = np.zeros((100, 3))
    commands[:50, 0], commands[50:, 0] = 1.0, -1.0
    velocity = np.zeros_like(commands)
    velocity[:52, 0] = 1.0
    velocity[52:55, 0] = 0.0
    velocity[55:60, 0] = -0.6
    velocity[60:, 0] = -1.0
    records = MODULE.transition_measurements(
        commands, velocity, np.zeros_like(velocity), np.ones(100, dtype=bool), np.zeros(100, dtype=bool), np.zeros(3), 0.02, 0,
    )
    result = records[0]
    assert result["zero_cross"]["latency_s"] == pytest.approx(0.06)
    assert result["reach_50"]["latency_s"] == pytest.approx(0.12)
    assert result["reach_90"]["latency_s"] == pytest.approx(0.22)
    assert result["steady_state_gain"]["value"] == pytest.approx(1.0)


def test_strict_stop_does_not_cancel_opposite_raw_velocities():
    commands = np.zeros((100, 3))
    commands[:20, 0] = 1.0
    velocity = np.zeros_like(commands)
    velocity[:, 0] = np.where(np.arange(100) % 2, -0.2, 0.2)
    valid, falls = np.ones(100, dtype=bool), np.zeros(100, dtype=bool)
    positions = np.cumsum(velocity, axis=0) * 0.02
    mean_velocity = MODULE.causal_mean(velocity, valid, 5)
    filtered = MODULE.transition_measurements(
        commands, mean_velocity, positions, valid, falls, np.zeros(3), 0.02, 0,
    )[0]["stop"]
    strict = MODULE.strict_stop_measurements(
        commands, velocity, positions, valid, falls, np.zeros(3), 0.02,
    )[0]
    assert filtered["status"] == "reached"
    assert strict["status"] == "not_reached"
    assert strict["latency_s"] is None


def test_observation_command_slice_is_derived_and_asserted():
    manager = SimpleNamespace(
        active_terms={"policy": ["base_ang_vel", "gravity", "velocity_commands", "joints"]},
        group_obs_term_dim={"policy": [(3,), (3,), (3,), (25,)]},
    )
    assert MODULE.command_observation_slice(manager, "policy") == slice(6, 9)
    manager.active_terms["policy"][2] = "wrong_term"
    with pytest.raises(ValueError, match="Missing"):
        MODULE.command_observation_slice(manager, "policy")


def test_controller_updates_actor_and_critic_inputs_before_action_without_history_refresh():
    import torch

    term = SimpleNamespace(
        _resample_command=lambda ids: None,
        _update_command=lambda: None,
        vel_command_b=torch.zeros(2, 3),
        time_left=torch.zeros(2),
        is_heading_env=torch.zeros(2, dtype=torch.bool),
        is_standing_env=torch.zeros(2, dtype=torch.bool),
        heading_target=torch.zeros(2),
        command_mode=torch.zeros(2, dtype=torch.long),
        transform_teleop_command=lambda commands: commands.clone(),
    )
    term.command = term.vel_command_b
    manager = SimpleNamespace(
        active_terms={
            "policy": ["base_ang_vel", "gravity", "velocity_commands", "joints"],
            "critic": ["base_lin_vel", "base_ang_vel", "gravity", "velocity_commands", "joints"],
        },
        group_obs_term_dim={"policy": [(3,), (3,), (3,), (25,)], "critic": [(3,), (3,), (3,), (3,), (25,)]},
    )
    env = SimpleNamespace(device="cpu", command_manager=SimpleNamespace(get_term=lambda name: term), observation_manager=manager)
    commands = np.zeros((2, 2, 3), dtype=np.float32)
    commands[1, :, 0] = 0.6
    controller = MODULE.CommandController(env, commands)
    try:
        observation, state = torch.zeros(2, 84), torch.zeros(2, 87)
        next_observation, next_state = controller.before_act(1, observation, state)
        np.testing.assert_allclose(next_observation[:, 6:9].numpy(), commands[1])
        np.testing.assert_allclose(next_state[:, 9:12].numpy(), commands[1])
        np.testing.assert_allclose(term.command.numpy(), commands[1])
        assert not observation.any() and not state.any()
        term._update_command()  # End-of-step update cannot replace this edge.
        assert torch.equal(term.command, next_observation[:, 6:9])
    finally:
        controller.close()
