"""Physics-onset capture and marker geometry, without starting Isaac or Kit."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from test_impact_runtime import M, _cfg, _env


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "impact_visualization_cpu_test", ROOT / "scripts/cusrl/impact_visualization.py"
)
V = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = V
assert SPEC.loader is not None
SPEC.loader.exec_module(V)


def make_runtime(n=1, **cfg_overrides):
    env = _env(n)
    data = env.scene["robot"].data
    data.body_link_pos_w = torch.zeros(n, len(env.scene["robot"].body_names), 3)
    return env, M.ImpactRuntime(env, _cfg(**cfg_overrides))


def hit(*, direction=(1.0, 0.0, 0.0), peak=1000.0, stress=False):
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    return V.ImpactHit(
        env_index=0, started_at=0.0, point_w=(1.0, -2.0, 3.0),
        direction_w=tuple(direction), peak_n=peak, duration_s=0.012,
        stress=stress, body_name="LINK_TORSO_YAW",
    )


def rotate(quaternion, vector):
    """Independent wxyz rotation used to inspect the generated USD transforms."""
    w = quaternion[0]
    xyz = np.asarray(quaternion[1:])
    vector = np.asarray(vector)
    return vector + 2 * w * np.cross(xyz, vector) + 2 * np.cross(xyz, np.cross(xyz, vector))


def test_short_pulse_is_visible_after_it_ends_between_policy_steps():
    env, runtime = make_runtime(fixed_duration_s=0.012)
    history = V.ImpactHistory(hold_s=0.3)
    clock = [0.0]
    runtime.hit_observer = lambda ids: history.capture(runtime, ids, clock[0])

    # Four 5 ms physics steps make one 20 ms policy step. The pulse is already
    # over by the third physics step, so polling active in a PlayerHook misses it.
    for step in range(4):
        clock[0] = step * env.physics_dt
        runtime.physics_step()
    assert not runtime.active.any()
    assert not runtime.forces_b.any()
    visible = history.visible(0.020)
    assert len(visible) == 1
    assert visible[0].started_at == 0.0
    assert visible[0].duration_s == pytest.approx(0.012)
    assert len(history.visible(0.299)) == 1
    assert history.visible(0.301) == []


def test_capture_uses_link_frame_point_and_freezes_the_world_space_event():
    env, runtime = make_runtime()
    runtime.physics_step()
    data = env.scene["robot"].data
    data.body_link_pos_w[0, 0] = torch.tensor([1.0, 2.0, 3.0])
    data.body_link_quat_w[0, 0] = torch.tensor([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
    runtime.hit_point_b[0] = torch.tensor([0.1, 0.2, 0.3])
    runtime.force_direction_w[0] = torch.tensor([0.0, 1.0, 0.0])
    history = V.ImpactHistory()
    event = history.capture(runtime, torch.tensor([0]), 1.0)[0]
    np.testing.assert_allclose(event.point_w, [0.8, 2.1, 3.3], atol=1e-6)
    assert event.direction_w == (0.0, 1.0, 0.0)
    assert event.body_name == "LINK_TORSO_YAW"

    # A turn or translation after contact must not move a retained world-space
    # arrow, and its immutable snapshots must not alias the simulator tensors.
    data.body_link_pos_w.fill_(100)
    data.body_link_quat_w[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    runtime.force_direction_w[0] = torch.tensor([1.0, 0.0, 0.0])
    runtime.hit_point_b.zero_()
    assert history.visible(1.1) == [event]
    np.testing.assert_allclose(event.point_w, [0.8, 2.1, 3.3], atol=1e-6)
    assert event.direction_w == (0.0, 1.0, 0.0)


def test_combination_hits_coexist_and_history_is_bounded_and_reset_per_environment():
    _, runtime = make_runtime(2)
    runtime.physics_step()
    runtime.episode_kind[1] = 2
    history = V.ImpactHistory(hold_s=0.3, max_hits=3)
    first = history.capture(runtime, torch.tensor([0, 1]), 0.0)
    second = history.capture(runtime, torch.tensor([0, 1]), 0.15)
    assert not first[0].stress and first[1].stress
    assert history.visible(0.2) == [first[1], *second]

    history.reset(torch.tensor([0]))
    assert history.visible(0.2) == [first[1], second[1]]
    # The first strike expires while the later strike in the same combination
    # remains visible. Expiration uses the caller's simulation timestamp.
    assert history.visible(0.31) == [second[1]]
    history.reset([1])
    assert history.visible(0.31) == []


@pytest.mark.parametrize("hold_s", [0.0, -0.1, float("nan"), float("inf")])
def test_invalid_hold_time_is_rejected(hold_s):
    with pytest.raises(ValueError):
        V.ImpactHistory(hold_s=hold_s)


@pytest.mark.parametrize("direction", [
    (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0), (1.0, -2.0, 3.0),
])
def test_arrow_tip_and_shaft_follow_applied_force_including_vertical_impacts(direction):
    event = hit(direction=direction)
    positions, quaternions, scales, indices = V.marker_arrays([event])
    assert positions.shape == (4, 3)
    assert quaternions.shape == (4, 4)
    assert scales.shape == (4, 3)
    np.testing.assert_array_equal(indices, [0, 1, 0, 2])
    assert np.isfinite(positions).all()
    assert np.isfinite(quaternions).all()
    assert np.isfinite(scales).all() and (scales > 0).all()
    np.testing.assert_allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-6)

    direction_w = np.asarray(event.direction_w)
    for quaternion in quaternions[:3]:
        np.testing.assert_allclose(rotate(quaternion, [0.0, 0.0, 1.0]), direction_w, atol=1e-6)
    cone_tip = positions[1] + rotate(quaternions[1], [0.0, 0.0, scales[1, 2] / 2])
    np.testing.assert_allclose(cone_tip, np.asarray(event.point_w) - direction_w * V.ARROW_STANDOFF_M, atol=1e-6)
    np.testing.assert_allclose(positions[3], event.point_w, atol=1e-6)
    guide_end = positions[2] + rotate(quaternions[2], [0.0, 0.0, scales[2, 2] / 2])
    guide_start = positions[2] - rotate(quaternions[2], [0.0, 0.0, scales[2, 2] / 2])
    np.testing.assert_allclose(guide_end, event.point_w, atol=1e-6)
    np.testing.assert_allclose(guide_start, cone_tip, atol=1e-6)
    shaft_end = positions[0] + rotate(quaternions[0], [0.0, 0.0, scales[0, 2] / 2])
    cone_base = positions[1] - rotate(quaternions[1], [0.0, 0.0, scales[1, 2] / 2])
    np.testing.assert_allclose(shaft_end, cone_base, atol=1e-6)


def test_marker_strength_and_pressure_color_are_distinguishable_and_empty_history_is_valid():
    weak = hit(peak=40)
    strong = hit(peak=1500, stress=True)
    positions, quaternions, scales, indices = V.marker_arrays([weak, strong])
    np.testing.assert_array_equal(indices, [0, 1, 0, 2, 3, 4, 3, 5])
    assert sum(scales[4:6, 2]) > sum(scales[:2, 2])
    assert np.isfinite(positions).all() and np.isfinite(quaternions).all()
    positions, quaternions, scales, indices = V.marker_arrays([])
    assert positions.shape == (0, 3)
    assert quaternions.shape == (0, 4)
    assert scales.shape == (0, 3)
    assert indices.shape == (0,)


def test_hit_observation_does_not_change_forces_torques_rng_or_checkpoint_contract():
    settings = dict(
        fixed_duration_s=0.012, recovery_pause_s=(0.04, 0.06),
        common_combo_probability=1.0, combo_gap_s=(0.03, 0.04),
    )
    env_a, plain = make_runtime(2, **settings)
    env_b, observed = make_runtime(2, **settings)
    history = V.ImpactHistory()
    clock = [0.0]
    captured = []

    def observer(ids):
        captured.extend(history.capture(observed, ids, clock[0]))

    assert plain.hit_observer is None
    observed.hit_observer = observer
    for step in range(100):
        clock[0] = step * env_a.physics_dt
        if step in (20, 70):
            plain.reset([0])
            observed.reset([0])
            history.reset([0])
        plain.physics_step()
        observed.physics_step()
        assert torch.equal(plain.forces_b, observed.forces_b)
        assert torch.equal(plain.torques_b, observed.torques_b)
        assert torch.equal(plain._generator.get_state(), observed._generator.get_state())
        assert torch.equal(
            env_a.scene["robot"].instantaneous_wrench_composer.forces,
            env_b.scene["robot"].instantaneous_wrench_composer.forces,
        )
    assert len(captured) > 4
    assert plain.config_signature == observed.config_signature
    assert plain.state_dict().keys() == observed.state_dict().keys()
    for key, value in plain.state_dict().items():
        other = observed.state_dict()[key]
        assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
