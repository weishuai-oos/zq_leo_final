"""CPU checks for first-fall censoring and sample-weighted benchmark aggregation."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "impact_evaluation", Path(__file__).resolve().parents[1] / "scripts/cusrl/evaluate_impact.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_fixed_commands_are_balanced_and_have_expected_modes():
    names, commands, modes = MODULE.trial_commands(32)
    assert len(names) == 32
    assert commands.shape == (32, 3)
    assert max(names.count(name) for name in names) - min(names.count(name) for name in names) <= 1
    assert modes[0] == 0
    assert commands[names.index("forward_1.0")].tolist() == [1.0, 0.0, 0.0]
    assert np.linalg.norm(commands[names.index("forward_left_0.6"), :2]) == pytest.approx(0.6)


def test_first_fall_keeps_terminal_frame_and_excludes_autoreset_frames():
    metrics = MODULE.TrialMetrics(2, 0.02)
    # First trial falls immediately; second trial survives with +1m/s error.
    velocity = metrics.commands.copy()
    velocity[:, 0] += [2.0, 1.0]
    metrics.add(velocity, [1.0, 0.1], [0.3, 0.9], [True, False], [
        {"hits": 1, "impulse_ns": 10, "sampled_peak_n": 2000}, {},
    ], [["base_contact"], []])
    velocity = metrics.commands.copy()
    velocity[:, 0] += [1000.0, 1.0]
    metrics.add(velocity, [3.0, 0.2], [50.0, 0.9], [False, False], [
        {"hits": 100}, {"hits": 2, "impulse_ns": 5, "sampled_peak_n": 1000},
    ])
    report = metrics.report()
    total = report["aggregate"]
    assert total["trials"] == 2
    assert total["falls"] == 1
    assert total["failure_rate"] == 0.5
    assert total["valid_samples"] == 3
    assert total["alive_seconds_total"] == pytest.approx(0.06)
    # Sample weighting: sqrt((2**2 + 1**2 + 1**2)/3), not mean of trial RMSEs.
    assert total["tracking_rmse_xy"] == pytest.approx(np.sqrt(2))
    assert total["tilt_max_rad"] == 1.0
    assert total["height_min_m"] == 0.3
    assert total["impacts"]["hits"] == 3
    assert total["impacts"]["impulse_total_ns"] == 15
    assert total["impacts"]["continuous_pulse_peak_max_n"] == 2000
    assert report["per_trial"][0]["valid_samples"] == 1
    assert report["per_trial"][0]["failure_terms"] == ["base_contact"]


def test_recovery_average_uses_completed_events_and_zero_count_is_null():
    metrics = MODULE.TrialMetrics(2, 0.02)
    metrics.add(metrics.commands, [0, 0], [1, 1], [False, False], [
        {"recoveries": 1, "recovery_time_s": 0.2, "recovery_expired": 1},
        {"recoveries": 3, "recovery_time_s": 2.4},
    ])
    report = metrics.report()
    assert report["aggregate"]["impacts"]["completed_recovery_mean_s"] == pytest.approx(0.65)
    assert report["aggregate"]["impacts"]["recovery_expired"] == 1
    empty = MODULE.TrialMetrics(1, 0.02)
    empty.add(empty.commands, [0], [1], [False], [{}])
    assert empty.report()["aggregate"]["impacts"]["completed_recovery_mean_s"] is None


def test_live_nonfinite_states_fail_fast_but_already_failed_trials_are_censored():
    metrics = MODULE.TrialMetrics(1, 0.02)
    with pytest.raises(ValueError, match="Nonfinite"):
        metrics.add([[float("nan"), 0, 0]], [0], [1], [False], [{}])
    metrics.add([[0, 0, 0]], [0], [1], [True], [{}])
    metrics.add([[float("nan"), 0, 0]], [float("nan")], [float("nan")], [False], [{}])
    assert metrics.report()["aggregate"]["valid_samples"] == 1


def test_recovery_protocol_requires_persistent_physical_knockdown_and_guard_hold():
    state = MODULE.RecoveryTrialState(0.1)
    # A short hand/foot support event is recorded but never becomes a fall.
    for _ in range(1):
        result = state.observe(tilt_rad=0.1, angular_speed_rad_s=0.2, command_error_mps=0.1,
                               root_height_m=0.9, nonfoot_support=True, guard_error_rad=1.0,
                               hand_knee_contact=True)
    assert not result["knocked_down"]
    assert state.contact_events == 1
    # A persistent low body state crosses the 0.30 s hysteresis threshold.
    for _ in range(3):
        result = state.observe(tilt_rad=1.0, angular_speed_rad_s=2.0, command_error_mps=1.0,
                               root_height_m=0.4, nonfoot_support=False)
    assert result["knocked_down"]
    assert not result["recovered"]


def test_recovery_requires_balance_then_guard_and_timeout_does_not_extend():
    state = MODULE.RecoveryTrialState(0.05)
    state.observe(tilt_rad=0.5, angular_speed_rad_s=2.0, command_error_mps=1.0,
                  root_height_m=0.8, impact=True)
    # Stable balance is not enough to claim guard return.
    for _ in range(8):
        state.observe(tilt_rad=0.1, angular_speed_rad_s=0.2, command_error_mps=0.1,
                      root_height_m=0.9, guard_error_rad=1.0, impact=False)
    assert state.recovery_time is not None
    assert not state.recovered
    while not state.timeout:
        state.observe(tilt_rad=0.5, angular_speed_rad_s=2.0, command_error_mps=1.0,
                      root_height_m=0.8, guard_error_rad=1.0, impact=True)
    assert not state.recovered
    assert state.recovery_time is not None


def test_impact_plan_is_deterministic_and_independent_of_policy_results():
    first, digest_a = MODULE.build_impact_plan(8, 20.0, 42, "stress")
    second, digest_b = MODULE.build_impact_plan(8, 20.0, 42, "stress")
    third, digest_c = MODULE.build_impact_plan(8, 20.0, 43, "stress")
    assert first == second
    assert digest_a == digest_b
    assert digest_a != digest_c
    assert len(first) >= 8 and len(third) >= 8
    assert {event.env_index for event in first} == set(range(8))


def test_recovery_metrics_are_separate_from_legacy_first_terminal_metric():
    metrics = MODULE.TrialMetrics(1, 0.1, protocol=MODULE.RECOVERY_PROTOCOL)
    metrics.add(metrics.commands, [0.0], [0.9], [False], [{}])
    metrics.add_recovery_observations([{
        "tilt_rad": 0.1, "angular_speed_rad_s": 0.2, "command_error_mps": 0.1,
        "root_height_m": 0.9, "guard_error_rad": 0.1, "impact": True,
    }])
    report = metrics.report()
    assert "recovery_v1" in report["aggregate"]
    assert report["aggregate"]["recovery_v1"]["knocked_down"] == 0


def test_recovery_guard_hold_cannot_succeed_during_a_relapse_and_horizon_is_censored():
    state = MODULE.RecoveryTrialState(.05)
    state.observe(tilt_rad=.5, angular_speed_rad_s=2., command_error_mps=1., root_height_m=.9, impact=True)
    for _ in range(6):
        state.observe(tilt_rad=.1, angular_speed_rad_s=.1, command_error_mps=.1, root_height_m=.9, guard_error_rad=1.)
    assert state.recovery_time is not None
    for _ in range(8):
        state.observe(tilt_rad=.5, angular_speed_rad_s=2., command_error_mps=1., root_height_m=.9, guard_error_rad=0.)
    report = state.report()
    assert not state.recovered
    assert report["events"][0]["status"] == "censored_horizon"
    assert report["events"][0]["guard_time_s"] is None


def test_contacts_are_edge_events_and_hand_contact_never_becomes_body_knockdown():
    state = MODULE.RecoveryTrialState(.005)
    for _ in range(100):
        state.observe(tilt_rad=.1, angular_speed_rad_s=.1, command_error_mps=.1,
                      root_height_m=.9, guard_error_rad=0., nonfoot_support=True,
                      hand_knee_contact=True, foot_side_contact=True)
    assert not state.knocked_down
    assert state.contact_events == 2  # one support onset plus one foot-side onset
    for _ in range(29):
        state.observe(tilt_rad=.1, angular_speed_rad_s=.1, command_error_mps=.1,
                      root_height_m=.9, body_ground_contact=True)
    assert not state.knocked_down
    state.observe(tilt_rad=.1, angular_speed_rad_s=.1, command_error_mps=.1,
                  root_height_m=.9, body_ground_contact=True)
    assert state.knocked_down


def test_multiple_hit_windows_and_completed_denominator_exclude_horizon_censoring():
    state = MODULE.RecoveryTrialState(.05)
    basic = dict(tilt_rad=.0, angular_speed_rad_s=.1, command_error_mps=.1, root_height_m=.9, guard_error_rad=.1)
    state.observe(**basic, impact=True)
    for _ in range(20):
        state.observe(**basic)
    assert state.events[0]["status"] == "resisted"
    state.observe(**basic, impact=True)
    assert len(state.report()["events"]) == 2
    assert state.report()["events"][-1]["status"] == "censored_horizon"


def test_plan_matches_calibrated_sites_caps_and_common_combinations():
    plan, _ = MODULE.build_impact_plan(48, 20., 42, "common")
    sites = {item[0]: item for item in MODULE.IMPACT_TARGETS}
    sequences = {}
    for event in plan:
        body, group, weight, extent, duration, common, stress = sites[event.target]
        assert event.group == group
        assert duration[0] <= event.duration_s <= duration[1]
        assert 2 * event.peak_n * event.duration_s / np.pi <= common + 1e-8
        assert np.linalg.norm(event.direction_initial) == pytest.approx(1.)
        assert np.all(np.abs(event.point_b) <= np.asarray(extent))
        sequences.setdefault((event.env_index, event.sequence), []).append(event)
    assert any(len(events) > 1 for events in sequences.values())
    assert len({event.target for event in plan}) > 8
    # Extending the number of trials cannot change an existing trial's stream.
    extended, _ = MODULE.build_impact_plan(50, 20., 42, "common")
    assert plan == tuple(event for event in extended if event.env_index < 48)


def _fake_runtime(num_envs=2):
    import torch
    from types import SimpleNamespace
    class Composer:
        def set_forces_and_torques(self, **kwargs):
            self.forces = kwargs["forces"].clone()
            self.torques = kwargs["torques"].clone()
    class Runtime:
        @property
        def enabled(self):
            return self.cfg.enabled
        def physics_step(self):
            assert not self.enabled
            self.forces_b.zero_()
            self.torques_b.zero_()
            self.active.zero_()
    runtime = Runtime()
    runtime.num_envs, runtime.device, runtime.dt = num_envs, torch.device("cpu"), .005
    runtime.cfg = SimpleNamespace(enabled=True, mode="stress")
    quat = torch.zeros(num_envs, 1, 4); quat[:, :, 0] = 1.
    runtime.robot = SimpleNamespace(body_names=["LINK_BASE"], data=SimpleNamespace(
        root_quat_w=quat[:, 0], body_link_quat_w=quat, root_pos_w=torch.zeros(num_envs, 3)),
        instantaneous_wrench_composer=Composer())
    runtime.forces_b, runtime.torques_b = torch.zeros(num_envs, 1, 3), torch.zeros(num_envs, 1, 3)
    runtime.active = torch.zeros(num_envs, dtype=torch.bool)
    runtime.body_id = torch.zeros(num_envs, dtype=torch.long)
    runtime.peak_n = torch.zeros(num_envs)
    runtime.duration, runtime.age = torch.ones(num_envs), torch.zeros(num_envs)
    runtime.force_direction_w, runtime.hit_point_b = torch.zeros(num_envs, 3), torch.zeros(num_envs, 3)
    runtime._since_hit = torch.ones(num_envs)
    runtime._stats = {key: torch.zeros(num_envs) for key in ("hits", "common_hits", "stress_hits", "sampled_peak_n", "impulse_ns", "peak_n")}
    runtime.DIRECTIONS, runtime.GROUPS = ("front", "side", "rear"), ("torso", "pelvis", "arm", "thigh", "calf", "head")
    runtime._direction_hits, runtime._group_hits = torch.zeros(num_envs, 3), torch.zeros(num_envs, 6)
    runtime.hit_observer = None
    return runtime


def test_actual_plan_executor_preserves_readonly_enabled_and_exact_pulse_impulse_and_torque():
    import torch
    runtime = _fake_runtime()
    events = (MODULE.ImpactEvent(0, .002, 100., .025, "LINK_BASE", "front", group="pelvis",
                                 point_b=(0., .1, 0.), direction_initial=(-1., 0., 0.)),)
    executor = MODULE.PlannedImpactExecutor(runtime, events, lambda: np.ones(2, dtype=bool))
    torque_peak = 0.
    for _ in range(20):
        executor()
        torque_peak = max(torque_peak, float(runtime.torques_b.norm()))
    assert runtime.enabled is True
    expected = 2 * 100 * .025 / np.pi
    assert executor.report()[0]["delivered_impulse_ns"] == pytest.approx(expected, abs=1e-5)
    assert runtime._stats["hits"].tolist() == [1., 0.]
    assert runtime._stats["stress_hits"].tolist() == [1., 0.]
    assert runtime._direction_hits[0].tolist() == [1., 0., 0.]
    assert runtime._group_hits[0, 1] == 1
    assert torque_peak > 0.
    assert torch.count_nonzero(runtime.forces_b) == 0


def test_plan_applied_to_other_env_is_unchanged_when_one_trial_dies_and_later_hits_are_skipped():
    events = tuple(MODULE.ImpactEvent(env, onset, 100., .025, "LINK_BASE", "front", group="pelvis")
                   for onset in (.01, .1) for env in range(2))
    r1, r2 = _fake_runtime(), _fake_runtime()
    live = np.ones(2, dtype=bool)
    first = MODULE.PlannedImpactExecutor(r1, events, lambda: live)
    second = MODULE.PlannedImpactExecutor(r2, events, lambda: np.ones(2, dtype=bool))
    for i in range(40):
        if i == 15:
            live[0] = False
        first(); second()
    report1, report2 = first.report(), second.report()
    for i, event in enumerate(events):
        if event.env_index == 1:
            assert report1[i]["delivered_impulse_ns"] == report2[i]["delivered_impulse_ns"]
    assert report1[2]["skipped_after_trial_end"]
    assert not report2[2]["skipped_after_trial_end"]


def test_neutral_manager_uses_public_api_even_with_list_term_configs():
    import torch
    from types import SimpleNamespace
    class Manager:
        def __init__(self):
            self._term_cfgs = [SimpleNamespace(func=lambda env: "old", params={"threshold": 1.})]
        def get_term_cfg(self, name):
            return self._term_cfgs[0]
        def set_term_cfg(self, name, cfg):
            self._term_cfgs[0] = cfg
    env = SimpleNamespace(num_envs=2, step_dt=.02, physics_dt=.005, device="cpu", termination_manager=Manager())
    hook = MODULE.EvaluationHook(env, protocol=MODULE.RECOVERY_PROTOCOL)
    old = env.termination_manager.get_term_cfg("base_contact")
    hook._install_neutral_termination()
    cfg = env.termination_manager.get_term_cfg("base_contact")
    assert cfg is not old
    assert cfg.func(env).tolist() == [False, False]
    hook.metrics.recovery[1].knocked_down = True
    assert cfg.func(env).tolist() == [False, True]
    assert old.params == {"threshold": 1.}


def test_guard_and_filtered_body_support_measurements_are_finite_and_distinct():
    import torch
    from types import SimpleNamespace
    class Scene(dict):
        pass
    joint_names = ["arm_a", "arm_b", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R"]
    class Robot:
        def __init__(self):
            self.data = SimpleNamespace(root_ang_vel_b=torch.zeros(2, 3), joint_pos=torch.full((2, 6), .5))
        def find_joints(self, names, preserve_order):
            return [joint_names.index(name) for name in names], names
    scene = Scene(robot=Robot())
    def sensor(active):
        force = torch.zeros(2, 1, 1, 3); force[active, 0, 0, 2] = 10.
        return SimpleNamespace(data=SimpleNamespace(force_matrix_w=force))
    scene.sensors = {"impact_ground_LINK_BASE": sensor([0]), "impact_ground_LINK_WRIST_END_L": sensor([1])}
    params = {"term_params": {"target_joint_pos": (.5, .5)}, "asset_cfg": SimpleNamespace(joint_names=["arm_a", "arm_b"])}
    env = SimpleNamespace(num_envs=2, step_dt=.02, physics_dt=.005, device="cpu", scene=scene,
                          cfg=SimpleNamespace(rewards=SimpleNamespace(standard_arm_pose_l1=SimpleNamespace(params=params))))
    hook = MODULE.EvaluationHook(env, protocol=MODULE.RECOVERY_PROTOCOL)
    hook._prepare_guard_reference()
    hook._frame = lambda: np.asarray([[0., 0., 0., 0., .9], [0., 0., 0., 0., .9]])
    observations = hook._physical_observations()
    assert observations[0]["guard_error_rad"] == 0.
    assert observations[0]["body_ground_contact"] is True
    assert observations[1]["body_ground_contact"] is False
    assert observations[1]["hand_knee_contact"] is True


def test_hook_samples_physics_persistence_and_keeps_pre_autoreset_terminal_frame():
    import torch
    from types import SimpleNamespace
    runtime = _fake_runtime(1)
    runtime.robot.data.joint_vel = torch.zeros(1, 4)
    class Scene(dict):
        env_origins = torch.zeros(1, 3)
        def update(self, dt):
            pass
    class Manager:
        active_terms = ["base_contact"]
        def __init__(self):
            self._term_cfgs = [SimpleNamespace(func=lambda env: torch.zeros(1, dtype=torch.bool), params={})]
        def get_term_cfg(self, name):
            return self._term_cfgs[0]
        def set_term_cfg(self, name, cfg):
            self._term_cfgs[0] = cfg
        def get_term(self, name):
            return self.done
    class Term:
        def __init__(self):
            self.vel_command_b = torch.zeros(1, 3)
            self.time_left = torch.zeros(1)
            self.is_heading_env = self.is_standing_env = torch.zeros(1, dtype=torch.bool)
            self.heading_target = torch.zeros(1)
            self.command_mode = torch.zeros(1, dtype=torch.long)
        def transform_teleop_command(self, value):
            return value
        def _resample_command(self, ids):
            pass
        def _update_command(self):
            pass
    class Env:
        def __init__(self):
            self.num_envs, self.step_dt, self.physics_dt, self.device = 1, .02, .005, "cpu"
            self.impact_runtime = runtime
            self.scene = Scene(robot=runtime.robot)
            self.termination_manager = Manager()
            term = Term()
            self.command_manager = SimpleNamespace(get_term=lambda name: term)
            self._sim_step_counter = 0
            self.height = .2
        def _reset_idx(self, ids):
            self.height = 100.  # must never enter terminal physical metrics
        def step(self, action):
            self.height = .2
            for _ in range(4):
                self._sim_step_counter += 1
                self.impact_runtime.physics_step()
                self.scene.update(self.physics_dt)
            self.reset_terminated = self.termination_manager.get_term_cfg("base_contact").func(self)
            self.termination_manager.done = self.reset_terminated
            ids = self.reset_terminated.nonzero().flatten()
            if ids.numel():
                self._reset_idx(ids)
    env = Env()
    hook = MODULE.EvaluationHook(env, protocol=MODULE.RECOVERY_PROTOCOL)
    def prepare():
        hook._ankle_ids, hook._foot_ids = [0, 1, 2, 3], None
    hook._prepare_guard_reference = prepare
    hook._physical_observations = lambda: [dict(tilt_rad=.0, angular_speed_rad_s=.0,
        command_error_mps=.0, root_height_m=env.height, body_ground_contact=True,
        nonfoot_support=True, guard_error_rad=.0)]
    hook._snapshot = lambda: (np.asarray([[0., 0., 0., 0., env.height]]), [{}])
    hook.initial_state_hash = "fixture"
    original_step, original_scene_update = env.step, env.scene.update
    original_term_cfg = env.termination_manager.get_term_cfg("base_contact")
    hook.init(SimpleNamespace())
    for index in range(12):
        env.step(None)
        hook.step(index, None, None)
    result = hook.metrics.report()
    assert result["aggregate"]["falls"] == 1
    assert result["aggregate"]["valid_samples"] == 8  # .15s detected at 5ms, checked at .16s
    assert result["aggregate"]["height_mean_m"] == pytest.approx(.2)
    assert hook.metrics.recovery[0].t == pytest.approx(.15)
    hook.close()
    assert env.step == original_step
    assert env.scene.update == original_scene_update
    assert env.termination_manager.get_term_cfg("base_contact") is original_term_cfg
