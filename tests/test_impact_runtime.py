from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
IMPACT_DIR = ROOT / "source/t800_lab/t800_lab/tasks/t800_impact"


def _load_modules():
    package_name = "impact_runtime_cpu"
    package = ModuleType(package_name)
    package.__path__ = [str(IMPACT_DIR)]
    sys.modules[package_name] = package
    modules = []
    for stem in ("impact_config", "impact_runtime"):
        spec = importlib.util.spec_from_file_location(f"{package_name}.{stem}", IMPACT_DIR / f"{stem}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


C, M = _load_modules()


class FakeComposer:
    def __init__(self, n, bodies):
        self.forces = torch.zeros(n, bodies, 3)
        self.torques = torch.zeros_like(self.forces)
        self.calls = 0

    def set_forces_and_torques(self, *, forces, torques, is_global):
        assert not is_global
        self.forces.copy_(forces)
        self.torques.copy_(torques)
        self.calls += 1

    def reset(self, env_ids=None):
        if env_ids is None:
            self.forces.zero_()
            self.torques.zero_()
        else:
            self.forces[env_ids] = 0
            self.torques[env_ids] = 0


def _env(n=2, names=("LINK_TORSO_YAW", "LINK_HEAD_YAW", "LINK_BASE")):
    root_quat = torch.zeros(n, 4)
    root_quat[:, 0] = 1
    body_quat = root_quat[:, None, :].repeat(1, len(names), 1)
    data = SimpleNamespace(
        root_quat_w=root_quat,
        body_link_quat_w=body_quat,
        root_pos_w=torch.zeros(n, 3),
        root_ang_vel_b=torch.zeros(n, 3),
        root_lin_vel_b=torch.zeros(n, 3),
        root_lin_vel_w=torch.zeros(n, 3),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(n, 1),
    )
    commands = torch.zeros(n, 3)
    return SimpleNamespace(
        device="cpu",
        num_envs=n,
        physics_dt=0.005,
        cfg=SimpleNamespace(seed=17, sim=SimpleNamespace(dt=0.005)),
        scene={
            "robot": SimpleNamespace(
                body_names=names, data=data, instantaneous_wrench_composer=FakeComposer(n, len(names))
            )
        },
        command_manager=SimpleNamespace(get_command=lambda _: commands),
        reset_terminated=torch.zeros(n, dtype=torch.bool),
    )


def _cfg(**kwargs):
    defaults = dict(
        mode="common",
        tactical_front_offset_deg=0,
        first_hit_delay_s=(0, 0),
        recovery_pause_s=(100, 100),
        common_combo_probability=0,
        stress_combo_probability=0,
        direction_elevation_deg=(0, 0),
        fixed_peak_n=1000,
        fixed_duration_s=0.023,
        targets=(C.ImpactTargetCfg("LINK_TORSO_YAW", "torso", 1, (0.1, 0.1, 0.1), (0.005, 0.080), 100, 170),),
    )
    defaults.update(kwargs)
    return C.ImpactCfg(**defaults)


@pytest.mark.parametrize("duration", (0.003, 0.023, 0.080))
def test_half_sine_analytically_preserves_impulse_for_nonintegral_step_width(duration):
    peak = torch.tensor([1378.0], dtype=torch.float64)
    width = torch.tensor([duration], dtype=torch.float64)
    impulse = torch.zeros(1, dtype=torch.float64)
    for step in range(math.ceil(duration / 0.005) + 2):
        contribution = M.half_sine_step_impulse(peak, width, torch.tensor([step * 0.005]), 0.005)
        assert 0 <= contribution.item() <= peak.item() * 0.005 + 1e-10
        impulse += contribution
    assert math.isclose(impulse.item(), 2 * peak.item() * duration / math.pi, rel_tol=2e-7)


def test_force_stays_fixed_in_world_and_torque_uses_current_link_quaternion():
    env = _env(1)
    runtime = M.ImpactRuntime(env, _cfg())
    runtime.physics_step()
    world_direction = runtime.force_direction_w.clone()
    runtime.hit_point_b[0] = torch.tensor([0.1, 0.2, 0.3])
    # Rotating either the root or the hit body must not rotate the incoming
    # impulse in world coordinates after a pulse has begun.
    q = torch.tensor([math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
    env.scene["robot"].data.root_quat_w[0] = q
    env.scene["robot"].data.body_link_quat_w[0, 0] = q
    runtime.physics_step()
    local = runtime.forces_b[:, 0]
    force_w = M._quat_rotate(q[None], local)
    assert torch.allclose(force_w / force_w.norm(dim=-1, keepdim=True), world_direction, atol=1e-6)
    assert torch.allclose(runtime.torques_b[0, 0], torch.cross(runtime.hit_point_b[0], local[0], dim=-1), atol=1e-5)
    assert env.scene["robot"].instantaneous_wrench_composer.calls == 2


def test_pulse_total_applied_impulse_and_explicit_clear_on_following_step():
    env = _env(1)
    runtime = M.ImpactRuntime(env, _cfg())
    actual_impulse = 0
    for _ in range(5):
        runtime.physics_step()
        actual_impulse += runtime.forces_b[0, 0].norm().item() * env.physics_dt
    assert math.isclose(actual_impulse, 2 * 1000 * 0.023 / math.pi, rel_tol=1e-6)
    assert not runtime.active[0]
    runtime.physics_step()
    assert not runtime.forces_b.any() and not runtime.torques_b.any()
    logs = runtime.log_reset([0])
    assert math.isclose(logs["Impact/impulse_ns"], actual_impulse, rel_tol=1e-6)
    assert logs["Impact/hits"] == 1
    assert 0 < logs["Impact/peak_n"] <= 1000


def test_reset_subset_does_not_clear_other_environment_or_global_curriculum():
    env = _env(2)
    runtime = M.ImpactRuntime(env, _cfg())
    runtime.set_update(3200)
    runtime.physics_step()
    other_force = runtime.forces_b[1].clone()
    runtime.last_step_style_scale[0] = 0.35
    runtime.reset([0])
    assert runtime.completed_updates == 3200
    assert not runtime.active[0] and runtime.active[1]
    assert not runtime.forces_b[0].any()
    assert torch.equal(runtime.forces_b[1], other_force)
    assert torch.equal(env.scene["robot"].instantaneous_wrench_composer.forces[1], other_force)
    assert runtime.last_step_style_scale[0] == pytest.approx(0.35)  # preceding AMP transition survives reset
    assert runtime.log_reset([0])["Impact/hits"] == 0


def test_warmup_no_hits_and_clean_episode_never_receives_a_force():
    runtime = M.ImpactRuntime(_env(32), _cfg(mode="train", warmup_updates=5, ramp_updates=10))
    for _ in range(5):
        runtime.physics_step()
    assert not runtime._stats["hits"].any()
    runtime.set_update(15)
    runtime.reset(None)
    clean = runtime.episode_kind == 0
    assert clean.any()
    for _ in range(10):
        runtime.physics_step()
    assert not runtime._stats["hits"][clean].any()
    assert not runtime.forces_b[clean].any()
    assert runtime._stats["hits"][~clean].sum() > 0


def test_zero_ramp_starts_full_curriculum_after_warmup_and_failure_rates_use_group_denominators():
    env = _env(3)
    runtime = M.ImpactRuntime(env, _cfg(mode="train", warmup_updates=2, ramp_updates=0))
    assert runtime.curriculum_progress == 0
    runtime.set_update(2)
    assert runtime.curriculum_progress == 1
    runtime.episode_kind[:] = torch.tensor([0, 1, 2])
    runtime.physics_step()
    env.reset_terminated[:] = torch.tensor([False, True, False])
    logs = runtime.log_reset(None)
    assert logs["Impact/hit_episode_failure_fraction"] == 0.5
    assert logs["Impact/common_failure_fraction"] == 1
    assert logs["Impact/stress_failure_fraction"] == 0


def test_episode_and_direction_mixture_and_stress_progress():
    env = _env(12000)
    cfg = _cfg(mode="train", warmup_updates=2, ramp_updates=10)
    runtime = M.ImpactRuntime(env, cfg)
    assert not (runtime.episode_kind == 2).any()
    runtime.set_update(12)
    runtime.reset(None)
    fractions = torch.bincount(runtime.episode_kind, minlength=3) / env.num_envs
    assert torch.allclose(fractions, torch.tensor([0.25, 0.65, 0.10]), atol=0.015)
    runtime.physics_step()
    selected = runtime.episode_kind != 0
    directions = torch.bincount(runtime.direction_sector[selected], minlength=3) / selected.sum()
    assert torch.allclose(directions, torch.tensor([0.60, 0.35, 0.05]), atol=0.015)
    bearing = torch.atan2(-runtime.force_direction_w[selected, 1], -runtime.force_direction_w[selected, 0]).abs()
    sector = runtime.direction_sector[selected]
    assert (bearing[sector == 0] <= math.pi / 3 + 1e-6).all()
    assert (bearing[sector == 1] >= math.pi / 3 - 1e-6).all()
    assert (bearing[sector == 1] <= math.radians(140) + 1e-6).all()
    assert (bearing[sector == 2] >= math.radians(140) - 1e-6).all()


def test_body_impulse_and_peak_caps_survive_fixed_extreme_evaluation_setting():
    target = C.ImpactTargetCfg("LINK_HEAD_YAW", "head", 1, (0.07, 0.07, 0.08), (0.01, 0.03), 20, 40)
    runtime = M.ImpactRuntime(
        _env(64), _cfg(mode="stress", fixed_peak_n=5000, fixed_duration_s=0.03, targets=(target,))
    )
    runtime.physics_step()
    assert (runtime.peak_n <= 5000).all()
    assert (2 * runtime.peak_n * runtime.duration / math.pi <= 40.00001).all()
    runtime = M.ImpactRuntime(_env(64), _cfg(fixed_peak_n=5000, targets=(target,)))
    runtime.physics_step()
    assert (runtime.peak_n <= 2000).all()
    assert (2 * runtime.peak_n * runtime.duration / math.pi <= 20.00001).all()


def test_weak_hit_never_forces_step_or_loosens_style_without_instability():
    runtime = M.ImpactRuntime(_env(1), _cfg(fixed_peak_n=40))
    for _ in range(50):
        runtime.physics_step()
        assert torch.equal(runtime.recovery_scale(), torch.ones(1))
    assert runtime._stats["hits"][0] == 1


def test_tilted_body_velocity_is_compared_with_walking_command_in_world_yaw_frame():
    env = _env(1)
    data = env.scene["robot"].data
    angle = math.pi / 3
    data.root_quat_w[0] = torch.tensor([math.cos(angle / 2), 0, math.sin(angle / 2), 0])
    data.projected_gravity_b[0] = torch.tensor([math.sin(angle), 0, -math.cos(angle)])
    env.command_manager.get_command("base_velocity")[0, 0] = 1
    data.root_lin_vel_w[0, 0] = 1
    data.root_lin_vel_b[0] = torch.tensor([math.cos(angle), 0, math.sin(angle)])
    runtime = M.ImpactRuntime(
        env,
        _cfg(fixed_peak_n=40, recovery_tilt_threshold_rad=1.5, recovery_velocity_error_threshold=0.2),
    )
    for _ in range(10):
        runtime.physics_step()
        assert runtime.recovery_scale()[0] == 1


def test_instability_gate_has_deadline_even_under_repeated_hits_and_rearms_only_after_stability():
    env = _env(1)
    runtime = M.ImpactRuntime(env, _cfg(fixed_peak_n=40, recovery_max_s=0.10))
    runtime.physics_step()
    env.scene["robot"].data.root_ang_vel_b[0, 0] = 3
    for i in range(30):
        runtime._since_hit[0] = 0  # emulate arbitrary continued hits
        runtime.physics_step()
        if i < 15:
            assert runtime.recovery_scale()[0] == pytest.approx(0.35)
    assert runtime.recovery_scale()[0] == 1
    assert runtime._stats["recovery_expired"][0] == 1
    env.scene["robot"].data.root_ang_vel_b.zero_()
    for _ in range(90):
        runtime.physics_step()
    runtime._since_hit[0] = 0
    env.scene["robot"].data.root_ang_vel_b[0, 0] = 3
    runtime.physics_step()
    assert runtime.recovery_scale()[0] < 1


def test_recovery_success_and_command_relative_extra_displacement_metrics():
    env = _env(1)
    command = env.command_manager.get_command("base_velocity")
    command[:, 0] = 1
    env.scene["robot"].data.root_lin_vel_b[:, 0] = 1
    env.scene["robot"].data.root_lin_vel_w[:, 0] = 1
    runtime = M.ImpactRuntime(env, _cfg())
    runtime.physics_step()
    for _ in range(30):
        env.scene["robot"].data.root_pos_w[:, 0] += env.physics_dt
        runtime.physics_step()
    assert runtime.log_reset([0])["Impact/extra_displacement_m"] < 1e-6
    env.scene["robot"].data.root_pos_w[:, 1] += 0.2
    runtime.physics_step()
    assert runtime.log_reset([0])["Impact/extra_displacement_m"] == pytest.approx(0.2, abs=1e-3)
    runtime._since_hit[0] = 0
    env.scene["robot"].data.root_ang_vel_b[0, 0] = 3
    runtime.physics_step()
    env.scene["robot"].data.root_ang_vel_b.zero_()
    for _ in range(45):
        runtime.physics_step()
    logs = runtime.log_reset([0])
    assert logs["Impact/recoveries"] == 1
    assert 0.19 <= logs["Impact/mean_recovery_time_s"] <= 0.25


def test_short_combinations_obey_gap_then_take_recovery_pause():
    cfg = _cfg(common_combo_probability=1, common_combo_max_hits=3, combo_gap_s=(0.15, 0.15), recovery_pause_s=(2, 2))
    runtime = M.ImpactRuntime(_env(1), cfg)
    onsets = []
    previous_hits = 0
    for step in range(120):
        runtime.physics_step()
        hits = runtime._stats["hits"][0].item()
        if hits > previous_hits:
            onsets.append(step * runtime.dt)
            previous_hits = hits
    assert 2 <= len(onsets) <= 3
    assert all(0.149 <= b - a <= 0.161 for a, b in zip(onsets, onsets[1:]))
    assert runtime.cooldown[0] > 1


@pytest.mark.parametrize("mode", ["stress", "train"])
def test_stress_training_and_play_use_isolated_hits_then_full_recovery_pause(mode):
    runtime = M.ImpactRuntime(_env(1), _cfg(
        mode=mode, warmup_updates=0, ramp_updates=0,
        clean_episode_probability=0, stress_episode_probability=1,
        stress_combo_probability=0, stress_combo_max_hits=1,
        combo_gap_s=(0.03, 0.03), recovery_pause_s=(0.15, 0.15),
    ))
    onsets = []
    previous_hits = 0
    for step in range(240):
        runtime.physics_step()
        assert not runtime.combo_remaining.any()
        hits = runtime._stats["hits"][0].item()
        if hits > previous_hits:
            onsets.append(step * runtime.dt)
            previous_hits = hits
    assert len(onsets) >= 5
    # Evaluation uses 23 ms pulses; training samples up to 80 ms. Neither may
    # schedule another onset at the much shorter 30 ms combination interval.
    assert all(b - a >= 0.15 for a, b in zip(onsets, onsets[1:]))


@pytest.mark.parametrize("mode,maximum", [("common", 3), ("train", 5)])
def test_common_play_and_stress_training_keep_their_combo_distribution(mode, maximum):
    runtime = M.ImpactRuntime(_env(64), _cfg(
        mode=mode, warmup_updates=0, ramp_updates=0,
        clean_episode_probability=0, stress_episode_probability=1,
        common_combo_probability=1, common_combo_max_hits=3,
        stress_combo_probability=1, stress_combo_max_hits=5,
    ))
    runtime.physics_step()
    assert (runtime.combo_remaining >= 1).all()
    assert (runtime.combo_remaining < maximum).all()
    assert (runtime.combo_remaining == maximum - 1).any()
    expected_kind = 1 if mode == "common" else 2
    assert (runtime.episode_kind == expected_kind).all()


def test_single_hit_keeps_rng_draw_count_when_combination_maximum_is_one():
    settings = dict(
        mode="stress", stress_combo_probability=0,
        recovery_pause_s=(0.15, 0.20), combo_gap_s=(0.03, 0.05),
    )
    first = M.ImpactRuntime(_env(4), _cfg(stress_combo_max_hits=1, **settings))
    second = M.ImpactRuntime(_env(4), _cfg(stress_combo_max_hits=5, **settings))
    for _ in range(100):
        first.physics_step()
        second.physics_step()
        assert torch.equal(first.forces_b, second.forces_b)
        assert torch.equal(first.torques_b, second.torques_b)
        assert torch.equal(first._generator.get_state(), second._generator.get_state())


def test_stress_play_recognizes_only_the_exact_legacy_combo_signature_and_keeps_state():
    training = M.ImpactRuntime(_env(1), C.ImpactCfg(
        mode="train", stress_combo_probability=0.70, stress_combo_max_hits=5,
    ))
    # Signature of the released training defaults before isolated stress hits.
    assert training.config_signature == "41fcd09cf34822efc5c2de2f232d1e663e488777d2f7649ae6c77b9a13020040"
    training.set_update(29500)
    state = training.state_dict()
    evaluation = M.ImpactRuntime(_env(1), C.ImpactCfg(mode="stress"))
    evaluation.load_state_dict(state)
    assert evaluation.config_signature != training.config_signature
    assert evaluation.legacy_stress_combo_signature == training.config_signature
    assert evaluation.completed_updates == 29500
    assert torch.equal(evaluation._generator.get_state(), training._generator.get_state())
    assert evaluation.state_dict()["config_signature"] == evaluation.config_signature
    changed_other_settings = M.ImpactRuntime(_env(1), C.ImpactCfg(mode="stress", ramp_updates=6000))
    with pytest.raises(ValueError, match="configuration mismatch"):
        changed_other_settings.load_state_dict(state)


def test_training_legacy_distribution_requires_explicit_migration_and_new_resume_needs_no_flag():
    old = M.ImpactRuntime(_env(2), C.ImpactCfg(
        mode="train", stress_combo_probability=0.70, stress_combo_max_hits=5,
    ))
    old.set_update(29500)
    state = old.state_dict()
    current = M.ImpactRuntime(_env(2), C.ImpactCfg(mode="train"))
    with pytest.raises(ValueError, match="impact-single-hit-migration"):
        current.load_state_dict(state)
    assert current.completed_updates == 0
    current.allow_legacy_stress_combo_migration = True
    current.load_state_dict(state)
    assert current.completed_updates == 29500
    assert torch.equal(current._generator.get_state(), old._generator.get_state())
    restored = M.ImpactRuntime(_env(2), C.ImpactCfg(mode="train"))
    restored.load_state_dict(current.state_dict())
    assert restored.completed_updates == 29500
    assert torch.equal(restored._generator.get_state(), current._generator.get_state())


def test_training_resume_restores_rng_and_curriculum_and_rejects_incompatible_settings():
    cfg = _cfg(mode="train")
    runtime = M.ImpactRuntime(_env(10), cfg)
    runtime.set_update(2300)
    state = runtime.state_dict()
    runtime.reset(None)
    expected_kind = runtime.episode_kind.clone()
    expected_delay = runtime.cooldown.clone()
    other = M.ImpactRuntime(_env(10), cfg)
    other.load_state_dict(state)
    assert other.completed_updates == 2300
    other.reset(None)
    assert torch.equal(other.episode_kind, expected_kind)
    assert torch.equal(other.cooldown, expected_delay)
    other.reset([1])
    assert other.completed_updates == 2300
    incompatible = M.ImpactRuntime(_env(10), replace(cfg, ramp_updates=6000))
    with pytest.raises(ValueError, match="configuration mismatch"):
        incompatible.load_state_dict(state)
    with pytest.raises(ValueError, match="version mismatch"):
        other.load_state_dict({**state, "version": 99})


def test_rank_zero_checkpoint_derives_reproducible_independent_other_rank_stream(monkeypatch):
    cfg = _cfg(mode="train", first_hit_delay_s=(1, 3))
    monkeypatch.setenv("RANK", "0")
    rank_zero = M.ImpactRuntime(_env(64), cfg)
    rank_zero.set_update(2400)
    state = rank_zero.state_dict()
    rank_zero.load_state_dict(state)
    rank_zero.reset(None)
    zero_delays = rank_zero.cooldown.clone()
    monkeypatch.setenv("RANK", "1")
    rank_one = M.ImpactRuntime(_env(64), cfg)
    rank_one.load_state_dict(state)
    rank_one.reset(None)
    one_delays = rank_one.cooldown.clone()
    assert rank_one.completed_updates == rank_zero.completed_updates == 2400
    assert not torch.equal(zero_delays, one_delays)
    second_rank_one = M.ImpactRuntime(_env(64), cfg)
    second_rank_one.load_state_dict(state)
    second_rank_one.reset(None)
    assert torch.equal(one_delays, second_rank_one.cooldown)
    saved_rank_one = rank_one.state_dict()
    rank_one.reset(None)
    expected = rank_one.cooldown.clone()
    second_rank_one.load_state_dict(saved_rank_one)
    second_rank_one.reset(None)
    assert torch.equal(expected, second_rank_one.cooldown)


def test_cross_device_evaluation_restores_progress_but_training_rejects_rng_change():
    evaluation = M.ImpactRuntime(_env(), _cfg())
    evaluation.set_update(1800)
    saved = {**evaluation.state_dict(), "generator_device": "cuda"}
    evaluation.load_state_dict(saved)
    assert evaluation.completed_updates == 1800
    clean_export = M.ImpactRuntime(_env(), _cfg(mode="off"))
    clean_export.load_state_dict(saved)
    assert clean_export.completed_updates == 1800
    training = M.ImpactRuntime(_env(), _cfg(mode="train"))
    with pytest.raises(ValueError, match="device differs"):
        training.load_state_dict(saved)


def test_headless_asset_excludes_head_and_disabled_runtime_is_quiet():
    env = _env(2, names=("LINK_TORSO_YAW", "LINK_BASE"))
    runtime = M.ImpactRuntime(env, C.ImpactCfg(mode="off"))
    assert all(t.group != "head" for t in runtime.targets)
    for _ in range(20):
        runtime.physics_step()
    assert not runtime.forces_b.any() and not runtime._stats["hits"].any()
    assert runtime.log_reset([]) == {}


@pytest.mark.parametrize(
    "setting",
    (
        dict(mode="bad"),
        dict(warmup_updates=-1),
        dict(direction_probabilities=(1, 1, 1)),
        dict(fixed_peak_n=5001),
        dict(combo_gap_s=(0, 0.5)),
    ),
)
def test_bad_config_fails_explicitly(setting):
    with pytest.raises(ValueError):
        M.ImpactRuntime(_env(), _cfg(**setting))
