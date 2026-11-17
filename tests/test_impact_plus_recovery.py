"""CPU contracts for bounded balance/guard recovery and reward interpolation."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "source/t800_lab/t800_lab/tasks/t800_impact_v2_plus"
package = ModuleType("_plus_core_test")
package.__path__ = [str(PACKAGE)]
sys.modules[package.__name__] = package


def load(stem):
    name = package.__name__ + "." + stem
    spec = importlib.util.spec_from_file_location(name, PACKAGE / (stem + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R, MDP, IMP = load("recovery"), load("mdp"), load("impact")
RecoveryConfig, RecoveryController = R.RecoveryConfig, R.RecoveryController


def observe(controller, *, tilt=0.0, angular=0.0, velocity=0.0, height=0.8, guard=0.0, **kwargs):
    controller.step(tilt, angular, velocity, height=height, guard_error=guard, **kwargs)


def run(controller, seconds, **kwargs):
    for _ in range(round(seconds / controller.cfg.dt)):
        observe(controller, **kwargs)


def test_normal_state_has_full_weights_and_velocity_commands_alone_do_not_trigger():
    c = RecoveryController(2, "cpu")
    run(c, 3.0, velocity=1.2)
    assert not c.active.any() and not c.timeout.any()
    assert torch.equal(c.guard_blend, torch.ones(2))
    assert torch.equal(c.scale, torch.ones(2))


def test_transient_support_balances_then_completes_guard_and_does_not_later_timeout():
    c = RecoveryController(1, "cpu")
    observe(c, contact=True)
    assert c.recovering.item() and c.scale.item() == pytest.approx(0.1)
    run(c, 0.25)
    assert c.guard_returning.item() and not c.recovering.item()
    assert 0.0 < c.guard_blend.item() < 1.0
    run(c, 0.5)
    assert not c.active.item() and not c.armed.item()
    assert c.guard_blend.item() == 1.0
    run(c, 4.0)
    assert not c.timeout.item() and not c.knockdown.item() and c.armed.item()


def test_missing_guard_or_tracking_prevents_false_recovery():
    c = RecoveryController(1, "cpu")
    observe(c, contact=True)
    run(c, 1.0, velocity=0.6)
    assert c.recovering.item()
    run(c, 1.0, guard=0.5)
    assert c.guard_returning.item() and c.active.item()
    run(c, 0.6, guard=0.5)
    assert c.timeout.item() and not c.knockdown.item()


def test_repeated_hits_and_guard_relapse_do_not_extend_original_deadline():
    c = RecoveryController(1, "cpu")
    observe(c, contact=True)
    run(c, 0.35)
    assert c.guard_returning.item()
    original_age = c.age.item()
    run(c, 0.25, contact=True, recent_hit=True)
    assert c.recovering.item() and c.age.item() > original_age
    run(c, 2.0, contact=True, recent_hit=True)
    assert c.timeout.item()
    assert c.age.item() == pytest.approx(2.5, abs=0.006)
    run(c, 1.0)
    observe(c, contact=True)
    assert c.timeout.item() and not c.active.item() and not c.armed.item()


def test_low_body_height_is_persistent_knockdown_and_cannot_count_as_balanced():
    c = RecoveryController(1, "cpu")
    run(c, 0.295, height=0.4)
    assert c.active.item() and not c.knockdown.item()
    observe(c, height=0.4)
    assert c.knockdown.item() and not c.timeout.item()
    run(c, 1.0)
    assert c.knockdown.item() and not c.active.item()


def test_body_contact_and_severe_tilt_use_independent_persistence_windows():
    c = RecoveryController(1, "cpu")
    run(c, 0.1, persistent_body_contact=True)
    assert not c.knockdown.item()
    observe(c)
    run(c, 0.145, persistent_body_contact=True)
    assert not c.knockdown.item()
    observe(c, persistent_body_contact=True)
    assert c.knockdown.item()
    c.reset()
    run(c, 0.295, tilt=math.radians(85))
    assert not c.knockdown.item()
    observe(c, tilt=math.radians(85))
    assert c.knockdown.item()


def test_reset_subset_clears_every_failure_and_restores_normal_scale():
    c = RecoveryController(2, "cpu", RecoveryConfig(max_s=0.2))
    run(c, 0.25, contact=True)
    c.reset([0])
    assert c.timeout.tolist() == [False, True]
    assert c.armed.tolist() == [True, False]
    assert c.guard_blend.tolist() == [1.0, 1.0]
    assert c.age[0].item() == 0.0


@pytest.mark.parametrize("values", [(math.nan, 0.0, 0.0), (0.0, math.inf, 0.0)])
def test_nonfinite_physical_inputs_are_rejected(values):
    c = RecoveryController(1, "cpu")
    with pytest.raises(ValueError, match="finite"):
        c.step(*values)


def test_reward_scales_keep_normal_cost_and_recovery_minimum_exact():
    c = RecoveryController(1, "cpu")
    env = SimpleNamespace(num_envs=1, device="cpu", impact_runtime=SimpleNamespace(recovery=c))
    constant = lambda env: torch.tensor([2.0])
    assert MDP.recovery_scaled(env, constant, {}, recovery_factor=0.1).item() == 2.0
    observe(c, contact=True)
    assert MDP.recovery_scaled(env, constant, {}, recovery_factor=0.1).item() == pytest.approx(0.2)
    assert MDP.recovery_scaled(env, constant, {}, recovery_factor=0.5).item() == pytest.approx(1.0)
    run(c, 0.4)
    scaled = MDP.recovery_scaled(env, constant, {}, recovery_factor=0.5).item()
    assert 1.0 < scaled < 2.0


class Scene(dict):
    pass


def make_runtime():
    from test_impact_runtime import _env
    env = _env(1)
    env._sim_step_counter = 0
    env.step_dt = 0.02
    env.cfg.recovery = RecoveryConfig()
    env.scene = Scene(env.scene)
    env.scene.env_origins = torch.zeros(1, 3)
    env.scene.sensors = {}
    robot = env.scene["robot"]
    robot.data.root_pos_w[:, 2] = 0.8
    robot.data.joint_pos = torch.zeros(1, 2)
    robot.find_joints = lambda names, preserve_order: ([0, 1], names)
    env.cfg.rewards = SimpleNamespace(standard_arm_pose_l1=SimpleNamespace(params={
        "asset_cfg": SimpleNamespace(joint_names=["ARM_L", "ARM_R"]),
        "term_params": {"target_joint_pos": (0.0, 0.0)},
    }))
    runtime = IMP.ImpactRuntime(env, IMP.ImpactCfg(mode="off"))
    env.impact_runtime = runtime
    return env, runtime


def test_controller_timing_uses_each_fresh_physics_sample_once_and_no_history():
    env, runtime = make_runtime()
    torso = SimpleNamespace(data=SimpleNamespace(
        force_matrix_w=torch.zeros(1, 1, 1, 3),
        force_matrix_w_history=torch.ones(1, 4, 1, 1, 3) * 100,
    ))
    env.scene.sensors[MDP.GROUND_SENSOR_PREFIX + "LINK_TORSO_YAW"] = torso
    # A stale history contact does not trigger current-contact persistence.
    runtime.post_physics_step()
    assert not runtime.recovery.active.item()
    torso.data.force_matrix_w[..., 2] = 10
    for counter in range(1, 30):
        env._sim_step_counter = counter
        runtime.post_physics_step()
        runtime.post_physics_step()  # scene.refresh/autoreset must not double tick
    assert not runtime.recovery.knockdown.item()
    assert runtime.recovery.body_contact_age.item() == pytest.approx(0.145)
    env._sim_step_counter = 30
    runtime.post_physics_step()
    assert runtime.recovery.knockdown.item()


def test_signed_progress_does_not_pay_repeated_oscillation_and_clears_on_reset():
    env, runtime = make_runtime()
    runtime.begin_control_step()
    runtime._update_recovery()  # initializes potential in normal standing
    runtime._contact_event.fill_(True)
    env.scene["robot"].data.root_pos_w[:, 2] = 0.7
    runtime._update_recovery()
    env.scene["robot"].data.root_pos_w[:, 2] = 0.8
    runtime._update_recovery()
    assert runtime._control_upright_progress.item() == pytest.approx(0.0, abs=1e-6)
    runtime.last_step_style_scale.fill_(0.1)
    runtime.reset([0])
    assert not runtime._upright_potential_valid.item()
    assert runtime._control_upright_progress.item() == 0.0
    assert runtime.last_step_style_scale.item() == pytest.approx(0.1)


def test_runtime_signature_binds_plus_recovery_thresholds():
    env, runtime = make_runtime()
    before = runtime.config_signature
    env.cfg.recovery = replace(env.cfg.recovery, max_s=3.0)
    assert runtime._config_signature() != before
