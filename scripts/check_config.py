#!/usr/bin/env python3
"""Inspect real IsaacLab config, optionally compare the historical task or step it.

Examples (run in the existing IsaacLab/Leo Python environment)::

    python scripts/check_config.py --headless
    python scripts/check_config.py --headless --steps 4 --num-envs 2
    python scripts/check_config.py --headless --compare-original ../zq_leo

Configuration inspection initializes Kit but does not create a simulation unless
--steps is positive. The original project is only an optional read-only reference.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import importlib
import json
import math
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/cusrl"))
import bootstrap  # noqa: E402


def freeze(value):
    """Compare values while ignoring the package name and relocation of resources."""
    if callable(value):
        return {"callable": value.__name__}
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        for marker in ("/amp_dataset", "/resources/t800"):
            if marker in value:
                return value[value.index(marker) + 1 :]
        return value
    if isinstance(value, dict):
        return {str(k): freeze(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [freeze(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return {k: freeze(v) for k, v in vars(value).items() if not k.startswith("_")}
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return repr(value)


def checkpoint_metadata_for_report(adapter):
    """Report continuation metadata without the impact runtime's large RNG tensor."""
    return freeze({key: value for key, value in adapter.state_dict().items() if key != "impact"})


def observation_terms(group):
    return {k: v for k, v in vars(group).items() if v is not None and hasattr(v, "func")}


def check_contract(cfg, *, extra_expected_weights=None):
    expected_weights = {
        "track_lin_vel_xy_exp": 1.6,
        "track_ang_vel_z_exp": 1.0,
        "lin_vel_z_l2": -0.5,
        "ang_vel_xy_l2": -0.5,
        "flat_orientation_l2": -0.6,
        "stand_still_without_cmd": -0.15,
        "energy": -1e-5,
        "dof_acc_l2": -2e-7,
        "action_rate_l2": -0.0045,
        "undesired_contacts": -1.0,
        "feet_slide": -0.5,
        "feet_force": -0.0005,
        "fly": -1.0,
        "feet_air_time": 0.5,
        "dof_pos_limits": -1.0,
        "joint_deviation_hip": -0.1,
        "joint_deviation_arms": 0.0,
        "joint_deviation_head": 0.0,
        "joint_deviation_torso": -1.0,
        "joint_deviation_knee": -0.05,
        "termination_penalty": -200.0,
        "track_lin_vel_xy_error_l2": -1.0,
        "track_ang_vel_z_error_l2": -0.5,
        "standard_arm_pose_l1": -0.2,
        "standard_head_pose_l1": -0.2,
        "ankle_target_rate_l2": -0.01,
        "ankle_target_acceleration_l2": -0.002,
        "ankle_overspeed_hinge_l2": -0.025,
        "ankle_acceleration_l2": -0.001,
        "ankle_target_soft_limit_hinge_l2": -0.2,
        "ankle_actual_soft_limit_hinge_l2": -0.2,
    }
    for name, (_, final) in getattr(cfg, "ankle_finetune_reward_weights", {}).items():
        expected_weights[name] = final
    if extra_expected_weights:
        expected_weights.update(extra_expected_weights)
    rewards = {k: v for k, v in vars(cfg.rewards).items() if v is not None and hasattr(v, "weight")}
    actual_weights = {k: v.weight for k, v in rewards.items()}
    assert actual_weights == expected_weights, f"reward contract mismatch: expected={expected_weights}, actual={actual_weights}"
    assert cfg.actions.joint_pos.use_default_offset
    assert cfg.actions.joint_pos.preserve_order
    assert len(cfg.actions.joint_pos.joint_names) == 25
    for name in ("J04_ANKLE_PITCH_L", "J10_ANKLE_PITCH_R"):
        assert cfg.scene.robot.init_state.joint_pos[name] == -0.14
    dimensions = {}
    orders = {}
    for name in ("policy", "critic", "amp_policy", "amp_reference"):
        terms = observation_terms(getattr(cfg.observations, name))
        orders[name] = list(terms)
        dimension = 0
        for term_name, term in terms.items():
            if term_name.startswith("joint_"):
                width = len(term.params["asset_cfg"].joint_names)
            else:
                width = 25 if term_name == "actions" else 3
            dimension += width * max(1, term.history_length)
        dimensions[name] = dimension
    assert dimensions == {"policy": 84, "critic": 87, "amp_policy": 96, "amp_reference": 96}
    assert orders["policy"] == [
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        "actions",
    ]
    assert orders["critic"] == ["base_lin_vel", *orders["policy"]]
    assert (
        orders["amp_policy"]
        == orders["amp_reference"]
        == ["velocity_commands", "projected_gravity", "joint_pos_rel", "joint_vel"]
    )
    assert math.isclose(cfg.sim.dt * cfg.decimation, 0.02)
    assert math.isclose(cfg.scene.contact_forces.update_period, 0.005)
    assert math.isclose(cfg.scene.motion_reference.update_period, 0.02)
    generator = cfg.scene.terrain.terrain_generator
    proportions = [v.proportion for v in generator.sub_terrains.values()]
    assert proportions == [0.20, 0.08, 0.12, 0.12, 0.08, 0.40]
    assert generator.num_rows == 10 and generator.num_cols == 50
    cumulative = []
    for value in proportions:
        cumulative.append(sum(proportions[: len(cumulative) + 1]) / sum(proportions))
    selected = [next(i for i, bound in enumerate(cumulative) if col / 50 + 0.001 < bound) for col in range(50)]
    columns = [selected.count(i) for i in range(6)]
    assert columns == [10, 4, 6, 6, 4, 20]
    assert tuple(cfg.commands.base_velocity.resampling_time_range) == (4.0, 6.0)
    return {
        "observation_dimensions": dimensions,
        "observation_order": orders,
        "action_joints": cfg.actions.joint_pos.joint_names,
        "initial_joint_pos_and_action_offsets": cfg.scene.robot.init_state.joint_pos,
        "control_dt": cfg.sim.dt * cfg.decimation,
        "terrain_proportions": proportions,
        "terrain_columns": columns,
        "commands": freeze(cfg.commands),
        "rewards": freeze(cfg.rewards),
        "ankle_variant": getattr(cfg, "ankle_variant", None),
        "ankle_pd": freeze(cfg.scene.robot.actuators["feet"]),
    }


def check_ankle_contract(cfg, task):
    from t800_lab.tasks.t800_base.env_cfg import T800BaseEnvCfg

    expected = {
        "T800_base-AnkleSmooth": {
            "ankle_target_rate_l2": (-0.01, -0.04),
            "ankle_target_acceleration_l2": (-0.002, -0.008),
        },
        "T800_base-AnkleDamping": {},
        "T800_base-AnkleTargetLimit": {"ankle_target_soft_limit_hinge_l2": (-0.2, -0.8)},
        "T800_base-AnklePose": {"ankle_support_joint_pose_l2": (0.0, -0.02)},
    }
    assert cfg.ankle_finetune_reward_weights == expected[task]
    result = check_contract(cfg)
    base = T800BaseEnvCfg()
    expected_damping = 0.6 if task.endswith("AnkleDamping") else 0.3
    feet = cfg.scene.robot.actuators["feet"]
    assert set(feet.damping.values()) == {expected_damping}
    assert set(feet.stiffness.values()) == {40.0}
    assert set(base.scene.robot.actuators["feet"].damping.values()) == {0.3}
    for name in ("observations", "actions", "commands", "terminations", "events", "curriculum"):
        assert freeze(getattr(cfg, name)) == freeze(getattr(base, name)), name
    result["variant_reward_weights"] = expected[task]
    return result


def check_combined_ankle_contract(cfg):
    """Check the standalone from-scratch ankle combination task."""
    from t800_lab.tasks.t800_base.env_cfg import T800BaseEnvCfg

    result = check_contract(
        cfg,
        extra_expected_weights={
            "ankle_target_rate_l2": -0.04,
            "ankle_target_acceleration_l2": -0.008,
            "ankle_support_joint_pose_l2": -0.02,
        },
    )
    assert getattr(cfg, "ankle_variant", None) == "AnkleCombined"
    assert getattr(cfg, "run_name", None) == "T800_base-AnkleCombined"
    assert not hasattr(cfg, "ankle_finetune_reward_weights")
    rewards = cfg.rewards
    assert rewards.ankle_target_rate_l2.weight == -0.04
    assert rewards.ankle_target_acceleration_l2.weight == -0.008
    pose = rewards.ankle_support_joint_pose_l2
    assert pose.weight == -0.02
    assert pose.params["threshold"] == 1.0
    assert pose.params["normalizer_rad"] == 0.1
    from t800_lab.tasks.t800_base.ankle_stable_mdp import ANKLE_JOINT_NAMES

    assert tuple(pose.params["asset_cfg"].joint_names) == tuple(ANKLE_JOINT_NAMES)
    feet = cfg.scene.robot.actuators["feet"]
    assert set(feet.stiffness.values()) == {40.0}
    assert set(feet.damping.values()) == {1.2}
    base = T800BaseEnvCfg()
    assert set(base.scene.robot.actuators["feet"].stiffness.values()) == {40.0}
    assert set(base.scene.robot.actuators["feet"].damping.values()) == {0.3}
    result["variant_reward_weights"] = {
        "ankle_target_rate_l2": -0.04,
        "ankle_target_acceleration_l2": -0.008,
        "ankle_support_joint_pose_l2": -0.02,
    }
    result["combined_from_scratch"] = True
    return result


def check_combined_v2_ankle_contract(cfg):
    from t800_lab.tasks.t800_ankle.env_cfg import T800AnkleCombinedEnvCfg

    result = check_contract(cfg, extra_expected_weights={
        "ankle_target_rate_l2": -0.04,
        "ankle_target_acceleration_l2": -0.008,
        "ankle_support_joint_pose_l2": -0.02,
        "ankle_physics_acceleration_l2": -0.02,
        "ankle_support_foot_angular_velocity_l2": -0.2,
    })
    combined = T800AnkleCombinedEnvCfg()
    for name in ("actions", "observations", "commands", "terminations", "curriculum"):
        assert freeze(getattr(cfg, name)) == freeze(getattr(combined, name)), name
    assert freeze(cfg.scene.terrain) == freeze(combined.scene.terrain)
    assert cfg.ankle_variant == "AnkleCombinedV2"
    assert cfg.run_name == bootstrap.COMBINED_V2_ANKLE_TASK
    assert cfg.ankle_finetune_reward_weights == {
        "ankle_physics_acceleration_l2": (0.0, -0.02),
        "ankle_support_foot_angular_velocity_l2": (0.0, -0.2),
    }
    assert set(cfg.scene.robot.actuators["feet"].stiffness.values()) == {40.0}
    assert set(cfg.scene.robot.actuators["feet"].damping.values()) == {2.0}
    assert cfg.events.collider_offsets.params["contact_offset_distribution_params"] == (0.002, 0.002)
    assert cfg.events.collider_offsets.params["rest_offset_distribution_params"] is None
    assert cfg.ankle_substep_stability["support_settle_time_s"] == 0.03
    result["substep_stability"] = freeze(cfg.ankle_substep_stability)
    result["collider_offsets"] = freeze(cfg.events.collider_offsets)
    return result


def compare_original(cfg, original):
    """Check every effective non-terrain config field against real old Isaac cfg."""
    sys.path.insert(0, str(original.resolve() / "source/leo_lab"))
    module = importlib.import_module("leo_lab.tasks.locomotion.config.t800_amp.ankle_stable_env_cfg")
    old_cls = module.T800LowCylinderTerrainAmpMaskBaseNarrowStdTargetPlusAnkleStable1EnvCfg
    old = freeze(old_cls())
    new = freeze(cfg)
    for data in (old, new):
        data.pop("ANKLE_STABLE_LEVEL", None)
        # IsaacLab ManagerBasedRLEnvCfg has no custom LeoLab monitor manager.
        monitor = data.pop("monitors", None)
        assert monitor in (None, {})
        data.pop("run_name", None)
        for group in ("amp_policy", "amp_reference"):
            for term in ("base_lin_vel", "base_ang_vel"):
                if term in data["observations"][group]:
                    assert data["observations"][group].pop(term) is None
    terrain = old["scene"]["terrain"]["terrain_generator"]
    terrain["num_cols"] = 50
    # The new flat family uses IsaacLab MeshPlaneTerrainCfg defaults. Compare
    # every old family's remaining parameters; validate the flat separately.
    flat = new["scene"]["terrain"]["terrain_generator"]["sub_terrains"].pop("fight_mat_flat")
    assert flat["proportion"] == 0.20
    for value in terrain["sub_terrains"].values():
        value["proportion"] = round(value["proportion"] * 0.8, 10)
    assert old == new, "Real IsaacLab configuration differs from the original beyond the documented exceptions."
    return "matched all effective fields after terrain mix, run_name, empty monitors, unused class constant and four disabled AMP-term normalization"


def check_impact_contract(cfg):
    from t800_lab.tasks.t800_base.env_cfg import T800BaseEnvCfg

    if cfg.run_name in bootstrap.DYNAMICS_DR_TASKS:
        from dataclasses import asdict

        result = check_impact_contract(cfg.dr_parent_cfg)
        assert freeze(cfg.rewards) == freeze(cfg.dr_parent_cfg.rewards)
        assert freeze(cfg.scene.robot.actuators) == freeze(cfg.dr_parent_cfg.scene.robot.actuators)
        assert cfg.actions.joint_pos.joint_names == cfg.dr_parent_cfg.actions.joint_pos.joint_names
        assert cfg.actions.joint_pos.scale == cfg.dr_parent_cfg.actions.joint_pos.scale
        result["dynamics_randomization"] = asdict(cfg.dynamics_randomization)
        result["dr_source"] = {"task": cfg.dr_parent_task, "iteration": cfg.dr_source_iteration}
        return result

    is_v2 = cfg.run_name == bootstrap.IMPACT_V2_TASK
    is_plus = cfg.run_name == bootstrap.IMPACT_V2_PLUS_TASK
    if is_v2:
        from t800_lab.tasks.t800_impact_v2 import mdp

        head_mdp = mdp
    elif is_plus:
        from t800_lab.tasks.t800_impact_v2_plus import mdp

        head_mdp = mdp
    else:
        from t800_lab.tasks.t800_impact import head_mdp, mdp

    base = T800BaseEnvCfg()
    check_contract(base)
    assert cfg.actions.joint_pos.joint_names == base.actions.joint_pos.joint_names
    assert cfg.actions.joint_pos.scale == base.actions.joint_pos.scale
    assert cfg.actions.joint_pos.use_default_offset and cfg.actions.joint_pos.preserve_order
    for name in ("policy", "critic", "amp_policy", "amp_reference"):
        assert list(observation_terms(getattr(cfg.observations, name))) == list(
            observation_terms(getattr(base.observations, name)))
    assert cfg.observations.policy.joint_pos.func is head_mdp.joint_pos_masked
    assert cfg.observations.policy.joint_pos.noise is None
    assert cfg.observations.policy.joint_vel.noise is None
    expected_fall = mdp.ground_fall_plus if is_plus else mdp.ground_fall
    assert cfg.terminations.base_contact.func is expected_fall
    assert cfg.events.push_robot is None and cfg.events.base_external_force_torque is None
    assert cfg.rewards.standard_head_pose_l1 is None and cfg.rewards.joint_deviation_head is None
    assert cfg.rewards.feet_air_time is None
    generator = cfg.scene.terrain.terrain_generator
    proportions = [value.proportion for value in generator.sub_terrains.values()]
    assert proportions == [0.8, 0.2]
    assert generator.num_cols == 10
    assert generator.sub_terrains["fight_mat_horizontal_obstacles"].terrain_kind == "horizontal"
    assert cfg.sim.dt == 0.005 and cfg.decimation == 4
    sensors = {name: value for name, value in vars(cfg.scene).items()
               if name.startswith(mdp.GROUND_SENSOR_PREFIX) and value is not None}
    assert len(sensors) >= 20
    for sensor in sensors.values():
        assert ".*" not in sensor.prim_path
        assert sensor.filter_prim_paths_expr == ["/World/ground/terrain/mesh"]
        assert sensor.history_length == 4 and sensor.update_period == 0.005
    if is_v2 or is_plus:
        feet = cfg.scene.robot.actuators["feet"]
        assert set(feet.stiffness.values()) == {40.0}
        assert set(feet.damping.values()) == {2.0}
        assert cfg.events.collider_offsets.params["contact_offset_distribution_params"] == (0.002, 0.002)
        assert cfg.events.collider_offsets.params["rest_offset_distribution_params"] is None
        for name, weight in (
            ("ankle_target_rate_l2", -0.04), ("ankle_target_acceleration_l2", -0.008),
            ("ankle_support_joint_pose_l2", -0.02), ("ankle_physics_acceleration_l2", -0.02),
            ("ankle_support_foot_angular_velocity_l2", -0.2),
        ):
            term = getattr(cfg.rewards, name)
            assert term.weight == weight and term.func is mdp.recovery_scaled
        assert "sensor_cfg" in cfg.rewards.ankle_support_joint_pose_l2.params
        assert len(cfg.impact_v2_source_play_signature) == 64
    if is_plus:
        assert len(cfg.impact_v2_parent_configuration_signature) == 64
        assert cfg.recovery.max_s == 2.5
    # Inherited config objects must not have been mutated by the new task.
    check_contract(T800BaseEnvCfg())
    result = {
        "observation_dimensions": {"policy": 84, "critic": 87, "amp_policy": 96, "amp_reference": 96},
        "action_joints": cfg.actions.joint_pos.joint_names,
        "control_dt": cfg.sim.dt * cfg.decimation,
        "terrain_proportions": proportions,
        "terrain_columns": [8, 2],
        "ground_filtered_body_sensors": len(sensors),
        "impact": freeze(cfg.impact),
        "rewards": freeze(cfg.rewards),
    }
    if is_plus:
        result["recovery"] = freeze(cfg.recovery)
        result["source_contracts"] = {
            "combined_v2": cfg.impact_v2_source_play_signature,
            "impact_v2": cfg.impact_v2_parent_configuration_signature,
            "impact_v2_runtime": cfg.impact_v2_parent_runtime_signature,
        }
    return result


@contextmanager
def capture_impact_impulse(env):
    """Include pulse samples cleared by autoresets during a smoke run."""
    raw = env.unwrapped
    runtime = raw.impact_runtime
    original_reset = raw._reset_idx
    evidence = {"impulse_ns": 0.0}

    def capture_reset(env_ids):
        evidence["impulse_ns"] += float(runtime._stats["impulse_ns"][env_ids].sum().item())
        return original_reset(env_ids)

    raw._reset_idx = capture_reset
    try:
        yield evidence
    finally:
        raw._reset_idx = original_reset
        evidence["impulse_ns"] += float(runtime._stats["impulse_ns"].sum().item())


def check_impact_ground_contact(env):
    """Verify actual filtered sole forces and non-sole fall resets on PhysX."""
    import torch
    module = importlib.import_module(
        "t800_lab.tasks.t800_impact_v2.mdp"
        if bootstrap.impact_parent_task(env.unwrapped.cfg.run_name) == bootstrap.IMPACT_V2_TASK
        else "t800_lab.tasks.t800_impact_v2_plus.mdp"
        if bootstrap.impact_parent_task(env.unwrapped.cfg.run_name) == bootstrap.IMPACT_V2_PLUS_TASK
        else "t800_lab.tasks.t800_impact.mdp"
    )
    FOOT_BODIES = module.FOOT_BODIES
    GROUND_SENSOR_PREFIX = module.GROUND_SENSOR_PREFIX
    nonsole_ground_contacts = module.nonsole_ground_contacts

    raw = env.unwrapped
    runtime = raw.impact_runtime
    robot = raw.scene["robot"]
    saved_mode, saved_reset = runtime.cfg.mode, raw._reset_idx
    contacts = torch.zeros(raw.num_envs, device=raw.device, dtype=torch.bool)

    def capture_reset(env_ids):
        contacts[env_ids] |= nonsole_ground_contacts(raw)[env_ids] > 0
        return saved_reset(env_ids)

    try:
        runtime.cfg.mode = "off"
        env.reset()
        actions = torch.zeros((raw.num_envs, 25), device=raw.device)
        sole_peak = 0.0
        for _ in range(25):
            env.step(actions)
            for name in FOOT_BODIES:
                sensor = raw.scene.sensors[GROUND_SENSOR_PREFIX + name]
                sole_peak = max(sole_peak, float(sensor.data.force_matrix_w.norm(dim=-1).max().item()))
        if sole_peak <= 1.0:
            raise RuntimeError("Ground contact smoke saw no filtered sole support force.")
        env.reset()
        state = robot.data.default_root_state.clone()
        state[:, :3] = raw.scene.env_origins + state.new_tensor((0.0, 0.0, 0.25))
        state[:, 3:7] = state.new_tensor((2**-0.5, 0.0, 2**-0.5, 0.0))
        state[:, 7:] = 0.0
        robot.write_root_state_to_sim(state)
        raw.scene.write_data_to_sim()
        raw.sim.forward()
        raw.scene.update(raw.step_dt)
        raw._reset_idx = capture_reset
        terminated_once = torch.zeros_like(contacts)
        for index in range(25):
            _, _, terminated, _, _ = env.step(actions)
            terminated_once |= terminated
            if bool(terminated_once.all()) and bool(contacts.all()):
                return {
                    "sole_filtered_force_peak_n": sole_peak,
                    "nonsole_ground_contact_per_env": contacts.tolist(),
                    "fallen_terminated_per_env": terminated_once.tolist(),
                    "all_falls_detected_within_s": (index + 1) * raw.step_dt,
                }
        raise RuntimeError("Forced fallen robots did not all terminate on non-sole ground contact within 0.5 s.")
    finally:
        raw._reset_idx = saved_reset
        runtime.cfg.mode = saved_mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-original", type=Path)
    parser.add_argument("--task", choices=bootstrap.TRAIN_TASKS, default="T800_base")
    parser.add_argument("--checkpoint", type=Path, help="Strict full-agent restore for the smoke check")
    parser.add_argument("--finetune-lr-scale", type=float, default=0.5)
    parser.add_argument("--finetune-warmup-iterations", type=int, default=None)
    parser.add_argument("--finetune-origin-iteration", type=int, default=None)
    parser.add_argument("--impact-single-hit-migration", action="store_true")
    parser.add_argument("--impact-mode", choices=("train", "off", "common", "stress"), default="common")
    parser.add_argument("--head-mode", choices=("full", "headless"), default="full")
    parser.add_argument("--output", type=Path, help="Optional JSON evidence file")
    parser.add_argument("--steps", type=int, default=0, help="Create an env and step zero actions this many times")
    parser.add_argument(
        "--smoke", action="store_true", help="Run 4 zero-action adapter steps (2 environments by default)"
    )
    parser.add_argument("--num-envs", "--num_envs", type=int, default=2)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.impact_single_hit_migration and args.task != "T800_base-Impact":
        parser.error("--impact-single-hit-migration requires T800_base-Impact")
    if args.smoke and args.steps == 0:
        args.steps = 4
    if args.steps < 0 or args.num_envs < 1:
        parser.error("--steps must be nonnegative and --num-envs must be positive")
    if args.task in bootstrap.IMPACT_TASKS and args.steps and args.checkpoint is None:
        parser.error("Impact smoke requires --checkpoint for strict stage-one/stage-two restore")
    if args.task == bootstrap.COMBINED_V2_ANKLE_TASK and args.steps and args.checkpoint is None:
        parser.error("AnkleCombinedV2 smoke requires a Combined or matching V2 --checkpoint")
    sys.path.insert(0, str(ROOT / "source/t800_lab"))
    sys.path.insert(0, str(ROOT / "scripts/cusrl"))
    launcher = AppLauncher(args)
    env = None
    usd_cache = None
    try:
        import gymnasium as gym
        import t800_lab.tasks  # noqa: F401
        from bootstrap import configure_usd_cache, load_task_config

        impact_task = args.task in bootstrap.IMPACT_TASKS
        cfg = load_task_config(args.task, "env_cfg_entry_point")
        cfg.validate()
        if impact_task:
            result = check_impact_contract(cfg)
        elif args.task == bootstrap.COMBINED_V2_ANKLE_TASK:
            result = check_combined_v2_ankle_contract(cfg)
        elif args.task == "T800_base-AnkleCombined":
            result = check_combined_ankle_contract(cfg)
        elif args.task in bootstrap.ANKLE_TASKS:
            result = check_ankle_contract(cfg, args.task)
        else:
            result = check_contract(cfg)
        play = load_task_config(args.task + "-Play", "env_cfg_entry_point")
        play.validate()
        if args.task in (
            bootstrap.COMBINED_ANKLE_TASK, bootstrap.COMBINED_V2_ANKLE_TASK,
            bootstrap.IMPACT_V2_TASK, bootstrap.IMPACT_V2_PLUS_TASK, *bootstrap.DYNAMICS_DR_TASKS, *bootstrap.ANKLE_TASKS,
        ):
            assert freeze(play.rewards) == freeze(cfg.rewards)
            assert freeze(play.scene.robot.actuators) == freeze(cfg.scene.robot.actuators)
            assert play.events.push_robot is None and play.events.base_external_force_torque is None
        assert tuple(play.commands.base_velocity.resampling_time_range) == (4.0, 6.0)
        ids = sorted(name for name in gym.registry if name.startswith("T800_"))
        assert ids == sorted(bootstrap.ALL_TASKS)
        result["registered_tasks"] = ids
        if args.compare_original:
            if impact_task or args.task in (*bootstrap.ANKLE_TASKS, bootstrap.COMBINED_ANKLE_TASK, bootstrap.COMBINED_V2_ANKLE_TASK):
                parser.error("--compare-original applies only to the unchanged first stage")
            result["original_comparison"] = compare_original(cfg, args.compare_original)
        if args.steps:
            import torch
            from t800_lab.utils.wrappers.cusrl.vecenv_wrapper import CusrlT800Adapter

            cfg.scene.num_envs = args.num_envs
            cfg.sim.device = args.device
            cfg.commands.base_velocity.debug_vis = False
            if impact_task:
                cfg.impact.mode = args.impact_mode
                cfg.impact.head_mode = args.head_mode
            usd_cache = configure_usd_cache(cfg)
            env = gym.make(args.task, cfg=cfg)
            if impact_task:
                from t800_lab.utils.wrappers.cusrl.impact import ImpactCusrlAdapter, ImpactV2CusrlAdapter, ImpactV2PlusCusrlAdapter

                adapter_type = (
                    ImpactV2CusrlAdapter if args.task == bootstrap.IMPACT_V2_TASK
                    else ImpactV2PlusCusrlAdapter if args.task == bootstrap.IMPACT_V2_PLUS_TASK
                    else ImpactCusrlAdapter
                )
                if args.task in bootstrap.DYNAMICS_DR_TASKS:
                    from t800_lab.utils.wrappers.cusrl.dynamics_dr import DynamicsDRCusrlAdapter

                    adapter_type = DynamicsDRCusrlAdapter
                adapter = adapter_type(env)
                env.unwrapped.impact_runtime.allow_legacy_stress_combo_migration = args.impact_single_hit_migration
            elif args.task == "T800_base-AnkleCombined":
                from t800_lab.utils.wrappers.cusrl.ankle import AnkleCombinedCusrlAdapter

                adapter = AnkleCombinedCusrlAdapter(env)
            elif args.task in (*bootstrap.ANKLE_TASKS, bootstrap.COMBINED_V2_ANKLE_TASK):
                from t800_lab.utils.wrappers.cusrl.ankle import AnkleCusrlAdapter

                adapter = AnkleCusrlAdapter(
                    env, task=args.task, lr_scale=args.finetune_lr_scale,
                    warmup_iterations=args.finetune_warmup_iterations,
                    origin_iteration=args.finetune_origin_iteration,
                )
            else:
                adapter = CusrlT800Adapter(env)
            observation, state, _ = adapter.reset()
            assert observation.shape == (args.num_envs, 84) and torch.isfinite(observation).all()
            assert state.shape == (args.num_envs, 87) and torch.isfinite(state).all()
            reference = adapter.collect_reference_motions(args.num_envs)
            assert reference.shape == (args.num_envs, 96) and torch.isfinite(reference).all()
            actual = env.unwrapped.observation_manager.compute()
            runtime_dimensions = {
                name: sum(int(value[0].numel()) for value in group.values()) for name, group in actual.items()
            }
            assert runtime_dimensions == result["observation_dimensions"], runtime_dimensions
            assert env.unwrapped.action_manager.total_action_dim == 25
            agent = None
            if args.checkpoint is not None:
                import cusrl
                from t800_lab.utils.wrappers.cusrl.checkpoint import load_agent_checkpoint_strict

                trainer_cfg = load_task_config(args.task, "cusrl_cfg_entry_point")
                trainer_cfg.agent_factory.device = args.device
                agent = trainer_cfg.agent_factory(adapter.spec)
                trial = cusrl.Trial(str(args.checkpoint))
                saved = trial.load_checkpoint(map_location=args.device)
                load_agent_checkpoint_strict(saved["agent"], agent)
                if args.task == bootstrap.COMBINED_V2_ANKLE_TASK:
                    from t800_lab.utils.wrappers.cusrl.ankle import validate_ankle_training_checkpoint

                    validate_ankle_training_checkpoint(saved, args.task, cfg)
                elif args.task in bootstrap.DYNAMICS_DR_TASKS:
                    from t800_lab.utils.wrappers.cusrl.dynamics_dr import validate_dynamics_dr_checkpoint

                    validate_dynamics_dr_checkpoint(saved, cfg, allow_source=True)
                elif args.task == bootstrap.IMPACT_V2_TASK:
                    from t800_lab.utils.wrappers.cusrl.impact import validate_impact_v2_checkpoint

                    validate_impact_v2_checkpoint(saved, cfg, allow_source=True)
                elif args.task == bootstrap.IMPACT_V2_PLUS_TASK:
                    from t800_lab.utils.wrappers.cusrl.impact import validate_impact_v2_plus_checkpoint

                    validate_impact_v2_plus_checkpoint(saved, cfg, allow_source=True)
                adapter.load_state_dict(saved.get("environment", {}))
                result["checkpoint"] = {"path": str(trial.checkpoint_path), "full_restore_verified": True}
                if args.task == bootstrap.COMBINED_V2_ANKLE_TASK:
                    from types import SimpleNamespace
                    from t800_lab.utils.wrappers.cusrl.ankle import AnkleFinetuneCallback

                    AnkleFinetuneCallback()(SimpleNamespace(environment=adapter, agent=agent, iteration=saved["iteration"]))
                    result["checkpoint"]["finetune_metadata"] = checkpoint_metadata_for_report(adapter)
                elif args.task in bootstrap.DYNAMICS_DR_TASKS:
                    from types import SimpleNamespace
                    from t800_lab.utils.wrappers.cusrl.dynamics_dr import DynamicsDRFinetuneCallback

                    DynamicsDRFinetuneCallback()(SimpleNamespace(environment=adapter, agent=agent, iteration=saved["iteration"]))
                    result["checkpoint"]["finetune_metadata"] = checkpoint_metadata_for_report(adapter)
                elif args.task in (bootstrap.IMPACT_V2_TASK, bootstrap.IMPACT_V2_PLUS_TASK):
                    from types import SimpleNamespace
                    from t800_lab.utils.wrappers.cusrl.impact import ImpactFinetuneCallback

                    ImpactFinetuneCallback()(SimpleNamespace(environment=adapter, agent=agent, iteration=saved["iteration"]))
                    result["checkpoint"]["finetune_metadata"] = checkpoint_metadata_for_report(adapter)
            if impact_task and args.impact_mode != "off":
                # Exercise the actual wrench path in a short smoke run rather
                # than waiting for the regular 1.5--3 s first-hit delay.
                env.unwrapped.impact_runtime.cooldown.zero_()
            actions = torch.zeros((args.num_envs, 25), device=env.unwrapped.device)
            with capture_impact_impulse(env) if impact_task else nullcontext(None) as impulse_evidence:
                for _ in range(args.steps):
                    if agent is not None:
                        # Exercise the restored recurrent actor as well as the env.
                        actions = agent.act(observation, state)
                    observation, state, rewards, _, _, extras = adapter.step(actions)
                    for value in (observation, state, rewards, extras["amp_obs"]):
                        assert torch.isfinite(value).all()
                    assert extras["amp_obs"].shape == (args.num_envs, 96)
                    if impact_task:
                        assert extras["impact_style_scale"].shape == (args.num_envs, 1)
            result["simulation"] = {
                "num_envs": args.num_envs,
                "steps": args.steps,
                "observation_dimensions": runtime_dimensions,
            }
            if args.task in (bootstrap.COMBINED_ANKLE_TASK, bootstrap.COMBINED_V2_ANKLE_TASK):
                robot = env.unwrapped.scene["robot"]
                feet = robot.actuators["feet"]
                ids = [int(index) for index in feet.joint_indices]
                assert len(ids) == 4
                actual_kp = robot.root_physx_view.get_dof_stiffnesses()[:, ids]
                actual_kd = robot.root_physx_view.get_dof_dampings()[:, ids]
                expected_kd = 2.0 if args.task == bootstrap.COMBINED_V2_ANKLE_TASK else 1.2
                for values, expected in (
                    (actual_kp, 40.0), (robot.data.joint_stiffness[:, ids], 40.0), (feet.stiffness, 40.0),
                    (actual_kd, expected_kd), (robot.data.joint_damping[:, ids], expected_kd), (feet.damping, expected_kd),
                ):
                    assert torch.allclose(values, torch.full_like(values, expected))
                result["simulation"]["ankle_physx_pd"] = {
                    "kp": actual_kp.cpu().tolist(), "kd": actual_kd.cpu().tolist(),
                }
            if args.task == bootstrap.COMBINED_V2_ANKLE_TASK:
                from t800_lab.tasks.t800_ankle.env import T800AnkleStabilityEnv

                assert isinstance(env.unwrapped, T800AnkleStabilityEnv)
                runtime = env.unwrapped.ankle_stability
                assert runtime.sample_count == cfg.decimation
                offsets = robot.root_physx_view.get_contact_offsets()
                rest_offsets = robot.root_physx_view.get_rest_offsets()
                assert torch.allclose(offsets, torch.full_like(offsets, 0.002))
                assert torch.allclose(rest_offsets, torch.zeros_like(rest_offsets))
                result["simulation"]["substep_stability"] = {
                    "samples_per_control_step": runtime.sample_count,
                    "contact_offset_m": [float(offsets.min()), float(offsets.max())],
                    "rest_offset_m": [float(rest_offsets.min()), float(rest_offsets.max())],
                    "metrics": {k: float(v) for k, v in runtime.metrics().items()},
                }
            if impact_task:
                runtime = env.unwrapped.impact_runtime
                impulse = impulse_evidence["impulse_ns"]
                if args.impact_mode in ("common", "stress"):
                    assert impulse > 0, "Impact smoke did not exercise a force pulse"
                adapter.reset()
                assert not runtime.active.any()
                assert not runtime.forces_b.any() and not runtime.torques_b.any()
                result["simulation"]["applied_impulse_ns"] = impulse
                result["simulation"]["reset_wrenches_cleared"] = True
                if args.smoke:
                    result["simulation"]["ground_contact_smoke"] = check_impact_ground_contact(env)
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        print(rendered)
        if args.output:
            args.output.write_text(rendered + "\n")
    except BaseException:
        # Kit may terminate its process during close; print the cause first.
        traceback.print_exc()
        import omni.kit.app

        omni.kit.app.get_app().post_quit(1)
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            # URDF conversion can still be writing after a constructor error.
            # Stop Kit before removing its temporary USD directory.
            try:
                launcher.app.close()
            finally:
                if usd_cache is not None:
                    usd_cache.cleanup()


if __name__ == "__main__":
    main()
