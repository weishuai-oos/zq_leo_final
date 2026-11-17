"""Export a trained T800_base CusRL policy as ONNX or TorchScript."""

import argparse
from contextlib import nullcontext
import json
import math
import os
import traceback

import bootstrap
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task", choices=[
        "T800_base", "T800_base-Play",
        *(name for task in bootstrap.IMPACT_TASKS for name in (task, task + "-Play")),
    ],
    default="T800_base-Play",
)
parser.add_argument("--head_mode", choices=["full", "headless"], default=None)
parser.add_argument("--output_dir", default="exported/cusrl")
parser.add_argument("--format", choices=["onnx", "jit"], default="onnx")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--opset_version", type=int, default=None)
parser.add_argument("--dynamo", action="store_true")
parser.add_argument("--no_optimize", action="store_false", dest="optimize")
parser.add_argument("--silent", action="store_true")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--load_run", required=True, help="Checkpoint file or run directory.")
args_cli = parser.parse_args()
is_impact_task = args_cli.task.startswith("T800_base-Impact")
is_impact_v2_plus_task = args_cli.task.removesuffix("-Play") == bootstrap.IMPACT_V2_PLUS_TASK
is_dynamics_dr_task = args_cli.task.removesuffix("-Play") in bootstrap.DYNAMICS_DR_TASKS
if not is_impact_task and args_cli.head_mode is not None:
    parser.error("--head_mode requires an Impact task.")
args_cli.headless = True
if args_cli.batch_size <= 0:
    parser.error("--batch_size must be positive.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import cusrl  # noqa: E402
from t800_lab.utils.wrappers.cusrl import (  # noqa: E402
    CusrlT800Adapter,
    load_agent_checkpoint_strict,
    validate_amp_checkpoint_compatibility,
)
from t800_lab.utils.wrappers.cusrl.impact import (  # noqa: E402
    ImpactCusrlAdapter,
    ImpactV2CusrlAdapter,
    ImpactV2PlusCusrlAdapter,
    impact_export_masks,
    validate_impact_v2_checkpoint,
)


def main():
    cusrl.config.enable_flash_attention(False)
    env_cfg = bootstrap.load_task_config(args_cli.task, "env_cfg_entry_point")
    agent_cfg = bootstrap.load_task_config(args_cli.task, "cusrl_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    agent_cfg.agent_factory.device = env_cfg.sim.device
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.push_robot = None
    env_cfg.commands.base_velocity.debug_vis = False
    if is_impact_task:
        env_cfg.impact.mode = "off"
        if args_cli.head_mode is not None:
            env_cfg.impact.head_mode = args_cli.head_mode

    with bootstrap.configure_usd_cache(env_cfg):
        adapter_type = (
            ImpactV2CusrlAdapter if args_cli.task.removesuffix("-Play") == bootstrap.IMPACT_V2_TASK
            else ImpactV2PlusCusrlAdapter if is_impact_v2_plus_task
            else ImpactCusrlAdapter if is_impact_task else CusrlT800Adapter
        )
        if is_dynamics_dr_task:
            from t800_lab.utils.wrappers.cusrl.dynamics_dr import DynamicsDRCusrlAdapter

            adapter_type = DynamicsDRCusrlAdapter
        env = adapter_type(
            gym.make(args_cli.task, cfg=env_cfg, disable_env_checker=True)
        )
        try:
            agent = agent_cfg.agent_factory.from_environment(env)
            trial = cusrl.Trial(args_cli.load_run, verbose=not args_cli.silent)
            checkpoint = trial.load_checkpoint(map_location=agent.device)
            if is_dynamics_dr_task:
                from t800_lab.utils.wrappers.cusrl.dynamics_dr import validate_dynamics_dr_checkpoint

                validate_dynamics_dr_checkpoint(checkpoint, env_cfg)
            elif args_cli.task.removesuffix("-Play") == bootstrap.IMPACT_V2_TASK:
                validate_impact_v2_checkpoint(checkpoint, env_cfg)
            elif is_impact_v2_plus_task:
                from t800_lab.utils.wrappers.cusrl.impact import validate_impact_v2_plus_checkpoint

                validate_impact_v2_plus_checkpoint(checkpoint, env_cfg)
            validate_amp_checkpoint_compatibility(
                checkpoint,
                env.amp_observation_dim,
                checkpoint_path=str(trial.checkpoint_path),
                task_name=args_cli.task,
            )
            if is_impact_task:
                load_agent_checkpoint_strict(checkpoint["agent"], agent)
            else:
                agent.load_state_dict(checkpoint["agent"])
            env.load_state_dict(checkpoint["environment"])
            del checkpoint
            output_dir = os.path.abspath(os.path.join(args_cli.output_dir, agent_cfg.experiment_name, trial.name))
            head_position = None
            if is_impact_task:
                from importlib import import_module

                head_mdp = import_module(
                    "t800_lab.tasks.t800_impact_v2.mdp"
                    if bootstrap.impact_parent_task(args_cli.task) == bootstrap.IMPACT_V2_TASK
                    else "t800_lab.tasks.t800_impact_v2_plus.mdp"
                    if bootstrap.impact_parent_task(args_cli.task) == bootstrap.IMPACT_V2_PLUS_TASK
                    else "t800_lab.tasks.t800_impact.head_mdp"
                )
                head_position = tuple(float(value) for value in head_mdp.head_reference_positions(env.unwrapped)[0])
            with impact_export_masks(agent, head_position, task=args_cli.task) if is_impact_task else nullcontext():
                agent.export(
                    output_dir=output_dir,
                    target_format=args_cli.format,
                    optimize=args_cli.optimize,
                    batch_size=args_cli.batch_size,
                    opset_version=args_cli.opset_version,
                    dynamo=args_cli.dynamo,
                    verbose=not args_cli.silent,
                )
            if is_impact_task:
                LOGICAL_JOINT_NAMES = head_mdp.LOGICAL_JOINT_NAMES

                robot = env.unwrapped.scene["robot"]
                physical_names = list(robot.joint_names)
                physical_defaults = robot.data.default_joint_pos[0]
                default_offsets = []
                for index, name in enumerate(LOGICAL_JOINT_NAMES):
                    if name in physical_names:
                        default_offsets.append(float(physical_defaults[physical_names.index(name)]))
                    elif index >= 23:
                        default_offsets.append(head_position[index - 23])
                    else:
                        raise ValueError(f"Impact export is missing required physical joint {name!r}.")
                if not all(math.isfinite(value) for value in default_offsets):
                    raise ValueError("Impact export default joint offsets must all be finite.")
                contract = {
                    "version": 1, "task": args_cli.task, "checkpoint": str(trial.checkpoint_path),
                    "actor_observation_dim": 84, "critic_observation_dim": 87,
                    "action_dim": 25, "amp_observation_dim": 96,
                    "control_period_s": float(env_cfg.sim.dt * env_cfg.decimation),
                    "joint_order": list(LOGICAL_JOINT_NAMES),
                    "observation_order": [
                        {"name": name, "width": width} for name, width in
                        (("base_ang_vel", 3), ("projected_gravity", 3), ("velocity_commands", 3),
                         ("joint_pos", 25), ("joint_vel", 25), ("previous_actions", 25))
                    ],
                    "head_mask_built_into_model": True,
                    "head_position_columns": [32, 33], "head_position_values_rad": list(head_position),
                    "head_zero_observation_columns": [57, 58, 82, 83],
                    "head_zero_action_columns": [23, 24],
                    "head_output_semantics": "Normalized zero; apply default joint position offset for fixed target.",
                    "action_scale": env_cfg.actions.joint_pos.scale,
                    "use_default_joint_position_offset": env_cfg.actions.joint_pos.use_default_offset,
                    "default_joint_position_offsets_rad": default_offsets,
                    "head_mode_for_export_validation": env_cfg.impact.head_mode,
                    "recurrent_memory": {"type": "LSTM", "layers": 2, "hidden_size": 256,
                                         "reset": "Reset all recurrent memories on episode/controller reset.",
                                         "torch_inference": "Use torch.inference_mode() for JIT inference and reset."},
                }
                with open(os.path.join(output_dir, "deployment_contract.json"), "w", encoding="utf-8") as stream:
                    json.dump(contract, stream, indent=2, ensure_ascii=False)
                    stream.write("\n")
            print(f"[INFO] Export completed: {output_dir}")
        finally:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        import omni.kit.app

        omni.kit.app.get_app().post_quit(1)
        raise
    finally:
        simulation_app.close()
