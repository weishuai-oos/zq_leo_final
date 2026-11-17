"""Train T800 locomotion from scratch or resume a compatible task checkpoint."""

import argparse
import json
import math
import os
import traceback

import bootstrap
from isaaclab.app import AppLauncher


def _resolve_local_num_envs(total_num_envs: int, world_size: int) -> int:
    """Interpret --num_envs as a global count under torchrun."""
    if total_num_envs <= 0 or world_size <= 0:
        raise ValueError("Number of environments and distributed world size must be positive.")
    if total_num_envs % world_size:
        raise ValueError(
            "Global number of environments must be divisible by the distributed world size: "
            f"num_envs={total_num_envs}, world_size={world_size}."
        )
    return total_num_envs // world_size


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=bootstrap.TRAIN_TASKS, default="T800_base")
parser.add_argument("--num_envs", type=int, default=None, help="Global environment count, including all ranks.")
parser.add_argument("--seed", type=int, default=42)
budget = parser.add_mutually_exclusive_group()
budget.add_argument("--max_iterations", type=int, default=None, help="Absolute final checkpoint iteration.")
budget.add_argument(
    "--additional_iterations", type=int, default=None,
    help="Continuation updates: ankle default 30000; Impact default 40000.",
)
parser.add_argument("--save_interval", type=int, default=None)
parser.add_argument("--log_dir", default="logs/cusrl")
parser.add_argument("--log_interval", type=int, default=1)
parser.add_argument("--run_name", default=None, help="Defaults to the selected task name.")
parser.add_argument("--logger", choices=["tensorboard", "wandb"], default="tensorboard")
parser.add_argument("--checkpoint", default=None, help="Checkpoint file or run directory to resume.")
parser.add_argument(
    "--impact-single-hit-migration", action="store_true",
    help="Migrate only the known legacy Impact stress-combo checkpoint to single-hit stress training.",
)
parser.add_argument("--finetune_lr_scale", type=float, default=None)
parser.add_argument("--finetune_warmup_iterations", type=int, default=None)
parser.add_argument("--finetune_origin_iteration", type=int, default=None)
parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument(
    "--no-debug-vis", action="store_true", help="Disable command visualization before creating the environment."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

for name in ("num_envs", "max_iterations", "additional_iterations", "save_interval", "log_interval"):
    value = getattr(args_cli, name)
    if value is not None and value <= 0:
        parser.error(f"--{name} must be positive.")
is_impact_task = args_cli.task in bootstrap.IMPACT_TASKS
is_impact_v2_task = args_cli.task == bootstrap.IMPACT_V2_TASK
is_impact_v2_plus_task = args_cli.task == bootstrap.IMPACT_V2_PLUS_TASK
is_dynamics_dr_task = args_cli.task in bootstrap.DYNAMICS_DR_TASKS
if args_cli.impact_single_hit_migration and args_cli.task != "T800_base-Impact":
    parser.error("--impact-single-hit-migration requires T800_base-Impact.")
is_combined_v2_task = args_cli.task == bootstrap.COMBINED_V2_ANKLE_TASK
is_ankle_task = args_cli.task in bootstrap.ANKLE_TASKS or is_combined_v2_task
is_combined_task = args_cli.task == bootstrap.COMBINED_ANKLE_TASK
if (is_impact_task or is_ankle_task) and not args_cli.checkpoint:
    parser.error(f"{args_cli.task} requires --checkpoint for continuation.")
if not (is_impact_task or is_ankle_task) and args_cli.additional_iterations is not None:
    parser.error("--additional_iterations requires an ankle or Impact continuation task.")
for name in ("finetune_lr_scale", "finetune_warmup_iterations", "finetune_origin_iteration"):
    value = getattr(args_cli, name)
    if value is not None and not is_ankle_task:
        parser.error(f"--{name} requires an ankle continuation task.")
if args_cli.finetune_lr_scale is not None and (
    not math.isfinite(args_cli.finetune_lr_scale) or not 0 < args_cli.finetune_lr_scale <= 1
):
    parser.error("--finetune_lr_scale must be in (0, 1].")
for name in ("finetune_warmup_iterations", "finetune_origin_iteration"):
    value = getattr(args_cli, name)
    if value is not None and value < 0:
        parser.error(f"--{name} must be nonnegative.")
if args_cli.run_name is None:
    args_cli.run_name = args_cli.task

import torch  # noqa: E402

is_distributed_launch = "LOCAL_RANK" in os.environ
if is_distributed_launch:
    missing = [name for name in ("RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE") if name not in os.environ]
    if missing:
        raise RuntimeError("torchrun is missing rank variables: " + ", ".join(missing))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    args_cli.device = f"cuda:{local_rank}"
args_cli.distributed = is_distributed_launch

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import cusrl  # noqa: E402
from t800_lab.utils.wrappers.cusrl import (  # noqa: E402
    CusrlT800Adapter,
    validate_amp_checkpoint_compatibility,
    verify_agent_checkpoint_restored,
)
from t800_lab.utils.wrappers.cusrl.impact import (  # noqa: E402
    ImpactCusrlAdapter,
    ImpactFinetuneCallback,
    ImpactV2CusrlAdapter,
    StrictCheckpointAgentFactory,
    resolve_impact_iteration_budget,
    validate_impact_v2_checkpoint,
)
from t800_lab.utils.wrappers.cusrl.ankle import (  # noqa: E402
    AnkleCombinedCusrlAdapter,
    AnkleCusrlAdapter,
    AnkleFinetuneCallback,
    resolve_ankle_iteration_budget,
    validate_ankle_training_checkpoint,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    cusrl.set_global_seed(args_cli.seed, deterministic=args_cli.deterministic)
    env_cfg = bootstrap.load_task_config(args_cli.task, "env_cfg_entry_point")
    agent_cfg = bootstrap.load_task_config(args_cli.task, "cusrl_cfg_entry_point")
    # IsaacLab reseeds RNGs during scene construction. Preserve CusRL's rank
    # separation for the new DR runs instead of repeating physical samples.
    env_cfg.seed = args_cli.seed + int(os.environ.get("RANK", 0)) if is_dynamics_dr_task else args_cli.seed
    env_cfg.sim.device = args_cli.device
    agent_cfg.agent_factory.device = env_cfg.sim.device
    if is_ankle_task or is_combined_task or is_impact_task:
        agent_cfg.experiment_name = args_cli.task
    global_num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    world_size = int(os.environ["WORLD_SIZE"]) if is_distributed_launch else 1
    env_cfg.scene.num_envs = _resolve_local_num_envs(int(global_num_envs), world_size)
    if args_cli.no_debug_vis:
        env_cfg.commands.base_velocity.debug_vis = False
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.save_interval is not None:
        agent_cfg.save_interval = args_cli.save_interval

    # Motion buffers partition selected trajectories by rank during scene construction.
    if is_distributed_launch:
        cusrl.utils.configure_distributed()

    with bootstrap.configure_usd_cache(env_cfg):
        wrapped = gym.make(args_cli.task, cfg=env_cfg, disable_env_checker=True)
        if is_ankle_task:
            env = AnkleCusrlAdapter(
                wrapped, task=args_cli.task,
                lr_scale=0.5 if args_cli.finetune_lr_scale is None else args_cli.finetune_lr_scale,
                warmup_iterations=args_cli.finetune_warmup_iterations,
                origin_iteration=args_cli.finetune_origin_iteration,
            )
        elif is_combined_task:
            env = AnkleCombinedCusrlAdapter(wrapped)
        elif is_dynamics_dr_task:
            from t800_lab.utils.wrappers.cusrl.dynamics_dr import DynamicsDRCusrlAdapter

            env = DynamicsDRCusrlAdapter(wrapped)
        elif is_impact_v2_task:
            env = ImpactV2CusrlAdapter(wrapped)
        elif is_impact_v2_plus_task:
            from t800_lab.utils.wrappers.cusrl.impact import ImpactV2PlusCusrlAdapter

            env = ImpactV2PlusCusrlAdapter(wrapped)
        else:
            env = (ImpactCusrlAdapter if is_impact_task else CusrlT800Adapter)(wrapped)
        if is_impact_task:
            env.unwrapped.impact_runtime.allow_legacy_stress_combo_migration = args_cli.impact_single_hit_migration
        trainer = None
        try:
            checkpoint = None
            checkpoint_path = args_cli.checkpoint
            agent_factory = agent_cfg.agent_factory
            if args_cli.checkpoint is not None:
                trial = cusrl.Trial(args_cli.checkpoint)
                checkpoint_path = str(trial.checkpoint_path)
                checkpoint = trial.load_checkpoint(map_location="cpu")
                validate_amp_checkpoint_compatibility(
                    checkpoint,
                    env.amp_observation_dim,
                    checkpoint_path=str(trial.checkpoint_path),
                    task_name=args_cli.task,
                )
                saved_environment = checkpoint.get("environment", {})
                if is_dynamics_dr_task:
                    from t800_lab.utils.wrappers.cusrl.dynamics_dr import validate_dynamics_dr_checkpoint

                    validate_dynamics_dr_checkpoint(checkpoint, env_cfg, allow_source=True)
                elif is_impact_v2_task:
                    validate_impact_v2_checkpoint(checkpoint, env_cfg, allow_source=True)
                elif is_impact_v2_plus_task:
                    from t800_lab.utils.wrappers.cusrl.impact import validate_impact_v2_plus_checkpoint

                    validate_impact_v2_plus_checkpoint(checkpoint, env_cfg, allow_source=True)
                elif is_ankle_task or is_combined_task or (
                    isinstance(saved_environment, dict)
                    and any(key in saved_environment for key in ("ankle_combined", "ankle_finetune"))
                ):
                    validate_ankle_training_checkpoint(checkpoint, args_cli.task, env_cfg)
                if is_impact_task or is_ankle_task or is_combined_task:
                    if type(checkpoint.get("iteration")) is not int or checkpoint["iteration"] < 0:
                        raise ValueError("Resume requires a nonnegative checkpoint iteration.")
                    if not isinstance(checkpoint.get("environment"), dict) or not isinstance(
                        checkpoint.get("stats"), dict
                    ):
                        raise ValueError("Resume requires complete environment and trainer state.")
                    agent_factory = StrictCheckpointAgentFactory(agent_factory, checkpoint["agent"])
                if is_impact_task or is_ankle_task:
                    resolve_budget = resolve_ankle_iteration_budget if is_ankle_task else resolve_impact_iteration_budget
                    agent_cfg.max_iterations = resolve_budget(
                        checkpoint["iteration"], max_iterations=args_cli.max_iterations,
                        additional_iterations=args_cli.additional_iterations,
                    )
                elif is_combined_task and agent_cfg.max_iterations <= checkpoint["iteration"]:
                    raise ValueError("AnkleCombined --max_iterations must exceed the restored iteration.")

            trainer = cusrl.Trainer(
                env,
                agent_factory,
                logger_factory=cusrl.make_logger_factory(
                    args_cli.logger,
                    log_dir=f"{args_cli.log_dir}/{agent_cfg.experiment_name}",
                    name=args_cli.run_name,
                    interval=args_cli.log_interval,
                ),
                num_iterations=agent_cfg.max_iterations,
                save_interval=agent_cfg.save_interval,
                checkpoint_path=checkpoint_path,
            )
            if checkpoint is not None and (is_impact_task or is_ankle_task or is_combined_task):
                verify_agent_checkpoint_restored(checkpoint["agent"], trainer.agent)
            if is_dynamics_dr_task:
                from t800_lab.utils.wrappers.cusrl.dynamics_dr import configure_dynamics_dr_amp_batch

                amp_batch = configure_dynamics_dr_amp_batch(trainer.agent, world_size)
                print(f"[INFO] Dynamics DR AMP batch: {amp_batch} per GPU, 1024 global.", flush=True)
            if is_ankle_task:
                trainer.register_callback(AnkleFinetuneCallback())
                metadata = env.finetune_metadata
                print(f"[INFO] Ankle continuation: {json.dumps(metadata, sort_keys=True)}, endpoint={trainer.num_iterations}", flush=True)
                trainer.dump_object(env_cfg, "environment_config.pkl")
                if trainer.logger is not None:
                    trainer.logger.save_info(json.dumps(metadata, indent=2), "ankle_finetune.json")
            elif is_combined_task:
                metadata = env.configuration_metadata
                print(
                    f"[INFO] AnkleCombined: {'from scratch' if checkpoint is None else 'resume'}, "
                    f"iteration={trainer.iteration}, endpoint={trainer.num_iterations}; "
                    "configured rewards active immediately; no finetune LR cap or warmup.", flush=True,
                )
                trainer.dump_object(env_cfg, "environment_config.pkl")
                if trainer.logger is not None:
                    trainer.logger.save_info(json.dumps(metadata, indent=2), "ankle_combined.json")
            elif is_impact_task:
                # register_callback invokes this immediately, before the first
                # checkpoint save, environment reset or rollout.
                if is_dynamics_dr_task:
                    from t800_lab.utils.wrappers.cusrl.dynamics_dr import DynamicsDRFinetuneCallback

                    trainer.register_callback(DynamicsDRFinetuneCallback())
                else:
                    trainer.register_callback(ImpactFinetuneCallback())
                metadata = env.finetune_metadata
                print(
                    f"[INFO] Impact continuation: origin={metadata['stage_origin_iteration']}, "
                    f"completed={metadata['completed_updates']}, endpoint={trainer.num_iterations}, "
                    f"lr_caps={metadata['lr_caps']}.", flush=True,
                )
            del checkpoint, agent_factory
            trainer.run_training_loop()
        finally:
            if trainer is not None and trainer.logger is not None and args_cli.logger == "tensorboard":
                trainer.logger.provider.close()
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
