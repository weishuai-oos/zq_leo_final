"""Play and diagnose a trained T800_base AMP policy."""

import argparse
import math
import traceback
import os

import bootstrap
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task", choices=bootstrap.ALL_TASKS,
    default="T800_base-Play",
)
parser.add_argument("--impact_mode", choices=["off", "common", "stress"], default=None)
parser.add_argument("--head_mode", choices=["full", "headless"], default=None)
parser.add_argument(
    "--impact-vis", action=argparse.BooleanOptionalAction, default=None,
    help="Show incoming force arrows and hit points. Defaults to on for Impact GUI Play, off when headless.",
)
parser.add_argument(
    "--impact-vis-hold", type=float, default=None,
    help="Keep each impact marker visible for this many simulation seconds (default: 0.3).",
)
parser.add_argument("--num_envs", type=int, default=50, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument(
    "--teleop",
    type=str,
    default="none",
    choices=["auto", "keyboard", "gamepad", "none"],
    help="Teleoperation mode for supported play tasks. Defaults to 'none'.",
)
parser.add_argument(
    "--auto_command_test",
    type=str,
    default=None,
    choices=["boxing_practical", "boxing_stability_sweep"],
    help=(
        "Run a non-interactive practical or stability-sweep command sequence. "
        "The sequence is expressed in the boxing tactical frame and uses the same frame conversion as teleop."
    ),
)
parser.add_argument(
    "--auto_command_loops",
    type=int,
    default=1,
    help="Number of times to repeat the automatic command sequence.",
)
parser.add_argument(
    "--auto_command_hold_s",
    type=float,
    default=None,
    help="Optional diagnostic override for the duration of every automatic command segment, in seconds.",
)
parser.add_argument(
    "--free_camera",
    action="store_true",
    help="Start near --camera_env, then leave the viewport camera free to move without robot tracking.",
)
parser.add_argument("--camera_env", type=int, default=0, help="Environment to frame initially (zero-based; default: 0).")
parser.add_argument(
    "--no-debug-vis",
    action="store_true",
    help=(
        "Disable command and waypoint debug visualization before environment creation. "
        "This avoids loading remote USD marker assets when playing headless/offline."
    ),
)
parser.add_argument(
    "--debug_reward_terms",
    nargs="*",
    default=None,
    help="Reward terms to print during play. Omit this flag to disable reward debug output.",
)
parser.add_argument("--debug_reward_env", type=int, default=0, help="Environment index to print reward terms for.")
parser.add_argument(
    "--debug_reward_interval",
    type=int,
    default=1,
    help="Print selected reward terms every N simulation steps.",
)
parser.add_argument(
    "--terrain_level",
    type=int,
    default=None,
    help=(
        "Force generated-terrain Play environments to one exact zero-based terrain level and keep it fixed "
        "across resets. Omit to retain the task's default random initialization and curriculum."
    ),
)
parser.add_argument(
    "--terrain_type",
    type=str,
    default=None,
    help=(
        "Select a generated-terrain family by its configured sub-terrain name. "
        "Must be combined with --terrain_level; use --terrain_column for an exact column."
    ),
)
parser.add_argument(
    "--terrain_column",
    type=int,
    default=None,
    help=(
        "Select an exact zero-based generated-terrain column. Must be combined with --terrain_level "
        "and cannot be combined with --terrain_type."
    ),
)
parser.add_argument(
    "--record_dir",
    type=str,
    default=None,
    help=(
        "Enable T800 AMP Play diagnostics and create a timestamped recording session under this directory. "
        "Ordinary Play behavior is unchanged when omitted."
    ),
)
parser.add_argument(
    "--record_envs",
    type=int,
    nargs="+",
    default=[0],
    help="Environment indices to record when --record_dir is enabled. Defaults to env 0.",
)
parser.add_argument(
    "--record_episodes",
    type=int,
    default=0,
    help="Stop after this many completed recorded episodes; 0 records until Play is interrupted.",
)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--load_run", "--checkpoint", dest="load_run", type=str, required=True,
                    help="Path to the checkpoint file or run directory to load.")
args_cli = parser.parse_args()
is_impact_task = args_cli.task.startswith("T800_base-Impact")
is_impact_v2_plus_task = args_cli.task.removesuffix("-Play") == bootstrap.IMPACT_V2_PLUS_TASK
is_dynamics_dr_task = args_cli.task.removesuffix("-Play") in bootstrap.DYNAMICS_DR_TASKS
is_combined_task = args_cli.task.removesuffix("-Play") == bootstrap.COMBINED_ANKLE_TASK
is_combined_v2_task = args_cli.task.removesuffix("-Play") == bootstrap.COMBINED_V2_ANKLE_TASK
if not is_impact_task and any(value is not None for value in (
    args_cli.impact_mode, args_cli.head_mode, args_cli.impact_vis, args_cli.impact_vis_hold,
)):
    parser.error("--impact_mode/--head_mode/--impact-vis/--impact-vis-hold require an Impact task.")
if args_cli.impact_vis_hold is not None and (
    not math.isfinite(args_cli.impact_vis_hold) or args_cli.impact_vis_hold <= 0
):
    parser.error("--impact-vis-hold must be positive and finite.")
if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive.")
if not 0 <= args_cli.camera_env < args_cli.num_envs:
    parser.error("--camera_env must be between 0 and --num_envs - 1.")
if args_cli.auto_command_test is not None and args_cli.teleop != "none":
    parser.error("--auto_command_test cannot be combined with --teleop.")
if args_cli.auto_command_loops < 1:
    parser.error("--auto_command_loops must be positive.")
if args_cli.auto_command_hold_s is not None and args_cli.auto_command_hold_s <= 0.0:
    parser.error("--auto_command_hold_s must be positive when provided.")
if args_cli.terrain_type is not None and args_cli.terrain_column is not None:
    parser.error("--terrain_type and --terrain_column cannot be combined.")
if (args_cli.terrain_type is not None or args_cli.terrain_column is not None) and args_cli.terrain_level is None:
    parser.error("--terrain_type/--terrain_column must be combined with --terrain_level.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import cusrl  # noqa: E402
from cusrl.template.player import PlayerHook  # noqa: E402
from t800_lab.utils.wrappers.cusrl import (  # noqa: E402
    CusrlT800Adapter,
    validate_amp_checkpoint_compatibility,
    verify_agent_checkpoint_restored,
)
from t800_lab.utils.wrappers.cusrl.impact import (  # noqa: E402
    ImpactCusrlAdapter,
    ImpactV2CusrlAdapter,
    StrictCheckpointAgentFactory,
    validate_impact_v2_checkpoint,
)
from t800_lab.utils.wrappers.cusrl.ankle import AnkleCombinedCusrlAdapter  # noqa: E402
from terrain_play import (  # noqa: E402
    apply_fixed_terrain_level,
    configure_fixed_terrain_level,
    configure_fixed_terrain_selection,
    inspect_terrain_level,
)


class RewardTermsDebugHook(PlayerHook):
    """Print selected reward terms for a single environment during play."""

    def __init__(self, env, reward_terms: list[str], env_index: int = 0, print_every: int = 1):
        self._env = env
        self._reward_terms = reward_terms
        self._env_index = env_index
        self._print_every = max(1, print_every)
        self._resolved_terms: list[tuple[str, str]] = []

    def init(self, player):
        super().init(player)
        if self._env_index < 0 or self._env_index >= self._env.num_envs:
            raise ValueError(f"debug_reward_env={self._env_index} is out of range for num_envs={self._env.num_envs}.")

        reward_manager = getattr(self._env, "reward_manager", None)
        if reward_manager is None or not hasattr(reward_manager, "get_active_iterable_terms"):
            print("[WARN] Reward debug requested, but this environment does not expose reward term buffers.")
            return

        self._resolved_terms = self._resolve_requested_terms()
        if not self._resolved_terms:
            available_terms = [name for name, _ in reward_manager.get_active_iterable_terms(env_idx=self._env_index)]
            print(f"[WARN] None of the requested reward terms were found: {', '.join(self._reward_terms)}")
            if available_terms:
                print(f"[INFO] Available reward terms for env {self._env_index}: {', '.join(available_terms)}")
            return

        resolved_terms_text = ", ".join(
            requested if requested == resolved else f"{requested}->{resolved}"
            for requested, resolved in self._resolved_terms
        )
        print(
            f"[INFO] Reward debug enabled for env {self._env_index}: "
            f"terms={resolved_terms_text}, every={self._print_every} step(s)."
        )

    def step(self, step: int, transition, metrics):
        if self._resolved_terms and step % self._print_every == 0:
            self._print_debug()

    def reset(self, indices):
        if self._resolved_terms and self._env_index in indices:
            self._print_debug()

    def _resolve_requested_terms(self) -> list[tuple[str, str]]:
        available_terms = [
            name for name, _ in self._env.reward_manager.get_active_iterable_terms(env_idx=self._env_index)
        ]
        resolved_terms = []
        missing_terms = []

        for requested_term in self._reward_terms:
            resolved_term = None
            if requested_term in available_terms:
                resolved_term = requested_term
            else:
                suffix_matches = [name for name in available_terms if name.split("-", 1)[-1] == requested_term]
                if len(suffix_matches) > 1:
                    print(f"[WARN] Multiple reward terms matched '{requested_term}'. Using '{suffix_matches[0]}'.")
                if suffix_matches:
                    resolved_term = suffix_matches[0]

            if resolved_term is None:
                missing_terms.append(requested_term)
                continue

            resolved_terms.append((requested_term, resolved_term))

        if missing_terms:
            print(f"[WARN] Requested reward terms not found: {', '.join(missing_terms)}")

        return resolved_terms

    def _format_total_step_reward(self) -> str:
        reward_buf = getattr(self._env, "reward_buf", None)
        if reward_buf is None:
            return "nan"
        if isinstance(reward_buf, dict):
            group_rewards = [
                f"{group_name}={float(group_value[self._env_index].item()):.6f}"
                for group_name, group_value in reward_buf.items()
            ]
            return "{" + ", ".join(group_rewards) + "}"
        return f"{float(reward_buf[self._env_index].item()):.6f}"

    def _print_debug(self):
        reward_terms = {
            name: float(values[0])
            for name, values in self._env.reward_manager.get_active_iterable_terms(env_idx=self._env_index)
        }
        timestep = self.player.timestep
        if timestep is None:
            timestep = getattr(self._env, "step_dt", None)
        sim_time_s = self.player.step * timestep if timestep is not None else float("nan")
        reward_text = ", ".join(
            f"{requested}={reward_terms.get(resolved, float('nan')):.6f}"
            for requested, resolved in self._resolved_terms
        )
        print(
            "[INFO] Reward debug: "
            f"step={self.player.step}, time={sim_time_s:.3f}s, env={self._env_index}, "
            f"total_step_reward={self._format_total_step_reward()}, {reward_text}"
        )


def _configure_play_viewer(env_cfg, free_camera: bool, camera_env: int = 0) -> None:
    if not hasattr(env_cfg, "viewer"):
        return

    env_cfg.viewer.env_index = camera_env
    if not free_camera:
        return
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.asset_name = None
    env_cfg.viewer.body_name = None
    print("[INFO] Free camera enabled: viewer origin_type=world; viewport will not follow the robot.")


class InitialFreeCameraHook(PlayerHook):
    """Frame a robot once after the Player's startup reset, then release the camera."""

    def __init__(self, env, env_index: int):
        self._env = env
        self._env_index = env_index
        self._positioned = False

    def step(self, step: int, transition, metrics):
        if self._positioned:
            return
        # Player resets again when its loop starts. Read the live root after
        # that reset/first step, rather than using the world or terrain origin.
        root = self._env.scene["robot"].data.root_pos_w[self._env_index].detach().cpu().tolist()
        viewer = self._env.cfg.viewer
        eye = tuple(origin + offset for origin, offset in zip(root, viewer.eye))
        lookat = tuple(origin + offset for origin, offset in zip(root, viewer.lookat))
        self._env.viewport_camera_controller.update_view_location(eye=eye, lookat=lookat)
        self._positioned = True
        print(
            f"[INFO] Initial free camera: env={self._env_index}, eye={eye}, lookat={lookat}; "
            "manual movement remains free, including after episode resets."
        )


def main():
    cusrl.set_global_seed(args_cli.seed)
    trial = cusrl.Trial(os.path.abspath(args_cli.load_run))
    env_cfg = bootstrap.load_task_config(args_cli.task, "env_cfg_entry_point")
    agent_cfg = bootstrap.load_task_config(args_cli.task, "cusrl_cfg_entry_point")
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    agent_cfg.agent_factory.device = env_cfg.sim.device
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.push_robot = None
    if is_impact_task:
        env_cfg.impact.mode = args_cli.impact_mode or "common"
        if args_cli.head_mode is not None:
            env_cfg.impact.head_mode = args_cli.head_mode
    _configure_play_viewer(env_cfg, args_cli.free_camera, args_cli.camera_env)
    if args_cli.no_debug_vis:
        env_cfg.commands.base_velocity.debug_vis = False
    if args_cli.terrain_type is None and args_cli.terrain_column is None:
        level = configure_fixed_terrain_level(env_cfg, args_cli.terrain_level)
        selection = configure_fixed_terrain_selection(env_cfg, level)
    else:
        selection = configure_fixed_terrain_selection(
            env_cfg,
            args_cli.terrain_level,
            terrain_name=args_cli.terrain_type,
            terrain_column=args_cli.terrain_column,
        )

    with bootstrap.configure_usd_cache(env_cfg):
        env = gym.make(args_cli.task, cfg=env_cfg, disable_env_checker=True)
        teleop_bridge = None
        try:
            if selection.level is not None:
                if selection.column is None:
                    apply_fixed_terrain_level(env, selection.level)
                else:
                    apply_fixed_terrain_level(env, selection.level, selection.column)
                env.reset()
                terrain_state = inspect_terrain_level(env)
                if terrain_state.level != selection.level:
                    raise RuntimeError(
                        f"Fixed terrain level changed after reset: expected {selection.level}, got {terrain_state.level}."
                    )
                print(
                    "[INFO] Fixed Play terrain: "
                    f"level={terrain_state.level}/{terrain_state.num_levels - 1}, "
                    f"terrain_types={terrain_state.terrain_types}, "
                    f"requested_type={selection.terrain_name}, requested_column={selection.column}."
                )

            if args_cli.teleop != "none" and not args_cli.headless:
                from t800_lab.tasks.locomotion.teleop import attach_default_locomotion_teleop

                teleop_bridge = attach_default_locomotion_teleop(env.unwrapped, teleop_mode=args_cli.teleop)

            hooks = []
            if args_cli.free_camera and not args_cli.headless:
                hooks.append(InitialFreeCameraHook(env.unwrapped, args_cli.camera_env))
            show_impacts = is_impact_task and (
                args_cli.impact_vis if args_cli.impact_vis is not None else not args_cli.headless
            )
            if show_impacts:
                from impact_visualization import ImpactVisualizationHook

                hooks.append(ImpactVisualizationHook(
                    env.unwrapped, hold_s=0.3 if args_cli.impact_vis_hold is None else args_cli.impact_vis_hold,
                ))
            if args_cli.auto_command_test is not None:
                from auto_command_test import PracticalBoxingCommandHook

                hooks.append(
                    PracticalBoxingCommandHook(
                        env=env.unwrapped,
                        profile=args_cli.auto_command_test,
                        loops=args_cli.auto_command_loops,
                        hold_s=args_cli.auto_command_hold_s,
                    )
                )
            if args_cli.debug_reward_terms:
                hooks.append(
                    RewardTermsDebugHook(
                        env=env.unwrapped,
                        reward_terms=args_cli.debug_reward_terms,
                        env_index=args_cli.debug_reward_env,
                        print_every=args_cli.debug_reward_interval,
                    )
                )
            if args_cli.record_dir is not None:
                from play_diagnostics import T800AmpPlayDiagnosticsHook

                hooks.append(
                    T800AmpPlayDiagnosticsHook(
                        env=env.unwrapped,
                        output_directory=args_cli.record_dir,
                        task_name=args_cli.task,
                        checkpoint_path=str(trial.checkpoint_path),
                        seed=args_cli.seed,
                        env_indices=args_cli.record_envs,
                        completed_episode_limit=args_cli.record_episodes,
                    )
                )

            if is_combined_task:
                env = AnkleCombinedCusrlAdapter(env)
            elif is_dynamics_dr_task:
                from t800_lab.utils.wrappers.cusrl.dynamics_dr import DynamicsDRCusrlAdapter

                env = DynamicsDRCusrlAdapter(env)
            elif args_cli.task.removesuffix("-Play") == bootstrap.IMPACT_V2_TASK:
                env = ImpactV2CusrlAdapter(env)
            elif is_impact_v2_plus_task:
                from t800_lab.utils.wrappers.cusrl.impact import ImpactV2PlusCusrlAdapter

                env = ImpactV2PlusCusrlAdapter(env)
            else:
                env = (ImpactCusrlAdapter if is_impact_task else CusrlT800Adapter)(env)
            checkpoint = trial.load_checkpoint(map_location="cpu")
            if is_dynamics_dr_task:
                from t800_lab.utils.wrappers.cusrl.dynamics_dr import validate_dynamics_dr_checkpoint

                validate_dynamics_dr_checkpoint(checkpoint, env.unwrapped.cfg)
            elif args_cli.task.removesuffix("-Play") == bootstrap.IMPACT_V2_TASK:
                validate_impact_v2_checkpoint(checkpoint, env.unwrapped.cfg)
            elif is_impact_v2_plus_task:
                from t800_lab.utils.wrappers.cusrl.impact import validate_impact_v2_plus_checkpoint

                validate_impact_v2_plus_checkpoint(checkpoint, env.unwrapped.cfg)
            elif not is_impact_task:
                from t800_lab.utils.wrappers.cusrl.ankle import validate_ankle_play_checkpoint

                validate_ankle_play_checkpoint(checkpoint, args_cli.task, env.unwrapped.cfg)
            validate_amp_checkpoint_compatibility(
                checkpoint,
                env.amp_observation_dim,
                checkpoint_path=str(trial.checkpoint_path),
                task_name=args_cli.task,
            )
            print(f"[INFO] Loading checkpoint: {trial.checkpoint_path}; AMP dimension={env.amp_observation_dim}.")
            factory = (
                StrictCheckpointAgentFactory(agent_cfg.agent_factory, checkpoint["agent"])
                if is_impact_task or is_combined_task or is_combined_v2_task else agent_cfg.agent_factory
            )
            player = cusrl.Player(
                environment=env,
                agent=factory,
                checkpoint_path=trial,
                num_steps=int(1e10),
                timestep=env.spec.timestep,
                deterministic=True,
                verbose=True,
                hooks=hooks,
            )
            if is_impact_task or is_combined_task or is_combined_v2_task:
                verify_agent_checkpoint_restored(checkpoint["agent"], player.agent)
            del checkpoint, factory
            player.run_playing_loop()
        finally:
            if teleop_bridge is not None:
                teleop_bridge.close()
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
