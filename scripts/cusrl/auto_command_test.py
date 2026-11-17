"""Practical, non-teleop command sequences for T800 AMP Play checks.

The sequence is deliberately semantic: its translation components are written
in the boxing tactical frame (forward/left), then passed through the same
``transform_teleop_command`` path used by keyboard teleop.  This makes the
script useful for checking both locomotion quality and the train/Play frame
contract without depending on training-time random command sampling.
"""

from __future__ import annotations

import math
import types

import torch

from cusrl.template.player import PlayerHook


class PracticalCommandSegment:
    """One semantic command and the amount of time it is held."""

    __slots__ = ("name", "tactical_command", "duration_s", "mode")

    def __init__(
        self,
        name: str,
        tactical_command: tuple[float, float, float],
        duration_s: float,
        mode: int,
    ):
        self.name = name
        self.tactical_command = tactical_command
        self.duration_s = float(duration_s)
        self.mode = int(mode)

    def __repr__(self) -> str:
        return (
            "PracticalCommandSegment("
            f"name={self.name!r}, tactical_command={self.tactical_command!r}, "
            f"duration_s={self.duration_s!r}, mode={self.mode!r})"
        )


# These are use-oriented test values, not a copy of the training distribution:
# moderate translation keeps the policy observable, while mixed segments test
# the real boxing use case of moving and turning at the same time.
MODE_STAND = 0
MODE_CARDINAL = 1
MODE_TURN = 2
MODE_OTHER = 3
MODE_MIXED = 4

AUTO_COMMAND_TEST_PROFILES = ("boxing_practical", "boxing_stability_sweep")


def _apply_hold_override(
    segments: tuple[PracticalCommandSegment, ...], hold_s: float | None
) -> tuple[PracticalCommandSegment, ...]:
    if hold_s is None:
        return segments
    if hold_s <= 0.0:
        raise ValueError(f"hold_s must be positive when provided, got {hold_s}.")
    return tuple(
        PracticalCommandSegment(segment.name, segment.tactical_command, float(hold_s), segment.mode)
        for segment in segments
    )


def build_practical_boxing_schedule(hold_s: float | None = None) -> tuple[PracticalCommandSegment, ...]:
    """Return one practical boxing movement sequence.

    ``hold_s`` is an optional diagnostic override that applies to every
    segment.  Keeping the default durations explicit makes the resulting
    recording easy to interpret and avoids coupling this test to command
    resampling intervals used during training.
    """

    segments = (
        PracticalCommandSegment("stand", (0.0, 0.0, 0.0), 2.0, MODE_STAND),
        PracticalCommandSegment("forward", (0.80, 0.0, 0.0), 4.0, MODE_CARDINAL),
        PracticalCommandSegment("backward", (-0.80, 0.0, 0.0), 4.0, MODE_CARDINAL),
        PracticalCommandSegment("left", (0.0, 0.60, 0.0), 4.0, MODE_CARDINAL),
        PracticalCommandSegment("right", (0.0, -0.60, 0.0), 4.0, MODE_CARDINAL),
        PracticalCommandSegment("forward_left", (0.68, 0.45, 0.0), 4.0, MODE_OTHER),
        PracticalCommandSegment("forward_right", (0.68, -0.45, 0.0), 4.0, MODE_OTHER),
        PracticalCommandSegment("backward_left", (-0.68, 0.45, 0.0), 4.0, MODE_OTHER),
        PracticalCommandSegment("backward_right", (-0.68, -0.45, 0.0), 4.0, MODE_OTHER),
        PracticalCommandSegment("turn_left", (0.0, 0.0, 0.75), 3.0, MODE_TURN),
        PracticalCommandSegment("turn_right", (0.0, 0.0, -0.75), 3.0, MODE_TURN),
        PracticalCommandSegment("advance_turn_left", (0.65, 0.0, 0.55), 4.0, MODE_MIXED),
        PracticalCommandSegment("retreat_turn_right", (-0.65, 0.0, -0.55), 4.0, MODE_MIXED),
        PracticalCommandSegment("recover", (0.0, 0.0, 0.0), 2.0, MODE_STAND),
    )
    return _apply_hold_override(segments, hold_s)


def build_boxing_stability_sweep_schedule(
    hold_s: float | None = None,
) -> tuple[PracticalCommandSegment, ...]:
    """Build a long stress sequence for estimating the policy's stable command envelope.

    Each speed level contains abrupt sign changes, so the test exercises
    stopping and re-accelerating rather than only steady-state tracking.  The
    default sweep is intentionally longer than the practical smoke test:
    approximately 179 seconds at the configured segment durations.
    """

    segments: list[PracticalCommandSegment] = [
        PracticalCommandSegment("sweep_settle", (0.0, 0.0, 0.0), 3.0, MODE_STAND)
    ]

    translation_speeds = (0.60, 0.80, 1.00, 1.20, 1.40)
    for speed in translation_speeds:
        token = f"{speed:g}"
        segments.extend(
            (
                PracticalCommandSegment(f"cardinal_forward_{token}", (speed, 0.0, 0.0), 2.0, MODE_CARDINAL),
                PracticalCommandSegment(f"cardinal_backward_{token}", (-speed, 0.0, 0.0), 2.0, MODE_CARDINAL),
                PracticalCommandSegment(f"cardinal_left_{token}", (0.0, speed, 0.0), 2.0, MODE_CARDINAL),
                PracticalCommandSegment(f"cardinal_right_{token}", (0.0, -speed, 0.0), 2.0, MODE_CARDINAL),
            )
        )

    for speed in translation_speeds:
        component = speed / math.sqrt(2.0)
        token = f"{speed:g}"
        segments.extend(
            (
                PracticalCommandSegment(f"diagonal_forward_left_{token}", (component, component, 0.0), 2.0, MODE_OTHER),
                PracticalCommandSegment(
                    f"diagonal_backward_right_{token}", (-component, -component, 0.0), 2.0, MODE_OTHER
                ),
                PracticalCommandSegment(
                    f"diagonal_forward_right_{token}", (component, -component, 0.0), 2.0, MODE_OTHER
                ),
                PracticalCommandSegment(
                    f"diagonal_backward_left_{token}", (-component, component, 0.0), 2.0, MODE_OTHER
                ),
            )
        )

    # One-second reversals specifically probe braking and foot-placement delay.
    for speed in (0.80, 1.00, 1.20, 1.40):
        token = f"{speed:g}"
        rapid_commands = (
            ("forward", (speed, 0.0, 0.0)),
            ("backward", (-speed, 0.0, 0.0)),
            ("left", (0.0, speed, 0.0)),
            ("right", (0.0, -speed, 0.0)),
            ("forward", (speed, 0.0, 0.0)),
            ("backward", (-speed, 0.0, 0.0)),
            ("left", (0.0, speed, 0.0)),
            ("right", (0.0, -speed, 0.0)),
        )
        segments.extend(
            PracticalCommandSegment(f"rapid_{direction}_{token}_{index}", command, 1.0, MODE_CARDINAL)
            for index, (direction, command) in enumerate(rapid_commands)
        )

    for yaw_rate in (0.50, 0.75, 1.00, 1.25, 1.50, 1.80, 2.20):
        token = f"{yaw_rate:g}"
        segments.extend(
            (
                PracticalCommandSegment(f"turn_left_{token}", (0.0, 0.0, yaw_rate), 2.0, MODE_TURN),
                PracticalCommandSegment(f"turn_right_{token}", (0.0, 0.0, -yaw_rate), 2.0, MODE_TURN),
            )
        )

    mixed_levels = ((0.80, 0.50), (1.00, 0.75), (1.20, 1.00), (1.40, 1.25))
    for speed, yaw_rate in mixed_levels:
        token = f"{speed:g}_{yaw_rate:g}"
        segments.extend(
            (
                PracticalCommandSegment(f"advance_turn_left_{token}", (speed, 0.0, yaw_rate), 2.0, MODE_MIXED),
                PracticalCommandSegment(f"retreat_turn_right_{token}", (-speed, 0.0, -yaw_rate), 2.0, MODE_MIXED),
                PracticalCommandSegment(f"advance_turn_right_{token}", (speed, 0.0, -yaw_rate), 2.0, MODE_MIXED),
                PracticalCommandSegment(f"retreat_turn_left_{token}", (-speed, 0.0, yaw_rate), 2.0, MODE_MIXED),
            )
        )

    segments.append(PracticalCommandSegment("sweep_recover", (0.0, 0.0, 0.0), 4.0, MODE_STAND))
    return _apply_hold_override(tuple(segments), hold_s)


def build_command_schedule(profile: str, hold_s: float | None = None) -> tuple[PracticalCommandSegment, ...]:
    if profile == "boxing_practical":
        return build_practical_boxing_schedule(hold_s)
    if profile == "boxing_stability_sweep":
        return build_boxing_stability_sweep_schedule(hold_s)
    raise ValueError(f"Unknown automatic command profile: {profile!r}.")


def segment_step_counts(segments: tuple[PracticalCommandSegment, ...], timestep: float) -> tuple[int, ...]:
    """Convert segment durations into at least one simulation step each."""

    if timestep <= 0.0:
        raise ValueError(f"timestep must be positive, got {timestep}.")
    return tuple(max(1, int(round(segment.duration_s / timestep))) for segment in segments)


def command_segment_at_step(
    segments: tuple[PracticalCommandSegment, ...],
    step_counts: tuple[int, ...],
    step: int,
    loops: int = 1,
) -> tuple[int, PracticalCommandSegment] | None:
    """Resolve a zero-based sequence step, returning ``None`` after all loops."""

    if len(segments) != len(step_counts) or not segments:
        raise ValueError("segments and step_counts must be non-empty and have equal length.")
    if any(count <= 0 for count in step_counts):
        raise ValueError("step_counts must contain positive values.")
    if loops < 1:
        raise ValueError(f"loops must be positive, got {loops}.")
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}.")

    cycle_steps = sum(step_counts)
    if step >= cycle_steps * loops:
        return None

    cycle_step = step % cycle_steps
    cursor = 0
    for index, count in enumerate(step_counts):
        if cycle_step < cursor + count:
            return index, segments[index]
        cursor += count
    raise RuntimeError("Failed to resolve a command sequence step.")


def _scheduled_resample(term, env_ids):
    """Keep IsaacLab's command timer from replacing the scripted command."""

    hook: PracticalBoxingCommandHook = term._practical_command_test_hook
    if hasattr(term, "time_left"):
        if env_ids is None:
            ids = torch.arange(term.num_envs, dtype=torch.long, device=term.device)
        elif isinstance(env_ids, slice):
            ids = torch.arange(term.num_envs, dtype=torch.long, device=term.device)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=term.device)
        if ids.numel():
            term.time_left[ids] = 1.0e6
    hook._apply_current_command()


def _scheduled_update(term):
    hook: PracticalBoxingCommandHook = term._practical_command_test_hook
    hook._apply_current_command()


class PracticalBoxingCommandHook(PlayerHook):
    """Drive a T800 ``base_velocity`` command term through a practical sequence."""

    def __init__(self, *, env, profile: str = "boxing_practical", loops: int = 1, hold_s: float | None = None):
        self._env = env
        self._profile = profile
        self._loops = int(loops)
        self._hold_s = hold_s
        self._segments = build_command_schedule(profile, hold_s)
        self._step_counts: tuple[int, ...] = ()
        self._elapsed_steps = 0
        self._total_steps = 0
        self._term = None
        self._original_resample_command = None
        self._original_update_command = None
        self._last_segment_index = None
        self._closed = False

    def init(self, player):
        super().init(player)
        if self._loops < 1:
            raise ValueError(f"auto command loops must be positive, got {self._loops}.")

        try:
            self._term = self._env.command_manager.get_term("base_velocity")
        except (KeyError, ValueError) as error:
            raise ValueError("Automatic boxing command testing requires the 'base_velocity' term.") from error

        command_shape = getattr(self._term, "command", None)
        if command_shape is None or command_shape.shape[-1] != 3:
            raise ValueError("Automatic boxing command testing requires a 3D SE(2) base_velocity command.")

        timestep = player.timestep
        if timestep is None:
            timestep = getattr(self._env, "step_dt", None)
        if timestep is None:
            raise ValueError("Could not determine the simulation timestep for automatic command testing.")
        self._step_counts = segment_step_counts(self._segments, float(timestep))
        self._total_steps = sum(self._step_counts) * self._loops
        self._install_term_hooks()
        self._apply_current_command()

        print(
            "[INFO] Automatic practical boxing command test enabled: "
            f"profile={self._profile}, loops={self._loops}, duration={self._total_steps * float(timestep):.1f}s, "
            f"segments={len(self._segments)}, frame=tactical->T800 root"
        )
        print("[INFO] Command sequence: " + " -> ".join(segment.name for segment in self._segments))

    def step(self, step: int, transition, metrics):
        del transition, metrics
        # The current callback follows one environment step.  The next
        # command-term update therefore sees this incremented sequence index.
        self._elapsed_steps = min(self._total_steps, max(self._elapsed_steps, int(step) + 1))
        if self._elapsed_steps >= self._total_steps:
            print("[INFO] Automatic practical boxing command test finished.")
            self.player.interrupted = True

    def reset(self, indices):
        del indices

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._term is not None:
            self._term._resample_command = self._original_resample_command
            self._term._update_command = self._original_update_command
            if getattr(self._term, "_practical_command_test_hook", None) is self:
                del self._term._practical_command_test_hook

    def _install_term_hooks(self):
        if hasattr(self._term, "_teleop_bridge"):
            raise ValueError("Automatic command testing cannot be combined with an active teleop bridge.")
        self._original_resample_command = self._term._resample_command
        self._original_update_command = self._term._update_command
        self._term._practical_command_test_hook = self
        self._term._resample_command = types.MethodType(_scheduled_resample, self._term)
        self._term._update_command = types.MethodType(_scheduled_update, self._term)

    def _apply_current_command(self):
        resolved = command_segment_at_step(self._segments, self._step_counts, self._elapsed_steps, self._loops)
        if resolved is None:
            return
        segment_index, segment = resolved
        if segment_index != self._last_segment_index:
            self._last_segment_index = segment_index
            print(
                "[INFO] Auto command segment: "
                f"{segment.name} tactical={segment.tactical_command} "
                f"steps={self._step_counts[segment_index]}"
            )

        device = self._term.device
        tactical = torch.as_tensor(segment.tactical_command, dtype=torch.float32, device=device)
        tactical = tactical.view(1, 3).expand(self._term.num_envs, -1)
        transform = getattr(self._term, "transform_teleop_command", None)
        command = transform(tactical) if callable(transform) else tactical
        self._term.vel_command_b[:] = command
        if hasattr(self._term, "is_heading_env"):
            self._term.is_heading_env.fill_(False)
        if hasattr(self._term, "is_standing_env"):
            self._term.is_standing_env.fill_(segment.mode == MODE_STAND)
        if hasattr(self._term, "heading_target"):
            self._term.heading_target.zero_()
        if hasattr(self._term, "command_mode"):
            self._term.command_mode.fill_(segment.mode)
        if hasattr(self._term, "is_stop_signal_env"):
            self._term.is_stop_signal_env.fill_(False)
