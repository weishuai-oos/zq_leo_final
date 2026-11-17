from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass

from .command_frames import rotate_planar_velocity

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class T800BoxingGuardVelocityCommand(UniformVelocityCommand):
    """Sample boxing locomotion commands in the tactical-forward frame.

    The existing T800 locomotion rewards consume ``[vx, vy, wz]`` in the
    gravity-aligned root-yaw frame. This term samples semantic directions in a
    boxing tactical frame and rotates them into that existing frame, keeping
    the command observation and reward interfaces unchanged.
    """

    cfg: "T800BoxingGuardVelocityCommandCfg"

    MODE_STAND = 0
    MODE_CARDINAL = 1
    MODE_TURN = 2
    MODE_OTHER = 3
    MODE_MIXED = 4

    def __init__(self, cfg: "T800BoxingGuardVelocityCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._validate_cfg()

        self.command_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._mode_weights = torch.as_tensor(cfg.mode_weights, dtype=torch.float, device=self.device)
        self._mode_weights /= self._mode_weights.sum()

        tactical_front_offset = math.radians(cfg.tactical_front_offset_deg)
        self._tactical_front_cos = math.cos(tactical_front_offset)
        self._tactical_front_sin = math.sin(tactical_front_offset)

    def __str__(self) -> str:
        return (
            "T800BoxingGuardVelocityCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}\n"
            f"\tMode weights: {self.cfg.mode_weights}\n"
            f"\tTactical-front offset: {self.cfg.tactical_front_offset_deg:.1f} deg"
        )

    def _validate_cfg(self) -> None:
        if len(self.cfg.mode_weights) != 5:
            raise ValueError("mode_weights must contain stand, cardinal, turn, other, and mixed weights.")
        if any(weight < 0.0 for weight in self.cfg.mode_weights) or sum(self.cfg.mode_weights) <= 0.0:
            raise ValueError("mode_weights must be non-negative and contain at least one positive weight.")

        for name in (
            "cardinal_speed_range",
            "other_speed_range",
            "turn_rate_range",
            "mixed_speed_range",
            "mixed_turn_rate_range",
        ):
            lower, upper = getattr(self.cfg, name)
            if lower < 0.0 or upper < lower:
                raise ValueError(f"{name} must be a non-negative, ascending range; got {(lower, upper)}.")

        jitter_lower, jitter_upper = self.cfg.other_direction_jitter_deg
        if jitter_upper < jitter_lower:
            raise ValueError("other_direction_jitter_deg must be an ascending range.")
        if self.cfg.heading_command:
            raise ValueError("T800 boxing commands use direct yaw-rate commands; heading_command must be False.")

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        num_samples = env_ids.numel()
        sampled_modes = torch.multinomial(self._mode_weights, num_samples, replacement=True)
        self.command_mode[env_ids] = sampled_modes
        self.is_standing_env[env_ids] = sampled_modes == self.MODE_STAND
        self.is_heading_env[env_ids] = False
        self.heading_target[env_ids] = 0.0

        tactical_velocity = torch.zeros(num_samples, 2, device=self.device)
        yaw_rate = torch.zeros(num_samples, device=self.device)

        cardinal_mask = sampled_modes == self.MODE_CARDINAL
        cardinal_count = int(cardinal_mask.sum().item())
        if cardinal_count:
            cardinal_angles = torch.tensor(
                (0.0, 0.5 * math.pi, math.pi, -0.5 * math.pi),
                dtype=torch.float,
                device=self.device,
            )
            direction_ids = torch.randint(0, cardinal_angles.numel(), (cardinal_count,), device=self.device)
            speeds = self._sample_uniform(cardinal_count, self.cfg.cardinal_speed_range)
            tactical_velocity[cardinal_mask] = self._polar_to_xy(speeds, cardinal_angles[direction_ids])

        turn_mask = sampled_modes == self.MODE_TURN
        turn_count = int(turn_mask.sum().item())
        if turn_count:
            yaw_rate[turn_mask] = self._sample_signed_magnitude(turn_count, self.cfg.turn_rate_range)

        other_mask = sampled_modes == self.MODE_OTHER
        other_count = int(other_mask.sum().item())
        if other_count:
            diagonal_angles = torch.tensor(
                (0.25 * math.pi, 0.75 * math.pi, -0.75 * math.pi, -0.25 * math.pi),
                dtype=torch.float,
                device=self.device,
            )
            direction_ids = torch.randint(0, diagonal_angles.numel(), (other_count,), device=self.device)
            jitter = torch.deg2rad(self._sample_uniform(other_count, self.cfg.other_direction_jitter_deg))
            speeds = self._sample_uniform(other_count, self.cfg.other_speed_range)
            tactical_velocity[other_mask] = self._polar_to_xy(speeds, diagonal_angles[direction_ids] + jitter)

        mixed_mask = sampled_modes == self.MODE_MIXED
        mixed_count = int(mixed_mask.sum().item())
        if mixed_count:
            directions = self._sample_uniform(mixed_count, (-math.pi, math.pi))
            speeds = self._sample_uniform(mixed_count, self.cfg.mixed_speed_range)
            tactical_velocity[mixed_mask] = self._polar_to_xy(speeds, directions)
            yaw_rate[mixed_mask] = self._sample_signed_magnitude(mixed_count, self.cfg.mixed_turn_rate_range)

        root_velocity = rotate_planar_velocity(
            tactical_velocity,
            cosine=self._tactical_front_cos,
            sine=self._tactical_front_sin,
        )

        self.vel_command_b[env_ids, :2] = root_velocity
        self.vel_command_b[env_ids, 2] = yaw_rate

    def transform_teleop_command(self, command: torch.Tensor) -> torch.Tensor:
        """Map a tactical-frame teleop command to the frame used during training."""
        if command.shape[-1] != 3:
            raise ValueError(f"Expected an SE(2) teleop command with final dimension 3, got {tuple(command.shape)}.")

        transformed = command.clone()
        transformed[..., :2] = rotate_planar_velocity(
            command[..., :2],
            cosine=self._tactical_front_cos,
            sine=self._tactical_front_sin,
        )
        return transformed

    def _sample_uniform(self, count: int, value_range: tuple[float, float]) -> torch.Tensor:
        return torch.empty(count, device=self.device).uniform_(*value_range)

    def _sample_signed_magnitude(self, count: int, value_range: tuple[float, float]) -> torch.Tensor:
        magnitudes = self._sample_uniform(count, value_range)
        signs = torch.where(
            torch.rand(count, device=self.device) < 0.5,
            -torch.ones(count, device=self.device),
            torch.ones(count, device=self.device),
        )
        return magnitudes * signs

    @staticmethod
    def _polar_to_xy(magnitudes: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        return torch.stack((magnitudes * torch.cos(angles), magnitudes * torch.sin(angles)), dim=-1)


@configclass
class T800BoxingGuardVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for tactical-frame boxing-guard velocity commands."""

    class_type: type = T800BoxingGuardVelocityCommand

    # Order: stand, cardinal translation, in-place turn, other translation, mixed.
    mode_weights: tuple[float, float, float, float, float] = (
        0.10,
        0.50,
        0.20,
        0.15,
        0.05,
    )
    tactical_front_offset_deg: float = 37.0
    cardinal_speed_range: tuple[float, float] = (0.65, 1.10)
    other_speed_range: tuple[float, float] = (0.80, 1.35)
    turn_rate_range: tuple[float, float] = (1.70, 2.65)
    mixed_speed_range: tuple[float, float] = (0.80, 1.25)
    mixed_turn_rate_range: tuple[float, float] = (0.50, 1.80)
    other_direction_jitter_deg: tuple[float, float] = (-15.0, 15.0)
