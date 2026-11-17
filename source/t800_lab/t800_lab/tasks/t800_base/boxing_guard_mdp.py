"""T800 boxing-guard task-specific observations and rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def velocity_command_in_root_yaw_frame(
    root_quat_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
) -> torch.Tensor:
    """Return ``[vx, vy, wz]`` in the same frames used by T800 command tracking."""
    root_lin_vel_yaw = quat_apply_inverse(yaw_quat(root_quat_w), root_lin_vel_w)
    return torch.cat((root_lin_vel_yaw[:, :2], root_ang_vel_w[:, 2:3]), dim=-1)


def reference_velocity_command(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
) -> torch.Tensor:
    """Derive the command represented by the currently sampled reference frame."""
    motion_reference = env.scene[asset_cfg.name]
    reference_frame = motion_reference.reference_frame
    return velocity_command_in_root_yaw_frame(
        reference_frame.base_quat_w[:, 0],
        reference_frame.base_lin_vel_w[:, 0],
        reference_frame.base_ang_vel_w[:, 0],
    )


def track_lin_vel_xy_yaw_frame_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return unsaturated squared XY command-tracking error in the root-yaw frame."""
    asset = env.scene[asset_cfg.name]
    actual_command = velocity_command_in_root_yaw_frame(
        asset.data.root_quat_w,
        asset.data.root_lin_vel_w[:, :3],
        asset.data.root_ang_vel_w[:, :3],
    )
    command = env.command_manager.get_command(command_name)
    return torch.sum(torch.square(command[:, :2] - actual_command[:, :2]), dim=-1)


def track_ang_vel_z_world_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return unsaturated squared world-Z yaw-rate command-tracking error."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    return torch.square(command[:, 2] - asset.data.root_ang_vel_w[:, 2])
