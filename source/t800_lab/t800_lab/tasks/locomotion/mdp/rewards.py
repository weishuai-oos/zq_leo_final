from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation


def feet_air_time_positive_biped_tuned(
    env,
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float = 0.03,
) -> torch.Tensor:
    """Variant of the biped air-time reward that still encourages stepping under small commands.

    This function keeps the original implementation untouched and only lowers the
    command gate for tuned locomotion tasks that should still step under small
    non-zero commands.
    #原版奖励函数中，只有当指令速度 > 0.1 m/s 时才会给予迈步奖励（否则为 0）
    这个“变体”修改目的是让机器人在接收到很小的指令（比如 0.05 m/s）时仍然愿意迈步，而不是原地不动
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def energy(env: ManagerBasedRLEnv, k: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize excessive energy usage based on torque and joint velocity."""

    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.relu(k * torch.square(asset.data.applied_torque) + asset.data.applied_torque * asset.data.joint_vel),
        dim=-1,
    )


def fly(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Detect near-zero contact to identify airborne states."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return (torch.sum(is_contact, dim=-1) < 0.5).float()


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize foot sliding while in contact with the ground."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)


def contact_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 450,
    max_reward: float = 400,
) -> torch.Tensor:
    """Penalize overly large contact forces on specified bodies."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    return reward.clamp(min=0, max=max_reward)


def stand_still_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize posture deviation when no command is given."""

    asset: Articulation = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(diff_angle), dim=1)
    cmd = env.command_manager.get_command(command_name).clone()
    cmd[:, 2] *= 10
    reward *= torch.norm(cmd, dim=1) < command_threshold
    return reward


def joint_pos_target_l1(
    env: ManagerBasedRLEnv,
    target_joint_pos: tuple[float, ...] | list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize_by_num_joints: bool = False,
) -> torch.Tensor:
    """Penalize absolute joint position error to an explicit target pose."""

    asset: Articulation = env.scene[asset_cfg.name]
    target = torch.as_tensor(target_joint_pos, dtype=asset.data.joint_pos.dtype, device=asset.data.joint_pos.device)
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_error = torch.sum(torch.abs(joint_pos - target.unsqueeze(0)), dim=1)
    if normalize_by_num_joints:
        joint_error = joint_error / joint_pos.shape[1]
    return joint_error
