"""Rewards used only by the controlled T800 ankle continuation variants."""

from __future__ import annotations

import torch


def support_joint_pose_l2(
    env,
    asset_cfg,
    sensor_cfg,
    threshold: float = 1.0,
    normalizer_rad: float = 0.1,
) -> torch.Tensor:
    """Penalize supported ankle posture away from each joint's default angle."""
    asset = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    error = (q - q0) / float(normalizer_rad)
    axis_weight = torch.tensor((1.0, 2.0, 1.0, 2.0), device=q.device, dtype=q.dtype)
    posture = error.square() * axis_weight.unsqueeze(0)
    history = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    # Keep each foot's support mask separate.  The four canonical joints are
    # pitch-L, roll-L, pitch-R, roll-R, so a swing foot must not gate the
    # supported foot's posture term (or vice versa).
    foot_contact = torch.linalg.vector_norm(history, dim=-1).amax(dim=1) > float(threshold)
    joint_contact = foot_contact.repeat_interleave(2, dim=1)
    return (posture * joint_contact.to(dtype=posture.dtype)).mean(dim=1)

