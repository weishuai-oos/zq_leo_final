from __future__ import annotations
from typing import Literal
import numpy as np
import torch
from t800_lab.utils.math import quat_angular_velocity, quat_slerp_batch


def motion_interpolate_bilinear(
    root_trans: torch.Tensor,
    root_quat: torch.Tensor,
    joint_pos: torch.Tensor,
    source_framerate: float,
    target_framerate: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The bilinear interpolation function for the motion data. Interface referring to `.motion_reference_cfg.MotionReferenceManagerCfg.motion_interpolate_func`."""

    # shape[0] - 1 because the time is measured by the frame interval. The 0-th frame does not count as time.
    time_max = (root_trans.shape[0] - 1) / source_framerate
    # NOTE: don't delete the frame_idx clipping, because the frame_idx might be greater than root_trans.shape[0] - 1
    num_frames_ = np.ceil(time_max * target_framerate)
    frame_idx = (
        torch.arange(num_frames_).to(dtype=torch.float, device=root_trans.device) / target_framerate * source_framerate
    )
    # Must be no greater than the last frame index, NOTE: assuming ascending order of the frame_idx
    frame_idx = frame_idx[frame_idx <= (root_trans.shape[0] - 1)]

    front_frame_idx = frame_idx.floor().long()
    back_frame_idx = frame_idx.ceil().long()
    ratio = frame_idx - front_frame_idx.float()
    _root_trans = (
        ratio.unsqueeze(-1) * root_trans[back_frame_idx] + (1 - ratio).unsqueeze(-1) * root_trans[front_frame_idx]
    )
    _root_quat = quat_slerp_batch(
        root_quat[front_frame_idx],
        root_quat[back_frame_idx],
        ratio,
    )
    _joint_pos = (
        ratio.unsqueeze(-1) * joint_pos[back_frame_idx] + (1 - ratio).unsqueeze(-1) * joint_pos[front_frame_idx]
    )

    return _root_trans, _root_quat, _joint_pos


def estimate_velocity(
    positions: torch.Tensor,  # (batch_size, num_frames, data_dim)
    dt: float,
    estimation_type: Literal["frontward", "backward", "frontbackward"] = "frontbackward",
):
    """Estimate the linear / joint velocity of the robot by using the position difference between two frames.
    Args:
        positions: the position data to be estimated, shape (batch_size, num_frames, data_dim)
        dt: the time interval between two frames
        estimation_type: the type of estimation, can be "frontward", "backward" or "frontbackward"
            - "frontward": estimate the velocity by using the position difference between the current frame and the next frame
            - "backward": estimate the velocity by using the position difference between the current frame and the previous frame
            - "frontbackward": estimate the velocity by using the position difference between the next frame and the previous frame
    Returns:
        velocity: the estimated velocity, shape (batch_size, num_frames, data_dim)
    """
    assert len(positions.shape) == 3, "The position data must be in shape (batch_size, num_frames, data_dim)."
    if estimation_type == "frontward":
        next_frame = torch.roll(positions, -1, dims=1)
        next_frame[:, -1] = positions[:, -1]
        prev_frame = positions
        velocity = (next_frame - prev_frame) / dt
    elif estimation_type == "backward":
        prev_frame = torch.roll(positions, 1, dims=1)
        prev_frame[:, 0] = positions[:, 0]
        next_frame = positions
        velocity = (next_frame - prev_frame) / dt
    elif estimation_type == "frontbackward":
        prev_frame = torch.roll(positions, 1, dims=1)
        prev_frame[:, 0] = positions[:, 0]
        next_frame = torch.roll(positions, -1, dims=1)
        next_frame[:, -1] = positions[:, -1]
        velocity = (next_frame - prev_frame) / (2 * dt)
    else:
        raise ValueError(f"Unknown estimation type: {estimation_type}.")

    return velocity


def estimate_angular_velocity(
    quaternions: torch.Tensor,  # (batch_size, num_frames, 4)
    dt: float,
    estimation_type: Literal["frontward", "backward", "frontbackward"] = "frontbackward",
):
    """Estimate the angular velocity of the robot by using the quaternion difference between two frames.
    Args:
        quaternions: the quaternion data to be estimated, shape (batch_size, num_frames, 4)
        dt: the time interval between two frames
        estimation_type: the type of estimation, can be "frontward", "backward" or "frontbackward"
            - "frontward": estimate the angular velocity by using the quaternion difference between the current frame and the next frame
            - "backward": estimate the angular velocity by using the quaternion difference between the current frame and the previous frame
            - "frontbackward": estimate the angular velocity by using the quaternion difference between the next frame and the previous frame
    Returns:
        angular_velocity: the estimated angular velocity, shape (batch_size, num_frames, 3)
    """
    assert quaternions.shape[-1] == 4, "The last dimension of the quaternion data must be 4."
    assert len(quaternions.shape) == 3, "The quaternion data must be in shape (batch_size, num_frames, 4)."
    if estimation_type == "frontward":
        next_frame = torch.roll(quaternions, -1, dims=1)
        next_frame[:, -1] = quaternions[:, -1]
        prev_frame = quaternions
        angular_velocity = quat_angular_velocity(
            prev_frame.reshape(-1, 4),
            next_frame.reshape(-1, 4),
            dt,
        ).reshape(-1, quaternions.shape[1], 3)
    elif estimation_type == "backward":
        prev_frame = torch.roll(quaternions, 1, dims=1)
        prev_frame[:, 0] = quaternions[:, 0]
        next_frame = quaternions
        angular_velocity = quat_angular_velocity(
            prev_frame.reshape(-1, 4),
            next_frame.reshape(-1, 4),
            dt,
        ).reshape(-1, quaternions.shape[1], 3)
    elif estimation_type == "frontbackward":
        prev_frame = torch.roll(quaternions, 1, dims=1)
        prev_frame[:, 0] = quaternions[:, 0]
        next_frame = torch.roll(quaternions, -1, dims=1)
        next_frame[:, -1] = quaternions[:, -1]
        angular_velocity = quat_angular_velocity(
            prev_frame.reshape(-1, 4),
            next_frame.reshape(-1, 4),
            2 * dt,
        ).reshape(-1, quaternions.shape[1], 3)
    else:
        raise ValueError(f"Unknown estimation type: {estimation_type}.")

    return angular_velocity
