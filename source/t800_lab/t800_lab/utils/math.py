import torch
import isaaclab.utils.math as math_utils


@torch.jit.script
def quat_slerp_batch(q1: torch.Tensor, q2: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Interpolate and sample across the quaternion (batch) based on the given t (batch).
    Args:
        q1, q2: The quaternion in (w, x, y, z). Shape is (B, 4)
        tau: The interpolation factor. Shape is (B,). Type is float where 0 <= t <= 1
    Return:
        The interpolated quaternion in (w, x, y, z). Shape is (B, 4)
    """
    # ensure the input is in the right shape
    assert q1.shape[-1] == 4, "The quaternion must be in (w, x, y, z) format."
    assert q2.shape[-1] == 4, "The quaternion must be in (w, x, y, z) format."
    assert tau.shape == q1.shape[:-1], "The batch size must be the same for all inputs."
    assert q1.shape[0] == q2.shape[0] == tau.shape[0], "The batch size must be the same for all inputs."
    assert (tau >= 0).all() and (tau <= 1).all(), "The interpolation factor must be in (0, 1) range."

    # if the dot product is negative, flip the quaternion
    dot_product = torch.sum(q1 * q2, dim=-1, keepdim=True).clip(-1, 1)  # shape (B, 1)
    q2 = torch.where(dot_product < 0, -q2, q2)
    dot_product = torch.where(dot_product < 0, -dot_product, dot_product)

    # calculate the angle between the two quaternions
    theta = torch.acos(dot_product)
    sin_theta = torch.sin(theta)
    q_too_similar = (dot_product > (1 - 1e-9)) | (torch.abs(theta) < (1e-9))

    # avoid division by zero
    sin_theta = torch.where(sin_theta == 0, torch.ones_like(sin_theta), sin_theta)

    # calculate the interpolation factor
    s1 = torch.sin((1 - tau).unsqueeze(-1) * theta) / sin_theta
    s2 = torch.sin(tau.unsqueeze(-1) * theta) / sin_theta

    # calculate the interpolated quaternion
    interpolated_quat = (s1 * q1 + s2 * q2) * (~q_too_similar) + q_too_similar * q1
    interpolated_quat = math_utils.normalize(interpolated_quat)

    return interpolated_quat


@torch.jit.script
def quat_angular_velocity(q_prev: torch.Tensor, q_next: torch.Tensor, dt: float) -> torch.Tensor:
    """Compute the angular velocity between two quaternions. (from q_prev to q_next in dt seconds)
    Args:
        q_prev, q_next: The quaternion in (w, x, y, z). Shape is (..., 4)
        dt: The time difference between the two quaternions.
    Returns:
        The angular velocity in the local frame. Shape is (..., 3)
    """
    # ensure the input is in the right shape
    assert q_prev.shape[-1] == 4, "The quaternion must be in (w, x, y, z) format."
    assert q_next.shape[-1] == 4, "The quaternion must be in (w, x, y, z) format."
    assert q_prev.shape == q_next.shape, "The shape of the two quaternions must be the same."
    assert dt > 0, "The time difference must be positive."

    # if the dot product is negative, flip the quaternion
    dot_product = torch.sum(q_prev * q_next, dim=-1, keepdim=True).clip(-1, 1)  # shape (..., 1)
    q_next = torch.where(dot_product < 0, -q_next, q_next)

    quat_diff = math_utils.quat_mul(q_next, math_utils.quat_conjugate(q_prev))  # q_next * q_prev^-1
    axis_angle_diff = math_utils.axis_angle_from_quat(quat_diff)
    angular_velocity = axis_angle_diff / dt

    return angular_velocity
