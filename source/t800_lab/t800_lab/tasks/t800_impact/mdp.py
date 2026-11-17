"""Recovery rewards and ground-only fall detection, usable in CPU tests."""

from __future__ import annotations

import math

import torch


GROUND_SENSOR_PREFIX = "impact_ground_"
FOOT_BODIES = ("LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R")


def _ground_sensors(env):
    for name, sensor in env.scene.sensors.items():
        if name.startswith(GROUND_SENSOR_PREFIX):
            yield name[len(GROUND_SENSOR_PREFIX):], sensor


def _ground_force(sensor):
    history = sensor.data.force_matrix_w_history
    if history is None:
        raise RuntimeError("Impact fall detection requires a ground-filtered contact sensor per body.")
    # [environment, history, single source body, ground filter, xyz].
    return torch.linalg.vector_norm(history, dim=-1).flatten(1).amax(dim=1)


def nonsole_ground_contacts(env, threshold: float = 1.0):
    contacts = torch.zeros(env.num_envs, device=env.device)
    for body, sensor in _ground_sensors(env):
        if body not in FOOT_BODIES:
            contacts += (_ground_force(sensor) > threshold).float()
    return contacts


def _quat_rotate_inverse(q, vector):
    xyz = q[..., 1:]
    cross = torch.cross(xyz, vector, dim=-1)
    return vector - 2 * q[..., :1] * cross + 2 * torch.cross(xyz, cross, dim=-1)


def foot_side_ground_contact(env, threshold: float = 1.0):
    """Catch inverted/side-loaded feet in addition to non-foot ground contact.

    IsaacLab reports the average filtered contact point, so this is an
    approximation for a box-shaped foot; it is not a per-triangle referee.
    """
    invalid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for body, sensor in _ground_sensors(env):
        if body not in FOOT_BODIES:
            continue
        # Pair current pose/point with current force, never the force history.
        current_force = sensor.data.force_matrix_w
        if current_force is None:
            raise RuntimeError("Foot contact requires a filtered force matrix.")
        contact = torch.linalg.vector_norm(current_force, dim=-1).flatten(1).amax(dim=1) > threshold
        quaternion = sensor.data.quat_w[:, 0]
        up = quaternion.new_zeros(env.num_envs, 3)
        up[:, 2] = 1.0
        local_up = _quat_rotate_inverse(quaternion, up)
        # A tilted lower corner is still a legal sole contact. A near-horizontal
        # sole normal or contact above the collider's midplane indicates a side.
        invalid |= contact & (local_up[:, 2] < math.cos(math.radians(70.0)))
        point = sensor.data.contact_pos_w
        if point is not None:
            delta = point[:, 0, 0] - sensor.data.pos_w[:, 0]
            local_point = _quat_rotate_inverse(quaternion, delta)
            invalid |= contact & torch.isfinite(local_point).all(dim=-1) & (local_point[:, 2] > -0.049)
    return invalid


def ground_fall(env, threshold: float = 1.0):
    invalid = (nonsole_ground_contacts(env, threshold) > 0) | foot_side_ground_contact(env, threshold)
    prior_substeps = getattr(env, "_impact_foot_ground_flags", None)
    return invalid if prior_substeps is None else invalid | prior_substeps


def recovery_scale(env):
    runtime = getattr(env, "impact_runtime", None)
    if runtime is None:
        return torch.ones(env.num_envs, device=env.device)
    scale = runtime.recovery_scale()
    # Preserve the pre-reset value for the corresponding transition's AMP term.
    runtime.last_step_style_scale = scale.clone()
    return scale


def recovery_scaled(env, term_func, term_params: dict, asset_cfg=None):
    """Relax soft tracking/posture costs only during measurable disturbance."""
    if asset_cfg is not None:
        term_params = {**term_params, "asset_cfg": asset_cfg}
    return term_func(env, **term_params) * recovery_scale(env)


def body_action_rate(env):
    # Fixed/failed head outputs must not influence the policy reward either.
    action = env.action_manager.action[:, :23]
    previous = env.action_manager.prev_action[:, :23]
    return (action - previous).square().sum(dim=-1) * recovery_scale(env)


def capture_recovery_scale(env):
    """A zero-weight-in-effect diagnostic term also snapshots clean transitions."""
    recovery_scale(env)
    return torch.zeros(env.num_envs, device=env.device)


def arena_margin(env, apothem: float = 4.0):
    """Root clearance from an 8 m regular octagon, for evaluation diagnostics."""
    position = env.scene["robot"].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    angles = torch.arange(8, device=position.device, dtype=position.dtype) * (math.pi / 4)
    normals = torch.stack((angles.cos(), angles.sin()), dim=-1)
    return apothem - (position @ normals.T).amax(dim=-1)
