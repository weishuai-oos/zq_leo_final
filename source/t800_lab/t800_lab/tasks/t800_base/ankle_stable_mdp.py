"""Small, stateful ankle-stability reward terms for T800 continuation runs.

The terms intentionally use the articulation's resolved joint ids only after
resolving the four canonical joint names.  This keeps the reward independent
of simulator joint ordering and makes the target angle the post-scale target
actually sent to the articulation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


ANKLE_JOINT_NAMES = (
    "J04_ANKLE_PITCH_L",
    "J05_ANKLE_ROLL_L",
    "J10_ANKLE_PITCH_R",
    "J11_ANKLE_ROLL_R",
)
ANKLE_PITCH_INDICES = (0, 2)
ANKLE_ROLL_INDICES = (1, 3)

# Physical normalizers, not hardware limits.  The overspeed thresholds are
# experiment settings for the reward only (pitch/roll: 8/5 rad/s).
TARGET_VELOCITY_SCALE_RAD_S = 4.0
TARGET_ACCELERATION_SCALE_RAD_S2 = 80.0
ACTUAL_ACCELERATION_SCALE_RAD_S2 = 80.0
PITCH_OVERSPEED_THRESHOLD_RAD_S = 8.0
ROLL_OVERSPEED_THRESHOLD_RAD_S = 5.0


def _as_env_ids(env_ids: Sequence[int] | torch.Tensor | None, count: int, device: torch.device):
    if env_ids is None:
        return torch.arange(count, device=device)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long).reshape(-1)


def _step_token(env: Any):
    counter = getattr(env, "common_step_counter", None)
    if counter is None:
        counter = getattr(env, "episode_length_buf", 0)
    return counter.detach().clone() if torch.is_tensor(counter) else counter


def _safe_square_mean(values: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Return a finite penalty; non-finite physics becomes a large penalty."""
    normalized = values / scale
    # Keep a finite, clearly non-zero fault penalty without creating a device
    # scalar or allowing a 1e38 reward to destabilize PPO value regression.
    normalized = torch.nan_to_num(normalized, nan=1000.0, posinf=1000.0, neginf=-1000.0)
    normalized = normalized.clamp(min=-1000.0, max=1000.0)
    return normalized.square().mean(dim=1)


def _joint_ids(asset: Any) -> torch.Tensor:
    """Resolve ankle ids in canonical order, never by action/sim order."""
    if hasattr(asset, "find_joints"):
        ids, _ = asset.find_joints(list(ANKLE_JOINT_NAMES), preserve_order=True)
        ids = torch.as_tensor(ids, device=asset.data.joint_pos.device, dtype=torch.long)
    else:
        names = list(getattr(asset, "joint_names", ()))
        missing = [name for name in ANKLE_JOINT_NAMES if name not in names]
        if missing:
            raise ValueError(f"Missing ankle joints: {missing}")
        ids = torch.tensor(
            [names.index(name) for name in ANKLE_JOINT_NAMES],
            device=asset.data.joint_pos.device,
            dtype=torch.long,
        )
    if ids.numel() != len(ANKLE_JOINT_NAMES):
        raise ValueError(f"Expected four ankle joints, got ids {ids.tolist()}")
    return ids


def _state(env: Any, asset: Any, dt: float) -> dict[str, torch.Tensor]:
    """Get one per-step snapshot; all ankle terms share this update."""
    state = getattr(env, "_ankle_stable_state", None)
    token = _step_token(env)
    if state is not None and state["token"] is not None:
        old_token = state["token"]
        same_step = bool(torch.equal(token, old_token) if torch.is_tensor(token) else token == old_token)
        if same_step:
            return state["snapshot"]

    if state is not None and state.get("asset_id") == id(asset):
        ids = state["joint_ids"]
    else:
        ids = _joint_ids(asset)
    q = asset.data.joint_pos.index_select(1, ids)
    qdes = asset.data.joint_pos_target.index_select(1, ids)
    vel = asset.data.joint_vel.index_select(1, ids)
    count = q.shape[0]
    if state is None or state["qdes"].shape != qdes.shape:
        zeros = torch.zeros_like(qdes)
        state = {
            "qdes": zeros.clone(),
            "qdes_prev": zeros.clone(),
            "q": zeros.clone(),
            "vel": zeros.clone(),
            "valid_1": torch.zeros(count, device=q.device, dtype=torch.bool),
            "valid_2": torch.zeros(count, device=q.device, dtype=torch.bool),
            "progress": torch.full((count,), -1, device=q.device, dtype=torch.long),
            "asset_id": id(asset),
            "joint_ids": ids,
            "token": None,
            "snapshot": {},
        }
        env._ankle_stable_state = state

    episode_buf = getattr(env, "episode_length_buf", None)
    if episode_buf is not None:
        episode_buf = torch.as_tensor(episode_buf, device=q.device)
        reset_mask = (state["progress"] >= 0) & (episode_buf < state["progress"])
        state["valid_1"] &= ~reset_mask
        state["valid_2"] &= ~reset_mask

    valid_1 = state["valid_1"]
    valid_2 = state["valid_2"]
    target_rate = (qdes - state["qdes"]) / max(float(dt), torch.finfo(qdes.dtype).eps)
    target_acc = (qdes - 2.0 * state["qdes"] + state["qdes_prev"]) / max(float(dt) ** 2, torch.finfo(qdes.dtype).eps)
    actual_acc = (vel - state["vel"]) / max(float(dt), torch.finfo(vel.dtype).eps)
    rate = torch.where(valid_1[:, None], target_rate, torch.zeros_like(target_rate))
    target_acc = torch.where((valid_1 & valid_2)[:, None], target_acc, torch.zeros_like(target_acc))
    actual_acc = torch.where(valid_1[:, None], actual_acc, torch.zeros_like(actual_acc))

    limits = asset.data.soft_joint_pos_limits.index_select(1, ids)
    span = (limits[..., 1] - limits[..., 0]).clamp_min(torch.finfo(q.dtype).eps)
    target_limit = torch.relu(limits[..., 0] - qdes) + torch.relu(qdes - limits[..., 1])
    actual_limit = torch.relu(limits[..., 0] - q) + torch.relu(q - limits[..., 1])
    target_limit = target_limit / span
    actual_limit = actual_limit / span
    thresholds = torch.tensor(
        [PITCH_OVERSPEED_THRESHOLD_RAD_S, ROLL_OVERSPEED_THRESHOLD_RAD_S] * 2,
        device=vel.device,
        dtype=vel.dtype,
    )
    overspeed = torch.relu(torch.abs(vel) - thresholds) / thresholds

    state["snapshot"] = {
        "target_rate": _safe_square_mean(rate, TARGET_VELOCITY_SCALE_RAD_S),
        "target_acc": _safe_square_mean(target_acc, TARGET_ACCELERATION_SCALE_RAD_S2),
        "overspeed": _safe_square_mean(overspeed),
        "actual_acc": _safe_square_mean(actual_acc, ACTUAL_ACCELERATION_SCALE_RAD_S2),
        "target_limit": _safe_square_mean(target_limit),
        "actual_limit": _safe_square_mean(actual_limit),
    }
    state["qdes_prev"][:] = state["qdes"]
    state["qdes"][:] = qdes
    state["q"][:] = q
    state["vel"][:] = vel
    state["valid_2"][:] = state["valid_1"]
    state["valid_1"][:] = True
    if episode_buf is not None:
        state["progress"][:] = episode_buf.to(dtype=state["progress"].dtype)
    state["token"] = token
    return state["snapshot"]


def reset_ankle_stable_history(env: Any, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
    """Clear history for all or a subset of environments."""
    state = getattr(env, "_ankle_stable_state", None)
    if state is None:
        return
    ids = _as_env_ids(env_ids, state["valid_1"].shape[0], state["valid_1"].device)
    state["valid_1"][ids] = False
    state["valid_2"][ids] = False
    state["progress"][ids] = -1
    state["snapshot"] = {}
    state["token"] = None


def _term(env: Any, key: str, asset_cfg: Any = None, dt: float = 0.02) -> torch.Tensor:
    asset_name = getattr(asset_cfg, "name", "robot")
    return _state(env, env.scene[asset_name], dt)[key]


def ankle_target_rate_l2(env: Any, asset_cfg: Any = None, dt: float = 0.02) -> torch.Tensor:
    return _term(env, "target_rate", asset_cfg, dt)


def ankle_target_acceleration_l2(env: Any, asset_cfg: Any = None, dt: float = 0.02) -> torch.Tensor:
    return _term(env, "target_acc", asset_cfg, dt)


def ankle_overspeed_hinge_l2(env: Any, asset_cfg: Any = None, dt: float = 0.02) -> torch.Tensor:
    return _term(env, "overspeed", asset_cfg, dt)


def ankle_acceleration_l2(env: Any, asset_cfg: Any = None, dt: float = 0.02) -> torch.Tensor:
    return _term(env, "actual_acc", asset_cfg, dt)


def ankle_target_soft_limit_hinge_l2(env: Any, asset_cfg: Any = None, dt: float = 0.02) -> torch.Tensor:
    return _term(env, "target_limit", asset_cfg, dt)


def ankle_actual_soft_limit_hinge_l2(env: Any, asset_cfg: Any = None, dt: float = 0.02) -> torch.Tensor:
    return _term(env, "actual_limit", asset_cfg, dt)
