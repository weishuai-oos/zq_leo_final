"""Opt-in diagnostics recorder for T800 AMP play sessions.

The Isaac Lab manager environment automatically resets completed environments
inside ``step``.  This recorder therefore intercepts ``_reset_idx`` and copies
the terminal physics state before reset.  It also captures the command before
each step because command resampling happens after reward computation.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cusrl.template.player import PlayerHook


# Version 2 adds the explicit ankle diagnostic schema.  Summarization remains
# field-tolerant so version-1 NPZ files continue to be readable.
FORMAT_VERSION = 2
T800_FOOT_BODY_NAMES = ("LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R")
T800_ANKLE_JOINT_NAMES = (
    "J04_ANKLE_PITCH_L",
    "J05_ANKLE_ROLL_L",
    "J10_ANKLE_PITCH_R",
    "J11_ANKLE_ROLL_R",
)
ANKLE_VELOCITY_REVERSAL_DEADBAND_RAD_S = 0.10

STUMBLE_TILT_THRESHOLD_RAD = math.radians(20.0)
STUMBLE_ROOT_HEIGHT_DROP_M = 0.08
STUMBLE_ANGULAR_VELOCITY_RAD_S = 2.5
FOOT_SLIP_SPEED_M_S = 0.20
FOOT_CONTACT_FORCE_N = 1.0
ACTION_SATURATION_ABS = 0.95

# Keep this list explicit so aggregate output remains stable as new per-frame
# diagnostics are added to episode summaries.
_AGGREGATE_MEAN_METRICS = (
    "duration_s",
    "tracking_rmse_tactical_x_m_s",
    "tracking_rmse_tactical_y_m_s",
    "tracking_rmse_yaw_rad_s",
    "tracking_rmse_translation_m_s",
    "tilt_p95_deg",
    "tilt_max_deg",
    "root_height_max_drop_m",
    "stumble_events",
    "stumble_fraction",
    "foot_contact_force_peak_n",
    "nonfoot_contact_force_peak_n",
    "foot_slip_mean_m_s",
    "foot_slip_peak_m_s",
    "foot_slip_fraction",
    "action_saturation_fraction",
    "joint_limit_min_margin_rad",
    "amp_style_reward_mean",
    "arm_joint_mean_std_rad",
    "arm_joint_max_std_rad",
    "head_joint_mean_std_rad",
    "head_joint_max_std_rad",
    "ankle_diagnostic_sample_rate_hz",
    "ankle_left_support_actual_soft_limit_events",
    "ankle_right_support_actual_soft_limit_events",
    "ankle_left_swing_actual_soft_limit_events",
    "ankle_right_swing_actual_soft_limit_events",
    "ankle_left_support_actual_velocity_rms",
    "ankle_right_support_actual_velocity_rms",
    "ankle_left_swing_actual_velocity_rms",
    "ankle_right_swing_actual_velocity_rms",
    "ankle_left_support_target_rate_reversal_events_deadband",
    "ankle_right_support_target_rate_reversal_events_deadband",
    "ankle_all_target_soft_limit_events",
    "ankle_all_actual_soft_limit_events",
    "ankle_all_target_hard_limit_events",
    "ankle_all_actual_hard_limit_events",
    "ankle_all_actual_velocity_rms",
    "ankle_all_target_rate_reversal_events_deadband",
    "ankle_all_actual_soft_limit_fraction",
    "ankle_all_actual_soft_limit_max_violation_rad",
    "ankle_left_support_actual_soft_limit_fraction",
    "ankle_right_support_actual_soft_limit_fraction",
    "ankle_left_swing_actual_soft_limit_fraction",
    "ankle_right_swing_actual_soft_limit_fraction",
)


def _finite_percentile(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, percentile)) if values.size else float("nan")


def _event_stats(mask: np.ndarray, duration_weights: np.ndarray) -> tuple[int, float]:
    """Return contiguous event count and time, without treating frames as events."""
    mask = np.asarray(mask, dtype=bool)
    weights = np.asarray(duration_weights, dtype=np.float64)
    if mask.size == 0 or weights.size != mask.size:
        return 0, float("nan")
    return int(_true_run_lengths(mask).size), float(np.sum(weights[mask]))


def _sign_reversal_count(values: np.ndarray, *, deadband: float, eligible: np.ndarray | None = None) -> int:
    """Count adjacent, eligible non-deadband sign changes (no filtered-gap joins)."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    eligible = np.ones(values.size, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    count = 0
    previous_sign = 0
    for index, value in enumerate(values):
        if not eligible[index] or not np.isfinite(value):
            previous_sign = 0
            continue
        if abs(value) <= deadband:
            # A deadband sample is still inside this contiguous eligible phase;
            # it does not bridge an ineligible/NaN gap because those reset above.
            continue
        sign = 1 if value > 0.0 else -1
        if previous_sign and sign != previous_sign and index > 0 and eligible[index - 1]:
            count += 1
        previous_sign = sign
    return count


def _diagnostic_time_axis(
    arrays: Mapping[str, np.ndarray], timestep: float, num_frames: int
) -> tuple[np.ndarray, bool]:
    if "episode_time_s" in arrays:
        times = np.asarray(arrays["episode_time_s"], dtype=np.float64).reshape(-1)
    else:
        times = np.arange(1, num_frames + 1, dtype=np.float64) * float(timestep)
    if times.size != num_frames:
        # Keep downstream duration/phase aggregation shape-safe, but do not
        # allow the fallback clock to make derivatives appear valid.
        return np.arange(1, num_frames + 1, dtype=np.float64) * float(timestep), False
    if num_frames < 2:
        return times, True
    differences = np.diff(times)
    observed_dt = float(np.nanmedian(differences)) if np.all(np.isfinite(differences)) else float("nan")
    uniform = bool(
        np.all(np.isfinite(differences))
        and np.all(differences > 0.0)
        and np.isclose(observed_dt, float(timestep), rtol=1.0e-3, atol=1.0e-7)
        and np.allclose(differences, observed_dt, rtol=1.0e-4, atol=1.0e-7)
    )
    return times, uniform


def add_ankle_diagnostic_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    timestep: float,
    ankle_indices: Sequence[int] | None = None,
    ankle_joint_names: Sequence[str] | None = None,
    joint_names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Add four-ankle target/actual diagnostics to one already isolated episode.

    Derivatives are control-cycle (Play-step) derivatives, not physics-substep
    samples.  The first rate and first two accelerations remain NaN by design.
    """
    result = dict(arrays)
    required = ("joint_pos", "joint_pos_target", "joint_vel")
    if any(name not in arrays for name in required):
        result["ankle_diagnostics_available"] = np.bool_(False)
        return result
    joint_pos = np.asarray(arrays["joint_pos"], dtype=np.float64)
    target = np.asarray(arrays["joint_pos_target"], dtype=np.float64)
    joint_vel = np.asarray(arrays["joint_vel"], dtype=np.float64)
    if ankle_indices is None and ankle_joint_names is not None and joint_names is not None:
        joint_name_to_index = {str(name): index for index, name in enumerate(joint_names)}
        resolved = [joint_name_to_index.get(str(name)) for name in ankle_joint_names]
        if len(resolved) == 4 and all(index is not None for index in resolved):
            ankle_indices = tuple(int(index) for index in resolved)
    if ankle_indices is None:
        result["ankle_diagnostics_unavailable_reason"] = np.str_("explicit ankle name mapping missing")
        result["ankle_diagnostics_available"] = np.bool_(False)
        return result
    indices = np.asarray(tuple(ankle_indices), dtype=np.int64)
    if indices.size != 4 or np.any(indices < 0) or np.any(indices >= joint_pos.shape[-1]):
        result["ankle_diagnostics_available"] = np.bool_(False)
        return result
    actual = joint_pos[:, indices]
    desired = target[:, indices]
    velocity = joint_vel[:, indices]
    num_frames = actual.shape[0]
    times, uniform = _diagnostic_time_axis(arrays, timestep, num_frames)
    result["ankle_actual_joint_pos_rad"] = actual.astype(np.float32)
    result["ankle_target_joint_pos_rad"] = desired.astype(np.float32)
    result["ankle_tracking_error_rad"] = (desired - actual).astype(np.float32)
    result["ankle_actual_velocity_rad_s"] = velocity.astype(np.float32)
    result["ankle_target_rate_rad_s"] = np.full_like(desired, np.nan, dtype=np.float64).astype(np.float32)
    result["ankle_target_acceleration_rad_s2"] = np.full_like(desired, np.nan, dtype=np.float64).astype(np.float32)
    result["ankle_actual_acceleration_rad_s2"] = np.full_like(desired, np.nan, dtype=np.float64).astype(np.float32)
    if num_frames >= 2 and uniform:
        dt = np.diff(times)
        target_rate = np.full_like(desired, np.nan)
        actual_acceleration = np.full_like(velocity, np.nan)
        target_rate[1:] = np.diff(desired, axis=0) / dt[:, None]
        actual_acceleration[1:] = np.diff(velocity, axis=0) / dt[:, None]
        target_acceleration = np.full_like(desired, np.nan)
        if num_frames >= 3:
            target_acceleration[2:] = np.diff(target_rate, axis=0)[1:] / (((dt[1:] + dt[:-1]) * 0.5)[:, None])
        result["ankle_target_rate_rad_s"] = target_rate.astype(np.float32)
        result["ankle_target_acceleration_rad_s2"] = target_acceleration.astype(np.float32)
        result["ankle_actual_acceleration_rad_s2"] = actual_acceleration.astype(np.float32)
    result["ankle_diagnostic_timebase_uniform"] = np.bool_(uniform)
    result["ankle_diagnostics_available"] = np.bool_(uniform or num_frames < 2)
    for bound_name in (
        "ankle_soft_joint_pos_limits_rad",
        "ankle_hard_joint_pos_limits_rad",
        "ankle_velocity_limits_rad_s",
    ):
        if bound_name not in result:
            result[bound_name] = (
                np.full((num_frames, 4, 2), np.nan, dtype=np.float32)
                if "pos_limits" in bound_name
                else np.full((num_frames, 4), np.nan, dtype=np.float32)
            )
    soft = np.asarray(result["ankle_soft_joint_pos_limits_rad"], dtype=np.float64)
    hard = np.asarray(result["ankle_hard_joint_pos_limits_rad"], dtype=np.float64)
    result["ankle_target_soft_violation_rad"] = np.maximum.reduce(
        (soft[..., 0] - desired, desired - soft[..., 1], np.zeros_like(desired))
    ).astype(np.float32)
    result["ankle_actual_soft_violation_rad"] = np.maximum.reduce(
        (soft[..., 0] - actual, actual - soft[..., 1], np.zeros_like(actual))
    ).astype(np.float32)
    result["ankle_target_hard_violation_rad"] = np.maximum.reduce(
        (hard[..., 0] - desired, desired - hard[..., 1], np.zeros_like(desired))
    ).astype(np.float32)
    result["ankle_actual_hard_violation_rad"] = np.maximum.reduce(
        (hard[..., 0] - actual, actual - hard[..., 1], np.zeros_like(actual))
    ).astype(np.float32)
    return result


def _ankle_summary(
    arrays: Mapping[str, np.ndarray],
    *,
    metadata: Mapping[str, Any],
    timestep: float,
) -> dict[str, Any]:
    names = tuple(metadata.get("ankle_joint_names", T800_ANKLE_JOINT_NAMES))
    output: dict[str, Any] = {"ankle_diagnostics_available": False}
    required = ("ankle_actual_joint_pos_rad", "ankle_target_joint_pos_rad", "ankle_tracking_error_rad")
    if any(name not in arrays for name in required):
        output["ankle_diagnostics_missing_fields"] = [name for name in required if name not in arrays]
        return output
    actual = np.asarray(arrays["ankle_actual_joint_pos_rad"], dtype=np.float64)
    target = np.asarray(arrays["ankle_target_joint_pos_rad"], dtype=np.float64)
    error = np.asarray(arrays["ankle_tracking_error_rad"], dtype=np.float64)
    if actual.ndim != 2 or actual.shape[1] != 4:
        output["ankle_diagnostics_missing_fields"] = ["four_ankle_columns"]
        return output
    num_frames = actual.shape[0]
    times, uniform = _diagnostic_time_axis(arrays, timestep, num_frames)
    dt = np.diff(times, prepend=times[0])
    if num_frames > 1:
        dt[0] = float(np.nanmedian(np.diff(times))) if np.all(np.isfinite(np.diff(times))) else float(timestep)
    else:
        dt[0] = float(timestep)
    dt = np.where(np.isfinite(dt) & (dt > 0.0), dt, float(timestep))
    output["ankle_diagnostics_available"] = bool(uniform or num_frames < 2)
    output["ankle_diagnostic_timebase_uniform"] = bool(uniform)
    output["ankle_diagnostic_sample_rate_hz"] = float(1.0 / np.nanmedian(dt)) if dt.size else float("nan")
    output["ankle_target_rate_available"] = bool(
        uniform and np.isfinite(arrays.get("ankle_target_rate_rad_s", np.nan)).any()
    )
    output["ankle_target_acceleration_available"] = bool(
        uniform and np.isfinite(arrays.get("ankle_target_acceleration_rad_s2", np.nan)).any()
    )
    output["ankle_actual_acceleration_available"] = bool(
        uniform and np.isfinite(arrays.get("ankle_actual_acceleration_rad_s2", np.nan)).any()
    )

    soft = np.asarray(
        arrays.get("ankle_soft_joint_pos_limits_rad", np.full((num_frames, 4, 2), np.nan)), dtype=np.float64
    )
    hard = np.asarray(
        arrays.get("ankle_hard_joint_pos_limits_rad", np.full((num_frames, 4, 2), np.nan)), dtype=np.float64
    )
    target_soft = np.maximum.reduce((soft[..., 0] - target, target - soft[..., 1], np.zeros_like(target)))
    actual_soft = np.maximum.reduce((soft[..., 0] - actual, actual - soft[..., 1], np.zeros_like(actual)))
    target_hard = np.maximum.reduce((hard[..., 0] - target, target - hard[..., 1], np.zeros_like(target)))
    actual_hard = np.maximum.reduce((hard[..., 0] - actual, actual - hard[..., 1], np.zeros_like(actual)))
    soft_valid = np.isfinite(soft[..., 0]) & np.isfinite(soft[..., 1]) & (soft[..., 0] <= soft[..., 1])
    hard_valid = np.isfinite(hard[..., 0]) & np.isfinite(hard[..., 1]) & (hard[..., 0] <= hard[..., 1])
    output["ankle_limit_metadata_available"] = bool(soft_valid.any() or hard_valid.any())
    contact_value = arrays.get("foot_contact")
    contact_available = contact_value is not None and np.asarray(contact_value).shape == (num_frames, 2)
    contact = np.asarray(contact_value, dtype=bool) if contact_available else np.zeros((num_frames, 2), dtype=bool)
    ankle_rates = np.asarray(arrays.get("ankle_target_rate_rad_s", np.full((num_frames, 4), np.nan)), dtype=np.float64)
    ankle_acc = np.asarray(
        arrays.get("ankle_target_acceleration_rad_s2", np.full((num_frames, 4), np.nan)), dtype=np.float64
    )
    actual_acc = np.asarray(
        arrays.get("ankle_actual_acceleration_rad_s2", np.full((num_frames, 4), np.nan)), dtype=np.float64
    )

    def add_group(
        prefix: str,
        columns: Sequence[int],
        eligible: np.ndarray,
        *,
        phase_available: bool = True,
    ) -> None:
        eligible = np.asarray(eligible, dtype=bool)
        phase_observed = bool(phase_available and eligible.any())
        output[f"{prefix}_phase_available"] = bool(phase_available)
        output[f"{prefix}_phase_observed"] = phase_observed
        output[f"{prefix}_phase_observed_duration_s"] = float(np.sum(dt[eligible])) if phase_observed else float("nan")
        output[f"{prefix}_phase_observed_fraction"] = (
            float(np.sum(dt[eligible]) / np.sum(dt)) if phase_observed and np.sum(dt) > 0.0 else float("nan")
        )
        for label, values, valid_bounds in (
            ("target_soft", target_soft, soft_valid),
            ("actual_soft", actual_soft, soft_valid),
            ("target_hard", target_hard, hard_valid),
            ("actual_hard", actual_hard, hard_valid),
        ):
            selected = values[:, columns]
            valid = valid_bounds[:, columns] & np.isfinite(selected)
            observed = eligible & np.any(valid, axis=1)
            mask = eligible & np.any(valid & (selected > 0.0), axis=1)
            label_available = bool(phase_available and observed.any())
            output[f"{prefix}_{label}_limit_available"] = label_available
            if label_available:
                events, duration = _event_stats(mask, dt)
                observed_duration = float(np.sum(dt[observed]))
                output[f"{prefix}_{label}_limit_events"] = events
                output[f"{prefix}_{label}_limit_duration_s"] = duration
                output[f"{prefix}_{label}_limit_fraction"] = (
                    duration / observed_duration if observed_duration > 0.0 else float("nan")
                )
                masked_selected = np.where(valid, selected, np.nan)
                observed_values = masked_selected[eligible]
                output[f"{prefix}_{label}_limit_max_violation_rad"] = float(np.nanmax(observed_values))
            else:
                output[f"{prefix}_{label}_limit_events"] = None
                output[f"{prefix}_{label}_limit_duration_s"] = float("nan")
                output[f"{prefix}_{label}_limit_fraction"] = float("nan")
                output[f"{prefix}_{label}_limit_max_violation_rad"] = float("nan")
        metric_specs = (
            ("tracking_error", error, True),
            (
                "actual_velocity",
                np.asarray(arrays.get("ankle_actual_velocity_rad_s", np.full_like(error, np.nan))),
                True,
            ),
            ("target_rate", ankle_rates, output["ankle_target_rate_available"]),
            ("target_acceleration", ankle_acc, output["ankle_target_acceleration_available"]),
            ("actual_acceleration", actual_acc, output["ankle_actual_acceleration_available"]),
        )
        for metric_name, values, metric_available in metric_specs:
            selected = values[:, columns][eligible] if phase_available else np.empty((0, len(columns)))
            magnitude = np.abs(selected)
            available = bool(metric_available and phase_available and np.isfinite(magnitude).any())
            output[f"{prefix}_{metric_name}_available"] = available
            output[f"{prefix}_{metric_name}_rms"] = (
                float(np.sqrt(np.nanmean(np.square(magnitude)))) if available else float("nan")
            )
            output[f"{prefix}_{metric_name}_p99"] = _finite_percentile(magnitude, 99.0) if available else float("nan")
            output[f"{prefix}_{metric_name}_peak"] = float(np.nanmax(magnitude)) if available else float("nan")
        reversal_available = bool(
            output["ankle_target_rate_available"]
            and phase_available
            and eligible.any()
            and np.isfinite(ankle_rates[eligible][:, columns]).any()
        )
        if reversal_available:
            output[f"{prefix}_target_rate_reversal_events_deadband"] = sum(
                _sign_reversal_count(
                    ankle_rates[:, column], deadband=ANKLE_VELOCITY_REVERSAL_DEADBAND_RAD_S, eligible=eligible
                )
                for column in columns
            )
        else:
            output[f"{prefix}_target_rate_reversal_events_deadband"] = None

    for column, name in enumerate(names):
        add_group(f"ankle_{name}", (column,), np.ones(num_frames, dtype=bool))
    add_group("ankle_all", tuple(range(4)), np.ones(num_frames, dtype=bool))
    # Side contact is applied after full-episode derivatives were computed.
    for side, foot_index in (("left", 0), ("right", 1)):
        columns = tuple(index for index, item in enumerate(names) if item.endswith(f"_{side[0].upper()}"))
        add_group(f"ankle_{side}_support", columns, contact[:, foot_index], phase_available=contact_available)
        add_group(f"ankle_{side}_swing", columns, ~contact[:, foot_index], phase_available=contact_available)
    return output


def quaternion_wxyz_to_rpy(quaternion: np.ndarray) -> np.ndarray:
    """Convert a WXYZ quaternion to roll, pitch, yaw in radians."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {quaternion.shape}.")

    norm = np.linalg.norm(quaternion)
    if norm <= np.finfo(np.float64).eps:
        return np.full(3, np.nan, dtype=np.float32)
    w, x, y, z = quaternion / norm

    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_sine = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    pitch = math.asin(pitch_sine)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray((roll, pitch, yaw), dtype=np.float32)


def world_velocity_to_yaw_frame(linear_velocity_w: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a world-frame linear velocity into the gravity-aligned root-yaw frame."""
    linear_velocity_w = np.asarray(linear_velocity_w, dtype=np.float64)
    if linear_velocity_w.shape != (3,):
        raise ValueError(f"Expected linear velocity shape (3,), got {linear_velocity_w.shape}.")
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.asarray(
        (
            cosine * linear_velocity_w[0] + sine * linear_velocity_w[1],
            -sine * linear_velocity_w[0] + cosine * linear_velocity_w[1],
            linear_velocity_w[2],
        ),
        dtype=np.float32,
    )


def root_yaw_to_tactical_velocity(root_yaw_velocity: np.ndarray, front_offset_deg: float) -> np.ndarray:
    """Undo the tactical-front rotation used by the T800 boxing command term."""
    root_yaw_velocity = np.asarray(root_yaw_velocity, dtype=np.float64)
    if root_yaw_velocity.shape != (3,):
        raise ValueError(f"Expected command shape (3,), got {root_yaw_velocity.shape}.")
    angle = math.radians(front_offset_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        (
            cosine * root_yaw_velocity[0] + sine * root_yaw_velocity[1],
            -sine * root_yaw_velocity[0] + cosine * root_yaw_velocity[1],
            root_yaw_velocity[2],
        ),
        dtype=np.float32,
    )


def stack_episode_frames(frames: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Stack homogeneous frame dictionaries into arrays suitable for ``np.savez``."""
    if not frames:
        raise ValueError("Cannot stack an empty episode.")
    expected_keys = tuple(frames[0])
    for frame_index, frame in enumerate(frames[1:], start=1):
        if tuple(frame) != expected_keys:
            missing = sorted(set(expected_keys) - set(frame))
            extra = sorted(set(frame) - set(expected_keys))
            raise ValueError(f"Frame {frame_index} schema mismatch: missing={missing}, extra={extra}.")
    return {key: np.stack([np.asarray(frame[key]) for frame in frames], axis=0) for key in expected_keys}


def _true_run_lengths(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return np.empty(0, dtype=np.int64)
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return ends - starts


def _safe_metric_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def terrain_name_from_column(generator_cfg, terrain_type: int) -> str:
    """Resolve a curriculum column to its configured sub-terrain name."""
    sub_terrains = getattr(generator_cfg, "sub_terrains", {}) or {}
    num_cols = int(getattr(generator_cfg, "num_cols", 0) or 0)
    column = int(terrain_type)
    if not sub_terrains or num_cols <= 0 or column < 0 or column >= num_cols:
        return "generated"

    proportions = [float(cfg.proportion) for cfg in sub_terrains.values()]
    total = sum(proportions)
    if total <= 0.0:
        return "generated"

    threshold = column / num_cols + 0.001
    cumulative = 0.0
    for name, proportion in zip(sub_terrains, proportions, strict=True):
        cumulative += proportion / total
        if threshold < cumulative:
            return str(name)
    return "generated"


def summarize_episode(
    arrays: Mapping[str, np.ndarray],
    *,
    timestep: float,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute compact stability, tracking, contact, and posture diagnostics."""
    metadata = metadata or {}
    num_frames = int(np.asarray(arrays["root_pos_w"]).shape[0])
    if num_frames == 0:
        raise ValueError("Cannot summarize an empty episode.")

    arrays = add_ankle_diagnostic_arrays(
        arrays,
        timestep=timestep,
        ankle_indices=metadata.get("ankle_joint_indices"),
        ankle_joint_names=metadata.get("ankle_joint_names"),
        joint_names=metadata.get("joint_names"),
    )
    command = np.asarray(arrays["tracking_command_tactical"], dtype=np.float64)
    actual = np.asarray(arrays["actual_tactical_velocity"], dtype=np.float64)
    tracking_error = actual - command
    root_rpy = np.asarray(arrays["root_rpy"], dtype=np.float64)
    root_ang_vel = np.asarray(arrays["root_ang_vel_w"], dtype=np.float64)
    root_height = np.asarray(arrays["root_height_env"], dtype=np.float64).reshape(-1)

    baseline_frames = max(1, min(num_frames, int(round(0.5 / max(timestep, 1.0e-9)))))
    baseline_height = float(np.nanmedian(root_height[:baseline_frames]))
    height_drop = np.maximum(0.0, baseline_height - root_height)
    tilt = np.max(np.abs(root_rpy[:, :2]), axis=1)
    horizontal_ang_vel = np.linalg.norm(root_ang_vel[:, :2], axis=1)
    unstable = (
        (tilt > STUMBLE_TILT_THRESHOLD_RAD)
        | (height_drop > STUMBLE_ROOT_HEIGHT_DROP_M)
        | (horizontal_ang_vel > STUMBLE_ANGULAR_VELOCITY_RAD_S)
    )
    unstable_runs = _true_run_lengths(unstable)

    foot_contact = np.asarray(arrays["foot_contact"], dtype=bool)
    foot_velocity = np.linalg.norm(np.asarray(arrays["foot_lin_vel_w"])[..., :2], axis=-1)
    contact_slip_speeds = foot_velocity[foot_contact]
    foot_force = np.asarray(arrays["foot_contact_force_history_max_norm"], dtype=np.float64)
    nonfoot_force = np.asarray(arrays["nonfoot_contact_force_history_max_norm"], dtype=np.float64)
    action = np.asarray(arrays["policy_action"], dtype=np.float64)
    joint_margin = np.asarray(arrays["joint_pos_limit_margin"], dtype=np.float64)
    done = np.asarray(arrays["done"], dtype=bool).reshape(-1)
    terminated = np.asarray(arrays["terminated"], dtype=bool).reshape(-1)
    truncated = np.asarray(arrays["truncated"], dtype=bool).reshape(-1)

    summary: dict[str, Any] = {
        "num_frames": num_frames,
        "duration_s": float(num_frames * timestep),
        "terminal": bool(done[-1]),
        "terminated": bool(terminated[-1]),
        "truncated": bool(truncated[-1]),
        "tracking_rmse_tactical_x_m_s": float(np.sqrt(np.mean(np.square(tracking_error[:, 0])))),
        "tracking_rmse_tactical_y_m_s": float(np.sqrt(np.mean(np.square(tracking_error[:, 1])))),
        "tracking_rmse_yaw_rad_s": float(np.sqrt(np.mean(np.square(tracking_error[:, 2])))),
        "tracking_rmse_translation_m_s": float(np.sqrt(np.mean(np.sum(np.square(tracking_error[:, :2]), axis=1)))),
        "tilt_p95_deg": float(np.degrees(np.nanpercentile(tilt, 95))),
        "tilt_max_deg": float(np.degrees(np.nanmax(tilt))),
        "root_height_baseline_m": baseline_height,
        "root_height_min_m": float(np.nanmin(root_height)),
        "root_height_max_drop_m": float(np.nanmax(height_drop)),
        "stumble_events": int(unstable_runs.size),
        "stumble_fraction": float(np.mean(unstable)),
        "max_stumble_duration_s": float(unstable_runs.max(initial=0) * timestep),
        "foot_contact_force_peak_n": float(np.nanmax(foot_force, initial=0.0)),
        "nonfoot_contact_force_peak_n": float(np.nanmax(nonfoot_force, initial=0.0)),
        "foot_slip_mean_m_s": float(np.mean(contact_slip_speeds)) if contact_slip_speeds.size else 0.0,
        "foot_slip_peak_m_s": float(np.max(contact_slip_speeds, initial=0.0)),
        "foot_slip_fraction": (
            float(np.mean(contact_slip_speeds > FOOT_SLIP_SPEED_M_S)) if contact_slip_speeds.size else 0.0
        ),
        "action_saturation_fraction": float(np.mean(np.abs(action) >= ACTION_SATURATION_ABS)),
        "joint_limit_min_margin_rad": float(np.nanmin(joint_margin)),
    }
    if "terrain_level" in arrays:
        summary["terrain_level"] = int(np.asarray(arrays["terrain_level"])[-1])
    if "terrain_type" in arrays:
        summary["terrain_type"] = int(np.asarray(arrays["terrain_type"])[-1])
    if "terrain_name" in arrays:
        summary["terrain_name"] = str(np.asarray(arrays["terrain_name"])[-1])

    amp_reward = np.asarray(arrays["amp_style_reward"], dtype=np.float64).reshape(-1)
    amp_reset_boundary = np.asarray(arrays["amp_transition_reset_boundary"], dtype=bool).reshape(-1)
    valid_amp = np.isfinite(amp_reward) & ~amp_reset_boundary
    summary["amp_style_reward_mean"] = float(np.mean(amp_reward[valid_amp])) if np.any(valid_amp) else float("nan")

    joint_names = list(metadata.get("joint_names", ()))
    joint_pos = np.asarray(arrays["joint_pos"], dtype=np.float64)
    for label, tokens in {
        "arm": ("SHOULDER", "ELBOW"),
        "head": ("HEAD",),
    }.items():
        indices = [index for index, name in enumerate(joint_names) if any(token in name for token in tokens)]
        if indices:
            joint_std = np.std(joint_pos[:, indices], axis=0)
            summary[f"{label}_joint_mean_std_rad"] = float(np.mean(joint_std))
            summary[f"{label}_joint_max_std_rad"] = float(np.max(joint_std))

    reward_terms = np.asarray(arrays["reward_terms"], dtype=np.float64)
    for index, name in enumerate(metadata.get("reward_term_names", ())):
        if index < reward_terms.shape[1]:
            safe_name = _safe_metric_name(str(name))
            summary[f"reward_mean_{safe_name}"] = float(np.mean(reward_terms[:, index]))

    summary.update(_ankle_summary(arrays, metadata=metadata, timestep=timestep))

    return summary


def aggregate_episode_summaries(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate episode summaries by terrain name and curriculum level.

    Missing or non-finite metrics are omitted from a group rather than turning
    the whole aggregate into NaN.  The returned list is sorted by the grouping
    key so reports are reproducible regardless of episode completion order.
    A successful episode is a completed time-limit truncation without an
    environment termination; interrupted partial segments are excluded from
    episode outcome rates but remain part of exposure and stability summaries.
    """
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for summary in summaries:
        name = str(summary.get("terrain_name", "unknown"))
        try:
            level = int(summary.get("terrain_level", -1))
        except (TypeError, ValueError):
            level = -1
        groups.setdefault((name, level), []).append(summary)

    aggregate: list[dict[str, Any]] = []
    for (terrain_name, terrain_level), episodes in sorted(groups.items()):
        row: dict[str, Any] = {
            "terrain_name": terrain_name,
            "terrain_level": terrain_level,
            "sample_n": len(episodes),
            "total_exposure_s": float(
                sum(
                    float(summary["duration_s"])
                    for summary in episodes
                    if isinstance(summary.get("duration_s"), (int, float, np.integer, np.floating))
                    and math.isfinite(float(summary["duration_s"]))
                )
            ),
        }
        terrain_types = sorted(
            {
                int(summary["terrain_type"])
                for summary in episodes
                if isinstance(summary.get("terrain_type"), (int, np.integer))
            }
        )
        if terrain_types:
            row["terrain_types"] = terrain_types

        for metric in _AGGREGATE_MEAN_METRICS:
            values = []
            for summary in episodes:
                value = summary.get(metric)
                if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
                    values.append(float(value))
            if values:
                row[f"mean_{metric}"] = float(np.mean(values))

        complete_values = [summary.get("complete") for summary in episodes if isinstance(summary.get("complete"), bool)]
        if complete_values:
            row["complete_rate"] = float(np.mean(complete_values))

        completed_episodes = [summary for summary in episodes if summary.get("complete") is True]
        row["episode_n"] = len(completed_episodes)
        for field in ("terminal", "terminated", "truncated"):
            values = [summary.get(field) for summary in completed_episodes if isinstance(summary.get(field), bool)]
            if values:
                row[f"{field}_rate"] = float(np.mean(values))

        if completed_episodes:
            success_count = sum(
                summary.get("truncated") is True and summary.get("terminated") is not True
                for summary in completed_episodes
            )
            row["success_n"] = success_count
            row["success_rate"] = success_count / len(completed_episodes)

        termination_reason_counts: dict[str, int] = {}
        for summary in completed_episodes:
            reason = summary.get("termination_reason")
            if reason is not None:
                reason_name = str(reason)
                termination_reason_counts[reason_name] = termination_reason_counts.get(reason_name, 0) + 1
        if termination_reason_counts:
            row["termination_reason_counts"] = dict(sorted(termination_reason_counts.items()))
            row["termination_reason_rates"] = {
                reason: count / len(completed_episodes) for reason, count in sorted(termination_reason_counts.items())
            }
        aggregate.append(row)
    return aggregate


def _to_numpy_row(value: torch.Tensor | np.ndarray, env_index: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value[env_index].detach().cpu().numpy().copy()
    return np.asarray(value[env_index]).copy()


def _optional_numpy_row(value: Any, env_index: int, shape: tuple[int, ...]) -> np.ndarray:
    if value is None:
        return np.full(shape, np.nan, dtype=np.float32)
    try:
        return _to_numpy_row(value, env_index).astype(np.float32, copy=False)
    except (IndexError, TypeError, AttributeError):
        return np.full(shape, np.nan, dtype=np.float32)


def _transition_row(transition: Mapping[str, Any], name: str, env_index: int, fallback: np.ndarray) -> np.ndarray:
    value = transition.get(name)
    if isinstance(value, torch.Tensor):
        return value[env_index].detach().cpu().numpy().copy()
    if isinstance(value, np.ndarray):
        return value[env_index].copy()
    return np.asarray(fallback).copy()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False) as file:
        json.dump(_json_safe(payload), file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
        temporary_path = Path(file.name)
    os.replace(temporary_path, path)


def _resolve_play_timestep_s(player, env) -> float:
    """Resolve the simulation step used by Play diagnostics.

    Some CusRL/IsaacLab environment specs leave ``spec.timestep`` unset even
    though the manager-based environment exposes its actual step as
    ``step_dt``.  Play passes the former to ``Player``, so diagnostics must
    fall back to the latter instead of multiplying by ``None``.
    """

    timestep = getattr(player, "timestep", None)
    if timestep is None:
        timestep = getattr(env, "step_dt", None)
    if timestep is None:
        timestep = getattr(getattr(env, "spec", None), "timestep", None)
    if timestep is None:
        raise ValueError("Play diagnostics could not resolve a positive simulation timestep.")

    timestep = float(timestep)
    if timestep <= 0.0:
        raise ValueError(f"Play diagnostics requires a positive simulation timestep, got {timestep}.")
    return timestep


def _unique_session_directory(base_directory: Path, task_name: str, checkpoint_path: str) -> Path:
    base_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    task_token = _safe_metric_name(task_name).replace(".", "_")[-80:]
    checkpoint_token = _safe_metric_name(Path(checkpoint_path).stem)
    stem = f"{timestamp}_{task_token}_{checkpoint_token}"
    candidate = base_directory / stem
    suffix = 1
    while candidate.exists():
        candidate = base_directory / f"{stem}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    (candidate / "episodes").mkdir()
    return candidate


class T800AmpPlayDiagnosticsHook(PlayerHook):
    """Record selected Play environments to per-episode NPZ files and summaries."""

    def __init__(
        self,
        *,
        env,
        output_directory: str | os.PathLike[str],
        task_name: str,
        checkpoint_path: str,
        seed: int,
        env_indices: Sequence[int] = (0,),
        completed_episode_limit: int = 0,
    ):
        self._env = env
        self._base_output_directory = Path(output_directory).expanduser().resolve()
        self._task_name = task_name
        self._checkpoint_path = checkpoint_path
        self._seed = int(seed)
        self._env_indices = tuple(dict.fromkeys(int(index) for index in env_indices))
        self._completed_episode_limit = max(0, int(completed_episode_limit))

        self._frames: dict[int, list[dict[str, Any]]] = {index: [] for index in self._env_indices}
        self._step_context: dict[int, dict[str, Any]] = {}
        self._terminal_snapshots: dict[int, dict[str, Any]] = {}
        self._episode_counts = {index: 0 for index in self._env_indices}
        self._summaries: list[dict[str, Any]] = []
        self._completed_episodes = 0
        self._saved_segments = 0
        self._closed = False

        self._original_env_step = None
        self._original_reset_idx = None
        self._amp_hook = None
        self._metadata: dict[str, Any] = {}

    @property
    def session_directory(self) -> Path:
        return self._session_directory

    def init(self, player):
        super().init(player)
        self._validate_environment()
        self._resolve_runtime_handles()
        self._session_directory = _unique_session_directory(
            self._base_output_directory,
            self._task_name,
            self._checkpoint_path,
        )
        self._metadata = self._build_metadata()
        _atomic_write_json(self._session_directory / "metadata.json", self._metadata)
        self._install_environment_interceptors()
        print(
            "[INFO] T800 AMP Play diagnostics enabled: "
            f"envs={list(self._env_indices)}, completed_episode_limit={self._completed_episode_limit or 'none'}, "
            f"output={self._session_directory}"
        )

    def step(self, step: int, transition, metrics):
        del metrics
        post_step_context = {env_index: self._capture_command_context(env_index) for env_index in self._env_indices}
        for env_index in self._env_indices:
            done = self._transition_bool(transition, "done", env_index)
            frame = self._terminal_snapshots.pop(env_index, None) if done else None
            if frame is None:
                frame = self._capture_environment_frame(env_index)

            tracking_context = self._step_context.get(env_index, post_step_context[env_index])
            frame.update(self._prefixed_context("tracking", tracking_context))
            frame.update(self._prefixed_context("post_step", post_step_context[env_index]))
            frame["global_step"] = np.int64(step)
            frame["policy_action"] = _transition_row(
                transition,
                "action",
                env_index,
                np.zeros(len(self._metadata["action_names"]), dtype=np.float32),
            ).astype(np.float32, copy=False)
            frame["agent_reward"] = _transition_row(
                transition,
                "reward",
                env_index,
                np.zeros(len(self._metadata["agent_reward_names"]), dtype=np.float32),
            ).astype(np.float32, copy=False)
            frame["terminated"] = np.bool_(self._transition_bool(transition, "terminated", env_index))
            frame["truncated"] = np.bool_(self._transition_bool(transition, "truncated", env_index))
            frame["done"] = np.bool_(done)
            frame.update(self._capture_amp_metrics(transition, env_index, reset_boundary=done))
            self._frames[env_index].append(frame)

            if done:
                self._save_episode(env_index, complete=True)

        self._step_context.clear()
        if self._completed_episode_limit and self._completed_episodes >= self._completed_episode_limit:
            self.player.interrupted = True

    def reset(self, indices):
        del indices

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._restore_environment_interceptors()
        for env_index in self._env_indices:
            if self._frames[env_index]:
                self._save_episode(env_index, complete=False, fallback_reason="interrupted")
        self._metadata.update(
            {
                "closed_at": datetime.now().astimezone().isoformat(),
                "completed_episodes": self._completed_episodes,
                "saved_segments": self._saved_segments,
            }
        )
        _atomic_write_json(self._session_directory / "metadata.json", self._metadata)
        self._write_summary_files()
        print(
            "[INFO] T800 AMP Play diagnostics closed: "
            f"completed_episodes={self._completed_episodes}, saved_segments={self._saved_segments}, "
            f"output={self._session_directory}"
        )

    def _validate_environment(self) -> None:
        if not self._env_indices:
            raise ValueError("record_envs must contain at least one environment index.")
        invalid = [index for index in self._env_indices if index < 0 or index >= self._env.num_envs]
        if invalid:
            raise ValueError(f"record_envs contains indices outside [0, {self._env.num_envs - 1}]: {invalid}.")
        try:
            robot = self._env.scene["robot"]
            contact_sensor = self._env.scene["contact_forces"]
        except KeyError as error:
            raise ValueError("Play diagnostics requires scene assets 'robot' and 'contact_forces'.") from error
        missing_feet = [name for name in T800_FOOT_BODY_NAMES if name not in robot.body_names]
        missing_contact_feet = [name for name in T800_FOOT_BODY_NAMES if name not in contact_sensor.body_names]
        if missing_feet or missing_contact_feet:
            raise ValueError(
                "Play diagnostics currently targets the T800 body contract; "
                f"missing_robot_feet={missing_feet}, missing_contact_feet={missing_contact_feet}."
            )

    def _resolve_runtime_handles(self) -> None:
        self._robot = self._env.scene["robot"]
        self._contact_sensor = self._env.scene["contact_forces"]
        self._robot_foot_ids = tuple(self._robot.body_names.index(name) for name in T800_FOOT_BODY_NAMES)
        self._contact_foot_ids = tuple(self._contact_sensor.body_names.index(name) for name in T800_FOOT_BODY_NAMES)
        missing_ankles = [name for name in T800_ANKLE_JOINT_NAMES if name not in self._robot.joint_names]
        if missing_ankles:
            raise ValueError(f"Play diagnostics requires explicit T800 ankle joints; missing={missing_ankles}.")
        self._ankle_joint_ids = tuple(self._robot.joint_names.index(name) for name in T800_ANKLE_JOINT_NAMES)
        self._command_term = self._env.command_manager.get_term("base_velocity")
        self._front_offset_deg = float(getattr(self._command_term.cfg, "tactical_front_offset_deg", 0.0))
        for hook in getattr(self.agent, "hook", ()):
            if hasattr(hook, "discriminator") and hasattr(hook, "reward_scale"):
                self._amp_hook = hook
                break

    def _build_metadata(self) -> dict[str, Any]:
        action_names = self._action_names()
        reward_term_names = [
            name for name, _ in self._env.reward_manager.get_active_iterable_terms(env_idx=self._env_indices[0])
        ]
        termination_names = list(self._env.termination_manager.active_terms)
        environment_reward_names = self._environment_reward_names()
        agent_reward_dim = int(getattr(self.environment.spec, "reward_dim", len(environment_reward_names)))
        terrain_metadata = self._terrain_metadata()
        return {
            "format_version": FORMAT_VERSION,
            "profile": "T800 AMP boxing-guard locomotion diagnostics",
            "task": self._task_name,
            "checkpoint": str(Path(self._checkpoint_path).expanduser().resolve()),
            "seed": self._seed,
            "created_at": datetime.now().astimezone().isoformat(),
            "timestep_s": _resolve_play_timestep_s(self.player, self._env),
            "recorded_env_indices": list(self._env_indices),
            "completed_episode_limit": self._completed_episode_limit,
            "quaternion_order": "wxyz",
            "world_frame": "+Z up; root frame +X forward; +Y left",
            "tactical_front_offset_deg": self._front_offset_deg,
            "command_semantics": {
                "tracking_command": "command present before the physics step and used by velocity rewards",
                "post_step_command": "command present in the observation returned by that step",
            },
            "terminal_capture": "physics/contact/reward state copied immediately before Isaac Lab auto-reset",
            "amp_terminal_note": (
                "AMP transition on terminal frames contains the auto-reset next observation; "
                "amp_transition_reset_boundary marks those frames and summaries exclude them from AMP means"
            ),
            "joint_names": list(self._robot.joint_names),
            "ankle_joint_names": list(T800_ANKLE_JOINT_NAMES),
            "ankle_joint_indices": list(self._ankle_joint_ids),
            "ankle_joint_order_source": "robot.joint_names lookup by explicit T800_ANKLE_JOINT_NAMES; never action order",
            "body_names": list(self._robot.body_names),
            "contact_body_names": list(self._contact_sensor.body_names),
            "foot_body_names": list(T800_FOOT_BODY_NAMES),
            "action_names": action_names,
            "reward_term_names": reward_term_names,
            "termination_term_names": termination_names,
            "environment_reward_names": environment_reward_names,
            "agent_reward_names": [f"reward_{index}" for index in range(agent_reward_dim)],
            "diagnostic_thresholds": {
                "stumble_tilt_rad": STUMBLE_TILT_THRESHOLD_RAD,
                "stumble_root_height_drop_m": STUMBLE_ROOT_HEIGHT_DROP_M,
                "stumble_horizontal_angular_velocity_rad_s": STUMBLE_ANGULAR_VELOCITY_RAD_S,
                "foot_contact_force_n": FOOT_CONTACT_FORCE_N,
                "foot_slip_speed_m_s": FOOT_SLIP_SPEED_M_S,
                "action_saturation_abs": ACTION_SATURATION_ABS,
                "ankle_velocity_reversal_deadband_rad_s": ANKLE_VELOCITY_REVERSAL_DEADBAND_RAD_S,
            },
            "ankle_limits": self._ankle_limit_metadata(),
            "ankle_sampling": {
                "clock": "Play control cycle",
                "timestep_s": _resolve_play_timestep_s(self.player, self._env),
                "not_physics_substeps": True,
                "derivative_timebase": "episode_time_s; uniformity validated per episode",
            },
            "terrain": terrain_metadata,
        }

    def _ankle_limit_metadata(self) -> dict[str, Any]:
        robot_data = self._robot.data
        soft = _optional_numpy_row(
            getattr(robot_data, "soft_joint_pos_limits", None),
            self._env_indices[0],
            (len(self._robot.joint_names), 2),
        )
        hard = _optional_numpy_row(
            getattr(robot_data, "joint_pos_limits", None),
            self._env_indices[0],
            (len(self._robot.joint_names), 2),
        )
        velocity = _optional_numpy_row(
            getattr(robot_data, "joint_vel_limits", None),
            self._env_indices[0],
            (len(self._robot.joint_names),),
        )
        return {
            "units": {"position": "rad", "velocity": "rad_s"},
            "available": {
                "soft_position": bool(np.isfinite(soft[list(self._ankle_joint_ids)]).any()),
                "hard_position": bool(np.isfinite(hard[list(self._ankle_joint_ids)]).any()),
                "velocity": bool(np.isfinite(velocity[list(self._ankle_joint_ids)]).any()),
            },
            "soft_position_rad": soft[list(self._ankle_joint_ids)].tolist(),
            "hard_position_rad": hard[list(self._ankle_joint_ids)].tolist(),
            "velocity_rad_s": velocity[list(self._ankle_joint_ids)].tolist(),
        }

    def _action_names(self) -> list[str]:
        names: list[str] = []
        for term_name in self._env.action_manager.active_terms:
            term = self._env.action_manager.get_term(term_name)
            term_joint_names = list(getattr(term, "_joint_names", ()))
            if term_joint_names:
                names.extend(term_joint_names)
            else:
                names.extend(f"{term_name}_{index}" for index in range(term.action_dim))
        return names

    def _environment_reward_names(self) -> list[str]:
        active_terms = getattr(self._env.reward_manager, "active_terms", None)
        if isinstance(active_terms, Mapping):
            return [str(name) for name in active_terms]
        return ["environment_reward"]

    def _terrain_metadata(self) -> dict[str, Any]:
        terrain = getattr(self._env.scene, "terrain", None)
        terrain_cfg = getattr(terrain, "cfg", None)
        generator_cfg = getattr(terrain_cfg, "terrain_generator", None)
        sub_terrains = getattr(generator_cfg, "sub_terrains", {}) or {}
        return {
            "terrain_type": getattr(terrain_cfg, "terrain_type", None),
            "generator_type": type(getattr(terrain, "terrain_generator", None)).__name__,
            "num_rows": getattr(generator_cfg, "num_rows", None),
            "num_cols": getattr(generator_cfg, "num_cols", None),
            "sub_terrains": [
                {
                    "name": name,
                    "cfg_type": type(cfg).__name__,
                    "proportion": float(cfg.proportion),
                }
                for name, cfg in sub_terrains.items()
            ],
        }

    def _install_environment_interceptors(self) -> None:
        self._original_env_step = self._env.step
        self._original_reset_idx = self._env._reset_idx

        def intercepted_step(action):
            self._step_context = {
                env_index: self._capture_command_context(env_index) for env_index in self._env_indices
            }
            return self._original_env_step(action)

        def intercepted_reset_idx(env_ids):
            self._capture_terminal_states(env_ids)
            return self._original_reset_idx(env_ids)

        self._env.step = intercepted_step
        self._env._reset_idx = intercepted_reset_idx

    def _restore_environment_interceptors(self) -> None:
        if self._original_env_step is not None:
            self._env.step = self._original_env_step
            self._original_env_step = None
        if self._original_reset_idx is not None:
            self._env._reset_idx = self._original_reset_idx
            self._original_reset_idx = None

    def _capture_terminal_states(self, env_ids) -> None:
        if isinstance(env_ids, torch.Tensor):
            reset_ids = set(int(index) for index in env_ids.detach().cpu().tolist())
        else:
            reset_ids = set(int(index) for index in env_ids)
        done_buffer = self._env.termination_manager.dones
        for env_index in self._env_indices:
            if env_index in reset_ids and bool(done_buffer[env_index].item()):
                self._terminal_snapshots[env_index] = self._capture_environment_frame(env_index)

    def _capture_command_context(self, env_index: int) -> dict[str, Any]:
        command_root_yaw = _to_numpy_row(self._env.command_manager.get_command("base_velocity"), env_index).astype(
            np.float32,
            copy=False,
        )
        command_mode = getattr(self._command_term, "command_mode", None)
        return {
            "command_root_yaw": command_root_yaw,
            "command_tactical": root_yaw_to_tactical_velocity(command_root_yaw, self._front_offset_deg),
            "command_mode": np.int64(command_mode[env_index].item()) if command_mode is not None else np.int64(-1),
            "command_counter": np.int64(self._command_term.command_counter[env_index].item()),
            "command_time_left_s": np.float32(self._command_term.time_left[env_index].item()),
        }

    @staticmethod
    def _prefixed_context(prefix: str, context: Mapping[str, Any]) -> dict[str, Any]:
        return {f"{prefix}_{name}": value for name, value in context.items()}

    def _capture_environment_frame(self, env_index: int) -> dict[str, Any]:
        robot_data = self._robot.data
        root_pos_w = _to_numpy_row(robot_data.root_pos_w, env_index).astype(np.float32, copy=False)
        root_quat_w = _to_numpy_row(robot_data.root_quat_w, env_index).astype(np.float32, copy=False)
        root_lin_vel_w = _to_numpy_row(robot_data.root_lin_vel_w, env_index).astype(np.float32, copy=False)
        root_ang_vel_w = _to_numpy_row(robot_data.root_ang_vel_w, env_index).astype(np.float32, copy=False)
        root_rpy = quaternion_wxyz_to_rpy(root_quat_w)
        root_yaw_velocity = np.concatenate(
            (
                world_velocity_to_yaw_frame(root_lin_vel_w, float(root_rpy[2]))[:2],
                root_ang_vel_w[2:3],
            )
        ).astype(np.float32, copy=False)
        actual_tactical_velocity = root_yaw_to_tactical_velocity(root_yaw_velocity, self._front_offset_deg)

        env_origin = _to_numpy_row(self._env.scene.env_origins, env_index).astype(np.float32, copy=False)
        joint_pos = _to_numpy_row(robot_data.joint_pos, env_index).astype(np.float32, copy=False)
        joint_limits = _to_numpy_row(robot_data.soft_joint_pos_limits, env_index).astype(np.float32, copy=False)
        joint_limit_margin = np.minimum(joint_pos - joint_limits[:, 0], joint_limits[:, 1] - joint_pos)
        hard_joint_limits = _optional_numpy_row(
            getattr(robot_data, "joint_pos_limits", None), env_index, joint_limits.shape
        )
        joint_velocity_limits = _optional_numpy_row(
            getattr(robot_data, "joint_vel_limits", None), env_index, (joint_pos.shape[0],)
        )

        body_pos_w = _to_numpy_row(robot_data.body_pos_w, env_index).astype(np.float32, copy=False)
        body_lin_vel_w = _to_numpy_row(robot_data.body_lin_vel_w, env_index).astype(np.float32, copy=False)
        foot_pos_w = body_pos_w[list(self._robot_foot_ids)]
        foot_lin_vel_w = body_lin_vel_w[list(self._robot_foot_ids)]

        contact_data = self._contact_sensor.data
        contact_force_w = _to_numpy_row(contact_data.net_forces_w, env_index).astype(np.float32, copy=False)
        contact_history = _to_numpy_row(contact_data.net_forces_w_history, env_index).astype(np.float32, copy=False)
        contact_history_max_norm = np.linalg.norm(contact_history, axis=-1).max(axis=0).astype(np.float32)
        foot_contact_force_history_max_norm = contact_history_max_norm[list(self._contact_foot_ids)]
        foot_contact = foot_contact_force_history_max_norm > FOOT_CONTACT_FORCE_N
        nonfoot_force = contact_history_max_norm.copy()
        nonfoot_force[list(self._contact_foot_ids)] = -np.inf
        nonfoot_index = int(np.argmax(nonfoot_force)) if nonfoot_force.size else -1
        nonfoot_max = float(nonfoot_force[nonfoot_index]) if nonfoot_index >= 0 else 0.0

        processed_action_parts = []
        for term_name in self._env.action_manager.active_terms:
            processed_action_parts.append(
                _to_numpy_row(
                    self._env.action_manager.get_term(term_name).processed_actions,
                    env_index,
                ).reshape(-1)
            )
        processed_action = np.concatenate(processed_action_parts).astype(np.float32, copy=False)

        reward_terms = np.asarray(
            [float(values[0]) for _, values in self._env.reward_manager.get_active_iterable_terms(env_idx=env_index)],
            dtype=np.float32,
        )
        termination_terms = np.asarray(
            [
                bool(self._env.termination_manager.get_term(name)[env_index].item())
                for name in self._metadata["termination_term_names"]
            ],
            dtype=bool,
        )

        terrain_level, terrain_type, terrain_name = self._terrain_state(env_index)
        return {
            "episode_step": np.int64(self._env.episode_length_buf[env_index].item()),
            "episode_time_s": np.float32(
                self._env.episode_length_buf[env_index].item() * _resolve_play_timestep_s(self.player, self._env)
            ),
            "env_origin_w": env_origin,
            "root_pos_w": root_pos_w,
            "root_pos_env": root_pos_w - env_origin,
            "root_height_env": np.float32(root_pos_w[2] - env_origin[2]),
            "root_quat_w": root_quat_w,
            "root_rpy": root_rpy,
            "root_lin_vel_w": root_lin_vel_w,
            "root_ang_vel_w": root_ang_vel_w,
            "root_projected_gravity_b": _to_numpy_row(robot_data.projected_gravity_b, env_index).astype(
                np.float32,
                copy=False,
            ),
            "actual_root_yaw_velocity": root_yaw_velocity,
            "actual_tactical_velocity": actual_tactical_velocity,
            "joint_pos": joint_pos,
            "joint_vel": _to_numpy_row(robot_data.joint_vel, env_index).astype(np.float32, copy=False),
            "joint_pos_target": _to_numpy_row(robot_data.joint_pos_target, env_index).astype(np.float32, copy=False),
            "joint_pos_limit_margin": joint_limit_margin.astype(np.float32, copy=False),
            "ankle_soft_joint_pos_limits_rad": joint_limits[list(self._ankle_joint_ids)],
            "ankle_hard_joint_pos_limits_rad": hard_joint_limits[list(self._ankle_joint_ids)],
            "ankle_velocity_limits_rad_s": joint_velocity_limits[list(self._ankle_joint_ids)],
            "computed_torque": _to_numpy_row(robot_data.computed_torque, env_index).astype(np.float32, copy=False),
            "applied_torque": _to_numpy_row(robot_data.applied_torque, env_index).astype(np.float32, copy=False),
            "processed_action": processed_action,
            "foot_pos_w": foot_pos_w,
            "foot_pos_env": foot_pos_w - env_origin,
            "foot_lin_vel_w": foot_lin_vel_w,
            "contact_force_w": contact_force_w,
            "contact_force_history_max_norm": contact_history_max_norm,
            "foot_contact_force_history_max_norm": foot_contact_force_history_max_norm,
            "foot_contact": foot_contact,
            "foot_current_air_time_s": _to_numpy_row(contact_data.current_air_time, env_index)[
                list(self._contact_foot_ids)
            ].astype(np.float32, copy=False),
            "foot_current_contact_time_s": _to_numpy_row(contact_data.current_contact_time, env_index)[
                list(self._contact_foot_ids)
            ].astype(np.float32, copy=False),
            "nonfoot_contact_body_index": np.int64(nonfoot_index),
            "nonfoot_contact_force_history_max_norm": np.float32(max(0.0, nonfoot_max)),
            "reward_terms": reward_terms,
            "environment_reward": self._environment_reward(env_index),
            "termination_terms": termination_terms,
            "terrain_level": np.int64(terrain_level),
            "terrain_type": np.int64(terrain_type),
            "terrain_name": np.str_(terrain_name),
        }

    def _environment_reward(self, env_index: int) -> np.ndarray:
        reward_buf = self._env.reward_buf
        if isinstance(reward_buf, Mapping):
            return np.asarray(
                [float(value[env_index].item()) for value in reward_buf.values()],
                dtype=np.float32,
            )
        return np.asarray([float(reward_buf[env_index].item())], dtype=np.float32)

    def _terrain_state(self, env_index: int) -> tuple[int, int, str]:
        terrain = getattr(self._env.scene, "terrain", None)
        levels = getattr(terrain, "terrain_levels", None)
        types = getattr(terrain, "terrain_types", None)
        if levels is None or types is None:
            return -1, -1, "plane"
        level = int(levels[env_index].item())
        terrain_type = int(types[env_index].item())
        generator = getattr(terrain, "terrain_generator", None)
        specific_cfg = None
        if generator is not None and hasattr(generator, "get_subterrain_cfg"):
            specific_cfg = generator.get_subterrain_cfg(level, terrain_type)
        if specific_cfg is None:
            generator_cfg = getattr(terrain.cfg, "terrain_generator", None)
            return level, terrain_type, terrain_name_from_column(generator_cfg, terrain_type)
        generator_cfg = getattr(terrain.cfg, "terrain_generator", None)
        for name, cfg in (getattr(generator_cfg, "sub_terrains", {}) or {}).items():
            if type(cfg) is type(specific_cfg) and getattr(cfg, "function", None) is getattr(
                specific_cfg, "function", None
            ):
                return level, terrain_type, name
        return level, terrain_type, type(specific_cfg).__name__

    def _capture_amp_metrics(self, transition, env_index: int, *, reset_boundary: bool) -> dict[str, Any]:
        logit_value = float("nan")
        reward_value = float("nan")
        agent_transition = transition.get("agent_transition")
        if self._amp_hook is not None and isinstance(agent_transition, torch.Tensor):
            with torch.no_grad(), self.agent.autocast():
                logit = self._amp_hook.discriminator(agent_transition[env_index : env_index + 1])
            probability = torch.sigmoid(logit)
            reward = -torch.log(torch.clamp(1.0 - probability, min=1.0e-4)) * self._amp_hook.reward_scale
            logit_value = float(logit.reshape(-1)[0].detach().cpu().item())
            reward_value = float(reward.reshape(-1)[0].detach().cpu().item())
        return {
            "amp_discriminator_logit": np.float32(logit_value),
            "amp_style_reward": np.float32(reward_value),
            "amp_transition_reset_boundary": np.bool_(reset_boundary),
        }

    @staticmethod
    def _transition_bool(transition: Mapping[str, Any], name: str, env_index: int) -> bool:
        value = transition.get(name)
        if isinstance(value, torch.Tensor):
            return bool(value[env_index].reshape(-1)[0].item())
        if isinstance(value, np.ndarray):
            return bool(value[env_index].reshape(-1)[0])
        return False

    def _termination_reason(self, arrays: Mapping[str, np.ndarray], fallback: str) -> str:
        flags = np.asarray(arrays["termination_terms"], dtype=bool)[-1]
        reasons = [name for name, active in zip(self._metadata["termination_term_names"], flags, strict=True) if active]
        return "+".join(reasons) if reasons else fallback

    def _save_episode(self, env_index: int, *, complete: bool, fallback_reason: str = "done") -> None:
        frames = self._frames[env_index]
        if not frames:
            return
        arrays = stack_episode_frames(frames)
        arrays = add_ankle_diagnostic_arrays(
            arrays,
            timestep=_resolve_play_timestep_s(self.player, self._env),
            ankle_indices=self._metadata.get("ankle_joint_indices"),
        )
        self._frames[env_index] = []
        episode_index = self._episode_counts[env_index]
        self._episode_counts[env_index] += 1
        self._saved_segments += 1
        if complete:
            self._completed_episodes += 1

        filename = f"episode_{self._saved_segments:04d}_env{env_index:04d}_local{episode_index:04d}.npz"
        output_path = self._session_directory / "episodes" / filename
        with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".npz", delete=False) as file:
            temporary_path = Path(file.name)
            np.savez_compressed(file, **arrays)
        os.replace(temporary_path, output_path)

        summary = summarize_episode(
            arrays,
            timestep=_resolve_play_timestep_s(self.player, self._env),
            metadata=self._metadata,
        )
        summary.update(
            {
                "segment_id": self._saved_segments,
                "env_index": env_index,
                "env_episode_index": episode_index,
                "complete": complete,
                "termination_reason": self._termination_reason(arrays, fallback_reason),
                "terrain_name": str(np.asarray(arrays["terrain_name"])[-1]),
                "terrain_level": int(np.asarray(arrays["terrain_level"])[-1]),
                "terrain_type": int(np.asarray(arrays["terrain_type"])[-1]),
                "file": str(output_path.relative_to(self._session_directory)),
            }
        )
        self._summaries.append(summary)
        self._write_summary_files()
        print(
            "[INFO] Saved Play diagnostic episode: "
            f"env={env_index}, frames={summary['num_frames']}, reason={summary['termination_reason']}, "
            f"file={output_path}"
        )

    def _write_summary_files(self) -> None:
        terrain_aggregates = aggregate_episode_summaries(self._summaries)
        _atomic_write_json(
            self._session_directory / "summary.json",
            {
                "format_version": FORMAT_VERSION,
                "task": self._task_name,
                "episodes": self._summaries,
                "terrain_aggregates": terrain_aggregates,
            },
        )
        _atomic_write_json(
            self._session_directory / "terrain_aggregates.json",
            {
                "format_version": FORMAT_VERSION,
                "task": self._task_name,
                "groups": terrain_aggregates,
            },
        )
        if not self._summaries:
            return
        field_names = list(dict.fromkeys(key for summary in self._summaries for key in summary))
        csv_path = self._session_directory / "episodes.csv"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=csv_path.parent, delete=False) as file:
            writer = csv.DictWriter(file, fieldnames=field_names)
            writer.writeheader()
            for summary in self._summaries:
                writer.writerow({key: _json_safe(value) for key, value in summary.items()})
            temporary_path = Path(file.name)
        os.replace(temporary_path, csv_path)
