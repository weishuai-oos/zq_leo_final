"""Simulator-independent settings for the second-stage impact curriculum.

Force peaks were estimated by the operator. Pulse widths, impulse limits and
local hit-point boxes below are provisional engineering assumptions: they must
be calibrated from instrumented impacts before making hardware claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ImpactTargetCfg:
    name: str
    group: str
    weight: float
    point_half_extent_m: tuple[float, float, float]
    duration_s: tuple[float, float]
    common_impulse_limit_ns: float
    stress_impulse_limit_ns: float


def _targets() -> tuple[ImpactTargetCfg, ...]:
    # Points are relative to the link origin, not its centre of mass. The
    # equivalent local torque is calculated explicitly by ImpactRuntime.
    targets = [
        ImpactTargetCfg("LINK_TORSO_YAW", "torso", 3.0, (0.10, 0.13, 0.18), (0.025, 0.080), 100.0, 170.0),
        ImpactTargetCfg("LINK_BASE", "pelvis", 1.2, (0.10, 0.12, 0.10), (0.025, 0.080), 100.0, 170.0),
        ImpactTargetCfg("LINK_HEAD_YAW", "head", 0.6, (0.07, 0.07, 0.08), (0.012, 0.035), 20.0, 40.0),
    ]
    for side in ("L", "R"):
        targets.extend(
            (
                ImpactTargetCfg(
                    f"LINK_SHOULDER_ROLL_{side}", "arm", 0.7, (0.06, 0.06, 0.10), (0.018, 0.060), 55.0, 100.0
                ),
                ImpactTargetCfg(
                    f"LINK_SHOULDER_YAW_{side}", "arm", 0.8, (0.06, 0.06, 0.12), (0.018, 0.060), 55.0, 100.0
                ),
                ImpactTargetCfg(f"LINK_ELBOW_YAW_{side}", "arm", 1.0, (0.06, 0.06, 0.13), (0.018, 0.060), 55.0, 100.0),
                ImpactTargetCfg(f"LINK_HIP_YAW_{side}", "thigh", 0.8, (0.07, 0.07, 0.15), (0.025, 0.090), 65.0, 110.0),
                ImpactTargetCfg(f"LINK_KNEE_PITCH_{side}", "calf", 0.6, (0.06, 0.06, 0.17), (0.025, 0.090), 45.0, 80.0),
            )
        )
    return tuple(targets)


@dataclass
class ImpactCfg:
    enabled: bool = True
    enabled_eval: bool = True
    mode: str = "train"  # train / off / common / stress (latter two bypass warmup)
    seed_offset: int = 7301
    command_name: str = "base_velocity"
    tactical_front_offset_deg: float = 37.0
    warmup_updates: int = 500
    ramp_updates: int = 5000
    initial_strength: float = 0.15
    eval_strength: float = 1.0
    fixed_peak_n: float | None = None
    fixed_duration_s: float | None = None
    fixed_interval_s: float | None = None

    clean_episode_probability: float = 0.25
    stress_episode_probability: float = 0.10
    direction_probabilities: tuple[float, float, float] = (0.60, 0.35, 0.05)
    # Incoming bearings: front ±60°, sides 60..140°, rear 140..180°.
    direction_elevation_deg: tuple[float, float] = (-15.0, 15.0)
    common_peak_bands_n: tuple[tuple[float, float], ...] = ((40.0, 300.0), (300.0, 900.0), (900.0, 2000.0))
    common_peak_band_probabilities: tuple[float, ...] = (0.35, 0.40, 0.25)
    stress_peak_n: tuple[float, float] = (1600.0, 5000.0)
    targets: tuple[ImpactTargetCfg, ...] = field(default_factory=_targets)
    first_hit_delay_s: tuple[float, float] = (1.5, 3.0)
    recovery_pause_s: tuple[float, float] = (2.0, 5.0)
    combo_gap_s: tuple[float, float] = (0.15, 0.50)  # onset-to-onset, includes pulse width
    common_combo_probability: float = 0.30
    # Both training stress episodes and explicit stress Play use isolated hits.
    stress_combo_probability: float = 0.0
    common_combo_max_hits: int = 3
    stress_combo_max_hits: int = 1

    # A recent force alone does not reduce the guard/style constraints. Actual
    # tilt, angular speed or command-relative velocity must cross a threshold.
    recovery_min_scale: float = 0.35
    recovery_recent_hit_s: float = 0.40
    recovery_max_s: float = 1.20
    recovery_stable_hold_s: float = 0.20
    recovery_rearm_stable_s: float = 0.40
    recovery_tilt_threshold_rad: float = 0.18
    recovery_angular_speed_threshold: float = 1.00
    recovery_velocity_error_threshold: float = 0.55
    measurement_window_s: float = 2.0

    # Shared with the independently implemented head controller. These are
    # physical fault probabilities; actor observations have no fault flag.
    head_mode: str = "full"
    head_fault_warmup_updates: int = 500
    head_passive_probability: float = 0.15
    head_weak_probability: float = 0.10
    head_fault_delay_s: tuple[float, float] = (2.0, 8.0)
