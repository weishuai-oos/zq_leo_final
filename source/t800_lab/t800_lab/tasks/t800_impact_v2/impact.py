"""Simulator-independent settings for the second-stage impact curriculum.

Force peaks were estimated by the operator. Pulse widths, impulse limits and
local hit-point boxes below are provisional engineering assumptions: they must
be calibrated from instrumented impacts before making hardware claims.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import torch


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


def half_sine_step_impulse(peak: torch.Tensor, duration: torch.Tensor, age: torch.Tensor, dt: float) -> torch.Tensor:
    """Exact integral of a finite half sine over [age, age + dt], in N s."""
    start = age.clamp_min(0.0).minimum(duration)
    end = (age + dt).clamp_min(0.0).minimum(duration)
    return peak * duration / math.pi * (torch.cos(math.pi * start / duration) - torch.cos(math.pi * end / duration))


def _quat_rotate(q: torch.Tensor, v: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    """Rotate with Isaac Lab's wxyz quaternion convention."""
    xyz = -q[..., 1:] if inverse else q[..., 1:]
    twice = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + q[..., :1] * twice + torch.cross(xyz, twice, dim=-1)


def _yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


class ImpactRuntime:
    """One pulse per environment, with short nonoverlapping combinations.

    Checkpoints persist the curriculum and dedicated RNG, not simulator episode
    state. A resumed environment starts fresh physical episodes. ``reset`` never
    resets global update progress. Episode metrics are simulator diagnostics,
    not observations supplied to the policy.
    """

    STATE_VERSION = 2
    GROUPS = ("torso", "pelvis", "arm", "thigh", "calf", "head")
    DIRECTIONS = ("front", "side", "rear")

    def __init__(self, env: Any, cfg: ImpactCfg):
        self.env, self.cfg = env, cfg
        self.robot = env.scene["robot"]
        self.device = torch.device(env.device)
        self.num_envs = int(env.num_envs)
        self.dt = float(getattr(env, "physics_dt", env.cfg.sim.dt))
        self._validate_config()
        self._generator = torch.Generator(device=self.device)
        self.rank = int(os.environ.get("RANK", "0"))
        if self.rank < 0:
            raise ValueError("Distributed rank must be nonnegative.")
        self._base_seed = int(getattr(env.cfg, "seed", 0) or 0) + cfg.seed_offset + self.rank
        self._generator.manual_seed(self._base_seed)
        self.completed_updates = 0
        # Optional Play-only observer. It does not enter observations, RNG or checkpoints.
        self.hit_observer: Callable[[torch.Tensor], None] | None = None
        # An explicit training-entry flag permits the one known distribution
        # change. Evaluation can always compare old policies under the new load.
        self.allow_legacy_stress_combo_migration = False
        self.config_signature = self._config_signature()
        self.legacy_stress_combo_signature = (
            self._config_signature(legacy_stress_combo=True)
            if cfg.stress_combo_probability == 0 and cfg.stress_combo_max_hits == 1
            else None
        )
        names = list(self.robot.body_names)
        self.targets = tuple(
            t for t in cfg.targets if t.name in names and not (cfg.head_mode == "headless" and t.group == "head")
        )
        if not self.targets:
            raise ValueError("Impact configuration has no targets present in robot.body_names.")

        def tensor(values, dtype=torch.float32):
            return torch.tensor(values, dtype=dtype, device=self.device)

        self._target_body = tensor([names.index(t.name) for t in self.targets], torch.long)
        self._target_weights = tensor([t.weight for t in self.targets])
        self._target_groups = tensor([self.GROUPS.index(t.group) for t in self.targets], torch.long)
        self._point_extents = tensor([t.point_half_extent_m for t in self.targets])
        self._durations = tensor([t.duration_s for t in self.targets])
        self._impulse_limits = tensor([(t.common_impulse_limit_ns, t.stress_impulse_limit_ns) for t in self.targets])
        self._direction_weights = tensor(cfg.direction_probabilities)
        self._band_weights = tensor(cfg.common_peak_band_probabilities)
        self._peak_bands = tensor(cfg.common_peak_bands_n)
        n = self.num_envs

        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(*shape, device=self.device, dtype=dtype)

        self.episode_kind = zeros(n, dtype=torch.long)  # 0 clean / 1 common / 2 stress
        self.active = zeros(n, dtype=torch.bool)
        self.cooldown = zeros(n)
        self.age = zeros(n)
        self.duration = torch.ones(n, device=self.device)
        self.peak_n = zeros(n)
        self.body_id = zeros(n, dtype=torch.long)
        self.target_id = zeros(n, dtype=torch.long)
        self.direction_sector = zeros(n, dtype=torch.long)
        self.force_direction_w = zeros(n, 3)
        self.hit_point_b = zeros(n, 3)
        self.combo_remaining = zeros(n, dtype=torch.long)
        self._onset_gap = zeros(n)
        self.forces_b = zeros(n, len(names), 3)
        self.torques_b = torch.zeros_like(self.forces_b)
        self._all_ids = torch.arange(n, device=self.device)
        self._since_hit = torch.full((n,), float("inf"), device=self.device)
        self._recovering = zeros(n, dtype=torch.bool)
        self._recovery_armed = torch.ones(n, device=self.device, dtype=torch.bool)
        self._recovery_age = zeros(n)
        self._stable_age = zeros(n)
        self._recovery_scale = torch.ones(n, device=self.device)
        self.last_step_style_scale = torch.ones(n, device=self.device)
        self._measurement_remaining = zeros(n)
        self._expected_xy = zeros(n, 2)
        self._measurement_error = zeros(n)
        self._stats = {
            name: zeros(n)
            for name in (
                "hits",
                "impulse_ns",
                "peak_n",
                "sampled_peak_n",
                "recoveries",
                "recovery_time_s",
                "recovery_expired",
                "extra_displacement_m",
                "measurement_count",
                "common_hits",
                "stress_hits",
            )
        }
        self._direction_hits = zeros(n, 3)
        self._group_hits = zeros(n, len(self.GROUPS))
        self.reset(self._all_ids)

    def _validate_config(self) -> None:
        c = self.cfg
        if c.mode not in ("train", "off", "common", "stress"):
            raise ValueError("Impact mode must be train, off, common, or stress.")
        if c.warmup_updates < 0 or c.ramp_updates < 0 or self.dt <= 0:
            raise ValueError("Impact update counts must be nonnegative and physics dt positive.")
        if not 0.0 <= c.initial_strength <= 1.0 or not 0.0 <= c.eval_strength <= 1.0:
            raise ValueError("Impact strength must lie in [0, 1].")
        if (
            c.clean_episode_probability < 0
            or c.stress_episode_probability < 0
            or c.clean_episode_probability + c.stress_episode_probability > 1
        ):
            raise ValueError("Clean/stress probabilities must be nonnegative and sum to at most one.")
        if (
            len(c.direction_probabilities) != 3
            or any(p < 0 for p in c.direction_probabilities)
            or not math.isclose(sum(c.direction_probabilities), 1.0)
        ):
            raise ValueError("Direction probabilities must be three nonnegative values summing to one.")
        if (
            len(c.common_peak_bands_n) != len(c.common_peak_band_probabilities)
            or any(p < 0 for p in c.common_peak_band_probabilities)
            or sum(c.common_peak_band_probabilities) <= 0
        ):
            raise ValueError("Common force bands need matching nonnegative probabilities.")
        ranges = (
            c.first_hit_delay_s,
            c.recovery_pause_s,
            c.combo_gap_s,
            c.direction_elevation_deg,
            c.stress_peak_n,
            *c.common_peak_bands_n,
        )
        if any(not all(math.isfinite(v) for v in r) or r[1] < r[0] for r in ranges):
            raise ValueError("Impact ranges must be finite and ascending.")
        if min(c.first_hit_delay_s[0], c.recovery_pause_s[0]) < 0 or c.combo_gap_s[0] <= 0:
            raise ValueError("Impact delays must be nonnegative and combination gaps positive.")
        for probability, maximum in (
            (c.common_combo_probability, c.common_combo_max_hits),
            (c.stress_combo_probability, c.stress_combo_max_hits),
        ):
            if (
                not 0 <= probability <= 1
                or type(maximum) is not int
                or maximum < 1
                or (probability > 0 and maximum < 2)
            ):
                raise ValueError("Combination probabilities or maximum lengths are invalid.")
        if (
            not 0 < c.recovery_min_scale <= 1
            or min(
                c.recovery_recent_hit_s,
                c.recovery_max_s,
                c.recovery_stable_hold_s,
                c.recovery_rearm_stable_s,
                c.measurement_window_s,
                c.recovery_tilt_threshold_rad,
                c.recovery_angular_speed_threshold,
                c.recovery_velocity_error_threshold,
            )
            <= 0
        ):
            raise ValueError("Recovery scales, windows and thresholds must be positive.")
        if any(
            v is not None and (not math.isfinite(v) or v <= 0)
            for v in (c.fixed_peak_n, c.fixed_duration_s, c.fixed_interval_s)
        ):
            raise ValueError("Fixed impact parameters must be positive and finite.")
        if c.fixed_peak_n is not None and c.fixed_peak_n > 5000:
            raise ValueError("Fixed peak exceeds the 5000 N stress envelope.")
        for t in c.targets:
            if (
                t.group not in self.GROUPS
                or t.weight <= 0
                or min(t.point_half_extent_m) < 0
                or t.duration_s[0] <= 0
                or t.duration_s[1] < t.duration_s[0]
                or min(t.common_impulse_limit_ns, t.stress_impulse_limit_ns) <= 0
            ):
                raise ValueError(f"Invalid impact target {t.name}.")
        if not c.targets or len({t.name for t in c.targets}) != len(c.targets):
            raise ValueError("Impact targets must be nonempty and unique.")

    def _config_signature(self, *, legacy_stress_combo: bool = False) -> str:
        settings = asdict(self.cfg)
        if legacy_stress_combo:
            settings["stress_combo_probability"] = 0.70
            settings["stress_combo_max_hits"] = 5
        # Evaluation selection is intentionally independent of training state.
        for key in (
            "enabled",
            "enabled_eval",
            "mode",
            "eval_strength",
            "fixed_peak_n",
            "fixed_duration_s",
            "fixed_interval_s",
            "head_mode",
        ):
            settings.pop(key)
        return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()

    def validate_checkpoint_signature(self, signature: Any) -> bool:
        """Return whether the exact known old stress distribution needs migration."""
        if signature == self.config_signature:
            return False
        if self.legacy_stress_combo_signature is not None and signature == self.legacy_stress_combo_signature:
            if self.cfg.mode == "train" and not self.allow_legacy_stress_combo_migration:
                raise ValueError(
                    "Impact checkpoint uses the old stress combination distribution. "
                    "Pass --impact-single-hit-migration to explicitly continue with isolated stress hits."
                )
            return True
        raise ValueError("Impact training configuration mismatch; refusing to reset or reinterpret the curriculum.")

    def set_update(self, completed_updates: int) -> None:
        if isinstance(completed_updates, bool) or not isinstance(completed_updates, int) or completed_updates < 0:
            raise ValueError("completed_updates must be a nonnegative integer.")
        self.completed_updates = completed_updates

    @property
    def curriculum_progress(self) -> float:
        if self.cfg.mode != "train":
            return 1.0
        if self.completed_updates < self.cfg.warmup_updates:
            return 0.0
        if self.cfg.ramp_updates == 0:
            return 1.0
        return min(1.0, (self.completed_updates - self.cfg.warmup_updates) / self.cfg.ramp_updates)

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled and self.cfg.mode != "off" and (self.cfg.mode == "train" or self.cfg.enabled_eval)

    def _uniform(self, n: int, bounds: tuple[float, float]) -> torch.Tensor:
        return bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n, device=self.device, generator=self._generator)

    def _ids(self, env_ids: Any) -> torch.Tensor:
        if env_ids is None or isinstance(env_ids, slice):
            return self._all_ids if env_ids is None else self._all_ids[env_ids]
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()

    def reset(self, env_ids: Any) -> None:
        ids = self._ids(env_ids)
        if ids.numel() == 0:
            return
        n = ids.numel()
        draw = self._uniform(n, (0.0, 1.0))
        stress_probability = self.cfg.stress_episode_probability * self.curriculum_progress
        kind = torch.where(
            draw < self.cfg.clean_episode_probability, 0, torch.where(draw >= 1.0 - stress_probability, 2, 1)
        )
        if self.cfg.mode in ("common", "stress"):
            kind.fill_(1 if self.cfg.mode == "common" else 2)
        self.episode_kind[ids] = kind
        self.active[ids] = False
        self.combo_remaining[ids] = 0
        self.age[ids] = self.peak_n[ids] = 0
        self.cooldown[ids] = self._uniform(n, self.cfg.first_hit_delay_s)
        self._since_hit[ids] = float("inf")
        self._recovering[ids] = False
        self._recovery_armed[ids] = True
        self._recovery_age[ids] = self._stable_age[ids] = 0
        self._recovery_scale[ids] = 1
        # last_step_style_scale belongs to the pre-reset reward transition and
        # must survive reset until the caller passes it to the AMP hook.
        self._measurement_remaining[ids] = self._measurement_error[ids] = 0
        for values in self._stats.values():
            values[ids] = 0
        self._direction_hits[ids] = self._group_hits[ids] = 0
        self.forces_b[ids] = self.torques_b[ids] = 0
        self.robot.instantaneous_wrench_composer.reset(env_ids=ids)

    def _start_hits(self, ids: torch.Tensor) -> None:
        n = ids.numel()
        target = torch.multinomial(self._target_weights, n, replacement=True, generator=self._generator)
        self.target_id[ids] = target
        self.body_id[ids] = self._target_body[target]
        self.hit_point_b[ids] = (
            2.0 * torch.rand((n, 3), device=self.device, generator=self._generator) - 1.0
        ) * self._point_extents[target]
        duration = self._durations[target, 0] + self._uniform(n, (0, 1)) * (
            self._durations[target, 1] - self._durations[target, 0]
        )
        if self.cfg.mode != "train" and self.cfg.fixed_duration_s is not None:
            duration.fill_(self.cfg.fixed_duration_s)
            # Fixed evaluation width is still constrained by the hit site's
            # calibrated envelope, as is the requested peak below.
            duration = duration.maximum(self._durations[target, 0]).minimum(self._durations[target, 1])
        self.duration[ids] = duration
        band = torch.multinomial(self._band_weights, n, replacement=True, generator=self._generator)
        common_peak = self._peak_bands[band, 0] + self._uniform(n, (0, 1)) * (
            self._peak_bands[band, 1] - self._peak_bands[band, 0]
        )
        stress = self.episode_kind[ids] == 2
        peak = torch.where(stress, self._uniform(n, self.cfg.stress_peak_n), common_peak)
        strength = (
            self.cfg.eval_strength
            if self.cfg.mode != "train"
            else self.cfg.initial_strength + (1.0 - self.cfg.initial_strength) * self.curriculum_progress
        )
        peak = 40.0 + (peak - 40.0) * strength
        if self.cfg.mode != "train" and self.cfg.fixed_peak_n is not None:
            peak.fill_(self.cfg.fixed_peak_n)
        # Common episodes never acquire an extreme hit through a fixed setting.
        peak = peak.clamp_min(0.0).minimum(torch.where(stress, 5000.0, 2000.0))
        limit = self._impulse_limits[target, stress.long()]
        self.peak_n[ids] = peak.minimum(limit * math.pi / (2.0 * duration))
        sectors = torch.multinomial(self._direction_weights, n, replacement=True, generator=self._generator)
        self.direction_sector[ids] = sectors
        magnitude = torch.where(
            sectors == 0,
            self._uniform(n, (0, 60)),
            torch.where(sectors == 1, self._uniform(n, (60, 140)), self._uniform(n, (140, 180))),
        )
        signs = torch.where(self._uniform(n, (0, 1)) < 0.5, -1.0, 1.0)
        bearing = _yaw(self.robot.data.root_quat_w[ids]) + torch.deg2rad(
            magnitude * signs + self.cfg.tactical_front_offset_deg
        )
        elevation = torch.deg2rad(self._uniform(n, self.cfg.direction_elevation_deg))
        # Bearing is the opponent's location; force travels towards the robot.
        self.force_direction_w[ids] = torch.stack(
            (
                -torch.cos(bearing) * torch.cos(elevation),
                -torch.sin(bearing) * torch.cos(elevation),
                torch.sin(elevation),
            ),
            dim=-1,
        )
        new_sequence = self.combo_remaining[ids] == 0
        combo_probability = torch.where(stress, self.cfg.stress_combo_probability, self.cfg.common_combo_probability)
        maximum = torch.where(stress, self.cfg.stress_combo_max_hits, self.cfg.common_combo_max_hits)
        length = (2 + torch.floor(self._uniform(n, (0, 1)) * (maximum - 1)).long()).minimum(maximum)
        remaining = torch.where(self._uniform(n, (0, 1)) < combo_probability, length - 1, 0)
        self.combo_remaining[ids] = torch.where(new_sequence, remaining, self.combo_remaining[ids] - 1)
        self._onset_gap[ids] = self._uniform(n, self.cfg.combo_gap_s).maximum(duration + self.dt)
        self.age[ids] = 0
        self.active[ids] = True
        self._since_hit[ids] = 0
        first_measurement = self._measurement_remaining[ids] <= 0
        first_ids = ids[first_measurement]
        # Prediction is for the *next* physical state, so command-following
        # motion itself does not look like a one-physics-step displacement.
        self._expected_xy[first_ids] = (
            self.robot.data.root_pos_w[first_ids, :2] + self._world_command_velocity()[first_ids] * self.dt
        )
        self._measurement_error[first_ids] = 0
        self._measurement_remaining[first_ids] = self.cfg.measurement_window_s
        self._stats["hits"][ids] += 1
        self._stats["common_hits"][ids] += (~stress).float()
        self._stats["stress_hits"][ids] += stress.float()
        self._stats["sampled_peak_n"][ids] = torch.maximum(self._stats["sampled_peak_n"][ids], self.peak_n[ids])
        self._direction_hits[ids, sectors] += 1
        self._group_hits[ids, self._target_groups[target]] += 1
        if self.hit_observer is not None:
            self.hit_observer(ids)

    def _commands(self) -> torch.Tensor:
        # The walking command is already in the gravity-aligned root-yaw frame.
        return self.env.command_manager.get_command(self.cfg.command_name)

    def _world_command_velocity(self, command: torch.Tensor | None = None) -> torch.Tensor:
        command = self._commands() if command is None else command
        yaw = _yaw(self.robot.data.root_quat_w)
        vx, vy = command[:, 0], command[:, 1]
        return torch.stack(
            (torch.cos(yaw) * vx - torch.sin(yaw) * vy, torch.sin(yaw) * vx + torch.cos(yaw) * vy), dim=-1
        )

    def _update_recovery(self) -> None:
        data, c = self.robot.data, self.cfg
        command = self._commands()
        gravity = data.projected_gravity_b
        tilt = torch.atan2(torch.linalg.vector_norm(gravity[:, :2], dim=-1), -gravity[:, 2])
        angular = torch.linalg.vector_norm(data.root_ang_vel_b[:, :2], dim=-1)
        world_velocity = self._world_command_velocity(command)
        velocity_error = torch.linalg.vector_norm(data.root_lin_vel_w[:, :2] - world_velocity, dim=-1)
        severity = torch.maximum(
            tilt / c.recovery_tilt_threshold_rad,
            torch.maximum(
                angular / c.recovery_angular_speed_threshold, velocity_error / c.recovery_velocity_error_threshold
            ),
        )
        stable = severity < 1.0
        self._stable_age = torch.where(stable, self._stable_age + self.dt, 0.0)
        self._recovery_armed |= self._stable_age >= c.recovery_rearm_stable_s
        start = (~self._recovering) & self._recovery_armed & (self._since_hit <= c.recovery_recent_hit_s) & (~stable)
        self._recovering |= start
        self._recovery_age[start] = 0
        self._recovery_armed[start] = False
        self._recovery_age += self._recovering * self.dt
        recovered = self._recovering & (self._stable_age >= c.recovery_stable_hold_s)
        expired = self._recovering & (self._recovery_age >= c.recovery_max_s)
        self._stats["recoveries"] += recovered
        self._stats["recovery_time_s"] += self._recovery_age * recovered
        self._stats["recovery_expired"] += expired & (~recovered)
        self._recovering &= ~(recovered | expired)
        # Deadline is attached to the first destabilization, never extended by
        # another hit. After expiry, stable rearming is required for a new gate.
        loosen = (severity - 1.0).clamp(0.0, 1.0)
        self._recovery_scale = torch.where(self._recovering, 1.0 - (1.0 - c.recovery_min_scale) * loosen, 1.0)
        measuring = self._measurement_remaining > 0
        error = torch.linalg.vector_norm(data.root_pos_w[:, :2] - self._expected_xy, dim=-1)
        self._measurement_error = torch.where(
            measuring, torch.maximum(self._measurement_error, error), self._measurement_error
        )
        self._stats["extra_displacement_m"] = torch.maximum(
            self._stats["extra_displacement_m"], self._measurement_error
        )
        self._expected_xy += world_velocity * self.dt * measuring[:, None]
        ended = measuring & (self._measurement_remaining <= self.dt)
        self._stats["measurement_count"] += ended
        self._measurement_remaining = (self._measurement_remaining - self.dt).clamp_min(0)
        self._since_hit += self.dt

    def physics_step(self) -> None:
        self.forces_b.zero_()
        self.torques_b.zero_()
        self._update_recovery()
        if not self.enabled or (self.cfg.mode == "train" and self.completed_updates < self.cfg.warmup_updates):
            self.active.zero_()
            self.combo_remaining.zero_()
        else:
            self.cooldown = torch.where(self.active, self.cooldown, self.cooldown - self.dt)
            due = (~self.active) & (self.cooldown <= 0) & (self.episode_kind != 0)
            ids = self._all_ids[due]
            if ids.numel():
                self._start_hits(ids)
            ids = self._all_ids[self.active]
            if ids.numel():
                impulse = half_sine_step_impulse(self.peak_n[ids], self.duration[ids], self.age[ids], self.dt)
                average = impulse / self.dt
                force_w = average[:, None] * self.force_direction_w[ids]
                body_quats = self.robot.data.body_link_quat_w[ids, self.body_id[ids]]
                force_b = _quat_rotate(body_quats, force_w, inverse=True)
                self.forces_b[ids, self.body_id[ids]] = force_b
                self.torques_b[ids, self.body_id[ids]] = torch.cross(self.hit_point_b[ids], force_b, dim=-1)
                self._stats["impulse_ns"][ids] += impulse
                self._stats["peak_n"][ids] = torch.maximum(self._stats["peak_n"][ids], average)
                self.age[ids] += self.dt
                done = ids[self.age[ids] + 1e-7 >= self.duration[ids]]
                if done.numel():
                    self.active[done] = False
                    pause = self._uniform(done.numel(), self.cfg.recovery_pause_s)
                    if self.cfg.mode != "train" and self.cfg.fixed_interval_s is not None:
                        pause.fill_(self.cfg.fixed_interval_s)
                    self.cooldown[done] = torch.where(
                        self.combo_remaining[done] > 0,
                        (self._onset_gap[done] - self.duration[done]).clamp_min(self.dt),
                        pause,
                    )
        # One full composition also zeroes every body after pulse completion.
        self.robot.instantaneous_wrench_composer.set_forces_and_torques(
            forces=self.forces_b, torques=self.torques_b, is_global=False
        )

    def recovery_scale(self) -> torch.Tensor:
        return self._recovery_scale

    def log_reset(self, env_ids: Any) -> dict[str, float]:
        ids = self._ids(env_ids)
        if not ids.numel():
            return {}
        stats = self._stats
        hits = stats["hits"][ids].sum().clamp_min(1)
        recoveries = stats["recoveries"][ids].sum().clamp_min(1)
        result = {
            f"Impact/{key}": values[ids].mean().item()
            for key, values in stats.items()
            if key not in ("recovery_time_s", "measurement_count")
        }
        result["Impact/mean_recovery_time_s"] = (stats["recovery_time_s"][ids].sum() / recoveries).item()
        result["Impact/clean_episode_fraction"] = (self.episode_kind[ids] == 0).float().mean().item()
        result["Impact/curriculum_progress"] = self.curriculum_progress
        terminated = getattr(self.env, "reset_terminated", None)
        if terminated is not None:
            hit_episodes = stats["hits"][ids] > 0
            result["Impact/hit_episode_failure_fraction"] = (
                (hit_episodes & terminated[ids]).sum() / hit_episodes.sum().clamp_min(1)
            ).item()
            for kind, name in enumerate(("clean", "common", "stress")):
                selected = self.episode_kind[ids] == kind
                result[f"Impact/{name}_failure_fraction"] = (
                    (terminated[ids] & selected).sum() / selected.sum().clamp_min(1)
                ).item()
        for i, name in enumerate(self.DIRECTIONS):
            result[f"Impact/direction_{name}_fraction"] = (self._direction_hits[ids, i].sum() / hits).item()
        for i, name in enumerate(self.GROUPS):
            result[f"Impact/target_{name}_fraction"] = (self._group_hits[ids, i].sum() / hits).item()
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "config_signature": self.config_signature,
            "completed_updates": self.completed_updates,
            "source_rank": self.rank,
            "generator_device": self.device.type,
            "generator_state": self._generator.get_state().cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != self.STATE_VERSION:
            raise ValueError(
                f"Impact state version mismatch: expected {self.STATE_VERSION}, got {state.get('version')}."
            )
        self.validate_checkpoint_signature(state.get("config_signature"))
        different_device = state.get("generator_device") != self.device.type
        if different_device and self.cfg.mode == "train":
            raise ValueError(
                "Impact RNG device differs; exact curriculum RNG restoration requires the same device type."
            )
        self.set_update(state["completed_updates"])
        source_rank = state.get("source_rank")
        if not isinstance(source_rank, int) or source_rank < 0:
            raise ValueError("Impact checkpoint is missing a valid RNG source_rank.")
        if source_rank == self.rank and not different_device:
            self._generator.set_state(state["generator_state"].cpu())
        else:
            # CusRL saves rank zero only. Copying its generator to every rank
            # would correlate all perturbations. Other ranks deterministically
            # derive a distinct stream. Evaluation on a different device also
            # uses a fresh deterministic stream; it is not a training replay.
            seed = (self._base_seed + self.completed_updates * 0x9E3779B1) % (2**63 - 1)
            self._generator.manual_seed(seed)
