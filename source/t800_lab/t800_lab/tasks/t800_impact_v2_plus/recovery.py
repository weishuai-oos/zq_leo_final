"""Physical recovery states; no simulator imports or policy observation flags."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class RecoveryConfig:
    dt: float = 0.005
    max_s: float = 2.5
    stable_hold_s: float = 0.25
    rearm_stable_s: float = 0.40
    guard_ramp_s: float = 0.50
    guard_hold_s: float = 0.25
    guard_error_rad: float = 0.25
    min_scale: float = 0.10
    unstable_tilt_rad: float = 0.18
    unstable_angular_speed: float = 1.0
    unstable_velocity_error: float = 0.55
    unassisted_trigger_tilt_rad: float = 0.35
    unassisted_trigger_angular_speed: float = 2.0
    balance_height_m: float = 0.75
    low_height_m: float = 0.45
    knockdown_tilt_rad: float = math.radians(80.0)
    persistent_s: float = 0.30
    body_persistent_s: float = 0.15
    severe_tilt_s: float = 0.30

    def __post_init__(self):
        if any(not math.isfinite(value) or value <= 0 for value in asdict(self).values()):
            raise ValueError("Recovery thresholds and durations must be positive and finite.")
        if self.min_scale > 1 or self.low_height_m >= self.balance_height_m:
            raise ValueError("Invalid recovery scale or height hysteresis.")


class RecoveryController:
    """Balance first, then restore the guard before the original deadline.

    All timings use physics samples. Foot-side and nonfoot support contacts
    request recovery; only persistent body contact, low height, or severe tilt
    count as knockdown. Timeout is a separate latched training termination.
    ``recovered`` and ``balance_recovered`` are one-sample event pulses.
    """

    def __init__(self, num_envs: int, device: torch.device | str, cfg: RecoveryConfig | None = None):
        if num_envs < 1:
            raise ValueError("Recovery needs at least one environment.")
        self.cfg = cfg or RecoveryConfig()
        self.device = torch.device(device)
        self.n = int(num_envs)
        for name in ("active", "recovering", "guard_returning", "timeout", "knockdown", "recovered", "balance_recovered"):
            setattr(self, name, torch.zeros(self.n, device=self.device, dtype=torch.bool))
        self.armed = torch.ones(self.n, device=self.device, dtype=torch.bool)
        for name in ("age", "stable_age", "rearm_age", "knockdown_age", "body_contact_age", "severe_tilt_age", "guard_age", "guard_hold_age"):
            setattr(self, name, torch.zeros(self.n, device=self.device))
        self.scale = torch.ones(self.n, device=self.device)
        self.guard_blend = torch.ones(self.n, device=self.device)

    def reset(self, ids=None):
        ids = torch.arange(self.n, device=self.device) if ids is None else torch.as_tensor(ids, device=self.device, dtype=torch.long).flatten()
        for name in ("active", "recovering", "guard_returning", "timeout", "knockdown", "recovered", "balance_recovered"):
            getattr(self, name)[ids] = False
        self.armed[ids] = True
        for name in ("age", "stable_age", "rearm_age", "knockdown_age", "body_contact_age", "severe_tilt_age", "guard_age", "guard_hold_age"):
            getattr(self, name)[ids] = 0.0
        self.scale[ids] = self.guard_blend[ids] = 1.0

    def _vector(self, value, *, boolean=False):
        tensor = torch.as_tensor(value, device=self.device, dtype=torch.bool if boolean else torch.float32)
        if tensor.ndim == 0:
            tensor = tensor.expand(self.n)
        if tensor.shape != (self.n,):
            raise ValueError(f"Recovery requires [{self.n}] physical measurements.")
        # GPU synchronization is avoided here; the rollout checker validates
        # nonfinite observations. CPU numerical contract tests reject them.
        if self.device.type == "cpu" and not boolean and not torch.isfinite(tensor).all():
            raise ValueError("Recovery physical measurements must be finite.")
        return tensor

    def step(self, tilt, angular_speed, velocity_error, contact=None, persistent_body_contact=None,
             low_height=None, height=None, recent_hit=None, guard_error=None):
        c, dt = self.cfg, self.cfg.dt
        tilt, angular_speed, velocity_error = (self._vector(value) for value in (tilt, angular_speed, velocity_error))
        height = self._vector(c.balance_height_m if height is None else height)
        guard_error = self._vector(0.0 if guard_error is None else guard_error)
        contact = self._vector(False if contact is None else contact, boolean=True)
        body = self._vector(False if persistent_body_contact is None else persistent_body_contact, boolean=True)
        low = (height < c.low_height_m) | self._vector(False if low_height is None else low_height, boolean=True)
        recent_hit = self._vector(False if recent_hit is None else recent_hit, boolean=True)
        self.recovered.zero_()
        self.balance_recovered.zero_()

        self.knockdown_age = torch.where(low, self.knockdown_age + dt, 0.0)
        self.body_contact_age = torch.where(body, self.body_contact_age + dt, 0.0)
        self.severe_tilt_age = torch.where(tilt >= c.knockdown_tilt_rad, self.severe_tilt_age + dt, 0.0)
        self.knockdown |= ((self.knockdown_age + dt * 1e-3 >= c.persistent_s)
                           | (self.body_contact_age + dt * 1e-3 >= c.body_persistent_s)
                           | (self.severe_tilt_age + dt * 1e-3 >= c.severe_tilt_s))
        failed = self.knockdown | self.timeout
        disturbed = ((tilt >= c.unstable_tilt_rad) | (angular_speed >= c.unstable_angular_speed)
                     | (velocity_error >= c.unstable_velocity_error))
        balanced = (~disturbed) & (height >= c.balance_height_m) & (~contact) & (~body) & (~low)
        trigger = (contact | body | low | (recent_hit & disturbed)
                   | (tilt >= c.unassisted_trigger_tilt_rad)
                   | (angular_speed >= c.unassisted_trigger_angular_speed))
        start = (~self.active) & self.armed & trigger & (~failed)
        self.active |= start
        self.recovering |= start
        self.armed[start] = False
        self.age[start] = self.stable_age[start] = self.guard_age[start] = self.guard_hold_age[start] = 0.0
        self.age += self.active.float() * dt

        # A relapse returns to balance recovery without resetting event age.
        relapse = self.guard_returning & (~balanced)
        self.guard_returning &= ~relapse
        self.recovering |= relapse
        self.guard_age[relapse] = self.guard_hold_age[relapse] = self.stable_age[relapse] = 0.0
        self.stable_age = torch.where(self.recovering & balanced, self.stable_age + dt, 0.0)
        self.balance_recovered = self.recovering & (self.stable_age + dt * 1e-3 >= c.stable_hold_s) & (~failed)
        self.recovering &= ~self.balance_recovered
        self.guard_returning |= self.balance_recovered
        self.guard_age = torch.where(self.guard_returning & balanced, self.guard_age + dt, self.guard_age)
        self.guard_hold_age = torch.where(self.guard_returning & balanced & (guard_error < c.guard_error_rad),
                                          self.guard_hold_age + dt, 0.0)
        success = (self.guard_returning & (self.guard_age + dt * 1e-3 >= c.guard_ramp_s)
                   & (self.guard_hold_age + dt * 1e-3 >= c.guard_hold_s) & (~failed))
        self.recovered = success
        self.active &= ~success
        self.guard_returning &= ~success
        self.rearm_age[success] = 0.0
        self.timeout |= self.active & (self.age + dt * 1e-3 >= c.max_s) & (~self.knockdown)
        failed = self.knockdown | self.timeout
        self.active &= ~failed
        self.recovering &= self.active
        self.guard_returning &= self.active
        self.rearm_age = torch.where((~self.active) & balanced & (~failed) & (~success), self.rearm_age + dt, 0.0)
        self.armed |= (self.rearm_age + dt * 1e-3 >= c.rearm_stable_s) & (~failed)
        self.armed &= ~failed
        blend = (self.guard_age / c.guard_ramp_s).clamp(0.0, 1.0)
        self.guard_blend = torch.where(self.active, torch.where(self.guard_returning, blend, 0.0), 1.0)
        self.scale = c.min_scale + (1.0 - c.min_scale) * self.guard_blend

    def state(self):
        return {name: getattr(self, name) for name in
                ("active", "recovering", "guard_returning", "scale", "guard_blend", "timeout", "knockdown", "recovered")}
