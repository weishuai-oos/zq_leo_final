"""Physics-step ankle costs, accumulated before the control-step reward pass."""

from __future__ import annotations

import math

import torch


class AnkleStabilityAccumulator:
    """Keep actual motion visible even when it cancels between policy samples."""

    def __init__(self, num_envs, device, settings, physics_dt, decimation):
        self.settings = dict(settings)
        if not all(math.isfinite(float(v)) and float(v) >= 0 for v in settings.values()):
            raise ValueError("Ankle stability settings must be finite and nonnegative.")
        for key in ("acceleration_scale_rad_s2", "foot_roll_scale_rad_s", "foot_pitch_scale_rad_s", "normalized_error_cap"):
            if settings[key] <= 0:
                raise ValueError(f"{key} must be positive.")
        if settings["contact_force_enter_n"] <= settings["contact_force_exit_n"]:
            raise ValueError("Contact entry force must exceed exit force.")
        if physics_dt <= 0 or type(decimation) is not int or decimation < 1:
            raise ValueError("A positive physics dt and integer decimation are required.")
        self.physics_dt = float(physics_dt)
        self.decimation = decimation
        self.support_age = torch.zeros(num_envs, 2, device=device)
        self._contact_gap = torch.zeros_like(self.support_age)
        self._previous_velocity = torch.zeros(num_envs, 4, device=device)
        self._acceleration_sum = torch.zeros(num_envs, device=device)
        self._rotation_sum = torch.zeros_like(self._acceleration_sum)
        self._acceleration_square_sum = torch.zeros_like(self._acceleration_sum)
        self._angular_square_sum = torch.zeros(num_envs, 2, device=device)
        self._support_samples = torch.zeros_like(self._acceleration_sum)
        self._angular_deadband = torch.tensor(
            [settings["foot_roll_deadband_rad_s"], settings["foot_pitch_deadband_rad_s"]], device=device,
        )
        self._angular_scale = torch.tensor(
            [settings["foot_roll_scale_rad_s"], settings["foot_pitch_scale_rad_s"]], device=device,
        )
        self.sample_count = 0
        self._last_step = None

    def begin(self, joint_velocity, step_counter):
        self._previous_velocity.copy_(joint_velocity)
        self._last_step = int(step_counter)
        self.sample_count = 0
        for value in (self._acceleration_sum, self._rotation_sum, self._acceleration_square_sum,
                      self._angular_square_sum, self._support_samples):
            value.zero_()

    def reset(self, env_ids=None):
        # The reward pass precedes autoreset. Retain this control step's costs
        # and global sample count for logging; begin() starts its successor.
        ids = slice(None) if env_ids is None else env_ids
        self.support_age[ids] = 0
        self._contact_gap[ids] = 0

    def _bounded_square(self, normalized):
        cap = self.settings["normalized_error_cap"]
        return torch.nan_to_num(normalized, nan=cap, posinf=cap, neginf=cap).clamp(0, cap).square()

    def observe(self, joint_velocity, foot_ang_vel_w, foot_quat_w, contact_force_w, step_counter, dt):
        if not math.isclose(float(dt), self.physics_dt, rel_tol=0, abs_tol=1e-9):
            raise RuntimeError("Ankle statistics require one physics step per observation.")
        if step_counter == self._last_step:
            return
        if self._last_step is None or step_counter != self._last_step + 1:
            raise RuntimeError("Ankle statistics skipped a physics sample.")
        if self.sample_count >= self.decimation:
            raise RuntimeError("Too many ankle physics samples in one control step.")
        cfg = self.settings
        acceleration = (joint_velocity - self._previous_velocity) / self.physics_dt
        excess = (acceleration.abs() - cfg["acceleration_deadband_rad_s2"]).clamp_min(0)
        self._acceleration_sum += self._bounded_square(excess / cfg["acceleration_scale_rad_s2"]).mean(dim=-1)
        self._acceleration_square_sum += torch.nan_to_num(
            acceleration, nan=1e5, posinf=1e5, neginf=-1e5,
        ).clamp(-1e5, 1e5).square().mean(dim=-1)

        # Hysteresis preserves support age through short force dropouts, but
        # neither rewards nor the settling timer accumulate during those gaps.
        force_z = contact_force_w[..., 2]
        contact = (force_z >= cfg["contact_force_enter_n"]) | (
            (self.support_age > 0) & (force_z > cfg["contact_force_exit_n"])
        )
        self._contact_gap = torch.where(contact, 0.0, self._contact_gap + self.physics_dt)
        self.support_age = torch.where(
            contact, self.support_age + self.physics_dt,
            torch.where(self._contact_gap <= cfg["contact_gap_tolerance_s"] + 1e-7, self.support_age, 0.0),
        )
        settled = contact & (self.support_age + 1e-7 >= cfg["support_settle_time_s"])

        # Rotate world angular velocity into the foot frame. Penalize changes
        # of the sole normal (roll/pitch), while allowing yaw about that normal.
        xyz = foot_quat_w[..., 1:]
        cross = torch.cross(xyz, foot_ang_vel_w, dim=-1)
        local_omega = foot_ang_vel_w - 2 * foot_quat_w[..., :1] * cross + 2 * torch.cross(xyz, cross, dim=-1)
        angular_excess = (local_omega[..., :2].abs() - self._angular_deadband).clamp_min(0)
        rotation = self._bounded_square(angular_excess / self._angular_scale).mean(dim=-1)
        self._rotation_sum += (rotation * settled).mean(dim=-1)
        self._support_samples += settled.sum(dim=-1)
        self._angular_square_sum += (
            torch.nan_to_num(local_omega[..., :2], nan=1e3, posinf=1e3, neginf=-1e3)
            .clamp(-1e3, 1e3).square() * settled.unsqueeze(-1)
        ).sum(dim=1)
        self._previous_velocity.copy_(joint_velocity)
        self._last_step = int(step_counter)
        self.sample_count += 1

    def values(self):
        if self.sample_count != self.decimation:
            raise RuntimeError(f"Expected {self.decimation} ankle physics samples, got {self.sample_count}.")
        return self._acceleration_sum / self.sample_count, self._rotation_sum / self.sample_count

    def metrics(self):
        acceleration, rotation = self.values()
        support_count = self._support_samples.sum().clamp_min(1)
        angular_rms = (self._angular_square_sum.sum(dim=0) / support_count).sqrt()
        return {
            "AnkleV2/physics_acceleration_cost": acceleration.mean(),
            "AnkleV2/support_foot_rotation_cost": rotation.mean(),
            "AnkleV2/physics_acceleration_rms_rad_s2": (self._acceleration_square_sum.mean() / self.sample_count).sqrt(),
            "AnkleV2/support_fraction": self._support_samples.mean() / (2 * self.sample_count),
            "AnkleV2/support_foot_roll_rms_rad_s": angular_rms[0],
            "AnkleV2/support_foot_pitch_rms_rad_s": angular_rms[1],
        }


def physics_acceleration_l2(env):
    return env.ankle_stability.values()[0]


def support_foot_angular_velocity_l2(env):
    return env.ankle_stability.values()[1]
