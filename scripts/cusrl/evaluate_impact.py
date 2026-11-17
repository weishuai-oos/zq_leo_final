"""Finite, first-fall T800 walking/impact benchmark; one fixed command per environment.

Run separately for each checkpoint and impact mode.  Metrics include the terminal
physical frame captured before IsaacLab autoresets, and exclude every later frame.
The module's numerical aggregation can be tested without launching Isaac Sim.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import traceback
import types
from hashlib import sha256

import numpy as np


COMMANDS = (
    ("stand", (0.0, 0.0, 0.0), 0),
    ("forward_0.6", (0.6, 0.0, 0.0), 1),
    ("forward_1.0", (1.0, 0.0, 0.0), 1),
    ("backward_0.6", (-0.6, 0.0, 0.0), 1),
    ("left_0.6", (0.0, 0.6, 0.0), 1),
    ("right_0.6", (0.0, -0.6, 0.0), 1),
    ("forward_left_0.6", (0.6 / math.sqrt(2), 0.6 / math.sqrt(2), 0.0), 3),
    ("forward_right_0.6", (0.6 / math.sqrt(2), -0.6 / math.sqrt(2), 0.0), 3),
    ("turn_left_0.75", (0.0, 0.0, 0.75), 2),
    ("turn_right_0.75", (0.0, 0.0, -0.75), 2),
    ("advance_turn_left", (0.6, 0.0, 0.75), 4),
    ("advance_turn_right", (0.6, 0.0, -0.75), 4),
)


RECOVERY_PROTOCOL = "recovery_v1"
RECOVERY_THRESHOLDS = {
    "knockdown_ground_s": 0.15,
    "knockdown_low_height_s": 0.30,
    "knockdown_height_m": 0.45,
    "knockdown_tilt_rad": math.radians(80.0),
    "balance_tilt_rad": 0.18,
    "balance_ang_speed_rad_s": 1.0,
    "balance_command_error_mps": 0.55,
    "balance_height_m": 0.75,
    "balance_hold_s": 0.25,
    "recovery_timeout_s": 2.5,
    "guard_error_rad": 0.25,
    "guard_hold_s": 0.25,
}


# Frozen benchmark load envelope, matching the V2/V2Plus training distribution.
# (body, group, weight, local point half extents, pulse duration, common/stress Ns)
IMPACT_TARGETS = (
    ("LINK_TORSO_YAW", "torso", 3.0, (.10, .13, .18), (.025, .080), 100., 170.),
    ("LINK_BASE", "pelvis", 1.2, (.10, .12, .10), (.025, .080), 100., 170.),
    ("LINK_HEAD_YAW", "head", .6, (.07, .07, .08), (.012, .035), 20., 40.),
    *((f"{body}_{side}", group, weight, extent, duration, common, stress)
      for side in ("L", "R")
      for body, group, weight, extent, duration, common, stress in (
          ("LINK_SHOULDER_ROLL", "arm", .7, (.06, .06, .10), (.018, .060), 55., 100.),
          ("LINK_SHOULDER_YAW", "arm", .8, (.06, .06, .12), (.018, .060), 55., 100.),
          ("LINK_ELBOW_YAW", "arm", 1., (.06, .06, .13), (.018, .060), 55., 100.),
          ("LINK_HIP_YAW", "thigh", .8, (.07, .07, .15), (.025, .090), 65., 110.),
          ("LINK_KNEE_PITCH", "calf", .6, (.06, .06, .17), (.025, .090), 45., 80.),
      )),
)


class ImpactEvent:
    """One fully specified pulse in a fixed initial-root tactical frame."""

    def __init__(self, env_index, onset_s, peak_n, duration_s, target, direction,
                 *, group=None, point_b=(0., 0., 0.), direction_initial=(1., 0., 0.),
                 sequence=0, sampled_peak_n=None, impulse_limit_ns=None):
        self.env_index, self.onset_s = int(env_index), float(onset_s)
        self.peak_n, self.duration_s = float(peak_n), float(duration_s)
        self.target, self.direction = target, direction
        self.group = group or target
        self.point_b = tuple(float(x) for x in point_b)
        self.direction_initial = tuple(float(x) for x in direction_initial)
        self.sequence = int(sequence)
        self.sampled_peak_n = self.peak_n if sampled_peak_n is None else float(sampled_peak_n)
        self.impulse_limit_ns = impulse_limit_ns

    def __eq__(self, other):
        return isinstance(other, ImpactEvent) and self.as_dict() == other.as_dict()

    def as_dict(self):
        return dict(vars(self))


def build_impact_plan(num_envs: int, duration_s: float, seed: int, mode: str):
    """Precompute all site/direction/point/force draws, independently per env.

    Common sequences have the original 30% two/three-hit combinations. Stress
    uses isolated hits; pause after each sequence is 2--5 s. Directions are
    frozen relative to initial root yaw plus the 37-degree tactical bearing,
    so a responding policy cannot rotate the future external load schedule.
    """
    if num_envs <= 0 or duration_s <= 0 or not math.isfinite(duration_s):
        raise ValueError("num_envs and duration_s must be positive and finite")
    if mode not in {"off", "common", "stress"} or seed < 0:
        raise ValueError("invalid impact mode or seed")
    rows = []
    weights = np.asarray([x[2] for x in IMPACT_TARGETS], dtype=float)
    weights /= weights.sum()
    for env_index in range(num_envs):
        if mode == "off":
            break
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), env_index, 7301]))
        onset, sequence = float(rng.uniform(1.5, 3.)), 0
        while onset < duration_s:
            length = int(rng.integers(2, 4)) if mode == "common" and rng.random() < .3 else 1
            for _ in range(length):
                body, group, _, extent, duration_range, common_cap, stress_cap = IMPACT_TARGETS[int(rng.choice(len(weights), p=weights))]
                duration = float(rng.uniform(*duration_range))
                if mode == "stress":
                    sampled_peak = float(rng.uniform(1600., 5000.))
                else:
                    band = ((40., 300.), (300., 900.), (900., 2000.))[int(rng.choice(3, p=(.35, .40, .25)))]
                    sampled_peak = float(rng.uniform(*band))
                cap = stress_cap if mode == "stress" else common_cap
                peak = min(sampled_peak, cap * math.pi / (2. * duration))
                sector = int(rng.choice(3, p=(.60, .35, .05)))
                magnitude = float(rng.uniform(*((0., 60.), (60., 140.), (140., 180.))[sector]))
                bearing = math.radians(37. + magnitude * (-1. if rng.random() < .5 else 1.))
                elevation = math.radians(float(rng.uniform(-15., 15.)))
                direction = (-math.cos(bearing) * math.cos(elevation), -math.sin(bearing) * math.cos(elevation), math.sin(elevation))
                point = tuple(rng.uniform(-1., 1., 3) * np.asarray(extent))
                if onset < duration_s:
                    rows.append(ImpactEvent(env_index, onset, peak, duration, body, ("front", "side", "rear")[sector],
                                            group=group, point_b=point, direction_initial=direction,
                                            sequence=sequence, sampled_peak_n=sampled_peak, impulse_limit_ns=cap))
                onset += max(float(rng.uniform(.15, .50)), duration + .005)
            # Runtime pauses from pulse end; remove the last unused onset gap.
            onset = rows[-1].onset_s + rows[-1].duration_s + float(rng.uniform(2., 5.)) if rows and rows[-1].env_index == env_index else duration_s
            sequence += 1
    events = tuple(sorted(rows, key=lambda e: (e.onset_s, e.env_index)))
    payload = {"version": 1, "mode": mode, "num_envs": num_envs, "duration_s": duration_s,
               "events": [e.as_dict() for e in events]}
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return events, digest


class RecoveryTrialState:
    """Neutral physical referee, with multiple bounded recovery events per trial."""

    def __init__(self, dt: float, thresholds=None):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        self.dt, self.t = float(dt), 0.
        self.thresholds = {**RECOVERY_THRESHOLDS, **(thresholds or {})}
        self.recovering = self.recovered = self.knocked_down = self.timeout = False
        self.balance_age = self.guard_age = self.ground_age = self.low_age = self.rearm_age = 0.
        self.event_time = self.recovery_time = self.guard_time = None
        self.contact_events = self.foot_side_events = self.support_events = 0
        self.legacy_first_terminal_s = None
        self._last_foot = self._last_support = False
        self._armed = True
        self.active = None
        self.events = []
        self.followup_hits_while_unresolved = 0

    def _finish(self, status):
        self.active["status"] = status
        self.active["end_s"] = self.t
        self.events.append(self.active)
        self.active = None
        self.recovering = False
        self.balance_age = self.guard_age = 0.

    def observe(self, *, tilt_rad, angular_speed_rad_s, command_error_mps, root_height_m,
                nonfoot_support=False, body_ground_contact=False, guard_error_rad=math.inf,
                impact=False, foot_side_contact=False, hand_knee_contact=False):
        values = (tilt_rad, angular_speed_rad_s, command_error_mps, root_height_m)
        if not all(math.isfinite(float(v)) for v in values) or math.isnan(float(guard_error_rad)):
            raise ValueError("recovery observations must be finite")
        if self.knocked_down:
            return {"knocked_down": True, "recovered": self.recovered, "timeout": self.timeout}
        self.t += self.dt
        c = self.thresholds
        support = bool(nonfoot_support or hand_knee_contact or body_ground_contact)
        foot = bool(foot_side_contact)
        self.foot_side_events += int(foot and not self._last_foot)
        self.support_events += int(support and not self._last_support)
        self.contact_events = self.foot_side_events + self.support_events
        self._last_foot, self._last_support = foot, support
        if (support or foot) and self.legacy_first_terminal_s is None:
            self.legacy_first_terminal_s = self.t
        self.ground_age = self.ground_age + self.dt if body_ground_contact else 0.
        low = root_height_m < c["knockdown_height_m"] or tilt_rad > c["knockdown_tilt_rad"]
        self.low_age = self.low_age + self.dt if low else 0.
        balanced = (tilt_rad < c["balance_tilt_rad"] and angular_speed_rad_s < c["balance_ang_speed_rad_s"]
                    and command_error_mps < c["balance_command_error_mps"] and root_height_m >= c["balance_height_m"]
                    and not support and not foot)
        unstable = not balanced
        if self.active is None:
            self.rearm_age = self.rearm_age + self.dt if balanced else 0.
            self._armed |= self.rearm_age + 1e-9 >= .40
            if impact and self._armed:
                self.active = {"onset_s": self.t - self.dt, "status": "pending", "hits": 1,
                               "destabilized": unstable, "balance_time_s": None, "guard_time_s": None}
                self.event_time = self.active["onset_s"]
                self.recovery_time = self.guard_time = None
                self.recovering, self._armed = True, False
                self.rearm_age = self.balance_age = self.guard_age = 0.
            elif impact:
                self.followup_hits_while_unresolved += 1
        elif impact:
            self.active["hits"] += 1
            self.balance_age = self.guard_age = 0.
        if self.ground_age + 1e-9 >= c["knockdown_ground_s"] or self.low_age + 1e-9 >= c["knockdown_low_height_s"]:
            self.knocked_down = True
            if self.active is not None:
                self._finish("knockdown")
            return {"knocked_down": True, "recovered": False, "timeout": self.timeout}
        if self.active is not None:
            self.active["destabilized"] |= unstable
            elapsed = self.t - self.active["onset_s"]
            prior_balance_held = self.balance_age + 1e-9 >= c["balance_hold_s"]
            self.balance_age = self.balance_age + self.dt if balanced else 0.
            balance_held = self.balance_age + 1e-9 >= c["balance_hold_s"]
            if balance_held and self.active["balance_time_s"] is None:
                self.active["balance_time_s"] = self.recovery_time = elapsed
            self.guard_age = self.guard_age + self.dt if balance_held and prior_balance_held and guard_error_rad < c["guard_error_rad"] else 0.
            if self.guard_age + 1e-9 >= c["guard_hold_s"]:
                self.active["guard_time_s"] = self.guard_time = elapsed
                self.recovered = True
                self._finish("recovered" if self.active["destabilized"] else "resisted")
            elif elapsed + 1e-9 >= c["recovery_timeout_s"]:
                self.timeout = True
                self._finish("timeout")
        return {"knocked_down": self.knocked_down, "recovered": self.recovered, "timeout": self.timeout}

    def report(self):
        events = deepcopy(self.events)
        if self.active is not None:
            censored = deepcopy(self.active)
            censored.update(status="censored_horizon", observed_s=self.t - censored["onset_s"])
            events.append(censored)
        return {"knocked_down": self.knocked_down, "recovered": self.recovered, "recovery_timeout": self.timeout,
                "contact_events": self.contact_events, "foot_side_events": self.foot_side_events,
                "support_events": self.support_events, "balance_recovery_time_s": self.recovery_time,
                "guard_return_time_s": self.guard_time, "legacy_first_terminal_s": self.legacy_first_terminal_s,
                "followup_hits_while_unresolved": self.followup_hits_while_unresolved, "events": events}


def trial_commands(num_envs: int):
    """Distribute independent fixed-command trials as evenly as possible."""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    selected = [COMMANDS[index % len(COMMANDS)] for index in range(num_envs)]
    return (
        [entry[0] for entry in selected],
        np.asarray([entry[1] for entry in selected], dtype=np.float64),
        np.asarray([entry[2] for entry in selected], dtype=np.int64),
    )


class TrialMetrics:
    """Aggregate actual samples, never averages of differently sized trials."""

    def __init__(self, num_envs: int, timestep: float, protocol: str = "legacy"):
        self.names, self.commands, self.modes = trial_commands(num_envs)
        self.dt = float(timestep)
        if protocol not in ("legacy", RECOVERY_PROTOCOL):
            raise ValueError(f"unknown evaluation protocol: {protocol}")
        self.protocol = protocol
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("timestep must be positive and finite")
        self.alive = np.ones(num_envs, dtype=bool)
        self.falls = np.zeros(num_envs, dtype=bool)
        self.frames: list[np.ndarray] = []
        self.valid: list[np.ndarray] = []
        self.impact_stats = [{} for _ in range(num_envs)]
        self.failure_terms = [[] for _ in range(num_envs)]
        self.recovery = [RecoveryTrialState(self.dt) for _ in range(num_envs)] if protocol == RECOVERY_PROTOCOL else None

    def add_recovery_observations(self, observations):
        """Feed neutral physical recovery measurements, one mapping per env."""
        if self.recovery is None:
            raise RuntimeError("recovery observations require protocol=recovery_v1")
        if len(observations) != len(self.names):
            raise ValueError("Expected one recovery observation mapping per environment")
        for i, (tracker, values) in enumerate(zip(self.recovery, observations, strict=True)):
            if self.alive[i]:
                tracker.observe(**values)

    def add(self, velocity, tilt, height, terminated, impact_stats, failure_terms=None):
        frame = np.column_stack((velocity, tilt, height)).astype(np.float64, copy=True)
        if frame.shape != (len(self.names), 5):
            raise ValueError("Expected velocity[N,3], tilt[N], height[N]")
        if not np.isfinite(frame[self.alive]).all():
            raise ValueError("Nonfinite physical state in a live evaluation trial")
        terminated = np.asarray(terminated, dtype=bool)
        if terminated.shape != self.alive.shape:
            raise ValueError("Expected terminated[N]")
        self.frames.append(frame)
        self.valid.append(self.alive.copy())
        for index in np.flatnonzero(self.alive):
            self.impact_stats[index] = deepcopy(impact_stats[index])
            if terminated[index] and failure_terms is not None:
                self.failure_terms[index] = list(failure_terms[index])
        self.falls |= self.alive & terminated
        self.alive &= ~terminated

    def _summarize(self, indices):
        indices = np.asarray(indices, dtype=int)
        frames = np.asarray(self.frames)[:, indices]
        valid = np.asarray(self.valid)[:, indices]
        samples = frames[valid]
        target = np.broadcast_to(self.commands[indices], frames[:, :, :3].shape)[valid]
        error = samples[:, :3] - target
        stats = [self.impact_stats[index] for index in indices]

        def total(key):
            return float(sum(stat.get(key, 0.0) for stat in stats))

        def maximum(key):
            return float(max((stat.get(key, 0.0) for stat in stats), default=0.0))

        recoveries = total("recoveries")
        hit_counts = {
            name: total("direction_" + name) for name in ("front", "side", "rear")
        }
        target_counts = {
            name: total("target_" + name) for name in ("torso", "pelvis", "arm", "thigh", "calf", "head")
        }
        result = {
            "trials": len(indices),
            "falls": int(self.falls[indices].sum()),
            "failure_rate": float(self.falls[indices].mean()),
            "valid_samples": int(valid.sum()),
            "alive_seconds_total": float(valid.sum() * self.dt),
            "alive_seconds_mean": float(valid.sum() * self.dt / len(indices)),
            "actual_velocity_mean_tactical": samples[:, :3].mean(axis=0).tolist(),
            "tracking_rmse_xyz": np.sqrt(np.mean(error**2, axis=0)).tolist(),
            "tracking_rmse_xy": float(np.sqrt(np.mean(np.sum(error[:, :2] ** 2, axis=-1)))),
            "tracking_rmse_yaw": float(np.sqrt(np.mean(error[:, 2] ** 2))),
            "tilt_p95_rad": float(np.quantile(samples[:, 3], 0.95)),
            "tilt_max_rad": float(samples[:, 3].max()),
            "height_mean_m": float(samples[:, 4].mean()),
            "height_min_m": float(samples[:, 4].min()),
            "impacts": {
                "hits": int(total("hits")),
                "common_hits": int(total("common_hits")),
                "stress_hits": int(total("stress_hits")),
                "impulse_total_ns": total("impulse_ns"),
                "applied_physics_step_force_max_n": maximum("peak_n"),
                "continuous_pulse_peak_max_n": maximum("sampled_peak_n"),
                "directions": hit_counts,
                "targets": target_counts,
                "recoveries": int(recoveries),
                "recovery_expired": int(total("recovery_expired")),
                "completed_recovery_mean_s": total("recovery_time_s") / recoveries if recoveries else None,
                "extra_displacement_max_m": maximum("extra_displacement_m"),
                "completed_measurement_windows": int(total("measurement_count")),
            },
        }
        if self.recovery is not None:
            selected = [self.recovery[index].report() for index in indices]
            events = [event for item in selected for event in item["events"]]
            finished = [event for event in events if event["status"] != "censored_horizon"]
            successful = [event for event in finished if event["status"] in ("resisted", "recovered")]
            disturbed = [event for event in finished if event["destabilized"]]
            restored = [event for event in successful if event["destabilized"]]
            balance_times = [event["balance_time_s"] for event in events if event["balance_time_s"] is not None]
            guard_times = [event["guard_time_s"] for event in successful]
            result["recovery_v1"] = {
                "knocked_down": int(sum(item["knocked_down"] for item in selected)),
                "knockdown_rate": float(sum(item["knocked_down"] for item in selected) / len(selected)),
                "recovered": len(restored), "resisted_without_destabilization": len(successful) - len(restored),
                "recovery_timeout": sum(event["status"] == "timeout" for event in events),
                "censored_at_horizon": sum(event["status"] == "censored_horizon" for event in events),
                "events": len(events), "finished_events": len(finished), "destabilized_finished_events": len(disturbed),
                "recovery_rate": len(restored) / len(disturbed) if disturbed else None,
                "guard_return_rate": len(successful) / len(finished) if finished else None,
                "contact_events": int(sum(item["contact_events"] for item in selected)),
                "foot_side_events": int(sum(item["foot_side_events"] for item in selected)),
                "support_events": int(sum(item["support_events"] for item in selected)),
                "legacy_contact_failures": sum(item["legacy_first_terminal_s"] is not None for item in selected),
                "balance_recovery_mean_s": float(np.mean(balance_times)) if balance_times else None,
                "guard_return_mean_s": float(np.mean(guard_times)) if guard_times else None,
                "trial_details": selected,
            }
        return result

    def report(self):
        if not self.frames:
            raise RuntimeError("No simulation samples were collected")
        trials = []
        for index, name in enumerate(self.names):
            trials.append({
                "env_index": index,
                "command_name": name,
                "command_tactical": self.commands[index].tolist(),
                "failure_terms": self.failure_terms[index],
                **self._summarize([index]),
            })
        report = {
            "aggregate": self._summarize(range(len(self.names))),
            "commands": {
                name: self._summarize([i for i, item in enumerate(self.names) if item == name])
                for name in dict.fromkeys(self.names)
            },
            "per_trial": trials,
        }
        return report


def _quat_rotate(q, vector, inverse=False):
    import torch
    xyz = -q[..., 1:] if inverse else q[..., 1:]
    cross = 2. * torch.cross(xyz, vector, dim=-1)
    return vector + q[..., :1] * cross + torch.cross(xyz, cross, dim=-1)


class PlannedImpactExecutor:
    """Run the fixed plan on an unmodified V2 or Plus disturbance runtime."""

    def __init__(self, runtime, events, alive, on_hit=None):
        import torch
        self.runtime, self.events, self.alive, self.on_hit = runtime, tuple(events), alive, on_hit
        self.original_step = runtime.physics_step
        self.elapsed, self.cursor = 0., 0
        self.started = np.zeros(len(events), dtype=bool)
        self.skipped = np.zeros(len(events), dtype=bool)
        self.delivered = torch.zeros(len(events), device=runtime.device)
        self.current = np.full(runtime.num_envs, -1, dtype=int)
        self.initial_yaw = None
        self.world_directions = None
        bodies = {name: i for i, name in enumerate(runtime.robot.body_names)}
        missing = {event.target for event in events}.difference(bodies)
        if missing:
            raise ValueError(f"Planned impact bodies are missing: {sorted(missing)}")
        self.body_ids = torch.tensor([bodies[e.target] for e in events], device=runtime.device, dtype=torch.long)
        self.points = torch.tensor([e.point_b for e in events], device=runtime.device, dtype=torch.float32).reshape(-1, 3)
        self.local_directions = torch.tensor([e.direction_initial for e in events], device=runtime.device, dtype=torch.float32).reshape(-1, 3)
        self.env_ids = torch.tensor([e.env_index for e in events], device=runtime.device, dtype=torch.long)
        self.peaks = torch.tensor([e.peak_n for e in events], device=runtime.device)
        self.durations = torch.tensor([e.duration_s for e in events], device=runtime.device)
        self.onsets = torch.tensor([e.onset_s for e in events], device=runtime.device)

    def __call__(self):
        import torch
        r = self.runtime
        # Keep the real runtime's recovery/displacement bookkeeping, but never
        # allow its policy/reset-dependent RNG to schedule benchmark pulses.
        enabled = r.cfg.enabled
        r.cfg.enabled = False
        try:
            self.original_step()
        finally:
            r.cfg.enabled = enabled
        if self.initial_yaw is None:
            q = r.robot.data.root_quat_w
            self.initial_yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2].square() + q[:, 3].square()))
            angles = self.initial_yaw[self.env_ids]
            x, y, z = self.local_directions.unbind(-1)
            self.world_directions = torch.stack((angles.cos() * x - angles.sin() * y,
                                                 angles.sin() * x + angles.cos() * y, z), -1)
        live = np.asarray(self.alive(), dtype=bool)
        end = self.elapsed + r.dt
        due = []
        while self.cursor < len(self.events) and self.events[self.cursor].onset_s < end - 1e-10:
            index, self.cursor = self.cursor, self.cursor + 1
            event = self.events[index]
            if live[event.env_index]:
                self.current[event.env_index] = index
                self.started[index] = True
                due.append(index)
            else:
                self.skipped[index] = True
        if due:
            indices = torch.tensor(due, device=r.device, dtype=torch.long)
            ids = self.env_ids[indices]
            sectors = torch.tensor([r.DIRECTIONS.index(self.events[i].direction) for i in due], device=r.device)
            groups = torch.tensor([r.GROUPS.index(self.events[i].group) for i in due], device=r.device)
            r._stats["hits"][ids] += 1
            stress = r.cfg.mode == "stress"
            r._stats["stress_hits" if stress else "common_hits"][ids] += 1
            r._stats["sampled_peak_n"][ids] = torch.maximum(r._stats["sampled_peak_n"][ids], self.peaks[indices])
            r._direction_hits[ids, sectors] += 1
            r._group_hits[ids, groups] += 1
            r._since_hit[ids] = 0.
            if hasattr(r, "_measurement_remaining"):
                fresh = r._measurement_remaining[ids] <= 0
                fresh_ids = ids[fresh]
                if fresh_ids.numel():
                    r._expected_xy[fresh_ids] = r.robot.data.root_pos_w[fresh_ids, :2] + r._world_command_velocity()[fresh_ids] * r.dt
                    r._measurement_error[fresh_ids] = 0.
                    r._measurement_remaining[fresh_ids] = r.cfg.measurement_window_s
            if self.on_hit is not None:
                self.on_hit(ids)
            if r.hit_observer is not None:
                r.hit_observer(ids)
        valid_envs = np.flatnonzero((self.current >= 0) & live)
        if len(valid_envs):
            event_indices = self.current[valid_envs]
            event_ends = np.asarray([self.events[i].onset_s + self.events[i].duration_s for i in event_indices])
            valid_envs = valid_envs[event_ends > self.elapsed + 1e-10]
            event_indices = self.current[valid_envs]
            if len(valid_envs):
                indices = torch.tensor(event_indices, device=r.device, dtype=torch.long)
                ids = torch.tensor(valid_envs, device=r.device, dtype=torch.long)
                duration, peak = self.durations[indices], self.peaks[indices]
                age = self.elapsed - self.onsets[indices]
                start = age.clamp_min(0.).minimum(duration)
                finish = (age + r.dt).clamp_min(0.).minimum(duration)
                impulse = peak * duration / math.pi * (torch.cos(math.pi * start / duration) - torch.cos(math.pi * finish / duration))
                force_w = impulse[:, None] / r.dt * self.world_directions[indices]
                bodies = self.body_ids[indices]
                force_b = _quat_rotate(r.robot.data.body_link_quat_w[ids, bodies], force_w, inverse=True)
                r.forces_b[ids, bodies] = force_b
                r.torques_b[ids, bodies] = torch.cross(self.points[indices], force_b, dim=-1)
                r.active[ids] = True
                r.body_id[ids], r.peak_n[ids], r.duration[ids] = bodies, peak, duration
                r.force_direction_w[ids], r.hit_point_b[ids] = self.world_directions[indices], self.points[indices]
                r.age[ids] = (age + r.dt).clamp_min(0.)
                r._stats["impulse_ns"][ids] += impulse
                r._stats["peak_n"][ids] = torch.maximum(r._stats["peak_n"][ids], impulse / r.dt)
                self.delivered[indices] += impulse
        r.robot.instantaneous_wrench_composer.set_forces_and_torques(forces=r.forces_b, torques=r.torques_b, is_global=False)
        self.elapsed = end

    def report(self):
        impulse = self.delivered.detach().cpu().tolist()
        directions = self.world_directions.detach().cpu().tolist() if self.world_directions is not None else [None] * len(self.events)
        return [{**e.as_dict(), "started": bool(self.started[i]), "skipped_after_trial_end": bool(self.skipped[i]),
                 "delivered_impulse_ns": impulse[i], "full_pulse_impulse_ns": 2 * e.peak_n * e.duration_s / math.pi,
                 "force_direction_w": directions[i]} for i, e in enumerate(self.events)]


class EvaluationHook:
    """Neutral physical referee sampled before autoreset, at physics rate."""

    def __init__(self, env, protocol="legacy"):
        self.env, self.protocol = env, protocol
        self.metrics = TrialMetrics(env.num_envs, env.step_dt, protocol=protocol)
        if protocol == RECOVERY_PROTOCOL:
            self.metrics.recovery = [RecoveryTrialState(env.physics_dt) for _ in range(env.num_envs)]
        self.tracking, self.terminal = False, {}
        self.impact_plan, self.impact_plan_hash = (), ""
        self._pending_hits = np.zeros(env.num_envs, dtype=bool)
        self._original_contact_cfg = None
        self._last_sim_counter = None
        self.initial_state_hash = None
        self.ankle_samples = 0
        self.ankle_velocity_square = self.ankle_acceleration_square = 0.
        self._previous_ankle_velocity = None
        self._ankle_velocity_lowpass = None
        self.ankle_highpass_square = 0.
        self.support_foot_samples = 0
        self.support_foot_roll_square = self.support_foot_pitch_square = 0.

    def install_impact_plan(self, events, digest):
        # Installation is deliberately deferred until init after strict restore.
        self.impact_plan, self.impact_plan_hash = tuple(events), str(digest)

    def init(self, player):
        import torch
        self.player = player
        env = self.env
        self.term = env.command_manager.get_term("base_velocity")
        self.tactical = torch.tensor(self.metrics.commands, device=env.device, dtype=torch.float32)
        self.modes = torch.tensor(self.metrics.modes, device=env.device)
        self.original_resample, self.original_update = self.term._resample_command, self.term._update_command
        self.original_reset, self.original_step, self.original_scene_update = env._reset_idx, env.step, env.scene.update
        if self.protocol == RECOVERY_PROTOCOL:
            self._install_neutral_termination()
            self._prepare_guard_reference()
        self.plan_executor = PlannedImpactExecutor(env.impact_runtime, self.impact_plan, self._live_for_plan, self._on_hit)
        env.impact_runtime.physics_step = self.plan_executor

        def resample(term, env_ids):
            term.time_left[env_ids] = 1e6
            self._apply_commands()

        def update(term):
            self._apply_commands()

        def scene_update(dt):
            self.original_scene_update(dt)
            counter = getattr(env, "_sim_step_counter", None)
            if self.tracking and (counter is None or counter != self._last_sim_counter):
                self._last_sim_counter = counter
                self._observe_physics()

        def reset(raw_env, env_ids):
            if self.tracking:
                frame, stats = self._snapshot()
                for index in env_ids.detach().cpu().numpy().astype(int):
                    if not self.metrics.alive[index]:
                        continue
                    if not bool(raw_env.reset_terminated[index].item()):
                        raise RuntimeError("Unexpected timeout before the evaluation horizon")
                    terms = [name for name in raw_env.termination_manager.active_terms
                             if bool(raw_env.termination_manager.get_term(name)[index].item())]
                    self.terminal[index] = (frame[index].copy(), stats[index], terms)
            return self.original_reset(env_ids)

        def step(raw_env, action):
            if self.initial_state_hash is None:
                data = env.scene["robot"].data
                initial = torch.cat((data.root_pos_w - env.scene.env_origins, data.root_quat_w,
                                     data.root_lin_vel_w, data.root_ang_vel_b, data.joint_pos, data.joint_vel), -1)
                self.initial_state_hash = sha256(initial.detach().cpu().numpy().tobytes()).hexdigest()
            self.terminal.clear()
            self.tracking = True
            try:
                return self.original_step(action)
            finally:
                self.tracking = False

        self.term._resample_command = types.MethodType(resample, self.term)
        self.term._update_command = types.MethodType(update, self.term)
        env._reset_idx, env.step, env.scene.update = types.MethodType(reset, env), types.MethodType(step, env), scene_update
        self._apply_commands()

    def _live_for_plan(self):
        if self.metrics.recovery is None:
            return self.metrics.alive
        return self.metrics.alive & np.asarray([not item.knocked_down for item in self.metrics.recovery])

    def _on_hit(self, ids):
        self._pending_hits[ids.detach().cpu().numpy()] = True

    def _install_neutral_termination(self):
        import torch
        manager = self.env.termination_manager
        self._original_contact_cfg = manager.get_term_cfg("base_contact")
        cfg = deepcopy(self._original_contact_cfg)
        def neutral_ground(raw_env):
            return torch.tensor([x.knocked_down for x in self.metrics.recovery], device=raw_env.device)
        cfg.func, cfg.params = neutral_ground, {}
        manager.set_term_cfg("base_contact", cfg)

    def _prepare_guard_reference(self):
        import torch
        robot = self.env.scene["robot"]
        params = self.env.cfg.rewards.standard_arm_pose_l1.params
        target = params.get("target_joint_pos", params.get("term_params", {}).get("target_joint_pos"))
        asset = params.get("asset_cfg")
        if target is None or asset is None:
            raise RuntimeError("Evaluation requires the configured boxing arm target and joint selectors")
        ids, names = robot.find_joints(asset.joint_names, preserve_order=True)
        if len(ids) != len(target):
            raise RuntimeError("Evaluation boxing arm target dimension differs from robot joints")
        self._guard_ids = ids
        self._guard_target = torch.tensor(target, device=self.env.device, dtype=robot.data.joint_pos.dtype)
        ankle_names = ("J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R")
        self._ankle_ids, _ = robot.find_joints(list(ankle_names), preserve_order=True)
        self._foot_ids = robot.find_bodies(["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"], preserve_order=True)[0] if hasattr(robot, "find_bodies") else None

    def _apply_commands(self):
        term = self.term
        term.vel_command_b[:] = term.transform_teleop_command(self.tactical)
        term.time_left.fill_(1e6)
        term.is_heading_env.fill_(False)
        term.is_standing_env[:] = self.modes == 0
        term.heading_target.zero_()
        term.command_mode[:] = self.modes

    def _physical_observations(self):
        import torch
        env = self.env
        data = env.scene["robot"].data
        frame = self._frame()
        angular = data.root_ang_vel_b[:, :2].norm(dim=-1)
        guard = (data.joint_pos[:, self._guard_ids] - self._guard_target).abs().mean(-1)
        nonfoot = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        body = torch.zeros_like(nonfoot)
        foot = torch.zeros_like(nonfoot)
        for name, sensor in env.scene.sensors.items():
            if not name.startswith("impact_ground_"):
                continue
            part = name[len("impact_ground_"):]
            values = sensor.data
            force = values.force_matrix_w
            if force is None:
                raise RuntimeError("Neutral impact evaluation requires ground-filtered force matrices")
            contact = force.norm(dim=-1).flatten(1).amax(1) > 1.
            if part not in ("LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"):
                nonfoot |= contact
                if part in ("LINK_TORSO_YAW", "LINK_BASE", "LINK_HEAD_YAW", "LINK_HEAD_PITCH"):
                    body |= contact
                continue
            up = torch.zeros(env.num_envs, 3, device=env.device)
            up[:, 2] = 1.
            quat = values.quat_w[:, 0]
            foot |= contact & (_quat_rotate(quat, up, inverse=True)[:, 2] < math.cos(math.radians(70.)))
            point = values.contact_pos_w
            if point is not None:
                local = _quat_rotate(quat, point[:, 0, 0] - values.pos_w[:, 0], inverse=True)
                foot |= contact & torch.isfinite(local).all(-1) & (local[:, 2] > -.049)
        packed = torch.stack((angular, guard, nonfoot.float(), body.float(), foot.float()), -1).detach().cpu().numpy()
        observations = []
        for i, row in enumerate(packed):
            observations.append({"tilt_rad": float(frame[i, 3]), "angular_speed_rad_s": float(row[0]),
                                 "command_error_mps": float(np.linalg.norm(frame[i, :2] - self.metrics.commands[i, :2])),
                                 "root_height_m": float(frame[i, 4]), "guard_error_rad": float(row[1]),
                                 "nonfoot_support": bool(row[2]), "body_ground_contact": bool(row[3]),
                                 "foot_side_contact": bool(row[4]), "hand_knee_contact": bool(row[2] and not row[3]),
                                 "impact": bool(self._pending_hits[i])})
        self._pending_hits.fill(False)
        return observations

    def _observe_physics(self):
        if self.metrics.recovery is None:
            return
        import torch
        live = self._live_for_plan()
        observations = self._physical_observations()
        for i in np.flatnonzero(live):
            self.metrics.recovery[i].observe(**observations[i])
        data = self.env.scene["robot"].data
        velocity = data.joint_vel[:, self._ankle_ids]
        if self._ankle_velocity_lowpass is None:
            self._ankle_velocity_lowpass = velocity.clone()
        alpha = 1. - math.exp(-2. * math.pi * 8. * self.env.physics_dt)
        self._ankle_velocity_lowpass += alpha * (velocity - self._ankle_velocity_lowpass)
        if self._previous_ankle_velocity is not None:
            mask = torch.tensor(live, device=velocity.device)
            valid = velocity[mask]
            acceleration = (velocity - self._previous_ankle_velocity)[mask] / self.env.physics_dt
            self.ankle_velocity_square += float(valid.square().sum().item())
            self.ankle_acceleration_square += float(acceleration.square().sum().item())
            self.ankle_highpass_square += float((velocity - self._ankle_velocity_lowpass)[mask].square().sum().item())
            self.ankle_samples += valid.numel()
            if self._foot_ids is not None:
                foot_mask = torch.stack([
                    self.env.scene.sensors["impact_ground_" + name].data.force_matrix_w.norm(dim=-1).flatten(1).amax(1) > 50.
                    for name in ("LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R")], -1) & mask[:, None]
                local_angular = _quat_rotate(data.body_quat_w[:, self._foot_ids], data.body_ang_vel_w[:, self._foot_ids], inverse=True)
                supported = local_angular[foot_mask]
                self.support_foot_samples += supported.shape[0]
                self.support_foot_roll_square += float(supported[:, 0].square().sum().item())
                self.support_foot_pitch_square += float(supported[:, 1].square().sum().item())
        self._previous_ankle_velocity = velocity.clone()

    def _frame(self):
        import torch
        data = self.env.scene["robot"].data
        q = data.root_quat_w
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2].square() + q[:, 3].square()))
        angle = yaw + math.radians(self.env.cfg.commands.base_velocity.tactical_front_offset_deg)
        vx, vy = data.root_lin_vel_w[:, 0], data.root_lin_vel_w[:, 1]
        gravity = data.projected_gravity_b
        frame = torch.stack((angle.cos() * vx + angle.sin() * vy, -angle.sin() * vx + angle.cos() * vy,
                             data.root_ang_vel_b[:, 2], torch.atan2(gravity[:, :2].norm(dim=-1), -gravity[:, 2]),
                             data.root_pos_w[:, 2] - self.env.scene.env_origins[:, 2]), -1).detach().cpu().numpy()
        return frame

    def _snapshot(self):
        import torch
        frame = self._frame()
        runtime = self.env.impact_runtime
        names = list(runtime._stats) + ["direction_" + name for name in runtime.DIRECTIONS] + ["target_" + name for name in runtime.GROUPS]
        packed = torch.column_stack((*runtime._stats.values(), runtime._direction_hits, runtime._group_hits)).detach().cpu().tolist()
        return frame, [dict(zip(names, row, strict=True)) for row in packed]

    def step(self, step, transition, metrics):
        frame, stats = self._snapshot()
        done, terms = np.zeros(self.env.num_envs, dtype=bool), [[] for _ in range(self.env.num_envs)]
        for index, (terminal_frame, terminal_stats, terminal_terms) in self.terminal.items():
            frame[index], stats[index], done[index], terms[index] = terminal_frame, terminal_stats, True, terminal_terms
        self.metrics.add(frame[:, :3], frame[:, 3], frame[:, 4], done, stats, terms)

    def reset(self, indices):
        pass

    def diagnostics(self):
        return {"initial_physical_state_hash": self.initial_state_hash,
                "ankle_physics_samples": self.ankle_samples,
                "ankle_velocity_rms_rad_s": math.sqrt(self.ankle_velocity_square / self.ankle_samples) if self.ankle_samples else None,
                "ankle_acceleration_rms_rad_s2": math.sqrt(self.ankle_acceleration_square / self.ankle_samples) if self.ankle_samples else None,
                "ankle_velocity_highpass_8hz_rms_rad_s": math.sqrt(self.ankle_highpass_square / self.ankle_samples) if self.ankle_samples else None,
                "highpass_filter": "first-order 8Hz cutoff at physical timestep; RMS residual, not a brick-wall spectrum",
                "supported_foot_samples": self.support_foot_samples,
                "supported_foot_roll_rms_rad_s": math.sqrt(self.support_foot_roll_square / self.support_foot_samples) if self.support_foot_samples else None,
                "supported_foot_pitch_rms_rad_s": math.sqrt(self.support_foot_pitch_square / self.support_foot_samples) if self.support_foot_samples else None,
                "delivered_plan": self.plan_executor.report()}

    def close(self):
        self.env._reset_idx, self.env.step, self.env.scene.update = self.original_reset, self.original_step, self.original_scene_update
        self.env.impact_runtime.physics_step = self.plan_executor.original_step
        self.term._resample_command, self.term._update_command = self.original_resample, self.original_update
        if self._original_contact_cfg is not None:
            self.env.termination_manager.set_term_cfg("base_contact", self._original_contact_cfg)


def configure_native_baseline(cfg):
    """Keep stage-one head inputs and outputs while retaining impact injection."""
    from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
    from t800_lab.tasks.t800_base.env_cfg import T800BaseEnvCfg
    from importlib import import_module

    module = import_module(
        "t800_lab.tasks.t800_impact_v2.env"
        if cfg.run_name == "T800_base-ImpactV2" else "t800_lab.tasks.t800_impact.actions"
    )
    ImpactJointPositionAction = module.ImpactJointPositionAction

    class NativeImpactJointPositionAction(ImpactJointPositionAction):
        def _fix_head_targets(self):
            pass

        def process_actions(self, actions):
            if actions.shape != self._logical_raw_actions.shape:
                raise ValueError("Native baseline action shape differs from the stage-one logical shape")
            self._logical_raw_actions.copy_(actions)
            JointPositionAction.process_actions(self, actions.index_select(-1, self._physical_to_logical))

    base = T800BaseEnvCfg()
    for group_name, terms in (
        ("policy", ("joint_pos", "joint_vel", "actions")),
        ("critic", ("joint_pos", "joint_vel", "actions")),
        ("amp_policy", ("joint_pos_rel", "joint_vel")),
        ("amp_reference", ("joint_pos_rel", "joint_vel")),
    ):
        for name in terms:
            setattr(getattr(cfg.observations, group_name), name, deepcopy(getattr(getattr(base.observations, group_name), name)))
    cfg.actions.joint_pos.class_type = NativeImpactJointPositionAction
    cfg.observations.policy.enable_corruption = False


def run(args):
    import bootstrap
    import cusrl
    import gymnasium as gym
    import isaaclab.sim as sim_utils

    from t800_lab.utils.wrappers.cusrl import validate_amp_checkpoint_compatibility, verify_agent_checkpoint_restored
    from t800_lab.utils.wrappers.cusrl.impact import (
        ImpactCusrlAdapter, ImpactV2CusrlAdapter, ImpactV2PlusCusrlAdapter,
        StrictCheckpointAgentFactory, validate_impact_v2_checkpoint, validate_impact_v2_plus_checkpoint,
    )
    from terrain_play import apply_fixed_terrain_level, configure_fixed_terrain_selection

    task = args.task
    cusrl.set_global_seed(args.seed)
    cfg = bootstrap.load_task_config(task, "env_cfg_entry_point")
    agent_cfg = bootstrap.load_task_config(task, "cusrl_cfg_entry_point")
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    agent_cfg.agent_factory.device = args.device
    cfg.episode_length_s = args.duration_s + 5.0
    cfg.impact.mode = args.impact_mode
    cfg.impact.eval_strength = 1.0
    cfg.impact.head_mode = "full"
    # Do not change cfg.impact fault probabilities: they are in checkpoint signatures.
    cfg.events.head_faults = None
    cfg.observations.policy.enable_corruption = False
    cfg.commands.base_velocity.debug_vis = False
    # These two presentation assets otherwise resolve against remote Nucleus.
    cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35))
    cfg.scene.sky_light.spawn.texture_file = None
    generator = cfg.scene.terrain.terrain_generator
    flat = deepcopy(generator.sub_terrains["fight_mat_flat"])
    flat.proportion = 1.0
    generator.sub_terrains = {"fight_mat_flat": flat}
    generator.num_rows = generator.num_cols = 1
    generator.size = (64.0, 64.0)
    generator.curriculum = False
    configure_fixed_terrain_selection(cfg, 0, terrain_name="fight_mat_flat")
    plan, plan_hash = build_impact_plan(args.num_envs, args.duration_s, args.seed, args.impact_mode)
    trial = cusrl.Trial(str(Path(args.checkpoint).resolve()))
    checkpoint = trial.load_checkpoint(map_location="cpu")
    task_base = task.removesuffix("-Play")
    if task_base in bootstrap.DYNAMICS_DR_TASKS:
        from t800_lab.utils.wrappers.cusrl.dynamics_dr import validate_dynamics_dr_checkpoint

        if args.native_baseline:
            raise ValueError("A DR evaluation requires its own checkpoint.")
        validate_dynamics_dr_checkpoint(checkpoint, cfg)
    elif task_base == bootstrap.IMPACT_V2_TASK:
        validate_impact_v2_checkpoint(checkpoint, cfg, allow_source=args.native_baseline)
    elif task_base == bootstrap.IMPACT_V2_PLUS_TASK:
        if args.native_baseline:
            raise ValueError("Use ImpactV2-Play for the CombinedV2 native baseline, not Plus-Play")
        validate_impact_v2_plus_checkpoint(checkpoint, cfg)
    if args.native_baseline:
        configure_native_baseline(cfg)
    with bootstrap.configure_usd_cache(cfg):
        gym_env = gym.make(task, cfg=cfg, disable_env_checker=True)
        adapter_type = {bootstrap.IMPACT_V2_TASK: ImpactV2CusrlAdapter,
                        bootstrap.IMPACT_V2_PLUS_TASK: ImpactV2PlusCusrlAdapter}.get(task_base, ImpactCusrlAdapter)
        if task_base in bootstrap.DYNAMICS_DR_TASKS:
            from t800_lab.utils.wrappers.cusrl.dynamics_dr import DynamicsDRCusrlAdapter

            adapter_type = DynamicsDRCusrlAdapter
        env = adapter_type(gym_env)
        try:
            apply_fixed_terrain_level(gym_env, 0, 0)
            raw = gym_env.unwrapped
            hook = EvaluationHook(raw, protocol=args.protocol)
            hook.install_impact_plan(plan, plan_hash)
            num_steps = math.ceil(args.duration_s / raw.step_dt)
            validate_amp_checkpoint_compatibility(
                checkpoint, env.amp_observation_dim, checkpoint_path=str(trial.checkpoint_path), task_name=task,
            )
            player = cusrl.Player(
                environment=env,
                agent=StrictCheckpointAgentFactory(agent_cfg.agent_factory, checkpoint["agent"]),
                checkpoint_path=None,
                num_steps=num_steps,
                timestep=0,
                deterministic=True,
                verbose=True,
                hooks=[],
            )
            player.agent.load_state_dict(checkpoint["agent"])
            env.load_state_dict(checkpoint["environment"])
            verify_agent_checkpoint_restored(checkpoint["agent"], player.agent)
            if args.nominal_dynamics:
                # Benchmark override after strict restoration, like the neutral
                # fall referee. Use the same masses/inertias/materials for old
                # and DR policies instead of comparing different startup draws.
                import torch

                robot = raw.scene["robot"]
                ids = torch.arange(raw.num_envs, dtype=torch.int, device="cpu")
                mass = robot.data.default_mass.detach().cpu().clone()
                inertia = robot.data.default_inertia.detach().cpu().clone()
                view = robot.root_physx_view
                view.set_masses(mass, ids)
                view.set_inertias(inertia, ids)
                material = torch.ones_like(view.get_material_properties())
                material[..., 2] = 0.0
                view.set_material_properties(material, ids)
                torch.testing.assert_close(view.get_masses(), mass)
                torch.testing.assert_close(view.get_inertias(), inertia)
                torch.testing.assert_close(view.get_material_properties(), material)
            player.agent.set_inference_mode(deterministic=True)
            runtime = raw.impact_runtime
            # Checkpoint RNG is training state. Independent benchmark seeds start here.
            runtime._generator.manual_seed(args.seed + runtime.cfg.seed_offset)
            config_signature = runtime.config_signature
            # Recovery/termination and deterministic pulse overrides start only
            # after the trained configuration and complete checkpoint passed.
            player.register_hook(hook)
            player.run_playing_loop()
            if player.step != num_steps:
                raise RuntimeError(f"Evaluation interrupted after {player.step}/{num_steps} steps")
            report = {
                "schema_version": 2,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "label": args.label,
                "checkpoint": str(trial.checkpoint_path),
                "checkpoint_iteration": int(trial.iteration),
                "task": task,
                "seed": args.seed,
                "num_envs": args.num_envs,
                "duration_requested_s": args.duration_s,
                "duration_simulated_s": num_steps * raw.step_dt,
                "control_timestep_s": raw.step_dt,
                "impact_mode": args.impact_mode,
                "evaluation_protocol": args.protocol,
                "nominal_dynamics": args.nominal_dynamics,
                "impact_plan_hash": plan_hash,
                "impact_plan_events": [event.as_dict() for event in plan],
                "recovery_thresholds": RECOVERY_THRESHOLDS if args.protocol == RECOVERY_PROTOCOL else None,
                "physical_diagnostics": hook.diagnostics(),
                "native_baseline": args.native_baseline,
                "head_faults": "disabled; full physical head retained",
                "impact_config_signature": config_signature,
                "terrain": {"type": "fight_mat_flat", "level": 0, "size_m": [64, 64]},
                "measurement_notes": [
                    "Exactly one fixed-command trial per environment; no replacement trials after termination.",
                    "All physical metrics, including RMSE, stop at first termination and include its pre-reset frame.",
                    "Aggregates weight individual valid simulation samples; xy RMSE is the planar error norm.",
                    "Velocity is in the 37-degree tactical yaw frame; angular rate is root body-z rad/s.",
                    "Recovery_v1 uses true knockdown persistence and separate recovery/guard timing; support and foot-side contacts are events.",
                    "Impact plan is pre-generated and hashed before simulation so policy termination cannot alter later load parameters.",
                    "Recovery mean counts completed recoveries only; expired windows and falls are reported separately.",
                    "Extra displacement is the runtime's maximum command-relative 2-second measurement deviation.",
                ],
                **hook.metrics.report(),
            }
            output = Path(args.output).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(output)
            print(f"[INFO] Evaluation JSON: {output}")
            print(json.dumps(report["aggregate"], indent=2))
        finally:
            env.close()


def main():
    import bootstrap  # noqa: F401 -- registers local source path before IsaacLab launch
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=[task + "-Play" for task in bootstrap.IMPACT_TASKS], default="T800_base-Impact-Play",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--impact_mode", choices=("off", "common", "stress"), default="off")
    parser.add_argument("--protocol", choices=("legacy", RECOVERY_PROTOCOL), default="legacy",
                        help="legacy first-terminal benchmark or neutral recovery_v1 criteria")
    parser.add_argument("--native-baseline", action="store_true")
    parser.add_argument("--nominal-dynamics", action="store_true",
                        help="Use nominal link mass/inertia and common friction=1, restitution=0 after strict restore.")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cuda:0")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.num_envs <= 0 or not math.isfinite(args.duration_s) or args.duration_s <= 0:
        parser.error("--num_envs and --duration-s must be positive and finite")
    if not Path(args.checkpoint).is_file():
        parser.error("--checkpoint must name an existing checkpoint file")
    launcher = AppLauncher(args)
    try:
        run(args)
    except BaseException:
        traceback.print_exc()
        import omni.kit.app

        omni.kit.app.get_app().post_quit(1)
        raise
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
