"""Finite T800 command-response benchmark with commands injected before actor inference.

Each environment runs one 15-second scenario.  Termination is captured before
autoreset and censors its remaining samples; segment changes retain LSTM history.
Numerical schedules and response measurements are CPU-testable without Isaac Sim.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import traceback
import types

import numpy as np


@dataclass(frozen=True)
class Segment:
    name: str
    duration_s: float
    command: tuple[float, float, float]


@dataclass(frozen=True)
class Scenario:
    name: str
    axis: int | None
    segments: tuple[Segment, ...]


def _axis_command(axis: int, value: float):
    command = [0.0, 0.0, 0.0]
    command[axis] = value
    return tuple(command)


def build_scenarios() -> tuple[Scenario, ...]:
    zero = (0.0, 0.0, 0.0)
    scenarios = []
    for axis, prefix, amplitudes, signs in (
        (0, "fb", (0.6, 1.0, 1.4), (1,)),
        (1, "lr", (0.6, 1.0, 1.4), (1,)),
        (2, "yaw", (0.75, 1.25, 1.8), (1, -1)),
    ):
        for amplitude in amplitudes:
            for sign in signs:
                suffix = "_negative_first" if sign < 0 else ""
                scenarios.append(Scenario(f"{prefix}_{amplitude:g}{suffix}", axis, (
                    Segment("settle", 2.0, zero),
                    Segment("first", 3.0, _axis_command(axis, sign * amplitude)),
                    Segment("stop_first", 2.0, zero),
                    Segment("reverse", 3.0, _axis_command(axis, -sign * amplitude)),
                    Segment("stop_reverse", 2.0, zero),
                    Segment("hold_stand", 3.0, zero),
                )))
    for axis, name, amplitude in ((0, "fast_fb", 0.8), (1, "fast_lr", 0.6), (2, "fast_yaw", 0.75)):
        segments = [Segment("settle", 2.0, zero)]
        segments.extend(
            Segment(f"reverse_{index:02d}", 0.5, _axis_command(axis, amplitude * (-1) ** index))
            for index in range(16)
        )
        segments.append(Segment("stop", 5.0, zero))
        scenarios.append(Scenario(name, axis, tuple(segments)))
    scenarios.append(Scenario("mixed_chase_evade", None, (
        Segment("settle", 2.0, zero),
        Segment("advance_turn_left", 2.0, (0.8, 0.0, 0.75)),
        Segment("retreat_turn_right", 2.0, (-0.8, 0.0, -0.75)),
        Segment("left_turn_left", 2.0, (0.0, 0.6, 0.75)),
        Segment("right_turn_right", 2.0, (0.0, -0.6, -0.75)),
        Segment("advance", 2.0, (0.8, 0.0, 0.0)),
        Segment("stop", 3.0, zero),
    )))
    assert len(scenarios) == 16
    return tuple(scenarios)


def compile_schedule(scenario: Scenario, timestep: float):
    commands, segment_ids = [], []
    for index, segment in enumerate(scenario.segments):
        count = int(round(segment.duration_s / timestep))
        if count < 1 or not math.isclose(count * timestep, segment.duration_s, abs_tol=1e-9):
            raise ValueError("Segment duration must be an exact positive number of control steps")
        commands.extend([segment.command] * count)
        segment_ids.extend([index] * count)
    return np.asarray(commands, dtype=np.float32), np.asarray(segment_ids, dtype=np.int16)


def causal_mean(values: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    """Trailing sample mean; no future samples and no output after first fall."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    result = np.full_like(values, np.nan)
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        selected = valid[start : index + 1]
        if valid[index] and selected.any():
            result[index] = values[start : index + 1][selected].mean(axis=0)
    return result


def first_fall_valid_mask(falls: np.ndarray) -> np.ndarray:
    """Include the first terminal physical frame and exclude all subsequent ones."""
    falls = np.asarray(falls, dtype=bool)
    return np.cumsum(falls, axis=0) - falls.astype(np.int64) == 0


def sustained_reach(condition, valid, falls, timestep: float, hold_s: float = 0.1):
    """Find a run spanning at least hold_s between its first and last samples.

    Input sample k is the post-physics state at (k+1)*dt after the command.
    Latency is the first observed qualifying state, reported only after the
    complete hold has been observed. Confirmation includes the hold duration.
    """
    condition = np.asarray(condition, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    falls = np.asarray(falls, dtype=bool)
    if not valid.any():
        return {"status": "not_exposed", "latency_s": None, "confirmed_s": None}
    hold_steps = int(math.ceil(hold_s / timestep - 1e-9)) + 1
    run = 0
    for index in range(len(condition)):
        # A terminal pose cannot count as a successful stable response.
        run = run + 1 if condition[index] and valid[index] and not falls[index] else 0
        if run >= hold_steps:
            first = index + 1 - hold_steps
            return {
                "status": "reached",
                "latency_s": float((first + 1) * timestep),
                "confirmed_s": float((index + 1) * timestep),
                "first_sample": first,
                "confirmed_sample": index,
            }
    return {
        "status": "fell_before_reach" if np.any(falls & valid) else "not_reached",
        "latency_s": None,
        "confirmed_s": None,
    }


def _not_applicable():
    return {"status": "not_applicable", "latency_s": None, "confirmed_s": None}


def transition_measurements(
    commands, velocity, positions, valid, falls, initial_position, timestep: float, axis: int | None,
):
    """Measure every actual command edge; unsuccessful responses remain explicit."""
    commands = np.asarray(commands, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    falls = np.asarray(falls, dtype=bool)
    edges = np.flatnonzero(np.any(np.diff(commands, axis=0) != 0, axis=-1)) + 1
    results = []
    for order, start in enumerate(edges):
        end = int(edges[order + 1]) if order + 1 < len(edges) else len(commands)
        old, new = commands[start - 1], commands[start]
        current_velocity = velocity[start:end]
        window_valid, window_falls = valid[start:end], falls[start:end]
        record = {
            "transition_index": order,
            "start_step": int(start),
            "start_s": float(start * timestep),
            "end_step_exclusive": end,
            "window_duration_s": float((end - start) * timestep),
            "valid_exposure_s": float(window_valid.sum() * timestep),
            "command_before": old.tolist(),
            "command_after": new.tolist(),
            "zero_cross": _not_applicable(),
            "reach_50": _not_applicable(),
            "reach_90": _not_applicable(),
            "stop": _not_applicable(),
            "steady_state_gain": {"status": "not_applicable", "value": None},
        }
        if axis is not None and abs(new[axis]) > 1e-8:
            directed = current_velocity[:, axis] * np.sign(new[axis])
            if old[axis] * new[axis] < 0:
                record["zero_cross"] = sustained_reach(directed >= 0, window_valid, window_falls, timestep)
            for fraction, name in ((0.5, "reach_50"), (0.9, "reach_90")):
                record[name] = sustained_reach(
                    directed >= fraction * abs(new[axis]), window_valid, window_falls, timestep,
                )
            steady_count = max(1, int(round(0.5 / timestep)))
            steady_valid = window_valid[-steady_count:] & ~window_falls[-steady_count:]
            complete = len(steady_valid) == steady_count and steady_valid.all()
            status = "complete" if complete else "not_exposed" if not steady_valid.any() else "incomplete_fall"
            record["steady_state_gain"] = {
                "status": status,
                "value": float(np.mean(current_velocity[-steady_count:, axis]) / new[axis]) if complete else None,
                "valid_samples": int(steady_valid.sum()),
                "required_samples": steady_count,
            }
        if np.all(np.abs(new) < 1e-8) and np.any(np.abs(old) > 1e-8):
            stopped = (np.linalg.norm(current_velocity[:, :2], axis=-1) < 0.1) & (np.abs(current_velocity[:, 2]) < 0.15)
            stop = sustained_reach(stopped, window_valid, window_falls, timestep)
            stop.update({"stop_distance_xy_m": None, "stop_path_xy_m": None})
            if stop["status"] == "reached":
                before = np.asarray(initial_position) if start == 0 else positions[start - 1]
                confirmed = start + stop["confirmed_sample"]
                path = np.vstack((before[:2], positions[start : confirmed + 1, :2]))
                stop["stop_distance_xy_m"] = float(np.linalg.norm(path[-1] - path[0]))
                stop["stop_path_xy_m"] = float(np.linalg.norm(np.diff(path, axis=0), axis=-1).sum())
            record["stop"] = stop
        results.append(record)
    return results


def summarize_trial(velocity, command, tilt, height, valid, falls, timestep):
    valid = np.asarray(valid, dtype=bool)
    error = np.asarray(velocity)[valid] - np.asarray(command)[valid]
    first = np.flatnonzero(falls)
    return {
        "fell": bool(len(first)),
        "first_fall_s": float((first[0] + 1) * timestep) if len(first) else None,
        "valid_samples": int(valid.sum()),
        "alive_seconds": float(valid.sum() * timestep),
        "tracking_rmse_components": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "tracking_rmse_xy": float(np.sqrt(np.mean(np.sum(error[:, :2] ** 2, axis=-1)))),
        "tracking_rmse_yaw": float(np.sqrt(np.mean(error[:, 2] ** 2))),
        "tilt_p95_rad": float(np.quantile(np.asarray(tilt)[valid], 0.95)),
        "tilt_max_rad": float(np.max(np.asarray(tilt)[valid])),
        "height_min_m": float(np.min(np.asarray(height)[valid])),
    }


def strict_stop_measurements(commands, raw_velocity, positions, valid, falls, initial_position, timestep):
    """Test motion magnitude before smoothing so opposite velocities cannot cancel."""
    raw_velocity = np.asarray(raw_velocity)
    speed = np.column_stack((np.linalg.norm(raw_velocity[:, :2], axis=-1), np.abs(raw_velocity[:, 2])))
    smoothed_speed = causal_mean(speed, valid, max(1, int(round(0.1 / timestep))))
    proxy = np.column_stack((smoothed_speed[:, 0], np.zeros(len(speed)), smoothed_speed[:, 1]))
    records = transition_measurements(commands, proxy, positions, valid, falls, initial_position, timestep, None)
    return [record["stop"] for record in records]


def command_observation_slice(manager, group: str):
    """Derive command offsets from the actual observation-manager term order."""
    terms = manager.active_terms[group]
    dimensions = manager.group_obs_term_dim[group]
    cursor = 0
    for name, shape in zip(terms, dimensions, strict=True):
        width = int(np.prod(shape))
        if name == "velocity_commands":
            if width != 3:
                raise ValueError(f"{group}.velocity_commands must have dimension three, got {shape}")
            result = slice(cursor, cursor + width)
            expected = (6, 9) if group == "policy" else (9, 12)
            if (result.start, result.stop) != expected:
                raise ValueError(f"Unexpected {group} command location: {result}; expected {expected}")
            return result
        cursor += width
    raise ValueError(f"Missing {group}.velocity_commands observation term")


class CommandController:
    """Set the physical command now, and prevent end-of-step resampling."""

    def __init__(self, env, commands):
        import torch

        self.env = env
        self.term = env.command_manager.get_term("base_velocity")
        self.commands = torch.as_tensor(commands, device=env.device)
        self.current = self.commands[0]
        self.original_resample = self.term._resample_command
        self.original_update = self.term._update_command

        def resample(term, env_ids):
            self._apply()

        def update(term):
            self._apply()

        self.term._resample_command = types.MethodType(resample, self.term)
        self.term._update_command = types.MethodType(update, self.term)
        self.policy_slice = command_observation_slice(env.observation_manager, "policy")
        self.critic_slice = command_observation_slice(env.observation_manager, "critic")
        self._apply()

    def _apply(self):
        import torch

        term = self.term
        term.vel_command_b[:] = term.transform_teleop_command(self.current)
        term.time_left.fill_(1e6)
        term.is_heading_env.fill_(False)
        translating = torch.linalg.vector_norm(self.current[:, :2], dim=-1) > 1e-8
        turning = self.current[:, 2].abs() > 1e-8
        term.is_standing_env[:] = ~(translating | turning)
        term.heading_target.zero_()
        term.command_mode[:] = torch.where(
            translating, torch.where(turning, 4, 1), torch.where(turning, 2, 0),
        )
        if hasattr(term, "is_stop_signal_env"):
            term.is_stop_signal_env.fill_(False)

    def before_act(self, step, observation, state):
        self.current = self.commands[step]
        self._apply()
        # Only replace the direct command input before normalization. Computing
        # all observation groups here would advance/rewrite AMP history.
        observation = observation.clone()
        state = state.clone()
        observation[:, self.policy_slice] = self.term.command
        state[:, self.critic_slice] = self.term.command
        return observation, state

    def close(self):
        self.term._resample_command = self.original_resample
        self.term._update_command = self.original_update


class TraceCapture:
    """Capture physical frames before autoreset and keep only each first trial."""

    def __init__(self, env, num_steps):
        self.env = env
        self.in_step = False
        self.terminal = {}
        self.alive = np.ones(env.num_envs, dtype=bool)
        self.original_reset = env._reset_idx
        shape = (num_steps, env.num_envs)
        self.arrays = {
            "actual_velocity_tactical_raw": np.full((*shape, 3), np.nan, dtype=np.float32),
            "root_pos_w": np.full((*shape, 3), np.nan, dtype=np.float32),
            "tilt_rad": np.full(shape, np.nan, dtype=np.float32),
            "height_m": np.full(shape, np.nan, dtype=np.float32),
            "alive": np.zeros(shape, dtype=bool),
            "first_fall": np.zeros(shape, dtype=bool),
            "amp_style_reward": np.full(shape, np.nan, dtype=np.float32),
            "amp_transition_valid": np.zeros(shape, dtype=bool),
        }
        manager = env.reward_manager
        desired = ("action_rate_l2", "dof_acc_l2", "ankle_target_rate_l2", "ankle_target_acceleration_l2")
        self.reward_names = [name for name in desired if name in manager.active_terms]
        self.reward_indices = [manager.active_terms.index(name) for name in self.reward_names]
        self.reward_weights = [float(manager.get_term_cfg(name).weight) for name in self.reward_names]
        self.arrays["reward_weighted_per_second"] = np.full((*shape, len(self.reward_names)), np.nan, dtype=np.float32)

        def reset(raw_env, env_ids):
            if self.in_step:
                frame = self.snapshot()
                for index in env_ids.detach().cpu().tolist():
                    if self.alive[index]:
                        if not bool(raw_env.reset_terminated[index].item()):
                            raise RuntimeError("Unexpected timeout inside a 15-second agility scenario")
                        self.terminal[index] = frame[index].copy()
            return self.original_reset(env_ids)

        env._reset_idx = types.MethodType(reset, env)

    def snapshot(self):
        import torch

        data = self.env.scene["robot"].data
        q = data.root_quat_w
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        angle = yaw + math.radians(self.env.cfg.commands.base_velocity.tactical_front_offset_deg)
        vx, vy = data.root_lin_vel_w[:, 0], data.root_lin_vel_w[:, 1]
        gravity = data.projected_gravity_b
        physical = torch.column_stack((
            angle.cos() * vx + angle.sin() * vy,
            -angle.sin() * vx + angle.cos() * vy,
            data.root_ang_vel_w[:, 2],
            data.root_pos_w,
            torch.atan2(torch.linalg.vector_norm(gravity[:, :2], dim=-1), -gravity[:, 2]),
            data.root_pos_w[:, 2] - self.env.scene.env_origins[:, 2],
            self.env.reward_manager._step_reward[:, self.reward_indices],
        ))
        return physical.detach().cpu().numpy()

    def record(self, step, terminated, amp_reward):
        frame = self.snapshot()
        done = terminated.detach().cpu().numpy().reshape(-1).astype(bool)
        for index, terminal in self.terminal.items():
            frame[index] = terminal
        newly_fallen = done & self.alive
        if set(np.flatnonzero(newly_fallen)) != set(self.terminal):
            raise RuntimeError("Missing pre-autoreset physical snapshot for a new fall")
        mask = self.alive.copy()
        arrays = self.arrays
        arrays["actual_velocity_tactical_raw"][step, mask] = frame[mask, :3]
        arrays["root_pos_w"][step, mask] = frame[mask, 3:6]
        arrays["tilt_rad"][step, mask] = frame[mask, 6]
        arrays["height_m"][step, mask] = frame[mask, 7]
        arrays["reward_weighted_per_second"][step, mask] = frame[mask, 8:]
        arrays["alive"][step] = mask
        arrays["first_fall"][step] = newly_fallen
        if amp_reward is not None:
            eligible = mask & ~done
            arrays["amp_style_reward"][step, eligible] = amp_reward[eligible]
            arrays["amp_transition_valid"][step] = eligible
        self.alive &= ~done

    def close(self):
        self.env._reset_idx = self.original_reset


def _amp_style_reward(agent):
    import torch

    transition = agent.transition.get("agent_transition")
    if transition is None:
        return None
    for hook in agent.hook:
        if hasattr(hook, "discriminator") and hasattr(hook, "reward_scale"):
            with torch.no_grad(), agent.autocast():
                probability = torch.sigmoid(hook.discriminator(transition))
                reward = -torch.log((1.0 - probability).clamp_min(1e-4)) * hook.reward_scale
            return reward.detach().float().cpu().numpy().reshape(-1)
    return None


def write_results(args, trial, raw, scenarios, assignments, commands, segment_ids, capture, initial_position):
    dt = float(raw.step_dt)
    arrays = capture.arrays
    steps, num_envs = arrays["alive"].shape
    angle = math.radians(raw.cfg.commands.base_velocity.tactical_front_offset_deg)
    root_commands = commands.copy()
    root_commands[:, :, 0] = math.cos(angle) * commands[:, :, 0] - math.sin(angle) * commands[:, :, 1]
    root_commands[:, :, 1] = math.sin(angle) * commands[:, :, 0] + math.cos(angle) * commands[:, :, 1]
    arrays.update({
        "command_tactical": commands,
        "command_root_yaw": root_commands,
        "command_time_s": np.arange(steps, dtype=np.float64) * dt,
        "state_time_s": np.arange(1, steps + 1, dtype=np.float64) * dt,
        "scenario_index": assignments,
        "scenario_names": np.asarray([scenario.name for scenario in scenarios]),
        "segment_index": segment_ids,
        "initial_root_pos_w": initial_position,
        "reward_term_names": np.asarray(capture.reward_names),
        "reward_term_weights": np.asarray(capture.reward_weights),
    })
    smooth = np.full_like(arrays["actual_velocity_tactical_raw"], np.nan)
    for index in range(num_envs):
        smooth[:, index] = causal_mean(arrays["actual_velocity_tactical_raw"][:, index], arrays["alive"][:, index], 5)
    arrays["actual_velocity_tactical_causal_100ms"] = smooth
    per_trial = []
    for index, scene_index in enumerate(assignments):
        scene = scenarios[scene_index]
        valid = arrays["alive"][:, index]
        falls = arrays["first_fall"][:, index]
        transitions = transition_measurements(
            commands[:, index], smooth[:, index], arrays["root_pos_w"][:, index],
            valid, falls, initial_position[index], dt, scene.axis,
        )
        strict_stops = strict_stop_measurements(
            commands[:, index], arrays["actual_velocity_tactical_raw"][:, index],
            arrays["root_pos_w"][:, index], valid, falls, initial_position[index], dt,
        )
        for transition, stop in zip(transitions, strict_stops, strict=True):
            transition["stop_raw_speed"] = stop
        per_trial.append({
            "env_index": index,
            "scenario": scene.name,
            **summarize_trial(
                arrays["actual_velocity_tactical_raw"][:, index], commands[:, index],
                arrays["tilt_rad"][:, index], arrays["height_m"][:, index], valid, falls, dt,
            ),
            "transitions": transitions,
        })
    scenario_reports = {}
    for scene in scenarios:
        selected = [record for record in per_trial if record["scenario"] == scene.name]
        scenario_reports[scene.name] = {
            "trials": len(selected),
            "falls": sum(record["fell"] for record in selected),
            "failure_rate": sum(record["fell"] for record in selected) / len(selected) if selected else None,
            "env_indices": [record["env_index"] for record in selected],
            "axis": scene.axis,
            "segments": [{"name": segment.name, "duration_s": segment.duration_s, "command": list(segment.command)}
                         for segment in scene.segments],
        }
    valid = arrays["alive"]
    errors = (arrays["actual_velocity_tactical_raw"] - commands)[valid]
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    trace_path = output.with_suffix(".npz")
    if trace_path == output:
        raise ValueError("--output is a JSON path, not an .npz path")
    temporary_trace = trace_path.with_name(trace_path.name + ".tmp")
    with temporary_trace.open("wb") as file:
        np.savez_compressed(file, **arrays)
    temporary_trace.replace(trace_path)
    command_cfg = raw.cfg.commands.base_velocity
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "checkpoint": str(trial.checkpoint_path),
        "checkpoint_iteration": int(trial.iteration),
        "task": "T800_base-Impact-Play",
        "seed": args.seed,
        "num_envs": num_envs,
        "duration_s": steps * dt,
        "control_timestep_s": dt,
        "impact_mode": args.impact_mode,
        "impact_config_signature": raw.impact_runtime.config_signature,
        "native_baseline": args.native_baseline,
        "head_faults": "disabled; full physical head retained",
        "terrain": {"type": "fight_mat_flat", "level": 0, "size_m": [64, 64]},
        "trace_file": str(trace_path),
        "training_command_ranges": {
            "cardinal_speed_m_s": list(command_cfg.cardinal_speed_range),
            "pure_turn_rad_s": list(command_cfg.turn_rate_range),
            "other_speed_m_s": list(command_cfg.other_speed_range),
            "mixed_speed_m_s": list(command_cfg.mixed_speed_range),
            "mixed_turn_rad_s": list(command_cfg.mixed_turn_rate_range),
            "note": "0.6/1.4m/s and 0.75rad/s cases intentionally probe outside their sampling mode's training range.",
        },
        "measurement_notes": [
            "At each step the actual command and actor/critic command input slices change before actor.act.",
            "Commands are unfiltered; no AMP history recomputation and no LSTM reset at command edges.",
            "State sample k is at (k+1)*0.02 seconds; command k acts over [k*0.02,(k+1)*0.02].",
            "Velocity xy uses tactical yaw frame (37 degrees); angular tracking is world-z angular velocity approximation.",
            "Raw RMSE includes first terminal physical frame, excludes every later autoreset frame, and weights valid samples.",
            "Reach/stop latencies use trailing 100ms mean velocity (5 samples; initial min_periods=1).",
            "Thresholds must persist for 100ms between first/last samples (6 samples); falls never count as successes.",
            "latency_s is first qualifying state once hold confirmed; confirmed_s is when hold confirmation becomes available.",
            "reach_50/reach_90 compare new signed target amplitude; zero_cross applies only direct sign reversals.",
            "stop tests signed mean velocity; counter-oscillations may cancel. It is a drift-stopping measure.",
            "stop_raw_speed tests 100ms mean(raw planar speed norm)<0.1m/s AND mean(abs(raw yaw))<0.15rad/s, held 100ms.",
            "Both stopping distance/path measures end at confirmation; raw-speed stops cannot cancel opposite motion.",
            "Steady gain uses the final 0.5 seconds of an unchanged command window; incomplete windows have null gain.",
            "Not-exposed, fell-before-reach and not-reached are explicit statuses with null latency, never zero.",
            "AMP trace is raw discriminator style reward before recovery scaling; terminal AMP transitions are invalid.",
            "Reward term traces equal IsaacLab _step_reward = term*weight, a weighted per-second contribution.",
            "Body/friction randomization is retained; paired seeds do not guarantee paired draws after different falls.",
        ],
        "reward_terms": [
            {"name": name, "weight": weight}
            for name, weight in zip(capture.reward_names, capture.reward_weights, strict=True)
        ],
        "aggregate": {
            "trials": num_envs,
            "falls": int(arrays["first_fall"].any(axis=0).sum()),
            "failure_rate": float(arrays["first_fall"].any(axis=0).mean()),
            "valid_samples": int(valid.sum()),
            "alive_seconds_total": float(valid.sum() * dt),
            "tracking_rmse_components": np.sqrt(np.mean(errors**2, axis=0)).tolist(),
            "tracking_rmse_xy": float(np.sqrt(np.mean(np.sum(errors[:, :2] ** 2, axis=-1)))),
            "tracking_rmse_yaw": float(np.sqrt(np.mean(errors[:, 2] ** 2))),
        },
        "scenarios": scenario_reports,
        "per_trial": per_trial,
    }
    temporary_json = output.with_name(output.name + ".tmp")
    temporary_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary_json.replace(output)
    print(f"[INFO] Agility JSON: {output}; trace: {trace_path}")
    print(json.dumps(report["aggregate"], indent=2))


def run(args):
    import bootstrap
    import cusrl
    import gymnasium as gym
    import isaaclab.sim as sim_utils

    from evaluate_impact import configure_native_baseline
    from t800_lab.utils.wrappers.cusrl import validate_amp_checkpoint_compatibility, verify_agent_checkpoint_restored
    from t800_lab.utils.wrappers.cusrl.impact import ImpactCusrlAdapter, StrictCheckpointAgentFactory
    from terrain_play import apply_fixed_terrain_level, configure_fixed_terrain_selection

    class AgilityPlayer(cusrl.Player):
        def run_playing_loop(self):
            observation, state, _ = self.environment.reset()
            initial = raw.scene["robot"].data.root_pos_w.detach().cpu().numpy().copy()
            for step in range(self.num_steps):
                observation, state = controller.before_act(step, observation, state)
                action = self.agent.act(observation, state)
                capture.terminal.clear()
                capture.in_step = True
                try:
                    observation, state, reward, terminated, truncated, info = self.environment.step(action)
                finally:
                    capture.in_step = False
                if bool(truncated.any().item()):
                    raise RuntimeError("Agility scenario encountered an unexpected episode timeout")
                # CusRL owns recurrent done handling, even in deterministic inference.
                self.agent.step(observation, reward, terminated, truncated, state, **info)
                capture.record(step, terminated, _amp_style_reward(self.agent))
                self.step = step + 1
                if self.step % 100 == 0:
                    print(f"[INFO] Agility step {self.step}/{self.num_steps}; surviving={int(capture.alive.sum())}")
            return initial

    task = "T800_base-Impact-Play"
    cusrl.set_global_seed(args.seed)
    cfg = bootstrap.load_task_config(task, "env_cfg_entry_point")
    agent_cfg = bootstrap.load_task_config(task, "cusrl_cfg_entry_point")
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    agent_cfg.agent_factory.device = args.device
    cfg.episode_length_s = 20.0
    cfg.impact.mode = args.impact_mode
    cfg.impact.eval_strength = 1.0
    cfg.impact.head_mode = "full"
    cfg.events.head_faults = None
    cfg.observations.policy.enable_corruption = False
    cfg.commands.base_velocity.debug_vis = False
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
    if args.native_baseline:
        configure_native_baseline(cfg)
    trial = cusrl.Trial(str(Path(args.checkpoint).resolve()))
    checkpoint = trial.load_checkpoint(map_location="cpu")
    with bootstrap.configure_usd_cache(cfg):
        gym_env = gym.make(task, cfg=cfg, disable_env_checker=True)
        env = ImpactCusrlAdapter(gym_env)
        controller = capture = None
        try:
            raw = gym_env.unwrapped
            dt = float(raw.step_dt)
            if not math.isclose(dt, 0.02, abs_tol=1e-9):
                raise ValueError(f"Agility benchmark requires 20ms control sampling, got {dt}")
            apply_fixed_terrain_level(gym_env, 0, 0)
            scenarios = build_scenarios()
            compiled = [compile_schedule(scene, dt) for scene in scenarios]
            assignments = np.arange(args.num_envs, dtype=np.int16) % len(scenarios)
            commands = np.stack([compiled[index][0] for index in assignments], axis=1)
            segment_ids = np.stack([compiled[index][1] for index in assignments], axis=1)
            controller = CommandController(raw, commands)
            capture = TraceCapture(raw, len(commands))
            validate_amp_checkpoint_compatibility(
                checkpoint, env.amp_observation_dim, checkpoint_path=str(trial.checkpoint_path), task_name=task,
            )
            player = AgilityPlayer(
                environment=env,
                agent=StrictCheckpointAgentFactory(agent_cfg.agent_factory, checkpoint["agent"]),
                checkpoint_path=None,
                num_steps=len(commands),
                timestep=0,
                deterministic=True,
                verbose=False,
            )
            player.agent.load_state_dict(checkpoint["agent"])
            env.load_state_dict(checkpoint["environment"])
            verify_agent_checkpoint_restored(checkpoint["agent"], player.agent)
            player.agent.set_inference_mode(deterministic=True)
            raw.impact_runtime._generator.manual_seed(args.seed + raw.impact_runtime.cfg.seed_offset)
            initial_position = player.run_playing_loop()
            write_results(args, trial, raw, scenarios, assignments, commands, segment_ids, capture, initial_position)
        finally:
            if capture is not None:
                capture.close()
            if controller is not None:
                controller.close()
            env.close()


def main():
    import bootstrap  # noqa: F401 -- local source path before AppLauncher
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--impact_mode", choices=("off", "common"), default="off")
    parser.add_argument("--native-baseline", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cuda:0")
    args = parser.parse_args()
    if args.num_envs <= 0:
        parser.error("--num_envs must be positive")
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
