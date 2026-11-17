"""Play-only, offline markers for physical impact onsets.

An impact can start and end between two policy steps. Capture its link-frame
application point at the physics-step onset, then retain a world-space snapshot
briefly. Marker lifetime never extends the actual force pulse.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from cusrl.template.player import PlayerHook


ARROW_STANDOFF_M = 0.35


@dataclass(frozen=True)
class ImpactHit:
    env_index: int
    started_at: float
    point_w: tuple[float, float, float]
    direction_w: tuple[float, float, float]
    peak_n: float
    duration_s: float
    stress: bool
    body_name: str


class ImpactHistory:
    """Bounded snapshots, expired using simulation time rather than wall time."""

    def __init__(self, hold_s: float = 0.3, max_hits: int = 512):
        if not math.isfinite(hold_s) or hold_s <= 0:
            raise ValueError("Impact marker hold time must be positive and finite.")
        if not isinstance(max_hits, int) or max_hits <= 0:
            raise ValueError("Impact marker capacity must be a positive integer.")
        self.hold_s = hold_s
        self.max_hits = max_hits
        self._hits: list[ImpactHit] = []

    def capture(self, runtime, env_ids: torch.Tensor, now: float) -> list[ImpactHit]:
        ids = torch.as_tensor(env_ids, device=runtime.device, dtype=torch.long).flatten()
        bodies = runtime.body_id[ids]
        data = runtime.robot.data
        quaternion = data.body_link_quat_w[ids, bodies]
        local_point = runtime.hit_point_b[ids]
        twice = 2.0 * torch.cross(quaternion[:, 1:], local_point, dim=-1)
        rotated = local_point + quaternion[:, :1] * twice + torch.cross(quaternion[:, 1:], twice, dim=-1)
        points = (data.body_link_pos_w[ids, bodies] + rotated).detach().cpu().tolist()
        directions = runtime.force_direction_w[ids].detach().cpu().tolist()
        peaks = runtime.peak_n[ids].detach().cpu().tolist()
        durations = runtime.duration[ids].detach().cpu().tolist()
        kinds = runtime.episode_kind[ids].detach().cpu().tolist()
        body_indices = bodies.detach().cpu().tolist()
        hits = [
            ImpactHit(
                env_index=env_index, started_at=now, point_w=tuple(points[i]),
                direction_w=tuple(directions[i]), peak_n=peaks[i], duration_s=durations[i],
                stress=kinds[i] == 2, body_name=runtime.robot.body_names[body_indices[i]],
            )
            for i, env_index in enumerate(ids.detach().cpu().tolist())
        ]
        self.visible(now)
        self._hits = (self._hits + hits)[-self.max_hits:]
        return hits

    def visible(self, now: float) -> list[ImpactHit]:
        self._hits = [hit for hit in self._hits if 0 <= now - hit.started_at < self.hold_s]
        return list(self._hits)

    def reset(self, env_ids) -> None:
        indices = set(torch.as_tensor(env_ids).flatten().cpu().tolist())
        self._hits = [hit for hit in self._hits if hit.env_index not in indices]


def marker_arrays(hits: list[ImpactHit]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Arrow outside the body, a thin guide to the application point, and a sphere.

    The prototypes have unit radius/height and their long axes point along +Z.
    Arrow length indicates the clamped pulse peak, not instantaneous force.
    The guide keeps arrowheads readable even when the force point is inside a mesh.
    """
    positions, orientations, scales, indices = [], [], [], []
    for hit in hits:
        point = np.asarray(hit.point_w, dtype=np.float64)
        direction = np.asarray(hit.direction_w, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        # Shortest rotation from +Z to the incoming world-space force direction.
        if direction[2] < -1.0 + 1e-10:
            quaternion = np.asarray((0.0, 1.0, 0.0, 0.0))
        else:
            quaternion = np.asarray((1.0 + direction[2], -direction[1], direction[0], 0.0))
            quaternion /= np.linalg.norm(quaternion)
        length = float(np.clip(0.20 + 0.00015 * hit.peak_n, 0.20, 0.95))
        head_length = min(0.10, length * 0.30)
        shaft_length = length - head_length
        arrow_tip = point - direction * ARROW_STANDOFF_M
        positions.extend((
            arrow_tip - direction * (head_length + shaft_length / 2),
            arrow_tip - direction * (head_length / 2),
            point - direction * (ARROW_STANDOFF_M / 2),
            point,
        ))
        orientations.extend((quaternion, quaternion, quaternion, (1.0, 0.0, 0.0, 0.0)))
        scales.extend((
            (0.012, 0.012, shaft_length), (0.035, 0.035, head_length),
            (0.0035, 0.0035, ARROW_STANDOFF_M), (0.025, 0.025, 0.025),
        ))
        first = 3 if hit.stress else 0
        indices.extend((first, first + 1, first, first + 2))
    return (
        np.asarray(positions, dtype=np.float32).reshape(-1, 3),
        np.asarray(orientations, dtype=np.float32).reshape(-1, 4),
        np.asarray(scales, dtype=np.float32).reshape(-1, 3),
        np.asarray(indices, dtype=np.int32),
    )


def _create_markers():
    # Imported only after Kit starts, and only when visualization is requested.
    # Built-in geometry avoids downloading remote USD arrow assets.
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    prototypes = {}
    for name, color in (("common", (1.0, 0.35, 0.02)), ("stress", (1.0, 0.02, 0.02))):
        material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=color, emissive_color=(color[0] * 0.3, color[1] * 0.3, color[2] * 0.3),
        )
        prototypes[f"{name}_shaft"] = sim_utils.CylinderCfg(
            radius=1.0, height=1.0, axis="Z", visual_material=material,
        )
        prototypes[f"{name}_head"] = sim_utils.ConeCfg(
            radius=1.0, height=1.0, axis="Z", visual_material=material,
        )
        point_color = (1.0, 0.9, 0.05) if name == "common" else color
        prototypes[f"{name}_point"] = sim_utils.SphereCfg(
            radius=1.0, visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=point_color,
                emissive_color=(point_color[0] * 0.3, point_color[1] * 0.3, point_color[2] * 0.3),
            ),
        )
    markers = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/ImpactHits", markers=prototypes))
    markers.set_visibility(False)
    return markers


class ImpactVisualizationHook(PlayerHook):
    """Observe physical hit onsets without changing the policy or force sampler."""

    def __init__(self, env, hold_s: float = 0.3):
        self.env = env
        self.history = ImpactHistory(hold_s)
        self.runtime = env.impact_runtime
        self.markers = None
        self._observer = self._capture

    def _time(self) -> float:
        return self.env._sim_step_counter * self.env.physics_dt

    def init(self, player):
        super().init(player)
        if self.runtime.hit_observer is not None:
            raise RuntimeError("An impact hit observer is already attached.")
        self.markers = _create_markers()
        self.runtime.hit_observer = self._observer
        print(
            f"[INFO] Impact visualization enabled: orange=common, red=stress, sphere=hit point; "
            f"arrows point along the applied force, hold={self.history.hold_s:g}s of simulation time.", flush=True,
        )

    def _capture(self, env_ids):
        hits = self.history.capture(self.runtime, env_ids, self._time())
        for hit in hits:
            if hit.env_index == 0:
                print(
                    f"[HIT] env=0 body={hit.body_name} kind={'stress' if hit.stress else 'common'} "
                    f"peak={hit.peak_n:.0f}N pulse={hit.duration_s * 1000:.1f}ms", flush=True,
                )

    def step(self, step, transition, metrics):
        del step, transition, metrics
        self._refresh()

    def reset(self, indices):
        self.history.reset(indices)
        self._refresh()

    def _refresh(self):
        if self.markers is None:
            return
        hits = self.history.visible(self._time())
        self.markers.set_visibility(bool(hits))
        if hits:
            positions, orientations, scales, indices = marker_arrays(hits)
            self.markers.visualize(
                translations=positions, orientations=orientations, scales=scales, marker_indices=indices,
            )

    def close(self):
        if self.runtime.hit_observer is self._observer:
            self.runtime.hit_observer = None
        if self.markers is not None:
            self.markers.set_visibility(False)
