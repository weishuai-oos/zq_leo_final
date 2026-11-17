"""Pure tensor helpers and startup events for T800 dynamics randomization.

The randomization is deliberately based on the immutable PhysX defaults.  This
is important for partial environment resets: a second startup call must never
randomize from an already randomized value.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import torch

from t800_lab.tasks.t800_impact_v2 import mdp as impact_mdp


BODY_JOINTS = 23
HEAD_SLOTS = 2
TORSO_BODY_NAME = "LINK_TORSO_YAW"


@dataclass(frozen=True)
class DynamicsRandomizationCfg:
    """Serializable DR contract shared by training and play configurations."""

    joint_offset_range: tuple[float, float] = (-0.01, 0.01)
    mass_scale_range: tuple[float, float] = (0.9, 1.1)
    torso_mass_delta_range: tuple[float, float] = (-2.0, 2.0)
    torso_com_offset_range: tuple[float, float, float] = (0.025, 0.05, 0.05)
    version: str = "t800-dynamics-dr-v1"

    def signature(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def sample_joint_offsets(
    num_envs: int,
    device: torch.device | str,
    low: float = -0.01,
    high: float = 0.01,
    *,
    enabled: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one 23-joint offset per episode; the two head slots stay zero."""
    offsets = torch.zeros((num_envs, BODY_JOINTS + HEAD_SLOTS), device=device)
    if enabled and num_envs:
        offsets[:, :BODY_JOINTS] = torch.empty((num_envs, BODY_JOINTS), device=device).uniform_(
            low, high, generator=generator
        )
    return offsets


def add_physical_target_offsets(
    physical_targets: torch.Tensor,
    physical_to_logical: torch.Tensor,
    logical_offsets: torch.Tensor,
) -> torch.Tensor:
    """Add logical offsets after the parent action has applied its affine map."""
    if logical_offsets.shape[-1] != BODY_JOINTS + HEAD_SLOTS:
        raise ValueError("T800 DR offsets must have 25 logical slots")
    mapped = logical_offsets.index_select(-1, physical_to_logical)
    return physical_targets + mapped


def subtract_joint_offsets(values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Pair target offsets with absolute or relative robot observations."""
    if values.shape[-1] < BODY_JOINTS or offsets.shape[-1] != BODY_JOINTS + HEAD_SLOTS:
        raise ValueError("Unexpected T800 joint-offset shape")
    result = values.clone()
    result[..., :BODY_JOINTS] -= offsets[..., :BODY_JOINTS]
    return result


def subtract_amp_joint_offsets(values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Compensate AMP's 19 body-joint projection and its two masked head slots."""
    indices = torch.tensor(
        [impact_mdp.LOGICAL_JOINT_NAMES.index(name) for name in impact_mdp.AMP_JOINT_NAMES],
        device=values.device,
    )
    result = values.clone()
    result[..., : len(indices)] -= offsets.index_select(-1, indices)
    return result


def _offsets(env: Any, values: torch.Tensor) -> torch.Tensor:
    offsets = getattr(env, "_dynamics_joint_offsets", None)
    if offsets is None:
        return torch.zeros((values.shape[0], BODY_JOINTS + HEAD_SLOTS), device=values.device, dtype=values.dtype)
    return offsets.to(device=values.device, dtype=values.dtype)


def joint_pos_masked(env: Any, asset_cfg: Any = None, noise_range: tuple[float, float] | None = None) -> torch.Tensor:
    return subtract_joint_offsets(impact_mdp.joint_pos_masked(env, asset_cfg, noise_range), _offsets(env, env.scene[asset_cfg.name if asset_cfg else "robot"].data.joint_pos))


def joint_pos_rel_masked(env: Any, asset_cfg: Any = None) -> torch.Tensor:
    values = impact_mdp.joint_pos_rel_masked(env, asset_cfg)
    return subtract_joint_offsets(values, _offsets(env, values))


def amp_joint_pos_rel_masked(env: Any, asset_cfg: Any = None) -> torch.Tensor:
    values = impact_mdp.amp_joint_pos_rel_masked(env, asset_cfg)
    return subtract_amp_joint_offsets(values, _offsets(env, values))


def amp_joint_vel_masked(env: Any, asset_cfg: Any = None) -> torch.Tensor:
    return impact_mdp.amp_joint_vel_masked(env, asset_cfg)


def reference_amp_joint_pos_rel_masked(env: Any, asset_cfg: Any = None, robot_cfg: Any = None) -> torch.Tensor:
    return impact_mdp.reference_amp_joint_pos_rel_masked(env, asset_cfg, robot_cfg)


def reference_amp_joint_vel_masked(env: Any, asset_cfg: Any = None, robot_cfg: Any = None, mask: bool = True) -> torch.Tensor:
    return impact_mdp.reference_amp_joint_vel_masked(env, asset_cfg, robot_cfg, mask)


def _resolve_body_ids(asset_cfg: Any, asset: Any) -> tuple[torch.Tensor, list[str]]:
    names = list(getattr(asset, "body_names", ()))
    selected = getattr(asset_cfg, "body_ids", slice(None))
    if selected == slice(None):
        ids = list(range(getattr(asset, "num_bodies", len(names))))
    elif isinstance(selected, (tuple, list)):
        ids = list(selected)
    else:
        ids = [int(selected)]
    if names and max(ids, default=-1) >= len(names):
        raise IndexError("DR body selector is outside the articulation")
    return torch.tensor(ids, dtype=torch.long, device="cpu"), [names[i] for i in ids] if names else [str(i) for i in ids]


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-6):
        raise RuntimeError(f"PhysX DR readback mismatch: {label}")


def randomize_dynamics(
    env: Any,
    env_ids: torch.Tensor | None,
    asset_cfg: Any,
    mass_scale_range: tuple[float, float] = (0.9, 1.1),
    torso_mass_delta_range: tuple[float, float] = (-2.0, 2.0),
    torso_com_offset_range: tuple[float, float, float] = (0.025, 0.05, 0.05),
    distribution: str = "uniform",
) -> None:
    """Write all-link mass/inertia and torso COM from nominal defaults, once."""
    del distribution
    if not bool(getattr(getattr(env, "cfg", None), "dr_enabled", True)):
        return
    asset = env.scene[asset_cfg.name]
    view = asset.root_physx_view
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, dtype=torch.long, device="cpu")
    else:
        env_ids = env_ids.detach().cpu().to(torch.long)
    body_ids, body_names = _resolve_body_ids(asset_cfg, asset)
    if not hasattr(asset, "_t800_dr_nominal_mass"):
        asset._t800_dr_nominal_mass = view.get_masses().clone()
        asset._t800_dr_nominal_inertia = view.get_inertias().clone()
        asset._t800_dr_nominal_coms = view.get_coms().clone()
    defaults_mass = asset._t800_dr_nominal_mass
    masses = view.get_masses().clone()
    ratios = torch.empty((len(env_ids), len(body_ids)), device=masses.device).uniform_(*mass_scale_range)
    torso_id = body_names.index(TORSO_BODY_NAME) if TORSO_BODY_NAME in body_names else None
    target_mass = defaults_mass[env_ids[:, None], body_ids] * ratios
    if torso_id is not None:
        target_mass[:, torso_id] += torch.empty(len(env_ids), device=masses.device).uniform_(*torso_mass_delta_range)
    if not torch.isfinite(target_mass).all() or not (target_mass > 0).all():
        raise ValueError("T800 DR sampled non-positive or non-finite link mass")
    ratios = target_mass / defaults_mass[env_ids[:, None], body_ids]
    masses[env_ids[:, None], body_ids] = target_mass
    view.set_masses(masses, env_ids)
    inertias = view.get_inertias().clone()
    defaults_inertia = asset._t800_dr_nominal_inertia
    inertias[env_ids[:, None], body_ids] = defaults_inertia[env_ids[:, None], body_ids] * ratios.unsqueeze(-1)
    view.set_inertias(inertias, env_ids)
    if torso_id is not None:
        coms = view.get_coms().clone()
        nominal = asset._t800_dr_nominal_coms
        delta = torch.stack([torch.empty(len(env_ids), device=coms.device).uniform_(-bound, bound) for bound in torso_com_offset_range], -1)
        target_com = nominal[env_ids, body_ids[torso_id]].clone()
        target_com[..., :3] += delta
        coms[env_ids, body_ids[torso_id], :3] = target_com[..., :3]
        view.set_coms(coms, env_ids)
    _assert_close(view.get_masses()[env_ids[:, None], body_ids], masses[env_ids[:, None], body_ids], "mass")
    _assert_close(view.get_inertias()[env_ids[:, None], body_ids], inertias[env_ids[:, None], body_ids], "inertia")
    if torso_id is not None:
        _assert_close(view.get_coms()[env_ids, body_ids[torso_id], :3], target_com[..., :3], "torso COM")
    env.extras.setdefault("log", {})["DR/mass_scale_min"] = ratios.min()
    env.extras["log"]["DR/mass_scale_max"] = ratios.max()
    if torso_id is not None:
        env.extras["log"]["DR/torso_com_offset_max_m"] = delta.abs().max()


