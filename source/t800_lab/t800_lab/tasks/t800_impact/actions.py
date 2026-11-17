"""Stage-one logical actions with fixed head targets and physics-step impacts."""

from __future__ import annotations

import copy
import re

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

from .head_mdp import HEAD_JOINT_NAMES, LOGICAL_JOINT_NAMES, update_head_faults


def _remove_missing_head_patterns(values, physical_names):
    """Discard only head-only selectors absent from the headless articulation."""
    def keep(pattern):
        return any(re.fullmatch(pattern, name) for name in physical_names) or not any(
            re.fullmatch(pattern, name) for name in HEAD_JOINT_NAMES
        )

    if isinstance(values, dict):
        return {pattern: value for pattern, value in values.items() if keep(pattern)}
    if isinstance(values, (tuple, list)):
        return [pattern for pattern in values if keep(pattern)]
    return values


class ImpactJointPositionAction(JointPositionAction):
    """Keep sampled logical actions intact while only controlling body outputs.

    The PPO action manager and this term's ``raw_actions`` retain the original
    25-dimensional draw. Only execution maps to physical joints and fixes head
    targets. This preserves the sampled-action/log-probability relationship.
    """

    def __init__(self, cfg, env):
        self._initializing = True
        physical_names = env.scene[cfg.asset_name].joint_names
        physical_cfg = copy.copy(cfg)
        for field in ("joint_names", "scale", "offset", "clip"):
            setattr(physical_cfg, field, _remove_missing_head_patterns(getattr(cfg, field), physical_names))
        super().__init__(physical_cfg, env)
        missing_body = set(LOGICAL_JOINT_NAMES[:-2]).difference(self._joint_names)
        unknown = set(self._joint_names).difference(LOGICAL_JOINT_NAMES)
        if missing_body or unknown:
            raise ValueError(f"Invalid impact action joint mapping; missing body={sorted(missing_body)}, unknown={sorted(unknown)}")
        self._physical_to_logical = torch.tensor(
            [LOGICAL_JOINT_NAMES.index(name) for name in self._joint_names], device=self.device, dtype=torch.long,
        )
        self._head_columns = [i for i, name in enumerate(self._joint_names) if name in HEAD_JOINT_NAMES]
        self._logical_raw_actions = torch.zeros(self.num_envs, len(LOGICAL_JOINT_NAMES), device=self.device)
        self._head_default = self._asset.data.default_joint_pos[:, self._joint_ids][:, self._head_columns].clone()
        if not torch.isfinite(self._head_default).all():
            raise ValueError("Head default targets must be finite.")
        self._initializing = False
        self._fix_head_targets()

    @property
    def action_dim(self) -> int:
        # The base initializer must size affine transforms for real joints.
        return self._num_joints if self._initializing else len(LOGICAL_JOINT_NAMES)

    @property
    def raw_actions(self) -> torch.Tensor:
        if self._initializing:
            return self._raw_actions
        return self._logical_raw_actions

    def _fix_head_targets(self) -> None:
        self._processed_actions[:, self._head_columns] = self._head_default

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._logical_raw_actions.shape:
            raise ValueError(f"Expected logical action shape {tuple(self._logical_raw_actions.shape)}, got {tuple(actions.shape)}")
        self._logical_raw_actions.copy_(actions)
        physical = actions.index_select(-1, self._physical_to_logical)
        physical[:, self._head_columns] = 0.0
        super().process_actions(physical)
        self._fix_head_targets()

    def apply_actions(self) -> None:
        super().apply_actions()
        record_ground = getattr(self._env, "_record_impact_ground_contact", None)
        if record_ground is not None:
            record_ground()
        update_head_faults(self._env, float(self._env.physics_dt))
        runtime = getattr(self._env, "impact_runtime", None)
        if runtime is not None:
            runtime.physics_step()

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._logical_raw_actions[env_ids] = 0.0
        self._fix_head_targets()


@configclass
class ImpactJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = ImpactJointPositionAction
