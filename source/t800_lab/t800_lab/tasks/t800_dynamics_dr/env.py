"""Impact environments with episode-scoped target-offset randomization."""

from __future__ import annotations

import torch
from isaaclab.utils import configclass

from t800_lab.tasks.t800_impact_v2.env import ImpactJointPositionAction as V2ImpactAction
from t800_lab.tasks.t800_impact_v2.env import ImpactJointPositionActionCfg as V2ImpactActionCfg
from t800_lab.tasks.t800_impact_v2.env import T800ImpactV2Env
from t800_lab.tasks.t800_impact_v2_plus.env import ImpactJointPositionAction as PlusImpactAction
from t800_lab.tasks.t800_impact_v2_plus.env import ImpactJointPositionActionCfg as PlusImpactActionCfg
from t800_lab.tasks.t800_impact_v2_plus.env import T800ImpactV2PlusEnv

from . import mdp


class _OffsetActionMixin:
    """Keep the parent 25-slot action contract while shifting physical targets."""

    def _setup_dr_offsets(self) -> None:
        cfg = self._env.cfg
        dr_cfg = cfg.dynamics_randomization
        enabled = bool(getattr(cfg, "dr_enabled", True))
        self._env._dynamics_joint_offsets = mdp.sample_joint_offsets(
            self.num_envs,
            self.device,
            *dr_cfg.joint_offset_range,
            enabled=enabled,
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        if not hasattr(self._env, "_dynamics_joint_offsets"):
            self._setup_dr_offsets()
        offsets = self._env._dynamics_joint_offsets.index_select(-1, self._physical_to_logical)
        self._processed_actions.add_(offsets)
        self._fix_head_targets()

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if not hasattr(self._env, "_dynamics_joint_offsets"):
            self._setup_dr_offsets()
        if env_ids is None or isinstance(env_ids, slice):
            selection = env_ids if isinstance(env_ids, slice) else slice(None)
            env_ids = torch.arange(self.num_envs, device=self.device)
            env_ids = env_ids[selection]
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        cfg = self._env.cfg
        self._env._dynamics_joint_offsets[env_ids] = mdp.sample_joint_offsets(
            len(env_ids), self.device, *cfg.dynamics_randomization.joint_offset_range,
            enabled=bool(getattr(cfg, "dr_enabled", True)),
        )
        self._env.extras.setdefault("log", {})["DR/joint_offset_abs_max_rad"] = self._env._dynamics_joint_offsets.abs().max()
        self._fix_head_targets()


class DynamicsDRImpactJointPositionActionV2(_OffsetActionMixin, V2ImpactAction):
    pass


class DynamicsDRImpactJointPositionActionPlus(_OffsetActionMixin, PlusImpactAction):
    pass


class T800DynamicsDRV2Env(T800ImpactV2Env):
    pass


class T800DynamicsDRPlusEnv(T800ImpactV2PlusEnv):
    pass


@configclass
class DynamicsDRImpactJointPositionActionV2Cfg(V2ImpactActionCfg):
    class_type: type = DynamicsDRImpactJointPositionActionV2


@configclass
class DynamicsDRImpactJointPositionActionPlusCfg(PlusImpactActionCfg):
    class_type: type = DynamicsDRImpactJointPositionActionPlus
