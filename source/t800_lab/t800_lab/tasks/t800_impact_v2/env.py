"""Stage-one logical actions with fixed head targets and physics-step impacts."""

from __future__ import annotations

import copy
import re

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

from t800_lab.tasks.t800_base.ankle_stable_mdp import ANKLE_JOINT_NAMES
from t800_lab.tasks.t800_base.constants import T800_FOOT_BODY_NAMES

from .impact import ImpactRuntime
from .mdp import (
    HEAD_JOINT_NAMES,
    LOGICAL_JOINT_NAMES,
    AnkleStabilityAccumulator,
    arena_margin,
    foot_side_ground_contact,
    update_head_faults,
)


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
            raise ValueError(
                f"Invalid impact action joint mapping; missing body={sorted(missing_body)}, unknown={sorted(unknown)}"
            )
        self._physical_to_logical = torch.tensor(
            [LOGICAL_JOINT_NAMES.index(name) for name in self._joint_names],
            device=self.device,
            dtype=torch.long,
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
            raise ValueError(
                f"Expected logical action shape {tuple(self._logical_raw_actions.shape)}, got {tuple(actions.shape)}"
            )
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


class T800ImpactV2Env(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        self._capture_ankle_physics = False
        super().__init__(cfg, **kwargs)
        self._validate_ground_contact_filters()
        self.impact_runtime = ImpactRuntime(self, cfg.impact)
        self._ankle_robot = self.scene["robot"]
        self._ankle_contact = self.scene["contact_forces"]
        self._ankle_joint_ids, names = self._ankle_robot.find_joints(list(ANKLE_JOINT_NAMES), preserve_order=True)
        self._ankle_foot_ids, foot_names = self._ankle_robot.find_bodies(T800_FOOT_BODY_NAMES, preserve_order=True)
        self._ankle_sensor_ids, sensor_names = self._ankle_contact.find_bodies(
            T800_FOOT_BODY_NAMES, preserve_order=True
        )
        if (
            tuple(names) != tuple(ANKLE_JOINT_NAMES)
            or foot_names != T800_FOOT_BODY_NAMES
            or sensor_names != T800_FOOT_BODY_NAMES
        ):
            raise ValueError("ImpactV2 ankle mapping is incomplete.")
        self.ankle_stability = AnkleStabilityAccumulator(
            self.num_envs,
            self.device,
            cfg.ankle_substep_stability,
            self.physics_dt,
            cfg.decimation,
        )
        self._original_scene_update = self.scene.update
        self.scene.update = self._update_scene_and_ankles

    def _update_scene_and_ankles(self, dt):
        self._original_scene_update(dt)
        if self._capture_ankle_physics:
            data = self._ankle_robot.data
            self.ankle_stability.observe(
                data.joint_vel[:, self._ankle_joint_ids],
                data.body_ang_vel_w[:, self._ankle_foot_ids],
                data.body_quat_w[:, self._ankle_foot_ids],
                self._ankle_contact.data.net_forces_w[:, self._ankle_sensor_ids],
                self._sim_step_counter,
                dt,
            )

    def _validate_ground_contact_filters(self):
        from pxr import UsdPhysics

        from .mdp import GROUND_SENSOR_PREFIX

        paths = {
            path
            for name, sensor in self.scene.sensors.items()
            if name.startswith(GROUND_SENSOR_PREFIX)
            for path in sensor.cfg.filter_prim_paths_expr
        }
        if not paths:
            raise RuntimeError("Impact fall detection requires ground contact filters.")
        for path in paths:
            prim = self.sim.stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
                raise RuntimeError(
                    f"Impact ground contact filter {path!r} is not a collision prim. "
                    "Filter the terrain collision Mesh, not its parent Xform."
                )

    def _reset_idx(self, env_ids):
        runtime = getattr(self, "impact_runtime", None)
        metrics = runtime.log_reset(env_ids) if runtime is not None else {}
        super()._reset_idx(env_ids)
        if runtime is not None:
            runtime.reset(env_ids)
            self.extras.setdefault("log", {}).update(metrics)

        if getattr(self, "ankle_stability", None) is not None:
            self.ankle_stability.reset(env_ids)

    def step(self, action):
        self._impact_foot_ground_flags = None
        self.ankle_stability.begin(self._ankle_robot.data.joint_vel[:, self._ankle_joint_ids], self._sim_step_counter)
        self._capture_ankle_physics = True
        try:
            result = super().step(action)
        finally:
            self._capture_ankle_physics = False
        # The reward pass happened before terminated environments were reset.
        # Do not recompute this from the post-reset state.
        result[-1]["impact_style_scale"] = self.impact_runtime.last_step_style_scale.clone().unsqueeze(-1)
        result[-1].setdefault("log", {})["Impact/arena_root_margin_m"] = arena_margin(self).mean()
        result[-1].setdefault("log", {}).update(self.ankle_stability.metrics())
        return result

    def close(self):
        self._capture_ankle_physics = False
        scene = getattr(self, "scene", None)
        if scene is not None and scene.update == self._update_scene_and_ankles:
            scene.update = self._original_scene_update
        super().close()

    def _record_impact_ground_contact(self):
        # Called immediately before a physics step. The first call sees the
        # previous control step, already checked; the others see this step's
        # first three samples. Termination checks the final sample directly.
        if self._sim_step_counter % self.cfg.decimation == 1:
            return
        invalid = foot_side_ground_contact(self)
        if self._impact_foot_ground_flags is None:
            self._impact_foot_ground_flags = invalid
        else:
            self._impact_foot_ground_flags |= invalid
