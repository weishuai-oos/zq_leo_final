"""Thin continuation configurations: parent behavior is unchanged except DR."""

from __future__ import annotations

import copy

import torch

from isaaclab.envs import mdp as sim_mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from t800_lab.tasks.t800_impact_v2.env_cfg import T800ImpactV2EnvCfg
from t800_lab.tasks.t800_impact_v2_plus.env_cfg import T800ImpactV2PlusEnvCfg

from . import mdp
from .env import DynamicsDRImpactJointPositionActionPlusCfg, DynamicsDRImpactJointPositionActionV2Cfg


class _MaterialReadbackTerm(sim_mdp.randomize_rigid_body_material):
    """Parent material sampler plus a startup PhysX readback validation."""

    def __call__(self, env, env_ids, static_friction_range, dynamic_friction_range, restitution_range,
                 num_buckets, asset_cfg, make_consistent=False):
        del static_friction_range, dynamic_friction_range, restitution_range, num_buckets, make_consistent
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()
        total = self.asset.root_physx_view.max_shapes
        bucket_ids = torch.randint(0, self.material_buckets.shape[0], (len(env_ids), total), device="cpu")
        samples = self.material_buckets[bucket_ids]
        expected = self.asset.root_physx_view.get_material_properties().clone()
        if self.num_shapes_per_body is None:
            expected[env_ids] = samples
        else:
            for body_id in asset_cfg.body_ids:
                start = sum(self.num_shapes_per_body[:body_id])
                end = start + self.num_shapes_per_body[body_id]
                expected[env_ids, start:end] = samples[:, start:end]
        self.asset.root_physx_view.set_material_properties(expected, env_ids)
        actual = self.asset.root_physx_view.get_material_properties()
        if not torch.allclose(actual[env_ids], expected[env_ids], rtol=1e-5, atol=1e-6):
            raise RuntimeError("PhysX material readback does not match the requested friction values")


def _freeze_parent(cfg, task_name: str, iteration: int) -> None:
    parent_cfg = copy.deepcopy(cfg)
    cfg.dr_parent_cfg = parent_cfg
    cfg.dr_parent_task = task_name
    cfg.dr_source_iteration = iteration
    cfg.dynamics_randomization = mdp.DynamicsRandomizationCfg()
    cfg.dr_enabled = True


def _install_dr(cfg, action_cfg) -> None:
    cfg.actions.joint_pos = action_cfg(
        asset_name="robot",
        joint_names=cfg.actions.joint_pos.joint_names,
        preserve_order=True,
        scale=cfg.actions.joint_pos.scale,
        use_default_offset=True,
    )
    cfg.observations.policy.joint_pos.func = mdp.joint_pos_masked
    cfg.observations.critic.joint_pos.func = mdp.joint_pos_rel_masked
    cfg.observations.amp_policy.joint_pos_rel.func = mdp.amp_joint_pos_rel_masked
    # Reference AMP is deliberately not compensated: it is the nominal target.
    cfg.observations.amp_reference.joint_pos_rel.func = mdp.reference_amp_joint_pos_rel_masked
    cfg.events.add_base_mass = EventTermCfg(
        func=mdp.randomize_dynamics,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_scale_range": cfg.dynamics_randomization.mass_scale_range,
            "torso_mass_delta_range": cfg.dynamics_randomization.torso_mass_delta_range,
            "torso_com_offset_range": cfg.dynamics_randomization.torso_com_offset_range,
        },
    )
    material_params = dict(cfg.events.physics_material.params)
    cfg.events.physics_material = EventTermCfg(
        func=_MaterialReadbackTerm,
        mode="startup",
        params=material_params,
    )
    cfg.dr_signature = cfg.dynamics_randomization.signature()


@configclass
class T800DynamicsDRV2EnvCfg(T800ImpactV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _freeze_parent(self, "T800_base-ImpactV2", 154500)
        self.run_name = "T800_base-ImpactV2-From154500-DR"
        _install_dr(self, DynamicsDRImpactJointPositionActionV2Cfg)


@configclass
class T800DynamicsDRV2EnvCfg_PLAY(T800DynamicsDRV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.dr_enabled = False
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.curriculum.terrain_levels = None
        self.impact.mode = "common"


@configclass
class T800DynamicsDRPlusEnvCfg(T800ImpactV2PlusEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _freeze_parent(self, "T800_base-ImpactV2Plus", 173000)
        self.run_name = "T800_base-ImpactV2Plus-From173000-DR"
        _install_dr(self, DynamicsDRImpactJointPositionActionPlusCfg)


@configclass
class T800DynamicsDRPlusEnvCfg_PLAY(T800DynamicsDRPlusEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.dr_enabled = False
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.curriculum.terrain_levels = None
        self.impact.mode = "common"
