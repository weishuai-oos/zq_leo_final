from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import omni.physics.tensors.impl.api as physx
import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.assets import Articulation
    from t800_lab.motion_reference import MotionReferenceManager


class projected_gravity_reference_as_state(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.motion_ref: MotionReferenceManager = (
            env.scene[cfg.params["asset_cfg"].name] if "asset_cfg" in cfg.params else env.scene["motion_reference"]
        )

        # Obtain global physics sim view
        physics_sim_view = physx.create_simulation_view("torch")
        physics_sim_view.set_subspace_roots("/")
        gravity = physics_sim_view.get_gravity()
        # Convert to direction vector
        gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]), device=self.device)
        gravity_dir = math_utils.normalize(gravity_dir.unsqueeze(0)).squeeze(0)

        self.GRAVITY_VEC_W = gravity_dir.repeat(len(self.motion_ref.ALL_INDICES), 1)

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    ):
        """The projected gravity in the robot base frame as state.
        Returns:
            (num_envs, 3)
        """
        motion_reference: MotionReferenceManager = env.scene[asset_cfg.name]
        root_quat_w = motion_reference.reference_frame.base_quat_w[:, 0]
        projected_gravity_w = math_utils.quat_apply_inverse(root_quat_w, self.GRAVITY_VEC_W)
        return projected_gravity_w


def joint_pos_rel_reference_as_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Joint positions (relative to robot's default joint pos) in the robot base frame as state, but from motion reference
    Returns:
        (num_envs, num_joints)
    """
    motion_reference: MotionReferenceManager = env.scene[asset_cfg.name]
    joint_pos = motion_reference.reference_frame.joint_pos[:, 0, asset_cfg.joint_ids]
    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos_rel = joint_pos - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    return joint_pos_rel * motion_reference.reference_frame.joint_pos_mask[:, 0, asset_cfg.joint_ids]


def joint_vel_rel_reference_as_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mask: bool = True,
):
    """Joint velocities (relative to robot's default joint vel) in the robot base frame as state, but from motion reference
    Returns:
        (num_envs, num_joints)
    """
    motion_reference: MotionReferenceManager = env.scene[asset_cfg.name]
    joint_vel = motion_reference.reference_frame.joint_vel[:, 0, asset_cfg.joint_ids]
    if mask:
        joint_vel *= motion_reference.reference_frame.joint_vel_mask[:, 0, asset_cfg.joint_ids]
    robot: Articulation = env.scene[robot_cfg.name]
    joint_vel_rel = joint_vel - robot.data.default_joint_vel[:, asset_cfg.joint_ids]
    return joint_vel_rel
