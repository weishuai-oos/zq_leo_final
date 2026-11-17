"""T800_base continuation: finite physical strikes and head-independent control."""

import xml.etree.ElementTree as ET

from isaaclab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from t800_lab.assets.t800 import T800_URDF_PATH
from t800_lab.tasks.t800_base.env_cfg import T800BaseEnvCfg

from . import head_mdp, mdp
from .actions import ImpactJointPositionActionCfg
from .impact_config import ImpactCfg
from .terrain_cfg import make_terrain


def reset_head_faults_event(env, env_ids):
    # IsaacLab's event validator requires env_ids to be a mandatory argument.
    head_mdp.reset_head_faults(env, env_ids)


def _ground_contact_sensors(scene):
    # ContactSensor supports one-to-many filtering, not many-to-many. Use one
    # source body per sensor so self/opponent/cage contacts never count as falls.
    bodies = [link.attrib["name"] for link in ET.parse(T800_URDF_PATH).getroot().findall("link")
              if link.find("collision") is not None]
    for body in bodies:
        foot = body in mdp.FOOT_BODIES
        setattr(scene, mdp.GROUND_SENSOR_PREFIX + body, ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + body,
            update_period=0.005,
            history_length=4,
            # Generated terrain's parent is an Xform. PhysX contact filters
            # must target the collision Mesh, otherwise filtered forces stay zero.
            filter_prim_paths_expr=["/World/ground/terrain/mesh"],
            track_pose=foot,
            track_contact_points=foot,
            max_contact_data_count_per_prim=16 if foot else 4,
        ))


@configclass
class T800ImpactEnvCfg(T800BaseEnvCfg):
    impact: ImpactCfg = ImpactCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "T800_base-Impact"
        self.scene.terrain = make_terrain()
        self.commands.base_velocity.debug_vis = False
        self.impact.tactical_front_offset_deg = self.commands.base_velocity.tactical_front_offset_deg
        self.actions.joint_pos = ImpactJointPositionActionCfg(
            asset_name="robot",
            joint_names=self.actions.joint_pos.joint_names,
            preserve_order=True,
            scale=self.actions.joint_pos.scale,
            use_default_offset=True,
        )
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.head_faults = EventTermCfg(func=reset_head_faults_event, mode="reset")
        # Reduce artificial rebound for the soft mat; friction retains the
        # stage-one range pending actual mat measurements.
        self.events.physics_material.params["restitution_range"] = (0.0, 0.10)
        _ground_contact_sensors(self.scene)
        self.terminations.base_contact = TerminationTermCfg(func=mdp.ground_fall)

        robot = SceneEntityCfg("robot")
        policy, critic = self.observations.policy, self.observations.critic
        # Noise is applied inside these functions, before the final head mask.
        policy.joint_pos = ObservationTermCfg(func=head_mdp.joint_pos_masked,
            params={"asset_cfg": robot, "noise_range": (-0.01, 0.01)})
        policy.joint_vel = ObservationTermCfg(func=head_mdp.joint_vel_masked,
            params={"asset_cfg": robot, "noise_range": (-1.5, 1.5)})
        policy.actions = ObservationTermCfg(func=head_mdp.last_action_masked)
        critic.joint_pos = ObservationTermCfg(func=head_mdp.joint_pos_rel_masked, params={"asset_cfg": robot})
        critic.joint_vel = ObservationTermCfg(func=head_mdp.joint_vel_masked, params={"asset_cfg": robot})
        critic.actions = ObservationTermCfg(func=head_mdp.last_action_masked)
        for group_name, functions, asset in (
            ("amp_policy", (head_mdp.amp_joint_pos_rel_masked, head_mdp.amp_joint_vel_masked), robot),
            ("amp_reference", (head_mdp.reference_amp_joint_pos_rel_masked,
                               head_mdp.reference_amp_joint_vel_masked), SceneEntityCfg("motion_reference")),
        ):
            group = getattr(self.observations, group_name)
            for name, function in zip(("joint_pos_rel", "joint_vel"), functions, strict=True):
                term = getattr(group, name)
                term.func = function
                term.params = {"asset_cfg": asset}

        self.rewards.joint_deviation_head = None
        self.rewards.standard_head_pose_l1 = None
        # No requirement to step after an impact, including when v_cmd is zero.
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts.func = mdp.nonsole_ground_contacts
        self.rewards.undesired_contacts.params = {"threshold": 1.0}
        self.rewards.action_rate_l2.func = mdp.body_action_rate
        self.rewards.action_rate_l2.params = {}
        for name in (
            "stand_still_without_cmd", "track_lin_vel_xy_error_l2", "track_ang_vel_z_error_l2",
            "standard_arm_pose_l1", "joint_deviation_torso", "joint_deviation_hip", "joint_deviation_knee",
            "dof_acc_l2", "ankle_target_rate_l2", "ankle_target_acceleration_l2", "ankle_acceleration_l2",
        ):
            term = getattr(self.rewards, name)
            original_params = dict(term.params)
            # IsaacLab only resolves top-level SceneEntityCfg values.
            asset_cfg = original_params.pop("asset_cfg", None)
            term.params = {"term_func": term.func, "term_params": original_params}
            if asset_cfg is not None:
                term.params["asset_cfg"] = asset_cfg
            term.func = mdp.recovery_scaled
        self.rewards.impact_scale_snapshot = RewardTermCfg(func=mdp.capture_recovery_scale, weight=1.0)


@configclass
class T800ImpactEnvCfg_PLAY(T800ImpactEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.curriculum.terrain_levels = None
        self.impact.mode = "common"
