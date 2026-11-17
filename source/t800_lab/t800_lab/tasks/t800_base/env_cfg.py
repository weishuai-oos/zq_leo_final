"""Explicit effective configuration of the T800 AMP AnkleStable-1 task.

The historical experiment inheritance is flattened here. The requested terrain
mix is the only behavioral change: 20% flat plus 80% of the old terrain mix.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg, mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as Event
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import t800_lab.envs.mdp as leo_mdp
import t800_lab.tasks.locomotion.mdp as locomotion_mdp
from t800_lab.assets.t800 import T800_CFG, T800_DFS_JOINT_NAMES, T800_URDF_PATH
from t800_lab.motion_reference import MotionReferenceManagerCfg
from t800_lab.motion_reference.motion_files.amass_motion_cfg import (
    AmassMotionCfg as AmassMotionCfgBase,
)
from t800_lab.motion_reference.utils import motion_interpolate_bilinear

from . import ankle_stable_mdp, boxing_guard_mdp
from .boxing_guard_velocity_command import T800BoxingGuardVelocityCommandCfg
from .constants import (
    AMP_HISTORY_LENGTH,
    DEFAULT_MOTION_DATA_PATH,
    DEFAULT_MOTION_SELECTION_PATH,
    T800_AMP_ACTION_SCALE,
    T800_AMP_JOINT_NAMES,
    T800_AMP_LINKS_OF_INTEREST,
    T800_ANKLE_JOINT_NAMES,
    T800_ARM_JOINT_NAMES,
    T800_BOXING_GUARD_NOMINAL_JOINT_POS,
    T800_FOOT_BODY_NAMES,
    T800_HEAD_JOINT_NAMES,
    T800_HIP_DEVIATION_JOINT_NAMES,
    T800_KNEE_JOINT_NAMES,
    T800_LEG_ACC_JOINT_NAMES,
    T800_STAND_STILL_JOINT_NAMES,
    _load_t800_standard_upper_pose,
)
from .terrain_cfg import make_terrain


@configclass
class T800MotionCfg(AmassMotionCfgBase):
    path = DEFAULT_MOTION_DATA_PATH
    retargetting_func = None
    filtered_motion_selection_filepath = DEFAULT_MOTION_SELECTION_PATH
    motion_start_from_middle_range = [0.0, 0.8]
    motion_start_height_offset = 0.0
    ensure_link_below_zero_ground = False
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontward"


motion_reference_cfg = MotionReferenceManagerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/LINK_BASE",
    robot_model_path=T800_URDF_PATH,
    reference_prim_path=None,
    frame_interval_s=0.02,
    update_period=0.02,
    num_frames=AMP_HISTORY_LENGTH,
    motion_buffers={"boxing_guard": T800MotionCfg()},
    link_of_interests=T800_AMP_LINKS_OF_INTEREST,
    mp_split_method="None",
)


@configclass
class T800SceneCfg(InteractiveSceneCfg):
    terrain = make_terrain()
    motion_reference: MotionReferenceManagerCfg = motion_reference_cfg
    robot = T800_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class T800ActionsCfg:
    joint_pos = leo_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=T800_DFS_JOINT_NAMES,
        preserve_order=True,
        scale=T800_AMP_ACTION_SCALE,
        use_default_offset=True,
    )


@configclass
class T800CommandsCfg:
    """Narrow boxing commands with a moderate pure-turn training range."""

    base_velocity = T800BoxingGuardVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 6.0),
        rel_standing_envs=0.10,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.35, 1.35),
            lin_vel_y=(-1.35, 1.35),
            ang_vel_z=(-2.10, 2.10),
            heading=None,
        ),
        mode_weights=(0.10, 0.50, 0.20, 0.15, 0.05),
        cardinal_speed_range=(0.65, 1.10),
        other_speed_range=(0.80, 1.35),
        turn_rate_range=(1.00, 2.10),
        mixed_speed_range=(0.80, 1.25),
        mixed_turn_rate_range=(0.50, 1.80),
    )


@configclass
class T800PolicyObsCfg(ObsGroup):
    base_ang_vel = ObsTerm(func=leo_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(
        func=leo_mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(
        func=mdp.joint_pos,
        noise=Unoise(n_min=-0.01, n_max=0.01),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=T800_DFS_JOINT_NAMES, preserve_order=True)},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel,
        noise=Unoise(n_min=-1.5, n_max=1.5),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=T800_DFS_JOINT_NAMES, preserve_order=True)},
    )
    actions = ObsTerm(func=leo_mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False


@configclass
class T800CriticObsCfg(ObsGroup):
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=leo_mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=leo_mdp.projected_gravity)
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=T800_DFS_JOINT_NAMES, preserve_order=True)},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=T800_DFS_JOINT_NAMES, preserve_order=True)},
    )
    actions = ObsTerm(func=leo_mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class T800AmpPolicyObsCfg(ObsGroup):
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
        history_length=AMP_HISTORY_LENGTH,
        flatten_history_dim=True,
    )
    projected_gravity = ObsTerm(
        func=leo_mdp.projected_gravity,
        params={"asset_cfg": SceneEntityCfg("robot")},
        history_length=AMP_HISTORY_LENGTH,
    )
    joint_pos_rel = ObsTerm(
        func=mdp.joint_pos_rel,
        history_length=AMP_HISTORY_LENGTH,
        flatten_history_dim=True,
        params={
            "asset_cfg": SceneEntityCfg(
                name="robot",
                joint_names=T800_AMP_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        scale=0.05,
        history_length=AMP_HISTORY_LENGTH,
        flatten_history_dim=True,
        params={
            "asset_cfg": SceneEntityCfg(
                name="robot",
                joint_names=T800_AMP_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class T800AmpReferenceObsCfg(ObsGroup):
    velocity_commands = ObsTerm(
        func=boxing_guard_mdp.reference_velocity_command,
        params={"asset_cfg": SceneEntityCfg(name="motion_reference")},
        history_length=AMP_HISTORY_LENGTH,
        flatten_history_dim=True,
    )
    projected_gravity = ObsTerm(
        func=leo_mdp.projected_gravity_reference_as_state,
        params={"asset_cfg": SceneEntityCfg(name="motion_reference")},
        history_length=AMP_HISTORY_LENGTH,
    )
    joint_pos_rel = ObsTerm(
        func=leo_mdp.joint_pos_rel_reference_as_state,
        history_length=AMP_HISTORY_LENGTH,
        flatten_history_dim=True,
        params={
            "asset_cfg": SceneEntityCfg(
                name="motion_reference",
                joint_names=T800_AMP_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )
    joint_vel = ObsTerm(
        func=leo_mdp.joint_vel_rel_reference_as_state,
        scale=0.05,
        history_length=AMP_HISTORY_LENGTH,
        flatten_history_dim=True,
        params={
            "asset_cfg": SceneEntityCfg(
                name="motion_reference",
                joint_names=T800_AMP_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class T800ObservationsCfg:
    policy: T800PolicyObsCfg = T800PolicyObsCfg()
    critic: T800CriticObsCfg = T800CriticObsCfg()
    amp_policy: T800AmpPolicyObsCfg = T800AmpPolicyObsCfg()
    amp_reference: T800AmpReferenceObsCfg = T800AmpReferenceObsCfg()


_ANKLE_CFG = SceneEntityCfg("robot", joint_names=list(ankle_stable_mdp.ANKLE_JOINT_NAMES), preserve_order=True)


@configclass
class T800RewardsCfg:
    track_lin_vel_xy_exp = RewTerm(
        func=locomotion_mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.6,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=locomotion_mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.6)
    stand_still_without_cmd = RewTerm(
        func=locomotion_mdp.stand_still_without_cmd,
        weight=-0.15,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=T800_STAND_STILL_JOINT_NAMES, preserve_order=True),
            "command_threshold": 0.02,
        },
    )
    energy = RewTerm(func=locomotion_mdp.energy, weight=-1e-05)
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2e-07,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=T800_LEG_ACC_JOINT_NAMES)},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.0045)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["^(?!LINK_ANKLE_ROLL_L$)(?!LINK_ANKLE_ROLL_R$).+$"],
            ),
            "threshold": 1.0,
        },
    )
    feet_slide = RewTerm(
        func=locomotion_mdp.feet_slide,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=T800_FOOT_BODY_NAMES),
            "asset_cfg": SceneEntityCfg("robot", body_names=T800_FOOT_BODY_NAMES),
        },
    )
    feet_force = RewTerm(
        func=locomotion_mdp.contact_force,
        weight=-0.0005,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=T800_FOOT_BODY_NAMES),
            "threshold": 1000,
            "max_reward": 400,
        },
    )
    fly = RewTerm(
        func=locomotion_mdp.fly,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=T800_FOOT_BODY_NAMES),
            "threshold": 1.0,
        },
    )
    feet_air_time = RewTerm(
        func=locomotion_mdp.feet_air_time_positive_biped_tuned,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=T800_FOOT_BODY_NAMES),
            "threshold": 0.18,
            "command_threshold": 0.08,
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=T800_ANKLE_JOINT_NAMES)},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=T800_HIP_DEVIATION_JOINT_NAMES)},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=T800_ARM_JOINT_NAMES)},
    )
    joint_deviation_head = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=T800_HEAD_JOINT_NAMES)},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["J12_TORSO_YAW"])},
    )
    joint_deviation_knee = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=T800_KNEE_JOINT_NAMES)},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_error_l2 = RewTerm(
        func=boxing_guard_mdp.track_lin_vel_xy_yaw_frame_l2,
        weight=-1.0,
        params={"command_name": "base_velocity"},
    )
    track_ang_vel_z_error_l2 = RewTerm(
        func=boxing_guard_mdp.track_ang_vel_z_world_l2,
        weight=-0.5,
        params={"command_name": "base_velocity"},
    )
    standard_arm_pose_l1 = RewTerm(
        func=locomotion_mdp.joint_pos_target_l1,
        weight=-0.2,
        params={
            "target_joint_pos": tuple((_load_t800_standard_upper_pose()[name] for name in T800_ARM_JOINT_NAMES)),
            "asset_cfg": SceneEntityCfg("robot", joint_names=T800_ARM_JOINT_NAMES, preserve_order=True),
            "normalize_by_num_joints": True,
        },
    )
    standard_head_pose_l1 = RewTerm(
        func=locomotion_mdp.joint_pos_target_l1,
        weight=-0.2,
        params={
            "target_joint_pos": tuple((_load_t800_standard_upper_pose()[name] for name in T800_HEAD_JOINT_NAMES)),
            "asset_cfg": SceneEntityCfg("robot", joint_names=T800_HEAD_JOINT_NAMES, preserve_order=True),
            "normalize_by_num_joints": True,
        },
    )
    ankle_target_rate_l2 = RewTerm(
        func=ankle_stable_mdp.ankle_target_rate_l2,
        weight=-0.01,
        params={"asset_cfg": _ANKLE_CFG, "dt": 0.02},
    )
    ankle_target_acceleration_l2 = RewTerm(
        func=ankle_stable_mdp.ankle_target_acceleration_l2,
        weight=-0.002,
        params={"asset_cfg": _ANKLE_CFG, "dt": 0.02},
    )
    ankle_overspeed_hinge_l2 = RewTerm(
        func=ankle_stable_mdp.ankle_overspeed_hinge_l2,
        weight=-0.025,
        params={"asset_cfg": _ANKLE_CFG, "dt": 0.02},
    )
    ankle_acceleration_l2 = RewTerm(
        func=ankle_stable_mdp.ankle_acceleration_l2,
        weight=-0.001,
        params={"asset_cfg": _ANKLE_CFG, "dt": 0.02},
    )
    ankle_target_soft_limit_hinge_l2 = RewTerm(
        func=ankle_stable_mdp.ankle_target_soft_limit_hinge_l2,
        weight=-0.2,
        params={"asset_cfg": _ANKLE_CFG, "dt": 0.02},
    )
    ankle_actual_soft_limit_hinge_l2 = RewTerm(
        func=ankle_stable_mdp.ankle_actual_soft_limit_hinge_l2,
        weight=-0.2,
        params={"asset_cfg": _ANKLE_CFG, "dt": 0.02},
    )


T800_FIGHT_MAT_STATIC_FRICTION_RANGE = (0.55, 1.25)
T800_FIGHT_MAT_DYNAMIC_FRICTION_RANGE = (0.45, 1.10)
T800_FIGHT_MAT_PLUS_RESTITUTION_RANGE = (0.0, 0.4)
T800_FIGHT_MAT_CONTACT_OFFSET_RANGE = (0.006, 0.014)
T800_TERRAIN_RESET_HEIGHT_OFFSET_RANGE = (0.28, 0.34)


@configclass
class T800EventsCfg:
    physics_material = Event(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": T800_FIGHT_MAT_STATIC_FRICTION_RANGE,
            "dynamic_friction_range": T800_FIGHT_MAT_DYNAMIC_FRICTION_RANGE,
            "restitution_range": T800_FIGHT_MAT_PLUS_RESTITUTION_RANGE,
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    add_base_mass = Event(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="LINK_BASE"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    base_external_force_torque = Event(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="LINK_BASE"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_base = Event(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": T800_TERRAIN_RESET_HEIGHT_OFFSET_RANGE,
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.1, 0.0),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = Event(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.8, 1.2), "velocity_range": (-1.0, 1.0)},
    )
    push_robot = Event(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
    collider_offsets = Event(
        func=mdp.randomize_rigid_body_collider_offsets,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "rest_offset_distribution_params": None,
            "contact_offset_distribution_params": T800_FIGHT_MAT_CONTACT_OFFSET_RANGE,
            "distribution": "uniform",
        },
    )
    ankle_stable_reset = Event(func=ankle_stable_mdp.reset_ankle_stable_history, mode="reset")


@configclass
class T800TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        time_out=False,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "LINK_BASE",
                    "LINK_TORSO_YAW",
                    "LINK_HEAD_.*",
                    "LINK_SHOULDER_.*",
                    "LINK_ELBOW_.*",
                    "LINK_WRIST_END_.*",
                    "LINK_HIP_.*",
                    "LINK_KNEE_PITCH_.*",
                ],
            ),
            "threshold": 1.0,
        },
    )
    terrain_out_of_bounds = DoneTerm(
        func=leo_mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
    )


@configclass
class T800CurriculumCfg:
    terrain_levels = CurrTerm(func=locomotion_mdp.terrain_levels_vel_with_stats)


@configclass
class T800BaseEnvCfg(ManagerBasedRLEnvCfg):
    scene: T800SceneCfg = T800SceneCfg(num_envs=4096, env_spacing=2.5)
    actions: T800ActionsCfg = T800ActionsCfg()
    commands: T800CommandsCfg = T800CommandsCfg()
    observations: T800ObservationsCfg = T800ObservationsCfg()
    rewards: T800RewardsCfg = T800RewardsCfg()
    terminations: T800TerminationsCfg = T800TerminationsCfg()
    events: T800EventsCfg = T800EventsCfg()
    curriculum: T800CurriculumCfg = T800CurriculumCfg()
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.0, 2.0, 0.5),
        lookat=(0.0, 0.0, 0.0),
        origin_type="asset_root",
        asset_name="robot",
    )

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        # The original tuned-v4 chain set ankle pitch before the AMP posture
        # override. That override intentionally leaves all four ankles intact.
        self.scene.robot.init_state.joint_pos.update(
            {
                "J04_ANKLE_PITCH_L": -0.14,
                "J10_ANKLE_PITCH_R": -0.14,
            }
        )
        self.scene.robot.init_state.joint_pos.update(T800_BOXING_GUARD_NOMINAL_JOINT_POS)
        self.mirror_action = None
        self.mirror_observation = None
        self.mirror_state = None
        self.ankle_stable_level = 1
        self.run_name = "T800_base"


@configclass
class T800BaseEnvCfg_PLAY(T800BaseEnvCfg):
    scene: T800SceneCfg = T800SceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.events.base_external_force_torque = None
        self.events.push_robot = None
