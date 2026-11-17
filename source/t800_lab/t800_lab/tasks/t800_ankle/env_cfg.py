"""T800_base continuation variants for isolating ankle high-frequency motion."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from t800_lab.tasks.t800_base import ankle_stable_mdp
from t800_lab.tasks.t800_base.constants import T800_FOOT_BODY_NAMES
from t800_lab.tasks.t800_base.env_cfg import T800BaseEnvCfg

from . import mdp, stability


_ANKLE_CFG = SceneEntityCfg("robot", joint_names=list(ankle_stable_mdp.ANKLE_JOINT_NAMES), preserve_order=True)
_FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=list(T800_FOOT_BODY_NAMES), preserve_order=True)


def _configure_play(cfg):
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 2.5
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None


@configclass
class T800AnkleSmoothEnvCfg(T800BaseEnvCfg):
    ankle_variant = "AnkleSmooth"
    run_name = "T800_base-AnkleSmooth"
    ankle_finetune_reward_weights = {
        "ankle_target_rate_l2": (-0.01, -0.04),
        "ankle_target_acceleration_l2": (-0.002, -0.008),
    }

    def __post_init__(self):
        super().__post_init__()
        self.rewards.ankle_target_rate_l2.weight = -0.04
        self.rewards.ankle_target_acceleration_l2.weight = -0.008
        self.run_name = "T800_base-AnkleSmooth"


@configclass
class T800AnkleSmoothEnvCfg_PLAY(T800AnkleSmoothEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)


@configclass
class T800AnkleDampingEnvCfg(T800BaseEnvCfg):
    ankle_variant = "AnkleDamping"
    run_name = "T800_base-AnkleDamping"
    ankle_finetune_reward_weights = {}

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators["feet"].damping = {
            ".*_ANKLE_PITCH.*": 0.6,
            ".*_ANKLE_ROLL.*": 0.6,
        }
        self.run_name = "T800_base-AnkleDamping"


@configclass
class T800AnkleDampingEnvCfg_PLAY(T800AnkleDampingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)


@configclass
class T800AnkleTargetLimitEnvCfg(T800BaseEnvCfg):
    ankle_variant = "AnkleTargetLimit"
    run_name = "T800_base-AnkleTargetLimit"
    ankle_finetune_reward_weights = {
        "ankle_target_soft_limit_hinge_l2": (-0.2, -0.8),
    }

    def __post_init__(self):
        super().__post_init__()
        self.rewards.ankle_target_soft_limit_hinge_l2.weight = -0.8
        self.run_name = "T800_base-AnkleTargetLimit"


@configclass
class T800AnkleTargetLimitEnvCfg_PLAY(T800AnkleTargetLimitEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)


@configclass
class T800AnklePoseEnvCfg(T800BaseEnvCfg):
    ankle_variant = "AnklePose"
    run_name = "T800_base-AnklePose"
    ankle_finetune_reward_weights = {"ankle_support_joint_pose_l2": (0.0, -0.02)}

    def __post_init__(self):
        super().__post_init__()
        self.rewards.ankle_support_joint_pose_l2 = RewTerm(
            func=mdp.support_joint_pose_l2,
            weight=-0.02,
            params={
                "asset_cfg": _ANKLE_CFG,
                "sensor_cfg": _FOOT_SENSOR_CFG,
                "threshold": 1.0,
                "normalizer_rad": 0.1,
            },
        )
        self.run_name = "T800_base-AnklePose"


@configclass
class T800AnklePoseEnvCfg_PLAY(T800AnklePoseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)


@configclass
class T800AnkleCombinedEnvCfg(T800BaseEnvCfg):
    """From-scratch combination of target smoothing, ankle damping and pose support."""

    ankle_variant = "AnkleCombined"
    run_name = "T800_base-AnkleCombined"

    def __post_init__(self):
        super().__post_init__()
        self.rewards.ankle_target_rate_l2.weight = -0.04
        self.rewards.ankle_target_acceleration_l2.weight = -0.008
        self.scene.robot.actuators["feet"].damping = {
            ".*_ANKLE_PITCH.*": 1.2,
            ".*_ANKLE_ROLL.*": 1.2,
        }
        self.rewards.ankle_support_joint_pose_l2 = RewTerm(
            func=mdp.support_joint_pose_l2,
            weight=-0.02,
            params={
                "asset_cfg": _ANKLE_CFG,
                "sensor_cfg": _FOOT_SENSOR_CFG,
                "threshold": 1.0,
                "normalizer_rad": 0.1,
            },
        )
        self.run_name = "T800_base-AnkleCombined"


@configclass
class T800AnkleCombinedEnvCfg_PLAY(T800AnkleCombinedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)


@configclass
class T800AnkleCombinedV2EnvCfg(T800AnkleCombinedEnvCfg):
    """Continue Combined with measured motion costs at every physics step."""

    ankle_variant = "AnkleCombinedV2"
    run_name = "T800_base-AnkleCombinedV2"
    ankle_substep_stability = {
        "contact_force_enter_n": 50.0,
        "contact_force_exit_n": 10.0,
        "contact_gap_tolerance_s": 0.02,
        "support_settle_time_s": 0.03,
        "acceleration_deadband_rad_s2": 100.0,
        "acceleration_scale_rad_s2": 100.0,
        "foot_roll_deadband_rad_s": 0.5,
        "foot_pitch_deadband_rad_s": 1.0,
        "foot_roll_scale_rad_s": 1.0,
        "foot_pitch_scale_rad_s": 2.0,
        "normalized_error_cap": 20.0,
    }
    ankle_finetune_reward_weights = {
        "ankle_physics_acceleration_l2": (0.0, -0.02),
        "ankle_support_foot_angular_velocity_l2": (0.0, -0.2),
    }

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators["feet"].damping = {".*_ANKLE_PITCH.*": 2.0, ".*_ANKLE_ROLL.*": 2.0}
        self.events.collider_offsets.params["contact_offset_distribution_params"] = (0.002, 0.002)
        # Retain the asset's zero rest offset and the original RNG consumption.
        self.events.collider_offsets.params["rest_offset_distribution_params"] = None
        self.rewards.ankle_physics_acceleration_l2 = RewTerm(func=stability.physics_acceleration_l2, weight=-0.02)
        self.rewards.ankle_support_foot_angular_velocity_l2 = RewTerm(
            func=stability.support_foot_angular_velocity_l2, weight=-0.2,
        )
        self.run_name = "T800_base-AnkleCombinedV2"


@configclass
class T800AnkleCombinedV2EnvCfg_PLAY(T800AnkleCombinedV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)

