"""Standalone CombinedV2 to impact continuation configuration."""

import hashlib
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

from isaaclab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from t800_lab.assets.t800 import T800_URDF_PATH
from t800_lab.tasks.t800_base.agent_cfg import T800BaseTrainerCfg
from t800_lab.tasks.t800_base.env_cfg import T800BaseEnvCfg
from t800_lab.tasks.t800_base.terrain_cfg import make_terrain as make_base_terrain

from . import mdp
from .env import ImpactJointPositionActionCfg
from .impact import ImpactCfg
from .recovery import RecoveryConfig

_ANKLE_CFG = SceneEntityCfg(
    "robot",
    joint_names=["J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R"],
    preserve_order=True,
)
_FOOT_SENSOR_CFG = SceneEntityCfg(
    "contact_forces", body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"], preserve_order=True
)


def reset_head_faults_event(env, env_ids):
    # IsaacLab's event validator requires env_ids to be a mandatory argument.
    mdp.reset_head_faults(env, env_ids)


def _ground_contact_sensors(scene):
    # ContactSensor supports one-to-many filtering, not many-to-many. Use one
    # source body per sensor so self/opponent/cage contacts never count as falls.
    bodies = [
        link.attrib["name"]
        for link in ET.parse(T800_URDF_PATH).getroot().findall("link")
        if link.find("collision") is not None
    ]
    for body in bodies:
        foot = body in mdp.FOOT_BODIES
        setattr(
            scene,
            mdp.GROUND_SENSOR_PREFIX + body,
            ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + body,
                update_period=0.005,
                history_length=4,
                # Generated terrain's parent is an Xform. PhysX contact filters
                # must target the collision Mesh, otherwise filtered forces stay zero.
                filter_prim_paths_expr=["/World/ground/terrain/mesh"],
                track_pose=foot,
                track_contact_points=foot,
                max_contact_data_count_per_prim=16 if foot else 4,
            ),
        )


@configclass
class T800ImpactV2PlusEnvCfg(T800BaseEnvCfg):
    impact: ImpactCfg = ImpactCfg()
    recovery: RecoveryConfig = RecoveryConfig()

    def __post_init__(self):
        super().__post_init__()
        # Build the exact V2 source contract first; the task identity is
        # switched to Plus only after those immutable snapshots are captured.
        self.run_name = "T800_base-ImpactV2"
        self.ankle_variant = "ImpactV2"
        self.ankle_substep_stability = {
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
        self.rewards.ankle_target_rate_l2.weight = -0.04
        self.rewards.ankle_target_acceleration_l2.weight = -0.008
        self.rewards.ankle_physics_acceleration_l2 = RewardTermCfg(func=mdp.physics_acceleration_l2, weight=-0.02)
        self.rewards.ankle_support_foot_angular_velocity_l2 = RewardTermCfg(
            func=mdp.support_foot_angular_velocity_l2, weight=-0.2
        )
        self.rewards.ankle_support_joint_pose_l2 = RewardTermCfg(
            func=mdp.support_joint_pose_l2,
            weight=-0.02,
            params={"asset_cfg": _ANKLE_CFG, "sensor_cfg": _FOOT_SENSOR_CFG, "threshold": 1.0, "normalizer_rad": 0.1},
        )
        self.scene.robot.actuators["feet"].stiffness = {".*_ANKLE_PITCH.*": 40.0, ".*_ANKLE_ROLL.*": 40.0}
        self.scene.robot.actuators["feet"].damping = {".*_ANKLE_PITCH.*": 2.0, ".*_ANKLE_ROLL.*": 2.0}
        self.events.collider_offsets.params["contact_offset_distribution_params"] = (0.002, 0.002)
        self.events.collider_offsets.params["rest_offset_distribution_params"] = None
        self.ankle_finetune_reward_weights = {
            "ankle_physics_acceleration_l2": (0.0, -0.02),
            "ankle_support_foot_angular_velocity_l2": (0.0, -0.2),
        }
        # Capture the source contract only after all V2 PD/contact/substep and
        # ankle reward settings exist, but before any Impact masking/wrapping.
        from t800_lab.utils.wrappers.cusrl.impact import combined_v2_source_signature

        self.impact_v2_source_play_signature = combined_v2_source_signature(self)
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
        policy.joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_masked, params={"asset_cfg": robot, "noise_range": (-0.01, 0.01)}
        )
        policy.joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_masked, params={"asset_cfg": robot, "noise_range": (-1.5, 1.5)}
        )
        policy.actions = ObservationTermCfg(func=mdp.last_action_masked)
        critic.joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel_masked, params={"asset_cfg": robot})
        critic.joint_vel = ObservationTermCfg(func=mdp.joint_vel_masked, params={"asset_cfg": robot})
        critic.actions = ObservationTermCfg(func=mdp.last_action_masked)
        for group_name, functions, asset in (
            ("amp_policy", (mdp.amp_joint_pos_rel_masked, mdp.amp_joint_vel_masked), robot),
            (
                "amp_reference",
                (mdp.reference_amp_joint_pos_rel_masked, mdp.reference_amp_joint_vel_masked),
                SceneEntityCfg("motion_reference"),
            ),
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
            "stand_still_without_cmd",
            "track_lin_vel_xy_error_l2",
            "track_ang_vel_z_error_l2",
            "standard_arm_pose_l1",
            "joint_deviation_torso",
            "joint_deviation_hip",
            "joint_deviation_knee",
            "dof_acc_l2",
            "ankle_target_rate_l2",
            "ankle_target_acceleration_l2",
            "ankle_acceleration_l2",
            "ankle_physics_acceleration_l2",
            "ankle_support_foot_angular_velocity_l2",
            "ankle_support_joint_pose_l2",
        ):
            term = getattr(self.rewards, name)
            original_params = dict(term.params)
            # IsaacLab only resolves top-level SceneEntityCfg values.
            asset_cfg = original_params.pop("asset_cfg", None)
            sensor_cfg = original_params.pop("sensor_cfg", None)
            term.params = {"term_func": term.func, "term_params": original_params}
            if asset_cfg is not None:
                term.params["asset_cfg"] = asset_cfg
            if sensor_cfg is not None:
                term.params["sensor_cfg"] = sensor_cfg
            term.func = mdp.recovery_scaled
        self.rewards.impact_scale_snapshot = RewardTermCfg(func=mdp.capture_recovery_scale, weight=1.0)
        # Snapshot the complete original V2 contract before Plus-only reward
        # relaxation or deferred-contact termination is applied.
        from t800_lab.utils.wrappers.cusrl.impact import impact_v2_parent_configuration_signature

        self.impact_v2_parent_configuration_signature = impact_v2_parent_configuration_signature(self)
        runtime_settings = asdict(self.impact)
        for key in ("enabled", "enabled_eval", "mode", "eval_strength", "fixed_peak_n", "fixed_duration_s", "fixed_interval_s", "head_mode"):
            runtime_settings.pop(key)
        self.impact_v2_parent_runtime_signature = hashlib.sha256(
            json.dumps(runtime_settings, sort_keys=True).encode()
        ).hexdigest()
        self.run_name = "T800_base-ImpactV2Plus"
        self.ankle_variant = "ImpactV2Plus"
        self.terminations.base_contact = TerminationTermCfg(func=mdp.ground_fall_plus)
        # Normal-stage weights are unchanged. Only a measured recovery stage
        # interpolates each soft cost toward its specified minimum multiplier.
        for name, term in vars(self.rewards).items():
            if term is not None and getattr(term, "func", None) is mdp.recovery_scaled:
                term.params["recovery_factor"] = 0.10 if name in {
                    "stand_still_without_cmd", "standard_arm_pose_l1", "joint_deviation_torso",
                    "joint_deviation_hip", "joint_deviation_knee", "ankle_support_joint_pose_l2",
                } else 0.50
        for name in ("feet_force", "feet_slide", "undesired_contacts"):
            term = getattr(self.rewards, name)
            parameters = dict(term.params)
            term.params = {"term_func": term.func, "term_params": parameters, "recovery_factor": 0.50}
            for key in ("asset_cfg", "sensor_cfg"):
                if key in parameters:
                    term.params[key] = parameters.pop(key)
            term.func = mdp.recovery_scaled
        for name in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"):
            term = getattr(self.rewards, name)
            term.params = {"term_func": term.func, "term_params": dict(term.params)}
            term.func = mdp.upright_tracking
        self.rewards.recovery_upright = RewardTermCfg(func=mdp.recovery_upright_progress, weight=0.20)


@configclass
class T800ImpactV2PlusEnvCfg_PLAY(T800ImpactV2PlusEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.curriculum.terrain_levels = None
        self.impact.mode = "common"


def make_terrain():
    terrain = make_base_terrain()
    generator = terrain.terrain_generator
    flat = generator.sub_terrains["fight_mat_flat"].copy()
    horizontal = generator.sub_terrains["fight_mat_horizontal_obstacles"].copy()
    flat.proportion = 0.8
    horizontal.proportion = 0.2
    generator.sub_terrains = {
        "fight_mat_flat": flat,
        "fight_mat_horizontal_obstacles": horizontal,
    }
    generator.num_cols = 10
    return terrain


HEAD_LINK_NAMES = frozenset(("LINK_HEAD_PITCH", "LINK_HEAD_YAW"))
HEAD_JOINT_NAMES = frozenset(("J27_HEAD_PITCH", "J28_HEAD_YAW"))


def make_headless_urdf(source_urdf: str | Path, destination: str | Path | None = None) -> str:
    """Remove both head links and their joints; retain absolute mesh references.

    The result is intended for a separate evaluation scene. A passive attached
    head and a head hanging from wires are mechanically different scenarios.
    """
    source = Path(source_urdf).expanduser().resolve(strict=True)
    source_bytes = source.read_bytes()
    root = ET.fromstring(source_bytes)
    links = {element.attrib["name"] for element in root.findall("link")}
    if not HEAD_LINK_NAMES.issubset(links):
        raise ValueError(f"Source URDF does not contain the two expected head links: {source}")
    removed = set(HEAD_LINK_NAMES)
    # Also remove any attached descendants to avoid dangling joints if an
    # evaluation URDF later adds fixed visual links beneath the head.
    changed = True
    while changed:
        changed = False
        for joint in root.findall("joint"):
            parent, child = joint.find("parent"), joint.find("child")
            if parent is not None and child is not None and parent.attrib["link"] in removed:
                if child.attrib["link"] not in removed:
                    removed.add(child.attrib["link"])
                    changed = True
    for element in list(root):
        if element.tag == "link" and element.attrib.get("name") in removed:
            root.remove(element)
        elif element.tag == "joint":
            referenced = {part.attrib.get("link") for tag in ("parent", "child") for part in element.findall(tag)}
            if element.attrib.get("name") in HEAD_JOINT_NAMES or referenced.intersection(removed):
                root.remove(element)
        elif element.tag in ("transmission", "gazebo"):
            references = {part.attrib.get("name") for part in element.iter("joint")}
            if references.intersection(HEAD_JOINT_NAMES) or element.attrib.get("reference") in removed:
                root.remove(element)
    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if filename and "://" not in filename and not Path(filename).is_absolute():
            mesh.attrib["filename"] = str((source.parent / filename).resolve())
    if destination is None:
        digest = hashlib.sha256(str(source).encode() + source_bytes).hexdigest()[:20]
        destination = Path(tempfile.gettempdir()) / "t800_impact_assets" / digest / "headless.urdf"
    target = Path(destination).expanduser().resolve()
    if target == source:
        raise ValueError("Headless URDF destination must differ from the source.")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    # Concurrent ranks may request the same generated asset. Atomic replacement
    # prevents an importer from seeing a partially written document.
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".urdf", delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(payload)
            output.close()
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return str(target)


@dataclass
class T800ImpactV2PlusTrainerCfg(T800BaseTrainerCfg):
    experiment_name = "T800_base-ImpactV2Plus"
