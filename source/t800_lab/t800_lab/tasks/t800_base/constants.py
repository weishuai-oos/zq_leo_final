"""Joint groups, AMP selection and posture targets for the single T800 task."""

from functools import lru_cache
from pathlib import Path

import numpy as np

from t800_lab.assets.t800 import T800_ACTION_SCALE, T800_DFS_JOINT_NAMES

T800_FOOT_BODY_NAMES = ["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"]
T800_LEG_ACC_JOINT_NAMES = [
    "J00_HIP_PITCH_L",
    "J01_HIP_ROLL_L",
    "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L",
    "J06_HIP_PITCH_R",
    "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R",
]
T800_HIP_DEVIATION_JOINT_NAMES = [
    "J01_HIP_ROLL_L",
    "J02_HIP_YAW_L",
    "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R",
]
T800_ARM_JOINT_NAMES = [
    "J13_SHOULDER_PITCH_L",
    "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L",
    "J17_ELBOW_YAW_L",
    "J20_SHOULDER_PITCH_R",
    "J21_SHOULDER_ROLL_R",
    "J22_SHOULDER_YAW_R",
    "J23_ELBOW_PITCH_R",
    "J24_ELBOW_YAW_R",
]
T800_HEAD_JOINT_NAMES = ["J27_HEAD_PITCH", "J28_HEAD_YAW"]
T800_ANKLE_JOINT_NAMES = [
    "J04_ANKLE_PITCH_L",
    "J05_ANKLE_ROLL_L",
    "J10_ANKLE_PITCH_R",
    "J11_ANKLE_ROLL_R",
]
T800_KNEE_JOINT_NAMES = ["J03_KNEE_PITCH_L", "J09_KNEE_PITCH_R"]

AMP_HISTORY_LENGTH = 2

DEFAULT_MOTION_DATA_PATH = str(Path(__file__).resolve().parent / "amp_dataset")

DEFAULT_MOTION_SELECTION_PATH = str(Path(DEFAULT_MOTION_DATA_PATH) / "boxing_guard_motion_selection.yaml")

DEFAULT_STANDARD_POSE_PATH = Path(DEFAULT_MOTION_DATA_PATH) / "标准姿势_t800_controller_retargeted.npz"

T800_AMP_LINKS_OF_INTEREST = [
    "LINK_BASE",
    "LINK_TORSO_YAW",
    "LINK_SHOULDER_ROLL_L",
    "LINK_SHOULDER_ROLL_R",
    "LINK_ELBOW_YAW_L",
    "LINK_ELBOW_YAW_R",
    "LINK_WRIST_END_L",
    "LINK_WRIST_END_R",
    "LINK_HIP_ROLL_L",
    "LINK_HIP_ROLL_R",
    "LINK_KNEE_PITCH_L",
    "LINK_KNEE_PITCH_R",
    "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_R",
]

T800_AMP_MASKED_JOINT_NAMES = set(T800_ANKLE_JOINT_NAMES)

T800_AMP_JOINT_NAMES = [
    joint_name for joint_name in T800_DFS_JOINT_NAMES if joint_name not in T800_AMP_MASKED_JOINT_NAMES
]

T800_STANDARD_POSE_JOINT_NAMES = T800_ARM_JOINT_NAMES + T800_HEAD_JOINT_NAMES

T800_STAND_STILL_JOINT_NAMES = [
    joint_name for joint_name in T800_DFS_JOINT_NAMES if joint_name not in set(T800_STANDARD_POSE_JOINT_NAMES)
]

T800_BOXING_GUARD_NOMINAL_JOINT_POS = {
    "J00_HIP_PITCH_L": -0.548942,
    "J01_HIP_ROLL_L": 0.220542,
    "J02_HIP_YAW_L": -0.119339,
    "J03_KNEE_PITCH_L": 0.279160,
    "J06_HIP_PITCH_R": -0.235541,
    "J07_HIP_ROLL_R": -0.223292,
    "J08_HIP_YAW_R": -0.102872,
    "J09_KNEE_PITCH_R": 0.257664,
    "J12_TORSO_YAW": 0.006373,
    "J13_SHOULDER_PITCH_L": -1.006467,
    "J14_SHOULDER_ROLL_L": 0.540767,
    "J15_SHOULDER_YAW_L": 0.313282,
    "J16_ELBOW_PITCH_L": -1.331819,
    "J17_ELBOW_YAW_L": 0.124102,
    "J20_SHOULDER_PITCH_R": -1.138289,
    "J21_SHOULDER_ROLL_R": -0.136425,
    "J22_SHOULDER_YAW_R": 0.197952,
    "J23_ELBOW_PITCH_R": -1.815755,
    "J24_ELBOW_YAW_R": -0.623839,
    "J27_HEAD_PITCH": 0.052337,
    "J28_HEAD_YAW": 0.621251,
}

T800_AMP_ACTION_SCALE = dict(T800_ACTION_SCALE)
T800_AMP_ACTION_SCALE.update(
    {
        "J13_SHOULDER_PITCH_L": 0.4,
        "J14_SHOULDER_ROLL_L": 0.3,
        "J15_SHOULDER_YAW_L": 0.3,
        "J16_ELBOW_PITCH_L": 0.4,
        "J17_ELBOW_YAW_L": 0.3,
        "J20_SHOULDER_PITCH_R": 0.4,
        "J21_SHOULDER_ROLL_R": 0.3,
        "J22_SHOULDER_YAW_R": 0.3,
        "J23_ELBOW_PITCH_R": 0.4,
        "J24_ELBOW_YAW_R": 0.4,
        "J27_HEAD_PITCH": 0.2,
        "J28_HEAD_YAW": 0.3,
    }
)


@lru_cache(maxsize=1)
def _load_t800_standard_upper_pose() -> dict[str, float]:
    """Load the edited one-frame upper-body target for the arm/head posture reward."""
    if not DEFAULT_STANDARD_POSE_PATH.is_file():
        raise FileNotFoundError(f"The T800 standard-pose experiments require '{DEFAULT_STANDARD_POSE_PATH}'.")

    with np.load(DEFAULT_STANDARD_POSE_PATH, allow_pickle=False) as data:
        required_keys = {"joint_names", "joint_pos"}
        missing_keys = required_keys.difference(data.files)
        if missing_keys:
            raise ValueError(
                f"Standard-pose file '{DEFAULT_STANDARD_POSE_PATH}' is missing keys: {sorted(missing_keys)}"
            )
        joint_names = [str(name) for name in np.asarray(data["joint_names"]).reshape(-1)]
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)

    if len(joint_names) != len(set(joint_names)):
        raise ValueError(f"Standard-pose file '{DEFAULT_STANDARD_POSE_PATH}' contains duplicate joint names.")
    if joint_pos.ndim != 2 or joint_pos.shape != (1, len(joint_names)):
        raise ValueError(
            f"Standard-pose file '{DEFAULT_STANDARD_POSE_PATH}' must contain exactly one joint_pos frame "
            f"with shape (1, {len(joint_names)}), got {joint_pos.shape}."
        )

    joint_index = {name: index for index, name in enumerate(joint_names)}
    missing_joints = [name for name in T800_STANDARD_POSE_JOINT_NAMES if name not in joint_index]
    if missing_joints:
        raise ValueError(
            f"Standard-pose file '{DEFAULT_STANDARD_POSE_PATH}' is missing upper-body joints: {missing_joints}"
        )

    upper_pose = {name: float(joint_pos[0, joint_index[name]]) for name in T800_STANDARD_POSE_JOINT_NAMES}
    invalid_joints = [name for name, value in upper_pose.items() if not np.isfinite(value)]
    if invalid_joints:
        raise ValueError(
            f"Standard-pose file '{DEFAULT_STANDARD_POSE_PATH}' contains non-finite values for: {invalid_joints}"
        )
    return upper_pose
