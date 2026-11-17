from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
MDP_PATH = ROOT / "source/t800_lab/t800_lab/tasks/t800_ankle/mdp.py"
SPEC = importlib.util.spec_from_file_location("t800_ankle_variant_mdp", MDP_PATH)
assert SPEC and SPEC.loader
MDP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MDP)


class _Cfg:
    name = "robot"
    joint_ids = torch.tensor([0, 1, 2, 3])


class _SensorCfg:
    name = "contact_forces"
    body_ids = torch.tensor([0, 1])


def _env(contact: bool):
    q0 = torch.zeros((1, 4))
    q = torch.tensor([[0.1, 0.1, 0.0, 0.0]])
    force = torch.zeros((1, 3, 2, 3))
    if contact:
        force[..., 0, 2] = 10.0
    return SimpleNamespace(
        scene={
            "robot": SimpleNamespace(data=SimpleNamespace(joint_pos=q, default_joint_pos=q0)),
            "contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w_history=force)),
        }
    )


def test_pose_reward_is_gated_by_support_contact():
    assert MDP.support_joint_pose_l2(_env(False), _Cfg(), _SensorCfg()).item() == 0.0
    assert MDP.support_joint_pose_l2(_env(True), _Cfg(), _SensorCfg()).item() > 0.0


def test_pose_reward_weights_roll_twice_pitch():
    env = _env(True)
    env.scene["robot"].data.joint_pos[:] = torch.tensor([[0.1, 0.0, 0.0, 0.0]])
    pitch = MDP.support_joint_pose_l2(env, _Cfg(), _SensorCfg()).item()
    env.scene["robot"].data.joint_pos[:] = torch.tensor([[0.0, 0.1, 0.0, 0.0]])
    roll = MDP.support_joint_pose_l2(env, _Cfg(), _SensorCfg()).item()
    assert roll == 2.0 * pitch


def test_pose_reward_uses_canonical_ids_and_contact_history():
    env = _env(False)
    env.scene["robot"].data.joint_pos[:] = torch.tensor([[0.0, 0.0, 0.0, 0.1]])
    env.scene["contact_forces"].data.net_forces_w_history[:, 1, 1, 2] = 2.0
    result = MDP.support_joint_pose_l2(env, _Cfg(), _SensorCfg()).item()
    assert result > 0.0


def test_pose_reward_keeps_swing_foot_out_of_support_term():
    env = _env(False)
    env.scene["robot"].data.joint_pos[:] = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
    env.scene["contact_forces"].data.net_forces_w_history[..., 0, 2] = 10.0
    assert MDP.support_joint_pose_l2(env, _Cfg(), _SensorCfg()).item() == 0.0


def test_combined_task_is_a_distinct_from_scratch_task():
    import importlib.util

    path = ROOT / "source/t800_lab/t800_lab/task_names.py"
    spec = importlib.util.spec_from_file_location("t800_task_names", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ANKLE_TASKS = module.ANKLE_TASKS
    ALL_TASKS = module.ALL_TASKS
    COMBINED_ANKLE_TASK = module.COMBINED_ANKLE_TASK
    TRAIN_TASKS = module.TRAIN_TASKS

    assert COMBINED_ANKLE_TASK == "T800_base-AnkleCombined"
    assert COMBINED_ANKLE_TASK not in ANKLE_TASKS
    assert COMBINED_ANKLE_TASK in TRAIN_TASKS
    assert {COMBINED_ANKLE_TASK, COMBINED_ANKLE_TASK + "-Play"} <= set(ALL_TASKS)

