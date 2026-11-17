"""Head-independent observations and physical head drive faults.

This module deliberately has no Isaac Sim imports so its numerical contract can
also be used by exporters and checked on a CPU.  The logical order is the stage
one policy's DFS order; a headless articulation still exposes all 25 slots.
"""

from __future__ import annotations

import math
from typing import Any

import torch


LOGICAL_JOINT_NAMES = (
    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L", "J03_KNEE_PITCH_L",
    "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L", "J06_HIP_PITCH_R", "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R", "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
    "J12_TORSO_YAW", "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L", "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L",
    "J20_SHOULDER_PITCH_R", "J21_SHOULDER_ROLL_R", "J22_SHOULDER_YAW_R",
    "J23_ELBOW_PITCH_R", "J24_ELBOW_YAW_R", "J27_HEAD_PITCH", "J28_HEAD_YAW",
)
HEAD_JOINT_NAMES = LOGICAL_JOINT_NAMES[-2:]
HEAD_LOGICAL_INDICES = (23, 24)
_ANKLE_NAMES = {"J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R"}
AMP_JOINT_NAMES = tuple(name for name in LOGICAL_JOINT_NAMES if name not in _ANKLE_NAMES)
# These are the stage-one init_state defaults, not the separately edited NPZ
# posture reward targets. Full and headless assets must use the same constants.
HEAD_REFERENCE_POSITIONS = (0.052337, 0.621251)


class HeadObservationMask(torch.nn.Module):
    """Mask the 84 actor features before their normalizer and recurrent model."""

    def __init__(self, head_position: tuple[float, float] = HEAD_REFERENCE_POSITIONS):
        super().__init__()
        self.head_position: torch.Tensor
        self.register_buffer("head_position", torch.tensor(head_position, dtype=torch.float32))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # [omega(3), gravity(3), command(3), q(25), dq(25), action(25)].
        head = self.head_position.to(observation).expand_as(observation[..., 32:34])
        zero = torch.zeros_like(observation[..., 57:59])
        return torch.cat(
            (observation[..., :32], head, observation[..., 34:57], zero,
             observation[..., 59:82], zero), dim=-1,
        )


class HeadActionMask(torch.nn.Module):
    """Return normalized logical actions; default pose offset fixes the head."""

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return torch.cat((action[..., :23], torch.zeros_like(action[..., 23:25])), dim=-1)


def _asset(env: Any, asset_cfg: Any = None, default_name: str = "robot") -> Any:
    return env.scene[asset_cfg.name if asset_cfg is not None else default_name]


def _indices(asset: Any, names: tuple[str, ...], device: torch.device) -> torch.Tensor:
    """Resolve body columns once, independently of simulator joint ordering."""
    cache = getattr(asset, "_impact_joint_indices", None)
    if cache is None:
        cache = asset._impact_joint_indices = {}
    key = (names, str(device))
    if key not in cache:
        actual = list(asset.joint_names)
        missing = set(names).difference(actual)
        if missing:
            raise ValueError(f"Impact observation is missing body joints: {sorted(missing)}")
        cache[key] = torch.tensor([actual.index(name) for name in names], device=device, dtype=torch.long)
    return cache[key]


def head_reference_positions(env: Any, asset_cfg: Any = None) -> torch.Tensor:
    """Return finite per-env fixed head targets, also for the 23-joint model."""
    asset = _asset(env, asset_cfg)
    default = asset.data.default_joint_pos
    fallback = default.new_tensor(HEAD_REFERENCE_POSITIONS).expand(default.shape[0], -1)
    values = []
    for i, name in enumerate(HEAD_JOINT_NAMES):
        if name in asset.joint_names:
            value = default[:, list(asset.joint_names).index(name)]
            values.append(torch.where(torch.isfinite(value), value, fallback[:, i]))
        else:
            values.append(fallback[:, i])
    return torch.stack(values, dim=-1)


def _corrupt(env: Any, values: torch.Tensor, noise_range: tuple[float, float] | None) -> torch.Tensor:
    if noise_range is None:
        return values
    observations = getattr(getattr(env, "cfg", None), "observations", None)
    policy = getattr(observations, "policy", None)
    if policy is not None and not policy.enable_corruption:
        return values
    return values + torch.empty_like(values).uniform_(*noise_range)


def _body_values(asset: Any, values: torch.Tensor, names: tuple[str, ...]) -> torch.Tensor:
    # Head columns are never read, including NaN/Inf telemetry and reference data.
    return values.index_select(-1, _indices(asset, names[:-2], values.device))


def _joint_values(
    env: Any, asset_cfg: Any, field: str, *, relative: bool = False,
    amp: bool = False, noise_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    names = AMP_JOINT_NAMES if amp else LOGICAL_JOINT_NAMES
    values = _body_values(asset, getattr(asset.data, field), names)
    if relative:
        default_field = "default_joint_pos" if field == "joint_pos" else "default_joint_vel"
        values = values - _body_values(asset, getattr(asset.data, default_field), names)
    values = _corrupt(env, values, noise_range)
    head = head_reference_positions(env, asset_cfg) if field == "joint_pos" and not relative else values.new_zeros(values.shape[0], 2)
    return torch.cat((values, head), dim=-1)


def joint_pos_masked(env: Any, asset_cfg: Any = None, noise_range: tuple[float, float] | None = None) -> torch.Tensor:
    return _joint_values(env, asset_cfg, "joint_pos", noise_range=noise_range)


def joint_pos_rel_masked(env: Any, asset_cfg: Any = None) -> torch.Tensor:
    return _joint_values(env, asset_cfg, "joint_pos", relative=True)


def joint_vel_masked(env: Any, asset_cfg: Any = None, noise_range: tuple[float, float] | None = None) -> torch.Tensor:
    return _joint_values(env, asset_cfg, "joint_vel", noise_range=noise_range)


def last_action_masked(env: Any, action_name: str | None = None) -> torch.Tensor:
    actions = env.action_manager.action if action_name is None else env.action_manager.get_term(action_name).raw_actions
    if actions.shape[-1] != len(LOGICAL_JOINT_NAMES):
        raise ValueError(f"Impact requires 25 logical sampled actions, got {actions.shape[-1]}.")
    return torch.cat((actions[..., :23], torch.zeros_like(actions[..., 23:])), dim=-1)


def amp_joint_pos_rel_masked(env: Any, asset_cfg: Any = None) -> torch.Tensor:
    return _joint_values(env, asset_cfg, "joint_pos", relative=True, amp=True)


def amp_joint_vel_masked(env: Any, asset_cfg: Any = None) -> torch.Tensor:
    return _joint_values(env, asset_cfg, "joint_vel", relative=True, amp=True)


def _reference_values(env: Any, asset_cfg: Any, robot_cfg: Any, field: str, mask: bool = True) -> torch.Tensor:
    reference = _asset(env, asset_cfg, "motion_reference")
    robot = _asset(env, robot_cfg)
    frame = reference.reference_frame
    values = _body_values(reference, getattr(frame, field)[:, 0], AMP_JOINT_NAMES)
    defaults = _body_values(robot, getattr(robot.data, "default_" + field), AMP_JOINT_NAMES)
    feature_mask = _body_values(reference, getattr(frame, field + "_mask")[:, 0], AMP_JOINT_NAMES) if mask else 1.0
    if field == "joint_pos":
        values = (values - defaults) * feature_mask
    else:
        # Retain stage-one semantics (reference masking precedes default subtraction).
        values = values * feature_mask - defaults
    return torch.cat((values, values.new_zeros(values.shape[0], 2)), dim=-1)


def reference_amp_joint_pos_rel_masked(env: Any, asset_cfg: Any = None, robot_cfg: Any = None) -> torch.Tensor:
    return _reference_values(env, asset_cfg, robot_cfg, "joint_pos")


def reference_amp_joint_vel_masked(env: Any, asset_cfg: Any = None, robot_cfg: Any = None, mask: bool = True) -> torch.Tensor:
    return _reference_values(env, asset_cfg, robot_cfg, "joint_vel", mask)


class _HeadFaultState:
    def __init__(self, env: Any, robot: Any):
        self.robot = robot
        default = robot.data.default_joint_pos
        self.ids = [list(robot.joint_names).index(name) for name in HEAD_JOINT_NAMES if name in robot.joint_names]
        if len(self.ids) not in (0, 2):
            raise ValueError("Impact supports the full head or headless model; a partially removed head needs a separate model.")
        self.clock = 0.0
        self.next_due = math.inf
        self.kind = torch.zeros(default.shape[0], device=default.device, dtype=torch.long)
        self.active = torch.zeros_like(self.kind, dtype=torch.bool)
        self.due_at = torch.full_like(self.kind, math.inf, dtype=default.dtype)
        self.scales = default.new_ones(default.shape[0], 3)
        self.baseline = {}
        if self.ids:
            for name in ("stiffness", "damping", "effort_limits"):
                field = "joint_" + name
                healthy = getattr(robot.data, "default_" + field, None)
                if healthy is None:
                    healthy = getattr(robot.data, field)
                self.baseline[name] = healthy[:, self.ids].clone()

    def write(self, env_ids: torch.Tensor, scales: torch.Tensor) -> None:
        for column, name in enumerate(("stiffness", "damping", "effort_limits")):
            values = self.baseline[name][env_ids] * scales[:, column:column + 1]
            writer = "write_joint_effort_limit_to_sim" if name == "effort_limits" else "write_joint_" + name + "_to_sim"
            getattr(self.robot, writer)(values, joint_ids=self.ids, env_ids=env_ids)
            # IsaacLab's simulation writers explicitly do not update actuator models.
            # Synchronize their buffers so logged applied torque matches the drive.
            for actuator in getattr(self.robot, "actuators", {}).values():
                joint_names = list(actuator.joint_names)
                for head_column, head_name in enumerate(HEAD_JOINT_NAMES):
                    if head_name not in joint_names:
                        continue
                    local = joint_names.index(head_name)
                    fields = ("effort_limit", "effort_limit_sim") if name == "effort_limits" else (name,)
                    for field in fields:
                        buffer = getattr(actuator, field, None)
                        if isinstance(buffer, torch.Tensor):
                            buffer[env_ids, local] = values[:, head_column]

    def apply_due(self) -> None:
        if not self.ids or self.clock < self.next_due:
            return
        env_ids = torch.nonzero(~self.active & (self.due_at <= self.clock), as_tuple=False).flatten()
        if env_ids.numel():
            self.write(env_ids, self.scales[env_ids])
            self.active[env_ids] = True
        self.next_due = self.due_at.masked_fill(self.active, math.inf).min().item()


def reset_head_faults(env: Any, env_ids: Any = None, asset_cfg: Any = None) -> None:
    """Restore healthy drives, then independently schedule weak/passive failures.

    Feedback is always masked, independently of output failure. Passive means a
    retained head with zero drive stiffness, damping and torque; it is not a
    detached or wire-suspended head. The headless variant is a separate asset.
    Training keeps healthy drives during the first 500 completed updates by
    default; explicit common/stress evaluations bypass that adaptation period.
    """
    robot = _asset(env, asset_cfg)
    state = getattr(env, "_impact_head_faults", None)
    if state is None:
        state = env._impact_head_faults = _HeadFaultState(env, robot)
    if not state.ids:
        return
    ids = torch.arange(state.kind.numel(), device=state.kind.device) if env_ids is None else torch.as_tensor(env_ids, device=state.kind.device, dtype=torch.long)
    if not ids.numel():
        return
    cfg = env.cfg.impact
    weak = float(getattr(cfg, "head_weak_probability", 0.10))
    passive = float(getattr(cfg, "head_passive_probability", 0.15))
    delay = tuple(getattr(cfg, "head_fault_delay_s", (2.0, 8.0)))
    warmup = int(getattr(cfg, "head_fault_warmup_updates", 500))
    if weak < 0.0 or passive < 0.0 or weak + passive > 1.0 or delay[0] < 0.0 or delay[1] < delay[0] or warmup < 0:
        raise ValueError("Invalid head fault probabilities or delay range.")
    runtime = getattr(env, "impact_runtime", None)
    completed_updates = getattr(runtime, "completed_updates", 0)
    if getattr(cfg, "mode", "train") == "train" and completed_updates < warmup:
        # The first stage learned to control the head. Let the body first adapt
        # to the fixed output and masked feedback before adding drive failures.
        weak = passive = 0.0
    state.write(ids, state.scales.new_ones(ids.numel(), 3))
    sample = torch.rand(ids.numel(), device=ids.device)
    kind = torch.where(sample < passive, 2, torch.where(sample < passive + weak, 1, 0))
    scales = state.scales.new_ones(ids.numel(), 3)
    weak_mask = kind == 1
    scales[:, 0] = torch.where(weak_mask, torch.empty_like(sample).uniform_(0.1, 0.4), scales[:, 0])
    scales[:, 1] = torch.where(weak_mask, torch.empty_like(sample).uniform_(0.2, 0.7), scales[:, 1])
    scales[:, 2] = torch.where(weak_mask, torch.empty_like(sample).uniform_(0.1, 0.4), scales[:, 2])
    scales[kind == 2] = 0.0
    state.kind[ids] = kind
    state.active[ids] = False
    state.scales[ids] = scales
    sampled_delay = torch.empty_like(sample).uniform_(*delay)
    state.due_at[ids] = torch.where(kind != 0, state.clock + sampled_delay, math.inf)
    state.next_due = state.due_at.masked_fill(state.active, math.inf).min().item()
    state.apply_due()


def update_head_faults(env: Any, dt: float) -> None:
    """Advance delayed physical drive failures once per physics step."""
    state = getattr(env, "_impact_head_faults", None)
    if state is not None:
        state.clock += dt
        state.apply_due()
