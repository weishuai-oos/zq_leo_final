"""Head isolation, ground-only fall detection and physics-step ankle costs.

This module deliberately has no Isaac Sim imports so its numerical contract can
also be used by exporters and checked on a CPU.  The logical order is the stage
one policy's DFS order; a headless articulation still exposes all 25 slots.
"""

import math
from typing import Any

import torch


LOGICAL_JOINT_NAMES = (
    "J00_HIP_PITCH_L",
    "J01_HIP_ROLL_L",
    "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L",
    "J04_ANKLE_PITCH_L",
    "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R",
    "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R",
    "J10_ANKLE_PITCH_R",
    "J11_ANKLE_ROLL_R",
    "J12_TORSO_YAW",
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
    "J27_HEAD_PITCH",
    "J28_HEAD_YAW",
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
            (observation[..., :32], head, observation[..., 34:57], zero, observation[..., 59:82], zero),
            dim=-1,
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
    env: Any,
    asset_cfg: Any,
    field: str,
    *,
    relative: bool = False,
    amp: bool = False,
    noise_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    names = AMP_JOINT_NAMES if amp else LOGICAL_JOINT_NAMES
    values = _body_values(asset, getattr(asset.data, field), names)
    if relative:
        default_field = "default_joint_pos" if field == "joint_pos" else "default_joint_vel"
        values = values - _body_values(asset, getattr(asset.data, default_field), names)
    values = _corrupt(env, values, noise_range)
    head = (
        head_reference_positions(env, asset_cfg)
        if field == "joint_pos" and not relative
        else values.new_zeros(values.shape[0], 2)
    )
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


def reference_amp_joint_vel_masked(
    env: Any, asset_cfg: Any = None, robot_cfg: Any = None, mask: bool = True
) -> torch.Tensor:
    return _reference_values(env, asset_cfg, robot_cfg, "joint_vel", mask)


class _HeadFaultState:
    def __init__(self, env: Any, robot: Any):
        self.robot = robot
        default = robot.data.default_joint_pos
        self.ids = [list(robot.joint_names).index(name) for name in HEAD_JOINT_NAMES if name in robot.joint_names]
        if len(self.ids) not in (0, 2):
            raise ValueError(
                "Impact supports the full head or headless model; a partially removed head needs a separate model."
            )
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
            values = self.baseline[name][env_ids] * scales[:, column : column + 1]
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
    ids = (
        torch.arange(state.kind.numel(), device=state.kind.device)
        if env_ids is None
        else torch.as_tensor(env_ids, device=state.kind.device, dtype=torch.long)
    )
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


GROUND_SENSOR_PREFIX = "impact_ground_"
FOOT_BODIES = ("LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R")


def _ground_sensors(env):
    for name, sensor in env.scene.sensors.items():
        if name.startswith(GROUND_SENSOR_PREFIX):
            yield name[len(GROUND_SENSOR_PREFIX) :], sensor


def _ground_force(sensor):
    history = sensor.data.force_matrix_w_history
    if history is None:
        raise RuntimeError("Impact fall detection requires a ground-filtered contact sensor per body.")
    # [environment, history, single source body, ground filter, xyz].
    return torch.linalg.vector_norm(history, dim=-1).flatten(1).amax(dim=1)


def nonsole_ground_contacts(env, threshold: float = 1.0):
    contacts = torch.zeros(env.num_envs, device=env.device)
    for body, sensor in _ground_sensors(env):
        if body not in FOOT_BODIES:
            contacts += (_ground_force(sensor) > threshold).float()
    return contacts


def _quat_rotate_inverse(q, vector):
    xyz = q[..., 1:]
    cross = torch.cross(xyz, vector, dim=-1)
    return vector - 2 * q[..., :1] * cross + 2 * torch.cross(xyz, cross, dim=-1)


def foot_side_ground_contact(env, threshold: float = 1.0):
    """Catch inverted/side-loaded feet in addition to non-foot ground contact.

    IsaacLab reports the average filtered contact point, so this is an
    approximation for a box-shaped foot; it is not a per-triangle referee.
    """
    invalid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for body, sensor in _ground_sensors(env):
        if body not in FOOT_BODIES:
            continue
        # Pair current pose/point with current force, never the force history.
        current_force = sensor.data.force_matrix_w
        if current_force is None:
            raise RuntimeError("Foot contact requires a filtered force matrix.")
        contact = torch.linalg.vector_norm(current_force, dim=-1).flatten(1).amax(dim=1) > threshold
        quaternion = sensor.data.quat_w[:, 0]
        up = quaternion.new_zeros(env.num_envs, 3)
        up[:, 2] = 1.0
        local_up = _quat_rotate_inverse(quaternion, up)
        # A tilted lower corner is still a legal sole contact. A near-horizontal
        # sole normal or contact above the collider's midplane indicates a side.
        invalid |= contact & (local_up[:, 2] < math.cos(math.radians(70.0)))
        point = sensor.data.contact_pos_w
        if point is not None:
            delta = point[:, 0, 0] - sensor.data.pos_w[:, 0]
            local_point = _quat_rotate_inverse(quaternion, delta)
            invalid |= contact & torch.isfinite(local_point).all(dim=-1) & (local_point[:, 2] > -0.049)
    return invalid


def ground_fall(env, threshold: float = 1.0):
    invalid = (nonsole_ground_contacts(env, threshold) > 0) | foot_side_ground_contact(env, threshold)
    prior_substeps = getattr(env, "_impact_foot_ground_flags", None)
    return invalid if prior_substeps is None else invalid | prior_substeps


def ground_fall_plus(env, threshold: float = 1.0):
    """Terminate persistent body/low-height failures, while allowing recovery.

    A single foot-side, hand, knee, or elbow contact is an event for the
    recovery controller rather than an immediate episode reset.
    """
    runtime = getattr(env, "impact_runtime", None)
    if runtime is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    # Contact sampling is performed after every physics substep by the env;
    # termination only reads the latched state here.
    return runtime.recovery.knockdown | runtime.recovery.timeout


def current_ground_contacts(env, threshold: float = 1.0):
    """Return current support trouble/body contact without history smearing.

    Legal sole contact is excluded. Four-sample sensor history is useful for
    reward diagnostics, but would lengthen a 5ms contact into false persistence.
    """
    trouble = foot_side_ground_contact(env, threshold)
    body_contact = torch.zeros_like(trouble)
    body_links = {"LINK_TORSO_YAW", "LINK_BASE", "LINK_HEAD_YAW", "LINK_HEAD_PITCH"}
    for body, sensor in _ground_sensors(env):
        if body in FOOT_BODIES:
            continue
        current = sensor.data.force_matrix_w
        if current is None:
            raise RuntimeError("Plus recovery requires current ground-filtered contact forces.")
        contact = torch.linalg.vector_norm(current, dim=-1).flatten(1).amax(dim=1) > threshold
        trouble |= contact
        if body in body_links:
            body_contact |= contact
    return trouble, body_contact


def recovery_scale(env):
    runtime = getattr(env, "impact_runtime", None)
    if runtime is None:
        return torch.ones(env.num_envs, device=env.device)
    scale = runtime.recovery_scale()
    # Preserve the pre-reset value for the corresponding transition's AMP term.
    runtime.last_step_style_scale = scale.clone()
    return scale


def recovery_scaled(env, term_func, term_params: dict, asset_cfg=None, sensor_cfg=None, recovery_factor: float = 1.0):
    """Relax soft tracking/posture costs only during measurable disturbance."""
    if asset_cfg is not None:
        term_params = {**term_params, "asset_cfg": asset_cfg}
    if sensor_cfg is not None:
        term_params = {**term_params, "sensor_cfg": sensor_cfg}
    runtime = getattr(env, "impact_runtime", None)
    blend = torch.ones(env.num_envs, device=env.device) if runtime is None else runtime.recovery.guard_blend
    multiplier = float(recovery_factor) + (1.0 - float(recovery_factor)) * blend
    return term_func(env, **term_params) * multiplier


def body_action_rate(env):
    # Fixed/failed head outputs must not influence the policy reward either.
    action = env.action_manager.action[:, :23]
    previous = env.action_manager.prev_action[:, :23]
    runtime = getattr(env, "impact_runtime", None)
    blend = 1.0 if runtime is None else runtime.recovery.guard_blend
    return (action - previous).square().sum(dim=-1) * (0.5 + 0.5 * blend)


def capture_recovery_scale(env):
    """A zero-weight-in-effect diagnostic term also snapshots clean transitions."""
    recovery_scale(env)
    return torch.zeros(env.num_envs, device=env.device)


def recovery_upright_progress(env):
    """Signed physical progress; integrated reward is bounded per trajectory."""
    runtime = getattr(env, "impact_runtime", None)
    if runtime is None:
        return torch.zeros(env.num_envs, device=env.device)
    return runtime._control_upright_progress / env.step_dt


def upright_tracking(env, term_func, term_params: dict):
    """Preserve ordinary tracking rewards; remove the low/lying-state loophole."""
    data = env.scene["robot"].data
    gravity = data.projected_gravity_b
    tilt = torch.atan2(torch.linalg.vector_norm(gravity[:, :2], dim=-1), -gravity[:, 2])
    height = data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    height_gate = ((height - 0.45) / 0.30).clamp(0.0, 1.0)
    tilt_gate = ((math.radians(80.0) - tilt) / (math.radians(80.0) - 0.35)).clamp(0.0, 1.0)
    return term_func(env, **term_params) * height_gate * tilt_gate


def arena_margin(env, apothem: float = 4.0):
    """Root clearance from an 8 m regular octagon, for evaluation diagnostics."""
    position = env.scene["robot"].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    angles = torch.arange(8, device=position.device, dtype=position.dtype) * (math.pi / 4)
    normals = torch.stack((angles.cos(), angles.sin()), dim=-1)
    return apothem - (position @ normals.T).amax(dim=-1)


class AnkleStabilityAccumulator:
    """Keep actual motion visible even when it cancels between policy samples."""

    def __init__(self, num_envs, device, settings, physics_dt, decimation):
        self.settings = dict(settings)
        if not all(math.isfinite(float(v)) and float(v) >= 0 for v in settings.values()):
            raise ValueError("Ankle stability settings must be finite and nonnegative.")
        for key in (
            "acceleration_scale_rad_s2",
            "foot_roll_scale_rad_s",
            "foot_pitch_scale_rad_s",
            "normalized_error_cap",
        ):
            if settings[key] <= 0:
                raise ValueError(f"{key} must be positive.")
        if settings["contact_force_enter_n"] <= settings["contact_force_exit_n"]:
            raise ValueError("Contact entry force must exceed exit force.")
        if physics_dt <= 0 or type(decimation) is not int or decimation < 1:
            raise ValueError("A positive physics dt and integer decimation are required.")
        self.physics_dt = float(physics_dt)
        self.decimation = decimation
        self.support_age = torch.zeros(num_envs, 2, device=device)
        self._contact_gap = torch.zeros_like(self.support_age)
        self._previous_velocity = torch.zeros(num_envs, 4, device=device)
        self._acceleration_sum = torch.zeros(num_envs, device=device)
        self._rotation_sum = torch.zeros_like(self._acceleration_sum)
        self._acceleration_square_sum = torch.zeros_like(self._acceleration_sum)
        self._angular_square_sum = torch.zeros(num_envs, 2, device=device)
        self._support_samples = torch.zeros_like(self._acceleration_sum)
        self._angular_deadband = torch.tensor(
            [settings["foot_roll_deadband_rad_s"], settings["foot_pitch_deadband_rad_s"]],
            device=device,
        )
        self._angular_scale = torch.tensor(
            [settings["foot_roll_scale_rad_s"], settings["foot_pitch_scale_rad_s"]],
            device=device,
        )
        self.sample_count = 0
        self._last_step = None

    def begin(self, joint_velocity, step_counter):
        self._previous_velocity.copy_(joint_velocity)
        self._last_step = int(step_counter)
        self.sample_count = 0
        for value in (
            self._acceleration_sum,
            self._rotation_sum,
            self._acceleration_square_sum,
            self._angular_square_sum,
            self._support_samples,
        ):
            value.zero_()

    def reset(self, env_ids=None):
        # The reward pass precedes autoreset. Retain this control step's costs
        # and global sample count for logging; begin() starts its successor.
        ids = slice(None) if env_ids is None else env_ids
        self.support_age[ids] = 0
        self._contact_gap[ids] = 0

    def _bounded_square(self, normalized):
        cap = self.settings["normalized_error_cap"]
        return torch.nan_to_num(normalized, nan=cap, posinf=cap, neginf=cap).clamp(0, cap).square()

    def observe(self, joint_velocity, foot_ang_vel_w, foot_quat_w, contact_force_w, step_counter, dt):
        if not math.isclose(float(dt), self.physics_dt, rel_tol=0, abs_tol=1e-9):
            raise RuntimeError("Ankle statistics require one physics step per observation.")
        if step_counter == self._last_step:
            return
        if self._last_step is None or step_counter != self._last_step + 1:
            raise RuntimeError("Ankle statistics skipped a physics sample.")
        if self.sample_count >= self.decimation:
            raise RuntimeError("Too many ankle physics samples in one control step.")
        cfg = self.settings
        acceleration = (joint_velocity - self._previous_velocity) / self.physics_dt
        excess = (acceleration.abs() - cfg["acceleration_deadband_rad_s2"]).clamp_min(0)
        self._acceleration_sum += self._bounded_square(excess / cfg["acceleration_scale_rad_s2"]).mean(dim=-1)
        self._acceleration_square_sum += (
            torch.nan_to_num(
                acceleration,
                nan=1e5,
                posinf=1e5,
                neginf=-1e5,
            )
            .clamp(-1e5, 1e5)
            .square()
            .mean(dim=-1)
        )

        # Hysteresis preserves support age through short force dropouts, but
        # neither rewards nor the settling timer accumulate during those gaps.
        force_z = contact_force_w[..., 2]
        contact = (force_z >= cfg["contact_force_enter_n"]) | (
            (self.support_age > 0) & (force_z > cfg["contact_force_exit_n"])
        )
        self._contact_gap = torch.where(contact, 0.0, self._contact_gap + self.physics_dt)
        self.support_age = torch.where(
            contact,
            self.support_age + self.physics_dt,
            torch.where(self._contact_gap <= cfg["contact_gap_tolerance_s"] + 1e-7, self.support_age, 0.0),
        )
        settled = contact & (self.support_age + 1e-7 >= cfg["support_settle_time_s"])

        # Rotate world angular velocity into the foot frame. Penalize changes
        # of the sole normal (roll/pitch), while allowing yaw about that normal.
        xyz = foot_quat_w[..., 1:]
        cross = torch.cross(xyz, foot_ang_vel_w, dim=-1)
        local_omega = foot_ang_vel_w - 2 * foot_quat_w[..., :1] * cross + 2 * torch.cross(xyz, cross, dim=-1)
        angular_excess = (local_omega[..., :2].abs() - self._angular_deadband).clamp_min(0)
        rotation = self._bounded_square(angular_excess / self._angular_scale).mean(dim=-1)
        self._rotation_sum += (rotation * settled).mean(dim=-1)
        self._support_samples += settled.sum(dim=-1)
        self._angular_square_sum += (
            torch.nan_to_num(local_omega[..., :2], nan=1e3, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3).square()
            * settled.unsqueeze(-1)
        ).sum(dim=1)
        self._previous_velocity.copy_(joint_velocity)
        self._last_step = int(step_counter)
        self.sample_count += 1

    def values(self):
        if self.sample_count != self.decimation:
            raise RuntimeError(f"Expected {self.decimation} ankle physics samples, got {self.sample_count}.")
        return self._acceleration_sum / self.sample_count, self._rotation_sum / self.sample_count

    def metrics(self):
        acceleration, rotation = self.values()
        support_count = self._support_samples.sum().clamp_min(1)
        angular_rms = (self._angular_square_sum.sum(dim=0) / support_count).sqrt()
        return {
            "AnkleV2/physics_acceleration_cost": acceleration.mean(),
            "AnkleV2/support_foot_rotation_cost": rotation.mean(),
            "AnkleV2/physics_acceleration_rms_rad_s2": (
                self._acceleration_square_sum.mean() / self.sample_count
            ).sqrt(),
            "AnkleV2/support_fraction": self._support_samples.mean() / (2 * self.sample_count),
            "AnkleV2/support_foot_roll_rms_rad_s": angular_rms[0],
            "AnkleV2/support_foot_pitch_rms_rad_s": angular_rms[1],
        }


def physics_acceleration_l2(env):
    return env.ankle_stability.values()[0]


def support_foot_angular_velocity_l2(env):
    return env.ankle_stability.values()[1]


def support_joint_pose_l2(env, asset_cfg, sensor_cfg, threshold: float = 1.0, normalizer_rad: float = 0.1):
    """Apply the weak default-pose cost to each supporting foot independently."""
    asset, sensor = env.scene[asset_cfg.name], env.scene[sensor_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    axis_weight = q.new_tensor((1.0, 2.0, 1.0, 2.0))
    posture = ((q - q0) / float(normalizer_rad)).square() * axis_weight
    history = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    foot_contact = torch.linalg.vector_norm(history, dim=-1).amax(dim=1) > float(threshold)
    return (posture * foot_contact.repeat_interleave(2, dim=1).to(posture.dtype)).mean(dim=1)
