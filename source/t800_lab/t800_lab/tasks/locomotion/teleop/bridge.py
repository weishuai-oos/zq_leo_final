from __future__ import annotations
from dataclasses import dataclass
import types
import torch
from isaaclab.devices import create_teleop_device
from .devices import Se2VelGamepad, Se2VelGamepadCfg, Se2VelKeyboard, Se2VelKeyboardCfg


@dataclass
class LocoTeleopProfile:
    profile: str
    command_name: str
    command_dim: int


class LocoTeleopBridge:
    def __init__(self, env, profile: LocoTeleopProfile, devices):
        self.env = env
        self.profile = profile
        self.devices = devices
        self.term = env.command_manager.get_term(profile.command_name)
        self._original_resample_command = self.term._resample_command
        self._original_update_command = self.term._update_command
        self._install_hooks()

    def _install_hooks(self):
        self.term._resample_command = types.MethodType(lambda term, env_ids: None, self.term)
        if self.profile.profile == "se2_velocity":
            self.term._update_command = types.MethodType(_update_se2_command, self.term)
        else:
            raise ValueError(f"Unsupported locomotion teleop profile: {self.profile.profile}")
        self.term._teleop_bridge = self

    def close(self):
        self.term._resample_command = self._original_resample_command
        self.term._update_command = self._original_update_command

    def get_command(self) -> torch.Tensor:
        commands = []
        for device in self.devices:
            command = device.advance()
            if isinstance(command, torch.Tensor):
                command = command.to(self.env.device)
            else:
                command = torch.as_tensor(command, dtype=torch.float32, device=self.env.device)
            commands.append(command)

        if not commands:
            return torch.zeros(self.profile.command_dim, device=self.env.device)

        for command in commands:
            if torch.any(torch.abs(command) > 1.0e-6):
                return command
        return commands[0]


def _update_se2_command(term):
    bridge: LocoTeleopBridge = term._teleop_bridge
    command = bridge.get_command().view(1, 3).expand(term.num_envs, -1)
    transform = getattr(term, "transform_teleop_command", None)
    if callable(transform):
        command = transform(command)
    term.vel_command_b[:] = command
    if hasattr(term, "is_heading_env"):
        term.is_heading_env.fill_(False)
    if hasattr(term, "is_standing_env"):
        term.is_standing_env.fill_(False)
    if hasattr(term, "heading_target"):
        term.heading_target.zero_()


def detect_locomotion_teleop_profile(env) -> LocoTeleopProfile | None:
    if "base_velocity" not in env.command_manager.active_terms:
        return None
    return LocoTeleopProfile("se2_velocity", "base_velocity", 3)


def attach_default_locomotion_teleop(env, teleop_mode: str = "auto") -> LocoTeleopBridge | None:
    if teleop_mode == "none":
        return None
    profile = detect_locomotion_teleop_profile(env)
    if profile is None:
        return None
    term = env.command_manager.get_term(profile.command_name)
    kwargs = dict(
        sim_device=env.device,
        v_x_range=term.cfg.ranges.lin_vel_x,
        v_y_range=term.cfg.ranges.lin_vel_y,
        omega_z_range=term.cfg.ranges.ang_vel_z,
    )
    devices_cfg = {
        "keyboard": Se2VelKeyboardCfg(class_type=Se2VelKeyboard, **kwargs),
        "gamepad": Se2VelGamepadCfg(class_type=Se2VelGamepad, **kwargs),
    }
    order = ["gamepad", "keyboard"] if teleop_mode == "auto" else [teleop_mode]
    devices = [create_teleop_device(name, devices_cfg) for name in order]
    return LocoTeleopBridge(env, profile, devices)
