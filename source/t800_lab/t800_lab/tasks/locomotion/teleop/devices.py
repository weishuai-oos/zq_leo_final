from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
import weakref
import numpy as np
import torch
import carb
import carb.input
import omni
from isaaclab.devices import DeviceBase, DeviceCfg


def _keyboard_event_name(event) -> str | None:
    """Return a normalized keyboard event name across Isaac Sim variants."""
    key_input = getattr(event, "input", None)
    if key_input is None:
        return None
    if isinstance(key_input, str):
        return key_input
    return getattr(key_input, "name", None)


@dataclass
class Se2VelKeyboardCfg(DeviceCfg):
    v_x_range: tuple[float, float] = (-1.0, 1.0)
    v_y_range: tuple[float, float] = (-1.0, 1.0)
    omega_z_range: tuple[float, float] = (-1.0, 1.0)
    num_gears: int = 4
    initial_gear: int = 1
    class_type: type[DeviceBase] | None = None


class Se2VelKeyboard(DeviceBase):
    """Keyboard controller for SE(2) velocity with asymmetric ranges and gear switching."""

    def __init__(self, cfg: Se2VelKeyboardCfg):
        super().__init__()
        self.v_x_range = cfg.v_x_range
        self.v_y_range = cfg.v_y_range
        self.omega_z_range = cfg.omega_z_range
        self.num_gears = max(1, int(cfg.num_gears))
        self.gear = int(np.clip(cfg.initial_gear, 1, self.num_gears))
        self._sim_device = cfg.sim_device

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        self._additional_callbacks: dict[str, Callable] = {}
        self._toggle_key_held = False
        self._pressed = {
            "up": False,
            "down": False,
            "left": False,
            "right": False,
            "yaw_neg": False,
            "yaw_pos": False,
        }
        self._base_command = np.zeros(3, dtype=np.float32)
        self._key_to_action = {
            "NUMPAD_8": "up",
            "UP": "up",
            "NUMPAD_2": "down",
            "DOWN": "down",
            "NUMPAD_6": "left",
            "LEFT": "left",
            "NUMPAD_4": "right",
            "RIGHT": "right",
            "Z": "yaw_neg",
            "NUMPAD_9": "yaw_neg",
            "X": "yaw_pos",
            "NUMPAD_7": "yaw_pos",
        }

    def __del__(self):
        if hasattr(self, "_input") and self._keyboard_sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def reset(self):
        for key in self._pressed:
            self._pressed[key] = False
        self._toggle_key_held = False
        self._base_command.fill(0.0)

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        return torch.tensor(self._base_command, dtype=torch.float32, device=self._sim_device)

    def _on_keyboard_event(self, event, *args, **kwargs):
        key_name = _keyboard_event_name(event)
        if key_name is None:
            return True
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key_name == "L":
                self.reset()
            elif key_name == "C":
                if not self._toggle_key_held:
                    self.gear = (self.gear % self.num_gears) + 1
                    self._toggle_key_held = True
                    self._update_base_command()
            elif key_name in self._key_to_action:
                self._pressed[self._key_to_action[key_name]] = True
                self._update_base_command()
            if key_name in self._additional_callbacks:
                self._additional_callbacks[key_name]()
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if key_name == "C":
                self._toggle_key_held = False
            elif key_name in self._key_to_action:
                self._pressed[self._key_to_action[key_name]] = False
                self._update_base_command()
        return True

    def _update_base_command(self):
        self._base_command.fill(0.0)
        scale = self.gear / self.num_gears
        if self._pressed["up"] ^ self._pressed["down"]:
            self._base_command[0] = (self.v_x_range[1] if self._pressed["up"] else self.v_x_range[0]) * scale
        if self._pressed["left"] ^ self._pressed["right"]:
            self._base_command[1] = (self.v_y_range[1] if self._pressed["left"] else self.v_y_range[0]) * scale
        if self._pressed["yaw_neg"] ^ self._pressed["yaw_pos"]:
            self._base_command[2] = (
                self.omega_z_range[0] if self._pressed["yaw_neg"] else self.omega_z_range[1]
            ) * scale


@dataclass
class Se2VelGamepadCfg(DeviceCfg):
    v_x_range: tuple[float, float] = (-1.0, 1.0)
    v_y_range: tuple[float, float] = (-1.0, 1.0)
    omega_z_range: tuple[float, float] = (-1.0, 1.0)
    dead_zone: float = 0.01
    class_type: type[DeviceBase] | None = None


class Se2VelGamepad(DeviceBase):
    """Gamepad controller for SE(2) velocity with range-aware scaling."""

    def __init__(self, cfg: Se2VelGamepadCfg):
        super().__init__()
        carb.settings.get_settings().set_bool("/persistent/app/omniverse/gamepadCameraControl", False)
        self.v_x_range = cfg.v_x_range
        self.v_y_range = cfg.v_y_range
        self.omega_z_range = cfg.omega_z_range
        self.dead_zone = cfg.dead_zone
        self._sim_device = cfg.sim_device

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._gamepad = self._appwindow.get_gamepad(0)
        self._gamepad_sub = self._input.subscribe_to_gamepad_events(
            self._gamepad,
            lambda event, *args, obj=weakref.proxy(self): obj._on_gamepad_event(event, *args),
        )
        self._base_command_raw = np.zeros([2, 3], dtype=np.float32)
        self._additional_callbacks = {}
        self._input_mapping = {
            carb.input.GamepadInput.LEFT_STICK_UP: (0, 0, abs(self.v_x_range[1])),
            carb.input.GamepadInput.LEFT_STICK_DOWN: (1, 0, abs(self.v_x_range[0])),
            carb.input.GamepadInput.LEFT_STICK_LEFT: (0, 1, abs(self.v_y_range[0])),
            carb.input.GamepadInput.LEFT_STICK_RIGHT: (1, 1, abs(self.v_y_range[1])),
            carb.input.GamepadInput.RIGHT_STICK_LEFT: (0, 2, abs(self.omega_z_range[1])),
            carb.input.GamepadInput.RIGHT_STICK_RIGHT: (1, 2, abs(self.omega_z_range[0])),
        }

    def __del__(self):
        if hasattr(self, "_input") and self._gamepad_sub is not None:
            self._input.unsubscribe_to_gamepad_events(self._gamepad, self._gamepad_sub)
            self._gamepad_sub = None

    def reset(self):
        self._base_command_raw.fill(0.0)

    def add_callback(self, key: carb.input.GamepadInput, func: Callable):
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        command_sign = self._base_command_raw[1, :] > self._base_command_raw[0, :]
        command = self._base_command_raw.max(axis=0)
        command[command_sign] *= -1
        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)

    def _on_gamepad_event(self, event: carb.input.GamepadEvent, *args, **kwargs):
        current_value = event.value
        if abs(current_value) < self.dead_zone:
            current_value = 0.0
        if event.input in self._input_mapping:
            direction, axis, sensitivity = self._input_mapping[event.input]
            self._base_command_raw[direction, axis] = sensitivity * current_value
        if event.input in self._additional_callbacks:
            self._additional_callbacks[event.input]()
        return True
