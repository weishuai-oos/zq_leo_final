"""CPU configuration model for extraction regression tests.

This executes repository config classes and post_init methods while recording external
IsaacLab constructors instead of importing Kit. It does not simulate IsaacLab defaults
or dynamics; scripts/check_config.py provides the separate real-IsaacLab check.
"""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


class Node:
    def __init__(self, *args, **kwargs):
        for cls in reversed(type(self).__mro__):
            for name, value in vars(cls).items():
                is_config_function = name in {"function", "class_type", "retargetting_func", "motion_interpolate_func"}
                if (
                    not name.startswith("_")
                    and (not callable(value) or is_config_function)
                    and not isinstance(value, property)
                ):
                    setattr(self, name, copy.deepcopy(value))
        if args:
            self.args = args
        self.__dict__.update(kwargs)
        post_init = getattr(self, "__post_init__", None)
        if post_init is not None:
            post_init()

    def copy(self):
        return copy.deepcopy(self)

    def replace(self, **kwargs):
        result = self.copy()
        result.__dict__.update(kwargs)
        return result


class EnvNode(Node):
    sim = Node(physx=Node())


class Symbol:
    """Unexecuted external function, namespace, or config constructor."""

    def __init__(self, path):
        self.path = path

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return Symbol(f"{self.path}.{name}")

    def __call__(self, *args, **kwargs):
        node = Node(*args, **kwargs)
        node._constructor = self.path
        return node

    def __mro_entries__(self, bases):
        return (Node,)

    def __deepcopy__(self, memo):
        return self


def configclass(cls):
    cls.__init__ = Node.__init__
    return cls


class ConfigModel:
    """Load only config/helper definitions; no installed task package is imported."""

    def __init__(self, project: Path, package: str):
        self.project = Path(project).resolve()
        self.package = package
        self.package_root = self.project / "source" / package / package
        self.modules: dict[str, ModuleType] = {}

    def _is_config(self, name: str) -> bool:
        if name in {f"{self.package}.assets.t800", f"{self.package}.terrains.terrain_generator_cfg"}:
            return True
        if self.package == "leo_lab":
            return (
                ".config." in name
                and not name.endswith(("_mdp", "command_frames"))
                or name == "leo_lab.terrains.terrain_generator_cfg"
            )
        return name.startswith("t800_lab.tasks.t800_base.") and name.rsplit(".", 1)[-1] in {
            "constants",
            "env_cfg",
            "terrain_cfg",
            "boxing_guard_velocity_command",
        }

    def _external(self, name, attr=None):
        full = name if attr is None else f"{name}.{attr}"
        if full == f"{self.package}.terrains.trimesh.StructuredFightMatPlusTerrainCfg":
            path = self.package_root / "terrains/trimesh/mesh_terrains_cfg.py"
            classes = [
                node
                for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.ClassDef)
                and node.name in {"WallTerrainCfgMixin", "StructuredFightMatPlusTerrainCfg"}
            ]
            namespace = {
                "SubTerrainBaseCfg": Node,
                "configclass": configclass,
                "mesh_terrains": Symbol("mesh_terrains"),
            }
            module = ast.Module(
                body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *classes],
                type_ignores=[],
            )
            exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
            return namespace["StructuredFightMatPlusTerrainCfg"]
        if full.endswith(".ankle_stable_mdp"):
            path = self.package_root.joinpath(*full.split(".")[1:]).with_suffix(".py")
            spec = importlib.util.spec_from_file_location("_cpu_ankle_stable_mdp", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        if full == "isaaclab.utils.configclass":
            return configclass
        if full in {
            "isaaclab.envs.ManagerBasedRLEnvCfg",
            "isaaclab.envs.manager_based_rl_env_cfg.ManagerBasedRLEnvCfg",
        }:
            return EnvNode
        if full == "isaaclab.utils.assets.ISAAC_NUCLEUS_DIR":
            return "ISAAC_NUCLEUS_DIR"
        if full == "isaaclab.utils.assets.ISAACLAB_NUCLEUS_DIR":
            return "ISAACLAB_NUCLEUS_DIR"
        if full in {
            "leo_lab.terrains.terrain_generator.FiledTerrainGenerator",
            "t800_lab.terrains.terrain_generator.FiledTerrainGenerator",
        }:
            return Symbol("FiledTerrainGenerator")
        if full in {
            "leo_lab.terrains.terrain_generator.FiledTerrainGeneratorCfg",
            "t800_lab.terrains.terrain_generator.FiledTerrainGeneratorCfg",
        }:
            return Symbol("FiledTerrainGeneratorCfg")
        return Symbol(full)

    def _import(self, namespace, statement, module_name):
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                name = alias.name
                if name.split(".")[0] in {
                    "math",
                    "os",
                    "copy",
                    "numpy",
                    "dataclasses",
                    "functools",
                    "pathlib",
                    "typing",
                }:
                    namespace[alias.asname or name.split(".")[0]] = importlib.import_module(
                        name if alias.asname else name.split(".")[0]
                    )
                elif self._is_config(name):
                    namespace[alias.asname or name] = self.load(name)
                else:
                    namespace[alias.asname or name.split(".")[0]] = self._external(name)
            return
        if statement.module == "__future__":
            return
        target = statement.module or ""
        if statement.level:
            target = importlib.util.resolve_name("." * statement.level + target, module_name.rpartition(".")[0])
        for alias in statement.names:
            name = alias.name
            if name == "*":
                raise ValueError(f"Star import is outside the config-model boundary: {target}")
            if target.split(".")[0] in {"math", "os", "copy", "numpy", "dataclasses", "functools", "pathlib", "typing"}:
                value = getattr(importlib.import_module(target), name)
            elif not statement.module:
                child = f"{target}.{name}"
                value = self.load(child) if self._is_config(child) else self._external(child)
            elif self._is_config(target):
                value = getattr(self.load(target), name)
            else:
                value = self._external(target, name)
            namespace[alias.asname or name] = value

    def load(self, name: str) -> ModuleType:
        if name in self.modules:
            return self.modules[name]
        path = self.package_root.joinpath(*name.split(".")[1:]).with_suffix(".py")
        if not path.is_file():
            raise FileNotFoundError(path)
        module = ModuleType(name)
        module.__file__ = str(path)
        module.__package__ = name.rpartition(".")[0]
        self.modules[name] = module
        for statement in ast.parse(path.read_text(), filename=str(path)).body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                self._import(module.__dict__, statement, name)
                continue
            code = ast.Module(
                body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), statement],
                type_ignores=[],
            )
            exec(compile(ast.fix_missing_locations(code), str(path), "exec"), module.__dict__)
        return module


def normalize(value: Any):
    if isinstance(value, Symbol):
        return {"callable": value.path.rsplit(".", 1)[-1]}
    if callable(value):
        return {"callable": value.__name__}
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        for marker in ("/amp_dataset", "/resources/t800"):
            if marker in value:
                return value[value.index(marker) + 1 :]
        return value
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalize(v) for v in value]
    if isinstance(value, set):
        return sorted(normalize(v) for v in value)
    if hasattr(value, "__dict__"):
        return {name: normalize(v) for name, v in vars(value).items() if not name.startswith("_")}
    if value is None or isinstance(value, (float, int, bool)):
        return value
    raise TypeError(f"Unsupported config value {type(value).__name__}: {value!r}")
