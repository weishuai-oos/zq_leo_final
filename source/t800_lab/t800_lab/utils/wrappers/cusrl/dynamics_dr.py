"""Strict dynamics-randomization continuation for the Impact stages.

The DR stage deliberately has its own checkpoint namespace.  In particular, a
DR checkpoint must never be mistaken for an ordinary ImpactV2/Plus checkpoint:
the latter validators are still the authority for validating the *source*.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from typing import Any

from .finetune import cap_learning_rate
from .impact import (
    FINETUNE_CONFIG_VERSION,
    IMPACT_V2_PLUS_TASK,
    IMPACT_V2_TASK,
    ImpactCusrlAdapter,
    _validate_metadata,
    impact_v2_configuration_signature,
    impact_v2_plus_configuration_signature,
    validate_impact_v2_checkpoint,
    validate_impact_v2_plus_checkpoint,
)

DR_CONFIG_VERSION = 1
_DR_REQUIRED_RANGES = (
    "joint_offset_range", "mass_scale_range", "torso_mass_delta_range", "torso_com_offset_range",
)


def configure_dynamics_dr_amp_batch(agent: Any, world_size: int) -> int:
    """Keep the two-GPU sources' 2 × 512 discriminator samples per update."""
    if type(world_size) is not int or world_size < 1 or 1024 % world_size:
        raise ValueError("DR world size must divide the source AMP global batch of 1024.")
    hooks = [hook for hook in agent.hook if hook.name == "scaled_adversarial_motion_prior"]
    if len(hooks) != 1:
        raise ValueError("DR continuation requires exactly one scaled AMP hook.")
    hooks[0].update_attribute("batch_size", 1024 // world_size)
    return 1024 // world_size


def _payload(value: Any) -> Any:
    if is_dataclass(value):
        return _payload(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_payload(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Dynamics-randomization ranges must be finite.")
        return value
    # Config objects are normally dataclasses.  Refuse opaque values so that a
    # signature cannot silently omit a newly-added randomization parameter.
    raise ValueError(f"Unsupported dynamics-randomization value: {type(value).__name__}.")


def _validate_ranges(value: Any, path: str = "dynamics_randomization") -> Any:
    value = _payload(value)
    if isinstance(value, Mapping):
        if path == "dynamics_randomization":
            if not value:
                raise ValueError("Dynamics-randomization configuration must not be empty.")
            missing = [name for name in _DR_REQUIRED_RANGES if name not in value]
            if missing:
                raise ValueError(f"Dynamics-randomization configuration is missing ranges: {', '.join(missing)}.")
        return {key: _validate_ranges(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, list):
        result = [_validate_ranges(item, f"{path}[]") for item in value]
        if result and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in result):
            if path.endswith("torso_com_offset_range"):
                if len(result) != 3 or any(item < 0 for item in result):
                    raise ValueError("COM offset bounds must have three nonnegative amplitudes.")
            elif len(result) != 2 or result[0] > result[1]:
                raise ValueError(f"{path} is not an ordered range.")
            if path.endswith("mass_scale_range") and any(item <= 0 for item in result):
                raise ValueError("Mass scale ranges must be positive.")
        return result
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"{path} contains a non-finite value.")
        return value
    return value


def _config_signature(cfg: Any, parent_task: str) -> str:
    if parent_task == IMPACT_V2_TASK:
        return impact_v2_configuration_signature(cfg)
    if parent_task == IMPACT_V2_PLUS_TASK:
        return impact_v2_plus_configuration_signature(cfg)
    raise ValueError(f"Unsupported DR parent task: {parent_task!r}.")


def dynamics_dr_configuration_signature(cfg: Any) -> str:
    """Hash only the immutable DR contract and its exact parent contract."""
    parent_task = getattr(cfg, "dr_parent_task", None)
    source_iteration = getattr(cfg, "dr_source_iteration", None)
    parent_cfg = getattr(cfg, "dr_parent_cfg", None)
    if parent_task not in (IMPACT_V2_TASK, IMPACT_V2_PLUS_TASK):
        raise ValueError("DR configuration must name an ImpactV2 or ImpactV2Plus parent task.")
    if type(source_iteration) is not int or source_iteration < 0:
        raise ValueError("DR source iteration must be a nonnegative integer.")
    if parent_cfg is None:
        raise ValueError("DR configuration must retain the pre-DR parent configuration.")
    ranges = _validate_ranges(getattr(cfg, "dynamics_randomization", None))
    contract = {
        "version": DR_CONFIG_VERSION,
        "parent_task": parent_task,
        "source_iteration": source_iteration,
        "parent_configuration_signature": _config_signature(parent_cfg, parent_task),
        "dr_environment_signature": _config_signature(cfg, parent_task),
        "run_name": getattr(cfg, "run_name", None),
        "dynamics_randomization": ranges,
    }
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source_checkpoint(checkpoint: Mapping[str, Any], cfg: Any, *, allow_source: bool) -> tuple[dict[str, Any], int, str, int]:
    parent_task = getattr(cfg, "dr_parent_task", None)
    source_iteration = getattr(cfg, "dr_source_iteration", None)
    if parent_task not in (IMPACT_V2_TASK, IMPACT_V2_PLUS_TASK):
        raise ValueError("DR checkpoint parent task is unsupported.")
    if type(source_iteration) is not int or source_iteration < 0:
        raise ValueError("DR source iteration must be a nonnegative integer.")
    iteration = checkpoint.get("iteration")
    if type(iteration) is not int or iteration != source_iteration:
        raise ValueError("DR source checkpoint iteration does not match dr_source_iteration.")
    parent_cfg = getattr(cfg, "dr_parent_cfg", None)
    if parent_cfg is None:
        raise ValueError("DR source validation requires dr_parent_cfg.")
    if parent_task == IMPACT_V2_TASK:
        validate_impact_v2_checkpoint(checkpoint, parent_cfg, allow_source=allow_source)
    else:
        validate_impact_v2_plus_checkpoint(checkpoint, parent_cfg, allow_source=allow_source)
    environment = checkpoint.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("DR source checkpoint environment metadata is incomplete.")
    if not isinstance(environment.get("impact"), Mapping) or not isinstance(environment.get("finetune"), Mapping):
        raise ValueError("DR source checkpoint must contain impact runtime and finetune metadata.")
    source_runtime_completed = environment["impact"].get("completed_updates")
    if type(source_runtime_completed) is not int or source_runtime_completed < 0:
        raise ValueError("DR source runtime progress is invalid.")
    return deepcopy(dict(environment)), iteration, parent_task, source_runtime_completed


def validate_dynamics_dr_checkpoint(checkpoint: Mapping[str, Any], cfg: Any, *, allow_source: bool = False) -> None:
    """Validate either a parent source (training-only) or a DR checkpoint."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Dynamics DR requires a checkpoint mapping.")
    environment = checkpoint.get("environment")
    if allow_source and isinstance(environment, Mapping) and "dynamics_dr" not in environment:
        _source_checkpoint(checkpoint, cfg, allow_source=True)
        return
    if not isinstance(environment, Mapping) or set(environment) != {"impact", "finetune", "dynamics_dr"}:
        raise ValueError("Dynamics DR checkpoint metadata must contain impact, finetune and dynamics_dr only.")
    dr = environment["dynamics_dr"]
    if not isinstance(dr, Mapping):
        raise ValueError("Dynamics DR provenance is incomplete.")
    required = {"version", "task", "parent_task", "source_task", "source_iteration", "source_environment",
                "parent_configuration_signature", "dr_environment_signature", "run_name",
                "dr_configuration_signature", "curriculum_offset", "stage_origin_iteration", "completed_updates"}
    if set(dr) != required or dr["version"] != DR_CONFIG_VERSION:
        raise ValueError("Dynamics DR provenance keys/version differ.")
    if dr["task"] != getattr(cfg, "run_name", None):
        raise ValueError("Dynamics DR checkpoint task differs.")
    if dr["parent_task"] != getattr(cfg, "dr_parent_task", None) or dr["source_task"] != dr["parent_task"]:
        raise ValueError("Dynamics DR parent/source task differs.")
    if dr["source_iteration"] != getattr(cfg, "dr_source_iteration", None):
        raise ValueError("Dynamics DR source iteration differs.")
    if (
        type(dr["stage_origin_iteration"]) is not int
        or type(dr["completed_updates"]) is not int
        or dr["stage_origin_iteration"] != dr["source_iteration"]
        or dr["completed_updates"] < 0
    ):
        raise ValueError("Dynamics DR stage progress is invalid.")
    if checkpoint.get("iteration") != dr["stage_origin_iteration"] + dr["completed_updates"]:
        raise ValueError("Dynamics DR checkpoint iteration does not match stage progress.")
    if type(dr["curriculum_offset"]) is not int or dr["curriculum_offset"] < 0:
        raise ValueError("Dynamics DR curriculum offset is invalid.")
    expected_parent = _config_signature(getattr(cfg, "dr_parent_cfg"), dr["parent_task"])
    if dr["parent_configuration_signature"] != expected_parent:
        raise ValueError("Dynamics DR parent configuration signature differs.")
    if dr.get("dr_environment_signature") != _config_signature(cfg, dr["parent_task"]):
        raise ValueError("Dynamics DR environment configuration signature differs.")
    if dr.get("run_name") != getattr(cfg, "run_name", None):
        raise ValueError("Dynamics DR run name differs.")
    if dr["dr_configuration_signature"] != dynamics_dr_configuration_signature(cfg):
        raise ValueError("Dynamics DR randomization configuration differs.")
    source = dr["source_environment"]
    if not isinstance(source, Mapping):
        raise ValueError("Dynamics DR source environment provenance is incomplete.")
    # Source provenance is checked with the original validator, not by trusting
    # the copied metadata in this checkpoint.
    _, _, _, source_runtime_completed = _source_checkpoint(
        {"environment": source, "iteration": dr["source_iteration"]}, cfg, allow_source=False,
    )
    if dr["curriculum_offset"] != source_runtime_completed:
        raise ValueError("Dynamics DR curriculum offset differs from the source runtime progress.")
    if environment["finetune"].get("lr_caps") != source["finetune"].get("lr_caps"):
        raise ValueError("Dynamics DR must retain the source learning-rate caps.")
    metadata = _validate_metadata(environment["finetune"], environment["impact"].get("config_signature"), dr["task"])
    if metadata["stage_origin_iteration"] != dr["stage_origin_iteration"] or metadata["completed_updates"] != dr["completed_updates"]:
        raise ValueError("Dynamics DR finetune and provenance progress differ.")
    if environment["impact"].get("completed_updates") != dr["curriculum_offset"] + dr["completed_updates"]:
        raise ValueError("Dynamics DR runtime and curriculum progress differ.")


class DynamicsDRCusrlAdapter(ImpactCusrlAdapter):
    """Adapter whose only migration route is the exact declared parent source."""

    def __init__(self, wrapped, **spec_kwargs):
        self.dr_source_environment = None
        self.dr_source_task = None
        self.dr_source_iteration = None
        self.curriculum_offset = 0
        super().__init__(wrapped, **spec_kwargs)

    @property
    def task(self):
        return getattr(self.unwrapped.cfg, "run_name", None)

    def state_dict(self) -> dict[str, Any]:
        runtime = self.impact_runtime.state_dict()
        metadata = deepcopy(self.finetune_metadata)
        if metadata is None or self.dr_source_environment is None:
            return {"impact": runtime, "finetune": metadata, "dynamics_dr": None}
        cfg = self.unwrapped.cfg
        return {
            "impact": runtime,
            "finetune": metadata,
            "dynamics_dr": {
                "version": DR_CONFIG_VERSION, "task": self.task,
                "parent_task": cfg.dr_parent_task, "source_task": self.dr_source_task,
                "source_iteration": self.dr_source_iteration,
                "source_environment": deepcopy(self.dr_source_environment),
                "parent_configuration_signature": _config_signature(cfg.dr_parent_cfg, cfg.dr_parent_task),
                "dr_environment_signature": _config_signature(cfg, cfg.dr_parent_task),
                "run_name": self.task,
                "dr_configuration_signature": dynamics_dr_configuration_signature(cfg),
                "curriculum_offset": self.curriculum_offset,
                "stage_origin_iteration": metadata["stage_origin_iteration"],
                "completed_updates": metadata["completed_updates"],
            },
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            raise ValueError("Dynamics DR checkpoint environment state must be a mapping.")
        cfg = self.unwrapped.cfg
        checkpoint = {"environment": state_dict, "iteration": getattr(cfg, "dr_source_iteration", -1)}
        if "dynamics_dr" in state_dict:
            dr = state_dict["dynamics_dr"]
            if not isinstance(dr, Mapping):
                raise ValueError("Dynamics DR provenance is incomplete.")
            checkpoint["iteration"] = dr.get("stage_origin_iteration", -1) + dr.get("completed_updates", -1)
            validate_dynamics_dr_checkpoint(checkpoint, cfg)
            self.impact_runtime.load_state_dict(state_dict["impact"])
            self.finetune_metadata = _validate_metadata(state_dict["finetune"], state_dict["impact"].get("config_signature"), self.task)
            self.dr_source_environment = deepcopy(dr["source_environment"])
            self.dr_source_task, self.dr_source_iteration = dr["source_task"], dr["source_iteration"]
            self.curriculum_offset = dr["curriculum_offset"]
            return
        source, iteration, task, offset = _source_checkpoint(checkpoint, cfg, allow_source=False)
        self.impact_runtime.load_state_dict(source["impact"])
        self.dr_source_environment, self.dr_source_task = source, task
        self.dr_source_iteration, self.curriculum_offset = iteration, offset
        self.finetune_metadata = None


class DynamicsDRFinetuneCallback:
    """Continue the parent runtime offset without warmup or LR halving."""

    def __init__(self):
        self.initialized = False

    def __call__(self, trainer) -> None:
        environment = trainer.environment
        runtime = environment.impact_runtime
        groups = trainer.agent.optimizer.param_groups
        if not self.initialized:
            metadata = environment.finetune_metadata
            if metadata is None:
                if environment.dr_source_environment is None or trainer.iteration != environment.dr_source_iteration:
                    raise ValueError("Dynamics DR must restore its exact validated source before training.")
                source_caps = environment.dr_source_environment["finetune"].get("lr_caps")
                if not isinstance(source_caps, list) or len(source_caps) != len(groups):
                    raise ValueError("Dynamics DR source learning-rate caps do not match optimizer groups.")
                metadata = {
                    "config_version": FINETUNE_CONFIG_VERSION, "task": environment.task,
                    "stage_origin_iteration": int(trainer.iteration), "completed_updates": 0,
                    "lr_caps": deepcopy(source_caps), "config_signature": runtime.state_dict()["config_signature"],
                    "config_migrations": [],
                }
            else:
                metadata = _validate_metadata(metadata, runtime.state_dict()["config_signature"], environment.task)
                if trainer.iteration != metadata["stage_origin_iteration"] + metadata["completed_updates"]:
                    raise ValueError("Dynamics DR checkpoint iteration does not match stage progress.")
            environment.finetune_metadata = metadata
            self.initialized = True
        else:
            metadata = environment.finetune_metadata
            if trainer.iteration != metadata["stage_origin_iteration"] + metadata["completed_updates"]:
                raise ValueError("Unexpected Dynamics DR callback iteration.")
            metadata["completed_updates"] += 1
        self._cap_learning_rate(trainer.agent, metadata["lr_caps"])
        runtime.set_update(environment.curriculum_offset + metadata["completed_updates"])

    _cap_learning_rate = staticmethod(cap_learning_rate)
