"""Checkpoint contracts for ankle ablations and combined training from scratch."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math

from t800_lab.task_names import ANKLE_TASKS, COMBINED_ANKLE_TASK, COMBINED_V2_ANKLE_TASK

from .finetune import cap_learning_rate
from .vecenv_wrapper import CusrlT800Adapter


def resolve_ankle_iteration_budget(restored_iteration, *, max_iterations=None, additional_iterations=None):
    if type(restored_iteration) is not int or restored_iteration < 0:
        raise ValueError("Ankle continuation requires a nonnegative checkpoint iteration.")
    if max_iterations is not None and additional_iterations is not None:
        raise ValueError("--max_iterations and --additional_iterations are mutually exclusive.")
    additional = 30000 if additional_iterations is None else additional_iterations
    if type(additional) is not int or additional <= 0:
        raise ValueError("--additional_iterations must be positive.")
    endpoint = restored_iteration + additional if max_iterations is None else max_iterations
    if type(endpoint) is not int or endpoint <= restored_iteration:
        raise ValueError("Ankle continuation endpoint must exceed the restored iteration.")
    return endpoint


def _config_value(value):
    """Hash declarative entity names, excluding scene-resolved numeric IDs."""
    if isinstance(value, Mapping):
        return {key: _config_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_config_value(item) for item in value]
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if hasattr(value, "preserve_order") and hasattr(value, "name"):
        return {
            key: _config_value(getattr(value, key, None))
            for key in ("name", "joint_names", "body_names", "preserve_order")
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported ankle configuration value: {type(value).__name__}")


def ankle_configuration_signature(cfg, task):
    """Cover ankle PD, reward formulas and parameters, including Pose gates."""
    terms = {
        name: {"func": term.func, "weight": term.weight, "params": term.params}
        for name, term in vars(cfg.rewards).items()
        if name.startswith("ankle_") and term is not None and hasattr(term, "func")
    }
    feet = cfg.scene.robot.actuators["feet"]
    signature = {
        "task": task.removesuffix("-Play"),
        "reward_weights": getattr(cfg, "ankle_finetune_reward_weights", {}),
        "reward_terms": terms,
        "stiffness": feet.stiffness,
        "damping": feet.damping,
    }
    # Preserve every existing checkpoint signature.  V2 adds physical-substep
    # rewards, so its contract must also cover their timing and contact model.
    if hasattr(cfg, "ankle_substep_stability"):
        collider = cfg.events.collider_offsets
        signature["substep_physics"] = {
            "physics_dt": cfg.sim.dt,
            "decimation": cfg.decimation,
            "stability": cfg.ankle_substep_stability,
            "collider_offsets": {
                "func": collider.func, "mode": collider.mode, "params": collider.params,
            },
        }
    signature = _config_value(signature)
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()


def _validate_finetune_metadata(metadata, task):
    required = {
        "version", "task", "config_signature", "play_config_signature",
        "origin_iteration", "completed_updates", "lr_caps",
    }
    if not isinstance(metadata, Mapping) or set(metadata) != required:
        raise ValueError("Ankle checkpoint finetune metadata is incomplete.")
    if type(metadata["version"]) is not int or metadata["version"] != 1 or metadata["task"] != task:
        raise ValueError("Ankle checkpoint task/version differs.")
    for name in ("config_signature", "play_config_signature"):
        if not isinstance(metadata[name], str) or len(metadata[name]) != 64:
            raise ValueError(f"Ankle checkpoint {name} is invalid.")
    for name in ("origin_iteration", "completed_updates"):
        if type(metadata[name]) is not int or metadata[name] < 0:
            raise ValueError(f"Ankle checkpoint {name} must be a nonnegative integer.")
    caps = metadata["lr_caps"]
    if not isinstance(caps, list) or not caps or any(
        type(value) not in (float, int) or not math.isfinite(value) or value <= 0 for value in caps
    ):
        raise ValueError("Ankle checkpoint LR caps are invalid.")


def _combined_source_cfg(source_combined_cfg=None):
    if source_combined_cfg is not None:
        return source_combined_cfg
    # Called after AppLauncher starts for actual training; CPU tests inject a
    # small source fixture instead of importing IsaacLab configuration classes.
    from t800_lab.tasks.t800_ankle.env_cfg import T800AnkleCombinedEnvCfg

    return T800AnkleCombinedEnvCfg()


def validate_ankle_training_checkpoint(checkpoint, task, env_cfg, source_combined_cfg=None):
    """Allow only the explicit Combined-to-V2 training migration or a V2 resume."""
    state = checkpoint.get("environment", {})
    if task.removesuffix("-Play") == COMBINED_V2_ANKLE_TASK and (
        isinstance(state, Mapping) and set(state) == {"ankle_combined"}
    ):
        validate_ankle_play_checkpoint(
            checkpoint, COMBINED_ANKLE_TASK, _combined_source_cfg(source_combined_cfg),
        )
        return
    validate_ankle_play_checkpoint(checkpoint, task, env_cfg)


def validate_ankle_play_checkpoint(checkpoint, task, env_cfg):
    """Prevent evaluating a trained damping policy with an unrelated PD config."""
    state = checkpoint.get("environment", {})
    if task.removesuffix("-Play") == COMBINED_V2_ANKLE_TASK:
        if not isinstance(state, Mapping) or set(state) != {"ankle_finetune"}:
            raise ValueError("AnkleCombinedV2 Play requires its own checkpoint; Combined is a training source only.")
        metadata = state["ankle_finetune"]
        _validate_finetune_metadata(metadata, COMBINED_V2_ANKLE_TASK)
        if metadata["play_config_signature"] != ankle_configuration_signature(env_cfg, task):
            raise ValueError("AnkleCombinedV2 PD/reward/physics configuration differs from the checkpoint.")
        return
    if task.removesuffix("-Play") == COMBINED_ANKLE_TASK or (
        isinstance(state, Mapping) and "ankle_combined" in state
    ):
        if task.removesuffix("-Play") != COMBINED_ANKLE_TASK:
            raise ValueError("AnkleCombined checkpoint must use its matching task, including during Play.")
        if not isinstance(state, Mapping) or set(state) != {"ankle_combined"}:
            raise ValueError("AnkleCombined requires its own checkpoint; base/finetune checkpoints are incompatible.")
        metadata = state["ankle_combined"]
        if not isinstance(metadata, Mapping) or set(metadata) != {"version", "task", "play_config_signature"}:
            raise ValueError("AnkleCombined checkpoint configuration metadata is incomplete.")
        if type(metadata["version"]) is not int or metadata["version"] != 1 or metadata["task"] != COMBINED_ANKLE_TASK:
            raise ValueError("AnkleCombined checkpoint task/version differs.")
        if metadata["play_config_signature"] != ankle_configuration_signature(env_cfg, task):
            raise ValueError("AnkleCombined PD/reward configuration differs from the checkpoint.")
        return
    if isinstance(state, Mapping) and "ankle_finetune" in state:
        metadata = state["ankle_finetune"]
        if not isinstance(metadata, Mapping) or metadata.get("task") != task.removesuffix("-Play"):
            raise ValueError("Ankle checkpoint must use its matching task, including during Play.")
        if metadata.get("play_config_signature") != ankle_configuration_signature(env_cfg, task):
            raise ValueError("Ankle Play PD/reward configuration differs from the checkpoint.")
    elif task.removesuffix("-Play") in ANKLE_TASKS and state:
        raise ValueError("Ankle tasks require a base or matching ankle checkpoint.")


class AnkleCombinedCusrlAdapter(CusrlT800Adapter):
    """Train normally from zero; retain the PD/reward contract for resume and Play."""

    def __init__(self, wrapped):
        self.configuration_metadata = {
            "version": 1,
            "task": COMBINED_ANKLE_TASK,
            "play_config_signature": ankle_configuration_signature(wrapped.unwrapped.cfg, COMBINED_ANKLE_TASK),
        }
        super().__init__(wrapped)

    def state_dict(self):
        return {"ankle_combined": deepcopy(self.configuration_metadata)}

    def load_state_dict(self, state_dict):
        validate_ankle_play_checkpoint(
            {"environment": state_dict}, COMBINED_ANKLE_TASK, self.unwrapped.cfg,
        )


class AnkleCusrlAdapter(CusrlT800Adapter):
    """Save warmup progress and LR caps alongside the original agent state."""

    def __init__(self, wrapped, *, task, lr_scale=0.5, warmup_iterations=None, origin_iteration=None,
                 source_combined_cfg=None):
        if task not in (*ANKLE_TASKS, COMBINED_V2_ANKLE_TASK):
            raise ValueError(f"Unsupported ankle continuation task: {task}")
        if warmup_iterations is None:
            warmup_iterations = 1000 if task == COMBINED_V2_ANKLE_TASK else 2000
        if not math.isfinite(lr_scale) or not 0 < lr_scale <= 1:
            raise ValueError("--finetune_lr_scale must be in (0, 1].")
        if type(warmup_iterations) is not int or warmup_iterations < 0:
            raise ValueError("--finetune_warmup_iterations must be nonnegative.")
        if origin_iteration is not None and (type(origin_iteration) is not int or origin_iteration < 0):
            raise ValueError("--finetune_origin_iteration must be nonnegative.")
        self.task = task
        self.lr_scale = float(lr_scale)
        self.warmup_iterations = warmup_iterations
        self.expected_origin = origin_iteration
        self.finetune_metadata = None
        self.source_combined_cfg = source_combined_cfg
        self.source_combined_metadata = None
        cfg = wrapped.unwrapped.cfg
        self.reward_weights = deepcopy(cfg.ankle_finetune_reward_weights)
        self.play_config_signature = ankle_configuration_signature(cfg, task)
        signature = {
            "play_config_signature": self.play_config_signature,
            "lr_scale": self.lr_scale,
            "warmup_iterations": warmup_iterations,
        }
        self.config_signature = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        super().__init__(wrapped)

    def state_dict(self):
        return {"ankle_finetune": deepcopy(self.finetune_metadata)}

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, Mapping):
            raise ValueError("Ankle checkpoint environment state must be a mapping.")
        if self.task == COMBINED_V2_ANKLE_TASK:
            validate_ankle_training_checkpoint(
                {"environment": state_dict}, self.task, self.unwrapped.cfg, self.source_combined_cfg,
            )
            if set(state_dict) == {"ankle_combined"}:
                self.source_combined_metadata = deepcopy(state_dict["ankle_combined"])
                return
        if not state_dict:  # The original T800_base saves an empty environment state.
            return
        if set(state_dict) != {"ankle_finetune"}:
            raise ValueError("Ankle checkpoint environment must contain only ankle_finetune state.")
        metadata = state_dict["ankle_finetune"]
        _validate_finetune_metadata(metadata, self.task)
        if metadata["config_signature"] != self.config_signature:
            raise ValueError("Ankle checkpoint configuration or warmup/LR settings differ.")
        if metadata["play_config_signature"] != self.play_config_signature:
            raise ValueError("Ankle checkpoint PD/reward configuration differs.")
        if self.expected_origin is not None and metadata["origin_iteration"] != self.expected_origin:
            raise ValueError("Ankle checkpoint stage origin differs from the requested origin.")
        self.finetune_metadata = deepcopy(dict(metadata))

    def set_warmup(self, completed_updates):
        alpha = min(completed_updates / self.warmup_iterations, 1.0) if self.warmup_iterations else 1.0
        manager = self.unwrapped.reward_manager
        for name, (original, final) in self.reward_weights.items():
            term = manager.get_term_cfg(name)
            term.weight = original + alpha * (final - original)
            manager.set_term_cfg(name, term)
        self.metrics.record(ankle_warmup_fraction=alpha)


class AnkleFinetuneCallback:
    """Runs once before training, then after updates and before iteration increments."""

    def __init__(self):
        self.initialized = False

    def __call__(self, trainer):
        env = trainer.environment
        groups = trainer.agent.optimizer.param_groups
        initial_call = not self.initialized
        if initial_call:
            metadata = env.finetune_metadata
            if metadata is None:
                if env.task == COMBINED_V2_ANKLE_TASK and env.source_combined_metadata is None:
                    raise ValueError("AnkleCombinedV2 requires a validated Combined checkpoint before finetuning.")
                if env.expected_origin is not None and trainer.iteration != env.expected_origin:
                    raise ValueError("Source checkpoint iteration must match --finetune_origin_iteration.")
                metadata = {
                    "version": 1,
                    "task": env.task,
                    "config_signature": env.config_signature,
                    "play_config_signature": env.play_config_signature,
                    "origin_iteration": int(trainer.iteration),
                    "completed_updates": 0,
                    "lr_caps": [float(group["lr"]) * env.lr_scale for group in groups],
                }
                env.finetune_metadata = metadata
            if len(metadata["lr_caps"]) != len(groups):
                raise ValueError("Ankle checkpoint LR cap group count differs.")
        else:
            metadata = env.finetune_metadata
        if trainer.iteration != metadata["origin_iteration"] + metadata["completed_updates"]:
            raise ValueError("Ankle checkpoint iteration does not match its warmup progress.")
        if not initial_call:
            metadata["completed_updates"] += 1
        cap_learning_rate(trainer.agent, metadata["lr_caps"])
        env.set_warmup(metadata["completed_updates"])
        self.initialized = True
