"""Stage-two state and strict continuation without changing CusRL hook identities."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from importlib import import_module
import json
import math
from types import SimpleNamespace
from typing import Any

from .checkpoint import validate_agent_checkpoint_compatibility
from .finetune import cap_learning_rate
from .vecenv_wrapper import CusrlT800Adapter

IMPACT_TASK = "T800_base-Impact"
IMPACT_V2_TASK = "T800_base-ImpactV2"
IMPACT_V2_PLUS_TASK = "T800_base-ImpactV2Plus"
SOURCE_V2_TASK = "T800_base-AnkleCombinedV2"
FINETUNE_CONFIG_VERSION = 2
DEFAULT_ADDITIONAL_ITERATIONS = 40000


def resolve_impact_iteration_budget(
    restored_iteration: int, *, max_iterations: int | None, additional_iterations: int | None
) -> int:
    """Keep the existing absolute endpoint flag; default to 40k new updates."""
    if max_iterations is not None and additional_iterations is not None:
        raise ValueError("--max_iterations and --additional_iterations are mutually exclusive.")
    additional = DEFAULT_ADDITIONAL_ITERATIONS if additional_iterations is None else additional_iterations
    if additional <= 0:
        raise ValueError("--additional_iterations must be positive.")
    endpoint = restored_iteration + additional if max_iterations is None else max_iterations
    if endpoint <= restored_iteration:
        raise ValueError(
            f"Training endpoint {endpoint} must exceed restored checkpoint iteration {restored_iteration}; "
            "use --additional_iterations for a new update budget."
        )
    return endpoint


def _validate_metadata(metadata: Any, config_signature: str, task: str = IMPACT_TASK) -> dict[str, Any]:
    keys = {
        "config_version", "task", "stage_origin_iteration", "completed_updates", "lr_caps", "config_signature"
    }
    if not isinstance(metadata, Mapping):
        raise ValueError("Impact checkpoint contains incomplete finetune metadata.")
    version = metadata.get("config_version")
    if type(version) is not int or version not in (1, FINETUNE_CONFIG_VERSION):
        raise ValueError("Impact checkpoint finetune configuration version differs.")
    if version == FINETUNE_CONFIG_VERSION:
        keys.add("config_migrations")
    if set(metadata) != keys:
        raise ValueError("Impact checkpoint contains incomplete finetune metadata.")
    if metadata["task"] != task:
        raise ValueError("Impact checkpoint finetune configuration version/task differs.")
    if metadata["config_signature"] != config_signature:
        raise ValueError("Impact checkpoint training configuration differs.")
    for key in ("stage_origin_iteration", "completed_updates"):
        if type(metadata[key]) is not int or metadata[key] < 0:
            raise ValueError(f"Impact checkpoint {key} must be a nonnegative integer.")
    caps = metadata["lr_caps"]
    if not isinstance(caps, list) or not caps or any(
        type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in caps
    ):
        raise ValueError("Impact checkpoint learning-rate caps are invalid.")
    migrations = metadata.get("config_migrations", [])
    if not isinstance(migrations, list):
        raise ValueError("Impact checkpoint configuration migration history is invalid.")
    previous_signature = None
    previous_update = -1
    for migration in migrations:
        if (
            not isinstance(migration, Mapping)
            or set(migration) != {"change", "from_signature", "to_signature", "completed_updates"}
            or migration["change"] != "stress_single_hit"
            or any(
                not isinstance(migration[key], str) or len(migration[key]) != 64
                or any(ch not in "0123456789abcdef" for ch in migration[key])
                for key in ("from_signature", "to_signature")
            )
            or migration["from_signature"] == migration["to_signature"]
            or (previous_signature is not None and migration["from_signature"] != previous_signature)
            or type(migration["completed_updates"]) is not int
            or not max(0, previous_update) <= migration["completed_updates"] <= metadata["completed_updates"]
        ):
            raise ValueError("Impact checkpoint configuration migration history is invalid.")
        previous_signature = migration["to_signature"]
        previous_update = migration["completed_updates"]
    if previous_signature is not None and previous_signature != config_signature:
        raise ValueError("Impact checkpoint configuration migration history does not reach its current signature.")
    result = deepcopy(dict(metadata))
    result["config_version"] = FINETUNE_CONFIG_VERSION
    result["config_migrations"] = deepcopy(migrations)
    return result


class ImpactCusrlAdapter(CusrlT800Adapter):
    """Persist curriculum/continuation metadata while resetting physical episodes normally."""

    task = IMPACT_TASK

    def __init__(self, wrapped, **spec_kwargs):
        self.finetune_metadata: dict[str, Any] | None = None
        super().__init__(wrapped, **spec_kwargs)

    @property
    def impact_runtime(self):
        return self.unwrapped.impact_runtime

    def state_dict(self) -> dict[str, Any]:
        return {
            "impact": self.impact_runtime.state_dict(),
            "finetune": deepcopy(self.finetune_metadata),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        # Stage one saved the base adapter's empty state. Loading it must not
        # mutate the newly constructed head or disturbance runtime.
        if not state_dict:
            return
        if set(state_dict) != {"impact", "finetune"}:
            raise ValueError("Impact checkpoint environment state must contain impact and finetune.")
        runtime = self.impact_runtime
        source_signature = state_dict["impact"].get("config_signature")
        migrating = runtime.validate_checkpoint_signature(source_signature)
        metadata = _validate_metadata(state_dict["finetune"], source_signature, self.task)
        if state_dict["impact"].get("completed_updates") != metadata["completed_updates"]:
            raise ValueError("Impact runtime and finetune update counts differ.")
        runtime.load_state_dict(state_dict["impact"])
        if migrating:
            signature = runtime.state_dict()["config_signature"]
            metadata["config_migrations"].append({
                "change": "stress_single_hit", "from_signature": source_signature,
                "to_signature": signature, "completed_updates": metadata["completed_updates"],
            })
            metadata["config_signature"] = signature
            print(
                f"[INFO] Impact checkpoint stress-load migration: 70% combinations / max 5 hits -> "
                f"isolated hits; completed_updates={metadata['completed_updates']}; "
                f"mode={runtime.cfg.mode}. Model, optimizer and stage progress are retained.", flush=True,
            )
        self.finetune_metadata = metadata


class StrictCheckpointAgentFactory:
    """Validate before Trainer's permissive loader sees a checkpoint."""

    def __init__(self, factory, saved_agent: Mapping[str, Any]):
        self.factory = factory
        self.saved_agent = saved_agent

    def from_environment(self, environment):
        agent = self.factory.from_environment(environment)
        validate_agent_checkpoint_compatibility(self.saved_agent, agent)
        return agent


def combined_v2_source_signature(cfg):
    """Freeze the historical V2 contract without importing its removable task.

    Called on the independent task's V2 configuration BEFORE impact changes.
    Callable identities are strings only in this hashing copy, never executable
    imports or mutations of the actual reward configuration.
    """
    from .ankle import ankle_configuration_signature

    rewards = deepcopy(cfg.rewards)
    for name, legacy_name in (
        ("ankle_support_joint_pose_l2", "mdp.support_joint_pose_l2"),
        ("ankle_physics_acceleration_l2", "stability.physics_acceleration_l2"),
        ("ankle_support_foot_angular_velocity_l2", "stability.support_foot_angular_velocity_l2"),
    ):
        getattr(rewards, name).func = "t800_lab.tasks.t800_ankle." + legacy_name
    source = SimpleNamespace(
        rewards=rewards, scene=cfg.scene, events=cfg.events, sim=cfg.sim, decimation=cfg.decimation,
        ankle_substep_stability=cfg.ankle_substep_stability,
        ankle_finetune_reward_weights=cfg.ankle_finetune_reward_weights,
    )
    return ankle_configuration_signature(source, SOURCE_V2_TASK)


def impact_v2_configuration_signature(cfg):
    """Bind resume/Play to the actual PD, substep costs and recovery formulas."""
    from .ankle import _config_value, ankle_configuration_signature

    signature = {
        "task": IMPACT_V2_TASK,
        "ankle": ankle_configuration_signature(cfg, IMPACT_V2_TASK),
        "rewards": {
            name: {"func": term.func, "weight": term.weight, "params": term.params}
            for name, term in vars(cfg.rewards).items()
            if term is not None and hasattr(term, "func")
        },
        "fall": {"func": cfg.terminations.base_contact.func, "params": cfg.terminations.base_contact.params},
        "action": cfg.actions.joint_pos.class_type,
        "material": cfg.events.physics_material.params,
    }
    return hashlib.sha256(json.dumps(_config_value(signature), sort_keys=True).encode()).hexdigest()


def impact_v2_parent_configuration_signature(cfg):
    """Freeze V2's contract before applying Plus changes, without importing V2.

    The independent Plus package initially builds the same configuration with
    local callable copies. Canonicalize only that package prefix in a hashing
    copy; reward weights, parameters, PD and contact settings remain checked.
    """
    from .ankle import _config_value

    def canonical(value):
        value = _config_value(value)
        if isinstance(value, str):
            return value.replace("t800_lab.tasks.t800_impact_v2_plus.", "t800_lab.tasks.t800_impact_v2.")
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in value.items()}
        if isinstance(value, list):
            return [canonical(item) for item in value]
        return value

    rewards = deepcopy(cfg.rewards)
    for term in vars(rewards).values():
        if term is not None and hasattr(term, "func"):
            term.func, term.params = canonical(term.func), canonical(term.params)
    actions = SimpleNamespace(joint_pos=SimpleNamespace(class_type=canonical(cfg.actions.joint_pos.class_type)))
    termination = SimpleNamespace(base_contact=SimpleNamespace(
        func=canonical(cfg.terminations.base_contact.func), params=canonical(cfg.terminations.base_contact.params),
    ))
    source = SimpleNamespace(
        rewards=rewards, scene=cfg.scene, events=cfg.events, sim=cfg.sim, decimation=cfg.decimation,
        ankle_substep_stability=cfg.ankle_substep_stability,
        ankle_finetune_reward_weights=cfg.ankle_finetune_reward_weights,
        actions=actions, terminations=termination,
    )
    return impact_v2_configuration_signature(source)


def impact_v2_plus_configuration_signature(cfg):
    """Bind Plus resumes to both physical settings and recovery thresholds."""
    from dataclasses import asdict
    from .ankle import _config_value

    contract = {
        "task": IMPACT_V2_PLUS_TASK,
        "environment": impact_v2_configuration_signature(cfg),
        "recovery": asdict(cfg.recovery),
        "parent_configuration": cfg.impact_v2_parent_configuration_signature,
        "parent_runtime": cfg.impact_v2_parent_runtime_signature,
    }
    return hashlib.sha256(json.dumps(_config_value(contract), sort_keys=True).encode()).hexdigest()


def _validate_v2_source(metadata, cfg, iteration):
    from .ankle import _validate_finetune_metadata

    _validate_finetune_metadata(metadata, SOURCE_V2_TASK)
    if metadata["play_config_signature"] != cfg.impact_v2_source_play_signature:
        raise ValueError("ImpactV2 source PD/reward/physics configuration differs from the frozen CombinedV2 contract.")
    source_contract = {
        "play_config_signature": metadata["play_config_signature"], "lr_scale": 0.5, "warmup_iterations": 1000,
    }
    expected = hashlib.sha256(json.dumps(source_contract, sort_keys=True).encode()).hexdigest()
    if metadata["config_signature"] != expected or metadata["completed_updates"] < 1000:
        raise ValueError("ImpactV2 requires a mature CombinedV2 source with the validated 1000-update warmup contract.")
    if type(iteration) is not int or iteration != metadata["origin_iteration"] + metadata["completed_updates"]:
        raise ValueError("ImpactV2 source iteration does not match its V2 progress.")


def validate_impact_v2_checkpoint(checkpoint, cfg, *, allow_source=False, _expected_configuration_signature=None):
    """Only training may migrate V2; Play and subsequent resumes use ImpactV2."""
    state = checkpoint.get("environment")
    iteration = checkpoint.get("iteration")
    if not isinstance(state, Mapping) or type(iteration) is not int or iteration < 0:
        raise ValueError("ImpactV2 requires a complete checkpoint with environment and iteration.")
    if set(state) == {"ankle_finetune"}:
        if not allow_source:
            raise ValueError("ImpactV2 Play requires its own checkpoint; CombinedV2 is a training source only.")
        _validate_v2_source(state["ankle_finetune"], cfg, iteration)
        return
    if set(state) != {"impact", "finetune", "impact_v2"}:
        raise ValueError("ImpactV2 requires a matching ImpactV2 checkpoint or an explicit CombinedV2 training source.")
    contract = state["impact_v2"]
    if not isinstance(contract, Mapping) or set(contract) != {"version", "task", "config_signature", "source_v2"}:
        raise ValueError("ImpactV2 checkpoint configuration metadata is incomplete.")
    if type(contract["version"]) is not int or contract["version"] != 1 or contract["task"] != IMPACT_V2_TASK:
        raise ValueError("ImpactV2 checkpoint task/version differs.")
    expected_signature = _expected_configuration_signature or impact_v2_configuration_signature(cfg)
    if contract["config_signature"] != expected_signature:
        raise ValueError("ImpactV2 checkpoint PD/reward/physics/recovery configuration differs.")
    source = contract["source_v2"]
    if not isinstance(source, Mapping):
        raise ValueError("ImpactV2 checkpoint is missing its validated V2 source.")
    origin, completed = source.get("origin_iteration"), source.get("completed_updates")
    if type(origin) is not int or type(completed) is not int:
        raise ValueError("ImpactV2 checkpoint V2 source progress must contain integers.")
    _validate_v2_source(source, cfg, origin + completed)
    if not isinstance(state["impact"], Mapping):
        raise ValueError("ImpactV2 checkpoint runtime metadata is incomplete.")
    metadata = _validate_metadata(state["finetune"], state["impact"].get("config_signature"), IMPACT_V2_TASK)
    if iteration != metadata["stage_origin_iteration"] + metadata["completed_updates"]:
        raise ValueError("ImpactV2 checkpoint iteration does not match its stage progress.")
    if metadata["stage_origin_iteration"] != source["origin_iteration"] + source["completed_updates"]:
        raise ValueError("ImpactV2 stage origin differs from its saved V2 source.")
    if state["impact"].get("completed_updates") != metadata["completed_updates"]:
        raise ValueError("ImpactV2 runtime and finetune update counts differ.")


class ImpactV2CusrlAdapter(ImpactCusrlAdapter):
    """Migrate V2 once, then retain independent impact and ankle contracts."""

    task = IMPACT_V2_TASK

    def __init__(self, wrapped, **spec_kwargs):
        self.source_v2_metadata = None
        self.configuration_signature = impact_v2_configuration_signature(wrapped.unwrapped.cfg)
        super().__init__(wrapped, **spec_kwargs)

    def state_dict(self):
        return {
            **super().state_dict(),
            "impact_v2": {
                "version": 1, "task": self.task, "config_signature": self.configuration_signature,
                "source_v2": deepcopy(self.source_v2_metadata),
            },
        }

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, Mapping):
            raise ValueError("ImpactV2 checkpoint environment state must be a mapping.")
        source = state_dict.get("ankle_finetune")
        progress = source if isinstance(source, Mapping) else state_dict.get("finetune", {})
        if not isinstance(progress, Mapping):
            raise ValueError("ImpactV2 checkpoint progress metadata is incomplete.")
        origin_key = "origin_iteration" if source is not None else "stage_origin_iteration"
        origin, completed = progress.get(origin_key), progress.get("completed_updates")
        if type(origin) is not int or type(completed) is not int:
            raise ValueError("ImpactV2 checkpoint progress must contain integer origin and updates.")
        validate_impact_v2_checkpoint(
            {"environment": state_dict, "iteration": origin + completed}, self.unwrapped.cfg, allow_source=True,
        )
        if source is not None:
            self.source_v2_metadata = deepcopy(source)
            return
        super().load_state_dict({name: state_dict[name] for name in ("impact", "finetune")})
        self.source_v2_metadata = deepcopy(state_dict["impact_v2"]["source_v2"])


def _plus_source_info(state, iteration, cfg):
    """Validate one of the two explicit migrations and return its curriculum offset."""
    if set(state) == {"ankle_finetune"}:
        _validate_v2_source(state["ankle_finetune"], cfg, iteration)
        return SOURCE_V2_TASK, 0
    if set(state) == {"impact", "finetune", "impact_v2"}:
        validate_impact_v2_checkpoint(
            {"environment": state, "iteration": iteration}, cfg,
            _expected_configuration_signature=cfg.impact_v2_parent_configuration_signature,
        )
        if state["impact"].get("config_signature") != cfg.impact_v2_parent_runtime_signature:
            raise ValueError("ImpactV2Plus source impact distribution/recovery configuration differs from V2.")
        return IMPACT_V2_TASK, state["finetune"]["completed_updates"]
    raise ValueError("ImpactV2Plus accepts only its own resume, CombinedV2 or ImpactV2 training sources.")


def validate_impact_v2_plus_checkpoint(checkpoint, cfg, *, allow_source=False):
    state, iteration = checkpoint.get("environment"), checkpoint.get("iteration")
    if not isinstance(state, Mapping) or type(iteration) is not int or iteration < 0:
        raise ValueError("ImpactV2Plus requires a complete checkpoint with environment and iteration.")
    if "impact_v2_plus" not in state:
        if not allow_source:
            raise ValueError("ImpactV2Plus Play requires its own checkpoint; sources are training-only.")
        _plus_source_info(state, iteration, cfg)
        return
    if set(state) != {"impact", "finetune", "impact_v2_plus"}:
        raise ValueError("ImpactV2Plus checkpoint environment metadata is incomplete.")
    contract = state["impact_v2_plus"]
    if (
        not isinstance(contract, Mapping)
        or set(contract) != {"version", "task", "config_signature", "source", "curriculum_offset"}
        or type(contract["version"]) is not int or contract["version"] != 1
        or contract["task"] != IMPACT_V2_PLUS_TASK
        or contract["config_signature"] != impact_v2_plus_configuration_signature(cfg)
    ):
        raise ValueError("ImpactV2Plus checkpoint PD/reward/physics/recovery configuration differs.")
    source = contract["source"]
    if not isinstance(source, Mapping) or set(source) != {"task", "iteration", "environment"}:
        raise ValueError("ImpactV2Plus checkpoint source provenance is incomplete.")
    if not isinstance(source["environment"], Mapping) or type(source["iteration"]) is not int:
        raise ValueError("ImpactV2Plus checkpoint source progress is invalid.")
    task, offset = _plus_source_info(source["environment"], source["iteration"], cfg)
    if task != source["task"] or type(contract["curriculum_offset"]) is not int or contract["curriculum_offset"] != offset:
        raise ValueError("ImpactV2Plus checkpoint source task/curriculum offset differs.")
    if not isinstance(state["impact"], Mapping):
        raise ValueError("ImpactV2Plus checkpoint runtime metadata is incomplete.")
    metadata = _validate_metadata(state["finetune"], state["impact"].get("config_signature"), IMPACT_V2_PLUS_TASK)
    if metadata["stage_origin_iteration"] != source["iteration"] or iteration != source["iteration"] + metadata["completed_updates"]:
        raise ValueError("ImpactV2Plus checkpoint iteration does not match its source and stage progress.")
    if state["impact"].get("completed_updates") != offset + metadata["completed_updates"]:
        raise ValueError("ImpactV2Plus runtime and inherited curriculum progress differ.")
    if task == IMPACT_V2_TASK and metadata["lr_caps"] != source["environment"]["finetune"]["lr_caps"]:
        raise ValueError("ImpactV2Plus continuation must retain its ImpactV2 learning-rate caps.")


class ImpactV2PlusCusrlAdapter(ImpactCusrlAdapter):
    """Two explicit source routes, then strict Plus-only resume and Play."""

    task = IMPACT_V2_PLUS_TASK

    def __init__(self, wrapped, **spec_kwargs):
        self.plus_source = None
        self.curriculum_offset = 0
        self.configuration_signature = impact_v2_plus_configuration_signature(wrapped.unwrapped.cfg)
        super().__init__(wrapped, **spec_kwargs)

    def state_dict(self):
        return {
            **super().state_dict(),
            "impact_v2_plus": {
                "version": 1, "task": self.task, "config_signature": self.configuration_signature,
                "source": deepcopy(self.plus_source), "curriculum_offset": self.curriculum_offset,
            },
        }

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, Mapping):
            raise ValueError("ImpactV2Plus checkpoint environment state must be a mapping.")
        source = state_dict.get("ankle_finetune")
        progress = source if isinstance(source, Mapping) else state_dict.get("finetune", {})
        if not isinstance(progress, Mapping):
            raise ValueError("ImpactV2Plus checkpoint progress metadata is incomplete.")
        key = "origin_iteration" if source is not None else "stage_origin_iteration"
        origin, completed = progress.get(key), progress.get("completed_updates")
        if type(origin) is not int or type(completed) is not int:
            raise ValueError("ImpactV2Plus checkpoint progress must contain integer origin and updates.")
        iteration = origin + completed
        cfg, runtime = self.unwrapped.cfg, self.impact_runtime
        validate_impact_v2_plus_checkpoint({"environment": state_dict, "iteration": iteration}, cfg, allow_source=True)
        if "impact_v2_plus" in state_dict:
            contract = state_dict["impact_v2_plus"]
            runtime.load_state_dict(state_dict["impact"])
            metadata = _validate_metadata(state_dict["finetune"], runtime.config_signature, self.task)
            self.plus_source = deepcopy(contract["source"])
            self.curriculum_offset = contract["curriculum_offset"]
            self.finetune_metadata = metadata
            return
        task, offset = _plus_source_info(state_dict, iteration, cfg)
        if task == IMPACT_V2_TASK:
            # Configuration was strictly checked against the frozen V2 parent.
            # Only the new recovery semantics change; preserve generator and
            # already learned impact/head curriculum progress across migration.
            migrated = deepcopy(state_dict["impact"])
            migrated["config_signature"] = runtime.config_signature
            runtime.load_state_dict(migrated)
        self.plus_source = {"task": task, "iteration": iteration, "environment": deepcopy(state_dict)}
        self.curriculum_offset = offset
        self.finetune_metadata = None


@contextmanager
def impact_export_masks(agent, head_position: tuple[float, float], *, task: str = IMPACT_TASK):
    """Include head isolation in CusRL's ONNX and both JIT export graphs."""
    import torch
    from t800_lab.task_names import impact_parent_task

    modules = {
        IMPACT_V2_TASK: "t800_lab.tasks.t800_impact_v2.mdp",
        IMPACT_V2_PLUS_TASK: "t800_lab.tasks.t800_impact_v2_plus.mdp",
    }
    module = import_module(modules.get(impact_parent_task(task), "t800_lab.tasks.t800_impact.head_mdp"))
    HeadActionMask, HeadObservationMask = module.HeadActionMask, module.HeadObservationMask

    if agent.environment_spec.observation_normalization is not None:
        raise ValueError("Impact export expects the original hook-based observation normalization.")
    original_pre = agent.hook.pre_export
    original_post = agent.hook.post_export

    def pre_export(graph):
        original_pre(graph)
        graph.add_node(
            HeadObservationMask(head_position).to(agent.device), module_name="impact_head_observation_mask",
            input_names={"observation": "observation"}, output_names="observation", expose_outputs=False,
            prepend=True,
        )

    def post_export(graph):
        original_post(graph)
        graph.add_node(
            HeadActionMask().to(agent.device), module_name="impact_head_action_mask",
            input_names={"action": "action"}, output_names="action", expose_outputs=False,
        )

    agent.hook.pre_export = pre_export
    agent.hook.post_export = post_export
    try:
        # CusRL initializes recurrent memories by forwarding the actor during
        # export. They must be inference buffers, without an autograd graph.
        with torch.no_grad():
            yield
    finally:
        agent.hook.pre_export = original_pre
        agent.hook.post_export = original_post


class ImpactFinetuneCallback:
    """Initialize once before rollout, then advance the stage after each update.

    CusRL invokes callbacks once at registration, then after update and before
    incrementing trainer.iteration. Keep that offset explicit to prevent a
    checkpoint resume from repeating the head/impact warmup.
    """

    def __init__(self):
        self.initialized = False

    def __call__(self, trainer) -> None:
        environment = trainer.environment
        runtime = environment.impact_runtime
        groups = trainer.agent.optimizer.param_groups
        if not self.initialized:
            metadata = environment.finetune_metadata
            if metadata is None:
                if environment.task == IMPACT_V2_TASK:
                    source = environment.source_v2_metadata
                    if source is None:
                        raise ValueError("ImpactV2 must load a validated CombinedV2 checkpoint before training.")
                    if trainer.iteration != source["origin_iteration"] + source["completed_updates"]:
                        raise ValueError("ImpactV2 source iteration does not match the restored trainer.")
                source_caps = None
                if environment.task == IMPACT_V2_PLUS_TASK:
                    source = environment.plus_source
                    if source is None or trainer.iteration != source["iteration"]:
                        raise ValueError("ImpactV2Plus must first restore one validated source checkpoint.")
                    if source["task"] == IMPACT_V2_TASK:
                        source_caps = source["environment"]["finetune"]["lr_caps"]
                metadata = {
                    "config_version": FINETUNE_CONFIG_VERSION,
                    "task": environment.task,
                    "stage_origin_iteration": int(trainer.iteration),
                    "completed_updates": 0,
                    "lr_caps": deepcopy(source_caps) if source_caps is not None else [float(group["lr"]) * 0.5 for group in groups],
                    "config_signature": runtime.state_dict()["config_signature"],
                    "config_migrations": [],
                }
            else:
                metadata = _validate_metadata(metadata, runtime.state_dict()["config_signature"], environment.task)
            if trainer.iteration != metadata["stage_origin_iteration"] + metadata["completed_updates"]:
                raise ValueError("Impact checkpoint iteration does not match stage progress.")
            if len(metadata["lr_caps"]) != len(groups):
                raise ValueError("Impact checkpoint learning-rate cap group count differs.")
            environment.finetune_metadata = metadata
            self.initialized = True
        else:
            metadata = environment.finetune_metadata
            if trainer.iteration != metadata["stage_origin_iteration"] + metadata["completed_updates"]:
                raise ValueError("Unexpected CusRL callback iteration; refusing to corrupt stage progress.")
            metadata["completed_updates"] += 1

        self._cap_learning_rate(trainer.agent, metadata["lr_caps"])
        runtime.set_update(getattr(environment, "curriculum_offset", 0) + metadata["completed_updates"])

    _cap_learning_rate = staticmethod(cap_learning_rate)
