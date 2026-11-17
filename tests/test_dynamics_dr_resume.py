"""CPU-only contract tests for the strict dynamics-DR continuation stage."""

from copy import deepcopy
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = ROOT / "source/t800_lab/t800_lab/utils/wrappers/cusrl"


@pytest.fixture(scope="module")
def dr_module():
    source_root = str(ROOT / "source/t800_lab")
    sys.path.insert(0, source_root)
    package_name = "_dynamics_dr_test"
    package = ModuleType(package_name)
    package.__path__ = [str(WRAPPERS)]
    sys.modules[package_name] = package
    base = ModuleType(f"{package_name}.vecenv_wrapper")
    base.CusrlT800Adapter = object
    sys.modules[base.__name__] = base
    for name in ("checkpoint", "finetune", "impact", "dynamics_dr"):
        spec = importlib.util.spec_from_file_location(f"{package_name}.{name}", WRAPPERS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    yield sys.modules[f"{package_name}.dynamics_dr"]
    for name in list(sys.modules):
        if name == package_name or name.startswith(package_name + "."):
            del sys.modules[name]
    sys.path.remove(source_root)


@dataclass(frozen=True)
class Ranges:
    joint_offset_range: tuple[float, float] = (-.01, .01)
    mass_scale_range: tuple[float, float] = (.9, 1.1)
    torso_mass_delta_range: tuple[float, float] = (-2., 2.)
    torso_com_offset_range: tuple[float, float, float] = (.025, .05, .05)


def config(parent_task="T800_base-ImpactV2", iteration=154500):
    parent = SimpleNamespace(marker="parent")
    return SimpleNamespace(
        run_name="T800_base-ImpactV2-From154500-DR", dr_parent_task=parent_task,
        dr_source_iteration=iteration, dr_parent_cfg=parent, dynamics_randomization=Ranges(),
    )


def source_environment(task, *, completed=14500):
    return {
        "impact": {"config_signature": "runtime", "completed_updates": completed},
        "finetune": {
            "config_version": 2, "task": task, "stage_origin_iteration": 140000,
            "completed_updates": completed, "lr_caps": [.00002], "config_signature": "runtime",
            "config_migrations": [],
        },
        "impact_v2": {"source": "fixture"} if task.endswith("ImpactV2") else None,
    }


def patch_parent_validation(monkeypatch, module):
    monkeypatch.setattr(module, "validate_impact_v2_checkpoint", lambda checkpoint, cfg, **kwargs: None)
    monkeypatch.setattr(module, "validate_impact_v2_plus_checkpoint", lambda checkpoint, cfg, **kwargs: None)
    monkeypatch.setattr(module, "_config_signature", lambda cfg, task: "parent-sig")


def make_dr_checkpoint(module, cfg, *, completed=0):
    source = source_environment(cfg.dr_parent_task, completed=14500 if cfg.dr_parent_task.endswith("ImpactV2") else 33000)
    source["finetune"]["task"] = cfg.dr_parent_task
    source["finetune"]["stage_origin_iteration"] = 140000
    source["impact_v2_plus"] = {"fixture": True} if cfg.dr_parent_task.endswith("Plus") else None
    source["impact_v2"] = {"fixture": True} if cfg.dr_parent_task.endswith("ImpactV2") else None
    offset = source["impact"]["completed_updates"]
    # The validator is monkeypatched above; this fixture focuses on DR metadata.
    dr = {
        "version": 1, "task": cfg.run_name, "parent_task": cfg.dr_parent_task,
        "source_task": cfg.dr_parent_task, "source_iteration": cfg.dr_source_iteration,
        "source_environment": deepcopy(source), "parent_configuration_signature": "parent-sig",
        "dr_environment_signature": "parent-sig", "run_name": cfg.run_name,
        "dr_configuration_signature": module.dynamics_dr_configuration_signature(cfg),
        "curriculum_offset": offset, "stage_origin_iteration": cfg.dr_source_iteration,
        "completed_updates": completed,
    }
    runtime_done = offset + completed
    metadata = {
        "config_version": 2, "task": cfg.run_name, "stage_origin_iteration": cfg.dr_source_iteration,
        "completed_updates": completed, "lr_caps": [.00002], "config_signature": "runtime",
        "config_migrations": [],
    }
    return {"environment": {"impact": {"config_signature": "runtime", "completed_updates": runtime_done},
                             "finetune": metadata, "dynamics_dr": dr},
            "iteration": cfg.dr_source_iteration + completed}


def test_source_requires_exact_iteration_and_parent_validator(dr_module, monkeypatch):
    module = dr_module
    cfg = config()
    patch_parent_validation(monkeypatch, module)
    source = {"environment": source_environment(cfg.dr_parent_task), "iteration": cfg.dr_source_iteration}
    module.validate_dynamics_dr_checkpoint(source, cfg, allow_source=True)
    source["iteration"] += 1
    with pytest.raises(ValueError, match="iteration"):
        module.validate_dynamics_dr_checkpoint(source, cfg, allow_source=True)


def test_bad_randomization_ranges_are_rejected(dr_module):
    cfg = config()
    cfg.dynamics_randomization = SimpleNamespace(joint_offset_range=(float("nan"), .01))
    with pytest.raises(ValueError):
        dr_module.dynamics_dr_configuration_signature(cfg)


def test_amp_global_batch_preserved_across_gpu_counts(dr_module):
    hook = SimpleNamespace(name="scaled_adversarial_motion_prior", batch_size=512)
    hook.update_attribute = lambda name, value: setattr(hook, name, value)
    agent = SimpleNamespace(hook=[hook])
    assert dr_module.configure_dynamics_dr_amp_batch(agent, 1) == 1024
    assert hook.batch_size == 1024
    assert dr_module.configure_dynamics_dr_amp_batch(agent, 2) == 512
    assert hook.batch_size == 512
    with pytest.raises(ValueError, match="world size"):
        dr_module.configure_dynamics_dr_amp_batch(agent, 3)
    with pytest.raises(ValueError, match="exactly one"):
        dr_module.configure_dynamics_dr_amp_batch(SimpleNamespace(hook=[]), 1)


def test_checkpoint_preserves_offset_and_metadata(dr_module, monkeypatch):
    module = dr_module
    cfg = config()
    patch_parent_validation(monkeypatch, module)
    checkpoint = make_dr_checkpoint(module, cfg, completed=7)
    module.validate_dynamics_dr_checkpoint(checkpoint, cfg)
    assert checkpoint["environment"]["dynamics_dr"]["curriculum_offset"] == 14500
    assert checkpoint["environment"]["dynamics_dr"]["stage_origin_iteration"] == 154500


def test_callback_does_not_warmup_or_halve_learning_rate(dr_module, monkeypatch):
    module = dr_module
    cfg = config()
    patch_parent_validation(monkeypatch, module)
    source = source_environment(cfg.dr_parent_task)
    runtime = SimpleNamespace(
        state_dict=lambda: {"config_signature": "runtime"},
        set_update=lambda update: setattr(runtime, "last_update", update),
    )
    env = SimpleNamespace(
        task=cfg.run_name, impact_runtime=runtime, finetune_metadata=None,
        dr_source_environment=source, dr_source_iteration=154500, curriculum_offset=14500,
    )
    parameter = torch.nn.Parameter(torch.ones(1))
    scheduler = SimpleNamespace(name="adaptive_lr_schedule", _lr_scale=.001, scale_all_params=True)
    agent = SimpleNamespace(optimizer=torch.optim.SGD([parameter], lr=.001), hook=[scheduler])
    trainer = SimpleNamespace(environment=env, agent=agent, iteration=154500)
    callback = module.DynamicsDRFinetuneCallback()
    callback(trainer)
    assert agent.optimizer.param_groups[0]["lr"] == pytest.approx(.00002)
    assert runtime.last_update == 14500
    # CusRL invokes the post-update callback before incrementing iteration.
    callback(trainer)
    assert runtime.last_update == 14501
    assert agent.optimizer.param_groups[0]["lr"] == pytest.approx(.00002)
    trainer.iteration += 1
    callback(trainer)
    assert runtime.last_update == 14502
