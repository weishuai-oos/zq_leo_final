"""CPU-only checks for strict restoration and durable stage-two update state."""

from copy import deepcopy
from dataclasses import dataclass
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = ROOT / "source/t800_lab/t800_lab/utils/wrappers/cusrl"


@pytest.fixture(scope="module")
def modules():
    source_root = str(ROOT / "source/t800_lab")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    package_name = "_impact_resume_test"
    package = ModuleType(package_name)
    package.__path__ = [str(WRAPPERS)]
    sys.modules[package_name] = package
    base = ModuleType(f"{package_name}.vecenv_wrapper")
    base.CusrlT800Adapter = object
    sys.modules[base.__name__] = base
    spec = importlib.util.spec_from_file_location(f"{package_name}.impact", WRAPPERS / "impact.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    checkpoint = sys.modules[f"{package_name}.checkpoint"]
    yield module, checkpoint
    for name in list(sys.modules):
        if name == package_name or name.startswith(package_name + "."):
            del sys.modules[name]
    if source_root in sys.path:
        sys.path.remove(source_root)


class FakeAgent:
    OPTIMIZERS = ["optimizer"]

    def __init__(self):
        self.actor = torch.nn.Linear(4, 2)
        self.critic = torch.nn.Linear(5, 1)
        self.discriminator = torch.nn.Linear(6, 1)
        self.scheduler = SimpleNamespace(name="adaptive_lr_schedule", _lr_scale=0.031, scale_all_params=False)
        self.hook = [self.scheduler]
        parameters = [
            (f"{name}.{key}", parameter)
            for name in ("actor", "critic", "discriminator")
            for key, parameter in getattr(self, name).named_parameters()
        ]
        self.optimizer = torch.optim.AdamW([{
            "params": [parameter for _, parameter in parameters],
            "param_names": [name for name, _ in parameters], "lr": 4e-5,
        }])
        sum(parameter.square().sum() for _, parameter in parameters).backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.rms = torch.arange(4, dtype=torch.float32)

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "hook": {
                "observation_normalization": {"mean": self.rms.clone()},
                "scaled_adversarial_motion_prior": self.discriminator.state_dict(),
                "adaptive_lr_schedule": {"lr_scale": self.scheduler._lr_scale},
            },
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.discriminator.load_state_dict(state["hook"]["scaled_adversarial_motion_prior"])
        self.rms.copy_(state["hook"]["observation_normalization"]["mean"])
        self.scheduler._lr_scale = state["hook"]["adaptive_lr_schedule"]["lr_scale"]
        self.optimizer.load_state_dict(state["optimizer"])


class FakeRuntime:
    def __init__(self):
        self.completed_updates = 0

    def state_dict(self):
        return {"config_signature": "fixture-v1", "version": 1, "completed_updates": self.completed_updates}

    def set_update(self, update):
        self.completed_updates = update

    def load_state_dict(self, state):
        assert state["config_signature"] == "fixture-v1"
        self.completed_updates = state["completed_updates"]

    def validate_checkpoint_signature(self, signature):
        if signature != "fixture-v1":
            raise ValueError("Impact training configuration mismatch")
        return False


def make_adapter(module):
    adapter = module.ImpactCusrlAdapter.__new__(module.ImpactCusrlAdapter)
    adapter.unwrapped = SimpleNamespace(impact_runtime=FakeRuntime(), head_marker=object())
    adapter.finetune_metadata = None
    return adapter


def test_complete_agent_restores_exactly(modules):
    _, checkpoint = modules
    saved = deepcopy(FakeAgent().state_dict())
    restored = FakeAgent()
    checkpoint.load_agent_checkpoint_strict(saved, restored)
    checkpoint.verify_agent_checkpoint_restored(saved, restored)


def test_known_ddp_wrapper_names_preserve_strict_parameter_order(modules):
    _, checkpoint = modules
    saved = deepcopy(FakeAgent().state_dict())
    group = saved["optimizer"]["param_groups"][0]
    group["param_names"] = [name.replace(".", ".module.", 1) for name in group["param_names"]]
    restored = FakeAgent()
    checkpoint.load_agent_checkpoint_strict(saved, restored)
    checkpoint.verify_agent_checkpoint_restored(saved, restored)
    group["param_names"].reverse()
    with pytest.raises(ValueError, match="parameter names/order"):
        checkpoint.validate_agent_checkpoint_compatibility(saved, restored)


@pytest.mark.parametrize("damage", ["module", "hook", "shape", "moment", "moment_shape", "parameter_order"])
def test_incomplete_or_incompatible_checkpoint_is_rejected(modules, damage):
    _, checkpoint = modules
    agent = FakeAgent()
    saved = deepcopy(agent.state_dict())
    if damage == "module":
        del saved["critic"]
    elif damage == "hook":
        del saved["hook"]["observation_normalization"]
    elif damage == "shape":
        saved["actor"]["weight"] = torch.zeros(2, 7)
    elif damage == "moment":
        del saved["optimizer"]["state"][0]["exp_avg"]
    elif damage == "moment_shape":
        saved["optimizer"]["state"][0]["exp_avg"] = torch.zeros(1)
    else:
        saved["optimizer"]["param_groups"][0]["param_names"].reverse()
    with pytest.raises(ValueError, match="Checkpoint"):
        checkpoint.validate_agent_checkpoint_compatibility(saved, agent)


def test_permissive_loader_or_mutated_restore_cannot_pass(modules):
    _, checkpoint = modules
    agent = FakeAgent()
    saved = deepcopy(agent.state_dict())
    with torch.no_grad():
        agent.actor.weight[0, 0] += 1
    with pytest.raises(ValueError, match="not restored exactly"):
        checkpoint.verify_agent_checkpoint_restored(saved, agent)


def test_empty_stage_one_environment_does_not_mutate_runtime_or_head(modules):
    impact, _ = modules
    adapter = make_adapter(impact)
    adapter.impact_runtime.set_update(13)
    head = adapter.unwrapped.head_marker
    adapter.load_state_dict({})
    assert adapter.impact_runtime.completed_updates == 13
    assert adapter.unwrapped.head_marker is head
    assert adapter.finetune_metadata is None


def test_stage_resume_preserves_origin_warmup_and_lr_cap(modules):
    impact, checkpoint = modules
    adapter = make_adapter(impact)
    agent = FakeAgent()
    trainer = SimpleNamespace(environment=adapter, agent=agent, iteration=26500)
    callback = impact.ImpactFinetuneCallback()
    callback(trainer)  # The immediate register_callback invocation, before rollout.
    cap = 2e-5
    assert agent.optimizer.param_groups[0]["lr"] == pytest.approx(cap)
    assert agent.scheduler._lr_scale == pytest.approx(0.031 * 0.5)
    assert adapter.impact_runtime.completed_updates == 0

    for _ in range(7):
        agent.optimizer.param_groups[0]["lr"] *= 1.5
        agent.scheduler._lr_scale *= 1.5
        callback(trainer)  # CusRL invokes callbacks before incrementing iteration.
        trainer.iteration += 1
    saved_environment = deepcopy(adapter.state_dict())
    saved_agent = deepcopy(agent.state_dict())
    assert adapter.impact_runtime.completed_updates == 7
    assert agent.optimizer.param_groups[0]["lr"] == pytest.approx(cap)

    restored = make_adapter(impact)
    restored.load_state_dict(saved_environment)
    restored_agent = FakeAgent()
    checkpoint.load_agent_checkpoint_strict(saved_agent, restored_agent)
    resumed = SimpleNamespace(environment=restored, agent=restored_agent, iteration=26507)
    resume_callback = impact.ImpactFinetuneCallback()
    resume_callback(resumed)
    assert restored.finetune_metadata["stage_origin_iteration"] == 26500
    assert restored.impact_runtime.completed_updates == 7
    assert restored_agent.optimizer.param_groups[0]["lr"] == pytest.approx(cap)
    assert restored_agent.scheduler._lr_scale == pytest.approx(agent.scheduler._lr_scale)
    resume_callback(resumed)
    assert restored.impact_runtime.completed_updates == 8


def test_resume_preserves_an_lr_below_cap(modules):
    impact, _ = modules
    adapter = make_adapter(impact)
    trainer = SimpleNamespace(environment=adapter, agent=FakeAgent(), iteration=10)
    impact.ImpactFinetuneCallback()(trainer)
    trainer.agent.optimizer.param_groups[0]["lr"] = 1e-5
    impact.ImpactFinetuneCallback()(trainer)
    assert trainer.agent.optimizer.param_groups[0]["lr"] == 1e-5


def test_inconsistent_stage_progress_is_rejected(modules):
    impact, _ = modules
    adapter = make_adapter(impact)
    trainer = SimpleNamespace(environment=adapter, agent=FakeAgent(), iteration=10)
    impact.ImpactFinetuneCallback()(trainer)
    trainer.iteration = 11
    with pytest.raises(ValueError, match="iteration"):
        impact.ImpactFinetuneCallback()(trainer)
    state = adapter.state_dict()
    state["impact"]["completed_updates"] += 1
    with pytest.raises(ValueError, match="update counts"):
        make_adapter(impact).load_state_dict(state)


def test_impact_iteration_budget_retains_absolute_semantics(modules):
    impact, _ = modules
    resolve = impact.resolve_impact_iteration_budget
    assert resolve(85500, max_iterations=None, additional_iterations=None) == 125500
    assert resolve(30000, max_iterations=None, additional_iterations=1000) == 31000
    assert resolve(26500, max_iterations=30000, additional_iterations=None) == 30000
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve(26500, max_iterations=30000, additional_iterations=1000)
    with pytest.raises(ValueError, match="must exceed"):
        resolve(26500, max_iterations=20000, additional_iterations=None)


def _mature_combined_v2_source():
    """Build the frozen, completed CombinedV2 source metadata contract."""
    play_signature = "bd85bb8e6735e14844b9074f68cffc504cea28421fc0ea63b4f160e6a9ffeb9d"
    config_signature = hashlib.sha256(json.dumps({
        "play_config_signature": play_signature, "lr_scale": 0.5, "warmup_iterations": 1000,
    }, sort_keys=True).encode()).hexdigest()
    return {
        "version": 1,
        "task": "T800_base-AnkleCombinedV2",
        "config_signature": config_signature,
        "play_config_signature": play_signature,
        "origin_iteration": 100000,
        "completed_updates": 40000,
        "lr_caps": [2.0e-5],
    }


def test_impact_v2_accepts_only_a_mature_combined_v2_source(modules):
    impact, _ = modules
    source = _mature_combined_v2_source()
    cfg = SimpleNamespace(impact_v2_source_play_signature=source["play_config_signature"])
    checkpoint = {"environment": {"ankle_finetune": source}, "iteration": 140000}
    impact.validate_impact_v2_checkpoint(checkpoint, cfg, allow_source=True)
    with pytest.raises(ValueError, match="training source"):
        impact.validate_impact_v2_checkpoint(checkpoint, cfg, allow_source=False)


@pytest.mark.parametrize("mutation,match", [
    (lambda state: state.update(play_config_signature="0" * 64), "PD/reward/physics"),
    (lambda state: state.update(completed_updates=999), "mature"),
    (lambda state: state.update(origin_iteration=100001), "iteration"),
    (lambda state: state.update(task="T800_base-AnkleCombined"), "task/version"),
])
def test_impact_v2_rejects_mismatched_or_immature_source_metadata(modules, mutation, match):
    impact, _ = modules
    source = _mature_combined_v2_source()
    mutation(source)
    cfg = SimpleNamespace(impact_v2_source_play_signature=
                          "bd85bb8e6735e14844b9074f68cffc504cea28421fc0ea63b4f160e6a9ffeb9d")
    checkpoint = {"environment": {"ankle_finetune": source}, "iteration": 140000}
    with pytest.raises(ValueError, match=match):
        impact.validate_impact_v2_checkpoint(checkpoint, cfg, allow_source=True)


def test_impact_v2_rejects_old_impact_or_partial_environment_state(modules):
    impact, _ = modules
    cfg = SimpleNamespace(impact_v2_source_play_signature="bd85bb8e6735e14844b9074f68cffc504cea28421fc0ea63b4f160e6a9ffeb9d")
    for state in ({}, {"impact": {}, "finetune": {}}, {"ankle_combined": {}}):
        with pytest.raises(ValueError, match="ImpactV2"):
            impact.validate_impact_v2_checkpoint({"environment": state, "iteration": 140000}, cfg)


def make_impact_v2_adapter(module):
    def term(weight):
        return SimpleNamespace(func=FakeAgent, weight=weight, params={})

    cfg = SimpleNamespace(
        impact_v2_source_play_signature=_mature_combined_v2_source()["play_config_signature"],
        rewards=SimpleNamespace(ankle_physics_acceleration_l2=term(-0.02), action_rate_l2=term(-0.0045)),
        ankle_finetune_reward_weights={"ankle_physics_acceleration_l2": (0.0, -0.02)},
        ankle_substep_stability={"acceleration_deadband_rad_s2": 100.0},
        sim=SimpleNamespace(dt=0.005), decimation=4,
        scene=SimpleNamespace(robot=SimpleNamespace(actuators={
            "feet": SimpleNamespace(stiffness={"ankle": 40.0}, damping={"ankle": 2.0}),
        })),
        events=SimpleNamespace(
            collider_offsets=SimpleNamespace(func=FakeAgent, mode="startup", params={"contact_offset": 0.002}),
            physics_material=SimpleNamespace(params={"restitution_range": (0.0, 0.10)}),
        ),
        terminations=SimpleNamespace(base_contact=term(0.0)),
        actions=SimpleNamespace(joint_pos=SimpleNamespace(class_type=FakeAgent)),
    )
    adapter = module.ImpactV2CusrlAdapter.__new__(module.ImpactV2CusrlAdapter)
    adapter.unwrapped = SimpleNamespace(cfg=cfg, impact_runtime=FakeRuntime())
    adapter.source_v2_metadata = adapter.finetune_metadata = None
    adapter.configuration_signature = module.impact_v2_configuration_signature(cfg)
    return adapter


def impact_v2_initial_state(module):
    adapter = make_impact_v2_adapter(module)
    adapter.load_state_dict({"ankle_finetune": _mature_combined_v2_source()})
    agent = FakeAgent()
    agent.optimizer.param_groups[0]["lr"] = 1e-5
    trainer = SimpleNamespace(environment=adapter, agent=agent, iteration=140000)
    callback = module.ImpactFinetuneCallback()
    callback(trainer)
    return adapter, agent, trainer, callback


def test_impact_v2_migration_and_own_resume_keep_progress_optimizer_and_lr(modules):
    module, checkpoint = modules
    adapter, agent, trainer, callback = impact_v2_initial_state(module)
    assert adapter.finetune_metadata["stage_origin_iteration"] == 140000
    assert adapter.finetune_metadata["task"] == "T800_base-ImpactV2"
    assert adapter.impact_runtime.completed_updates == 0
    assert adapter.unwrapped.cfg.rewards.ankle_physics_acceleration_l2.weight == -0.02
    for _ in range(3):
        callback(trainer)
        trainer.iteration += 1
    saved_environment, saved_agent = deepcopy(adapter.state_dict()), deepcopy(agent.state_dict())
    module.validate_impact_v2_checkpoint(
        {"iteration": 140003, "environment": saved_environment}, adapter.unwrapped.cfg,
    )
    restored = make_impact_v2_adapter(module)
    restored.load_state_dict(saved_environment)
    restored_agent = FakeAgent()
    checkpoint.load_agent_checkpoint_strict(saved_agent, restored_agent)
    resumed = SimpleNamespace(environment=restored, agent=restored_agent, iteration=140003)
    module.ImpactFinetuneCallback()(resumed)
    checkpoint.verify_agent_checkpoint_restored(saved_agent, restored_agent)
    assert restored.finetune_metadata["stage_origin_iteration"] == 140000
    assert restored.finetune_metadata["completed_updates"] == 3
    assert restored.source_v2_metadata == _mature_combined_v2_source()
    assert restored.impact_runtime.completed_updates == 3
    assert restored_agent.optimizer.param_groups[0]["lr"] == pytest.approx(5e-6)


@pytest.mark.parametrize("change", ["pd", "contact", "substep", "recovery", "origin", "iteration"])
def test_impact_v2_own_resume_rejects_changed_configuration_or_progress(modules, change):
    module, _ = modules
    adapter, _, _, _ = impact_v2_initial_state(module)
    saved = {"environment": deepcopy(adapter.state_dict()), "iteration": 140000}
    cfg = adapter.unwrapped.cfg
    if change == "pd":
        cfg.scene.robot.actuators["feet"].damping["ankle"] = 0.3
    elif change == "contact":
        cfg.events.collider_offsets.params["contact_offset"] = 0.014
    elif change == "substep":
        cfg.sim.dt = 0.01
    elif change == "recovery":
        cfg.rewards.ankle_physics_acceleration_l2.params["recovery_scale"] = 0.1
    elif change == "origin":
        saved["environment"]["finetune"]["stage_origin_iteration"] += 1
    else:
        saved["iteration"] += 1
    with pytest.raises(ValueError, match="configuration differs|iteration|origin"):
        module.validate_impact_v2_checkpoint(saved, cfg)


def make_real_adapter(module, *, mode="train", legacy=False, **overrides):
    from test_impact_runtime import C, M, _env

    settings = dict(mode=mode)
    if legacy:
        settings.update(stress_combo_probability=0.70, stress_combo_max_hits=5)
    settings.update(overrides)
    adapter = module.ImpactCusrlAdapter.__new__(module.ImpactCusrlAdapter)
    adapter.unwrapped = SimpleNamespace(impact_runtime=M.ImpactRuntime(_env(2), C.ImpactCfg(**settings)))
    adapter.finetune_metadata = None
    return adapter


def old_combo_checkpoint(module):
    old = make_real_adapter(module, legacy=True)
    agent = FakeAgent()
    module.ImpactFinetuneCallback()(SimpleNamespace(environment=old, agent=agent, iteration=85500))
    old.impact_runtime.set_update(29500)
    old.finetune_metadata["completed_updates"] = 29500
    state = deepcopy(old.state_dict())
    # Reproduce the original on-disk version-1 metadata contract exactly.
    state["finetune"]["config_version"] = 1
    del state["finetune"]["config_migrations"]
    return state, deepcopy(agent.state_dict())


def test_explicit_stress_migration_preserves_training_progress_lr_and_agent_then_resumes_normally(modules):
    impact, checkpoint = modules
    state, saved_agent = old_combo_checkpoint(impact)
    restored = make_real_adapter(impact)
    with pytest.raises(ValueError, match="impact-single-hit-migration"):
        restored.load_state_dict(state)
    assert restored.finetune_metadata is None
    assert restored.impact_runtime.completed_updates == 0

    restored.impact_runtime.allow_legacy_stress_combo_migration = True
    restored.load_state_dict(state)
    agent = FakeAgent()
    checkpoint.load_agent_checkpoint_strict(saved_agent, agent)
    trainer = SimpleNamespace(environment=restored, agent=agent, iteration=115000)
    callback = impact.ImpactFinetuneCallback()
    callback(trainer)
    checkpoint.verify_agent_checkpoint_restored(saved_agent, agent)
    metadata = restored.finetune_metadata
    assert metadata["config_version"] == 2
    assert metadata["stage_origin_iteration"] == 85500
    assert metadata["completed_updates"] == 29500
    assert metadata["lr_caps"] == state["finetune"]["lr_caps"]
    assert metadata["config_migrations"] == [{
        "change": "stress_single_hit",
        "from_signature": state["impact"]["config_signature"],
        "to_signature": restored.impact_runtime.config_signature,
        "completed_updates": 29500,
    }]
    assert torch.equal(
        restored.impact_runtime.state_dict()["generator_state"], state["impact"]["generator_state"]
    )
    assert state["finetune"]["config_version"] == 1
    assert "config_migrations" not in state["finetune"]  # Caller state is not rewritten.

    callback(trainer)
    trainer.iteration += 1
    new_state = deepcopy(restored.state_dict())
    again = make_real_adapter(impact)
    again.load_state_dict(new_state)  # No migration flag needed for a new checkpoint.
    assert again.finetune_metadata == restored.finetune_metadata
    assert again.finetune_metadata["completed_updates"] == 29501
    assert again.impact_runtime.config_signature == new_state["impact"]["config_signature"]


@pytest.mark.parametrize("mode", ["off", "common", "stress"])
def test_legacy_policy_evaluation_automatically_records_only_known_stress_load_change(modules, mode):
    impact, _ = modules
    state, _ = old_combo_checkpoint(impact)
    evaluation = make_real_adapter(impact, mode=mode)
    evaluation.load_state_dict(state)
    assert evaluation.impact_runtime.completed_updates == 29500
    assert evaluation.finetune_metadata["config_signature"] == evaluation.impact_runtime.config_signature
    assert len(evaluation.finetune_metadata["config_migrations"]) == 1
    assert not evaluation.impact_runtime.allow_legacy_stress_combo_migration


@pytest.mark.parametrize("overrides", [dict(ramp_updates=6000), dict(stress_combo_probability=0.50)])
def test_migration_flag_does_not_allow_unrelated_configuration_changes(modules, overrides):
    impact, _ = modules
    old = make_real_adapter(impact, legacy=True, **overrides)
    impact.ImpactFinetuneCallback()(
        SimpleNamespace(environment=old, agent=FakeAgent(), iteration=85500)
    )
    current = make_real_adapter(impact)
    current.impact_runtime.allow_legacy_stress_combo_migration = True
    with pytest.raises(ValueError, match="configuration mismatch"):
        current.load_state_dict(old.state_dict())


@pytest.mark.parametrize("damage", ["unknown_key", "future_update", "signature_chain", "negative_update"])
def test_invalid_migration_lineage_is_rejected(modules, damage):
    impact, _ = modules
    state, _ = old_combo_checkpoint(impact)
    current = make_real_adapter(impact, mode="stress")
    current.load_state_dict(state)
    migrated = current.state_dict()
    entry = migrated["finetune"]["config_migrations"][0]
    if damage == "unknown_key":
        entry["extra"] = True
    elif damage == "future_update":
        entry["completed_updates"] += 1
    elif damage == "signature_chain":
        entry["to_signature"] = "0" * 64
    else:
        entry["completed_updates"] = -1
    with pytest.raises(ValueError, match="migration history"):
        make_real_adapter(impact).load_state_dict(migrated)


@dataclass
class _PlusRecoveryFixture:
    max_s: float = 2.5


class PlusFakeRuntime(FakeRuntime):
    config_signature = "fixture-plus-v1"

    def state_dict(self):
        return {"config_signature": self.config_signature, "version": 1, "completed_updates": self.completed_updates}

    def load_state_dict(self, state):
        if state["config_signature"] != self.config_signature:
            raise ValueError("Plus runtime configuration mismatch")
        self.completed_updates = state["completed_updates"]


def make_plus_adapter(module):
    cfg = make_impact_v2_adapter(module).unwrapped.cfg
    cfg.impact_v2_parent_configuration_signature = module.impact_v2_configuration_signature(cfg)
    cfg.impact_v2_parent_runtime_signature = "fixture-v1"
    cfg.recovery = _PlusRecoveryFixture()
    cfg.rewards.action_rate_l2.params["plus_min_scale"] = 0.5
    adapter = module.ImpactV2PlusCusrlAdapter.__new__(module.ImpactV2PlusCusrlAdapter)
    adapter.unwrapped = SimpleNamespace(cfg=cfg, impact_runtime=PlusFakeRuntime())
    adapter.finetune_metadata = adapter.plus_source = None
    adapter.curriculum_offset = 0
    adapter.configuration_signature = module.impact_v2_plus_configuration_signature(cfg)
    return adapter


@pytest.mark.parametrize("from_impact", [False, True])
def test_plus_both_sources_preserve_agent_and_then_resume_stage_and_curriculum(modules, from_impact):
    module, checkpoint = modules
    if from_impact:
        source, agent, source_trainer, source_callback = impact_v2_initial_state(module)
        for _ in range(6000):
            source_callback(source_trainer)
            source_trainer.iteration += 1
        state = deepcopy(source.state_dict())
        origin = source_trainer.iteration
        offset = 6000
        expected_lr = agent.optimizer.param_groups[0]["lr"]
    else:
        state = {"ankle_finetune": _mature_combined_v2_source()}
        agent = FakeAgent()
        origin, offset = 140000, 0
        expected_lr = agent.optimizer.param_groups[0]["lr"] * 0.5
    untouched = deepcopy(state)
    plus = make_plus_adapter(module)
    module.validate_impact_v2_plus_checkpoint(
        {"environment": state, "iteration": origin}, plus.unwrapped.cfg, allow_source=True,
    )
    with pytest.raises(ValueError, match="Play requires its own"):
        module.validate_impact_v2_plus_checkpoint({"environment": state, "iteration": origin}, plus.unwrapped.cfg)
    plus.load_state_dict(state)
    trainer = SimpleNamespace(environment=plus, agent=agent, iteration=origin)
    callback = module.ImpactFinetuneCallback()
    callback(trainer)
    assert plus.impact_runtime.completed_updates == offset
    assert plus.finetune_metadata["completed_updates"] == 0
    assert plus.finetune_metadata["stage_origin_iteration"] == origin
    assert agent.optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
    for _ in range(3):
        callback(trainer)
        trainer.iteration += 1
    saved_agent, saved_env = deepcopy(agent.state_dict()), deepcopy(plus.state_dict())
    module.validate_impact_v2_plus_checkpoint(
        {"environment": saved_env, "iteration": origin + 3}, plus.unwrapped.cfg,
    )
    resumed, restored_agent = make_plus_adapter(module), FakeAgent()
    checkpoint.load_agent_checkpoint_strict(saved_agent, restored_agent)
    resumed.load_state_dict(saved_env)
    module.ImpactFinetuneCallback()(SimpleNamespace(environment=resumed, agent=restored_agent, iteration=origin + 3))
    checkpoint.verify_agent_checkpoint_restored(saved_agent, restored_agent)
    assert resumed.impact_runtime.completed_updates == offset + 3
    assert resumed.finetune_metadata["completed_updates"] == 3
    assert resumed.finetune_metadata["lr_caps"] == plus.finetune_metadata["lr_caps"]
    assert state.keys() == untouched.keys()
    assert state.get("finetune") == untouched.get("finetune")


@pytest.mark.parametrize("damage", ["parent_pd", "parent_distribution", "source_iteration", "offset", "lr", "recovery"])
def test_plus_rejects_incompatible_source_or_mutated_resume(modules, damage):
    module, _ = modules
    source, agent, trainer, _ = impact_v2_initial_state(module)
    state = deepcopy(source.state_dict())
    plus = make_plus_adapter(module)
    if damage in ("parent_pd", "parent_distribution"):
        if damage == "parent_pd":
            state["impact_v2"]["config_signature"] = "wrong-pd"
        else:
            state["impact"]["config_signature"] = state["finetune"]["config_signature"] = "wrong-distribution"
        with pytest.raises(ValueError, match="configuration differs"):
            plus.load_state_dict(state)
        return
    plus.load_state_dict(state)
    module.ImpactFinetuneCallback()(SimpleNamespace(environment=plus, agent=agent, iteration=trainer.iteration))
    saved = {"iteration": trainer.iteration, "environment": deepcopy(plus.state_dict())}
    contract = saved["environment"]["impact_v2_plus"]
    if damage == "source_iteration":
        contract["source"]["iteration"] += 1
    elif damage == "offset":
        contract["curriculum_offset"] += 1
    elif damage == "lr":
        saved["environment"]["finetune"]["lr_caps"][0] *= 0.5
    else:
        plus.unwrapped.cfg.recovery.max_s = 4.0
    with pytest.raises(ValueError):
        module.validate_impact_v2_plus_checkpoint(saved, plus.unwrapped.cfg)
