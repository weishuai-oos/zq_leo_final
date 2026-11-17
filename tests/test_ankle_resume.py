"""Continuation must preserve optimizer/RMS state and never restart warmup on resume."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from test_impact_resume import FakeAgent

ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = ROOT / "source/t800_lab/t800_lab/utils/wrappers/cusrl"


@pytest.fixture(scope="module")
def ankle_module():
    sys.path.insert(0, str(ROOT / "source/t800_lab"))
    package_name = "_ankle_resume_test"
    package = ModuleType(package_name)
    package.__path__ = [str(WRAPPERS)]
    sys.modules[package_name] = package
    base = ModuleType(f"{package_name}.vecenv_wrapper")

    class BaseAdapter:
        def __init__(self, wrapped):
            self.unwrapped = wrapped.unwrapped
            self.metrics = SimpleNamespace(record=lambda **kw: None)

    base.CusrlT800Adapter = BaseAdapter
    sys.modules[base.__name__] = base
    spec = importlib.util.spec_from_file_location(f"{package_name}.ankle", WRAPPERS / "ankle.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.path.remove(str(ROOT / "source/t800_lab"))
    for name in list(sys.modules):
        if name == package_name or name.startswith(package_name + "."):
            del sys.modules[name]


def make_adapter(module, task="T800_base-AnkleSmooth", **kwargs):
    terms = {
        "ankle_target_rate_l2": SimpleNamespace(weight=-0.04),
        "ankle_target_acceleration_l2": SimpleNamespace(weight=-0.008),
    }
    cfg = SimpleNamespace(
        rewards=SimpleNamespace(**terms),
        ankle_finetune_reward_weights={
            "ankle_target_rate_l2": (-0.01, -0.04),
            "ankle_target_acceleration_l2": (-0.002, -0.008),
        },
        scene=SimpleNamespace(robot=SimpleNamespace(actuators={
            "feet": SimpleNamespace(stiffness={"ankle": 40.0}, damping={"ankle": 0.3}),
        })),
    )
    for term in terms.values():
        term.func = make_adapter
        term.params = {
            "threshold": 1.0, "normalizer_rad": 0.1,
            "asset_cfg": SimpleNamespace(name="robot", joint_names=["ankle"], body_names=None, preserve_order=True),
        }
    manager = SimpleNamespace(get_term_cfg=lambda name: terms[name], set_term_cfg=lambda name, term: terms.update({name: term}))
    wrapped = SimpleNamespace(unwrapped=SimpleNamespace(cfg=cfg, reward_manager=manager))
    return module.AnkleCusrlAdapter(wrapped, task=task, **kwargs), terms


def test_warmup_lr_cap_and_resume_preserve_stage(ankle_module):
    m = ankle_module
    env, terms = make_adapter(m, origin_iteration=85500)
    trainer = SimpleNamespace(environment=env, agent=FakeAgent(), iteration=85500)
    callback = m.AnkleFinetuneCallback()
    callback(trainer)
    assert terms["ankle_target_rate_l2"].weight == pytest.approx(-0.01)
    assert trainer.agent.optimizer.param_groups[0]["lr"] == pytest.approx(2e-5)
    assert trainer.agent.scheduler._lr_scale == pytest.approx(0.031 * 0.5)
    for _ in range(1000):
        callback(trainer)
        trainer.iteration += 1
    assert terms["ankle_target_rate_l2"].weight == pytest.approx(-0.025)
    saved = deepcopy(env.state_dict())
    restored, new_terms = make_adapter(m, origin_iteration=85500)
    restored.load_state_dict(saved)
    resumed = SimpleNamespace(environment=restored, agent=trainer.agent, iteration=86500)
    new_callback = m.AnkleFinetuneCallback()
    new_callback(resumed)
    assert restored.finetune_metadata["completed_updates"] == 1000
    assert resumed.agent.optimizer.param_groups[0]["lr"] == pytest.approx(2e-5)  # Not halved twice.
    assert new_terms["ankle_target_rate_l2"].weight == pytest.approx(-0.025)
    for _ in range(1000):
        new_callback(resumed)
        resumed.iteration += 1
    assert new_terms["ankle_target_rate_l2"].weight == pytest.approx(-0.04)
    assert new_terms["ankle_target_acceleration_l2"].weight == pytest.approx(-0.008)


def test_checkpoint_task_config_origin_and_progress_are_checked(ankle_module):
    m = ankle_module
    env, _ = make_adapter(m)
    trainer = SimpleNamespace(environment=env, agent=FakeAgent(), iteration=85500)
    m.AnkleFinetuneCallback()(trainer)
    state = env.state_dict()
    for kwargs in ({"task": "T800_base-AnklePose"}, {"warmup_iterations": 1000}, {"lr_scale": 0.25}, {"origin_iteration": 1}):
        other, _ = make_adapter(m, **kwargs)
        with pytest.raises(ValueError):
            other.load_state_dict(state)
    other, _ = make_adapter(m)
    other.load_state_dict(state)
    with pytest.raises(ValueError, match="progress"):
        m.AnkleFinetuneCallback()(SimpleNamespace(environment=other, agent=FakeAgent(), iteration=85501))
    source, _ = make_adapter(m, origin_iteration=85500)
    with pytest.raises(ValueError, match="Source checkpoint iteration"):
        m.AnkleFinetuneCallback()(SimpleNamespace(environment=source, agent=FakeAgent(), iteration=85000))


def test_matching_play_task_is_required(ankle_module):
    m = ankle_module
    env, _ = make_adapter(m, task="T800_base-AnkleDamping")
    cfg = env.unwrapped.cfg
    checkpoint = {"environment": {"ankle_finetune": {
        "task": env.task, "play_config_signature": env.play_config_signature,
    }}}
    m.validate_ankle_play_checkpoint(checkpoint, "T800_base-AnkleDamping-Play", cfg)
    m.validate_ankle_play_checkpoint({"environment": {}}, "T800_base-AnkleDamping-Play", cfg)
    for wrong in ("T800_base-Play", "T800_base-AnklePose-Play"):
        with pytest.raises(ValueError, match="matching task"):
            m.validate_ankle_play_checkpoint(checkpoint, wrong, cfg)
    cfg.scene.robot.actuators["feet"].damping = {"ankle": 0.6}
    with pytest.raises(ValueError, match="PD/reward"):
        m.validate_ankle_play_checkpoint(checkpoint, "T800_base-AnkleDamping-Play", cfg)


@pytest.mark.parametrize("parameter,new_value", [("threshold", 2.0), ("normalizer_rad", 0.2)])
def test_pose_parameters_are_in_configuration_signature(ankle_module, parameter, new_value):
    env, terms = make_adapter(ankle_module)
    cfg = env.unwrapped.cfg
    signature = ankle_module.ankle_configuration_signature(cfg, env.task)
    # IsaacLab resolves entity IDs after scene construction; this is not a config change.
    terms["ankle_target_rate_l2"].params["asset_cfg"].joint_ids = [3]
    assert ankle_module.ankle_configuration_signature(cfg, env.task) == signature
    terms["ankle_target_rate_l2"].params[parameter] = new_value
    assert ankle_module.ankle_configuration_signature(cfg, env.task) != signature


def test_iteration_budget_and_zero_warmup(ankle_module):
    m = ankle_module
    assert m.resolve_ankle_iteration_budget(85500) == 115500
    assert m.resolve_ankle_iteration_budget(90000, max_iterations=115500) == 115500
    for endpoint in (85000, 85500):
        with pytest.raises(ValueError):
            m.resolve_ankle_iteration_budget(85500, max_iterations=endpoint)
    env, terms = make_adapter(m, warmup_iterations=0)
    m.AnkleFinetuneCallback()(SimpleNamespace(environment=env, agent=FakeAgent(), iteration=85500))
    assert terms["ankle_target_rate_l2"].weight == -0.04


def make_combined_adapter(module):
    previous, _ = make_adapter(module)
    cfg = deepcopy(previous.unwrapped.cfg)
    del cfg.ankle_finetune_reward_weights
    cfg.scene.robot.actuators["feet"].damping = {"ankle": 1.2}
    cfg.rewards.ankle_support_joint_pose_l2 = SimpleNamespace(
        weight=-0.02, func=make_combined_adapter, params={"normalizer_rad": 0.1, "threshold": 1.0},
    )
    return module.AnkleCombinedCusrlAdapter(SimpleNamespace(unwrapped=SimpleNamespace(cfg=cfg))), cfg


def test_combined_checkpoint_roundtrip_and_matching_play(ankle_module):
    env, cfg = make_combined_adapter(ankle_module)
    state = env.state_dict()
    assert set(state) == {"ankle_combined"}
    assert cfg.rewards.ankle_target_rate_l2.weight == -0.04
    assert cfg.rewards.ankle_support_joint_pose_l2.weight == -0.02
    restored, _ = make_combined_adapter(ankle_module)
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    ankle_module.validate_ankle_play_checkpoint(
        {"environment": state}, "T800_base-AnkleCombined-Play", cfg,
    )
    # Callers cannot accidentally modify the adapter's saved contract.
    state["ankle_combined"]["task"] = "T800_base"
    assert env.state_dict()["ankle_combined"]["task"] == "T800_base-AnkleCombined"


@pytest.mark.parametrize("state", [{}, {"ankle_finetune": {}}, None])
def test_combined_rejects_base_and_ablation_checkpoints(ankle_module, state):
    env, _ = make_combined_adapter(ankle_module)
    with pytest.raises(ValueError, match="own checkpoint"):
        env.load_state_dict(state)


@pytest.mark.parametrize("change", ["damping", "reward", "pose_gate"])
def test_combined_resume_and_play_reject_changed_pd_or_rewards(ankle_module, change):
    env, cfg = make_combined_adapter(ankle_module)
    state = env.state_dict()
    if change == "damping":
        cfg.scene.robot.actuators["feet"].damping = {"ankle": 0.3}
    elif change == "reward":
        cfg.rewards.ankle_target_rate_l2.weight = -0.01
    else:
        cfg.rewards.ankle_support_joint_pose_l2.params["threshold"] = 2.0
    with pytest.raises(ValueError, match="PD/reward"):
        env.load_state_dict(state)
    with pytest.raises(ValueError, match="PD/reward"):
        ankle_module.validate_ankle_play_checkpoint(
            {"environment": state}, "T800_base-AnkleCombined-Play", cfg,
        )


@pytest.mark.parametrize("task", ["T800_base-Play", "T800_base-AnkleSmooth-Play", "T800_base-AnklePose"])
def test_combined_checkpoint_cannot_play_under_another_task(ankle_module, task):
    env, cfg = make_combined_adapter(ankle_module)
    with pytest.raises(ValueError, match="matching task"):
        ankle_module.validate_ankle_play_checkpoint({"environment": env.state_dict()}, task, cfg)


def test_combined_cli_starts_from_zero_and_keeps_old_continuation_requirement():
    from test_impact_launch import train_cli_namespace

    args = train_cli_namespace(["--task", "T800_base-AnkleCombined", "--max_iterations", "100000"])
    assert args.checkpoint is None
    assert args.run_name == "T800_base-AnkleCombined"
    assert args.finetune_lr_scale is args.finetune_warmup_iterations is None
    for task in ("T800_base-AnkleSmooth", "T800_base-AnkleDamping", "T800_base-AnkleTargetLimit", "T800_base-AnklePose"):
        with pytest.raises(SystemExit):
            train_cli_namespace(["--task", task])


@pytest.mark.parametrize("flag,value", [
    ("--additional_iterations", "30000"),
    ("--finetune_lr_scale", "0.5"),
    ("--finetune_warmup_iterations", "2000"),
])
def test_combined_cli_rejects_ablation_finetune_settings(flag, value):
    from test_impact_launch import train_cli_namespace

    with pytest.raises(SystemExit):
        train_cli_namespace(["--task", "T800_base-AnkleCombined", flag, value])


def make_combined_v2_adapter(module, **kwargs):
    source, source_cfg = make_combined_adapter(module)
    cfg = deepcopy(source_cfg)
    cfg.scene.robot.actuators["feet"].damping = {"ankle": 2.0}
    cfg.sim = SimpleNamespace(dt=0.005)
    cfg.decimation = 4
    cfg.events = SimpleNamespace(collider_offsets=SimpleNamespace(
        func=make_combined_v2_adapter, mode="startup",
        params={"contact_offset_distribution_params": (0.002, 0.002), "rest_offset_distribution_params": None},
    ))
    cfg.ankle_substep_stability = {
        "contact_force_enter_n": 50.0,
        "contact_force_exit_n": 10.0,
        "support_settle_time_s": 0.03,
        "acceleration_deadband_rad_s2": 100.0,
    }
    cfg.ankle_finetune_reward_weights = {
        "ankle_physics_acceleration_l2": (0.0, -0.02),
        "ankle_support_foot_angular_velocity_l2": (0.0, -0.2),
    }
    for name, (_, final) in cfg.ankle_finetune_reward_weights.items():
        setattr(cfg.rewards, name, SimpleNamespace(weight=final, func=make_combined_v2_adapter, params={}))
    # IsaacLab's RewardManager owns a copy of its configuration.  Warming the
    # live manager must not rewrite the declarative checkpoint contract.
    terms = deepcopy(vars(cfg.rewards))
    manager = SimpleNamespace(get_term_cfg=lambda name: terms[name], set_term_cfg=lambda name, term: terms.update({name: term}))
    wrapped = SimpleNamespace(unwrapped=SimpleNamespace(cfg=cfg, reward_manager=manager))
    adapter = module.AnkleCusrlAdapter(
        wrapped, task="T800_base-AnkleCombinedV2", source_combined_cfg=source_cfg, **kwargs,
    )
    return adapter, terms, source


def test_combined_v2_migration_warms_only_new_rewards_and_preserves_resume(ankle_module):
    m = ankle_module
    env, terms, source = make_combined_v2_adapter(m, origin_iteration=100000)
    assert env.warmup_iterations == 1000
    source_state = source.state_dict()
    env.load_state_dict(source_state)
    assert env.source_combined_metadata == source_state["ankle_combined"]
    assert env.finetune_metadata is None
    trainer = SimpleNamespace(environment=env, agent=FakeAgent(), iteration=100000)
    callback = m.AnkleFinetuneCallback()
    callback(trainer)
    assert terms["ankle_physics_acceleration_l2"].weight == 0
    assert terms["ankle_support_foot_angular_velocity_l2"].weight == 0
    assert terms["ankle_target_rate_l2"].weight == -0.04
    assert terms["ankle_target_acceleration_l2"].weight == -0.008
    assert terms["ankle_support_joint_pose_l2"].weight == -0.02
    for _ in range(500):
        callback(trainer)
        trainer.iteration += 1
    assert terms["ankle_physics_acceleration_l2"].weight == pytest.approx(-0.01)
    assert terms["ankle_support_foot_angular_velocity_l2"].weight == pytest.approx(-0.1)
    state = deepcopy(env.state_dict())
    restored, new_terms, _ = make_combined_v2_adapter(m, origin_iteration=100000)
    restored.load_state_dict(state)
    resumed = SimpleNamespace(environment=restored, agent=trainer.agent, iteration=100500)
    new_callback = m.AnkleFinetuneCallback()
    new_callback(resumed)
    assert restored.finetune_metadata["origin_iteration"] == 100000
    assert restored.finetune_metadata["completed_updates"] == 500
    assert resumed.agent.optimizer.param_groups[0]["lr"] == pytest.approx(2e-5)
    assert new_terms["ankle_physics_acceleration_l2"].weight == pytest.approx(-0.01)
    for _ in range(500):
        new_callback(resumed)
        resumed.iteration += 1
    assert new_terms["ankle_physics_acceleration_l2"].weight == -0.02
    assert new_terms["ankle_support_foot_angular_velocity_l2"].weight == -0.2
    m.validate_ankle_play_checkpoint(
        {"environment": restored.state_dict()}, "T800_base-AnkleCombinedV2-Play", restored.unwrapped.cfg,
    )


@pytest.mark.parametrize("state", [{}, {"ankle_finetune": {}}, None, {"ankle_combined": {}}, {"impact": {}}])
def test_combined_v2_rejects_missing_or_unrelated_training_source(ankle_module, state):
    env, _, _ = make_combined_v2_adapter(ankle_module)
    with pytest.raises(ValueError):
        env.load_state_dict(state)


def test_combined_v2_validates_source_signature_and_does_not_mutate_it(ankle_module):
    m = ankle_module
    env, _, source = make_combined_v2_adapter(m)
    state = source.state_dict()
    before = deepcopy(state)
    m.validate_ankle_training_checkpoint(
        {"environment": state}, env.task, env.unwrapped.cfg, source.unwrapped.cfg,
    )
    env.load_state_dict(state)
    assert state == before
    state["ankle_combined"]["play_config_signature"] = "0" * 64
    with pytest.raises(ValueError, match="PD/reward"):
        env.load_state_dict(state)


def test_combined_v2_callback_cannot_start_without_loading_source(ankle_module):
    env, _, _ = make_combined_v2_adapter(ankle_module)
    with pytest.raises(ValueError, match="validated Combined checkpoint"):
        ankle_module.AnkleFinetuneCallback()(
            SimpleNamespace(environment=env, agent=FakeAgent(), iteration=100000),
        )


def test_combined_v2_play_does_not_allow_training_migration(ankle_module):
    m = ankle_module
    env, _, source = make_combined_v2_adapter(m)
    checkpoint = {"environment": source.state_dict()}
    with pytest.raises(ValueError, match="training source only"):
        m.validate_ankle_play_checkpoint(checkpoint, env.task + "-Play", env.unwrapped.cfg)
    env.load_state_dict(source.state_dict())
    m.AnkleFinetuneCallback()(SimpleNamespace(environment=env, agent=FakeAgent(), iteration=100000))
    checkpoint = {"environment": env.state_dict()}
    for wrong in ("T800_base-Play", "T800_base-AnkleCombined-Play", "T800_base-AnkleSmooth-Play"):
        with pytest.raises(ValueError):
            m.validate_ankle_play_checkpoint(checkpoint, wrong, env.unwrapped.cfg)
    m.validate_ankle_play_checkpoint(checkpoint, env.task + "-Play", env.unwrapped.cfg)


@pytest.mark.parametrize("change", ["dt", "decimation", "contact_offset", "rest_offset", "substep_threshold"])
def test_combined_v2_signature_covers_physics_and_motion_thresholds(ankle_module, change):
    m = ankle_module
    env, _, source = make_combined_v2_adapter(m)
    env.load_state_dict(source.state_dict())
    m.AnkleFinetuneCallback()(SimpleNamespace(environment=env, agent=FakeAgent(), iteration=100000))
    checkpoint = {"environment": env.state_dict()}
    cfg = env.unwrapped.cfg
    if change == "dt":
        cfg.sim.dt = 0.002
    elif change == "decimation":
        cfg.decimation = 5
    elif change == "contact_offset":
        cfg.events.collider_offsets.params["contact_offset_distribution_params"] = (0.006, 0.014)
    elif change == "rest_offset":
        cfg.events.collider_offsets.params["rest_offset_distribution_params"] = (0.001, 0.001)
    else:
        cfg.ankle_substep_stability["support_settle_time_s"] = 0.1
    with pytest.raises(ValueError, match="physics configuration"):
        m.validate_ankle_play_checkpoint(checkpoint, env.task + "-Play", cfg)


def test_combined_v2_custom_warmup_and_lr_settings_must_match_on_resume(ankle_module):
    m = ankle_module
    env, _, source = make_combined_v2_adapter(m, warmup_iterations=250, lr_scale=0.25)
    env.load_state_dict(source.state_dict())
    m.AnkleFinetuneCallback()(SimpleNamespace(environment=env, agent=FakeAgent(), iteration=100000))
    restored, _, _ = make_combined_v2_adapter(m, warmup_iterations=250, lr_scale=0.25)
    restored.load_state_dict(env.state_dict())
    default, _, _ = make_combined_v2_adapter(m)
    with pytest.raises(ValueError, match="warmup/LR"):
        default.load_state_dict(env.state_dict())


def test_combined_v2_cli_requires_checkpoint_and_accepts_continuation_settings():
    from test_impact_launch import train_cli_namespace

    with pytest.raises(SystemExit):
        train_cli_namespace(["--task", "T800_base-AnkleCombinedV2"])
    args = train_cli_namespace([
        "--task", "T800_base-AnkleCombinedV2", "--checkpoint", "/tmp/combined/ckpt_100000.pt",
        "--additional_iterations", "30000", "--finetune_lr_scale", "0.5",
        "--finetune_warmup_iterations", "1000",
    ])
    assert args.additional_iterations == 30000
    assert args.finetune_lr_scale == 0.5
    assert args.finetune_warmup_iterations == 1000
    assert args.run_name == "T800_base-AnkleCombinedV2"
