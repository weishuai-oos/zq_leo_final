"""Register only T800 training and its evaluation configuration."""

import gymnasium as gym

_ANKLE_VARIANT_TASKS = {
    "AnkleSmooth": "T800AnkleSmoothEnvCfg",
    "AnkleDamping": "T800AnkleDampingEnvCfg",
    "AnkleTargetLimit": "T800AnkleTargetLimitEnvCfg",
    "AnklePose": "T800AnklePoseEnvCfg",
    "AnkleCombined": "T800AnkleCombinedEnvCfg",
    "AnkleCombinedV2": "T800AnkleCombinedV2EnvCfg",
}

for _variant, _cfg_class in _ANKLE_VARIANT_TASKS.items():
    _env_entry = (
        "t800_lab.tasks.t800_ankle.env:T800AnkleStabilityEnv"
        if _variant == "AnkleCombinedV2" else "isaaclab.envs:ManagerBasedRLEnv"
    )
    gym.register(
        id=f"T800_base-{_variant}",
        entry_point=_env_entry,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"t800_lab.tasks.t800_ankle.env_cfg:{_cfg_class}",
            "cusrl_cfg_entry_point": "t800_lab.tasks.t800_base.agent_cfg:T800BaseTrainerCfg",
        },
    )
    gym.register(
        id=f"T800_base-{_variant}-Play",
        entry_point=_env_entry,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"t800_lab.tasks.t800_ankle.env_cfg:{_cfg_class}_PLAY",
            "cusrl_cfg_entry_point": "t800_lab.tasks.t800_base.agent_cfg:T800BaseTrainerCfg",
        },
    )

gym.register(
    id="T800_base",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "t800_lab.tasks.t800_base.env_cfg:T800BaseEnvCfg",
        "cusrl_cfg_entry_point": "t800_lab.tasks.t800_base.agent_cfg:T800BaseTrainerCfg",
    },
)
gym.register(
    id="T800_base-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "t800_lab.tasks.t800_base.env_cfg:T800BaseEnvCfg_PLAY",
        "cusrl_cfg_entry_point": "t800_lab.tasks.t800_base.agent_cfg:T800BaseTrainerCfg",
    },
)

gym.register(
    id="T800_base-Impact",
    entry_point="t800_lab.tasks.t800_impact.env:T800ImpactEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "t800_lab.tasks.t800_impact.env_cfg:T800ImpactEnvCfg",
        "cusrl_cfg_entry_point": "t800_lab.tasks.t800_impact.agent_cfg:T800ImpactTrainerCfg",
    },
)
gym.register(
    id="T800_base-Impact-Play",
    entry_point="t800_lab.tasks.t800_impact.env:T800ImpactEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "t800_lab.tasks.t800_impact.env_cfg:T800ImpactEnvCfg_PLAY",
        "cusrl_cfg_entry_point": "t800_lab.tasks.t800_impact.agent_cfg:T800ImpactTrainerCfg",
    },
)

for _suffix, _cfg_suffix in (("", ""), ("-Play", "_PLAY")):
    gym.register(
        id="T800_base-ImpactV2" + _suffix,
        entry_point="t800_lab.tasks.t800_impact_v2.env:T800ImpactV2Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "t800_lab.tasks.t800_impact_v2.env_cfg:T800ImpactV2EnvCfg" + _cfg_suffix,
            "cusrl_cfg_entry_point": "t800_lab.tasks.t800_impact_v2.env_cfg:T800ImpactV2TrainerCfg",
        },
    )

for _task, _stem in (
    ("T800_base-ImpactV2-From154500-DR", "T800DynamicsDRV2"),
    ("T800_base-ImpactV2Plus-From173000-DR", "T800DynamicsDRPlus"),
):
    for _suffix, _cfg_suffix in (("", ""), ("-Play", "_PLAY")):
        gym.register(
            id=_task + _suffix,
            entry_point=f"t800_lab.tasks.t800_dynamics_dr.env:{_stem}Env",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"t800_lab.tasks.t800_dynamics_dr.env_cfg:{_stem}EnvCfg" + _cfg_suffix,
                "cusrl_cfg_entry_point": "t800_lab.tasks.t800_base.agent_cfg:T800BaseTrainerCfg",
            },
        )

for _suffix, _cfg_suffix in (("", ""), ("-Play", "_PLAY")):
    gym.register(
        id="T800_base-ImpactV2Plus" + _suffix,
        entry_point="t800_lab.tasks.t800_impact_v2_plus.env:T800ImpactV2PlusEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "t800_lab.tasks.t800_impact_v2_plus.env_cfg:T800ImpactV2PlusEnvCfg" + _cfg_suffix,
            "cusrl_cfg_entry_point": "t800_lab.tasks.t800_impact_v2_plus.env_cfg:T800ImpactV2PlusTrainerCfg",
        },
    )
