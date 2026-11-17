"""Same factory, module names and parameter order as the first stage."""

from dataclasses import dataclass

from t800_lab.tasks.t800_base.agent_cfg import T800BaseTrainerCfg


@dataclass
class T800ImpactTrainerCfg(T800BaseTrainerCfg):
    experiment_name = "T800_base-Impact"
