"""LSTM PPO/AMP trainer for the single T800_base task."""

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

import cusrl
from cusrl.environment.isaaclab import TrainerCfg
from cusrl.utils.dict_utils import get_first


class ScaledAdversarialMotionPrior(cusrl.hook.AdversarialMotionPrior):
    """Local AMP hook wrapper that actually applies ``reward_scale`` to the style reward."""

    @torch.no_grad()
    def post_step(self, transition):
        if (agent_transition := cast(Tensor, transition.pop("amp_obs", None))) is None:
            if self.state_indices is None:
                raise ValueError("AMP observation is not provided and indices are not specified.")

            state = cast(Tensor, get_first(transition, "state", "observation"))
            next_state = cast(Tensor, get_first(transition, "next_state", "next_observation"))
            amp_state = state[..., self.state_indices]
            amp_next_state = next_state[..., self.state_indices]
            agent_transition = torch.cat([amp_state, amp_next_state], dim=-1)

        expert_transition = self._sample_demonstration(agent_transition.size(0))
        self.transition_rms.update(agent_transition)
        self.transition_rms.update(expert_transition)
        agent_transition = self.transition_rms.normalize(agent_transition)
        expert_transition = self.transition_rms.normalize(expert_transition)
        transition["agent_transition"] = agent_transition
        transition["expert_transition"] = expert_transition

        with self.agent.autocast():
            logit = self.discriminator(agent_transition)
        style_reward = -torch.log(torch.clamp(1 - 1 / (1 + torch.exp(-logit)), min=1e-4))
        style_reward = style_reward * self.reward_scale
        if (impact_scale := transition.pop("impact_style_scale", None)) is not None:
            if impact_scale.shape != style_reward.shape or not torch.isfinite(impact_scale).all():
                raise ValueError("Invalid per-transition impact style scale.")
            style_reward = style_reward * impact_scale.clamp(0.0, 1.0)

        cast(Tensor, transition["reward"]).add_(style_reward)
        self.agent.record(amp_reward=style_reward)


@dataclass
class T800BaseTrainerCfg(TrainerCfg):
    """LSTM CusRL setup for T800 flat locomotion with AMP style supervision."""

    max_iterations = 133000
    save_interval = 1000
    experiment_name = "T800_base"
    agent_factory = cusrl.ActorCritic.Factory(
        num_steps_per_update=24,
        actor_factory=cusrl.Actor.Factory(
            backbone_factory=cusrl.Lstm.Factory(hidden_size=256, num_layers=2),
            distribution_factory=cusrl.NormalDist.Factory(),
        ),
        critic_factory=cusrl.Value.Factory(
            backbone_factory=cusrl.Lstm.Factory(hidden_size=256, num_layers=2),
        ),
        optimizer_factory=cusrl.OptimizerFactory(torch.optim.AdamW, defaults=dict(lr=1e-3)),
        sampler=cusrl.AutoMiniBatchSampler(num_epochs=5, num_mini_batches=4),
        hooks=[
            cusrl.hook.ModuleInitialization(),
            cusrl.hook.ObservationNormalization(),
            ScaledAdversarialMotionPrior(
                discriminator_factory=cusrl.Mlp.Factory(hidden_dims=(512, 256)),
                batch_size=512,
                reward_scale=0.03,
                loss_weight=1.0,
                grad_penalty_weight=5.0,
            ),
            cusrl.hook.ValueComputation(),
            cusrl.hook.GeneralizedAdvantageEstimation(),
            cusrl.hook.AdvantageNormalization(),
            cusrl.hook.ValueLoss(),
            cusrl.hook.OnPolicyPreparation(),
            cusrl.hook.PpoSurrogateLoss(),
            cusrl.hook.EntropyLoss(0.003),
            cusrl.hook.GradientClipping(1.0),
            cusrl.hook.OnPolicyStatistics(cusrl.sampler.AutoMiniBatchSampler()),
            cusrl.hook.AdaptiveLRSchedule(0.015),
        ],
    )
