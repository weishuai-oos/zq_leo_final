from __future__ import annotations
from typing import Any, cast
import cusrl
import gymnasium as gym
import torch
from cusrl.environment.isaaclab import IsaacLabEnvAdapter
from cusrl.template import Environment
from cusrl.utils.typing import Slice


class CusrlT800Adapter(IsaacLabEnvAdapter):
    """Wraps a LeoLabRlEnv (or any ManagerBasedRLEnv) for CusRL,
    with support for multi-reward and multi-dimensional observation terms."""

    def __init__(self, wrapped: gym.Env, **spec_kwargs: Any):
        from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

        self.wrapped = wrapped
        self.unwrapped: DirectRLEnv | ManagerBasedRLEnv = wrapped.unwrapped
        self.device = self.unwrapped.device
        self.metrics = cusrl.utils.Metrics()
        self._amp_reference_pool: torch.Tensor | None = None
        self._amp_observation_dim: int | None = None
        self._amp_reference_capacity = 32768
        self._uses_adapter_amp_sampler = False

        reward_dim = getattr(self.unwrapped, "num_rewards", 1)
        demonstration_sampler = getattr(self.unwrapped, "collect_reference_motions", None)
        if (
            demonstration_sampler is None
            and self._has_observation_group("amp_policy")
            and self._has_observation_group("amp_reference")
        ):
            demonstration_sampler = self.collect_reference_motions
            self._uses_adapter_amp_sampler = True

        # Call Environment.__init__ directly to pass reward_dim
        # (IsaacLabEnvAdapter.__init__ hardcodes reward_dim=1)
        Environment.__init__(
            self,
            num_instances=self.unwrapped.num_envs,
            observation_dim=self._get_observation_dim(),
            action_dim=self._get_action_dim(),
            state_dim=self._get_state_dim(),
            reward_dim=reward_dim,
            autoreset=True,
            demonstration_sampler=demonstration_sampler,
            final_state_is_missing=True,
            **spec_kwargs,
        )

        # Pass symmetry specs and any task-specific extras from env config if available.
        cfg = self.unwrapped.cfg
        for attr in ("mirror_observation", "mirror_state", "mirror_action"):
            val = getattr(cfg, attr, None)
            if val is not None:
                setattr(self.spec, attr, val)
        spec_extras = getattr(cfg, "cusrl_spec_extras", None)
        if isinstance(spec_extras, dict):
            for key, value in spec_extras.items():
                setattr(self.spec, key, value)

        if self._uses_adapter_amp_sampler:
            # Prime the expert pool once so the AMP hook can immediately sample from it during agent initialization.
            self.reset()

        print("\033[0m", end="")

    def _get_observation_dim(self) -> int:
        """Compute flattened observation dim, supporting multi-dim obs terms."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_manager = self.unwrapped.observation_manager
            if hasattr(obs_manager, "group_obs_term_dim") and "policy" in obs_manager.group_obs_term_dim:
                term_dims = obs_manager.group_obs_term_dim["policy"]
                return int(sum(torch.prod(torch.tensor(dim, device="cpu")).item() for dim in term_dims))
            shape = obs_manager.group_obs_dim["policy"]
            if len(shape) != 1:
                raise ValueError("Only 1D observation space is supported without group_obs_term_dim.")
            return shape[0]
        return gym.spaces.flatdim(self.unwrapped.single_observation_space["policy"])

    def _get_state_dim(self) -> int | None:
        """Compute flattened critic observation dim, supporting multi-dim obs terms."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_manager = self.unwrapped.observation_manager
            if hasattr(obs_manager, "group_obs_term_dim"):
                term_dims = obs_manager.group_obs_term_dim.get("critic")
                if term_dims is None:
                    return None
                return int(sum(torch.prod(torch.tensor(dim, device="cpu")).item() for dim in term_dims))
            shape = obs_manager.group_obs_dim.get("critic")
            if shape is None:
                return None
            if len(shape) != 1:
                raise ValueError("Only 1D state space is supported without group_obs_term_dim.")
            return shape[0]
        space = self.unwrapped.single_observation_space.get("critic")
        if space is None:
            return None
        return gym.spaces.flatdim(space)

    def step(self, action: torch.Tensor):
        observation_dict, reward, terminated, truncated, extras = self.wrapped.step(action)
        pose_visualizer = getattr(self.unwrapped, "_play_pose_visualizer", None)
        if pose_visualizer is not None:
            pose_visualizer.on_step()

        observation = self._extract_and_flatten(observation_dict, "policy")
        state = self._extract_and_flatten(observation_dict, "critic")
        amp_observation = self._extract_and_flatten(observation_dict, "amp_policy")
        amp_reference = self._extract_and_flatten(observation_dict, "amp_reference")
        self._validate_amp_observation_pair(amp_observation, amp_reference)
        self._update_amp_reference_pool(amp_reference)

        # Multi-reward: dict from MultiRewardManager -> stacked [N, num_groups]
        if isinstance(reward, dict):
            reward = torch.stack(list(reward.values()), dim=-1)
        else:
            reward = cast(torch.Tensor, reward).unsqueeze(-1)

        terminated = cast(torch.Tensor, terminated).unsqueeze(-1)
        truncated = cast(torch.Tensor, truncated).unsqueeze(-1)
        extras = cast(dict, extras)
        self.metrics.record(**extras.pop("log", {}))

        step_extras = observation_dict | extras
        if amp_observation is not None:
            step_extras["amp_obs"] = amp_observation

        return observation, state, reward, terminated, truncated, step_extras

    def reset(
        self,
        *,
        indices: torch.Tensor | Slice | None = None,
        randomize_episode_progress: bool = False,
    ):
        if indices is None:
            observation_dict, _ = self.wrapped.reset()
            observation = self._extract_and_flatten(observation_dict, "policy")
            state = self._extract_and_flatten(observation_dict, "critic")
            amp_observation = self._extract_and_flatten(observation_dict, "amp_policy")
            amp_reference = self._extract_and_flatten(observation_dict, "amp_reference")
            self._validate_amp_observation_pair(amp_observation, amp_reference)
            self._update_amp_reference_pool(amp_reference)
            extras = {k: v for k, v in observation_dict.items() if k not in ("policy", "critic")}
        else:
            if isinstance(indices, slice):
                indices = torch.arange(self.num_instances, device=self.device)[indices]
            indices = torch.as_tensor(indices, device=self.device)
            observation_dict, _ = self.unwrapped.reset(env_ids=indices)

            observation = self._extract_and_flatten(observation_dict, "policy")
            state = self._extract_and_flatten(observation_dict, "critic")
            amp_observation = self._extract_and_flatten(observation_dict, "amp_policy")
            amp_reference = self._extract_and_flatten(observation_dict, "amp_reference")
            self._validate_amp_observation_pair(amp_observation, amp_reference)
            self._update_amp_reference_pool(amp_reference)
            extras = {}
            for key, value in observation_dict.items():
                if key not in ("policy", "critic"):
                    extras[key] = value[indices] if isinstance(value, torch.Tensor) else value

            if observation is not None:
                observation = observation[indices]
            if state is not None:
                state = state[indices]

        if randomize_episode_progress:
            target = indices if indices is not None else slice(None)
            self.unwrapped.episode_length_buf[target] = torch.randint_like(
                self.unwrapped.episode_length_buf[target], int(self.unwrapped.max_episode_length)
            )

        pose_visualizer = getattr(self.unwrapped, "_play_pose_visualizer", None)
        if pose_visualizer is not None:
            pose_visualizer.on_reset(indices)

        return observation, state, extras

    def collect_reference_motions(self, num_samples: int):
        """Samples flattened AMP reference observations from the latest cached reference pool."""
        if self._amp_reference_pool is None or self._amp_reference_pool.size(0) == 0:
            self.reset()
        if self._amp_reference_pool is None or self._amp_reference_pool.size(0) == 0:
            raise RuntimeError("AMP reference pool is empty after reset.")
        indices = torch.randint(self._amp_reference_pool.size(0), (num_samples,), device=self.device)
        return self._amp_reference_pool[indices]

    def _extract_and_flatten(self, obs_dict: dict, group_name: str) -> torch.Tensor | None:
        """Extract an observation group and flatten it to [N, D]."""
        group = obs_dict.pop(group_name, None)
        if group is None:
            return None
        if isinstance(group, torch.Tensor):
            return group
        # Dict of terms (concatenate_terms=False): flatten each and concat
        parts = []
        for value in group.values():
            parts.append(value.flatten(start_dim=1))
        return torch.cat(parts, dim=1)

    def _has_observation_group(self, group_name: str) -> bool:
        obs_manager = getattr(self.unwrapped, "observation_manager", None)
        group_obs_dim = getattr(obs_manager, "group_obs_dim", None)
        return isinstance(group_obs_dim, dict) and group_name in group_obs_dim

    @property
    def amp_observation_dim(self) -> int | None:
        """Flattened AMP dimension produced by both policy and reference observations."""
        return self._amp_observation_dim

    def _validate_amp_observation_pair(
        self,
        amp_observation: torch.Tensor | None,
        amp_reference: torch.Tensor | None,
    ) -> None:
        if amp_observation is None and amp_reference is None:
            return
        if amp_observation is None or amp_reference is None:
            raise RuntimeError("AMP policy and reference observation groups must either both exist or both be absent.")

        policy_dim = int(amp_observation.size(-1))
        reference_dim = int(amp_reference.size(-1))
        if policy_dim != reference_dim:
            raise RuntimeError(
                f"AMP policy/reference dimension mismatch: policy={policy_dim}, reference={reference_dim}."
            )
        if self._amp_observation_dim is not None and self._amp_observation_dim != policy_dim:
            raise RuntimeError(
                f"AMP observation dimension changed across steps: {self._amp_observation_dim} vs {policy_dim}."
            )
        self._amp_observation_dim = policy_dim

    def _update_amp_reference_pool(self, amp_reference: torch.Tensor | None) -> None:
        if amp_reference is None:
            return

        amp_reference = amp_reference.detach().clone()
        if self._amp_reference_pool is None:
            self._amp_reference_pool = amp_reference
            return

        if self._amp_reference_pool.size(-1) != amp_reference.size(-1):
            raise RuntimeError(
                "AMP reference dimension changed across steps: "
                f"{self._amp_reference_pool.size(-1)} vs {amp_reference.size(-1)}."
            )

        self._amp_reference_pool = torch.cat([self._amp_reference_pool, amp_reference], dim=0)
        if self._amp_reference_pool.size(0) > self._amp_reference_capacity:
            keep = torch.randperm(self._amp_reference_pool.size(0), device=self.device)[: self._amp_reference_capacity]
            self._amp_reference_pool = self._amp_reference_pool[keep]
