"""Evaluation hooks for periodic reward-vector evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import torch

from gepa_dapo_grn.config import RewardMixerConfig
from gepa_dapo_grn.policy_interfaces import Policy
from gepa_dapo_grn.reward_mixers import mix_reward_vectors


@dataclass
class EvalResult:
    """Container for evaluation metrics."""

    metrics: Dict[str, float]
    scalar_rewards: torch.Tensor


def _summarize_reward_vectors(reward_vectors: List[Dict[str, float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    if not reward_vectors:
        return summary
    keys: set[str] = set()
    for vector in reward_vectors:
        keys.update(vector.keys())
    for key in keys:
        values = torch.tensor(
            [vector.get(key, 0.0) for vector in reward_vectors],
            dtype=torch.float32,
        )
        summary[f"eval/{key}/mean"] = values.mean().item()
        summary[f"eval/{key}/std"] = values.std().item()
    return summary


def run_eval(
    policy: Policy,
    reward_fn: Callable[[Policy], List[Dict[str, float]]],
    mixer_config: RewardMixerConfig,
) -> EvalResult:
    """Run evaluation and return scalarized reward metrics."""

    reward_vectors = reward_fn(policy)
    scalar_rewards, mixer_stats = mix_reward_vectors(reward_vectors, mixer_config)
    metrics = _summarize_reward_vectors(reward_vectors)
    metrics.update({f"eval/{key}": value for key, value in mixer_stats.items()})
    metrics["eval/scalar_mean"] = scalar_rewards.mean().item()
    metrics["eval/scalar_std"] = scalar_rewards.std().item()
    return EvalResult(metrics=metrics, scalar_rewards=scalar_rewards)


class EvalHook:
    """Hook to run evaluation periodically during training."""

    def __init__(
        self,
        reward_fn: Callable[[Policy], List[Dict[str, float]]],
        mixer_config: RewardMixerConfig,
        interval_steps: int = 100,
    ) -> None:
        self.reward_fn = reward_fn
        self.mixer_config = mixer_config
        self.interval_steps = interval_steps

    def maybe_run(self, policy: Policy, step: int) -> EvalResult | None:
        if step % self.interval_steps != 0:
            return None
        return run_eval(policy, self.reward_fn, self.mixer_config)
