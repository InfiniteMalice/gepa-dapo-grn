"""Evaluation hooks for GEPA-style feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import torch

from gepa_dapo_grn.config import RewardMixerConfig
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback
from gepa_dapo_grn.policy_interfaces import Policy
from gepa_dapo_grn.reward_mixers import mix_reward_vectors


@dataclass
class EvalResult:
    """Container for evaluation metrics."""

    metrics: Dict[str, float]
    scalar_rewards: torch.Tensor


def _summarize_feedback(feedbacks: Iterable[GEPAFeedback]) -> Dict[str, float]:
    rewards: Dict[str, List[float]] = {}
    tags: Dict[str, List[float]] = {}
    abstained = []
    for feedback in feedbacks:
        for key, value in feedback.rewards.items():
            rewards.setdefault(key, []).append(float(value))
        for key, value in feedback.tags.items():
            tags.setdefault(key, []).append(float(value))
        abstained.append(float(feedback.abstained))

    summary: Dict[str, float] = {}
    for key, values in rewards.items():
        tensor = torch.tensor(values, dtype=torch.float32)
        summary[f"eval/reward/{key}/mean"] = tensor.mean().item()
        summary[f"eval/reward/{key}/std"] = tensor.std().item()
    for key, values in tags.items():
        tensor = torch.tensor(values, dtype=torch.float32)
        summary[f"eval/tag/{key}/mean"] = tensor.mean().item()
        summary[f"eval/tag/{key}/std"] = tensor.std().item()
    if abstained:
        abstained_tensor = torch.tensor(abstained, dtype=torch.float32)
        summary["eval/abstained/mean"] = abstained_tensor.mean().item()
    return summary


def run_eval(
    policy: Policy,
    feedback_fn: Callable[[Policy], List[GEPAFeedback]],
    mixer_config: RewardMixerConfig,
) -> EvalResult:
    """Run evaluation and return scalarized reward metrics."""

    feedbacks = feedback_fn(policy)
    reward_vectors = [feedback.rewards for feedback in feedbacks]
    scalar_rewards, mixer_stats = mix_reward_vectors(reward_vectors, mixer_config)
    metrics = _summarize_feedback(feedbacks)
    metrics.update({f"eval/{key}": value for key, value in mixer_stats.items()})
    metrics["eval/scalar_mean"] = scalar_rewards.mean().item()
    metrics["eval/scalar_std"] = scalar_rewards.std().item()
    return EvalResult(metrics=metrics, scalar_rewards=scalar_rewards)


class EvalHook:
    """Hook to run evaluation periodically during training."""

    def __init__(
        self,
        feedback_fn: Callable[[Policy], List[GEPAFeedback]],
        mixer_config: RewardMixerConfig,
        interval_steps: int = 100,
    ) -> None:
        self.feedback_fn = feedback_fn
        self.mixer_config = mixer_config
        self.interval_steps = interval_steps

    def maybe_run(self, policy: Policy, step: int) -> EvalResult | None:
        if step % self.interval_steps != 0:
            return None
        return run_eval(policy, self.feedback_fn, self.mixer_config)
