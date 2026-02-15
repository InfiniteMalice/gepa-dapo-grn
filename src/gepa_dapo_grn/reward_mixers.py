"""Reward mixing utilities for vector-valued rewards.

Warning:
    Do not include deception penalties as built-in defaults. Keep deception-like
    signals as tags/safety controls unless an application explicitly chooses
    custom reward weights externally.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch

from gepa_dapo_grn.config import RewardMixerConfig


def _collect_keys(
    reward_vectors: Iterable[Dict[str, float]],
    config: RewardMixerConfig,
) -> List[str]:
    if config.weights:
        return sorted(config.weights.keys())
    keys = set()
    for vector in reward_vectors:
        keys.update(vector.keys())
    return sorted(keys)


def _weights_for_keys(keys: Iterable[str], config: RewardMixerConfig) -> torch.Tensor:
    weights = []
    for key in keys:
        weights.append(config.weights.get(key, config.default_weight))
    return torch.tensor(weights, dtype=torch.float32)


def multi_objective_scalarize(
    reward_vectors: List[Dict[str, float]],
    *,
    weights: Mapping[str, float],
    normalize: bool = True,
    gates: Optional[Mapping[str, Tuple[Optional[float], Optional[float]]]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Key-agnostic scalarization helper with optional per-dimension clamps."""

    if not reward_vectors:
        raise ValueError("reward_vectors must contain at least one element")
    keys = sorted(weights.keys())
    rewards = torch.zeros((len(reward_vectors), len(keys)), dtype=torch.float32)
    for row, vector in enumerate(reward_vectors):
        for col, key in enumerate(keys):
            value = float(vector.get(key, 0.0))
            if gates and key in gates:
                low, high = gates[key]
                if low is not None:
                    value = max(low, value)
                if high is not None:
                    value = min(high, value)
            rewards[row, col] = value

    stats: Dict[str, float] = {}
    if normalize and keys:
        mean = rewards.mean(dim=0, keepdim=True)
        std = rewards.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        rewards = (rewards - mean) / std
        stats.update({f"reward_mean/{key}": mean[0, idx].item() for idx, key in enumerate(keys)})
        stats.update({f"reward_std/{key}": std[0, idx].item() for idx, key in enumerate(keys)})

    weights_tensor = torch.tensor([float(weights[key]) for key in keys], dtype=torch.float32)
    scalar = (
        rewards @ weights_tensor if keys else torch.zeros(len(reward_vectors), dtype=torch.float32)
    )
    stats.update(
        {
            f"reward_weight/{key}": float(weight)
            for key, weight in zip(keys, weights_tensor.tolist())
        }
    )
    stats["reward_scalar/mean"] = scalar.mean().item()
    stats["reward_scalar/std"] = scalar.std(unbiased=False).item()
    return scalar, stats


def mix_reward_vectors(
    reward_vectors: List[Dict[str, float]],
    config: RewardMixerConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Convert vector rewards into scalar rewards.

    Unknown keys are ignored by default (default_weight=0.0), unless configured.
    """

    if not reward_vectors:
        raise ValueError("reward_vectors must contain at least one element")

    keys = _collect_keys(reward_vectors, config)
    weights = {key: config.weights.get(key, config.default_weight) for key in keys}
    scalar, stats = multi_objective_scalarize(
        reward_vectors,
        weights=weights,
        normalize=config.normalize,
    )

    if config.clip_min is not None or config.clip_max is not None:
        scalar = torch.clamp(scalar, min=config.clip_min, max=config.clip_max)
        stats["reward_scalar/mean"] = scalar.mean().item()
        stats["reward_scalar/std"] = scalar.std(unbiased=False).item()

    return scalar, stats


def describe_reward_mixer(config: RewardMixerConfig) -> dict:
    """Return a JSON-serializable description of the reward mixer configuration."""

    return asdict(config)
