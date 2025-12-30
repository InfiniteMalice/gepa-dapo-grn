"""Reward mixing utilities for vector-valued rewards."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, List, Tuple

import torch

from gepa_dapo_grn.config import RewardMixerConfig


def _collect_keys(
    reward_vectors: Iterable[Dict[str, float]],
    config: RewardMixerConfig,
) -> List[str]:
    if config.weights:
        return list(config.weights.keys())
    keys = set()
    for vector in reward_vectors:
        keys.update(vector.keys())
    return sorted(keys)


def _weights_for_keys(keys: Iterable[str], config: RewardMixerConfig) -> torch.Tensor:
    weights = []
    for key in keys:
        weights.append(config.weights.get(key, config.default_weight))
    return torch.tensor(weights, dtype=torch.float32)


def mix_reward_vectors(
    reward_vectors: List[Dict[str, float]],
    config: RewardMixerConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Convert vector rewards into scalar rewards.

    Args:
        reward_vectors: List of reward dictionaries per sample.
        config: Reward mixer configuration.

    Returns:
        Tuple of scalar rewards tensor and a dictionary of summary statistics.
    """

    if not reward_vectors:
        raise ValueError("reward_vectors must contain at least one element")

    keys = _collect_keys(reward_vectors, config)
    rewards = torch.zeros((len(reward_vectors), len(keys)), dtype=torch.float32)
    for row, vector in enumerate(reward_vectors):
        for col, key in enumerate(keys):
            rewards[row, col] = float(vector.get(key, 0.0))

    stats: Dict[str, float] = {}
    if config.normalize:
        mean = rewards.mean(dim=0, keepdim=True)
        std = rewards.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        rewards = (rewards - mean) / std
        stats.update({f"reward_mean/{key}": mean[0, idx].item() for idx, key in enumerate(keys)})
        stats.update({f"reward_std/{key}": std[0, idx].item() for idx, key in enumerate(keys)})

    weights = _weights_for_keys(keys, config)
    scalar = rewards @ weights

    if config.clip_min is not None or config.clip_max is not None:
        scalar = torch.clamp(scalar, min=config.clip_min, max=config.clip_max)

    stats.update({f"reward_weight/{key}": weight for key, weight in zip(keys, weights.tolist())})
    stats["reward_scalar/mean"] = scalar.mean().item()
    stats["reward_scalar/std"] = scalar.std(unbiased=False).item()
    return scalar, stats


def describe_reward_mixer(config: RewardMixerConfig) -> dict:
    """Return a JSON-serializable description of the reward mixer configuration."""

    return asdict(config)
