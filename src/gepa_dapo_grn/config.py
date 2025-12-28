"""Configuration dataclasses for DAPO training, GRN, and reward mixing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(slots=True)
class DAPOConfig:
    """Configuration for DAPO-style policy optimization.

    Args:
        clip_ratio: Clipping range for the policy ratio.
        clip_advantage: Clipping range for advantages.
        kl_coeff: Initial KL penalty coefficient.
        target_kl: Target KL value for adaptive coefficient updates.
        kl_horizon: Update horizon for adaptive KL coefficient.
        adaptive_kl: Whether to adapt KL coefficient toward the target.
        max_grad_norm: Maximum gradient norm for clipping.
        value_coef: Weighting for the value loss term.
        group_size: Optional group size for group-based advantage normalization.
    """

    clip_ratio: float = 0.2
    clip_advantage: float = 5.0
    kl_coeff: float = 0.1
    target_kl: float = 0.01
    kl_horizon: float = 1000.0
    adaptive_kl: bool = True
    max_grad_norm: float = 1.0
    value_coef: float = 0.5
    group_size: Optional[int] = None


@dataclass(slots=True)
class GRNConfig:
    """Configuration for Global Response Normalization modules."""

    enabled: bool = False
    apply_to_policy: bool = False
    apply_to_value: bool = False
    epsilon: float = 1e-6


@dataclass(slots=True)
class RewardMixerConfig:
    """Configuration for vector reward scalarization.

    Args:
        weights: Optional mapping of reward keys to weights.
        normalize: Whether to z-score normalize reward dimensions before mixing.
        clip_min: Optional minimum clip value for the scalar reward.
        clip_max: Optional maximum clip value for the scalar reward.
        default_weight: Weight to apply for keys without explicit weights.
    """

    weights: Dict[str, float] = field(default_factory=dict)
    normalize: bool = True
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None
    default_weight: float = 1.0
