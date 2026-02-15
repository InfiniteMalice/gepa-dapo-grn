"""Configuration dataclasses for DAPO training, GRN, and reward mixing."""

from __future__ import annotations

from dataclasses import field
from typing import Dict, List, Optional

from gepa_dapo_grn._compat import dataclass


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
        use_soft_gating: Use smooth ratio gating instead of hard clipping.
        gating_temperature: Temperature for soft ratio gating.
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
    use_soft_gating: bool = False
    gating_temperature: float = 1.0


@dataclass(slots=True)
class GRNConfig:
    """Configuration for Global Response Normalization modules."""

    enabled: bool = False
    apply_to_policy: bool = False
    apply_to_value: bool = False
    epsilon: float = 1e-6
    include_modules: List[str] = field(default_factory=list)
    exclude_modules: List[str] = field(default_factory=list)
    protect_probe_modules: bool = True
    probe_name_patterns: List[str] = field(
        default_factory=lambda: ["probe", "interpret", "explainability", "attribution"]
    )


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
    default_weight: float = 0.0
