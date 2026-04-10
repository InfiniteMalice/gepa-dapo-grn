"""Configuration dataclasses for DAPO training, GRN, and reward mixing."""

from __future__ import annotations

import math
from dataclasses import field
from numbers import Real
from typing import Dict, List, Optional

from gepa_dapo_grn._compat import dataclass


@dataclass(slots=True)
class TrainerBackendConfig:
    """Top-level trainer backend selection."""

    backend: str = "dapo"

    def validated_backend(self) -> str:
        if not isinstance(self.backend, str):
            raise ValueError("backend must be one of: 'dapo', 'maxrl'")
        backend = self.backend.strip().lower()
        if backend not in {"dapo", "maxrl"}:
            raise ValueError("backend must be one of: 'dapo', 'maxrl'")
        return backend


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


@dataclass(slots=True)
class MaxRLConfig:
    """Configuration for verifier-first MaxRL-inspired optimization."""

    enabled: bool = False
    num_samples: int = 4
    success_tag_key: str = "verifier_success"
    use_binary_success_only: bool = True
    normalize_by_successes: bool = True
    min_success_count: int = 1
    max_success_weight: float = 10.0
    zero_success_kl_coeff: float = 0.05
    grad_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        if not isinstance(self.use_binary_success_only, bool):
            raise ValueError("use_binary_success_only must be a bool")
        if not isinstance(self.normalize_by_successes, bool):
            raise ValueError("normalize_by_successes must be a bool")
        if not isinstance(self.success_tag_key, str) or not self.success_tag_key.strip():
            raise ValueError("success_tag_key must be a non-empty string")
        self.success_tag_key = self.success_tag_key.strip()

        if not isinstance(self.num_samples, int) or isinstance(self.num_samples, bool):
            raise ValueError("num_samples must be an integer")
        if self.num_samples < 1:
            raise ValueError("num_samples must be >= 1")

        if not isinstance(self.min_success_count, int) or isinstance(self.min_success_count, bool):
            raise ValueError("min_success_count must be an integer")
        if self.min_success_count < 0:
            raise ValueError("min_success_count must be >= 0")

        if not isinstance(self.max_success_weight, Real) or isinstance(
            self.max_success_weight, bool
        ):
            raise ValueError("max_success_weight must be numeric")
        if not math.isfinite(self.max_success_weight):
            raise ValueError("max_success_weight must be numeric")
        if self.max_success_weight < 0.0:
            raise ValueError("max_success_weight must be >= 0.0")
        if not isinstance(self.zero_success_kl_coeff, Real) or isinstance(
            self.zero_success_kl_coeff, bool
        ):
            raise ValueError("zero_success_kl_coeff must be numeric")
        if not math.isfinite(self.zero_success_kl_coeff):
            raise ValueError("zero_success_kl_coeff must be numeric")
        if self.zero_success_kl_coeff < 0.0:
            raise ValueError("zero_success_kl_coeff must be >= 0.0")
        if not isinstance(self.grad_clip_norm, Real) or isinstance(self.grad_clip_norm, bool):
            raise ValueError("grad_clip_norm must be numeric")
        if not math.isfinite(self.grad_clip_norm):
            raise ValueError("grad_clip_norm must be numeric")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be > 0.0")