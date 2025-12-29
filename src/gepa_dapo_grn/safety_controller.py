"""Safety controller that adapts optimization settings based on GEPA feedback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from gepa_dapo_grn.config import DAPOConfig, GRNConfig
from gepa_dapo_grn.ema_utils import update_ema
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback


@dataclass(slots=True)
class SafetyState:
    """EMA statistics tracked across all feedback."""

    reward_ema: Dict[str, float] = field(default_factory=dict)
    tag_ema: Dict[str, float] = field(default_factory=dict)
    count: int = 0


class SafetyController:
    """Adjust training configs based on safety-related feedback."""

    def __init__(
        self,
        decay: float = 0.9,
        reward_risk_weights: Optional[Dict[str, float]] = None,
        tag_risk_weights: Optional[Dict[str, float]] = None,
        risk_tolerance: float = 0.0,
        adjustment_scale: float = 0.5,
        min_clip_ratio: float = 0.05,
        max_kl_coeff: float = 10.0,
        grn_enable_threshold: float = 0.2,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be between 0 and 1")
        self.decay = decay
        self.reward_risk_weights = reward_risk_weights or {}
        self.tag_risk_weights = tag_risk_weights or {}
        self.risk_tolerance = risk_tolerance
        self.adjustment_scale = adjustment_scale
        self.min_clip_ratio = min_clip_ratio
        self.max_kl_coeff = max_kl_coeff
        self.grn_enable_threshold = grn_enable_threshold
        self.state = SafetyState()
        self._baseline_clip_ratio: Optional[float] = None
        self._baseline_kl_coeff: Optional[float] = None
        self._baseline_grn_enabled: Optional[bool] = None

    def update(self, feedback: GEPAFeedback) -> SafetyState:
        """Update EMA statistics based on feedback."""

        for key, value in feedback.rewards.items():
            current = self.state.reward_ema.get(key, float(value))
            self.state.reward_ema[key] = update_ema(current, float(value), self.decay)
        for key, value in feedback.tags.items():
            current = self.state.tag_ema.get(key, float(value))
            self.state.tag_ema[key] = update_ema(current, float(value), self.decay)
        self.state.count += 1
        return self.state

    def _risk_score(self) -> float:
        score = 0.0
        for key, weight in self.reward_risk_weights.items():
            score += weight * self.state.reward_ema.get(key, 0.0)
        for key, weight in self.tag_risk_weights.items():
            score += weight * self.state.tag_ema.get(key, 0.0)
        return score

    def adjust_configs(self, dapo_config: DAPOConfig, grn_config: GRNConfig) -> None:
        """Mutate configs in place based on current safety state."""

        if self._baseline_clip_ratio is None:
            self._baseline_clip_ratio = dapo_config.clip_ratio
        if self._baseline_kl_coeff is None:
            self._baseline_kl_coeff = dapo_config.kl_coeff
        if self._baseline_grn_enabled is None:
            self._baseline_grn_enabled = grn_config.enabled

        risk_score = self._risk_score()
        risk_delta = max(0.0, risk_score - self.risk_tolerance)
        adjustment = 1.0 + self.adjustment_scale * risk_delta

        dapo_config.clip_ratio = max(
            self.min_clip_ratio,
            self._baseline_clip_ratio / adjustment,
        )
        dapo_config.kl_coeff = min(
            self.max_kl_coeff,
            self._baseline_kl_coeff * adjustment,
        )

        if self._baseline_grn_enabled:
            grn_config.enabled = True
        else:
            grn_config.enabled = risk_delta > self.grn_enable_threshold

    def describe(self) -> Dict[str, float]:
        """Return a summary of safety-related EMA statistics."""

        summary = {
            **{f"safety_reward_ema/{key}": value for key, value in self.state.reward_ema.items()},
            **{f"safety_tag_ema/{key}": value for key, value in self.state.tag_ema.items()},
            "safety_count": float(self.state.count),
            "safety_risk_score": self._risk_score(),
        }
        return summary
