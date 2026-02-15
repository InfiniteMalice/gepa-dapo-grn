"""DAPO training core utilities."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from gepa_dapo_grn.config import DAPOConfig, GRNConfig, RewardMixerConfig
from gepa_dapo_grn.curriculum import CurriculumTracker
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback
from gepa_dapo_grn.grn import maybe_wrap_policy_heads, restore_policy_heads
from gepa_dapo_grn.policy_interfaces import Policy
from gepa_dapo_grn.reward_mixers import mix_reward_vectors
from gepa_dapo_grn.safety_controller import SafetyController


@dataclass
class DAPOBatch:
    """Batch data needed for a DAPO training step."""

    inputs: Dict[str, torch.Tensor]
    actions: torch.Tensor
    logp_old: torch.Tensor


@dataclass
class DAPOStepResult:
    """Outputs from a DAPO training step."""

    loss: torch.Tensor
    metrics: Dict[str, float]


def _group_normalize(values: torch.Tensor, group_size: int) -> torch.Tensor:
    if values.numel() % group_size != 0:
        raise ValueError("Batch size must be divisible by group_size")
    grouped = values.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    normalized = (grouped - mean) / std
    return normalized.view(-1)


def _clip_advantages(advantages: torch.Tensor, clip_value: float) -> torch.Tensor:
    return torch.clamp(advantages, min=-clip_value, max=clip_value)


def _soft_ratio_weights(log_ratio: torch.Tensor, temperature: float) -> torch.Tensor:
    safe_temperature = max(1e-6, float(temperature))
    weights = torch.exp(-torch.abs(log_ratio) / safe_temperature)
    return torch.clamp(weights, min=0.0, max=1.0)


def _approx_kl(logp_new: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    return (logp_new - logp_ref).mean()


def _broadcast_rewards(rewards: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if rewards.ndim == target.ndim:
        return rewards
    if rewards.ndim == 1 and target.ndim > 1:
        shape = (rewards.shape[0],) + (1,) * (target.ndim - 1)
        return rewards.view(shape).expand_as(target)
    if rewards.ndim > target.ndim:
        raise ValueError("rewards cannot have higher rank than target for broadcasting")
    trailing_dims = target.ndim - rewards.ndim
    shape = rewards.shape + (1,) * trailing_dims
    return rewards.view(shape).expand_as(target)


_FEEDBACK_BATCH_MISMATCH = "feedbacks must align with batch size"


class DAPOTrainer:
    """Decoupled Advantage Policy Optimization trainer with GEPA feedback support."""

    def __init__(
        self,
        policy: Policy,
        optimizer: torch.optim.Optimizer,
        config: Optional[DAPOConfig] = None,
        grn_config: Optional[GRNConfig] = None,
        reward_mixer: Optional[RewardMixerConfig] = None,
        curriculum: Optional[CurriculumTracker] = None,
        safety_controller: Optional[SafetyController] = None,
    ) -> None:
        self.policy = policy
        self.optimizer = optimizer
        self.config = config or DAPOConfig()
        self.grn_config = grn_config or GRNConfig()
        self.reward_mixer = reward_mixer or RewardMixerConfig()
        self.curriculum = curriculum or CurriculumTracker()
        self.safety_controller = safety_controller or SafetyController()
        self.ref_policy = policy.clone()
        self._original_heads: Dict[str, nn.Module] = {}
        self._sync_grn_wrapping()

    def update_reference(self) -> None:
        """Refresh the reference policy used for KL regularization."""

        self.ref_policy = self.policy.clone()

    def _sync_grn_wrapping(self) -> None:
        if self.grn_config.enabled:
            if not self._original_heads:
                self._original_heads = maybe_wrap_policy_heads(self.policy, self.grn_config)
        else:
            if self._original_heads:
                restore_policy_heads(self.policy, self._original_heads)
                self._original_heads = {}

    def _compute_policy_loss(
        self,
        logp_new: torch.Tensor,
        logp_old: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        ratio = torch.exp(logp_new - logp_old)
        advantages_clipped = _clip_advantages(advantages, self.config.clip_advantage)

        if self.config.use_soft_gating:
            weights = _soft_ratio_weights(logp_new - logp_old, self.config.gating_temperature)
            return -((ratio * advantages_clipped) * weights).mean()

        ratio_clipped = torch.clamp(
            ratio,
            1.0 - self.config.clip_ratio,
            1.0 + self.config.clip_ratio,
        )
        return -(ratio_clipped * advantages_clipped).mean()

    def _compute_value_loss(self, values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        return 0.5 * (returns - values).pow(2).mean()

    def _update_kl_coeff(self, kl_value: float) -> None:
        if not self.config.adaptive_kl:
            return
        multiplier = torch.exp(
            torch.tensor((kl_value - self.config.target_kl) / self.config.kl_horizon)
        ).item()
        self.config.kl_coeff *= multiplier

    def train_step(
        self,
        batch: DAPOBatch,
        feedbacks: List[GEPAFeedback],
        extra_loss: Optional[torch.Tensor] = None,
    ) -> DAPOStepResult:
        """Run a single DAPO training step."""

        if len(feedbacks) != batch.actions.shape[0]:
            raise ValueError(_FEEDBACK_BATCH_MISMATCH)

        for feedback in feedbacks:
            task_id = feedback.meta.get("task_id", "default")
            self.curriculum.update(task_id, feedback)
            self.safety_controller.update(feedback)

        self.safety_controller.adjust_configs(self.config, self.grn_config)
        self._sync_grn_wrapping()

        reward_vectors = [feedback.rewards for feedback in feedbacks]
        scalar_rewards, reward_stats = mix_reward_vectors(reward_vectors, self.reward_mixer)

        self.policy.train()
        outputs = self.policy(**batch.inputs)
        logp_new = self.policy.logprobs(batch.actions, **batch.inputs)
        with torch.no_grad():
            logp_ref = self.ref_policy.logprobs(batch.actions, **batch.inputs)

        rewards_for_adv = _broadcast_rewards(scalar_rewards, logp_new)
        advantages = rewards_for_adv
        value_loss = torch.tensor(0.0, device=logp_new.device)
        if outputs.values is not None:
            returns = _broadcast_rewards(scalar_rewards, outputs.values)
            if returns.shape == outputs.values.shape:
                advantages = returns - outputs.values
                value_loss = self._compute_value_loss(outputs.values, returns)
            else:
                warnings.warn(
                    "Value head shape mismatch; skipping value loss computation.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if self.config.group_size:
            flat_adv = advantages.view(-1)
            normalized = _group_normalize(flat_adv, self.config.group_size)
            advantages = normalized.view_as(advantages)

        policy_loss = self._compute_policy_loss(logp_new, batch.logp_old, advantages)
        kl_value = _approx_kl(logp_new, logp_ref)
        kl_loss = self.config.kl_coeff * kl_value

        total_loss = policy_loss + kl_loss + self.config.value_coef * value_loss
        if extra_loss is not None:
            total_loss = total_loss + extra_loss

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self._update_kl_coeff(kl_value.item())

        metrics = {
            "loss/total": total_loss.item(),
            "loss/policy": policy_loss.item(),
            "loss/value": value_loss.item() if value_loss.numel() else 0.0,
            "loss/kl": kl_loss.item(),
            "kl/value": kl_value.item(),
            "kl/coeff": self.config.kl_coeff,
            "reward/mean": scalar_rewards.mean().item(),
            "reward/std": scalar_rewards.std(unbiased=False).item(),
            "policy/use_soft_gating": float(self.config.use_soft_gating),
            "policy/gating_temperature": float(self.config.gating_temperature),
        }
        metrics.update(reward_stats)
        metrics.update(self.safety_controller.describe())
        return DAPOStepResult(loss=total_loss, metrics=metrics)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize trainer state."""

        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": asdict(self.config),
            "grn_config": asdict(self.grn_config),
            "curriculum_state": self.curriculum.tasks,
            "safety_state": {
                "reward_ema": self.safety_controller.state.reward_ema,
                "tag_ema": self.safety_controller.state.tag_ema,
                "verifier_ema": self.safety_controller.state.verifier_ema,
                "count": self.safety_controller.state.count,
            },
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load trainer state."""

        self.policy.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])
        if "config" in state:
            config_state = state["config"]
            self.config = (
                DAPOConfig(**config_state) if isinstance(config_state, dict) else config_state
            )
        if "grn_config" in state:
            grn_state = state["grn_config"]
            self.grn_config = GRNConfig(**grn_state) if isinstance(grn_state, dict) else grn_state
        if "curriculum_state" in state:
            self.curriculum.tasks = state["curriculum_state"]
        if "safety_state" in state:
            safety = state["safety_state"]
            self.safety_controller.state.reward_ema = safety.get("reward_ema", {})
            self.safety_controller.state.tag_ema = safety.get("tag_ema", {})
            self.safety_controller.state.verifier_ema = safety.get("verifier_ema", {})
            self.safety_controller.state.count = safety.get("count", 0)
        self._sync_grn_wrapping()
