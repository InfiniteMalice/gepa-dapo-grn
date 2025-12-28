"""DAPO training core utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn

from gepa_dapo_grn.config import DAPOConfig
from gepa_dapo_grn.policy_interfaces import Policy


@dataclass
class DAPOBatch:
    """Batch data needed for a DAPO training step."""

    inputs: Dict[str, torch.Tensor]
    actions: torch.Tensor
    logp_old: torch.Tensor
    advantages: torch.Tensor
    returns: Optional[torch.Tensor] = None


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
    std = grouped.std(dim=1, keepdim=True).clamp_min(1e-6)
    normalized = (grouped - mean) / std
    return normalized.view(-1)


def _clip_advantages(advantages: torch.Tensor, clip_value: float) -> torch.Tensor:
    return torch.clamp(advantages, min=-clip_value, max=clip_value)


def _approx_kl(logp_new: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    return (logp_new - logp_ref).mean()


class DAPOTrainer:
    """Decoupled Advantage Policy Optimization trainer."""

    def __init__(
        self,
        policy: Policy,
        optimizer: torch.optim.Optimizer,
        config: Optional[DAPOConfig] = None,
    ) -> None:
        self.policy = policy
        self.optimizer = optimizer
        self.config = config or DAPOConfig()
        self.kl_coeff = self.config.kl_coeff
        self.ref_policy = policy.clone()

    def update_reference(self) -> None:
        """Refresh the reference policy used for KL regularization."""

        self.ref_policy = self.policy.clone()

    def _compute_policy_loss(
        self,
        logp_new: torch.Tensor,
        logp_old: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        ratio = torch.exp(logp_new - logp_old)
        ratio_clipped = torch.clamp(
            ratio,
            1.0 - self.config.clip_ratio,
            1.0 + self.config.clip_ratio,
        )
        advantages_clipped = _clip_advantages(advantages, self.config.clip_advantage)
        return -(ratio_clipped * advantages_clipped).mean()

    def _compute_value_loss(self, values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        return 0.5 * (returns - values).pow(2).mean()

    def _update_kl_coeff(self, kl_value: float) -> None:
        if not self.config.adaptive_kl:
            return
        multiplier = torch.exp(
            torch.tensor((kl_value - self.config.target_kl) / self.config.kl_horizon)
        ).item()
        self.kl_coeff *= multiplier

    def train_step(
        self,
        batch: DAPOBatch,
        extra_loss: Optional[torch.Tensor] = None,
    ) -> DAPOStepResult:
        """Run a single DAPO training step."""

        self.policy.train()
        outputs = self.policy(**batch.inputs)
        logp_new = self.policy.log_probs(batch.actions, **batch.inputs)
        with torch.no_grad():
            logp_ref = self.ref_policy.log_probs(batch.actions, **batch.inputs)

        advantages = batch.advantages
        if self.config.group_size:
            advantages = _group_normalize(advantages, self.config.group_size)

        policy_loss = self._compute_policy_loss(logp_new, batch.logp_old, advantages)
        kl_value = _approx_kl(logp_new, logp_ref)
        kl_loss = self.kl_coeff * kl_value

        value_loss = torch.tensor(0.0, device=logp_new.device)
        if outputs.values is not None and batch.returns is not None:
            value_loss = self._compute_value_loss(outputs.values, batch.returns)

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
            "kl/coeff": self.kl_coeff,
        }
        return DAPOStepResult(loss=total_loss, metrics=metrics)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize trainer state."""

        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "kl_coeff": self.kl_coeff,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load trainer state."""

        self.policy.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.kl_coeff = float(state["kl_coeff"])
