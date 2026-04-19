"""MaxRL-inspired verifier-first trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn

from gepa_dapo_grn.config import GRNConfig, MaxRLConfig
from gepa_dapo_grn.curriculum import CurriculumTracker
from gepa_dapo_grn.dapo_core import _approx_kl
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback
from gepa_dapo_grn.grn import maybe_wrap_policy_heads, restore_policy_heads
from gepa_dapo_grn.policy_interfaces import Policy
from gepa_dapo_grn.safety_controller import SafetyController

BACKEND_METRIC_FLAG = 1.0


@dataclass
class MaxRLBatch:
    """Batch data for MaxRL training."""

    inputs: Dict[str, torch.Tensor]
    actions: torch.Tensor
    task_ids: List[str]


@dataclass
class MaxRLStepResult:
    """Outputs from a MaxRL training step."""

    loss: torch.Tensor
    metrics: Dict[str, float]


class MaxRLTrainer:
    """Verifier-first trainer using successful rollouts as max-likelihood targets."""

    def __init__(
        self,
        policy: Policy,
        optimizer: torch.optim.Optimizer,
        config: Optional[MaxRLConfig] = None,
        grn_config: Optional[GRNConfig] = None,
        curriculum: Optional[CurriculumTracker] = None,
        safety_controller: Optional[SafetyController] = None,
        reference_policy: Optional[Policy] = None,
    ) -> None:
        self.policy = policy
        self.optimizer = optimizer
        self.config = config or MaxRLConfig(enabled=True)
        if not self.config.enabled:
            raise ValueError("MaxRL backend requires MaxRLConfig.enabled=True")
        self.grn_config = grn_config or GRNConfig()
        self.curriculum = curriculum or CurriculumTracker()
        self.safety_controller = safety_controller or SafetyController()
        self._initial_reference_policy = reference_policy
        self._original_heads: Dict[str, nn.Module] = {}
        self._sync_grn_wrapping()
        self._refresh_reference()

    def update_reference(self) -> None:
        self._refresh_reference()

    def _refresh_reference(self, source_policy: Optional[Policy] = None) -> None:
        # Intentional order: clone after potential GRN wrapping so KL compares against the
        # post-wrapped structure used for MaxRL updates.
        # When an explicit constructor reference is provided, retain it as the source for
        # future refreshes triggered by update_reference() and GRN state transitions.
        if source_policy is None:
            source_policy = self._initial_reference_policy or self.policy
        source = source_policy.clone()
        if self.grn_config.enabled:
            maybe_wrap_policy_heads(source, self.grn_config)
        self.ref_policy = source
        self.ref_policy.eval()
        for parameter in self.ref_policy.parameters():
            parameter.requires_grad_(False)
        self._ref_grn_enabled = self.grn_config.enabled

    def _sync_grn_wrapping(self) -> None:
        if self.grn_config.enabled and not self._original_heads:
            self._original_heads = maybe_wrap_policy_heads(self.policy, self.grn_config)
        if not self.grn_config.enabled and self._original_heads:
            restore_policy_heads(self.policy, self._original_heads)
            self._original_heads = {}

    def _success_value(self, feedback: GEPAFeedback) -> float:
        if self.config.success_tag_key in feedback.tags:
            value = float(feedback.tags[self.config.success_tag_key])
        elif self.config.success_tag_key in feedback.verifier:
            value = float(feedback.verifier[self.config.success_tag_key])
        elif "verifier_pass" in feedback.tags:
            value = float(feedback.tags["verifier_pass"])
        elif "verifier_pass" in feedback.verifier:
            value = float(feedback.verifier["verifier_pass"])
        else:
            value = 0.0
        if self.config.use_binary_success_only:
            return 1.0 if value >= 0.5 else 0.0
        return min(self.config.max_success_weight, max(0.0, value))

    def train_step(self, batch: MaxRLBatch, feedbacks: List[GEPAFeedback]) -> MaxRLStepResult:
        batch_size = int(batch.actions.shape[0])
        if batch_size == 0 or len(feedbacks) == 0 or len(batch.task_ids) == 0:
            raise ValueError("batch is empty; cannot compute loss")
        if len(feedbacks) != batch_size:
            raise ValueError(
                f"feedbacks length {len(feedbacks)} does not match batch_size {batch_size}"
            )
        if len(batch.task_ids) != batch_size:
            raise ValueError(
                f"task_ids length {len(batch.task_ids)} does not match batch_size {batch_size}"
            )

        inputs = dict(batch.inputs)
        if "batch_size" in inputs:
            input_batch_size = inputs["batch_size"]
            if isinstance(input_batch_size, torch.Tensor):
                input_batch_size = input_batch_size.item()
            if int(input_batch_size) != batch_size:
                raise ValueError("batch.inputs['batch_size'] must match batch.actions.shape[0]")
        else:
            inputs["batch_size"] = torch.tensor(batch_size, device=batch.actions.device)
        expected_batch = int(batch_size)
        for key, value in inputs.items():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim >= 1
                and value.shape[0] != expected_batch
            ):
                raise ValueError(
                    f"batch.inputs['{key}'] leading dimension {value.shape[0]} "
                    f"does not match batch_size {expected_batch}"
                )

        for task_id, feedback in zip(batch.task_ids, feedbacks):
            self.curriculum.update(task_id, feedback)
            self.safety_controller.update(feedback)
        self.safety_controller.adjust_grn_config(self.grn_config)
        self._sync_grn_wrapping()
        if self._ref_grn_enabled != self.grn_config.enabled:
            self._refresh_reference()

        self.policy.train()
        logp_new = self.policy.logprobs(batch.actions, **inputs)
        with torch.no_grad():
            logp_ref = self.ref_policy.logprobs(batch.actions, **inputs)

        success_weights = torch.tensor(
            [self._success_value(fb) for fb in feedbacks],
            dtype=logp_new.dtype,
            device=logp_new.device,
        )
        success_weights = torch.clamp(success_weights, max=float(self.config.max_success_weight))
        success_count = int((success_weights > 0).sum().item())
        effective_batch_size = max(1, int(success_weights.numel()))
        success_rate = float(success_count / effective_batch_size)
        zero_success = float(success_count == 0)
        insufficient_success = float(0 < success_count < self.config.min_success_count)

        kl_value = _approx_kl(logp_new, logp_ref)
        kl_fallback = self.config.zero_success_kl_coeff * torch.clamp(kl_value, min=0.0)

        if success_count >= self.config.min_success_count:
            denom = float(
                success_count if self.config.normalize_by_successes else effective_batch_size
            )
            mle_loss = -((success_weights * logp_new).sum() / max(1.0, denom))
        elif success_count > 0:
            mle_loss = kl_fallback
        else:
            mle_loss = torch.zeros((), dtype=logp_new.dtype, device=logp_new.device)

        zero_success_kl = (
            kl_fallback
            if zero_success
            else torch.zeros((), dtype=logp_new.dtype, device=logp_new.device)
        )
        total_loss = mle_loss + zero_success_kl

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()

        task_groups: Dict[str, List[float]] = {}
        coverage_values: List[float] = []
        for task_id, feedback, success in zip(batch.task_ids, feedbacks, success_weights.tolist()):
            task_groups.setdefault(task_id, []).append(float(success))
            verifier_coverage = feedback.verifier.get("verifier_coverage", 1.0)
            coverage_values.append(float(feedback.tags.get("verifier_coverage", verifier_coverage)))
        saturated_tasks = sum(
            int(sum(values) >= 1.0 and len(values) >= self.config.num_samples)
            for values in task_groups.values()
        )

        metrics = {
            "backend": BACKEND_METRIC_FLAG,  # backend-presence flag for dashboard grouping.
            "maxrl/objective": float((-mle_loss).item()),
            "maxrl/loss": float(total_loss.item()),
            "maxrl/success_count": float(success_count),
            "maxrl/success_rate": success_rate,
            "maxrl/insufficient_success_rate": insufficient_success,
            "maxrl/num_samples": float(self.config.num_samples),
            "maxrl/zero_success_batch_rate": zero_success,
            "maxrl/per_task_saturation": float(saturated_tasks / max(1, len(task_groups))),
            "maxrl/verifier_coverage": float(sum(coverage_values) / max(1, len(coverage_values))),
            "maxrl/kl": float(kl_value.item()),
        }
        return MaxRLStepResult(loss=total_loss, metrics=metrics)
