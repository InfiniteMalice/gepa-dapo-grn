from __future__ import annotations

import inspect
from dataclasses import fields

import torch
from torch import nn

from gepa_dapo_grn import (
    CurriculumTracker,
    DAPOConfig,
    DAPOTrainer,
    GEPAFeedback,
    GlobalResponseNorm,
    GRNConfig,
    RewardMixerConfig,
    SafetyController,
)
from gepa_dapo_grn.dapo_core import DAPOBatch
from gepa_dapo_grn.policy_interfaces import Policy, PolicyOutput


class TinyPolicy(Policy):
    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.logits_param = nn.Parameter(torch.zeros(num_actions))
        self.value_head = nn.Linear(num_actions, 1)

    def forward(self, **inputs: torch.Tensor) -> PolicyOutput:
        batch_size = inputs["batch_size"]
        logits = self.logits_param.repeat(batch_size, 1)
        values = self.value_head(logits).squeeze(-1)
        return PolicyOutput(logits=logits, values=values)

    def logprobs(self, actions: torch.Tensor, **inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(**inputs)
        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        return torch.gather(log_probs, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

    def clone(self) -> "TinyPolicy":
        cloned = TinyPolicy(self.logits_param.numel())
        cloned.load_state_dict(self.state_dict())
        cloned.eval()
        for param in cloned.parameters():
            param.requires_grad_(False)
        return cloned


def test_public_api_exports() -> None:
    assert DAPOTrainer
    assert DAPOConfig
    assert GRNConfig
    assert RewardMixerConfig
    assert GEPAFeedback
    assert CurriculumTracker
    assert SafetyController
    assert GlobalResponseNorm


def test_public_api_schema_fields() -> None:
    assert {field.name for field in fields(GEPAFeedback)} == {
        "rewards",
        "tags",
        "meta",
        "abstained",
    }
    assert {field.name for field in fields(DAPOConfig)} == {
        "clip_ratio",
        "clip_advantage",
        "kl_coeff",
        "target_kl",
        "kl_horizon",
        "adaptive_kl",
        "max_grad_norm",
        "value_coef",
        "group_size",
    }
    assert {field.name for field in fields(GRNConfig)} == {
        "enabled",
        "apply_to_policy",
        "apply_to_value",
        "epsilon",
    }
    assert {field.name for field in fields(RewardMixerConfig)} == {
        "weights",
        "normalize",
        "clip_min",
        "clip_max",
        "default_weight",
    }


def test_train_step_signature_and_metrics() -> None:
    signature = inspect.signature(DAPOTrainer.train_step)
    assert list(signature.parameters) == ["self", "batch", "feedbacks", "extra_loss"]

    policy = TinyPolicy(num_actions=2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    trainer = DAPOTrainer(policy, optimizer)

    actions = torch.zeros(2, dtype=torch.long)
    inputs = {"batch_size": actions.shape[0]}
    logp_old = trainer.ref_policy.logprobs(actions, **inputs)
    batch = DAPOBatch(inputs=inputs, actions=actions, logp_old=logp_old)

    feedbacks = [GEPAFeedback(rewards={"reward": 1.0}) for _ in range(actions.shape[0])]
    result = trainer.train_step(batch, feedbacks)

    assert "loss/total" in result.metrics
    assert "kl/value" in result.metrics
