import copy

import torch
from torch import nn

from gepa_dapo_grn.config import MaxRLConfig, TrainerBackendConfig
from gepa_dapo_grn.dapo_core import DAPOTrainer
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback
from gepa_dapo_grn.maxrl_core import MaxRLBatch, MaxRLTrainer
from gepa_dapo_grn.policy_interfaces import Policy, PolicyOutput
from gepa_dapo_grn.trainer import make_trainer
from gepa_dapo_grn.verifiers import VerifierResult


class SimplePolicy(Policy):
    def __init__(self, num_actions: int = 2) -> None:
        super().__init__()
        self.logits_param = nn.Parameter(torch.zeros(num_actions))

    def forward(self, **inputs: torch.Tensor) -> PolicyOutput:
        batch_size = int(inputs["batch_size"].item())
        logits = self.logits_param.repeat(batch_size, 1)
        return PolicyOutput(logits=logits, values=None)

    def logprobs(self, actions: torch.Tensor, **inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(**inputs)
        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        return torch.gather(log_probs, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

    def clone(self) -> "SimplePolicy":
        cloned = copy.deepcopy(self)
        cloned.eval()
        for param in cloned.parameters():
            param.requires_grad_(False)
        return cloned


def test_maxrl_smoke_success_and_zero_success() -> None:
    policy = SimplePolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    trainer = MaxRLTrainer(
        policy=policy,
        optimizer=optimizer,
        config=MaxRLConfig(enabled=True, num_samples=2, min_success_count=1),
    )
    actions = torch.zeros(4, dtype=torch.long)
    batch = MaxRLBatch(
        inputs={"batch_size": torch.tensor(actions.shape[0])},
        actions=actions,
        task_ids=["task-a", "task-a", "task-b", "task-b"],
    )

    success_feedbacks = [
        GEPAFeedback(tags={"verifier_success": 1.0}),
        GEPAFeedback(tags={"verifier_success": 0.0}),
        GEPAFeedback(tags={"verifier_success": 1.0}),
        GEPAFeedback(tags={"verifier_success": 0.0}),
    ]
    success_result = trainer.train_step(batch, success_feedbacks)
    assert torch.isfinite(success_result.loss)
    assert success_result.metrics["maxrl/success_count"] == 2.0

    zero_feedbacks = [GEPAFeedback(tags={"verifier_success": 0.0}) for _ in range(4)]
    zero_result = trainer.train_step(batch, zero_feedbacks)
    assert torch.isfinite(zero_result.loss)
    assert zero_result.metrics["maxrl/zero_success_batch_rate"] == 1.0


def test_backend_selection_factory() -> None:
    policy = SimplePolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    assert isinstance(make_trainer(policy, optimizer), DAPOTrainer)
    assert isinstance(
        make_trainer(
            policy,
            optimizer,
            backend_config=TrainerBackendConfig(backend="maxrl"),
            maxrl_config=MaxRLConfig(enabled=True),
        ),
        MaxRLTrainer,
    )


def test_verifier_result_maps_into_feedback_tags() -> None:
    result = VerifierResult(passed=True, score=1.0, coverage=0.75, diagnostics={"x": 0.3})
    tags = result.as_tags()
    feedback = GEPAFeedback(tags=tags)

    assert feedback.tags["verifier_success"] == 1.0
    assert feedback.tags["verifier_coverage"] == 0.75
    assert feedback.tags["x"] == 0.3


def test_maxrl_has_no_special_deception_penalty_path() -> None:
    policy = SimplePolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    trainer = MaxRLTrainer(policy=policy, optimizer=optimizer, config=MaxRLConfig(enabled=True))
    actions = torch.zeros(2, dtype=torch.long)
    batch = MaxRLBatch(
        inputs={"batch_size": torch.tensor(actions.shape[0])},
        actions=actions,
        task_ids=["task-a", "task-a"],
    )
    feedbacks = [
        GEPAFeedback(tags={"deception": 1.0, "verifier_success": 1.0}),
        GEPAFeedback(tags={"deception": 0.0, "verifier_success": 0.0}),
    ]
    result = trainer.train_step(batch, feedbacks)
    assert torch.isfinite(result.loss)
    assert "deception" not in result.metrics
