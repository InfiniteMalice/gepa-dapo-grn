"""CPU-safe demo: MaxRL backend with a tiny verifier-backed classification task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from gepa_dapo_grn import (
    GEPAFeedback,
    MaxRLBatch,
    MaxRLConfig,
    MaxRLTrainer,
    VerifierResult,
)
from gepa_dapo_grn.policy_interfaces import Policy, PolicyOutput


class TinyPolicy(Policy):
    def __init__(self, num_actions: int = 2) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_actions))
        self.feature_proj = nn.Linear(1, num_actions)

    def forward(self, **inputs: torch.Tensor) -> PolicyOutput:
        task_feature = inputs["task_feature"].float()
        batch_size = task_feature.shape[0]
        base_logits = self.logits.repeat(batch_size, 1)
        conditioned_logits = base_logits + self.feature_proj(task_feature)
        return PolicyOutput(logits=conditioned_logits)

    def logprobs(self, actions: torch.Tensor, **inputs: torch.Tensor) -> torch.Tensor:
        output = self.forward(**inputs)
        log_probs = torch.log_softmax(output.logits, dim=-1)
        return torch.gather(log_probs, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

    def clone(self) -> "TinyPolicy":
        cloned = TinyPolicy(num_actions=self.logits.numel()).to(
            device=self.logits.device, dtype=self.logits.dtype
        )
        cloned.load_state_dict(self.state_dict())
        cloned.eval()
        for param in cloned.parameters():
            param.requires_grad_(False)
        return cloned


@dataclass
class ToyItem:
    task_id: str
    expected_action: int


def verify_action(action: int, expected_action: int) -> VerifierResult:
    is_success = float(action == expected_action)
    return VerifierResult(
        passed=bool(is_success),
        score=is_success,
        coverage=1.0,
        diagnostics={"toy_margin": is_success},
    )


def main() -> None:
    torch.manual_seed(0)
    items: List[ToyItem] = [
        ToyItem(task_id=f"task-{i // 4}", expected_action=(i % 2)) for i in range(8)
    ]
    task_feature = torch.tensor(
        [[float(item.expected_action)] for item in items],
        dtype=torch.float32,
    )
    policy = TinyPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.1)
    trainer = MaxRLTrainer(policy, optimizer, config=MaxRLConfig(enabled=True, num_samples=4))

    for step in range(4):
        with torch.no_grad():
            outputs = policy(task_feature=task_feature)
            probs = torch.softmax(outputs.logits, dim=-1)
            sampled_actions = torch.distributions.Categorical(probs=probs).sample()
        feedbacks: List[GEPAFeedback] = []
        task_ids: List[str] = []
        for item, action in zip(items, sampled_actions.tolist()):
            verification = verify_action(action, item.expected_action)
            feedbacks.append(
                GEPAFeedback(
                    rewards={"task_reward": verification.score or 0.0},
                    tags=verification.as_tags(),
                    meta={"task_id": item.task_id, "step": str(step)},
                )
            )
            task_ids.append(item.task_id)

        batch = MaxRLBatch(
            inputs={"batch_size": torch.tensor(len(items)), "task_feature": task_feature},
            actions=sampled_actions,
            task_ids=task_ids,
        )
        result = trainer.train_step(batch, feedbacks)
        success_rate = result.metrics.get("maxrl/success_rate", float("nan"))
        print(f"step={step} loss={result.loss.item():.4f} success_rate={success_rate:.3f}")


if __name__ == "__main__":
    main()
