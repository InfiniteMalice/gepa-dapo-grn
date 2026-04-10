"""CPU-safe demo: MaxRL backend with a tiny verifier-backed classification task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from gepa_dapo_grn import (
    GEPAFeedback,
    MaxRLConfig,
    MaxRLTrainer,
    VerifierResult,
)
from gepa_dapo_grn.maxrl_core import MaxRLBatch
from gepa_dapo_grn.policy_interfaces import Policy, PolicyOutput


class TinyPolicy(Policy):
    def __init__(self, num_actions: int = 2) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_actions))

    def forward(self, **inputs: torch.Tensor) -> PolicyOutput:
        batch_size = int(inputs["batch_size"].item())
        return PolicyOutput(logits=self.logits.repeat(batch_size, 1))

    def logprobs(self, actions: torch.Tensor, **inputs: torch.Tensor) -> torch.Tensor:
        output = self.forward(**inputs)
        log_probs = torch.log_softmax(output.logits, dim=-1)
        return torch.gather(log_probs, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

    def clone(self) -> "TinyPolicy":
        cloned = TinyPolicy(num_actions=self.logits.numel())
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
    policy = TinyPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.1)
    trainer = MaxRLTrainer(policy, optimizer, config=MaxRLConfig(enabled=True, num_samples=4))

    for step in range(4):
        sampled_actions = torch.randint(low=0, high=2, size=(len(items),), dtype=torch.long)
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
            inputs={"batch_size": torch.tensor(len(items))},
            actions=sampled_actions,
            task_ids=task_ids,
        )
        result = trainer.train_step(batch, feedbacks)
        print(
            f"step={step} loss={result.loss.item():.4f} "
            f"success_rate={result.metrics['maxrl/success_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
