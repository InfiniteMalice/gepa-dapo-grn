"""Simple bandit demo with vector rewards, curriculum tracking, and safety control."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from gepa_dapo_grn.config import DAPOConfig, RewardMixerConfig
from gepa_dapo_grn.curriculum import CurriculumTracker
from gepa_dapo_grn.dapo_core import DAPOBatch, DAPOTrainer
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback
from gepa_dapo_grn.logging_utils import MetricsLogger
from gepa_dapo_grn.policy_interfaces import Policy, PolicyOutput
from gepa_dapo_grn.safety_controller import SafetyController


@dataclass
class BanditPolicy(Policy):
    """Minimal categorical policy for a bandit task."""

    num_actions: int

    def __post_init__(self) -> None:
        super().__init__()
        self.policy_head = nn.Parameter(torch.zeros(self.num_actions))
        self.value_head = nn.Linear(self.num_actions, 1)

    def forward(self, **inputs: torch.Tensor) -> PolicyOutput:
        batch_size = inputs["batch_size"]
        logits = self.policy_head.repeat(batch_size, 1)
        values = self.value_head(logits).squeeze(-1)
        return PolicyOutput(logits=logits, values=values)

    def logprobs(self, actions: torch.Tensor, **inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(**inputs)
        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        return torch.gather(log_probs, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

    def clone(self) -> "BanditPolicy":
        cloned = copy.deepcopy(self)
        cloned.eval()
        for param in cloned.parameters():
            param.requires_grad_(False)
        return cloned


def reward_vector(action: int) -> Dict[str, float]:
    return {"engagement": float(action == 1), "diversity": float(action == 2)}


def feedback_for_action(action: int, task_id: str) -> GEPAFeedback:
    deception = float(action == 0)
    return GEPAFeedback(
        rewards=reward_vector(action),
        tags={"deception": deception},
        meta={"task_id": task_id},
        abstained=False,
    )


def main() -> None:
    torch.manual_seed(0)
    policy = BanditPolicy(num_actions=3)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

    mixer_config = RewardMixerConfig(weights={"engagement": 1.0, "diversity": 0.5})
    curriculum = CurriculumTracker(
        decay=0.8,
        reward_weights={"engagement": 1.0},
        tag_weights={"deception": -1.0},
    )
    safety = SafetyController(tag_risk_weights={"deception": 1.0})

    trainer = DAPOTrainer(
        policy,
        optimizer,
        DAPOConfig(),
        reward_mixer=mixer_config,
        curriculum=curriculum,
        safety_controller=safety,
    )

    logger = MetricsLogger(prefix="bandit")
    task_id = "bandit-demo"

    for step in range(20):
        batch_size = 6
        inputs = {"batch_size": batch_size}
        logits = policy(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
        logp_old = policy.logprobs(actions, **inputs)

        feedbacks = [feedback_for_action(int(action), task_id) for action in actions]
        batch = DAPOBatch(inputs=inputs, actions=actions, logp_old=logp_old)
        result = trainer.train_step(batch, feedbacks)

        metrics = {
            **result.metrics,
            "sample_weight": curriculum.sample_weight(task_id),
        }
        if step % 5 == 0:
            logger.log(metrics)


if __name__ == "__main__":
    main()
