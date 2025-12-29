"""Simple bandit demo with vector rewards and curriculum tracking."""

from __future__ import annotations

import copy
from typing import Dict

import torch
from torch import nn

from gepa_dapo_grn.config import DAPOConfig, RewardMixerConfig
from gepa_dapo_grn.dapo_core import DAPOBatch, DAPOTrainer
from gepa_dapo_grn.logging_utils import MetricsLogger
from gepa_dapo_grn.policy_interfaces import Policy, PolicyOutput
from gepa_dapo_grn.reward_mixers import mix_reward_vectors
from gepa_dapo_grn.sampling import CurriculumTracker


class BanditPolicy(Policy):
    """Minimal categorical policy for a bandit task."""

    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.logits = nn.Parameter(torch.zeros(self.num_actions))
        self.value_head = nn.Linear(self.num_actions, 1)

    def forward(self, **inputs: torch.Tensor) -> PolicyOutput:
        batch_size = inputs["batch_size"]
        logits = self.logits.repeat(batch_size, 1)
        values = self.value_head(logits).squeeze(-1)
        return PolicyOutput(logits=logits, values=values)

    def log_probs(self, actions: torch.Tensor, **inputs: torch.Tensor) -> torch.Tensor:
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


def main() -> None:
    torch.manual_seed(0)
    policy = BanditPolicy(num_actions=3)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    trainer = DAPOTrainer(policy, optimizer, DAPOConfig())

    mixer_config = RewardMixerConfig(weights={"engagement": 1.0, "diversity": 0.5})
    tracker = CurriculumTracker(decay=0.8)
    logger = MetricsLogger(prefix="bandit")

    for step in range(20):
        batch_size = 6
        inputs = {"batch_size": batch_size}
        logits = policy(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
        logp_old = policy.log_probs(actions, **inputs)

        reward_vectors = [reward_vector(int(action)) for action in actions]
        scalar_rewards, stats = mix_reward_vectors(reward_vectors, mixer_config)
        tracker.update(reward_vectors)

        advantages = scalar_rewards - scalar_rewards.mean()
        batch = DAPOBatch(
            inputs=inputs,
            actions=actions,
            logp_old=logp_old,
            advantages=advantages,
            returns=scalar_rewards,
        )
        result = trainer.train_step(batch)

        metrics = {**stats, **result.metrics, **tracker.describe()}
        if step % 5 == 0:
            logger.log(metrics)


if __name__ == "__main__":
    main()
