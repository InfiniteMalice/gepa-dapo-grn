"""Simple bandit demo comparing hard clipping and soft ratio gating."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn

from gepa_dapo_grn.config import DAPOConfig, RewardMixerConfig
from gepa_dapo_grn.curriculum import CurriculumTracker
from gepa_dapo_grn.dapo_core import DAPOBatch, DAPOTrainer
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback
from gepa_dapo_grn.logging_utils import MetricsLogger, summarize_feedback
from gepa_dapo_grn.policy_interfaces import Policy, PolicyOutput
from gepa_dapo_grn.safety_controller import SafetyController


@dataclass(eq=False)
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
    verifier_pass = float(action != 0)
    return GEPAFeedback(
        rewards=reward_vector(action),
        tags={"risk_score": float(action == 0)},
        verifier={"verifier_pass": verifier_pass, "verifier_fail_rate": 1.0 - verifier_pass},
        meta={"task_id": task_id},
        abstained=False,
    )


def run_training(use_soft_gating: bool) -> Tuple[float, float]:
    policy = BanditPolicy(num_actions=3)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

    mixer_config = RewardMixerConfig(weights={"engagement": 1.0, "diversity": 0.5})
    curriculum = CurriculumTracker(decay=0.8, reward_weights={"engagement": 1.0})
    safety = SafetyController(
        tag_risk_weights={"risk_score": 1.0},
        verifier_risk_weights={"verifier_fail_rate": 1.0},
    )

    trainer = DAPOTrainer(
        policy,
        optimizer,
        DAPOConfig(use_soft_gating=use_soft_gating, gating_temperature=0.7),
        reward_mixer=mixer_config,
        curriculum=curriculum,
        safety_controller=safety,
    )

    task_id = "bandit-demo"
    losses = []
    pass_rates = []
    for _ in range(20):
        batch_size = 6
        inputs = {"batch_size": batch_size}
        with torch.no_grad():
            logits = policy(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
            logp_old = policy.logprobs(actions, **inputs)

        feedbacks = [feedback_for_action(int(action), task_id) for action in actions]
        batch = DAPOBatch(inputs=inputs, actions=actions, logp_old=logp_old)
        result = trainer.train_step(batch, feedbacks)
        summary = summarize_feedback(feedbacks)
        losses.append(result.metrics["loss/total"])
        pass_rates.append(summary.get("feedback/verifier/verifier_pass", 0.0))

    return float(sum(losses) / len(losses)), float(sum(pass_rates) / len(pass_rates))


def main() -> None:
    torch.manual_seed(0)
    logger = MetricsLogger(prefix="bandit")
    hard_loss, hard_pass = run_training(use_soft_gating=False)
    soft_loss, soft_pass = run_training(use_soft_gating=True)
    logger.log(
        {
            "hard_clip_loss": hard_loss,
            "soft_gating_loss": soft_loss,
            "hard_clip_verifier_pass": hard_pass,
            "soft_gating_verifier_pass": soft_pass,
        }
    )


if __name__ == "__main__":
    main()
