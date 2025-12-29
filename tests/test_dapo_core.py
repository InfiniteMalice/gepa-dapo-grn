import copy

import torch
from torch import nn

from gepa_dapo_grn.config import DAPOConfig
from gepa_dapo_grn.dapo_core import DAPOBatch, DAPOTrainer
from gepa_dapo_grn.policy_interfaces import Policy, PolicyOutput


class SimplePolicy(Policy):
    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_actions))
        self.value_head = nn.Linear(num_actions, 1)

    def forward(self, **inputs: torch.Tensor) -> PolicyOutput:
        batch_size = inputs["batch_size"]
        logits = self.logits.repeat(batch_size, 1)
        values = self.value_head(logits).squeeze(-1)
        return PolicyOutput(logits=logits, values=values)

    def log_probs(self, actions: torch.Tensor, **inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(**inputs)
        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        return torch.gather(log_probs, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

    def clone(self) -> "SimplePolicy":
        cloned = copy.deepcopy(self)
        cloned.eval()
        for param in cloned.parameters():
            param.requires_grad_(False)
        return cloned


def test_policy_loss_decoupled_clipping() -> None:
    policy = SimplePolicy(num_actions=2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    config = DAPOConfig(clip_ratio=0.1, clip_advantage=1.0, adaptive_kl=False)
    trainer = DAPOTrainer(policy, optimizer, config)

    logp_new = torch.tensor([0.0, -0.2])
    logp_old = torch.tensor([-0.1, -0.1])
    advantages = torch.tensor([2.0, -2.0])

    loss = trainer._compute_policy_loss(logp_new, logp_old, advantages)
    ratio = torch.exp(logp_new - logp_old)
    ratio_clipped = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
    adv_clipped = torch.clamp(advantages, min=-config.clip_advantage, max=config.clip_advantage)
    expected = -(ratio_clipped * adv_clipped).mean()
    assert torch.allclose(loss, expected)


def test_kl_coeff_adapts() -> None:
    policy = SimplePolicy(num_actions=2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    config = DAPOConfig(target_kl=0.0, kl_horizon=1.0, adaptive_kl=True)
    trainer = DAPOTrainer(policy, optimizer, config)

    with torch.no_grad():
        policy.logits.copy_(torch.tensor([2.0, 0.0]))

    actions = torch.zeros(4, dtype=torch.long)
    inputs = {"batch_size": actions.shape[0]}
    logp_old = trainer.ref_policy.log_probs(actions, **inputs)
    advantages = torch.ones_like(logp_old)

    batch = DAPOBatch(inputs=inputs, actions=actions, logp_old=logp_old, advantages=advantages)
    initial_coeff = trainer.kl_coeff
    trainer.train_step(batch)
    assert trainer.kl_coeff > initial_coeff
