import copy

import torch
from torch import nn

from gepa_dapo_grn.config import DAPOConfig, RewardMixerConfig
from gepa_dapo_grn.dapo_core import DAPOBatch, DAPOTrainer, _soft_ratio_weights
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback
from gepa_dapo_grn.policy_interfaces import Policy, PolicyOutput


class SimplePolicy(Policy):
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


def test_soft_gating_weights_monotonic() -> None:
    log_ratio = torch.tensor([0.0, 0.2, 0.8])
    weights = _soft_ratio_weights(log_ratio, temperature=0.5)
    assert torch.isfinite(weights).all()
    assert weights[0] > weights[1] > weights[2]


def test_soft_gating_policy_loss_is_finite() -> None:
    policy = SimplePolicy(num_actions=2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    config = DAPOConfig(
        clip_ratio=0.1,
        clip_advantage=1.0,
        adaptive_kl=False,
        use_soft_gating=True,
        gating_temperature=0.5,
    )
    trainer = DAPOTrainer(policy, optimizer, config)
    loss = trainer._compute_policy_loss(
        torch.tensor([0.0, -1.0, 1.0]),
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, -1.0]),
    )
    assert torch.isfinite(loss)


def test_kl_coeff_adapts_and_loss_finite() -> None:
    policy = SimplePolicy(num_actions=2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    config = DAPOConfig(target_kl=0.0, kl_horizon=1.0, adaptive_kl=True)
    trainer = DAPOTrainer(policy, optimizer, config, reward_mixer=RewardMixerConfig())

    with torch.no_grad():
        policy.logits_param.copy_(torch.tensor([2.0, 0.0]))

    actions = torch.zeros(4, dtype=torch.long)
    inputs = {"batch_size": actions.shape[0]}
    logp_old = trainer.ref_policy.logprobs(actions, **inputs)
    batch = DAPOBatch(inputs=inputs, actions=actions, logp_old=logp_old)

    feedbacks = [GEPAFeedback(rewards={"reward": 1.0}) for _ in range(actions.shape[0])]
    initial_coeff = trainer.config.kl_coeff
    result = trainer.train_step(batch, feedbacks)
    assert torch.isfinite(result.loss)
    assert trainer.config.kl_coeff > initial_coeff


def test_train_step_group_normalize_single_sample_is_finite() -> None:
    policy = SimplePolicy(num_actions=2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    config = DAPOConfig(group_size=1, adaptive_kl=False)
    trainer = DAPOTrainer(policy, optimizer, config, reward_mixer=RewardMixerConfig())

    actions = torch.zeros(1, dtype=torch.long)
    inputs = {"batch_size": actions.shape[0]}
    logp_old = trainer.ref_policy.logprobs(actions, **inputs)
    batch = DAPOBatch(inputs=inputs, actions=actions, logp_old=logp_old)

    feedbacks = [GEPAFeedback(rewards={"reward": 1.0})]
    result = trainer.train_step(batch, feedbacks)

    assert torch.isfinite(result.loss)
    assert result.metrics["reward/std"] == 0.0


def test_soft_gating_large_log_ratio_is_finite() -> None:
    policy = SimplePolicy(num_actions=2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    config = DAPOConfig(use_soft_gating=True, gating_temperature=1.0, adaptive_kl=False)
    trainer = DAPOTrainer(policy, optimizer, config)

    logp_new = torch.tensor([1000.0, -1000.0])
    logp_old = torch.tensor([0.0, 0.0])
    advantages = torch.tensor([1.0, 1.0])

    loss = trainer._compute_policy_loss(logp_new, logp_old, advantages)
    assert torch.isfinite(loss)
