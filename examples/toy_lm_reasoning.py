"""Toy language-model reasoning example with DAPO and vector rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from gepa_dapo_grn.config import DAPOConfig, GRNConfig, RewardMixerConfig
from gepa_dapo_grn.dapo_core import DAPOBatch, DAPOTrainer
from gepa_dapo_grn.grn import maybe_apply_grn
from gepa_dapo_grn.logging_utils import MetricsLogger
from gepa_dapo_grn.policy_interfaces import HuggingFaceLMPolicy
from gepa_dapo_grn.reward_mixers import mix_reward_vectors


@dataclass
class SimpleTokenizer:
    vocab: Dict[str, int]

    def __call__(self, text: str) -> Dict[str, torch.Tensor]:
        ids = [self.vocab[token] for token in text.split()]
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def build_vocab() -> Dict[str, int]:
    tokens = [str(i) for i in range(10)] + ["+", "="]
    return {token: idx for idx, token in enumerate(tokens)}


def build_prompts() -> List[Tuple[str, int]]:
    prompts = []
    for a in range(3):
        for b in range(3):
            prompt = f"{a} + {b} ="
            prompts.append((prompt, a + b))
    return prompts


def reward_vector(prediction: int, target: int) -> Dict[str, float]:
    correct = float(prediction == target)
    distance = -abs(prediction - target)
    return {"accuracy": correct, "distance": distance, "brevity": 1.0}


def main() -> None:
    vocab = build_vocab()
    tokenizer = SimpleTokenizer(vocab)

    config = GPT2Config(
        vocab_size=len(vocab),
        n_layer=2,
        n_head=2,
        n_embd=32,
        n_positions=16,
    )
    model = GPT2LMHeadModel(config)

    grn_config = GRNConfig(enabled=False, apply_to_policy=True)
    model.lm_head = maybe_apply_grn(model.lm_head, grn_config, grn_config.apply_to_policy)

    policy = HuggingFaceLMPolicy(model, tokenizer)
    optimizer = torch.optim.Adam(policy.parameters(), lr=2e-4)

    trainer = DAPOTrainer(policy, optimizer, DAPOConfig())
    mixer_config = RewardMixerConfig(weights={"accuracy": 1.0, "distance": 0.2, "brevity": 0.05})

    prompts = build_prompts()
    logger = MetricsLogger(prefix="train")

    for step in range(30):
        batch_inputs = []
        batch_targets = []
        for prompt, target in prompts:
            tokens = tokenizer(prompt)
            batch_inputs.append(tokens)
            batch_targets.append(target)

        input_ids = torch.cat([item["input_ids"] for item in batch_inputs], dim=0)
        attention_mask = torch.cat([item["attention_mask"] for item in batch_inputs], dim=0)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        with torch.no_grad():
            logits = policy(**inputs).logits
            last_logits = logits[:, -1, :]
            probs = torch.softmax(last_logits, dim=-1)
            actions_last = torch.multinomial(probs, num_samples=1).squeeze(-1)

        actions = input_ids.clone()
        actions[:, -1] = actions_last
        with torch.no_grad():
            logp_old = policy.log_probs(actions, **inputs)

        reward_vectors = [
            reward_vector(int(actions_last[idx]), target)
            for idx, target in enumerate(batch_targets)
        ]
        scalar_rewards, reward_stats = mix_reward_vectors(reward_vectors, mixer_config)
        advantages = scalar_rewards - scalar_rewards.mean()

        advantages_full = torch.zeros_like(logp_old)
        advantages_full[:, -1] = advantages

        batch = DAPOBatch(
            inputs=inputs,
            actions=actions,
            logp_old=logp_old,
            advantages=advantages_full,
        )
        result = trainer.train_step(batch)

        metrics = {**reward_stats, **result.metrics}
        if step % 10 == 0:
            logger.log(metrics)


if __name__ == "__main__":
    main()
