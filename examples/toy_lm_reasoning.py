"""Toy language-model reasoning example with DAPO, GEPA feedback, and GRN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from gepa_dapo_grn.config import DAPOConfig, GRNConfig, RewardMixerConfig
from gepa_dapo_grn.curriculum import CurriculumTracker
from gepa_dapo_grn.dapo_core import DAPOBatch, DAPOTrainer
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback
from gepa_dapo_grn.grn import maybe_apply_grn
from gepa_dapo_grn.integration.hf_lm import HuggingFaceLMPolicy
from gepa_dapo_grn.logging_utils import MetricsLogger
from gepa_dapo_grn.safety_controller import SafetyController


@dataclass
class SimpleTokenizer:
    vocab: Dict[str, int]

    def __call__(self, text: str, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
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


def feedback_for_prediction(prediction: int, target: int, task_id: str) -> GEPAFeedback:
    correct = float(prediction == target)
    distance = -abs(prediction - target)
    calibration_error = abs(prediction - target) / 10.0
    return GEPAFeedback(
        rewards={"truth": correct, "distance": distance, "brevity": 1.0},
        tags={"calibration_error": calibration_error, "harmlessness": 1.0},
        meta={"task_id": task_id},
        abstained=False,
    )


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

    mixer_config = RewardMixerConfig(weights={"truth": 1.0, "distance": 0.2, "brevity": 0.05})
    curriculum = CurriculumTracker(
        decay=0.9,
        reward_weights={"truth": 1.0},
        tag_weights={"calibration_error": -1.0},
    )
    safety = SafetyController(tag_risk_weights={"calibration_error": 1.0})

    trainer = DAPOTrainer(
        policy,
        optimizer,
        DAPOConfig(),
        grn_config=grn_config,
        reward_mixer=mixer_config,
        curriculum=curriculum,
        safety_controller=safety,
    )

    prompts = build_prompts()
    logger = MetricsLogger(prefix="train")
    task_id = "toy-lm"

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
            logp_old = policy.logprobs(actions, **inputs)

        feedbacks = [
            feedback_for_prediction(int(actions_last[idx]), target, task_id)
            for idx, target in enumerate(batch_targets)
        ]

        batch = DAPOBatch(
            inputs=inputs,
            actions=actions,
            logp_old=logp_old,
        )
        result = trainer.train_step(batch, feedbacks)

        if step % 10 == 0:
            logger.log(result.metrics)


if __name__ == "__main__":
    main()
