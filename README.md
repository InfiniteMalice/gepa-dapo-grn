# gepa-dapo-grn

`gepa-dapo-grn` is a lightweight Python library for DAPO-style reinforcement learning on
sequence models. It supports vector-valued rewards (GEPA-like decompositions) and optional
Global Response Normalization (GRN) that can be applied to policy/value heads without
changing default behavior.

## Key concepts

- **DAPO (Decoupled Advantage Policy Optimization)**: PPO-like ratios with separate clipping
  for the policy ratio and advantages, plus KL regularization to a frozen reference policy.
- **Vector-valued rewards**: reward dictionaries are mixed into scalar rewards via configurable
  reward mixers (weights, normalization, clipping).
- **GRN (Global Response Normalization)**: optional module that rescales activations by their
  global feature norm using a learnable scale.

## Installation

```bash
pip install -e .
```

## Quick start

### Wrap a HuggingFace LM

```python
from transformers import GPT2Config, GPT2LMHeadModel

from gepa_dapo_grn.policy_interfaces import HuggingFaceLMPolicy

config = GPT2Config(vocab_size=128, n_layer=2, n_head=2, n_embd=64)
model = GPT2LMHeadModel(config)
policy = HuggingFaceLMPolicy(model, tokenizer=None)
```

### Define a reward-vector function

```python
from typing import Dict, List

from gepa_dapo_grn.policy_interfaces import Policy


def reward_fn(policy: Policy) -> List[Dict[str, float]]:
    # Return a list of reward dicts, one per sample.
    return [
        {"accuracy": 1.0, "brevity": 0.2},
        {"accuracy": 0.0, "brevity": 0.5},
    ]
```

### Run a simple DAPO training loop

```python
import torch

from gepa_dapo_grn.config import DAPOConfig, RewardMixerConfig
from gepa_dapo_grn.dapo_core import DAPOBatch, DAPOTrainer
from gepa_dapo_grn.reward_mixers import mix_reward_vectors

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
trainer = DAPOTrainer(policy, optimizer, DAPOConfig())

reward_vectors = reward_fn(policy)
scalar_rewards, _ = mix_reward_vectors(reward_vectors, RewardMixerConfig())
advantages = scalar_rewards - scalar_rewards.mean()

batch = DAPOBatch(
    inputs={"input_ids": torch.zeros((2, 8), dtype=torch.long)},
    actions=torch.zeros((2, 8), dtype=torch.long),
    logp_old=torch.zeros((2, 8)),
    advantages=advantages,
)
trainer.train_step(batch)
```

### Toggle GRN

```python
from gepa_dapo_grn.config import GRNConfig
from gepa_dapo_grn.grn import maybe_apply_grn

grn_config = GRNConfig(enabled=True, apply_to_policy=True)
policy_head = maybe_apply_grn(model.lm_head, grn_config, grn_config.apply_to_policy)
```

## Examples

- `examples/toy_lm_reasoning.py`: toy GPT-style model on a synthetic math task using
  vector rewards and DAPO.
- `examples/simple_bandit_demo.py`: small bandit policy demonstrating reward mixing and
  curriculum tracking.

## Library structure

- `gepa_dapo_grn/config.py`: dataclass configs for DAPO, GRN, and reward mixing.
- `gepa_dapo_grn/dapo_core.py`: DAPO trainer implementing decoupled clipping and KL regularization.
- `gepa_dapo_grn/reward_mixers.py`: mix vector rewards into scalar rewards.
- `gepa_dapo_grn/grn.py`: GRN module and head-wrapping helpers.
- `gepa_dapo_grn/policy_interfaces.py`: abstract policy interface + HF LM adapter.
- `gepa_dapo_grn/sampling.py`: curriculum tracking via EMA of reward signals.
- `gepa_dapo_grn/logging_utils.py`: stdout + JSONL metric logging.
- `gepa_dapo_grn/eval_hooks.py`: evaluation hooks for reward-vector functions.

## Notes

This library does **not** import or depend on any external GEPA repositories. Reward vectors
are treated as generic dictionaries and can represent any decomposed scoring scheme.
