import torch

from gepa_dapo_grn.config import RewardMixerConfig
from gepa_dapo_grn.reward_mixers import mix_reward_vectors


def test_reward_mixer_weights_and_clipping() -> None:
    reward_vectors = [
        {"a": 1.0, "b": 0.0},
        {"a": 2.0, "b": 1.0},
    ]
    config = RewardMixerConfig(weights={"a": 1.0, "b": 2.0}, normalize=False, clip_max=2.5)
    scalar, stats = mix_reward_vectors(reward_vectors, config)
    expected = torch.tensor([1.0, 4.0])
    clipped = torch.clamp(expected, max=2.5)
    assert torch.allclose(scalar, clipped)
    assert stats["reward_weight/a"] == 1.0
    assert stats["reward_weight/b"] == 2.0
