import torch
from torch import nn

from gepa_dapo_grn.config import GRNConfig
from gepa_dapo_grn.grn import GlobalResponseNorm, maybe_apply_grn


def test_grn_scales_by_global_norm() -> None:
    grn = GlobalResponseNorm(epsilon=0.0)
    activations = torch.tensor([[3.0, 4.0]])
    output = grn(activations)
    expected = activations / torch.linalg.norm(activations, dim=-1, keepdim=True)
    assert torch.allclose(output, expected)


def test_grn_disabled_returns_head() -> None:
    head = nn.Linear(4, 2)
    config = GRNConfig(enabled=False)
    wrapped = maybe_apply_grn(head, config, apply_flag=True)
    assert wrapped is head


def test_grn_outputs_are_finite() -> None:
    grn = GlobalResponseNorm(epsilon=1e-6)
    activations = torch.randn(8, 4)
    output = grn(activations)
    assert torch.isfinite(output).all()
