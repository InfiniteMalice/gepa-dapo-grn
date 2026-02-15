import torch
from torch import nn

from gepa_dapo_grn.config import GRNConfig
from gepa_dapo_grn.grn import (
    GlobalResponseNorm,
    GRNWrappedHead,
    maybe_apply_grn,
    maybe_wrap_policy_heads,
)


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


def test_grn_probe_module_protected_unless_included() -> None:
    class ProbePolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy_probe_head = nn.Linear(4, 2)

    policy = ProbePolicy()
    config = GRNConfig(enabled=True, apply_to_policy=True)
    maybe_wrap_policy_heads(
        policy, config, policy_attr="policy_probe_head", value_attr="value_head"
    )
    assert not isinstance(policy.policy_probe_head, GRNWrappedHead)

    config.include_modules = ["policy_probe_head"]
    maybe_wrap_policy_heads(
        policy, config, policy_attr="policy_probe_head", value_attr="value_head"
    )
    assert isinstance(policy.policy_probe_head, GRNWrappedHead)
