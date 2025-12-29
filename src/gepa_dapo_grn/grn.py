"""Global Response Normalization modules and helpers."""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Dict, Optional

import torch
from torch import nn

from gepa_dapo_grn.config import GRNConfig


class GlobalResponseNorm(nn.Module):
    """Normalize activations by global feature norm with a learnable scale.

    This module rescales activations based on the L2 norm across the feature
    dimension. The scale is a learnable parameter.
    """

    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.norm(activations, dim=-1, keepdim=True)
        scaled = activations * (self.scale / (norm + self.epsilon))
        return scaled


class GRNWrappedHead(nn.Module):
    """Wrapper that applies GRN before an existing head module."""

    def __init__(self, head: nn.Module, grn: GlobalResponseNorm) -> None:
        super().__init__()
        self.grn = grn
        self.head = head

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.grn(inputs))


def wrap_head_with_grn(head: nn.Module, config: GRNConfig) -> nn.Module:
    """Optionally wrap a head with GRN depending on configuration."""

    if not config.enabled:
        return head
    if isinstance(head, GRNWrappedHead):
        return head
    return GRNWrappedHead(head, GlobalResponseNorm(epsilon=config.epsilon))


def maybe_apply_grn(
    head: nn.Module,
    config: GRNConfig,
    apply_flag: bool,
    builder: Optional[Callable[[nn.Module], nn.Module]] = None,
) -> nn.Module:
    """Apply GRN wrapping to a module if enabled and flagged.

    Args:
        head: Module to wrap.
        config: GRN configuration.
        apply_flag: Flag indicating whether to apply GRN to this head.
        builder: Optional custom builder for wrapping.
    """

    if not config.enabled or not apply_flag:
        return head
    if builder is None:

        def _builder(module: nn.Module) -> nn.Module:
            return wrap_head_with_grn(module, config)

        builder = _builder
    return builder(head)


def maybe_wrap_policy_heads(
    policy: nn.Module,
    config: GRNConfig,
    policy_attr: str = "policy_head",
    value_attr: str = "value_head",
) -> Dict[str, nn.Module]:
    """Wrap policy/value heads with GRN when configured.

    Returns a mapping of attribute names to original modules for restoration.
    """

    originals: Dict[str, nn.Module] = {}
    if not config.enabled:
        return originals
    if config.apply_to_policy and hasattr(policy, policy_attr):
        head = getattr(policy, policy_attr)
        if not isinstance(head, GRNWrappedHead):
            originals[policy_attr] = head
            setattr(policy, policy_attr, wrap_head_with_grn(head, config))
    if config.apply_to_value and hasattr(policy, value_attr):
        head = getattr(policy, value_attr)
        if not isinstance(head, GRNWrappedHead):
            originals[value_attr] = head
            setattr(policy, value_attr, wrap_head_with_grn(head, config))
    return originals


def restore_policy_heads(policy: nn.Module, originals: Dict[str, nn.Module]) -> None:
    """Restore policy/value heads to their original modules."""

    for attr, module in originals.items():
        setattr(policy, attr, module)


def describe_grn_config(config: GRNConfig) -> dict:
    """Return a JSON-serializable description of the GRN configuration."""

    return asdict(config)
