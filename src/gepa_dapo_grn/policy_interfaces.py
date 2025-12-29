"""Policy interfaces for DAPO training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(slots=True)
class PolicyOutput:
    """Container for policy outputs.

    Args:
        logits: Action logits for the policy.
        values: Optional value predictions aligned to the logits.
    """

    logits: torch.Tensor
    values: Optional[torch.Tensor] = None


class Policy(ABC, nn.Module):
    """Abstract policy interface for DAPO training."""

    @abstractmethod
    def forward(self, **inputs: torch.Tensor) -> PolicyOutput:
        """Compute policy outputs."""

    @abstractmethod
    def logprobs(self, actions: torch.Tensor, **inputs: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for the provided actions."""

    def log_probs(self, actions: torch.Tensor, **inputs: torch.Tensor) -> torch.Tensor:
        """Backward-compatible alias for logprobs."""

        return self.logprobs(actions, **inputs)

    @abstractmethod
    def clone(self) -> "Policy":
        """Return a detached clone of the policy for use as a reference."""
