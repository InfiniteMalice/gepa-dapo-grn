"""Policy interfaces and a lightweight HuggingFace LM adapter."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

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
    def log_probs(self, actions: torch.Tensor, **inputs: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for the provided actions."""

    @abstractmethod
    def clone(self) -> "Policy":
        """Return a detached clone of the policy for use as a reference."""


class HuggingFaceLMPolicy(Policy):
    """Adapter for HuggingFace-style causal language models.

    The adapter does not import transformers directly; users supply the model and tokenizer.
    """

    def __init__(self, model: nn.Module, tokenizer: Any) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, **inputs: torch.Tensor) -> PolicyOutput:
        outputs = self.model(**inputs)
        logits = outputs.logits
        values = getattr(outputs, "values", None)
        return PolicyOutput(logits=logits, values=values)

    def log_probs(self, actions: torch.Tensor, **inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(**inputs)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        return torch.gather(log_probs, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

    def clone(self) -> "Policy":
        cloned = copy.deepcopy(self)
        cloned.eval()
        for param in cloned.parameters():
            param.requires_grad_(False)
        return cloned

    def generate(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward generate calls to the underlying model."""

        return self.model.generate(*args, **kwargs)

    def encode(self, text: str, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Tokenize input text using the provided tokenizer."""

        tokens = self.tokenizer(text, return_tensors="pt", **kwargs)
        return {key: value.to(self.device) for key, value in tokens.items()}

    @property
    def device(self) -> torch.device:
        """Return the primary device for the policy."""

        return next(self.parameters()).device
