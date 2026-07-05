"""Adaptive per-prompt reference storage."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from gepa_dapo_grn._compat import dataclass


@dataclass(slots=True)
class ActiveReference:
    """Reference output and best policy candidate for a prompt."""

    prompt_id: str
    reference_output: str
    reference_score: float
    best_policy_output: Optional[str] = None
    best_policy_score: Optional[float] = None
    source: str = "dataset"
    update_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "reference_output": self.reference_output,
            "reference_score": self.reference_score,
            "best_policy_output": self.best_policy_output,
            "best_policy_score": self.best_policy_score,
            "source": self.source,
            "update_count": self.update_count,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActiveReference":
        best_score = payload.get("best_policy_score")
        return cls(
            prompt_id=str(payload["prompt_id"]),
            reference_output=str(payload["reference_output"]),
            reference_score=float(payload["reference_score"]),
            best_policy_output=(
                str(payload["best_policy_output"])
                if payload.get("best_policy_output") is not None
                else None
            ),
            best_policy_score=float(best_score) if best_score is not None else None,
            source=str(payload.get("source", "dataset")),
            update_count=int(payload.get("update_count", 0)),
            metadata=dict(payload.get("metadata") or {}),
        )


class ActiveReferenceStore:
    """In-memory active reference store with JSON persistence."""

    def __init__(self, references: Optional[Mapping[str, ActiveReference]] = None) -> None:
        self._references: Dict[str, ActiveReference] = dict(references or {})

    def get(self, prompt_id: str) -> Optional[ActiveReference]:
        return self._references.get(prompt_id)

    def set(self, reference: ActiveReference) -> None:
        self._references[reference.prompt_id] = reference

    def update_candidate(
        self,
        prompt_id: str,
        output: str,
        score: float,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ActiveReference:
        reference = self._require(prompt_id)
        if reference.best_policy_score is None or score > reference.best_policy_score:
            reference.best_policy_output = output
            reference.best_policy_score = float(score)
            if metadata:
                reference.metadata.setdefault("best_policy_metadata", {}).update(dict(metadata))
        return reference

    def promote(
        self,
        prompt_id: str,
        output: str,
        score: float,
        source: str = "policy",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ActiveReference:
        reference = self._require(prompt_id)
        reference.reference_output = output
        reference.reference_score = float(score)
        reference.source = source
        reference.update_count += 1
        if metadata:
            reference.metadata.update(dict(metadata))
        return reference

    def to_dict(self) -> Dict[str, Any]:
        return {
            prompt_id: reference.to_dict()
            for prompt_id, reference in sorted(self._references.items())
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActiveReferenceStore":
        references = {}
        for prompt_id, reference_payload in payload.items():
            reference = ActiveReference.from_dict(reference_payload)
            if str(prompt_id) != reference.prompt_id:
                raise ValueError(
                    "active reference key does not match payload prompt_id: "
                    f"{prompt_id!r} != {reference.prompt_id!r}"
                )
            references[reference.prompt_id] = reference
        return cls(references)

    def save_json(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=str(target.parent),
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_file.write(payload)
                temp_path = temp_file.name
            os.replace(temp_path, target)
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                os.unlink(temp_path)

    @classmethod
    def load_json(cls, path: str) -> "ActiveReferenceStore":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def _require(self, prompt_id: str) -> ActiveReference:
        reference = self.get(prompt_id)
        if reference is None:
            raise KeyError(f"unknown active reference prompt_id: {prompt_id}")
        return reference
