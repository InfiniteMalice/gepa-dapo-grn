"""Promotion gates for adaptive references."""

from __future__ import annotations

from dataclasses import field
from typing import Any, Dict, Optional

from gepa_dapo_grn._compat import dataclass
from gepa_dapo_grn.active_grpo.references import ActiveReference
from gepa_dapo_grn.graph.metrics import (
    contradiction_count,
    critical_unsupported_claim_count,
    unsupported_claim_count,
)
from gepa_dapo_grn.graph.schema import ReasoningGraph
from gepa_dapo_grn.graph.validators import ValidationResult, validate_graph_for_promotion


@dataclass(slots=True)
class PromotionGateConfig:
    """Strict gates for promoting policy output to a new reference."""

    margin: float = 0.02
    require_external_verifier: bool = True
    max_unsupported_claims: int = 0
    max_critical_unsupported_claims: int = 0
    max_contradictions: int = 0
    require_no_safety_violation: bool = True


@dataclass(slots=True)
class PromotionCandidate:
    """Candidate output considered for reference promotion."""

    prompt_id: str
    output: str
    score: float
    verifier_source: str
    graph: Optional[ReasoningGraph] = None
    safety_violation: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


def evaluate_promotion(
    candidate: PromotionCandidate,
    reference: ActiveReference,
    config: PromotionGateConfig,
) -> ValidationResult:
    """Evaluate strict non-self-confirming reference-promotion gates."""

    errors = []
    warnings = []
    metrics: Dict[str, Any] = {
        "candidate_score": float(candidate.score),
        "reference_score": float(reference.reference_score),
        "margin": float(config.margin),
        "score_delta": float(candidate.score) - float(reference.reference_score),
    }

    if candidate.score <= reference.reference_score + config.margin:
        errors.append("candidate score does not beat reference by required margin")
    if config.require_external_verifier and candidate.verifier_source == "self":
        errors.append("external verifier is required for promotion")
    if config.require_no_safety_violation and candidate.safety_violation:
        errors.append("candidate has a safety violation")

    if candidate.graph is not None:
        graph_result = validate_graph_for_promotion(candidate.graph)
        metrics.update(graph_result.metrics)
        if not graph_result.passed:
            errors.extend(graph_result.errors)
        unsupported = unsupported_claim_count(candidate.graph)
        critical_unsupported = critical_unsupported_claim_count(candidate.graph)
        contradictions = contradiction_count(candidate.graph)
        if unsupported > config.max_unsupported_claims:
            errors.append("candidate exceeds unsupported claim limit")
        if critical_unsupported > config.max_critical_unsupported_claims:
            errors.append("candidate has unsupported critical claims")
        if contradictions > config.max_contradictions:
            errors.append("candidate exceeds contradiction limit")
    else:
        warnings.append("candidate has no graph artifact")

    return ValidationResult(
        passed=not errors,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
    )
