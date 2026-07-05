"""Validation helpers for public reasoning graphs."""

from __future__ import annotations

from dataclasses import field
from typing import Dict, List, Union

from gepa_dapo_grn._compat import dataclass
from gepa_dapo_grn.graph.metrics import (
    claim_support_coverage,
    contradiction_count,
    critical_unsupported_claim_count,
    graph_completeness,
    unsupported_claim_count,
)
from gepa_dapo_grn.graph.schema import ReasoningGraph

MetricValue = Union[float, int, bool]


@dataclass(slots=True)
class ValidationResult:
    """Structured validation result for promotion and diagnostics."""

    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, MetricValue] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
        }


def _metrics(graph: ReasoningGraph) -> Dict[str, MetricValue]:
    return {
        "graph_completeness": graph_completeness(graph),
        "claim_support_coverage": claim_support_coverage(graph),
        "unsupported_claims": unsupported_claim_count(graph),
        "critical_unsupported_claims": critical_unsupported_claim_count(graph),
        "contradictions": contradiction_count(graph),
    }


def validate_claim_support(graph: ReasoningGraph) -> ValidationResult:
    """Validate that critical claims have public support."""

    errors = graph.validate_basic()
    unsupported = unsupported_claim_count(graph)
    critical_unsupported = critical_unsupported_claim_count(graph)
    warnings = []
    if unsupported:
        warnings.append(f"{unsupported} claim(s) lack public support")
    if critical_unsupported:
        errors.append(f"{critical_unsupported} critical claim(s) lack public support")
    return ValidationResult(
        passed=not errors, errors=errors, warnings=warnings, metrics=_metrics(graph)
    )


def validate_no_orphan_evidence(graph: ReasoningGraph) -> ValidationResult:
    """Validate that evidence links point to existing claims and evidence nodes."""

    errors = [error for error in graph.validate_basic() if error.startswith("evidence link ")]
    return ValidationResult(passed=not errors, errors=errors, metrics=_metrics(graph))


def validate_no_unresolved_critical_contradictions(graph: ReasoningGraph) -> ValidationResult:
    """Validate that critical contradictions are explicitly resolved."""

    errors = graph.validate_basic()
    for contradiction in graph.contradictions:
        if contradiction.critical and not contradiction.resolved:
            errors.append(f"critical contradiction is unresolved: {contradiction.id}")
    for edge in graph.edges:
        if edge.type == "contradicts" and bool(edge.metadata.get("critical", False)):
            if not bool(edge.metadata.get("resolved", False)):
                errors.append(f"critical contradicts edge is unresolved: {edge.id}")
    return ValidationResult(passed=not errors, errors=errors, metrics=_metrics(graph))


def validate_graph_for_promotion(graph: ReasoningGraph) -> ValidationResult:
    """Validate a graph under strict reference-promotion gates."""

    errors = graph.validate_basic()
    warnings: List[str] = []
    unsupported = unsupported_claim_count(graph)
    critical_unsupported = critical_unsupported_claim_count(graph)
    contradictions = contradiction_count(graph)

    if unsupported:
        errors.append(f"{unsupported} unsupported claim(s) present")
    if critical_unsupported:
        errors.append(f"{critical_unsupported} unsupported critical claim(s) present")
    if contradictions:
        errors.append(f"{contradictions} contradiction(s) present")

    metrics = _metrics(graph)
    metrics["valid_for_promotion"] = not errors
    return ValidationResult(
        passed=not errors,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
    )
