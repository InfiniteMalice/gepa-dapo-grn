"""GEPA-compatible graph feedback builders."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

from gepa_dapo_grn.gepa_interfaces import GEPAFeedback
from gepa_dapo_grn.graph.metrics import (
    claim_support_coverage,
    contradiction_count,
    critical_unsupported_claim_count,
    graph_completeness,
    graph_to_answer_alignment_score,
    unsupported_claim_count,
)
from gepa_dapo_grn.graph.schema import ReasoningGraph
from gepa_dapo_grn.graph.validators import validate_graph_for_promotion


def build_graph_feedback(
    graph: ReasoningGraph,
    final_answer: str,
    base_rewards: Optional[Mapping[str, float]] = None,
    base_tags: Optional[Mapping[str, Any]] = None,
    judge_fn: Optional[Callable[[ReasoningGraph, str], float]] = None,
) -> GEPAFeedback:
    """Build GEPA feedback from public graph artifacts only."""

    unsupported = unsupported_claim_count(graph)
    critical_unsupported = critical_unsupported_claim_count(graph)
    contradictions = contradiction_count(graph)
    alignment = graph_to_answer_alignment_score(graph, final_answer, judge_fn=judge_fn)
    promotion_result = validate_graph_for_promotion(graph)

    rewards: Dict[str, float] = dict(base_rewards or {})
    rewards.update(
        {
            "graph_completeness": graph_completeness(graph),
            "claim_support": claim_support_coverage(graph),
            "contradiction_penalty": -float(contradictions),
            "unsupported_claim_penalty": -float(unsupported),
            "graph_answer_alignment": alignment,
        }
    )

    tags: Dict[str, float] = {}
    for key, value in dict(base_tags or {}).items():
        try:
            tags[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    tags.update(
        {
            "has_graph": 1.0,
            "unsupported_claims": float(unsupported),
            "critical_unsupported_claims": float(critical_unsupported),
            "contradictions": float(contradictions),
            "graph_valid_for_promotion": 1.0 if promotion_result.passed else 0.0,
        }
    )

    return GEPAFeedback(
        rewards=rewards,
        tags=tags,
        meta={
            "reasoning_graph": graph.to_dict(),
            "graph_metrics": {
                "graph_completeness": rewards["graph_completeness"],
                "claim_support": rewards["claim_support"],
                "contradictions": contradictions,
                "unsupported_claims": unsupported,
                "critical_unsupported_claims": critical_unsupported,
                "graph_answer_alignment": alignment,
            },
        },
    )
