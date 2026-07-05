"""Deterministic metrics for public reasoning graphs."""

from __future__ import annotations

import re
from typing import Callable, Optional, Set

from gepa_dapo_grn.graph.schema import Claim, ReasoningGraph

_SUPPORT_EDGE_TYPES = {"supports", "tests", "implements"}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def _claim_node_ids(graph: ReasoningGraph, claim: Claim) -> Set[str]:
    ids = {claim.id}
    ids.update(node.id for node in graph.nodes if node.type == "claim" and node.label == claim.text)
    return ids


def _supported_claim_ids(graph: ReasoningGraph) -> Set[str]:
    supported = {link.claim_id for link in graph.evidence_links}
    for claim in graph.claims:
        claim_ids = _claim_node_ids(graph, claim)
        if any(
            edge.target in claim_ids and edge.type in _SUPPORT_EDGE_TYPES for edge in graph.edges
        ):
            supported.add(claim.id)
    return supported


def graph_completeness(graph: ReasoningGraph) -> float:
    """Score presence of core public reasoning artifact components."""

    has_evidence = bool(graph.evidence_links) or any(
        node.type == "evidence" for node in graph.nodes
    )
    components = [
        bool(graph.claims) or any(node.type == "claim" for node in graph.nodes),
        has_evidence,
        any(node.type == "mechanism" for node in graph.nodes),
        any(node.type == "assumption" for node in graph.nodes),
        any(node.type == "constraint" for node in graph.nodes),
    ]
    return sum(1.0 for present in components if present) / float(len(components))


def claim_support_coverage(graph: ReasoningGraph) -> float:
    """Return supported claims divided by total claims."""

    if not graph.claims:
        return 1.0
    return len(_supported_claim_ids(graph)) / float(len(graph.claims))


def contradiction_count(graph: ReasoningGraph) -> int:
    """Count explicit contradiction records plus contradicts edges."""

    return len(graph.contradictions) + sum(1 for edge in graph.edges if edge.type == "contradicts")


def unsupported_claim_count(graph: ReasoningGraph) -> int:
    """Count claims with no public evidence/support relation."""

    supported = _supported_claim_ids(graph)
    return sum(1 for claim in graph.claims if claim.id not in supported)


def critical_unsupported_claim_count(graph: ReasoningGraph) -> int:
    """Count critical claims without support."""

    supported = _supported_claim_ids(graph)
    return sum(1 for claim in graph.claims if claim.critical and claim.id not in supported)


def evidence_path_count(graph: ReasoningGraph, claim_id: str) -> int:
    """Count direct public evidence/support paths to a claim."""

    count = sum(1 for link in graph.evidence_links if link.claim_id == claim_id)
    claim = next((item for item in graph.claims if item.id == claim_id), None)
    claim_ids = {claim_id} if claim is None else _claim_node_ids(graph, claim)
    count += sum(
        1 for edge in graph.edges if edge.target in claim_ids and edge.type in _SUPPORT_EDGE_TYPES
    )
    return count


def _tokens(text: str) -> Set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def graph_to_answer_alignment_score(
    graph: ReasoningGraph,
    final_answer: str,
    judge_fn: Optional[Callable[[ReasoningGraph, str], float]] = None,
) -> float:
    """Score whether public claims are reflected in the final answer."""

    if judge_fn is not None:
        return max(0.0, min(1.0, float(judge_fn(graph, final_answer))))
    if not graph.claims:
        return 1.0

    answer_tokens = _tokens(final_answer)
    if not answer_tokens:
        return 0.0

    aligned = 0
    for claim in graph.claims:
        claim_tokens = _tokens(claim.text)
        if not claim_tokens:
            continue
        overlap = claim_tokens.intersection(answer_tokens)
        if len(overlap) / float(len(claim_tokens)) >= 0.35:
            aligned += 1
    return aligned / float(len(graph.claims))
