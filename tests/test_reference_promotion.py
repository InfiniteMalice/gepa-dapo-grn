from gepa_dapo_grn.active_grpo import (
    ActiveReference,
    PromotionCandidate,
    PromotionGateConfig,
    evaluate_promotion,
)
from gepa_dapo_grn.graph import Claim, EvidenceLink, GraphEdge, GraphNode, ReasoningGraph


def _valid_graph() -> ReasoningGraph:
    return ReasoningGraph(
        nodes=[
            GraphNode(id="e1", type="evidence", label="public verifier"),
            GraphNode(id="c1", type="claim", label="Verified claim"),
        ],
        edges=[GraphEdge(id="s1", source="e1", target="c1", type="supports")],
        claims=[Claim(id="c1", text="Verified claim", critical=True)],
        evidence_links=[EvidenceLink(id="l1", claim_id="c1", evidence_node_id="e1")],
    )


def test_self_scored_candidate_fails_when_external_verifier_required() -> None:
    reference = ActiveReference("p1", "ref", 0.5)
    candidate = PromotionCandidate("p1", "better", 0.8, "self", graph=_valid_graph())

    result = evaluate_promotion(candidate, reference, PromotionGateConfig())

    assert not result.passed
    assert "external verifier is required for promotion" in result.errors


def test_candidate_must_clear_reference_margin() -> None:
    reference = ActiveReference("p1", "ref", 0.5)
    candidate = PromotionCandidate("p1", "near", 0.51, "unit-test", graph=_valid_graph())

    result = evaluate_promotion(candidate, reference, PromotionGateConfig(margin=0.02))

    assert not result.passed
    assert "candidate score does not beat reference by required margin" in result.errors


def test_safety_violation_fails() -> None:
    reference = ActiveReference("p1", "ref", 0.5)
    candidate = PromotionCandidate(
        "p1",
        "better",
        0.8,
        "unit-test",
        graph=_valid_graph(),
        safety_violation=True,
    )

    result = evaluate_promotion(candidate, reference, PromotionGateConfig())

    assert not result.passed
    assert "candidate has a safety violation" in result.errors


def test_unsupported_critical_claim_fails() -> None:
    reference = ActiveReference("p1", "ref", 0.5)
    graph = ReasoningGraph(claims=[Claim(id="c1", text="Critical", critical=True)])
    candidate = PromotionCandidate("p1", "better", 0.8, "unit-test", graph=graph)

    result = evaluate_promotion(candidate, reference, PromotionGateConfig())

    assert not result.passed
    assert result.metrics["critical_unsupported_claims"] == 1


def test_graph_limits_report_unsupported_claims_and_contradictions() -> None:
    reference = ActiveReference("p1", "ref", 0.5)
    graph = ReasoningGraph(
        nodes=[
            GraphNode(id="c1", type="claim", label="A"),
            GraphNode(id="c2", type="counterclaim", label="not A"),
        ],
        edges=[GraphEdge(id="e1", source="c2", target="c1", type="contradicts")],
        claims=[Claim(id="c1", text="A"), Claim(id="c2", text="not A")],
    )
    candidate = PromotionCandidate("p1", "better", 0.8, "unit-test", graph=graph)

    result = evaluate_promotion(candidate, reference, PromotionGateConfig())

    assert not result.passed
    assert result.metrics["unsupported_claims"] == 2
    assert result.metrics["contradictions"] == 1
    assert "candidate exceeds unsupported claim limit" in result.errors
    assert "candidate exceeds contradiction limit" in result.errors


def test_no_graph_candidate_warns_but_can_pass_other_gates() -> None:
    reference = ActiveReference("p1", "ref", 0.5)
    candidate = PromotionCandidate("p1", "better", 0.8, "unit-test")

    result = evaluate_promotion(candidate, reference, PromotionGateConfig())

    assert result.passed
    assert result.warnings == ["candidate has no graph artifact"]


def test_verified_better_candidate_passes() -> None:
    reference = ActiveReference("p1", "ref", 0.5)
    candidate = PromotionCandidate("p1", "better", 0.8, "unit-test", graph=_valid_graph())

    result = evaluate_promotion(candidate, reference, PromotionGateConfig())

    assert result.passed
