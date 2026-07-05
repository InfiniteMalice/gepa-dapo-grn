from gepa_dapo_grn.graph import (
    Claim,
    Contradiction,
    EvidenceLink,
    GraphEdge,
    GraphNode,
    ReasoningGraph,
    validate_claim_support,
    validate_graph_for_promotion,
    validate_no_unresolved_critical_contradictions,
)


def test_unsupported_critical_claim_is_detected() -> None:
    graph = ReasoningGraph(claims=[Claim(id="c1", text="Critical", critical=True)])

    result = validate_claim_support(graph)

    assert not result.passed
    assert result.metrics["critical_unsupported_claims"] == 1


def test_contradicted_claim_is_detected() -> None:
    graph = ReasoningGraph(
        nodes=[
            GraphNode(id="c1", type="claim", label="A"),
            GraphNode(id="c2", type="counterclaim", label="not A"),
        ],
        edges=[
            GraphEdge(
                id="e1",
                source="c2",
                target="c1",
                type="contradicts",
                metadata={"critical": True},
            )
        ],
        claims=[Claim(id="c1", text="A"), Claim(id="c2", text="not A")],
        contradictions=[
            Contradiction(
                id="k1",
                claim_id="c1",
                counterclaim_id="c2",
                critical=True,
                resolved=False,
            )
        ],
    )

    result = validate_no_unresolved_critical_contradictions(graph)

    assert not result.passed
    assert result.metrics["contradictions"] == 2


def test_valid_graph_passes_promotion_validation() -> None:
    graph = ReasoningGraph(
        nodes=[
            GraphNode(id="e1", type="evidence", label="paper"),
            GraphNode(id="c1", type="claim", label="Supported"),
        ],
        edges=[GraphEdge(id="s1", source="e1", target="c1", type="supports")],
        claims=[Claim(id="c1", text="Supported", critical=True)],
        evidence_links=[EvidenceLink(id="l1", claim_id="c1", evidence_node_id="e1")],
    )

    result = validate_graph_for_promotion(graph)

    assert result.passed
    assert result.metrics["valid_for_promotion"] is True
