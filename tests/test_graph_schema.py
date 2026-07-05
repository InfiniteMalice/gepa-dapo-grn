from gepa_dapo_grn.graph import (
    Claim,
    Contradiction,
    EvidenceLink,
    GraphEdge,
    GraphNode,
    ReasoningGraph,
)


def test_graph_schema_roundtrip() -> None:
    graph = ReasoningGraph(
        nodes=[
            GraphNode(id="e1", type="evidence", label="public experiment"),
            GraphNode(id="c1", type="claim", label="The treatment improves recall"),
        ],
        edges=[GraphEdge(id="edge1", source="e1", target="c1", type="supports")],
        claims=[Claim(id="c1", text="The treatment improves recall", critical=True)],
        evidence_links=[EvidenceLink(id="link1", claim_id="c1", evidence_node_id="e1", score=0.9)],
    )

    assert graph.validate_basic() == []

    from_dict = ReasoningGraph.from_dict(graph.to_dict())
    from_json = ReasoningGraph.from_json(graph.to_json())

    assert from_dict.to_dict() == graph.to_dict()
    assert from_json.to_dict() == graph.to_dict()


def test_graph_basic_validation_reports_missing_edge_endpoint() -> None:
    graph = ReasoningGraph(
        nodes=[GraphNode(id="a", type="concept", label="A")],
        edges=[GraphEdge(id="bad", source="a", target="missing", type="supports")],
    )

    assert "edge bad target does not exist: missing" in graph.validate_basic()


def test_graph_basic_validation_reports_duplicate_evidence_and_contradiction_ids() -> None:
    graph = ReasoningGraph(
        claims=[Claim(id="c1", text="A"), Claim(id="c2", text="not A")],
        evidence_links=[
            EvidenceLink(id="dup", claim_id="c1"),
            EvidenceLink(id="dup", claim_id="c1"),
        ],
        contradictions=[
            Contradiction(id="dup", claim_id="c1", counterclaim_id="c2"),
            Contradiction(id="dup", claim_id="c1", counterclaim_id="c2"),
        ],
    )

    errors = graph.validate_basic()

    assert "duplicate evidence link id: dup" in errors
    assert "duplicate contradiction id: dup" in errors


def test_graph_basic_validation_requires_evidence_link_node_type() -> None:
    graph = ReasoningGraph(
        nodes=[GraphNode(id="not-evidence", type="claim", label="A claim node")],
        claims=[Claim(id="c1", text="A")],
        evidence_links=[EvidenceLink(id="link1", claim_id="c1", evidence_node_id="not-evidence")],
    )

    assert (
        "evidence link link1 evidence node is not evidence: not-evidence" in graph.validate_basic()
    )
