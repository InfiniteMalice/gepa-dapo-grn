"""Schema objects for public graph reasoning artifacts."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import field
from typing import Any, Dict, List, Mapping, Optional

from gepa_dapo_grn._compat import dataclass

ALLOWED_NODE_TYPES = {
    "concept",
    "mechanism",
    "assumption",
    "evidence",
    "constraint",
    "risk",
    "claim",
    "counterclaim",
    "experiment",
    "test",
    "result",
}

ALLOWED_EDGE_TYPES = {
    "supports",
    "contradicts",
    "causes",
    "depends_on",
    "tests",
    "bounds",
    "generalizes_to",
    "fails_under",
    "refines",
    "implements",
}


def _dict(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(value or {})


@dataclass(slots=True)
class GraphNode:
    """A public reasoning graph node."""

    id: str
    type: str
    label: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GraphNode":
        return cls(
            id=str(payload["id"]),
            type=str(payload["type"]),
            label=str(payload.get("label", payload["id"])),
            metadata=_dict(payload.get("metadata")),
        )


@dataclass(slots=True)
class GraphEdge:
    """A typed relation between public graph nodes."""

    id: str
    source: str
    target: str
    type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GraphEdge":
        return cls(
            id=str(payload["id"]),
            source=str(payload["source"]),
            target=str(payload["target"]),
            type=str(payload["type"]),
            metadata=_dict(payload.get("metadata")),
        )


@dataclass(slots=True)
class Claim:
    """A public claim that can be supported, contradicted, or marked critical."""

    id: str
    text: str
    critical: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "critical": bool(self.critical),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Claim":
        return cls(
            id=str(payload["id"]),
            text=str(payload.get("text", "")),
            critical=bool(payload.get("critical", False)),
            metadata=_dict(payload.get("metadata")),
        )


@dataclass(slots=True)
class EvidenceLink:
    """A public evidence pointer attached to a claim."""

    id: str
    claim_id: str
    evidence_node_id: Optional[str] = None
    description: str = ""
    url: Optional[str] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "claim_id": self.claim_id,
            "evidence_node_id": self.evidence_node_id,
            "description": self.description,
            "url": self.url,
            "score": self.score,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceLink":
        score = payload.get("score")
        return cls(
            id=str(payload["id"]),
            claim_id=str(payload["claim_id"]),
            evidence_node_id=(
                str(payload["evidence_node_id"])
                if payload.get("evidence_node_id") is not None
                else None
            ),
            description=str(payload.get("description", "")),
            url=str(payload["url"]) if payload.get("url") is not None else None,
            score=float(score) if score is not None else None,
            metadata=_dict(payload.get("metadata")),
        )


@dataclass(slots=True)
class Contradiction:
    """A public contradiction record between claims."""

    id: str
    claim_id: str
    counterclaim_id: Optional[str] = None
    description: str = ""
    resolved: bool = False
    critical: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "claim_id": self.claim_id,
            "counterclaim_id": self.counterclaim_id,
            "description": self.description,
            "resolved": bool(self.resolved),
            "critical": bool(self.critical),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Contradiction":
        return cls(
            id=str(payload["id"]),
            claim_id=str(payload["claim_id"]),
            counterclaim_id=(
                str(payload["counterclaim_id"])
                if payload.get("counterclaim_id") is not None
                else None
            ),
            description=str(payload.get("description", "")),
            resolved=bool(payload.get("resolved", False)),
            critical=bool(payload.get("critical", False)),
            metadata=_dict(payload.get("metadata")),
        )


@dataclass(slots=True)
class ReasoningGraph:
    """A public, serializable graph artifact for reward metadata."""

    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    claims: List[Claim] = field(default_factory=list)
    evidence_links: List[EvidenceLink] = field(default_factory=list)
    contradictions: List[Contradiction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "claims": [claim.to_dict() for claim in self.claims],
            "evidence_links": [link.to_dict() for link in self.evidence_links],
            "contradictions": [item.to_dict() for item in self.contradictions],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReasoningGraph":
        return cls(
            nodes=[GraphNode.from_dict(item) for item in payload.get("nodes", [])],
            edges=[GraphEdge.from_dict(item) for item in payload.get("edges", [])],
            claims=[Claim.from_dict(item) for item in payload.get("claims", [])],
            evidence_links=[
                EvidenceLink.from_dict(item) for item in payload.get("evidence_links", [])
            ],
            contradictions=[
                Contradiction.from_dict(item) for item in payload.get("contradictions", [])
            ],
            metadata=_dict(payload.get("metadata")),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "ReasoningGraph":
        return cls.from_dict(json.loads(payload))

    def validate_basic(self) -> List[str]:
        """Return validation errors for basic graph well-formedness."""

        errors: List[str] = []
        node_ids = [node.id for node in self.nodes]
        edge_ids = [edge.id for edge in self.edges]
        claim_ids = [claim.id for claim in self.claims]
        evidence_link_ids = [link.id for link in self.evidence_links]
        contradiction_ids = [contradiction.id for contradiction in self.contradictions]

        for item_name, values in (
            ("node", node_ids),
            ("edge", edge_ids),
            ("claim", claim_ids),
            ("evidence link", evidence_link_ids),
            ("contradiction", contradiction_ids),
        ):
            duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
            errors.extend(f"duplicate {item_name} id: {value}" for value in duplicates)

        node_id_set = set(node_ids)
        nodes_by_id = {node.id: node for node in self.nodes}
        claim_id_set = set(claim_ids)

        for node in self.nodes:
            if node.type not in ALLOWED_NODE_TYPES:
                errors.append(f"unsupported node type for {node.id}: {node.type}")

        for edge in self.edges:
            if edge.type not in ALLOWED_EDGE_TYPES:
                errors.append(f"unsupported edge type for {edge.id}: {edge.type}")
            if edge.source not in node_id_set:
                errors.append(f"edge {edge.id} source does not exist: {edge.source}")
            if edge.target not in node_id_set:
                errors.append(f"edge {edge.id} target does not exist: {edge.target}")

        for link in self.evidence_links:
            if link.claim_id not in claim_id_set:
                errors.append(f"evidence link {link.id} claim does not exist: {link.claim_id}")
            if link.evidence_node_id is not None and link.evidence_node_id not in node_id_set:
                errors.append(
                    f"evidence link {link.id} evidence node does not exist: "
                    f"{link.evidence_node_id}"
                )
            elif link.evidence_node_id is not None:
                evidence_node = nodes_by_id[link.evidence_node_id]
                if evidence_node.type != "evidence":
                    errors.append(
                        f"evidence link {link.id} evidence node is not evidence: "
                        f"{link.evidence_node_id}"
                    )

        for contradiction in self.contradictions:
            if contradiction.claim_id not in claim_id_set:
                errors.append(
                    f"contradiction {contradiction.id} claim does not exist: "
                    f"{contradiction.claim_id}"
                )
            if (
                contradiction.counterclaim_id is not None
                and contradiction.counterclaim_id not in claim_id_set
            ):
                errors.append(
                    f"contradiction {contradiction.id} counterclaim does not exist: "
                    f"{contradiction.counterclaim_id}"
                )

        return errors
