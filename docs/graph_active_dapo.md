# Graph-Active-DAPO

Graph-Active-DAPO is an optional layer for adding public graph feedback and adaptive references to
the existing GEPA-shaped DAPO workflow. It does not replace `DAPOTrainer`, MaxRL, GRN, verifier
hooks, or reward mixers.

## Graph Feedback

Graph feedback represents public reasoning artifacts as structured `ReasoningGraph` data:

- nodes such as claims, evidence, mechanisms, assumptions, constraints, risks, tests, and results;
- typed edges such as supports, contradicts, tests, bounds, refines, and implements;
- public claims, evidence links, and contradiction records.

The graph metrics are deterministic by default:

- `graph_completeness`
- `claim_support_coverage`
- `unsupported_claim_count`
- `critical_unsupported_claim_count`
- `contradiction_count`
- `graph_to_answer_alignment_score`

`build_graph_feedback(...)` stores the graph and metrics in GEPA-style feedback metadata and emits
reward dimensions for completeness, support, contradiction penalties, unsupported-claim penalties,
and answer alignment.

```python
from gepa_dapo_grn.graph import Claim, EvidenceLink, GraphNode, ReasoningGraph, build_graph_feedback

graph = ReasoningGraph(
    nodes=[GraphNode(id="e1", type="evidence", label="external check")],
    claims=[Claim(id="c1", text="The answer passed an external check", critical=True)],
    evidence_links=[EvidenceLink(id="l1", claim_id="c1", evidence_node_id="e1")],
)
feedback = build_graph_feedback(graph, "The answer passed an external check.")
```

## Active-GRPO-Style References

The active reference utilities keep a per-prompt reference output and score, track the best policy
candidate, and choose a training mode:

- `imitate` when no better policy candidate exists;
- `mixed` when reference and policy scores are close;
- `reinforce` when the policy candidate beats the reference by the configured margin.

```python
from gepa_dapo_grn.active_grpo import ActiveGRPOScheduler

scheduler = ActiveGRPOScheduler(margin=0.02, mixed_band=0.01)
mode = scheduler.choose_mode(reference_score=0.8, best_policy_score=0.83)
```

## Safety And Alignment

This module does not reward hidden chain-of-thought. It only scores public artifacts: graph JSON,
assumptions, claims, evidence links, tests, answer summaries, and verifier outputs.

Promotion gates are intentionally strict. A policy candidate cannot become a new reference from
self-score alone when an external verifier is required. Promotion also fails on safety violations,
unsupported critical claims, unsupported claims beyond the configured limit, and contradictions
beyond the configured limit.

## Wrapper Usage

`GraphActiveDapoTrainer` is a lightweight wrapper. It builds feedback, chooses an active-reference
mode, optionally evaluates promotion, and delegates to an existing trainer object if that object
exposes `record_feedback`, `consume_feedback`, or `train_on_feedback`.

```python
from gepa_dapo_grn.trainers import GraphActiveDapoTrainer

def verifier(prompt, output, reference_output):
    return {"score": 0.9 if "verified" in output else 0.5}

wrapper = GraphActiveDapoTrainer(base_trainer=trainer, verifier_fn=verifier)
decision = wrapper.decide(
    prompt_id="task-1",
    prompt="Solve the task.",
    reference_output="dataset reference",
    reference_score=0.7,
    policy_candidates=["verified policy answer"],
)
```

Graph feedback and active references can be used independently. Set
`GraphActiveDapoConfig(use_graph_feedback=False)` to run without graph feedback, or import and use
`gepa_dapo_grn.graph` directly without the active-reference scheduler.

## Plugging In A Domain Verifier

Pass a verifier function that accepts `(prompt, output, reference_output)` and returns either a
float score or a mapping with a `score` key. The package makes no provider calls, downloads no
datasets, and does not require network access.
