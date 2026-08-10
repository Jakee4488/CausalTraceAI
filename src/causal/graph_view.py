"""The causal model, shaped for the UI's graph pane.

TracerLensAi's graph pane rendered a *task* DAG built by an LLM decomposer. This
project has no decomposer — but it does have a real causal graph, the one the
engine actually computes over: independent root causes and the decision, all
feeding a deterministic utility.

So the pane renders that instead, emitting the identical
``{nodes, edges, critical_path, version}`` payload ``to_ui_graph()`` produced,
which means ``CausalGraph.tsx``, its dagre layout and its status colours need no
changes at all.

Node status carries the interesting part
----------------------------------------
The topology is fixed — it is the same four-into-one fan on every turn, and a
picture of it alone would be wallpaper. What changes per query is which causes
were *clamped* and which stayed uncertain, so that is what status encodes:

    intervened on — ``do(C = c)``        -> active   (being manipulated)
    observed in context — ``C = c``      -> done     (known)
    neither: left stochastic             -> pending  (still uncertain)

That mapping is the reason to keep the pane: read the graph and you can see, at
a glance, exactly how much of the world the answer had pinned down.
"""

from __future__ import annotations

from typing import Any, Optional

# UI node kinds, reused from the renderer's existing colour mapping.
KIND_CAUSE = "input"
KIND_DECISION = "process"
KIND_UTILITY = "outcome"

DECISION_NODE_ID = "decision"
UTILITY_NODE_ID = "utility"

# Structural edges, asserted by the model rather than inferred, so they carry
# full confidence — the renderer softens anything below 1.0.
_EDGE_CONFIDENCE = 1.0


def _cause_status(
    name: str,
    context: dict[str, float],
    interventions: dict[str, float],
) -> str:
    if name in interventions:
        return "active"
    if name in context:
        return "done"
    return "pending"


def build_ui_graph(
    graph_description: dict[str, Any],
    context: Optional[dict[str, float]] = None,
    interventions: Optional[dict[str, float]] = None,
    chosen_decision: Optional[Any] = None,
) -> dict[str, Any]:
    """Render ``CausalDecisionAPI.describe_graph()`` as the UI graph payload.

    Args:
        graph_description: the output of ``describe_graph()``.
        context: causes the user observed, so they were clamped to a value.
        interventions: causes forced via ``do(C = c)``.
        chosen_decision: the recommended action, if the query produced one —
            it marks the decision node ``done`` rather than ``pending``.
    """
    context = context or {}
    interventions = interventions or {}
    causes = graph_description.get("causes") or {}
    decisions = graph_description.get("decisions") or {}

    nodes: list[dict[str, Any]] = [
        {
            "id": name,
            "label": name.replace("_", " "),
            "kind": KIND_CAUSE,
            "status": _cause_status(name, context, interventions),
            "description": (schema or {}).get("description", ""),
        }
        for name, schema in causes.items()
    ]

    # One node for D, labelled with the recommended action once there is one.
    # Before that it is the open question the whole graph exists to answer.
    if chosen_decision is not None:
        decision_label = decisions.get(
            str(chosen_decision), str(chosen_decision)
        )
        decision_status = "done"
    else:
        decision_label = "decision"
        decision_status = "pending"

    nodes.append(
        {
            "id": DECISION_NODE_ID,
            "label": decision_label,
            "kind": KIND_DECISION,
            "status": decision_status,
            "description": "; ".join(
                f"{key}: {value}" for key, value in decisions.items()
            )[:200],
        }
    )

    nodes.append(
        {
            "id": UTILITY_NODE_ID,
            "label": "utility",
            "kind": KIND_UTILITY,
            "status": "done" if chosen_decision is not None else "pending",
            "description": (
                graph_description.get("utility_description", "") or ""
            )[:200],
        }
    )

    edges = [
        {
            "source": name,
            "target": UTILITY_NODE_ID,
            "relation": "causes",
            "confidence": _EDGE_CONFIDENCE,
            "rationale": "Every cause is a parent of utility.",
        }
        for name in causes
    ]
    edges.append(
        {
            "source": DECISION_NODE_ID,
            "target": UTILITY_NODE_ID,
            "relation": "causes",
            "confidence": _EDGE_CONFIDENCE,
            "rationale": "Utility is deterministic given the decision and causes.",
        }
    )

    return {
        "nodes": nodes,
        "edges": edges,
        # The path the answer is *about*: choosing D to move U.
        "critical_path": [DECISION_NODE_ID, UTILITY_NODE_ID],
        "version": 1,
    }


def interventions_from_parsed(parsed: dict[str, Any]) -> dict[str, float]:
    """Pull ``{variable: value}`` out of a ParsedCausalQuery dump.

    Only the target arm: the baseline is the comparison, and showing both would
    paint one node with two conflicting statuses.
    """
    intervention = (parsed or {}).get("intervention")
    if not intervention:
        return {}
    variable = intervention.get("variable")
    if not variable:
        return {}
    return {variable: intervention.get("value")}
