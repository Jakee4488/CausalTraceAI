"""The graph pane payload.

These are pure dict tests — no torch needed for most of them — because
``build_ui_graph`` takes ``describe_graph()`` output, not an engine.
"""

from __future__ import annotations

import pytest

from src.causal.graph_view import (
    DECISION_NODE_ID,
    UTILITY_NODE_ID,
    build_ui_graph,
    interventions_from_parsed,
)

DESCRIPTION = {
    "causes": {
        "demand": {"name": "demand", "type": "continuous", "description": "Units."},
        "market_growth": {
            "name": "market_growth",
            "type": "continuous",
            "description": "Rate.",
        },
        "competitor_active": {
            "name": "competitor_active",
            "type": "binary",
            "allowed_values": [0, 1],
            "description": "Rival campaign.",
        },
    },
    "decision_variable": "D",
    "decisions": {"0": "Conservative", "1": "Balanced", "2": "Aggressive"},
    "utility_variable": "U",
    "utility_description": "Profit-like reward.",
}

CAUSE_IDS = {"demand", "market_growth", "competitor_active"}


def node_by_id(graph, node_id):
    return next(n for n in graph["nodes"] if n["id"] == node_id)


def test_topology_is_every_cause_plus_decision_into_utility():
    graph = build_ui_graph(DESCRIPTION)

    assert {n["id"] for n in graph["nodes"]} == CAUSE_IDS | {
        DECISION_NODE_ID,
        UTILITY_NODE_ID,
    }
    # Every edge terminates at utility: that is the model's whole structure.
    assert {e["target"] for e in graph["edges"]} == {UTILITY_NODE_ID}
    assert {e["source"] for e in graph["edges"]} == CAUSE_IDS | {DECISION_NODE_ID}
    assert graph["critical_path"] == [DECISION_NODE_ID, UTILITY_NODE_ID]


def test_status_distinguishes_clamped_from_stochastic():
    """The reason the pane is worth rendering: which causes the answer had
    pinned down, and which it integrated over."""
    graph = build_ui_graph(
        DESCRIPTION,
        context={"demand": 13.0},
        interventions={"competitor_active": 1.0},
    )

    assert node_by_id(graph, "demand")["status"] == "done"          # observed
    assert node_by_id(graph, "competitor_active")["status"] == "active"  # do()
    assert node_by_id(graph, "market_growth")["status"] == "pending"     # stochastic


def test_every_cause_is_pending_when_nothing_is_known():
    graph = build_ui_graph(DESCRIPTION)
    assert all(
        node_by_id(graph, name)["status"] == "pending" for name in CAUSE_IDS
    )


def test_decision_node_takes_the_recommended_action_label():
    before = build_ui_graph(DESCRIPTION)
    assert node_by_id(before, DECISION_NODE_ID)["label"] == "decision"
    assert node_by_id(before, DECISION_NODE_ID)["status"] == "pending"

    after = build_ui_graph(DESCRIPTION, chosen_decision=1)
    assert node_by_id(after, DECISION_NODE_ID)["label"] == "Balanced"
    assert node_by_id(after, DECISION_NODE_ID)["status"] == "done"
    assert node_by_id(after, UTILITY_NODE_ID)["status"] == "done"


def test_payload_matches_the_renderer_contract():
    """Field-for-field what CausalGraph.tsx reads. Drift here renders an empty
    pane with no error anywhere."""
    graph = build_ui_graph(DESCRIPTION, context={"demand": 13.0})

    assert set(graph) == {"nodes", "edges", "critical_path", "version"}
    for node in graph["nodes"]:
        assert set(node) == {"id", "label", "kind", "status", "description"}
        assert node["kind"] in {"input", "process", "outcome"}
        assert node["status"] in {
            "pending", "active", "done", "failed", "invalidated", "replanned"
        }
    for edge in graph["edges"]:
        assert set(edge) == {
            "source", "target", "relation", "confidence", "rationale"
        }
        assert 0.0 <= edge["confidence"] <= 1.0


def test_descriptions_are_capped_for_the_renderer():
    """The old model validators capped these at 200 chars and the pane's layout
    still assumes it."""
    wide = dict(DESCRIPTION, utility_description="x" * 500)
    graph = build_ui_graph(wide)
    assert len(node_by_id(graph, UTILITY_NODE_ID)["description"]) <= 200


def test_interventions_from_parsed_takes_only_the_target_arm():
    parsed = {
        "intervention": {
            "variable": "competitor_active",
            "value": 1.0,
            "baseline_value": 0.0,
        }
    }
    # Not the baseline: one node cannot carry two conflicting statuses.
    assert interventions_from_parsed(parsed) == {"competitor_active": 1.0}
    assert interventions_from_parsed({}) == {}
    assert interventions_from_parsed({"intervention": None}) == {}


def test_matches_the_live_engine_description():
    """Guards against describe_graph() and the view drifting apart."""
    pytest.importorskip("torch")
    from src.causal.problem import build_causal_api

    graph = build_ui_graph(build_causal_api().describe_graph())
    assert {n["id"] for n in graph["nodes"]} == CAUSE_IDS | {
        DECISION_NODE_ID,
        UTILITY_NODE_ID,
    }
