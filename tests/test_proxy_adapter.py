"""The seam between the agent and the proxy.

This is the one place where two independently-tested halves have to agree on a
wire format, and nothing else checks it: the agent suite asserts what the graph
emits, the proxy suite asserts what the endpoint returns, and a mismatch between
them produces a working run with a blank panel and no error anywhere.

So these tests drive the *real* compiled graph, capture its actual
``stream_mode="updates"`` chunks, and push them through the proxy's own parsing
functions — the same code path a live turn takes.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("langgraph")
pytest.importorskip("fastapi", reason="proxy dependencies are not installed")

from proxy.main import (  # noqa: E402
    CAUSAL_STATE_PREFIX,
    STAGE_BY_NODE,
    _iter_node_deltas,
    _merge_usage,
    _resolve_stage,
)
from src.causal import state as st  # noqa: E402
from src.causal.graph_app import build_causal_langgraph_app  # noqa: E402
from src.causal.models import ParsedCausalQuery  # noqa: E402
from src.causal.problem import build_causal_api  # noqa: E402
from tests.test_graph_app import StubModel  # noqa: E402


@pytest.fixture(scope="module")
def chunks():
    """Real chunks from a real run — not a fixture anyone hand-wrote."""
    api = build_causal_api()
    model = StubModel(
        parsed=ParsedCausalQuery(
            query_type="optimal_policy", context={"demand": 13.0}
        ),
        interpret_usage={
            "input_tokens": 900, "output_tokens": 40, "total_tokens": 940
        },
        explain_usage={
            "input_tokens": 500, "output_tokens": 200, "total_tokens": 700
        },
    )
    app = build_causal_langgraph_app(
        causal_api=api, model=model, default_num_samples=512, default_seed=123
    )
    return list(
        app.stream(
            {"user_input": "[[run:abc123]] Demand is 13."},
            stream_mode="updates",
        )
    )


def replay(chunks):
    """Re-run the proxy's consumption loop over captured chunks.

    Mirrors ``_agent_stream``'s inner loop exactly, including the two channels
    that must be re-reduced here — see the accumulation test below for why.
    """
    causal_state: dict = {}
    usage_totals: dict = {}
    stages: list[str] = []
    graph_frames: list[dict] = []
    forwarded: list[str] = []

    for chunk in chunks:
        for node, delta in _iter_node_deltas(chunk):
            resolved = _resolve_stage(node)
            if resolved:
                stages.append(resolved)
            for key, value in delta.items():
                if not key.startswith(CAUSAL_STATE_PREFIX):
                    continue
                if key == "causal_steps":
                    lines = value or []
                    forwarded.extend(lines)
                    causal_state.setdefault(key, []).extend(lines)
                elif key == "causal_usage":
                    _merge_usage(usage_totals, value)
                else:
                    causal_state[key] = value
            if delta.get("causal_graph"):
                graph_frames.append(delta["causal_graph"])

    return causal_state, stages, graph_frames, usage_totals, forwarded


def test_every_node_the_graph_runs_has_a_stage(chunks):
    """A node the proxy cannot map produces a chunk that lights nothing in the
    timeline — the failure is invisible, so assert coverage directly."""
    emitted = {node for chunk in chunks for node, _ in _iter_node_deltas(chunk)}
    unmapped = emitted - set(STAGE_BY_NODE)
    assert not unmapped, f"nodes with no stage mapping: {sorted(unmapped)}"


def test_stages_arrive_in_pipeline_order(chunks):
    _, stages, _, _, _ = replay(chunks)
    assert stages == ["interpret", "validate", "compute", "explain"]


def test_report_fields_are_all_populated(chunks):
    """Exactly the keys the endpoint copies into the `done` frame."""
    causal_state, _, _, _, _ = replay(chunks)

    assert causal_state["causal_final_answer"] == "Narrated answer."
    assert causal_state["causal_status"]["phase"] == "complete"
    assert causal_state["causal_decision"]["optimal_decision"] in (0, 1, 2)
    assert causal_state["causal_graph"]["nodes"]
    assert len(causal_state["causal_steps"]) >= 4
    assert causal_state["causal_run_id"] == "abc123"


def test_updates_mode_yields_raw_node_returns_not_reduced_channels(chunks):
    """The single most load-bearing fact about this transport.

    LangGraph's ``updates`` mode yields what the node *returned*, not the value
    the channel holds after its reducer ran. So both reduced channels have to be
    reduced a second time on the proxy side: ``causal_steps`` accumulated and
    ``causal_usage`` summed. Assuming otherwise costs the timeline every line
    except the last node's, and bills every turn at the explainer's usage alone.
    """
    steps_per_chunk = [
        delta["causal_steps"]
        for chunk in chunks
        for _, delta in _iter_node_deltas(chunk)
        if delta.get("causal_steps")
    ]
    # Disjoint, not cumulative: no chunk repeats an earlier chunk's lines.
    assert steps_per_chunk[0] == ["[interpret] optimal_policy; observed demand=13"]
    assert all(
        steps_per_chunk[0][0] not in later for later in steps_per_chunk[1:]
    )

    usage_per_chunk = [
        delta["causal_usage"]["total_tokens"]
        for chunk in chunks
        for _, delta in _iter_node_deltas(chunk)
        if delta.get("causal_usage")
    ]
    assert usage_per_chunk == [940, 700], "expected per-call usage, not totals"


def test_usage_sums_across_the_turn(chunks):
    _, _, _, usage_totals, _ = replay(chunks)
    assert usage_totals == {
        "input_tokens": 1400,
        "output_tokens": 240,
        "total_tokens": 1640,
    }


def test_graph_frames_are_emitted_more_than_once(chunks):
    """One at interpretation (topology + what is clamped), one at compute (with
    the recommended action). A single frame would mean the pane only fills in at
    the end."""
    _, _, frames, _, _ = replay(chunks)
    assert len(frames) >= 2
    first, last = frames[0], frames[-1]
    assert {n["id"] for n in first["nodes"]} == {n["id"] for n in last["nodes"]}
    decision = next(n for n in last["nodes"] if n["id"] == "decision")
    assert decision["status"] == "done"


def test_every_trace_line_is_forwarded_exactly_once(chunks):
    """What the UI timeline shows must equal what the report holds — no line
    dropped, none shown twice."""
    causal_state, _, _, _, forwarded = replay(chunks)
    assert forwarded == causal_state["causal_steps"]
    assert len(forwarded) == len(set(forwarded)), "a trace line was sent twice"


def test_state_keys_the_proxy_reads_match_the_agent_contract(chunks):
    """UI_KEYS is the agent's declaration of what it forwards. Anything listed
    there that a run never produces is a field the UI will always see as null."""
    causal_state, _, _, _, _ = replay(chunks)
    produced = set(causal_state)
    # causal_posteriors is only written by the posterior_summary route.
    expected = set(st.UI_KEYS) - {st.KEY_POSTERIORS, st.KEY_USAGE}
    assert expected <= produced, f"never produced: {sorted(expected - produced)}"


def test_malformed_chunks_are_skipped_not_fatal():
    """LanggraphAgent runs chunks through dumpd; a non-conforming payload must
    not take the stream down."""
    assert list(_iter_node_deltas({"node": "not a dict"})) == []
    assert list(_iter_node_deltas("not a dict")) == []
    assert list(_iter_node_deltas({})) == []
    assert list(_iter_node_deltas({"a": {"x": 1}, "b": {"y": 2}})) == [
        ("a", {"x": 1}),
        ("b", {"y": 2}),
    ]
