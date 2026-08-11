"""The LangGraph pipeline routes, validates and streams correctly.

Every test here drives the real compiled graph and the real Pyro engine — only
the model is stubbed, because the one thing worth faking is the LLM. That means
these also cover the transport: what a node returns is what the proxy will see
as a progress frame, so asserting on ``stream_mode="updates"`` chunks asserts on
the actual wire format.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("langgraph")

from src.causal import state as st  # noqa: E402
from src.causal.graph_app import (  # noqa: E402
    build_causal_langgraph_app,
    extract_run_id,
    strip_markers,
)
from src.causal.models import ParsedCausalQuery  # noqa: E402
from src.causal.problem import build_causal_api  # noqa: E402

# Small on purpose: these tests exercise routing and transport, not precision.
# The parity suite is what pins the numbers.
SAMPLES = 512


class StubMessage:
    """An AIMessage as far as the pipeline is concerned: content + usage.

    ``text`` mirrors ``AIMessage.text``: Gemini 3 returns a list of content
    blocks rather than a bare string, and the explain node reads the joined
    text blocks so a thought signature never reaches the user.
    """

    def __init__(self, content="Narrated answer.", usage=None):
        self.content = content
        self.usage_metadata = usage

    @property
    def text(self):
        if isinstance(self.content, str):
            return self.content
        return "".join(
            block.get("text", "")
            for block in self.content
            if isinstance(block, dict) and block.get("type") == "text"
        )


class StubStructuredModel:
    """The `include_raw=True` envelope: {"raw", "parsed", "parsing_error"}."""

    def __init__(self, parsed, error=None, usage=None, parsing_error=None):
        self._parsed = parsed
        self._error = error
        self._usage = usage
        self._parsing_error = parsing_error
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self._error:
            raise self._error
        return {
            "raw": StubMessage("", self._usage),
            "parsed": None if self._parsing_error else self._parsed,
            "parsing_error": self._parsing_error,
        }


class StubModel:
    """Stands in for ChatGoogleGenerativeAI.

    Counts both call sites separately so a test can assert the pipeline spent
    one LLM call, not two, on a short-circuited query.
    """

    def __init__(
        self,
        parsed=None,
        interpret_error=None,
        explain_error=None,
        interpret_usage=None,
        explain_usage=None,
        parsing_error=None,
        explain_content="Narrated answer.",
    ):
        self._structured = StubStructuredModel(
            parsed, interpret_error, interpret_usage, parsing_error
        )
        self._explain_error = explain_error
        self._explain_usage = explain_usage
        self._explain_content = explain_content
        self.explain_calls = 0

    def with_structured_output(self, schema, include_raw=False):
        assert include_raw, "usage metadata is only reachable with include_raw"
        return self._structured

    def invoke(self, messages):
        self.explain_calls += 1
        if self._explain_error:
            raise self._explain_error
        return StubMessage(self._explain_content, self._explain_usage)

    @property
    def interpret_calls(self):
        return self._structured.calls


@pytest.fixture(scope="module")
def api():
    return build_causal_api()


def build(api, parsed=None, **kwargs):
    model = StubModel(parsed=parsed, **kwargs)
    app = build_causal_langgraph_app(
        causal_api=api, model=model, default_num_samples=SAMPLES, default_seed=123
    )
    return app, model


def run(app, text="Demand is 13. Which policy is best?"):
    return app.invoke({"user_input": text})


# ── Marker handling ─────────────────────────────────────────────────────────


def test_run_id_is_extracted_and_stripped():
    text = "[[run:abc-123]] Demand is 13. Which policy?"
    assert extract_run_id(text) == "abc-123"
    assert strip_markers(text) == "Demand is 13. Which policy?"
    assert extract_run_id("no marker here") is None


# ── Routing ─────────────────────────────────────────────────────────────────


def test_optimal_policy_route(api):
    parsed = ParsedCausalQuery(
        query_type="optimal_policy", context={"demand": 13.0}
    )
    app, model = build(api, parsed)
    final = run(app)

    assert final[st.KEY_ROUTE] == "optimal_policy"
    assert final[st.KEY_DECISION]["query_type"] == "optimal_policy"
    assert final[st.KEY_DECISION]["optimal_decision"] in api.decisions
    assert final[st.KEY_FINAL] == "Narrated answer."
    assert final[st.KEY_STATUS]["phase"] == "complete"
    # Two calls: interpret, then explain.
    assert (model.interpret_calls, model.explain_calls) == (1, 1)


def test_intervention_effect_route(api):
    parsed = ParsedCausalQuery(
        query_type="intervention_effect",
        context={"demand": 13.0},
        intervention={
            "variable": "competitor_active",
            "value": 1.0,
            "baseline_value": 0.0,
            "mode": "reoptimise_policy",
        },
    )
    app, _ = build(api, parsed)
    final = run(app)

    assert final[st.KEY_ROUTE] == "intervention_effect"
    result = final[st.KEY_DECISION]
    assert result["estimand"] == "reoptimised_policy_effect"
    assert "causal_effect" in result


def test_posterior_summary_route(api):
    parsed = ParsedCausalQuery(query_type="posterior_summary")
    app, _ = build(api, parsed)
    final = run(app)

    assert final[st.KEY_ROUTE] == "posterior_summary"
    assert set(final[st.KEY_POSTERIORS]) == set(api.causes)


def test_clarification_short_circuits_without_a_second_llm_call(api):
    parsed = ParsedCausalQuery(
        query_type="clarification_required",
        clarification_question="Which decision should be held fixed?",
    )
    app, model = build(api, parsed)
    final = run(app, "What happens if we force demand to change?")

    assert final[st.KEY_FINAL] == "Which decision should be held fixed?"
    assert final[st.KEY_STATUS]["phase"] == "clarification"
    # The whole point of terminating early: one call, not two.
    assert (model.interpret_calls, model.explain_calls) == (1, 0)


# ── Validation rejects what the prompt merely asked for ─────────────────────


def test_unknown_context_variable_is_rejected(api):
    parsed = ParsedCausalQuery(
        query_type="optimal_policy", context={"not_a_cause": 1.0}
    )
    app, model = build(api, parsed)
    final = run(app)

    assert final[st.KEY_ROUTE] == "error"
    assert "not_a_cause" in final[st.KEY_FINAL]
    assert final[st.KEY_STATUS]["phase"] == "failed"
    assert model.explain_calls == 0


def test_unknown_intervention_variable_is_rejected(api):
    parsed = ParsedCausalQuery(
        query_type="intervention_effect",
        intervention={
            "variable": "interest_rate",
            "value": 1.0,
            "mode": "reoptimise_policy",
        },
    )
    app, _ = build(api, parsed)
    final = run(app)

    assert final[st.KEY_ROUTE] == "error"
    assert "interest_rate" in final[st.KEY_FINAL]


def test_out_of_range_fixed_decision_is_rejected(api):
    parsed = ParsedCausalQuery(
        query_type="intervention_effect",
        intervention={
            "variable": "competitor_active",
            "value": 1.0,
            "mode": "fixed_policy",
            "fixed_decision": 99,
        },
    )
    app, _ = build(api, parsed)
    final = run(app)

    assert final[st.KEY_ROUTE] == "error"
    assert "allowed decisions" in final[st.KEY_FINAL]


def test_observe_and_intervene_conflict_becomes_an_error(api):
    """The engine raises on a contradiction; the node must turn that into an
    answer rather than a stack trace."""
    parsed = ParsedCausalQuery(
        query_type="intervention_effect",
        context={"competitor_active": 0.0},
        intervention={
            "variable": "competitor_active",
            "value": 1.0,
            "mode": "reoptimise_policy",
        },
    )
    app, _ = build(api, parsed)
    final = run(app)

    assert final[st.KEY_STATUS]["phase"] == "failed"
    assert "competitor_active" in final[st.KEY_FINAL]


def test_empty_input_is_rejected_before_any_llm_call(api):
    app, model = build(api, ParsedCausalQuery(query_type="optimal_policy"))
    final = run(app, "   ")

    assert final[st.KEY_STATUS]["phase"] == "failed"
    assert model.interpret_calls == 0


# ── Degradation ─────────────────────────────────────────────────────────────


def test_interpretation_failure_costs_no_compute(api):
    app, model = build(
        api,
        ParsedCausalQuery(query_type="optimal_policy"),
        interpret_error=RuntimeError("quota exhausted"),
    )
    final = run(app)

    assert final[st.KEY_STATUS]["phase"] == "failed"
    assert "quota exhausted" in final[st.KEY_FINAL]
    assert st.KEY_DECISION not in final or final.get(st.KEY_DECISION) is None


def test_narration_failure_still_returns_the_numbers(api):
    """The numbers are the product. Losing them because the explainer failed
    would be the wrong failure mode."""
    parsed = ParsedCausalQuery(
        query_type="optimal_policy", context={"demand": 13.0}
    )
    app, _ = build(api, parsed, explain_error=RuntimeError("narration down"))
    final = run(app)

    assert final[st.KEY_STATUS]["phase"] == "complete"
    assert "optimal_decision" in final[st.KEY_FINAL]
    assert "narration down" in final[st.KEY_FINAL]


def test_block_content_is_flattened_to_text(api):
    """Gemini 3 narrates in content blocks and attaches a thought signature.
    Stringifying that list would ship the block repr — signature and all — to
    the user as the answer."""
    parsed = ParsedCausalQuery(
        query_type="optimal_policy", context={"demand": 13.0}
    )
    app, _ = build(
        api,
        parsed,
        explain_content=[
            {
                "type": "text",
                "text": "Narrated answer.",
                "extras": {"signature": "EpwDCpkDARFNMg"},
            },
        ],
    )
    final = run(app)

    assert final[st.KEY_FINAL] == "Narrated answer."


# ── Transport ───────────────────────────────────────────────────────────────


def test_updates_stream_yields_node_keyed_deltas(api):
    """The proxy destructures ``{node_name: delta}``. If LangGraph ever stopped
    producing that shape, every progress frame would silently go missing."""
    parsed = ParsedCausalQuery(
        query_type="optimal_policy", context={"demand": 13.0}
    )
    app, _ = build(api, parsed)

    chunks = list(
        app.stream({"user_input": "Demand is 13."}, stream_mode="updates")
    )
    assert chunks, "expected at least one update chunk"

    seen_nodes = []
    for chunk in chunks:
        assert isinstance(chunk, dict) and len(chunk) == 1
        node, delta = next(iter(chunk.items()))
        assert isinstance(delta, dict)
        seen_nodes.append(node)

    assert seen_nodes == [
        "interpret_query",
        "validate_query",
        "optimal_policy",
        "explain_result",
    ]


def test_trace_lines_accumulate_across_nodes(api):
    """``causal_steps`` carries an operator.add reducer. Without it only the
    last node's line would survive and the timeline would be empty."""
    parsed = ParsedCausalQuery(
        query_type="optimal_policy", context={"demand": 13.0}
    )
    app, _ = build(api, parsed)
    final = run(app)

    steps = final[st.KEY_STEPS]
    assert len(steps) >= 4
    tags = [line.split("]")[0] + "]" for line in steps]
    assert "[interpret]" in tags
    assert "[compute]" in tags
    assert "[decision]" in tags


def test_state_is_json_safe(api):
    """LanggraphAgent runs every chunk through dumpd and the proxy forwards it,
    so a stray tensor here surfaces as an unserialisable chunk at runtime."""
    import json

    parsed = ParsedCausalQuery(
        query_type="optimal_policy", context={"demand": 13.0}
    )
    app, _ = build(api, parsed)
    final = run(app)

    causal_state = {k: v for k, v in final.items() if k.startswith(st.STATE_PREFIX)}
    json.dumps(causal_state)  # raises if anything is not JSON-safe


def test_token_usage_sums_across_both_llm_calls(api):
    """The proxy bills from this. Last-write-wins would report only the
    explainer's usage and halve every turn's cost."""
    parsed = ParsedCausalQuery(
        query_type="optimal_policy", context={"demand": 13.0}
    )
    app, _ = build(
        api,
        parsed,
        interpret_usage={
            "input_tokens": 900, "output_tokens": 40, "total_tokens": 940
        },
        explain_usage={
            "input_tokens": 500, "output_tokens": 200, "total_tokens": 700
        },
    )
    final = run(app)

    assert final[st.KEY_USAGE] == {
        "input_tokens": 1400,
        "output_tokens": 240,
        "total_tokens": 1640,
    }


def test_usage_is_recorded_even_when_interpretation_is_unparseable(api):
    """The tokens were spent whether or not the schema came back valid."""
    app, _ = build(
        api,
        ParsedCausalQuery(query_type="optimal_policy"),
        parsing_error="schema mismatch",
        interpret_usage={
            "input_tokens": 900, "output_tokens": 10, "total_tokens": 910
        },
    )
    final = run(app)

    assert final[st.KEY_STATUS]["phase"] == "failed"
    assert final[st.KEY_USAGE]["total_tokens"] == 910


def test_missing_usage_metadata_is_absent_not_zero(api):
    """A model that reports nothing must not look like a free turn."""
    parsed = ParsedCausalQuery(
        query_type="optimal_policy", context={"demand": 13.0}
    )
    app, _ = build(api, parsed)
    final = run(app)

    assert final.get(st.KEY_USAGE) in ({}, None)


def test_graph_is_emitted_before_the_answer(api):
    """The pane should populate while the run is still going, not with the
    final frame."""
    parsed = ParsedCausalQuery(
        query_type="optimal_policy", context={"demand": 13.0}
    )
    app, _ = build(api, parsed)

    chunks = list(
        app.stream({"user_input": "Demand is 13."}, stream_mode="updates")
    )
    interpret_delta = chunks[0]["interpret_query"]
    assert st.KEY_GRAPH in interpret_delta
    assert interpret_delta[st.KEY_GRAPH]["nodes"]
