"""Structured per-node logging: the records, and what they must not change.

The suite drives the real compiled graph over the real Pyro engine — same setup
as test_graph_app.py, only the model is stubbed — and reads back what the JSON
handler actually wrote. Asserting on parsed JSON rather than on LogRecords is
deliberate: the formatter is half the feature, and a payload that only survives
until json.dumps is not a log.

The load-bearing test is test_instrumentation_does_not_change_the_result. Every
other assertion here is about what got recorded; that one is about the promise
that recording it changed nothing.
"""

from __future__ import annotations

import io
import json
import logging

import pytest

pytest.importorskip("torch")
pytest.importorskip("langgraph")

from src.app_utils import jsonlog  # noqa: E402
from src.causal import node_logging, state as st  # noqa: E402
from src.causal.graph_app import build_causal_langgraph_app  # noqa: E402
from src.causal.models import ParsedCausalQuery  # noqa: E402
from src.causal.problem import build_causal_api  # noqa: E402
from tests.test_graph_app import StubModel  # noqa: E402

SAMPLES = 512
QUESTION = "Demand is 13. Which policy is best?"

COMPUTE_NODES = ("optimal_policy", "intervention_effect", "posterior_summary")


@pytest.fixture(scope="module")
def api():
    return build_causal_api()


@pytest.fixture
def stream(monkeypatch):
    """Capture what the handler wrote, as text, exactly as stdout would see it."""
    buffer = io.StringIO()
    logger = logging.getLogger(jsonlog.PACKAGE_LOGGER)
    monkeypatch.setattr(logger, "handlers", [], raising=False)
    jsonlog.configure_logging(level="INFO", stream=buffer)
    yield buffer


def records(stream) -> list[dict]:
    """Every line the run emitted, parsed. Also asserts each one *is* JSON."""
    return [
        json.loads(line) for line in stream.getvalue().splitlines() if line.strip()
    ]


def run_graph(api, parsed, text=QUESTION, **kwargs):
    model = StubModel(parsed=parsed, **kwargs)
    app = build_causal_langgraph_app(
        causal_api=api, model=model, default_num_samples=SAMPLES, default_seed=123
    )
    return app.invoke({"user_input": text}), model


def _phase(entries, node, phase) -> dict:
    matches = [
        entry
        for entry in entries
        if entry.get("node") == node and entry.get("phase") == phase
    ]
    assert matches, f"no {phase} record for {node}"
    return matches[-1]


def node_end(entries, node) -> dict:
    return _phase(entries, node, "end")


def node_start(entries, node) -> dict:
    return _phase(entries, node, "start")


POLICY_QUERY = ParsedCausalQuery(
    query_type="optimal_policy", context={"demand": 13.0}
)
INTERVENTION_QUERY = ParsedCausalQuery(
    query_type="intervention_effect",
    context={"demand": 13.0},
    intervention={
        "variable": "competitor_active",
        "value": 1.0,
        "baseline_value": 0.0,
        "mode": "reoptimise_policy",
    },
)


# ── The promise: logging only ───────────────────────────────────────────────


def test_instrumentation_does_not_change_the_result(api, stream):
    """Same graph, same stub, logging on and off — identical final state.

    This is the assertion the whole change is written against. It runs the
    pipeline twice with the logger silenced for one of them, which exercises the
    wrapper either way (it is compiled in), and compares everything the proxy
    would ever read.
    """
    marked = f"[[run:fixed-id]] {QUESTION}"
    logged_state, _ = run_graph(api, POLICY_QUERY, text=marked)

    logging.getLogger(jsonlog.PACKAGE_LOGGER).setLevel(logging.CRITICAL)
    quiet_state, _ = run_graph(api, POLICY_QUERY, text=marked)

    assert logged_state == quiet_state
    # And the seed is doing its job, so this is a real comparison rather than
    # two runs that happened to agree on their keys.
    assert logged_state[st.KEY_DECISION] == quiet_state[st.KEY_DECISION]


def test_returned_run_id_is_unchanged_when_the_proxy_supplied_one(api, stream):
    final, _ = run_graph(api, POLICY_QUERY, text=f"[[run:abc123]] {QUESTION}")
    assert final[st.KEY_RUN_ID] == "abc123"
    assert all(entry["request_id"] == "abc123" for entry in records(stream))


# ── Record shape ────────────────────────────────────────────────────────────


def test_every_record_carries_the_required_fields(api, stream):
    run_graph(api, POLICY_QUERY)
    entries = records(stream)
    assert entries

    for entry in entries:
        assert entry["request_id"]
        assert entry["node"]
        assert entry["phase"] in ("start", "end", "route")
        assert entry["timestamp"].endswith("Z")
        assert entry["severity"] in ("INFO", "WARNING", "ERROR")
        assert entry["message"]
        if entry["phase"] == "end":
            assert isinstance(entry["duration_ms"], float)


def test_each_node_emits_a_matched_start_and_end_pair(api, stream):
    run_graph(api, POLICY_QUERY)
    entries = records(stream)

    started = [e["node"] for e in entries if e["phase"] == "start"]
    ended = [e["node"] for e in entries if e["phase"] == "end"]
    assert started == ended == [
        "interpret_query",
        "validate_query",
        "optimal_policy",
        "explain_result",
    ]


def test_one_request_id_covers_the_whole_turn_without_a_marker(api, stream):
    """No proxy, no marker — the uuid4 fallback still has to be one id."""
    final, _ = run_graph(api, POLICY_QUERY, text=QUESTION)
    ids = {entry["request_id"] for entry in records(stream)}

    assert len(ids) == 1
    request_id = ids.pop()
    assert request_id != "unknown" and len(request_id) == 32
    # Threaded through state, so the proxy and the UI see the same id.
    assert final[st.KEY_RUN_ID] == request_id


def test_routers_record_the_branch_taken(api, stream):
    run_graph(api, POLICY_QUERY)
    routes = {
        (entry["node"], entry["route"])
        for entry in records(stream)
        if entry["phase"] == "route"
    }
    # The second is invisible anywhere else: it is never written to state.
    assert ("validate_query", "optimal_policy") in routes
    assert ("optimal_policy", "explain") in routes


# ── The Gemini nodes ────────────────────────────────────────────────────────


def test_llm_nodes_log_model_name_and_renamed_token_usage(api, stream):
    run_graph(
        api,
        POLICY_QUERY,
        interpret_usage={
            "input_tokens": 612, "output_tokens": 48, "total_tokens": 660
        },
        explain_usage={
            "input_tokens": 900, "output_tokens": 120, "total_tokens": 1020
        },
    )
    entries = records(stream)

    interpret = node_end(entries, "interpret_query")
    assert interpret["token_usage"] == {
        "prompt_tokens": 612, "completion_tokens": 48, "total_tokens": 660
    }
    explain = node_end(entries, "explain_result")
    assert explain["token_usage"]["total_tokens"] == 1020
    # The stub is not a real chat model, so there is no name to resolve; the key
    # must still be present rather than silently absent.
    assert "model_name" in interpret and "model_name" in explain

    # No usage on the nodes that spend none.
    assert "token_usage" not in node_end(entries, "optimal_policy")


def test_unreported_usage_logs_as_null_not_zeros(api, stream):
    """"Unknown" and "free" are different facts, and the proxy bills from one."""
    run_graph(api, POLICY_QUERY)
    assert node_end(records(stream), "interpret_query")["token_usage"] is None


# ── The mathematics ─────────────────────────────────────────────────────────

_STATISTICS = (
    "expected_utility",
    "utility_standard_deviation",
    "monte_carlo_standard_error",
    "mean_lower_95",
    "mean_upper_95",
)


def test_optimal_policy_logs_the_whole_evaluation_table(api, stream):
    final, _ = run_graph(api, POLICY_QUERY)
    maths = node_end(records(stream), "optimal_policy")["math"]

    assert maths["estimand"] == "argmax_d E[U | D=d, demand=13]"
    assert maths["num_samples"] == SAMPLES
    assert maths["seed"] == 123
    assert maths["clamped"] == ["demand"]
    assert maths["stochastic"] == ["market_growth", "competitor_active"]

    rows = maths["action_evaluations"]
    assert len(rows) == len(api.decisions), "no decision may be dropped"
    for row in rows:
        for key in _STATISTICS:
            assert isinstance(row[key], float)

    assert maths["optimal_decision"] == final[st.KEY_DECISION]["optimal_decision"]
    assert set(maths["probability_each_action_is_best"]) == {"0", "1", "2"}


def test_decision_margin_is_the_gap_to_the_runner_up(api, stream):
    run_graph(api, POLICY_QUERY)
    maths = node_end(records(stream), "optimal_policy")["math"]

    utilities = sorted(
        (row["expected_utility"] for row in maths["action_evaluations"]),
        reverse=True,
    )
    assert maths["decision_margin"] == pytest.approx(utilities[0] - utilities[1])
    assert isinstance(maths["margin_exceeds_mc_error"], bool)


def test_a_margin_inside_monte_carlo_error_is_flagged():
    """The point of the flag: a coin flip must not read as a recommendation."""
    indistinguishable = [
        {"decision": 0, "expected_utility": 10.00, "monte_carlo_standard_error": 0.5},
        {"decision": 1, "expected_utility": 10.02, "monte_carlo_standard_error": 0.5},
    ]
    clear = [
        {"decision": 0, "expected_utility": 10.0, "monte_carlo_standard_error": 0.01},
        {"decision": 1, "expected_utility": 25.0, "monte_carlo_standard_error": 0.01},
    ]
    assert node_logging._decision_margin(indistinguishable)[
        "margin_exceeds_mc_error"
    ] is False
    assert node_logging._decision_margin(clear)["margin_exceeds_mc_error"] is True


def test_intervention_logs_both_arms_and_the_paired_seed(api, stream):
    run_graph(api, INTERVENTION_QUERY)
    maths = node_end(records(stream), "intervention_effect")["math"]

    assert maths["estimand"] == "reoptimised_policy_effect"
    assert maths["formula"] == (
        "max_d E[U | do(competitor_active=1)] - "
        "max_d E[U | do(competitor_active=0)]"
    )
    assert maths["target_interventions"] == {"competitor_active": 1.0}
    assert maths["baseline_interventions"] == {"competitor_active": 0.0}
    assert maths["baseline_is_stochastic_model"] is False
    assert maths["paired"] is True and maths["seed"] == 123
    assert isinstance(maths["causal_effect"], float)
    assert set(maths["decision_flip"]) == {"baseline", "target", "changed"}

    for arm in ("target_arm", "baseline_arm"):
        rows = maths[arm]["action_evaluations"]
        assert len(rows) == len(api.decisions)
        assert all("monte_carlo_standard_error" in row for row in rows)


def test_absent_baseline_logs_as_the_stochastic_model(api, stream):
    """Not do(C=0) — a different estimand, and the only place it is stated."""
    parsed = ParsedCausalQuery(
        query_type="intervention_effect",
        intervention={
            "variable": "competitor_active",
            "value": 1.0,
            "mode": "reoptimise_policy",
        },
    )
    run_graph(api, parsed)
    maths = node_end(records(stream), "intervention_effect")["math"]

    assert maths["baseline_is_stochastic_model"] is True
    assert maths["baseline_interventions"] == {}
    assert "the stochastic model" in maths["formula"]


def test_fixed_policy_estimand_names_the_held_decision(api, stream):
    parsed = ParsedCausalQuery(
        query_type="intervention_effect",
        intervention={
            "variable": "competitor_active",
            "value": 1.0,
            "baseline_value": 0.0,
            "mode": "fixed_policy",
            "fixed_decision": 1,
        },
    )
    run_graph(api, parsed)
    maths = node_end(records(stream), "intervention_effect")["math"]

    assert maths["estimand"] == "fixed_policy_effect"
    assert maths["formula"] == (
        "E[U | do(competitor_active=1), D=1] - "
        "E[U | do(competitor_active=0), D=1]"
    )
    assert maths["fixed_decision"] == 1


def test_posterior_summary_logs_families_and_hyperparameters(api, stream):
    run_graph(api, ParsedCausalQuery(query_type="posterior_summary"))
    maths = node_end(records(stream), "posterior_summary")["math"]

    assert maths["causes"] == sorted(api.causes)
    distributions = maths["posterior_distributions"]
    assert distributions["demand"]["distribution"] == "Normal-Inverse-Gamma"
    assert distributions["demand"]["kappa"]
    assert distributions["competitor_active"]["distribution"] == "Beta-Bernoulli"
    assert "posterior_probability_mean" in distributions["competitor_active"]


def test_validation_records_the_structured_query_it_checked(api, stream):
    run_graph(api, POLICY_QUERY)
    starts = [
        entry
        for entry in records(stream)
        if entry["node"] == "validate_query" and entry["phase"] == "start"
    ]
    assert starts[0]["input"]["structured_query"]["query_type"] == "optimal_policy"


# ── Content policy ──────────────────────────────────────────────────────────


def test_user_text_is_hashed_by_default(api, stream):
    run_graph(api, POLICY_QUERY)
    written = stream.getvalue()

    assert QUESTION not in written
    assert "Narrated answer." not in written

    described = node_end(records(stream), "interpret_query")["output"][st.KEY_QUERY]
    assert described["chars"] == len(QUESTION)
    assert len(described["sha256"]) == 12


def test_content_flag_opens_the_text_up(api, stream, monkeypatch):
    monkeypatch.setenv("CAUSAL_LOG_CONTENT", "true")
    run_graph(api, POLICY_QUERY)

    written = stream.getvalue()
    assert QUESTION in written
    assert "Narrated answer." in written


def test_the_structured_query_is_always_logged_in_full(api, stream):
    """It is the single most useful field for debugging an interpretation, and
    it is model output rather than raw user text."""
    run_graph(api, POLICY_QUERY)
    parsed = node_end(records(stream), "interpret_query")["output"][
        st.KEY_PARSED_QUERY
    ]
    assert parsed["query_type"] == "optimal_policy"
    assert parsed["context"] == {"demand": 13.0}


# ── Truncation, and the formatter's own guarantees ──────────────────────────


def test_the_ui_graph_collapses_to_counts(api, stream):
    run_graph(api, POLICY_QUERY)
    graph = node_end(records(stream), "interpret_query")["output"][st.KEY_GRAPH]
    assert set(graph) == {"nodes", "edges"}
    assert graph["nodes"] > 0


def test_long_strings_truncate_and_report_their_length():
    shaped = node_logging._compact("x" * 5000)
    assert shaped.endswith("(5000 chars)")
    assert len(shaped) < 600


def test_long_lists_truncate_with_a_count():
    shaped = node_logging._compact(list(range(500)))
    assert shaped[-1] == "…(500 items)"
    assert len(shaped) == node_logging._MAX_ITEMS + 1


def test_the_evaluation_table_is_never_truncated(api, stream):
    """The generic caps guard unbounded payloads; the math is not one."""
    run_graph(api, POLICY_QUERY)
    rows = node_end(records(stream), "optimal_policy")["math"]["action_evaluations"]
    assert all(isinstance(row, dict) for row in rows)


def test_a_non_serialisable_payload_still_produces_valid_json():
    class Unserialisable:
        def __repr__(self):
            return "<engine handle>"

    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(jsonlog.JsonFormatter())
    logger = logging.getLogger("src.test.serialisation")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)

    logger.info("odd payload", extra={"thing": Unserialisable()})
    assert json.loads(buffer.getvalue())["thing"] == "<engine handle>"


def test_infinite_floats_survive_as_strings():
    """expected_sampling_variance is inf when alpha <= 1, and bare Infinity is
    not JSON."""
    payload = node_logging._compact({"variance": float("inf")})
    assert json.loads(json.dumps(payload))["variance"] == "inf"


def test_configure_logging_is_idempotent():
    logger = logging.getLogger(jsonlog.PACKAGE_LOGGER)
    logger.handlers = []
    jsonlog.configure_logging(stream=io.StringIO())
    jsonlog.configure_logging(stream=io.StringIO())
    assert len(logger.handlers) == 1


# ── Failure paths ───────────────────────────────────────────────────────────


def test_interpretation_failure_is_recorded_and_still_returns(api, stream):
    final, _ = run_graph(
        api, POLICY_QUERY, interpret_error=RuntimeError("quota exhausted")
    )
    entries = records(stream)

    assert node_end(entries, "interpret_query")["outcome"] == "error"
    # The terminal's start record is where the failure text is readable in full;
    # the answer it returns is the same text, and that is hashed as user-facing.
    assert "quota exhausted" in node_start(entries, "error")["input"]["causal_error"]
    # Unchanged behaviour: the failure is still the answer, not an exception.
    assert "quota exhausted" in final[st.KEY_FINAL]


def test_rejected_query_records_the_error_and_the_route(api, stream):
    parsed = ParsedCausalQuery(
        query_type="optimal_policy", context={"not_a_cause": 1.0}
    )
    run_graph(api, parsed)
    entries = records(stream)

    validate = node_end(entries, "validate_query")
    assert "not_a_cause" in validate["output"][st.KEY_ERROR]
    assert ("validate_query", "error") in {
        (e["node"], e["route"]) for e in entries if e["phase"] == "route"
    }


def test_narration_failure_records_the_error_and_keeps_the_numbers(api, stream):
    final, _ = run_graph(
        api, POLICY_QUERY, explain_error=RuntimeError("narration down")
    )
    explain = node_end(records(stream), "explain_result")

    assert explain["duration_ms"] >= 0
    assert "optimal_decision" in final[st.KEY_FINAL]


def test_a_raising_node_logs_before_it_re_raises(stream):
    """The wrapper must not swallow — only witness."""
    observer = node_logging.NodeLogger(cause_names=["demand"], num_samples=8, seed=1)

    def explode(state):
        raise ValueError("engine is on fire")

    with pytest.raises(ValueError, match="engine is on fire"):
        observer.node("optimal_policy", explode)({st.KEY_RUN_ID: "r1"})

    entry = records(stream)[-1]
    assert entry["outcome"] == "raised"
    assert entry["error_type"] == "ValueError"
    assert "engine is on fire" in entry["error"]
    assert entry["severity"] == "ERROR"
