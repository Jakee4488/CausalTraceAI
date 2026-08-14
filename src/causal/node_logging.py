"""Per-node instrumentation for the LangGraph pipeline.

One ``start`` and one ``end`` record per node, correlated by ``request_id``, so
a single question can be followed from the LLM's interpretation through the
sampling to the narration. The records are emitted by a wrapper applied at
``add_node`` time in :mod:`src.causal.graph_app` — no node body writes a log
line, and no node's return value is touched. What the wrapper receives from the
node is exactly what it hands back to LangGraph.

What ends up in the record
--------------------------
Three things, deliberately weighted:

*Timing and identity* — ``request_id``, ``node``, ``phase``, ``timestamp`` and
``duration_ms``. The cheap part, and what makes the stream filterable.

*The mathematics* — for the three compute routes, the estimand that was
evaluated, the sampling settings, which causes were clamped versus integrated
over, and the complete per-decision statistics table the engine returned:
expected utility, standard deviation, Monte Carlo standard error, 95% bounds and
the probability each action is best. The engine already computes all of it and
today it is discarded once the narration is written; a summary that collapsed
``action_evaluations`` to a row count would throw away the only record of *how*
an answer was reached. These dicts are bounded — three decisions by a handful of
floats — so they are logged whole. Generic truncation is reserved for what is
genuinely unbounded.

*As little user content as possible* — the raw question and the narrated answer
reduce to a character count and a short hash unless ``CAUSAL_LOG_CONTENT`` is
set. That mirrors the NO_CONTENT span policy in :mod:`src.app_utils.telemetry`,
for the same reason: those two strings are already retained in Firestore, and
Cloud Logging is a different retention and access model.

This module imports neither torch nor langgraph — it reads dicts the engine
already produced — so it stays as cheap and hermetic as :mod:`src.causal.runtime`.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import math
import re
import time
import uuid
from typing import Any, Callable, Optional

from src.app_utils.jsonlog import content_logging_enabled
from src.causal import state as st

logger = logging.getLogger(__name__)

# The same marker the proxy injects. Compiled from state.py's pattern rather
# than importing graph_app's helper, which would be a circular import — the
# graph imports this module, not the other way round.
_RUN_ID_RE = re.compile(st.RUN_ID_MARKER_RE)

# Generic caps, for values with no dedicated handler below. Deliberately
# generous: the engine's own payloads are bounded and go through the extractors,
# so anything hitting these limits is unexpected and worth seeing a prefix of.
_MAX_STRING = 500
_MAX_ITEMS = 50
_MAX_DEPTH = 5

# The 95% convention decision.py already reports its bounds at.
_Z_95 = 1.96

# Carried in `math` / `token_usage`, so logging them again under `output` would
# duplicate the largest payloads in every record. causal_run_id is the
# request_id the record is already keyed by.
_OUTPUT_SKIP = frozenset({st.KEY_DECISION, st.KEY_POSTERIORS, st.KEY_USAGE,
                          st.KEY_RUN_ID})

# Reduced to a length and a digest unless content logging is on.
_TEXT_KEYS = frozenset({"user_input", st.KEY_QUERY, st.KEY_FINAL})


# ── Value shaping ────────────────────────────────────────────────────────────


def describe_text(text: Optional[str]) -> Any:
    """A string as either its content or its shape, per the content policy."""
    text = text or ""
    if content_logging_enabled():
        return _truncate_string(text)
    return {
        "chars": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
    }


def _truncate_string(text: str) -> str:
    if len(text) <= _MAX_STRING:
        return text
    return f"{text[:_MAX_STRING]}…({len(text)} chars)"


def _compact(value: Any, depth: int = 0) -> Any:
    """Generic fallback shaping for anything without a dedicated handler."""
    if depth >= _MAX_DEPTH:
        return f"<truncated at depth {_MAX_DEPTH}>"
    if isinstance(value, str):
        return _truncate_string(value)
    if isinstance(value, dict):
        return {
            str(key): _compact(item, depth + 1)
            for key, item in list(value.items())[:_MAX_ITEMS]
        }
    if isinstance(value, (list, tuple)):
        items = [_compact(item, depth + 1) for item in list(value)[:_MAX_ITEMS]]
        if len(value) > _MAX_ITEMS:
            items.append(f"…({len(value)} items)")
        return items
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        # json.dumps would emit bare NaN/Infinity, which is not valid JSON and
        # which Cloud Logging rejects. expected_sampling_variance is inf for a
        # Normal-Inverse-Gamma cause with alpha <= 1, so this is reachable.
        return str(value)
    return value


def _describe_graph(graph: Any) -> Any:
    """The UI graph payload is drawing instructions, not mathematics."""
    if not isinstance(graph, dict):
        return _compact(graph)
    return {
        "nodes": len(graph.get("nodes") or []),
        "edges": len(graph.get("edges") or []),
    }


def _describe_value(key: str, value: Any) -> Any:
    if key in _TEXT_KEYS:
        return describe_text(value if isinstance(value, str) else "")
    if key == st.KEY_GRAPH:
        return _describe_graph(value)
    return _compact(value)


def _fmt(value: Any) -> str:
    """Numbers as they read in a formula, not as Python reprs."""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


# ── The mathematics each compute node performed ──────────────────────────────


def _evaluation_row(row: Any) -> Any:
    """One decision's statistics, kept whole — this is the actual arithmetic."""
    if not isinstance(row, dict):
        return _compact(row)
    return {
        key: _compact(row[key])
        for key in (
            "decision",
            "decision_description",
            "expected_utility",
            "utility_standard_deviation",
            "monte_carlo_standard_error",
            "mean_lower_95",
            "mean_upper_95",
        )
        if key in row
    }


def _decision_margin(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How far clear the winner is, against the noise in the estimate.

    An expected utility is a Monte Carlo mean, so a margin smaller than the
    sampling error means the recommendation is a coin flip dressed as an answer
    — the one thing a decision log should be able to say and currently cannot.

    The two arms are evaluated on the *same* sampled worlds (one call to
    ``sample_causal_worlds`` feeds every decision), so their errors are
    positively correlated and the true standard error of the difference is
    smaller than the independent combination used here. That makes this test
    conservative: it can call a real difference indistinguishable, never the
    reverse.
    """
    utilities = [
        row for row in rows
        if isinstance(row, dict) and isinstance(row.get("expected_utility"), (int, float))
    ]
    if len(utilities) < 2:
        return {}

    ranked = sorted(utilities, key=lambda row: row["expected_utility"], reverse=True)
    best, runner_up = ranked[0], ranked[1]
    margin = float(best["expected_utility"]) - float(runner_up["expected_utility"])
    errors = [
        float(row.get("monte_carlo_standard_error") or 0.0)
        for row in (best, runner_up)
    ]
    combined = math.sqrt(errors[0] ** 2 + errors[1] ** 2)

    return {
        "decision_margin": margin,
        "runner_up_decision": runner_up.get("decision"),
        "margin_exceeds_mc_error": bool(margin > _Z_95 * combined),
    }


def _render_conditioning(
    context: dict[str, Any], interventions: dict[str, Any]
) -> str:
    terms = [f"{name}={_fmt(value)}" for name, value in sorted(context.items())]
    terms += [
        f"do({name}={_fmt(value)})" for name, value in sorted(interventions.items())
    ]
    return ", ".join(terms)


def _optimal_policy_math(
    result: dict[str, Any], cause_names: list[str], seed: int
) -> dict[str, Any]:
    context = result.get("context") or {}
    interventions = result.get("interventions") or {}
    clamped = set(context) | set(interventions)
    conditioning = _render_conditioning(context, interventions)
    rows = result.get("action_evaluations") or []

    return {
        "estimand": (
            f"argmax_d E[U | D=d, {conditioning}]"
            if conditioning
            else "argmax_d E[U | D=d]"
        ),
        "num_samples": result.get("num_samples"),
        "seed": seed,
        # The difference between "the model knows this" and "the model
        # integrated over it" — the same split runtime.summarize_stochastic_line
        # narrates for the user, recorded here in machine-readable form.
        "clamped": sorted(clamped),
        "stochastic": [name for name in cause_names if name not in clamped],
        "action_evaluations": [_evaluation_row(row) for row in rows],
        "probability_each_action_is_best": _compact(
            result.get("probability_each_action_is_best")
        ),
        "optimal_decision": result.get("optimal_decision"),
        "optimal_expected_utility": result.get("optimal_expected_utility"),
        **_decision_margin(rows),
    }


def _arm(result: Any, cause_names: list[str], seed: int) -> Any:
    """One side of an intervention contrast, minus the echoed query.

    ``context``/``interventions`` are dropped: the top level already states both
    arms, and repeating them per arm doubles the record for nothing.
    """
    if not isinstance(result, dict):
        return None
    arm = _optimal_policy_math(result, cause_names, seed)
    for key in ("clamped", "stochastic", "estimand"):
        arm.pop(key, None)
    return arm


def _intervention_math(
    result: dict[str, Any], cause_names: list[str], seed: int
) -> dict[str, Any]:
    estimand = result.get("estimand")
    variable = result.get("intervention_variable")
    value = result.get("intervention_value")
    baseline_value = result.get("baseline_value")

    target = f"do({variable}={_fmt(value)})"
    # An absent baseline is not do(C=0) — it is the ordinary stochastic model,
    # a different estimand entirely, and the distinction is invisible in the
    # result dict unless it is said out loud.
    baseline = (
        f"do({variable}={_fmt(baseline_value)})"
        if baseline_value is not None
        else "the stochastic model"
    )
    fixed = result.get("fixed_decision")
    formula = (
        f"max_d E[U | {target}] - max_d E[U | {baseline}]"
        if estimand == "reoptimised_policy_effect"
        else f"E[U | {target}, D={fixed}] - E[U | {baseline}, D={fixed}]"
    )

    payload: dict[str, Any] = {
        "estimand": estimand,
        "formula": formula,
        "intervention_variable": variable,
        "target_interventions": {variable: value},
        "baseline_interventions": (
            {variable: baseline_value} if baseline_value is not None else {}
        ),
        "baseline_is_stochastic_model": baseline_value is None,
        # Both arms are drawn under the same seed, so the contrast is paired and
        # the shared sampling noise cancels rather than adding.
        "paired": True,
        "seed": seed,
        "target_expected_utility": result.get("target_expected_utility"),
        "baseline_expected_utility": result.get("baseline_expected_utility"),
        "causal_effect": result.get("causal_effect"),
    }

    if estimand == "reoptimised_policy_effect":
        baseline_decision = result.get("baseline_optimal_decision")
        target_decision = result.get("target_optimal_decision")
        payload["decision_flip"] = {
            "baseline": baseline_decision,
            "target": target_decision,
            "changed": baseline_decision != target_decision,
        }
        payload["target_arm"] = _arm(result.get("target_result"), cause_names, seed)
        payload["baseline_arm"] = _arm(
            result.get("baseline_result"), cause_names, seed
        )
    else:
        payload["fixed_decision"] = fixed
        payload["target_arm"] = {
            "action_evaluations": [_evaluation_row(result.get("target_evaluation"))]
        }
        payload["baseline_arm"] = {
            "action_evaluations": [_evaluation_row(result.get("baseline_evaluation"))]
        }

    return payload


def _posterior_math(posteriors: Any) -> dict[str, Any]:
    """Distribution family, hyperparameters and moments — the whole node."""
    if not isinstance(posteriors, dict):
        return {"posterior_distributions": _compact(posteriors)}
    return {
        "estimand": "posterior hyperparameters and moments per cause",
        "causes": sorted(posteriors),
        "posterior_distributions": {
            name: _compact(summary) for name, summary in posteriors.items()
        },
    }


def describe_math(
    result: dict[str, Any], cause_names: list[str], seed: int
) -> Optional[dict[str, Any]]:
    """The mathematics behind a compute node's return value, or None."""
    decision = result.get(st.KEY_DECISION)
    if not isinstance(decision, dict):
        return None

    query_type = decision.get("query_type")
    try:
        if query_type == "optimal_policy":
            return _optimal_policy_math(decision, cause_names, seed)
        if query_type == "intervention_effect":
            return _intervention_math(decision, cause_names, seed)
        if query_type == "posterior_summary":
            return _posterior_math(
                result.get(st.KEY_POSTERIORS)
                or decision.get("posterior_distributions")
            )
    except Exception as exc:  # pragma: no cover - defensive
        # A malformed result is worth a log line saying so, never an exception
        # that fails a node which had already produced its answer.
        return {"extraction_error": f"{type(exc).__name__}: {exc}"}
    return None


# ── Node input and output ────────────────────────────────────────────────────


def _describe_input(
    node: str, state: dict[str, Any], num_samples: int, seed: int
) -> dict[str, Any]:
    raw_input = state.get("user_input") or ""
    if node == "interpret_query":
        return {
            "user_input": describe_text(_RUN_ID_RE.sub("", raw_input).strip()),
            "has_run_marker": bool(_RUN_ID_RE.search(raw_input)),
        }

    parsed = state.get(st.KEY_PARSED_QUERY) or {}
    if node == "validate_query":
        return {"structured_query": _compact(parsed)}

    if node in ("optimal_policy", "intervention_effect", "posterior_summary"):
        received = {
            "query_type": parsed.get("query_type"),
            "context": _compact(parsed.get("context") or {}),
            "num_samples": num_samples,
            "seed": seed,
        }
        if parsed.get("intervention"):
            received["intervention"] = _compact(parsed["intervention"])
        return received

    if node == "explain_result":
        decision = state.get(st.KEY_DECISION)
        return {
            "query_type": (decision or {}).get("query_type")
            if isinstance(decision, dict)
            else None,
            "has_numerical_result": decision is not None,
            "has_error": bool(state.get(st.KEY_ERROR)),
        }

    # clarification / error terminals.
    return {
        "query_type": parsed.get("query_type"),
        "causal_error": _truncate_string(state.get(st.KEY_ERROR) or "") or None,
    }


def _describe_output(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"returned": _compact(result)}
    return {
        key: _describe_value(key, value)
        for key, value in result.items()
        if key not in _OUTPUT_SKIP
    }


def _token_usage(result: Any) -> Optional[dict[str, int]]:
    """Usage in prompt/completion vocabulary, or None when unreported.

    ``_usage_of`` in graph_app returns ``{}`` for a model that reports nothing,
    to keep "unknown" distinct from "free". That distinction survives here as
    ``null`` — never a fabricated set of zeros.
    """
    usage = (result or {}).get(st.KEY_USAGE) if isinstance(result, dict) else None
    if not isinstance(usage, dict) or not usage:
        return None
    renamed = {
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }
    return {key: value for key, value in renamed.items() if value is not None}


# ── The wrapper ──────────────────────────────────────────────────────────────


class NodeLogger:
    """Wraps graph nodes and routers in structured start/end logging.

    Built once per compiled graph, closing over the settings that are fixed for
    the process — the model name, the cause names, the sample count and the seed
    — so a node never has to report them itself.
    """

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        cause_names: Optional[list[str]] = None,
        num_samples: int = 0,
        seed: int = 0,
    ) -> None:
        self.model_name = model_name
        self.cause_names = list(cause_names or [])
        self.num_samples = num_samples
        self.seed = seed

    def _request_id(self, state: dict[str, Any], entry: bool) -> str:
        existing = state.get(st.KEY_RUN_ID)
        if existing:
            return str(existing)
        if not entry:
            # Only reachable if a node ran before interpret_query, which the
            # graph's edges make impossible — flagged rather than silently
            # given a fresh id that would split the turn across two ids.
            return "unknown"
        marker = _RUN_ID_RE.search(state.get("user_input") or "")
        return marker.group(1) if marker else uuid.uuid4().hex

    def node(
        self,
        name: str,
        fn: Callable[[Any], dict],
        *,
        entry: bool = False,
        llm: bool = False,
    ) -> Callable[[Any], dict]:
        """Return ``fn`` wrapped in a start/end pair. Same signature, same
        return value, same exceptions."""

        @functools.wraps(fn)
        def wrapped(state: Any) -> dict:
            request_id = self._request_id(state, entry)
            if entry and not state.get(st.KEY_RUN_ID):
                # The entry node reads the id back out of state, so resolving it
                # here — before the call — is what lets the `start` record carry
                # the same id as everything downstream.
                state = {**state, st.KEY_RUN_ID: request_id}

            base = {"request_id": request_id, "node": name}
            started = time.perf_counter()
            logger.info(
                "node start",
                extra={
                    **base,
                    "phase": "start",
                    "input": _describe_input(
                        name, state, self.num_samples, self.seed
                    ),
                    **({"model_name": self.model_name} if llm else {}),
                },
            )

            try:
                result = fn(state)
            except Exception as exc:
                logger.error(
                    "node failed",
                    extra={
                        **base,
                        "phase": "end",
                        "outcome": "raised",
                        "duration_ms": _elapsed_ms(started),
                        "error_type": type(exc).__name__,
                        "error": _truncate_string(str(exc)),
                    },
                )
                raise

            record: dict[str, Any] = {
                **base,
                "phase": "end",
                "outcome": "error" if (result or {}).get(st.KEY_ERROR) else "ok",
                "duration_ms": _elapsed_ms(started),
                "output": _describe_output(result),
            }
            if llm:
                record["model_name"] = self.model_name
                record["token_usage"] = _token_usage(result)
            maths = describe_math(result or {}, self.cause_names, self.seed)
            if maths is not None:
                record["math"] = maths

            logger.info("node end", extra=record)
            return result

        return wrapped

    def router(
        self, name: str, fn: Callable[[Any], str]
    ) -> Callable[[Any], str]:
        """Log the branch a conditional edge chose.

        ``route_after_calculation`` returns "explain" or "error" and writes
        neither to state, so without this the branch taken leaves no trace at
        all.
        """

        @functools.wraps(fn)
        def wrapped(state: Any) -> str:
            route = fn(state)
            logger.info(
                "route",
                extra={
                    "request_id": self._request_id(state, entry=False),
                    "node": name,
                    "phase": "route",
                    "route": route,
                },
            )
            return route

        return wrapped


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
