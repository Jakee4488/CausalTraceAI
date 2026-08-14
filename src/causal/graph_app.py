"""The LangGraph state machine — cell 30, wired to the UI transport.

    interpret_query (LLM 1) -> validate_query (0 LLM) -> one of
        optimal_policy | intervention_effect | posterior_summary   (0 LLM)
      -> explain_result (LLM 2)

``clarification`` and ``error`` terminate without an LLM call, so a rejected or
ambiguous query costs one Gemini call rather than two.

What this adds over the notebook
--------------------------------
Every node also writes the ``causal_*`` keys the proxy streams: a trace line, a
status phase, and — once a query has been interpreted — the model graph with its
per-cause clamped/stochastic status. That is the same "one write, two purposes"
idea TracerLensAi used, expressed in LangGraph's channel model instead of ADK's
``state_delta``: the partial state a node returns *is* the progress frame.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from src.causal import runtime, state as st
from src.causal.causes import Intervention
from src.causal.graph_view import build_ui_graph, interventions_from_parsed
from src.causal.models import CausalStatus, ParsedCausalQuery
from src.causal.node_logging import NodeLogger
from src.causal.prompts import (
    build_query_interpreter_prompt,
    build_result_explanation_prompt,
)

logger = logging.getLogger(__name__)

_RUN_ID_RE = re.compile(st.RUN_ID_MARKER_RE)


def extract_run_id(text: str) -> Optional[str]:
    """Pull the per-turn correlation id the proxy injected, if present."""
    match = _RUN_ID_RE.search(text or "")
    return match.group(1) if match else None


def strip_markers(text: str) -> str:
    return _RUN_ID_RE.sub("", text or "").strip()


def _status(phase: str, query_type: str = "", note: str = "") -> dict:
    return CausalStatus(
        phase=phase, query_type=query_type, note=note
    ).model_dump(mode="json")


def _usage_of(message: Any) -> dict[str, int]:
    """Token counts off an AIMessage, or empty when the model does not report.

    The proxy bills from this. Missing metadata must therefore read as "unknown"
    (empty), never as a fabricated zero that looks like a free turn.
    """
    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, dict):
        return {}
    return {
        key: int(usage[key])
        for key in ("input_tokens", "output_tokens", "total_tokens")
        if isinstance(usage.get(key), int)
    }


def build_causal_langgraph_app(
    causal_api,
    model: Any,
    default_num_samples: int = st.DEFAULT_NUM_SAMPLES,
    default_seed: int = st.DEFAULT_SEED,
):
    """Construct and compile the LangGraph application.

    A closure over the engine, the model and the sampling settings: the sample
    count and seed are baked in at build time rather than passed per query, so
    two runs of the same question are always directly comparable.
    """
    # include_raw=True so the underlying AIMessage survives: it carries the
    # token usage the proxy bills from, which a bare parsed model discards. It
    # also surfaces a schema-parse failure as data rather than an exception.
    structured_query_model = model.with_structured_output(
        ParsedCausalQuery, include_raw=True
    )
    graph_description = causal_api.describe_graph()

    # Structured logging, wired in at add_node time below. Everything it needs
    # that is fixed for the process is resolved once, here, so no node has to
    # report its own settings: the model name, the cause set the clamped /
    # stochastic split is taken against, and the sampling settings.
    observer = NodeLogger(
        model_name=(
            getattr(model, "model_name", None) or getattr(model, "model", None)
        ),
        cause_names=list(causal_api.causes),
        num_samples=default_num_samples,
        seed=default_seed,
    )

    # ── Node 1: LLM interprets natural language ─────────────────────────────

    def interpret_query(state: st.CausalApplicationState) -> dict:
        raw_input = (state.get("user_input") or "").strip()
        # The logging wrapper resolves the correlation id before this node runs
        # — from the proxy's marker, or a fresh uuid when there is none — so its
        # `start` record carries the same id as every record downstream. Read it
        # back rather than re-deriving, and fall back to the marker so a direct
        # (unwrapped) call still behaves exactly as it always did.
        run_id = state.get(st.KEY_RUN_ID) or extract_run_id(raw_input)
        user_input = strip_markers(raw_input)

        base: dict[str, Any] = {
            st.KEY_RUN_ID: run_id,
            st.KEY_QUERY: user_input,
        }

        if not user_input:
            return {
                **base,
                st.KEY_ERROR: "The user input is empty.",
                st.KEY_ROUTE: "error",
                st.KEY_STATUS: _status("failed", note="empty input"),
                st.KEY_STEPS: [runtime.summarize_error_line("empty input")],
            }

        usage: dict[str, int] = {}
        try:
            envelope = structured_query_model.invoke(
                [
                    SystemMessage(
                        content=build_query_interpreter_prompt(graph_description)
                    ),
                    HumanMessage(content=user_input),
                ]
            )
            # include_raw=True yields {"raw", "parsed", "parsing_error"}. The
            # tokens were spent whether or not the schema came back valid, so
            # read usage before deciding the call succeeded.
            usage = _usage_of((envelope or {}).get("raw"))
            parsed = (envelope or {}).get("parsed")
            if parsed is None:
                raise ValueError(
                    (envelope or {}).get("parsing_error")
                    or "the model returned no parseable query"
                )
        except Exception as exc:
            logger.warning("Query interpretation failed: %s", exc)
            message = (
                "The LLM could not convert the request into a structured "
                f"causal query: {exc}"
            )
            return {
                **base,
                st.KEY_ERROR: message,
                st.KEY_ROUTE: "error",
                st.KEY_STATUS: _status("failed", note="interpretation failed"),
                st.KEY_STEPS: [runtime.summarize_error_line(message)],
                st.KEY_USAGE: usage,
            }

        parsed_dump = parsed.model_dump(mode="json")
        return {
            **base,
            st.KEY_PARSED_QUERY: parsed_dump,
            st.KEY_STATUS: _status("validating", parsed.query_type),
            st.KEY_STEPS: [runtime.summarize_interpretation_line(parsed_dump)],
            st.KEY_USAGE: usage,
            # The graph is drawable as soon as the query is understood — the UI
            # gets it before the sampling starts rather than with the answer.
            st.KEY_GRAPH: build_ui_graph(
                graph_description,
                context=parsed.context,
                interventions=interventions_from_parsed(parsed_dump),
            ),
        }

    # ── Node 2: deterministic validation ────────────────────────────────────

    def validate_query(state: st.CausalApplicationState) -> dict:
        """Re-check every LLM-supplied name in Python before any sampling.

        The prompt already says to use only names from the supplied graph. This
        is what makes that a guarantee rather than an instruction.
        """
        if state.get(st.KEY_ERROR):
            return {st.KEY_ROUTE: "error"}

        parsed = state.get(st.KEY_PARSED_QUERY) or {}
        query_type = parsed.get("query_type")

        if query_type == "clarification_required":
            return {
                st.KEY_ROUTE: "clarification",
                st.KEY_STATUS: _status("clarification", query_type),
            }

        known_causes = set(causal_api.causes)

        def _reject(message: str) -> dict:
            return {
                st.KEY_ERROR: message,
                st.KEY_ROUTE: "error",
                st.KEY_STATUS: _status("failed", query_type, message[:120]),
                st.KEY_STEPS: [runtime.summarize_error_line(message)],
            }

        unknown_context = set(parsed.get("context") or {}) - known_causes
        if unknown_context:
            return _reject(
                "The interpreted query contains unknown context variables: "
                f"{sorted(unknown_context)}"
            )

        if query_type == "intervention_effect":
            intervention = parsed.get("intervention")
            if not intervention:
                return _reject("The intervention is missing.")
            if intervention.get("variable") not in known_causes:
                return _reject(
                    "The query attempts to intervene on unknown variable "
                    f"'{intervention.get('variable')}'."
                )
            if (
                intervention.get("mode") == "fixed_policy"
                and intervention.get("fixed_decision")
                not in causal_api.decisions
            ):
                return _reject(
                    "The fixed decision is not one of the allowed decisions: "
                    f"{causal_api.decisions}."
                )

        return {
            st.KEY_ROUTE: query_type,
            st.KEY_STATUS: _status("computing", query_type),
        }

    # ── Node 3A: optimal policy ─────────────────────────────────────────────

    def run_optimal_policy(state: st.CausalApplicationState) -> dict:
        parsed = state.get(st.KEY_PARSED_QUERY) or {}
        context = parsed.get("context") or {}
        try:
            # Only `context` is passed, never `interventions`: routing means an
            # optimal_policy query never carries one, and silently applying it
            # here would answer a different question than the one shown.
            result = causal_api.optimal_policy(
                context=context,
                num_samples=default_num_samples,
                seed=default_seed,
            )
        except Exception as exc:
            return _compute_failure(f"Optimal-policy calculation failed: {exc}")

        return {
            st.KEY_DECISION: result,
            st.KEY_STATUS: _status("explaining", "optimal_policy"),
            st.KEY_STEPS: [
                runtime.summarize_samples_line(default_num_samples, default_seed),
                runtime.summarize_stochastic_line(
                    list(causal_api.causes), context, {}
                ),
                runtime.summarize_policy_line(result),
            ],
            st.KEY_GRAPH: build_ui_graph(
                graph_description,
                context=context,
                chosen_decision=result.get("optimal_decision"),
            ),
        }

    # ── Node 3B: intervention effect ────────────────────────────────────────

    def run_intervention_effect(state: st.CausalApplicationState) -> dict:
        parsed = state.get(st.KEY_PARSED_QUERY) or {}
        spec = parsed.get("intervention")
        if not spec:
            return _compute_failure("No intervention was supplied.")

        context = parsed.get("context") or {}
        intervention = Intervention(
            variable=spec["variable"],
            value=spec["value"],
            baseline_value=spec.get("baseline_value"),
            mode=spec.get("mode", "reoptimise_policy"),
            fixed_decision=spec.get("fixed_decision"),
        )

        try:
            result = causal_api.intervention_effect(
                intervention=intervention,
                context=context,
                num_samples=default_num_samples,
                seed=default_seed,
            )
        except Exception as exc:
            # Covers the observe-vs-intervene contradiction the engine raises on.
            return _compute_failure(f"Intervention calculation failed: {exc}")

        chosen = result.get("target_optimal_decision")
        if chosen is None:
            chosen = result.get("fixed_decision")

        return {
            st.KEY_DECISION: result,
            st.KEY_STATUS: _status("explaining", "intervention_effect"),
            st.KEY_STEPS: [
                runtime.summarize_samples_line(default_num_samples, default_seed),
                runtime.summarize_stochastic_line(
                    list(causal_api.causes),
                    context,
                    {intervention.variable: intervention.value},
                ),
                runtime.summarize_intervention_line(result),
            ],
            st.KEY_GRAPH: build_ui_graph(
                graph_description,
                context=context,
                interventions={intervention.variable: intervention.value},
                chosen_decision=chosen,
            ),
        }

    # ── Node 3C: posterior summary ──────────────────────────────────────────

    def run_posterior_summary(state: st.CausalApplicationState) -> dict:
        posteriors = causal_api.posterior_summaries()
        return {
            st.KEY_POSTERIORS: posteriors,
            st.KEY_DECISION: {
                "query_type": "posterior_summary",
                "posterior_distributions": posteriors,
            },
            st.KEY_STATUS: _status("explaining", "posterior_summary"),
            st.KEY_STEPS: [runtime.summarize_posterior_line(posteriors)],
        }

    # ── Node 3D: clarification ──────────────────────────────────────────────

    def return_clarification(state: st.CausalApplicationState) -> dict:
        parsed = state.get(st.KEY_PARSED_QUERY) or {}
        question = parsed.get("clarification_question") or (
            "Please provide the missing information needed to define the "
            "causal query."
        )
        return {
            st.KEY_FINAL: question,
            st.KEY_STATUS: _status("clarification", "clarification_required"),
            st.KEY_STEPS: ["[clarify] essential information was missing"],
        }

    # ── Node 4: LLM explains the Pyro result ────────────────────────────────

    def explain_result(state: st.CausalApplicationState) -> dict:
        if state.get(st.KEY_ERROR):
            return {
                st.KEY_FINAL: state[st.KEY_ERROR],
                st.KEY_STATUS: _status("failed"),
            }

        numerical_result = state.get(st.KEY_DECISION)
        payload = {
            "original_user_input": state.get(st.KEY_QUERY, ""),
            "structured_query": state.get(st.KEY_PARSED_QUERY, {}),
            "authoritative_numerical_result": numerical_result,
        }

        try:
            response = model.invoke(
                [
                    SystemMessage(content=build_result_explanation_prompt()),
                    HumanMessage(
                        content=json.dumps(payload, indent=2, default=str)
                    ),
                ]
            )
            # Gemini 3 answers as a list of content blocks (the text plus a
            # thought signature), so ``.content`` is no longer a bare string
            # and ``str()`` of it would ship the block repr to the user.
            # ``.text`` joins the text blocks and passes a plain string through.
            answer = response.text
            return {
                st.KEY_FINAL: answer,
                st.KEY_STATUS: _status("complete"),
                st.KEY_USAGE: _usage_of(response),
            }
        except Exception as exc:
            # The numbers are the product and they already exist. Losing them
            # because the narration failed would be the wrong failure mode, so
            # hand back the raw result rather than an apology.
            logger.warning("Explanation failed; returning raw result: %s", exc)
            return {
                st.KEY_FINAL: (
                    "The numerical calculation succeeded, but the LLM "
                    "explanation failed.\n\n"
                    + json.dumps(numerical_result, indent=2, default=str)
                    + f"\n\nExplanation error: {exc}"
                ),
                st.KEY_STATUS: _status("complete", note="narration failed"),
                st.KEY_STEPS: [
                    runtime.summarize_error_line(f"narration failed: {exc}")
                ],
            }

    # ── Error node ──────────────────────────────────────────────────────────

    def return_error(state: st.CausalApplicationState) -> dict:
        message = state.get(st.KEY_ERROR) or "An unknown application error occurred."
        return {
            st.KEY_FINAL: message,
            st.KEY_STATUS: _status("failed", note=message[:120]),
        }

    def _compute_failure(message: str) -> dict:
        return {
            st.KEY_ERROR: message,
            st.KEY_STATUS: _status("failed", note=message[:120]),
            st.KEY_STEPS: [runtime.summarize_error_line(message)],
        }

    # ── Routing ─────────────────────────────────────────────────────────────

    def route_after_validation(state: st.CausalApplicationState) -> str:
        return state.get(st.KEY_ROUTE, "error")

    def route_after_calculation(state: st.CausalApplicationState) -> str:
        return "error" if state.get(st.KEY_ERROR) else "explain"

    # ── Assemble ────────────────────────────────────────────────────────────

    builder = StateGraph(st.CausalApplicationState)

    # Every node is registered through the logger. The wrapper returns what the
    # node returned, unchanged; the node names stay byte-identical because the
    # proxy's STAGE_BY_NODE mapping keys off them.
    builder.add_node(
        "interpret_query",
        observer.node("interpret_query", interpret_query, entry=True, llm=True),
    )
    builder.add_node("validate_query", observer.node("validate_query", validate_query))
    builder.add_node(
        "optimal_policy", observer.node("optimal_policy", run_optimal_policy)
    )
    builder.add_node(
        "intervention_effect",
        observer.node("intervention_effect", run_intervention_effect),
    )
    builder.add_node(
        "posterior_summary",
        observer.node("posterior_summary", run_posterior_summary),
    )
    builder.add_node(
        "clarification", observer.node("clarification", return_clarification)
    )
    builder.add_node(
        "explain_result", observer.node("explain_result", explain_result, llm=True)
    )
    builder.add_node("error", observer.node("error", return_error))

    builder.add_edge(START, "interpret_query")
    builder.add_edge("interpret_query", "validate_query")

    builder.add_conditional_edges(
        "validate_query",
        observer.router("validate_query", route_after_validation),
        {
            "optimal_policy": "optimal_policy",
            "intervention_effect": "intervention_effect",
            "posterior_summary": "posterior_summary",
            "clarification": "clarification",
            "error": "error",
        },
    )

    for compute_node in (
        "optimal_policy",
        "intervention_effect",
        "posterior_summary",
    ):
        builder.add_conditional_edges(
            compute_node,
            observer.router(compute_node, route_after_calculation),
            {"explain": "explain_result", "error": "error"},
        )

    builder.add_edge("explain_result", END)
    builder.add_edge("clarification", END)
    builder.add_edge("error", END)

    # No checkpointer: each turn is self-contained, and the proxy already owns
    # conversation history. Adding one would make the same question answered
    # twice depend on which session it arrived in.
    return builder.compile()
