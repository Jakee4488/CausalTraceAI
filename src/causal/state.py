"""The graph's state, and the key names the proxy and UI read.

Everything the pipeline produces lives under a ``causal_`` prefix, because that
prefix is the whole transport contract: the proxy collects every ``causal_*``
key out of each LangGraph chunk and forwards the UI-facing ones. ``user_input``
is deliberately unprefixed — it is the *input* to a run, not a result, and
prefixing it would echo the question back as if the pipeline had produced it.

Reducers
--------
LangGraph merges a node's returned partial state into the running state one
channel at a time. The default is last-write-wins, which is right for every key
here except the trace: the router, the interpreter and the compute node each
append a line, and last-write-wins would mean only the final node's line
survived. ``causal_steps`` therefore carries an ``operator.add`` reducer so
returning ``{"causal_steps": ["[compute] ..."]}`` extends rather than replaces.

Everything stored must stay JSON-safe. ``LanggraphAgent.stream_query`` runs each
chunk through ``dumpd`` and the proxy forwards the result verbatim, so a stray
tensor or pydantic model here surfaces as an unserialisable chunk at runtime
rather than a type error at import.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


def add_usage(
    left: Optional[dict[str, int]], right: Optional[dict[str, int]]
) -> dict[str, int]:
    """Reducer that sums token counts across the turn's two LLM calls.

    Last-write-wins would report only the explainer's usage, halving every bill
    and letting the quota gate pass turns it should have stopped.
    """
    left = left or {}
    right = right or {}
    return {
        key: int(left.get(key, 0)) + int(right.get(key, 0))
        for key in set(left) | set(right)
    }


# ── Session-state keys (the proxy duplicates the prefix, not the names) ──────
KEY_RUN_ID = "causal_run_id"                # per-turn correlation id
KEY_QUERY = "causal_query"                  # marker-stripped user question
KEY_PARSED_QUERY = "causal_parsed_query"    # ParsedCausalQuery dump
KEY_ROUTE = "causal_route"                  # which compute node was chosen
KEY_DECISION = "causal_decision"            # the engine's numerical result
KEY_POSTERIORS = "causal_posteriors"        # posterior_summaries() dump
KEY_GRAPH = "causal_graph"                  # UI graph payload (graph_view.py)
KEY_STATUS = "causal_status"                # CausalStatus dump
KEY_STEPS = "causal_steps"                  # list[str] human-readable trace
KEY_FINAL = "causal_final_answer"           # the narrated answer
KEY_ERROR = "causal_error"                  # failure text, surfaced as answer
KEY_USAGE = "causal_usage"                  # summed token counts (see add_usage)

STATE_PREFIX = "causal_"

# What the proxy forwards to the browser. Kept here so agent and proxy have one
# place to disagree loudly rather than two places to drift quietly.
UI_KEYS = (
    KEY_STEPS,
    KEY_GRAPH,
    KEY_STATUS,
    KEY_DECISION,
    KEY_POSTERIORS,
    KEY_FINAL,
    KEY_RUN_ID,
)

# ── Defaults (env-overridable in agent.py) ───────────────────────────────────
# 20_000 worlds and seed 123 are the notebook's §4.7 settings; the parity tests
# pin results at exactly these values, so changing them changes every number the
# app reports.
DEFAULT_NUM_SAMPLES = 20_000
DEFAULT_SEED = 123

# Correlation id injected by the proxy on the same channel as any other marker.
RUN_ID_MARKER_RE = r"\[\[run:([A-Za-z0-9_-]{1,64})\]\]"


class CausalApplicationState(TypedDict, total=False):
    """The LangGraph channel set.

    ``total=False`` throughout: every node returns only the keys it actually
    wrote, which is what makes ``stream_mode="updates"`` chunks small enough to
    be useful as progress frames.
    """

    # Input.
    user_input: str

    # Correlation and the cleaned question.
    causal_run_id: Optional[str]
    causal_query: str

    # Interpretation and routing.
    causal_parsed_query: dict[str, Any]
    causal_route: str

    # Results.
    causal_decision: Optional[dict[str, Any]]
    causal_posteriors: Optional[dict[str, Any]]
    causal_graph: Optional[dict[str, Any]]

    # Presentation.
    causal_status: dict[str, Any]
    causal_steps: Annotated[list[str], operator.add]
    causal_final_answer: str
    causal_error: str

    # Billing. Summed rather than replaced — see add_usage.
    causal_usage: Annotated[dict[str, int], add_usage]
