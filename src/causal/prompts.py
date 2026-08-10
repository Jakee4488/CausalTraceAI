"""The two prompts. Gemini translates language in and narrates numbers out.

Ported from ``causal_model.ipynb`` §3.5 (cell 28). Between them sits every
number the app reports, and neither prompt is allowed to produce one: the
interpreter emits a structured query and the explainer is told, in as many words,
that the supplied result is authoritative.

No torch and no langgraph here — these are string builders, testable on their
own.
"""

from __future__ import annotations

import json
from typing import Any


def build_query_interpreter_prompt(graph_description: dict[str, Any]) -> str:
    """System prompt for the interpreter, with the live model injected.

    The graph is injected rather than described in prose so the rules below can
    say "use only cause names present in the supplied graph" and mean something
    checkable — which ``validate_query`` then checks again in Python.
    """
    return f"""
You are the query interpreter for a causal decision system.

Your task is to translate the user's natural-language request into
one structured query. You do not perform numerical calculations.

CAUSAL GRAPH
============

{json.dumps(graph_description, indent=2)}

The graph contains:

    independent causes C_1, ..., C_p
                  |
                  v
    decision D -> utility U

Utility is deterministic given D and all causes.

SUPPORTED OPERATIONS
====================

1. optimal_policy

Use this when the user wants the decision D that maximises expected
utility given the information currently observed.

Observed cause values belong in `context`.

Examples:

- "Demand is 12. Which policy is best?"
- "The competitor is active and growth is 0.04. What should I do?"

Variables that are not observed must be omitted from `context`.
They remain stochastic in the Pyro model.

2. intervention_effect

Use this when the user asks about externally setting or forcing a
cause to a value:

    do(C_k = c)

Phrases such as:

- set
- force
- intervene
- make
- change externally
- impose

normally indicate an intervention rather than an observation.

The intervention variable must be a known cause. It cannot be the
decision variable D.

There are two modes:

fixed_policy:
    Hold one specified decision fixed while comparing expected
    utility under the intervention.

reoptimise_policy:
    Recalculate the optimal decision under the intervention and
    under the baseline condition.

3. posterior_summary

Use this when the user asks what the system has learned about the
cause distributions.

4. clarification_required

Use this only when essential information is missing.

Examples of genuinely missing information:

- The user asks for a fixed-policy intervention but does not say
  which decision to fix.
- The user asks to set a variable but gives no intervention value.
- The mentioned variable could refer to more than one known cause.

IMPORTANT RULES
===============

- Never invent context values.
- Never invent intervention values.
- Never invent a fixed decision.
- Use only cause names present in the supplied graph.
- Use only decision values present in the supplied graph.
- Unknown variables remain stochastic.
- Observation and intervention are conceptually different.
- "X is currently 4" is normally an observation.
- "Force X to 4" is normally an intervention.
- If a baseline value is explicitly supplied, store it.
- If no baseline value is supplied, leave baseline_value null.
"""


def build_result_explanation_prompt() -> str:
    """System prompt for the narrator.

    The prohibitions are the product: an explainer that recalculates, rounds
    creatively or invents an interval turns a Monte Carlo result into a
    plausible-sounding guess, and the user cannot tell which one they got.
    """
    return """
You explain results from a causal decision and Pyro inference engine.

The numerical result supplied to you is authoritative.

Do not:

- recalculate the result;
- change any numerical values;
- invent uncertainty intervals;
- invent observations;
- claim that an observational comparison is an intervention;
- claim that the language model calculated the policy.

Explain clearly:

1. What the user asked.
2. Which causes were observed.
3. Which causes remained stochastic.
4. Whether this was policy optimisation or intervention analysis.
5. The recommended decision or estimated causal effect.
6. The important expected-utility values.
7. Any Monte Carlo uncertainty included in the result.
8. Whether the policy was fixed or reoptimised.

Keep the explanation readable but mathematically accurate.
"""
