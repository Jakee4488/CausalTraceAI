"""Trace-line summarizers — the human-readable record of a turn.

Each returns one ``[tag] text`` line. The proxy streams them as they appear and
the UI prints them under the stage that produced them, so they are the only
narration of a run the user sees while it is still running.

Pure string formatting over the engine's result dicts: no torch, no langgraph.
"""

from __future__ import annotations

from typing import Any

# Results are reported to four significant figures. Not cosmetic: the engine's
# Monte Carlo standard error is typically the third or fourth digit, so printing
# more would show noise as signal.
_SIG = ".4g"


def summarize_interpretation_line(parsed: dict[str, Any]) -> str:
    query_type = parsed.get("query_type", "unknown")
    context = parsed.get("context") or {}
    if context:
        observed = ", ".join(
            f"{name}={value:{_SIG}}" for name, value in context.items()
        )
        return f"[interpret] {query_type}; observed {observed}"
    return f"[interpret] {query_type}; nothing observed — all causes stochastic"


def summarize_stochastic_line(
    all_causes: list[str], context: dict[str, float], interventions: dict
) -> str:
    """Which causes the answer is still uncertain about.

    Worth its own line: it is the difference between "the model knows this" and
    "the model integrated over it", and nothing else in the trace says so.
    """
    clamped = set(context) | set(interventions)
    stochastic = [name for name in all_causes if name not in clamped]
    if not stochastic:
        return "[compute] every cause was clamped — no uncertainty integrated"
    return f"[compute] integrating over {len(stochastic)}: {', '.join(stochastic)}"


def summarize_policy_line(result: dict[str, Any]) -> str:
    utility = result.get("optimal_expected_utility", 0.0)
    description = result.get("optimal_decision_description") or result.get(
        "optimal_decision"
    )
    probability = (result.get("probability_each_action_is_best") or {}).get(
        str(result.get("optimal_decision")), 0.0
    )
    return (
        f"[decision] {description} — expected utility {utility:{_SIG}}, "
        f"best in {probability * 100:.1f}% of worlds"
    )


def summarize_intervention_line(result: dict[str, Any]) -> str:
    variable = result.get("intervention_variable", "?")
    value = result.get("intervention_value")
    baseline = result.get("baseline_value")
    effect = result.get("causal_effect", 0.0)
    estimand = result.get("estimand", "")

    against = (
        f"do({variable}={baseline:{_SIG}})"
        if baseline is not None
        else "the stochastic model"
    )
    mode = (
        "policy reoptimised"
        if estimand == "reoptimised_policy_effect"
        else f"policy held at {result.get('fixed_decision')}"
    )
    return (
        f"[decision] do({variable}={value:{_SIG}}) vs {against}: "
        f"effect {effect:+{_SIG}} ({mode})"
    )


def summarize_posterior_line(posteriors: dict[str, Any]) -> str:
    return f"[posterior] summarised {len(posteriors)} learned cause distributions"


def summarize_samples_line(num_samples: int, seed: int) -> str:
    return f"[compute] {num_samples:,} Monte Carlo worlds, seed {seed}"


def summarize_error_line(message: str) -> str:
    return f"[error] {message[:200]}"
