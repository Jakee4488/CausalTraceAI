"""The decision problem this app solves: capacity planning.

Ported from ``causal_model.ipynb`` §4 (cells 34–42) — the three causes, the
three capacity policies, the deterministic utility function, and the ten
training observations.

Why this is built once and never re-learned
-------------------------------------------
``CausalDecisionAPI.learn`` performs conjugate updates that *accumulate* into
the hyperparameters — the notebook flags §4.6 as non-idempotent for exactly this
reason. A process-global API that called ``learn`` per request would retrain on
the same ten rows on every turn, and the posteriors would silently sharpen as
traffic arrived: the same question would get a more confident answer in the
afternoon than in the morning, with nothing in the output to show why.

So the training set is folded in exactly once, inside :func:`build_causal_api`,
and :func:`get_causal_api` caches the result for the process. Nothing downstream
calls ``learn``.
"""

from __future__ import annotations

import functools

import torch

from src.causal.causes import (
    BayesianBernoulliCause,
    BayesianNormalCause,
    CauseDistribution,
)
from src.causal.decision import CausalDecisionAPI

# ── The action set ───────────────────────────────────────────────────────────

DECISIONS: list[int] = [0, 1, 2]

DECISION_DESCRIPTIONS: dict[int, str] = {
    0: "Conservative capacity policy",
    1: "Balanced capacity policy",
    2: "Aggressive capacity policy",
}

UTILITY_DESCRIPTION = (
    "Profit-like reward equal to revenue minus fixed cost, competitive loss "
    "and unused-capacity penalty."
)

# Per-policy economics. Read by utility_function below; kept as data so the
# three branches cannot drift apart.
_POLICY_PARAMETERS: dict[int, dict[str, float]] = {
    0: {
        "capacity": 8.0,
        "margin_per_unit": 6.0,
        "fixed_cost": 8.0,
        "unused_capacity_penalty": 0.5,
    },
    1: {
        "capacity": 14.0,
        "margin_per_unit": 7.0,
        "fixed_cost": 18.0,
        "unused_capacity_penalty": 1.2,
    },
    2: {
        "capacity": 22.0,
        "margin_per_unit": 8.0,
        "fixed_cost": 35.0,
        "unused_capacity_penalty": 2.5,
    },
}

# ── Observations folded in at construction ───────────────────────────────────

TRAINING_OBSERVATIONS: dict[str, list[float]] = {
    "demand": [9.5, 11.0, 12.5, 10.8, 13.2, 8.9, 11.7, 12.1, 10.2, 14.0],
    "market_growth": [0.01, 0.03, -0.01, 0.04, 0.02, 0.01, 0.05, 0.00, 0.02, 0.03],
    "competitor_active": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
}


def build_causes() -> dict[str, CauseDistribution]:
    """Fresh priors, before any learning. A new dict every call — the cause
    objects are mutable, so sharing them across APIs would let one API's
    ``learn`` leak into another's posteriors."""
    return {
        "demand": BayesianNormalCause(
            name="demand",
            mu=10.0,
            kappa=1.0,
            alpha=3.0,
            beta=8.0,
            description="Current customer demand measured in demand units.",
        ),
        "market_growth": BayesianNormalCause(
            name="market_growth",
            mu=0.02,
            kappa=2.0,
            alpha=4.0,
            beta=0.02,
            description=(
                "Market growth rate represented as a decimal. For example, "
                "0.05 means five percent growth."
            ),
        ),
        "competitor_active": BayesianBernoulliCause(
            name="competitor_active",
            alpha=2.0,
            beta=3.0,
            description=(
                "Whether a competing company is actively running a strong "
                "campaign. Zero means inactive and one means active."
            ),
        ),
    }


def utility_function(
    decision: int, causes: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Deterministic structural equation ``U = u(D, demand, market_growth,
    competitor_active)``.

    There is no additional random utility noise: any uncertainty in U comes from
    uncertainty in the causes.
    """
    if decision not in _POLICY_PARAMETERS:
        raise ValueError(f"Unknown decision: {decision}")

    parameters = _POLICY_PARAMETERS[decision]
    capacity = parameters["capacity"]
    margin_per_unit = parameters["margin_per_unit"]
    fixed_cost = parameters["fixed_cost"]
    unused_capacity_penalty = parameters["unused_capacity_penalty"]

    demand = causes["demand"]
    market_growth = causes["market_growth"]
    competitor_active = causes["competitor_active"]

    effective_demand = demand * (1.0 + market_growth)
    capacity_tensor = torch.full_like(effective_demand, capacity)

    units_sold = torch.minimum(effective_demand, capacity_tensor)
    revenue = margin_per_unit * units_sold

    competitive_loss = (
        competitor_active
        * 2.0
        * torch.clamp(effective_demand - 5.0, min=0.0)
    )

    unused_capacity = torch.clamp(capacity_tensor - effective_demand, min=0.0)
    capacity_penalty = unused_capacity_penalty * unused_capacity

    return revenue - fixed_cost - competitive_loss - capacity_penalty


def build_causal_api(learn: bool = True) -> CausalDecisionAPI:
    """Construct the engine on fresh priors, folding the training set in once.

    ``learn=False`` yields the prior-only engine — used by tests that assert the
    conjugate update actually moved the hyperparameters.
    """
    api = CausalDecisionAPI(
        causes=build_causes(),
        decisions=DECISIONS,
        utility_function=utility_function,
        decision_descriptions=DECISION_DESCRIPTIONS,
        utility_description=UTILITY_DESCRIPTION,
    )
    if learn:
        api.learn(TRAINING_OBSERVATIONS)
    return api


@functools.cache
def get_causal_api() -> CausalDecisionAPI:
    """The process-wide engine. Cached, so the training data is folded in
    exactly once per process no matter how many turns arrive."""
    return build_causal_api()
