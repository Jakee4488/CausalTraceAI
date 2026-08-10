"""Canned payloads for the offline dev path (no ``AGENT_ENGINE_ENDPOINT``).

These are not invented shapes. Every value below was produced by running the
real engine — ``optimal_policy(context={"demand": 13.0}, num_samples=20_000,
seed=123)`` against the §4 capacity-planning problem — and pasted here. That
matters: a hand-written mock drifts from the live payload silently, and the
first person to notice is a user looking at a panel that renders blank against
the real backend.

Regenerate after any change to ``problem.py`` or ``graph_view.py``::

    python -c "import json; \
      from src.causal.problem import get_causal_api; \
      print(json.dumps(get_causal_api().optimal_policy( \
          context={'demand': 13.0}, num_samples=20000, seed=123), indent=1))"
"""

MOCK_STEPS = [
    "[interpret] optimal_policy; observed demand=13",
    "[compute] 20,000 Monte Carlo worlds, seed 123",
    "[compute] integrating over 2: market_growth, competitor_active",
    "[decision] Balanced capacity policy — expected utility 66.91, "
    "best in 100.0% of worlds",
]

MOCK_DECISION = {
    "query_type": "optimal_policy",
    "optimal_decision": 1,
    "optimal_decision_description": "Balanced capacity policy",
    "optimal_expected_utility": 66.90581512451172,
    "context": {"demand": 13.0},
    "interventions": {},
    "num_samples": 20000,
    "action_evaluations": [
        {
            "decision": 0,
            "decision_description": "Conservative capacity policy",
            "expected_utility": 33.39320755004883,
            "utility_standard_deviation": 8.141141891479492,
            "monte_carlo_standard_error": 0.0575665682554245,
            "mean_lower_95": 33.28037643432617,
            "mean_upper_95": 33.506038665771484,
        },
        {
            "decision": 1,
            "decision_description": "Balanced capacity policy",
            "expected_utility": 66.90581512451172,
            "utility_standard_deviation": 9.28825569152832,
            "monte_carlo_standard_error": 0.06567788869142532,
            "mean_lower_95": 66.77708435058594,
            "mean_upper_95": 67.0345458984375,
        },
        {
            "decision": 2,
            "decision_description": "Aggressive capacity policy",
            "expected_utility": 42.644866943359375,
            "utility_standard_deviation": 10.632640838623047,
            "monte_carlo_standard_error": 0.07518412172794342,
            "mean_lower_95": 42.49750518798828,
            "mean_upper_95": 42.79222869873047,
        },
    ],
    "probability_each_action_is_best": {
        "0": 0.0,
        "1": 0.9998999834060669,
        "2": 9.999999747378752e-05,
    },
}

# demand is observed (done), the other two are integrated over (pending) — the
# offline path therefore exercises all three node statuses, not just one.
MOCK_GRAPH = {
    "nodes": [
        {
            "id": "demand",
            "label": "demand",
            "kind": "input",
            "status": "done",
            "description": "Current customer demand measured in demand units.",
        },
        {
            "id": "market_growth",
            "label": "market growth",
            "kind": "input",
            "status": "pending",
            "description": (
                "Market growth rate represented as a decimal. For example, "
                "0.05 means five percent growth."
            ),
        },
        {
            "id": "competitor_active",
            "label": "competitor active",
            "kind": "input",
            "status": "pending",
            "description": (
                "Whether a competing company is actively running a strong "
                "campaign. Zero means inactive and one means active."
            ),
        },
        {
            "id": "decision",
            "label": "Balanced capacity policy",
            "kind": "process",
            "status": "done",
            "description": (
                "0: Conservative capacity policy; 1: Balanced capacity policy; "
                "2: Aggressive capacity policy"
            ),
        },
        {
            "id": "utility",
            "label": "utility",
            "kind": "outcome",
            "status": "done",
            "description": (
                "Profit-like reward equal to revenue minus fixed cost, "
                "competitive loss and unused-capacity penalty."
            ),
        },
    ],
    "edges": [
        {
            "source": "demand",
            "target": "utility",
            "relation": "causes",
            "confidence": 1.0,
            "rationale": "Every cause is a parent of utility.",
        },
        {
            "source": "market_growth",
            "target": "utility",
            "relation": "causes",
            "confidence": 1.0,
            "rationale": "Every cause is a parent of utility.",
        },
        {
            "source": "competitor_active",
            "target": "utility",
            "relation": "causes",
            "confidence": 1.0,
            "rationale": "Every cause is a parent of utility.",
        },
        {
            "source": "decision",
            "target": "utility",
            "relation": "causes",
            "confidence": 1.0,
            "rationale": (
                "Utility is deterministic given the decision and causes."
            ),
        },
    ],
    "critical_path": ["decision", "utility"],
    "version": 1,
}

MOCK_POSTERIORS = {
    "demand": {
        "distribution": "Normal-Inverse-Gamma",
        "posterior_mean": 11.263635635375977,
        "kappa": 11.0,
        "alpha": 8.0,
        "beta": 20.682727813720703,
        "expected_sampling_variance": 2.9546754019601003,
    },
    "market_growth": {
        "distribution": "Normal-Inverse-Gamma",
        "posterior_mean": 0.019999997690320015,
        "kappa": 12.0,
        "alpha": 9.0,
        "beta": 0.02149999886751175,
        "expected_sampling_variance": 0.0026874998584389687,
    },
    "competitor_active": {
        "distribution": "Beta-Bernoulli",
        "alpha": 6.0,
        "beta": 9.0,
        "posterior_probability_mean": 0.4,
        "posterior_probability_variance": 0.015,
    },
}
