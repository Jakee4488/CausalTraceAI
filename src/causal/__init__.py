"""The causal decision engine.

Three layers, ported from ``causal_model.ipynb``:

- :mod:`causes` — conjugate Bayesian marginals for each independent root cause.
- :mod:`decision` — ``CausalDecisionAPI``: Monte Carlo worlds, utility, policy.
- :mod:`graph_app` — the LangGraph state machine Gemini drives.

Nothing here imports the serving stack, so the math is testable without
credentials or network.
"""
