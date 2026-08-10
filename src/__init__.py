"""CausalTraceAI package exports.

Lazy (PEP 562) so importing a pure submodule like ``src.causal.decision`` does
not pull in the serving stack, which resolves GCP credentials at import time.
"""

__all__ = ["agent", "root_app"]


def __getattr__(name):
    if name in ("agent", "root_app"):
        from src import agent as _agent

        return {"agent": _agent.agent, "root_app": _agent.agent}[name]
    raise AttributeError(f"module 'src' has no attribute {name!r}")
