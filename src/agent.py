"""CausalTraceAI — the Agent Engine entrypoint.

The compiled LangGraph app *is* the agent. ``LanggraphAgent`` exposes
``query`` / ``stream_query`` / ``get_state_history`` as the deployed schema and
handles tracing; its ``runnable_builder`` hook lets us return our own graph
instead of the default ReAct executor, so no adapter sits in between.

Construction here is deliberately inert. ``LanggraphAgent.__init__`` only
records its arguments — the model, the engine and the compiled graph are built
in ``set_up()``, which Agent Engine calls on the serving side. Importing this
module therefore needs no credentials, which is what lets the deploy script and
the tests import it freely.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv

from src.app_utils.telemetry import setup_telemetry
from src.causal import state as st
from src.causal.graph_app import build_causal_langgraph_app
from src.causal.problem import get_causal_api

load_dotenv()
logger = logging.getLogger(__name__)

MODEL = os.environ.get("CAUSAL_MODEL", "gemini-3.6-flash")


def _int_env(name: str, default: int) -> int:
    """Read a positive int from the environment, ignoring junk.

    A malformed value falls back to the default rather than failing the
    container at start-up: the sampling settings are a tuning knob, and a typo
    in one should not take the service down.
    """
    try:
        value = int(os.environ.get(name, "") or default)
    except ValueError:
        logger.warning("%s is not an integer; using %s", name, default)
        return default
    return value if value > 0 else default


NUM_SAMPLES = _int_env("CAUSAL_NUM_SAMPLES", st.DEFAULT_NUM_SAMPLES)
SEED = _int_env("CAUSAL_SEED", st.DEFAULT_SEED)

# Prompts and responses stay out of trace spans. Set before the agent is
# constructed, since the instrumentor reads this policy at set_up().
setup_telemetry()


def runnable_builder(model: Any, **kwargs: Any):
    """Return the compiled causal graph as the agent's runnable.

    Called by ``LanggraphAgent.set_up()`` with the built model (plus ``tools``,
    ``checkpointer`` and the ``*_kwargs`` it would have passed to the default
    ReAct builder — all irrelevant here, and swallowed by ``**kwargs`` so a new
    keyword in a future release does not break the signature).

    ``get_causal_api()`` is cached per process, so the ten training observations
    are folded into the priors exactly once no matter how many turns arrive.
    """
    return build_causal_langgraph_app(
        causal_api=get_causal_api(),
        model=model,
        default_num_samples=NUM_SAMPLES,
        default_seed=SEED,
    )


def build_agent(enable_tracing: bool = True):
    """Construct the deployable agent object.

    Deliberately *not* set up here: Agent Engine calls ``set_up()`` on the
    serving side, and doing it locally would initialise clients that cannot be
    serialised into the deployment package.
    """
    from vertexai.agent_engines.templates.langgraph import LanggraphAgent

    return LanggraphAgent(
        model=MODEL,
        runnable_builder=runnable_builder,
        enable_tracing=enable_tracing,
    )


agent = build_agent()


if __name__ == "__main__":
    # Local smoke test: needs ADC and a Vertex-enabled project.
    agent.set_up()
    for chunk in agent.stream_query(
        input={"user_input": "Demand is 13 units. Which policy should I choose?"},
        stream_mode="updates",
    ):
        print(chunk)
