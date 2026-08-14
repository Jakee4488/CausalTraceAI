"""Run the causal agent locally, speaking the Agent Engine's wire protocol.

Why this exists
---------------
The proxy talks to the agent over HTTP: it POSTs to the Agent Engine's
``:streamQuery`` endpoint and reads NDJSON lines, one per LangGraph ``updates``
chunk. Locally there are normally only two options, and neither exercises the
agent: ``MODE=mock`` serves canned frames from ``proxy/mockdata.py`` without
calling anything, and pointing at the deployed engine runs the graph in Vertex,
where its logs go to Cloud Logging rather than your terminal.

This serves that same endpoint from localhost with the *real* compiled graph in
process — real Gemini calls, real Pyro sampling, real per-node JSON logs on this
terminal. The proxy needs no change to use it: it is the same protocol.

    python run_local_agent.py          # → http://127.0.0.1:8081

Then point the proxy at it and open the UI:

    AGENT_ENGINE_ENDPOINT=http://127.0.0.1:8081/v1/local:query \
    ACCESS_STORE=memory uvicorn proxy.main:app --port 8080

Credentials: ``GOOGLE_API_KEY`` from ``.env``, which is what the local
``ChatGoogleGenerativeAI`` reads. The deployed agent instead gets its model from
``LanggraphAgent``'s own builder (Gemini through Vertex, ADC); everything below
the model — the graph, the engine, the sampling settings, the logging — is the
same code path via ``runnable_builder``.

Dev tool. Not shipped: ``deploy_agent.py`` packages ``src`` only.
"""

from __future__ import annotations

import json
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

load_dotenv()

# Importing src.agent is what configures JSON logging and telemetry, exactly as
# it does on the serving side. It is inert beyond that — no credentials are
# resolved at import — so this stays a cheap import.
from src.agent import MODEL, NUM_SAMPLES, SEED, runnable_builder  # noqa: E402

PORT = int(os.environ.get("LOCAL_AGENT_PORT", "8081"))

app = FastAPI(title="Local Agent Engine stand-in")
_graph = None


def _build_graph():
    """Compile the graph once, on first request.

    Deferred rather than done at import for the same reason ``set_up()`` is:
    building the model resolves credentials, and this module should be
    importable without them.
    """
    global _graph
    if _graph is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Put it in .env, or export it."
            )
        # Same hook the deployed LanggraphAgent calls, so the graph, the engine
        # and the sampling settings are assembled identically to production.
        _graph = runnable_builder(ChatGoogleGenerativeAI(model=MODEL))
    return _graph


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL, "num_samples": NUM_SAMPLES, "seed": SEED}


@app.post("/{path:path}")
async def stream_query(path: str, request: Request) -> StreamingResponse:
    """The ``:streamQuery`` contract, as the proxy uses it.

    Accepts any path so the proxy's ``:query`` → ``:streamQuery`` rewrite lands
    here whatever engine id it was given.
    """
    body = await request.json()
    payload = body.get("input") or {}
    graph_input = payload.get("input") or {}
    stream_mode = payload.get("stream_mode", "updates")

    def chunks():
        # NDJSON: one JSON object per line, which is what _pump_lines in
        # proxy/main.py reads. A trailing newline per chunk is required — the
        # proxy splits on lines, not on object boundaries.
        for chunk in _build_graph().stream(graph_input, stream_mode=stream_mode):
            yield json.dumps(chunk, default=str) + "\n"

    return StreamingResponse(chunks(), media_type="application/x-ndjson")


if __name__ == "__main__":
    print(
        f"[local-agent] model={MODEL} num_samples={NUM_SAMPLES} seed={SEED}\n"
        f"[local-agent] listening on http://127.0.0.1:{PORT}\n"
        f"[local-agent] point the proxy at "
        f"http://127.0.0.1:{PORT}/v1/local:query\n"
    )
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
