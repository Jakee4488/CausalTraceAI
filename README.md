# CausalTraceAI

Ask a business question in plain English. **Gemini** turns it into a precise
structured causal query, **Pyro** does all of the arithmetic by Monte Carlo
simulation, and **Gemini** narrates the resulting numbers back.

> **The LLM never does math.** It translates language *in* and narrates results
> *out*. Every number in an answer is produced by the Pyro engine.

The agent is a LangGraph state machine deployed to **Vertex AI Agent Engine**;
the browser talks to it through a FastAPI proxy on Cloud Run.

---

## What it answers

This is a **prescriptive** causal system, not a descriptive one. It does not
estimate "what is the effect of X on Y" — it answers:

| Query | Example |
|---|---|
| `optimal_policy` | "Demand is 13. Which capacity policy should I choose?" |
| `intervention_effect` | "What does forcing the competitor to be active cost me?" |
| `posterior_summary` | "What has the system learned about demand?" |

The decision problem is the capacity-planning model from
[`causal_model.ipynb`](causal_model.ipynb) §4: three independent root causes
(`demand`, `market_growth` as Normal-Inverse-Gamma; `competitor_active` as
Beta-Bernoulli), three capacity policies, and a deterministic profit-like
utility. Because every cause is a root and utility is deterministic given the
action and the causes, **all** the spread in U comes from uncertainty about the
causes — which is exactly what the Monte Carlo integrates over.

## Architecture

```
Browser ──► Cloud Run proxy ──► Vertex AI Agent Engine (LangGraph)
   ▲              │                        │
   └── SSE ───────┘◄─── {node: delta} ─────┘
```

```
interpret_query (LLM 1) → validate_query (0 LLM) → one of
    optimal_policy | intervention_effect | posterior_summary   (0 LLM, Pyro)
  → explain_result (LLM 2)
```

Two Gemini calls per answered question, zero for the math. `clarification` and
`error` terminate without the second call, so an ambiguous or rejected query
costs one.

`validate_query` re-checks every LLM-supplied name against the live engine
before any sampling: the prompt *asks* the model to use only known causes, and
this is what makes it a guarantee.

| Layer | Module | Ported from |
|---|---|---|
| Cause distributions | [`src/causal/causes.py`](src/causal/causes.py) | notebook §1 |
| Decision engine | [`src/causal/decision.py`](src/causal/decision.py) | notebook §2 |
| LangGraph pipeline | [`src/causal/graph_app.py`](src/causal/graph_app.py) | notebook §3 |
| The problem | [`src/causal/problem.py`](src/causal/problem.py) | notebook §4 |
| Graph pane payload | [`src/causal/graph_view.py`](src/causal/graph_view.py) | new |

### Two things worth knowing before changing anything

**`learn()` is not idempotent.** Conjugate updates accumulate into the
hyperparameters. `get_causal_api()` is process-cached so the ten training
observations are folded in exactly once; calling `learn` per request would make
the posteriors sharpen as traffic arrived, with nothing in the output to show
why.

**`stream_mode="updates"` yields what a node *returned*, not the reduced
channel.** Both channels with reducers (`causal_steps`, `causal_usage`) must
therefore be reduced a second time on the proxy side. Getting this wrong loses
every trace line except the last node's and bills each turn at the explainer's
token usage alone. [`tests/test_proxy_adapter.py`](tests/test_proxy_adapter.py)
pins it.

## Running it

```bash
# Agent + engine tests (needs torch/pyro)
pytest tests/ -q

# The UI, against the proxy's offline mock (no Agent Engine needed)
cd ui && npm install && npm run build
MOCK_FRAME_DELAY_S=0 uvicorn proxy.main:app --reload
```

With no `AGENT_ENGINE_ENDPOINT` the proxy serves a scripted run built from
[`proxy/mockdata.py`](proxy/mockdata.py) — real engine output, pasted, so the
offline path exercises the same payload shape the live one produces.

## Tests

| File | Guards |
|---|---|
| [`test_notebook_parity.py`](tests/test_notebook_parity.py) | The port reproduces the notebook **exactly** — it executes the notebook's own cells and diffs every float |
| [`test_graph_app.py`](tests/test_graph_app.py) | Routing, validation, degradation, transport shape, token accounting |
| [`test_proxy_adapter.py`](tests/test_proxy_adapter.py) | The agent↔proxy seam: real chunks through the proxy's real parsing |
| [`test_graph_view.py`](tests/test_graph_view.py) | The graph payload matches the renderer's contract |

The parity suite exists because the notebook ships with **cleared outputs** —
there is no committed ground truth to diff against, so it runs the notebook and
the port side by side under the same seed and requires bit-identical results. A
tolerance there would hide exactly the porting bug it exists to catch.

## Deploying

```bash
export AGENT_ENGINE_STAGING_BUCKET=gs://your-bucket
./deploy_to_gcp.sh                # agent → proxy → hosting, updating in place
./deploy_to_gcp.sh --only agent   # just the Agent Engine
python deploy_agent.py --dry-run  # show what would be sent

# A new engine + a new Cloud Run proxy, side by side with production
./deploy_new_stack.sh --stack v2 --dry-run
./deploy_new_stack.sh --stack v2 --app-url https://v2.causaltraceai.com
```

[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) is the full architecture and
deployment guide — read it before the first deploy of a new stack.
`deploy_to_gcp.sh` only ever updates the recorded engine in place;
`deploy_new_stack.sh` is what creates a new one, and it cannot touch production.

`deploy_agent.py` replaces `agents-cli`: that tool packages an ADK agent
directory, and this agent is a LangGraph runnable wrapped in `LanggraphAgent`,
which goes through the SDK's own `agent_engines.create`/`update`.

**Install the CPU torch wheel** (`--index-url https://download.pytorch.org/whl/cpu`)
or the deployed environment picks up CUDA builds and grows by roughly a gigabyte
for GPU support nothing here uses.

## Environment

| Variable | Default | Effect |
|---|---|---|
| `CAUSAL_MODEL` | `gemini-3.6-flash` | Model for both LLM calls |
| `CAUSAL_NUM_SAMPLES` | `20000` | Monte Carlo worlds per query |
| `CAUSAL_SEED` | `123` | Sampling seed — changing it changes every reported number |
| `AGENT_ENGINE_ENDPOINT` | — | Unset ⇒ the proxy serves its offline mock |
| `AGENT_ENGINE_STAGING_BUCKET` | — | Required by `deploy_agent.py` |

The proxy's own variables (access gate, Firestore, mail) are unchanged from the
upstream stack — see [`docs/access_control.md`](docs/access_control.md).
