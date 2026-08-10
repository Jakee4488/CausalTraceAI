"""The port is faithful to ``causal_model.ipynb``, or this fails.

The notebook ships with cleared outputs, so there is no committed ground truth
to diff against. Instead these tests execute the notebook's own §1–§4 cells in a
fresh namespace, build the ported engine alongside, drive both with the same
seed and sample count, and require every number to match **exactly**.

Exact, not approximate: both sides call ``pyro.set_rng_seed`` with the same seed
and walk the causes in the same order, so identical code must produce identical
floats. A tolerance here would hide precisely the porting bug this exists to
catch — a reordered sample site, a dropped hyperparameter, a changed default.

Requires torch/pyro; skipped where they are absent.
"""

from __future__ import annotations

import json
import math
import pathlib

import pytest

pytest.importorskip("torch")
pytest.importorskip("pyro")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "causal_model.ipynb"

# §1 layer-1 + §2 layer-2, then the §4 worked configuration. Cells 2 and 20
# import langgraph and build the LLM client; the math needs neither.
ENGINE_CELLS = [6, 8, 10, 12, 14, 16, 18]
CONFIG_CELLS = [34, 36, 38, 40, 42]

FULL_CONTEXT = {"demand": 13.0, "market_growth": 0.04, "competitor_active": 1.0}
PARTIAL_CONTEXT = {"demand": 13.0}


@pytest.fixture(scope="module")
def notebook_namespace() -> dict:
    """The notebook's own classes and its constructed, learned ``causal_api``."""
    if not NOTEBOOK.exists():  # pragma: no cover - repo layout guard
        pytest.skip(f"{NOTEBOOK.name} not present")
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict = {}
    for index in ENGINE_CELLS + CONFIG_CELLS:
        source = "".join(nb["cells"][index]["source"])
        exec(compile(source, f"<notebook cell {index}>", "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def notebook_api(notebook_namespace):
    return notebook_namespace["causal_api"]


@pytest.fixture(scope="module")
def port_api():
    from src.causal.problem import build_causal_api

    # Not get_causal_api(): the process-wide cache would let one test's engine
    # leak into another's, and these compare hyperparameters.
    return build_causal_api()


def assert_identical(left, right, path: str = "") -> None:
    """Deep compare with exact float equality and a path in the failure."""
    if isinstance(left, dict) and isinstance(right, dict):
        assert set(left) == set(right), (
            f"{path}: key mismatch — only-notebook="
            f"{sorted(set(left) - set(right))}, only-port="
            f"{sorted(set(right) - set(left))}"
        )
        for key in left:
            assert_identical(left[key], right[key], f"{path}.{key}")
        return

    if isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right), (
            f"{path}: length {len(left)} != {len(right)}"
        )
        for index, (a, b) in enumerate(zip(left, right)):
            assert_identical(a, b, f"{path}[{index}]")
        return

    if isinstance(left, float) or isinstance(right, float):
        if math.isinf(left) and math.isinf(right):
            assert left == right, f"{path}: {left} != {right}"
            return

    assert left == right, f"{path}: {left!r} != {right!r}"


def test_learned_posteriors_match(notebook_api, port_api):
    """The same ten observations produce the same conjugate hyperparameters."""
    assert_identical(
        notebook_api.posterior_summaries(),
        port_api.posterior_summaries(),
        "posterior_summaries",
    )


def test_graph_description_matches(notebook_api, port_api):
    """describe_graph() is the payload the interpreter prompt is built from, so
    drift here silently changes what the LLM is told the model contains."""
    assert_identical(
        notebook_api.describe_graph(),
        port_api.describe_graph(),
        "describe_graph",
    )


def test_learning_actually_moved_the_priors(port_api):
    """Guards the parity tests themselves: if learn() were a no-op on both
    sides they would agree on the priors and prove nothing."""
    from src.causal.problem import build_causal_api

    prior_only = build_causal_api(learn=False)
    assert prior_only.causes["demand"].kappa == 1.0
    assert port_api.causes["demand"].kappa == 11.0  # 1 + 10 observations
    assert prior_only.causes["competitor_active"].alpha == 2.0
    assert port_api.causes["competitor_active"].alpha == 6.0  # 2 + 4 ones


@pytest.mark.parametrize(
    "context, label",
    [(PARTIAL_CONTEXT, "partial"), (FULL_CONTEXT, "full")],
)
def test_optimal_policy_matches(notebook_api, port_api, context, label):
    """§5.1 and §5.2 — policy under partial information and full context."""
    assert_identical(
        notebook_api.optimal_policy(
            context=context, num_samples=20_000, seed=123
        ),
        port_api.optimal_policy(
            context=context, num_samples=20_000, seed=123
        ),
        f"optimal_policy({label})",
    )


@pytest.mark.parametrize(
    "mode, fixed_decision",
    [("fixed_policy", 1), ("reoptimise_policy", None)],
)
def test_intervention_effect_matches(
    notebook_api, notebook_namespace, port_api, mode, fixed_decision
):
    """§5.3 and §5.4 — both estimands, including the paired-seed contrast."""
    from src.causal.causes import Intervention

    kwargs = dict(
        variable="competitor_active",
        value=1.0,
        baseline_value=0.0,
        mode=mode,
        fixed_decision=fixed_decision,
    )
    call = dict(context=PARTIAL_CONTEXT, num_samples=20_000, seed=123)

    assert_identical(
        notebook_api.intervention_effect(
            intervention=notebook_namespace["Intervention"](**kwargs), **call
        ),
        port_api.intervention_effect(
            intervention=Intervention(**kwargs), **call
        ),
        f"intervention_effect({mode})",
    )


def test_same_seed_is_reproducible(port_api):
    """Two calls with the same seed and sample count are directly comparable —
    the property the whole common-random-numbers design rests on."""
    first = port_api.optimal_policy(
        context=PARTIAL_CONTEXT, num_samples=5_000, seed=7
    )
    second = port_api.optimal_policy(
        context=PARTIAL_CONTEXT, num_samples=5_000, seed=7
    )
    assert_identical(first, second, "reproducibility")


def test_probabilities_sum_to_one(port_api):
    result = port_api.optimal_policy(
        context=PARTIAL_CONTEXT, num_samples=20_000, seed=123
    )
    total = sum(result["probability_each_action_is_best"].values())
    assert total == pytest.approx(1.0, abs=1e-6)
