"""Layer 2 — ``CausalDecisionAPI``, the math engine.

One class holding everything the LangGraph layer is allowed to call. Constructed
from the causes (:mod:`src.causal.causes`), a list of decisions, and a
deterministic ``utility_function(decision, causes) -> Tensor``.

How ``do(...)`` is implemented
------------------------------
``_pyro_world_model`` walks the causes and picks one of three cases per cause::

    name in interventions  ->  clamp to a constant   do(C = c)
    name in context        ->  clamp to a constant   observed X = x
    neither                ->  cause.pyro_sample()   stays stochastic

The first two branches are identical code, and that is not an oversight — it is
a property of this graph. Every cause is a root with no parents, so there is
nothing to cut: ``P(U | do(C=c), D=d)`` and ``P(U | C=c, D=d)`` are the same
distribution here. The distinction is kept in the *interface* because it is the
honest description of what the user asked, and because the moment any cause
gains a parent the two branches must diverge. Where the distinction already
bites numerically is the fixed-vs-reoptimised axis of ``intervention_effect``.

Common random numbers
---------------------
``evaluate_all_decisions`` samples the worlds **once** and evaluates **every**
decision against **the same** worlds, so comparisons differ only by the
decision, not by sampling noise. ``intervention_effect`` extends the trick
across arms by passing the identical seed to both, making the contrast paired.
The practical consequence: the reported ``monte_carlo_standard_error`` is
correct for each decision's expected utility on its own, but **conservative for
comparisons** — the true standard error of a difference is smaller than the two
individual errors suggest, because they are positively correlated by
construction.

Ported from ``causal_model.ipynb`` §2 (cell 18).
"""

from __future__ import annotations

import math
from typing import Any, Callable

import pyro
import torch

from src.causal.causes import CauseDistribution, Decision, Intervention, Tensor


class CausalDecisionAPI:
    """Causal structure::

        C_1  ──┐
        C_2  ──┤
        ...    ├──> U
        C_p  ──┤
        D    ──┘

    Assumptions:

    1. Every C_j is a root cause.
    2. The causes are mutually independent.
    3. D is the decision variable.
    4. U is deterministic given D and C.
    5. Each cause has its own learnable marginal distribution.

    The expected value of decision d given observed context x is
    ``V(d, x) = E[u(d, x, C_unobserved)]``, and the optimal policy at context x
    is ``d*(x) = argmax_d V(d, x)``.
    """

    def __init__(
        self,
        causes: dict[str, CauseDistribution],
        decisions: list[Decision],
        utility_function: Callable[[Decision, dict[str, Tensor]], Tensor],
        decision_descriptions: dict[Decision, str] | None = None,
        utility_description: str = "",
    ) -> None:
        if not causes:
            raise ValueError("At least one cause is required.")
        if not decisions:
            raise ValueError("At least one decision is required.")
        if len(causes) != len(set(causes)):
            raise ValueError("Cause names must be unique.")

        for name, cause in causes.items():
            if name != cause.name:
                raise ValueError(
                    f"Dictionary key '{name}' does not match "
                    f"cause name '{cause.name}'."
                )

        self.causes = causes
        self.decisions = list(decisions)
        self.utility_function = utility_function
        self.decision_descriptions = decision_descriptions or {
            decision: str(decision) for decision in decisions
        }
        self.utility_description = utility_description

    # ── Model description ────────────────────────────────────────────────────

    def describe_graph(self) -> dict[str, Any]:
        """Metadata used both by developers and by the LLM query interpreter."""
        return {
            "causes": {
                name: cause.schema() for name, cause in self.causes.items()
            },
            "decision_variable": "D",
            "decisions": {
                str(decision): self.decision_descriptions.get(
                    decision, str(decision)
                )
                for decision in self.decisions
            },
            "utility_variable": "U",
            "utility_description": self.utility_description,
            "assumptions": [
                "All causes are mutually independent.",
                "All causes are parents of utility.",
                "The decision is a parent of utility.",
                "Utility is deterministic given the decision and causes.",
            ],
        }

    # ── Learning ─────────────────────────────────────────────────────────────

    def learn(self, observations: dict[str, list[Any]]) -> dict[str, Any]:
        """Update every supplied marginal distribution.

        NOT IDEMPOTENT — conjugate updates accumulate into the hyperparameters,
        so calling this twice with the same data trains on it twice. See
        :mod:`src.causal.problem`, which folds the training set in exactly once
        at construction and never calls this again at runtime.
        """
        unknown = set(observations) - set(self.causes)
        if unknown:
            raise ValueError(
                f"Unknown causes in training data: {sorted(unknown)}"
            )

        for name, values in observations.items():
            self.causes[name].update(values)

        return self.posterior_summaries()

    def posterior_summaries(self) -> dict[str, Any]:
        return {
            name: cause.posterior_summary()
            for name, cause in self.causes.items()
        }

    # ── Validation ───────────────────────────────────────────────────────────

    def _validate_context(self, context: dict[str, float] | None) -> None:
        unknown = set(context or {}) - set(self.causes)
        if unknown:
            raise ValueError(f"Unknown context variables: {sorted(unknown)}")

    def _validate_interventions(
        self, interventions: dict[str, float] | None
    ) -> None:
        unknown = set(interventions or {}) - set(self.causes)
        if unknown:
            raise ValueError(
                f"Unknown intervention variables: {sorted(unknown)}"
            )

    # ── Pyro causal model ────────────────────────────────────────────────────

    def _pyro_world_model(
        self,
        num_samples: int,
        context: dict[str, float] | None = None,
        interventions: dict[str, float] | None = None,
    ) -> dict[str, Tensor]:
        """Pyro model generating Monte Carlo causal worlds.

        Context variables are observed before D is selected. Interventions
        replace the structural mechanism for that variable with a constant.
        """
        context = context or {}
        interventions = interventions or {}
        sampled_causes: dict[str, Tensor] = {}

        for name, cause in self.causes.items():
            if name in interventions:
                sampled_causes[name] = torch.full(
                    (num_samples,),
                    float(interventions[name]),
                    dtype=torch.float32,
                )
            elif name in context:
                sampled_causes[name] = torch.full(
                    (num_samples,),
                    float(context[name]),
                    dtype=torch.float32,
                )
            else:
                sampled_causes[name] = cause.pyro_sample(
                    num_samples=num_samples
                )

        return sampled_causes

    def sample_causal_worlds(
        self,
        num_samples: int = 10_000,
        context: dict[str, float] | None = None,
        interventions: dict[str, float] | None = None,
        seed: int = 123,
    ) -> dict[str, Tensor]:
        """Execute the Pyro model and return sampled cause values."""
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2.")

        self._validate_context(context)
        self._validate_interventions(interventions)

        context = context or {}
        interventions = interventions or {}

        # A contradiction — the same variable observed as one value and
        # intervened to another — raises rather than silently preferring one.
        # Consistent duplicates are allowed.
        for variable in set(context) & set(interventions):
            observed_value = float(context[variable])
            intervention_value = float(interventions[variable])
            if observed_value != intervention_value:
                raise ValueError(
                    f"'{variable}' is observed as {observed_value} but "
                    f"intervened on as {intervention_value}."
                )

        pyro.set_rng_seed(seed)
        traced_model = pyro.poutine.trace(self._pyro_world_model)
        trace = traced_model.get_trace(
            num_samples=num_samples,
            context=context,
            interventions=interventions,
        )
        return trace.nodes["_RETURN"]["value"]

    # ── Utility evaluation ───────────────────────────────────────────────────

    def evaluate_decision(
        self, decision: Decision, sampled_causes: dict[str, Tensor]
    ) -> Tensor:
        if decision not in self.decisions:
            raise ValueError(f"Unknown decision '{decision}'.")

        utility = self.utility_function(decision, sampled_causes)
        if not isinstance(utility, torch.Tensor):
            utility = torch.as_tensor(utility, dtype=torch.float32)
        utility = utility.reshape(-1)

        expected_size = next(iter(sampled_causes.values())).numel()
        if utility.numel() != expected_size:
            raise ValueError(
                "The utility function must return exactly one utility value "
                "for each simulated causal world. "
                f"Expected {expected_size}, received {utility.numel()}."
            )

        return utility.float()

    def evaluate_all_decisions(
        self,
        context: dict[str, float] | None = None,
        interventions: dict[str, float] | None = None,
        num_samples: int = 10_000,
        seed: int = 123,
    ) -> dict[str, Any]:
        """Evaluate every decision against one shared set of sampled worlds."""
        sampled_causes = self.sample_causal_worlds(
            num_samples=num_samples,
            context=context,
            interventions=interventions,
            seed=seed,
        )

        utility_samples: dict[Decision, Tensor] = {}
        evaluations: list[dict[str, Any]] = []

        for decision in self.decisions:
            utilities = self.evaluate_decision(
                decision=decision, sampled_causes=sampled_causes
            )
            utility_samples[decision] = utilities

            mean = utilities.mean()
            standard_deviation = utilities.std(unbiased=True)
            standard_error = standard_deviation / math.sqrt(num_samples)

            evaluations.append(
                {
                    "decision": decision,
                    "decision_description": self.decision_descriptions.get(
                        decision, str(decision)
                    ),
                    "expected_utility": float(mean),
                    "utility_standard_deviation": float(standard_deviation),
                    "monte_carlo_standard_error": float(standard_error),
                    "mean_lower_95": float(mean - 1.96 * standard_error),
                    "mean_upper_95": float(mean + 1.96 * standard_error),
                }
            )

        utility_matrix = torch.stack(
            [utility_samples[decision] for decision in self.decisions], dim=1
        )
        winning_decision_indexes = utility_matrix.argmax(dim=1)

        # A different question from expected_utility: an action can have the
        # highest average payoff while winning in a minority of worlds. Ties in
        # argmax go to the lowest index, immaterial for continuous utilities.
        probability_best = {
            str(decision): float(
                (winning_decision_indexes == index).float().mean()
            )
            for index, decision in enumerate(self.decisions)
        }

        return {
            "context": context or {},
            "interventions": interventions or {},
            "num_samples": num_samples,
            "action_evaluations": evaluations,
            "probability_each_action_is_best": probability_best,
        }

    # ── Optimal policy ───────────────────────────────────────────────────────

    def optimal_policy(
        self,
        context: dict[str, float] | None = None,
        interventions: dict[str, float] | None = None,
        num_samples: int = 10_000,
        seed: int = 123,
    ) -> dict[str, Any]:
        """Calculate ``d*(x) = argmax_d E[U | D=d, X=x]``, optionally under an
        intervention."""
        evaluation = self.evaluate_all_decisions(
            context=context,
            interventions=interventions,
            num_samples=num_samples,
            seed=seed,
        )
        optimal_evaluation = max(
            evaluation["action_evaluations"],
            key=lambda row: row["expected_utility"],
        )
        return {
            "query_type": "optimal_policy",
            "optimal_decision": optimal_evaluation["decision"],
            "optimal_decision_description": optimal_evaluation[
                "decision_description"
            ],
            "optimal_expected_utility": optimal_evaluation["expected_utility"],
            **evaluation,
        }

    # ── Intervention effect ──────────────────────────────────────────────────

    def intervention_effect(
        self,
        intervention: Intervention,
        context: dict[str, float] | None = None,
        num_samples: int = 10_000,
        seed: int = 123,
    ) -> dict[str, Any]:
        """Supports two causal estimands.

        Fixed-policy effect::

            E[U | do(C_k=c_1), D=d] - E[U | do(C_k=c_0), D=d]

        Reoptimised-policy effect::

            max_d E[U | do(C_k=c_1), D=d] - max_d E[U | do(C_k=c_0), D=d]
        """
        if intervention.variable not in self.causes:
            raise ValueError(
                f"Unknown intervention variable '{intervention.variable}'."
            )

        if intervention.mode == "fixed_policy":
            if intervention.fixed_decision is None:
                raise ValueError("fixed_policy mode requires fixed_decision.")
            if intervention.fixed_decision not in self.decisions:
                raise ValueError(
                    f"Unknown fixed decision '{intervention.fixed_decision}'."
                )

        target_interventions = {intervention.variable: intervention.value}
        baseline_interventions = (
            {intervention.variable: intervention.baseline_value}
            if intervention.baseline_value is not None
            else {}
        )

        if intervention.mode == "reoptimise_policy":
            # Identical seed on both arms — the contrast is paired.
            target = self.optimal_policy(
                context=context,
                interventions=target_interventions,
                num_samples=num_samples,
                seed=seed,
            )
            baseline = self.optimal_policy(
                context=context,
                interventions=baseline_interventions,
                num_samples=num_samples,
                seed=seed,
            )
            effect = (
                target["optimal_expected_utility"]
                - baseline["optimal_expected_utility"]
            )
            return {
                "query_type": "intervention_effect",
                "estimand": "reoptimised_policy_effect",
                "intervention_variable": intervention.variable,
                "intervention_value": intervention.value,
                "baseline_value": intervention.baseline_value,
                "target_optimal_decision": target["optimal_decision"],
                "target_optimal_decision_description": target[
                    "optimal_decision_description"
                ],
                "baseline_optimal_decision": baseline["optimal_decision"],
                "baseline_optimal_decision_description": baseline[
                    "optimal_decision_description"
                ],
                "target_expected_utility": target["optimal_expected_utility"],
                "baseline_expected_utility": baseline[
                    "optimal_expected_utility"
                ],
                "causal_effect": effect,
                "target_result": target,
                "baseline_result": baseline,
            }

        fixed_decision = intervention.fixed_decision

        target_evaluation = self.evaluate_all_decisions(
            context=context,
            interventions=target_interventions,
            num_samples=num_samples,
            seed=seed,
        )
        baseline_evaluation = self.evaluate_all_decisions(
            context=context,
            interventions=baseline_interventions,
            num_samples=num_samples,
            seed=seed,
        )

        target_row = next(
            row
            for row in target_evaluation["action_evaluations"]
            if row["decision"] == fixed_decision
        )
        baseline_row = next(
            row
            for row in baseline_evaluation["action_evaluations"]
            if row["decision"] == fixed_decision
        )
        effect = (
            target_row["expected_utility"] - baseline_row["expected_utility"]
        )

        return {
            "query_type": "intervention_effect",
            "estimand": "fixed_policy_effect",
            "fixed_decision": fixed_decision,
            "fixed_decision_description": self.decision_descriptions.get(
                fixed_decision, str(fixed_decision)
            ),
            "intervention_variable": intervention.variable,
            "intervention_value": intervention.value,
            "baseline_value": intervention.baseline_value,
            "target_expected_utility": target_row["expected_utility"],
            "baseline_expected_utility": baseline_row["expected_utility"],
            "causal_effect": effect,
            "target_evaluation": target_row,
            "baseline_evaluation": baseline_row,
        }
