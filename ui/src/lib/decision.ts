// Reading a decision result: is the winner real, and what was actually computed?
//
// Both answers are derivable from `causal_decision` alone, which the proxy
// already forwards in full — no new transport. These functions mirror the
// extractors in src/causal/node_logging.py field for field, deliberately: the
// number in the panel and the number in the server log have to agree, and two
// implementations that drift are worse than none.

import type { ActionEvaluation, CausalGraph, DecisionResult } from "../types";

/** The 95% convention decision.py already reports its interval bounds at. */
const Z_95 = 1.96;

export interface DecisionVerdict {
  best: ActionEvaluation;
  runnerUp: ActionEvaluation;
  /** Best minus runner-up expected utility. */
  margin: number;
  /** Independent combination of the two arms' Monte Carlo standard errors. */
  combinedError: number;
  /** The bar `margin` has to clear: 1.96 × combinedError. */
  threshold: number;
  /** False ⇒ the recommendation is inside sampling noise. */
  separated: boolean;
}

/**
 * Whether the recommended action is distinguishable from the runner-up.
 *
 * An expected utility is a Monte Carlo mean, so a margin smaller than the
 * sampling error means the "recommendation" is a coin flip. The card renders
 * this because nothing else in the UI does: the 95% whiskers are drawn, but a
 * reader is left to eyeball whether they overlap.
 *
 * Conservative by construction. Every action is evaluated against the *same*
 * sampled worlds, so the two errors are positively correlated and the true
 * standard error of the difference is smaller than the independent combination
 * used here. That can call a real difference indistinguishable; it cannot do
 * the reverse.
 */
export function decisionVerdict(
  rows: ActionEvaluation[] | undefined,
): DecisionVerdict | null {
  const usable = (rows ?? []).filter((row) => Number.isFinite(row?.expected_utility));
  if (usable.length < 2) return null;

  const ranked = [...usable].sort((a, b) => b.expected_utility - a.expected_utility);
  const [best, runnerUp] = ranked;

  const margin = best.expected_utility - runnerUp.expected_utility;
  const errors = [best, runnerUp].map((row) => row.monte_carlo_standard_error || 0);
  const combinedError = Math.sqrt(errors[0] ** 2 + errors[1] ** 2);
  const threshold = Z_95 * combinedError;

  return {
    best,
    runnerUp,
    margin,
    combinedError,
    threshold,
    separated: margin > threshold,
  };
}

/** `%g`-style: enough digits to be exact, none of the trailing zeros. */
function fmtG(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "?";
  return String(Number(value.toPrecision(6)));
}

function conditioning(
  context: Record<string, number>,
  interventions: Record<string, number>,
): string {
  const terms = Object.keys(context)
    .sort()
    .map((name) => `${name}=${fmtG(context[name])}`);
  terms.push(
    ...Object.keys(interventions)
      .sort()
      .map((name) => `do(${name}=${fmtG(interventions[name])})`),
  );
  return terms.join(", ");
}

/**
 * The estimand this result answers, written out.
 *
 * "expected utility 66.77" is a number; "argmax_d E[U | D=d, demand=13]" is the
 * question it answers. Only the second tells you whether the answer is the one
 * you asked for — which matters most for interventions, where an absent
 * baseline silently changes the estimand from do(C=c₀) to the stochastic model.
 */
export function renderEstimand(result: DecisionResult | null): string | null {
  if (!result) return null;

  if (result.query_type === "optimal_policy") {
    const terms = conditioning(result.context ?? {}, result.interventions ?? {});
    return terms ? `argmax_d E[U | D=d, ${terms}]` : "argmax_d E[U | D=d]";
  }

  if (result.query_type === "intervention_effect") {
    const variable = result.intervention_variable ?? "?";
    const target = `do(${variable}=${fmtG(result.intervention_value)})`;
    const baseline =
      result.baseline_value != null
        ? `do(${variable}=${fmtG(result.baseline_value)})`
        : "the stochastic model";
    if (result.estimand === "reoptimised_policy_effect") {
      return `max_d E[U | ${target}] - max_d E[U | ${baseline}]`;
    }
    const held = result.fixed_decision;
    return `E[U | ${target}, D=${held}] - E[U | ${baseline}, D=${held}]`;
  }

  return null;
}

export interface CauseSplit {
  observed: string[];
  intervened: string[];
  stochastic: string[];
}

/**
 * Which causes the answer pinned down, and which it integrated over.
 *
 * Read off the graph payload rather than the decision dict, because the graph
 * is the one place the split is stated for *every* query type. An
 * intervention_effect result carries no top-level `context`/`interventions` at
 * all — they live inside its per-arm sub-results — so deriving this from the
 * decision made an observed cause look integrated-over, contradicting the trace
 * line right beneath it.
 *
 * build_ui_graph in src/causal/graph_view.py owns the mapping, and the graph
 * pane paints from the same three statuses, so the words and the picture cannot
 * disagree.
 */
export function causeSplit(graph: CausalGraph | null | undefined): CauseSplit | null {
  const causes = (graph?.nodes ?? []).filter((node) => node.kind === "input");
  if (!causes.length) return null;
  const withStatus = (status: string) =>
    causes.filter((node) => node.status === status).map((node) => node.id);
  return {
    intervened: withStatus("active"),
    observed: withStatus("done"),
    stochastic: withStatus("pending"),
  };
}
