// The engine's answer: which action, what it is worth, and how sure we are.
//
// Three sub-views, one per query_type, sharing the plot grid EffectChart
// established (gutter / track / meta) so the pane reads as one instrument.
//
// On colour: the actions are *alternatives being ranked*, not independent
// series, so this deliberately uses no categorical palette. Giving policy 0/1/2
// three hues would assert an identity that carries no meaning and would repaint
// if the action set ever changed. Instead one action is emphasised — the
// recommended one — and the rest stay recessive. Emphasis is never colour
// alone: the winner also carries a "✓ recommended" chip.
//
// Built from positioned HTML rather than SVG for the same reason as
// EffectChart: the plots are one-dimensional, so percentage positioning is
// responsive without a viewBox distorting the marks or the type.

import { Fragment } from "react";

import { causeSplit, decisionVerdict, renderEstimand } from "../../lib/decision";
import { fmtNum } from "../../lib/markdown";
import type {
  ActionEvaluation,
  CausalGraph,
  DecisionResult,
  PosteriorSummaries,
} from "../../types";

/** Linear scale across the data. Unlike EffectChart's, this does NOT force zero
 *  on-scale: utilities here are profit-like and all positive, and anchoring at
 *  zero would squash three actions into the right-hand third of the track. */
function makeScale(values: number[]) {
  const finite = values.filter((v) => Number.isFinite(v));
  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (!finite.length || min === max) {
    min = (finite[0] ?? 0) - 1;
    max = (finite[0] ?? 0) + 1;
  }
  const pad = (max - min) * 0.12;
  min -= pad;
  max += pad;
  return (value: number) => ((value - min) / (max - min)) * 100;
}

function actionScaleValues(rows: ActionEvaluation[]): number[] {
  const values: number[] = [];
  for (const row of rows) {
    values.push(row.expected_utility, row.mean_lower_95, row.mean_upper_95);
  }
  return values;
}

// ── Optimal policy ──────────────────────────────────────────────────────────

function ActionRow({
  row,
  pct,
  probability,
  isBest,
}: {
  row: ActionEvaluation;
  pct: (v: number) => number;
  probability: number;
  isBest: boolean;
}) {
  const point = pct(row.expected_utility);
  const low = pct(row.mean_lower_95);
  const high = pct(row.mean_upper_95);
  const state = isBest ? "best" : "alt";

  return (
    <>
      <span className={"plot-gutter action-name " + state} title={row.decision_description}>
        {row.decision_description}
      </span>
      <div className="plot-track action-track">
        {/* The 95% interval is on the MEAN, not on U — it is estimation
            precision, not the spread of outcomes. */}
        <span
          className={"action-whisker " + state}
          style={{ left: `${Math.min(low, high)}%`, width: `${Math.abs(high - low)}%` }}
        />
        <span
          className={"action-point " + state}
          style={{ left: `${point}%` }}
          title={
            `expected utility ${fmtNum(row.expected_utility)} ` +
            `± ${fmtNum(row.monte_carlo_standard_error)} (MC standard error)`
          }
        />
      </div>
      <span className="plot-meta action-value">{fmtNum(row.expected_utility)}</span>

      <span className="plot-gutter prob-label">wins in</span>
      <div className="plot-track prob-track" title={`best in ${(probability * 100).toFixed(1)}% of simulated worlds`}>
        <span className={"prob-fill " + state} style={{ width: `${probability * 100}%` }} />
      </div>
      <span className="plot-meta prob-value">{(probability * 100).toFixed(1)}%</span>
    </>
  );
}

function PolicyView({ result }: { result: DecisionResult }) {
  const rows = result.action_evaluations || [];
  const pct = makeScale(actionScaleValues(rows));
  const probabilities = result.probability_each_action_is_best || {};
  const verdict = decisionVerdict(rows);

  // A margin inside sampling error is not a recommendation, so the hero stops
  // presenting it as one. Everything below is unchanged — the reader still gets
  // the full table and can see for themselves how close it was.
  const tooClose = verdict != null && !verdict.separated;

  return (
    <>
      <div className={"decision-hero" + (tooClose ? " unresolved" : "")}>
        <span className="decision-hero-label">
          {tooClose ? "too close to call" : "recommended"}
        </span>
        <span className="decision-hero-value">
          {tooClose && verdict
            ? `${verdict.best.decision_description} or ${verdict.runnerUp.decision_description}`
            : result.optimal_decision_description ?? String(result.optimal_decision)}
        </span>
        <span className="decision-hero-utility">
          expected utility {fmtNum(result.optimal_expected_utility)}
        </span>
      </div>

      {verdict && (
        <div className={"decision-confidence " + (tooClose ? "warn" : "ok")}>
          <span className="confidence-icon" aria-hidden="true">
            {tooClose ? "≈" : "✓"}
          </span>
          <span className="confidence-text">
            {/* Four digits, not three: a margin this close to zero rounds to
                "0.01" at the card's default precision, which reads as a real
                gap rather than the near-tie it is. */}
            {tooClose ? (
              <>
                <strong>{fmtNum(verdict.margin, 4)}</strong> ahead of{" "}
                {verdict.runnerUp.decision_description}, against ±
                {fmtNum(verdict.threshold, 4)} of Monte Carlo error — the gap is
                inside the noise, so this ranking would not survive a different
                seed.
              </>
            ) : (
              <>
                <strong>{fmtNum(verdict.margin, 4)}</strong> clear of{" "}
                {verdict.runnerUp.decision_description}, against ±
                {fmtNum(verdict.threshold, 4)} of Monte Carlo error — the ranking
                is real, not a sampling artefact.
              </>
            )}
          </span>
        </div>
      )}

      <div className="decision-chart">
        <div className="decision-caption">
          point = expected utility, bar = 95% interval on that mean — all actions
          evaluated against the same {(result.num_samples ?? 0).toLocaleString()} simulated worlds
        </div>
        {rows.map((row) => (
          <ActionRow
            key={String(row.decision)}
            row={row}
            pct={pct}
            probability={probabilities[String(row.decision)] ?? 0}
            isBest={row.decision === result.optimal_decision}
          />
        ))}
      </div>
    </>
  );
}

// ── Intervention effect ─────────────────────────────────────────────────────

function InterventionView({ result }: { result: DecisionResult }) {
  const target = result.target_expected_utility ?? 0;
  const baseline = result.baseline_expected_utility ?? 0;
  const effect = result.causal_effect ?? 0;
  const pct = makeScale([target, baseline]);
  const reoptimised = result.estimand === "reoptimised_policy_effect";

  const arms = [
    {
      key: "target",
      label: `do(${result.intervention_variable} = ${fmtNum(result.intervention_value)})`,
      value: target,
      decision: result.target_optimal_decision_description,
    },
    {
      key: "baseline",
      label:
        result.baseline_value != null
          ? `do(${result.intervention_variable} = ${fmtNum(result.baseline_value)})`
          : "no intervention",
      value: baseline,
      decision: result.baseline_optimal_decision_description,
    },
  ];

  // Did the intervention change what you should *do*, or only what you earn?
  // Two different findings, and the arm rows below state each arm's choice
  // without ever comparing them.
  const flipped =
    reoptimised &&
    result.target_optimal_decision_description != null &&
    result.baseline_optimal_decision_description != null &&
    result.target_optimal_decision_description !==
      result.baseline_optimal_decision_description;

  return (
    <>
      <div className="decision-hero">
        <span className="decision-hero-label">causal effect</span>
        <span className={"decision-hero-value effect " + (effect >= 0 ? "up" : "down")}>
          {effect >= 0 ? "+" : ""}
          {fmtNum(effect)}
        </span>
        {/* Which estimand this is decides what the number means, so it is
            stated rather than left to the trace. */}
        <span className="decision-hero-utility">
          {reoptimised
            ? "policy reoptimised under each condition"
            : `policy held at ${result.fixed_decision_description ?? result.fixed_decision}`}
        </span>
      </div>

      {reoptimised && (
        <div className={"decision-confidence " + (flipped ? "warn" : "ok")}>
          <span className="confidence-icon" aria-hidden="true">{flipped ? "⤳" : "="}</span>
          <span className="confidence-text">
            {flipped
              ? "The intervention changes which policy is best, not just its payoff."
              : "The best policy is the same under both conditions — the intervention moves the payoff, not the choice."}
          </span>
        </div>
      )}

      <div className="decision-chart">
        <div className="decision-caption">
          both arms sampled with the same seed, so the contrast is paired
        </div>
        {arms.map((arm) => (
          // Keyed on the Fragment: the grid's cells are siblings, so wrapping
          // them in an element would break the three-column alignment.
          <Fragment key={arm.key}>
            <span className="plot-gutter action-name" title={arm.label}>
              {arm.label}
            </span>
            <div className="plot-track action-track">
              <span
                className={"action-point " + (arm.key === "target" ? "best" : "alt")}
                style={{ left: `${pct(arm.value)}%` }}
                title={`expected utility ${fmtNum(arm.value)}`}
              />
            </div>
            <span className="plot-meta action-value">{fmtNum(arm.value)}</span>
            {arm.decision && (
              <>
                <span className="plot-gutter prob-label">chooses</span>
                <span className="decision-arm-choice">{arm.decision}</span>
                <span className="plot-meta" />
              </>
            )}
          </Fragment>
        ))}
      </div>
    </>
  );
}

// ── Posterior summary ───────────────────────────────────────────────────────

function summaryLine(summary: Record<string, unknown>): string {
  if (summary.distribution === "Beta-Bernoulli") {
    return `P(1) = ${fmtNum(summary.posterior_probability_mean as number, 3)}`;
  }
  if (summary.distribution === "Normal-Inverse-Gamma") {
    return (
      `mean ${fmtNum(summary.posterior_mean as number, 3)}, ` +
      `variance ${fmtNum(summary.expected_sampling_variance as number, 3)}`
    );
  }
  return "";
}

function PosteriorView({ posteriors }: { posteriors: PosteriorSummaries }) {
  const entries = Object.entries(posteriors);
  return (
    <>
      <div className="decision-hero">
        <span className="decision-hero-label">learned distributions</span>
        <span className="decision-hero-value">{entries.length} causes</span>
      </div>
      <dl className="posterior-list">
        {entries.map(([name, summary]) => (
          <div className="posterior-row" key={name}>
            <dt>{name.replace(/_/g, " ")}</dt>
            <dd>
              <span className="posterior-dist">{String(summary.distribution ?? "")}</span>
              <span className="posterior-moments">{summaryLine(summary)}</span>
            </dd>
          </div>
        ))}
      </dl>
    </>
  );
}

// ── Card ────────────────────────────────────────────────────────────────────

interface Props {
  decision: DecisionResult | null;
  posteriors: PosteriorSummaries | null;
  /** Supplies the full cause list, which lives only on the graph payload. */
  graph?: CausalGraph | null;
}

export function DecisionCard({ decision, posteriors, graph }: Props) {
  if (!decision) return null;

  const estimand = renderEstimand(decision);
  const split = causeSplit(graph);

  return (
    <section className="decision-card">
      <header className="decision-card-head">
        <span className="decision-card-title">Decision</span>
        <span className="decision-kind">
          {String(decision.query_type ?? "").replace(/_/g, " ")}
        </span>
      </header>

      {/* What was computed, before what came out of it. The narrated answer
          restates the result; nothing restates the question the engine was
          actually given, and for interventions that is where the meaning is. */}
      {estimand && (
        <div className="decision-estimand">
          <span className="estimand-label">estimand</span>
          <code className="estimand-formula">{estimand}</code>
        </div>
      )}

      {split && (
        <div className="decision-split">
          {split.intervened.length > 0 && (
            <span className="split-group intervened">
              <span className="split-label">forced</span>
              {split.intervened.map((name) => (
                <span className="split-chip" key={name}>{name.replace(/_/g, " ")}</span>
              ))}
            </span>
          )}
          {split.observed.length > 0 && (
            <span className="split-group observed">
              <span className="split-label">observed</span>
              {split.observed.map((name) => (
                <span className="split-chip" key={name}>{name.replace(/_/g, " ")}</span>
              ))}
            </span>
          )}
          <span className="split-group stochastic">
            <span className="split-label">integrated over</span>
            {split.stochastic.length ? (
              split.stochastic.map((name) => (
                <span className="split-chip" key={name}>{name.replace(/_/g, " ")}</span>
              ))
            ) : (
              <span className="split-none">nothing — every cause was pinned</span>
            )}
          </span>
        </div>
      )}

      {decision.query_type === "optimal_policy" && <PolicyView result={decision} />}
      {decision.query_type === "intervention_effect" && <InterventionView result={decision} />}
      {decision.query_type === "posterior_summary" && (
        <PosteriorView posteriors={decision.posterior_distributions ?? posteriors ?? {}} />
      )}

      {/* Standing disclosure, not decoration: the whole design rests on the
          model never having produced these numbers. */}
      <footer className="decision-provenance">
        computed by Pyro Monte Carlo — the language model did not calculate this
      </footer>
    </section>
  );
}
