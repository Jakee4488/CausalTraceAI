import { memo, useEffect, useRef, useState } from "react";
import { stageDurationMs, type Stage } from "../../lib/stages";
import type { NodeRun } from "../../types";

function fmtMs(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;
}

/**
 * One graph node's execution, when a stage ran more than one.
 *
 * A stage that ran exactly one node needs no row of its own: the head already
 * carries that node's name, duration and spend, and repeating them produced two
 * near-identical lines per stage. This is for the case where the split actually
 * has something to split — several nodes under one stage.
 */
const NodeRow = memo(function NodeRow({ run, share }: { run: NodeRun; share: number }) {
  const tokens = run.usage?.total_tokens;
  return (
    <li className="node-row">
      <span className="node-name">{run.node}</span>
      {/* Width is the node's share of the stage, so the bar reads as "where the
          time went" without needing a second axis. */}
      <span className="node-bar" aria-hidden="true">
        <span className="node-bar-fill" style={{ width: `${Math.max(share * 100, 2)}%` }} />
      </span>
      {tokens != null && (
        <span
          className="node-tokens"
          title={
            `${run.usage?.prompt_tokens ?? "?"} prompt + ` +
            `${run.usage?.completion_tokens ?? "?"} completion`
          }
        >
          {tokens.toLocaleString()} tok
        </span>
      )}
      <span className="node-time">{fmtMs(run.duration_ms)}</span>
    </li>
  );
});

/**
 * Live elapsed counter.
 *
 * Isolated in its own leaf with its own rAF loop so ticking never re-renders
 * the timeline tree, and throttled to ~10fps because a per-frame text update
 * is pure layout cost nobody can read.
 */
function Elapsed({ startedMs, endedMs, running, overrideMs }: {
  startedMs: number | null;
  endedMs: number | null;
  running: boolean;
  /** Measured duration, when the proxy reported per-node timings. */
  overrideMs?: number | null;
}) {
  const [now, setNow] = useState(() => performance.now());
  const base = useRef(performance.now());

  useEffect(() => {
    if (!running) return;
    base.current = performance.now();
    let raf = 0;
    let last = 0;
    const tick = (t: number) => {
      if (t - last > 100) {
        setNow(t);
        last = t;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [running]);

  let ms: number;
  if (overrideMs != null) ms = overrideMs;
  else if (running) ms = now - base.current;
  else if (startedMs != null && endedMs != null) ms = endedMs - startedMs;
  else return null;

  if (ms < 0) ms = 0;
  return <span className="stage-time">{fmtMs(ms)}</span>;
}

const StageRow = memo(function StageRow({
  stage,
  expanded,
  onToggle,
}: {
  stage: Stage;
  expanded: boolean;
  onToggle: (id: string) => void;
}) {
  const running = stage.status === "active";
  const expandable = stage.steps.length > 0 || stage.nodes.length > 1;
  // Prefer the measured sum over frame-arrival arithmetic once the stage is
  // done; while it is running the live counter is still the honest number.
  const measured = running ? null : stageDurationMs(stage);
  const stageTotal = stage.nodes.reduce((sum, run) => sum + (run.duration_ms || 0), 0);
  // Surfaced on the head rather than only inside the expansion: the per-call
  // split is the one fact here that no other row duplicates, and the summed
  // badge in the header cannot show it.
  const stageTokens = stage.nodes.reduce(
    (sum, run) => sum + (run.usage?.total_tokens ?? 0),
    0,
  );
  const soleNode = stage.nodes.length === 1 ? stage.nodes[0] : null;

  return (
    <li className={"stage-row " + stage.status} data-stage={stage.id}>
      <span className="stage-dot" aria-hidden="true" />
      <div className="stage-body">
        {/* A row with nothing recorded under it is not a control, so it does
            not render as one — the old version made every row look clickable
            and then opened a full-height dialog holding a single line. */}
        {expandable ? (
          <button
            type="button"
            className="stage-head"
            aria-expanded={expanded}
            onClick={() => onToggle(stage.id)}
          >
            <span className="stage-caret" aria-hidden="true">{expanded ? "▾" : "▸"}</span>
            <span className="stage-label">{stage.label}</span>
            {/* The exact node id, which is also the string to grep for in the
                agent's logs — and for the compute stage, the only place the UI
                says which of the three routes actually ran. */}
            {soleNode && <span className="stage-node">{soleNode.node}</span>}
            {stage.current && (
              <span className="stage-counter">{`step ${stage.current.index} of ${stage.current.total}`}</span>
            )}
            {stageTokens > 0 && (
              <span className="stage-tokens" title="Tokens this stage's Gemini call spent">
                {stageTokens.toLocaleString()} tok
              </span>
            )}
            <Elapsed
              startedMs={stage.startedMs}
              endedMs={stage.endedMs}
              running={running}
              overrideMs={measured}
            />
          </button>
        ) : (
          <div className="stage-head static">
            <span className="stage-caret" aria-hidden="true" />
            <span className="stage-label">{stage.label}</span>
            {/* The exact node id, which is also the string to grep for in the
                agent's logs — and for the compute stage, the only place the UI
                says which of the three routes actually ran. */}
            {soleNode && <span className="stage-node">{soleNode.node}</span>}
            {stage.current && (
              <span className="stage-counter">{`step ${stage.current.index} of ${stage.current.total}`}</span>
            )}
            {stageTokens > 0 && (
              <span className="stage-tokens" title="Tokens this stage's Gemini call spent">
                {stageTokens.toLocaleString()} tok
              </span>
            )}
            <Elapsed
              startedMs={stage.startedMs}
              endedMs={stage.endedMs}
              running={running}
              overrideMs={measured}
            />
          </div>
        )}

        {expanded && expandable && (
          <>
            {stage.nodes.length > 1 && (
              <ul className="stage-nodes">
                {stage.nodes.map((run) => (
                  <NodeRow
                    key={`${run.node}-${run.at_ms}`}
                    run={run}
                    share={stageTotal > 0 ? (run.duration_ms || 0) / stageTotal : 0}
                  />
                ))}
              </ul>
            )}
            {stage.steps.length > 0 && (
              <ul className="stage-steps">
                {stage.steps.map((step, i) => (
                  <li key={i}>{step}</li>
                ))}
              </ul>
            )}
          </>
        )}
      </div>
    </li>
  );
});

interface Props {
  stages: Stage[];
  /** Hide stages that never ran, once the run is over. */
  compact?: boolean;
}

/**
 * The pipeline for a causal turn: stages light up in order, with each stage's
 * trace lines available underneath it.
 *
 * Laid out vertically. The horizontal form pinned every stage to a 58px column
 * inside a ~379px pane, which both ellipsised all eight labels and pushed the
 * last three — including Synthesize, the stage that wrote the answer — outside
 * the clip box, with the scrollbar suppressed in CSS.
 */
export function WorkflowTimeline({ stages, compact = false }: Props) {
  const visible = compact ? stages.filter((s) => s.status !== "skipped") : stages;
  // A set, not a single id: this became an execution record rather than a
  // disclosure widget, and an accordion made "read the whole trace" a
  // four-click exercise where three of the clicks close what you just opened.
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(() => new Set());

  const toggle = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (!next.delete(id)) next.add(id);
      return next;
    });

  return (
    <div className="workflow-timeline" role="status" aria-label="Causal pipeline progress">
      <ol className="stage-list">
        {visible.map((stage) => (
          <StageRow
            key={stage.id}
            stage={stage}
            expanded={expanded.has(stage.id)}
            onToggle={toggle}
          />
        ))}
      </ol>
    </div>
  );
}
