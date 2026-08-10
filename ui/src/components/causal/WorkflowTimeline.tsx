import { memo, useEffect, useRef, useState } from "react";
import type { Stage } from "../../lib/stages";

/**
 * Live elapsed counter.
 *
 * Isolated in its own leaf with its own rAF loop so ticking never re-renders
 * the timeline tree, and throttled to ~10fps because a per-frame text update
 * is pure layout cost nobody can read.
 */
function Elapsed({ startedMs, endedMs, running }: {
  startedMs: number | null;
  endedMs: number | null;
  running: boolean;
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
  if (running) ms = now - base.current;
  else if (startedMs != null && endedMs != null) ms = endedMs - startedMs;
  else return null;

  if (ms < 0) ms = 0;
  return <span className="stage-time">{ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`}</span>;
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
  const expandable = stage.steps.length > 0;

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
            {stage.current && (
              <span className="stage-counter">{`step ${stage.current.index} of ${stage.current.total}`}</span>
            )}
            <Elapsed startedMs={stage.startedMs} endedMs={stage.endedMs} running={running} />
          </button>
        ) : (
          <div className="stage-head static">
            <span className="stage-caret" aria-hidden="true" />
            <span className="stage-label">{stage.label}</span>
            {stage.current && (
              <span className="stage-counter">{`step ${stage.current.index} of ${stage.current.total}`}</span>
            )}
            <Elapsed startedMs={stage.startedMs} endedMs={stage.endedMs} running={running} />
          </div>
        )}

        {expanded && expandable && (
          <ul className="stage-steps">
            {stage.steps.map((step, i) => (
              <li key={i}>{step}</li>
            ))}
          </ul>
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
  const [expanded, setExpanded] = useState<string | null>(null);

  const toggle = (id: string) => setExpanded((prev) => (prev === id ? null : id));

  return (
    <div className="workflow-timeline" role="status" aria-label="Causal pipeline progress">
      <ol className="stage-list">
        {visible.map((stage) => (
          <StageRow
            key={stage.id}
            stage={stage}
            expanded={expanded === stage.id}
            onToggle={toggle}
          />
        ))}
      </ol>
    </div>
  );
}
