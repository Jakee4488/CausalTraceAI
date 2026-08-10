// Causal reasoning panel: timeline, model graph, decision result, trace.

import { useCallback, useEffect, useState } from "react";
import { CausalGraph } from "./CausalGraph";
import { DecisionCard } from "./DecisionCard";
import { WorkflowTimeline } from "./WorkflowTimeline";
import { downloadRun } from "../../lib/export";
import type { Stage } from "../../lib/stages";
import type { Report, CausalGraph as CausalGraphType } from "../../types";

const STEP_TAG_RE = /^\[([a-z_ -]+)\]\s*(.*)$/i;

function StepLine({ step }: { step: string }) {
  const match = STEP_TAG_RE.exec(String(step));
  if (!match) return <li>{String(step)}</li>;
  const tag = match[1].toLowerCase();
  let cls = "step-tag";
  if (tag === "decision") cls += " ok";
  if (tag === "error") cls += " fail";
  return (
    <li>
      <span className={cls}>{match[1]}</span>
      {match[2]}
    </li>
  );
}

interface Props {
  report: Report;
  stages?: Stage[];
  liveGraph?: CausalGraphType | null;
  /** Rendered into the pane's fixed header, so it never scrolls away. */
  onClose?: () => void;
}

export function CausalPanel({ report, stages, liveGraph, onClose }: Props) {
  const steps = report.causal_reasoning_steps || [];
  const graph = liveGraph || report.causal_graph;
  const hasGraph = !!(graph && graph.nodes && graph.nodes.length);
  const phase = report.causal_status?.phase;
  // Mid-run App passes a placeholder report and drives the panel from
  // liveGraph, so there is nothing worth serialising yet.
  const finished = !!report.response;
  const [highlighted, setHighlighted] = useState<string | null>(null);
  // The node drill-down drawer went with the change ledger it read from: this
  // pipeline has no per-step record to drill into. Clicking a node still
  // highlights it, which is what the answer's [Node: …] citations need.
  const openNode = useCallback((id: string) => setHighlighted(id), []);

  useEffect(() => {
    const handleHighlight = (e: Event) => {
      const customEvent = e as CustomEvent<{ id: string | null }>;
      const targetIdOrLabel = customEvent.detail.id;
      setHighlighted(targetIdOrLabel);

      if (targetIdOrLabel && graph && graph.nodes) {
        const targetNode = graph.nodes.find(
          (n) => n.id === targetIdOrLabel || n.label === targetIdOrLabel,
        );
        if (targetNode) openNode(targetNode.id);
      }
    };
    window.addEventListener("highlight-node", handleHighlight);
    return () => window.removeEventListener("highlight-node", handleHighlight);
  }, [graph, openNode]);

  return (
    <div className="causal-panel">
      {/* Fixed header. The close control used to be absolutely positioned
          inside the scroll container, so it sat at y=-97 the moment you
          scrolled to anything worth reading. */}
      <div className="causal-head">
        <span className="causal-head-title">
          <span aria-hidden="true">⚯</span> Causal reasoning
        </span>
        {phase && <span className="phase-badge">{String(phase).replace(/_/g, " ")}</span>}
        {/* An exportable run is what turns the pane from something you read
            into something you can attach to a ticket or diff. Disabled until
            the run lands — mid-flight the panel is driven by the live graph and
            an export would be a near-empty file. */}
        <button
          className="export-run-btn"
          onClick={() => downloadRun(report, stages)}
          disabled={!finished}
          title={finished ? "Download this run as JSON" : "Available once the run finishes"}
        >
          ↓ Export
        </button>
        {onClose && (
          <button className="close-pane-btn" onClick={onClose} aria-label="Close causal panel">
            ✕
          </button>
        )}
      </div>

      {report.run_id && (
        <div className="run-id-strip" title="Correlation id: joins this answer to its server log line and trace">
          <span className="run-id-label">run</span>
          <code>{report.run_id}</code>
        </div>
      )}

      <div className="causal-panel-scroll">
        {stages && (
          <div className="pane-timeline">
            <WorkflowTimeline stages={stages} />
          </div>
        )}

        {hasGraph && (
          <div className="graph-container">
            <CausalGraph graph={graph} onOpenNode={openNode} highlightedId={highlighted} />
          </div>
        )}

        <DecisionCard
          decision={report.causal_decision}
          posteriors={report.causal_posteriors}
        />

        {steps.length > 0 && (
          <details className="causal-steps-details" open>
            <summary>Causal Reasoning Trace</summary>
            <ul className="causal-steps">
              {steps.map((step, i) => (
                <StepLine key={i} step={step} />
              ))}
            </ul>
          </details>
        )}
      </div>

    </div>
  );
}

/** Whether a report has anything causal worth showing a panel for. */
// Moved to ../../lib/causal so callers can ask the question without importing
// this module (and with it ReactFlow + dagre). Re-exported for convenience.
export { hasCausalContent } from "../../lib/causal";
