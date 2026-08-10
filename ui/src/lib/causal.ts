import type { Report } from "../types";

/**
 * Whether a report has anything worth opening the causal panel for.
 *
 * Lives here rather than beside the panel so that asking the question does not
 * drag in the panel itself — CausalPanel pulls ReactFlow and dagre, and both
 * App and MessageList need this predicate during an ordinary render. Keeping it
 * dependency-free is what lets the panel be loaded lazily.
 */
export function hasCausalContent(report: Report): boolean {
  return (
    (report.causal_reasoning_steps || []).length > 0 ||
    !!report.causal_graph?.nodes?.length ||
    !!report.causal_decision
  );
}
