import type { Report } from "../types";
import type { Stage } from "./stages";

/**
 * Serialize one run to a single auditable file.
 *
 * The pane already holds everything needed to reconstruct a turn — the model
 * graph, the decision result with its full per-action evaluation, and the trace
 * — but it was only readable by scrolling. A file makes it attachable to a
 * ticket and diffable against another run.
 *
 * The decision result carries its own `num_samples`, and the engine is seeded,
 * so two exports of the same question are byte-comparable.
 */
export function buildRunExport(report: Report, stages?: Stage[]): string {
  return JSON.stringify(
    {
      exported_at: new Date().toISOString(),
      run_id: report.run_id ?? null,
      response: report.response,
      total_token_count: report.total_token_count,
      status: report.causal_status ?? null,
      graph: report.causal_graph,
      decision: report.causal_decision,
      posteriors: report.causal_posteriors,
      reasoning_trace: report.causal_reasoning_steps ?? [],
      // Client-side timings; absent on a turn replayed from history.
      stage_timings: (stages ?? []).map((s) => ({
        id: s.id,
        label: s.label,
        status: s.status,
        started_ms: s.startedMs,
        ended_ms: s.endedMs,
      })),
    },
    null,
    2,
  );
}

export function downloadRun(report: Report, stages?: Stage[]): void {
  const blob = new Blob([buildRunExport(report, stages)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `causaltrace-run-${report.run_id ?? Date.now().toString(36)}.json`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
