import type { CausalGraph, NodeRun } from "../types";

export type StageStatus = "pending" | "active" | "done" | "failed" | "skipped";

export interface Stage {
  id: string;
  label: string;
  status: StageStatus;
  steps: string[];
  /** The graph nodes that actually ran under this stage, in order. */
  nodes: NodeRun[];
  startedMs: number | null;
  endedMs: number | null;
  current?: { index: number; total: number };
}

export interface ProgressFrame {
  stage?: string | null;
  message?: string | null;
  step?: string | null;
  steps?: string[];
  nodes?: NodeRun[];
  elapsed_ms?: number;
  index?: number;
  total?: number;
  graph?: CausalGraph | null;
}

/**
 * A stage's duration from its nodes, when the proxy reported them.
 *
 * Better than the frame-arrival arithmetic below, which measures "time until
 * the next stage started" and so charges each stage for the gap after it.
 * Returns null when there is no node data — mock mode and replayed history
 * predate it — and the caller keeps the old number.
 */
export function stageDurationMs(stage: Stage): number | null {
  if (!stage.nodes.length) return null;
  return stage.nodes.reduce((total, run) => total + (run.duration_ms || 0), 0);
}

// One entry per stage the proxy can report (proxy/main.py STAGE_BY_NODE). The
// three compute nodes — optimal_policy / intervention_effect /
// posterior_summary — are alternatives, only one of which runs per turn, so
// they share a single "compute" stage rather than leaving two rows permanently
// grey and implying work that was skipped.
const STAGE_ORDER: Array<Pick<Stage, "id" | "label">> = [
  { id: "interpret", label: "Interpret query" },
  { id: "validate", label: "Validate" },
  { id: "compute", label: "Simulate worlds" },
  { id: "explain", label: "Explain" },
];

// Raw LangGraph node names, accepted as aliases so a frame that reached the UI
// unmapped still lands on the right row instead of vanishing.
const ALIASES: Record<string, string> = {
  interpret_query: "interpret",
  validate_query: "validate",
  optimal_policy: "compute",
  intervention_effect: "compute",
  posterior_summary: "compute",
  explain_result: "explain",
  clarification: "explain",
  error: "explain",
};

function normalizeStage(id: string | null | undefined): string | null {
  if (!id) return null;
  const key = id.toLowerCase().trim();
  return ALIASES[key] ?? key;
}

function cloneStage(stage: Stage): Stage {
  return {
    ...stage,
    steps: [...stage.steps],
    nodes: [...stage.nodes],
    current: stage.current ? { ...stage.current } : undefined,
  };
}

export function initialStages(): Stage[] {
  return STAGE_ORDER.map((stage) => ({
    ...stage,
    // Every stage runs on every turn now — the old "web" pre-skip existed for
    // a branch that was off unless a toggle was set, and there is no such
    // branch here.
    status: "pending",
    steps: [],
    nodes: [],
    startedMs: null,
    endedMs: null,
  }));
}

export function applyProgress(prev: Stage[], frame: ProgressFrame): Stage[] {
  const stages = prev.map(cloneStage);
  const stageId = normalizeStage(frame.stage);
  const elapsed = frame.elapsed_ms ?? 0;

  // Routed by each run's own stage rather than the frame's: one chunk can carry
  // several nodes, and the frame reports only the last one it saw.
  for (const run of frame.nodes ?? []) {
    const target = normalizeStage(run.stage ?? frame.stage);
    const owner = stages.find((stage) => stage.id === target);
    if (owner && !owner.nodes.some((seen) => seen.at_ms === run.at_ms && seen.node === run.node)) {
      owner.nodes.push(run);
    }
  }

  if (!stageId) return stages;
  const idx = stages.findIndex((stage) => stage.id === stageId);
  if (idx < 0) return stages;

  for (let i = 0; i < idx; i += 1) {
    if (stages[i].status === "pending" || stages[i].status === "active") {
      stages[i].status = "done";
      if (stages[i].startedMs == null) stages[i].startedMs = 0;
      stages[i].endedMs = elapsed;
    }
  }

  const stage = stages[idx];
  if (stage.status === "pending" || stage.status === "skipped") {
    stage.status = "active";
    stage.startedMs = elapsed;
    stage.endedMs = null;
  }

  const line = frame.step ?? frame.message;
  if (line && (stage.steps.length === 0 || stage.steps[stage.steps.length - 1] !== line)) {
    stage.steps.push(line);
  }
  
  if (frame.steps && frame.steps.length > 0) {
    stage.steps.push(...frame.steps);
  }

  if (typeof frame.index === "number" && typeof frame.total === "number") {
    stage.current = { index: frame.index, total: frame.total };
  }

  return stages;
}

/**
 * Rebuild a finished timeline from a persisted node trace.
 *
 * Live runs capture their stages as they go; a turn replayed out of history has
 * none, and used to show no timeline at all. The node trace persists with the
 * rest of the causal payload, which is enough to reconstruct what ran and how
 * long each node took — everything except the trace lines, which the report
 * carries separately.
 */
export function stagesFromNodeTrace(trace: NodeRun[] | undefined): Stage[] | null {
  if (!trace?.length) return null;
  const stages = initialStages();
  for (const run of trace) {
    const owner = stages.find((stage) => stage.id === normalizeStage(run.stage ?? run.node));
    if (owner) owner.nodes.push(run);
  }
  return stages.map((stage) => ({
    ...stage,
    status: stage.nodes.length ? "done" : "skipped",
  }));
}

export function finalizeStages(prev: Stage[], elapsedMs: number, failed: boolean): Stage[] {
  return prev.map((stage) => {
    const next = cloneStage(stage);
    if (next.status === "active") {
      next.status = failed ? "failed" : "done";
      next.endedMs = elapsedMs;
    } else if (next.status === "pending") {
      next.status = "skipped";
    }
    return next;
  });
}
