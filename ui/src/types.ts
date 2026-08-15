// Shapes the proxy returns. Names mirror the `causal_*` channel names in
// src/causal/state.py, because a persisted turn replays through exactly the
// same fields as a live one.

export type NodeStatus =
  | "pending" | "active" | "done" | "failed" | "invalidated" | "replanned";

export interface GraphNode {
  id: string;
  label: string;
  kind: string;
  status: NodeStatus;
  /** Why this component exists. Capped at 200 chars by the model validator. */
  description?: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  relation: string;
  confidence: number;
  /** Why this link was asserted. Capped at 200 chars by the model validator. */
  rationale?: string;
}

/** to_ui_graph() output — see src/causal/graph_engine.py. */
export interface CausalGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  critical_path: string[];
  version: number;
}

/** One action's evaluation — src/causal/decision.py evaluate_all_decisions. */
export interface ActionEvaluation {
  decision: number;
  decision_description: string;
  expected_utility: number;
  /** Spread of U across worlds — the *risk*, not the estimation error. */
  utility_standard_deviation: number;
  /** std / sqrt(n) — precision of the mean estimate. */
  monte_carlo_standard_error: number;
  /** 95% interval on the mean, not on U itself. */
  mean_lower_95: number;
  mean_upper_95: number;
}

/** Posterior summary per cause — shape varies by conjugate family. */
export type PosteriorSummaries = Record<string, Record<string, unknown>>;

/** Whatever the engine computed this turn. One union rather than three types:
 *  the proxy forwards a single `causal_decision` key and `query_type` says
 *  which arm of the union it holds. */
export interface DecisionResult {
  query_type: "optimal_policy" | "intervention_effect" | "posterior_summary";
  num_samples?: number;
  context?: Record<string, number>;
  interventions?: Record<string, number>;

  // optimal_policy
  optimal_decision?: number;
  optimal_decision_description?: string;
  optimal_expected_utility?: number;
  action_evaluations?: ActionEvaluation[];
  probability_each_action_is_best?: Record<string, number>;

  // intervention_effect
  estimand?: "fixed_policy_effect" | "reoptimised_policy_effect";
  intervention_variable?: string;
  intervention_value?: number;
  baseline_value?: number | null;
  target_expected_utility?: number;
  baseline_expected_utility?: number;
  causal_effect?: number;
  fixed_decision?: number;
  fixed_decision_description?: string;
  target_optimal_decision_description?: string;
  baseline_optimal_decision_description?: string;

  // posterior_summary
  posterior_distributions?: PosteriorSummaries;
}

/** Per-Gemini-call token split. Absent on the nodes that spend none. */
export interface NodeUsage {
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
}

/**
 * One LangGraph node execution, as timed by the proxy.
 *
 * `duration_ms` is the gap between this node's chunk and the previous one, so
 * it includes transport — the agent's own structured logs carry the measured
 * figure. Close enough to find the slow node, not close enough to quote.
 */
export interface NodeRun {
  node: string;
  stage: string | null;
  duration_ms: number;
  at_ms: number;
  usage?: NodeUsage;
}

/** The terminal `done` frame: everything needed to render a finished turn. */
export interface Report {
  status: string;
  response: string;
  /** Correlation id joining this answer to its proxy log line and trace. */
  run_id?: string;
  total_token_count: number;
  causal_reasoning_steps: string[];
  causal_graph: CausalGraph | null;
  causal_status: { phase?: string } | null;
  /** The engine's numerical result — see DecisionResult. */
  causal_decision: DecisionResult | null;
  /** Only populated by a posterior_summary query. */
  causal_posteriors: PosteriorSummaries | null;
  /** Every node the graph ran, in order. Absent on mock and older turns. */
  causal_node_trace?: NodeRun[];
}

export interface ChatMessage {
  key: string;
  role: "user" | "ai" | "error" | "greeting";
  content: string;
  attachments?: string[];
  report?: Report;
  /** Captured when the run finished; absent on replayed history. */
  stages?: import("./lib/stages").Stage[];
}

export interface Attachment {
  localId: string;
  id: string | null;
  name: string;
  size: number;
  status: "uploading" | "done" | "error";
  error?: string;
}

export interface Conversation {
  chat_id: string;
  title: string;
  total_tokens: number;
  updated_at: string;
}
