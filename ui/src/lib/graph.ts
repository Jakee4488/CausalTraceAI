import dagre from "@dagrejs/dagre";
import type { Edge, Node } from "@xyflow/react";
import type { CausalGraph, NodeStatus } from "../types";

// Layout box, not the painted box: it carries the node plus the gap dagre
// should leave around it. The painted node is sized by .dag-node in styles.css.
const NODE_WIDTH = 200;
const NODE_HEIGHT = 72;
const RANKSEP = 52;
const NODESEP = 44;

/** The graph is a bounded, zoomable canvas — not a block that grows forever. */
const MIN_CANVAS = 300;
const MAX_CANVAS = 560;

/** Roles a node can play in the DAG. This is the durable semantic. */
export type NodeKind = "cause" | "mediator" | "confounder" | "outcome" | "other";

const KIND_ALIASES: Record<string, NodeKind> = {
  cause: "cause",
  treatment: "cause",
  exposure: "cause",
  mediator: "mediator",
  mechanism: "mediator",
  step: "mediator",
  confounder: "confounder",
  collider: "confounder",
  latent: "confounder",
  outcome: "outcome",
  effect: "outcome",
  result: "outcome",
};

export function normalizeKind(kind: string | null | undefined): NodeKind {
  return KIND_ALIASES[String(kind ?? "").toLowerCase().trim()] ?? "other";
}

/** Short role tag rendered inside the node, so kind is legible without hover. */
export const KIND_LABEL: Record<NodeKind, string> = {
  cause: "cause",
  mediator: "mediator",
  confounder: "confounder",
  outcome: "outcome",
  other: "component",
};

export interface CausalNodeData {
  [key: string]: unknown;
  label: string;
  kind: NodeKind;
  status: NodeStatus;
  hop: number;
  onCritical: boolean;
  highlighted?: boolean;
  /** The decomposer's "why this component exists". */
  description?: string;
}

export type CausalNode = Node<CausalNodeData, "causal">;

/** Below this an edge is drawn as "soft" — the claim is weakly supported. */
const SOFT_CONFIDENCE = 0.45;

function criticalHopMap(graph: CausalGraph): Map<string, number> {
  const hops = new Map<string, number>();
  for (const [index, id] of (graph.critical_path || []).entries()) hops.set(id, index);
  return hops;
}

export function topologyKey(graph: CausalGraph): string {
  const nodes = (graph.nodes || []).map((node) => node.id).sort().join("|");
  const edges = (graph.edges || [])
    .map((edge) => `${edge.source}->${edge.target}`)
    .sort()
    .join("|");
  return `${nodes}::${edges}`;
}

/**
 * Fingerprint of everything `buildEdges` actually reads.
 *
 * Deliberately wider than `topologyKey`: an edge's appearance also depends on
 * its confidence band and on the critical path, and neither is part of the
 * topology. Memoising edges on `topologyKey` alone would leave an edge frozen
 * when only its confidence moved.
 */
export function edgeAppearanceKey(graph: CausalGraph): string {
  const edges = (graph.edges || [])
    .map((edge) => `${edge.source}->${edge.target}@${edge.confidence}`)
    .sort()
    .join("|");
  return `${edges}::${(graph.critical_path || []).join(">")}`;
}

export function layoutNodes(graph: CausalGraph): CausalNode[] {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "TB", ranksep: RANKSEP, nodesep: NODESEP, marginx: 24, marginy: 24 });
  g.setDefaultEdgeLabel(() => ({}));

  const hops = criticalHopMap(graph);
  for (const node of graph.nodes || []) {
    g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  }
  for (const edge of graph.edges || []) g.setEdge(edge.source, edge.target);

  dagre.layout(g);

  return (graph.nodes || []).map((node) => {
    const positioned = g.node(node.id) as { x: number; y: number } | undefined;
    const hop = hops.get(node.id) ?? 0;
    return {
      id: node.id,
      type: "causal",
      position: {
        x: (positioned?.x ?? 0) - NODE_WIDTH / 2,
        y: (positioned?.y ?? 0) - NODE_HEIGHT / 2,
      },
      data: {
        label: node.label,
        kind: normalizeKind(node.kind),
        status: node.status,
        hop,
        onCritical: hops.has(node.id),
        description: node.description,
      },
    } satisfies CausalNode;
  });
}

export function buildEdges(graph: CausalGraph): Edge[] {
  const criticalSet = new Set<string>();
  const path = graph.critical_path || [];
  for (let i = 0; i < path.length - 1; i += 1) {
    criticalSet.add(`${path[i]}->${path[i + 1]}`);
  }

  return (graph.edges || []).map((edge, index) => {
    const key = `${edge.source}->${edge.target}`;
    const critical = criticalSet.has(key);
    const soft = edge.confidence < SOFT_CONFIDENCE;
    const band = critical ? "critical" : soft ? "soft" : "firm";

    return {
      id: `${key}-${index}`,
      source: edge.source,
      target: edge.target,
      // Long-range edges (a confounder pointing at the outcome) were drawn as
      // straight lines through every node box between the two ends. smoothstep
      // routes around them.
      type: "smoothstep",
      animated: critical,
      // Verb only. `relation (confidence%)` needed ~110px in a ~400px column,
      // so two labels landing in the same horizontal band was the default
      // rather than the edge case. Confidence now reads off the stroke, which
      // had nothing to say before.
      label: edge.relation,
      // The decomposer's justification for the link, shown on hover. It was
      // captured by the model and dropped in to_ui_graph until now.
      ariaLabel: edge.rationale
        ? `${edge.relation}: ${edge.rationale}`
        : `${edge.source} ${edge.relation} ${edge.target}`,
      data: { rationale: edge.rationale ?? "", confidence: edge.confidence },
      // Colour, dash and label fill all come from styles.css via these classes,
      // so both themes apply and the legend can be generated from the same
      // three bands the renderer actually uses.
      className: `causal-edge ${band}`,
      // The critical path has to out-weigh everything else on the canvas; React
      // Flow's `animated` dash pattern thins it optically, so it needs the
      // width back.
      style: { strokeWidth: critical ? 3.4 : 1.2 + edge.confidence * 1.6 },
      labelBgPadding: [6, 4] as [number, number],
      labelBgBorderRadius: 4,
      markerEnd: { type: "arrowclosed" },
    } satisfies Edge;
  });
}

/**
 * Height of the graph canvas.
 *
 * Bounded on purpose. Deriving this from the tallest node made a seven-node
 * chain 1072px tall inside a 945px viewport, which is what pushed the estimand
 * card past two screens of scroll. The canvas now fits the pane and the graph
 * is navigated by zoom, like every other DAG viewer.
 */
export function graphHeight(nodes: CausalNode[]): number {
  if (nodes.length === 0) return MIN_CANVAS;
  const maxY = Math.max(...nodes.map((node) => node.position.y + NODE_HEIGHT));
  return Math.min(Math.max(MIN_CANVAS, Math.ceil(maxY + 32)), MAX_CANVAS);
}
