import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  Controls,
  Handle,
  Position,
  ReactFlow,
  type NodeProps,
  type NodeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  KIND_LABEL,
  buildEdges,
  edgeAppearanceKey,
  graphHeight,
  layoutNodes,
  topologyKey,
  type CausalNode,
  type CausalNodeData,
} from "../../lib/graph";
import type { CausalGraph as CausalGraphType } from "../../types";

/** Node status. Order matches the lifecycle, so the key reads as a sequence. */
const STATUS_LEGEND: [string, string][] = [
  ["pending", "queued"],
  ["active", "running"],
  ["done", "verified"],
  ["failed", "failed"],
];

/**
 * Link bands, generated from the same three classes `buildEdges` applies.
 * The previous key documented a dotted-grey "low confidence" edge the renderer
 * never produced, and called critical edges orange when they were violet.
 */
const EDGE_LEGEND: [string, string][] = [
  ["critical", "critical path"],
  ["firm", "supported"],
  ["soft", "weakly supported"],
];

/**
 * One component in the causal DAG.
 *
 * memo'd on its data: the executor loop re-serializes the whole graph on every
 * iteration, so without this every node re-renders on every frame.
 * `--hop` drives the ripple's transition-delay purely in CSS — no JS timers.
 */
const CausalNodeView = memo(function CausalNodeView({ data }: NodeProps) {
  const { label, kind, status, hop, onCritical, highlighted } = data as unknown as CausalNodeData;
  return (
    <div
      className={
        `dag-node ${status} kind-${kind}` +
        (onCritical ? " critical" : "") +
        (highlighted ? " highlighted" : "")
      }
      style={{ ["--hop" as string]: hop }}
      data-status={status}
    >
      <Handle type="target" position={Position.Top} isConnectable={false} />
      {/* Role is the durable semantic — a confounder is a different object from
          a mediator — so it is rendered, not hidden in a title attribute. */}
      <span className="dag-node-kind">{KIND_LABEL[kind]}</span>
      <span className="dag-node-label">{label}</span>
      <Handle type="source" position={Position.Bottom} isConnectable={false} />
    </div>
  );
});

const NODE_TYPES: NodeTypes = { causal: CausalNodeView };

interface Props {
  graph: CausalGraphType;
  /** Opens the drill-down drawer for a component. */
  onOpenNode?: (id: string, label: string) => void;
  highlightedId?: string | null;
}

export function CausalGraph({ graph, onOpenNode, highlightedId }: Props) {
  const key = topologyKey(graph);

  // Read through a ref so the memo below can depend on the topology key alone.
  // That is the whole point: layout must not recompute when only a status
  // changed, which is what every frame after the graph is built carries.
  const graphRef = useRef(graph);
  graphRef.current = graph;

  // Synchronous, so the container is correctly sized on the very first paint.
  // `key` is the only dependency by design: it is the topology fingerprint, and
  // depending on `graph` instead would re-run layout on every status frame —
  // exactly the jump this is here to prevent.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const base = useMemo(() => layoutNodes(graphRef.current), [key]);
  const [nodes, setNodes] = useState<CausalNode[]>(base);

  useEffect(() => setNodes(base), [base]);

  // Status-only update: patch data in place, preserving node identity so React
  // Flow keeps the same DOM element and CSS transitions it rather than
  // remounting.
  useEffect(() => {
    setNodes((prev) => {
      const byId = new Map((graph.nodes || []).map((n) => [n.id, n.status]));
      let changed = false;
      const next = prev.map((node) => {
        const status = byId.get(node.id);
        if (!status || status === node.data.status) return node;
        changed = true;
        return { ...node, data: { ...node.data, status } };
      });
      return changed ? next : prev;
    });
  }, [graph]);

  // Same reasoning as the layout memo above: `graph` is a fresh object on every
  // status frame, so depending on it rebuilt the whole edge array dozens of
  // times per run. The key covers everything buildEdges reads — endpoints,
  // confidence, and the critical path — and nothing it doesn't.
  const edgeKey = edgeAppearanceKey(graph);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const edges = useMemo(() => buildEdges(graphRef.current), [edgeKey]);
  const height = useMemo(() => graphHeight(base), [base]);

  // React Flow warns — and computes a useless first fitView — when it mounts
  // into a box with no width. In the side pane that is the first frame every
  // time, so hold the canvas until the container has actually been laid out.
  const boxRef = useRef<HTMLDivElement>(null);
  const [measured, setMeasured] = useState(false);
  useLayoutEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const check = () => setMeasured(el.clientWidth > 0 && el.clientHeight > 0);
    check();
    const observer = new ResizeObserver(check);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const rendered = useMemo(
    () =>
      nodes.map((node) => {
        const data = node.data as unknown as CausalNodeData;
        const highlighted = highlightedId === node.id || highlightedId === data.label;
        return {
          ...node,
          // Selectable so React Flow gives the node pointer events at all: it
          // sets pointer-events:none inline unless something needs them, and an
          // onClick inside the custom node is invisible to that decision.
          selectable: true,
          focusable: true,
          ariaLabel: `${data.label}. ${KIND_LABEL[data.kind]}, ${data.status}.` +
            (data.onCritical ? " On the critical path." : ""),
          // Reuse the existing data object when the highlight hasn't changed.
          // CausalNodeView is memo'd on shallow prop equality, so handing it a
          // freshly-built `data` on every frame defeated that memo completely —
          // every node re-rendered on every status update regardless.
          data: data.highlighted === highlighted ? node.data : { ...node.data, highlighted },
        };
      }),
    [nodes, highlightedId],
  );

  // Mid-run the ledger is still partial, so the streaming graph is rendered
  // without a click handler. Don't advertise an affordance that isn't there.
  const interactive = !!onOpenNode;

  // React Flow gives focused nodes no activation key of their own, so Enter and
  // Space are handled here against whichever node currently holds focus.
  const onKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      if (!onOpenNode || (event.key !== "Enter" && event.key !== " ")) return;
      const active = (event.target as HTMLElement).closest<HTMLElement>(".react-flow__node");
      const id = active?.dataset.id;
      if (!id) return;
      event.preventDefault();
      const node = nodes.find((n) => n.id === id);
      if (node) onOpenNode(node.id, (node.data as unknown as CausalNodeData).label);
    },
    [nodes, onOpenNode],
  );

  return (
    <div className="graph-card">
      <div
        ref={boxRef}
        className={"causal-graph-container" + (interactive ? " interactive" : "")}
        style={{ height }}
        onKeyDown={onKeyDown}
      >
        {measured && (
          <ReactFlow
            nodes={rendered}
            edges={edges}
            nodeTypes={NODE_TYPES}
            fitView
            fitViewOptions={{ padding: 0.14 }}
            onNodeClick={(_, node) =>
              onOpenNode?.(node.id, (node.data as CausalNodeData).label)
            }
            nodesDraggable={false}
            nodesConnectable={false}
            elementsSelectable
            nodesFocusable
            // Edges carry no action and no accessible name; as tab stops they
            // were seven dead keystrokes ahead of the first real one.
            edgesFocusable={false}
            // Wheel stays page scroll — the canvas sits inside a scrolling
            // pane, so hijacking it would trap the reader. Zoom is explicit,
            // via the controls below.
            panOnScroll={false}
            zoomOnScroll={false}
            preventScrolling={false}
            minZoom={0.15}
            maxZoom={1.8}
            proOptions={{ hideAttribution: false }}
          >
            <Controls showInteractive={false} position="bottom-right" />
          </ReactFlow>
        )}
      </div>

      <div className="graph-legend">
        <span className="legend-group-label">nodes</span>
        {STATUS_LEGEND.map(([cls, label]) => (
          <span key={cls}>
            <i className={"lg-dot " + cls} />
            {label}
          </span>
        ))}
        <span className="legend-divider" aria-hidden="true">|</span>
        <span className="legend-group-label">links</span>
        {EDGE_LEGEND.map(([cls, label]) => (
          <span key={cls}>
            <i className={"lg-line " + cls} />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}
