import { memo, useEffect, useLayoutEffect, useMemo, useRef } from "react";
import { enhanceMarkdown, renderMarkdown } from "../lib/markdown";
import { hasCausalContent } from "../lib/causal";
import type { Stage } from "../lib/stages";
import type { CausalGraph as CausalGraphType, ChatMessage, Report } from "../types";

function Markdown({ text }: { text: string }) {
  const ref = useRef<HTMLDivElement>(null);
  // marked + DOMPurify on every render was costing a full re-parse of every
  // message in the transcript on every progress frame — of which a causal run
  // emits dozens. The output depends only on the text.
  const html = useMemo(() => renderMarkdown(text), [text]);
  // Citation buttons are built here, over the mounted DOM, so re-run whenever
  // the rendered HTML changes.
  useEffect(() => enhanceMarkdown(ref.current), [html]);

  const handleClick = (e: React.MouseEvent) => {
    const target = (e.target as HTMLElement).closest(".node-citation");
    const label = target?.getAttribute("data-node");
    if (label) {
      window.dispatchEvent(new CustomEvent("highlight-node", { detail: { id: label } }));
    }
  };

  return <div className="md-content" ref={ref} onClick={handleClick} dangerouslySetInnerHTML={{ __html: html }} />;
}

/**
 * What the details button is worth opening for.
 *
 * The button used to say only "View Causal Details", which asked the reader to
 * spend a click to find out whether there was anything behind it. These are the
 * three facts the panel leads with anyway.
 */
function causalSummary(report: Report): string[] {
  const bits: string[] = [];
  const decision = report.causal_decision;

  if (decision?.query_type === "optimal_policy") {
    if (decision.optimal_decision_description) {
      bits.push(decision.optimal_decision_description);
    }
    // How often the winner actually won says more than its average payoff: an
    // action can lead on expected utility while winning a minority of worlds.
    const best = decision.probability_each_action_is_best?.[String(decision.optimal_decision)];
    if (best != null) bits.push(`best in ${(best * 100).toFixed(0)}% of worlds`);
  } else if (decision?.query_type === "intervention_effect") {
    const effect = decision.causal_effect;
    if (effect != null) {
      bits.push(`effect ${effect >= 0 ? "+" : ""}${effect.toFixed(2)}`);
    }
  } else if (decision?.query_type === "posterior_summary") {
    bits.push("posterior summary");
  }

  // How much of the world stayed uncertain — the honest counterweight to a
  // confident-looking recommendation.
  const stochastic = (report.causal_graph?.nodes || []).filter(
    (n) => n.kind === "input" && n.status === "pending",
  ).length;
  if (stochastic) bits.push(`${stochastic} cause${stochastic === 1 ? "" : "s"} uncertain`);

  return bits;
}

// Memoised: MessageList re-renders on every progress frame (its `stages` prop
// is a fresh array each time), and without this every settled message in the
// transcript re-rendered along with it.
const AiMessage = memo(function AiMessage(
  { message, onSelect }: { message: ChatMessage; onSelect?: (msg: ChatMessage) => void },
) {
  const report = message.report;
  const hasCausal = report && hasCausalContent(report);
  const summary = hasCausal ? causalSummary(report) : [];

  return (
    <div className="msg ai">
      <div className="avatar" />
      <div className="bubble">
        <Markdown text={message.content || "No response received."} />
        {/* Below the answer: it annotates what was just read, rather than
            asking for a click before there is anything to annotate. */}
        {hasCausal && onSelect && (
          <button className="view-details-btn" onClick={() => onSelect(message)}>
            <span className="vdb-mark" aria-hidden="true">⚯</span>
            <span className="vdb-text">How this was derived</span>
            {summary.length > 0 && <span className="vdb-summary">{summary.join(" · ")}</span>}
            <span className="vdb-arrow" aria-hidden="true">→</span>
          </button>
        )}
      </div>
    </div>
  );
});

const UserMessage = memo(function UserMessage({ message }: { message: ChatMessage }) {
  return (
    <div className="msg user">
      <div className="bubble">
        <p>{message.content}</p>
        {(message.attachments || []).map((name, i) => (
          <span className="bubble-attachment" key={`${name}-${i}`}>
            {"⎘ " + name}
          </span>
        ))}
      </div>
    </div>
  );
});

function PendingBubble({ stages }: { stages: Stage[] }) {
  const active = stages.find((s) => s.status === "active");
  return (
    <div className="msg ai">
      <div className="avatar" />
      <div className="bubble">
        <div className="causal-summary-box live-pulse">
          <div className="causal-summary-header">
            <span className="causal-spinner">⚯</span> Causal reasoning:{" "}
            <span className="active-step-label" key={active?.id || "thinking"}>
              {active?.label || "Thinking..."}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

interface Props {
  messages: ChatMessage[];
  isSending: boolean;
  stages: Stage[];
  liveGraph: CausalGraphType | null;
  onSelectMessage?: (message: ChatMessage) => void;
  onPromptClick?: (text: string) => void;
}

export function MessageList({ messages, isSending, stages, liveGraph: _liveGraph, onSelectMessage, onPromptClick }: Props) {
  const areaRef = useRef<HTMLDivElement>(null);

  // Only follow the tail if the reader is already at it. `stages` changes on
  // every progress frame, so an unconditional scroll-to-bottom yanked anyone
  // who scrolled up to re-read an earlier answer straight back down, within a
  // fraction of a second, for the whole run.
  const pinnedToBottom = useRef(true);

  const onScroll = () => {
    const area = areaRef.current;
    if (!area) return;
    const slack = area.scrollHeight - area.scrollTop - area.clientHeight;
    pinnedToBottom.current = slack < 80;
  };

  useLayoutEffect(() => {
    const area = areaRef.current;
    if (area && pinnedToBottom.current) area.scrollTop = area.scrollHeight;
  }, [messages, isSending, stages]);

  return (
    <div className="messages" id="messages-area" ref={areaRef} onScroll={onScroll}>
      <div className="messages-inner" id="messages-inner">
        {messages.length === 1 && messages[0].role === "greeting" && (
          <div className="empty-state">
            <div className="empty-state-greeting">
              <div className="avatar" />
              <p>{messages[0].content}</p>
            </div>
            <div className="prompt-cards">
              <button onClick={() => {
                if(onPromptClick) onPromptClick("Why does ice float?");
              }}>🧊 Why does ice float?</button>
              <button onClick={() => {
                if(onPromptClick) onPromptClick("Explain how water boils");
              }}>🔥 Explain how water boils</button>
            </div>
          </div>
        )}
        
        {messages.map((message) => {
          if (message.role === "greeting") return null; // handled in empty state
          if (message.role === "user") return <UserMessage key={message.key} message={message} />;
          if (message.role === "error") {
            return (
              <div className="msg ai" key={message.key}>
                <div className="avatar" />
                <div className="bubble error">
                  <p>{"Error: " + message.content}</p>
                </div>
              </div>
            );
          }
          return <AiMessage key={message.key} message={message} onSelect={onSelectMessage} />;
        })}
        {isSending && <PendingBubble stages={stages} />}
      </div>
    </div>
  );
}
