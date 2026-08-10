import { useEffect, useRef } from "react";
import { ALLOWED_EXTENSIONS, formatSize } from "../hooks/useAttachments";
import type { Attachment } from "../types";

interface Props {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onStop?: () => void;
  isSending?: boolean;
  disabled: boolean;
  /** Behind the access gate: the whole composer is inert, not just Send. */
  locked?: boolean;
  onUnlock?: () => void;
  attachments: Attachment[];
  onFiles: (files: FileList | null) => void;
  onRemoveAttachment: (localId: string) => void;
}

function Chip({ att, onRemove }: { att: Attachment; onRemove: () => void }) {
  const classes = ["attach-chip"];
  if (att.status === "uploading") classes.push("uploading");
  if (att.status === "error") classes.push("error");
  return (
    <span className={classes.join(" ")} data-att-id={att.localId} title={att.error}>
      <span className="chip-name">{att.name}</span>
      <span className="size">{"· " + formatSize(att.size)}</span>
      <button type="button" className="chip-x" aria-label={"Remove " + att.name} onClick={onRemove}>
        ✕
      </button>
    </span>
  );
}

export function Composer({
  value,
  onChange,
  onSend,
  onStop,
  isSending,
  disabled,
  locked = false,
  onUnlock,
  attachments,
  onFiles,
  onRemoveAttachment,
}: Props) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-resize to fit content, matching the vanilla behaviour.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = el.scrollHeight + "px";
  }, [value]);

  return (
    <div className="composer">
      <div className="composer-inner">
        <div className="chips-row" id="attachment-chips">
          {attachments.map((att) => (
            <Chip key={att.localId} att={att} onRemove={() => onRemoveAttachment(att.localId)} />
          ))}
        </div>
        <div className={locked ? "input-pill locked" : "input-pill"} onClick={locked ? onUnlock : undefined}>
          <button
            className="attach-btn"
            id="attach-btn"
            aria-label="Attach files"
            title="Attach files"
            disabled={locked}
            onClick={() => fileInputRef.current?.click()}
          >
            +
          </button>
          <input
            type="file"
            id="file-input"
            multiple
            hidden
            ref={fileInputRef}
            accept={ALLOWED_EXTENSIONS.join(",")}
            onChange={(e) => {
              onFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <textarea
            id="chat-input"
            rows={1}
            ref={textareaRef}
            disabled={locked}
            placeholder={locked
              ? "Log in with your email to use the agent"
              : "Ask about cause and effect… (Enter to send, Shift+Enter for newline)"}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                onSend();
              }
            }}
          />
          {isSending ? (
            <button className="stop-btn" id="stop-btn" aria-label="Stop" onClick={onStop}>
              ■
            </button>
          ) : (
            <button className="send-btn" id="send-btn" aria-label="Send" disabled={disabled} onClick={onSend}>
              ➤
            </button>
          )}
        </div>
        <div className="disclaimer">Agent evaluations are simulated dry-run predictions.</div>
      </div>
    </div>
  );
}
