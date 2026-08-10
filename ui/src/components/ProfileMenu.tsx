// Header chip for a signed-in visitor: who they are, how much quota is left,
// and the two things they might want to do about it.

import { useEffect, useRef, useState } from "react";
import type { Access } from "../hooks/useAccess";
import { UsageBar } from "./AccessGate";

interface Props {
  access: Access;
  onRequestTokens: () => void;
}

export function ProfileMenu({ access, onRequestTokens }: Props) {
  const { state } = access;
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const remaining = Math.max(0, state.token_limit - state.tokens_used);

  const confirmErase = () => {
    const ok = window.confirm(
      "Delete your email address and usage record, and every conversation stored for it? " +
        "This can't be undone.",
    );
    if (ok) void access.eraseMyData();
  };

  return (
    <div className="profile-wrap" ref={wrapRef}>
      <button
        className="profile-btn"
        id="profile-btn"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <span className="profile-avatar" aria-hidden="true">
          {(state.email || "?").charAt(0).toUpperCase()}
        </span>
        <span className="profile-email">{state.email}</span>
      </button>

      {open && (
        <div className="profile-menu" id="profile-menu" role="menu">
          <div className="profile-header">{state.email}</div>

          <div className="profile-usage">
            <div className="profile-usage-row">
              <span>Tokens left</span>
              <strong id="tokens-remaining">{remaining.toLocaleString()}</strong>
            </div>
            <UsageBar used={state.tokens_used} limit={state.token_limit} />
            <div className="profile-usage-sub">
              {state.tokens_used.toLocaleString()} of {state.token_limit.toLocaleString()} used
            </div>
          </div>

          {state.extension_status === "pending" ? (
            <div className="profile-note">More tokens requested — waiting for approval.</div>
          ) : (
            <button className="profile-item" role="menuitem" onClick={() => { setOpen(false); onRequestTokens(); }}>
              Request more tokens
            </button>
          )}

          <button className="profile-item" role="menuitem" onClick={() => { setOpen(false); access.logOut(); }}>
            Log out
          </button>
          <button className="profile-item danger" role="menuitem" onClick={confirmErase}>
            Delete my data
          </button>
          <div className="profile-footer">
            Chats are deleted from the server within 24 hours.
          </div>
        </div>
      )}
    </div>
  );
}
