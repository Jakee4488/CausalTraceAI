// The email gate. One component, one `mode` derived from access state, so
// there is no second near-duplicate modal for the token-cap case.
//
// Dialog mechanics follow the pattern already proven in causal/StepDrawer.tsx:
// role="dialog" + aria-modal, initial focus, a fixed backdrop, no portal. The
// difference is dismissal — the sign-in states are a gate and ignore Escape,
// while the quota states let you close and re-read the transcript.

import { useEffect, useRef, useState } from "react";
import type { Access } from "../hooks/useAccess";

interface Props {
  access: Access;
  /** Open the extension form on request, not only when the cap is hit. */
  showExtension?: boolean;
  onCloseExtension?: () => void;
}

const CONTACT = "jacobbinu4488code@gmail.com";

/** Mirrors RESEND_COOLDOWN_S in proxy/access.py. */
const RESEND_COOLDOWN_S = 30;

export function AccessGate({ access, showExtension = false, onCloseExtension }: Props) {
  const { state, busy, error } = access;
  const [email, setEmail] = useState(state.email || "");
  const [reason, setReason] = useState("");
  const [dismissed, setDismissed] = useState(false);
  const [resendIn, setResendIn] = useState(0);
  const panelRef = useRef<HTMLDivElement>(null);

  const atLimit = state.status === "limit_reached";
  const mode = atLimit || !showExtension ? state.status : "limit_reached";
  const blocking = mode === "logged_out" || mode === "pending" || mode === "denied" || mode === "link_sent";
  const open = !dismissed && mode !== "ok" && mode !== "unknown";

  // A new mode is new information, so an earlier dismissal shouldn't hide it.
  useEffect(() => setDismissed(false), [mode]);

  // Arriving at "check your inbox" means a link was just sent, so the cooldown
  // starts here rather than on the first click — otherwise the button looks
  // available and answers 429.
  useEffect(() => {
    if (mode === "link_sent") setResendIn(RESEND_COOLDOWN_S);
  }, [mode]);

  useEffect(() => {
    if (resendIn <= 0) return;
    const timer = window.setTimeout(() => setResendIn((s) => s - 1), 1000);
    return () => window.clearTimeout(timer);
  }, [resendIn]);

  const resend = async () => {
    if (!state.email) return;
    await access.submitEmail(state.email);
    // Restart the wait even on failure: the server counts from its own last
    // successful send, and hammering it produces 429s rather than email.
    setResendIn(RESEND_COOLDOWN_S);
  };

  const close = () => {
    setDismissed(true);
    onCloseExtension?.();
  };

  useEffect(() => {
    if (!open) return;
    panelRef.current?.focus();
    if (blocking) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
    // `close` is stable enough for this dialog's lifetime; re-binding on every
    // render would churn the listener without changing behaviour.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, blocking]);

  if (!open) return null;

  const submitEmail = (e: React.FormEvent) => {
    e.preventDefault();
    if (email.trim() && !busy) void access.submitEmail(email.trim());
  };

  const submitExtension = (e: React.FormEvent) => {
    e.preventDefault();
    if (!busy) void access.askForMoreTokens(reason.trim());
  };

  return (
    <>
      <div className="access-backdrop" onClick={blocking ? undefined : close} />
      <div
        className="access-modal"
        id="access-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="access-title"
        tabIndex={-1}
        ref={panelRef}
      >
        {!blocking && (
          <button className="access-close" aria-label="Close" onClick={close}>
            ×
          </button>
        )}

        {mode === "logged_out" && (
          <form onSubmit={submitEmail}>
            <h2 className="access-title" id="access-title">Sign in to Tracer Lens</h2>
            <p className="access-lede">
              This is a beta prototype with a limited budget, so access is granted by hand.
              Enter your email — if you're already approved, I'll send you a sign-in link.
            </p>
            <label className="access-label" htmlFor="access-email">Email address</label>
            <input
              className="access-input"
              id="access-email"
              type="email"
              autoComplete="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
            <button className="access-primary" id="access-submit" type="submit" disabled={busy}>
              {busy ? "Sending…" : "Continue"}
            </button>
            {error && <p className="access-error" id="access-error">{error}</p>}
            <PrivacyNotice />
          </form>
        )}

        {mode === "link_sent" && (
          <div id="access-status">
            <h2 className="access-title" id="access-title">Check your inbox</h2>
            <p className="access-lede">
              A sign-in link is on its way to <strong>{state.email}</strong>. It works once and
              expires in 15 minutes. Signing in keeps you in for a day, or until you close
              your browser.
            </p>
            <div className="access-actions">
              <button
                className="access-primary"
                id="access-resend"
                onClick={() => void resend()}
                disabled={busy || resendIn > 0}
              >
                {busy
                  ? "Sending…"
                  : resendIn > 0
                    ? `Send another link (${resendIn}s)`
                    : "Send another link"}
              </button>
              <button className="access-secondary" onClick={() => access.logOut()}>
                Use a different address
              </button>
            </div>
            <p className="access-hint">
              Not arrived? Check spam — a new link replaces the previous one.
            </p>
            {error && <p className="access-error">{error}</p>}
          </div>
        )}

        {mode === "pending" && (
          <div id="access-status">
            <h2 className="access-title" id="access-title">Waiting for approval</h2>
            <p className="access-lede">
              Thanks — your request for <strong>{state.email}</strong> has been sent. You'll get an
              email with a sign-in link once it's approved. This page will notice on its own, so
              you can leave it open.
            </p>
            <div className="access-actions">
              <button className="access-secondary" onClick={() => void access.refresh()} disabled={busy}>
                Check again
              </button>
              <button className="access-secondary" onClick={() => access.logOut()}>
                Use a different address
              </button>
            </div>
            <PrivacyNotice />
          </div>
        )}

        {mode === "denied" && (
          <div id="access-status">
            <h2 className="access-title" id="access-title">Access not granted</h2>
            <p className="access-lede">
              Access wasn't granted for <strong>{state.email}</strong>. This is a small personal
              prototype with a limited budget, so access is given out sparingly. If you think
              that's a mistake, email <a href={`mailto:${CONTACT}`}>{CONTACT}</a>.
            </p>
            <button className="access-secondary" onClick={() => access.logOut()}>
              Use a different address
            </button>
          </div>
        )}

        {mode === "limit_reached" && state.extension_status === "pending" && (
          <div id="access-status">
            <h2 className="access-title" id="access-title">Request sent</h2>
            <p className="access-lede">
              Your request for more tokens has been sent. You'll get an email once it's approved,
              and you can carry on right where you left off — no need to sign in again.
            </p>
          </div>
        )}

        {mode === "limit_reached" && state.extension_status !== "pending" && (
          <form onSubmit={submitExtension}>
            <h2 className="access-title" id="access-title">
              {atLimit ? "You've used your tokens" : "Request more tokens"}
            </h2>
            <p className="access-lede">
              {atLimit
                ? `You've used all ${state.token_limit.toLocaleString()} tokens allocated to this demo. ` +
                  "Want more? Tell me what you're using it for and I'll take a look."
                : "Tell me what you're using it for and I'll take a look. You'll get an email once " +
                  "it's approved."}
            </p>
            <UsageBar used={state.tokens_used} limit={state.token_limit} />
            <label className="access-label" htmlFor="extension-message">
              Why do you need more? <span className="access-optional">(optional)</span>
            </label>
            <textarea
              className="access-input access-textarea"
              id="extension-message"
              rows={3}
              placeholder="Comparing causal tooling for a write-up…"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
            />
            <button className="access-primary" id="extension-submit" type="submit" disabled={busy}>
              {busy ? "Sending…" : "Request more tokens"}
            </button>
            {error && <p className="access-error">{error}</p>}
          </form>
        )}
      </div>
    </>
  );
}

export function UsageBar({ used, limit }: { used: number; limit: number }) {
  const pct = limit > 0 ? Math.min(100, (used / limit) * 100) : 0;
  return (
    <div className="usage-bar" title={`${used.toLocaleString()} of ${limit.toLocaleString()} tokens`}>
      <div className="usage-bar-fill" style={{ width: `${pct.toFixed(1)}%` }} />
    </div>
  );
}

/**
 * UK GDPR notice. Every claim here is enforced in code — the retention by
 * expires_at + the sweep in proxy/admin.py, the erasure by DELETE /account —
 * so keep the two in step if either changes.
 */
function PrivacyNotice() {
  return (
    <details className="access-privacy">
      <summary>What I collect, and for how long</summary>
      <p>
        <strong>Your data.</strong> I store your <strong>email address</strong> and a{" "}
        <strong>token-usage counter</strong> — that's what approves access and enforces the quota.
        Kept while you have access, and deleted whenever you ask: there's a <em>Delete my data</em>{" "}
        button in your profile menu, or email <a href={`mailto:${CONTACT}`}>{CONTACT}</a>. Lawful
        basis: legitimate interests in running and protecting a personally-funded prototype.
      </p>
      <p>
        <strong>Your chats.</strong> Conversations are{" "}
        <strong>automatically deleted from the server within 24 hours</strong> and are never used
        for training. <strong>Please don't paste sensitive or personal information</strong> — while
        a chat is live it's processed by Google Cloud (Vertex AI) to produce the answer.
      </p>
      <p>
        Your address goes to Resend solely to send you these emails. It isn't shared with anyone
        else, used for marketing, or added to any list.
      </p>
    </details>
  );
}
