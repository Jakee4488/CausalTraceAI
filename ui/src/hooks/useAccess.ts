// Owns the email gate's state machine so App.tsx doesn't grow another six
// useState calls. Resolves the session on boot (including arriving via a
// sign-in link), then polls slowly while a request is pending.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AccessError,
  deleteAccount,
  exchangeAuth,
  fetchAccessStatus,
  login,
  requestExtension,
} from "../lib/api";
import {
  EMPTY_ACCESS,
  clearSession,
  getSession,
  getStoredEmail,
  storeSession,
  takeAuthParam,
  type AccessState,
  type AccessStatus,
} from "../lib/access";

// Slow on purpose: an approval is a human action minutes or hours away, and a
// tab left open overnight shouldn't hammer the endpoint.
const POLL_INTERVAL_MS = 30_000;

export interface Access {
  state: AccessState;
  approved: boolean;
  busy: boolean;
  error: string | null;
  submitEmail: (email: string) => Promise<void>;
  refresh: () => Promise<void>;
  askForMoreTokens: (message: string) => Promise<void>;
  eraseMyData: () => Promise<void>;
  logOut: () => void;
  /** Push a gate refusal that came back from another endpoint into this state. */
  applyRefusal: (error: AccessError) => void;
}

export function useAccess(): Access {
  const [state, setState] = useState<AccessState>(() => ({
    ...EMPTY_ACCESS,
    status: getSession() ? "unknown" : "logged_out",
    email: getStoredEmail() || undefined,
  }));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bootedRef = useRef(false);

  const forgetSession = useCallback((status: AccessStatus = "logged_out") => {
    clearSession();
    setState({ ...EMPTY_ACCESS, status });
  }, []);

  const refresh = useCallback(async () => {
    if (!getSession()) {
      setState((prev) => (prev.status === "logged_out" ? prev : { ...EMPTY_ACCESS, status: "logged_out" }));
      return;
    }
    try {
      setState(await fetchAccessStatus());
    } catch (e) {
      if (e instanceof AccessError && e.code === "logged_out") forgetSession();
      else console.error("Failed to read access status:", e);
    }
  }, [forgetSession]);

  // Boot: a sign-in link takes priority over any stored session, since it is
  // the more recent intent (and may belong to a different address).
  useEffect(() => {
    if (bootedRef.current) return;
    bootedRef.current = true;

    const auth = takeAuthParam();
    if (!auth) {
      void refresh();
      return;
    }
    setBusy(true);
    exchangeAuth(auth)
      .then(({ token, ...next }) => {
        storeSession(token, next.email || "");
        setState(next);
      })
      .catch((e: Error) => {
        forgetSession();
        setError(e.message);
      })
      .finally(() => setBusy(false));
  }, [refresh, forgetSession]);

  // Poll only while something is actually expected to change.
  useEffect(() => {
    if (state.status !== "pending" && state.extension_status !== "pending") return;
    const timer = window.setInterval(() => void refresh(), POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [state.status, state.extension_status, refresh]);

  const submitEmail = useCallback(async (email: string) => {
    setBusy(true);
    setError(null);
    try {
      const next = await login(email);
      // "link_sent" means an approved address just got a fresh sign-in link;
      // the session arrives when they click it, not now.
      setState({ ...EMPTY_ACCESS, ...next });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, []);

  const askForMoreTokens = useCallback(async (message: string) => {
    setBusy(true);
    setError(null);
    try {
      setState(await requestExtension(message));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, []);

  const eraseMyData = useCallback(async () => {
    setBusy(true);
    try {
      await deleteAccount();
    } catch (e) {
      console.error("Failed to delete account:", e);
    } finally {
      setBusy(false);
      forgetSession();
    }
  }, [forgetSession]);

  const applyRefusal = useCallback((refusal: AccessError) => {
    if (refusal.code === "logged_out") {
      forgetSession();
      return;
    }
    setState((prev) => ({
      ...prev,
      status: refusal.code,
      tokens_used: refusal.usage || prev.tokens_used,
      token_limit: refusal.limit || prev.token_limit,
    }));
  }, [forgetSession]);

  return {
    state,
    approved: state.status === "ok",
    busy,
    error,
    submitEmail,
    refresh,
    askForMoreTokens,
    eraseMyData,
    logOut: forgetSession,
    applyRefusal,
  };
}
