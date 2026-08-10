import type { AccessState, AccessStatus } from "./access";
import { getSession } from "./access";
import type { CausalGraph, Conversation, Report } from "../types";
import { getAnonId } from "./ids";
import { readSse } from "./sse";

export interface PromptBody {
  prompt: string;
  /** Retained for the proxy's schema; this app has a single pathway and always
   *  runs it, so there is nothing for the flag to switch. */
  causal_reasoning: boolean;
  model_name: string;
  chat_id: string;
  attachments?: string[];
  /** Correlation id for this turn; echoed back on the report. */
  run_id: string;
}

export interface ProgressFrame {
  stage?: string;
  message?: string;
  step?: string;
  elapsed_ms: number;
  index?: number;
  total?: number;
}

interface AnalyzeHandlers {
  signal?: AbortSignal;
  onProgress?: (frame: ProgressFrame) => void;
  onGraph?: (graph: CausalGraph | null) => void;
}

interface HistoryPage {
  conversations: Conversation[];
  next_cursor: string | null;
}

interface ConversationMessage {
  role: "user" | "ai";
  content?: string;
  attachments?: string[];
  causal?: Partial<Report>;
}

export interface ConversationPayload {
  chat_id: string;
  title?: string;
  total_tokens: number;
  messages: ConversationMessage[];
}

interface UploadPayload {
  file_id: string;
}

/**
 * A refusal by the access gate rather than a failed request.
 *
 * The chat UI treats these as a *state* (show the right modal), not as an
 * error bubble in the transcript — being over quota isn't a failed turn.
 */
export class AccessError extends Error {
  constructor(
    readonly code: AccessStatus,
    message: string,
    readonly usage = 0,
    readonly limit = 0,
  ) {
    super(message);
    this.name = "AccessError";
  }
}

interface AccessDetail {
  code?: AccessStatus;
  message?: string;
  usage?: number;
  limit?: number;
}

function apiBase(): string {
  const value = (window as typeof window & { TRACERLENS_API_BASE?: string }).TRACERLENS_API_BASE;
  return value ? value.replace(/\/$/, "") : "";
}

function authHeaders(contentType?: string): HeadersInit {
  const headers: Record<string, string> = { "X-Anon-Id": getAnonId() };
  if (contentType) headers["Content-Type"] = contentType;
  const session = getSession();
  if (session) headers.Authorization = "Bearer " + session;
  return headers;
}

async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const response = await fetch(`${apiBase()}${path}`, init);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    let detail: AccessDetail | undefined;
    try {
      const payload = (await response.json()) as { detail?: string | AccessDetail };
      if (typeof payload.detail === "string") message = payload.detail;
      else if (payload.detail) {
        detail = payload.detail;
        message = detail.message || message;
      }
    } catch {
      // Keep status message fallback.
    }
    // A revoked or expired session reads as logged out, so the UI can drop it
    // and show the login modal instead of an unexplained failure.
    if (response.status === 401) throw new AccessError("logged_out", message);
    if (response.status === 403 && detail?.code) {
      throw new AccessError(detail.code, message, detail.usage, detail.limit);
    }
    throw new Error(message);
  }
  return response;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await request(path, {
    method: "POST",
    headers: authHeaders("application/json"),
    body: JSON.stringify(body),
  });
  return (await response.json()) as T;
}

// ── Access gate ──────────────────────────────────────────────────────────────

/** Submit an email: the server decides whether that's a sign-in or a request. */
export function login(email: string): Promise<AccessState> {
  return postJson<AccessState>("/auth/login", { email });
}

/** Trade a single-use link token for a session. */
export function exchangeAuth(auth: string): Promise<AccessState & { token: string }> {
  return postJson<AccessState & { token: string }>("/auth/exchange", { auth });
}

export async function fetchAccessStatus(): Promise<AccessState> {
  const response = await request("/access/status", { headers: authHeaders() });
  return (await response.json()) as AccessState;
}

export function requestExtension(message: string): Promise<AccessState> {
  return postJson<AccessState>("/access/extension", { message });
}

export async function deleteAccount(): Promise<void> {
  await request("/account", { method: "DELETE", headers: authHeaders() });
}

export async function analyzePrompt(body: PromptBody, handlers: AnalyzeHandlers = {}): Promise<Report> {
  const response = await request("/analyze-prompt", {
    method: "POST",
    headers: authHeaders("application/json"),
    body: JSON.stringify(body),
    signal: handlers.signal,
  });

  if (!response.headers.get("content-type")?.includes("text/event-stream")) {
    return (await response.json()) as Report;
  }

  let done: Report | null = null;
  for await (const frame of readSse(response)) {
    if (frame.event === "progress") {
      handlers.onProgress?.(JSON.parse(frame.data) as ProgressFrame);
    } else if (frame.event === "graph") {
      handlers.onGraph?.(JSON.parse(frame.data) as CausalGraph | null);
    } else if (frame.event === "done") {
      done = JSON.parse(frame.data) as Report;
    } else if (frame.event === "error") {
      const payload = JSON.parse(frame.data) as { error?: string; detail?: string };
      throw new Error(payload.error || payload.detail || "Streaming error");
    }
  }

  if (!done) throw new Error("Missing terminal response from server");
  return done;
}

export async function uploadFile(file: File): Promise<UploadPayload> {
  const form = new FormData();
  form.append("file", file);
  const response = await request("/upload", {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });
  return (await response.json()) as UploadPayload;
}

export async function fetchHistory(cursor?: string): Promise<HistoryPage | null> {
  const query = cursor ? `?cursor=${encodeURIComponent(cursor)}` : "";
  const response = await request(`/history${query}`, { headers: await authHeaders() });
  return (await response.json()) as HistoryPage;
}

export async function fetchConversation(chatId: string): Promise<ConversationPayload | null> {
  const response = await request(`/history/${encodeURIComponent(chatId)}`, {
    headers: authHeaders(),
  });
  return (await response.json()) as ConversationPayload;
}
