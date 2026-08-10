const ANON_COOKIE = "tracerlens-anon-id";
let messageCounter = 0;

function randomHex(length: number): string {
  const bytes = new Uint8Array(Math.ceil(length / 2));
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("").slice(0, length);
}

export function generateSessionId(): string {
  return `chat-${Date.now().toString(36)}-${randomHex(8)}`;
}

export function nextMessageKey(role: string): string {
  messageCounter += 1;
  return `${role}-${Date.now().toString(36)}-${messageCounter}`;
}

/**
 * Correlation id for one turn.
 *
 * Minted here so the client can identify a run even if the response never
 * arrives. Hex only, to match the `[[run:...]]` marker the proxy injects — the
 * proxy re-validates and replaces anything that doesn't fit.
 */
export function generateRunId(): string {
  return randomHex(32);
}

export function getAnonId(): string {
  const existing = document.cookie
    .split(";")
    .map((cookie) => cookie.trim())
    .find((cookie) => cookie.startsWith(`${ANON_COOKIE}=`));
  if (existing) return decodeURIComponent(existing.split("=").slice(1).join("="));

  const anonId = `anon-${randomHex(24)}`;
  const maxAge = 60 * 60 * 24 * 365;
  document.cookie = `${ANON_COOKIE}=${encodeURIComponent(anonId)}; Max-Age=${maxAge}; Path=/; SameSite=Lax`;
  return anonId;
}
