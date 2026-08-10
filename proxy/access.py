"""Email-gated access control, token quota, and usage metrics.

Tracer Lens is a beta prototype on a public site, so the agent sits behind an
approval gate keyed on email address rather than being open to the world.

The flow, end to end:

    visitor enters email  →  POST /auth/login
        no record  →  status "pending", admin notified
        approved   →  a signed, single-use login link is emailed to the visitor
        clicked    →  POST /auth/exchange trades it for a 30-day session token

The session token is the identity for every subsequent request: it rides on the
existing ``Authorization: Bearer`` header, so nothing about the CORS allow-list
or the UI's header plumbing had to change when this replaced Firebase auth.

Trust model: access is bound to control of the inbox, not to a typed string —
the email a visitor types only ever *starts* a request, it never grants
anything. The signature is checked on every call, and ``token_version`` on the
record lets a deny or a delete revoke every live session at once.

Storage is Firestore (``agent_access`` for people, ``agent_runs`` for per-turn
metrics). Both hold the minimum the feature needs and nothing else — no IP
addresses, no prompt text, no user agents — because the privacy notice shown in
the access modal promises exactly that.
"""
import asyncio
import base64
import functools
import hashlib
import hmac
import json
import os
import re
import secrets
import smtplib
import ssl
import sys
import time
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import NamedTuple, Optional

import httpx
import firebase_admin
from fastapi import Header, HTTPException
from firebase_admin import firestore as firebase_firestore
from google.cloud import firestore as gcf

from proxy import memstore

ACCESS_COLLECTION = "agent_access"
RUNS_COLLECTION = "agent_runs"
OTP_COLLECTION = "admin_otp"

# Purposes are baked into every signature so a token minted for one job cannot
# be replayed as another — a login link can't become a session, and neither can
# stand in for an admin one-click action.
PURPOSE_LOGIN = "login"
PURPOSE_SESSION = "session"
PURPOSE_ADMIN_ACT = "admin_act"
PURPOSE_ADMIN_SESSION = "admin_session"

LOGIN_LINK_TTL_S = 15 * 60
# 24 hours, paired with the UI holding the token in sessionStorage: closing the
# browser ends the session, and a browser left open all week still expires.
# Neither bound is sufficient alone — sessionStorage survives indefinitely in a
# tab nobody closes, and a TTL alone leaves a live token on a shared machine.
# Re-entry is cheap (one emailed link), so this costs little for the reduction
# in how long a leaked token stays useful.
SESSION_TTL_S = 24 * 60 * 60
ADMIN_ACT_TTL_S = 7 * 24 * 60 * 60
ADMIN_SESSION_TTL_S = 12 * 60 * 60

# Don't re-email the admin every time an impatient visitor resubmits.
NOTIFY_COOLDOWN_S = 15 * 60
NOTIFY_MAX_ATTEMPTS = 5
# Per-address gap between sign-in link resends. Matches the admin OTP cooldown.
# Without it a held-down button turns a personal Gmail into a bulk sender,
# against a ~500/day cap and a spam reputation that is hard to win back.
RESEND_COOLDOWN_S = 30

_RECORD_CACHE_TTL_S = 30.0
# Only a trigger for sweeping expired entries, not a hard ceiling: everything
# past the TTL is dead weight anyway, so the cache settles at roughly the number
# of addresses actually active within a 30-second window.
_RECORD_CACHE_MAX = 512


# ── Configuration ────────────────────────────────────────────────────────────
# Read at call time rather than import time so tests can monkeypatch the
# environment, and so a Cloud Run env-var update takes effect on the next
# request instead of needing a cold start.

def token_limit() -> int:
    return int(os.getenv("ACCESS_TOKEN_LIMIT", "200000"))


def token_grant() -> int:
    return int(os.getenv("ACCESS_TOKEN_GRANT", "200000"))


def chat_retention_hours() -> float:
    return float(os.getenv("CHAT_RETENTION_HOURS", "24"))


def run_metrics_retention_days() -> float:
    return float(os.getenv("RUN_METRICS_RETENTION_DAYS", "30"))


def notify_email() -> str:
    return os.getenv("ACCESS_NOTIFY_EMAIL", "jacobbinu4488code@gmail.com")


def from_email() -> str:
    explicit = os.getenv("ACCESS_FROM_EMAIL")
    if explicit:
        return explicit
    # Over SMTP the provider rewrites From to the authenticated mailbox anyway
    # (Gmail silently replaces a mismatched sender), so defaulting to the login
    # account is what the recipient will actually see. Falling back to the
    # Resend sandbox sender here would produce a header that contradicts the
    # envelope and reads as spoofed.
    smtp = _smtp_config()
    if smtp:
        return f"Tracer Lens <{smtp.user}>"
    return "Tracer Lens <onboarding@resend.dev>"


LOCAL_APP_URL = "http://localhost:8080"


class AppUrlNotConfigured(RuntimeError):
    """APP_URL is missing somewhere its absence would break every sign-in."""


def is_managed_runtime() -> bool:
    """True on Cloud Run / Cloud Functions / App Engine.

    Those runtimes set these themselves, so this needs no configuration of its
    own — which matters, because a variable you must remember to set cannot be
    the thing that catches the variable you forgot to set.
    """
    return bool(os.getenv("K_SERVICE") or os.getenv("GAE_ENV")
                or os.getenv("FUNCTION_TARGET"))


def app_url() -> str:
    """Public origin every emailed link is built from.

    Deliberately *not* derived from the request's Host header. Magic-link
    email is the textbook target for host-header injection: an attacker sends
    a login request carrying ``Host: evil.com``, the victim receives a mail
    whose link points there, and clicking it hands the single-use token to the
    attacker. The origin therefore comes from configuration only.

    An empty value counts as unset — os.getenv(name, default) returns the empty
    string rather than the default when a var is present but blank, which a
    compose file or a CI secret that resolved to nothing produces easily, and
    the resulting links would be rooted at ``/?auth=...`` with no origin.
    """
    configured = (os.getenv("APP_URL") or "").strip().rstrip("/")
    if configured:
        return configured
    # Falling back to localhost in production would mail out links pointing at
    # the recipient's own machine — delivered successfully, dead on arrival,
    # and silent. Refusing is the safer failure.
    if is_managed_runtime() and not os.getenv("ALLOW_LOCALHOST_APP_URL"):
        raise AppUrlNotConfigured(
            "APP_URL is not set, so every sign-in and admin link would point at "
            "http://localhost:8080 instead of the real site. Set APP_URL to the "
            "public origin (e.g. https://tracerlensai.com) on the service. To "
            "override deliberately, set ALLOW_LOCALHOST_APP_URL=1.")
    return LOCAL_APP_URL


# ── Firestore ────────────────────────────────────────────────────────────────

def get_firebase_app():
    """Initialize the Firebase Admin app lazily (uses ADC on Cloud Run)."""
    try:
        return firebase_admin.get_app()
    except ValueError:
        return firebase_admin.initialize_app()


@functools.cache
def get_db():
    """Firestore client.

    The project's database is named ``tracerlensai`` (not the default
    ``(default)``), so pass it explicitly. Overridable via
    FIRESTORE_DATABASE_ID for other environments.

    ``ACCESS_STORE=memory`` swaps in a process-local dict instead — the gate
    reads a record on every request, so without this, offline mock mode would
    need cloud credentials just to open the chat. See proxy/memstore.py.
    """
    if os.getenv("ACCESS_STORE") == "memory":
        print("WARNING: ACCESS_STORE=memory - access records live in memory "
              "and are lost on restart. Never use this in production.")
        return memstore.MemoryDb()
    get_firebase_app()
    database_id = os.getenv("FIRESTORE_DATABASE_ID", "tracerlensai")
    return firebase_firestore.client(database_id=database_id)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def log(text: str) -> None:
    """Print without ever raising on the console's encoding.

    These lines carry visitor-supplied content — an extension request message
    can hold any character at all — and a Windows console defaults to cp1252.
    A plain print() would turn one emoji into a 500 on the request path.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


# ── Signing ──────────────────────────────────────────────────────────────────

_ephemeral_secret: Optional[str] = None


def signing_secret() -> bytes:
    """The HMAC key for every token this module mints.

    An unset ACCESS_SIGNING_SECRET generates a random per-process key so mock
    mode, CI, and the E2E suite run without any secret configured. That is
    deliberately noisy: on a real deploy it would silently invalidate every
    session on each cold start, so it warns loudly rather than failing closed
    and taking the whole site down.
    """
    global _ephemeral_secret
    configured = os.getenv("ACCESS_SIGNING_SECRET")
    if configured:
        return configured.encode("utf-8")
    if _ephemeral_secret is None:
        _ephemeral_secret = secrets.token_urlsafe(32)
        print("WARNING: ACCESS_SIGNING_SECRET is unset - using an ephemeral "
              "per-process key. Sessions will not survive a restart.")
    return _ephemeral_secret.encode("utf-8")


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def sign(payload: dict, purpose: str, ttl_s: int) -> str:
    """Return ``<payload>.<signature>``, both base64url."""
    body = dict(payload)
    body["p"] = purpose
    body["exp"] = int(time.time()) + ttl_s
    encoded = _b64e(json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    digest = hmac.new(signing_secret(), encoded.encode("ascii"), hashlib.sha256).digest()
    return f"{encoded}.{_b64e(digest)}"


def unsign(token: Optional[str], purpose: str) -> Optional[dict]:
    """Verify signature, purpose, and expiry. Returns None on any failure.

    Every rejection collapses to None on purpose: distinguishing "bad
    signature" from "expired" in a response tells an attacker which half of a
    forged token to keep working on.
    """
    if not token or "." not in token:
        return None
    encoded, _, signature = token.rpartition(".")
    expected = hmac.new(signing_secret(), encoded.encode("ascii"), hashlib.sha256).digest()
    try:
        if not hmac.compare_digest(_b64d(signature), expected):
            return None
        payload = json.loads(_b64d(encoded))
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("p") != purpose:
        return None
    if int(payload.get("exp", 0)) < time.time():
        return None
    return payload


# ── Email identity ───────────────────────────────────────────────────────────

# Deliberately permissive: the only address that matters is one the visitor can
# actually receive mail at, and the login link proves that far better than any
# regex can. This just rejects obvious junk before it reaches the mail provider.
#
# The excluded punctuation is defence in depth, not correctness: an unapproved
# address is stored and rendered in the admin dashboard before anyone vets it,
# so quotes, brackets, parens and semicolons buy an attacker markup/script
# characters in a page an admin will open. RFC 5321 permits them in a quoted
# local part; no real mailbox needs them, and the injection surface is not
# worth the compatibility.
_EMAIL_RE = re.compile(r"^[^@\s\"'()<>;,\\]+@[^@\s.]+(\.[^@\s.]+)+$")
MAX_EMAIL_LEN = 254


def normalize_email(raw: Optional[str]) -> Optional[str]:
    """Lowercase and trim, or None when it isn't a plausible address."""
    if not isinstance(raw, str):
        return None
    email = raw.strip().lower()
    # fullmatch, not match: "$" also matches just before a trailing newline, so
    # .match would accept "victim@example.com\n".
    if not email or len(email) > MAX_EMAIL_LEN or not _EMAIL_RE.fullmatch(email):
        return None
    return email


def email_key(email: str) -> str:
    """Stable document id for an address.

    Hashed for id-safety (Firestore ids reject a few characters emails allow)
    and as light pseudonymisation of the collection's key space. The plaintext
    address is still a field on the document — it has to be, to email anyone.
    """
    return hashlib.sha256(email.encode("utf-8")).hexdigest()[:32]


# ── Record store ─────────────────────────────────────────────────────────────
#
# Every read on the request hot path is a single point read by derived doc id —
# no queries, no indexes — behind a short TTL cache. Admin listings stream the
# whole collection and filter in Python, which is the right trade at portfolio
# scale (tens of records) and keeps the query surface at zero.

_record_cache: dict = {}


def _cache_put(key: str, record: Optional[dict]) -> None:
    # Evict on write. invalidate() only ever removed the one key it was given,
    # so entries for addresses that never came back sat here for the life of
    # the process and the cache grew with every distinct visitor.
    if len(_record_cache) > _RECORD_CACHE_MAX:
        cutoff = time.monotonic() - _RECORD_CACHE_TTL_S
        for stale in [k for k, (at, _) in _record_cache.items() if at < cutoff]:
            _record_cache.pop(stale, None)
    _record_cache[key] = (time.monotonic(), record)


def invalidate(email: str) -> None:
    _record_cache.pop(email_key(email), None)


def _access_ref(key: str):
    return get_db().collection(ACCESS_COLLECTION).document(key)


def get_record(email: str, *, cached: bool = True) -> Optional[dict]:
    key = email_key(email)
    if cached:
        entry = _record_cache.get(key)
        if entry and time.monotonic() - entry[0] < _RECORD_CACHE_TTL_S:
            return entry[1]
    snapshot = _access_ref(key).get()
    record = snapshot.to_dict() if snapshot.exists else None
    _cache_put(key, record)
    return record


def list_records() -> list:
    """Every access record, newest request first."""
    records = []
    for doc in get_db().collection(ACCESS_COLLECTION).stream():
        data = doc.to_dict() or {}
        data["key"] = doc.id
        records.append(data)
    records.sort(key=lambda r: str(r.get("requested_at") or ""), reverse=True)
    return records


def create_or_touch_request(email: str) -> tuple:
    """Ensure a record exists for ``email``.

    Returns ``(record, should_notify)``. The cooldown keeps a visitor who
    resubmits five times from sending five identical emails, while still
    re-notifying if the first attempt was long enough ago to have been lost.
    """
    key = email_key(email)
    ref = _access_ref(key)
    snapshot = ref.get()
    now = utcnow()

    if snapshot.exists:
        record = snapshot.to_dict() or {}
        ref.update({"last_seen": now})
        record["last_seen"] = now
        should_notify = (
            record.get("status") == "pending"
            and seconds_since(record.get("notified_at")) > NOTIFY_COOLDOWN_S
        )
        _cache_put(key, record)
        return record, should_notify

    record = {
        "email": email,
        "status": "pending",
        "token_version": 1,
        "login_nonce": None,
        "tokens_used": 0,
        "token_limit": token_limit(),
        "extension_status": "none",
        "extension_message": None,
        "runs_total": 0,
        "runs_failed": 0,
        "latency_ms_sum": 0,
        "last_run_at": None,
        "notify_state": "pending",
        "notify_error": None,
        "notify_attempts": 0,
        "notified_at": None,
        "requested_at": now,
        "decided_at": None,
        "last_seen": now,
    }
    ref.set(record)
    _cache_put(key, record)
    return record, True


def seconds_since(moment) -> float:
    """Age of a Firestore timestamp in seconds; +inf when absent."""
    if moment is None:
        return float("inf")
    if isinstance(moment, str):
        try:
            moment = datetime.fromisoformat(moment)
        except ValueError:
            return float("inf")
    if not hasattr(moment, "tzinfo"):
        return float("inf")
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return (utcnow() - moment).total_seconds()


def update_record(email: str, fields: dict) -> None:
    _access_ref(email_key(email)).update(fields)
    invalidate(email)


def set_status(email: str, status: str) -> Optional[dict]:
    """Approve or deny. Denying revokes any live session immediately."""
    record = get_record(email, cached=False)
    if record is None:
        return None
    fields = {"status": status, "decided_at": utcnow()}
    if status == "denied":
        fields["token_version"] = int(record.get("token_version", 1)) + 1
    update_record(email, fields)
    return get_record(email, cached=False)


def grant_extension(email: str) -> Optional[dict]:
    """Add another grant on top of the current cap.

    Additive, not a reset: someone who has burned 250K of a 400K cap keeps that
    usage and gets a 600K ceiling. Resetting usage instead would silently
    forgive it and make the running total meaningless.
    """
    record = get_record(email, cached=False)
    if record is None:
        return None
    new_limit = int(record.get("token_limit", token_limit())) + token_grant()
    update_record(email, {
        "token_limit": new_limit,
        "extension_status": "none",
        "extension_message": None,
    })
    return get_record(email, cached=False)


def request_extension(email: str, message: Optional[str]) -> Optional[dict]:
    record = get_record(email, cached=False)
    if record is None:
        return None
    update_record(email, {
        "extension_status": "pending",
        "extension_message": (message or "").strip()[:2000] or None,
        "extension_requested_at": utcnow(),
    })
    return get_record(email, cached=False)


def issue_login_nonce(email: str) -> Optional[str]:
    """Mint a single-use nonce and stamp it on the record."""
    if get_record(email, cached=False) is None:
        return None
    nonce = secrets.token_urlsafe(16)
    update_record(email, {"login_nonce": nonce})
    return nonce


def consume_login_nonce(email: str, nonce: str) -> bool:
    """Burn the nonce, so a forwarded or leaked link works exactly once.

    Read-compare-write is not enough to deliver that "exactly once": two
    requests carrying the same leaked link can both read a live nonce before
    either clears it, and both would be handed a full-length session. The
    compare and the clear have to be a single atomic step, so on real
    Firestore this runs in a transaction and only one caller can observe the
    nonce still set. memstore has no transaction support and no concurrency to
    protect against either (single process, offline dev only), so it keeps the
    straight-line path.
    """
    def _matches(record: Optional[dict]) -> bool:
        return bool(record
                    and record.get("login_nonce")
                    and hmac.compare_digest(str(record["login_nonce"]), str(nonce)))

    db = get_db()
    ref = db.collection(ACCESS_COLLECTION).document(email_key(email))
    begin_transaction = getattr(db, "transaction", None)

    if begin_transaction is None:  # ACCESS_STORE=memory
        snapshot = ref.get()
        if not _matches(snapshot.to_dict() if snapshot.exists else None):
            return False
        ref.update({"login_nonce": None})
        invalidate(email)
        return True

    @gcf.transactional
    def _consume(txn) -> bool:
        snapshot = ref.get(transaction=txn)
        if not _matches(snapshot.to_dict() if snapshot.exists else None):
            return False
        txn.update(ref, {"login_nonce": None})
        return True

    consumed = _consume(begin_transaction())
    if consumed:
        invalidate(email)
    return consumed


def delete_user(email: str) -> bool:
    """Erase the access record and every conversation belonging to it.

    This is the GDPR erasure path, reachable both from the visitor's own
    profile menu and from the admin dashboard. Run metrics are keyed by hash
    and expire on their own; they carry no address and no chat content.
    """
    key = email_key(email)
    ref = _access_ref(key)
    if not ref.get().exists:
        return False
    delete_conversations(key)
    ref.delete()
    invalidate(email)
    return True


def delete_conversations(user_key: str) -> int:
    """Drop every conversation (and message) under one user key."""
    user_ref = get_db().collection("users").document(user_key)
    removed = 0
    for conv in user_ref.collection("conversations").stream():
        conv_ref = user_ref.collection("conversations").document(conv.id)
        for message in conv_ref.collection("messages").stream():
            conv_ref.collection("messages").document(message.id).delete()
        conv_ref.delete()
        removed += 1
    user_ref.delete()
    return removed


# ── Usage and run metrics ────────────────────────────────────────────────────

def record_usage(email: str, tokens: int) -> None:
    """Add a turn's tokens to the running total. Best effort by design.

    Called from the streaming path, where a raised exception would truncate a
    response the visitor has already paid for. An occasional lost increment is
    a far better failure than a broken answer.
    """
    if not email or not tokens:
        return
    try:
        _access_ref(email_key(email)).update({"tokens_used": gcf.Increment(int(tokens))})
        invalidate(email)
    except Exception as e:
        log(f"WARNING: failed to record usage for {email_key(email)}: {e}")


def record_run(email: Optional[str], *, ok: bool, latency_ms: int,
               tokens_in: int = 0, tokens_out: int = 0, tokens_total: int = 0,
               model: str = "", causal: bool = False, web: bool = False,
               error_kind: Optional[str] = None) -> None:
    """Write one row of per-turn telemetry for the dashboard.

    Metadata only — no prompt, no response, no error text that could echo
    either. That is what lets these rows outlive the 24-hour chat retention
    without contradicting the promise made in the privacy notice.
    """
    if not email:
        return
    try:
        now = utcnow()
        key = email_key(email)
        get_db().collection(RUNS_COLLECTION).add({
            "email_hash": key,
            "ts": now,
            "ok": bool(ok),
            "error_kind": error_kind,
            "latency_ms": int(latency_ms),
            "tokens_in": int(tokens_in),
            "tokens_out": int(tokens_out),
            "tokens_total": int(tokens_total),
            "model": model,
            "causal": bool(causal),
            "web": bool(web),
            "expires_at": now + timedelta(days=run_metrics_retention_days()),
        })
        _access_ref(key).update({
            "runs_total": gcf.Increment(1),
            "runs_failed": gcf.Increment(0 if ok else 1),
            "latency_ms_sum": gcf.Increment(int(latency_ms)),
            "last_run_at": now,
        })
        invalidate(email)
    except Exception as e:
        log(f"WARNING: failed to record run metrics for {email}: {e}")


def list_runs(limit: int = 100, email_hash: Optional[str] = None) -> list:
    """Most recent runs first, optionally for one user."""
    query = (get_db().collection(RUNS_COLLECTION)
             .order_by("ts", direction=gcf.Query.DESCENDING)
             .limit(max(1, min(limit * 4 if email_hash else limit, 1000))))
    rows = []
    for doc in query.stream():
        data = doc.to_dict() or {}
        if email_hash and data.get("email_hash") != email_hash:
            continue
        data["id"] = doc.id
        rows.append(data)
        if len(rows) >= limit:
            break
    return rows


# ── The gate ─────────────────────────────────────────────────────────────────

class AccessState(NamedTuple):
    code: str                      # ok | no_session | pending | denied | limit_reached
    email: Optional[str] = None
    usage: int = 0
    limit: int = 0
    extension_status: str = "none"

    @property
    def allowed(self) -> bool:
        return self.code == "ok"

    def as_detail(self) -> dict:
        return {
            "code": self.code,
            "message": ACCESS_MESSAGES.get(self.code, "Access denied"),
            "usage": self.usage,
            "limit": self.limit,
            "extension_status": self.extension_status,
        }


ACCESS_MESSAGES = {
    "no_session": "Sign in with your email to use the agent.",
    "pending": "Your access request is waiting for approval.",
    "denied": "Access to this demo was not granted for this address.",
    "limit_reached": "You have reached your token limit for this demo.",
}


def check_access(caller: Optional[dict]) -> AccessState:
    """Decide whether this caller may run the agent."""
    if not caller or not caller.get("email"):
        return AccessState("no_session")
    email = caller["email"]
    record = get_record(email)
    if record is None:
        return AccessState("no_session", email)
    status = record.get("status")
    if status == "denied":
        return AccessState("denied", email)
    usage = int(record.get("tokens_used", 0))
    limit = int(record.get("token_limit", token_limit()))
    extension = record.get("extension_status", "none")
    if status != "approved":
        return AccessState("pending", email, usage, limit, extension)
    if usage >= limit:
        return AccessState("limit_reached", email, usage, limit, extension)
    return AccessState("ok", email, usage, limit, extension)


def require_access(caller: Optional[dict]) -> AccessState:
    """check_access, but raising the 403 the UI keys its modal off."""
    state = check_access(caller)
    if not state.allowed:
        raise HTTPException(status_code=403, detail=state.as_detail())
    return state


# ── Session identity ─────────────────────────────────────────────────────────

async def get_caller(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """Resolve the session token: absent → None, invalid → 401.

    Mirrors the contract of the Firebase ``get_current_user`` this replaced, so
    every endpoint that took an optional user still behaves the same way.
    """
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Malformed Authorization header")
    payload = unsign(token, PURPOSE_SESSION)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    email = normalize_email(payload.get("email"))
    if not email:
        raise HTTPException(status_code=401, detail="Invalid session")
    record = get_record(email)
    if record is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    # A denied or deleted user's live sessions die here rather than lingering
    # until their 30-day expiry.
    if int(payload.get("ver", 0)) != int(record.get("token_version", 1)):
        raise HTTPException(status_code=401, detail="Session revoked — please sign in again")
    return {"email": email, "email_key": email_key(email), "verified": True}


def mint_session(email: str, record: dict) -> str:
    return sign({"email": email, "ver": int(record.get("token_version", 1))},
                PURPOSE_SESSION, SESSION_TTL_S)


def login_link(email: str, nonce: str) -> str:
    token = sign({"email": email, "n": nonce}, PURPOSE_LOGIN, LOGIN_LINK_TTL_S)
    return f"{app_url()}/?auth={token}"


def admin_action_link(action: str, email: str) -> str:
    token = sign({"a": action, "email": email}, PURPOSE_ADMIN_ACT, ADMIN_ACT_TTL_S)
    return f"{app_url()}/admin/act?t={token}"


# ── Notifications ────────────────────────────────────────────────────────────

class _SmtpConfig(NamedTuple):
    host: str
    port: int
    user: str
    password: str


def _smtp_config() -> Optional[_SmtpConfig]:
    """SMTP settings, or None when SMTP isn't configured.

    All four parts must be present: a half-configured mailer that silently
    falls through to a different transport is worse than one that is plainly
    off, because the failure only shows up as mail arriving from the wrong
    sender. SMTP_PORT defaults to 587 (STARTTLS), which is what Gmail,
    Fastmail and Outlook all expect.
    """
    host = os.getenv("SMTP_HOST")
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    if not (host and user and password):
        return None
    try:
        port = int(os.getenv("SMTP_PORT", "587"))
    except ValueError:
        port = 587
    return _SmtpConfig(host, port, user, password)


def _send_via_smtp_blocking(cfg: _SmtpConfig, sender: str, to: str, subject: str,
                            text: str, html: Optional[str]) -> tuple:
    """Blocking SMTP send. Runs off the event loop via asyncio.to_thread.

    Uses stdlib smtplib rather than adding a dependency, matching how the
    signing code sticks to stdlib hmac.
    """
    message = EmailMessage()
    message["From"] = sender
    message["To"] = to
    message["Subject"] = subject
    message.set_content(text)
    if html:
        # Both parts, so a plain-text client still gets the link — these mails
        # exist to carry a URL, and an HTML-only body would strand anyone
        # reading in a terminal client.
        message.add_alternative(html, subtype="html")

    context = ssl.create_default_context()
    try:
        if cfg.port == 465:
            with smtplib.SMTP_SSL(cfg.host, cfg.port, context=context, timeout=20) as server:
                server.login(cfg.user, cfg.password)
                server.send_message(message)
        else:
            with smtplib.SMTP(cfg.host, cfg.port, timeout=20) as server:
                server.starttls(context=context)
                server.login(cfg.user, cfg.password)
                server.send_message(message)
        return True, ""
    except smtplib.SMTPAuthenticationError as e:
        # By far the most common setup failure, and the raw message ("Username
        # and Password not accepted") sends people to reset their password when
        # the real answer is that Gmail needs an App Password.
        return False, (f"smtp auth rejected for {cfg.user} — Gmail requires a 16-character "
                       f"App Password (with 2-Step Verification on), not the account "
                       f"password: {e}")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


async def send_email(to: str, subject: str, text: str, html: Optional[str] = None) -> tuple:
    """Send via SMTP or Resend. Returns ``(sent, error)``.

    Transport precedence is SMTP → Resend → print, so configuring SMTP_* is
    enough to route mail through a personal mailbox without touching the
    Resend settings that a deployment may still carry.

    With neither configured the message is printed. Whether that *counts* as
    delivered is the difference between a dev box and a deployment, so it is an
    explicit choice rather than a guess: ACCESS_MAIL_TRANSPORT=console says
    "printing is the transport here", which is what lets mock mode, CI, and the
    E2E suite exercise the whole approval flow without a single credential.

    Left unset, an unconfigured mailer is reported as the failure it is. That
    asymmetry is deliberate — the access gate is entirely email-driven, so a
    deployment whose mail silently no-ops does not degrade, it locks every
    visitor out while reporting success. Better a loud 502 on the first
    sign-in than a site that looks healthy and admits nobody.
    """
    if not to:
        # getenv(name, default) only falls back when a variable is *absent*, so
        # an empty ACCESS_NOTIFY_EMAIL (easy to produce from a compose file or
        # a CI secret that resolved to nothing) would otherwise reach the
        # provider as a send to nobody and fail far from its cause.
        log("ERROR event=notify_failed reason=empty_recipient "
            f"subject={subject!r}")
        return False, "no recipient address configured"

    smtp = _smtp_config()
    if smtp:
        sent, error = await asyncio.to_thread(
            _send_via_smtp_blocking, smtp, from_email(), to, subject, text, html)
        if not sent:
            log(f"ERROR event=notify_failed to={to} {error}")
        return sent, error

    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        console = os.getenv("ACCESS_MAIL_TRANSPORT", "").strip().lower() == "console"
        rule = "-" * 46
        label = ("EMAIL (console transport — printed, not sent)" if console
                 else "EMAIL (not sent, no SMTP_* or RESEND_API_KEY configured)")
        log(f"\n{rule}\n{label}\n"
            f"To: {to}\nSubject: {subject}\n\n{text}\n{rule}\n")
        if console:
            # The log *is* the inbox in this mode: the E2E suite reads its
            # sign-in links straight back out of this output, so reporting a
            # send here is accurate rather than a stub.
            return True, ""
        return False, "no mail transport configured"
    payload = {"from": from_email(), "to": [to], "subject": subject, "text": text}
    if html:
        payload["html"] = html
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://api.resend.com/emails",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
        if response.status_code >= 400:
            error = f"resend {response.status_code}: {response.text[:300]}"
            log(f"ERROR event=notify_failed to={to} {error}")
            return False, error
        return True, ""
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        log(f"ERROR event=notify_failed to={to} {error}")
        return False, error


def _record_notify_outcome(email: str, sent: bool, error: str) -> None:
    """Persist delivery state so a silent failure becomes a visible one.

    The dashboard reads these fields and surfaces failures in its alert strip
    with a retry button — a dropped notification should delay a decision, never
    lose the request.
    """
    try:
        update_record(email, {
            "notify_state": "sent" if sent else "failed",
            "notify_error": None if sent else error[:500],
            "notify_attempts": gcf.Increment(1),
            "notified_at": utcnow(),
        })
    except Exception as e:
        log(f"WARNING: failed to record notify outcome for {email}: {e}")


def _wrap_html(title: str, body: str) -> str:
    return (
        '<div style="font-family:system-ui,-apple-system,Segoe UI,sans-serif;'
        'max-width:520px;margin:0 auto;padding:24px;color:#1a1a1a">'
        f'<h2 style="margin:0 0 16px;font-size:18px">{title}</h2>{body}'
        '<p style="margin-top:28px;font-size:12px;color:#888">'
        'Tracer Lens — automated message</p></div>'
    )


def _button(href: str, label: str, color: str) -> str:
    return (f'<a href="{href}" style="display:inline-block;padding:10px 18px;'
            f'margin:4px 8px 4px 0;background:{color};color:#fff;'
            f'text-decoration:none;border-radius:6px;font-weight:600">{label}</a>')


async def notify_access_request(email: str) -> None:
    """Tell the admin someone wants in, with one-click decision links."""
    approve = admin_action_link("approve", email)
    deny = admin_action_link("deny", email)
    when = utcnow().strftime("%Y-%m-%d %H:%M UTC")
    text = (f"New Tracer Lens access request\n\n"
            f"Email: {email}\nRequested: {when}\n\n"
            f"Approve: {approve}\nDeny:    {deny}\n\n"
            f"Dashboard: {app_url()}/admin\n")
    html = _wrap_html("New access request", (
        f'<p><strong>{email}</strong><br>'
        f'<span style="color:#666">Requested {when}</span></p>'
        f'<p>{_button(approve, "Approve", "#16a34a")}{_button(deny, "Deny", "#dc2626")}</p>'
        f'<p style="font-size:13px"><a href="{app_url()}/admin">Open the dashboard</a></p>'
    ))
    sent, error = await send_email(notify_email(), f"Tracer Lens access request — {email}", text, html)
    _record_notify_outcome(email, sent, error)


async def notify_extension_request(email: str, message: Optional[str], usage: int, limit: int) -> None:
    grant = admin_action_link("grant", email)
    text = (f"Token extension request\n\n"
            f"Email: {email}\nUsage: {usage:,} / {limit:,} tokens\n"
            f"Message: {message or '(none)'}\n\n"
            f"Grant +{token_grant():,}: {grant}\n\nDashboard: {app_url()}/admin\n")
    html = _wrap_html("Token extension request", (
        f'<p><strong>{email}</strong><br>'
        f'<span style="color:#666">{usage:,} / {limit:,} tokens used</span></p>'
        f'<p style="background:#f4f4f5;padding:12px;border-radius:6px;white-space:pre-wrap">'
        f'{(message or "(no message)")}</p>'
        f'<p>{_button(grant, f"Grant +{token_grant():,}", "#16a34a")}</p>'
    ))
    sent, error = await send_email(
        notify_email(), f"Tracer Lens extension request — {email}", text, html)
    _record_notify_outcome(email, sent, error)


async def send_login_link(email: str, link: str) -> tuple:
    text = (f"Here is your sign-in link for Tracer Lens:\n\n{link}\n\n"
            f"It works once and expires in 15 minutes.\n"
            f"If you didn't ask for this, you can ignore it.\n")
    html = _wrap_html("Sign in to Tracer Lens", (
        f'<p>{_button(link, "Sign in", "#6366f1")}</p>'
        '<p style="font-size:13px;color:#666">This link works once and expires in '
        '15 minutes. If you didn\'t ask for it, ignore this email.</p>'
    ))
    return await send_email(email, "Your Tracer Lens sign-in link", text, html)


async def send_approval_email(email: str, link: str) -> tuple:
    """Approval doubles as the sign-in email — approve, click, you're in."""
    text = (f"You've been approved to use Tracer Lens.\n\n"
            f"Sign in here: {link}\n\n"
            f"This link works once and expires in 15 minutes; you can always request "
            f"a new one from the site.\n\n"
            f"Your data: I store your email address and a token-usage counter, to "
            f"approve access and enforce the usage quota. Conversations are deleted "
            f"from the server within 24 hours. To have your data erased, use the "
            f"'Delete my data' button in your profile menu or reply to this email.\n")
    html = _wrap_html("You're approved", (
        '<p>Your access to the Tracer Lens demo has been approved.</p>'
        f'<p>{_button(link, "Sign in", "#6366f1")}</p>'
        '<p style="font-size:13px;color:#666">This link works once and expires in '
        '15 minutes. You can request a new one from the site at any time.</p>'
        '<p style="font-size:12px;color:#888;border-top:1px solid #eee;padding-top:12px">'
        'I store your email address and a token-usage counter, to approve access and '
        'enforce the quota. Conversations are deleted from the server within 24 hours. '
        'Use <em>Delete my data</em> in your profile menu to erase everything.</p>'
    ))
    return await send_email(email, "Your Tracer Lens access is approved", text, html)


async def send_denied_email(email: str) -> tuple:
    text = ("Thanks for your interest in Tracer Lens.\n\n"
            "Access wasn't granted for this address this time. This is a small "
            "personal prototype with a limited budget, so access is granted "
            "sparingly.\n")
    return await send_email(email, "Tracer Lens access request", text,
                            _wrap_html("Access request", f"<p>{text}</p>"))


async def send_extension_granted(email: str, new_limit: int) -> tuple:
    text = (f"Your Tracer Lens token limit has been raised to {new_limit:,} tokens.\n\n"
            f"You can carry on where you left off — no need to sign in again.\n")
    return await send_email(email, "Tracer Lens: more tokens granted", text,
                            _wrap_html("More tokens granted", (
                                f'<p>Your limit is now <strong>{new_limit:,} tokens</strong>.</p>'
                                '<p>Carry on where you left off — no need to sign in again.</p>')))


async def retry_failed_notifications(limit: int = 20) -> int:
    """Re-send notifications that previously failed. Returns how many recovered.

    Called opportunistically from the dashboard and the status poll, so a
    transient Resend outage heals itself without anyone noticing.
    """
    recovered = 0
    for record in list_records()[:limit]:
        if record.get("notify_state") != "failed":
            continue
        if int(record.get("notify_attempts", 0)) >= NOTIFY_MAX_ATTEMPTS:
            continue
        email = record.get("email")
        if not email:
            continue
        if record.get("status") == "pending":
            await notify_access_request(email)
        elif record.get("extension_status") == "pending":
            await notify_extension_request(
                email, record.get("extension_message"),
                int(record.get("tokens_used", 0)), int(record.get("token_limit", 0)))
        else:
            continue
        # get_record returns None for a record deleted while this batch ran
        # (a user exercising "Delete my data" mid-retry). Without the guard the
        # AttributeError 500s the endpoint and skips every remaining record.
        refreshed = get_record(email, cached=False) or {}
        if refreshed.get("notify_state") == "sent":
            recovered += 1
    return recovered
