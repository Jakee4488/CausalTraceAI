"""CausalTraceAI - Lightweight Proxy Backend.

Serves the static UI and proxies requests to the LangGraph agent running on
Vertex AI Agent Engine, translating its streamed chunks into Server-Sent
Events for the browser.
"""
import asyncio
import json
import os
import re
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone

import httpx
import google.auth
import google.auth.transport.requests
from google.cloud import firestore as gcf
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional

from proxy import access, admin
from proxy.access import get_caller, get_db
from proxy.mockdata import (
    MOCK_DECISION,
    MOCK_GRAPH,
    MOCK_POSTERIORS,
    MOCK_STEPS,
)

app = FastAPI(title="CausalTraceAI Proxy")
# TEMP: admin panel disabled for local docker-compose smoke testing.
# Re-enable before any real deploy: app.include_router(admin.router)


@app.on_event("startup")
def _verify_public_origin():
    """Fail a misconfigured revision at boot rather than at someone's sign-in.

    APP_URL is only read when a link is built, so without this a deploy that
    forgot it starts healthy, serves the UI, accepts logins, and mails out
    links pointing at localhost — the breakage surfaces to a visitor, not to
    whoever deployed. Raising here makes Cloud Run's health check fail the
    revision, so the bad config never takes traffic.
    """
    resolved = access.app_url()   # raises AppUrlNotConfigured in production
    access.log(f"INFO event=startup app_url={resolved} "
               f"managed_runtime={access.is_managed_runtime()}")


# Allow the frontend to call this API cross-origin when it is served from a
# different host than the Cloud Run service — e.g. the app is on
# causaltraceai.com (Firebase Hosting, 60s cap) but hits api.causaltraceai.com
# (Cloud Run direct, no cap) for long causal runs. Comma-separated origins in
# CORS_ORIGINS; empty (default) leaves the middleware inert for same-origin.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        # DELETE is here for /account — the visitor's own "Delete my data"
        # button. Omitting it would make the GDPR erasure path fail in
        # production only, where the app and the API are on different origins.
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        # Auth is a bearer header, not a cookie, so credentials stay off.
        # X-Anon-Id must be listed or the cross-origin path silently drops it
        # and every signed-out caller falls back to the shared identity.
        allow_headers=["Authorization", "Content-Type", "X-Anon-Id"],
        allow_credentials=False,
    )

# ── Identity ─────────────────────────────────────────────────────────────────
#
# Callers are identified by the email-session token minted in proxy/access.py
# and verified by get_caller (imported above), which keeps the same optional
# contract the Firebase dependency it replaced had: absent → None, invalid →
# 401. Everything user-scoped keys on ``email_key`` — a hash of the address —
# so the access record and the conversation history describe the same person no
# matter how many times they sign in.

# Matches what the client actually mints: "anon-" + hex (ui/src/lib/ids.ts
# getAnonId, currently 24 chars). The previous bare-UUID pattern matched nothing
# the UI has ever sent, so the fallback below silently collapsed every
# signed-out caller onto one shared id — the exact leak its docstring says it
# exists to prevent. The length is a range so tuning the client's entropy
# cannot quietly reintroduce that.
_ANON_ID_RE = re.compile(r"anon-[0-9a-f]{16,64}")


def _agent_user_id(user: Optional[dict], anon_id: Optional[str]) -> str:
    """Resolve the agent-side user_id for this caller.

    Signed-in callers key on their email hash. The anonymous fallback is
    unreachable through /analyze-prompt now that the gate rejects sessionless
    callers, but it stays as a defence in depth: agent session state persists
    in VertexAiSessionService, so a shared id would leak one visitor's
    conversation and causal_* state to the next.
    """
    if user:
        return user["email_key"]
    if anon_id and _ANON_ID_RE.fullmatch(anon_id):
        return f"anon:{anon_id}"
    return "anon:unknown"


def _chat_expiry(now: datetime) -> datetime:
    """When a conversation written at ``now`` must be gone.

    Backs the "deleted within 24 hours" promise in the access modal. Firestore
    TTL policies on this field do the routine work; POST /admin/sweep is the
    backstop, because TTL deletion is only guaranteed to happen *within 24
    hours of expiry* and the promise is a flat 24 hours from writing.
    """
    return now + timedelta(hours=access.chat_retention_hours())


def _save_exchange(user: dict, chat_id: str, prompt: str, response_text: str, token_count: int,
                   attachments: Optional[list] = None, causal: Optional[dict] = None,
                   run_id: Optional[str] = None):
    """Persist a user/AI message pair under users/{email_key}/conversations/{chat_id}."""
    db = get_db()
    now = datetime.now(timezone.utc)
    expires_at = _chat_expiry(now)

    user_ref = db.collection("users").document(user["email_key"])
    user_ref.set({"email": user.get("email"), "last_seen": now}, merge=True)

    conv_ref = user_ref.collection("conversations").document(chat_id)
    if conv_ref.get().exists:
        # The window slides with activity: an active conversation stays alive,
        # and 24 hours after the last message the whole thing goes.
        conv_ref.update({"updated_at": now, "expires_at": expires_at,
                         "total_tokens": gcf.Increment(token_count)})
    else:
        conv_ref.set({
            "title": prompt[:50],
            "created_at": now,
            "updated_at": now,
            "expires_at": expires_at,
            "total_tokens": token_count,
        })

    messages = conv_ref.collection("messages")
    user_msg = {"role": "user", "content": prompt, "created_at": now, "expires_at": expires_at}
    if attachments:
        user_msg["attachments"] = attachments
    # Stamped on both sides of the exchange so a question about a specific
    # answer traces back to its proxy log line and Cloud Trace span. Set before
    # the write — mutating the dict afterwards would not reach Firestore.
    if run_id:
        user_msg["run_id"] = run_id
    messages.add(user_msg)

    ai_msg = {"role": "ai", "content": response_text,
              "created_at": datetime.now(timezone.utc), "expires_at": expires_at}
    if run_id:
        ai_msg["run_id"] = run_id
    if causal:
        # Stored under the same key names the UI renderer reads off the live
        # response, so a reloaded turn replays identically (diagram, estimand
        # card, steps) instead of degrading to plain Markdown.
        ai_msg["causal"] = causal
    messages.add(ai_msg)


# ── Static Files ─────────────────────────────────────────────────────────────

def _ui_directory() -> Optional[str]:
    """Locate the built UI bundle.

    The React app builds to ui/dist, which is where both local dev and the E2E
    suite find it, and where Dockerfile.proxy copies its build stage's output.
    Returns None on a fresh checkout that hasn't run `npm run build` yet — the
    API still serves, so backend tests don't need Node.
    """
    for candidate in (os.getenv("UI_DIST"), "ui/dist"):
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


UI_DIR = _ui_directory()
if UI_DIR:
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


@app.get("/")
def read_root(request: Request):
    """Redirect root to the UI, query string intact.

    Sign-in links point at the site root (``/?auth=…``). Dropping the query
    here would strip the token and land every emailed link back on the login
    modal — invisible in production, where Firebase Hosting serves the SPA at
    the root, but broken everywhere the proxy is the origin.
    """
    if not UI_DIR:
        raise HTTPException(
            status_code=503,
            detail="UI bundle not built. Run `npm ci && npm run build` in ui/.")
    query = request.url.query
    return RedirectResponse(url="/static/index.html" + (f"?{query}" if query else ""))

@app.get("/health")
def health_check():
    """Health probe."""
    return {"status": "ok"}


# ── Dev-only session bootstrap ────────────────────────────────────────────────
# Only mounted when ACCESS_STORE=memory (local / CI). Never reaches production
# because production never sets ACCESS_STORE=memory.

if os.getenv("ACCESS_STORE") == "memory":
    @app.get("/dev/session")
    def dev_session(email: str = "browser-test@local.dev"):
        """Return a live session token for an approved test user.

        Skips the email magic-link flow so automated browser tests can sign in
        without an inbox. Removed from the app when ACCESS_STORE != memory.
        """
        record, _ = access.create_or_touch_request(email)
        access.set_status(email, "approved")
        record = access.get_record(email, cached=False)
        token = access.mint_session(email, record)
        return {"token": token, "email": email}

    @app.get("/dev/autologin")
    async def dev_autologin(email: str = "browser-test@local.dev"):
        """Redirect browser through the real ?auth= magic-link flow.

        Creates an approved user, issues a single-use login nonce, and redirects
        to /?auth=<token> so the SPA calls /auth/exchange and stores the session
        in sessionStorage itself — no JS injection required.
        Only available when ACCESS_STORE=memory (never in production).
        """
        access.create_or_touch_request(email)
        access.set_status(email, "approved")
        nonce = access.issue_login_nonce(email)
        link = access.login_link(email, nonce)
        return RedirectResponse(url=link)


# ── Access API ───────────────────────────────────────────────────────────────
#
# One email field on the client drives all of this: the visitor types an
# address and the server decides whether that is a sign-in or an access
# request. See proxy/access.py for the trust model.


class LoginRequest(BaseModel):
    email: str


class ExchangeRequest(BaseModel):
    auth: str


class ExtensionRequest(BaseModel):
    message: Optional[str] = None


def _access_payload(email: str, state: access.AccessState) -> dict:
    return {
        "status": state.code,
        "email": email,
        "tokens_used": state.usage,
        "token_limit": state.limit,
        "extension_status": state.extension_status,
    }


@app.post("/auth/login")
async def auth_login(body: LoginRequest):
    """Sign in, or request access — the record decides which.

    Deliberately returns the same shape whichever branch runs, and never
    reveals whether an address was already known: the response for an unknown
    address and a freshly created one are identical.
    """
    email = access.normalize_email(body.email)
    if not email:
        raise HTTPException(status_code=400, detail="Enter a valid email address")

    record, should_notify = access.create_or_touch_request(email)

    if record.get("status") == "approved":
        # Resending is the same operation as signing in, so the UI's "Send
        # another link" reuses this path rather than duplicating it. The
        # cooldown lives here because every resend issues a fresh nonce, which
        # invalidates the previous link: without it, an impatient visitor
        # clicking twice kills the link they are waiting on and ends up holding
        # several dead ones.
        waited = access.seconds_since(record.get("login_sent_at"))
        if waited < access.RESEND_COOLDOWN_S:
            raise HTTPException(
                status_code=429,
                detail=f"A sign-in link was just sent. Please wait "
                       f"{int(access.RESEND_COOLDOWN_S - waited)}s before asking "
                       f"for another, then check your spam folder too.")

        nonce = access.issue_login_nonce(email)
        sent, error = await access.send_login_link(email, access.login_link(email, nonce))
        if not sent:
            # "Check your inbox" for a mail that was never accepted by any
            # provider sends the visitor to hunt through spam for something
            # that does not exist. The delivery detail stays server-side (it
            # can carry credentials and provider internals); the visitor gets
            # the one fact that changes what they do next.
            access.log(f"ERROR event=login_link_failed email={email} {error}")
            raise HTTPException(
                status_code=502,
                detail="Could not send the sign-in email just now. Please try again "
                       "in a moment — if it keeps failing, the site owner has been "
                       "notified.")
        # Stamped only after a confirmed send, so a failed delivery doesn't
        # lock the visitor out of retrying for the length of the cooldown.
        access.update_record(email, {"login_sent_at": access.utcnow()})
        return {"status": "link_sent", "email": email}

    if should_notify:
        await access.notify_access_request(email)

    if record.get("status") == "denied":
        return {"status": "denied", "email": email}
    return {"status": "pending", "email": email}


@app.post("/auth/exchange")
async def auth_exchange(body: ExchangeRequest):
    """Trade a single-use login link for a 30-day session."""
    payload = access.unsign(body.auth, access.PURPOSE_LOGIN)
    if not payload:
        raise HTTPException(status_code=401, detail="This sign-in link has expired")
    email = access.normalize_email(payload.get("email"))
    record = access.get_record(email, cached=False) if email else None
    if not record:
        raise HTTPException(status_code=401, detail="This sign-in link is no longer valid")
    if record.get("status") != "approved":
        raise HTTPException(status_code=403, detail=access.check_access(
            {"email": email}).as_detail())
    if not access.consume_login_nonce(email, payload.get("n", "")):
        raise HTTPException(
            status_code=401,
            detail="This sign-in link has already been used — request a new one")

    state = access.check_access({"email": email})
    return {"token": access.mint_session(email, record), **_access_payload(email, state)}


@app.get("/access/status")
async def access_status(user: Optional[dict] = Depends(get_caller)):
    """Current gate state — polled by the UI while a request is pending."""
    if not user:
        return {"status": "logged_out"}
    email = user["email"]
    record = access.get_record(email, cached=False)
    # Heal a notification that failed earlier, so a Resend blip doesn't leave
    # someone waiting on an email that will never arrive.
    if record and record.get("notify_state") == "failed" and record.get("status") == "pending":
        if int(record.get("notify_attempts", 0)) < access.NOTIFY_MAX_ATTEMPTS:
            await access.notify_access_request(email)
    return _access_payload(email, access.check_access(user))


@app.post("/access/extension")
async def access_extension(body: ExtensionRequest, user: Optional[dict] = Depends(get_caller)):
    """Ask for more tokens. The caller stays blocked until it's granted."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    email = user["email"]
    record = access.request_extension(email, body.message)
    if record is None:
        raise HTTPException(status_code=404, detail="No access record for this address")
    await access.notify_extension_request(
        email, body.message,
        int(record.get("tokens_used", 0)), int(record.get("token_limit", 0)))
    return _access_payload(email, access.check_access(user))


@app.delete("/account")
async def delete_account(user: Optional[dict] = Depends(get_caller)):
    """Self-serve erasure: the access record and every conversation, gone."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    access.delete_user(user["email"])
    return {"status": "deleted"}


# ── History API ──────────────────────────────────────────────────────────────

@app.get("/history")
async def list_history(
    cursor: Optional[str] = None,
    limit: int = 30,
    user: Optional[dict] = Depends(get_caller),
):
    """List the signed-in user's conversations, most recent first.

    Pass the previous page's ``next_cursor`` (an ISO-8601 ``updated_at``) to
    page backwards in time. ``next_cursor`` is null on the last page.

    Only the last 24 hours are ever here — conversations expire (see
    ``_chat_expiry``), so this is a rolling window rather than an archive.
    """
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    page_size = max(1, min(limit, 100))
    db = get_db()
    query = (
        db.collection("users").document(user["email_key"]).collection("conversations")
        .order_by("updated_at", direction=gcf.Query.DESCENDING)
    )
    if cursor:
        try:
            # A '+' in the UTC offset decodes to a space in a query string, so
            # accept the cursor whether or not the caller percent-encoded it.
            after = datetime.fromisoformat(cursor.replace(" ", "+"))
        except ValueError:
            # 400 rather than silently restarting at page 1: a silent reset
            # would make the sidebar's "load older" page forever.
            raise HTTPException(status_code=400, detail="Malformed cursor")
        query = query.start_after({"updated_at": after})
    query = query.limit(page_size)

    conversations = []
    for doc in query.stream():
        data = doc.to_dict() or {}
        updated_at = data.get("updated_at")
        conversations.append({
            "chat_id": doc.id,
            "title": data.get("title") or "(untitled)",
            "total_tokens": data.get("total_tokens", 0),
            "updated_at": updated_at.isoformat() if hasattr(updated_at, "isoformat") else updated_at,
        })
    # A short page means we reached the end; a full one may or may not have
    # more, so hand back a cursor and let the next call settle it.
    next_cursor = conversations[-1]["updated_at"] if len(conversations) == page_size else None
    return {"conversations": conversations, "next_cursor": next_cursor}

@app.get("/history/{chat_id}")
async def get_history(chat_id: str, user: Optional[dict] = Depends(get_caller)):
    """Return the messages of one of the signed-in user's conversations."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    db = get_db()
    conv_ref = (
        db.collection("users").document(user["email_key"])
        .collection("conversations").document(chat_id)
    )
    snapshot = conv_ref.get()
    if not snapshot.exists:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conv = snapshot.to_dict() or {}
    messages = []
    for doc in conv_ref.collection("messages").order_by("created_at").stream():
        data = doc.to_dict() or {}
        message = {"role": data.get("role"), "content": data.get("content")}
        # Both are absent on most docs: attachments only on user messages that
        # had them, causal only on AI messages from a causal run.
        if data.get("attachments"):
            message["attachments"] = data["attachments"]
        if data.get("causal"):
            message["causal"] = data["causal"]
        messages.append(message)
    return {
        "chat_id": chat_id,
        "title": conv.get("title"),
        "total_tokens": conv.get("total_tokens", 0),
        "messages": messages,
    }

# ── Upload API ───────────────────────────────────────────────────────────────
#
# Dev-first storage: uploads live in an in-process dict (optionally mirrored to
# UPLOAD_DIR so `uvicorn --reload` restarts keep them). On Cloud Run this store
# is per-instance and ephemeral — if the service ever scales past one instance,
# swap _put_upload/_get_upload for a GCS-backed implementation keyed by
# uploads/{uid}/{file_id}; the call sites don't need to change.

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
MAX_ATTACHMENT_TEXT_CHARS = 200_000
ALLOWED_UPLOAD_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".yaml", ".yml",
    ".toml", ".xml", ".html", ".css", ".js", ".ts", ".py", ".java", ".go",
    ".rs", ".c", ".cpp", ".h", ".sh", ".sql", ".log",
}

_uploads: dict = {}


def _upload_dir() -> Optional[str]:
    path = os.getenv("UPLOAD_DIR")
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def _sweep_uploads() -> None:
    """Drop expired uploads from the process-local index.

    Entries used to be evicted only when someone read an already-expired id, so
    an upload nobody referenced again stayed resident for the life of the
    instance — a slow leak bounded only by MAX_UPLOAD_BYTES x how long the
    revision lives.
    """
    for file_id in [k for k, v in _uploads.items() if _expired_upload(v)]:
        _uploads.pop(file_id, None)


def _put_upload(record: dict) -> str:
    file_id = uuid.uuid4().hex
    # Cheap: runs once per upload, over a dict holding at most a day of them.
    _sweep_uploads()
    _uploads[file_id] = record
    directory = _upload_dir()
    if directory:
        with open(os.path.join(directory, f"{file_id}.json"), "w", encoding="utf-8") as fh:
            json.dump(record, fh)
    return file_id


def _expired_upload(record: dict) -> bool:
    """Uploads ride the same 24-hour clock as the chats they belong to."""
    return access.seconds_since(record.get("expires_at")) > 0


def _get_upload(file_id: str) -> Optional[dict]:
    record = _uploads.get(file_id)
    if record is None:
        directory = _upload_dir()
        if directory and re.fullmatch(r"[0-9a-f]{32}", file_id or ""):
            sidecar = os.path.join(directory, f"{file_id}.json")
            if os.path.exists(sidecar):
                with open(sidecar, encoding="utf-8") as fh:
                    record = json.load(fh)
                _uploads[file_id] = record
    if record is None:
        return None
    if _expired_upload(record):
        _uploads.pop(file_id, None)
        return None
    return record


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), user: Optional[dict] = Depends(get_caller)):
    """Accept a text-extractable file and return an id to reference in chat."""
    # Gated like the agent itself: without this the upload endpoint is an open
    # file sink for anyone who finds the URL.
    access.require_access(user)
    filename = os.path.basename(file.filename or "").strip() or "upload"
    extension = os.path.splitext(filename)[1].lower()
    if extension not in ALLOWED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS))
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{extension or filename}'. Allowed: {allowed}",
        )
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )
    text = data.decode("utf-8", errors="replace")[:MAX_ATTACHMENT_TEXT_CHARS]
    now = datetime.now(timezone.utc)
    file_id = _put_upload({
        "filename": filename,
        "size": len(data),
        "content_type": file.content_type or "text/plain",
        "text": text,
        "owner_uid": user["email_key"] if user else None,
        "created_at": now.isoformat(),
        "expires_at": _chat_expiry(now).isoformat(),
    })
    return {
        "file_id": file_id,
        "filename": filename,
        "size": len(data),
        "content_type": file.content_type or "text/plain",
        "text_chars": len(text),
    }


def _resolve_attachments(ids: list, user: Optional[dict]) -> list:
    """Look up attachment ids, enforcing uploader-only access.

    Unknown ids and other users' uploads both 404 so ids aren't probeable.
    """
    owner = user["email_key"] if user else None
    files = []
    for file_id in ids or []:
        record = _get_upload(file_id)
        if record is None or record.get("owner_uid") != owner:
            raise HTTPException(status_code=404, detail=f"Unknown attachment id: {file_id}")
        files.append(record)
    return files


def _attachment_context(files: list) -> str:
    """Render attached file contents as context blocks for the agent."""
    if not files:
        return ""
    blocks = [
        f"--- Attached file: {f['filename']} ---\n{f['text']}\n--- End of file: {f['filename']} ---"
        for f in files
    ]
    return "\n".join(blocks) + "\n\n"


# ── Agent Proxy API ─────────────────────────────────────────────────────────

# Must match src/causal/state.py (the proxy image does not ship src/).
#
# There is no mode marker here. TracerLensAi routed between a general assistant
# and a causal pipeline on `[[causal:on]]`; this app has one pathway, so every
# turn is a causal turn and there is nothing to signal. The run-id marker stays,
# because correlation is still needed.
CAUSAL_STATE_PREFIX = "causal_"

# Per-turn correlation id. Minted by the UI, echoed on the report, stored on the
# history document, injected to the agent on the marker channel, and logged
# beside the Cloud Trace context — so one id joins what the user saw, what the
# agent recorded, and what the platform traced.
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _safe_run_id(value: Optional[str]) -> str:
    """Accept only an id shaped like one we minted; otherwise mint a fresh one.

    The value reaches the agent inside the message text, so it is never
    interpolated unvalidated.
    """
    if value and _RUN_ID_RE.match(value):
        return value
    return uuid.uuid4().hex


# ── SSE transport ───────────────────────────────────────────────────────────
# /analyze-prompt streams the causal pipeline's progress instead of buffering
# the whole run. Frames: `progress` (stage + newly appended trace lines),
# `graph` (a to_ui_graph payload, one per graph mutation), `done` (the full
# report), `error`. The `done` payload is a superset of the JSON body this
# endpoint used to return, so history persistence and replay are unchanged.

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Without this, nginx-style intermediaries buffer the whole body and the
    # client sees a single flush at the end — which defeats the point.
    "X-Accel-Buffering": "no",
}

# Emitted when the upstream goes quiet, so an intermediary doesn't reap a
# connection during a slow LLM stage (the decomposer alone can run ~30s).
SSE_PING_INTERVAL_S = 15.0

# Pipeline stages, keyed on the LangGraph node name.
#
# With stream_mode="updates" the agent yields one {node_name: state_update} dict
# per node that ran, so the node name plays exactly the role the ADK event
# `author` used to: it says which stage produced this delta.
STAGE_BY_NODE = {
    "interpret_query": "interpret",
    "validate_query": "validate",
    # The three compute nodes are alternatives — only one runs per turn — so
    # they collapse to a single stage rather than three that mostly stay grey.
    "optimal_policy": "compute",
    "intervention_effect": "compute",
    "posterior_summary": "compute",
    "explain_result": "explain",
    "clarification": "explain",
    "error": "explain",
}


def _sse(event: str, data) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def _resolve_stage(node: Optional[str]) -> Optional[str]:
    """Map a LangGraph chunk's node name to a pipeline stage id."""
    return STAGE_BY_NODE.get(node or "")


def _iter_node_deltas(chunk: dict):
    """Yield (node_name, delta) pairs from one streamed chunk.

    LangGraph emits ``{node_name: partial_state}`` under stream_mode="updates".
    Normally that is a single entry, but parallel branches can produce several
    in one chunk, so this iterates rather than assuming one. Anything that is
    not a dict-of-dicts is skipped: LanggraphAgent runs chunks through `dumpd`,
    and a non-conforming payload should be ignored rather than crash the stream.
    """
    if not isinstance(chunk, dict):
        return
    for node, delta in chunk.items():
        if isinstance(node, str) and isinstance(delta, dict):
            yield node, delta


def _merge_usage(totals: dict, value) -> None:
    """Add one node's token counts into the running totals, in place.

    ADK emitted `usage_metadata` on every event; LangChain hangs it off the
    AIMessage instead, and our nodes do not put messages in state — so the graph
    reports what it spent under `causal_usage` and this sums it. Without this
    every turn bills as zero and the quota gate never trips.

    Summed, not assigned: under stream_mode="updates" LangGraph yields the value
    the node *returned*, not the reduced channel, so each chunk carries only its
    own call's usage. (The graph's add_usage reducer produces the total for the
    final state; the stream never shows it.)
    """
    if not isinstance(value, dict):
        return
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        count = value.get(key)
        if isinstance(count, int):
            totals[key] = totals.get(key, 0) + count


# Pacing for the offline mock stream. Small enough that the E2E suite stays
# fast, large enough that a human can watch the stages land. Read per-call so
# unit tests can set MOCK_FRAME_DELAY_S=0 and skip the pacing entirely.
MOCK_FRAME_DELAY_S = 0.15


async def _mock_stream(report: dict):
    """Scripted SSE stream for offline dev and the E2E suite.

    Replays the *shape* of a real run: the four pipeline stages in order, the
    model graph appearing as soon as the query is interpreted, and its cause
    nodes resolving from pending to their final clamped/stochastic status. The
    final `done` frame is the same report the endpoint would otherwise return.

    Unlike the ADK version this walks only the *cause* nodes. The real pipeline
    never puts a cause into "active" as a transition — status there means
    "intervened on" — so animating every node through active would teach the
    offline path a state the live one cannot produce.
    """
    delay = float(os.getenv("MOCK_FRAME_DELAY_S", MOCK_FRAME_DELAY_S))
    graph = report.get("causal_graph") or {}
    nodes = list(graph.get("nodes") or [])
    # Consumed left-to-right so each trace line lands under the stage that
    # would really have produced it.
    pending_steps = list(report.get("causal_reasoning_steps") or [])
    final_status = {n["id"]: n.get("status") or "pending" for n in nodes}
    statuses = {n["id"]: "pending" for n in nodes}
    elapsed = 0

    def _graph_frame():
        return _sse("graph", {**graph, "nodes": [
            {**n, "status": statuses[n["id"]]} for n in nodes]})

    def _take_step():
        return [pending_steps.pop(0)] if pending_steps else []

    script = [
        ("interpret", "interpreting"),
        ("validate", "validating"),
        ("compute", "computing"),
    ]

    for index, (stage, phase) in enumerate(script):
        await asyncio.sleep(delay)
        elapsed += int(delay * 1000)
        yield _sse("progress", {"stage": stage, "phase": phase,
                                "steps": _take_step(), "current_step": None,
                                "elapsed_ms": elapsed})
        if stage == "interpret":
            # The graph is drawable the moment the query is understood, which
            # is exactly when the live pipeline emits it.
            yield _graph_frame()

    # Resolve the causes one at a time so the pane has real transitions.
    causes = [n for n in nodes if n.get("kind") == "input"]
    for index, node in enumerate(causes, start=1):
        await asyncio.sleep(delay)
        elapsed += int(delay * 1000)
        statuses[node["id"]] = final_status[node["id"]]
        yield _sse("progress", {
            "stage": "compute", "phase": "computing",
            "steps": _take_step(),
            "current_step": {"id": node["id"], "index": index,
                             "total": len(causes)},
            "elapsed_ms": elapsed})
        yield _graph_frame()

    # The decision and utility nodes settle together: both are determined once
    # the argmax is taken.
    for node in nodes:
        statuses[node["id"]] = final_status[node["id"]]
    yield _graph_frame()

    await asyncio.sleep(delay)
    elapsed += int(delay * 1000)
    # Anything left over lands on the final stage rather than being dropped.
    yield _sse("progress", {"stage": "explain", "phase": "explaining",
                            "steps": pending_steps, "current_step": None,
                            "elapsed_ms": elapsed})
    yield _sse("done", report)


class PromptRequest(BaseModel):
    """Request body from the UI.

    ``causal_reasoning`` is retained as an accepted field but no longer switches
    anything: this app has a single pathway and every turn runs it. Keeping the
    field means an older UI build, or a replayed request, still validates.
    """
    prompt: str
    causal_reasoning: bool = True
    model_name: str = "gemini-3.6-flash"
    chat_id: Optional[str] = None  # Maps to Agent Session ID
    attachments: list = Field(default_factory=list)  # file_ids from POST /upload
    run_id: Optional[str] = None   # correlation id for this turn (see _safe_run_id)


def _causal_payload(report: dict) -> Optional[dict]:
    """Pull the causal_* fields out of a response body for persistence.

    Returns None when the turn produced nothing causal, so non-causal messages
    keep their original document shape.
    """
    payload = {key: value for key, value in report.items()
               if key.startswith(CAUSAL_STATE_PREFIX) and value}
    return payload or None


def _persist_if_signed_in(user: Optional[dict], req: "PromptRequest", response_text: str, token_count: int,
                          attachment_names: Optional[list] = None, causal: Optional[dict] = None,
                          run_id: Optional[str] = None):
    """Best-effort history write; never fails the chat response."""
    if not user or not req.chat_id:
        return
    try:
        _save_exchange(user, req.chat_id, req.prompt, response_text, token_count,
                       attachment_names, causal, run_id)
    except Exception as e:
        access.log(f"WARNING: failed to persist history for {user['email_key']}: {e}")


def _record_turn(user: Optional[dict], req: "PromptRequest", *, ok: bool, started: float,
                 tokens_in: int = 0, tokens_out: int = 0, tokens_total: int = 0,
                 error_kind: Optional[str] = None):
    """Charge a turn's tokens to the caller and log it for the dashboard.

    The single place usage is incremented, so the quota can't drift from what
    actually ran. Both calls are best-effort inside proxy/access.py: a metrics
    write must never be the reason a paid-for answer fails to arrive.
    """
    if not user:
        return
    access.record_usage(user["email"], tokens_total)
    access.record_run(
        user["email"], ok=ok, latency_ms=int((time.monotonic() - started) * 1000),
        tokens_in=tokens_in, tokens_out=tokens_out, tokens_total=tokens_total,
        # causal=True unconditionally: this app has one pathway. web=False —
        # there is no retrieval branch. Both are kept in the call so the admin
        # dashboard's run records keep their existing columns.
        model=req.model_name, causal=True, web=False,
        error_kind=error_kind)


@app.post("/analyze-prompt")
async def analyze_prompt(
    req: PromptRequest,
    user: Optional[dict] = Depends(get_caller),
    x_anon_id: Optional[str] = Header(None),
    x_cloud_trace_context: Optional[str] = Header(None),
):
    """Proxy the request to the Vertex AI Agent Engine (streaming)."""

    # Before anything costs money: 403 with a machine-readable code the UI
    # turns into the right modal (waiting for approval, or quota reached).
    # TEMP: access gate disabled for local docker-compose smoke testing.
    # Re-enable before any real deploy: access.require_access(user)

    agent_engine_base = os.getenv("AGENT_ENGINE_ENDPOINT")
    started = time.monotonic()
    attachment_files = _resolve_attachments(req.attachments, user)
    attachment_names = [f["filename"] for f in attachment_files]

    # One line per turn carrying the correlation id and the platform's trace id
    # (Cloud Run sets X-Cloud-Trace-Context), so a run in the UI can be found in
    # Cloud Logging and followed into Cloud Trace.
    run_id = _safe_run_id(req.run_id)
    trace_id = (x_cloud_trace_context or "").split("/")[0] or "-"
    print(f"run_id={run_id} trace_id={trace_id} chat_id={req.chat_id or '-'} "
          f"attachments={len(attachment_names)}")

    if not agent_engine_base:
        # Mock response for local development if Agent Runtime is not configured
        mock_text = "Agent Proxy configured. (Set AGENT_ENGINE_ENDPOINT to connect to Agent Runtime). Prompt: " + req.prompt
        if attachment_names:
            # Deterministic acknowledgment so UI/E2E tests can verify uploads.
            mock_text += f"\n\nAttached files ({len(attachment_names)}): {', '.join(attachment_names)}"
        mock_graph = MOCK_GRAPH
        mock_steps = list(MOCK_STEPS)
        mock_decision = MOCK_DECISION
        mock_posteriors = MOCK_POSTERIORS
        report = {
            "status": "success",
            "response": mock_text,
            "run_id": run_id,
            "total_token_count": 10,
            "causal_reasoning_steps": mock_steps,
            "causal_graph": mock_graph,
            "causal_status": {"phase": "complete", "query_type": "optimal_policy"},
            "causal_decision": mock_decision,
            "causal_posteriors": mock_posteriors,
        }
        # Persist from the same dict that goes back to the UI so the offline
        # dev path exercises causal history, quota accounting and metrics too.
        _persist_if_signed_in(user, req, mock_text, 10, attachment_names,
                              _causal_payload(report), run_id)
        _record_turn(user, req, ok=True, started=started,
                     tokens_in=6, tokens_out=4, tokens_total=10)
        return StreamingResponse(
            _mock_stream(report), media_type="text/event-stream",
            headers=SSE_HEADERS)

    # Derive the streaming endpoint from the base query URL
    # e.g. .../reasoningEngines/ID:query  →  .../reasoningEngines/ID:streamQuery
    stream_url = agent_engine_base.replace(":query", ":streamQuery")

    # Use Application Default Credentials (ADC) for Vertex AI auth
    try:
        credentials = await _agent_credentials()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to obtain ADC credentials: {e}")

    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json"
    }

    # Attached file contents ride as context blocks ahead of the prompt. The
    # correlation id rides in the message text on the marker channel; the graph
    # strips it into causal_run_id so every write this turn is attributable.
    # The clean prompt (req.prompt) is what gets persisted.
    outbound_message = (
        f"[[run:{run_id}]] {_attachment_context(attachment_files)}{req.prompt}"
    )

    payload = {
        "class_method": "stream_query",
        "input": {
            # LanggraphAgent forwards `input` straight into the compiled graph.
            # A bare string would be wrapped as {"input", "messages"}, which our
            # state does not read — so pass the mapping the graph expects.
            "input": {"user_input": outbound_message},
            # Without this LangGraph 1.x streams "values" (the entire state on
            # every step), and the trace-line high-water mark below would resend
            # every line each frame.
            "stream_mode": "updates",
        },
    }

    return StreamingResponse(
        _agent_stream(req, user, attachment_names, stream_url, payload, headers,
                      started, run_id),
        media_type="text/event-stream", headers=SSE_HEADERS)


# ── Agent credentials ────────────────────────────────────────────────────────
#
# ADC used to be resolved and refreshed on every /analyze-prompt. Both calls are
# synchronous: google.auth.default() walks the filesystem or metadata server and
# .refresh() is a blocking HTTPS round trip. Inside an `async def`, on a
# single-worker container serving up to 80 concurrent requests, each one froze
# the whole event loop — every other in-flight SSE stream included — to re-mint
# a token that stays valid for an hour.

_CREDENTIALS = None
_CREDENTIALS_LOCK = asyncio.Lock()
# Refresh a little early rather than racing the boundary mid-request.
_CREDENTIALS_SKEW_S = 300


async def _agent_credentials():
    """ADC for the Agent Engine call, resolved once and refreshed on expiry.

    The lock keeps a burst of concurrent turns from refreshing N times over;
    the refresh itself runs in a worker thread so the loop keeps serving.
    """
    global _CREDENTIALS

    def _fresh_enough(creds) -> bool:
        if creds is None or not getattr(creds, "token", None):
            return False
        expiry = getattr(creds, "expiry", None)
        if expiry is None:
            # Nothing to measure against — some credential types never set one.
            return True
        # google-auth stores expiry as a naive UTC datetime.
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        return (expiry - now).total_seconds() > _CREDENTIALS_SKEW_S

    if _fresh_enough(_CREDENTIALS):
        return _CREDENTIALS

    async with _CREDENTIALS_LOCK:
        if _fresh_enough(_CREDENTIALS):  # another task won the race
            return _CREDENTIALS
        creds = _CREDENTIALS
        if creds is None:
            creds, _ = await asyncio.to_thread(
                google.auth.default,
                scopes=["https://www.googleapis.com/auth/cloud-platform"])
        await asyncio.to_thread(
            creds.refresh, google.auth.transport.requests.Request())
        _CREDENTIALS = creds
        return _CREDENTIALS


async def _pump_lines(resp, queue: asyncio.Queue):
    """Feed upstream NDJSON lines into a queue.

    The read runs in its own task so the emitting side can wait on the queue
    with a timeout and send keepalives. Racing a timeout directly against
    ``aiter_lines()`` would cancel a partially-consumed read and corrupt the
    stream; cancelling ``queue.get()`` is safe.
    """
    try:
        async for line in resp.aiter_lines():
            await queue.put(("line", line))
    except Exception as e:  # upstream died mid-stream
        await queue.put(("error", str(e)))
    finally:
        await queue.put(("eof", None))


async def _agent_stream(req: "PromptRequest", user: Optional[dict],
                        attachment_names: list, stream_url: str,
                        payload: dict, headers: dict, started: float, run_id: str):
    """Forward the Agent Engine run as SSE, then emit the assembled report."""
    collected_text = []
    causal_state = {}
    # One entry per node the graph ran, in order. Derived here rather than sent
    # by the agent: a chunk arrives when its node finishes, so the gap between
    # consecutive arrivals is that node's execution time. That keeps the agent's
    # node returns untouched — the graph reports no timing of its own — at the
    # cost of including transport in the measurement, which is why the UI labels
    # these as measured at the proxy. The authoritative per-node duration is the
    # `duration_ms` in the agent's own structured logs.
    node_runs: list = []
    last_boundary = started
    # Summed across the turn's LLM calls by _merge_usage; the three counts below
    # are derived from it after every chunk so a stream that dies mid-run still
    # bills for what it burned.
    usage_totals: dict = {}
    total_token_count = 0
    prompt_token_count = 0
    candidates_token_count = 0
    last_stage = None

    recorded = False

    def _record(ok: bool, error_kind: Optional[str] = None):
        """Record this turn exactly once, whatever way the stream ends.

        Every exit path routes through here so a turn can never be billed
        twice, and — more importantly — can never escape being billed at all.
        """
        nonlocal recorded
        if recorded:
            return
        recorded = True
        _record_turn(user, req, ok=ok, started=started, error_kind=error_kind,
                     tokens_in=prompt_token_count, tokens_out=candidates_token_count,
                     tokens_total=total_token_count)

    def _fail(kind: str):
        """Record a failed turn. Tokens already burned still count."""
        _record(ok=False, error_kind=kind)

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", stream_url, json=payload, headers=headers) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    _fail("upstream_http")
                    yield _sse("error", {
                        "detail": f"Agent Engine error: {body.decode()}"})
                    return

                # Start the per-node clock here rather than at `started`:
                # everything before this point is ADC, DNS and TLS, and charging
                # it to the first node overstated interpret_query by ~600ms.
                last_boundary = time.monotonic()

                queue: asyncio.Queue = asyncio.Queue()
                pump = asyncio.create_task(_pump_lines(resp, queue))
                try:
                    while True:
                        try:
                            kind, value = await asyncio.wait_for(
                                queue.get(), timeout=SSE_PING_INTERVAL_S)
                        except asyncio.TimeoutError:
                            yield ": ping\n\n"
                            continue
                        if kind == "eof":
                            break
                        if kind == "error":
                            _fail("stream_failed")
                            yield _sse("error", {"detail": f"Agent Engine stream failed: {value}"})
                            return

                        line = (value or "").strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except ValueError:
                            collected_text.append(line)  # plain text line
                            continue

                        # LangGraph streams {node_name: state_update} under
                        # stream_mode="updates". A chunk normally carries one
                        # entry; parallel branches can carry several, so this
                        # merges them all before deciding what to forward.
                        #
                        # Critically, the value for a channel is what the node
                        # *returned*, not the reduced channel — so the two
                        # channels with reducers have to be reduced again here.
                        graph_payload = None
                        stage = None
                        new_steps: list = []
                        new_runs: list = []

                        for node, delta in _iter_node_deltas(event):
                            resolved = _resolve_stage(node)
                            if resolved:
                                stage = resolved

                            now = time.monotonic()
                            run = {
                                "node": node,
                                "stage": resolved,
                                "duration_ms": int((now - last_boundary) * 1000),
                                "at_ms": int((now - started) * 1000),
                            }
                            # Per-call usage, not the running total: under
                            # stream_mode="updates" a chunk carries only what
                            # its own node spent, which is exactly the split the
                            # summed figure loses.
                            spent = delta.get("causal_usage")
                            if isinstance(spent, dict) and spent:
                                run["usage"] = {
                                    "prompt_tokens": spent.get("input_tokens"),
                                    "completion_tokens": spent.get("output_tokens"),
                                    "total_tokens": spent.get("total_tokens"),
                                }
                            node_runs.append(run)
                            new_runs.append(run)
                            last_boundary = now

                            for key, state_value in delta.items():
                                if not key.startswith(CAUSAL_STATE_PREFIX):
                                    continue
                                if key == "causal_steps":
                                    # Accumulate: each chunk carries only that
                                    # node's own lines, so assigning would leave
                                    # the report holding the last node's alone.
                                    lines = state_value or []
                                    new_steps.extend(lines)
                                    causal_state.setdefault(key, []).extend(lines)
                                elif key == "causal_usage":
                                    _merge_usage(usage_totals, state_value)
                                else:
                                    causal_state[key] = state_value

                            if delta.get("causal_graph"):
                                graph_payload = delta["causal_graph"]

                        prompt_token_count = usage_totals.get("input_tokens", 0)
                        candidates_token_count = usage_totals.get("output_tokens", 0)
                        total_token_count = usage_totals.get("total_tokens", 0)

                        # ── Forward this chunk as progress ──────────────────
                        # No high-water mark needed: a chunk only ever carries
                        # lines this node just produced, so they are new by
                        # construction.
                        status = causal_state.get("causal_status")
                        phase = status.get("phase") if isinstance(status, dict) else None

                        if stage or new_steps:
                            if stage:
                                last_stage = stage
                            yield _sse("progress", {
                                "stage": stage or last_stage,
                                "phase": phase,
                                "steps": new_steps,
                                "nodes": new_runs,
                                "current_step": None,
                                "elapsed_ms": int((time.monotonic() - started) * 1000),
                            })

                        if graph_payload:
                            yield _sse("graph", graph_payload)
                finally:
                    pump.cancel()
    except (GeneratorExit, asyncio.CancelledError):
        # The client went away — the UI's Stop button aborts the fetch, which
        # closes this generator and raises GeneratorExit at the suspended
        # yield. Both of these derive from BaseException, so the `except
        # Exception` below never sees them and the accounting after this block
        # never runs. Tokens the upstream already burned still have to count,
        # or Stop is an unlimited-quota bypass. Re-raise: a generator must not
        # swallow its own teardown.
        _record(ok=False, error_kind="aborted")
        raise
    except Exception as e:
        traceback.print_exc()
        _fail("proxy_exception")
        yield _sse("error", {"detail": str(e)})
        return

    try:
        # Every terminal node writes causal_final_answer — explain_result,
        # clarification and error alike — so it is always the answer when the
        # run reached an end. collected_text is the fallback for a stream that
        # died mid-run and left only loose text behind.
        response_text = (
            causal_state.get("causal_final_answer") or "".join(collected_text)
        )
        response_text = response_text.strip() or "(no response)"

        report = {
            "status": "success",
            "response": response_text,
            "run_id": run_id,
            "total_token_count": total_token_count,
            "input_token_count": prompt_token_count,
            "output_token_count": candidates_token_count,
            "causal_reasoning_steps": causal_state.get("causal_steps") or [],
            "causal_graph": causal_state.get("causal_graph"),
            "causal_status": causal_state.get("causal_status"),
            "causal_decision": causal_state.get("causal_decision"),
            "causal_posteriors": causal_state.get("causal_posteriors"),
            # Prefixed, so _causal_payload carries it into Firestore with
            # everything else and a replayed turn shows the same node trace as
            # a live one.
            "causal_node_trace": node_runs,
        }
        # _causal_payload filters on the causal_ prefix, so the new keys
        # persist to Firestore and replay through history without extra work.
        _persist_if_signed_in(user, req, response_text, total_token_count, attachment_names,
                              _causal_payload(report), run_id)
        _record(ok=True)
    except Exception as e:
        traceback.print_exc()
        _fail("assembly_failed")
        yield _sse("error", {"detail": str(e)})
        return

    yield _sse("done", report)
