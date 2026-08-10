import json

import pytest

# The proxy stack (FastAPI, firebase-admin, httpx) is only needed by the proxy
# tests. The engine tests are hermetic — pure torch/pyro over pydantic — and
# must stay runnable in an environment that has the math but not the web layer,
# which is exactly the WSL kernel the notebook was validated on. A bare
# module-level import here fails collection for the *whole* suite instead, so
# the proxy fixtures are gated on the import succeeding.
try:
    from fastapi.testclient import TestClient

    import proxy.access as proxy_access
    import proxy.main as proxy_main
    from proxy import memstore
    from proxy.main import app

    PROXY_AVAILABLE = True
    _PROXY_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment-dependent
    PROXY_AVAILABLE = False
    _PROXY_IMPORT_ERROR = exc
    proxy_access = proxy_main = memstore = app = TestClient = None


def _need_proxy():
    if not PROXY_AVAILABLE:
        pytest.skip(f"proxy dependencies unavailable: {_PROXY_IMPORT_ERROR}")


@pytest.fixture
def client():
    _need_proxy()
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _fast_mock_stream(monkeypatch):
    """Drop the offline mock stream's pacing so unit tests don't sleep.

    The 150ms/frame default exists so a human (and the Playwright suite) can
    watch stages land; it is pure cost here.
    """
    monkeypatch.setenv("MOCK_FRAME_DELAY_S", "0")


@pytest.fixture(autouse=True)
def _clean_access_state(monkeypatch):
    """Isolate the access layer between tests.

    Also unsets every mail transport unconditionally: with real credentials in
    the developer's environment, any test that triggers a notification would
    send actual email. The notifier prints instead when none is configured.
    SMTP_* is covered as well as RESEND_API_KEY — SMTP takes precedence in
    send_email, so leaving it set would route test mail through a personal
    mailbox and silently defeat this guard.
    """
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    for name in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    if not PROXY_AVAILABLE:
        # Autouse, so it runs for the hermetic engine tests too. The env
        # scrubbing above is still worth doing; the module globals below do not
        # exist to reset.
        yield
        return
    proxy_access._record_cache.clear()
    # The proxy caches ADC across requests (re-minting a 1h token per turn used
    # to block the event loop). It is a module global, so without this a stub
    # credential from one test would still be serving the next one.
    proxy_main._CREDENTIALS = None
    yield
    proxy_access._record_cache.clear()
    proxy_main._CREDENTIALS = None


# ── Fake Firestore ───────────────────────────────────────────────
# The suite used to carry its own copy of this, class for class. The two had
# already drifted — the copy raised KeyError where memstore uses setdefault, and
# TypeError-sorted None where memstore filters — so the tests were asserting
# against semantics the shipped ACCESS_STORE=memory path does not have, and
# proxy/memstore.py itself had no coverage at all. Using the real thing fixes
# both halves of that.


@pytest.fixture
def fake_store(monkeypatch):
    """Swap Firestore for the fake.

    Both modules need patching: proxy/access.py owns ``get_db`` and calls it
    for the access records, while proxy/main.py holds an imported reference it
    uses for conversation history.
    """
    _need_proxy()
    store = memstore.MemoryDb()
    monkeypatch.setattr(proxy_access, "get_db", lambda: store)
    monkeypatch.setattr(proxy_main, "get_db", lambda: store)
    monkeypatch.setattr(proxy_access, "get_firebase_app", lambda: None)
    return store


@pytest.fixture
def no_email(monkeypatch):
    """Capture outbound email instead of sending (or printing) it."""
    sent = []

    async def _capture(to, subject, text, html=None):
        sent.append({"to": to, "subject": subject, "text": text})
        return True, ""

    monkeypatch.setattr(proxy_access, "send_email", _capture)
    return sent


# ── Access helpers ───────────────────────────────────────────────────────────

TEST_EMAIL = "visitor@example.com"


def approve_email(email=TEST_EMAIL, **overrides):
    """Create an approved access record and return it."""
    proxy_access.create_or_touch_request(email)
    proxy_access.set_status(email, "approved")
    if overrides:
        proxy_access.update_record(email, overrides)
    return proxy_access.get_record(email, cached=False)


def session_headers(email=TEST_EMAIL):
    """Authorization header for a live session belonging to ``email``."""
    record = proxy_access.get_record(email, cached=False) or {}
    return {"Authorization": "Bearer " + proxy_access.mint_session(email, record)}


@pytest.fixture
def approved(fake_store):
    """An approved visitor plus the headers to act as them."""
    approve_email()
    return session_headers()


# ── SSE helpers ──────────────────────────────────────────────────────────────
# /analyze-prompt streams Server-Sent Events. Tests assert on the frames rather
# than a JSON body.

def sse_frames(response):
    """Parse an SSE response body into a list of (event_name, payload)."""
    frames = []
    for block in response.text.split("\n\n"):
        if not block.strip():
            continue
        name = "message"
        data_lines = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if data_lines:  # skip ": ping" keepalive comments
            frames.append((name, json.loads("\n".join(data_lines))))
    return frames


def sse_report(response):
    """Return the terminal `done` payload — the report body of a run."""
    for name, payload in sse_frames(response):
        if name == "done":
            return payload
    raise AssertionError(
        f"no `done` frame in stream; frames were "
        f"{[n for n, _ in sse_frames(response)]}")
