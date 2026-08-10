"""Admin surface: two-factor login, review endpoints, and the dashboard.

Kept out of ``proxy/main.py``, which is already long enough, and out of the
React bundle — the dashboard is server-rendered HTML so it needs no build step
and adds no route to a deliberately router-less SPA.

Auth is a shared password (ADMIN_TOKEN) **plus** a one-time code emailed to
ACCESS_NOTIFY_EMAIL, so a leaked password alone is not enough. The one-click
Approve/Deny links in notification emails are exempt: they are signed,
expiring, and single-purpose, and reaching one already requires access to that
inbox — the same second factor an OTP would prove, at the cost of the one click
that makes them worth having.
"""
import hashlib
import hmac
import os
import secrets
import time
from datetime import timedelta
from typing import Optional

import httpx
import google.auth
import google.auth.transport.requests
from google.cloud import firestore as gcf
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from proxy import access

router = APIRouter()

OTP_TTL_S = 10 * 60
OTP_MAX_ATTEMPTS = 5
OTP_REQUEST_COOLDOWN_S = 30

_last_otp_request = {"at": 0.0}


# ── Two-factor login ─────────────────────────────────────────────────────────

def _admin_password() -> Optional[str]:
    return os.getenv("ADMIN_TOKEN")


def _hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8") + access.signing_secret()).hexdigest()


class OtpStart(BaseModel):
    password: str


class OtpVerify(BaseModel):
    challenge_id: str
    code: str


@router.post("/admin/auth/start")
async def admin_auth_start(body: OtpStart):
    """Check the password, then email a 6-digit code to the admin address."""
    password = _admin_password()
    if not password:
        raise HTTPException(status_code=503, detail="ADMIN_TOKEN is not configured")
    if not hmac.compare_digest(body.password or "", password):
        # Same delay-free generic error either way; the password is the only
        # thing being tested here and a distinct message would confirm it.
        raise HTTPException(status_code=401, detail="Invalid credentials")

    now = time.time()
    if now - _last_otp_request["at"] < OTP_REQUEST_COOLDOWN_S:
        raise HTTPException(status_code=429, detail="Please wait before requesting another code")
    _last_otp_request["at"] = now

    code = f"{secrets.randbelow(1000000):06d}"
    challenge_id = secrets.token_urlsafe(16)
    access.get_db().collection(access.OTP_COLLECTION).document(challenge_id).set({
        "code_hash": _hash_code(code),
        "expires_at": access.utcnow() + timedelta(seconds=OTP_TTL_S),
        "attempts": 0,
        "created_at": access.utcnow(),
    })
    await access.send_email(
        access.notify_email(),
        "Tracer Lens admin code",
        f"Your admin sign-in code is {code}\n\nIt expires in 10 minutes.\n"
        f"If you didn't try to sign in, someone has your admin password — rotate it.\n",
    )
    return {"challenge_id": challenge_id}


@router.post("/admin/auth/verify")
async def admin_auth_verify(body: OtpVerify):
    """Trade a correct code for a 12-hour admin session."""
    ref = access.get_db().collection(access.OTP_COLLECTION).document(body.challenge_id or "_")
    snapshot = ref.get()
    if not snapshot.exists:
        raise HTTPException(status_code=401, detail="Invalid or expired code")
    challenge = snapshot.to_dict() or {}

    if access.seconds_since(challenge.get("created_at")) > OTP_TTL_S:
        ref.delete()
        raise HTTPException(status_code=401, detail="Invalid or expired code")

    # Increment-then-read, not read-increment-write: parallel verifies would
    # otherwise all read attempts=0 and all write 1, so OTP_MAX_ATTEMPTS would
    # cap nothing and the six-digit space could be brute-forced concurrently.
    # Increment is applied atomically, so each caller reads back a count that
    # already includes every attempt racing it.
    ref.update({"attempts": gcf.Increment(1)})
    attempts = int((ref.get().to_dict() or {}).get("attempts", 0))
    if attempts > OTP_MAX_ATTEMPTS:
        ref.delete()
        raise HTTPException(status_code=429, detail="Too many attempts — start again")

    if not hmac.compare_digest(_hash_code(body.code or ""), str(challenge.get("code_hash"))):
        raise HTTPException(status_code=401, detail="Invalid or expired code")

    ref.delete()
    return {"token": access.sign({"admin": True}, access.PURPOSE_ADMIN_SESSION,
                                 access.ADMIN_SESSION_TTL_S)}


async def require_admin(authorization: Optional[str] = Header(None)) -> dict:
    """Dependency guarding every admin JSON endpoint."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Admin session required")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not access.unsign(token, access.PURPOSE_ADMIN_SESSION):
        raise HTTPException(status_code=401, detail="Admin session required")
    return {"admin": True}


# ── Review data ──────────────────────────────────────────────────────────────

def _public_record(record: dict) -> dict:
    """Shape one record for the dashboard, with derived stats."""
    runs = int(record.get("runs_total", 0) or 0)
    failed = int(record.get("runs_failed", 0) or 0)
    latency_sum = int(record.get("latency_ms_sum", 0) or 0)
    return {
        "email": record.get("email"),
        "key": record.get("key"),
        "status": record.get("status"),
        "tokens_used": int(record.get("tokens_used", 0) or 0),
        "token_limit": int(record.get("token_limit", 0) or 0),
        "extension_status": record.get("extension_status", "none"),
        "extension_message": record.get("extension_message"),
        "runs_total": runs,
        "runs_failed": failed,
        "failure_rate": round(failed / runs * 100, 1) if runs else 0.0,
        "avg_latency_ms": int(latency_sum / runs) if runs else 0,
        "notify_state": record.get("notify_state"),
        "notify_error": record.get("notify_error"),
        "requested_at": _iso(record.get("requested_at")),
        "last_run_at": _iso(record.get("last_run_at")),
        "last_seen": _iso(record.get("last_seen")),
    }


def _iso(value):
    return value.isoformat() if hasattr(value, "isoformat") else value


@router.get("/admin/users")
async def admin_users(_: dict = Depends(require_admin)):
    """Every record, plus the alerts the dashboard's top strip renders."""
    records = [_public_record(r) for r in access.list_records()]
    alerts = []
    for record in records:
        if record["notify_state"] == "failed":
            alerts.append({
                "kind": "notify_failed", "email": record["email"],
                "detail": record["notify_error"] or "notification could not be sent",
            })
    stale = [r for r in records
             if r["status"] == "pending" and access.seconds_since(r["requested_at"]) > 86400]
    for record in stale:
        alerts.append({"kind": "stale_request", "email": record["email"],
                       "detail": "pending for more than 24 hours"})
    return {
        "users": records,
        "pending": [r for r in records if r["status"] == "pending"],
        "extensions": [r for r in records if r["extension_status"] == "pending"],
        "alerts": alerts,
    }


@router.get("/admin/runs")
async def admin_runs(limit: int = 100, email_hash: Optional[str] = None,
                     _: dict = Depends(require_admin)):
    rows = access.list_runs(limit=max(1, min(limit, 500)), email_hash=email_hash)
    for row in rows:
        row["ts"] = _iso(row.get("ts"))
        row.pop("expires_at", None)
    return {"runs": rows}


@router.get("/admin/pending-count")
async def admin_pending_count(_: dict = Depends(require_admin)):
    """Heartbeat probe: how many requests are waiting, and for how long."""
    records = access.list_records()
    pending = [r for r in records if r.get("status") == "pending"]
    oldest = max((access.seconds_since(r.get("requested_at")) for r in pending), default=0)
    return {"pending": len(pending), "oldest_age_s": int(oldest if oldest != float("inf") else 0)}


# ── Decisions ────────────────────────────────────────────────────────────────

class EmailBody(BaseModel):
    email: str


def _require_sent(sent: bool, error: str, email: str, action: str) -> None:
    """Raise on a failed send instead of reporting success either way.

    Every one of these calls changes the record's status *before* attempting
    to notify — set_status/grant_extension already committed by the time this
    runs. So a failed send here does not roll that back; it makes the failure
    visible instead of silently telling the admin "Approved" while the
    visitor never got a way in. The dashboard's Alerts tab and /admin's
    printed-email fallback are the recovery paths this points them at.
    """
    if not sent:
        access.log(f"ERROR event={action}_email_failed email={email} {error}")
        raise HTTPException(
            status_code=502,
            detail=f"{action.title()} recorded, but the email failed to send "
                   f"({error}). The status change stuck — retry the notification "
                   f"from the Alerts tab, or resend from the dashboard.")


async def _approve(email: str) -> dict:
    """Approve and immediately email a sign-in link — approve, click, in."""
    record = access.set_status(email, "approved")
    if record is None:
        raise HTTPException(status_code=404, detail="No such request")
    nonce = access.issue_login_nonce(email)
    sent, error = await access.send_approval_email(email, access.login_link(email, nonce))
    _require_sent(sent, error, email, "approve")
    return {"status": "approved", "email": email}


async def _deny(email: str) -> dict:
    record = access.set_status(email, "denied")
    if record is None:
        raise HTTPException(status_code=404, detail="No such request")
    sent, error = await access.send_denied_email(email)
    _require_sent(sent, error, email, "deny")
    return {"status": "denied", "email": email}


async def _grant(email: str) -> dict:
    record = access.grant_extension(email)
    if record is None:
        raise HTTPException(status_code=404, detail="No such request")
    new_limit = int(record.get("token_limit", 0))
    sent, error = await access.send_extension_granted(email, new_limit)
    _require_sent(sent, error, email, "grant")
    return {"status": "granted", "email": email, "token_limit": new_limit}


@router.post("/admin/access/approve")
async def admin_approve(body: EmailBody, _: dict = Depends(require_admin)):
    return await _approve(_require_email(body.email))


@router.post("/admin/access/deny")
async def admin_deny(body: EmailBody, _: dict = Depends(require_admin)):
    return await _deny(_require_email(body.email))


@router.post("/admin/extension/approve")
async def admin_grant(body: EmailBody, _: dict = Depends(require_admin)):
    return await _grant(_require_email(body.email))


@router.post("/admin/user/delete")
async def admin_delete_user(body: EmailBody, _: dict = Depends(require_admin)):
    """Erasure on request: access record and conversations both go."""
    email = _require_email(body.email)
    if not access.delete_user(email):
        raise HTTPException(status_code=404, detail="No such user")
    return {"status": "deleted", "email": email}


@router.post("/admin/notify/retry")
async def admin_retry_notifications(_: dict = Depends(require_admin)):
    return {"recovered": await access.retry_failed_notifications()}


def _require_email(raw: str) -> str:
    email = access.normalize_email(raw)
    if not email:
        raise HTTPException(status_code=400, detail="Invalid email address")
    return email


# ── One-click links from notification emails ─────────────────────────────────

_ACTIONS = {"approve": _approve, "deny": _deny, "grant": _grant}


@router.get("/admin/act", response_class=HTMLResponse)
async def admin_act(t: str = ""):
    """Handle an Approve/Deny/Grant link straight from the admin's inbox."""
    payload = access.unsign(t, access.PURPOSE_ADMIN_ACT)
    if not payload:
        return HTMLResponse(_result_page("Link expired", "This link is no longer valid. "
                                         "Use the dashboard instead."), status_code=400)
    handler = _ACTIONS.get(payload.get("a"))
    email = access.normalize_email(payload.get("email"))
    if not handler or not email:
        return HTMLResponse(_result_page("Invalid link", "That action isn't recognised."),
                            status_code=400)
    try:
        result = await handler(email)
    except HTTPException as e:
        return HTMLResponse(_result_page("Nothing to do", str(e.detail)), status_code=e.status_code)
    return HTMLResponse(_result_page(
        f"{result['status'].title()}: {email}",
        "Done. They have been emailed."))


def _result_page(title: str, body: str) -> str:
    return (
        '<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<div style="font-family:system-ui,sans-serif;max-width:420px;margin:18vh auto;'
        'padding:28px;text-align:center;background:#0b0e14;color:#e6e8ee;border-radius:14px">'
        f'<h2 style="margin:0 0 10px;font-size:19px">{title}</h2>'
        f'<p style="color:#9aa0ae;margin:0">{body}</p>'
        '<p style="margin-top:22px"><a href="/admin" style="color:#22d3ee">Open dashboard</a></p>'
        '</div>')


# ── Retention sweep ──────────────────────────────────────────────────────────

@router.post("/admin/sweep")
async def admin_sweep(_: dict = Depends(require_admin)):
    return await run_sweep()


async def run_sweep() -> dict:
    """Hard-delete anything past its expiry.

    Firestore TTL policies are the primary mechanism, but they delete
    *typically within 24 hours of expiry* — which alone would make the flat
    "deleted within 24 hours" promise in the privacy notice false. This is the
    backstop that makes the promise true.
    """
    db = access.get_db()
    conversations = messages = runs = sessions = 0

    for user in db.collection("users").stream():
        user_ref = db.collection("users").document(user.id)
        for conv in user_ref.collection("conversations").stream():
            data = conv.to_dict() or {}
            if not _expired(data.get("expires_at")):
                continue
            conv_ref = user_ref.collection("conversations").document(conv.id)
            for message in conv_ref.collection("messages").stream():
                conv_ref.collection("messages").document(message.id).delete()
                messages += 1
            conv_ref.delete()
            conversations += 1
            if await _delete_agent_session(conv.id):
                sessions += 1

    # Ordered ascending, so the first live row means everything after it lives.
    for doc in (db.collection(access.RUNS_COLLECTION)
                .order_by("expires_at").limit(500).stream()):
        if not _expired((doc.to_dict() or {}).get("expires_at")):
            break
        db.collection(access.RUNS_COLLECTION).document(doc.id).delete()
        runs += 1

    return {"conversations": conversations, "messages": messages,
            "runs": runs, "agent_sessions": sessions}


def _expired(value) -> bool:
    """True only when a timestamp is definitely in the past.

    An unreadable value counts as *not* expired: a sweep that deletes what it
    cannot date is worse than one that leaves a stray document behind.
    """
    if value is None:
        return False
    age = access.seconds_since(value)
    return age != float("inf") and age > 0


async def _delete_agent_session(chat_id: str) -> bool:
    """Best-effort delete of the Vertex AI Agent Engine session.

    Agent Engine keeps conversation state server-side (VertexAiSessionService),
    so dropping only the Firestore copy would leave the 24-hour promise half
    true. Failures are swallowed and logged: this runs inside a sweep that must
    not abort partway through, and the Firestore delete — the copy this service
    actually controls — has already happened by the time we get here.
    """
    base = os.getenv("AGENT_ENGINE_ENDPOINT")
    if not base:
        return False
    engine = base.split(":query")[0].split(":streamQuery")[0]
    try:
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(google.auth.transport.requests.Request())
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.delete(
                f"{engine}/sessions/{chat_id}",
                headers={"Authorization": f"Bearer {credentials.token}"})
        return response.status_code < 400
    except Exception as e:
        access.log(f"WARNING: agent session delete failed for {chat_id}: {e}")
        return False


# ── Dashboard ────────────────────────────────────────────────────────────────

@router.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    """Self-contained dashboard. All data arrives via the JSON endpoints."""
    return HTMLResponse(DASHBOARD_HTML)


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tracer Lens — Admin</title>
<style>
:root{--bg:#0b0e14;--s1:#131722;--s2:#1a1f2e;--tx:#e6e8ee;--tx2:#9aa0ae;
--cy:#22d3ee;--vi:#8b5cf6;--ok:#22c55e;--warn:#f59e0b;--err:#ef4444;--bd:#252b3b}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--tx)}
.wrap{max-width:1100px;margin:0 auto;padding:24px 18px 60px}
h1{font-size:20px;margin:0 0 2px}h1 span{background:linear-gradient(90deg,var(--cy),var(--vi));
-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{color:var(--tx2);font-size:13px;margin-bottom:22px}
.card{background:var(--s1);border:1px solid var(--bd);border-radius:12px;padding:18px;margin-bottom:16px}
label{display:block;font-size:12px;color:var(--tx2);margin-bottom:5px}
input,textarea{width:100%;padding:10px 12px;background:var(--s2);border:1px solid var(--bd);
border-radius:8px;color:var(--tx);font:inherit;margin-bottom:12px}
input:focus{outline:2px solid var(--cy);outline-offset:-1px}
button{padding:8px 14px;border:0;border-radius:8px;background:var(--s2);color:var(--tx);
font:inherit;font-weight:600;cursor:pointer;border:1px solid var(--bd)}
button:hover{border-color:var(--cy)}button.primary{background:linear-gradient(90deg,var(--cy),var(--vi));color:#06121a;border:0}
button.ok{background:var(--ok);color:#04140a;border:0}button.no{background:var(--err);color:#fff;border:0}
button.sm{padding:5px 10px;font-size:12px}
.tabs{display:flex;gap:6px;margin-bottom:14px;flex-wrap:wrap}
.tabs button{background:transparent}.tabs button[aria-selected=true]{background:var(--s2);border-color:var(--cy)}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--tx2);font-weight:600;font-size:11px;text-transform:uppercase;
letter-spacing:.04em;padding:8px 10px;border-bottom:1px solid var(--bd)}
td{padding:9px 10px;border-bottom:1px solid var(--bd);vertical-align:middle}
tr:last-child td{border-bottom:0}
.scroll{overflow-x:auto}
.pill{display:inline-block;padding:2px 9px;border-radius:99px;font-size:11px;font-weight:600}
.pill.approved{background:rgba(34,197,94,.15);color:var(--ok)}
.pill.pending{background:rgba(245,158,11,.15);color:var(--warn)}
.pill.denied{background:rgba(239,68,68,.15);color:var(--err)}
.pill.ok{background:rgba(34,197,94,.15);color:var(--ok)}
.pill.fail{background:rgba(239,68,68,.15);color:var(--err)}
.alert{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.4);border-radius:10px;
padding:12px 14px;margin-bottom:14px;font-size:13px}
.alert b{color:var(--err)}
.bar{height:6px;background:var(--s2);border-radius:99px;overflow:hidden;min-width:80px}
.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--cy),var(--vi))}
.muted{color:var(--tx2)}.right{text-align:right}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.msg{background:var(--s2);border-radius:8px;padding:8px 10px;margin-top:6px;
font-size:12px;white-space:pre-wrap;color:var(--tx2)}
.hidden{display:none}
.toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:var(--s2);
border:1px solid var(--cy);padding:10px 18px;border-radius:10px;font-size:13px}
</style></head><body><div class="wrap">
<h1>Tracer Lens <span>Admin</span></h1>
<div class="sub" id="sub">Sign in to review access requests.</div>

<div class="card" id="login">
  <div id="step1">
    <label for="pw">Admin password</label>
    <input id="pw" type="password" autocomplete="current-password" placeholder="ADMIN_TOKEN">
    <button class="primary" onclick="startAuth()">Send me a code</button>
  </div>
  <div id="step2" class="hidden">
    <label for="code">6-digit code sent to your inbox</label>
    <input id="code" inputmode="numeric" autocomplete="one-time-code" placeholder="000000">
    <button class="primary" onclick="verifyAuth()">Verify</button>
    <button onclick="reset()">Back</button>
  </div>
  <div id="err" class="muted" style="margin-top:8px"></div>
</div>

<div id="app" class="hidden">
  <div id="alerts"></div>
  <div class="tabs">
    <button onclick="show('requests')" aria-selected="true" id="t-requests">Requests</button>
    <button onclick="show('users')" aria-selected="false" id="t-users">Users</button>
    <button onclick="show('activity')" aria-selected="false" id="t-activity">Activity</button>
    <button onclick="refresh()">Refresh</button>
    <button onclick="sweep()">Run retention sweep</button>
    <button onclick="logout()">Log out</button>
  </div>
  <div class="card" id="panel"></div>
</div>
</div>
<script>
const KEY='tl-admin-token';
let token=sessionStorage.getItem(KEY)||'', data={users:[],pending:[],extensions:[],alerts:[]},
    runs=[], tab='requests', challenge='';

const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=n=>Number(n||0).toLocaleString();
const ago=t=>{if(!t)return '—';const d=(Date.now()-new Date(t))/1000;
  if(d<60)return 'just now';if(d<3600)return Math.floor(d/60)+'m ago';
  if(d<86400)return Math.floor(d/3600)+'h ago';return Math.floor(d/86400)+'d ago';};

function toast(m){const el=document.createElement('div');el.className='toast';el.textContent=m;
  document.body.appendChild(el);setTimeout(()=>el.remove(),2600);}
function fail(m){document.getElementById('err').textContent=m||'';}

async function api(path,opts={}){
  const r=await fetch(path,{...opts,headers:{'Content-Type':'application/json',
    ...(token?{Authorization:'Bearer '+token}:{}),...(opts.headers||{})}});
  // Only a *session* 401 means expired. The auth endpoints answer 401 for a
  // wrong password or code, and that message has to reach the form.
  if(r.status===401&&token&&!path.startsWith('/admin/auth/')){
    logout();throw new Error('Session expired');}
  const body=await r.json().catch(()=>({}));
  if(!r.ok)throw new Error(typeof body.detail==='string'?body.detail:'Request failed');
  return body;
}

async function startAuth(){
  fail('');
  try{
    const r=await api('/admin/auth/start',{method:'POST',
      body:JSON.stringify({password:document.getElementById('pw').value})});
    challenge=r.challenge_id;
    document.getElementById('step1').classList.add('hidden');
    document.getElementById('step2').classList.remove('hidden');
    document.getElementById('code').focus();
  }catch(e){fail(e.message);}
}
async function verifyAuth(){
  fail('');
  try{
    const r=await api('/admin/auth/verify',{method:'POST',
      body:JSON.stringify({challenge_id:challenge,code:document.getElementById('code').value})});
    token=r.token;sessionStorage.setItem(KEY,token);boot();
  }catch(e){fail(e.message);}
}
function reset(){document.getElementById('step1').classList.remove('hidden');
  document.getElementById('step2').classList.add('hidden');fail('');}
function logout(){token='';sessionStorage.removeItem(KEY);
  document.getElementById('app').classList.add('hidden');
  document.getElementById('login').classList.remove('hidden');reset();}

async function boot(){
  if(!token)return;
  document.getElementById('login').classList.add('hidden');
  document.getElementById('app').classList.remove('hidden');
  await refresh();
}
async function refresh(){
  try{
    data=await api('/admin/users');
    runs=(await api('/admin/runs?limit=80')).runs;
    document.getElementById('sub').textContent=
      data.users.length+' users · '+data.pending.length+' pending · '+data.extensions.length+' extension requests';
    render();
  }catch(e){toast(e.message);}
}
async function act(path,email,label){
  try{await api(path,{method:'POST',body:JSON.stringify({email})});toast(label);await refresh();}
  catch(e){toast(e.message);}
}
function approve(e){act('/admin/access/approve',e,'Approved — sign-in link sent');}
function deny(e){act('/admin/access/deny',e,'Denied');}
function grant(e){act('/admin/extension/approve',e,'Extension granted');}
function del(e){if(confirm('Delete '+e+' and all their conversations? This cannot be undone.'))
  act('/admin/user/delete',e,'Deleted');}
function retry(){act('/admin/notify/retry','','Retried failed notifications');}
async function sweep(){try{const r=await api('/admin/sweep',{method:'POST'});
  toast('Swept '+r.conversations+' conversations, '+r.runs+' metric rows');}catch(e){toast(e.message);}}

function show(next){tab=next;
  ['requests','users','activity'].forEach(t=>
    document.getElementById('t-'+t).setAttribute('aria-selected',String(t===next)));
  render();}

function render(){
  document.getElementById('alerts').innerHTML=(data.alerts||[]).map(a=>
    '<div class="alert"><b>'+(a.kind==='notify_failed'?'Notification failed':'Waiting over 24h')+
    '</b> — '+esc(a.email)+'<br><span class="muted">'+esc(a.detail)+'</span>'+
    (a.kind==='notify_failed'?' <button class="sm" data-act="retry">Retry send</button>':'')+
    '</div>').join('');
  document.getElementById('panel').innerHTML=
    tab==='requests'?requestsView():tab==='users'?usersView():activityView();
}

function requestsView(){
  let html='<h3 style="margin:0 0 12px;font-size:15px">Pending access</h3>';
  html+= data.pending.length? '<div class="scroll"><table><tr><th>Email</th><th>Requested</th>'+
    '<th>Notified</th><th class="right">Decision</th></tr>'+data.pending.map(u=>
    '<tr><td>'+esc(u.email)+'</td><td class="muted">'+ago(u.requested_at)+'</td>'+
    '<td>'+pill(u.notify_state==='sent'?'ok':'fail',u.notify_state||'—')+'</td>'+
    '<td class="right"><button class="sm ok" data-act="approve" data-email="'+esc(u.email)+'">Approve</button> '+
    '<button class="sm no" data-act="deny" data-email="'+esc(u.email)+'">Deny</button></td></tr>').join('')+
    '</table></div>' : '<p class="muted">Nothing waiting.</p>';

  html+='<h3 style="margin:22px 0 12px;font-size:15px">Extension requests</h3>';
  html+= data.extensions.length? data.extensions.map(u=>
    '<div style="border-top:1px solid var(--bd);padding:12px 0"><div class="row">'+
    '<strong>'+esc(u.email)+'</strong><span class="muted">'+num(u.tokens_used)+' / '+
    num(u.token_limit)+' tokens</span>'+
    '<button class="sm ok" data-act="grant" data-email="'+esc(u.email)+'">Grant +200,000</button></div>'+
    (u.extension_message?'<div class="msg">'+esc(u.extension_message)+'</div>':'')+
    '</div>').join('') : '<p class="muted">Nothing waiting.</p>';
  return html;
}

function usersView(){
  if(!data.users.length)return '<p class="muted">No users yet.</p>';
  return '<div class="scroll"><table><tr><th>Email</th><th>Status</th><th>Tokens</th>'+
    '<th>Runs</th><th>Failures</th><th>Avg latency</th><th>Last run</th><th></th></tr>'+
    data.users.map(u=>{
      const pct=u.token_limit?Math.min(100,u.tokens_used/u.token_limit*100):0;
      return '<tr><td>'+esc(u.email)+'</td><td>'+pill(u.status,u.status)+'</td>'+
      '<td><div class="bar"><i style="width:'+pct.toFixed(1)+'%"></i></div>'+
      '<span class="muted" style="font-size:11px">'+num(u.tokens_used)+' / '+num(u.token_limit)+'</span></td>'+
      '<td>'+num(u.runs_total)+'</td>'+
      '<td>'+(u.runs_failed?'<span class="pill fail">'+u.runs_failed+' · '+u.failure_rate+'%</span>':'<span class="muted">0</span>')+'</td>'+
      '<td class="muted">'+(u.avg_latency_ms?(u.avg_latency_ms/1000).toFixed(1)+'s':'—')+'</td>'+
      '<td class="muted">'+ago(u.last_run_at)+'</td>'+
      '<td class="right"><button class="sm no" data-act="del" data-email="'+esc(u.email)+'">Delete data</button></td></tr>';
    }).join('')+'</table></div>';
}

function activityView(){
  if(!runs.length)return '<p class="muted">No runs recorded yet.</p>';
  const byHash={};data.users.forEach(u=>byHash[u.key]=u.email);
  return '<div class="scroll"><table><tr><th>When</th><th>User</th><th>Result</th>'+
    '<th>Latency</th><th>Tokens (in / out)</th><th>Mode</th></tr>'+
    runs.map(r=>'<tr><td class="muted">'+ago(r.ts)+'</td>'+
      '<td>'+esc(byHash[r.email_hash]||r.email_hash.slice(0,10)+'…')+'</td>'+
      '<td>'+(r.ok?'<span class="pill ok">ok</span>':
        '<span class="pill fail">'+esc(r.error_kind||'failed')+'</span>')+'</td>'+
      '<td class="muted">'+(r.latency_ms/1000).toFixed(1)+'s</td>'+
      '<td class="muted">'+num(r.tokens_total)+' <span style="opacity:.6">('+num(r.tokens_in)+
      ' / '+num(r.tokens_out)+')</span></td>'+
      '<td class="muted">'+(r.causal?'causal':'chat')+(r.web?' + web':'')+'</td></tr>').join('')+
    '</table></div>';
}

function pill(cls,text){return '<span class="pill '+esc(cls)+'">'+esc(text)+'</span>';}

// Row actions are delegated rather than inline onclick handlers. An email is
// attacker-chosen text that reaches this page before anyone approves it, and
// inside a double-quoted attribute the parser entity-decodes esc()'s &#39;
// back to a quote *before* the JS is parsed — so interpolating one into
// onclick="fn('...')" is injectable no matter how it was escaped. dataset
// hands back the decoded value with no JS parsing step at all.
const ACTIONS={approve,deny,grant,del,retry};
document.addEventListener('click',e=>{
  const btn=e.target.closest('button[data-act]');
  if(!btn)return;
  const fn=ACTIONS[btn.dataset.act];
  if(fn)fn(btn.dataset.email||'');
});

document.getElementById('pw').addEventListener('keydown',e=>{if(e.key==='Enter')startAuth();});
document.getElementById('code').addEventListener('keydown',e=>{if(e.key==='Enter')verifyAuth();});
boot();
</script></body></html>
"""
