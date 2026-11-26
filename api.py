"""CallSight web app. Run: uvicorn api:app --port 8012"""
import hashlib
import hmac
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import Cookie, Depends, FastAPI, Form, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from typing import Literal

from pydantic import BaseModel, Field

from callsight import ask, extract, pipeline, redact, store, stt
from callsight.models import ActionItem, Segment
from callsight.config import CLIENTS, ROOT

app = FastAPI(title="CallSight", docs_url=None, redoc_url=None, openapi_url=None)
conn = store.connect()
COOKIE = "cs_session"
MAX_AUDIO = 25 * 1024 * 1024  # ~25 min of compressed phone audio
MAX_MINUTES = float(os.getenv("MAX_AUDIO_MINUTES", "10"))  # Whisper on CPU: ~20 s of compute per minute of audio
AUDIO = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}
log = logging.getLogger("uvicorn.error")

# --- auth: same scheme as LeadDesk (no DEMO_PASS = open, for local dev) -----------------------------------------
SESSION_TTL = 7 * 86400
FAILS_PER_IP, FAILS_GLOBAL, WINDOW = 5, 30, 15 * 60
_fails: dict[str, list[float]] = {}  # ponytail: in-memory, one worker process


def _token(expiry: int, user: str) -> str:
    """The username is in the cookie (signed), so the log can say who did what for the whole 7 days, not only at sign-in."""
    key = hashlib.sha256(f"callsight:{os.getenv('DEMO_USER', '')}:{os.getenv('DEMO_PASS', '')}:{os.getenv('EXTRA_LOGINS', '')}".encode()).digest()
    return f"{expiry}.{user}." + hmac.new(key, f"{expiry}.{user}".encode(), hashlib.sha256).hexdigest()


def _user(session: str | None) -> str | None:
    """Who this cookie belongs to, or None if it's missing, expired or forged. "" when no login is configured."""
    if not _logins():  # no login configured: open, for local dev
        return ""
    expiry, _, rest = (session or "").partition(".")
    user = rest.rpartition(".")[0]
    try:
        ok = int(expiry) > time.time() and hmac.compare_digest(session.encode(), _token(int(expiry), user).encode())
    except ValueError:
        return None
    return user if ok and any(u == user for u, _ in _logins()) else None


def _authed(session: str | None) -> bool:
    return _user(session) is not None


def _login_ok(username: str, password: str) -> bool:
    """DEMO_USER/DEMO_PASS, or any "user:pass" in EXTRA_LOGINS (comma separated), e.g. one login per reviewer."""
    return any(hmac.compare_digest(username.encode(), u.encode()) & hmac.compare_digest(password.encode(), p.encode())
               for u, p in _logins())


def _logins() -> list[tuple[str, str]]:
    pairs = [(os.getenv("DEMO_USER", ""), os.getenv("DEMO_PASS", ""))]
    pairs += [(u.strip(), p.strip()) for u, _, p in (x.partition(":") for x in os.getenv("EXTRA_LOGINS", "").split(","))]
    return [(u, p) for u, p in pairs if u and p]


def _ip(request: Request) -> str:
    return request.headers.get("x-forwarded-for", "").split(",")[0].strip() or request.client.host  # Caddy sets it from CF


# Per-IP limits on the endpoints that spend model tokens or CPU. With no login this is what stops one visitor using up
# the day's budget (DAILY_BUDGET_USD) for everyone.
LIMITS = {"call": (int(os.getenv("CALLS_PER_HOUR", "10")), 3600), "ask": (int(os.getenv("ASKS_PER_HOUR", "30")), 3600),
          "live": (int(os.getenv("LIVE_CHUNKS_PER_HOUR", "400")), 3600)}  # ~5 s each: 10-min recordings, a few an hour
_hits: dict[tuple[str, str], list[float]] = {}  # ponytail: in-memory, one worker; never pruned of idle IPs


def rate_limit(kind: str, request: Request) -> None:
    n, window = LIMITS[kind]
    now, key = time.time(), (kind, _ip(request))
    hits = [t for t in _hits.get(key, []) if t > now - window]
    if len(hits) >= n:
        raise HTTPException(429, f"limit of {n} per hour reached for this demo; try again later")
    _hits[key] = hits + [now]


def _recent(ip: str, now: float) -> list[float]:
    hits = [t for t in _fails.get(ip, []) if t > now - WINDOW]
    if hits:
        _fails[ip] = hits
    else:
        _fails.pop(ip, None)
    return hits


def require_session(cs_session: str | None = Cookie(default=None)) -> str:
    if (user := _user(cs_session)) is None:
        raise HTTPException(401, "sign in required")
    return user


class Login(BaseModel):
    username: str
    password: str


@app.post("/api/login")
def login(body: Login, request: Request, response: Response):
    ip = _ip(request)
    now = time.time()
    if len(_recent(ip, now)) >= FAILS_PER_IP or sum(len(_recent(k, now)) for k in list(_fails)) >= FAILS_GLOBAL:
        log.warning("login locked out ip=%s", ip)
        raise HTTPException(429, "Too many attempts. Try again in 15 minutes.")
    if not _login_ok(body.username, body.password):
        _fails.setdefault(ip, []).append(now)
        log.warning("login failed ip=%s", ip)  # never log what was typed
        raise HTTPException(401, "Incorrect username or password")
    _fails.pop(ip, None)
    log.info("login ok user=%s ip=%s", body.username, ip)
    response.set_cookie(COOKIE, _token(int(now) + SESSION_TTL, body.username), httponly=True, secure=True, samesite="lax", max_age=SESSION_TTL)
    return {"ok": True}


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE, httponly=True, secure=True, samesite="lax")
    return {"ok": True}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "web" / "index.html")


# --- app API (cookie protected) ----------------------------------------------------------------------------------
api = [Depends(require_session)]


@app.get("/api/me")
def me(request: Request, cs_session: str | None = Cookie(default=None)):
    user = _user(cs_session)
    if user:  # every page load by a signed-in user, so returning visitors show up too
        log.info("visit user=%s ip=%s", user, _ip(request))
    return {"login_required": bool(_logins()), "signed_in": user is not None}


@app.get("/api/settings", dependencies=api)
def settings():
    """What the settings page can offer, and the server defaults it starts from."""
    from callsight.config import RULES
    return {"providers": [p for p, k in (("claude", "ANTHROPIC_API_KEY"), ("grok", "XAI_API_KEY")) if os.getenv(k)],
            "default_provider": extract.default_provider(), "keepable": sorted(redact.KEEPABLE),
            "churn_alert": RULES["churn_alert_min"]}


@app.get("/api/clients", dependencies=api)
def clients():
    return [c.model_dump() for c in CLIENTS.values()]


SAMPLES = [{k: c[k] for k in ("id", "client_id", "title", "transcript")}
           for f in ("calls_long.json", "calls.json", "calls_new.json") for c in json.loads((ROOT / "data" / f).read_text())]


@app.get("/api/samples", dependencies=api)
def samples():
    return SAMPLES


SAMPLE_AUDIO = {"1": "c1.wav", "21": "c21.mp3", "22": "c22.mp3", "23": "c23.mp3"}  # stereo test calls (tools/make_audio.py)


@app.get("/api/sample-audio", dependencies=api)
def sample_audio(call: str = "1"):
    path = ROOT / "samples" / SAMPLE_AUDIO.get(call, "missing")
    if not path.is_file():
        raise HTTPException(404, "no such sample recording on this server")
    return FileResponse(path, filename=f"callsight-sample-call-{call}{path.suffix}")


@app.get("/api/calls", dependencies=api)
def calls():
    return [{"id": r.id, "client": CLIENTS[r.client_id].name, "churn_risk": r.insight.churn_risk,
             "sentiment": r.insight.sentiment, "alerts": len(r.alerts)} for r in store.all_calls(conn)]


@app.get("/api/calls/{call_id}", dependencies=api)
def call(call_id: str):
    if not (r := store.get(conn, call_id)):
        raise HTTPException(404, "no such call")
    return r.model_dump() | {"client": CLIENTS[r.client_id].model_dump()}


MAX_CHUNK = 2 * 1024 * 1024  # a 30 s chunk of 16 kHz mono 16-bit WAV is ~1 MB


@app.post("/api/live", dependencies=api)
async def live_chunk(request: Request, audio: UploadFile):
    """One few-second piece of a microphone recording -> its words, with times from the start of the piece. The browser
    adds the offset and keeps the running transcript; nothing is stored here, and the audio is deleted at once."""
    rate_limit("live", request)
    data = await audio.read(MAX_CHUNK + 1)
    if len(data) > MAX_CHUNK:
        raise HTTPException(413, "chunk too large")
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        f.write(data)
        f.flush()
        if _minutes(f.name) > 0.5:
            raise HTTPException(413, "chunk too long (30 s max)")
        segs = await run_in_threadpool(stt.transcribe, f.name)
    return [{"start": s.start, "end": s.end, "text": s.text} for s in segs]


class LiveSegment(BaseModel):
    start: float = Field(ge=0, le=3600)
    end: float = Field(ge=0, le=3600)
    text: str = Field(max_length=2000)


@app.post("/api/calls", dependencies=api)
async def new_call(request: Request, user: str = Depends(require_session), client_id: str = Form(...), transcript: str | None = Form(None),
                   audio: UploadFile | None = None, segments: str | None = Form(None), provider: str | None = Form(None),
                   keep: str = Form(""), churn_alert: float | None = Form(None, ge=0.3, le=0.9)):
    if client_id not in CLIENTS:
        raise HTTPException(422, "unknown client")
    if provider not in (None, "", "claude", "grok"):
        raise HTTPException(422, "provider must be claude or grok")
    if [bool(transcript and transcript.strip()), bool(audio and audio.filename), bool(segments)].count(True) != 1:
        raise HTTPException(422, "send one of: a transcript, an audio file, or a live recording's transcript")
    live = None
    if segments:  # a microphone recording, already transcribed piece by piece through /api/live
        try:
            live = [Segment(speaker="Speaker", **LiveSegment.model_validate(s).model_dump()) for s in json.loads(segments)]
        except Exception:
            raise HTTPException(422, "couldn't read the live transcript")
        if not live or len(live) > 3000 or sum(len(s.text) for s in live) > 60_000:
            raise HTTPException(413, "live transcript empty or too long")
    keep_set = {k.strip().upper() for k in keep.split(",") if k.strip()}
    if keep_set - redact.KEEPABLE:
        raise HTTPException(422, f"only {', '.join(sorted(redact.KEEPABLE))} can be left unmasked")
    rate_limit("call", request)
    call_id = f"CALL-{time.strftime('%m%d')}-{uuid.uuid4().hex[:5]}"
    log.info("call user=%s id=%s client=%s %s", user, call_id, client_id, "live" if live else "audio" if audio and audio.filename else "text")
    kw = {"provider": provider or None, "keep": keep_set, "churn_min": churn_alert, "write_out": False}  # Monday/GHL on approval
    try:
        if live:
            return (await run_in_threadpool(pipeline.process, call_id, client_id, conn, segments=live, **kw)).model_dump()
        if transcript:
            if len(transcript) > 60_000:
                raise HTTPException(413, "transcript too long (60k characters max)")
            return (await run_in_threadpool(pipeline.process, call_id, client_id, conn, text=transcript, **kw)).model_dump()
        suffix = Path(audio.filename).suffix.lower()
        if suffix not in AUDIO:
            raise HTTPException(415, f"audio must be one of {', '.join(sorted(AUDIO))}")
        data = await audio.read(MAX_AUDIO + 1)
        if len(data) > MAX_AUDIO:
            raise HTTPException(413, "audio too large (25 MB max)")
        with tempfile.NamedTemporaryFile(suffix=suffix) as f:  # deleted as soon as it's transcribed
            f.write(data)
            f.flush()
            if _minutes(f.name) > MAX_MINUTES:
                raise HTTPException(413, f"recording too long ({MAX_MINUTES:g} minutes max)")
            return (await run_in_threadpool(pipeline.process, call_id, client_id, conn, audio_path=f.name, **kw)).model_dump()
    except redact.RedactionUnavailable as e:
        raise HTTPException(503, str(e))
    except extract.ModelUnavailable as e:
        raise HTTPException(502, str(e))


def _minutes(path: str) -> float:
    import av
    try:
        with av.open(path) as c:
            return (c.duration or 0) / av.time_base / 60  # unknown duration -> 0; Whisper still runs under the CPU cap
    except Exception:
        raise HTTPException(415, "couldn't read that audio file")


class Edit(BaseModel):
    summary: str | None = Field(None, min_length=1, max_length=2000)
    sentiment: Literal["positive", "neutral", "frustrated", "angry"] | None = None
    churn_risk: float | None = Field(None, ge=0, le=1)
    action_items: list[ActionItem] | None = Field(None, max_length=50)


def _get(call_id: str):
    if not (r := store.get(conn, call_id)):
        raise HTTPException(404, "no such call")
    return r


@app.patch("/api/calls/{call_id}", dependencies=api)
async def edit_call(call_id: str, body: Edit, user: str = Depends(require_session)):
    if any(len(a.task) > 300 or len(a.quote) > 500 or len(a.due or "") > 80 or len(a.speaker or "") > 80 for a in body.action_items or []):
        raise HTTPException(422, "a follow-up is too long")
    r = await run_in_threadpool(pipeline.edit, conn, _get(call_id), user, **body.model_dump(exclude_none=True, exclude={"action_items"}),
                                action_items=body.action_items)
    log.info("edit user=%s id=%s", user, call_id)
    return r.model_dump() | {"client": CLIENTS[r.client_id].model_dump()}


@app.post("/api/calls/{call_id}/approve", dependencies=api)
async def approve_call(call_id: str, user: str = Depends(require_session)):
    r = await run_in_threadpool(pipeline.approve, conn, _get(call_id), user)
    log.info("approve user=%s id=%s monday=%s", user, call_id, r.monday_item_id)
    return r.model_dump() | {"client": CLIENTS[r.client_id].model_dump(),
                             "connected": {"monday": bool(os.getenv("MONDAY_API_TOKEN") and os.getenv("MONDAY_CALLS_BOARD_ID")),
                                           "ghl": bool(os.getenv("GHL_API_KEY") and os.getenv("GHL_CONTACTS"))}}


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    provider: str | None = None


@app.post("/api/ask", dependencies=api)
async def ask_calls(q: Question, request: Request, user: str = Depends(require_session)):
    rate_limit("ask", request)
    log.info("ask user=%s q=%r", user, q.question[:120])
    try:
        return await run_in_threadpool(ask.answer, conn, q.question, q.provider if q.provider in ("claude", "grok") else None)
    except extract.ModelUnavailable as e:
        raise HTTPException(502, str(e))
