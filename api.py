"""Callsight web app. Run: uvicorn api:app --port 8012"""
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
from pydantic import BaseModel, Field

from callsight import ask, extract, pipeline, redact, store
from callsight.config import CLIENTS, ROOT

app = FastAPI(title="Callsight", docs_url=None, redoc_url=None, openapi_url=None)
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


def _token(expiry: int) -> str:
    key = hashlib.sha256(f"callsight:{os.getenv('DEMO_USER', '')}:{os.getenv('DEMO_PASS', '')}".encode()).digest()
    return f"{expiry}." + hmac.new(key, str(expiry).encode(), hashlib.sha256).hexdigest()


def _authed(session: str | None) -> bool:
    if not os.getenv("DEMO_PASS"):
        return True
    try:
        expiry = int((session or "").split(".", 1)[0])
    except ValueError:
        return False
    return expiry > time.time() and hmac.compare_digest(session.encode(), _token(expiry).encode())


def _ip(request: Request) -> str:
    return request.headers.get("x-forwarded-for", "").split(",")[0].strip() or request.client.host  # Caddy sets it from CF


# Per-IP limits on the endpoints that spend model tokens or CPU. With no login this is what stops one visitor using up
# the day's budget (DAILY_BUDGET_USD) for everyone.
LIMITS = {"call": (int(os.getenv("CALLS_PER_HOUR", "10")), 3600), "ask": (int(os.getenv("ASKS_PER_HOUR", "30")), 3600)}
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


def require_session(cs_session: str | None = Cookie(default=None)):
    if not _authed(cs_session):
        raise HTTPException(401, "sign in required")


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
    ok = hmac.compare_digest(body.username.encode(), os.getenv("DEMO_USER", "").encode()) & \
        hmac.compare_digest(body.password.encode(), os.getenv("DEMO_PASS", "").encode())
    if not ok:
        _fails.setdefault(ip, []).append(now)
        log.warning("login failed ip=%s", ip)  # never log what was typed
        raise HTTPException(401, "Incorrect username or password")
    _fails.pop(ip, None)
    response.set_cookie(COOKIE, _token(int(now) + SESSION_TTL), httponly=True, secure=True, samesite="lax", max_age=SESSION_TTL)
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
def me(cs_session: str | None = Cookie(default=None)):
    return {"login_required": bool(os.getenv("DEMO_PASS")), "signed_in": _authed(cs_session)}


@app.get("/api/clients", dependencies=api)
def clients():
    return [c.model_dump() for c in CLIENTS.values()]


SAMPLES = [{k: c[k] for k in ("id", "client_id", "title", "transcript")}
           for f in ("calls.json", "calls_new.json") for c in json.loads((ROOT / "data" / f).read_text())]


@app.get("/api/samples", dependencies=api)
def samples():
    return SAMPLES


@app.get("/api/sample-audio", dependencies=api)
def sample_audio():  # a two-voice stereo test recording of CALL-01 (tools/make_audio.py)
    wav = ROOT / "samples" / "c1.wav"
    if not wav.exists():
        raise HTTPException(404, "no sample recording on this server")
    return FileResponse(wav, media_type="audio/wav", filename="callsight-sample-call.wav")


@app.get("/api/calls", dependencies=api)
def calls():
    return [{"id": r.id, "client": CLIENTS[r.client_id].name, "churn_risk": r.insight.churn_risk,
             "sentiment": r.insight.sentiment, "alerts": len(r.alerts)} for r in store.all_calls(conn)]


@app.get("/api/calls/{call_id}", dependencies=api)
def call(call_id: str):
    if not (r := store.get(conn, call_id)):
        raise HTTPException(404, "no such call")
    return r.model_dump() | {"client": CLIENTS[r.client_id].model_dump()}


@app.post("/api/calls", dependencies=api)
async def new_call(request: Request, client_id: str = Form(...), transcript: str | None = Form(None),
                   audio: UploadFile | None = None, provider: str | None = Form(None)):
    if client_id not in CLIENTS:
        raise HTTPException(422, "unknown client")
    if provider not in (None, "", "claude", "grok"):
        raise HTTPException(422, "provider must be claude or grok")
    if bool(transcript and transcript.strip()) == bool(audio and audio.filename):
        raise HTTPException(422, "send a transcript or an audio file, not both")
    rate_limit("call", request)
    call_id = f"CALL-{time.strftime('%m%d')}-{uuid.uuid4().hex[:5]}"
    kw = {"provider": provider or None}
    try:
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


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    provider: str | None = None


@app.post("/api/ask", dependencies=api)
async def ask_calls(q: Question, request: Request):
    rate_limit("ask", request)
    try:
        return await run_in_threadpool(ask.answer, conn, q.question, q.provider if q.provider in ("claude", "grok") else None)
    except extract.ModelUnavailable as e:
        raise HTTPException(502, str(e))
