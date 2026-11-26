"""Callsight web app. Run: uvicorn api:app --port 8012"""
import hashlib
import hmac
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
    ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or request.client.host  # Caddy sets it from CF
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


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "web" / "index.html")


# --- app API (cookie protected) ----------------------------------------------------------------------------------
api = [Depends(require_session)]


@app.get("/api/clients", dependencies=api)
def clients():
    return [c.model_dump() for c in CLIENTS.values()]


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
async def new_call(client_id: str = Form(...), transcript: str | None = Form(None), audio: UploadFile | None = None,
                   provider: str | None = Form(None)):
    if client_id not in CLIENTS:
        raise HTTPException(422, "unknown client")
    if provider not in (None, "", "claude", "grok"):
        raise HTTPException(422, "provider must be claude or grok")
    if bool(transcript and transcript.strip()) == bool(audio and audio.filename):
        raise HTTPException(422, "send a transcript or an audio file, not both")
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
            return (await run_in_threadpool(pipeline.process, call_id, client_id, conn, audio_path=f.name, **kw)).model_dump()
    except redact.RedactionUnavailable as e:
        raise HTTPException(503, str(e))
    except extract.ModelUnavailable as e:
        raise HTTPException(502, str(e))


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    provider: str | None = None


@app.post("/api/ask", dependencies=api)
async def ask_calls(q: Question):
    return await run_in_threadpool(ask.answer, conn, q.question, q.provider if q.provider in ("claude", "grok") else None)
