"""audio or transcript -> segments -> PII Shield -> LLM extraction -> quote check -> rules -> store -> Monday / GHL."""
import time
from functools import partial

from . import ask, extract, redact, rules, sinks, store, stt
from .config import CLIENTS
from .models import ActionItem, CallResult, Segment


def process(call_id: str, client_id: str, conn, *, audio_path: str | None = None, text: str | None = None,
            segments: list[Segment] | None = None, provider: str | None = None, keep: set[str] = frozenset(),
            churn_min: float | None = None, write_out: bool = True, owner: str | None = None,
            speakers: dict[str, str] | None = None, rerun_of: str | None = None, on_progress=None,
            analyze_fn=None, redact_fn=None, embed_fn=None) -> CallResult:
    """One of audio_path, text, or segments (already transcribed, e.g. live from a microphone).
    write_out=False leaves Monday/GHL until a person approves the call (the web app); the CLI and evals write at once."""
    client = CLIENTS.get(client_id)
    if client is None:
        raise KeyError(f"unknown client {client_id}")
    if sum(x is not None for x in (audio_path, text, segments)) != 1:
        raise ValueError("give exactly one of audio_path, text or segments")
    start = time.perf_counter()
    extract.WHO.set({"user": owner, "call": call_id})  # for the usage log
    step = on_progress or (lambda stage, **kw: None)
    names = speakers or {}
    if audio_path:
        segments = stt.transcribe(audio_path, on_progress=lambda f, d: step("transcribing", fraction=f, seconds=d))
    elif text is not None:  # speaker names are only swapped for aliases when names are being masked
        segments = stt.from_text(text)
        if "NAME" not in keep:
            names = {a: name for name, a in redact.aliases(segments).items()}
            segments = redact.alias_speakers(segments)
    step("masking")
    segments = (redact_fn or partial(redact.redact, keep=keep))(segments)  # raises if PII Shield is down: nothing unmasked goes further
    provider = provider or extract.default_provider()
    step("analysing", provider=provider)
    insight = extract.analyze(segments, client, provider, analyze_fn)
    step("checking")
    for a in insight.action_items:
        a.done = False  # a person sets this, never the model
    checks = extract.check_quotes(insight, segments)
    result = CallResult(id=call_id, client_id=client_id, segments=segments, insight=insight, checks=checks,
                        alerts=rules.alerts(insight, checks, client, segments, churn_min), provider=provider,
                        unmasked=sorted(keep), speakers=names, churn_min=churn_min, owner=owner, rerun_of=rerun_of,
                        latency_ms=int((time.perf_counter() - start) * 1000))
    store.save(conn, result)  # saved before any write-out, so a Monday/GHL failure never loses the call
    ask.index(conn, result, embed_fn or ask.embed)
    if write_out:
        send(conn, result)
    return result


def send(conn, r: CallResult) -> CallResult:
    """Write the call to Monday and GHL (each is off unless configured)."""
    if not r.monday_item_id and (item := sinks.to_monday(r)):
        r.monday_item_id = item
        store.save(conn, r)
    sinks.to_ghl(r)
    return r


def edit(conn, r: CallResult, user: str, *, summary: str | None = None, sentiment: str | None = None,
         churn_risk: float | None = None, action_items: list[ActionItem] | None = None, embed_fn=None) -> CallResult:
    """A person's corrections. The alerts are re-run from the rules; the quote checks stay as the model left them.
    An edit after approval needs approving again."""
    changes = {k: v for k, v in (("summary", summary), ("sentiment", sentiment), ("churn_risk", churn_risk),
                                 ("action_items", action_items)) if v is not None}
    r.insight = r.insight.model_copy(update=changes)
    open_items = r.insight.model_copy(update={"action_items": [a for a in r.insight.action_items if not a.done]})
    r.alerts = rules.alerts(open_items, r.checks, CLIENTS[r.client_id], r.segments, r.churn_min)
    r.edited_by, r.approved_by, r.approved_at = user, None, None
    store.save(conn, r)
    if summary is not None:  # the summary is part of what "ask across calls" searches
        ask.index(conn, r, embed_fn or ask.embed)
    return r


def approve(conn, r: CallResult, user: str) -> CallResult:
    r.approved_by, r.approved_at = user, time.time()
    store.save(conn, r)
    return send(conn, r)
