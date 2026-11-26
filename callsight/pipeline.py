"""audio or transcript -> segments -> PII Shield -> LLM extraction -> quote check -> rules -> store -> Monday / GHL."""
import os
import time
from functools import partial

from . import ask, extract, redact, rules, sinks, store, stt, voice
from .config import CLIENTS, RULES
from .models import ActionItem, CallResult, Segment


def process(call_id: str, client_id: str, conn, *, audio_path: str | None = None, text: str | None = None,
            segments: list[Segment] | None = None, provider: str | None = None, keep: set[str] = frozenset(),
            churn_min: float | None = None, write_out: bool = True, owner: str | None = None,
            speakers: dict[str, str] | None = None, rerun_of: str | None = None, on_progress=None,
            analyze_fn=None, second_fn=None, redact_fn=None, embed_fn=None) -> CallResult:
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
    threshold = RULES["churn_alert_min"] if churn_min is None else churn_min
    second = _second_opinion(segments, client, provider, insight, threshold, analyze_fn, second_fn, step)
    checks = extract.check_quotes(insight, segments)
    heard = voice.summarise(segments)
    result = CallResult(id=call_id, client_id=client_id, segments=segments, insight=insight, checks=checks,
                        alerts=rules.alerts(insight, checks, client, segments, churn_min)
                        + rules.review_flags(insight, heard, second, threshold, provider), provider=provider,
                        voice=heard, second_opinion=second,
                        unmasked=sorted(keep), speakers=names, churn_min=churn_min, owner=owner, rerun_of=rerun_of,
                        latency_ms=int((time.perf_counter() - start) * 1000))
    store.save(conn, result)  # saved before any write-out, so a Monday/GHL failure never loses the call
    ask.index(conn, result, embed_fn or ask.embed)
    if write_out:
        send(conn, result)
    return result


def _second_opinion(segments, client, provider, insight, threshold, analyze_fn, second_fn, step) -> dict | None:
    """A borderline reading gets a second model's view (SECOND_OPINION=0 turns it off). Grok costs ~0.2 cents."""
    other = "grok" if provider == "claude" else "claude"
    if not rules.borderline(insight, threshold) or os.getenv("SECOND_OPINION", "1") == "0":
        return None
    if second_fn is None and (analyze_fn is not None or not os.getenv("XAI_API_KEY" if other == "grok" else "ANTHROPIC_API_KEY")):
        return None
    step("second opinion", provider=other)
    try:
        i2 = extract.analyze(segments, client, other, second_fn)
    except extract.ModelUnavailable:
        return None
    return {"provider": other, "churn_risk": i2.churn_risk, "sentiment": i2.sentiment}


def send(conn, r: CallResult) -> CallResult:
    """Write the call to Monday and GHL (each is off unless configured). Approved again after changes: the item
    already exists, so the new report goes on it as an update instead of a second item."""
    if r.monday_item_id:
        sinks.monday_update(r.monday_item_id, [f"Approved again after changes by {r.approved_by or 'a user'}.", *sinks.report(r)])
    elif item := sinks.to_monday(r):
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
    what = lambda i: (i.summary, i.sentiment, i.churn_risk, [(a.owner, a.speaker, a.task, a.due) for a in i.action_items])
    before, was_done = what(r.insight), {a.task: a.done for a in r.insight.action_items}
    r.insight = r.insight.model_copy(update=changes)
    content_changed = what(r.insight) != before  # ticking a follow-up off isn't a change to what the call says
    open_items = r.insight.model_copy(update={"action_items": [a for a in r.insight.action_items if not a.done]})
    threshold = RULES["churn_alert_min"] if r.churn_min is None else r.churn_min
    r.alerts = rules.alerts(open_items, r.checks, CLIENTS[r.client_id], r.segments, r.churn_min) + rules.review_flags(
        r.insight, r.voice, None if churn_risk is not None or sentiment is not None else r.second_opinion, threshold, r.provider)
    if content_changed:
        r.edited_by, r.approved_by, r.approved_at = user, None, None  # needs approving again
    ticked = [a for a in r.insight.action_items if a.task in was_done and a.done != was_done[a.task]]
    if ticked and r.monday_item_id and r.approved_by is not None:  # keep the Monday item in step
        sinks.monday_update(r.monday_item_id, [f"{'Done' if a.done else 'Reopened'}: {a.task} (ticked by {user or 'a user'})" for a in ticked])
    store.save(conn, r)
    if summary is not None:  # the summary is part of what "ask across calls" searches
        ask.index(conn, r, embed_fn or ask.embed)
    return r


def approve(conn, r: CallResult, user: str) -> CallResult:
    r.approved_by, r.approved_at = user, time.time()
    store.save(conn, r)
    return send(conn, r)
