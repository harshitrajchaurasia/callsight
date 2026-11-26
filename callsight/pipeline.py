"""audio or transcript -> segments -> PII Shield -> LLM extraction -> quote check -> rules -> store."""
import time

from . import extract, redact, rules, store, stt
from .config import CLIENTS
from .models import CallResult


def process(call_id: str, client_id: str, conn, *, audio_path: str | None = None, text: str | None = None,
            provider: str | None = None, analyze_fn=None, redact_fn=None) -> CallResult:
    client = CLIENTS.get(client_id)
    if client is None:
        raise KeyError(f"unknown client {client_id}")
    if (audio_path is None) == (text is None):
        raise ValueError("give exactly one of audio_path or text")
    start = time.perf_counter()
    segments = stt.transcribe(audio_path) if audio_path else stt.from_text(text)
    segments = (redact_fn or redact.redact)(segments)  # raises if PII Shield is down: nothing unmasked goes further
    provider = provider or extract.default_provider()
    insight = extract.analyze(segments, client, provider, analyze_fn)
    checks = extract.check_quotes(insight, segments)
    result = CallResult(id=call_id, client_id=client_id, segments=segments, insight=insight, checks=checks,
                        alerts=rules.alerts(insight, checks, client), provider=provider,
                        latency_ms=int((time.perf_counter() - start) * 1000))
    store.save(conn, result)
    return result
