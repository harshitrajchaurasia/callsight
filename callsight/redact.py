"""PII masking through PII Shield (github.com/harshitrajchaurasia/pii-shield, running locally).

Fails closed: if PII Shield is down or returns something odd, the call stops here and nothing reaches an LLM.
"""
import os

import requests

from .models import Segment


class RedactionUnavailable(RuntimeError):
    pass


def redact(segments: list[Segment], *, post=requests.post) -> list[Segment]:
    url = os.getenv("PII_SHIELD_URL", "http://127.0.0.1:18081") + "/api/redact-text"
    # one request per call: PII Shield keeps line breaks, so line n of the answer is segment n
    lines = [s.text.replace("\n", " ") for s in segments]
    try:
        r = post(url, json={"text": "\n".join(lines)}, timeout=30)
        r.raise_for_status()
        out = r.json()["redacted_text"].split("\n")
    except Exception as e:
        raise RedactionUnavailable(f"PII Shield failed ({type(e).__name__}); call not processed") from e
    if len(out) != len(lines):
        raise RedactionUnavailable("PII Shield changed the line count; call not processed")
    return [s.model_copy(update={"text": t}) for s, t in zip(segments, out)]
