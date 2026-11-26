"""PII masking through PII Shield (github.com/harshitrajchaurasia/pii-shield, running locally).

Fails closed: if PII Shield is down or returns something odd, the call stops here and nothing reaches an LLM.
"""
import os
import re

import requests

from .models import Segment


class RedactionUnavailable(RuntimeError):
    pass


def redact(segments: list[Segment], *, post=requests.post) -> list[Segment]:
    url = os.getenv("PII_SHIELD_URL", "http://127.0.0.1:18081") + "/api/redact-text"
    # one request per call: PII Shield keeps line breaks, so line n of the answer is segment n. The speaker label is
    # sent too (a pasted transcript labels people by name) and split back off at the first ": ".
    lines = [f"{s.speaker}: {s.text}".replace("\n", " ") for s in segments]
    try:
        r = post(url, json={"text": "\n".join(lines)}, timeout=30)
        r.raise_for_status()
        out = r.json()["redacted_text"].split("\n")
    except Exception as e:
        raise RedactionUnavailable(f"PII Shield failed ({type(e).__name__}); call not processed") from e
    if len(out) != len(lines):
        raise RedactionUnavailable("PII Shield changed the line count; call not processed")
    parts = [line.partition(": ") for line in out]
    if not all(sep for _, sep, _ in parts):
        raise RedactionUnavailable("PII Shield changed a speaker label; call not processed")
    return [s.model_copy(update={"speaker": who, "text": t}) for s, (who, _, t) in zip(segments, parts)]


def alias_speakers(segments: list[Segment]) -> list[Segment]:
    """Pasted transcripts label people by name ("Greg (Summit): ..."). Name-finding models miss a bare first name, but
    the labels say exactly which words are names: swap each for "Speaker N" in the labels and the text. A company in
    brackets is kept, so the model can still tell our side from the client's."""
    alias: dict[str, str] = {}
    for s in segments:
        name = s.speaker.split(" (")[0].strip()
        if name and name.lower() not in ("speaker", "account manager", "client"):
            alias.setdefault(name.lower(), f"Speaker {len(alias) + 1}")
    if not alias:
        return segments
    words = re.compile(r"\b(" + "|".join(map(re.escape, sorted(alias, key=len, reverse=True))) + r")\b", re.I)
    swap = lambda t: words.sub(lambda m: alias[m.group(1).lower()], t)
    return [s.model_copy(update={"speaker": swap(s.speaker), "text": swap(s.text)}) for s in segments]
