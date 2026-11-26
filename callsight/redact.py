"""PII masking through PII Shield (github.com/harshitrajchaurasia/pii-shield, running locally).

Fails closed: if PII Shield is down or returns something odd, the call stops here and nothing reaches an LLM.
"""
import os
import re
from difflib import SequenceMatcher

import requests

from .config import CLIENTS
from .models import Segment


class RedactionUnavailable(RuntimeError):
    pass


# Speech-to-text writes details the way they're spoken, which PII Shield's patterns don't expect:
# "jo52 at example.com" and card digits as "4,111,1,100,11,...". Emails are rewritten to the written form before
# masking; any run of 9+ digits still left afterwards (a misheard card or account number) is masked here.
SPOKEN_EMAIL = re.compile(r"\b([\w.+-]+) at ([\w-]+(?:\.[a-z]{2,})+)\b", re.I)
LONG_NUMBER = re.compile(r"\d(?:[\d,. -]*\d)?")


def _mask_long_numbers(text: str) -> str:
    return LONG_NUMBER.sub(lambda m: "[NUMBER]" if sum(ch.isdigit() for ch in m.group()) >= 9 else m.group(), text)


# Types a caller may choose to send unmasked. Contact, ID and money details are always masked.
KEEPABLE = {"NAME", "LOCATION"}
# Company names are not personal data, and "(Acme)" in a speaker label is how the model tells our side from the
# client's, but PII Shield masks some of them ("Acme" as a [LOCATION]). These are always put back.
COMPANIES = {c.name.lower() for c in CLIENTS.values()} | {"acme", "acme leads"}
WORDS = re.compile(r"\[[A-Z_]+\]|\w+|\s+|[^\w\s]")
TOKENS = re.compile(r"(?:\s*\[[A-Z_]+\]\s*)+")


def _unmask(sent: str, got: str, keep: set[str]) -> str:
    """Put back the original words where PII Shield's only change was a token of a kept type (e.g. [NAME]) or a
    known company name. Diffed word by word: a character diff lines up the capitals in "Marcus" with "[NAME]"."""
    kept = re.compile(r"(?:\s*\[(?:" + "|".join(sorted(keep)) + r")\]\s*)+") if keep else None
    a, b = WORDS.findall(sent), WORDS.findall(got)
    out = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        old, new = "".join(a[i1:i2]), "".join(b[j1:j2])
        restore = op == "replace" and ((kept and kept.fullmatch(new)) or (TOKENS.fullmatch(new) and old.strip().lower() in COMPANIES))
        out.append(old if restore else new)
    return "".join(out)  # ponytail: a token the diff can't pair with its words stays masked, which is the safe side


def redact(segments: list[Segment], *, post=requests.post, keep: set[str] = frozenset()) -> list[Segment]:
    if keep - KEEPABLE:
        raise ValueError(f"only {', '.join(sorted(KEEPABLE))} can be left unmasked")
    url = os.getenv("PII_SHIELD_URL", "http://127.0.0.1:18081") + "/api/redact-text"
    # one request per call: PII Shield keeps line breaks, so line n of the answer is segment n. The speaker label is
    # sent too (a pasted transcript labels people by name) and split back off at the first ": ".
    # all whitespace collapsed: PII Shield turns a stray \r into a line break and trims trailing spaces
    lines = [" ".join(SPOKEN_EMAIL.sub(r"\1@\2", f"{s.speaker}: {s.text}").split()) for s in segments]
    try:
        r = post(url, json={"text": "\n".join(lines)}, timeout=30)
        r.raise_for_status()
        out = r.json()["redacted_text"].split("\n")
        if len(out) == len(lines):
            out = [_unmask(a, b, keep) for a, b in zip(lines, out)]
    except Exception as e:
        raise RedactionUnavailable(f"PII Shield failed ({type(e).__name__}); call not processed") from e
    if len(out) != len(lines):
        raise RedactionUnavailable("PII Shield changed the line count; call not processed")
    parts = [line.partition(":") for line in out]
    if not all(sep for _, sep, _ in parts):
        raise RedactionUnavailable("PII Shield changed a speaker label; call not processed")
    return [s.model_copy(update={"speaker": who.strip(), "text": _mask_long_numbers(t.strip())}) for s, (who, _, t) in zip(segments, parts)]


def aliases(segments: list[Segment]) -> dict[str, str]:
    """Speaker name, as first written -> "Person N" ("Speaker 2" won't do: PII Shield masks the word "Speaker")."""
    out: dict[str, str] = {}
    for s in segments:
        name = s.speaker.split(" (")[0].strip()
        if (name and name.lower() not in ("speaker", "account manager", "client") and not name.startswith("Person ")
                and name.lower() not in {k.lower() for k in out}):
            out[name] = f"Person {len(out) + 1}"
    return out


def alias_speakers(segments: list[Segment]) -> list[Segment]:
    """Pasted transcripts label people by name ("Greg (Summit): ..."). Name-finding models miss a bare first name, but
    the labels say exactly which words are names: swap each for "Person N" in the labels and the text. A company in
    brackets is kept, so the model can still tell our side from the client's."""
    alias = {name.lower(): a for name, a in aliases(segments).items()}
    if not alias:
        return segments
    words = re.compile(r"\b(" + "|".join(map(re.escape, sorted(alias, key=len, reverse=True))) + r")\b", re.I)
    swap = lambda t: words.sub(lambda m: alias[m.group(1).lower()], t)
    return [s.model_copy(update={"speaker": swap(s.speaker), "text": swap(s.text)}) for s in segments]
