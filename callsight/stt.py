"""Speech-to-text with faster-whisper on CPU.

Stereo recordings (most phone systems can record each side on its own channel) are transcribed per channel, which
labels the speakers for free. Mono files get one speaker label; the extraction model works out who is who.
ponytail: no diarization model (pyannote needs a GPU to be quick). Add one if mono recordings are the norm.
"""
import os
from functools import lru_cache

from .config import CLIENTS, RULES
from .models import Segment


# Words Whisper should expect. Names are where accents hurt most ("Priya Shah" heard as "Maria Shaw"), so the
# team's and the clients' names go in too.
PROMPT = ", ".join([RULES["vocabulary"], *sorted({c.account_manager for c in CLIENTS.values()}), *(c.name for c in CLIENTS.values())])


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel  # heavy import: only when audio actually comes in
    return WhisperModel(os.getenv("WHISPER_MODEL", "small.en"), device="cpu", compute_type="int8",
                        cpu_threads=int(os.getenv("WHISPER_THREADS", "4")))


def _run(audio, speaker: str, gap: float = 1.0, on_progress=None) -> list[Segment]:
    """Whisper's segments can span a pause in which the other channel spoke, which puts turns out of order. Word
    timestamps let us cut a segment wherever this speaker is silent for more than `gap` seconds."""
    segs, info = _model().transcribe(audio, language="en", beam_size=1, vad_filter=True, word_timestamps=True, initial_prompt=PROMPT)
    words = []
    for s in segs:  # a generator: Whisper decodes as we iterate, so this is where the time goes
        words += s.words
        if on_progress:
            on_progress(min(1.0, s.end / max(info.duration, 1)), info.duration)
    out: list[list] = []
    for w in words:
        if not out or w.start - out[-1][-1].end > gap:
            out.append([])
        out[-1].append(w)
    return [Segment(start=round(ws[0].start, 2), end=round(ws[-1].end, 2), speaker=speaker,
                    text="".join(w.word for w in ws).strip()) for ws in out if "".join(w.word for w in ws).strip()]


def transcribe(path: str, on_progress=None) -> list[Segment]:
    """on_progress(fraction, seconds_of_audio) is called as Whisper works through the recording."""
    import av
    from faster_whisper.audio import decode_audio
    with av.open(path) as f:
        stereo = f.streams.audio[0].channels == 2
    if not stereo:
        return _run(path, "Speaker", on_progress=on_progress)
    names = os.getenv("CHANNEL_SPEAKERS", "Account manager,Client").split(",")
    left, right = decode_audio(path, split_stereo=True)
    half = (lambda base: lambda f, d: on_progress(base + f / 2, d)) if on_progress else (lambda base: None)
    return sorted(_run(left, names[0], on_progress=half(0)) + _run(right, names[1], on_progress=half(0.5)), key=lambda s: s.start)


def from_text(text: str) -> list[Segment]:
    """A pasted 'Speaker: words' transcript -> segments (no timestamps, so line numbers stand in)."""
    out = []
    for n, line in enumerate(x for x in text.splitlines() if x.strip()):
        who, sep, words = line.partition(":")
        if not sep or len(who) > 40:
            who, words = "Speaker", line
        out.append(Segment(start=n, end=n, speaker=who.strip(), text=words.strip()))
    return out
