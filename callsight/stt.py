"""Speech-to-text with faster-whisper on CPU.

Stereo recordings (most phone systems can record each side on its own channel) are transcribed per channel, which
labels the speakers for free. Mono files get one speaker label; the extraction model works out who is who.
ponytail: no diarization model (pyannote needs a GPU to be quick). Add one if mono recordings are the norm.
"""
import os
from functools import lru_cache

from .config import RULES
from .models import Segment


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel  # heavy import: only when audio actually comes in
    return WhisperModel(os.getenv("WHISPER_MODEL", "small.en"), device="cpu", compute_type="int8",
                        cpu_threads=int(os.getenv("WHISPER_THREADS", "4")))


def _run(audio, speaker: str) -> list[Segment]:
    segs, _ = _model().transcribe(audio, beam_size=1, vad_filter=True, initial_prompt=RULES["vocabulary"])
    return [Segment(start=round(s.start, 2), end=round(s.end, 2), speaker=speaker, text=s.text.strip()) for s in segs]


def transcribe(path: str) -> list[Segment]:
    import av
    from faster_whisper.audio import decode_audio
    with av.open(path) as f:
        stereo = f.streams.audio[0].channels == 2
    if not stereo:
        return _run(path, "Speaker")
    names = os.getenv("CHANNEL_SPEAKERS", "Account manager,Client").split(",")
    left, right = decode_audio(path, split_stereo=True)
    return sorted(_run(left, names[0]) + _run(right, names[1]), key=lambda s: s.start)


def from_text(text: str) -> list[Segment]:
    """A pasted 'Speaker: words' transcript -> segments (no timestamps, so line numbers stand in)."""
    out = []
    for n, line in enumerate(x for x in text.splitlines() if x.strip()):
        who, sep, words = line.partition(":")
        if not sep or len(who) > 40:
            who, words = "Speaker", line
        out.append(Segment(start=n, end=n, speaker=who.strip(), text=words.strip()))
    return out
