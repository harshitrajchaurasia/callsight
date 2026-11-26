"""Turn a 'Speaker: line' transcript into a two-voice stereo WAV with Piper (local TTS), for testing speech-to-text.
The first speaker is on the left channel and everyone else on the right, like a dual-channel call recording.

Synthetic audio is cleaner than a real phone call, so word error rates measured on it are optimistic.
Usage: python tools/make_audio.py transcript.txt out.wav
"""
import sys
import wave

from piper import PiperVoice

VOICES = ["models/en_US-lessac-medium.onnx", "models/en_US-ryan-medium.onnx"]


def main(src: str, dst: str) -> None:
    voices = [PiperVoice.load(v) for v in VOICES]
    speakers: dict[str, int] = {}
    rate = voices[0].config.sample_rate
    with wave.open(dst, "wb") as out:
        out.setnchannels(2); out.setsampwidth(2); out.setframerate(rate)
        for line in open(src, encoding="utf-8"):
            who, _, text = line.partition(":")
            if not text.strip():
                continue
            side = min(speakers.setdefault(who.split(" (")[0].strip(), len(speakers)), 1)
            pcm = b"".join(c.audio_int16_bytes for c in voices[side].synthesize(text.strip()))
            frames = [pcm[i:i + 2] for i in range(0, len(pcm), 2)]
            out.writeframes(b"".join(f + b"\0\0" for f in frames) if side == 0 else b"".join(b"\0\0" + f for f in frames))
            out.writeframes(b"\0\0\0\0" * int(rate * 0.4))  # pause between turns


if __name__ == "__main__":
    main(*sys.argv[1:3])
