"""Speech-to-text by accent and line quality: American vs Indian-accented English, clean vs phone audio.

Voices are synthetic (Piper): two US voices, and three speakers of the VCTK voice who were recorded with Indian
accents (p248, p251, p376). "Phone" = mono, band-limited to 300-3400 Hz at 8 kHz, mu-law, white noise at 15 dB SNR.
Real recordings will differ; this compares models and conditions, it doesn't promise a number for a real call.
Usage: python -m evals.stt_bench small.en medium.en ...   (VCTK voice path in VCTK_VOICE)
"""
import json
import os
import sys
import time

import numpy as np
from faster_whisper import WhisperModel
from piper import PiperVoice, SynthesisConfig

from callsight.stt import PROMPT
from evals.run_eval import wer, words

LINES = [
    "Honestly the Louisiana leads have been a real problem since mid October.",
    "We closed six policies off your leads and the contact rate is way up.",
    "If that doesn't change by January we'll move the budget to a cheaper vendor.",
    "Can you send me the consent records for those two leads before Friday?",
    "I'd like to add Arizona and maybe Nevada next quarter if the price is right.",
    "Our agents prefer the morning appointment reminders, the evening ones get ignored.",
    "The final expense pilot sounds interesting, send me a proposal with the numbers.",
    "We were charged twice in November, four hundred and ninety dollars.",
    "Create a task for Priya Shah to send the Texas quote to Summit Annuity Group by Friday.",
    "Remind Marcus Lee to call Golden Years Final Expense back tomorrow morning.",
    "Assign Dana Brooks the consent export for Heritage Final Expense Co, due next Wednesday.",
    "Add a note to BlueOak IUL Partners that they want Nevada leads in January.",
    "Log that Harbor Retirement Advisors renewed for another year at sixty leads a month.",
    "Ask Northstar Retirement Planning for the webinar dates and put it on my list.",
]
NAMES = ["priya shah", "marcus lee", "dana brooks", "summit annuity", "golden years", "heritage final expense",
         "blueoak", "harbor retirement", "northstar retirement"]


def voices():
    out = [("US", "lessac", PiperVoice.load("models/en_US-lessac-medium.onnx"), None),
           ("US", "ryan", PiperVoice.load("models/en_US-ryan-medium.onnx"), None)]
    vctk = PiperVoice.load(os.environ["VCTK_VOICE"])
    for spk in ("p248", "p251", "p376"):
        out.append(("IN", spk, vctk, vctk.config.speaker_id_map[spk]))
    return out


def synth(voice, sid, text) -> tuple[np.ndarray, int]:
    cfg = SynthesisConfig(speaker_id=sid) if sid is not None else None
    pcm = b"".join(c.audio_int16_bytes for c in voice.synthesize(text, syn_config=cfg))
    return np.frombuffer(pcm, np.int16).astype(np.float32) / 32768, voice.config.sample_rate


def resample(x, sr, to):
    t = np.arange(int(len(x) * to / sr)) * sr / to
    return np.interp(t, np.arange(len(x)), x).astype(np.float32)


def bandpass(x, sr, lo, hi):
    f = np.fft.rfft(x); hz = np.fft.rfftfreq(len(x), 1 / sr)
    f[(hz < lo) | (hz > hi)] = 0
    return np.fft.irfft(f, len(x)).astype(np.float32)


def phone(x, sr, rng):
    y = resample(bandpass(x, sr, 300, 3400), sr, 8000)
    y = y + rng.normal(0, np.sqrt(np.mean(y ** 2) / 10 ** 1.5), len(y)).astype(np.float32)  # 15 dB SNR
    mu = 255.0
    y = np.sign(y) * np.log1p(mu * np.minimum(np.abs(y), 1)) / np.log1p(mu)
    y = np.round(y * 127) / 127  # 8-bit mu-law
    y = np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu
    return resample(y, 8000, 16000)


def main(models):
    conds = os.getenv("CONDS", "clean,phone").split(","); groups = os.getenv("GROUPS", "US,IN").split(",")
    rng = np.random.default_rng(0)
    clips = []
    for group, name, v, sid in voices():
        if group not in groups:
            continue
        for line in LINES:
            x, sr = synth(v, sid, line)
            clips.append((group, name, line, {"clean": resample(x, sr, 16000), "phone": phone(x, sr, rng)}))
    results = {}
    for spec in models:
        m, _, beam = spec.partition(":")  # "small.en:5" = beam size 5
        model = WhisperModel(m, device="cpu", compute_type="int8", cpu_threads=int(os.getenv("WHISPER_THREADS", "2")))
        for cond in conds:
            for group in groups:
                ref, hyp, names_ok, names_n, audio_s, cpu_s = [], [], 0, 0, 0.0, 0.0
                for g, _, line, audio in clips:
                    if g != group:
                        continue
                    t = time.perf_counter()
                    segs, _ = model.transcribe(audio[cond], language="en", beam_size=int(beam or 1), vad_filter=True, initial_prompt=PROMPT)
                    text = " ".join(s.text for s in segs)
                    cpu_s += time.perf_counter() - t; audio_s += len(audio[cond]) / 16000
                    ref += words(line); hyp += words(text)
                    for n in NAMES:
                        if n in line.lower():
                            names_n += 1; names_ok += n in " ".join(words(text))
                key = f"{spec}|{cond}|{group}"
                results[key] = {"wer": round(wer(ref, hyp), 3), "names": f"{names_ok}/{names_n}", "rtf": round(cpu_s / audio_s, 2)}
                print(key, results[key], flush=True)
    old = json.load(open("evals/stt_bench.json")) if os.path.exists("evals/stt_bench.json") else {}
    json.dump(old | results, open("evals/stt_bench.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["small.en"])
