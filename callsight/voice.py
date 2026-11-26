"""How things were said, measured from the audio in code: pitch, pitch movement, loudness and speaking rate per
segment, then compared with the same speaker's own baseline on the call.

This is signal processing, not an emotion model. It doesn't decide churn or sentiment; it finds moments where a
speaker's voice rises well above their own normal (louder, higher, faster, more animated), and a rule flags a call
whose voice and words disagree. Comparing each speaker with themself, not with a global norm, keeps it fair across
voices and accents. ponytail: off-the-shelf speech-emotion models are trained on acted speech and do worst on accents
and phone lines; add one when there are real labelled calls to check it against.
"""
import numpy as np

SR = 16000
FRAME, HOP = 640, 320  # 40 ms frames, 20 ms apart
LAGS = (SR // 400, SR // 70)  # pitch between 70 and 400 Hz


def features(audio: np.ndarray, text: str) -> dict | None:
    """One segment's audio (16 kHz mono float) -> pitch (Hz), pitch movement (semitones), loudness (dB), words/min."""
    if len(audio) < SR // 2:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(audio, FRAME)[::HOP]
    rms = np.sqrt((frames ** 2).mean(axis=1)) + 1e-9
    loud = frames[rms > max(rms.max() * 0.1, 1e-4)]  # skip near-silent frames
    if not len(loud):
        return None
    w = loud * np.hanning(FRAME)
    spec = np.fft.rfft(w, 2 * FRAME)
    ac = np.fft.irfft(spec * np.conj(spec))[:, :FRAME]
    ac = ac / (ac[:, :1] + 1e-9)
    lag = LAGS[0] + ac[:, LAGS[0]:LAGS[1]].argmax(axis=1)
    voiced = ac[np.arange(len(ac)), lag] > 0.45
    f0 = SR / lag[voiced]
    if len(f0) < 5:
        return None
    med = float(np.median(f0))
    return {"pitch": round(med, 1), "pitch_var": round(float(np.std(12 * np.log2(f0 / med))), 2),
            "loud": round(float(20 * np.log10(np.median(np.sqrt((loud ** 2).mean(axis=1))))), 1),
            "wpm": round(len(text.split()) / (len(audio) / SR) * 60)}


KEYS = ("loud", "pitch", "pitch_var", "wpm")


def summarise(segments) -> dict:
    """Per speaker: baseline (median) and talk share; per segment: how far above their own baseline (robust z)."""
    by: dict[str, list] = {}
    for i, s in enumerate(segments):
        if s.voice:
            by.setdefault(s.speaker.split(" (")[0], []).append((i, s))
    total = sum(s.end - s.start for s in segments) or 1
    speakers, moments = {}, []
    for who, items in by.items():
        vals = {k: np.array([s.voice[k] for _, s in items], float) for k in KEYS}
        base = {k: float(np.median(v)) for k, v in vals.items()}
        spread = {k: float(np.median(np.abs(v - base[k]))) * 1.4826 or 1.0 for k, v in vals.items()}
        floor = {"loud": 2.0, "pitch": base["pitch"] * 0.06, "pitch_var": 0.4, "wpm": 15}  # ignore tiny wobbles
        lift = []
        for i, s in items:
            z = {k: (s.voice[k] - base[k]) / max(spread[k], floor[k]) for k in KEYS}
            score = float(np.mean([max(0.0, z[k]) for k in KEYS]))
            lift.append(score)
            if score >= 1.5 and len(items) >= 4:
                up = [k for k in KEYS if z[k] >= 1.5]
                moments.append({"at": s.start, "speaker": who, "score": round(score, 1), "why": up})
        third = max(1, len(lift) // 3)
        speakers[who] = {"talk_share": round(sum(s.end - s.start for _, s in items) / total, 2),
                         "pitch": round(base["pitch"]), "loud": round(base["loud"], 1), "wpm": round(base["wpm"]),
                         "rise": round(float(np.mean(lift[-third:]) - np.mean(lift[:third])), 2) if len(lift) >= 6 else 0.0,
                         "segments": len(items)}
    return {"speakers": speakers, "moments": sorted(moments, key=lambda m: -m["score"])[:8]}
