"""Voice measurement on synthetic signals with known answers, and the rules that use it."""
import numpy as np

from callsight import pipeline, rules, store, voice
from callsight.models import Segment
from tests.test_core import fake_embed, insight

SR = voice.SR


def tone(hz, amp, secs=2.0, wobble=0.0):
    t = np.arange(int(SR * secs)) / SR
    f = hz * (1 + wobble * np.sin(2 * np.pi * 3 * t))
    return (amp * np.sin(2 * np.pi * np.cumsum(f) / SR)).astype(np.float32)


def test_features_measure_pitch_loudness_and_pace():
    a, b = voice.features(tone(150, 0.1), "one two three four"), voice.features(tone(220, 0.4), "one two three four five six")
    assert abs(a["pitch"] - 150) < 6 and abs(b["pitch"] - 220) < 8
    assert b["loud"] - a["loud"] > 10  # 4x the amplitude is +12 dB
    assert (a["wpm"], b["wpm"]) == (120, 180)
    assert voice.features(tone(220, 0.4, wobble=0.1), "x")["pitch_var"] > a["pitch_var"] + 0.5
    assert voice.features(np.zeros(SR, np.float32), "quiet") is None


def _call(raised_last=3):
    segs = []
    for n in range(9):
        hot = n >= 9 - raised_last
        f = voice.features(tone(240 if hot else 160, 0.5 if hot else 0.1), "word " * (8 if hot else 5))
        segs.append(Segment(start=n * 3, end=n * 3 + 2, speaker="Client", text="...", voice=f))
    segs.append(Segment(start=40, end=42, speaker="Account manager", text="ok", voice=voice.features(tone(120, 0.1), "ok then")))
    return segs


def test_a_voice_rising_above_its_own_normal_is_found():
    s = voice.summarise(_call())
    assert s["speakers"]["Client"]["rise"] >= 0.8 and len(s["moments"]) == 3
    assert {"loud", "pitch", "wpm"} <= set(s["moments"][0]["why"])
    assert voice.summarise(_call(raised_last=0))["moments"] == []  # a steady voice: nothing to flag


def test_voice_and_words_disagreeing_goes_to_a_person():
    heard = voice.summarise(_call())
    calm = insight(sentiment="neutral", churn_risk=0.2)
    assert any("voice rose" in f for f in rules.review_flags(calm, heard, None, 0.6))
    assert not rules.review_flags(insight(sentiment="angry"), heard, None, 0.6)  # the words already say it
    assert not rules.review_flags(calm, voice.summarise(_call(raised_last=0)), None, 0.6)


def test_borderline_readings_get_a_second_opinion(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "connect", store.connect)
    conn = store.connect(str(tmp_path / "t.db"))
    asked = []
    second = lambda segs, client: asked.append(1) or insight(churn_risk=0.1, sentiment="positive")
    r = pipeline.process("C1", "C-03", conn, text="AM: hi", provider="claude", write_out=False, redact_fn=lambda s: s,
                         embed_fn=fake_embed, analyze_fn=lambda s, c: insight(churn_risk=0.55, sentiment="frustrated"), second_fn=second)
    assert asked and r.second_opinion == {"provider": "grok", "churn_risk": 0.1, "sentiment": "positive"}
    assert any(a.startswith("Second opinion disagrees") for a in r.alerts)
    r2 = pipeline.process("C2", "C-03", conn, text="AM: hi", provider="claude", write_out=False, redact_fn=lambda s: s,
                          embed_fn=fake_embed, analyze_fn=lambda s, c: insight(churn_risk=0.95, sentiment="angry"), second_fn=second)
    assert len(asked) == 1 and r2.second_opinion is None  # a clear reading isn't second-guessed
