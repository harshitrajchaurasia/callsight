import pytest

from callsight import extract, pipeline, redact, rules, store, stt
from callsight.config import CLIENTS
from callsight.models import ActionItem, Insight, Segment, Signal

SEGS = stt.from_text("AM: How have the IUL leads been?\nClient: Honestly we're looking at other vendors.\n"
                     "Client: Half the numbers are disconnected, third time I'm telling you.\nAM: I'll call you back by Friday.")


def insight(**kw):
    base = dict(summary="s", sentiment="frustrated", churn_risk=0.8, upsell_signals=[], objections=[],
                churn_reasons=[Signal(text="other vendors", quote="we're looking at other vendors")],
                action_items=[ActionItem(owner="us", task="call back", due="Friday", quote="I'll call you back by Friday")])
    return Insight(**(base | kw))


def test_from_text_keeps_speakers():
    assert [s.speaker for s in SEGS] == ["AM", "Client", "Client", "AM"]
    assert stt.from_text("no speaker here")[0].speaker == "Speaker"


def test_find_quote_exact_fuzzy_and_missing():
    assert extract.find_quote("We're looking at other vendors", SEGS) == 1
    assert extract.find_quote("half the numbers are disconnected third time", SEGS) == 2  # punctuation dropped
    assert extract.find_quote("vendors. Half the numbers", SEGS) == 1  # spans two lines
    assert extract.find_quote("we are cancelling today", SEGS) is None
    assert extract.find_quote("IUL", SEGS) is None  # too short to mean anything


def test_churn_alert_needs_a_found_quote():
    client = CLIENTS["C-03"]
    i = insight()
    assert any("call back today" in a for a in rules.alerts(i, extract.check_quotes(i, SEGS), client))
    fake = insight(churn_reasons=[Signal(text="cancelling", quote="we will cancel at the end of the month")])
    out = rules.alerts(fake, extract.check_quotes(fake, SEGS), client)
    assert any("no supporting quote" in a for a in out) and any("unverified" in a for a in out)
    assert not any("call back today" in a for a in out)


def test_unverified_upsell_does_not_alert():
    i = insight(churn_risk=0.1, churn_reasons=[], upsell_signals=[Signal(text="more states", quote="add Texas and Ohio next month")])
    assert not any(a.startswith("Upsell") for a in rules.alerts(i, extract.check_quotes(i, SEGS), CLIENTS["C-03"]))


class Resp:
    def __init__(self, body, code=200): self.body, self.code = body, code
    def raise_for_status(self):
        if self.code >= 400: raise RuntimeError(self.code)
    def json(self): return self.body


def test_redact_maps_lines_back():
    out = redact.redact(SEGS[:2], post=lambda url, **kw: Resp({"redacted_text": kw["json"]["text"].replace("vendors", "[X]")}))
    assert out[1].text == "Honestly we're looking at other [X]." and out[1].speaker == "Client"


@pytest.mark.parametrize("post", [lambda url, **kw: Resp({}, 503), lambda url, **kw: Resp({"redacted_text": "one line"}),
                                  lambda url, **kw: (_ for _ in ()).throw(ConnectionError())])
def test_redact_fails_closed(post):
    with pytest.raises(redact.RedactionUnavailable):
        redact.redact(SEGS, post=post)


def test_pipeline_never_calls_model_without_redaction(tmp_path):
    called = []
    def boom(_): raise redact.RedactionUnavailable("down")
    with pytest.raises(redact.RedactionUnavailable):
        pipeline.process("X", "C-03", store.connect(str(tmp_path / "t.db")), text="AM: hi there",
                         analyze_fn=lambda *a: called.append(1), redact_fn=boom)
    assert not called


def test_pipeline_end_to_end_offline(tmp_path):
    conn = store.connect(str(tmp_path / "t.db"))
    seen = []
    r = pipeline.process("CALL-T", "C-03", conn, text="\n".join(f"{s.speaker}: {s.text}" for s in SEGS), provider="claude",
                         redact_fn=lambda segs: [s.model_copy(update={"text": s.text.upper()}) for s in segs],
                         analyze_fn=lambda segs, client: seen.append(segs) or insight())
    assert seen[0][0].text.isupper()  # the model only saw redacted text
    assert all(c.found for c in r.checks) and store.get(conn, "CALL-T").insight.churn_risk == 0.8
    with pytest.raises(KeyError):
        pipeline.process("Y", "NOPE", conn, text="a: b", analyze_fn=lambda *a: insight(), redact_fn=lambda s: s)


def test_analyze_retries_once_then_gives_up():
    calls = []
    def flaky(segs, client):
        calls.append(1)
        if len(calls) == 1: raise ValueError("bad json")
        return insight()
    assert extract.analyze(SEGS, CLIENTS["C-03"], "claude", flaky).churn_risk == 0.8
    with pytest.raises(extract.ModelUnavailable):
        extract.analyze(SEGS, CLIENTS["C-03"], "claude", lambda *a: (_ for _ in ()).throw(ValueError()))
