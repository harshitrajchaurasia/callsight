import pytest

import numpy as np

from callsight import ask, extract, pipeline, redact, rules, store, stt
from callsight.config import CLIENTS
from callsight.models import ActionItem, Insight, Segment, Signal

SEGS = stt.from_text("AM: How have the IUL leads been?\nClient: Honestly we're looking at other vendors.\n"
                     "Client: Half the numbers are disconnected, third time I'm telling you.\nAM: I'll call you back by Friday.")


def fake_embed(texts):
    """Bag of words hashed into 64 dims, normalised: enough for 'shared words score higher'."""
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in extract._norm(t).split():
            out[i, sum(map(ord, w)) % 64] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


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
                         analyze_fn=lambda *a: called.append(1), redact_fn=boom, embed_fn=fake_embed)
    assert not called


def test_pipeline_end_to_end_offline(tmp_path):
    conn = store.connect(str(tmp_path / "t.db"))
    seen = []
    r = pipeline.process("CALL-T", "C-03", conn, text="\n".join(f"{s.speaker}: {s.text}" for s in SEGS), provider="claude",
                         redact_fn=lambda segs: [s.model_copy(update={"text": s.text.upper()}) for s in segs],
                         analyze_fn=lambda segs, client: seen.append(segs) or insight(), embed_fn=fake_embed)
    assert seen[0][0].text.isupper()  # the model only saw redacted text
    assert all(c.found for c in r.checks) and store.get(conn, "CALL-T").insight.churn_risk == 0.8
    with pytest.raises(KeyError):
        pipeline.process("Y", "NOPE", conn, text="a: b", analyze_fn=lambda *a: insight(), redact_fn=lambda s: s, embed_fn=fake_embed)


def test_analyze_retries_once_then_gives_up():
    calls = []
    def flaky(segs, client):
        calls.append(1)
        if len(calls) == 1: raise ValueError("bad json")
        return insight()
    assert extract.analyze(SEGS, CLIENTS["C-03"], "claude", flaky).churn_risk == 0.8
    with pytest.raises(extract.ModelUnavailable):
        extract.analyze(SEGS, CLIENTS["C-03"], "claude", lambda *a: (_ for _ in ()).throw(ValueError()))


def test_ask_finds_the_right_call_and_checks_citations(tmp_path):
    conn = store.connect(str(tmp_path / "t.db"))
    for cid, client, text in [("A", "C-03", "AM: hi\nClient: our budget was cut so we may pause in January"),
                              ("B", "C-01", "AM: hi\nClient: the Medicare leads are great, add Texas please")]:
        pipeline.process(cid, client, conn, text=text, redact_fn=lambda s: s, embed_fn=fake_embed, provider="claude",
                         analyze_fn=lambda *a: insight(churn_risk=0.1, churn_reasons=[], action_items=[]))
    hits = ask.search(conn, "budget cut pause", embed_fn=fake_embed)
    assert hits[0]["call_id"] == "A"
    out = ask.answer(conn, "who had budget cuts?", provider="claude", embed_fn=fake_embed,
                     complete_fn=lambda sys, user, p: "Harbor's client, budget cut [S1]. Also [S99].")
    assert [s["n"] for s in out["sources"]] == [1] and out["unknown_citations"] == [99]
    pipeline.process("A", "C-03", conn, text="AM: reprocessed", redact_fn=lambda s: s, embed_fn=fake_embed, provider="claude",
                     analyze_fn=lambda *a: insight(churn_risk=0.1, churn_reasons=[], action_items=[]))
    assert not any("budget" in t for (t,) in conn.execute("SELECT text FROM chunks WHERE call_id='A'"))  # re-index replaces


def test_ask_with_no_calls(tmp_path):
    out = ask.answer(store.connect(str(tmp_path / "t.db")), "anything", complete_fn=lambda *a: 1 / 0, embed_fn=fake_embed)
    assert out["sources"] == []


def test_sinks_monday_and_ghl(tmp_path, monkeypatch):
    sent = []
    subscribers = [{"id": "1"}]
    def post(url, **kw):
        sent.append((url, kw["json"]))
        if "subscribers" in kw["json"].get("query", ""):
            return Resp({"data": {"boards": [{"subscribers": subscribers}]}})
        return Resp({"data": {"x": {"id": 42}}})
    r = pipeline.process("CALL-M", "C-03", store.connect(str(tmp_path / "t.db")), text="AM: I'll call you back by Friday.",
                         provider="claude", redact_fn=lambda s: s, embed_fn=fake_embed, analyze_fn=lambda *a: insight())
    from callsight import sinks
    assert sinks.to_monday(r, post=post) is None and not sent  # off without env
    monkeypatch.setenv("MONDAY_API_TOKEN", "t"); monkeypatch.setenv("MONDAY_CALLS_BOARD_ID", "1")
    assert sinks.to_monday(r, post=post) is None  # no allowlist: never write
    monkeypatch.setenv("MONDAY_ALLOWED_SUBSCRIBERS", "1")
    subscribers.append({"id": "2"})
    assert sinks.to_monday(r, post=post) is None and not any("create_item" in j.get("query", "") for _, j in sent)
    subscribers.pop()
    assert sinks.to_monday(r, post=post) == "42"
    queries = " ".join(j["query"] for _, j in sent)
    assert "create_subitem" in queries and "notification" not in queries and "people" not in queries
    assert any("[unverified]" in j["variables"].get("body", "") for _, j in sent)  # churn quote isn't in this transcript
    monkeypatch.setenv("GHL_API_KEY", "k"); monkeypatch.setenv("GHL_CONTACTS", '{"C-03": "abc"}')
    assert sinks.to_ghl(r, post=post) and sent[-1][0].endswith("/contacts/abc/notes")
