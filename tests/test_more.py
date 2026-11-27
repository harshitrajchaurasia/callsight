"""Roles and visibility, delete, re-run, progress, usage, and voice commands. Offline: models, PII Shield and Monday are stubbed."""
import json

from fastapi.testclient import TestClient

import api
from callsight import ask, command, extract, redact, sinks, store
from callsight.models import Segment
from tests.test_core import fake_embed, insight


def _app(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "conn", store.connect(str(tmp_path / "t.db")))
    monkeypatch.setattr(redact, "redact", lambda segs, **kw: segs)
    monkeypatch.setattr(ask, "embed", fake_embed)
    usage = type("Usage", (), {"input_tokens": 100, "output_tokens": 10})()
    monkeypatch.setitem(extract.PROVIDERS, "claude", lambda segs, client: extract._record("claude", "m", usage, 0) or insight())
    monkeypatch.setitem(extract.PROVIDERS, "grok", lambda segs, client: insight(churn_risk=0.2))
    monkeypatch.setattr(extract, "USAGE_PATH", str(tmp_path / "usage.jsonl"))
    monkeypatch.setenv("DEMO_USER", "admin"); monkeypatch.setenv("DEMO_PASS", "p"); monkeypatch.setenv("EXTRA_LOGINS", "garrett:q")
    api._hits.clear(); api._fails.clear()
    return TestClient(api.app, base_url="https://testserver")


def _login(c, u, p):
    c.cookies.clear()
    assert c.post("/api/login", json={"username": u, "password": p}).status_code == 200


def test_members_see_shared_and_their_own_calls(tmp_path, monkeypatch):
    c = _app(tmp_path, monkeypatch)
    form = {"client_id": "C-03", "provider": "claude", "transcript": "AM: hi"}
    _login(c, "admin", "p"); mine = c.post("/api/calls", data=form).json()["id"]
    _login(c, "garrett", "q"); his = c.post("/api/calls", data=form).json()["id"]
    assert [x["id"] for x in c.get("/api/calls").json()] == [his]
    assert c.get(f"/api/calls/{mine}").status_code == 404
    assert c.get("/api/settings").json()["admin"] is False and c.get("/api/usage").status_code == 403
    _login(c, "admin", "p")
    assert {x["id"] for x in c.get("/api/calls").json()} == {mine, his}
    assert c.get("/api/usage").json()["by_user"].keys() == {"admin", "garrett"}  # usage is logged per user
    _login(c, "garrett", "q")
    assert c.delete(f"/api/calls/{his}").json() == {"ok": True}
    assert c.delete(f"/api/calls/{mine}").status_code == 404


def test_shared_calls_only_an_admin_deletes(tmp_path, monkeypatch):
    c = _app(tmp_path, monkeypatch)
    _login(c, "admin", "p")
    shared = c.post("/api/calls", data={"client_id": "C-03", "provider": "claude", "transcript": "AM: hi"}).json()["id"]
    r = store.get(api.conn, shared); r.owner = None; store.save(api.conn, r)  # like the seeded demo calls
    _login(c, "garrett", "q")
    assert c.get(f"/api/calls/{shared}").status_code == 200
    assert c.delete(f"/api/calls/{shared}").status_code == 403


def test_rerun_with_another_model_and_progress(tmp_path, monkeypatch):
    c = _app(tmp_path, monkeypatch); _login(c, "admin", "p")
    first = c.post("/api/calls", data={"client_id": "C-03", "provider": "claude", "transcript": "Marcus (Acme): hi",
                                        "job": "abcdefgh12"}).json()
    assert c.get("/api/progress/abcdefgh12").json()["stage"] == "checking"
    again = c.post(f"/api/calls/{first['id']}/rerun", json={"provider": "grok"}).json()
    assert again["rerun_of"] == first["id"] and again["provider"] == "grok" and again["insight"]["churn_risk"] == 0.2
    assert again["speakers"] == first["speakers"] == {"Person 1": "Marcus"}
    assert [s["text"] for s in again["segments"]] == [s["text"] for s in first["segments"]]


def test_command_matches_people_and_clients_in_code():
    plan = command.Plan(reply="ok", actions=[
        command.Action(kind="task", text="send the Texas quote", assignee="Prya", client="summit annuity", due="Friday"),
        command.Action(kind="task", text="call back", assignee="me", client=None, due=None),
        command.Action(kind="task", text="x", assignee="Bob Smith", client="Nowhere Inc", due=None)])
    out = command.understand("whatever", "garrett", "claude", model_fn=lambda *a: plan, redact_fn=lambda segs, **kw: segs)
    a, b, x = out["actions"]
    assert (a["assignee"], a["known"], a["client_id"]) == ("Priya Shah", True, "C-02")
    assert (b["assignee"], b["known"]) == ("garrett", True)
    assert (x["assignee"], x["known"], x["client_id"]) == ("Bob Smith", False, None)  # never guessed


def test_command_masks_contact_details_but_keeps_names():
    seen = []
    def shield(segs, keep):
        seen.append(keep); return [s.model_copy(update={"text": s.text.replace("555-0187", "[PHONE]")}) for s in segs]
    out = command.understand("ask Priya to call 555-0187", "", "claude", redact_fn=shield,
                             model_fn=lambda system, prompt, schema, provider: seen.append(prompt) or command.Plan(actions=[], reply="none"))
    assert seen[0] == {"NAME", "LOCATION"} and "555-0187" not in seen[1] and "Priya" in seen[1]


def test_confirmed_task_goes_to_monday_safely(tmp_path, monkeypatch):
    conn = store.connect(str(tmp_path / "t.db"))
    monkeypatch.setenv("MONDAY_API_TOKEN", "t"); monkeypatch.setenv("MONDAY_CALLS_BOARD_ID", "5")
    monkeypatch.setenv("MONDAY_ALLOWED_SUBSCRIBERS", "1")
    sent = []
    class R:
        def __init__(self, d): self.d = d
        def json(self): return self.d
    def post(url, json=None, **kw):
        sent.append(json["query"])
        if "subscribers" in json["query"]:
            return R({"data": {"boards": [{"subscribers": [{"id": 1}]}]}})
        return R({"data": {"x": {"id": "99"}}})
    a = command.run(conn, {"kind": "task", "text": "send the quote", "assignee": "Priya Shah", "client_id": "C-02", "due": "Friday"}, "garrett", post=post)
    assert a["monday_item_id"] == "99" and "people" not in " ".join(sent) and "notification" not in " ".join(sent)
    assert command.listing(conn, "garrett")[0]["text"] == "send the quote" and command.listing(conn, "other") == []
    monkeypatch.setenv("MONDAY_ALLOWED_SUBSCRIBERS", "2")  # someone else watches the board: write nothing
    assert command.run(conn, {"kind": "task", "text": "y"}, "g", post=post)["monday_item_id"] is None
    assert command.set_done(conn, a["id"], True, "garrett") and command.listing(conn, None)[-1]["done"]


def test_ticking_a_follow_up_keeps_approval_and_notes_it_on_monday(tmp_path, monkeypatch):
    from callsight import pipeline
    conn = store.connect(str(tmp_path / "t.db"))
    notes = []
    monkeypatch.setattr(sinks, "to_monday", lambda r: "77")
    monkeypatch.setattr(sinks, "to_ghl", lambda r: False)
    monkeypatch.setattr(sinks, "monday_update", lambda item, lines: notes.append((item, lines)) or True)
    r = pipeline.process("C1", "C-03", conn, text="AM: hi", write_out=False, redact_fn=lambda s: s, embed_fn=fake_embed,
                         analyze_fn=lambda s, c: insight(churn_risk=0.9, sentiment="angry"))
    r = pipeline.approve(conn, r, "priya")
    assert r.monday_item_id == "77" and not notes
    items = [a.model_copy(update={"done": True}) for a in r.insight.action_items]
    r = pipeline.edit(conn, r, "priya", action_items=items, embed_fn=fake_embed)
    assert r.approved_by == "priya" and notes[-1] == ("77", ["Done: call back (ticked by priya)"])
    r = pipeline.edit(conn, r, "priya", summary="changed", embed_fn=fake_embed)
    assert r.approved_by is None  # a real change needs approving again
    r = pipeline.approve(conn, r, "dana")
    assert notes[-1][0] == "77" and notes[-1][1][0] == "Approved again after changes by dana."  # same item, not a new one


def test_history_says_what_happened_and_members_see_only_their_part(tmp_path, monkeypatch):
    c = _app(tmp_path, monkeypatch)
    _login(c, "garrett", "q")
    r = c.post("/api/calls", data={"client_id": "C-03", "provider": "claude", "transcript": "Marcus (Acme): call me on [PHONE]"}).json()
    h = c.get(f"/api/calls/{r['id']}").json()["history"]
    kinds = [e["kind"] for e in reversed(h)]
    assert kinds[:3] == ["processed", "masked", "read"] and kinds[-1] == "waiting"
    assert "1 phone number" in h[-2]["text"] and "Person 1" in h[-2]["text"]  # what was masked, in words
    items = c.get(f"/api/calls/{r['id']}").json()["insight"]["action_items"]
    items[0]["done"] = True
    h = c.patch(f"/api/calls/{r['id']}", json={"action_items": items}).json()["history"]
    assert h[0]["kind"] == "ticked" and h[0]["text"].startswith("garrett marked done:")
    h = c.patch(f"/api/calls/{r['id']}", json={"churn_risk": 0.1}).json()["history"]
    assert h[0]["kind"] == "edited" and "risk of leaving" in h[0]["text"] and "to 10%" in h[0]["text"]
    h = c.post(f"/api/calls/{r['id']}/approve").json()["history"]
    assert h[0]["kind"] == "approved" and "Monday isn't connected" in h[0]["text"]
    _login(c, "admin", "p")
    mine = c.post("/api/calls", data={"client_id": "C-03", "provider": "claude", "transcript": "AM: hi"}).json()["id"]
    assert {e["kind"] for e in c.get("/api/activity").json()} >= {"login", "approved"}  # the admin sees everything
    _login(c, "garrett", "q")
    seen = c.get("/api/activity").json()
    assert all(e["call_id"] != mine for e in seen) and all(e["user"] == "garrett" for e in seen if e["kind"] == "login")
