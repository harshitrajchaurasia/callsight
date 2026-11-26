import time
from fastapi.testclient import TestClient

import api
from callsight import ask, extract, redact, store
from tests.test_core import fake_embed, insight


def test_api_login_validation_and_a_call(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "conn", store.connect(str(tmp_path / "t.db")))
    monkeypatch.setattr(redact, "redact", lambda segs, **kw: segs)
    monkeypatch.setattr(ask, "embed", fake_embed)
    monkeypatch.setitem(extract.PROVIDERS, "claude", lambda segs, client: insight())
    monkeypatch.setenv("DEMO_USER", "u"); monkeypatch.setenv("DEMO_PASS", "p")
    c = TestClient(api.app, base_url="https://testserver")
    assert c.get("/api/calls").status_code == 401
    assert c.post("/api/login", json={"username": "u", "password": "no"}).status_code == 401
    assert c.post("/api/login", json={"username": "u", "password": "p"}).status_code == 200
    assert c.post("/api/calls", data={"client_id": "NOPE", "transcript": "a: b"}).status_code == 422
    assert c.post("/api/calls", data={"client_id": "C-03", "transcript": "a: b", "keep": "NAME,PHONE"}).status_code == 422
    assert c.post("/api/calls", data={"client_id": "C-03"}).status_code == 422  # neither transcript nor audio
    assert c.post("/api/calls", data={"client_id": "C-03"}, files={"audio": ("x.exe", b"MZ")}).status_code == 415
    r = c.post("/api/calls", data={"client_id": "C-03", "provider": "claude",
                                   "transcript": "AM: we're looking at other vendors\nAM: I'll call you back by Friday"})
    assert r.status_code == 200 and r.json()["checks"][0]["found"]
    assert [x["id"] for x in c.get("/api/calls").json()] == [r.json()["id"]]
    monkeypatch.setattr(redact, "redact", lambda segs, **kw: (_ for _ in ()).throw(redact.RedactionUnavailable("down")))
    assert c.post("/api/calls", data={"client_id": "C-03", "transcript": "a: b", "provider": "claude"}).status_code == 503
    assert len(c.get("/api/samples").json()) == 23
    assert c.post("/api/logout").status_code == 200 and c.get("/api/calls").status_code == 401


def test_login_lockout(monkeypatch):
    monkeypatch.setenv("DEMO_USER", "u"); monkeypatch.setenv("DEMO_PASS", "p")
    api._fails.clear()
    c = TestClient(api.app, base_url="https://testserver")
    codes = [c.post("/api/login", json={"username": "u", "password": "x"}, headers={"x-forwarded-for": "9.9.9.9"}).status_code
             for _ in range(6)]
    assert codes == [401] * 5 + [429]
    api._fails.clear()


def test_public_mode_rate_limits_spending(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "conn", store.connect(str(tmp_path / "t.db")))
    monkeypatch.setattr(redact, "redact", lambda segs, **kw: segs)
    monkeypatch.setattr(ask, "embed", fake_embed)
    monkeypatch.setitem(extract.PROVIDERS, "claude", lambda segs, client: insight())
    monkeypatch.setenv("DEMO_PASS", ""); monkeypatch.delenv("EXTRA_LOGINS", raising=False)
    monkeypatch.setitem(api.LIMITS, "call", (2, 3600))
    api._hits.clear()
    c = TestClient(api.app, base_url="https://testserver")
    assert c.get("/api/me").json() == {"login_required": False, "signed_in": True}
    form = {"client_id": "C-03", "provider": "claude", "transcript": "AM: hi"}
    codes = [c.post("/api/calls", data=form, headers={"x-forwarded-for": "1.1.1.1"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    assert c.post("/api/calls", data=form, headers={"x-forwarded-for": "2.2.2.2"}).status_code == 200  # per IP
    api._hits.clear()


def test_extra_logins(monkeypatch):
    monkeypatch.setenv("DEMO_USER", "u"); monkeypatch.setenv("DEMO_PASS", "p"); monkeypatch.setenv("EXTRA_LOGINS", "g:q, h2:r ,h:,:x")
    api._fails.clear()
    assert api._login_ok("u", "p") and api._login_ok("g", "q")
    assert not api._login_ok("g", "p") and not api._login_ok("h", "") and not api._login_ok("u", "q")
    assert api._login_ok("h2", "r") and not api._login_ok("", "x")
    monkeypatch.setenv("DEMO_PASS", "")
    assert not api._authed(None)  # EXTRA_LOGINS alone still means a login is required


def test_session_cookie_names_the_user(monkeypatch, caplog):
    monkeypatch.setenv("DEMO_USER", "u"); monkeypatch.setenv("DEMO_PASS", "p"); monkeypatch.setenv("EXTRA_LOGINS", "garrett:q")
    api._fails.clear()
    c = TestClient(api.app, base_url="https://testserver")
    with caplog.at_level("INFO", logger="uvicorn.error"):
        assert c.post("/api/login", json={"username": "garrett", "password": "q"}).status_code == 200
        assert c.get("/api/me").json()["signed_in"]
    assert "login ok user=garrett" in caplog.text and "visit user=garrett" in caplog.text
    tok = c.cookies.get(api.COOKIE)
    assert api._user(tok) == "garrett"
    assert api._user(tok.replace(".garrett.", ".u.")) is None  # can't swap the name without the key
    assert api._user(api._token(int(time.time()) + 60, "nobody")) is None  # signed, but not a current login
    monkeypatch.setenv("EXTRA_LOGINS", "")
    assert api._user(tok) is None  # removing a login signs it out
