"""Write-outs: a Monday item per call (action items as subitems, summary + alerts as an update) and a GHL contact note.

Both are off unless their env vars are set, and both only create records: no notifications, @mentions or people
columns, so nobody watching the board is pinged by this code. Failures are logged and never lose the call (it's
already saved).
"""
import json
import logging
import os
from html import escape

import requests

from .config import CLIENTS
from .models import CallResult

log = logging.getLogger(__name__)
MONDAY = "https://api.monday.com/v2"
GHL = "https://services.leadconnectorhq.com"
MUTATIONS = {
    "item": "mutation ($board: ID!, $name: String!) { create_item (board_id: $board, item_name: $name) { id } }",
    "subitem": "mutation ($parent: ID!, $name: String!) { create_subitem (parent_item_id: $parent, item_name: $name) { id } }",
    "update": "mutation ($item: ID!, $body: String!) { create_update (item_id: $item, body: $body) { id } }",
}


def report(r: CallResult) -> list[str]:
    """The summary, alerts and checked points as plain lines, shared by Monday and GHL."""
    timed = any(s.end > s.start for s in r.segments)  # audio has times; a pasted transcript only has line numbers

    def _mmss(at: float | None) -> str:
        return "" if at is None else f" @{int(at) // 60:02d}:{int(at) % 60:02d}" if timed else f" (line {int(at) + 1})"
    i = r.insight
    lines = [i.summary, f"Sentiment {i.sentiment}, churn risk {i.churn_risk:.0%}", *r.alerts]
    lines += [f"{'' if c.found else '[unverified] '}{c.kind.replace('_', ' ')}: {c.text}{_mmss(c.at)}" for c in r.checks]
    return lines


def _monday(what: str, post):
    """A gql(kind, **variables) for the configured board, or None if Monday is off or the board isn't safe to write.
    Creating items can email a board's subscribers, so write only if every subscriber is on the allowlist."""
    token = os.getenv("MONDAY_API_TOKEN")
    board = os.getenv("MONDAY_TASKS_BOARD_ID" if what == "task" else "MONDAY_CALLS_BOARD_ID") or os.getenv("MONDAY_CALLS_BOARD_ID")
    if not (token and board):
        return None
    headers = {"Authorization": token, "API-Version": "2024-10"}
    allowed = set(filter(None, os.getenv("MONDAY_ALLOWED_SUBSCRIBERS", "").split(",")))
    try:
        resp = post(MONDAY, timeout=15, headers=headers,
                    json={"query": "query ($b: [ID!]) { boards(ids: $b) { subscribers { id } } }", "variables": {"b": [board]}}).json()
        subs = {str(u["id"]) for u in resp["data"]["boards"][0]["subscribers"]}
    except Exception as e:
        log.warning("monday subscriber check failed, not writing %s: %r", what, e)
        return None
    if not allowed or subs - allowed:
        log.warning("monday board %s has subscribers outside MONDAY_ALLOWED_SUBSCRIBERS; not writing %s", board, what)
        return None

    def gql(kind, **variables):
        resp = post(MONDAY, timeout=15, headers=headers, json={"query": MUTATIONS[kind], "variables": variables | (
            {"board": board} if kind == "item" else {})}).json()
        return str(next(iter(resp["data"].values()))["id"])
    return gql


def monday_task(name: str, body: str, *, post=requests.post) -> str | None:
    """One item (a task from a voice command) on the tasks board; the person goes in the name, never a people column."""
    if not (gql := _monday("task", post)):
        return None
    try:
        item = gql("item", name=name[:250])
        gql("update", item=item, body=f"<p>{escape(body)}</p>")
        return item
    except Exception as e:
        log.warning("monday task failed: %r", e)
        return None


def monday_update(item: str, lines: list[str], *, post=requests.post) -> bool:
    """An update (a comment) on an existing item: a follow-up ticked off, or the call re-approved after changes."""
    if not (gql := _monday(f"update on {item}", post)):
        return False
    try:
        gql("update", item=item, body="".join(f"<p>{escape(x)}</p>" for x in lines))
        return True
    except Exception as e:
        log.warning("monday update on %s failed: %r", item, e)
        return False


def board_url() -> str | None:
    """Link to the board for people (MONDAY_BOARD_URL, e.g. https://acct.monday.com/boards/123), if Monday is on."""
    return os.getenv("MONDAY_BOARD_URL") if os.getenv("MONDAY_API_TOKEN") and os.getenv("MONDAY_CALLS_BOARD_ID") else None


def to_monday(r: CallResult, *, post=requests.post) -> str | None:
    if not (gql := _monday(r.id, post)):
        return None
    i = r.insight
    try:
        item = gql("item", name=f"{CLIENTS[r.client_id].name}: {r.id} (churn {i.churn_risk:.0%}, {i.sentiment})"[:250])
    except Exception as e:
        log.warning("monday item for %s failed: %r", r.id, e)
        return None
    try:  # the item exists now: a failed subitem/update is logged, not retried into a duplicate item
        for a in i.action_items:
            who = r.speakers.get(a.speaker or "", a.speaker) or a.owner
            gql("subitem", parent=item, name=f"[{who}] {a.task}" + (f" (due {a.due})" if a.due else ""))
        gql("update", item=item, body="".join(f"<p>{escape(x)}</p>" for x in report(r)))
    except Exception as e:
        log.warning("monday item %s incomplete: %r", item, e)
    return item


def to_ghl(r: CallResult, *, post=requests.post) -> bool:
    return ghl_note(r.client_id, f"Call {r.id}\n" + "\n".join(report(r)), post=post)


def ghl_note(client_id: str, body: str, *, post=requests.post) -> bool:
    """A note on the client's GHL contact. GHL_CONTACTS maps our client ids to contact ids, e.g. {"C-03": "abc"}."""
    contact = json.loads(os.getenv("GHL_CONTACTS") or "{}").get(client_id)
    if not (os.getenv("GHL_API_KEY") and contact):
        return False
    try:
        post(f"{GHL}/contacts/{contact}/notes", timeout=15, json={"body": body},
             headers={"Authorization": f"Bearer {os.getenv('GHL_API_KEY')}", "Version": os.getenv("GHL_API_VERSION", "v3")}
             ).raise_for_status()
        return True
    except Exception as e:
        log.warning("ghl note for %s failed: %r", client_id, e)
        return False
