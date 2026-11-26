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


def _mmss(at: float | None) -> str:
    return "" if at is None else f" @{int(at) // 60:02d}:{int(at) % 60:02d}"


def report(r: CallResult) -> list[str]:
    """The summary, alerts and checked points as plain lines, shared by Monday and GHL."""
    i = r.insight
    lines = [i.summary, f"Sentiment {i.sentiment}, churn risk {i.churn_risk:.0%}", *r.alerts]
    lines += [f"{'' if c.found else '[unverified] '}{c.kind.replace('_', ' ')}: {c.text}{_mmss(c.at)}" for c in r.checks]
    return lines


def to_monday(r: CallResult, *, post=requests.post) -> str | None:
    token, board = os.getenv("MONDAY_API_TOKEN"), os.getenv("MONDAY_CALLS_BOARD_ID")
    if not (token and board):
        return None

    def gql(kind, **variables):
        resp = post(MONDAY, timeout=15, headers={"Authorization": token, "API-Version": "2024-10"},
                    json={"query": MUTATIONS[kind], "variables": variables}).json()
        return str(next(iter(resp["data"].values()))["id"])

    # Creating items can email a board's subscribers. Write only if every subscriber is on the allowlist.
    allowed = set(filter(None, os.getenv("MONDAY_ALLOWED_SUBSCRIBERS", "").split(",")))
    try:
        resp = post(MONDAY, timeout=15, headers={"Authorization": token, "API-Version": "2024-10"},
                    json={"query": "query ($b: [ID!]) { boards(ids: $b) { subscribers { id } } }", "variables": {"b": [board]}}).json()
        subs = {str(u["id"]) for u in resp["data"]["boards"][0]["subscribers"]}
    except Exception as e:
        log.warning("monday subscriber check failed, not writing %s: %r", r.id, e)
        return None
    if not allowed or subs - allowed:
        log.warning("monday board %s has subscribers outside MONDAY_ALLOWED_SUBSCRIBERS; not writing %s", board, r.id)
        return None
    i = r.insight
    try:
        item = gql("item", board=board, name=f"{CLIENTS[r.client_id].name}: {r.id} (churn {i.churn_risk:.0%}, {i.sentiment})"[:250])
    except Exception as e:
        log.warning("monday item for %s failed: %r", r.id, e)
        return None
    try:  # the item exists now: a failed subitem/update is logged, not retried into a duplicate item
        for a in i.action_items:
            gql("subitem", parent=item, name=f"[{a.owner}] {a.task}" + (f" (due {a.due})" if a.due else ""))
        gql("update", item=item, body="".join(f"<p>{escape(x)}</p>" for x in report(r)))
    except Exception as e:
        log.warning("monday item %s incomplete: %r", item, e)
    return item


def to_ghl(r: CallResult, *, post=requests.post) -> bool:
    """A note on the client's GHL contact. GHL_CONTACTS maps our client ids to contact ids, e.g. {"C-03": "abc"}."""
    contact = json.loads(os.getenv("GHL_CONTACTS") or "{}").get(r.client_id)
    if not (os.getenv("GHL_API_KEY") and contact):
        return False
    try:
        post(f"{GHL}/contacts/{contact}/notes", timeout=15, json={"body": f"Call {r.id}\n" + "\n".join(report(r))},
             headers={"Authorization": f"Bearer {os.getenv('GHL_API_KEY')}", "Version": os.getenv("GHL_API_VERSION", "v3")}
             ).raise_for_status()
        return True
    except Exception as e:
        log.warning("ghl note for %s failed: %r", r.id, e)
        return False
