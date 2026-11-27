"""What happened to a call, in sentences people read in the app (the History tab and the Activity page).
The server log keeps the terse version; these say what each step did and what it means."""
import re
from collections import Counter

from .models import CallResult

PROVIDER = {"claude": "Claude", "grok": "Grok"}
KIND = {"NAME": "name", "PHONE": "phone number", "EMAIL": "email", "LOCATION": "place", "NUMBER": "long number",
        "CREDIT_CARD": "card number", "SSN": "ID number", "ADDRESS": "address", "DATE_OF_BIRTH": "date of birth"}


def _who(user: str | None) -> str:
    return user or "someone"


def _n(k: int, word: str) -> str:
    return f"{k} {word}{'' if k == 1 else 's'}"


def processed(r: CallResult, user: str | None, source: str) -> list[tuple[str, str]]:
    """One line per pipeline step, in order."""
    i, out = r.insight, []
    out.append(("processed", f"{_who(user)} sent in {source}."))
    masked = Counter(m for s in r.segments for m in re.findall(r"\[([A-Z_]+)\]", s.text))
    kept = " and ".join({"NAME": "names", "LOCATION": "places"}[k] for k in r.unmasked if k in ("NAME", "LOCATION"))
    if masked:
        parts = ", ".join(_n(v, KIND.get(k, k.lower().replace("_", " "))) for k, v in masked.most_common())
        text = f"PII Shield masked {_n(sum(masked.values()), 'personal detail')} ({parts}) before any AI saw the call."
    else:
        text = "PII Shield checked the call for personal details before any AI saw it and found none to mask."
    if r.speakers:
        text += f" Speaker names were swapped for Person 1 to {len(r.speakers)}; the real names stay on this server."
    if kept:
        text += f" The {kept} were sent as said (a setting)."
    out.append(("masked", text))
    out.append(("read", f"{PROVIDER.get(r.provider, r.provider)} read the masked call: risk of leaving "
                        f"{round(i.churn_risk * 100)}%, client mood {i.sentiment}, {_n(len(i.action_items), 'follow-up')}."))
    if s := r.second_opinion:
        out.append(("second", f"That reading was borderline, so {PROVIDER.get(s['provider'], s['provider'])} read the call "
                              f"too: risk of leaving {round(s['churn_risk'] * 100)}%, client mood {s['sentiment']}."))
    found = sum(c.found for c in r.checks)
    miss = len(r.checks) - found
    out.append(("checked", f"Code found {found} of {_n(len(r.checks), 'quote')} in the transcript."
                           + (f" {miss} couldn't be found, so those points are marked unverified and can't raise alerts." if miss else "")))
    out.append(("alerts", f"The rules raised {_n(len(r.alerts), 'alert')}: " + "; ".join(r.alerts[:3]) + "." if r.alerts
                else "The rules raised no alerts."))
    out.append(("waiting", f"Finished in {r.latency_ms / 1000:.0f} s. Waiting for review: nothing has been sent to Monday yet."))
    return out


def edited(before: CallResult, after: CallResult, user: str | None) -> list[tuple[str, str]]:
    out = []
    old = {a.task: a for a in before.insight.action_items}
    noted = " It's noted on the Monday item." if after.approved_by is not None and after.monday_item_id else ""
    for a in after.insight.action_items:
        if a.task in old and old[a.task].done != a.done:
            out.append(("ticked", f"{_who(user)} {'marked done' if a.done else 'reopened'}: {a.task}.{noted}"))
    b, a = before.insight, after.insight
    parts = []
    if a.summary != b.summary:
        parts.append("the summary")
    if a.churn_risk != b.churn_risk:
        parts.append(f"risk of leaving {round(b.churn_risk * 100)}% to {round(a.churn_risk * 100)}%")
    if a.sentiment != b.sentiment:
        parts.append(f"client mood {b.sentiment} to {a.sentiment}")
    content = lambda i: [(x.owner, x.speaker, x.task, x.due) for x in i.action_items]
    if content(a) != content(b):
        parts.append(f"the follow-ups ({len(b.action_items)} before, {len(a.action_items)} now)")
    if parts:
        again = " It had been approved, so it's waiting for review again." if before.approved_by is not None and after.approved_by is None else ""
        out.append(("edited", f"{_who(user)} changed {', '.join(parts)}.{again}"))
    return out


def approved(r: CallResult, user: str | None, had_item: str | None, connected: bool) -> str:
    n = len(r.insight.action_items)
    if had_item:
        return f"{_who(user)} approved it again. The corrected report was added to Monday item #{had_item} as an update."
    if r.monday_item_id:
        return f"{_who(user)} approved it. Sent to Monday as item #{r.monday_item_id}, with {_n(n, 'follow-up')} as subitems."
    if connected:
        return f"{_who(user)} approved it, but the write to Monday didn't go through. Edit and approve again to retry."
    return f"{_who(user)} approved it. Monday isn't connected on this server, so nothing was sent."


if __name__ == "__main__":  # python -m callsight.activity: a quick look at the wording
    from .models import Insight, Segment
    i = Insight.model_validate({"summary": "s", "sentiment": "frustrated", "churn_risk": 0.65, "churn_reasons": [],
                                    "upsell_signals": [], "objections": [], "action_items": []})
    r = CallResult(id="CALL-1", client_id="C-01", segments=[Segment(start=0, end=1, speaker="A", text="call [PHONE] or [NAME]")],
                   insight=i, checks=[], alerts=[], provider="claude", latency_ms=12000)
    for k, t in processed(r, "ana", "a pasted transcript"):
        print(k, "|", t)
