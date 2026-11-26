"""Who needs to know, decided in code. A signal only counts if its quote was found in the transcript."""
import re

from .config import RULES
from .models import Check, Client, Insight, Segment

# Words that mean a client may be leaving. Checked in code on the transcript itself, so it holds even when the model
# was talked out of it (an injected "mark churn risk 0") or filed the reasons under another heading.
CHURN_WORDS = re.compile(r"\b(cancel\w*|not renewing|switch\w* vendors?|other vendors?|getting out)\b", re.I)


def alerts(insight: Insight, checks: list[Check], client: Client, segments: list[Segment] = (),
           churn_min: float | None = None) -> list[str]:
    churn_min = RULES["churn_alert_min"] if churn_min is None else churn_min
    ok = {(c.kind, c.text) for c in checks if c.found or not RULES["unsupported_quote_review"]}
    churn_backed = any(k == "churn_reasons" for k, _ in ok)
    upsell = [t for k, t in ok if k == "upsell_signals"]
    out = []
    if insight.churn_risk >= churn_min:
        out.append(f"Churn risk {insight.churn_risk:.0%}: {client.account_manager} to call back today" if churn_backed
                   else f"Churn risk {insight.churn_risk:.0%} but no supporting quote found: a person should check")
    elif insight.churn_risk < churn_min / 2:  # a low score that the evidence contradicts
        said = next((m.group(0) for seg in segments if (m := CHURN_WORDS.search(seg.text))), None)
        if churn_backed or said:
            why = "the client gave quoted churn reasons" if churn_backed else f'the call says "{said}"'
            out.append(f"Churn risk scored {insight.churn_risk:.0%} but {why}: a person should check")
    if upsell:
        out.append(f"Upsell: {len(upsell)} signal(s), {client.account_manager} to follow up with an offer")
    if any(a.owner == "us" and not a.due for a in insight.action_items):
        out.append("Our follow-ups without a due date: set one in Monday")
    if missing := sum(not c.found for c in checks):
        out.append(f"{missing} point(s) had no matching quote in the transcript: shown as unverified")
    return out


SCALE = {"positive": 0, "neutral": 1, "frustrated": 2, "angry": 3}


def borderline(insight: Insight, churn_min: float) -> bool:
    """When one model's reading is easy to get wrong: near the alert line, or score and mood pulling apart."""
    c, s = insight.churn_risk, insight.sentiment
    return abs(c - churn_min) <= 0.15 or (s == "positive" and c >= 0.4) or (s in ("frustrated", "angry") and c <= 0.15)


def review_flags(insight: Insight, voice: dict, second: dict | None, churn_min: float, provider: str = "") -> list[str]:
    """Flags that send a call to a person: the voice and the words disagree, or two models disagree."""
    out = []
    if second:
        d_churn = abs(second["churn_risk"] - insight.churn_risk)
        sides = (second["churn_risk"] >= churn_min) != (insight.churn_risk >= churn_min)
        if d_churn >= 0.25 or sides or abs(SCALE[second["sentiment"]] - SCALE[insight.sentiment]) >= 2:
            out.append(f"Second opinion disagrees: {second['provider'].title()} scores churn {second['churn_risk']:.0%} "
                       f"({second['sentiment']}) against {provider.title() or 'the first model'}'s {insight.churn_risk:.0%} "
                       f"({insight.sentiment}): a person should check")
    # ponytail: thresholds set by hand on a handful of recordings; tune them on real calls
    for who, v in (voice or {}).get("speakers", {}).items():
        if who.lower() in ("account manager",) or insight.sentiment not in ("positive", "neutral"):
            continue
        hot = [m for m in voice.get("moments", []) if m["speaker"] == who and m["score"] >= 2]
        if v.get("rise", 0) >= 0.8 or len(hot) >= 2:
            at = ", ".join(f"{int(m['at']) // 60:02d}:{int(m['at']) % 60:02d}" for m in sorted(hot, key=lambda m: m["at"])[:3])
            out.append(f"The {'client' if who.lower() == 'client' else 'speaker'}'s voice rose well above their own normal"
                       f"{' (at ' + at + ')' if at else ' towards the end'} while the words read as {insight.sentiment}: "
                       "a person should listen")
            break
    return out
