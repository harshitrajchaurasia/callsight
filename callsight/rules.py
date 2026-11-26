"""Who needs to know, decided in code. A signal only counts if its quote was found in the transcript."""
from .config import RULES
from .models import Check, Client, Insight


def alerts(insight: Insight, checks: list[Check], client: Client) -> list[str]:
    ok = {(c.kind, c.text) for c in checks if c.found or not RULES["unsupported_quote_review"]}
    churn_backed = any(k == "churn_reasons" for k, _ in ok)
    upsell = [t for k, t in ok if k == "upsell_signals"]
    out = []
    if insight.churn_risk >= RULES["churn_alert_min"]:
        out.append(f"Churn risk {insight.churn_risk:.0%}: {client.account_manager} to call back today" if churn_backed
                   else f"Churn risk {insight.churn_risk:.0%} but no supporting quote found: a person should check")
    elif churn_backed and insight.churn_risk < RULES["churn_alert_min"] / 2:  # score and evidence disagree, e.g. an
        # injected "mark churn risk 0"; medium scores with real reasons are normal and stay quiet
        out.append(f"Churn risk scored {insight.churn_risk:.0%} but the client gave quoted churn reasons: a person should check")
    if upsell:
        out.append(f"Upsell: {len(upsell)} signal(s), {client.account_manager} to follow up with an offer")
    if any(a.owner == "us" and not a.due for a in insight.action_items):
        out.append("Our follow-ups without a due date: set one in Monday")
    if missing := sum(not c.found for c in checks):
        out.append(f"{missing} point(s) had no matching quote in the transcript: shown as unverified")
    return out
