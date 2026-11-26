from typing import Literal

from pydantic import BaseModel, Field


class Client(BaseModel):
    id: str
    name: str
    vertical: str
    states: list[str]
    plan: str
    account_manager: str


class Segment(BaseModel):
    start: float  # seconds from the start of the call
    end: float
    speaker: str
    text: str


QUOTE = "Exact words copied from the transcript (5-25 words) that support this. Copy, don't paraphrase."


class Signal(BaseModel):
    text: str = Field(description="The point in a few words")
    quote: str = Field(description=QUOTE)


class ActionItem(BaseModel):
    owner: Literal["us", "client"] = Field(description="us = Acme Leads staff, client = the agent")
    task: str
    due: str | None = Field(description="Due date or timeframe if one was said, else null")
    quote: str = Field(description=QUOTE)


class Insight(BaseModel):
    """What the model returns. Every claim carries a quote so code can check it against the transcript."""
    summary: str = Field(description="2-3 neutral sentences; people by role, not name")
    sentiment: Literal["positive", "neutral", "frustrated", "angry"]
    churn_risk: float = Field(ge=0, le=1, description="0-1 likelihood the client cancels soon")
    churn_reasons: list[Signal] = Field(description="Why churn_risk is above 0.2; empty if it isn't")
    upsell_signals: list[Signal] = Field(description="Concrete signs they would buy more (states, volume, products); empty if none")
    objections: list[Signal] = Field(description="Complaints or concerns they raised; empty if none")
    action_items: list[ActionItem]


class Check(BaseModel):
    """A quote check: was the quote found in the transcript, and at what time."""
    kind: str
    text: str
    quote: str
    found: bool
    at: float | None = None


class CallResult(BaseModel):
    id: str
    client_id: str
    segments: list[Segment]  # PII-masked
    insight: Insight
    checks: list[Check]
    alerts: list[str]  # decided in code
    provider: str
    latency_ms: int
    monday_item_id: str | None = None
