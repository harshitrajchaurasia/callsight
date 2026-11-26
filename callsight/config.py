import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .models import Client

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
RULES = json.loads((ROOT / "config" / "rules.json").read_text())
CLIENTS = {c["id"]: Client(**c) for c in json.loads((ROOT / "data" / "clients.json").read_text())}
DB_PATH = os.getenv("CALLSIGHT_DB", str(ROOT / "data" / "callsight.db"))

if not 0 <= RULES["churn_alert_min"] <= 1:
    raise ValueError("config/rules.json: churn_alert_min must be between 0 and 1")
