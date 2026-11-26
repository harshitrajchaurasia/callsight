import os

import callsight.config  # noqa: F401  (loads .env first, so the strip below wins)

for k in [k for k in os.environ if k.startswith(("MONDAY_", "GHL_", "ANTHROPIC_", "XAI_"))]:
    del os.environ[k]  # tests are offline: no real Monday items, GHL notes or model calls
