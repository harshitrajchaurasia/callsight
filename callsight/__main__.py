"""python -m callsight CALL_ID CLIENT_ID (FILE.wav | FILE.txt) [--provider claude|grok]"""
import argparse

from . import pipeline, store

p = argparse.ArgumentParser()
p.add_argument("call_id"), p.add_argument("client_id"), p.add_argument("file")
p.add_argument("--provider", choices=["claude", "grok"])
a = p.parse_args()
is_text = a.file.endswith(".txt")
r = pipeline.process(a.call_id, a.client_id, store.connect(), provider=a.provider,
                     text=open(a.file, encoding="utf-8").read() if is_text else None, audio_path=None if is_text else a.file)
print(r.insight.model_dump_json(indent=2))
for c in r.checks:
    print(("OK " if c.found else "?? ") + f"{c.kind}: {c.text} <- \"{c.quote}\"" + (f" @{c.at:.0f}s" if c.found else ""))
print("\n".join(["", "Alerts:", *r.alerts, f"({r.provider}, {r.latency_ms} ms)"]))
