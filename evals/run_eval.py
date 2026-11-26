"""Eval: extraction (both models), quote checks, PII leaks, injection, speech-to-text WER, ask-across-calls.

python -m evals.run_eval [--providers claude,grok] [--audio] [--ask]     (~$0.35 for everything)
Uses the real PII Shield on :18081 and a throwaway DB, never the app's. Writes evals/results.json.
"""
import argparse
import json
import os
import re
import tempfile
from pathlib import Path

os.environ["USAGE_TAG"] = "eval"
from callsight import ask, pipeline, store, stt  # noqa: E402  (this loads .env, so strip after it)

for k in [k for k in os.environ if k.startswith(("MONDAY_", "GHL_"))]:
    del os.environ[k]  # the eval never writes to Monday or GHL

ROOT = Path(__file__).resolve().parent.parent
CALLS = json.loads((ROOT / "data/calls.json").read_text()) + json.loads((ROOT / "data/calls_new.json").read_text())
CHURN = {"low": (0, 0.3), "medium": (0.3, 0.7), "high": (0.6, 1)}


def score(call: dict, r) -> dict:
    e, i = call["expect"], r.insight
    lo, hi = CHURN[e["churn"]]
    masked = " ".join(s.text for s in r.segments) + " " + i.model_dump_json()
    return {"churn": lo <= i.churn_risk <= hi, "upsell": any(c.found for c in r.checks if c.kind == "upsell_signals") == e["upsell"],
            "actions": len(i.action_items) >= e["min_action_items"], "sentiment": i.sentiment in e["sentiment"],
            "quotes_found": sum(c.found for c in r.checks), "quotes": len(r.checks),
            "review": any("person should" in a for a in r.alerts), "churn_risk": i.churn_risk, "sentiment_was": i.sentiment,
            "second": r.second_opinion,
            "pii_leaks": [p for p in call.get("pii", []) if p.lower() in masked.lower()]}


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    out = {k: f"{sum(r[k] for r in rows)}/{n}" for k in ("churn", "upsell", "actions", "sentiment")}
    out["quotes_found"] = f"{sum(r['quotes_found'] for r in rows)}/{sum(r['quotes'] for r in rows)}"
    out["pii_leaks"] = sum(len(r["pii_leaks"]) for r in rows)
    out["flagged_for_review"] = [r["id"] for r in rows if r["review"]]
    out["misses"] = [f"{r['id']}:{k}" for r in rows for k in ("churn", "upsell", "actions", "sentiment") if not r[k]]
    return out


def words(t: str) -> list[str]:
    t = re.sub(r"^[^:\n]{1,40}:", "", t, flags=re.M).replace("%", " percent").lower()
    return re.findall(r"[a-z0-9']+", t)


def wer(ref: list[str], hyp: list[str]) -> float:
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        cur = [i]
        for j, b in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1] / max(len(ref), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", default="claude,grok")
    ap.add_argument("--calls", help="data files to use instead, e.g. calls_hard.json (the cases built to be misread)")
    ap.add_argument("--audio", action="store_true", help="synthesise each call, transcribe it, WER + grok on the audio")
    ap.add_argument("--ask", action="store_true", help="ask-across-calls questions (needs the first provider's run)")
    a = ap.parse_args()
    global CALLS
    if a.calls:
        CALLS = [c for f in a.calls.split(",") for c in json.loads((ROOT / "data" / f).read_text())]
    tmp = tempfile.mkdtemp()
    results: dict = {}
    conns = {}
    for prov in a.providers.split(","):
        conns[prov] = conn = store.connect(f"{tmp}/{prov}.db")
        rows = []
        for c in CALLS:
            r = pipeline.process(c["id"], c["client_id"], conn, text=c["transcript"], provider=prov)
            rows.append({"id": c["id"], **score(c, r)})
            print(prov, c["id"], rows[-1])
        results[prov] = {"summary": summarise(rows), "rows": rows}
    if a.audio:
        conn, rows, wers = store.connect(f"{tmp}/audio.db"), [], []
        for c in CALLS:
            txt, wav = f"{tmp}/{c['id']}.txt", f"{tmp}/{c['id']}.wav"
            Path(txt).write_text(c["transcript"])
            os.system(f"{ROOT}/.venv/bin/python {ROOT}/tools/make_audio.py {txt} {wav}")
            segs = stt.transcribe(wav)
            wers.append(round(wer(words(c["transcript"]), words(" ".join(s.text for s in segs))), 3))
            r = pipeline.process(c["id"], c["client_id"], conn, audio_path=wav, provider="grok")
            rows.append({"id": c["id"], "wer": wers[-1], **score(c, r)})
            print("audio", c["id"], rows[-1])
        results["audio_grok"] = {"summary": summarise(rows) | {"mean_wer": round(sum(wers) / len(wers), 3),
                                                              "model": os.getenv("WHISPER_MODEL", "small.en")}, "rows": rows}
    if a.ask:
        prov = a.providers.split(",")[0]
        qs, rows = json.loads((ROOT / "evals/questions.json").read_text()), []
        for q in qs:
            hits = {h["call_id"] for h in ask.search(conns[prov], q["q"])}
            out = ask.answer(conns[prov], q["q"], provider=prov)
            cited = {s["call_id"] for s in out["sources"]}
            ok = set(q["calls"]) | set(q["also_ok"])
            rows.append({"q": q["q"], "retrieved": f"{len(hits & set(q['calls']))}/{len(q['calls'])}",
                         "cited": f"{len(cited & set(q['calls']))}/{len(q['calls'])}", "wrong_cited": sorted(cited - ok),
                         "bad_refs": out["unknown_citations"], "answer": out["answer"]})
            print("ask", rows[-1])
        need = sum(len(q["calls"]) for q in qs)
        results["ask"] = {"summary": {"provider": prov,
                                      "retrieval_recall": f"{sum(int(r['retrieved'].split('/')[0]) for r in rows)}/{need}",
                                      "citation_recall": f"{sum(int(r['cited'].split('/')[0]) for r in rows)}/{need}",
                                      "wrong_citations": sum(len(r["wrong_cited"]) for r in rows)}, "rows": rows}
    out = "evals/results.json" if not a.calls else f"evals/results_{a.calls.split('.')[0].replace('calls_', '')}.json"
    (ROOT / out).write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v["summary"] for k, v in results.items()}, indent=1))


if __name__ == "__main__":
    main()
