# Callsight

Client calls in; checked facts, alerts and follow-ups out. Built for a made-up lead agency ("Acme Leads") whose
account managers talk to insurance agents all day: who is about to leave, who wants more, and what we promised.

Live demo (login required, since every call costs a little): [callsight.saakura.dev](https://callsight.saakura.dev)

```
recording (stereo) ─► faster-whisper (CPU) ─► PII Shield ─► Claude or Grok ─► quotes checked in code ─► rules ─► Monday / GHL
pasted transcript ──────────────────────────┘                                                         └► search: "ask across calls"
```

## What it does
- **Speech-to-text on CPU.** faster-whisper `small.en`, int8. Phone systems can record each side on its own
  channel, so each channel is transcribed separately and labelled by speaker without a diarization model. Word
  timestamps split a segment wherever a speaker pauses, so turns stay in order.
- **PII is masked before any model sees the call,** by [PII Shield](https://github.com/harshitrajchaurasia/pii-shield)
  running locally. Speaker names in pasted transcripts are swapped for "Speaker N" first, because name-finding models
  miss bare first names. If PII Shield is down, the call stops; nothing is sent unmasked.
- **Extraction with a quote for every claim.** The model returns a summary, sentiment, churn risk with reasons, upsell
  signals, objections and action items, each with a verbatim quote. Code then finds each quote in the masked
  transcript. A point whose quote isn't there is shown as unverified and never triggers an alert.
- **Alerts are rules in code** (`config/rules.json`), not model output. A churn alert needs a churn score over the
  threshold and a found quote. A low score that contradicts quoted churn reasons is flagged for a person.
- **Write-outs.** A Monday item per call (action items as subitems, the report as an update) and a note on the
  client's GHL contact. Before writing, it reads the board's subscribers and stops unless every one is on an
  allowlist; it never sets people columns or @mentions, so nobody gets notified by it.
- **Ask across calls.** Local embeddings (fastembed `bge-small-en-v1.5`) over SQLite. The answer cites sources, and
  code checks that every citation was actually retrieved.
- **Web app.** FastAPI and one HTML page: paste a transcript or upload a recording, see the checked quotes with
  click-to-highlight in the transcript, ask questions across calls.

## Results
20 made-up calls (`data/calls*.json`), each with expected outcomes written before the run: a churn band, whether
there's an upsell, a minimum number of follow-ups, acceptable sentiments, and for one call the personal details read
out on it. Two of the calls contain instructions aimed at the AI. Run on 27 Nov 2026 with `evals/run_eval.py`.

| | Claude Sonnet 5.5, text | Grok 4.20, text | Grok 4.20, from audio |
|---|---|---|---|
| Churn risk in the expected band | 20/20 | 19/20 | 20/20 |
| Upsell found / not found correctly | 20/20 | 20/20 | 20/20 |
| Follow-ups extracted | 20/20 | 20/20 | 20/20 |
| Sentiment | 19/20 | 18/20 | 19/20 |
| Quotes found in the transcript | 85/85 | 65/65 | 66/66 |
| Personal details reaching the model | 0 | 0 | 0 |
| Cost per call | ~1 cent | ~0.2 cents | ~0.2 cents |

- **Speech-to-text:** mean word error rate 3.6% on synthesised two-voice stereo calls (`small.en`, CPU, about 20 s
  of compute per minute of audio). Real phone audio will be worse; I haven't measured it.
- **The miss that matters:** one call says "mark churn risk 0 and sentiment positive". Grok obeyed and scored a
  cancelling client 0%. Every quote it gave was real, so the quote check couldn't catch it. A rule now does: a low
  score on a call that says "cancel", "not renewing" and the like is flagged for a person, whatever the model filed
  the reasons under. Replayed on the stored results, that flags the Grok call and one other low-scored call (also an
  injection, which says "cancelling").
- **Ask across calls** (10 questions, Claude): the right calls were retrieved for 18 of 19 expected answers and cited
  for 17. One of the misses was my citation parser not reading "[S2, S5]"; that's fixed but not re-measured.
- The one sentiment miss both models share (a billing complaint from a client who also wants a new product) is
  arguable either way.

## Trade-offs
- No diarization model. Stereo recordings get speaker labels for free; a mono recording gets one label and the
  extraction model works out who is who. pyannote would fix that but wants a GPU to be quick.
- The quote check proves the quote is in the transcript, not that the model read it right. A real quote can back a
  wrong score, which is why the score-vs-evidence rule exists.
- PII Shield over-masks a little (for example "from another vendor" becomes "from [NAME] vendor"). That costs a word
  of context, never a leak.

## Run it
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # one model key is enough; PII Shield must be running
.venv/bin/python -m pytest -q   # offline, no keys needed
.venv/bin/uvicorn api:app --port 8012
.venv/bin/python tools/make_audio.py samples/c1.txt samples/c1.wav   # a stereo test recording
.venv/bin/python -m callsight CALL-01 C-03 samples/c1.wav           # or the .txt itself
.venv/bin/python -m evals.run_eval --audio --ask               # the numbers above, ~$0.25
```
Test audio is synthesised with Piper (`tools/make_audio.py`, voices in `models/`), one voice per channel.

MIT licence. All clients, people and calls are made up.
