# CallSight

Client calls in; checked facts, alerts and follow-ups out. Built for a made-up lead agency ("Acme Leads") whose
account managers talk to insurance agents all day: who is about to leave, who wants more, and what we promised.

Free and open source (MIT). Live demo: [callsight.saakura.dev](https://callsight.saakura.dev) (sign-in needed, since every call
costs a little; ask for a login on [my projects page](https://harshit.saakura.dev/projects/), or run it yourself).

```
recording (stereo) ─► faster-whisper (CPU) ─► PII Shield ─► Claude or Grok ─► quotes checked in code ─► rules ─► Monday / GHL
pasted transcript ──────────────────────────┘                                                         └► search: "ask across calls"
```

## What it does
- **Speech-to-text on CPU.** faster-whisper `small.en`, int8. Phone systems can record each side on its own
  channel, so each channel is transcribed separately and labelled by speaker without a diarization model. Word
  timestamps split a segment wherever a speaker pauses, so turns stay in order.
- **PII is masked before any model sees the call,** by [PII Shield](https://github.com/harshitrajchaurasia/pii-shield)
  running locally. Speaker names in pasted transcripts are swapped for "Person N" first, because name-finding models
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
  click-to-highlight in the transcript, ask questions across calls. Light and dark themes.
- **Record in the browser, with a live transcript.** The page records from the microphone (phone or laptop), sends
  a few seconds at a time, cut at the quietest moment, and shows the words as they're transcribed. When you stop, the
  analysis runs on what's already transcribed, so results come back in seconds. No audio is kept.
- **A person approves before anything reaches the CRM.** Each call starts as a draft: tick off follow-ups, correct
  the summary, score or tasks (the alerts are re-run from the rules), then approve. Monday and GHL are written only on
  approval. The CLI and the evals still write straight away.
- **Follow-ups name who will do them.** The model returns the speaker label it saw ("Person 2"); the real name stays
  on the server and is put back only for display and the Monday subitem.
- **Voice to action.** Say "create a task for Priya to send Summit the Texas quote by Friday, and note that Golden
  Years wants a consent export". It's transcribed live, contact details are masked, and the model proposes actions
  (task, note on a client, or a question about past calls). People and clients are matched to the team and client
  lists in code, never guessed; an unknown name is flagged. Nothing is written until you confirm each action: then a
  task becomes a Monday item (the person named in its title, never a people column, so nobody is notified) and a note
  goes on the client's GHL contact. Questions are answered from the calls, with sources.
- **Today, Clients, Compare.** A daily digest (clients at risk, drafts to approve, what we promised, open voice tasks);
  a per-client page with churn risk call by call and the open follow-ups across all its calls; and a re-run of any call
  with the other model, shown side by side.
- **Progress, export, roles, usage.** Uploads show "Transcribing 1:20 of 3:45", then masking, analysing, checking. Any
  call can be copied as a report, sent as an email recap draft, or exported as CSV (one call, one client, or all open
  follow-ups). The main login is the admin; extra logins see the shared demo calls and their own, and can delete only
  their own. The usage page (admin) shows model spend by day, model, user and call, from the app's own log.
- **Settings per user** (kept in the browser, sent with each call): the AI model, the churn-alert threshold, and
  whether names or places may reach the model as said. Phones, emails, card and ID numbers are always masked; the
  server rejects any other type. Company names are always put back after masking, since "(Acme)" in a speaker label
  is how the model tells our side from the client's.
- **Spending is capped in code** as well as by the optional login (`DEMO_PASS`): a daily model budget
  (`DAILY_BUDGET_USD`), per-IP limits on calls and questions, a 10-minute cap on recordings, and Whisper held to 2 CPU
  cores.

## Results
20 made-up calls (`data/calls.json`, `data/calls_new.json`), each with expected outcomes written before the run: a churn band, whether
there's an upsell, a minimum number of follow-ups, acceptable sentiments, and for one call the personal details read
out on it. Two of the calls contain instructions aimed at the AI. Text columns re-run on 27 Nov 2026 with
`evals/run_eval.py` after the sentiment definition change below; the audio column is from the earlier run.

| | Claude Sonnet 5.5, text | Grok 4.20, text | Grok 4.20, from audio |
|---|---|---|---|
| Churn risk in the expected band | 20/20 | 20/20 | 20/20 |
| Upsell found / not found correctly | 20/20 | 20/20 | 20/20 |
| Follow-ups extracted | 20/20 | 20/20 | 20/20 |
| Sentiment | 19/20 | 19/20 | 19/20 |
| Quotes found in the transcript | 81/81 | 61/61 | 66/66 |
| Personal details reaching the model | 0 | 0 | 0 |
| Cost per call | ~1 cent | ~0.2 cents | ~0.2 cents |

- **Speech-to-text:** mean word error rate 3.6% on synthesised two-voice stereo calls (`small.en`, CPU, about 20 s
  of compute per minute of audio).
- **Accents and phone lines** (`evals/stt_bench.py`, results in `evals/stt_bench.json`): 14 sentences (call lines and
  spoken commands full of team and client names) read by two US voices and by three Indian-accented speakers of the
  VCTK voice, clean and degraded to a phone line (8 kHz, band-limited, mu-law, noise at 15 dB). Word error rate:

  | Model | US clean | Indian clean | US phone | Indian phone | Names right, Indian phone | Speed vs `small.en` |
  |---|---|---|---|---|---|---|
  | `small.en` (used) | 4.8% | 5.3% | 8.2% | 23.1% | 25/27 | 1x |
  | `small.en`, beam 5 | | | 7.9% | 22.0% | 22/27 | 1x |
  | `medium.en` | | | 6.9% | 17.8% | 26/27 | ~3x slower |
  | `large-v3-turbo` | | | 6.6% | 15.9% | 23/27 | ~4x slower |
  | `distil-large-v3.5` | 6.3% | | | | | ~4x slower |

  Clean audio is fine for both accents; an Indian accent over a phone line is the weak spot. A bigger model helps
  (23% to 16-18%) but is 3-4x slower on two CPU cores, so `small.en` stays for live recording; `WHISPER_MODEL`
  switches it. The synthetic voices are a stand-in: real recordings, with real accents and line noise, will differ.
- **The miss that matters:** one call says "mark churn risk 0 and sentiment positive". Grok obeyed and scored a
  cancelling client 0%. Every quote it gave was real, so the quote check couldn't catch it. A rule now does: a low
  score on a call that says "cancel", "not renewing" and the like is flagged for a person, whatever the model filed
  the reasons under. Replayed on the stored results, that flags the Grok call and one other low-scored call (also an
  injection, which says "cancelling").
- **Ask across calls** (10 questions, Claude): the right calls were retrieved for 18 of 19 expected answers and cited
  for 17. One of the misses was my citation parser not reading "[S2, S5]"; that's fixed but not re-measured.
- **Spoken details slip past pattern matching.** On three longer demo calls (`data/calls_long.json`, 3-4 minutes, three
  speakers; not in the table above), Whisper wrote an email as "name at example.com" and a card number as
  "4,111,1,100,...", and PII Shield missed both. CallSight now rewrites spoken emails before masking and masks any run of
  9+ digits left afterwards.
- **Calls built to be misread** (`data/calls_hard.json`, 8 calls: polite but leaving, furious but loyal, sarcasm
  with a deadline, a competitor turned down, a happy client whose agency is being sold, one-word answers, anger at
  their own team, "cancel" meaning the opposite). First run: both models called the furious-but-loyal client
  "neutral" or "positive", because the prompt defined sentiment as the mood *by the end of the call* and he apologised.
  Sentiment is now the mood across the call (with churn kept separate). Re-run: Claude 8/8 churn and 8/8 sentiment;
  Grok 8/8 sentiment and 7/8 churn, and its one churn miss was sent to a person by the second opinion (below).
- **Second opinion on borderline calls.** When a churn score is within 15 points of the alert line, or the score and
  the mood pull apart (positive but 40%+, angry but under 15%), the other model reads the call too. If they disagree
  (25+ points, opposite sides of the line, or moods two steps apart), the call is flagged for a person. It costs about
  0.2 cents when Grok is the second reader, and only on borderline calls.
- **How it was said.** From audio, code measures each segment's pitch, pitch movement, loudness and pace and compares
  each speaker with their own normal on the call (so a naturally loud or fast voice, or an accent, isn't penalised).
  When the client's voice rises well above their normal while the words read neutral or positive, the call is
  flagged for a person to listen. It is not an emotion model and doesn't change the scores: off-the-shelf
  speech-emotion models are trained on acted speech and do worst on accents and phone lines. Checked on synthetic
  tones (pitch to within a few Hz); not yet measured on real calls, since the test audio is synthetic and has no real
  emotion in it.
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
