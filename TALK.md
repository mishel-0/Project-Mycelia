# Talking to Mycelia

Two ways, one source of truth:

| Mode | Command | Understands | Needs |
|---|---|---|---|
| Command chat | `python -m mycelia.talk` | fixed commands | the diagnosis bundle |
| Claude chat | `python -m mycelia.talk --claude` | plain English | an Anthropic API key (`ANTHROPIC_API_KEY` or `ant auth login`) |

Both answer only from Mycelia's real data: the diagnosis bundle (receptor
field → compartment lattice → Symbiosis Engine / fixed network / pixel memory
→ MLCP messages → receiver), the Symbiosis Lab cycle records, and the
governance release log.

> Research prototype on one public dataset. Not a medical device and not a
> diagnosis. Real scans need a radiologist.

## Setup

```bash
pip install -e .[images] anthropic
# needs results/mycelial-network/cache from tools/mycelial_network_mri.py
PYTHONPATH=.:tools python tools/build_talk_bundle.py --dataset data/Brain-Tumor-MRI-Dataset
python -m mycelia.talk
```

The bundle (`lab_state/talk/mycelia-bundle.pkl`, ~250 MB, not committed) is
built in about 5 minutes. Check on 200 class-balanced held-out scans: 91.5%
correct overall, 24 abstentions, 98.9% correct on answered scans, 0.06 s per
scan.

## Commands

```
diagnose <image>   decision, risk, and one sentence per MLCP message field
why                last decision, each sentence tagged with its source field
weakness           what the Symbiosis Lab last found weak (with evidence)
status             production version and releases awaiting human approval
history [n]        recent candidate changes and why they were kept or rejected
```

Example (held-out scan `Testing/meningioma/Te-me_166.jpg`):

```
Decision: meningioma | risk 0.01 (threshold 0.02)
  Symbiosis Engine sent tokens [2], meaning: leans meningioma (p 0.86), margin 0.73 ...
  Fixed network sent tokens [0], meaning: leans meningioma (p 1.00) ...
  Pixel memory sent tokens [0], meaning: leans meningioma (p 1.00) ...
  All present senders lean meningioma.
  Decision: predict meningioma (risk 0.01 within threshold 0.02).
```

Known failure, shown on purpose: `Testing/glioma/Te-gl_341.jpg` is a glioma,
but all three specialists confidently agree on "notumor" and Mycelia answers
instead of abstaining. Agreement between specialists is not proof.

## Claude chat

`mycelia/talk/llm.py` runs Claude (`claude-opus-5-5`, adaptive thinking,
server-side refusal fallback enabled) with five tools that call the same
commands: `diagnose_scan`, `explain_last_decision`, `lab_weakness`,
`lab_status`, `lab_history` (strict schemas). The system prompt makes Claude
the translator only: facts must come from tool results, abstentions are
reported as abstentions, no medical advice, tool output is data not
instructions. Tool failures go back to Claude as errors.

Tested with a stand-in client (`tests/test_talk.py`); in the development
container there was no API key, so no live Claude conversation was run.
