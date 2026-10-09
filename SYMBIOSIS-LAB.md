# Mycelia Symbiosis Lab v0.1

A bounded self-improvement loop. The lab finds a weakness, gathers evidence,
proposes candidate versions, tests them against the frozen production
version, rejects regressions, and verifies the winner. It **cannot** read
the holdout set, change its safety policy, or promote anything; a separate
gate scores the holdout once, and only a human approves a release.

v0.1 has no language model: specialists are rule-based behind a
`run(task) -> dict` interface so a model can replace any of them later.

## Two loops, kept apart

| | Intelligence/evolution lab (`mycelia/symbiosis_lab/`) | Safety and release gate (`governance/`) |
|---|---|---|
| Data | search + selection tiers only | holdout tier, once per candidate |
| Can | evaluate, diagnose, research, propose, test, verify, request release | check policy hash, STOP, frozen data hashes, audit chains; score holdout; record |
| Cannot | read holdout, edit policy, import governance, approve | promote without a human |
| Who changes it | the loop | humans only (`python -m governance.approve`) |

## Components

- **Frozen benchmark** (`benchmark.py`): splits once into search (inner
  Training), selection (20% of Training) and holdout (Testing); SHA-256 of
  each file in `manifest.json`; any change stops the lab and the gate.
- **Scorecard**: accuracy, balanced accuracy, per-class recall with Wilson
  95% intervals, selective accuracy at 90% coverage, tumor presence,
  robustness (accuracy after losing 30% of memories and healing), memory,
  fit time; 3 seeds.
- **Registry** (`registry.py`): append-only, hash-chained JSONL. Each
  candidate records parent, config, hypothesis, expected benefit, mutation,
  compute budget, test plan, rollback target and every status change.
- **Memory** (`memory.py`): episodic (cycles), semantic (evidence with source
  and content hash), procedural (mutation success rates that bias future
  proposals and decay over time), failure (rejected candidates and reasons).
- **Specialists** (`specialists.py`):
  - Evaluator runs scorecards in the sandbox.
  - Diagnostician flags a class whose recall interval sits below all others.
  - Researcher fetches allowlisted pages and reads local documents; content
    is untrusted, only searched and quoted.
  - Engineer generates candidates from a fixed mutation grammar.
  - Verifier re-tests the winner on fresh seeds.
- **Gateway** (`gateway.py`): `fetch_page` (https, allowlist, GET, 10 s,
  2 MB, rate limit, scripts stripped, logged), `read_document`,
  `query_approved_dataset`, `run_sandboxed_experiment` (child process,
  CPU/memory/wall limits, secrets and proxy variables removed),
  `submit_candidate`.
- **Gate** (`governance/gate.py`, `policy.toml`, `policy.sha256`): refuses to
  run if the policy hash differs, STOP exists, benchmark hashes changed or an
  audit chain is broken. Releases only verified candidates descended from
  production, scores holdout once against production, flags regressions,
  sets `awaiting_human`. `approve` and `rollback` are human CLI actions that
  need an interactive terminal and typed confirmation.

## Acceptance rule (predeclared in `policy.toml`)

A candidate passes only if weak-class recall rises by at least 1.0 point
(mean over 3 seeds), and accuracy, selective accuracy and tumor presence
drop by no more than 0.5 point, robustness by no more than 1.0 point, and
memory grows by no more than 1.5x. The verifier requires at least +0.5 point
on fresh seeds with the same tolerances. No score can buy back a failed
guard.

## Running

```bash
PYTHONPATH=.:tools python tools/symbiosis_lab_demo.py --dataset data/Brain-Tumor-MRI-Dataset
python -m governance.approve status
python -m governance.approve approve <candidate-id>     # human, interactive
python -m governance.approve rollback                    # human, interactive
touch governance/STOP                                    # kill switch
```

The demo needs the compartment cache from `tools/mycelial_network_mri.py`.
Changing `policy.toml` requires a human to update `policy.sha256` too.

## First demonstration (MRI benchmark)

**Cycle 001 — no action.** On the selection tier the production engine's
glioma recall was 93.5% (95% CI 90.1–95.9%), overlapping the other classes, so
the absolute weakness rule fired nothing and the loop correctly did nothing.
Glioma is weak only on Testing (81.5%), which the lab cannot see: the glioma
scans in Testing differ from those in Training.

**Rule change (by the developer, after cycle 001).** A relative rule was added:
the worst class's CI lies below the best class's and its recall is at least 3
points under the others. Cycle 001's record is kept unchanged.

**Cycle 002 — full loop.**
- Weakness: glioma (relative rule), recall 93.5%.
- Research: arxiv fetch refused by the container's network policy (recorded);
  evidence quoted from `RECEPTOR-MEMORY.md` and `COLLECTIVE-DISSENT.md`.
- Six candidates tested on selection (3 seeds):

| Mutation | Glioma recall gain | Decision |
|---|---|---|
| weak_class_growth+0.45 | +1.9 pts | **accepted** |
| weak_class_growth+0.25 | +0.8 | rejected: gain < 1.0 |
| softer_recall | +0.4 | rejected: gain < 1.0 |
| larger_budget | 0.0 | rejected: gain < 1.0 |
| extra_epoch | −0.4 | rejected: gain < 1.0 |
| sharper_recall | −1.1 | rejected: gain, and selective accuracy −0.7 |

- Verification on fresh seeds: glioma +1.1 pts (93.9% → 95.0%), accuracy
  96.05% → 95.96% (within tolerance): replicated.
- Gate, holdout scored once (1,467 Testing scans):

| | Production | Candidate |
|---|---|---|
| Accuracy | 93.66% | 93.52% |
| Glioma recall | 81.5% | **83.4%** |
| Selective accuracy (90%) | 97.2% | 97.4% |
| Tumor presence | 97.41% | 97.41% |
| Memories | 1,805 | 2,470 |

  No guarded regression beyond tolerance; status `awaiting_human`.
  Production is still `baseline`. The glioma gain on holdout (+1.9 pts) is
  about 7 of 379 scans, within noise; a human should weigh that and the 37%
  larger memory before approving.

## Known limits

- Network: in the development container every outside host was blocked by
  the environment's network policy, so the research step's fetch is refused
  and recorded; local project documents are used as evidence instead.
- Candidates are configurations of existing learners, not code changes.
- The sandbox is a resource-limited child process, not a container or VM.
- Tests: `tests/test_symbiosis_lab.py`, `tests/test_governance.py`.
