# Pure Mycelia v0.1 — non-neural symbiotic rule learning

No neural networks, no pretrained models, no LLMs, no API calls. Agents are
rule-based learners that induce meanings and rules from demonstrations,
share verified discoveries on a message bus, check each other through a
critic, and keep provenance-tracked memory in SQLite.

Code: `mycelia/pure/` (worlds, `dsl`, `agents`, `bus`, `memory`,
`coordinator`, `rule_learner`), experiments `tools/pure_mycelia_experiments.py`,
tests `tests/test_pure.py`, results `results/pure/pure-results.json`.

> These are controlled symbolic worlds. Success here shows rule induction,
> composition and transfer in those worlds, not general intelligence or
> natural-language understanding.

## Worlds

- **Instruction world (dax/blick/wug):** each episode invents new words for 4
  primitives (each a two-action sequence), 2 modifiers and 2 connectives, with
  hidden operators drawn from a skewed distribution. One primitive is never
  shown alone. Test phrases are combinations never demonstrated.
- **Rule world:** a hidden f(a, b) (e.g. a+b, a·b, a+2b, (a+b)·2) seen through 5
  examples; tested on 20 unseen inputs.

Train episodes (seeds 0–199) are used only to learn search priors; all
results are on 200 test episodes (seeds 10000–10199) with fresh vocabularies.

## Agents

| Agent | How it learns |
|---|---|
| Observer | records demonstrations (episodic memory) |
| Lexicon | primitives from one-word demos; recovers never-shown-alone words by inverting operators the inducer verified |
| Rule inducer | searches an operator language (repeat k, reverse, mirror, concatenate, interleave …) in learned-prior order |
| Critic | re-checks every hypothesis against all demos it can evaluate; tolerates one outlier only with ≥2 agreeing demos and flags it as suspect |
| Executor | parses and runs new phrases from verified knowledge |

## Results (200 held-out test episodes)

### A. Symbiosis at equal compute (accuracy on unseen phrases)

| Evaluation budget | Network | One-pass (no feedback) | Single generalist | Vote of 3 | Memorizer |
|---|---|---|---|---|---|
| 100 | **94.6%** | 74.4% | 0% | 0% | 0% |
| 300 | **100%** | 74.4% | 0% | 0% | 0% |
| 3,000 | **100%** | 74.4% | 1.0% | 1.0% | 0% |
| 30,000 | **100%** | 74.4% | 34.0% | 26.0% | 0% |
| 100,000 | **100%** | 74.4% | 90.5% | 58.5% | 0% |

- Splitting the problem between agents that share verified discoveries
  needs ~90 evaluations per episode; one learner searching all word
  meanings jointly needs ~20,000–110,000.
- Feedback matters: when the lexicon cannot use the inducer's operators to
  recover the never-shown-alone word, accuracy is capped at 74.4%.
- Memorising demonstrations solves nothing: every test phrase is new.
- Voting among three generalists on a third of the budget each is worse than
  one generalist with the full budget.

### B. Procedural memory (search priors learned on train episodes)

| Budget | Without priors | With priors |
|---|---|---|
| 40 | 37.1% | **64.5%** |
| 60 | 78.9% | **81.8%** |
| 100 | 94.6% | **95.7%** |
| unlimited | 100% (90.3 evals) | 100% (84.6 evals) |

### C. Critic with one corrupted demonstration (3 demos per modifier)

| | Accuracy | Corrupted demo flagged |
|---|---|---|
| Critic on | **80.4%** | 125 / 200 |
| Critic off (strict consistency) | 52.7% | 0 |

A corrupted demo is unidentifiable when it is the only clean evidence for a
word (2 demos, one wrong); the critic correctly refuses to guess there.

### D. Retention

Rebuilt from verified memory alone (no demonstrations, 0 evaluations): **100%**
on all test episodes. 1,700 verified facts stored with source, episode and
evidence; refuted facts are superseded, never deleted.

### E. Sample efficiency (budget 3,000)

| Demos per modifier | Network | Generalist (3,000) | Generalist (300,000) |
|---|---|---|---|
| 1 | 22.9% | 1.2% | **99.0%** |
| 2 | **100%** | 1.0% | 100% |
| 3 | **100%** | 1.0% | 100% |

Limitation found: with one demo per modifier, the never-shown-alone word
appears only next to an unknown modifier. The lexicon needs the operator and
the inducer needs the word, so the network deadlocks. Joint search breaks this
tie at great cost. A future agent should propose joint hypotheses for exactly
these mutually dependent pairs.

### F. Rule world

| Budget | Without priors (exact rule) | With learned priors |
|---|---|---|
| 10 | 26% | **92%** |
| 50 | 61% | **100%** |
| 1,000+ | 93% (94 evals) | **100% (5 evals)** |

Without priors, shortest-first search sometimes finds a simpler rule that fits
the 5 examples but not the hidden one (7% of episodes). Priors learned from
verified rules on train episodes fix this. With one corrupted example: critic
on 100% accuracy (all 200 flagged), critic off 7.2%.

### G. Communication conventions

Verified concepts get short symbols (c0, c1, …): 400 symbols over 50
episodes, message bytes 253,891 → 195,972 (−23%), all references decoded.

## What this does and does not show

- Shows: rule induction from few examples, composition of unseen
  combinations, a measurable benefit of sharing verified discoveries over
  isolated, voting and joint-search baselines at equal compute, transfer of
  learned search strategies, error detection, and retention without search.
- Does not show: natural language, open-ended reasoning or coding. The
  operator language and agent roles are designed by hand; what is learned is
  word meanings, rules, search priors and conventions.
- Next steps: joint-hypothesis agent for mutual dependencies; let agents
  invent new operators by composing existing ones; larger grammars;
  program synthesis tasks; strategy changes proposed through the
  Symbiosis Lab gate.
