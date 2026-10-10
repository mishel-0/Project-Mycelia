# Pure Mycelia v0.2 — learning skills from online documentation

Milestone: Mycelia learns a skill from online documentation, tests it
itself, shares it with other agents, and uses it to solve a new problem —
with no neural networks, pretrained models or LLMs.

**Status: run live.** docs.python.org is blocked in the development
container, so the documentation was read from its source in the official
CPython repository on GitHub (`Doc/builtins/stdtypes.rst`, commit
`0ec3aee2`) through **Mycelia Reach**, a channel layer modelled on
[Agent-Reach](https://github.com/Panniantong/Agent-Reach). Same text that
docs.python.org renders; nothing simulated.

## Mycelia Reach (`mycelia/reach/channels.py`)

Agent-Reach's design, adapted: each channel says which sources it handles,
checks whether a backend really works, and reads through ordered fallbacks;
every read is logged (hash-chained) with a SHA-256 of the content.

| Channel | Backends | Status here |
|---|---|---|
| GitHub (`github://owner/repo/path`) | anonymous blob-less `git` read | works |
| Web (`https://`, allowlist) | direct GET → Jina Reader (Agent-Reach's web backend) | blocked by network policy |
| RSS (`rss+https://`) | feedparser | needs `pip install feedparser` |

Not adopted from Agent-Reach: browser-cookie extraction and logged-in
platforms (personal-account access), and Exa search (an external AI
service; Pure Mycelia uses no outside AI).

## Live results (15 held-out test tasks, 13 solvable)

Documentation: 46 `str` methods and 133 examples extracted (doctests plus
inline "``expr`` returns ``value``" sentences). **34 skills verified** by
Mycelia's own experiments; the injected altered page's `title` example was
**contested** by experiment and the invented `str.unscramble` never ran.

| Condition | Solved correctly | Wrong | Correct abstentions (of 2) |
|---|---|---|---|
| Blind enumeration (same verified skills, no doc words) | 7 / 13 | 0 | 2 |
| Documentation relevance | 9 / 13 | 0 | 2 |
| Documentation + priors learned on train tasks | **11 / 13** | 0 | 2 |
| Unverified docs trusted (incl. altered page) | 8 / 13 | 0 | 2 |

Solved only with documentation: "make every letter lowercase" (`x.lower()`;
blind search found two rival programs that fit and abstained), "capitalize
the first letter and lowercase the rest after trimming"
(`x.strip(' ').capitalize()`), "case insensitive form … without surrounding
spaces" (`x.casefold().strip(' ')`), "turn the comma list into uppercase words
separated by spaces" (`x.replace(',', ' ').upper()`).

On the 7 tasks every condition solved, total attempts: blind 2,514,
documentation 1,779 (−29%), with priors 1,816, unverified 1,955. Counts
include a 200-attempt look-ahead used to detect rival programs.

Still unsolved: "make each word start with a capital letter and trim the
ends" (needs `strip` + `title`; nothing links "trim" to `strip`, so the pair
ranks beyond the budget) and "check whether the path ends with a txt
extension" (needs the constant `.txt`, which the solver cannot invent).
Both abstained — no wrong answers.

Development notes (honest record): the first live run solved 2/13 because
of solver bugs (one-step programs starved by deeper search, zero-argument
methods tried with constants, task punctuation crowded out), and the task
"remove duplicate words" was mislabelled because its inputs had no
duplicates; it was regenerated with duplicates. All fixes are in the
history; numbers above are from the final run.

## Pipeline

| Step | Code | What happens |
|---|---|---|
| Fetch | `mycelia/symbiosis_lab/gateway.py` (`fetch_html`) | allowlisted https GET of `docs.python.org/3/library/stdtypes.html`, timeout, 2 MB cap, logged; content marked untrusted |
| Extract | `mycelia/pure/extract.py` | deterministic parsing of each `str.<method>` entry: signature, parameters (required/optional), description, doctest examples, source position |
| Discover | `skills.Discoverer` | candidate skill per entry; description words minus *learned* stop-words (words in >25% of entries) |
| Experiment | `skills.Experimenter` + `sandbox.py` | re-runs every documented example and probes edge inputs (empty, spaces, case, punctuation, non-ASCII) in a restricted evaluator; a skill is **verified** only by its own experiments; contradicted examples make it **contested**; anything outside the pure-method allowlist is **not_allowed** |
| Share | `bus.Bus` | verified skills are announced on the bus; the solver only receives skills announced as verified |
| Solve | `solver.Solver` | best-first search over compositions (≤3 steps) of verified skills, ordered by overlap between the task's English goal and each skill's documentation words, plus priors from earlier solved tasks; must reproduce every visible example; otherwise answers "not enough evidence" |

### Sandbox

No `eval`/`exec`. Expressions are parsed and walked by hand: literals, the
variable `x`, literal-int subscripts, and calls to allowlisted pure `str`
methods only, with integer and result-size caps. Tests confirm it blocks
`__import__`, dunder attributes, lambdas, comprehensions, `open`,
`str.format` and huge allocations. Downloaded code is never executed directly.

## Experiment (`tools/pure_web_skills.py`)

Frozen task set `datasets/pure_tasks.json` (generated by
`tools/make_pure_tasks.py`): 30 tasks with an English goal, 3 visible and 6
hidden examples; even ids train (priors only), odd ids test. 4 tasks cannot
be solved with `str` methods (reverse, sort, initials, dedupe): the correct
answer there is to abstain.

Conditions on the 15 test tasks, same search budget:
1. **Blind enumeration** — same verified skills, no documentation words.
2. **Documentation relevance** — skill order guided by the goal's words.
3. **Relevance + priors** learned from solved train tasks.
4. **Unverified docs trusted** — every documented entry used without
   experiments, including an injected altered page (a wrong `title` example
   and an invented `str.unscramble`).

Reported: correctly solved (hidden examples), wrong answers, correct
abstentions, attempts per solved task.

## Running

```bash
# allow docs.python.org in the environment's Network access settings first
python tools/make_pure_tasks.py          # already committed; regenerates identically
PYTHONPATH=.:tools python tools/pure_web_skills.py
```

If the domain is blocked the tool stops with the gateway error and writes
`results/pure-web/preflight.json`.

## Tests

`tests/test_pure_web.py` uses a small hand-written page in the same HTML
structure (`tests/fixtures/sphinx_str_methods.html`) and an altered
untrusted page: extraction, sandbox escapes, verification by experiment,
refutation of the altered example, rejection of the invented method,
solving a held-out task, and abstaining on reversal.
