# Pure Mycelia v0.3 — multi-source, multi-domain, symbiotic continual learning

Question: does symbiotic interaction help a network discover and use
knowledge better than one learner, a shared library, or fixed sharing? No
neural networks, pretrained models or LLMs. Sources read live from GitHub
via Mycelia Reach (web: blocked by environment policy, untested; RSS:
untested). Task suites frozen and committed (commit 75f66cb, SHA-256
manifest) before the solver was written; development used only the dev set.

Code: `mycelia/learn/` (sources, evidence, memory, solver, network,
explorer), `tools/learn_experiments.py`, `tests/test_learn.py`. Results:
`results/learn/learn-results.json`. Reproduce:
`PYTHONPATH=.:tools python tools/learn_experiments.py`.

## Sources (live, CPython commit 0ec3aee2 + gto76/python-cheatsheet)

| Source | Format | Claims |
|---|---|---|
| Doc/builtins/stdtypes.rst | rst | 692 |
| Doc/builtins/functions.rst | rst | 206 |
| Doc/library/math.rst | rst | 129 |
| Doc/library/statistics.rst | rst | 135 |
| Lib/statistics.py | code docstrings | 130 |
| python-cheatsheet README.md | markdown | 38 |

68 skills verified in scope by sandbox experiments. Lifecycle, evidence
strength and source reliability are stored separately; docs and code of the
same project are classified `same_lineage` (not independent); identical text
counts as a copy; `sorted` has two independent lineages.

## Results (3 frozen eval sets, 900 tasks, 814 solvable, 86 unsolvable)

**1. Multi-source.** Adding the code docstrings and the markdown cheatsheet
changed nothing: 68 skills, 577/814 solved, 34 wrong, 86/86 unsolvable
correctly abstained, in all three source configurations. Those sources
repeat the docs rather than add operations.

**2. Cross-domain.** 182/351 cross-domain tasks solved (50/106 novel
combinations). Of the 182, 181 could not be solved from any single
document's skills (neither-alone check); 1 violation recorded.

**3. Network necessity** (3 random task-to-node assignments; solved of 814):

| Budget | A isolated | B shared repository | C fixed ring | D adaptive |
|---|---|---|---|---|
| 1,000 | 75–96 | **486–493** | 243–248 | 217–239 |
| 10,000 | 75–96 | **549–552** | 259–265 | 228–247 |

| | A | B | C | D |
|---|---|---|---|---|
| Wrong answers (budget 10k) | 2–8 | 29 | 12–19 | 9–13 |
| Fake skill in a node's library | 0 | 900 | 352–373 | **0** |
| Skills transferred | 0 | (central copy) | 19,325 | 8,358 |

Node failure (statistics node lost, 61 statistics tasks): B 37, C 0, D 6.

**The core hypothesis is not supported on accuracy.** A shared repository
solves about twice as many tasks as adaptive symbiosis. D's selective
requests (a peer's top-12 most relevant skills) often omit the second or
third skill a composite task needs. D does block the injected fake skill
completely (0 vs 900 in B), gives fewer wrong answers, and transfers fewer
skills than fixed sharing — safety and efficiency gains, not capability.

**4. Exploration** (start: stdtypes; 5 reads from 336 catalog files):
goal-driven, curiosity and random all stayed at 192 solved; alphabetical
reached 348 by luck (Doc/builtins/ sorts first). **Bug:** the explorer
mapped built-in functions to Doc/library/builtins.rst instead of
Doc/builtins/functions.rst, so the guided strategies never found the
builtins documentation. Fixed in `explorer.py`; the exploration experiment
has not been re-run yet, so no valid conclusion about exploration exists.

**5. Continual learning** (eval set 101; 11 sequential sources): solved
59 → 121 → 159 → 198 as text, builtins, math and statistics were read; no
change from the 7 later sources. 11 tasks solved earlier failed after the
statistics episode — all **retrieval failures** (skill still present, not
found within the budget because the larger skill set changed the search
order), none forgetting. Rebuilding from SQLite memory with no network
reproduced the live result exactly (198 = 198).

## Caveats

Hand-written task templates; Python string/number/statistics operations
only; 34 wrong answers (programs fitting 3 visible examples but not hidden
ones); `math.factorial` unlearnable because current docs moved it to
`math.integer`.
