# Phase 10, Task 1 — Network design (5 layouts, 6 terminals each)

Means over 5 random layouts (1,200 growth steps; budgets conserved to ~1e-9):

| Method | Backbone length | Efficiency | Random 10% cut survival | Targeted cut survival | Terminals reached |
|---|---:|---:|---:|---:|---:|
| MYCELIA (grown blind) | 124.9 | 0.750 | **0.680** | 0.167 | 5.8 / 6 |
| MST (given locations) | 89.9 | 0.817 | 0.587 | 0.427 | 6 |
| Complete graph | 414.3 | 1.000 | 1.000 | 1.000 | 6 |

**Versus MST:** efficiency within 10% (0.750 vs 0.817, −8%) ✓; random-damage survival **better** (+16%) ✓; backbone cost 39% higher ✗; targeted-attack survival much worse ✗.
**Gate (within 10% of best baseline on 2 of 3 axes):** passes against MST on efficiency + random robustness; fails against the complete graph. Partial pass.

**Interpretation:** MYCELIA trades backbone cost for resilience to random damage, a property of real fungal networks, and it reached the food without being told where it was. Its main weakness is concentrated load on a few central cords (targeted attack). Total grown length (~1,390) includes blind exploration; MST and complete graph are handed terminal locations, so the next fair test is a blind-search baseline.
