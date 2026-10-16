# Phase 13: swarm coverage, shared trail vs coordinated frontier exploration (2026-10-16)

Script: `tools/swarm_coverage_benchmark.py` (Runpod cpu3g, seconds of compute). 32x32 maps, 20% obstacles, K robots start at one
entrance, goal = visit 95% of free cells (cap 8000 steps). 60 maps per cell, same maps and start for every method.
Metric: median steps to 95% coverage (all runs finished except random walk at K=1; random K=4 finished 97%/67% in two scenarios).

Methods: random walk; marks_private (avoid own visited cells); marks_shared (decaying marks left in the environment, step to
least-marked neighbour; local reads only, no radio); frontier (walk to the nearest unvisited, unclaimed cell using a global shared
map; needs full communication). A first version had all frontier robots move in lockstep with no comms (identical tie-breaking);
fixed with random BFS tie-breaking before the numbers below.

## Median steps to 95% coverage

| scenario | K | random | marks_private | marks_shared | frontier | marks_shared - frontier (95%) |
|---|---|---|---|---|---|---|
| normal | 1 | never | 1731 | 1929 | 1050 | +977 (+875, +1083) |
| normal | 4 | 3722 | 647 | 513 | 268 | +268 (+237, +301) |
| normal | 16 | 1312 | 237 | 189 | 84 | +110 (+101, +121) |
| normal | 64 | 615 | 134 | 99 | 49 | +54 (+47, +61) |
| half lost at step 100 | 4 | 6762 | 1039 | 884 | 436 | +474 (+420, +538) |
| half lost at step 100 | 16 | 2185 | 371 | 242 | 84 | +185 (+162, +213) |
| half lost at step 100 | 64 | 935 | 162 | 99 | 49 | +58 (+50, +67) |
| no global map (no comms) | 4 | 3722 | 647 | 513 | 513 | +18 (-13, +49) tie |
| no global map (no comms) | 16 | 1312 | 237 | 189 | 217 | -30 (-44, -15) marks faster |
| no global map (no comms) | 64 | 615 | 134 | 99 | 132 | -29 (-36, -22) marks faster |

## Reading
- With a shared global map, coordinated frontier exploration is about 1.9-2.2x faster than the trail. The trail never beats it there.
- The trail is 17-30% faster per step count than private marks and 4-6x faster than a random walk, and needs only local reads.
- When the global map is unavailable (no radio), the trail ties at K=4 and beats frontier by about 13-22% at K=16 and K=64.
- Single-point damage (half the robots lost) hurts both; frontier stays ahead. The trail's niche is communication-free coordination, not speed.
- Synthetic grid maps. Research simulation only.
