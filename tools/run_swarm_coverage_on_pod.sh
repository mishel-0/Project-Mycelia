#!/usr/bin/env bash
# Pod launcher for the swarm coverage benchmark (pure NumPy, no dataset download).
set -euo pipefail
cd /workspace 2>/dev/null || cd "$HOME"
[ -d Project-Mycelia ] || git clone --depth 1 https://github.com/mishel-0/Project-Mycelia.git
cd Project-Mycelia
pip install -q numpy
python -u tools/swarm_coverage_benchmark.py --workers "$(nproc)" --maps 60 --ks 1 4 16 64 --output runs/swarm-coverage
echo "=== BENCHMARK FINISHED ==="
