#!/usr/bin/env bash
# Runs the pod-scale swarm-vision benchmark from a fresh CPU pod, prints results to the log.
set -euo pipefail
cd /workspace 2>/dev/null || cd "$HOME"
[ -d Project-Mycelia ] || git clone --depth 1 https://github.com/mishel-0/Project-Mycelia.git
cd Project-Mycelia
pip install -q numpy scikit-learn pillow
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:tools python -u tools/swarm_vision_pod.py --workers "$(nproc)" --output runs/swarm-vision-pod
echo "=== BENCHMARK FINISHED ==="
