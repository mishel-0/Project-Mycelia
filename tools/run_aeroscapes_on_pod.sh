#!/usr/bin/env bash
# Pod launcher: download AeroScapes (CC-BY-SA-4.0) on the pod, run the benchmark, print results to the log.
set -euo pipefail
cd /workspace 2>/dev/null || cd "$HOME"
[ -d Project-Mycelia ] || git clone --depth 1 https://github.com/mishel-0/Project-Mycelia.git
cd Project-Mycelia
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
pip install -q numpy scikit-learn pillow gdown
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
DATA=/tmp/aeroscapes_dl; mkdir -p "$DATA"
gdown --fuzzy "https://drive.google.com/file/d/1WmXcm0IamIA0QPpyxRfWKnicxZByA60v/view" -O "$DATA/aeroscapes_archive" || { echo "DOWNLOAD FAILED (Google Drive blocked or quota)"; exit 2; }
SIZE=$(stat -c %s "$DATA/aeroscapes_archive"); echo "downloaded bytes: $SIZE"
[ "$SIZE" -lt 8000000000 ] || { echo "archive larger than 8 GB, aborting"; exit 3; }
mkdir -p "$DATA/x"; (cd "$DATA/x" && (unzip -q ../aeroscapes_archive || tar xf ../aeroscapes_archive))
ROOT=$(dirname "$(find "$DATA/x" -type d -name JPEGImages | head -1)"); echo "dataset root: $ROOT"; ls "$ROOT"
PYTHONPATH=.:tools python -u tools/aeroscapes_benchmark.py --root "$ROOT" --output runs/aeroscapes
echo "=== BENCHMARK FINISHED ==="
