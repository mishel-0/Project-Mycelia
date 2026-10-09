"""Export a compact inference snapshot; the original MYCELIA model is retained."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mycelia.frozen_visual import FrozenVisualMatcher
from mycelia.visual_memory import VisualMycelium


def compile_model(memory_path, output):
    memory_path, output = Path(memory_path), Path(output)
    started = time.perf_counter()
    digest = hashlib.sha256(memory_path.read_bytes()).hexdigest()
    model = VisualMycelium.load(memory_path)
    loaded = time.perf_counter()
    snapshot = FrozenVisualMatcher.from_model(model, digest)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save(output)
    result = {'source': str(memory_path), 'source_sha256': digest,
              'source_bytes': memory_path.stat().st_size, 'output': str(output),
              'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
              'output_bytes': output.stat().st_size, 'numeric_array_bytes': snapshot.array_bytes,
              'colony_count': snapshot.colony_count,
              'load_seconds': loaded-started, 'compile_and_save_seconds': time.perf_counter()-loaded,
              'scope': 'Inference optimization only; full checkpoint retained for graph state and learning.'}
    output.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compile_model(args.memory, args.output), indent=2))


if __name__ == '__main__':
    main()
