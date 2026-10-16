"""First Symbiosis Lab demonstration on the brain-tumor MRI benchmark.

Freezes the benchmark (once), bootstraps production from the documented
Symbiosis Engine setting if none exists (operator action, recorded), and runs
one evolution cycle. The cycle ends at `awaiting_human` at most; promotion
requires `python -m governance.approve approve <id>` in a human terminal.
"""
from __future__ import annotations
import argparse, getpass, json
from pathlib import Path
import numpy as np
from governance.gate import Gate
from governance.approve import DEFAULT_BASELINE
from mycelia.symbiosis_lab.benchmark import prepare
from mycelia.symbiosis_lab.loop import run_cycle
from train_mycelium import LABELS, load_images, clean_splits, write_json
from symbiosis_mri import compress


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--cache', type=Path, default=Path('results/mycelial-network/cache'))
    ap.add_argument('--lab', type=Path, default=Path('lab_state'))
    ap.add_argument('--output', type=Path, default=Path('results/symbiosis-lab'))
    a = ap.parse_args(); bench = a.lab / 'benchmark'; a.output.mkdir(parents=True, exist_ok=True)
    if not (bench / 'manifest.json').exists():
        records, _ = load_images(a.dataset, 16); clean, _ = clean_splits(records)
        y = np.array([LABELS.index(r['label']) for r in records])
        print('freezing benchmark', prepare(compress(a.cache, 16), y, clean['Training'], clean['Testing'], bench)['counts'], flush=True)
    gate = Gate(bench, a.lab / 'registry', state_root=a.lab / 'governance')
    if gate.production() is None:
        gate.bootstrap(DEFAULT_BASELINE, getpass.getuser() + ' (demo operator)')
    docs = [p for p in (Path('RECEPTOR-MEMORY.md'), Path('SYMBIOSIS-ENGINE.md'), Path('COLLECTIVE-DISSENT.md')) if p.exists()]
    record = run_cycle(gate, a.lab, bench.resolve(), documents=docs)
    n = len(list(a.output.glob('cycle-*.json'))) + 1
    write_json(a.output / f'cycle-{n:03d}.json', record)
    print('cycle written:', a.output / f'cycle-{n:03d}.json')
    print('production unchanged:', gate.production()['id'])


if __name__ == '__main__':
    main()
