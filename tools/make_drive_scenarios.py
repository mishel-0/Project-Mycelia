"""Freeze the driving evaluation suites (before any learner tuning).

Practice scenarios use seeds 0-9999 of each car's own family. Evaluation uses
seeds 100000+ : in-family (unseen variations) and held-out families. A
manifest stores SHA-256 hashes so the suites cannot be silently changed."""
import hashlib, json
from dataclasses import asdict
from pathlib import Path
from mycelia.autonomy.scenarios import make, TRAIN_FAMILIES, HELD_OUT

suites = {}
for s in (1, 2, 3):
    sc = [make(f, 100000 * s + i) for f in list(TRAIN_FAMILIES.values()) + HELD_OUT for i in range(50)]
    suites[f'drive_eval_{s}.json'] = [dict(asdict(x), held_out=x.family in HELD_OUT) for x in sc]
manifest = {}
for name, data in suites.items():
    text = json.dumps(data, sort_keys=True); Path('datasets', name).write_text(text)
    manifest[name] = {'sha256': hashlib.sha256(text.encode()).hexdigest(), 'scenarios': len(data)}
Path('datasets/drive_manifest.json').write_text(json.dumps(manifest, indent=1)); print(manifest)
