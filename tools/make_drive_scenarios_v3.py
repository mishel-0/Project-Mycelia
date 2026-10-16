"""Freeze the v0.3 driving suites before any v0.3 mechanism exists.
Each scenario is labelled with the obstacle types it contains, so results can
be reported per obstacle. Seeds 500000+ never overlap earlier suites."""
import hashlib, json
from dataclasses import asdict
from pathlib import Path
from mycelia.autonomy.scenarios import make, TRAIN_V2, HELD_OUT, SHIFTED_FAMILIES

manifest = {}
for s in (1, 2, 3):
    data = []
    for fam in [f for f, _ in TRAIN_V2.values()] + HELD_OUT:
        for i in range(40):
            sc = make(fam, 500000 * s + i, shifted=fam in SHIFTED_FAMILIES)
            kinds = sorted({h.kind for h in sc.hazards}) or ['none']
            data.append(dict(asdict(sc), held_out=fam in HELD_OUT, post_shift=fam in SHIFTED_FAMILIES, obstacles=kinds))
    name = f'drive_v3_eval_{s}.json'; text = json.dumps(data, sort_keys=True); Path('datasets', name).write_text(text)
    manifest[name] = {'sha256': hashlib.sha256(text.encode()).hexdigest(), 'scenarios': len(data)}
Path('datasets/drive_v3_manifest.json').write_text(json.dumps(manifest, indent=1)); print(manifest)
