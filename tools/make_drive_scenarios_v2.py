"""Freeze the v0.2 driving evaluation suites before any v0.2 mechanism is written.

Final world state is post-shift: practice families listed in SHIFTED_FAMILIES
are generated with the hidden shift. Held-out families are included as is.
Seeds 300000+ never overlap practice seeds."""
import hashlib, json
from dataclasses import asdict
from pathlib import Path
from mycelia.autonomy.scenarios import make, TRAIN_V2, HELD_OUT, SHIFTED_FAMILIES

manifest = {}
for s in (1, 2, 3):
    data = []
    for fam in [f for f, _ in TRAIN_V2.values()] + HELD_OUT:
        for i in range(40):
            sc = make(fam, 300000 * s + i, shifted=fam in SHIFTED_FAMILIES)
            data.append(dict(asdict(sc), held_out=fam in HELD_OUT, post_shift=fam in SHIFTED_FAMILIES))
    name = f'drive_v2_eval_{s}.json'; text = json.dumps(data, sort_keys=True); Path('datasets', name).write_text(text)
    manifest[name] = {'sha256': hashlib.sha256(text.encode()).hexdigest(), 'scenarios': len(data)}
Path('datasets/drive_v2_manifest.json').write_text(json.dumps(manifest, indent=1)); print(manifest)
