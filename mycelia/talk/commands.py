"""Mycelia's own answers: every reply is read from real data (the diagnosis
bundle, the Symbiosis Lab records, the governance release log). Nothing is
generated beyond the MLCP decoder's field-traceable sentences."""
from __future__ import annotations
import json
from pathlib import Path

HELP = """Commands:
  diagnose <image path>   run the scan through Mycelia (research only)
  why                     explain the last decision, sentence by sentence
  weakness                what the Symbiosis Lab last found weak
  status                  production version and releases awaiting a human
  history [n]             last n candidate changes and why they were kept or rejected
  help | quit"""


def _jsonl(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


class Mycelia:
    def __init__(self, bundle_path='lab_state/talk/mycelia-bundle.pkl', lab_root='lab_state',
                 cycles_dir='results/symbiosis-lab'):
        self.bundle_path, self.lab, self.cycles = Path(bundle_path), Path(lab_root), Path(cycles_dir)
        self._bundle = None; self.last = None

    def bundle(self):
        if self._bundle is None:
            if not self.bundle_path.exists():
                raise FileNotFoundError(f'no diagnosis bundle at {self.bundle_path}; run tools/build_talk_bundle.py')
            from .bundle import MyceliaBundle
            self._bundle = MyceliaBundle.load(self.bundle_path)
        return self._bundle

    def diagnose(self, path):
        if not Path(path).is_file():
            return {'error': f'image not found: {path}'}
        self.last = self.bundle().diagnose(path); return self.last

    def why(self):
        if self.last is None:
            return {'error': 'no scan diagnosed yet in this conversation'}
        return {'image': self.last['image'], 'decision': self.last['decision'],
                'sentences': [{'text': t, 'field': f} for t, f in zip(self.last['explanation'], self.last['explanation_fields'])]}

    def weakness(self):
        files = sorted(self.cycles.glob('cycle-*.json'))
        if not files:
            return {'error': 'no Symbiosis Lab cycles recorded'}
        rec = json.loads(files[-1].read_text()); steps = {s['step']: s for s in rec['steps']}
        if 'weakness' in steps:
            w = steps['weakness']['detail']
            return {'cycle': files[-1].name, 'weak_class': w['label'], 'recall': w['recall'], 'ci_95': w['ci'],
                    'rule': w.get('rule'), 'evidence': w['evidence']}
        return {'cycle': files[-1].name, 'weak_class': None, 'reason': steps.get('no_action', {}).get('reason')}

    def status(self):
        gov = self.lab / 'governance'; prod = gov / 'production.json'
        releases = _jsonl(gov / 'releases.jsonl'); latest = {}
        for r in releases:
            latest[r['candidate']] = r
        pending = [{'candidate': c, 'holdout': {k: {m: v[m] for m in ('accuracy', 'selective_accuracy', 'tumor_presence_accuracy')}
                                                for k, v in r['holdout'].items()}, 'regressions': r['holdout_regressions']}
                   for c, r in latest.items() if r['status'] == 'awaiting_human']
        return {'production': json.loads(prod.read_text()) if prod.exists() else None, 'awaiting_human_approval': pending}

    def history(self, n=5):
        cands = {}
        for r in _jsonl(self.lab / 'registry' / 'candidates.jsonl'):
            if r['event'] == 'proposed':
                cands[r['id']] = {'id': r['id'], 'mutation': r['mutation'], 'hypothesis': r['hypothesis']}
            elif r['id'] in cands:
                cands[r['id']].update({k: r[k] for k in ('status', 'gain', 'reasons') if k in r})
        return {'candidates': list(cands.values())[-int(n):]}

    def run(self, line):
        """Command-line front end: returns text."""
        parts = line.strip().split(maxsplit=1)
        if not parts:
            return ''
        cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else '')
        if cmd == 'help':
            return HELP
        if cmd == 'diagnose':
            r = self.diagnose(arg.strip().strip('"\''))
            if 'error' in r:
                return r['error']
            head = (f"Decision: {r['decision']}" + (f" (leaning {r['tentative_label']})" if r['decision'] == 'abstain' else '') +
                    f" | risk {r['risk']:.2f} (threshold {r['threshold']:.2f})")
            return '\n'.join([head, *('  ' + s for s in r['explanation']), r['notice']])
        if cmd == 'why':
            r = self.why()
            return r.get('error') or '\n'.join(f"  {s['text']}  [{s['field']}]" for s in r['sentences'])
        if cmd in ('weakness', 'status', 'history'):
            r = self.history(int(arg)) if cmd == 'history' and arg.isdigit() else getattr(self, cmd)()
            return json.dumps(r, indent=2, default=str)
        return f'Unknown command {cmd!r}.\n' + HELP
