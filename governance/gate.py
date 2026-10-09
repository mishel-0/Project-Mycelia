"""Release gate: policy integrity, kill switch, one-time holdout scoring.

The gate never promotes. A verified candidate gets exactly one holdout
evaluation, recorded in a hash-chained release log, and then waits for a
human (`python -m governance.approve`).
"""
from __future__ import annotations
import hashlib, json, tomllib
from pathlib import Path
import numpy as np
from mycelia.symbiosis_lab.benchmark import sha256, train, evaluate
from mycelia.symbiosis_lab.registry import AppendLog, Registry

HERE = Path(__file__).resolve().parent


class PolicyError(RuntimeError):
    pass


class Gate:
    def __init__(self, benchmark_root, registry_root, state_root=HERE / 'state', policy=HERE / 'policy.toml',
                 policy_hash=HERE / 'policy.sha256', stop_file=HERE / 'STOP'):
        self.bench = Path(benchmark_root); self.registry = Registry(registry_root)
        self.state = Path(state_root); self.state.mkdir(parents=True, exist_ok=True)
        self.policy_path, self.policy_hash, self.stop_file = Path(policy), Path(policy_hash), Path(stop_file)
        self.releases = AppendLog(self.state / 'releases.jsonl'); self.production_path = self.state / 'production.json'

    def policy(self):
        raw = self.policy_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.policy_hash.read_text().strip():
            raise PolicyError('policy.toml does not match policy.sha256: refusing to operate')
        return tomllib.loads(raw.decode())

    def stopped(self):
        return self.stop_file.exists()

    def production(self):
        return json.loads(self.production_path.read_text()) if self.production_path.exists() else None

    def _check_frozen(self):
        manifest = json.loads((self.bench / 'manifest.json').read_text())
        for name in ('lab.npz', 'holdout.npz'):
            if sha256(self.bench / name) != manifest['sha256'][name]:
                raise PolicyError(f'frozen benchmark {name} changed')

    def request_release(self, candidate_id):
        policy = self.policy()
        if self.stopped():
            raise PolicyError('STOP is set')
        if not self.registry.log.verify() or not self.releases.verify():
            raise PolicyError('audit log hash chain broken')
        self._check_frozen()
        cand = self.registry.get(candidate_id); prod = self.production()
        if cand['status'] != 'verified':
            raise PolicyError(f'candidate status is {cand["status"]}, must be verified')
        if prod is None or cand['parent'] != prod['id']:
            raise PolicyError('candidate must descend from the current production version')
        used = sum(r.get('candidate') == candidate_id for r in self.releases.records())
        if used >= policy['release']['holdout_scoring_per_candidate']:
            raise PolicyError('holdout already used for this candidate')
        lab = np.load(self.bench / 'lab.npz'); hold = np.load(self.bench / 'holdout.npz')
        scores = {}
        for name, config in (('candidate', cand['config']), ('production', prod['config'])):
            e = train(config, lab['train_x'], lab['train_flip'], lab['train_y'], 0)
            scores[name] = evaluate(e, hold['x'], hold['y']); scores[name]['memories'] = int(e.n)
        c, p = scores['candidate'], scores['production']
        flags = {k: c[k] < p[k] - policy['acceptance'][f'tolerance_{t}'] for k, t in (
            ('accuracy', 'accuracy'), ('selective_accuracy', 'selective_accuracy'),
            ('tumor_presence_accuracy', 'tumor_presence'))}
        record = self.releases.append({'candidate': candidate_id, 'parent': prod['id'], 'holdout': scores,
                                       'holdout_regressions': flags, 'status': 'awaiting_human'})
        self.registry.update(candidate_id, 'awaiting_human', release_hash=record['hash'])
        return record

    # Human-only operations (called from governance.approve) -----------------
    def bootstrap(self, config, approver):
        if self.production() is not None:
            raise PolicyError('production already exists')
        self._write_production({'id': 'baseline', 'config': config, 'approved_by': approver, 'parent': None})

    def approve(self, candidate_id, approver):
        self.policy()
        rec = [r for r in self.releases.records() if r.get('candidate') == candidate_id]
        if not rec or rec[-1]['status'] != 'awaiting_human':
            raise PolicyError('no release awaiting approval for ' + candidate_id)
        cand = self.registry.get(candidate_id); prev = self.production()
        self._write_production({'id': candidate_id, 'config': cand['config'], 'approved_by': approver, 'parent': prev['id']},
                               previous=prev)
        self.releases.append({'candidate': candidate_id, 'status': 'approved', 'approved_by': approver})
        self.registry.update(candidate_id, 'approved', approved_by=approver)

    def rollback(self, approver):
        hist = json.loads((self.state / 'production-history.json').read_text())
        if not hist:
            raise PolicyError('nothing to roll back to')
        prev = hist.pop(); cur = self.production()
        (self.state / 'production-history.json').write_text(json.dumps(hist))
        self.production_path.write_text(json.dumps(prev, indent=2))
        self.releases.append({'candidate': cur['id'], 'status': 'rolled_back', 'restored': prev['id'], 'approved_by': approver})
        if cur['id'] != 'baseline':
            self.registry.update(cur['id'], 'rolled_back', approved_by=approver)

    def _write_production(self, prod, previous=None):
        hist_path = self.state / 'production-history.json'
        hist = json.loads(hist_path.read_text()) if hist_path.exists() else []
        if previous is not None:
            hist.append(previous)
        hist_path.write_text(json.dumps(hist)); self.production_path.write_text(json.dumps(prod, indent=2))
