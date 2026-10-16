"""Append-only, hash-chained logs and the candidate registry.

Every line stores the hash of the previous line, so editing or deleting a
past record breaks `verify()`. Candidate status changes are new events, never
rewrites.
"""
from __future__ import annotations
import hashlib, json, time, uuid
from pathlib import Path

STATUSES = ('proposed', 'tested', 'rejected', 'verified', 'awaiting_human', 'approved', 'rolled_back')


class AppendLog:
    def __init__(self, path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)

    def records(self):
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def append(self, record):
        prev = self.records()[-1]['hash'] if self.path.exists() and self.records() else '0' * 64
        body = {**record, 'time': time.time(), 'prev': prev}
        body['hash'] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        with self.path.open('a') as f:
            f.write(json.dumps(body, sort_keys=True) + '\n')
        return body

    def verify(self):
        prev = '0' * 64
        for r in self.records():
            body = {k: v for k, v in r.items() if k != 'hash'}
            if r['prev'] != prev or hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != r['hash']:
                return False
            prev = r['hash']
        return True


class Registry:
    def __init__(self, root):
        self.log = AppendLog(Path(root) / 'candidates.jsonl')

    def propose(self, parent, config, hypothesis, expected_benefit, mutation, budget, test_plan):
        cid = 'cand-' + uuid.uuid4().hex[:10]
        self.log.append({'event': 'proposed', 'id': cid, 'parent': parent, 'config': config, 'hypothesis': hypothesis,
                         'expected_benefit': expected_benefit, 'mutation': mutation, 'compute_budget': budget,
                         'test_plan': test_plan, 'rollback': parent, 'status': 'proposed'})
        return cid

    def update(self, cid, status, **data):
        if status not in STATUSES:
            raise ValueError(f'unknown status {status}')
        if cid not in self.candidates():
            raise KeyError(cid)
        return self.log.append({'event': 'status', 'id': cid, 'status': status, **data})

    def candidates(self):
        out = {}
        for r in self.log.records():
            if r['event'] == 'proposed':
                out[r['id']] = dict(r)
            elif r['id'] in out:
                out[r['id']].update({k: v for k, v in r.items() if k not in ('event', 'hash', 'prev', 'time')})
        return out

    def get(self, cid):
        return self.candidates()[cid]
