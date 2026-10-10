"""SQLite knowledge store with provenance; refuted facts are superseded, not deleted."""
from __future__ import annotations
import json, sqlite3, time

KINDS = ('episodic', 'semantic', 'procedural', 'failure')
STATUSES = ('observed', 'inferred', 'verified', 'refuted', 'suspect')


class Memory:
    def __init__(self, path=':memory:'):
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY, kind TEXT, key TEXT, value TEXT,
            status TEXT, confidence REAL, source TEXT, episode INTEGER, evidence TEXT, superseded_by INTEGER, created REAL)""")

    def add(self, kind, key, value, status, source, episode, evidence=(), confidence=1.):
        if kind not in KINDS or status not in STATUSES:
            raise ValueError((kind, status))
        cur = self.db.execute('INSERT INTO facts (kind,key,value,status,confidence,source,episode,evidence,created) VALUES (?,?,?,?,?,?,?,?,?)',
                              (kind, key, json.dumps(value), status, confidence, source, episode, json.dumps(list(evidence)), time.time()))
        return cur.lastrowid

    def supersede(self, old_id, new_id):
        self.db.execute('UPDATE facts SET superseded_by=? WHERE id=?', (new_id, old_id))

    def refute(self, fact_id, reason, source):
        row = self.db.execute('SELECT kind,key,value,episode FROM facts WHERE id=?', (fact_id,)).fetchone()
        new = self.add(row[0], row[1], json.loads(row[2]), 'refuted', source, row[3], [reason])
        self.supersede(fact_id, new); return new

    def active(self, kind, episode=None, key=None):
        q = 'SELECT id,key,value,status,confidence,source,evidence FROM facts WHERE kind=? AND superseded_by IS NULL'
        args = [kind]
        if episode is not None:
            q += ' AND episode=?'; args.append(episode)
        if key is not None:
            q += ' AND key=?'; args.append(key)
        return [dict(id=r[0], key=r[1], value=json.loads(r[2]), status=r[3], confidence=r[4], source=r[5],
                     evidence=json.loads(r[6])) for r in self.db.execute(q, args)]

    def history(self, key):
        return [dict(id=r[0], status=r[1], superseded_by=r[2]) for r in
                self.db.execute('SELECT id,status,superseded_by FROM facts WHERE key=? ORDER BY id', (key,))]
