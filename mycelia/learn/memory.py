"""Four memories in one SQLite file: declarative, procedural, evidence, coordination.

Append-only with supersession (nothing is overwritten), so knowledge can be
rebuilt offline from memory without re-reading any source.
"""
from __future__ import annotations
import json, sqlite3, time


class LearnMemory:
    TABLES = ('declarative', 'procedural', 'evidence', 'coordination')

    def __init__(self, path=':memory:'):
        self.db = sqlite3.connect(path)
        for t in self.TABLES:
            self.db.execute(f'CREATE TABLE IF NOT EXISTS {t} (id INTEGER PRIMARY KEY, key TEXT, body TEXT, episode INTEGER, '
                            f'created REAL, superseded_by INTEGER)')

    def put(self, table, key, body, episode=0):
        prev = self.db.execute(f'SELECT id FROM {table} WHERE key=? AND superseded_by IS NULL', (key,)).fetchall()
        cur = self.db.execute(f'INSERT INTO {table} (key, body, episode, created) VALUES (?,?,?,?)',
                              (key, json.dumps(body, default=str), episode, time.time()))
        for (pid,) in prev:
            self.db.execute(f'UPDATE {table} SET superseded_by=? WHERE id=?', (cur.lastrowid, pid))
        return cur.lastrowid

    def current(self, table):
        return {k: json.loads(b) for k, b in self.db.execute(f'SELECT key, body FROM {table} WHERE superseded_by IS NULL')}

    def history(self, table, key):
        return [json.loads(b) for (b,) in self.db.execute(f'SELECT body FROM {table} WHERE key=? ORDER BY id', (key,))]

    def log_exchange(self, record):
        return self.put('coordination', f"{record['sender']}->{record['receiver']}:{record['skill']}:{time.time_ns()}", record)

    def store(self, skills, declarative, evidence, episode=0):
        for op, s in skills.items():
            self.put('procedural', op, s, episode)
        for op, d in declarative.items():
            self.put('declarative', op, d, episode)
        for op, e in evidence.items():
            self.put('evidence', op, e, episode)
