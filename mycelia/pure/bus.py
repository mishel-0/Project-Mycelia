"""Message bus with structured messages and learned short symbols for verified concepts.

A verified concept gets a short id (e.g. c3). Later messages may refer to it by
id; a receiver resolves the id from the shared table. Byte savings and
decode success are measured, so a convention is only useful if it decodes.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict

KINDS = ('observation', 'hypothesis', 'counterexample', 'verified', 'request')


@dataclass
class Message:
    sender: str
    kind: str
    content: dict
    evidence: list = field(default_factory=list)
    confidence: float = 1.
    status: str = 'unverified'


class Bus:
    def __init__(self):
        self.log, self.symbols, self.stats = [], {}, {'full_bytes': 0, 'symbol_bytes': 0, 'decoded': 0, 'decode_failures': 0}

    def post(self, msg):
        if msg.kind not in KINDS:
            raise ValueError(msg.kind)
        full = json.dumps(asdict(msg), sort_keys=True); key = json.dumps(msg.content, sort_keys=True)
        if msg.kind == 'verified' and key not in self.symbols:
            self.symbols[key] = f'c{len(self.symbols)}'
        sym = self.symbols.get(key)
        compact = json.dumps({'s': msg.sender, 'k': msg.kind, 'id': sym}) if sym else full
        self.stats['full_bytes'] += len(full); self.stats['symbol_bytes'] += len(compact)
        self.log.append(msg); return sym

    def resolve(self, sym):
        for key, s in self.symbols.items():
            if s == sym:
                self.stats['decoded'] += 1; return json.loads(key)
        self.stats['decode_failures'] += 1; return None
