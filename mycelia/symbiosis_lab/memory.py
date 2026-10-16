"""Lab memory with distinct roles; every record carries provenance.

episodic   - cycle records (what was tried, what happened)
semantic   - evidence from documents (source, retrieval time, content hash)
procedural - mutation type -> outcome statistics; these adaptive weights
             bias which mutations the Engineer tries next, decaying over time
failure    - rejected candidates and why
"""
from __future__ import annotations
import json
from pathlib import Path
from .registry import AppendLog

VERSION = 'symbiosis-lab-0.1'


class LabMemory:
    def __init__(self, root):
        root = Path(root); self.root = root
        self.episodic = AppendLog(root / 'episodic.jsonl'); self.semantic = AppendLog(root / 'semantic.jsonl')
        self.failure = AppendLog(root / 'failure.jsonl'); self.procedural_path = root / 'procedural.json'

    def remember(self, kind, record, source, confidence='unverified'):
        log = {'episodic': self.episodic, 'semantic': self.semantic, 'failure': self.failure}[kind]
        return log.append({**record, 'source': source, 'confidence': confidence, 'created_by': VERSION})

    def procedural(self):
        return json.loads(self.procedural_path.read_text()) if self.procedural_path.exists() else {}

    def weight(self, mutation):
        stats = self.procedural().get(mutation, {'wins': 0., 'trials': 0.})
        return (1 + stats['wins']) / (2 + stats['trials'])  # Laplace-smoothed success rate

    def reinforce(self, mutation, verified_gain, decay=.9):
        """Only verified outcomes update pathway weights; old evidence decays."""
        stats = self.procedural()
        for s in stats.values():
            s['wins'] *= decay; s['trials'] *= decay
        s = stats.setdefault(mutation, {'wins': 0., 'trials': 0.})
        s['trials'] += 1; s['wins'] += float(bool(verified_gain))
        self.procedural_path.parent.mkdir(parents=True, exist_ok=True)
        self.procedural_path.write_text(json.dumps(stats, indent=2))
