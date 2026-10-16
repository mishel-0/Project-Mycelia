"""Rule-based specialists behind one interface: `run(task) -> dict`.

Any of them can later be replaced by a language-model specialist with the
same inputs and outputs; the gate and acceptance rules do not change.
"""
from __future__ import annotations
import copy, hashlib, re
import numpy as np
from .benchmark import LABELS

ALLOWED_CONFIG_KEYS = {'engine', 'epochs', 'mirror'}
GUARDED = (('accuracy', 'tolerance_accuracy'), ('selective_accuracy', 'tolerance_selective_accuracy'),
           ('robust_accuracy', 'tolerance_robust_accuracy'), ('tumor_presence_accuracy', 'tolerance_tumor_presence'))


def validate_candidate(config):
    extra = set(config) - ALLOWED_CONFIG_KEYS
    if extra:
        raise PermissionError(f'candidate may only change {sorted(ALLOWED_CONFIG_KEYS)}, not {sorted(extra)}')


def judge(cand, parent, weak, acceptance, gain_key='min_gain'):
    """Predeclared rule: weak-class recall gain >= threshold and no guarded regression."""
    gain = cand['recall'][weak] - parent['recall'][weak]; reasons = []
    if gain < acceptance[gain_key]:
        reasons.append(f'{weak} recall gain {gain:+.4f} < {acceptance[gain_key]}')
    for key, tol in GUARDED:
        if key in cand and cand[key] < parent[key] - acceptance[tol]:
            reasons.append(f'{key} regressed {cand[key] - parent[key]:+.4f} (tolerance {acceptance[tol]})')
    if cand['memories'] > acceptance['max_memory_ratio'] * parent['memories']:
        reasons.append(f"memory {cand['memories']:.0f} > {acceptance['max_memory_ratio']}x parent")
    return not reasons, gain, reasons


class Evaluator:
    def __init__(self, gateway, benchmark_root, limits):
        self.gateway, self.benchmark_root, self.limits = gateway, str(benchmark_root), limits

    def run(self, task):
        validate_candidate(task['config'])
        spec = {'id': task.get('id'), 'config': task['config'], 'benchmark': self.benchmark_root,
                'seeds': list(task.get('seeds', (0, 1, 2)))}
        return self.gateway.run_sandboxed_experiment(spec, self.limits['sandbox_cpu_seconds'],
                                                     self.limits['sandbox_memory_bytes'], self.limits['sandbox_wall_seconds'])


class Diagnostician:
    def run(self, task):
        """Weakness = a class whose recall CI lies entirely below every other class's."""
        s = task['scorecard']['summary']; ci = s['recall_ci']
        worst = min(LABELS, key=lambda l: s['recall'][l])
        others = [ci[l][0] for l in LABELS if l != worst]
        base = {'label': worst, 'label_index': LABELS.index(worst), 'recall': s['recall'][worst], 'ci': ci[worst]}
        if ci[worst][1] < min(others):
            return {**base, 'rule': 'absolute', 'evidence': 'recall CI below every other class on 3 seeds'}
        # Relative rule (added after cycle 001 found no absolute weakness): CI below the
        # best class and recall at least 3 points under the mean of the other classes.
        gap = np.mean([s['recall'][l] for l in LABELS if l != worst]) - s['recall'][worst]
        if ci[worst][1] < max(others) and gap >= .03:
            return {**base, 'rule': 'relative', 'gap': float(gap),
                    'evidence': f'recall CI below the best class and {gap:.3f} under the others on 3 seeds'}
        return None


class Researcher:
    QUERIES = {'glioma': 'https://export.arxiv.org/api/query?search_query=all:glioma+AND+all:meningioma+AND+all:MRI&max_results=3'}

    def __init__(self, gateway, memory):
        self.gateway, self.memory = gateway, memory

    def run(self, task):
        label = task['weakness']['label']; found, attempts = [], []
        url = self.QUERIES.get(label)
        if url:
            try:
                page = self.gateway.fetch_page(url); found.append(self._cite(page['url'], page['text'], label))
                attempts.append({'url': url, 'ok': True})
            except Exception as e:
                attempts.append({'url': url, 'ok': False, 'reason': str(e)[:200]})
        for doc in task.get('documents', ()):
            try:
                d = self.gateway.read_document(doc); c = self._cite(d['source'], d['text'], label)
                if c['excerpts']:
                    found.append(c)
            except Exception as e:
                attempts.append({'document': doc, 'ok': False, 'reason': str(e)[:200]})
        for c in found:
            self.memory.remember('semantic', c, source=c['source'])
        return {'citations': found, 'attempts': attempts}

    @staticmethod
    def _cite(source, text, label):
        # Untrusted text is only searched and quoted; it never becomes instructions.
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if label in s.lower()]
        return {'source': source, 'content_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'excerpts': [s[:300] for s in sentences[:5]], 'untrusted': True}


class Engineer:
    """Fixed mutation grammar; procedural memory biases which mutations go first."""
    def __init__(self, memory):
        self.memory = memory

    def mutations(self, weak):
        return [('weak_class_growth+0.25', lambda c: self._margin(c, weak, .25),
                 f'Growing new traces more readily for {LABELS[weak]} adds boundary examples where errors cluster.'),
                ('weak_class_growth+0.45', lambda c: self._margin(c, weak, .45),
                 f'A stronger {LABELS[weak]} growth margin captures more of its variability.'),
                ('sharper_recall', lambda c: self._engine(c, 'gamma', c['engine'].get('gamma', 5.) * 1.5),
                 'Sharper recall reduces interference from look-alike classes.'),
                ('softer_recall', lambda c: self._engine(c, 'gamma', c['engine'].get('gamma', 5.) / 1.5),
                 'Softer recall pools more neighbours and may smooth noisy errors.'),
                ('extra_epoch', lambda c: {**c, 'epochs': int(c.get('epochs', 2)) + 1},
                 'One more pass lets failures grow traces that the first passes missed.'),
                ('larger_budget', lambda c: self._engine(c, 'budget', int(c['engine'].get('budget', 4000) * 1.5)),
                 'More memory capacity keeps rare variants instead of evicting them.')]

    @staticmethod
    def _engine(c, key, value):
        c = copy.deepcopy(c); c.setdefault('engine', {})[key] = value; return c

    @staticmethod
    def _margin(c, weak, delta):
        c = copy.deepcopy(c); e = c.setdefault('engine', {})
        e['label_margins'] = [[weak, e.get('grow_margin', .2) + delta]]; return c

    def run(self, task):
        weak, n = task['weakness']['label_index'], task['n']
        ranked = sorted(enumerate(self.mutations(weak)), key=lambda m: (-self.memory.weight(m[1][0]), m[0]))
        out = []
        for _, (name, fn, why) in ranked[:n]:
            cfg = fn(task['parent_config']); validate_candidate(cfg)
            out.append({'mutation': name, 'config': cfg, 'hypothesis': why})
        return out


class Verifier:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def run(self, task):
        seeds = (10, 11, 12)
        cand = self.evaluator.run({'id': task['id'], 'config': task['config'], 'seeds': seeds})['summary']
        parent = self.evaluator.run({'id': 'parent', 'config': task['parent_config'], 'seeds': seeds})['summary']
        ok, gain, reasons = judge(cand, parent, task['weak'], task['acceptance'], 'replication_min_gain')
        return {'replicated': ok, 'gain': gain, 'reasons': reasons, 'seeds': seeds, 'candidate': cand, 'parent': parent}
