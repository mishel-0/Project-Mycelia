"""Verification lab and evidence accounting.

Claim lifecycle per operation (status is kept separate from evidence strength
and from source reliability):
  observed          extracted from a source, nothing checked yet
  supported         not testable here, but reported by >=2 independent lineages
  verified_in_scope sandbox reproduced documented examples / probes, none contradicted
  contested         some documented examples reproduce, some are contradicted
  refuted_in_scope  every runnable documented example is contradicted
  unverified        not adequately checked
A sandbox run proves behaviour only for the recorded inputs (its scope).
"""
from __future__ import annotations
import hashlib, re
from collections import Counter, defaultdict
from mycelia.pure.sandbox import run, evaluate, PURE_STR_METHODS, SAFE_BUILTINS, SAFE_MODULE_FUNCTIONS

WORD = re.compile(r'[a-z]+')
STR_PROBES = ['', ' ', 'a', 'Hello World', '  mixed CASE text  ', 'a,b,,c', 'x-y_z', '123']
LIST_PROBES = [[1, 2, 3], [2, 8, 4, 4], [3.5, 1.0], ['b', 'a', 'c']]
NUM_PROBES = [0, 1, 7, 12, 2.5, -3]


def normalized(text):
    return ' '.join(WORD.findall(text.lower()))


def shingles(text, k=5):
    w = normalized(text).split()
    return {' '.join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def relation_between(a, b):
    """How two reports of the same subject relate: copy | same_lineage | independent."""
    if a['lineage'] == b['lineage']:
        same_text = normalized(a['text']) == normalized(b['text'])
        sa, sb = shingles(a['text']), shingles(b['text'])
        overlap = len(sa & sb) / max(1, len(sa | sb))
        return 'copy' if same_text or overlap >= .8 else 'same_lineage'
    sa, sb = shingles(a['text']), shingles(b['text'])
    return 'copy' if len(sa & sb) / max(1, len(sa | sb)) >= .8 else 'independent'


def executable(op):
    if op.startswith('str.'):
        return op[4:] in PURE_STR_METHODS
    if '.' in op:
        mod, name = op.split('.', 1)
        return name in SAFE_MODULE_FUNCTIONS.get(mod, ())
    return op in SAFE_BUILTINS


def parse_expected(text):
    try:
        return ('ok', evaluate(text))
    except Exception:
        m = re.search(r'(\w+Error)', text)
        return ('error', m.group(1)) if m else ('unparsed', text)


def call_expr(op, args=()):
    a = ', '.join(map(repr, args))
    return f'x.{op[4:]}({a})' if op.startswith('str.') else f'{op}(x{", " if a else ""}{a})'


class Lab:
    def __init__(self, constants=(',', ' ')):
        self.constants = list(constants)

    def test(self, op, examples, params):
        """Run documented examples and probes; return the evidence record."""
        rec = {'reproduced': [], 'contradicted': [], 'unrunnable': 0, 'probes': []}
        for ex in examples:
            got, want = run(ex['source']), parse_expected(ex['expected'])
            if got[0] == 'blocked' or want[0] == 'unparsed':
                rec['unrunnable'] += 1
            elif got == want or (got[0] == 'ok' and want[0] == 'ok' and isinstance(got[1], float)
                                 and isinstance(want[1], (int, float)) and abs(got[1] - want[1]) < 1e-9):
                rec['reproduced'].append({'expr': ex['source'], 'lineage': ex.get('lineage')})
            else:
                rec['contradicted'].append({'expr': ex['source'], 'documented': ex['expected'], 'observed': repr(got),
                                            'lineage': ex.get('lineage')})
        if executable(op):
            required = [p for p in params if not p['optional']]
            arg_sets = [()] if len(required) <= (0 if op.startswith('str.') else 1) else [(c,) for c in self.constants][:3]
            probes = STR_PROBES if op.startswith('str.') else STR_PROBES[:4] + LIST_PROBES + NUM_PROBES
            for p in probes:
                for args in arg_sets:
                    r = run(call_expr(op, args), {'x': p})
                    if r[0] == 'ok':
                        rec['probes'].append({'input': repr(p), 'args': list(args), 'output': repr(r[1])[:80]})
        return rec

    @staticmethod
    def status(rec, independent_lineages, runnable_op):
        if rec['contradicted'] and not rec['reproduced']:
            return 'refuted_in_scope'
        if rec['contradicted']:
            return 'contested'
        if rec['reproduced'] or (runnable_op and rec['probes']):
            return 'verified_in_scope'
        if independent_lineages >= 2:
            return 'supported'
        return 'unverified'


def build(claims, constants=None):
    """Claims from any sources -> (skills, declarative, evidence) with lifecycle and independence."""
    by = defaultdict(list)
    for c in claims:
        by[c['subject']].append(c)
    descriptions = [c['object'] for c in claims if c['relation'] == 'describes' and isinstance(c['object'], str)]
    doc_freq = Counter(w for d in descriptions for w in set(WORD.findall(d.lower())))
    stop = {w for w, n in doc_freq.items() if n > .2 * max(1, len(descriptions))}
    consts = constants or [k for k, _ in Counter(ch for c in claims if c['relation'] == 'example'
                                                for ch in re.findall(r"'([^\w']{1,3})'", c['object']['source'])).most_common(6)] or [',', ' ']
    lab = Lab(consts); skills, declarative, evidence = {}, {}, {}
    for op, cs in by.items():
        sig = next((c['object'] for c in cs if c['relation'] == 'signature'), {'signature': op + '()', 'params': []})
        exs = [dict(c['object'], lineage=c['lineage']) for c in cs if c['relation'] == 'example']
        reports = [{'lineage': c['lineage'], 'text': c['object'], 'source': c['source']} for c in cs if c['relation'] == 'describes' and c['object']]
        rels = [relation_between(a, b) for i, a in enumerate(reports) for b in reports[i + 1:]]
        lineages = {r['lineage'] for r in reports} | {e['lineage'] for e in exs}
        rec = lab.test(op, exs, sig['params'])
        status = Lab.status(rec, len(lineages), executable(op))
        words = sorted({w for r in reports for w in WORD.findall(r['text'].lower())} - stop | set(WORD.findall(op.lower())))
        evidence[op] = rec | {'pairwise_relations': Counter(rels), 'lineages': sorted(lineages),
                              'sources': sorted({c['source'] for c in cs})}
        declarative[op] = {'descriptions': reports, 'status': status, 'domains': sorted({c['domain'] for c in cs})}
        if executable(op) and status == 'verified_in_scope':
            required = sum(not p['optional'] for p in sig['params'])
            skills[op] = {'op': op, 'kind': 'method' if op.startswith('str.') else 'function', 'params': sig['params'],
                          'required': required - (0 if op.startswith('str.') else 1) if required else 0, 'words': words,
                          'domains': declarative[op]['domains'], 'sources': evidence[op]['sources'], 'status': status,
                          'evidence_strength': {'examples_reproduced': len(rec['reproduced']), 'probes_ok': len(rec['probes']),
                                                'independent_lineages': len(lineages)}}
    reliability = {}
    for op, rec in evidence.items():
        for e in rec['reproduced'] + rec['contradicted']:
            ok, n = reliability.get(e['lineage'], (0, 0)); reliability[e['lineage']] = (ok + (e in rec['reproduced']), n + 1)
    return skills, declarative, evidence, {k: ok / n for k, (ok, n) in reliability.items()}, consts
