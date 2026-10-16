"""Discover, experiment, verify and share skills learned from documentation.

Discoverer turns extracted entries into candidate skills. Experimenter runs
each documented example in the sandbox and probes edge inputs. A skill is
verified only by its own experiments; documentation claims alone are
'observed'. Contradicted examples are kept as counterexamples.
"""
from __future__ import annotations
import ast, re
from collections import Counter
from .bus import Bus, Message
from .sandbox import run, evaluate, SandboxError, PURE_STR_METHODS

PROBES = ['', ' ', 'a', 'Hello World', '  mixed CASE text  ', 'a,b,,c', 'x-y_z', '123', 'ÄÖü']
WORD = re.compile(r'[a-z]+')


def parse_expected(text):
    try:
        return ('ok', evaluate(text))
    except Exception:
        m = re.search(r'(\w+Error)', text)
        return ('error', m.group(1)) if m else ('unparsed', text)


def doc_constants(entries):
    """String/int literals used as arguments in documented examples (learned constants)."""
    found = Counter()
    for e in entries:
        for ex in e['examples']:
            try:
                tree = ast.parse(ex['source'], mode='eval')
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    for a in node.args:
                        if isinstance(a, ast.Constant) and isinstance(a.value, str) and len(a.value) <= 3:
                            found[a.value] += 1
    return [c for c, _ in found.most_common(8)]


def learned_stopwords(entries, share=.25):
    docs = [set(WORD.findall(e['description'].lower())) for e in entries]
    counts = Counter(w for d in docs for w in d)
    return {w for w, c in counts.items() if c > share * max(len(docs), 1)}


class Discoverer:
    name = 'discoverer'

    def run(self, entries, bus):
        stop = learned_stopwords(entries); out = []
        for e in entries:
            words = sorted(set(WORD.findall(e['description'].lower())) - stop)
            skill = {'method': e['method'], 'signature': e['signature'], 'params': e['params'], 'words': words,
                     'examples': e['examples'], 'source': e['source'], 'status': 'observed'}
            bus.post(Message(self.name, 'hypothesis', {'skill': e['method'], 'claims': len(e['examples'])}, [e['source']], .5))
            out.append(skill)
        return out


class Experimenter:
    name = 'experimenter'

    def __init__(self, constants=()):
        self.constants = list(constants)  # probe arguments learned from documented examples

    def run(self, skill, bus):
        m = skill['method']; record = {'examples_reproduced': 0, 'examples_contradicted': [], 'examples_unrunnable': 0, 'probes': {}}
        if m not in PURE_STR_METHODS:
            skill.update(status='not_allowed', evidence=record); return skill
        for ex in skill['examples']:
            got, want = run(ex['source']), parse_expected(ex['expected'])
            if got[0] == 'blocked' or want[0] == 'unparsed':
                record['examples_unrunnable'] += 1
            elif got == want:
                record['examples_reproduced'] += 1
            else:
                record['examples_contradicted'].append({'source': ex['source'], 'documented': ex['expected'], 'observed': repr(got)})
        required = [p for p in skill['params'] if not p['optional']]
        consts = list(dict.fromkeys(self.constants + [',', ' ']))  # learned constants plus two generic probes
        arg_sets = [()] if not required else ([(c,) for c in consts] if len(required) == 1 else
                                               [(a, b) for a in consts for b in consts if a != b][:6])
        for probe in PROBES:
            for args in arg_sets:
                call = f'{probe!r}.{m}({", ".join(map(repr, args))})' if m != 'join' else f'{args[0]!r}.join(x.split())' if args else None
                if call:
                    record['probes'][f'{probe!r} {args}'] = run(call if m == 'join' else f'x.{m}({", ".join(map(repr, args))})', {'x': probe})
        callable_ok = any(r[0] == 'ok' for r in record['probes'].values()) or record['examples_reproduced'] > 0
        if record['examples_contradicted']:
            skill['status'] = 'contested'
            bus.post(Message(self.name, 'counterexample', {'skill': m}, record['examples_contradicted'], 1., 'refuted'))
        elif callable_ok:
            skill['status'] = 'verified'
            bus.post(Message(self.name, 'verified', {'skill': m, 'words': skill['words']},
                             [f'{record["examples_reproduced"]} examples reproduced'], 1., 'verified'))
        else:
            skill['status'] = 'unverified'
        out_types = {type(v[1]).__name__ for v in record['probes'].values() if v[0] == 'ok'}
        skill.update(evidence=record, returns=sorted(out_types), required=len(required)); return skill


def learn_skills(pages, bus=None, verify=True):
    """pages: [(text, url[, "rst"])] -> skills, with the bus log as shared record."""
    from .extract import extract, extract_rst
    bus = bus or Bus()
    entries = [e for page in pages for e in (extract_rst(page[0], page[1]) if len(page) > 2 and page[2] == 'rst'
                                             else extract(page[0], page[1]))]
    skills = Discoverer().run(entries, bus)
    if verify:
        consts = doc_constants(entries); skills = [Experimenter(consts).run(s, bus) for s in skills]
    return skills, doc_constants(entries), bus
