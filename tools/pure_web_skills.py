"""Pure Mycelia v0.2: learn string skills from docs.python.org, verify them by
experiment, share them on the bus, and solve held-out tasks.

No neural networks, pretrained models or LLMs. The live page is fetched
through the allowlisted gateway; if the domain is blocked the run stops and
says so (nothing is simulated). An altered, untrusted page is injected to
test that unverified claims are rejected.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from mycelia.pure.bus import Bus
from mycelia.pure.sandbox import run
from mycelia.pure.skills import learn_skills
from mycelia.pure.solver import Solver
from mycelia.reach.channels import Reach, ChannelError
from receptor_memory_mri import wilson

URL = 'https://docs.python.org/3/library/stdtypes.html'


def evaluate(solver, tasks, budget, learn=False):
    rows = []
    for t in tasks:
        sol = solver.solve(t, budget)
        if sol['program'] is None:
            outcome = 'abstained'
        else:
            ok = all(run(sol['program'], {'x': x}) == ('ok', y) for x, y in t['hidden'])
            outcome = 'correct' if ok else 'wrong'
            if learn and ok:
                solver.learn_from(t, sol)
        rows.append({'id': t['id'], 'goal': t['goal'], 'solvable': t['solvable_by_str_methods'], 'outcome': outcome,
                     'program': sol['program'], 'attempts': sol['attempts']})
    solv = [r for r in rows if r['solvable']]; unsolv = [r for r in rows if not r['solvable']]
    k = sum(r['outcome'] == 'correct' for r in solv)
    return {'solved_correctly': k, 'solvable_tasks': len(solv), 'solved_wilson_95': wilson(k, len(solv)),
            'wrong': sum(r['outcome'] == 'wrong' for r in rows),
            'correct_abstentions_on_unsolvable': sum(r['outcome'] == 'abstained' for r in unsolv), 'unsolvable_tasks': len(unsolv),
            'mean_attempts_when_solved': float(np.mean([r['attempts'] for r in solv if r['outcome'] == 'correct'] or [0])),
            'rows': rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=Path('results/pure-web')); ap.add_argument('--budget', type=int, default=20000)
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True); start = time.monotonic()
    sources = [('github://python/cpython/Doc/builtins/stdtypes.rst', 'rst'), (URL, 'html')]
    reach = Reach(a.output / 'reach', web_allowlist=('docs.python.org',)); page, attempts = None, []
    for src, fmt in sources:  # Mycelia Reach channels (Agent-Reach style), first that works
        try:
            page = reach.read(src); page['format'] = fmt; break
        except ChannelError as e:
            attempts.append({'source': src, 'error': str(e)[:300]})
    if page is None:
        (a.output / 'preflight.json').write_text(json.dumps({'ok': False, 'attempts': attempts}, indent=2))
        sys.exit('No documentation source reachable; see preflight.json. Nothing was simulated.')
    doc_url = page.get('url', page['source'])
    (a.output / 'source.json').write_text(json.dumps({k: page[k] for k in ('source', 'channel', 'backend', 'sha256', 'retrieved')} |
                                                     {'url': doc_url, 'commit': page.get('commit'), 'failed_attempts': attempts}, indent=2))
    print('documentation from', doc_url, flush=True)
    tasks = json.loads(Path('datasets/pure_tasks.json').read_text())
    train, test = [t for t in tasks if t['split'] == 'train'], [t for t in tasks if t['split'] == 'test']

    # Discover + experiment + share. Only skills announced as verified on the bus reach the solver.
    bus = Bus(); untrusted = Path('tests/fixtures/untrusted_altered_docs.html').read_text()
    skills, consts, bus = learn_skills([(page['text'], doc_url, page['format']), (untrusted, 'untrusted-injected')], bus)
    shared = {m.content['skill'] for m in bus.log if m.kind == 'verified'}
    verified = [s for s in skills if s['status'] == 'verified' and s['source'] == doc_url and s['method'] in shared]
    observed, _, _ = learn_skills([(page['text'], doc_url, page['format']), (untrusted, 'untrusted-injected')], Bus(), verify=False)
    R = {'source': doc_url, 'channel': page['channel'], 'skills_extracted': len([s for s in skills if s['source'] == doc_url]),
         'skills_verified': sorted(s['method'] for s in verified),
         'skills_contested': [{'method': s['method'], 'source': s['source'], 'evidence': s['evidence']['examples_contradicted']}
                              for s in skills if s['status'] == 'contested'],
         'not_allowed': sorted({s['method'] for s in skills if s['status'] == 'not_allowed'}),
         'learned_constants': consts, 'bus': {'messages': len(bus.log), 'verified': len(shared)}}
    print('verified skills:', len(verified), 'contested:', len(R['skills_contested']), flush=True)

    R['blind_enumeration'] = evaluate(Solver(verified, consts, use_docs=False), test, a.budget)
    R['docs_relevance'] = evaluate(Solver(verified, consts), test, a.budget)
    learner = Solver(verified, consts); R['train_tasks'] = evaluate(learner, train, a.budget, learn=True)
    R['docs_relevance_plus_priors'] = evaluate(Solver(verified, consts, priors=dict(learner.priors)), test, a.budget)
    trusting = [dict(s, required=sum(not p['optional'] for p in s['params'])) for s in observed]
    R['unverified_docs_trusted'] = evaluate(Solver(trusting, consts), test, a.budget)
    for k in ('blind_enumeration', 'docs_relevance', 'docs_relevance_plus_priors', 'unverified_docs_trusted'):
        v = R[k]; print(k, v['solved_correctly'], '/', v['solvable_tasks'], 'wrong', v['wrong'], 'abstain-ok', v['correct_abstentions_on_unsolvable'],
                        'attempts', round(v['mean_attempts_when_solved'], 1), flush=True)
    R['seconds'] = time.monotonic() - start
    (a.output / 'results.json').write_text(json.dumps(R, indent=2, default=str))


if __name__ == '__main__':
    main()
