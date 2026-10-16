"""Pure Mycelia v0.3 experiments: multi-source, cross-domain, network necessity,
exploration and continual learning. Sources are read live from GitHub through
Mycelia Reach (the only working channel here); web/RSS are reported untested.

Development used only datasets/learn_tasks_dev.json; the three eval sets were
frozen (SHA-256 manifest, committed) before the solver existed and are used here.
"""
from __future__ import annotations
import argparse, hashlib, json, random, time
from collections import Counter, defaultdict
from pathlib import Path
from mycelia.learn.evidence import build
from mycelia.learn.explorer import Explorer
from mycelia.learn.memory import LearnMemory
from mycelia.learn.network import Network, Node, merge
from mycelia.learn.solver import Solver, score, execute
from mycelia.learn.sources import Source, extract
from mycelia.reach.channels import Reach
from receptor_memory_mri import wilson

CP = 'github://python/cpython/'
SOURCES = {
    'text': [Source('stdtypes', CP + 'Doc/builtins/stdtypes.rst', 'rst', 'python/cpython', 'text')],
    'builtins': [Source('functions', CP + 'Doc/builtins/functions.rst', 'rst', 'python/cpython', 'builtins')],
    'math': [Source('math', CP + 'Doc/library/math.rst', 'rst', 'python/cpython', 'math')],
    'statistics': [Source('statistics-doc', CP + 'Doc/library/statistics.rst', 'rst', 'python/cpython', 'statistics'),
                   Source('statistics-code', CP + 'Lib/statistics.py', 'code', 'python/cpython', 'statistics')],
    'cheatsheet': [Source('cheatsheet', 'github://gto76/python-cheatsheet/README.md', 'markdown', 'gto76/python-cheatsheet', 'general')],
}
EVAL = ['learn_tasks_eval_101.json', 'learn_tasks_eval_202.json', 'learn_tasks_eval_303.json']


def summarise(rows):
    solv = [r for r in rows if r['solvable']]; uns = [r for r in rows if not r['solvable']]
    k = sum(r['outcome'] == 'correct' for r in solv)
    fam = defaultdict(lambda: [0, 0])
    for r in solv:
        fam[r['family']][0] += r['outcome'] == 'correct'; fam[r['family']][1] += 1
    return {'solved': k, 'solvable': len(solv), 'wilson_95': wilson(k, len(solv)), 'wrong': sum(r['outcome'] == 'wrong' for r in rows),
            'abstained_on_unsolvable': sum(r['outcome'] == 'abstained' for r in uns), 'unsolvable': len(uns),
            'by_family': {f: {'solved': a, 'of': b} for f, (a, b) in sorted(fam.items())},
            'cross_domain': [sum(r['outcome'] == 'correct' for r in solv if r['cross_domain']), sum(r['cross_domain'] for r in solv)],
            'novel_combination': [sum(r['outcome'] == 'correct' for r in solv if r['novel']), sum(r['novel'] for r in solv)]}


def exploration(a, reach, docs, dev, run_all):
    R = {}
    commit, catalog = reach.github.list('python', 'cpython', 'Doc/')
    catalog = [f for f in catalog if f.endswith('.rst') and (f.startswith('Doc/library/') or f.startswith('Doc/builtins/'))]
    start = 'Doc/builtins/stdtypes.rst'; R.update({'catalog_files': len(catalog), 'start': start})
    for strategy in ('goal', 'curiosity', 'random', 'alphabetical'):
        runs = []
        for seed in ((0, 1, 2, 3, 4) if strategy == 'random' else (0,)):
            ex = Explorer(catalog, strategy, seed); read = {start: docs['stdtypes']['text']}
            ex.observe(start, read[start]); trace = []
            for step in range(1, 6):
                cl = [c for p, txt in read.items() for c in extract(txt, Source(p, CP + p, 'rst', 'python/cpython', p))]
                sk, dc, _, _, cs = build(cl)
                res = run_all(sk, cs, budget=2000)['total']; trace.append({'reads': len(read) - 1, 'solved': res['solved'], 'read': list(read)})
                unsolved = ()
                if strategy == 'goal':  # only training (dev) tasks guide reading; eval tasks stay invisible
                    rows = score(Solver(sk, cs, budget=2000), dev); unsolved = [t['goal'] for t, r in zip(dev, rows) if r['outcome'] != 'correct' and t['solvable']]
                nxt = ex.next(unsolved)
                if nxt is None:
                    break
                try:
                    txt = reach.read(CP + nxt)['text']
                except Exception:
                    txt = ''
                read[nxt] = txt; ex.observe(nxt, txt, [o for o, d in dc.items() if d['status'] == 'unverified'])
            runs.append(trace)
        R[strategy] = runs; print('4', strategy, [[s['solved'] for s in r] for r in runs], runs[0][-1]['read'][1:], flush=True)

    return R


def exploration_only(a, reach, docs, dev, evals, run_all):
    R = exploration(a, reach, docs, dev, run_all)
    (a.output / 'exploration-rerun.json').write_text(json.dumps(R, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--output', type=Path, default=Path('results/learn'))
    ap.add_argument('--cache', type=Path, default=Path('lab_state/repos')); ap.add_argument('--budget', type=int, default=10000)
    ap.add_argument('--only', choices=['exploration'], help='rerun one experiment into its own results file')
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True); t0 = time.monotonic(); R = {}
    manifest = json.loads(Path('datasets/learn_tasks_manifest.json').read_text())
    for name in ['learn_tasks_dev.json'] + EVAL:
        assert hashlib.sha256(Path('datasets', name).read_text().encode()).hexdigest() == manifest[name]['sha256'], name
    dev = json.loads(Path('datasets/learn_tasks_dev.json').read_text()); evals = {n: json.loads(Path('datasets', n).read_text()) for n in EVAL}
    reach = Reach(a.output / 'reach', cache_dir=a.cache); docs, claims = {}, {}
    for group, srcs in SOURCES.items():
        for s in srcs:
            d = reach.read(s.uri); docs[s.id] = d; claims[s.id] = extract(d['text'], s)
    R['sources'] = {k: {'uri': d['source'], 'commit': d.get('commit'), 'sha256': d['sha256'], 'claims': len(claims[k]),
                        'by_relation': Counter(c['relation'] for c in claims[k])} for k, d in docs.items()}
    R['channels'] = {'github': 'live', 'web': 'blocked by environment network policy (untested)', 'rss': 'untested'}
    print('sources read', {k: v['claims'] for k, v in R['sources'].items()}, flush=True)

    def knowledge(ids):
        return build([c for i in ids for c in claims[i]])

    def run_all(skills, consts, priors=None, budget=a.budget):
        out = {}
        for n, tasks in evals.items():
            out[n] = summarise(score(Solver(skills, consts, priors=priors, budget=budget), tasks))
        tot = lambda key: sum(v[key] for v in out.values())
        out['total'] = {'solved': tot('solved'), 'solvable': tot('solvable'), 'wilson_95': wilson(tot('solved'), tot('solvable')),
                        'wrong': tot('wrong'), 'abstained_on_unsolvable': tot('abstained_on_unsolvable'), 'unsolvable': tot('unsolvable'),
                        'cross_domain': [sum(v['cross_domain'][i] for v in out.values() if 'cross_domain' in v) for i in (0, 1)],
                        'novel_combination': [sum(v['novel_combination'][i] for v in out.values() if 'novel_combination' in v) for i in (0, 1)]}
        return out

    if a.only == 'exploration':
        return exploration_only(a, reach, docs, dev, evals, run_all)
    # 1. Multi-source learning + evidence accounting.
    rst = ['stdtypes', 'functions', 'math', 'statistics-doc']; R['multi_source'] = {}
    for label, ids in (('rst_docs', rst), ('rst_plus_code', rst + ['statistics-code']), ('rst_code_markdown', rst + ['statistics-code', 'cheatsheet'])):
        skills, decl, ev, rel, consts = knowledge(ids)
        R['multi_source'][label] = {'skills_verified_in_scope': len(skills), 'lifecycle': Counter(d['status'] for d in decl.values()),
                                    'source_reliability': rel, 'evaluation': run_all(skills, consts)['total']}
        print('1', label, len(skills), R['multi_source'][label]['evaluation']['solved'], flush=True)
    full = knowledge(rst + ['statistics-code', 'cheatsheet']); skills_all, decl_all, ev_all, _, consts_all = full
    R['evidence_examples'] = {op: {'lineages': ev_all[op]['lineages'], 'pairwise': ev_all[op]['pairwise_relations'],
                                   'status': decl_all[op]['status']} for op in ('statistics.median', 'statistics.mean', 'str.strip', 'sorted')}

    # 2. Cross-domain: every cross-domain task solved with all sources must fail with any single document's skills.
    single = {i: knowledge([i]) for i in rst}; R['cross_domain'] = {'checked': 0, 'neither_alone_holds': 0, 'violations': []}
    for n, tasks in evals.items():
        rows = {r['id']: r for r in score(Solver(skills_all, consts_all, budget=a.budget), [t for t in tasks if t['cross_domain']])}
        for t in tasks:
            if t['cross_domain'] and rows[t['id']]['outcome'] == 'correct':
                R['cross_domain']['checked'] += 1
                alone = [i for i, k in single.items() if score(Solver(k[0], k[4], budget=a.budget), [t])[0]['outcome'] == 'correct']
                R['cross_domain']['neither_alone_holds'] += not alone
                if alone:
                    R['cross_domain']['violations'].append({'task': t['id'], 'solved_by_single_source': alone})
    print('2', R['cross_domain']['checked'], R['cross_domain']['neither_alone_holds'], flush=True)

    # 3. Network necessity: A isolated, B repository, C fixed ring, D adaptive (+ a lying node).
    node_skills = {g: knowledge([s.id for s in srcs]) for g, srcs in SOURCES.items()}
    liar_skills = {'str.unscramble': {'op': 'str.unscramble', 'kind': 'method', 'params': [], 'required': 0, 'words': ['sort', 'order', 'letters'],
                                      'domains': ['general'], 'sources': ['untrusted'], 'status': 'verified_in_scope'},
                   'str.swapcase': {'op': 'str.swapcase', 'kind': 'method', 'params': [], 'required': 0,
                                    'words': ['trim', 'remove', 'spaces', 'both', 'ends', 'average', 'mean', 'largest', 'words'],
                                    'domains': ['general'], 'sources': ['untrusted'], 'status': 'verified_in_scope'}}
    R['network'] = {}
    for budget in (1000, a.budget):
        R['network'][budget] = {}
        for cond in 'ABCD':
            per_seed = []
            for seed in (0, 1, 2):
                mem = LearnMemory(); nodes = [Node(g, k[0], k[4]) for g, k in node_skills.items()] + [Node('untrusted', liar_skills, [',', ' '])]
                net = Network(nodes, mem); names = [n.name for n in nodes if n.name != 'untrusted']
                rng = random.Random(seed)
                if cond == 'D':  # coordination learning on dev tasks only
                    for t in dev:
                        net.solve(rng.choice(names), t, 'D', budget, learn=True)
                rows = []
                for n, tasks in evals.items():
                    for t in tasks:
                        recv = rng.choice(names); sol, lib = net.solve(recv, t, cond, budget)
                        out = 'abstained' if sol['program'] is None else ('correct' if all(
                            execute(sol['program'], x) == ('ok', y) for x, y in t['hidden']) else 'wrong')
                        rows.append({'solvable': t['solvable'], 'family': t['family'], 'cross_domain': t['cross_domain'],
                                     'novel': t['novel_combination'], 'outcome': out, 'attempts': sol['attempts'], 'fake_in_library': 'str.unscramble' in lib})
                s = summarise(rows); s['false_claims_in_libraries'] = sum(r['fake_in_library'] for r in rows)
                s['mean_attempts'] = sum(r['attempts'] for r in rows) / len(rows); s['comm'] = dict(net.stats)
                s['exchanges_logged'] = len(mem.current('coordination')); per_seed.append(s)
            R['network'][budget][cond] = per_seed
            print('3', budget, cond, [p['solved'] for p in per_seed], 'of', per_seed[0]['solvable'], 'wrong', [p['wrong'] for p in per_seed],
                  'fake', [p['false_claims_in_libraries'] for p in per_seed], flush=True)
    # Node failure: statistics node lost at evaluation time.
    R['node_failure'] = {}
    for cond in 'BCD':
        mem = LearnMemory(); nodes = [Node(g, k[0], k[4]) for g, k in node_skills.items()]; net = Network(nodes, mem)
        names = [n.name for n in nodes]; rng = random.Random(0)
        if cond == 'D':
            for t in dev:
                net.solve(rng.choice(names), t, 'D', a.budget, learn=True)
        stat_tasks = [t for t in evals[EVAL[0]] if any(o.startswith('statistics.') for o in t['required_ops'])]
        rows = []
        for t in stat_tasks:
            recv = rng.choice([x for x in names if x != 'statistics']); sol, _ = net.solve(recv, t, cond, a.budget, failed=('statistics',))
            rows.append({'solvable': True, 'family': t['family'], 'cross_domain': t['cross_domain'], 'novel': t['novel_combination'],
                         'outcome': 'correct' if sol['program'] and all(execute(sol['program'], x) == ('ok', y) for x, y in t['hidden']) else 'abstained'})
        R['node_failure'][cond] = summarise(rows); print('3f', cond, R['node_failure'][cond]['solved'], '/', len(rows), flush=True)

    R['exploration'] = exploration(a, reach, docs, dev, run_all)

    # 5. Continual learning: read sources in episodes, re-test everything solved before.
    order = ['stdtypes', 'functions', 'math', 'statistics-doc', 'statistics-code', 'cheatsheet']
    extra = ['Doc/library/datetime.rst', 'Doc/library/textwrap.rst', 'Doc/library/itertools.rst', 'Doc/library/random.rst', 'Doc/library/operator.rst']
    for p in extra:
        d = reach.read(CP + p); docs[p] = d; claims[p] = extract(d['text'], Source(p, CP + p, 'rst', 'python/cpython', 'unrelated'))
    mem = LearnMemory(str(a.output / 'continual-memory.sqlite')); solved_before = {}; episodes = []
    tasks = evals[EVAL[0]]
    for e, sid in enumerate(order + extra, 1):
        sk, dc, ev, _, cs = knowledge(order[:e] if e <= len(order) else order + extra[:e - len(order)])
        mem.store(sk, dc, ev, e)
        rows = {r['id']: r for r in score(Solver(sk, cs, budget=a.budget), tasks)}
        lost = []
        for tid, ep in solved_before.items():
            if rows[tid]['outcome'] != 'correct':
                t = next(x for x in tasks if x['id'] == tid)
                missing = [o for o in t['required_ops'] if o not in sk and o != 'str.join']
                lost.append({'task': tid, 'cause': 'forgetting (skill missing)' if missing else 'retrieval failure (skill present, not found in budget)'})
        for tid, r in rows.items():
            if r['outcome'] == 'correct':
                solved_before.setdefault(tid, e)
        episodes.append({'episode': e, 'source': sid, 'skills': len(sk), 'solved': sum(r['outcome'] == 'correct' for r in rows.values()),
                         'regressions': lost}); print('5', e, sid, len(sk), episodes[-1]['solved'], 'regress', len(lost), flush=True)
    offline = {k: v for k, v in mem.current('procedural').items()}
    rows_off = score(Solver(offline, cs, budget=a.budget), tasks)
    R['continual'] = {'episodes': episodes, 'recovered_from_memory_without_network': sum(r['outcome'] == 'correct' for r in rows_off),
                      'final_live': episodes[-1]['solved']}
    print('5 offline recovery', R['continual']['recovered_from_memory_without_network'], 'vs', episodes[-1]['solved'], flush=True)
    R['seconds'] = time.monotonic() - t0
    (a.output / 'learn-results.json').write_text(json.dumps(R, indent=1, default=str))


if __name__ == '__main__':
    main()
