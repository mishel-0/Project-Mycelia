"""One evolution cycle: evaluate -> diagnose -> research -> propose -> test ->
reject regressions -> verify -> request release (never promote)."""
from __future__ import annotations
import time
from .gateway import Gateway
from .memory import LabMemory
from .registry import Registry
from .specialists import Diagnostician, Engineer, Evaluator, Researcher, Verifier, judge


def run_cycle(gate, lab_root, benchmark_root, documents=(), opener=None, log=print):
    policy = gate.policy()  # raises if the policy was tampered with
    memory = LabMemory(lab_root / 'memory'); registry = Registry(lab_root / 'registry')
    record = {'started': time.time(), 'steps': []}
    def step(name, **data):
        record['steps'].append({'step': name, **data}); log(f'[{name}] ' + ', '.join(f'{k}={v}' for k, v in data.items() if not isinstance(v, (dict, list))))
    if gate.stopped():
        step('halted', reason='STOP set'); return record
    gateway = Gateway(lab_root / 'gateway', allowlist=policy['network']['allowlist'], documents=documents,
                      opener=opener, release_fn=gate.request_release)
    evaluator = Evaluator(gateway, benchmark_root, policy['limits'])
    prod = gate.production(); base = evaluator.run({'id': prod['id'], 'config': prod['config']})
    step('evaluate_production', id=prod['id'], summary=base['summary'])
    weakness = Diagnostician().run({'scorecard': base})
    if weakness is None:
        step('no_action', reason='no class recall separated from the others')
        memory.remember('episodic', record, source='loop'); return record
    step('weakness', label=weakness['label'], recall=round(weakness['recall'], 4), detail=weakness)
    research = Researcher(gateway, memory).run({'weakness': weakness, 'documents': [d.name for d in documents]})
    step('research', citations=len(research['citations']), detail=research)
    n = policy['limits']['max_candidates_per_cycle']
    proposals = Engineer(memory).run({'weakness': weakness, 'parent_config': prod['config'], 'n': n})
    tested = []
    for p in proposals:
        if gate.stopped():
            step('halted', reason='STOP set mid-cycle'); break
        cid = registry.propose(prod['id'], p['config'], p['hypothesis'], f"raise {weakness['label']} recall",
                               p['mutation'], policy['limits'], 'scorecard on selection tier, seeds 0-2, vs production')
        sc = evaluator.run({'id': cid, 'config': p['config']})['summary']
        ok, gain, reasons = judge(sc, base['summary'], weakness['label'], policy['acceptance'])
        registry.update(cid, 'tested' if ok else 'rejected', gain=gain, reasons=reasons, scorecard=sc)
        if not ok:
            memory.remember('failure', {'candidate': cid, 'mutation': p['mutation'], 'reasons': reasons}, source='evaluator')
            memory.reinforce(p['mutation'], False)
        tested.append({'id': cid, 'mutation': p['mutation'], 'accepted': ok, 'gain': gain, 'reasons': reasons, 'scorecard': sc})
        step('candidate', id=cid, mutation=p['mutation'], accepted=ok, gain=round(gain, 4), reasons=reasons)
    winners = sorted((t for t in tested if t['accepted']), key=lambda t: -t['gain'])
    if not winners:
        step('no_winner', reason='every candidate failed the predeclared rule')
    for w in winners[:1]:
        cand = registry.get(w['id'])
        v = Verifier(evaluator).run({'id': w['id'], 'config': cand['config'], 'parent_config': prod['config'],
                                     'weak': weakness['label'], 'acceptance': policy['acceptance']})
        memory.reinforce(w['mutation'], v['replicated'])
        step('verify', id=w['id'], replicated=v['replicated'], gain=round(v['gain'], 4), reasons=v['reasons'], detail=v)
        if not v['replicated']:
            registry.update(w['id'], 'rejected', reasons=['did not replicate'] + v['reasons']); break
        registry.update(w['id'], 'verified', verification=v)
        release = gateway.submit_candidate(w['id'])
        step('release_requested', id=w['id'], status=release['status'], holdout=release['holdout'],
             holdout_regressions=release['holdout_regressions'])
    record['candidates'] = tested; record['finished'] = time.time()
    memory.remember('episodic', {k: v for k, v in record.items() if k != 'steps'} | {'steps': [s['step'] for s in record['steps']]},
                    source='loop')
    return record
