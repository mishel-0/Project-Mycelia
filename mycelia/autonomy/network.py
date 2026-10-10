"""Sharing learned driving rules between cars under four conditions.

A isolated    each car keeps its own rules
B repository  every car uses every car's rules (static shared memory)
C fixed       each car uses its own rules + its two ring neighbours' rules
B_scope       ablation: shared repository, rules applied only inside their verified scope
A_scope       ablation: isolated cars, own rules applied only inside their verified scope
D adaptive    a received rule is adopted directly only if the receiver's own
              driving contexts lie inside the rule's verified scope; otherwise
              the receiver re-verifies it on its own scenarios (from the same
              episode budget) and adopts it only if it never collides and is no
              worse than the reflex. At run time D applies rules only inside
              their verified scope. Every exchange is logged in coordination
              memory; a sender whose rules failed twice in a receiver's
              context is no longer re-verified for that receiver.
All cars get the same total episode budget.
"""
from __future__ import annotations
from mycelia.learn.memory import LearnMemory
from mycelia.pure.bus import Bus, Message
from .node import Car
from .scenarios import TRAIN_FAMILIES

CONTEXT = ('visibility', 'surface', 'left_free', 'right_free')


def origin(rule_id):
    return rule_id.split('-r')[0]


def covers(rule, contexts):
    """True if every context the receiver drives in lies inside the rule's verified scope."""
    for ctx in contexts:
        obs = dict(zip(CONTEXT, ctx))
        if any(f in obs and obs[f] not in vals for f, vals in rule['scope'].items()):
            return False
    return True


def build(condition, budget=300, seed=0, reserve=60, memory=None):
    memory = memory or LearnMemory(); bus = Bus(); stats = {'adopted_in_scope': 0, 'reverified_ok': 0, 'reverified_failed': 0,
                                                              'skipped_untrusted': 0, 'messages': 0}
    cars = [Car(n, f, budget - (reserve if condition == 'D' else 0), seed=seed, scope_check=condition in ('D', 'B_scope', 'A_scope'))
            for n, f in TRAIN_FAMILIES.items()]
    for c in cars:
        c.practice()
        for r in c.rules:
            bus.post(Message(c.name, 'verified', {'rule': r['id'], 'cond': r['cond'], 'action': r['action']}, [], 1., 'verified'))
    extras = {c.name: [] for c in cars}
    names = [c.name for c in cars]
    for i, c in enumerate(cars):
        if condition in ('B', 'B_scope'):
            extras[c.name] = [r for o in cars if o is not c for r in o.rules]
        elif condition == 'C':
            nb = {names[(i - 1) % 4], names[(i + 1) % 4]}
            extras[c.name] = [r for o in cars if o.name in nb for r in o.rules]
        elif condition == 'D':
            c.budget.limit += reserve; failures = {}
            for o in cars:
                if o is c:
                    continue
                for r in o.rules:
                    stats['messages'] += 1
                    if covers(r, c.contexts):
                        extras[c.name].append(r); stats['adopted_in_scope'] += 1
                        memory.log_exchange({'sender': o.name, 'receiver': c.name, 'skill': r['id'], 'decision': 'adopted_in_scope'}); continue
                    if failures.get(o.name, 0) >= 2:
                        stats['skipped_untrusted'] += 1
                        memory.log_exchange({'sender': o.name, 'receiver': c.name, 'skill': r['id'], 'decision': 'skipped_untrusted'}); continue
                    ev = c.test(dict(r, scope={}), c.batch)
                    ok = ev is not None and ev['collisions'] == 0 and ev['success'] >= ev['baseline_success'] and ev['ticks'] <= ev['baseline_ticks'] + .5
                    if ok:
                        scope = {f: sorted(set(v) | ev['seen'].get(f, set())) for f, v in r['scope'].items()}
                        extras[c.name].append(dict(r, scope=scope)); stats['reverified_ok'] += 1
                    else:
                        failures[o.name] = failures.get(o.name, 0) + 1; stats['reverified_failed'] += 1
                    memory.log_exchange({'sender': o.name, 'receiver': c.name, 'skill': r['id'],
                                         'decision': 'reverified_ok' if ok else 'reverified_failed', 'evidence': ev and {k: ev[k] for k in ('success', 'collisions', 'ticks', 'baseline_success', 'baseline_ticks')}})
    return cars, extras, stats, memory
