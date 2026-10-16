"""v0.2 protocol: rounds of practice -> sharing -> operational driving, with a
hidden shift after round 2, a misleading car, hidden brake differences and
counted communication. Mechanisms are switched on individually:

  M1 re-verification  receiver tests a rule on its own scenarios with its own
                      brakes before adopting it (spends its episode budget)
  M2 source trust     Beta trust per sender from re-verifications and runtime
                      outcomes; reject low-trust senders without spending
                      budget, skip verification for well-established ones
  M3 monitoring       a rule that fired in a collision is suspended and a
                      retraction is broadcast to every car holding it
  M4 value sharing    at most k rule messages per receiver per round, chosen
                      by evidence x sender trust x fit to the receiver's contexts
"""
from __future__ import annotations
import json
from mycelia.learn.memory import LearnMemory
from mycelia.pure.bus import Bus, Message
from .network import covers, origin
from .node import Car
from .scenarios import TRAIN_V2, SHIFTED_FAMILIES
from .sim import simulate
from .trust import Trust

ALL_VALUES = {'visibility': ['clear', 'fog'], 'surface': ['dry', 'wet'], 'left_free': ['yes', 'no', 'unknown', 'none'],
              'right_free': ['yes', 'no', 'unknown', 'none'], 'distance': ['near', 'mid', 'far', 'none']}
CONDITIONS = {'A': None, 'B': None, 'B_scope': set(), 'M1': {'M1'}, 'M2': {'M2'}, 'M3': {'M3'}, 'M4': {'M4'},
              'D_full': {'M1', 'M2', 'M3', 'M4'}}


def liar_rules():
    """car5: corrupted rules claiming universal scope and perfect evidence."""
    fake = {'tests': 50, 'success': 50, 'collisions': 0}
    def r(i, cond, action, scope=None):
        return {'id': f'car5-r{i}', 'cond': cond, 'action': action, 'origin': 'car5', 'status': 'verified_in_scope', 'evidence': fake,
                'scope': scope or {k: v for k, v in ALL_VALUES.items() if k not in cond}}
    return [r(1, {'ahead': 'pedestrian', 'distance': 'near'}, 'keep'),     # deliberate: drive through pedestrians
            r(2, {'ahead': 'block', 'distance': 'near'}, 'keep'),          # deliberate: drive into blocks
            r(3, {'ahead': 'block', 'left_free': 'unknown'}, 'left'),      # faulty: blind lane change in fog
            r(4, {'ahead': 'vehicle', 'distance': 'mid'}, 'keep')]          # faulty, mostly harmless


def value(rule, sender, receiver, trust):
    ev = rule.get('evidence', {}); rate = (ev.get('success', 0) + 1) / (ev.get('tests', 0) + 2)
    fit = 1. if covers(rule, receiver.contexts) else .3
    return rate * trust.mean(sender) * fit


def oracle_unsafe(rule, car, n=40):
    """Held-out check (not available to the cars): is this rule unsafe for this
    receiver, with its own brakes, in its own post-shift world?"""
    from .scenarios import make
    scs = [make(car.family, 900000 + i, shifted=car.shifted) for i in range(n)]
    with_rule = sum(simulate(sc, car.policy([dict(rule, scope={})], False), brake=car.brake)['outcome'] == 'collision' for sc in scs)
    base = sum(simulate(sc, car.policy([], False), brake=car.brake)['outcome'] == 'collision' for sc in scs)
    return with_rule > base


def run(condition, seed=0, budget=400, rounds=4, ops=20, k=3, batch=6, reserve=36):
    """`reserve`: episodes per round that re-verifying cars (M1) keep for checking
    received rules; other cars spend them on practice. Total budget is equal."""
    mech = CONDITIONS[condition]; scope = mech is not None
    cars = [Car(n, f, budget, batch=batch, seed=seed, brake=b, scope_check=scope) for n, (f, b) in TRAIN_V2.items()]
    held = {c.name: {} for c in cars}; rejected = {c.name: set() for c in cars}
    trust = {c.name: Trust() for c in cars}; memory = LearnMemory(); bus = Bus()
    stats = {'messages': 0, 'bytes': 0, 'adopted': 0, 'rejected_by_verification': 0, 'rejected_by_trust': 0,
             'verifications': 0, 'retractions': 0, 'operational_collisions_by_round': [], 'liar_rules_held_end': 0}
    per_round = budget // rounds
    for rnd in range(1, rounds + 1):
        if rnd == 3:  # hidden shift: nothing in the observations changes
            for c in cars:
                c.shifted = c.family in SHIFTED_FAMILIES
        verifying = bool(mech and 'M1' in mech)
        for c in cars:
            c.budget.limit = per_round * rnd - (reserve if verifying else 0); c.practice()
            c.budget.limit = per_round * rnd
        if condition != 'A':
            for c in cars:
                pool = [(r, o.name) for o in cars if o is not c for r in o.rules] + [(r, 'car5') for r in liar_rules()]
                pool = [(r, s) for r, s in pool if r['id'] not in held[c.name] and r['id'] not in rejected[c.name]]
                if mech and 'M4' in mech:
                    pool = sorted(pool, key=lambda x: -value(x[0], x[1], c, trust[c.name]))[:k]
                for r, sender in pool:
                    stats['messages'] += 1; stats['bytes'] += len(json.dumps(r))
                    bus.post(Message(sender, 'hypothesis', {'rule': r['id']}, [], .5))
                    decision = 'adopted'
                    if mech and 'M2' in mech and trust[c.name].reject(sender):
                        decision = 'rejected_by_trust'
                    elif mech and 'M1' in mech and not ('M2' in mech and trust[c.name].skip_verify(sender)):
                        ev = c.test(dict(r, scope={}), c.batch)
                        if ev is None:
                            continue  # no budget left this round; may be offered again later
                        stats['verifications'] += 1
                        if not ev['fired']:  # never exercised in this car's world: no evidence either way
                            rejected[c.name].add(r['id']); stats['irrelevant_here'] = stats.get('irrelevant_here', 0) + 1
                            memory.log_exchange({'sender': sender, 'receiver': c.name, 'skill': r['id'], 'round': rnd, 'decision': 'irrelevant_here'})
                            continue
                        ok = ev['collisions'] == 0 and ev['success'] >= ev['baseline_success'] and ev['ticks'] <= ev['baseline_ticks'] + .5
                        trust[c.name].update(sender, ok)
                        if ok:
                            r = dict(r, scope={f: sorted(set(v) | ev['seen'].get(f, set())) for f, v in r['scope'].items()})
                        else:
                            decision = 'rejected_by_verification'
                    if decision == 'adopted':
                        held[c.name][r['id']] = (r, sender); stats['adopted'] += 1
                    else:
                        rejected[c.name].add(r['id']); stats[decision] += 1
                    memory.log_exchange({'sender': sender, 'receiver': c.name, 'skill': r['id'], 'round': rnd, 'decision': decision})
        collisions = 0
        for c in cars:  # operational driving in each car's current (possibly shifted) world
            for _ in range(ops):
                res = simulate(c.scenario(), c.policy([r for r, _ in held[c.name].values()]), brake=c.brake)
                collisions += res['outcome'] == 'collision'
                for rid in set(res['fired']):
                    sender = origin(rid)
                    if sender != c.name and mech and 'M2' in mech:
                        trust[c.name].update(sender, res['outcome'] != 'collision')
                    if res['outcome'] == 'collision' and mech and 'M3' in mech:
                        stats['retractions'] += 1
                        bus.post(Message(c.name, 'counterexample', {'rule': rid, 'retract': True}, [], 1., 'refuted'))
                        for o in cars:
                            held[o.name].pop(rid, None); rejected[o.name].add(rid)
                            o.rules = [x for x in o.rules if x['id'] != rid]
        stats['operational_collisions_by_round'].append(collisions)
    stats['liar_rules_held_end'] = sum(rid.startswith('car5') for c in cars for rid in held[c.name])
    stats['false_rule_acceptance'] = sum(oracle_unsafe(r, c) for c in cars for r, _ in held[c.name].values())
    stats['foreign_rules_held_end'] = sum(len(h) for h in held.values())
    stats['trust'] = {c.name: {s: round(trust[c.name].mean(s), 3) for s in trust[c.name].counts} for c in cars}
    return cars, {c.name: [r for r, _ in held[c.name].values()] for c in cars}, stats, memory
