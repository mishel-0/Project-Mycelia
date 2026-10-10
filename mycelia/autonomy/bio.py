"""v0.3: biologically inspired mechanisms, each switchable for ablation.

rat       world model: (situation, action) -> outcome counts, updated after every
          drive; picks actions when no rule applies; failure analysis also
          experiments at earlier moments before the failure (back-chaining)
crow      procedures: generalisation may drop the obstacle type itself, so one
          rule covers several obstacles where tests show it still holds
octopus   local veto: a car ignores any rule whose action its own experience
          shows crashing in this situation
bee       shared rules are advertised by evidence, used on probation, trusted
          network-wide after a quorum of 2 independent confirmations, and
          withdrawn everywhere after a crash
physarum  rules and car-to-car links carry a strength: reinforced by useful
          outcomes, weakened by failures, decaying each round, pruned when
          weak; message slots follow link strength
These are engineering mechanisms inspired by biology, not models of it.
"""
from __future__ import annotations
import json
from mycelia.learn.memory import LearnMemory
from mycelia.pure.bus import Bus, Message
from .network import origin
from .network_v2 import liar_rules, oracle_unsafe
from .node import Car, FEATURES, GENERALISE, matches
from .scenarios import TRAIN_V2, SHIFTED_FAMILIES
from .sim import ACTIONS, simulate, reflex

ALL = frozenset({'rat', 'crow', 'octopus', 'bee', 'physarum'})


def situation(obs):
    return tuple(obs[f] for f in FEATURES)


class BioCar(Car):
    def __init__(self, *a, mech=ALL, **kw):
        super().__init__(*a, **kw); self.mech = set(mech); self.model = {}

    # -- rat: learn from every encounter ------------------------------------
    def record(self, trace, outcome):
        for i, (obs, action, _) in enumerate(trace):
            late = i >= len(trace) - 4
            key = (situation(obs), action); n, col, ok = self.model.get(key, (0, 0, 0))
            self.model[key] = (n + 1, col + (late and outcome == 'collision'), ok + (outcome == 'success'))

    def stats(self, obs, action):
        n, col, ok = self.model.get((situation(obs), action), (0, 0, 0))
        return n, (col / n if n else 0.), (ok / n if n else 0.)

    def vetoed(self, obs, action):  # octopus
        n, col, _ = self.stats(obs, action); return 'octopus' in self.mech and n >= 3 and col > .34

    def model_choice(self, obs):
        base = reflex(obs); bn, bcol, bok = self.stats(obs, base)
        options = [(a,) + self.stats(obs, a) for a in ACTIONS]
        good = [o for o in options if o[1] >= 3]
        if not good or (bn >= 3 and bcol == 0 and bok >= .8):
            return base
        best = min(good, key=lambda o: (o[2], -o[3]))
        return best[0] if (best[2] < bcol or (best[2] == bcol and best[3] > bok)) and bn >= 1 else base

    def policy(self, extra=(), scope_check=None):
        check = self.scope_check if scope_check is None else scope_check
        rules = sorted(list(self.rules) + list(extra), key=lambda r: (-len(r['cond']), -r.get('g', 1.)))

        def act(obs):
            for r in rules:
                if matches(r, obs, check) and not self.vetoed(obs, r['action']):
                    return r['action'], r['id']
            if 'rat' in self.mech:
                return self.model_choice(obs), None
            return reflex(obs), None
        return act

    # -- practice: like Car.practice, plus back-chaining (rat) and abstraction (crow)
    def practice(self):
        generalise = (('ahead',) if 'crow' in self.mech else ()) + GENERALISE
        while self.budget.take(1):
            sc = self.scenario(); res = simulate(sc, self.policy(scope_check=False), trace=True, brake=self.brake)
            for obs, _, _ in res['trace']:
                self.contexts.add(tuple(obs[f] for f in ('visibility', 'surface', 'left_free', 'right_free')))
            if 'rat' in self.mech:
                self.record(res['trace'], res['outcome'])
            if res['outcome'] == 'collision':
                for rid in set(res['fired']):
                    self.rules = [r for r in self.rules if r['id'] != rid]
            if res['outcome'] == 'success' and res['idle'] <= 4:
                continue
            first = next((i for i, (o, a, _) in enumerate(res['trace']) if a == 'stop' and o['distance'] == 'near'), len(res['trace']) - 1)
            moments = [first] + ([first - k for k in (1, 2, 3) if first - k >= 0] if 'rat' in self.mech else [])
            for m in moments:
                trigger = res['trace'][m][0]
                if trigger['ahead'] == 'none':
                    continue
                cond = {f: trigger[f] for f in FEATURES}
                if any(all(r['cond'].get(f) == cond[f] for f in r['cond']) for r in self.rules + self.failures):
                    break
                learned = self._experiment(cond, trigger, generalise)
                if learned is None:
                    return
                if learned:
                    break

    def _experiment(self, cond, trigger, generalise):
        best = None
        for action in ACTIONS:
            if action == reflex(trigger):
                continue
            rule = {'id': f'{self.name}-r{next(Car._ids)}', 'cond': dict(cond), 'action': action, 'scope': {}, 'origin': self.name, 'g': 1.}
            ev = self.test(rule, self.batch)
            if ev is None:
                return None
            better = ev['success'] > ev['baseline_success'] or (ev['success'] == ev['baseline_success'] and ev['ticks'] <= ev['baseline_ticks'] - 1)
            if ev['collisions'] == 0 and better and (best is None or (ev['success'], -ev['ticks']) > (best[1]['success'], -best[1]['ticks'])):
                best = (rule, ev)
        if best is None:
            self.failures.append({'cond': cond, 'status': 'no_better_action'}); return False
        rule, ev = best
        for f in generalise:
            if f not in rule['cond']:
                continue
            general = dict(rule, cond={k: v for k, v in rule['cond'].items() if k != f})
            g = self.test(general, self.batch)
            if g is None:
                break
            if g['collisions'] == 0 and g['success'] >= ev['success'] and g['ticks'] <= ev['ticks'] + 1 and g['fired']:
                scope = dict(rule['scope']); scope[f] = sorted(g['seen'].get(f, {rule['cond'][f]}) | {rule['cond'][f]})
                rule, ev = dict(general, scope=scope), g
        rule.update(status='verified_in_scope', evidence={'tests': ev['n'], 'success': ev['success'], 'collisions': ev['collisions']})
        self.rules.append(rule); return True


def run(mech=ALL, seed=0, budget=400, rounds=4, ops=20, k=4, batch=6):
    mech = set(mech)
    cars = [BioCar(n, f, budget, batch=batch, seed=seed, brake=b, scope_check=True, mech=mech) for n, (f, b) in TRAIN_V2.items()]
    names = [c.name for c in cars]; held = {c.name: {} for c in cars}; blocked = {c.name: set() for c in cars}
    link = {(a, b): 1. for a in names + ['car5'] for b in names if a != b}
    confirmations, trusted = {}, set(); memory = LearnMemory(); bus = Bus()
    stats = {'messages': 0, 'bytes': 0, 'probation_adopted': 0, 'promoted_by_quorum': 0, 'withdrawn': 0, 'pruned': 0,
             'operational_collisions_by_round': []}
    per_round = budget // rounds

    def withdraw(rid):
        stats['withdrawn'] += 1; bus.post(Message('network', 'counterexample', {'rule': rid, 'retract': True}, [], 1., 'refuted'))
        for c in cars:
            held[c.name].pop(rid, None); blocked[c.name].add(rid); c.rules = [r for r in c.rules if r['id'] != rid]

    for rnd in range(1, rounds + 1):
        if rnd == 3:
            for c in cars:
                c.shifted = c.family in SHIFTED_FAMILIES
        for c in cars:
            c.budget.limit = per_round * rnd; c.practice()
        # sharing: advertise by evidence (x strength); receivers' slots follow link strength (physarum)
        for c in cars:
            pool = [(r, o.name) for o in cars if o is not c for r in o.rules] + [(r, 'car5') for r in liar_rules()]
            pool = [(r, s) for r, s in pool if r['id'] not in held[c.name] and r['id'] not in blocked[c.name]]
            def intensity(x):
                r, s = x; ev = r.get('evidence', {}); q = (ev.get('success', 0) + 1) / (ev.get('tests', 0) + 2)
                return q * (r.get('g', 1.) if 'physarum' in mech else 1.) * (link[(s, c.name)] if 'physarum' in mech else 1.)
            for r, s in sorted(pool, key=lambda x: -intensity(x))[:k]:
                stats['messages'] += 1; stats['bytes'] += len(json.dumps(r))
                r = dict(r, probation='bee' in mech and r['id'] not in trusted)
                held[c.name][r['id']] = r; stats['probation_adopted'] += r['probation']
                memory.log_exchange({'sender': s, 'receiver': c.name, 'skill': r['id'], 'round': rnd, 'probation': r['probation']})
        collisions = 0
        for c in cars:
            for _ in range(ops):
                res = simulate(c.scenario(), c.policy(list(held[c.name].values())), trace=True, brake=c.brake)
                collisions += res['outcome'] == 'collision'
                if 'rat' in mech:
                    c.record(res['trace'], res['outcome'])
                for rid in set(res['fired']):
                    sender = origin(rid); r = held[c.name].get(rid) or next((x for x in c.rules if x['id'] == rid), None)
                    if r is None:
                        continue
                    if 'physarum' in mech:
                        r['g'] = min(5., r.get('g', 1.) + .5) if res['outcome'] == 'success' else r.get('g', 1.) * (.2 if res['outcome'] == 'collision' else .9)
                        if sender != c.name:
                            link[(sender, c.name)] = link[(sender, c.name)] + .2 if res['outcome'] == 'success' else link[(sender, c.name)] * .5
                    if 'bee' in mech and sender != c.name:
                        if res['outcome'] == 'collision':
                            withdraw(rid)
                        elif res['outcome'] == 'success':
                            confirmations.setdefault(rid, set()).add(c.name)
                            if len(confirmations[rid]) >= 2 and rid not in trusted:
                                trusted.add(rid); stats['promoted_by_quorum'] += 1
                                for o in cars:
                                    if rid not in held[o.name] and rid not in blocked[o.name] and o.name != sender:
                                        held[o.name][rid] = dict(r, probation=False)
        stats['operational_collisions_by_round'].append(collisions)
        if 'physarum' in mech:  # decay and prune weak pathways
            for c in cars:
                for r in list(c.rules) + list(held[c.name].values()):
                    r['g'] = r.get('g', 1.) * .85
                for r in [r for r in c.rules if r['g'] < .3]:
                    c.rules.remove(r); stats['pruned'] += 1
                for rid in [rid for rid, r in held[c.name].items() if r['g'] < .3]:
                    held[c.name].pop(rid); blocked[c.name].add(rid); stats['pruned'] += 1
            for key in link:
                link[key] *= .9
    stats['liar_rules_held_end'] = sum(rid.startswith('car5') for c in cars for rid in held[c.name])
    stats['false_rule_acceptance'] = sum(oracle_unsafe(r, c) for c in cars for r in held[c.name].values())
    stats['own_rules_end'] = {c.name: len(c.rules) for c in cars}; stats['model_entries'] = {c.name: len(c.model) for c in cars}
    return cars, {c.name: list(held[c.name].values()) for c in cars}, stats, memory
