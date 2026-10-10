"""A car node that learns condition -> action rules by experiment.

Practice loop (own scenario family, fixed episode budget):
  1. drive; if the episode fails or wastes time, take the first observation
     where the reflex had to stop (or the last one before a collision);
  2. experiment: try each action as a rule for exactly that situation on a
     batch of scenario variations, against the reflex on the same batch;
  3. keep the best action only if it never collides and beats the reflex;
  4. generalise by dropping one condition at a time, re-testing on fresh
     variations; a dropped feature's *scope* is the set of values it actually
     took when the rule was tested (so the rule is verified only there);
  5. rules that cause a collision later are refuted (kept as failures).
"""
from __future__ import annotations
import itertools
from .scenarios import make
from .sim import ACTIONS, simulate, reflex

FEATURES = ('ahead', 'distance', 'left_free', 'right_free', 'visibility', 'surface')
GENERALISE = ('surface', 'visibility', 'right_free', 'left_free', 'distance')


def matches(rule, obs, scope_check):
    if any(obs[f] != v for f, v in rule['cond'].items()):
        return False
    return not scope_check or all(obs[f] in vals for f, vals in rule['scope'].items())


class Budget:
    def __init__(self, limit):
        self.limit, self.used = limit, 0

    def take(self, n=1):
        if self.used + n > self.limit:
            return False
        self.used += n; return True


class Car:
    _ids = itertools.count()

    def __init__(self, name, family, budget=300, batch=6, scope_check=False, seed=0):
        self.name, self.family, self.batch, self.scope_check, self.seed = name, family, batch, scope_check, seed
        self.budget = Budget(budget); self.rules, self.failures, self.contexts = [], [], set()
        self._seed = 0

    def next_seed(self):
        self._seed += 1; return 7919 * self.seed + self._seed

    def policy(self, extra=(), scope_check=None):
        check = self.scope_check if scope_check is None else scope_check
        rules = sorted(list(self.rules) + list(extra), key=lambda r: -len(r['cond']))

        def act(obs):
            for r in rules:
                if matches(r, obs, check):
                    return r['action'], r['id']
            return reflex(obs), None
        return act

    def test(self, rule, n, base=None):
        """Evaluate a candidate rule on n fresh variations from the car's own family."""
        if not self.budget.take(n * (2 if base is None else 1)):
            return None
        scs = [make(self.family, self.next_seed()) for _ in range(n)]
        with_rule = [simulate(sc, self.policy([rule], False), trace=True) for sc in scs]
        baseline = [simulate(sc, self.policy([], False)) for sc in scs] if base is None else base
        seen = {}
        for res in with_rule:
            for obs, _, rid in res['trace']:
                if rid == rule['id']:
                    for f in FEATURES:
                        seen.setdefault(f, set()).add(obs[f])
        ok = sum(r['outcome'] == 'success' for r in with_rule); col = sum(r['outcome'] == 'collision' for r in with_rule)
        ticks = lambda rs: sum(r['ticks'] for r in rs) / len(rs)
        return {'success': ok, 'collisions': col, 'ticks': ticks(with_rule), 'baseline_ticks': ticks(baseline),
                'baseline_success': sum(r['outcome'] == 'success' for r in baseline),
                'baseline_collisions': sum(r['outcome'] == 'collision' for r in baseline), 'seen': seen, 'n': n,
                'fired': sum(bool(r['fired']) for r in with_rule)}

    def practice(self):
        while self.budget.take(1):
            sc = make(self.family, self.next_seed()); res = simulate(sc, self.policy(scope_check=False), trace=True)
            for obs, _, _ in res['trace']:
                self.contexts.add(tuple(obs[f] for f in ('visibility', 'surface', 'left_free', 'right_free')))
            if res['outcome'] == 'collision':
                for rid in set(res['fired']):
                    for r in list(self.rules):
                        if r['id'] == rid:
                            self.rules.remove(r); self.failures.append(dict(r, status='refuted_in_scope', counterexample=sc.id))
            if res['outcome'] == 'success' and res['idle'] <= 4:
                continue
            trigger = next((o for o, a, _ in res['trace'] if a == 'stop' and o['distance'] == 'near'), res['trace'][-1][0])
            cond = {f: trigger[f] for f in FEATURES}
            if any(all(r['cond'].get(f) == cond[f] for f in r['cond']) for r in self.rules + self.failures):
                continue  # already learned (or already failed) here
            best = None
            for action in ACTIONS:
                if action == reflex(trigger):
                    continue
                rule = {'id': f'{self.name}-r{next(Car._ids)}', 'cond': dict(cond), 'action': action, 'scope': {}, 'origin': self.name}
                ev = self.test(rule, self.batch)
                if ev is None:
                    return
                better = ev['success'] > ev['baseline_success'] or (ev['success'] == ev['baseline_success'] and ev['ticks'] <= ev['baseline_ticks'] - 1)
                if ev['collisions'] == 0 and better and (best is None or (ev['success'], -ev['ticks']) > (best[1]['success'], -best[1]['ticks'])):
                    best = (rule, ev)
            if best is None:
                self.failures.append({'cond': cond, 'status': 'no_better_action'}); continue
            rule, ev = best
            for f in GENERALISE:
                if f not in rule['cond']:
                    continue
                general = dict(rule, cond={k: v for k, v in rule['cond'].items() if k != f})
                g = self.test(general, self.batch)
                if g is None:
                    break
                if g['collisions'] == 0 and g['success'] >= ev['success'] and g['ticks'] <= ev['ticks'] + 1 and g['fired']:
                    scope = dict(rule['scope']); scope[f] = sorted(g['seen'].get(f, {rule['cond'][f]}) | {rule['cond'][f]})
                    rule, ev = dict(general, scope=scope), g
            rule.update(status='verified_in_scope', evidence={'tests': ev['n'], 'success': ev['success'], 'collisions': ev['collisions'],
                                                              'ticks': ev['ticks'], 'baseline_success': ev['baseline_success'],
                                                              'baseline_ticks': ev['baseline_ticks']})
            self.rules.append(rule)
