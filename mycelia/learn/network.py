"""Symbiotic nodes and the network-necessity comparison.

Each node reads its own sources and verifies its own skills. Conditions:
  A isolated   - a node uses only its own skills
  B repository - every node uses a shared library of all published skills
  C fixed      - a node uses its own + its two ring neighbours' skills
  D adaptive   - a node asks peers chosen from coordination memory (which peers'
                 skills actually helped on which goal words), takes their most
                 relevant skills, re-verifies them on receipt, and learns from
                 every exchange (helped / unused / failed reproduction).
All conditions get the same solver budget per task.
"""
from __future__ import annotations
import re
from collections import defaultdict
from mycelia.pure.bus import Bus, Message
from .evidence import Lab, executable
from .solver import Solver, stem, match, WORD


def merge(*libraries):
    out = {}
    for lib in libraries:
        for op, s in lib.items():
            if op in out:
                out[op] = dict(out[op], words=sorted(set(out[op]['words']) | set(s['words'])),
                               sources=sorted(set(out[op]['sources']) | set(s['sources'])))
            else:
                out[op] = s
    return out


class Node:
    def __init__(self, name, skills, constants):
        self.name, self.skills, self.constants = name, dict(skills), constants
        self.received = {}  # verified skills kept from earlier exchanges (procedural memory)

    def offer(self, goal_words, k):
        """Most relevant skills for these goal words (what this node is willing to send)."""
        def rel(s):
            return sum(any(match(g, w) for w in s['words'] + WORD.findall(s['op'].split('.')[-1])) for g in goal_words)
        ranked = sorted(self.skills.values(), key=lambda s: (-rel(s), s['op']))
        return [s for s in ranked[:k] if rel(s) > 0] or ranked[:2]


def reverify(skill):
    """Receiver re-runs the sender's skill on its own probes; fakes and non-runnable ops fail."""
    if not executable(skill['op']):
        return False
    rec = Lab().test(skill['op'], [], skill.get('params', []))
    return bool(rec['probes'])


class Network:
    def __init__(self, nodes, memory=None, offer_k=12):
        self.nodes = {n.name: n for n in nodes}; self.memory = memory; self.offer_k = offer_k
        self.route = defaultdict(float)  # (goal word stem, sender) -> learned usefulness
        self.bus = Bus(); self.stats = defaultdict(int)

    def library(self, receiver, condition, goal_words, failed=()):
        node = self.nodes[receiver]; alive = {k: v for k, v in self.nodes.items() if k not in failed}
        if condition == 'A':
            return dict(node.skills)
        if condition == 'B':
            return merge(*[v.skills for v in self.nodes.values()])  # repository keeps copies even if a node fails
        if condition == 'C':
            names = list(self.nodes); i = names.index(receiver)
            nbrs = {names[(i - 1) % len(names)], names[(i + 1) % len(names)]} - set(failed)
            self.stats['C_skills_transferred'] += sum(len(self.nodes[n].skills) for n in nbrs)
            return merge(node.skills, *[self.nodes[n].skills for n in nbrs])
        # D: adaptive symbiosis
        peers = [p for p in alive if p != receiver]
        penalty = {p: self.route[('*', p)] for p in peers}  # failed reproductions on receipt
        scored = {p: sum(self.route[(stem(g), p)] for g in goal_words) + penalty[p] for p in peers}
        best = sorted((p for p in peers if scored[p] > 0), key=lambda p: -scored[p])[:2]
        chosen = best or [p for p in peers if penalty[p] > -2]  # exploit learned routes, else explore trusted peers
        lib = merge(node.skills, node.received)
        for p in chosen:
            self.stats['D_requests'] += 1
            for s in self.nodes[p].offer(goal_words, self.offer_k):
                self.stats['D_skills_transferred'] += 1
                ok = reverify(s)
                self.bus.post(Message(p, 'verified' if ok else 'counterexample', {'skill': s['op']}, [], 1.))
                if ok:
                    lib = merge(lib, {s['op']: dict(s, origin=p)})
                else:
                    self.stats['D_rejected_on_receipt'] += 1; self.route[('*', p)] -= 1
                    self.log(p, receiver, s['op'], reproduced=False, helped=False, goal=goal_words)
        return lib

    def log(self, sender, receiver, op, **kw):
        if self.memory is not None:
            self.memory.log_exchange({'sender': sender, 'receiver': receiver, 'skill': op, **kw})

    def learn(self, receiver, goal_words, library, solution):
        """Coordination memory: credit senders whose skills were used; debit unused offers."""
        used = set(solution.get('ops', [])) if solution.get('program') else set()
        for op, s in library.items():
            origin = s.get('origin')
            if origin and origin != receiver:
                helped = op in used
                for g in goal_words:
                    self.route[(stem(g), origin)] += 1. if helped else -.05
                if helped:
                    self.nodes[receiver].received[op] = s  # keep a verified copy
                self.log(origin, receiver, op, reproduced=True, helped=helped, goal=goal_words)

    def solve(self, receiver, task, condition, budget, failed=(), learn=False):
        goal = [w for w in WORD.findall(task['goal'].lower()) if len(w) > 2]
        lib = self.library(receiver, condition, goal, failed)
        sol = Solver(lib, self.nodes[receiver].constants, budget=budget).solve(task)
        if condition == 'D' and learn:
            self.learn(receiver, goal, lib, sol)
        return sol, lib
