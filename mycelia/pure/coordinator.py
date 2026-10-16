"""The symbiotic network and the baselines it must beat at equal compute."""
from __future__ import annotations
import itertools, random
from .agents import Observer, Lexicon, RuleInducer, Critic, execute
from .bus import Bus, Message
from .dsl import UNARY, BINARY, Budget, BudgetExhausted, Priors, describe
from .memory import Memory


def empty():
    return {'prims': {}, 'unary': {}, 'binary': {}}


class Model:
    def __init__(self, K, info=None):
        self.K, self.info = K, info or {}

    def predict(self, phrase):
        return execute(self.K, phrase)


class SymbioticNetwork:
    """Lexicon, inducer and critic exchange verified discoveries until nothing new is learned.

    share=False is the one-pass ablation: the lexicon's directly observed words reach
    the inducer once, but no discovery flows back (no inversion, no further rounds)."""
    def __init__(self, priors=None, critic=True, share=True, memory=None):
        self.priors = priors or Priors(); self.critic_on, self.share = critic, share
        self.memory = memory or Memory(); self.trust = {'lexicon': 0, 'inducer': 0}

    def learn(self, ep, budget_limit, episode_id=0, learn_priors=False):
        budget, bus, K = Budget(budget_limit), Bus(), empty()
        Observer().run(ep.demos, bus, self.memory, episode_id)
        lex, ind, critic = Lexicon(), RuleInducer(self.priors), Critic(tolerant=self.critic_on)
        suspects, rounds = [], 0
        try:
            while rounds < 10:
                rounds += 1; new = 0
                # Lexicon discoveries are reviewed first, so the inducer can use them in the same round.
                stages = [lambda: [(lex.name, h) for h in lex.propose(K, ep.demos, budget, use_operators=self.share)],
                          lambda: [(ind.name, h) for h in ind.propose(K, ep.demos, budget)]]
                for sender, h in (x for stage in stages for x in stage()):
                    slot, word, meaning, support = h[0], h[1], h[2], h[3]
                    bus.post(Message(sender, 'hypothesis', {'slot': slot, 'word': word, 'meaning': list(meaning)}, support, .5))
                    if self.critic_on:
                        status, agree, outliers = critic.review(slot, word, meaning, K, ep.demos, budget)
                    else:  # without a critic, only perfectly consistent proposals are accepted, unchecked
                        status, agree, outliers = ('verified' if not (len(h) > 4 and h[4]) else 'refuted'), support, []
                    if status == 'verified':
                        K[slot][word] = list(meaning) if slot == 'prims' else meaning; new += 1; self.trust[sender] += 1
                        self.memory.add('semantic', word, {'slot': slot, 'meaning': list(meaning)}, 'verified', sender, episode_id, agree)
                        bus.post(Message('critic', 'verified', {'slot': slot, 'word': word, 'meaning': list(meaning)}, agree, 1., 'verified'))
                        for d in outliers:
                            suspects.append(d)
                            self.memory.add('failure', ' '.join(ep.demos[d][0]), ep.demos[d][1], 'suspect', 'critic', episode_id,
                                            [f'contradicts verified meaning of {word}'])
                        if learn_priors and slot != 'prims':
                            self.priors.reinforce(meaning)
                    else:
                        self.trust[sender] -= 1
                        self.memory.add('failure', word, {'slot': slot, 'meaning': list(meaning)}, 'refuted', 'critic', episode_id, outliers)
                        bus.post(Message('critic', 'counterexample', {'slot': slot, 'word': word}, outliers, 1., 'refuted'))
                if not new or not self.share:
                    break
        except BudgetExhausted:
            pass
        return Model(K, {'evaluations': budget.used, 'rounds': rounds, 'suspects': sorted(set(suspects)),
                         'bus': bus.stats, 'symbols': len(bus.symbols), 'messages': len(bus.log)})

    def recall(self, episode_id):
        """Rebuild knowledge from verified memory alone: no demonstrations, no search."""
        K = empty()
        for f in self.memory.active('semantic', episode_id):
            v = f['value']; K[v['slot']][f['key']] = v['meaning'] if v['slot'] == 'prims' else tuple(v['meaning'])
        return Model(K, {'evaluations': 0})


class Generalist:
    """Single learner: one joint search over every unknown word's meaning."""
    def __init__(self, seed=None, priors=None):
        self.seed, self.priors = seed, priors or Priors()

    def learn(self, ep, budget_limit, **_):
        budget = Budget(budget_limit); K = empty()
        for p, o in ep.demos:
            if len(p) == 1:
                K['prims'][p[0]] = list(o)
        tokens = sorted({t for _, o in ep.demos for t in o}); slots = {}
        for p, _ in ep.demos:
            if len(p) == 2:
                slots.setdefault(p[0], 'prims'); slots[p[1]] = 'unary'
            if len(p) == 3:
                slots[p[1]] = 'binary'; slots.setdefault(p[0], 'prims'); slots.setdefault(p[2], 'prims')
        unknown = sorted(w for w, s in slots.items() if w not in K['prims'] or s != 'prims')
        unknown = [w for w in unknown if not (slots[w] == 'prims' and w in K['prims'])]
        options = {'prims': [list(c) for c in itertools.permutations(tokens, 2)],
                   'unary': self.priors.order(list(UNARY)), 'binary': self.priors.order(list(BINARY))}
        lists = []
        for w in unknown:
            opts = list(options[slots[w]])
            if self.seed is not None:
                random.Random(f'{self.seed}{w}').shuffle(opts)
            lists.append(opts)
        try:
            for combo in itertools.product(*lists):
                budget.spend(); trial = {k: dict(v) for k, v in K.items()}
                for w, m in zip(unknown, combo):
                    trial[slots[w]][w] = m
                if all(execute(trial, p) == list(o) for p, o in ep.demos):
                    return Model(trial, {'evaluations': budget.used, 'solved': True})
        except BudgetExhausted:
            pass
        return Model(K, {'evaluations': budget.used, 'solved': False})


class Vote:
    """Three generalists with different search orders, a third of the budget each; majority answer."""
    def learn(self, ep, budget_limit, **_):
        models = [Generalist(seed=s).learn(ep, budget_limit // 3) for s in range(3)]
        class Voted:
            info = {'evaluations': sum(m.info['evaluations'] for m in models)}
            def predict(self, phrase):
                answers = [tuple(a) for a in (m.predict(phrase) for m in models) if a is not None]
                return list(max(set(answers), key=answers.count)) if answers else None
        return Voted()


class Memorizer:
    def learn(self, ep, budget_limit, **_):
        table = {tuple(p): o for p, o in ep.demos}
        class Lookup:
            info = {'evaluations': 0}
            def predict(self, phrase):
                return table.get(tuple(phrase))
        return Lookup()


def explain(model, word):
    K = model.K
    if word in K['prims']:
        return f'"{word}" means the actions {" ".join(K["prims"][word])}'
    for slot in ('unary', 'binary'):
        if word in K[slot]:
            return f'"{word}" {describe(K[slot][word])}'
    return f'"{word}" is not known yet'
