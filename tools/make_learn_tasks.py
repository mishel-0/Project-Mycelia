"""Generate and freeze the v0.3 task suites (dev + three larger eval sets).

Reference answers are computed by plain Python here, from reference programs
written in Mycelia's step language. Each task records the operations it needs
and which documentation file describes each one; cross-domain tasks need
operations from >= 2 files. Some templates appear only in eval sets (novel
combinations). Some tasks cannot be solved by composing the allowed
operations; the right answer is to abstain. Hashes are written to a manifest
before any solver work, so the suites cannot be tuned against.
"""
import hashlib, json, math, random, statistics
from pathlib import Path

DOC = {'str.': 'Doc/builtins/stdtypes.rst', 'math.': 'Doc/library/math.rst', 'statistics.': 'Doc/library/statistics.rst'}
BUILTIN_DOC = 'Doc/builtins/functions.rst'


def doc_of(op):
    return next((d for p, d in DOC.items() if op.startswith(p)), BUILTIN_DOC)


def ops_of(program):
    out = []
    for step in program:
        if step[0] == 'map':
            out += ops_of([step[1]])
        elif step[0] != 'join':
            out.append(step[1])
        else:
            out.append('str.join')
    return out


def apply(step, v):
    kind = step[0]
    if kind == 'method':
        return getattr(v, step[1][4:])(*step[2])
    if kind == 'call':
        mod, _, name = step[1].rpartition('.')
        fn = {'math': math, 'statistics': statistics}[mod].__dict__[name] if mod else __builtins__.__dict__[name] if hasattr(__builtins__, '__dict__') else __builtins__[name]
        return fn(v, *step[2])
    if kind == 'map':
        return [apply(step[1], e) for e in v]
    if kind == 'join':
        return step[1].join(v)
    raise ValueError(step)


W = ['alice', 'bob', 'mycelia', 'spore', 'hypha', 'river', 'stone', 'green', 'apple', 'night', 'fungus', 'root']


def T(goals, gen, program=None, fn=None, solvable=True, eval_only=False, family='text'):
    return dict(goals=goals, gen=gen, program=program, fn=fn, solvable=solvable, eval_only=eval_only, family=family)


def templates(rng):
    def sentence():
        ws = [rng.choice([w, w.upper(), w.title()]) for w in rng.sample(W, rng.randint(2, 5))]
        return rng.choice(['', ' ', '  ']) + rng.choice([' ', '  ']).join(ws) + rng.choice(['', ' ', '  '])
    def words():
        return ' '.join(rng.sample(W, rng.randint(3, 6)))
    def nums():
        return ','.join(str(rng.randint(1, 99)) for _ in range(rng.randint(3, 7)))
    def numlist():
        return [rng.randint(1, 50) for _ in range(rng.randint(3, 7))]
    def number():
        return rng.choice([rng.randint(1, 400), round(rng.uniform(1, 400), 2)])
    def smallint():
        return rng.randint(1, 10)
    m = lambda op, *a: ('method', op, list(a)); c = lambda op, *a: ('call', op, list(a)); mp = lambda s: ('map', s); j = lambda sep: ('join', sep)
    return [
        T(['make every letter uppercase', 'convert the text to capital letters'], sentence, [m('str.upper')]),
        T(['make every letter lowercase', 'convert the text to small letters'], sentence, [m('str.lower')]),
        T(['remove the whitespace at both ends', 'trim the surrounding spaces'], sentence, [m('str.strip')]),
        T(['swap the case of every letter'], sentence, [m('str.swapcase')]),
        T(['make each word start with a capital letter after trimming the ends'], sentence, [m('str.strip'), m('str.title')]),
        T(['replace the commas with semicolons'], nums, [m('str.replace', ',', ';')]),
        T(['split the comma separated numbers into a list of strings'], nums, [m('str.split', ',')]),
        T(['count how many commas there are'], nums, [m('str.count', ',')]),
        T(['how many characters does the text have', 'length of the text'], words, [c('len')], family='text+builtins'),
        T(['how many words are there', 'count the words'], words, [m('str.split'), c('len')], family='text+builtins'),
        T(['sort the words alphabetically', 'put the words in sorted order'], words, [m('str.split'), c('sorted')], family='text+builtins'),
        T(['length of the longest word', 'size of the biggest word'], words, [m('str.split'), mp(c('len')), c('max')], family='text+builtins'),
        T(['length of the shortest word'], words, [m('str.split'), mp(c('len')), c('min')], family='text+builtins', eval_only=True),
        T(['sort the words and join them with commas'], words, [m('str.split'), c('sorted'), j(',')], family='text+builtins'),
        T(['the largest number in the comma separated text'], nums, [m('str.split', ','), mp(c('int')), c('max')], family='text+builtins'),
        T(['add up the comma separated numbers', 'total of the comma separated numbers'], nums, [m('str.split', ','), mp(c('int')), c('sum')], family='text+builtins'),
        T(['square root of the number'], number, [c('math.sqrt')], family='math'),
        T(['round the number down to an integer', 'floor of the number'], number, [c('math.floor')], family='math'),
        T(['round the number up to an integer', 'ceiling of the number'], number, [c('math.ceil')], family='math'),
        T(['factorial of the number'], smallint, [c('math.factorial')], family='math'),
        T(['product of all the numbers in the list', 'multiply the numbers together'], numlist, [c('math.prod')], family='math'),
        T(['arithmetic mean of the numbers', 'average of the list'], numlist, [c('statistics.mean')], family='statistics'),
        T(['median of the numbers', 'middle value of the list'], numlist, [c('statistics.median')], family='statistics'),
        T(['population standard deviation of the numbers'], numlist, [c('statistics.pstdev')], family='statistics'),
        T(['median word length', 'middle value of the word lengths'], words, [m('str.split'), mp(c('len')), c('statistics.median')], family='cross'),
        T(['average of the comma separated numbers', 'mean of the numbers in the text'], nums, [m('str.split', ','), mp(c('int')), c('statistics.mean')], family='cross'),
        T(['square root of the number of words'], words, [m('str.split'), c('len'), c('math.sqrt')], family='cross', eval_only=True),
        T(['round the mean of the list down to an integer', 'floor of the average'], numlist, [c('statistics.mean'), c('math.floor')], family='cross', eval_only=True),
        T(['product of the comma separated numbers'], nums, [m('str.split', ','), mp(c('int')), c('math.prod')], family='cross', eval_only=True),
        T(['median of the comma separated numbers'], nums, [m('str.split', ','), mp(c('int')), c('statistics.median')], family='cross', eval_only=True),
        # Not composable from the allowed operations: abstain.
        T(['reverse the order of the characters'], sentence, fn=lambda s: s[::-1], solvable=False, family='unsolvable'),
        T(['keep only the first letter of each word'], words, fn=lambda s: ''.join(w[0] for w in s.split()), solvable=False, family='unsolvable'),
        T(['sort the words by their length'], words, fn=lambda s: sorted(s.split(), key=lambda w: (len(w), w)), solvable=False, family='unsolvable'),
        T(['count the vowels'], words, fn=lambda s: sum(ch in 'aeiou' for ch in s), solvable=False, family='unsolvable'),
    ]


def make(seed, n=None, dev=False):
    rng = random.Random(seed); tpls = templates(rng); out = []
    pool = [t for t in tpls if not (dev and t['eval_only'])]
    k = 0
    while len(out) < (n or len(pool) * 2):
        t = pool[k % len(pool)] if n is None else rng.choice(pool); k += 1
        xs = []
        while len(xs) < 9:
            x = t['gen']()
            if x not in xs:
                xs.append(x)
        f = (lambda v, p=t['program']: [v := apply(s, v) for s in p][-1]) if t['program'] else t['fn']
        ops = ops_of(t['program']) if t['program'] else []
        docs = sorted({doc_of(o) for o in ops})
        out.append({'id': f'{seed}-{len(out)}', 'goal': rng.choice(t['goals']), 'family': t['family'],
                    'examples': [[x, f(x)] for x in xs[:3]], 'hidden': [[x, f(x)] for x in xs[3:]],
                    'solvable': t['solvable'], 'novel_combination': t['eval_only'], 'required_ops': ops,
                    'required_docs': docs, 'cross_domain': len(docs) >= 2,
                    'reference_program': t['program'], 'split': 'dev' if dev else 'eval'})
    return out


if __name__ == '__main__':
    out = Path('datasets'); out.mkdir(exist_ok=True); manifest = {}
    suites = {'learn_tasks_dev.json': make(7, dev=True)}
    for s in (101, 202, 303):
        suites[f'learn_tasks_eval_{s}.json'] = make(s, n=300)
    for name, tasks in suites.items():
        data = json.dumps(tasks, indent=1, sort_keys=True); (out / name).write_text(data)
        manifest[name] = {'sha256': hashlib.sha256(data.encode()).hexdigest(), 'tasks': len(tasks),
                          'cross_domain': sum(t['cross_domain'] for t in tasks), 'unsolvable': sum(not t['solvable'] for t in tasks),
                          'novel_combination': sum(t['novel_combination'] for t in tasks)}
    (out / 'learn_tasks_manifest.json').write_text(json.dumps(manifest, indent=1)); print(json.dumps(manifest, indent=1))
