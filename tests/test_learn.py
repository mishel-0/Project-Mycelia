import json, hashlib
from pathlib import Path
from mycelia.learn.evidence import build, relation_between, Lab
from mycelia.learn.explorer import Explorer
from mycelia.learn.memory import LearnMemory
from mycelia.learn.network import Network, Node, reverify
from mycelia.learn.solver import Solver, execute, match
from mycelia.learn.sources import Source, extract, subject_of
from mycelia.pure.sandbox import run

RST = """.. module:: statistics

.. function:: median(data)

   Return the median (middle value) of numeric data.

   .. doctest::

      >>> median([1, 3, 5])
      3

.. function:: mean(data)

   Return the sample arithmetic mean of data.  For example, ``mean([1, 2, 3])`` returns ``2``.
"""
CODE = '''def median(data):
    """Return the median (middle value) of numeric data.

    >>> median([1, 3, 5, 7])
    4.0
    """
'''
MD = """```python
<list> = sorted(<collection>)     # Returns a new sorted list.
<str>  = <str>.strip()            # Strips all whitespace characters from both ends.
```"""
S = lambda i, f, lin='python/cpython': Source(i, f'github://x/y/{i}', f, lin, 'statistics')


def test_adapters_emit_the_same_claim_shape():
    rst = extract(RST, S('statistics.rst', 'rst')); code = extract(CODE, Source('c', 'github://x/y/statistics.py', 'code', 'python/cpython', 's'))
    md = extract(MD, S('README.md', 'markdown', 'someone/cheatsheet'))
    assert {c['subject'] for c in rst} == {'statistics.median', 'statistics.mean'}
    assert any(c['relation'] == 'example' and c['object']['source'] == 'statistics.median([1, 3, 5, 7])' for c in code)
    assert {c['subject'] for c in md} == {'sorted', 'str.strip'} and all(c['relation'] == 'describes' for c in md)
    assert subject_of("'a,b'.split(',')") == 'str.split' and subject_of('math.sqrt(4)') == 'math.sqrt'


def test_lifecycle_and_lineage():
    cl = extract(RST, S('statistics.rst', 'rst')) + extract(CODE, Source('c', 'github://x/y/statistics.py', 'code', 'python/cpython', 's'))
    bad = [dict(subject='statistics.mean', relation='example', object={'source': 'statistics.mean([1, 2, 3])', 'expected': '5'},
                domain='s', source='liar', lineage='liar/repo', position=0)]
    skills, decl, ev, rel, _ = build(cl + bad)
    assert decl['statistics.median']['status'] == 'verified_in_scope'
    assert decl['statistics.mean']['status'] == 'contested' and rel['liar/repo'] == 0.
    assert ev['statistics.median']['pairwise_relations']['copy'] == 1  # identical docstring text counts once
    a = {'lineage': 'p', 'text': 'Return the median (middle value) of numeric data.'}
    assert relation_between(a, dict(a)) == 'copy' and relation_between(a, {'lineage': 'q', 'text': 'Middle element of sorted numbers'}) == 'independent'
    assert Lab.status({'contradicted': [1], 'reproduced': [], 'probes': []}, 1, True) == 'refuted_in_scope'


def test_solver_composes_across_domains_and_abstains():
    skills, *_ , consts = build(extract(RST, S('statistics.rst', 'rst')) + [
        dict(subject='str.split', relation='signature', object={'signature': 'str.split(sep=None)', 'params': [{'name': 'sep', 'optional': True}]},
             domain='t', source='s', lineage='p', position=0),
        dict(subject='str.split', relation='describes', object='Return a list of the words in the string', domain='t', source='s', lineage='p', position=0),
        dict(subject='len', relation='signature', object={'signature': 'len(s)', 'params': [{'name': 's', 'optional': False}]}, domain='b', source='f', lineage='p', position=0),
        dict(subject='len', relation='describes', object='Return the length of an object', domain='b', source='f', lineage='p', position=0)])
    task = {'goal': 'median word length', 'examples': [['ab abc a', 2], ['abcd ab', 3.0], ['a bb ccc dddd e', 2]]}
    sol = Solver(skills, consts, budget=5000).solve(task)
    assert sol['program'] and execute(sol['program'], 'xx y zzzz') == ('ok', 2)
    rev = {'goal': 'reverse the characters', 'examples': [['abc', 'cba'], ['xy', 'yx'], ['hello', 'olleh']]}
    assert Solver(skills, consts, budget=2000).solve(rev)['program'] is None
    assert match('length', 'len') and not match('le', 'len')


def test_network_rejects_fake_skills_and_records_exchanges():
    real = build(extract(RST, S('statistics.rst', 'rst')))[0]
    fake = {'str.unscramble': {'op': 'str.unscramble', 'kind': 'method', 'params': [], 'required': 0, 'words': ['median'],
                               'domains': [], 'sources': ['liar'], 'status': 'verified_in_scope'}}
    assert not reverify(fake['str.unscramble']) and reverify(real['statistics.median'])
    mem = LearnMemory(); net = Network([Node('a', {}, [',']), Node('stats', real, [',']), Node('liar', fake, [','])], mem)
    task = {'goal': 'median of the numbers', 'examples': [[[1, 2, 3], 2], [[4, 9, 1], 4], [[5, 5, 1], 5]]}
    sol, lib = net.solve('a', task, 'D', 2000, learn=True)
    assert sol['program'] and 'str.unscramble' not in lib and net.stats['D_rejected_on_receipt'] >= 1
    assert mem.current('coordination') and net.route[('medi', 'stats')] > 0
    _, lib_b = net.solve('a', task, 'B', 2000); assert 'str.unscramble' in lib_b  # the repository has no gate


def test_memory_is_append_only_and_rebuildable():
    m = LearnMemory(); m.put('procedural', 'len', {'v': 1}); m.put('procedural', 'len', {'v': 2})
    assert m.current('procedural')['len'] == {'v': 2} and [h['v'] for h in m.history('procedural', 'len')] == [1, 2]


def test_explorer_follows_references_and_never_sees_eval_tasks():
    cat = ['Doc/library/math.rst', 'Doc/library/statistics.rst', 'Doc/library/zipfile.rst']
    ex = Explorer(cat, 'curiosity'); ex.observe('Doc/builtins/stdtypes.rst', 'See :mod:`statistics` and :func:`statistics.mean`. Also :mod:`math`.')
    ex.observe('Doc/x.rst', 'Uses :mod:`statistics`.')
    assert ex.next() == 'Doc/library/statistics.rst'
    g = Explorer(cat, 'goal'); g.observe('Doc/builtins/stdtypes.rst', 'For square roots see :mod:`math`.')
    assert g.next(['square root of the number']) == 'Doc/library/math.rst'


def test_task_suites_match_frozen_manifest():
    man = json.loads(Path('datasets/learn_tasks_manifest.json').read_text())
    for name, meta in man.items():
        assert hashlib.sha256(Path('datasets', name).read_text().encode()).hexdigest() == meta['sha256']


def test_sandbox_allows_safe_functions_only():
    assert run('statistics.median([1, 5, 3])') == ('ok', 3) and run('sorted(["b", "a"])') == ('ok', ['a', 'b'])
    for bad in ('math.factorial(10000)', "sorted(['a'], key=len)", 'math.__dict__', '__import__("os")', 'eval("1")'):
        assert run(bad)[0] == 'blocked'
