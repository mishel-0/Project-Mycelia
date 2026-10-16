import json
from pathlib import Path
import pytest
from mycelia.pure.extract import extract
from mycelia.pure.sandbox import run, PURE_STR_METHODS
from mycelia.pure.skills import learn_skills
from mycelia.pure.solver import Solver

PAGE = Path('tests/fixtures/sphinx_str_methods.html').read_text()
BAD = Path('tests/fixtures/untrusted_altered_docs.html').read_text()


def test_extractor_reads_signatures_descriptions_examples():
    e = {x['method']: x for x in extract(PAGE, 'https://docs.python.org/3/library/stdtypes.html')}
    assert {'upper', 'split', 'replace', 'join'} <= set(e)
    assert e['split']['examples'][0] == {'source': "'1,2,3'.split(',')", 'expected': "['1', '2', '3']"}
    assert [p['optional'] for p in e['replace']['params']] == [False, False, True]
    assert 'uppercase' in e['upper']['description'] and e['upper']['source'].startswith('https://')


@pytest.mark.parametrize('src', ["__import__('os').system('echo hi')", "'a'.__class__.__mro__", "().__class__",
                                 "(lambda: 1)()", "open('/etc/passwd')", "'x' * 10**9", "'a'.format(1)", "[c for c in 'ab']"])
def test_sandbox_blocks_escapes(src):
    assert run(src)[0] == 'blocked'


def test_skills_verified_by_experiment_and_untrusted_claims_refuted():
    skills, consts, bus = learn_skills([(PAGE, 'good'), (BAD, 'untrusted')])
    status = {}
    for s in skills:
        status.setdefault(s['method'], []).append((s['source'], s['status']))
    assert ('good', 'verified') in status['upper'] and ('good', 'verified') in status['split']
    assert ('untrusted', 'contested') in status['title']        # altered example contradicted by experiment
    assert status['unscramble'] == [('untrusted', 'not_allowed')]  # invented method never runs
    assert status['format'][0][1] == 'not_allowed'
    assert ',' in consts and any(m.kind == 'counterexample' for m in bus.log)


def test_solver_uses_verified_skills_and_abstains_when_it_cannot():
    skills, consts, _ = learn_skills([(PAGE, 'good')])
    good = [s for s in skills if s['status'] == 'verified']
    assert {'replace', 'join'} <= {s['method'] for s in good}  # verified by probing with learned constants
    solver = Solver(good, consts)
    t = {'goal': 'make each word start with a capital letter and trim the ends',
         'examples': [['  alice SMITH ', 'Alice Smith'], ['bob', 'Bob'], [' x y', 'X Y']]}
    sol = solver.solve(t, 5000); assert sol['program'] and run(sol['program'], {'x': ' river stone '}) == ('ok', 'River Stone')
    rev = {'goal': 'reverse the order of the characters', 'examples': [['abc', 'cba'], ['xy', 'yx'], ['hello', 'olleh']]}
    assert solver.solve(rev, 3000)['program'] is None


def test_task_set_is_frozen_and_split():
    tasks = json.loads(Path('datasets/pure_tasks.json').read_text())
    assert {t['split'] for t in tasks} == {'train', 'test'} and all(len(t['hidden']) == 6 for t in tasks)
