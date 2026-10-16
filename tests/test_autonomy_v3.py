import hashlib, json
from pathlib import Path
from mycelia.autonomy import bio
from mycelia.autonomy.scenarios import make
from mycelia.autonomy.sim import simulate

OBS = {'ahead': 'pedestrian', 'distance': 'near', 'visibility': 'clear', 'surface': 'wet', 'left_free': 'none', 'right_free': 'none'}


def car(**kw):
    return bio.BioCar('car2', 'rain_pedestrians', 400, batch=6, seed=0, brake=2, scope_check=True, **kw)


def test_rat_records_and_octopus_vetoes_crashing_action():
    c = car(); trace = [(OBS, 'go', None)]
    for _ in range(3):
        c.record(trace, 'collision')
    assert c.stats(OBS, 'go')[:2] == (3, 1.) and c.vetoed(OBS, 'go')
    assert not car(mech={'rat'}).vetoed(OBS, 'go')


def test_crow_generalisation_may_drop_obstacle_type():
    seen = {}
    class Spy(bio.BioCar):
        def _experiment(self, cond, trigger, generalise):
            seen['g'] = generalise; return None
    for mech, has in (({'crow'}, True), (set(), False)):
        c = Spy('car2', 'rain_pedestrians', 400, brake=2, mech=mech); c.practice()
        assert ('ahead' in seen['g']) == has


def test_judge_removes_only_repeatedly_crashing_rules():
    c = car(); c.rules = [{'id': 'x', 'cond': {}, 'action': 'go'}]
    c.judge({'fired': ['x'], 'outcome': 'collision'})
    assert c.rules  # one crash is not enough
    c.judge({'fired': ['x'], 'outcome': 'collision'}); c.judge({'fired': ['x'], 'outcome': 'collision'})
    assert not c.rules


def test_weak_brake_car_learns_early_pedestrian_braking():
    c = car(); c.practice(); pol = c.policy(scope_check=True)
    ok = sum(simulate(make('rain_pedestrians', 9000 + i), pol, brake=2)['outcome'] == 'success' for i in range(50))
    assert ok >= 45


def test_network_run_bee_physarum_and_bandwidth():
    cars, held, stats, _ = bio.run(seed=0, budget=200, rounds=4, ops=5, k=2)
    assert stats['messages'] <= 2 * len(cars) * 4
    assert all(r['probation'] for h in held.values() for r in h if r['id'].startswith('car5'))  # never reach quorum
    assert all(r.get('g', 1.) >= .3 for h in held.values() for r in h)


def test_v3_suites_match_manifest():
    m = json.loads(Path('datasets/drive_v3_manifest.json').read_text())
    files = m.get('files', m)
    for name, digest in files.items():
        if isinstance(digest, dict):
            digest = digest.get('sha256')
        assert hashlib.sha256(Path('datasets', Path(name).name).read_bytes()).hexdigest() == digest
