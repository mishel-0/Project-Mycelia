import hashlib, json
from pathlib import Path
from mycelia.autonomy.network_v2 import run, liar_rules, oracle_unsafe
from mycelia.autonomy.node import Car
from mycelia.autonomy.scenarios import make
from mycelia.autonomy.sim import Scenario, Hazard, simulate, reflex, observe
from mycelia.autonomy.trust import Trust

POL = lambda o: (reflex(o), None)


def test_brake_profile_changes_outcome_but_not_observations():
    ped = Scenario('p', 1, 'clear', 'dry', [Hazard('pedestrian', 0, 6, t0=3, duration=4)])
    assert simulate(ped, POL, brake=0)['outcome'] == 'success' and simulate(ped, POL, brake=2)['outcome'] == 'collision'


def test_hidden_shift_changes_dynamics_with_identical_observations():
    a, b = make('urban_mixed', 5), make('urban_mixed', 5, shifted=True)
    assert observe(a, a.hazards, 0, 0, 0) == observe(b, b.hazards, 0, 0, 0) and (a.extra_roll, b.extra_roll) == (0, 1)


def test_trust_rejects_and_skips_on_evidence():
    t = Trust()
    for _ in range(3):
        t.update('bad', False)
    for _ in range(6):
        t.update('good', True)
    assert t.reject('bad') and not t.reject('good') and t.skip_verify('good') and not t.skip_verify('new')


def test_liar_rules_are_unsafe_and_caught_by_reverification():
    car = Car('car2', 'rain_pedestrians', brake=2, shifted=True)
    assert oracle_unsafe(liar_rules()[0], car)  # "drive through pedestrians" is unsafe for this car
    _, held_b, stats_b, _ = run('B_scope', 0, budget=160)
    _, held_m1, stats_m1, _ = run('M1', 0, budget=160)
    liar = lambda held: sum(r['id'].startswith('car5') for rs in held.values() for r in rs)
    assert liar(held_b) == 16 and liar(held_m1) < liar(held_b) and stats_m1['rejected_by_verification'] > 0


def test_monitoring_retracts_after_collision_everywhere():
    cars, held, stats, mem = run('M3', 0, budget=160)
    assert stats['retractions'] > 0
    assert stats['liar_rules_held_end'] < 16


def test_value_sharing_respects_bandwidth():
    _, _, stats, mem = run('M4', 0, budget=160, k=2)
    per = {}
    for rec in mem.current('coordination').values():
        per[(rec['receiver'], rec['round'])] = per.get((rec['receiver'], rec['round']), 0) + 1
    assert per and max(per.values()) <= 2


def test_v2_suites_match_frozen_manifest():
    man = json.loads(Path('datasets/drive_v2_manifest.json').read_text())
    for name, meta in man.items():
        assert hashlib.sha256(Path('datasets', name).read_text().encode()).hexdigest() == meta['sha256']
