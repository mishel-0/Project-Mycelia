import hashlib, json
from pathlib import Path
from mycelia.autonomy.network import build, covers
from mycelia.autonomy.node import Car, matches
from mycelia.autonomy.scenarios import make
from mycelia.autonomy.sim import Scenario, Hazard, simulate, reflex, observe


def test_sim_is_deterministic_and_reflex_gets_stuck_behind_block():
    sc = Scenario('t', 2, 'clear', 'dry', [Hazard('block', 0, 8)])
    a, b = simulate(sc, lambda o: (reflex(o), None)), simulate(sc, lambda o: (reflex(o), None))
    assert a == b and a['outcome'] == 'stuck' and a['unnecessary_stops'] == 0
    assert observe(Scenario('f', 2, 'fog', 'dry', []), [], 0, 0, 0)['left_free'] == 'none'
    assert observe(Scenario('f', 2, 'fog', 'dry', []), [], 0, 0, 0)['right_free'] == 'unknown'  # fog hides side lanes


def test_wet_roads_roll_one_cell_after_stopping():
    pol = lambda o: (reflex(o), None)  # stops as soon as the block is seen one cell ahead
    wet = simulate(Scenario('w', 1, 'clear', 'wet', [Hazard('block', 0, 1)]), pol)
    dry = simulate(Scenario('d', 1, 'clear', 'dry', [Hazard('block', 0, 1)]), pol)
    assert wet['outcome'] == 'collision' and dry['outcome'] == 'stuck'


def test_car_learns_lane_change_with_recorded_scope():
    car = Car('car1', 'city_blocks', budget=200); car.practice()
    assert car.rules and all(r['status'] == 'verified_in_scope' and r['evidence']['collisions'] == 0 for r in car.rules)
    lane_rules = [r for r in car.rules if r['action'] in ('left', 'right') and r['cond'].get('ahead') == 'block']
    assert lane_rules and all('visibility' in r['scope'] or r['cond'].get('visibility') for r in lane_rules)
    sc = make('city_blocks', 999999)
    assert simulate(sc, car.policy())['outcome'] == 'success'


def test_scope_check_blocks_use_outside_verified_conditions():
    rule = {'id': 'x-r1', 'cond': {'ahead': 'block'}, 'action': 'left', 'scope': {'surface': ['dry']}, 'origin': 'x'}
    obs = {'ahead': 'block', 'distance': 'near', 'left_free': 'yes', 'right_free': 'none', 'visibility': 'clear', 'surface': 'wet'}
    assert matches(rule, obs, False) and not matches(rule, obs, True)
    assert covers(rule, {('clear', 'dry', 'yes', 'none')}) and not covers(rule, {('clear', 'wet', 'yes', 'none')})


def test_adaptive_condition_logs_every_exchange():
    cars, extras, stats, memory = build('D', budget=150, seed=0)
    assert stats['messages'] == sum(1 for c in cars for o in cars if o is not c for _ in o.rules)
    assert len(memory.current('coordination')) == stats['messages'] - 0
    assert all(c.budget.used <= c.budget.limit for c in cars)


def test_eval_suites_match_frozen_manifest():
    man = json.loads(Path('datasets/drive_manifest.json').read_text())
    for name, meta in man.items():
        assert hashlib.sha256(Path('datasets', name).read_text().encode()).hexdigest() == meta['sha256']
