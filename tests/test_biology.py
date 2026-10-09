"""Mechanism interventions and analytic/transactional checks for the new core."""
from copy import deepcopy
from dataclasses import replace
import json
import math
import pytest
from mycelia.biology import Hypha, Parameters, Septum
from mycelia.biology.mechanisms import water_transport, nutrient_uptake


def inert_parameters(**changes):
    p = Parameters(membrane_permeability=0, uptake_density=0, metabolism_per_min=0,
                   maintenance_density=0, reserve_release_per_min=0,
                   vesicle_synthesis_per_min=0, motor_delivery_per_min=0,
                   exocytosis_per_min=0, diffusion_um2_min=0)
    return replace(p, **changes)


def assert_balanced(h):
    assert max(abs(e) for e in h.budgets().values()) < 1e-9
    h.validate()


def test_dimensional_constants_and_pore_conductance():
    p = Parameters()
    assert p.RT == pytest.approx(2.47895703, rel=1e-8)
    q_m3_s = math.pi * (p.pore_radius_um * 1e-6)**4 * 1e6 / (8 * p.effective_viscosity_Pa_s * p.septum_length_um * 1e-6)
    assert Septum(0, 1).conductance(p) == pytest.approx(q_m3_s * 1e15 * 60)
    assert p.wall_carbon_pmol_pL == pytest.approx(0.08)


@pytest.mark.parametrize('field,value', [('temperature_K', 0), ('bulk_modulus_MPa', -1), ('max_substep_min', .1), ('wall_yield_MPa', float('nan')), ('max_compartments', 2.5)])
def test_invalid_parameters(field, value):
    with pytest.raises(ValueError):
        replace(Parameters(), **{field: value})


def test_finite_bath_shared_uptake():
    h = Hypha(replace(inert_parameters(), uptake_density=100))
    h.cells[1].bath = 0
    bath = h.baths[0]
    before = bath.nutrient_pmol
    nutrient_uptake(h, 10)
    assert bath.nutrient_pmol == pytest.approx(0, abs=1e-12)
    assert sum(c.nutrient_pmol for c in h.cells) > before
    assert_balanced(h)


def test_hydraulic_pressure_difference_matches_implicit_analytic_solution():
    h = Hypha(inert_parameters())
    a, b = h.cells
    a.water_pL += .02
    b.water_pL -= .02
    p = h.parameters
    before = a.pressure(p) - b.pressure(p)
    capacitance = a.rest_volume_pL / p.bulk_modulus_MPa
    dt = .01
    expected = before / (1 + 2 * dt * h.septa[0].conductance(p) / capacitance)
    water_transport(h, dt)
    assert a.pressure(p) - b.pressure(p) == pytest.approx(expected, abs=1e-12)
    assert h.septa[0].last_flow_pL_min > 0
    assert_balanced(h)


def test_septal_closure_prevents_flow_and_diffusion():
    h = Hypha(inert_parameters(diffusion_um2_min=100))
    a, b = h.cells
    a.water_pL += .02; b.water_pL -= .02
    h.septa[0].opening = 0
    before = deepcopy(h.state_dict()['cells'])
    h.run(1)
    assert h.state_dict()['cells'] == before
    assert h.septa[0].last_flow_pL_min == 0
    assert_balanced(h)


@pytest.mark.parametrize('mechanism', ['wall_extensibility', 'motor_delivery_per_min', 'exocytosis_per_min'])
def test_growing_tip_requires_wall_motor_and_exocytosis(mechanism):
    control = Hypha(initial_cargo_pmol=0)
    inhibited = Hypha(replace(Parameters(), **{mechanism: 0}), initial_cargo_pmol=0)
    control.run(5); inhibited.run(5)
    assert control.summary()['permanent_extension_um'] > 1
    assert inhibited.summary()['permanent_extension_um'] == pytest.approx(0, abs=1e-12)
    assert_balanced(control); assert_balanced(inhibited)


def test_ATP_regeneration_is_required_when_initial_energy_zero():
    h = Hypha(replace(Parameters(), ATP_yield=0), initial_ATP_pmol=0, initial_cargo_pmol=0)
    h.run(5)
    assert h.summary()['permanent_extension_um'] == pytest.approx(0, abs=1e-12)
    assert h.summary()['tip_ATP_pmol'] == 0
    assert_balanced(h)


def test_no_growth_below_wall_yield():
    h = Hypha(replace(Parameters(), wall_yield_MPa=1))
    h.run(5)
    assert h.summary()['permanent_extension_um'] == pytest.approx(0, abs=1e-12)


def test_hyperosmotic_shock_loses_water_and_arrests_growth_without_negative_turgor():
    h = Hypha(); h.run(10)
    pre = h.summary()
    h.hyperosmotic_shock(.6); h.run(2)
    arrested = h.summary()
    assert arrested['tip_pressure_MPa'] == 0
    assert arrested['total_water_pL'] < pre['total_water_pL']
    h.run(1)
    assert h.summary()['permanent_extension_um'] == pytest.approx(arrested['permanent_extension_um'], abs=1e-12)
    assert h.limiter_events == 0
    assert_balanced(h)


def test_growth_generates_toward_tip_flow_in_initially_equilibrated_pair():
    p = inert_parameters(exocytosis_per_min=4)
    growing = Hypha(p, initial_cargo_pmol=.05)
    stopped = Hypha(replace(p, wall_extensibility=0), initial_cargo_pmol=.05)
    growing.run(.1); stopped.run(.1)
    assert growing.summary()['permanent_extension_um'] > 0
    assert growing.summary()['flow_toward_tip_pL_min'] > 0
    assert stopped.summary()['flow_toward_tip_pL_min'] == pytest.approx(0, abs=1e-12)
    assert_balanced(growing)


def test_septation_preserves_pools_and_tip_polarization():
    h = Hypha(); h.run(60)
    assert len(h.cells) >= 3
    assert sum(c.tip for c in h.cells) == 1
    assert all(c.apical_cargo_pmol == 0 for c in h.cells if not c.tip)
    assert len(h.septa) == len(h.cells) - 1
    assert_balanced(h)


def test_restart_matches_uninterrupted_evolution(tmp_path):
    h = Hypha(); h.run(2)
    path = tmp_path / 'state.json'
    h.save(path)
    resumed = Hypha.load(path)
    h.run(3); resumed.run(3)
    assert h.state_dict() == resumed.state_dict()
    invalid = deepcopy(h.state_dict()); invalid['schema'] = 'prototype'
    with pytest.raises(ValueError): Hypha.from_state_dict(invalid)


def test_failed_step_rolls_back_partial_reactions(monkeypatch):
    import mycelia.biology.engine as module
    h = Hypha(); before = h.state_dict()
    def broken(*args): raise ArithmeticError('injected solver failure')
    monkeypatch.setattr(module, 'water_transport', broken)
    with pytest.raises(ArithmeticError, match='injected'):
        h.step()
    assert h.state_dict() == before


def test_timestep_refinement_reduces_numerical_error():
    runs = [Hypha(replace(Parameters(), max_substep_min=dt)) for dt in (.02, .01, .0025)]
    for h in runs: h.run(5)
    lengths = [h.summary()['permanent_extension_um'] for h in runs]
    coarse_error, fine_error = [abs(x-lengths[2]) for x in lengths[:2]]
    assert fine_error < coarse_error
    assert fine_error / lengths[2] < .01
    assert all(h.limiter_events == 0 for h in runs)


def test_invalid_checkpoint_and_unbudgeted_edit_rejected():
    h = Hypha(); data = h.state_dict(); data['cells'][0]['water_pL'] += .5
    with pytest.raises(ValueError, match='balance'): Hypha.from_state_dict(data)
    data = h.state_dict(); data['septa'][0]['opening'] = 2
    with pytest.raises(ValueError, match='opening'): Hypha.from_state_dict(data)
    h.cells[0].nutrient_pmol += 1
    with pytest.raises(ValueError, match='balance'): h.step()


def test_biology_cli_run_resume_and_reject_invalid_duration(tmp_path):
    import subprocess, sys
    output = tmp_path / 'run'
    command = [sys.executable, '-m', 'mycelia.biology']
    result = subprocess.run(command + ['--duration', '.5', '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = json.loads((output / 'summary.json').read_text())
    assert data['time_min'] == pytest.approx(.5)
    assert (output / 'timeseries.csv').exists()
    assert 'Empirical' not in result.stdout  # Routine run has no claimed empirical pass.
    result = subprocess.run(command + ['--resume', str(output/'state.json'), '--duration', '.5', '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((output / 'summary.json').read_text())['time_min'] == pytest.approx(1)
    result = subprocess.run(command + ['--duration', '-1', '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 2


def test_validation_panel_serializes_and_reports_computational_scope(tmp_path):
    from mycelia.biology.experiments import run_validation
    result = run_validation(tmp_path)
    loaded = json.loads((tmp_path / 'summary.json').read_text())
    assert result['passed'] and all(loaded['checks'].values())
    assert loaded['empirical_validation'] == 'not performed'
    assert len(loaded['scenarios']) == 8
    assert 'not establish biological fidelity' in (tmp_path / 'report.html').read_text()
    for name in loaded['scenarios']:
        restored = Hypha.load(tmp_path / name / 'state.json')
        assert_balanced(restored)
        assert (tmp_path / name / 'timeseries.csv').read_text().startswith('time_min,step,')
