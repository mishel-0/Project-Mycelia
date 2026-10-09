"""Reproducible causal interventions; these are computational, not empirical fits."""
from dataclasses import replace, asdict
import csv
import json
from pathlib import Path
from .engine import Hypha
from .state import Parameters


def _rate(h, start):
    records = [row for row in h.history if row['time_min'] >= start - 1e-8]
    if len(records) < 2:
        return 0.0
    return (records[-1]['permanent_extension_um'] - records[0]['permanent_extension_um']) / (records[-1]['time_min'] - records[0]['time_min'])


def run_validation(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    p = Parameters()
    changes = dict(full={}, no_wall_creep={'wall_extensibility': 0},
                   no_motor_delivery={'motor_delivery_per_min': 0},
                   no_exocytosis={'exocytosis_per_min': 0},
                   no_ATP_regeneration={'ATP_yield': 0},
                   no_membrane_water={'membrane_permeability': 0},
                   closed_septa={'pore_radius_um': 0})
    models = {}
    for name, change in changes.items():
        h = Hypha(replace(p, **change), initial_ATP_pmol=0, initial_cargo_pmol=0, tip_bath_nutrient=False)
        h.run(30)
        models[name] = h
    shock = Hypha(initial_ATP_pmol=0, initial_cargo_pmol=0, tip_bath_nutrient=False)
    shock.run(10)
    before = shock.summary()
    shock.hyperosmotic_shock(.6)
    shock.run(20)
    models['hyperosmotic_shock'] = shock
    refined = []
    for dt in (.02, .01, .0025):
        h = Hypha(replace(p, max_substep_min=dt), initial_ATP_pmol=0, initial_cargo_pmol=0, tip_bath_nutrient=False)
        h.run(10)
        refined.append(dict(substep_min=dt, extension_um=float(h.summary()['permanent_extension_um']), limiter_events=h.limiter_events))
    reference = refined[-1]['extension_um']
    fine_error = abs(refined[1]['extension_um'] - reference)
    coarse_error = abs(refined[0]['extension_um'] - reference)
    # Isolate growth-induced flow: equilibrium, no uptake, metabolism,
    # membrane exchange or solute diffusion; only initially supplied tip cargo.
    isolated_p = replace(p, uptake_density=0, metabolism_per_min=0, maintenance_density=0,
                         reserve_release_per_min=0, vesicle_synthesis_per_min=0,
                         motor_delivery_per_min=0, membrane_permeability=0, diffusion_um2_min=0)
    isolated = Hypha(isolated_p, initial_cargo_pmol=.05)
    isolated.run(.1)
    isolated_off = Hypha(replace(isolated_p, wall_extensibility=0), initial_cargo_pmol=.05)
    isolated_off.run(.1)
    extensions = {name: float(h.summary()['permanent_extension_um']) for name, h in models.items()}
    checks = dict(
        full_grows=extensions['full'] > 1,
        wall_required=abs(extensions['no_wall_creep']) < 1e-10,
        motor_required=abs(extensions['no_motor_delivery']) < 1e-10,
        exocytosis_required=abs(extensions['no_exocytosis']) < 1e-10,
        ATP_required=abs(extensions['no_ATP_regeneration']) < 1e-10,
        water_exchange_supports_sustained_growth=extensions['no_membrane_water'] < extensions['full'],
        open_septa_support_remote_nutrient_supply=extensions['closed_septa'] < extensions['full'],
        shock_loses_water=shock.summary()['total_water_pL'] < before['total_water_pL'],
        shock_arrests_growth=abs(_rate(shock, 25)) < 1e-10,
        shock_turgor_is_zero=shock.summary()['tip_pressure_MPa'] == 0,
        growth_induces_flow=isolated.summary()['flow_toward_tip_pL_min'] > 0 and abs(isolated_off.summary()['flow_toward_tip_pL_min']) < 1e-10,
        refinement_reduces_error=fine_error < coarse_error,
        refined_extension_difference_below_one_percent=fine_error / reference < .01,
        conservative_budgets=all(max(abs(v) for v in h.budgets().values()) < 1e-9 for h in models.values()),
        no_water_limiter_activation=all(h.limiter_events == 0 for h in models.values()) and all(r['limiter_events'] == 0 for r in refined),
    )
    checks = {name: bool(value) for name, value in checks.items()}
    summary = dict(model='mechanistic hypha milestone 1', empirical_validation='not performed',
                   species_calibration='none; illustrative cross-species mechanism selection',
                   passed=all(checks.values()), checks=checks, duration_min=30,
                   osmotic_shock=dict(time_min=10, increment_MPa=.6, pre=before, final=shock.summary()),
                   parameters=asdict(p), scenarios={name: dict(h.summary(), terminal_extension_rate_um_min=_rate(h, 25)) for name, h in models.items()},
                   timestep_refinement=refined, refinement_relative_difference=fine_error / reference,
                   isolated_growth_flow=dict(growing=isolated.summary(), wall_disabled=isolated_off.summary()),
                   limitations=['No ion/Ca/channel or membrane voltage model',
                                'No osmotic adaptation: shock recovery is not reproduced',
                                'Straight prescribed axis, no branching or fusion in this core',
                                'Cargo pools are lumped; motor/exocyst identities not resolved',
                                'No quantitative experimental dataset has been fitted or predicted'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    for name, h in models.items():
        target = output / name
        target.mkdir(exist_ok=True)
        h.save(target / 'state.json')
        fields = [key for key, value in h.history[0].items() if isinstance(value, (int, float))]
        with (target / 'timeseries.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader(); writer.writerows(h.history)
    from .report import write_validation_report
    write_validation_report(output, summary, models)
    return summary
