"""Portable numerical reports; plots use matplotlib when explicitly installed."""
import csv
import html
import json
from pathlib import Path


STYLE = '''body{font:16px/1.6 system-ui,sans-serif;margin:0;background:#111923;color:#d9e6ef}
main{max-width:1100px;margin:auto;padding:40px 24px}h1{font-size:34px;line-height:1.2}h2{margin-top:32px;color:#94ddc8}
a{color:#94ddc8}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 12px;border-bottom:1px solid #314453;text-align:left}th{color:#94ddc8}.scroll{overflow:auto}.note{padding:18px;border-left:3px solid #daa263;background:#1c2935}img{width:100%;background:white;border-radius:4px}.pass{color:#94ddc8}.fail{color:#ffacac}code{color:#94ddc8}'''


def write_run_report(output, model):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    model.save(output / 'state.json')
    summary = model.summary()
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    fields = [key for key, value in model.history[0].items() if isinstance(value, (int, float))]
    with (output / 'timeseries.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(model.history)
    rows = ''.join(f'<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>' for k, v in summary.items() if k != 'budget_errors')
    errors = ''.join(f'<tr><td>{k}</td><td>{v:.3g}</td></tr>' for k, v in model.budgets().items())
    (output / 'report.html').write_text(f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MYCELIA mechanistic hypha</title><style>{STYLE}</style></head><body><main><h1>MYCELIA · Mechanistic hypha</h1><p class="note">Computational physiology experiment. Parameters are uncalibrated; no measured fungal dataset has been fitted or predicted. Length records permanent wall extension; elastic shape and branching are not resolved.</p><h2>State and observables</h2><div class="scroll"><table>{rows}</table></div><h2>Budget errors</h2><table>{errors}</table><p><a href="timeseries.csv">Time series</a> · <a href="state.json">Restart state</a> · <a href="summary.json">Summary JSON</a></p></main></body></html>')


def write_validation_report(output, summary, models):
    output = Path(output)
    rows = []
    for name, state in summary['scenarios'].items():
        rows.append(f'<tr><td>{html.escape(name.replace("_", " "))}</td><td>{state["permanent_extension_um"]:.6f}</td><td>{state["terminal_extension_rate_um_min"]:.6f}</td><td>{state["tip_pressure_MPa"]:.6f}</td><td>{state["flow_toward_tip_pL_min"]:.7f}</td></tr>')
    checks = ''.join(f'<tr><td>{html.escape(k.replace("_", " "))}</td><td class="{"pass" if v else "fail"}">{"PASS" if v else "FAIL"}</td></tr>' for k, v in summary['checks'].items())
    max_errors = {k: max(abs(h.budgets()[k]) for h in models.values()) for k in next(iter(models.values())).budgets()}
    budgets = ''.join(f'<tr><td>{k}</td><td>{v:.3g}</td></tr>' for k, v in max_errors.items())
    refinement = ''.join(f'<tr><td>{row["substep_min"]}</td><td>{row["extension_um"]:.8f}</td></tr>' for row in summary['timestep_refinement'])
    limitations = ''.join(f'<li>{html.escape(s)}</li>' for s in summary['limitations'])
    content = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MYCELIA · Biological mechanism experiments</title><style>{STYLE}</style></head><body><main><p>MYCELIA / MECHANISTIC CORE / MILESTONE 1</p><h1>What makes the hypha grow?</h1><p>Finite nutrient uptake → osmotic water exchange → turgor → funded wall expansion → growth-induced flow.</p><p class="note">These tests verify a coupled computational model. They do not establish biological fidelity or species calibration. All arms start with zero ATP and zero vesicle cargo; nutrients are available in the basal bath only. Seed nutrients, osmolytes and reserves remain present.</p><h2>Matched interventions · 30 minutes</h2><div class="scroll"><table><tr><th>Condition</th><th>Extension (µm)</th><th>Last 5 min rate (µm/min)</th><th>Tip turgor (MPa)</th><th>Flow toward tip (pL/min)</th></tr>{''.join(rows)}</table></div><p>Closed septa still allow local seed resources and tip water uptake. Removing water exchange still permits a little extension by relaxing the initially stretched wall. Neither condition is expected to abolish all growth.</p><h2>Computational checks</h2><table>{checks}</table><h2>Timestep refinement · 10 minutes</h2><table><tr><th>Substep (min)</th><th>Extension (µm)</th></tr>{refinement}</table><p>The 0.01-min versus 0.0025-min relative difference is {100*summary['refinement_relative_difference']:.4f}%. This is a numerical comparison, not biological accuracy.</p><h2>Maximum final budget residuals</h2><table>{budgets}</table><p>Carbon is measured in pmol of monomer equivalents, water in pL, energy in pmol of ATP equivalents, and inert bath osmolyte in pmol. ATP equivalents are an accounting proxy, not thermodynamic free energy.</p><h2>Biological limits</h2><ul>{limitations}</ul><p>Hyperosmotic shock of +0.6 MPa is applied at minute 10. This core intentionally does not produce regulatory recovery: ion uptake and osmolyte synthesis pathways require a later specified model.</p><p><a href="summary.json">Machine-readable results</a>. Each named condition folder contains a restart state and CSV time series.</p></main></body></html>'''
    (output / 'report.html').write_text(content)
    results = ['# MYCELIA mechanistic physiology — milestone 1', '',
               'Computational validation only. No species calibration or quantitative empirical validation.', '',
               '| Condition | Permanent extension (µm) | Final turgor (MPa) |', '|---|---:|---:|']
    for name, row in summary['scenarios'].items():
        results.append(f'| {name} | {row["permanent_extension_um"]:.6f} | {row["tip_pressure_MPa"]:.6f} |')
    results += ['', f'Computational checks: {sum(summary["checks"].values())}/{len(summary["checks"])} passed.',
                f'Timestep refinement relative difference: {summary["refinement_relative_difference"]:.8g}.', '',
                'No osmotic recovery, explicit ion channels/calcium, molecular cytoskeleton, branching/fusion or demonstrated biological memory are implemented in this new core.', '',
                'The older prototype is preserved separately. Its MRI results do not validate this model.']
    (output / 'RESULTS.md').write_text('\n'.join(results)+'\n')
