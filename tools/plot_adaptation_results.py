"""Export scientific adaptation curves and a standalone evidence report."""
import argparse,csv,html,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args();out=args.output
    data=json.loads((out/'cue-switch/summary.json').read_text());medical=json.loads((out/'mri-validation/summary.json').read_text())
    rows=list(csv.DictReader((out/'cue-switch/trajectories.csv').open()));arms=list(data['arms'])
    names={'legacy_fixed':'Fixed legacy','legacy_consolidated':'Aging legacy','dual_fast_slow':'Fast + slow','fast_only':'Fast only','slow_only':'Slow only'}
    colors={'legacy_fixed':'#687886','legacy_consolidated':'#a26837','dual_fast_slow':'#147c6d','fast_only':'#be425b','slow_only':'#6c59a0'}
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for arm in arms:
        y=[]
        for step in range(1,13):
            selected=[r for r in rows if r['arm']==arm and r['phase']=='B' and int(r['phase_step'])==step]
            y.append(np.mean([float(r['distance_B'])/float(r['reference_AB']) for r in selected]))
        axes[0].plot(range(1,13),y,label=names[arm],color=colors[arm])
    axes[0].axhline(.05,color='#333',linestyle=':',label='Adaptation threshold')
    axes[0].text(.40,.14,'Fast-only and dual responses overlap',transform=axes[0].transAxes,fontsize=8)
    axes[0].set(title='A. Response after switching A → B',xlabel='B exposures',ylabel='Normalized mismatch to B',ylim=(0,1.1));axes[0].legend(fontsize=8)
    retention=[data['arms'][arm]['mean_A_retention_after_B_normalized'] for arm in arms]
    axes[1].bar(range(5),retention,color=[colors[arm] for arm in arms]);axes[1].set_xticks(range(5),[names[arm].replace(' ','\n') for arm in arms]);axes[1].set(title='B. Old cue A before any reteaching',ylabel='Normalized mismatch to A',ylim=(0,1.25));axes[1].axhline(.1,color='#333',linestyle=':')
    for i,y in enumerate(retention):axes[1].text(i,y+.025,f'{y:.3f}',ha='center',fontsize=9)
    candidates=medical['candidates'];values=[c['validation']['accuracy']*100 for c in candidates]
    axes[2].bar(range(3),values,color=['#a26837','#147c6d','#be425b']);axes[2].set_xticks(range(3),['Aging\nlegacy','Fast +\nslow','Fast\nonly']);axes[2].set(title='C. Separate MRI Training validation',ylabel='Correct predictions (%)',ylim=(0,100))
    for i,y in enumerate(values):axes[2].text(i,y+2,f'{y:.2f}%',ha='center')
    fig.suptitle('MYCELIA · Candidate adaptation evidence\n30 paired synthetic trials; MRI validation n=1,082; no biological or clinical validation',fontsize=13)
    fig.savefig(out/'adaptation-results.svg');fig.savefig(out/'adaptation-results.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    for arm in ('legacy_fixed','dual_fast_slow','fast_only'):
        for cue,style in [('A','-'),('B','--')]:
            values=[]
            for step in range(1,65):
                selected=[r for r in rows if r['arm']==arm and int(r['exposure'])==step]
                values.append(np.mean([float(r['distance_'+cue])/float(r['reference_AB']) for r in selected]))
            ax.plot(range(1,65),values,color=colors[arm],linestyle=style,label=names[arm]+': cue '+cue)
    ax.axvspan(40.5,52.5,color='#f3d5ac',alpha=.4);ax.axvline(40.5,color='#777',linestyle=':');ax.axvline(52.5,color='#777',linestyle=':')
    ax.set(title='Full A → B → A history; mean across 30 paired trials',xlabel='Exposure (A: 1–40, B: 41–52, return A: 53–64)',ylabel='Normalized mismatch');ax.legend(ncol=3,fontsize=8);fig.savefig(out/'full-history.svg');fig.savefig(out/'full-history.png',dpi=160);plt.close(fig)
    def esc(value):return html.escape(str(value))
    arm_rows=''.join(f'<tr><td>{esc(names[arm])}</td><td>{d["adapted_to_B"]}/30</td><td>{d["adapted_and_retained_A"]}/30</td><td>{d["mean_B_restricted_latency"]:.1f}</td><td>{d["mean_A_retention_after_B_normalized"]:.4f}</td><td>{d["mean_synthesis_carbon"]:.3f}</td></tr>' for arm,d in data['arms'].items())
    mri_rows=''.join(f'<tr><td>{esc(c["name"])}</td><td>{c["validation"]["accuracy"]:.2%}</td><td>{c["validation"]["balanced_accuracy"]:.2%}</td></tr>' for c in candidates)
    before=json.loads((out/'debug-baseline.json').read_text());after=json.loads((out/'debug-post-fixes.json').read_text())
    debug_rows=''.join(f'<tr><td>{esc(k)}</td><td>{esc(v)}</td><td>{esc(after["checkpoint_probes"][k])}</td></tr>' for k,v in before['checkpoint_probes'].items())
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MYCELIA adaptive memory phase</title><style>body{{font:16px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:0 24px;background:#f4f8fa;color:#243547}}table{{border-collapse:collapse;background:white;max-width:100%}}td,th{{padding:8px 12px;border-bottom:1px solid #dae3e9;text-align:left}}img{{width:100%;background:white}}code,pre{{background:#e5edf2;padding:8px;overflow:auto}}a{{color:#096c60}}h1,h2{{line-height:1.2}}</style>
    <h1>MYCELIA: debugging and candidate adaptive memory</h1><p><b>Debugging preceded implementation.</b> Resource-paid fast/slow traces passed the A→B→A behavioral check in 30/30 paired trials. Cord chemistry can now update at capped radius. The MRI validation comparison did not improve: {candidates[1]['validation']['accuracy']:.2%} versus the previous rule's {candidates[0]['validation']['accuracy']:.2%}. The candidate remains opt-in.</p>
    <p>This extends the existing prototype lattice. It does not implement a growing dimensional network or establish a fungal molecular memory law. The dimensional core remains separate and untrained.</p>
    <img src="adaptation-results.svg" alt="Cue-switch response, retention and separate MRI validation comparison">
    <h2>Debugging evidence</h2><table><tr><th>Probe</th><th>Before</th><th>After</th></tr>{debug_rows}</table>
    <p>The original suite passed 109 tests; the debugging gate passed 142 before adding adaptive state. Stress probes covered 540 memory operations and 180 dimensional physiology steps. <a href="debug-tests.txt">Debug gate test log</a> · <a href="tests-project.txt">Final project test log</a> · <a href="adaptive-stress.json">Adaptive randomized stress</a> · <a href="legacy-checkpoint-replay.json">Preserved-model replay</a>.</p>
    <h2>A→B→A assay</h2><p>Same cue pair, initial geometry, seed pools, feed and physiology schedule in each paired trial. Forty A exposures, twelve B, then twelve A. No medical labels enter this assay. B adaptation requires normalized mismatch ≤0.05; retention of A is measured before reteaching and requires ≤0.10. These are specified computational thresholds, not accuracy percentages. Synthesis costs differ.</p>
    <table><tr><th>Arm</th><th>Adapted to B</th><th>Adapted and retained A</th><th>Restricted B latency</th><th>A error after B</th><th>Synthesis carbon</th></tr>{arm_rows}</table>
    <p>Restricted latency codes failures within the 12-exposure horizon as 13, rather than implying measured adaptation on exposure 13. Dual versus fixed paired latency difference: {data['paired_differences']['legacy_fixed']['B_restricted_latency_control_minus_dual']['mean']:.1f} exposures, with bootstrap 95% interval {esc(data['paired_differences']['legacy_fixed']['B_restricted_latency_control_minus_dual']['bootstrap_95_interval'])}. Maximum budget error: {data['max_absolute_budget_error']:.3g}.</p>
    <img src="full-history.svg" alt="Full acquisition, switch and return time courses">
    <p>Fast state can overwrite B on return to A; this is a finite retention tradeoff, not perfect lifelong image memorization. The rules were designed to express fast response and slower persistence, and this synthetic assay tests their implementation. It does not establish biological fidelity.</p>
    <h2>Paid saturation fixture</h2><p>Capped structural radius and material stayed unchanged. New dual-channel marker traces changed on {data['saturation_probe']['dual_fast_slow']['changed_cord_traces']} cords; the old rule changed {data['saturation_probe']['legacy_fixed']['changed_cord_traces']}. Marker carbon/energy costs are included in the ledgers.</p>
    <h2>Separate MRI validation</h2><p>Each model learned the same 2,000 inner-Training images twice and predicted the same 1,082 Training-validation images. Testing was not evaluated. Rates were specified before validation. Dual state stores two channels per colony and has greater state capacity than single-channel controls; acquisition cost is reported. This result does not replace the previous model's 76.89% independent Testing score or 81.21% all-data training recall.</p>
    <table><tr><th>Candidate</th><th>Validation accuracy</th><th>Balanced accuracy</th></tr>{mri_rows}</table><p>Better synthetic adaptation did not yield a better MRI classifier. Patient independence and clinical validation remain absent.</p>
    <p><a href="mri-validation/summary.json">MRI metrics/configuration</a> · <a href="mri-validation/predictions.csv">All validation predictions</a> · <a href="mri-validation/adaptive-validation-memory.json">Validation-only adaptive model</a> · <a href="cue-switch/summary.json">Assay metrics</a> · <a href="cue-switch/per-trial.csv">Per-trial data</a> · <a href="cue-switch/trajectories.csv">Full trajectories</a>.</p>
    <h2>Reproduce</h2><pre>python -m pip install -e '.[images,dev]'
python tools/adaptation_test.py --output runs/adaptation
python tools/adaptive_medical_validation.py --dataset Brain-Tumor-MRI-Dataset --output runs/adaptive-mri-validation</pre>
    <p>See <a href="ADAPTIVE-MEMORY.md">mechanism specification</a> for equations, limits and biological sources. Growing sensing structure, fusion, physiological transport-based learning, ions/calcium and measured species calibration remain later gates.</p></html>'''
    (out/'report.html').write_text(text);print(out/'report.html')


if __name__=='__main__':main()
