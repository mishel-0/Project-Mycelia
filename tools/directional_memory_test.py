"""Phase 2: blinded-direction regrowth after fresh-medium inoculum transfer."""
from pathlib import Path
import argparse,csv,json,math,time
import numpy as np
from mycelia import Config,Environment,Mycelium
from mycelia.foraging import transfer_inoculum,direction_metrics
from mycelia.report import write_report

CONDITIONS=('memory','erased','bias_off','inheritance_off','rotated','sham')

def bootstrap(values,seed=42):
    x=np.asarray(values,dtype=float);rng=np.random.default_rng(seed)
    boot=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(axis=1)
    return {'mean':float(x.mean()),'ci95':[float(v) for v in np.quantile(boot,[.025,.975])]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--trials',type=int,default=30);p.add_argument('--train-steps',type=int,default=120);p.add_argument('--probe-steps',type=int,default=100);p.add_argument('--output',type=Path,default=Path('runs/directional-memory-test'))
    a=p.parse_args()
    if min(a.trials,a.train_steps,a.probe_steps)<1:raise ValueError('trial and step counts must be positive')
    a.output.mkdir(parents=True,exist_ok=True);start=time.monotonic();rows=[];problems=[];training_diagnostics=[]
    config=Config(max_nodes=350,max_tips=24,branch_interval=6,foraging_memory_rate=.08,foraging_memory_gain=1.2,foraging_memory_decay=.001,foraging_memory_cost=.01,foraging_memory_inheritance=.85)
    rng=np.random.default_rng(2026);angles=rng.uniform(0,2*math.pi,a.trials)
    for trial,angle in enumerate(angles):
        size=64;center=size//2;yy,xx=np.indices((size,size));bx=center+12*math.cos(angle);by=center+12*math.sin(angle)
        # Field follows one fixed protocol; model receives no bait angle or label.
        field=.2+1.6*np.exp(-((xx-bx)**2+(yy-by)**2)/(2*8**2))
        trained=Mycelium(Environment(field,diffusion=config.diffusion),config,seed=5000+trial,position=(center,center))
        trajectory=[]
        for step in range(a.train_steps):
            trained.step()
            if step+1 in (10,20,40,80,a.train_steps):
                root=trained.nodes[0];strength=math.hypot(root.polarity_trace_x,root.polarity_trace_y)
                gx,gy=trained.environment.gradient(root.x,root.y);g=math.hypot(gx,gy)
                trajectory.append({"step":step+1,"trace_strength":strength,"trace_to_bait_cosine":(root.polarity_trace_x*math.cos(angle)+root.polarity_trace_y*math.sin(angle))/strength if strength else None,"local_gradient_to_bait_cosine":(gx*math.cos(angle)+gy*math.sin(angle))/g if g else None,"nearest_bait_distance":min(math.hypot(n.x-bx,n.y-by) for n in trained.nodes.values() if n.alive)})
        training_diagnostics.append({"trial":trial,"bait_angle":float(angle),"trajectory":trajectory,"within_bait_core_radius3":trajectory[-1]["nearest_bait_distance"]<=3})
        sham=Mycelium(Environment(np.full((size,size),field.mean()),diffusion=config.diffusion),config,seed=5000+trial,position=(center,center));sham.run(a.train_steps)
        for condition in CONDITIONS:
            o=transfer_inoculum(trained,condition=condition,sham=sham,seed=9000+trial)
            assert len(o.nodes)==1 and not o.segments
            assert np.all(o.environment.nutrient==.6)
            # Probe arrays are fresh and initial directions independent of bait.
            metrics=o.run(a.probe_steps);direction=direction_metrics(o,float(angle))
            err=max(abs(metrics[k]) for k in ('carbon_error','water_error','energy_error'))
            assert err<1e-8
            if direction['alignment'] is None:problems.append({'trial':trial,'condition':condition,'reason':'no regrowth'})
            rows.append({'trial':trial,'condition':condition,'bait_angle':float(angle),'training_trace_strength':math.hypot(trained.nodes[0].polarity_trace_x,trained.nodes[0].polarity_trace_y),'probe_trace_strength':math.hypot(o.nodes[0].polarity_trace_x,o.nodes[0].polarity_trace_y),'carbon_error':metrics['carbon_error'],'water_error':metrics['water_error'],'energy_error':metrics['energy_error'],**direction})
            if trial==0:
                write_report(o,a.output/'examples'/condition)
                if condition=='memory':o.save(a.output/'example-regrown-state.json')
        if trial==0:write_report(trained,a.output/'examples/training');write_report(sham,a.output/'examples/sham-training')
        print(f'Completed {trial+1}/{a.trials} paired trials; all 6 regrowth arms',flush=True)
    result={'assay_scope':'Gradient-history regrowth test; Gaussian field supplies food at the inoculum from the start. Discrete wood-bait colonization is not required and this is not a replication of the published fungal transplantation experiment.','purpose':'Test an explicit candidate mechanism: resource-associated local polarity consolidation affects later growth after cue removal. Positive results validate implemented behavior, not a measured fungal molecular mechanism.','protocol':{'trials':a.trials,'training_steps':a.train_steps,'probe_steps':a.probe_steps,'seed':2026,'bait_direction':'randomized independently each trial','training_field':'.2 baseline + Gaussian bait, amplitude1.6, sigma8, distance12','probe_field':'fresh uniform .6 nutrient, water8; training medium discarded','retained_tissue':'root compartment only; all old geometry and tips discarded','matching':'identical trained-source physical resource pools across arms; sham substitutes only trace from uniform history','probe_memory_updates':'disabled; pre-existing traces can decay and be inherited','new_tip_directions':'paired random rotation of four buds; independent of bait','primary_outcome':'paired length-weighted cosine alignment: memory minus erased','inference':'10,000 paired bootstrap resamples; exploratory mechanism benchmark, no held-out tuning','config':config.as_dict()},'training_diagnostics':training_diagnostics,'bait_core_encounter_trials':sum(d['within_bait_core_radius3'] for d in training_diagnostics),'problems':problems,'conditions':{},'paired_differences':{},'max_absolute_budget_error':max(abs(r[k]) for r in rows for k in ('carbon_error','water_error','energy_error'))}
    for condition in CONDITIONS:
        group=[r for r in rows if r['condition']==condition and r['alignment'] is not None]
        result['conditions'][condition]={'valid_trials':len(group),'alignment':bootstrap([r['alignment'] for r in group]) if group else None,'forward_fraction_mean':float(np.mean([r['forward_length_fraction'] for r in group])) if group else None,'new_length_mean':float(np.mean([r['new_length'] for r in group])) if group else None}
    memory={r['trial']:r['alignment'] for r in rows if r['condition']=='memory'}
    for condition in CONDITIONS[1:]:
        paired=[memory[r['trial']]-r['alignment'] for r in rows if r['condition']==condition and memory[r['trial']] is not None and r['alignment'] is not None]
        result['paired_differences']['memory_minus_'+condition]={'valid_pairs':len(paired),**bootstrap(paired)} if paired else {'valid_pairs':0}
    primary=result['paired_differences']['memory_minus_erased']
    result['behavioral_check_passed']=primary.get('ci95',[-1,-1])[0]>0 and not problems
    result['elapsed_seconds']=time.monotonic()-start
    (a.output/'summary.json').write_text(json.dumps(result,indent=2))
    with (a.output/'per-trial.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
