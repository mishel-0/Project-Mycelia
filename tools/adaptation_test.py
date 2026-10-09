"""Paired A→B→A chemical-cue assay for candidate fast/slow graph memory.

This tests implemented plasticity and forgetting, not fungal biological fidelity,
physical nutrient foraging, MRI accuracy, or dimensional-core learning.
"""
from dataclasses import asdict
from pathlib import Path
import argparse,csv,json,time
import numpy as np
from mycelia import MemoryColony,MemoryConfig,image_cues,__version__


def configs(size):
    common=dict(size=size,colonies_per_label=1,physiology_interval=20)
    return {'legacy_fixed':MemoryConfig(**common),'legacy_consolidated':MemoryConfig(**common,consolidation=True),
            'dual_fast_slow':MemoryConfig(**common,adaptive=True),'fast_only':MemoryConfig(**common,adaptive=True,slow_rate=0),
            'slow_only':MemoryConfig(**common,adaptive=True,fast_rate=0)}


def bootstrap_difference(values,seed=2026):
    values=np.asarray(values,dtype=float);rng=np.random.default_rng(seed)
    draws=values[rng.integers(0,len(values),(10000,len(values)))].mean(axis=1)
    return {'mean':float(values.mean()),'bootstrap_95_interval':np.quantile(draws,[.025,.975]).tolist()}


def saturate(colony):
    """Paid fixture: build every cord to its cap from existing local resources."""
    o=colony.organism
    for e in o.segments.values():
        cost=2*(o.config.radius-e.radius);energy=cost*colony.config.energy_cost;a,b=o.nodes[e.a],o.nodes[e.b]
        if min(a.nutrient,b.nutrient)<cost/2 or min(a.energy,b.energy)<energy/2:raise ValueError('insufficient fixture resources')
        a.nutrient-=cost/2;b.nutrient-=cost/2;e.material+=cost;e.radius=o.config.radius
        a.energy-=energy/2;b.energy-=energy/2;o.memory_spent+=energy
    o._validate()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--trials',type=int,default=30);p.add_argument('--size',type=int,default=8)
    a=p.parse_args()
    if a.trials<2 or a.size<3:p.error('trials >=2 and size >=3 required')
    a.output.mkdir(parents=True,exist_ok=True);start=time.monotonic();rng=np.random.default_rng(2026)
    rows=[];curves=[];configurations=configs(a.size);largest_budget_error=0.
    for trial in range(a.trials):
        gray_a=(rng.random((a.size,a.size))>.45).astype(float)*.85+.05
        gray_b=(rng.random((a.size,a.size))>.45).astype(float)*.85+.05
        if np.array_equal(gray_a,gray_b):gray_b[0,0]=.05 if gray_a[0,0]>.5 else .9
        cue_a=image_cues(gray_a);cue_b=image_cues(gray_b);reference=float(np.mean((cue_a-cue_b)**2))
        for name,config in configurations.items():
            c=MemoryColony(config);cost=energy=0.;trajectory=[]
            for phase,cue,steps in [('A',cue_a,40),('B',cue_b,12),('A_return',cue_a,12)]:
                if phase=='B':before_b=c.distance(cue_b)/reference
                if phase=='A_return':retention_a=c.distance(cue_a)/reference;after_b=c.distance(cue_b)/reference
                for step in range(steps):
                    update=c.expose(cue);cost+=update['synthesis_carbon'];energy+=update['synthesis_energy']
                    trajectory.append({'trial':trial,'arm':name,'phase':phase,'phase_step':step+1,'exposure':len(trajectory)+1,
                                       'distance_A':c.distance(cue_a),'distance_B':c.distance(cue_b),'reference_AB':reference})
            b_values=[r['distance_B']/reference for r in trajectory if r['phase']=='B']
            b_latency=next((i+1 for i,v in enumerate(b_values) if v<=.05),None)
            a_values=[r['distance_A']/reference for r in trajectory if r['phase']=='A_return']
            a_latency=0 if retention_a<=.05 else next((i+1 for i,v in enumerate(a_values) if v<=.05),None)
            summary=c.organism.summary();error=max(abs(summary[k]) for k in ('carbon_error','water_error','energy_error'));largest_budget_error=max(largest_budget_error,error)
            rows.append({'trial':trial,'arm':name,'before_B_normalized':before_b,'after_B_normalized':after_b,'A_retention_after_B_normalized':retention_a,
                         'B_latency':b_latency,'B_restricted_latency':b_latency if b_latency is not None else 13,
                         'A_return_latency':a_latency,'adapts_and_retains':int(b_latency is not None and retention_a<=.1),
                         'synthesis_carbon':cost,'synthesis_energy':energy,'max_budget_error':error})
            curves.extend(trajectory)
            if trial==0 and name=='dual_fast_slow':
                c.freeze();c.state_dict();model=c.state_dict();(a.output/'example-colony.json').write_text(json.dumps(model,allow_nan=False))
                np.savez_compressed(a.output/'example-cues.npz',A=gray_a,B=gray_b)
        print(f'Completed paired A→B→A trial {trial+1}/{a.trials}, five arms',flush=True)
    result={'version':__version__,'scope':'Synthetic chemical-cue switch on a pre-inoculated legacy graph; computational adaptation test, not fungal validation, physical resource foraging or MRI evaluation.',
            'protocol':{'trials':a.trials,'seed':2026,'size':a.size,'phases':{'A':40,'B':12,'A_return':12},'paired_cues':True,
                        'matching':'Identical initial geometry, resource pools, feed=.02 per living compartment per lesson, reward=1 and physiology schedule; acquisition costs can differ.',
                        'distance':'Minimum direct graph mismatch among available funded slow/fast traces, divided by common per-trial cue A–B node MSE.',
                        'B_adaptation_threshold':.05,'A_retention_threshold':.1,'B_latency_censoring':'No threshold crossing within 12 lessons is None; restricted comparison codes these failures as 13.',
                        'A_return_latency':'Zero means A was retained before reteaching; otherwise first threshold crossing after re-exposure.',
                        'configs':{name:asdict(config) for name,config in configurations.items()}},'arms':{},'paired_differences':{},'checks':{}}
    for name in configurations:
        selected=[r for r in rows if r['arm']==name]
        result['arms'][name]={'trials':len(selected),'adapted_to_B':sum(r['B_latency'] is not None for r in selected),'adapted_and_retained_A':sum(r['adapts_and_retains'] for r in selected),
                             'mean_B_restricted_latency':float(np.mean([r['B_restricted_latency'] for r in selected])),
                             'mean_A_retention_after_B_normalized':float(np.mean([r['A_retention_after_B_normalized'] for r in selected])),
                             'mean_synthesis_carbon':float(np.mean([r['synthesis_carbon'] for r in selected])),'mean_synthesis_energy':float(np.mean([r['synthesis_energy'] for r in selected]))}
    dual=[r for r in rows if r['arm']=='dual_fast_slow']
    for name in ('legacy_fixed','legacy_consolidated','fast_only','slow_only'):
        control=[r for r in rows if r['arm']==name]
        result['paired_differences'][name]={'B_restricted_latency_control_minus_dual':bootstrap_difference([x['B_restricted_latency']-y['B_restricted_latency'] for x,y in zip(control,dual)]),
                                          'A_retention_error_control_minus_dual':bootstrap_difference([x['A_retention_after_B_normalized']-y['A_retention_after_B_normalized'] for x,y in zip(control,dual)])}
    # Real material/energy-funded radius saturation, then a new cue.
    result['saturation_probe']={}
    for name in ('legacy_fixed','dual_fast_slow'):
        c=MemoryColony(configurations[name]);c.expose(cue_a);saturate(c)
        before=[(e.route_trace,e.route_fast_trace,e.radius,e.material) for e in c.organism.segments.values()];update=c.expose(cue_b)
        edges=list(c.organism.segments.values());s=c.organism.summary()
        result['saturation_probe'][name]={'changed_cord_traces':sum((e.route_trace,e.route_fast_trace)!=v[:2] for e,v in zip(edges,before)),
                                         'physical_radius_and_material_unchanged':all((e.radius,e.material)==v[2:] for e,v in zip(edges,before)),
                                         'synthesis_carbon':update['synthesis_carbon'],'max_budget_error':max(abs(s[k]) for k in ('carbon_error','water_error','energy_error'))}
    result['checks']={'all_budgets_within_1e-8':largest_budget_error<1e-8,
                      'all_dual_trials_adapt_and_retain':all(r['adapts_and_retains'] for r in dual),
                      'dual_updates_cords_at_radius_cap':result['saturation_probe']['dual_fast_slow']['changed_cord_traces']>0,
                      'saturated_radius_unchanged':result['saturation_probe']['dual_fast_slow']['physical_radius_and_material_unchanged'],
                      'dual_faster_than_fixed_paired_interval':result['paired_differences']['legacy_fixed']['B_restricted_latency_control_minus_dual']['bootstrap_95_interval'][0]>0}
    result['max_absolute_budget_error']=largest_budget_error;result['elapsed_seconds']=time.monotonic()-start
    result['computational_behavior_check_passed']=all(result['checks'].values())
    (a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    for filename,data in [('per-trial.csv',rows),('trajectories.csv',curves)]:
        with (a.output/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print(json.dumps({k:result[k] for k in ('arms','checks','max_absolute_budget_error','elapsed_seconds')},indent=2),flush=True)


if __name__=='__main__':main()
