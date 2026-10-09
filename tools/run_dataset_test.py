"""MRI-driven simulation validation, not tumor classification.
Run with Python containing mycelia, numpy and Pillow.
"""
from pathlib import Path
from collections import Counter
import argparse,csv,json,time
import numpy as np
from PIL import Image
from mycelia import Config, Environment, Mycelium
from mycelia.report import write_report

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,default=Path('/Users/misheladnan/Desktop/Project-Mycelia/Brain-Tumor-MRI-Dataset'))
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--steps',type=int,default=60)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.steps<1:raise ValueError('steps must be positive')
    start=time.monotonic();counts=Counter();invalid=[];files=sorted(a.dataset.glob('*/*/*.jpg'))
    if not files:raise ValueError('No dataset images found')
    for f in files:
     try:
      with Image.open(f) as im:im.load();assert im.width>0 and im.height>0
      counts['/'.join(f.relative_to(a.dataset).parts[:2])]+=1
     except Exception as e:invalid.append({'file':str(f),'error':str(e)})
    print(f'Image validation: {len(files)} files, {len(invalid)} failures',flush=True)
    rows=[];failures=[];representatives={}
    config=Config(max_nodes=150,max_tips=24)
    for i,f in enumerate([f for f in files if f.relative_to(a.dataset).parts[0]=='Testing']):
     label=f.parent.name
     try:
      with Image.open(f) as im:
       # Fixed intensity scale; no per-image min-max normalization.
       field=np.asarray(im.convert('L').resize((32,32),Image.Resampling.BOX),dtype=float)/255*.8
      o=Mycelium(Environment(field,diffusion=config.diffusion),config,seed=42)
      s=o.run(a.steps)
      assert max(abs(s[k]) for k in ('carbon_error','water_error','energy_error'))<1e-8
      assert all(np.all(np.isfinite(v)) for v in (o.environment.nutrient,o.environment.water))
      if label not in representatives:
       # Longer growth/branching/fusion exercise using the same image mapping.
       o.run(300-a.steps if a.steps<300 else 0)
       write_report(o,a.output/'examples'/label)
       representatives[label]={'file':str(f.relative_to(a.dataset)),**o.summary()}
      rows.append({'file':str(f.relative_to(a.dataset)),'label':label,'steps':a.steps,'nodes':s['nodes'],'active_tips':s['active_tips'],'length':s['total_length'],'branches':s['events'].get('branch',0),'fusions':s['events'].get('fusion',0),'flow':s['total_flow'],'carbon_error':s['carbon_error'],'water_error':s['water_error'],'energy_error':s['energy_error']})
     except Exception as e:failures.append({'file':str(f.relative_to(a.dataset)),'error':str(e)})
     if (i+1)%100==0:print(f'Tested {i+1}/1600; failures={len(failures)}',flush=True)
    if rows:
     with (a.output/'per-image-results.csv').open('w',newline='') as h:
      w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'purpose':'MRI intensity-driven substrate stress test; no classifier, predictions, accuracy or medical validation','mapping':'Pillow grayscale, BOX resize 32x32, nutrient=gray/255*0.8; water=8; seed=42; dt=0.5; no labels fed to simulator','counts':dict(counts),'images_validated':len(files),'invalid_images':invalid,'test_images_passed':len(rows),'simulation_failures':failures,'steps_per_test_image':a.steps,'elapsed_seconds':time.monotonic()-start,'representative_300_step_runs':representatives,'groups':{}}
    for label in sorted({r['label'] for r in rows}):
     group=[r for r in rows if r['label']==label]
     summary['groups'][label]={'count':len(group),'mean_nodes':float(np.mean([r['nodes'] for r in group])),'mean_length':float(np.mean([r['length'] for r in group])),'max_absolute_budget_error':max(abs(r[k]) for r in group for k in ('carbon_error','water_error','energy_error'))}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (a.output/'README.md').write_text('# Brain dataset test\n\nThis is an MRI-driven substrate test of the experimental MYCELIA simulator. It does not classify tumors. All images are decoded; all Testing images run independently for '+str(a.steps)+' steps using a fixed grayscale-to-nutrient mapping. Folder labels are used only for grouping results. Four representative images run for 300 steps.\n\nSee summary.json, per-image-results.csv and examples/*/report.html. Reproduce with `python run_dataset_test.py --dataset /path/to/Brain-Tumor-MRI-Dataset --output /path/to/results` after installing MYCELIA, NumPy and Pillow.\n')
    print(json.dumps({k:summary[k] for k in ('images_validated','test_images_passed','simulation_failures','elapsed_seconds','groups')},indent=2),flush=True)
    if invalid or failures:raise SystemExit(1)

if __name__=="__main__":
    main()
