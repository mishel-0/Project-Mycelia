"""Exploratory Training-only MRI validation for a fixed adaptation candidate.

The independent Testing folder is not evaluated here. The physiological
adaptation assay specifies candidate rates before these validation results.
"""
import argparse,csv,hashlib,json,time
from pathlib import Path
from dataclasses import asdict
from copy import deepcopy
import numpy as np
from mycelia import MemoryConfig,__version__
try:
    from .train_mycelium import load_images,clean_splits,validation_split,fit,metrics,labels_for,state_hash,write_json
except ImportError:
    from train_mycelium import load_images,clean_splits,validation_split,fit,metrics,labels_for,state_hash,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);start=time.monotonic();records,cues=load_images(a.dataset)
    clean,audit=clean_splits(records);inner,validation,calibration=validation_split(records,clean['Training'],42,500)
    candidates=[('legacy_consolidated_32',MemoryConfig(colonies_per_label=32,consolidation=True)),
                ('dual_fast_slow_32',MemoryConfig(colonies_per_label=32,adaptive=True)),
                ('fast_only_32',MemoryConfig(colonies_per_label=32,adaptive=True,slow_rate=0))]
    result={'version':__version__,'scope':'Exploratory Training-only MRI validation; 2,000 inner-Training images, two passes; 1,082 validation images from clean Training. Independent Testing is not scored, and no all-data final model is fitted.',
            'protocol':{'seed':42,'epochs':2,'training':len(calibration),'validation':len(validation),'candidate_parameters':'Fixed before MRI validation, from the synthetic cue-switch implementation; no parameter optimization on Testing.',
                        'matching':'Same grid, feed, initial resource pools, per-label colonies and physiology schedule. Dual state adds stored channels and pays extra acquisition costs; this is not a matched count of stored templates.'},
            'audit':audit,'candidates':[],'checks':{}}
    all_pred={}
    for name,config in candidates:
        model,evidence=fit(records,cues,calibration,config,2,42,name,a.output);before=state_hash(model)
        prediction=model.predict_many(cues[validation]);all_pred[name]=prediction
        candidate={'name':name,'config':asdict(config),'learning':evidence,'validation':metrics(labels_for(records,validation),prediction)}
        result['checks'][name+'_frozen']=before==state_hash(model)
        if name=='dual_fast_slow_32':
            path=a.output/'adaptive-validation-memory.json';model.save(path)
            result['heldout']={'scope':'MRI validation model trained on 2,000 clean inner-Training images only, two passes. Not the final all-data model; independent Testing is not evaluated.','model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'metrics':candidate['validation']}
            from mycelia import AssociativeMycelium
            restored=AssociativeMycelium.load(path)
            result['checks']['saved_adaptive_predictions_match']=restored.predict_many(cues[validation])==prediction
            del restored
            erased=deepcopy(model)
            for pop in erased.colonies.values():
                for colony in pop:
                    for n in colony.organism.nodes.values():n.receptor_trace=[];n.receptor_variance=[];n.receptor_fast_trace=[]
            result['checks']['erasing_both_channels_abstains']=all(p is None for p in erased.predict_many(cues[validation]));del erased
        result['candidates'].append(candidate);write_json(a.output/'summary.json',result)
        print(name+' validation: '+str(candidate['validation']['accuracy']),flush=True);del model
    with (a.output/'predictions.csv').open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(['file','actual',*all_pred])
        for j,i in enumerate(validation):w.writerow([records[i]['file'],records[i]['label'],*[pred[j] for pred in all_pred.values()]])
    reference=result['candidates'][0]['validation']['balanced_accuracy'];adaptive=result['candidates'][1]['validation']['balanced_accuracy']
    result['adaptive_minus_legacy_balanced_accuracy']=adaptive-reference
    result['MRI_improvement_observed']=adaptive>reference
    result['checks']['budgets_within_1e-8']=all(c['learning']['max_absolute_budget_error']<1e-8 for c in result['candidates'])
    result['checks_passed']=all(result['checks'].values());result['elapsed_seconds']=time.monotonic()-start
    write_json(a.output/'summary.json',result)
    print(json.dumps({'scores':{c['name']:c['validation']['accuracy'] for c in result['candidates']},'checks_passed':result['checks_passed'],'elapsed_seconds':result['elapsed_seconds']},indent=2),flush=True)
    if not result['checks_passed']:raise RuntimeError('candidate integrity failed')


if __name__=='__main__':main()
