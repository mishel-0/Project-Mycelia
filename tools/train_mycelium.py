"""Train actual compartment/cord memory; separate held-out testing from all-data fit.

Only NumPy and Pillow are needed. Labels reward a colony population; recognition
is the fixed graph mismatch in mycelia.memory, with no external learned readout.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,time
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import numpy as np
from PIL import Image
from mycelia import AssociativeMycelium,MemoryConfig,image_cues,__version__

LABELS=('glioma','meningioma','notumor','pituitary')


def write_json(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)


def load_images(dataset,size=16):
    records=[];cues=[]
    files=sorted(p for p in dataset.glob('*/*/*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'})
    if not files:raise ValueError('no images found in Training/Testing class directories')
    for path in files:
        relative=path.relative_to(dataset)
        split,label=relative.parts[:2]
        if split not in ('Training','Testing') or label not in LABELS:raise ValueError(f'unknown split or label: {relative}')
        with Image.open(path) as im:
            rgb=im.convert('RGB')
            source_hash=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            gray=np.asarray(rgb.convert('L').resize((size,size),Image.Resampling.BOX),dtype=np.uint8)
        cue_hash=hashlib.sha256(str(gray.shape).encode()+gray.tobytes()).hexdigest()
        records.append({'file':str(relative),'split':split,'label':label,'source_sha256':source_hash,'cue_sha256':cue_hash})
        cues.append(image_cues(gray.astype(float)/255))
    return records,np.stack(cues)


def clean_splits(records):
    source_labels={};cue_labels={}
    for r in records:
        source_labels.setdefault(r['source_sha256'],set()).add(r['label'])
        cue_labels.setdefault(r['cue_sha256'],set()).add(r['label'])
    bad_source={h for h,v in source_labels.items() if len(v)>1}
    bad_cue={h for h,v in cue_labels.items() if len(v)>1}
    clean={};removed=Counter();seen_source=set();seen_cue=set()
    # Track all Training hashes, including duplicate rows, to block hidden overlap.
    train_source={r['source_sha256'] for r in records if r['split']=='Training'}
    train_cue={r['cue_sha256'] for r in records if r['split']=='Training'}
    for split in ('Training','Testing'):
        seen_source=set();seen_cue=set();indices=[]
        for i,r in enumerate(records):
            if r['split']!=split:continue
            hs,hc=r['source_sha256'],r['cue_sha256']
            if hs in bad_source or hc in bad_cue:removed[split+'_label_conflict']+=1;continue
            if split=='Testing' and hs in train_source:removed['Testing_source_overlap']+=1;continue
            if split=='Testing' and hc in train_cue:removed['Testing_additional_encoded_overlap']+=1;continue
            if hs in seen_source or hc in seen_cue:removed[split+'_within_split_duplicate']+=1
            else:indices.append(i)
            seen_source.add(hs);seen_cue.add(hc)
        clean[split]=np.array(indices,dtype=int)
    if any(not len(v) for v in clean.values()):raise ValueError('empty clean training or held-out split')
    return clean,{'removed':dict(removed),'source_label_conflicts':len(bad_source),'encoded_label_conflicts':len(bad_cue)}


def validation_split(records,train,seed=42,limit=500):
    rng=np.random.default_rng(seed);inner=[];validation=[];calibration=[]
    for label in LABELS:
        indices=np.array([i for i in train if records[i]['label']==label],dtype=int)
        rng.shuffle(indices);n=max(1,int(round(.2*len(indices))))
        if len(indices)<=n:raise ValueError(f'insufficient Training images for {label}')
        validation.extend(indices[:n]);inner.extend(indices[n:]);calibration.extend(indices[n:n+limit])
    return np.array(inner,dtype=int),np.array(validation,dtype=int),np.array(calibration,dtype=int)


def metrics(actual,predicted):
    if len(actual)!=len(predicted) or not len(actual):raise ValueError('invalid metric sample')
    matrix=np.zeros((len(LABELS),len(LABELS)),dtype=int);classes={}
    for actual_label,predicted_label in zip(actual,predicted):
        if predicted_label is not None:matrix[LABELS.index(actual_label),LABELS.index(predicted_label)]+=1
    counts=Counter(actual)
    for j,label in enumerate(LABELS):
        tp=int(matrix[j,j]);recall=tp/max(counts[label],1);precision=tp/max(int(matrix[:,j].sum()),1)
        classes[label]={'count':counts[label],'recall':recall,'precision':precision,'f1':2*recall*precision/max(recall+precision,1e-12)}
    correct=sum(a==p for a,p in zip(actual,predicted));n=len(actual);accuracy=correct/n
    z=1.959963984540054;denominator=1+z*z/n
    center=(accuracy+z*z/(2*n))/denominator
    half=z*math.sqrt(accuracy*(1-accuracy)/n+z*z/(4*n*n))/denominator
    return {'n':n,'correct':correct,'accuracy':accuracy,'accuracy_wilson_95_interval':[center-half,center+half],'balanced_accuracy':float(np.mean([v['recall'] for v in classes.values()])),
            'abstentions':sum(p is None for p in predicted),'labels':LABELS,'confusion_matrix':matrix.tolist(),'classes':classes}


def fit(records,cues,indices,config,epochs,seed,phase,output):
    model=AssociativeMycelium(LABELS,config);rng=np.random.default_rng(seed)
    evidence={'exposures':0,'changed_nodes':0,'synthesis_carbon':0.,'synthesis_energy':0.}
    for epoch in range(epochs):
        for step,i in enumerate(rng.permutation(indices)):
            r=records[i];update=model.learn(cues[i],r['label'])
            evidence['exposures']+=1
            for key in ('changed_nodes','synthesis_carbon','synthesis_energy'):evidence[key]+=update[key]
            if (step+1)%250==0 or step+1==len(indices):
                progress={'phase':phase,'epoch':epoch+1,'epochs':epochs,'image':step+1,'images_per_epoch':len(indices),'total_exposures':evidence['exposures']}
                write_json(output/'progress.json',progress)
                print(f'{phase}: epoch {epoch+1}/{epochs}, {step+1}/{len(indices)} images',flush=True)
    model.freeze()
    summaries=[z.organism.summary() for pop in model.colonies.values() for z in pop]
    evidence.update({'colonies':{k:len(v) for k,v in model.colonies.items()},'total_memory_material':sum(s['memory_material'] for s in summaries),
                     'max_absolute_budget_error':max(abs(s[k]) for s in summaries for k in ('carbon_error','water_error','energy_error'))})
    return model,evidence


def state_hash(model):return hashlib.sha256(json.dumps(model.state_dict(),sort_keys=True,allow_nan=False).encode()).hexdigest()


def labels_for(records,indices):return [records[i]['label'] for i in indices]


def write_predictions(path,records,indices,predictions,scores):
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['file','source_split','actual','prediction','correct',*[label+'_mismatch' for label in LABELS]])
        for i,p,s in zip(indices,predictions,scores):
            r=records[i];writer.writerow([r['file'],r['split'],r['label'],p,int(p==r['label']),*s])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=2);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--selection-per-class',type=int,default=500)
    args=parser.parse_args()
    if args.epochs<1 or args.selection_per_class<1:parser.error('epochs and selection-per-class must be positive')
    args.output.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    records,cues=load_images(args.dataset);clean,audit=clean_splits(records)
    inner,validation,calibration=validation_split(records,clean['Training'],args.seed,args.selection_per_class)
    print(f'Loaded {len(records)} images; clean Training={len(clean["Training"])}, held-out Testing={len(clean["Testing"])}; validation={len(validation)}',flush=True)
    with (args.output/'image-manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    result={'version':__version__,'dataset':str(args.dataset.resolve()),'raw_counts':dict(Counter(r['split'] for r in records)),
            'clean_counts':{k:len(v) for k,v in clean.items()},'audit':audit,'seed':args.seed,'epochs':args.epochs,
            'encoder':'Fixed 16x16 grayscale BOX resize; grayscale and two signed local gradients as three chemical cue channels.',
            'mechanism':'Actual resource-funded compartment receptor traces and cord reinforcement in AssociativeMycelium. Direct graph mismatch recall; no CNN, neural network, backpropagation, learned encoder or external trained readout.',
            'limits':'Bio-inspired prototype hypotheses, not established fungal image cognition. Image-level split; patient identity absent. Research only; not clinically validated. Dimensional mycelia.biology.Hypha is a separate, untrained module.',
            'selection_protocol':{'selection_pool':len(calibration),'inner_training_pool':len(inner),'validation':len(validation),'maximum_selection_images_per_class':args.selection_per_class,
                                  'rule':'Highest Training-only validation balanced accuracy; break ties by fewer colonies. Testing labels never used for configuration selection.'},
            'candidates':[]}
    candidates=[('8_fixed',MemoryConfig(colonies_per_label=8)),('16_consolidated',MemoryConfig(colonies_per_label=16,consolidation=True)),('32_consolidated',MemoryConfig(colonies_per_label=32,consolidation=True))]
    for name,config in candidates:
        model,evidence=fit(records,cues,calibration,config,args.epochs,args.seed,'validation/'+name,args.output)
        prediction=model.predict_many(cues[validation])
        result['candidates'].append({'name':name,'config':asdict(config),'validation':metrics(labels_for(records,validation),prediction),'learning':evidence})
        write_json(args.output/'selection.json',result)
        print(name+' validation accuracy: '+str(result['candidates'][-1]['validation']['accuracy']),flush=True)
        del model
    chosen=max(result['candidates'],key=lambda c:(c['validation']['balanced_accuracy'],-c['config']['colonies_per_label']))
    result['selected_candidate']=chosen['name'];config=MemoryConfig(**chosen['config'])
    model,evidence=fit(records,cues,clean['Training'],config,args.epochs,args.seed,'heldout-model',args.output)
    path=args.output/'heldout-memory.json';model.save(path)
    before=state_hash(model);test=clean['Testing'];test_scores=model.score_many(cues[test]);test_pred=model.predict_many(cues[test])
    result['heldout']={'scope':'Frozen model trained on clean Training only. Exact source and actual encoded-input overlap removed from Testing.',
                      'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'learning':evidence,'metrics':metrics(labels_for(records,test),test_pred),
                      'training_recall':metrics(labels_for(records,clean['Training']),model.predict_many(cues[clean['Training']]))}
    result['checks']={'heldout_frozen_state_unchanged':before==state_hash(model)}
    sample=test[:min(32,len(test))]
    scalar=np.array([list(model.scores(cues[i]).values()) for i in sample])
    result['checks']['batch_scalar_max_error']=float(np.max(np.abs(scalar-model.score_many(cues[sample]))))
    restored=AssociativeMycelium.load(path)
    result['checks']['heldout_save_load_predictions_identical']=restored.predict_many(cues[test])==test_pred
    del restored
    write_predictions(args.output/'heldout-predictions.csv',records,test,test_pred,test_scores)
    ablated=deepcopy(model)
    for pop in ablated.colonies.values():
        for colony in pop:
            for e in colony.organism.segments.values():e.route_trace=0;e.radius=colony.organism.config.radius*.7
    result['heldout']['no_learned_cord_traces']=metrics(labels_for(records,test),ablated.predict_many(cues[test]))
    del ablated
    ablated=deepcopy(model)
    for pop in ablated.colonies.values():
        for colony in pop:
            for n in colony.organism.nodes.values():n.receptor_trace=[];n.receptor_variance=[];n.memory_exposures=0
    erased=ablated.predict_many(cues[test])
    result['heldout']['no_receptor_memory']=metrics(labels_for(records,test),erased)
    result['checks']['erasing_receptors_abstains_every_image']=all(p is None for p in erased)
    empty=AssociativeMycelium(LABELS,config)
    result['checks']['untrained_model_abstains_every_image']=all(p is None for p in empty.predict_many(cues[test]))
    majority=Counter(labels_for(records,clean['Training'])).most_common(1)[0][0]
    result['heldout']['majority_baseline']=metrics(labels_for(records,test),[majority]*len(test))
    write_json(args.output/'heldout-results.json',result)
    print('Held-out result: '+str(result['heldout']['metrics']['accuracy']),flush=True)
    del model,ablated
    # Explicitly separate final all-data learning: Testing is now included, so its
    # subsequent predictions measure training recall, never held-out accuracy.
    if audit['source_label_conflicts'] or audit['encoded_label_conflicts']:
        raise ValueError('all-data fit stopped: conflicting labels need resolution; held-out results preserved')
    all_indices=np.arange(len(records))
    model,evidence=fit(records,cues,all_indices,config,args.epochs,args.seed,'full-data-model',args.output)
    path=args.output/'full-data-memory.json';model.save(path);before=state_hash(model)
    scores=model.score_many(cues);prediction=model.predict_many(cues)
    result['full_data']={'scope':'Final fit on every valid original Training and Testing row, including repeated images. Metrics are resubstitution/training recall, not held-out generalization.',
                         'images':len(records),'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'learning':evidence,
                         'training_recall':metrics(labels_for(records,all_indices),prediction)}
    write_predictions(args.output/'full-data-recall-predictions.csv',records,all_indices,prediction,scores)
    result['checks']['full_data_frozen_state_unchanged']=state_hash(model)==before
    restored=AssociativeMycelium.load(path)
    result['checks']['full_data_save_load_predictions_identical']=restored.predict_many(cues)==prediction
    result['checks']['all_resource_budgets_within_1e-8']=max(result['heldout']['learning']['max_absolute_budget_error'],evidence['max_absolute_budget_error'])<1e-8
    result['checks']['all_valid_images_used_for_final_training']=evidence['exposures']==len(records)*args.epochs
    result['elapsed_seconds']=time.monotonic()-start
    result['checks_passed']=all(v for k,v in result['checks'].items() if k!='batch_scalar_max_error') and result['checks']['batch_scalar_max_error']<1e-12
    write_json(args.output/'summary.json',result);write_json(args.output/'progress.json',{'phase':'complete','checks_passed':result['checks_passed']})
    print(json.dumps({'selected':result['selected_candidate'],'heldout':result['heldout']['metrics'],'full_data_recall':result['full_data']['training_recall'],'checks':result['checks'],'elapsed_seconds':result['elapsed_seconds']},indent=2),flush=True)
    if not result['checks_passed']:raise RuntimeError('training integrity check failed')


if __name__=='__main__':main()
