"""Evaluate learning stored in MYCELIA compartments/cords, without ML readouts."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import argparse,csv,hashlib,json,time
import numpy as np
from PIL import Image
from mycelia import AssociativeMycelium,MemoryConfig,image_cues

LABELS=['glioma','meningioma','notumor','pituitary']

def metrics(actual,predicted):
    matrix=np.zeros((4,4),dtype=int)
    for a,p in zip(actual,predicted):
        if p is not None:matrix[LABELS.index(a),LABELS.index(p)]+=1
    counts=Counter(actual);per_class={}
    for i,label in enumerate(LABELS):
        tp=matrix[i,i];recall=float(tp/max(counts[label],1));precision=float(tp/max(matrix[:,i].sum(),1))
        per_class[label]={'count':counts[label],'recall':recall,'precision':precision,'f1':2*precision*recall/max(precision+recall,1e-12)}
    return {'accuracy':float(np.mean(np.asarray(actual)==np.asarray(predicted,dtype=object))),'balanced_accuracy':float(np.mean([v['recall'] for v in per_class.values()])),'abstentions':sum(p is None for p in predicted),'confusion_labels':LABELS,'confusion_matrix':matrix.tolist(),'classes':per_class}

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--colonies',type=int,default=8)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);start=time.monotonic();records=[];config=MemoryConfig(colonies_per_label=a.colonies)
    for f in sorted(a.dataset.glob('*/*/*.jpg')):
        with Image.open(f) as im:
            rgb=im.convert('RGB');h=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            gray=np.asarray(rgb.convert('L').resize((config.size,config.size),Image.Resampling.BOX),dtype=float)/255
        records.append({'path':str(f.relative_to(a.dataset)),'split':f.relative_to(a.dataset).parts[0],'label':f.parent.name,'hash':h,'cue':image_cues(gray)})
    if not records:raise ValueError('no MRI images found')
    labels_by_hash={}
    for r in records:labels_by_hash.setdefault(r['hash'],set()).add(r['label'])
    conflicts={h for h,v in labels_by_hash.items() if len(v)>1};seen=set();train=[];test=[];overlap=0
    for r in records:
        if r['split']!='Training' or r['hash'] in conflicts or r['hash'] in seen:continue
        seen.add(r['hash']);train.append(r)
    train_hashes=set(seen);seen=set()
    for r in records:
        if r['split']!='Testing':continue
        if r['hash'] in train_hashes:overlap+=1;continue
        if r['hash'] in conflicts or r['hash'] in seen:continue
        seen.add(r['hash']);test.append(r)
    if not train or not test:raise ValueError('empty clean split')
    model=AssociativeMycelium(LABELS,config)
    order=np.random.default_rng(42).permutation(len(train));changed=0;synthesis=0
    for j,i in enumerate(order):
        r=train[i];update=model.learn(r['cue'],r['label']);changed+=update['changed_nodes'];synthesis+=update['synthesis_carbon']
        if (j+1)%200==0:print(f'Local graph learning: {j+1}/{len(train)} exposures',flush=True)
    model.freeze();model.save(a.output/'network-memory.json')
    restored=AssociativeMycelium.load(a.output/'network-memory.json')
    before=hashlib.sha256(json.dumps(restored.state_dict(),sort_keys=True).encode()).hexdigest()
    print('Training done; frozen recall on training and held-out images',flush=True)
    train_pred=[restored.predict(r['cue']) for r in train];pred=[restored.predict(r['cue']) for r in test]
    assert before==hashlib.sha256(json.dumps(restored.state_dict(),sort_keys=True).encode()).hexdigest(),'Recall mutated learned biological state'
    for r,y in zip(test[:10],pred[:10]):assert model.predict(r['cue'])==y,'Saved memory changed predictions'
    with (a.output/'predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['file','actual','predicted','correct']);w.writerows((r['path'],r['label'],y,int(r['label']==y)) for r,y in zip(test,pred))
    result={'mechanism':'Local resource-paid receptor imprints and real cord reinforcement within Mycelium node/edge state. Winner competition within label-rewarded colony populations; direct graph mismatch recall. This is an explicit bio-inspired prototype memory model, not proven fungal MRI cognition. No SVM, neural network, CNN, backpropagation or learned external image encoder.','protocol':'Fixed 16x16 raw grayscale + signed local gradients converted to chemical cues; one shuffled Training-only exposure pass, seed42; 8 colony imprints per class; frozen independent Testing recall; exact decoded-pixel duplicate filtering.','raw_counts':dict(Counter(r['split'] for r in records)),'clean_counts':{'Training':len(train),'Testing':len(test)},'exact_test_train_overlap':overlap,'conflicting_hashes':len(conflicts),'training':metrics([r['label'] for r in train],train_pred),'heldout':metrics([r['label'] for r in test],pred),'learning_evidence':{'node_updates':changed,'memory_synthesis_carbon':synthesis,'serialized_model_prediction_match':True,'frozen_recall_state_hash_unchanged':True},'ablations':{}}
    # Re-run the identical images with only local learned state selectively removed.
    cord_off=deepcopy(restored)
    for pop in cord_off.colonies.values():
        for colony in pop:
            for e in colony.organism.segments.values():e.route_trace=0;e.radius=colony.organism.config.radius*.7
    result['ablations']['no_learned_cord_traces']=metrics([r['label'] for r in test],[cord_off.predict(r['cue']) for r in test])
    memory_off=deepcopy(restored)
    for pop in memory_off.colonies.values():
        for colony in pop:
            for n in colony.organism.nodes.values():n.receptor_trace=[];n.receptor_variance=[];n.memory_exposures=0
    result['ablations']['no_receptor_memory']=metrics([r['label'] for r in test],[memory_off.predict(r['cue']) for r in test])
    majority=Counter(r['label'] for r in train).most_common(1)[0][0]
    result['ablations']['majority_baseline']=metrics([r['label'] for r in test],[majority]*len(test))
    summaries=[colony.organism.summary() for pop in model.colonies.values() for colony in pop]
    result['learning_evidence']['max_absolute_budget_error']=max(abs(s[k]) for s in summaries for k in ('carbon_error','water_error','energy_error'))
    result['learning_evidence']['total_memory_material']=sum(s['memory_material'] for s in summaries)
    result['colony_graphs']=[{'label':label,'colony':i,'exposures':colony.exposures,**colony.organism.summary()} for label,pop in model.colonies.items() for i,colony in enumerate(pop)]
    result['elapsed_seconds']=time.monotonic()-start
    (a.output/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('clean_counts','training','heldout','learning_evidence','ablations','elapsed_seconds')},indent=2),flush=True)
if __name__=='__main__':main()
