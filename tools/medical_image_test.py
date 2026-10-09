"""Medical-image experiment for the dimensional core; external readouts are explicit.

No CNN, neural network, sklearn, SVM or learned encoder. The core is NOT trained.
A train-only NumPy nearest-centroid readout measures image-label information in
its physiological response. This is image-level research evaluation only.
"""
from pathlib import Path
from collections import Counter
from dataclasses import replace, asdict
import argparse, csv, hashlib, json, time, shutil
import numpy as np
from mycelia.biology import Hypha, Parameters
from mycelia.biology.images import (load_image, encode_image, new_image_hypha, apply_pulses,
                                    OBSERVABLES, SCALES, PROTOCOL, protocol_hash)

LABELS = ['glioma','meningioma','notumor','pituitary']


def implementation_fingerprint():
    import mycelia.biology.images as adapter
    base = Path(adapter.__file__).parent
    hashes = {name: hashlib.sha256((base/name).read_bytes()).hexdigest()
              for name in ('state.py','engine.py','mechanisms.py','images.py')}
    return hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()


def freeze_model(model):
    for key in ('mean','scale','centroids'):
        model[key] = np.asarray(model[key], dtype=float)
        model[key].setflags(write=False)
    return model


def fit_centroids(x, labels):
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels)
    if x.ndim != 2 or len(x) != len(labels) or not len(x) or not np.all(np.isfinite(x)) or set(labels) != set(LABELS):
        raise ValueError('finite training matrix and all four label classes are required')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1
    standardized = (x-mean)/scale
    centroids = np.array([standardized[labels==label].mean(axis=0) for label in LABELS])
    return freeze_model(dict(mean=mean, scale=scale, centroids=centroids, labels=LABELS.copy()))


def predict(model, x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(model['mean']) or not np.all(np.isfinite(x)):
        raise ValueError('invalid readout inputs')
    normalized = (x-model['mean'])/model['scale']
    distances = np.sum((normalized[:,None,:]-model['centroids'][None,:,:])**2,axis=2)
    return np.asarray(model['labels'])[np.argmin(distances,axis=1)]


def model_dict(model):
    return {key: value.tolist() if isinstance(value,np.ndarray) else value for key,value in model.items()}


def metrics(actual, predicted):
    actual,predicted = np.asarray(actual),np.asarray(predicted)
    matrix=np.zeros((4,4),dtype=int)
    for truth,guess in zip(actual,predicted):
        matrix[LABELS.index(str(truth)),LABELS.index(str(guess))]+=1
    n=len(actual);correct=int(np.trace(matrix));accuracy=correct/n
    z=1.959963984540054
    center=(accuracy+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(accuracy*(1-accuracy)/n+z*z/(4*n*n))/(1+z*z/n)
    per_class={}
    for i,label in enumerate(LABELS):
        recall=float(matrix[i,i]/max(matrix[i].sum(),1));precision=float(matrix[i,i]/max(matrix[:,i].sum(),1))
        per_class[label]=dict(count=int(matrix[i].sum()),recall=recall,precision=precision,
                              f1=2*recall*precision/max(recall+precision,1e-12))
    return dict(accuracy=accuracy,correct=correct,count=n,accuracy_95pct_Wilson_CI=[float(center-half),float(center+half)],
                balanced_accuracy=float(np.mean([r['recall'] for r in per_class.values()])),
                confusion_labels=LABELS,confusion_matrix=matrix.tolist(),classes=per_class)


def clean_splits(records):
    labels_by_hash={}
    for r in records: labels_by_hash.setdefault(r['hash'],set()).add(r['label'])
    conflicts={key for key,value in labels_by_hash.items() if len(value)>1}
    seen=set();train=[];test=[];overlap=0;duplicates_train=0;duplicates_test=0
    for i,r in enumerate(records):
        if r['split']!='Training' or r['hash'] in conflicts: continue
        if r['hash'] in seen:duplicates_train+=1;continue
        seen.add(r['hash']);train.append(i)
    train_hashes=set(seen);seen=set()
    for i,r in enumerate(records):
        if r['split']!='Testing':continue
        if r['hash'] in conflicts:continue
        if r['hash'] in train_hashes:overlap+=1;continue
        if r['hash'] in seen:duplicates_test+=1;continue
        seen.add(r['hash']);test.append(i)
    return np.asarray(train,dtype=int),np.asarray(test,dtype=int),dict(exact_test_train_overlap=overlap,
           within_training_duplicates=duplicates_train,within_testing_duplicates=duplicates_test,
           conflicting_pixel_hashes=len(conflicts),pixel_hash_definition='SHA256(decoded RGB dimensions + bytes)')



def encoded_records(records, images):
    """Exact collisions in the actual quantized stimulus, not just source pixels."""
    if len(records) != len(images):
        raise ValueError('encoded input count differs')
    return [dict(r, hash=hashlib.sha256(np.rint(np.asarray(gray)*255).astype(np.uint8).tobytes()).hexdigest())
            for r,gray in zip(records,images)]


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def controls(gray, expected):
    original=encode_image(gray,return_model=True)
    repeat=encode_image(gray)
    pairs=gray.ravel().reshape(-1,2)
    resumed=new_image_hypha()
    first=apply_pulses(resumed,pairs[:16]);resumed=Hypha.from_state_dict(json.loads(json.dumps(resumed.state_dict(),allow_nan=False)))
    second=apply_pulses(resumed,pairs[16:])
    shuffled=np.random.default_rng(42).permutation(gray.ravel()).reshape(8,8)
    altered=encode_image(shuffled)
    constant=encode_image(np.full((8,8),gray.mean()))
    refined=encode_image(gray,replace(Parameters(),max_substep_min=.005))
    off=replace(Parameters(),uptake_density=0,membrane_permeability=0,metabolism_per_min=0,
                maintenance_density=0,reserve_release_per_min=0,vesicle_synthesis_per_min=0,
                motor_delivery_per_min=0,exocytosis_per_min=0,diffusion_um2_min=0,wall_extensibility=0)
    disabled=encode_image(gray,off);disabled_black=encode_image(np.zeros((8,8)),off)
    normal=original['features'].reshape(-1,len(OBSERVABLES))
    delta=(refined['features'].reshape(normal.shape)-normal)/SCALES
    return dict(repeat_identical=bool(np.array_equal(original['features'],repeat['features'])),
                cache_matches_recomputation=bool(np.array_equal(expected,original['features'])),
                restart_identical=bool(np.array_equal(np.vstack([first,second]).ravel(),original['features'])),
                shuffle_preserves_intensity_histogram=bool(np.array_equal(np.sort(shuffled.ravel()),np.sort(gray.ravel()))),
                normalized_shuffle_response_RMS=float(np.sqrt(np.mean(((altered['features'].reshape(normal.shape)-normal)/SCALES)**2))),
                normalized_constant_mean_response_RMS=float(np.sqrt(np.mean(((constant['features'].reshape(normal.shape)-normal)/SCALES)**2))),
                refinement_normalized_RMS=float(np.sqrt(np.mean(delta**2))),
                refinement_normalized_max=float(np.max(np.abs(delta))),
                disabled_biology_input_insensitive=bool(np.array_equal(disabled['features'],disabled_black['features'])),
                no_water_limiter=all(r['limiter_events']==0 for r in [original,repeat,altered,constant,refined,disabled]),
                max_budget_error=max(r['maximum_budget_error'] for r in [original,repeat,altered,constant,refined,disabled])),original['model']


def run(dataset,output):
    dataset,output=Path(dataset),Path(output);output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();files=sorted(dataset.glob('*/*/*.jpg'))
    if not files: raise ValueError('No JPG images in Training/Testing class folders')
    records=[];invalid=[]
    for path in files:
        rel=path.relative_to(dataset)
        if rel.parts[0] not in {'Training','Testing'} or rel.parts[1] not in LABELS:raise ValueError(f'unexpected dataset folder: {rel}')
        try:
            gray,digest=load_image(path)
            records.append(dict(path=str(rel),split=rel.parts[0],label=rel.parts[1],hash=digest,gray=gray))
        except Exception as exc:
            invalid.append(dict(path=str(rel),error=str(exc)))
    print(f'Decoded {len(records)}/{len(files)} images; failures={len(invalid)}',flush=True)
    if invalid:
        (output/'image-failures.json').write_text(json.dumps(invalid,indent=2))
        raise ValueError('invalid images; full dataset protocol aborted')
    train,test,dedup=clean_splits(records)
    if not len(train) or not len(test): raise ValueError('no independent clean split')
    print(f'Clean splits: {len(train)} Training / {len(test)} Testing; exact overlap removed={dedup["exact_test_train_overlap"]}',flush=True)
    implementation_hash=implementation_fingerprint()
    signature=hashlib.sha256((protocol_hash()+implementation_hash+''.join(r['path']+r['hash'] for r in records)).encode()).hexdigest()
    cache=output/'features.npz';cache_reused=cache.exists();rows=[];failures=[]
    if cache.exists():
        with np.load(cache,allow_pickle=False) as d:
            if str(d['signature'])!=signature:raise ValueError('feature cache does not match decoded pixels and exact protocol')
            physiology=d['physiology'];image=d['image'];rows=json.loads(str(d['rows_json']))
        if physiology.shape!=(len(records),256) or not np.all(np.isfinite(physiology)):raise ValueError('invalid physiology cache')
        print('Verified content/protocol-matched feature cache',flush=True)
    else:
        feature_rows=[];image=[]
        for i,r in enumerate(records):
            try:
                encoded=encode_image(r['gray'])
                feature_rows.append(encoded['features']);image.append(r['gray'].ravel())
                final=encoded['final']
                rows.append(dict(file=r['path'],split=r['split'],label=r['label'],
                    duration_min=final['time_min'],extension_um=final['permanent_extension_um'],
                    final_turgor_MPa=final['tip_pressure_MPa'],final_ATP_pmol=final['tip_ATP_pmol'],
                    final_apical_cargo_pmol=final['tip_apical_cargo_pmol'],final_flow_pL_min=final['flow_toward_tip_pL_min'],
                    max_budget_error=encoded['maximum_budget_error'],limiter_events=encoded['limiter_events']))
            except Exception as exc:
                failures.append(dict(path=r['path'],error=str(exc)))
            if (i+1)%400==0:print(f'New-core MRI processing: {i+1}/{len(records)}; failures={len(failures)}',flush=True)
        if failures:
            (output/'simulation-failures.json').write_text(json.dumps(failures,indent=2))
            raise ArithmeticError('physiology failure; fitting aborted rather than dropping failed images')
        physiology=np.asarray(feature_rows);image=np.asarray(image)
        np.savez_compressed(cache,signature=signature,physiology=physiology,image=image,
                            paths=np.asarray([r['path'] for r in records]),pixel_hashes=np.asarray([r['hash'] for r in records]),
                            rows_json=json.dumps(rows))
    if not np.array_equal(image,np.asarray([r['gray'].ravel() for r in records])):
        raise ValueError('cached encoded stimuli differ from freshly decoded inputs')
    source_pixel_counts=dict(Training=len(train),Testing=len(test))
    raw_test=set(int(i) for i in test)
    encoded=encoded_records(records,image)
    train,test,encoded_dedup=clean_splits(encoded)
    encoded_dedup['pixel_hash_definition']='SHA256(fixed 8x8 uint8 grayscale stimulus)'
    if not len(train) or not len(test):raise ValueError('no independent encoded split')
    print(f'Exact 8x8 input filtering: {len(train)} Training / {len(test)} Testing',flush=True)
    additional_encoded_exclusions=len(raw_test-set(int(i) for i in test))
    write_csv(output/'per-image-physiology.csv',rows)
    actual=np.asarray([r['label'] for r in records]);predictions={};results={};models={}
    matrices=dict(physiology_only=physiology,image_only=image,image_plus_physiology=np.c_[image,physiology],
                  mean_intensity_only=image.mean(axis=1,keepdims=True))
    for name,x in matrices.items():
        model=fit_centroids(x[train],actual[train]);models[name]=model
        before=json.dumps(model_dict(model),sort_keys=True)
        guesses=predict(model,x[test]);predictions[name]=guesses
        assert before==json.dumps(model_dict(model),sort_keys=True),'readout changed during test predictions'
        results[name]=dict(training=metrics(actual[train],predict(model,x[train])),heldout=metrics(actual[test],guesses),
                           features=x.shape[1],frozen_test_state_unchanged=True)
        print(name,results[name]['heldout']['accuracy'],flush=True)
        write_csv(output/(name+'-predictions.csv'),[dict(file=records[int(i)]['path'],actual=str(actual[i]),predicted=str(y),correct=int(actual[i]==y)) for i,y in zip(test,guesses)])
    majority=Counter(actual[train]).most_common(1)[0][0]
    results['majority_baseline']=dict(heldout=metrics(actual[test],[majority]*len(test)),prediction=str(majority))
    shuffled_labels=np.random.default_rng(42).permutation(actual[train])
    permuted=fit_centroids(physiology[train],shuffled_labels)
    results['permuted_training_labels']=dict(heldout=metrics(actual[test],predict(permuted,physiology[test])),seed=42,
                                           scope='Single negative-control permutation; not a significance test')
    serialized=dict(schema='mycelia.external-image-centroid.v1',protocol=PROTOCOL,parameters=asdict(Parameters()),
                    model=model_dict(models['physiology_only']),protocol_hash=protocol_hash(),implementation_hash=implementation_hash,scope='External numerical readout. No biological learning law.')
    (output/'external-physiology-readout.json').write_text(json.dumps(serialized,indent=2))
    restored=freeze_model(json.loads((output/'external-physiology-readout.json').read_text())['model'])
    assert np.array_equal(predict(restored,physiology[test]),predictions['physiology_only'])
    representatives={};control_results=[]
    for label in LABELS:
        indexes=[int(i) for i in test if records[int(i)]['label']==label]
        for position,i in enumerate(indexes[:2]):
            r=records[i];control,h=controls(r['gray'],physiology[i]);control['file']=r['path'];control['label']=label;control_results.append(control)
            if position==0:
                destination=output/'examples'/label;destination.mkdir(parents=True,exist_ok=True)
                h.save(destination/'state.json')
                shutil.copy2(dataset/r['path'],destination/'input.jpg')
                write_csv(destination/'timeseries.csv',[{key:row[key] for key in ('time_min',*OBSERVABLES)} for row in h.history])
                representatives[label]=dict(file=r['path'],predicted=str(predictions['physiology_only'][list(test).index(i)]),actual=label,
                                            correct=bool(predictions['physiology_only'][list(test).index(i)]==label),final=h.summary())
    comparisons={}
    for name in ['physiology_only','image_plus_physiology']:
        differences=(predictions[name]==actual[test]).astype(float)-(predictions['image_only']==actual[test]).astype(float)
        rng=np.random.default_rng(42)
        samples=np.mean(differences[rng.integers(0,len(test),size=(2000,len(test)))],axis=1)
        comparisons[name+'_minus_image_only']=dict(accuracy_difference=float(differences.mean()),paired_bootstrap_95pct_CI=np.quantile(samples,[.025,.975]).tolist(),bootstrap_replicates=2000,
                    scope='Image-level sampling; patient correlation is not accounted for')
    checks=dict(all_images_decode=len(records)==len(files),all_images_simulated=len(rows)==len(records),
                labels_excluded_from_physiology=True,clean_train_test_hashes_disjoint=not ({records[int(i)]['hash'] for i in train}&{records[int(i)]['hash'] for i in test}),
                clean_encoded_inputs_disjoint=not ({encoded[int(i)]['hash'] for i in train}&{encoded[int(i)]['hash'] for i in test}),
                finite_response_features=bool(np.all(np.isfinite(physiology))),
                conservation_below_1e_9=all(r['max_budget_error']<1e-9 for r in rows),no_water_limiter_activation=all(r['limiter_events']==0 for r in rows),
                repeat_and_cache_identical=all(c['repeat_identical'] and c['cache_matches_recomputation'] for c in control_results),
                restart_identical=all(c['restart_identical'] for c in control_results),
                disabled_biology_input_insensitive=all(c['disabled_biology_input_insensitive'] for c in control_results),
                spatial_shuffle_changes_response=all(c['normalized_shuffle_response_RMS']>1e-8 for c in control_results),
                fine_timestep_normalized_error_below_0_02=all(c['refinement_normalized_max']<.02 for c in control_results),
                readout_reload_predictions_identical=True)
    summary=dict(purpose='New dimensional mechanistic-core MRI response test with separate external discrimination assay',
        protocol=PROTOCOL,protocol_hash=protocol_hash(),implementation_hash=implementation_hash,feature_cache_signature=signature,feature_cache_reused=cache_reused,parameters=asdict(Parameters()),
        raw_counts=dict(Counter(r['split'] for r in records)),clean_counts=dict(Training=len(train),Testing=len(test)),deduplication=dedup,encoded_input_deduplication=encoded_dedup,source_pixel_clean_counts=source_pixel_counts,additional_encoded_test_exclusions=additional_encoded_exclusions,
        images_decoded=len(records),images_simulated=len(rows),test_images_simulated=sum(r['split']=='Testing' for r in rows),simulation_failures=failures,
        physiology_budget_max=max(r['max_budget_error'] for r in rows),water_limiter_events=sum(r['limiter_events'] for r in rows),
        readout_algorithm='Train-only standardization and Euclidean nearest class centroid, NumPy only; no tuning on Testing',
        physiology_core_trained=False,readout_is_external=True,models=results,paired_comparisons=comparisons,
        controls=control_results,representatives=representatives,checks=checks,passed=bool(all(checks.values())),elapsed_seconds=time.monotonic()-start,
        limitations=['Synthetic intensity-to-nutrient encoding, not a medical or fungal sensing mechanism',
            'MRI image-level labels; patient IDs and provenance for near duplicates are unavailable',
            'Exact source-pixel and 8x8 stimulus duplicates removed; near-duplicate and patient-level leakage remain possible',
            'Accuracy and confidence intervals are exploratory image-level measurements, not clinical validation',
            'The new core has no image-learning law; only the separate numerical readout uses labels',
            'No electrical/Ca/polarity/branch/fusion or validated fungal-memory mechanism is added by this experiment'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    write_report(output,summary)
    print(json.dumps(dict(images_simulated=summary['images_simulated'],clean_counts=summary['clean_counts'],checks=checks,
                         physiology_accuracy=results['physiology_only']['heldout']['accuracy'],elapsed_seconds=summary['elapsed_seconds']),indent=2),flush=True)
    return summary


def write_report(output,s):
    table=[]
    for name,result in s['models'].items():
        r=result['heldout'];table.append(f'| {name} | {100*r["accuracy"]:.2f}% | {r["correct"]}/{r["count"]} | {100*r["balanced_accuracy"]:.2f}% |')
    lines=['# New MYCELIA core — brain MRI experiment','',
        f'{s["images_simulated"]} images processed with the dimensional core; no simulation failures. Labels never enter the physiology. Each image begins with the same seed state.','',
        '**The core was not trained.** Accuracy below belongs to an external, train-only NumPy nearest-centroid readout of its response. No CNN, neural network, SVM or sklearn is used.','',
        f'Clean split: {s["clean_counts"]["Training"]} Training / {s["clean_counts"]["Testing"]} Testing. {s["deduplication"]["exact_test_train_overlap"]} exact train/test-overlap images removed from accuracy evaluation. The actual 8×8 stimulus is additionally deduplicated, removing {s["additional_encoded_test_exclusions"]} further Testing images. All raw images still undergo the physiology test.','',
        '| Readout / control | Held-out accuracy | Correct | Balanced accuracy |','|---|---:|---:|---:|',*table,'',
        f'Computational checks: {sum(s["checks"].values())}/{len(s["checks"])} passed. Maximum recorded material/water/ATP ledger residual: {s["physiology_budget_max"]:.3g}. Water limiter activations: {s["water_limiter_events"]}.','',
        'Pixel mapping: 8×8 grayscale BOX resize, gray/255. Each row-major adjacent pair drives the basal and apical baths for 0.1 min at 0.02 × intensity M nutrient. All replaced external carbon is accounted for. Total exposure: 32 pulses, 3.2 min. This is an artificial encoding, not a measured MRI-to-chemistry conversion.','',
        'Controls use the first two clean held-out images in each class: exact repeat, cache recomputation, checkpoint restart halfway, histogram-preserving shuffle, a uniform image of the same mean intensity, disabled biology, and a 0.005-min internal timestep.','',
        'Limits: no patient-level split or near-duplicate audit is possible with the available metadata. Confidence intervals treat images as independent. These results do not establish clinical accuracy, tumor diagnosis, fungal cognition or biological image learning.','',
        'See summary.json for confusion matrices, per-class metrics, Wilson intervals, paired bootstrap comparisons and all controls. Per-image CSVs, frozen external readout JSON, and example states are included.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import html
    rows=''.join(f'<tr><td>{name}</td><td>{100*r["heldout"]["accuracy"]:.2f}%</td><td>{r["heldout"]["correct"]}/{r["heldout"]["count"]}</td><td>{100*r["heldout"]["balanced_accuracy"]:.2f}%</td></tr>' for name,r in s['models'].items())
    checks=''.join(f'<tr><td>{k.replace("_"," ")}</td><td>{"PASS" if v else "FAIL"}</td></tr>' for k,v in s['checks'].items())
    examples=''.join(f'<div><img src="examples/{label}/input.jpg" alt="Dataset example {label}"><p>Folder label: {label}<br>Readout prediction: {r["predicted"]}</p></div>' for label,r in s['representatives'].items())
    from mycelia.biology.report import STYLE
    (output/'report.html').write_text(f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MYCELIA · New core MRI test</title><style>{STYLE}.examples{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:20px}}.examples img{{height:220px;object-fit:contain;background:#080d12}}</style></head><body><main><p>MYCELIA / MECHANISTIC CORE / MRI RESPONSE EXPERIMENT</p><h1>Brain images through fungal physiology</h1><p>{s["images_simulated"]} images processed · {s["clean_counts"]["Training"]} clean Training · {s["clean_counts"]["Testing"]} clean Testing</p><p class="note">The biological core was not trained. Classification uses a separate NumPy nearest-centroid readout. Image pixels are synthetic nutrient pulses; this is neither measured fungal image perception nor clinical validation.</p><h2>Held-out discrimination</h2><div class="scroll"><table><tr><th>Readout/control</th><th>Accuracy</th><th>Correct</th><th>Balanced accuracy</th></tr>{rows}</table></div><h2>First clean held-out example per class</h2><div class="examples">{examples}</div><p>Examples were chosen by file order, not prediction correctness.</p><h2>Response and integrity checks</h2><table>{checks}</table><p>Maximum recorded conservation residual: {s["physiology_budget_max"]:.3g}; water limiter events: {s["water_limiter_events"]}. Eight matched control images use repeat, restart, preserved-histogram shuffle, constant mean, disabled biology and timestep refinement.</p><h2>Protocol and limits</h2><p>8×8 grayscale intensity scan → paired basal/apical nutrient pulses at 0.02×intensity M → 32 exposures of 0.1 min. All external carbon replacement is recorded. The hypha is fresh for each image, with identical initial physiology and no input labels.</p><p>Exact decoded-pixel and actual 8×8 stimulus duplicates are removed from readout evaluation. {s["additional_encoded_test_exclusions"]} additional resized-input Testing matches were excluded. Patient identities and near-duplicate provenance are unavailable; image-level leakage may remain. The intervals in JSON assume independent images. The core contains no biological image-learning mechanism.</p><p><a href="RESULTS.md">Detailed result note</a> · <a href="summary.json">Metrics, controls and intervals</a> · <a href="per-image-physiology.csv">Every image response</a> · <a href="external-physiology-readout.json">External frozen readout</a></p></main></body></html>')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=run(a.dataset,a.output)
    return 0 if result['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
