"""Leakage boundaries and replay of the direct graph-learning workflow."""
import numpy as np
from mycelia import MemoryConfig,image_cues
from tools.train_mycelium import LABELS,clean_splits,fit,metrics,validation_split


def record(split,label,source,cue):
    return {'file':split+'/'+label+'/'+source+'.jpg','split':split,'label':label,'source_sha256':source,'cue_sha256':cue}


def test_source_and_encoded_duplicates_cannot_cross_evaluation_boundaries():
    rows=[record('Training','glioma','a','x'),record('Training','glioma','a','x'),
          record('Training','glioma','b','x'),record('Training','glioma','c','y'),
          record('Testing','glioma','b','x'),record('Testing','glioma','d','y'),
          record('Testing','glioma','e','z'),record('Testing','glioma','f','z')]
    clean,audit=clean_splits(rows)
    assert clean['Training'].tolist()==[0,3]
    assert clean['Testing'].tolist()==[6]
    assert audit['removed']['Testing_source_overlap']==1
    assert audit['removed']['Testing_additional_encoded_overlap']==1
    assert audit['removed']['Testing_within_split_duplicate']==1


def test_conflicting_encoded_labels_are_quarantined():
    rows=[record('Training','glioma','a','same'),record('Testing','notumor','b','same'),
          record('Training','glioma','c','unique-c'),record('Testing','notumor','d','unique-d')]
    clean,audit=clean_splits(rows)
    assert audit['encoded_label_conflicts']==1
    assert clean['Training'].tolist()==[2]
    assert clean['Testing'].tolist()==[3]


def test_validation_uses_only_training_and_has_disjoint_inputs():
    rows=[record(split,label,f'{split}-{label}-{i}',f'{split}-{label}-{i}')
          for split in ('Training','Testing') for label in LABELS for i in range(10)]
    clean,_=clean_splits(rows)
    inner,validation,calibration=validation_split(rows,clean['Training'],limit=3)
    assert set(inner)|set(validation)==set(clean['Training'])
    assert not set(inner)&set(validation)
    assert set(calibration)<=set(inner)
    assert not (set(inner)|set(validation))&set(clean['Testing'])
    assert len(calibration)==12
    # Altering every Testing label cannot change Training-only selection inputs.
    for i in clean['Testing']:rows[i]['label']='pituitary'
    repeated=validation_split(rows,clean['Training'],limit=3)
    for expected,actual in zip((inner,validation,calibration),repeated):np.testing.assert_array_equal(expected,actual)


def test_training_consumes_all_requested_rows_and_is_reproducible(tmp_path):
    rows=[record(split,label,f'{split}-{label}',f'{split}-{label}')
          for split in ('Training','Testing') for label in LABELS]
    cues=np.stack([image_cues(np.full((3,3),LABELS.index(r['label'])/3)) for r in rows])
    indices=np.arange(len(rows));config=MemoryConfig(size=3,colonies_per_label=1,physiology_interval=2,consolidation=True)
    a,evidence=fit(rows,cues,indices,config,2,42,'test',tmp_path)
    b,_=fit(rows,cues,indices,config,2,42,'test',tmp_path)
    assert a.state_dict()==b.state_dict()
    assert evidence['exposures']==16
    assert sum(z.exposures for pop in a.colonies.values() for z in pop)==16
    assert a.predict_many(cues)==[r['label'] for r in rows]
    assert evidence['max_absolute_budget_error']<1e-8
    assert a.frozen


def test_abstentions_are_counted_as_errors_in_metrics():
    result=metrics(['glioma','notumor'],['glioma',None])
    assert result['n']==2 and result['correct']==1
    assert result['accuracy']==.5 and result['abstentions']==1
    assert result['classes']['notumor']['recall']==0


def test_saved_image_predictor_uses_same_encoding_and_does_not_rewrite_model(tmp_path):
    import json,subprocess,sys
    from pathlib import Path
    from PIL import Image
    from mycelia import AssociativeMycelium
    pixels=np.arange(25,dtype=np.uint8).reshape(5,5)*10
    image=tmp_path/'sample.png';Image.fromarray(pixels).save(image)
    model=AssociativeMycelium(['sample'],MemoryConfig(size=5))
    model.learn(image_cues(pixels.astype(float)/255),'sample');model.freeze()
    memory=tmp_path/'memory.json';model.save(memory);before=memory.read_bytes()
    target=tmp_path/'prediction.json'
    tool=Path(__file__).resolve().parents[1]/'tools/predict_memory_image.py'
    run=subprocess.run([sys.executable,str(tool),'--memory',str(memory),'--image',str(image),'--output',str(target)],capture_output=True,text=True,check=True)
    result=json.loads(run.stdout)
    assert result['prediction']=='sample'
    assert result==json.loads(target.read_text())
    assert result['chemical_mismatch_scores']['sample']==model.scores(image_cues(pixels.astype(float)/255))['sample']
    assert memory.read_bytes()==before
