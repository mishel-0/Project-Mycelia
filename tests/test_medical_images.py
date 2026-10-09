"""Integrity tests for the synthetic MRI input protocol and separate readout."""
from pathlib import Path
from dataclasses import replace
import importlib.util
import json
import numpy as np
import pytest
from mycelia.biology import Hypha, Parameters
from mycelia.biology.images import encode_image, new_image_hypha, apply_pulses, load_image, OBSERVABLES

spec=importlib.util.spec_from_file_location('medical_image_test',Path(__file__).parents[1]/'tools/medical_image_test.py')
readout=importlib.util.module_from_spec(spec);spec.loader.exec_module(readout)


def test_external_nutrient_replacement_accounts_for_addition_and_removal():
    h=Hypha();before=h._totals()['carbon_pmol']
    h.set_bath_nutrient(0,3);h.set_bath_nutrient(1,0)
    assert h.ledger['carbon_added_pmol']==1
    assert h.ledger['carbon_removed_pmol']==2
    assert h._totals()['carbon_pmol']==pytest.approx(before-1)
    h.run(.2);h.validate()
    assert max(abs(v) for v in h.budgets().values())<1e-9
    # Older milestone checkpoints have no intervention ledger entries.
    original=Hypha();loaded=Hypha.from_state_dict(original.state_dict())
    loaded.set_bath_nutrient(0,0);loaded.step();loaded.validate()


@pytest.mark.parametrize('bath,amount',[(2,0),(True,1),(0,-1),(0,float('inf'))])
def test_invalid_external_intervention_leaves_state_unchanged(bath,amount):
    h=Hypha();before=h.state_dict()
    with pytest.raises(ValueError):h.set_bath_nutrient(bath,amount)
    assert before==h.state_dict()


def test_scan_has_finite_reproducible_label_free_response():
    gray=np.linspace(0,1,64).reshape(8,8)
    a=encode_image(gray,return_model=True);b=encode_image(gray)
    assert a['features'].shape==(32*len(OBSERVABLES),)
    assert np.array_equal(a['features'],b['features'])
    assert a['final']['time_min']==pytest.approx(3.2)
    assert a['maximum_budget_error']<1e-9
    assert a['limiter_events']==0
    a['model'].validate()
    constant=encode_image(np.full((8,8),gray.mean()))
    assert not np.allclose(a['features'],constant['features'])


def test_image_exposure_restart_continues_the_same_physiology():
    gray=np.linspace(0,1,64).reshape(8,8);pairs=gray.ravel().reshape(-1,2)
    h=new_image_hypha();first=apply_pulses(h,pairs[:16])
    h=Hypha.from_state_dict(h.state_dict());second=apply_pulses(h,pairs[16:])
    assert np.array_equal(np.vstack([first,second]).ravel(),encode_image(gray)['features'])


@pytest.mark.parametrize('gray',[np.zeros((4,4)),np.full((8,8),np.nan),np.full((8,8),1.1)])
def test_invalid_image_protocol_inputs(gray):
    with pytest.raises(ValueError):encode_image(gray)


def test_decoded_pixel_hash_rejects_same_content_under_different_paths(tmp_path):
    Image=pytest.importorskip('PIL.Image')
    arr=np.arange(64,dtype=np.uint8).reshape(8,8)
    a=tmp_path/'a.png';b=tmp_path/'b.png';Image.fromarray(arr).save(a);Image.fromarray(arr).save(b)
    gray1,h1=load_image(a);gray2,h2=load_image(b)
    assert h1==h2 and np.array_equal(gray1,gray2)


def test_clean_split_excludes_exact_overlaps_duplicates_and_conflicts():
    records=[dict(split='Training',label='glioma',hash='a'),dict(split='Training',label='glioma',hash='a'),
             dict(split='Testing',label='glioma',hash='a'),dict(split='Testing',label='notumor',hash='b'),
             dict(split='Testing',label='notumor',hash='b'),dict(split='Training',label='glioma',hash='c'),
             dict(split='Testing',label='pituitary',hash='c')]
    train,test,report=readout.clean_splits(records)
    assert train.tolist()==[0] and test.tolist()==[3]
    assert report['exact_test_train_overlap']==1
    assert report['within_training_duplicates']==1 and report['within_testing_duplicates']==1
    assert report['conflicting_pixel_hashes']==1


def test_centroid_parameters_are_training_only_and_frozen():
    x=np.array([[0,0],[0,.1],[1,0],[1,.1],[2,0],[2,.1],[3,0],[3,.1]])
    labels=np.repeat(readout.LABELS,2)
    model=readout.fit_centroids(x,labels)
    assert np.array_equal(readout.predict(model,x),labels)
    before=json.dumps(readout.model_dict(model),sort_keys=True)
    readout.predict(model,np.array([[100,100]]))
    assert before==json.dumps(readout.model_dict(model),sort_keys=True)
    assert np.array_equal(model['mean'],x.mean(axis=0))
    for key in ('mean','scale','centroids'):
        with pytest.raises(ValueError):model[key].flat[0]=99
    restored=readout.freeze_model(json.loads(before))
    assert np.array_equal(readout.predict(restored,x),labels)


def test_metric_accounting_for_known_confusion():
    result=readout.metrics(readout.LABELS,['glioma']*4)
    assert result['accuracy']==.25 and result['correct']==1 and result['balanced_accuracy']==.25
    low,high=result['accuracy_95pct_Wilson_CI'];assert low<.25<high
    assert np.asarray(result['confusion_matrix']).sum()==4


def test_single_image_cli_uses_saved_external_readout_and_checks_protocol(tmp_path):
    import subprocess,sys
    from dataclasses import asdict
    Image=pytest.importorskip('PIL.Image')
    images=[np.full((8,8),value) for value in (0,85/255,170/255,1)]
    x=np.array([encode_image(gray)['features'] for gray in images])
    model=readout.fit_centroids(x,readout.LABELS)
    artifact=dict(schema='mycelia.external-image-centroid.v1',parameters=asdict(Parameters()),
        model=readout.model_dict(model),protocol_hash=readout.protocol_hash(),implementation_hash=readout.implementation_fingerprint())
    path=tmp_path/'readout.json';path.write_text(json.dumps(artifact))
    image=tmp_path/'input.png';Image.fromarray(np.full((8,8),85,dtype=np.uint8)).save(image)
    output=tmp_path/'prediction'
    command=[sys.executable,str(Path(__file__).parents[1]/'tools/predict_medical_image.py'),str(image),'--readout',str(path),'--output',str(output)]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    prediction=json.loads((output/'prediction.json').read_text())
    assert prediction['predicted_dataset_label']==readout.LABELS[1]
    assert prediction['physiology_core_trained'] is False
    assert 'not clinical diagnosis' in prediction['scope']
    Hypha.load(output/'state.json').validate()
    artifact['implementation_hash']='outdated';path.write_text(json.dumps(artifact))
    bad=subprocess.run(command,capture_output=True,text=True)
    assert bad.returncode!=0 and 'differs' in bad.stderr


def test_resized_input_collisions_excluded_even_when_source_hashes_differ():
    records=[dict(split='Training',label='glioma',hash='original-a'),
             dict(split='Testing',label='glioma',hash='original-b'),
             dict(split='Testing',label='pituitary',hash='original-c')]
    images=np.array([np.zeros(64),np.zeros(64),np.ones(64)])
    source_train,source_test,_=readout.clean_splits(records)
    assert source_test.tolist()==[1,2]
    mapped=readout.encoded_records(records,images)
    train,test,counts=readout.clean_splits(mapped)
    assert train.tolist()==[0] and test.tolist()==[2]
    assert counts['exact_test_train_overlap']==1
