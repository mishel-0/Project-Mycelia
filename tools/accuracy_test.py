"""Train/test a supervised readout; physiology remains independent of labels."""
from pathlib import Path
import argparse,csv,hashlib,json,time,warnings
import joblib,numpy as np
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix
from mycelia import Config,Environment,Mycelium

LABELS=['glioma','meningioma','notumor','pituitary']
PHYS=['nodes','segments','active_tips','total_length','total_flow','mean_turgor','mean_radius','nutrient','reserve','energy','biomass','uptake','respired_carbon']

def image_features(gray):
    # 9 unsigned gradient bins, 8x8 cells, overlapping 2x2 L2-Hys blocks.
    gy,gx=np.gradient(gray);mag=np.hypot(gx,gy)
    angle=(np.arctan2(gy,gx)%np.pi)*9/np.pi
    low=np.floor(angle).astype(int)%9;fraction=angle-np.floor(angle)
    hist=np.zeros((8,8,9))
    yy,xx=np.indices(gray.shape)
    np.add.at(hist,(yy//8,xx//8,low),mag*(1-fraction))
    np.add.at(hist,(yy//8,xx//8,(low+1)%9),mag*fraction)
    blocks=[]
    for y in range(7):
        for x in range(7):
            v=hist[y:y+2,x:x+2].ravel();v=v/np.sqrt(v@v+1e-6)
            v=np.minimum(v,.2);blocks.extend(v/np.sqrt(v@v+1e-6))
    intensity=gray.reshape(16,4,16,4).mean(axis=(1,3)).ravel()
    return np.r_[blocks,intensity]

def extract(path):
    with Image.open(path) as im:
        rgb=im.convert('RGB');h=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
        gray=np.asarray(rgb.convert('L').resize((64,64),Image.Resampling.BOX),dtype=float)/255
        field=np.asarray(rgb.convert('L').resize((32,32),Image.Resampling.BOX),dtype=float)/255*.8
    o=Mycelium(Environment(field),Config(max_nodes=150,max_tips=24),seed=42)
    s=o.run(60)
    f=[s[k] for k in PHYS]+[s['events'].get(k,0) for k in ('branch','fusion','tip_death','retraction')]
    for attr in ('x','y','pressure','nutrient'):
        vals=np.array([getattr(n,attr) for n in o.nodes.values() if n.alive])
        f.extend([float(vals.mean()),float(vals.std()),float(vals.min()),float(vals.max())])
    f=np.asarray(f)
    assert np.all(np.isfinite(f))
    assert max(abs(s[k]) for k in ('carbon_error','water_error','energy_error'))<1e-8
    return h,image_features(gray),f

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    files=sorted(a.dataset.glob('*/*/*.jpg'));cache=a.output/'features.npz'
    if cache.exists():
        d=np.load(cache,allow_pickle=False);paths=d['paths'];labels=d['labels'];splits=d['splits'];hashes=d['hashes'];img=d['image'];phys=d['physiology']
        assert paths.tolist()==[str(f.relative_to(a.dataset)) for f in files], 'Cache dataset paths differ'
    else:
        paths=[];labels=[];splits=[];hashes=[];img=[];phys=[]
        for i,f in enumerate(files):
            h,x,z=extract(f);paths.append(str(f.relative_to(a.dataset)));labels.append(f.parent.name);splits.append(f.relative_to(a.dataset).parts[0]);hashes.append(h);img.append(x);phys.append(z)
            if (i+1)%200==0:print(f'Image + physiology features: {i+1}/{len(files)}',flush=True)
        paths,labels,splits,hashes=map(np.asarray,(paths,labels,splits,hashes));img=np.asarray(img);phys=np.asarray(phys)
        np.savez_compressed(cache,paths=paths,labels=labels,splits=splits,hashes=hashes,image=img,physiology=phys)
    groups={}
    for h,y in zip(hashes,labels):groups.setdefault(h,set()).add(y)
    conflicts={h for h,v in groups.items() if len(v)>1};seen=set();train=[]
    for i in np.flatnonzero(splits=='Training'):
        if hashes[i] in conflicts or hashes[i] in seen:continue
        seen.add(hashes[i]);train.append(i)
    train_hashes=set(seen);seen=set();test=[];overlap=[]
    for i in np.flatnonzero(splits=='Testing'):
        if hashes[i] in train_hashes:overlap.append(int(i));continue
        if hashes[i] in conflicts or hashes[i] in seen:continue
        seen.add(hashes[i]);test.append(i)
    train=np.array(train);test=np.array(test)
    if not len(train) or not len(test):raise ValueError('No clean train/test images')
    result={'protocol':'Training-only supervised fitting; fixed C=0.1, no tuning on Testing. Primary Testing set excludes exact decoded-pixel training overlap, within-split duplicates and conflicting-label hashes. Labels never enter physiology.','scope':'Image-level evaluation only; patient identity and near-duplicate provenance unavailable. Not a clinical validation.','raw_counts':{'Training':int(sum(splits=='Training')),'Testing':int(sum(splits=='Testing'))},'clean_counts':{'Training':len(train),'Testing':len(test)},'exact_test_train_overlap':len(overlap),'conflicting_pixel_hashes':len(conflicts),'models':{}}
    matrices={'physiology_only':phys,'image_only':img,'image_plus_physiology':np.c_[img,phys]}
    for name,x in matrices.items():
        print(f'Fitting {name}: {x.shape[1]} features; {len(train)} train / {len(test)} test',flush=True)
        model=make_pipeline(StandardScaler(),LinearSVC(C=.1,max_iter=20000,dual='auto',random_state=42))
        with warnings.catch_warnings(record=True) as notes:
            model.fit(x[train],labels[train])
        pred=model.predict(x[test]);trainpred=model.predict(x[train]);untrained=DummyClassifier(strategy='most_frequent').fit(x[train],labels[train])
        result['models'][name]={'training_accuracy':accuracy_score(labels[train],trainpred),'heldout_accuracy':accuracy_score(labels[test],pred),'balanced_accuracy':balanced_accuracy_score(labels[test],pred),'majority_baseline_accuracy':accuracy_score(labels[test],untrained.predict(x[test])),'confusion_labels':LABELS,'confusion_matrix':confusion_matrix(labels[test],pred,labels=LABELS).tolist(),'classification_report':classification_report(labels[test],pred,labels=LABELS,output_dict=True,zero_division=0),'fit_warnings':[str(n.message) for n in notes]}
        joblib.dump(model,a.output/(name+'.joblib'))
        restored=joblib.load(a.output/(name+'.joblib'));assert np.array_equal(restored.predict(x[test]),pred)
        with (a.output/(name+'-predictions.csv')).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['file','actual','predicted','correct'])
            w.writerows((paths[i],labels[i],y,int(labels[i]==y)) for i,y in zip(test,pred))
        print(name,json.dumps({k:v for k,v in result['models'][name].items() if k.endswith('accuracy')}),flush=True)
    result['elapsed_seconds']=time.monotonic()-start
    (a.output/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
