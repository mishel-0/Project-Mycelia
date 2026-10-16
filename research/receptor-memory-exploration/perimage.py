import sys,csv,json,numpy as np
sys.path.insert(0,'/home/user/project-mycelia');sys.path.insert(0,'/home/user/project-mycelia/tools')
from mycelia.receptor_memory import TraceMemory
from train_mycelium import load_images,clean_splits,LABELS
records,_=load_images(__import__('pathlib').Path('data/Brain-Tumor-MRI-Dataset'),16)
d=np.load('d64.npz');y=d['y'];tr=d['tr'];te=d['te']
F=np.load('F5.npy');Ff=np.load('F5_flip.npy')
mem=TraceMemory(4,2.,.003).fit(F[tr],y[tr],[(Ff,y[tr])])
S=mem.scores(F[te]);p=S.argmax(1)
e=np.exp(5*(S-S.max(1,keepdims=True)));P=e/e.sum(1,keepdims=True)
tumor=1-P[:,2]
rows=[]
for k,i in enumerate(te):
    r=records[i];rows.append(dict(file=r['file'],actual=r['label'],predicted=LABELS[p[k]],correct=int(p[k]==y[i]),
      actual_tumor=int(y[i]!=2),predicted_tumor=int(p[k]!=2),tumor_score=round(float(tumor[k]),4),confidence=round(float(P[k].max()),4),
      **{f'{l}_score':round(float(P[k,j]),4) for j,l in enumerate(LABELS)}))
with open('per-image-predictions.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
c=np.array([r['correct'] for r in rows]);conf=P.max(1)
print('understood',c.sum(),'of',len(c),'| not understood',len(c)-c.sum())
for lo,hi in ((0,.5),(.5,.7),(.7,.9),(.9,1.01)):
    m=(conf>=lo)&(conf<hi);print(f'confidence {lo:.1f}-{min(hi,1):.1f}: {m.sum()} images, {c[m].mean() if m.sum() else 0:.3f} correct')
t=y[te]!=2
print('mean tumor score: real tumors',tumor[t].mean().round(4),'| healthy',tumor[~t].mean().round(4))
# AUC
from itertools import product
pos=tumor[t];neg=tumor[~t];auc=(np.sum(pos[:,None]>neg[None,:])+.5*np.sum(pos[:,None]==neg[None,:]))/(len(pos)*len(neg));print('tumor AUC',round(auc,4))
for l in LABELS:
    m=np.array([r['actual']==l for r in rows]);print(l,'understood',c[m].sum(),'/',m.sum())
print('worst misses:');
for r in sorted([r for r in rows if not r['correct']],key=lambda r:-r['confidence'])[:5]:print(' ',r['file'],r['actual'],'->',r['predicted'],r['confidence'])
