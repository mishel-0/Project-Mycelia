import numpy as np,json
d=np.load('d64.npz');y=d['y'];tr=d['tr'];te=d['te']
F=np.load('F5.npy');m=F[tr].mean(0,dtype=np.float64).astype(np.float32);s=(F[tr].std(0,dtype=np.float64)+1e-6).astype(np.float32)
def n(A):
    A-=m;A/=s;A/=np.linalg.norm(A,axis=1,keepdims=True);return A
base=n(F[tr].copy());T=n(F[te].copy());del F
augn={k:n(np.load(f)) for k,f in (('flip','F5_flip.npy'),('s1','s1.npy'),('s2','s2.npy'))}
def fit_predict(rows,query,keys,g,lam):
    A=np.concatenate([base[rows]]+[augn[k][rows] for k in keys]);yy=np.tile(y[tr][rows],1+len(keys))
    K=A@A.T;K-=1;K*=g;np.exp(K,out=K);K[np.diag_indices_from(K)]+=lam
    a=np.linalg.solve(K,(np.eye(4)[yy]*2-1).astype(np.float32));del K
    return (np.exp(g*(query@A.T-1))@a).argmax(1)
rng=np.random.default_rng(1);val=np.concatenate([rng.permutation(np.where(y[tr]==c)[0])[:int(.2*(y[tr]==c).sum())] for c in range(4)])
inner=np.setdiff1d(np.arange(len(tr)),val)
res=[]
for keys in ((),('flip',),('flip','s1','s2')):
  for g in (2,5):
    for lam in (.003,.01,.03):
      acc=(fit_predict(inner,base[val],keys,g,lam)==y[tr][val]).mean();res.append((acc,keys,g,lam));print(round(acc,4),keys,g,lam,flush=True)
best=max(res,key=lambda r:r[0]);print('selected on validation',best)
p=fit_predict(np.arange(len(tr)),T,best[1],best[2],best[3]);acc=(p==y[te]).mean()
L=['glioma','meningioma','notumor','pituitary'];M=np.zeros((4,4),int)
for a,b in zip(y[te],p):M[a,b]+=1
z=1.96;nn=len(te);c=(acc+z*z/(2*nn))/(1+z*z/nn);h=z*np.sqrt(acc*(1-acc)/nn+z*z/(4*nn*nn))/(1+z*z/nn)
out=dict(test_accuracy=acc,correct=int((p==y[te]).sum()),n=nn,wilson95=[c-h,c+h],balanced=float(np.mean(np.diag(M)/M.sum(1))),
 recall=dict(zip(L,(np.diag(M)/M.sum(1)).round(4).tolist())),precision=dict(zip(L,(np.diag(M)/M.sum(0)).round(4).tolist())),confusion=M.tolist(),
 selected=dict(augment=list(best[1]),gamma=best[2],lam=best[3],val_acc=best[0]),validation=[dict(acc=r[0],augment=list(r[1]),gamma=r[2],lam=r[3]) for r in res])
json.dump(out,open('final.json','w'),indent=1);print(json.dumps({k:v for k,v in out.items() if k!='validation'},indent=1))
