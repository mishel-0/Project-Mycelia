import numpy as np,sys
d=np.load('d64.npz');y=d['y'];tr=d['tr'];te=d['te']
F=np.load(sys.argv[1]).astype(np.float64);Ff=np.load(sys.argv[2]).astype(np.float64)
m=F[tr].mean(0);s=F[tr].std(0)+1e-6
n=lambda A:(lambda B:B/np.linalg.norm(B,axis=1,keepdims=True))((A-m)/s)
A=np.concatenate([n(F[tr]),n(Ff)]);yy=np.concatenate([y[tr],y[tr]]);T=n(F[te])
Y=np.eye(4)[yy]*2-1;G=A@A.T;Gt=T@A.T
print('1nn+flip',(yy[Gt.argmax(1)]==y[te]).mean())
for g in (2,5,8):
  for lam in (.003,.01,.03):
    a=np.linalg.solve(np.exp(g*(G-1))+lam*np.eye(len(A)),Y);print('rbf',g,lam,((np.exp(g*(Gt-1))@a).argmax(1)==y[te]).mean(),flush=True)
