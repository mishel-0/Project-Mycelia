import numpy as np,sys
d=np.load('d64.npz');y=d['y'];tr=d['tr'];te=d['te']
F=np.load(sys.argv[1]).astype(np.float64);m=F[tr].mean(0);s=F[tr].std(0)+1e-6;F=(F-m)/s
Y=np.eye(4)[y[tr]]*2-1;A=F[tr];K=A@A.T
for lam in (10,100,1000,3000):
    a=np.linalg.solve(K+lam*np.eye(len(tr)),Y);W=A.T@a
    print('ridge',lam,((F[te]@W).argmax(1)==y[te]).mean())
# RBF kernel teaching
Fn=F/np.linalg.norm(F,axis=1,keepdims=True);G=Fn[tr]@Fn[tr].T;Gt=Fn[te]@Fn[tr].T
for g in (2,5,10):
  for lam in (.01,.1):
    a=np.linalg.solve(np.exp(g*(G-1))+lam*np.eye(len(tr)),Y);print('rbf',g,lam,((np.exp(g*(Gt-1))@a).argmax(1)==y[te]).mean())
