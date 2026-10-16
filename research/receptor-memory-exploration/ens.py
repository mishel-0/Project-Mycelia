import numpy as np,sys
d=np.load('d64.npz');y=d['y'];tr=d['tr'];te=d['te']
def prep(F):
    m=F[tr].mean(0);s=F[tr].std(0)+1e-6;F=(F-m)/s;return F/np.linalg.norm(F,axis=1,keepdims=True)
sims={n:(lambda F:F[te]@F[tr].T)(prep(np.load(n))) for n in sys.argv[1:]}
def acc(sim):return (y[tr][sim.argmax(1)]==y[te]).mean()
for n,s in sims.items():print(n,acc(s))
print('sum all',acc(sum(sims.values())))
# raw pixel memory too
X=d['X'].reshape(len(y),-1).astype(np.float32);X=prep(X);print('raw',acc(X[te]@X[tr].T))
print('sum+raw',acc(sum(sims.values())+.5*(X[te]@X[tr].T)))
