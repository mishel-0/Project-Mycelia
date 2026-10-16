import numpy as np,sys,time
d=np.load(sys.argv[1]);X=d['X'].astype(np.float32)/255;y=d['y'];tr=d['tr'];te=d['te']
K=int(sys.argv[2]);P=int(sys.argv[3]);st=int(sys.argv[4]);G=int(sys.argv[5])
rng=np.random.default_rng(0);N,H,W=X.shape
def patches(img):
    w=np.lib.stride_tricks.sliding_window_view(img,(P,P))[::st,::st]
    return w.reshape(w.shape[0],w.shape[1],-1)
def norm(p):
    p=p-p.mean(-1,keepdims=True);return p/np.sqrt(p.var(-1,keepdims=True)+.01)
# unsupervised: sample patches from training images only, no labels
S=np.concatenate([patches(X[i]).reshape(-1,P*P)[rng.choice(((H-P)//st+1)**2,20,replace=False)] for i in tr]);S=norm(S)
mu=S.mean(0);C=np.cov(S-mu,rowvar=False);e,V=np.linalg.eigh(C);Wz=V@np.diag(1/np.sqrt(e+.1))@V.T
Z=(S-mu)@Wz
# competitive hyphal tips (spherical k-means)
D=Z[rng.choice(len(Z),K,replace=False)];D/=np.linalg.norm(D,axis=1,keepdims=True)
for it in range(15):
    a=np.argmax(Z@D.T,axis=1)
    for k in range(K):
        m=Z[a==k]
        D[k]=m.sum(0) if len(m) else Z[rng.integers(len(Z))]
    D/=np.linalg.norm(D,axis=1,keepdims=True)+1e-9
def encode(img):
    p=patches(img);g=p.shape[0];z=(norm(p.reshape(-1,P*P))-mu)@Wz
    r=z@D.T;f=np.maximum(0,np.abs(r)-.5*0)  # rectified
    f=np.concatenate([np.maximum(0,r-.25),np.maximum(0,-r-.25)],1).reshape(g,g,-1)
    out=[]
    for GG in sorted({1,2,G}):
        cut=np.array_split(np.arange(g),GG);out+=[f[np.ix_(a,b)].mean((0,1)) for a in cut for b in cut]
    return np.concatenate(out)
t=time.time();F=np.stack([encode(X[i]) for i in range(N)]).astype(np.float32);print('enc',F.shape,time.time()-t,flush=True)
np.save(sys.argv[6],F)
m=F[tr].mean(0);s=F[tr].std(0)+1e-6;F=(F-m)/s
F/=np.linalg.norm(F,axis=1,keepdims=True)
sim=F[te]@F[tr].T
for k in (1,3,5,9):
    idx=np.argsort(-sim,axis=1)[:,:k];votes=np.zeros((len(te),4))
    for j in range(k):np.add.at(votes,(np.arange(len(te)),y[tr][idx[:,j]]),sim[np.arange(len(te)),idx[:,j]])
    print('knn',k,(votes.argmax(1)==y[te]).mean())
cent=np.stack([F[tr][y[tr]==c].mean(0) for c in range(4)]);print('centroid',((F[te]@cent.T).argmax(1)==y[te]).mean())
