import pickle,numpy as np
from PIL import Image,ImageDraw
d=np.load('d64.npz');X=d['X'].astype(np.float32)/255;y=d['y'];te=d['te']
m=pickle.load(open('dict.pkl','rb'));mu,Wz,D,P=m['mu'],m['Wz'],m['D'],m['P'];Winv=np.linalg.inv(Wz)
def recon(img,k):
    w=np.lib.stride_tricks.sliding_window_view(img,(P,P));g=w.shape[0];p=w.reshape(-1,P*P)
    mean=p.mean(1,keepdims=True);sd=np.sqrt(p.var(1,keepdims=True)+.01)
    z=((p-mean)/sd-mu)@Wz;r=z@D.T
    if k=='pool':  # only the pooled 4x4 response, what the classifier memory keeps
        return None
    idx=np.argsort(-np.abs(r),1)[:,:k];zh=np.zeros_like(z)
    for i in range(len(z)):
        B=D[idx[i]];c,*_=np.linalg.lstsq(B.T,z[i],rcond=None);zh[i]=c@B
    ph=((zh@Winv)+mu)*sd+mean
    out=np.zeros_like(img);cnt=np.zeros_like(img)
    for i,(a,b) in enumerate(np.ndindex(g,g)):out[a:a+P,b:b+P]+=ph[i].reshape(P,P);cnt[a:a+P,b:b+P]+=1
    return np.clip(out/cnt,0,1)
def psnr(a,b):return 10*np.log10(1/np.mean((a-b)**2))
rng=np.random.default_rng(0);pick=[rng.choice(te[y[te]==c]) for c in range(4)]
L=['glioma','meningioma','notumor','pituitary'];ks=[1,3,10]
cols=[];
for k in ks:print('k',k,'mean PSNR on 40 test images',np.mean([psnr(X[i],recon(X[i],k)) for i in rng.choice(te,40,replace=False)]).round(2),'dB',flush=True)
S=4;W=64*S;canvas=Image.new('L',(W*(1+len(ks)),W*4+20*4),255);dr=ImageDraw.Draw(canvas)
for r,i in enumerate(pick):
    ims=[X[i]]+[recon(X[i],k) for k in ks]
    for c,im in enumerate(ims):
        canvas.paste(Image.fromarray((im*255).astype('uint8')).resize((W,W),Image.Resampling.NEAREST),(c*W,r*(W+20)+20))
    dr.text((4,r*(W+20)+4),f'{L[r]}: original | receptors k=1 | k=3 | k=10',fill=0)
canvas.save('/home/user/project-mycelia/results/receptor-memory/reconstruction.png')
