import sys,numpy as np,csv
sys.path.insert(0,'/home/user/project-mycelia/tools');sys.path.insert(0,'/home/user/project-mycelia')
from train_mycelium import load_images,clean_splits,LABELS
from PIL import Image
from pathlib import Path
D=Path(sys.argv[1]);R=int(sys.argv[2])
records,_=load_images(D,16);clean,audit=clean_splits(records)
X=np.stack([np.asarray(Image.open(D/r['file']).convert('L').resize((R,R),Image.Resampling.BOX),dtype=np.uint8) for r in records])
y=np.array([LABELS.index(r['label']) for r in records])
np.savez_compressed(sys.argv[3],X=X,y=y,tr=clean['Training'],te=clean['Testing'])
print(X.shape,len(clean['Training']),len(clean['Testing']))
