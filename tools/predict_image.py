"""Process one image with a locally saved supervised readout."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from accuracy_test import extract
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--features',choices=('physiology_only','image_only','image_plus_physiology'),default='image_plus_physiology')
    a=p.parse_args()
    _,image,physiology=extract(a.image)
    x={'physiology_only':physiology,'image_only':image,'image_plus_physiology':np.r_[image,physiology]}[a.features]
    model=joblib.load(a.model)
    print(json.dumps({'image':str(a.image),'prediction':str(model.predict(x.reshape(1,-1))[0]),'features':a.features,'scope':'experimental supervised image readout; not a medical diagnosis'},indent=2))

if __name__=="__main__":
    main()
