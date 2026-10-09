"""Read-only recognition using saved compartment/cord memory (JSON)."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from PIL import Image
from mycelia import AssociativeMycelium,image_cues

def main():
    p=argparse.ArgumentParser();p.add_argument('--memory',type=Path,required=True);p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,help='Optional JSON prediction file')
    a=p.parse_args();memory=AssociativeMycelium.load(a.memory);memory.freeze()
    with Image.open(a.image) as im:
        gray=np.asarray(im.convert('RGB').convert('L').resize((memory.config.size,memory.config.size),Image.Resampling.BOX),dtype=float)/255
    cue=image_cues(gray);scores=memory.scores(cue)
    prediction=None if all(math.isinf(v) for v in scores.values()) else min(scores,key=scores.get)
    result={'image':str(a.image.resolve()),'memory':str(a.memory.resolve()),'prediction':prediction,
            'chemical_mismatch_scores':{k:v if math.isfinite(v) else None for k,v in scores.items()},
            'score_meaning':'Lower mismatch fits the stored imprint better. These scores are not probabilities.',
            'scope':'Experimental bio-inspired associative memory; image-level research evaluation.'}
    # Report fitting scope only when an adjacent manifest matches this exact file.
    manifest=a.memory.parent/'summary.json'
    if manifest.exists():
        try:metadata=json.loads(manifest.read_text())
        except (ValueError,OSError):metadata={}
        digest=hashlib.sha256(a.memory.read_bytes()).hexdigest()
        for kind in ('heldout','full_data'):
            entry=metadata.get(kind,{})
            if isinstance(entry,dict) and entry.get('model_sha256')==digest:
                result['training_scope']=entry['scope'];result['memory_sha256']=digest
    content=json.dumps(result,indent=2,allow_nan=False)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(content+'\n')
    print(content)
if __name__=='__main__':main()
