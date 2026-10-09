"""Apply the explicit external physiology readout to a single image."""
import argparse, json
from pathlib import Path
from mycelia.biology import Parameters
from mycelia.biology.images import load_image, encode_image, protocol_hash
from medical_image_test import freeze_model, predict, implementation_fingerprint


def predict_file(image, artifact, output=None):
    data=json.loads(Path(artifact).read_text())
    if data.get('schema')!='mycelia.external-image-centroid.v1':
        raise ValueError('unsupported external readout schema')
    parameters=Parameters(**data['parameters'])
    if data.get('protocol_hash')!=protocol_hash(parameters) or data.get('implementation_hash')!=implementation_fingerprint():
        raise ValueError('readout protocol/core implementation differs; regenerate the test artifact')
    gray,digest=load_image(image)
    response=encode_image(gray,parameters,return_model=output is not None)
    model=freeze_model(data['model'])
    guess=str(predict(model,response['features'][None,:])[0])
    result=dict(image=str(Path(image)),decoded_pixel_hash=digest,predicted_dataset_label=guess,
                readout='External train-only NumPy nearest centroid',physiology_core_trained=False,
                scope='Exploratory dataset-label prediction; not clinical diagnosis or biological image learning',
                physiology=response['final'],max_budget_error=response['maximum_budget_error'])
    if output is not None:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        (output/'prediction.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        response['model'].save(output/'state.json')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('image',type=Path);p.add_argument('--readout',type=Path,required=True)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    print(json.dumps(predict_file(args.image,args.readout,args.output),indent=2,allow_nan=False))


if __name__=='__main__':main()
