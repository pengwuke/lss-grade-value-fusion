"""Check explicit exported checkpoints on artificial tensors; never reads MRI/CSV."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('HF_HUB_OFFLINE','1')
import torch
import paper2_runtime
from paper2_runtime.artifacts import read_manifest
from paper2_runtime.models import CNN_FAMILIES,REFERENCE_FAMILIES,load_model,probabilities


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('output exists; choose a new report path')
    torch.set_num_threads(1);torch.manual_seed(0)
    entries=read_manifest(args.models);rows=[]
    with torch.inference_mode():
        for e in entries:
            model,entry=load_model(args.models,e['group'],e['family'],e['fold'],device='cpu')
            shape=(1,64,64) if e['family'] in CNN_FAMILIES else ((3,224,224) if e['family']=='deit_tiny' else (1,224,224))
            count=shape[0]*shape[1]*shape[2]
            gradient=torch.linspace(-1.,1.,steps=count,dtype=torch.float32).reshape(1,*shape)
            inputs=torch.cat([torch.zeros_like(gradient),gradient],dim=0)
            values=probabilities(model,entry,inputs)
            assert values.shape==(2,4) and torch.isfinite(values).all() and (values>=0).all()
            tolerance=3.5e-6 if e['family'] in REFERENCE_FAMILIES else 1e-6
            error=float(torch.max(torch.abs(values.sum(1)-1)).item());assert error<=tolerance
            rows.append({'group':e['group'],'family':e['family'],'fold':e['fold'],
                         'strict_load':True,'output_shape':[2,4],'maximum_row_mass_error':error,
                         'synthetic_grades':values.argmax(1).tolist()})
            del model,values,inputs
    result={'entries':len(rows),'strict_architecture_load_pass':len(rows),'synthetic_forward_pass':len(rows),
            'artificial_inputs':['all-zero tensor','deterministic -1..1 gradient'],
            'patient_inference':False,'pretrained_downloads_required':False,'device':'cpu',
            'isolated_mode':bool(sys.flags.isolated),'runtime_package_path':paper2_runtime.__file__,
            'versions':{n:importlib.metadata.version(n) for n in ['torch','torchvision','timm','numpy','opencv-python-headless','Pillow']},
            'model_roles':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ['model_roles','runtime_package_path']}))


if __name__=='__main__':main()
