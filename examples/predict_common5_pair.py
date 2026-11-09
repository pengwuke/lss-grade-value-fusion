"""Apply one explicit frozen common5 model pair to two verified ROI representations."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from paper2_runtime.artifacts import read_fusion_config, sha256
from paper2_runtime.contracts import fold_index
from paper2_runtime.fusion import common5_fusion
from paper2_runtime.models import load_model, probabilities
from paper2_runtime.preprocessing import cnn_iq8_tensor, deit_png_tensor


def read_gray_png(path):
    """Decode the exact bytes hashed; no implicit colour/bit-depth conversion."""
    raw = Path(path).expanduser().read_bytes()
    with Image.open(BytesIO(raw)) as image:
        if image.format != 'PNG' or image.mode != 'L' or getattr(image, 'n_frames', 1) != 1:
            raise ValueError('This example requires a single-frame 8-bit L-mode grayscale PNG')
        image.load()
        values = np.asarray(image, dtype=np.uint8).copy()
    return values, hashlib.sha256(raw).hexdigest()


def predict_pair(models_root, cnn_iq8_path, deit_png_path, fold, strategy, roi_id):
    fold = fold_index(fold)
    if strategy not in {'R10', 'R11'}:
        raise ValueError('Expected frozen strategy R10 or R11')
    if not isinstance(roi_id, str) or not roi_id or roi_id != roi_id.strip():
        raise ValueError('roi_id must be a nonempty identifier without outer whitespace')
    models_root = Path(models_root).expanduser().resolve()
    config = read_fusion_config(models_root)
    cnn_image, cnn_sha = read_gray_png(cnn_iq8_path)
    deit_image, deit_sha = read_gray_png(deit_png_path)
    with torch.inference_mode():
        cnn, cnn_entry = load_model(models_root, 'common5_core', 'grade_value_coral_selected', fold, device='cpu')
        deit, deit_entry = load_model(models_root, 'common5_core', 'deit_tiny', fold, device='cpu')
        pc = probabilities(cnn, cnn_entry, cnn_iq8_tensor(cnn_image).unsqueeze(0)).cpu().numpy()
        pd = probabilities(deit, deit_entry, deit_png_tensor(deit_image).unsqueeze(0)).cpu().numpy()
    fused = common5_fusion(pc, pd, fold, strategy, config)
    root = Path(models_root)
    return {'roi_id': roi_id, 'fold': fold, 'strategy': strategy,
            'model_group': 'common5_core',
            'model_roles': [{'family': e['family'], 'fold': e['fold'], 'checkpoint_sha256': e['sha256']}
                            for e in [cnn_entry, deit_entry]],
            'input_sha256': {'cnn_iq8': cnn_sha, 'deit_png': deit_sha},
            'model_manifest_sha256': sha256(root / 'models_manifest.json'),
            'fusion_config_sha256': sha256(root / 'fusion_config.json'),
            'coral_weight': float(config['common5'][strategy + '_coral_weights'][fold]),
            'coral_probabilities': pc[0].tolist(), 'deit_probabilities': pd[0].tolist(),
            'fusion_probability_dtype': str(fused.dtype),
            'fused_probabilities': fused[0].tolist(), 'predicted_grade': int(fused.argmax(axis=1)[0]),
            'roi_pair_identity': 'Caller asserts the two files represent the same anatomical ROI; not verified automatically',
            'input_provenance': 'Grayscale format and bytes are checked; DICOM-IQ8 and historical-PNG origins remain caller supplied',
            'fold_assignment': 'Caller supplies the assigned held-out fold; no fold assignment is inferred',
            'scope': 'One frozen common5 pair and one ROI; no training, weight selection or performance estimate'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--cnn-iq8', type=Path, required=True, help='Tight DICOM-derived IQ8 saved as grayscale PNG')
    parser.add_argument('--deit-png', type=Path, required=True, help='Corresponding historical grayscale PNG ROI')
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    parser.add_argument('--strategy', choices=['R10', 'R11'], required=True)
    parser.add_argument('--roi-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output = args.output.expanduser()
    if args.output.exists():
        parser.error('output exists; select a new report path')
    result = predict_pair(args.models, args.cnn_iq8, args.deit_png, args.fold, args.strategy, args.roi_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'roi_id': result['roi_id'], 'fold': result['fold'],
                      'strategy': result['strategy'], 'predicted_grade': result['predicted_grade'],
                      'report_written': True}))


if __name__ == '__main__':
    main()
