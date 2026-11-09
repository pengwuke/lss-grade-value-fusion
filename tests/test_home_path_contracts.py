"""Artificial I/O cases for literal home paths; no patient files or model weights."""
import contextlib
import hashlib
import importlib.util
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

source = Path(__file__).resolve().parents[1] / 'examples/predict_common5_pair.py'
spec = importlib.util.spec_from_file_location('home_path_example', source)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class HomePathContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='paper2_r7_home_')
        self.root = Path(self.temp.name).resolve()
        self.home = self.root / 'ARTIFICIAL HOME'
        self.home.mkdir()
        self.env = patch.dict(os.environ, {'HOME':str(self.home), 'USERPROFILE':str(self.home)})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def argv(self, name):
        return ['--models','~/models','--cnn-iq8','~/cnn.png','--deit-png','~/deit.png',
                '--fold','0','--strategy','R11','--roi-id','ARTIFICIAL_ROI','--output','~/'+name]

    def test_home_png_reads_exact_pixels_and_same_byte_fingerprint(self):
        values = np.arange(35,dtype=np.uint8).reshape(5,7)
        target = self.home / 'ARTIFICIAL gray.png'
        Image.fromarray(values).save(target)
        actual, digest = example.read_gray_png('~/ARTIFICIAL gray.png')
        np.testing.assert_array_equal(actual, values)
        self.assertEqual(digest, hashlib.sha256(target.read_bytes()).hexdigest())

    def test_loading_and_provenance_use_one_expanded_model_root(self):
        model_root = (self.home/'models').resolve()
        calls = []
        def config(path):
            calls.append(Path(path)); return {'common5':{'R11_coral_weights':[.7]*5}}
        def load(path, group, family, fold, **kwargs):
            calls.append(Path(path)); return object(), {'family':family,'fold':fold,'sha256':'a'*64}
        def digest(path):
            calls.append(Path(path).parent); return 'b'*64
        with contextlib.ExitStack() as stack:
            for name, value in [('read_fusion_config',config),('load_model',load),('sha256',digest)]:
                stack.enter_context(patch.object(example,name,side_effect=value))
            stack.enter_context(patch.object(example,'read_gray_png',return_value=(np.zeros((3,7),dtype=np.uint8),'c'*64)))
            stack.enter_context(patch.object(example,'cnn_iq8_tensor',return_value=torch.zeros(1,64,64)))
            stack.enter_context(patch.object(example,'deit_png_tensor',return_value=torch.zeros(3,224,224)))
            stack.enter_context(patch.object(example,'probabilities',return_value=torch.tensor([[.1,.2,.3,.4]])))
            result = example.predict_pair('~/models','~/cnn.png','~/deit.png',0,'R11','ARTIFICIAL_ROI')
        self.assertEqual(calls,[model_root]*5)
        self.assertEqual(result['model_manifest_sha256'],'b'*64)

    def test_new_report_goes_to_expanded_home_destination(self):
        target = self.home/'reports/new.json'
        result = {'roi_id':'ARTIFICIAL_ROI','fold':0,'strategy':'R11','predicted_grade':2}
        with patch.object(example,'predict_pair',return_value=result), contextlib.redirect_stdout(StringIO()):
            example.main(self.argv('reports/new.json'))
        self.assertEqual(json.loads(target.read_text()),result)

    def test_existing_home_report_rejected_before_prediction(self):
        target = self.home/'existing.json';target.write_text('DO NOT OVERWRITE',encoding='utf-8')
        with patch.object(example,'predict_pair',side_effect=AssertionError('prediction must not start')), contextlib.redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            example.main(self.argv('existing.json'))
        self.assertEqual(target.read_text(),'DO NOT OVERWRITE')


if __name__=='__main__':unittest.main()
