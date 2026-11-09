import contextlib
import hashlib
import importlib.util
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

path = Path(__file__).resolve().parents[1] / 'examples/predict_common5_pair.py'
spec = importlib.util.spec_from_file_location('common5_example', path)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class Common5ExampleContracts(unittest.TestCase):
    def setUp(self):
        self.parent = Path(tempfile.gettempdir()).resolve()
        self.temp = tempfile.TemporaryDirectory(prefix='paper2_r6_example_', dir=self.parent)
        self.root = Path(self.temp.name).resolve()
        assert self.root.parent == self.parent and self.root.name.startswith('paper2_r6_example_')

    def tearDown(self):
        self.temp.cleanup()

    def test_grayscale_pixels_and_hash_use_the_same_file_bytes(self):
        values = np.arange(35, dtype=np.uint8).reshape(5, 7)
        target = self.root / 'ARTIFICIAL_gray.png'
        Image.fromarray(values).save(target)
        actual, digest = example.read_gray_png(target)
        np.testing.assert_array_equal(actual, values)
        self.assertEqual(digest, hashlib.sha256(target.read_bytes()).hexdigest())

    def test_no_silent_rgb_conversion(self):
        target = self.root / 'ARTIFICIAL_rgb.png'
        Image.fromarray(np.zeros((3, 7, 3), dtype=np.uint8)).save(target)
        with self.assertRaises(ValueError):
            example.read_gray_png(target)

    def test_no_silent_16bit_conversion(self):
        target = self.root / 'ARTIFICIAL_16bit.png'
        Image.fromarray(np.array([[0, 1024], [2048, 65535]], dtype=np.uint16)).save(target)
        with self.assertRaises(ValueError):
            example.read_gray_png(target)

    def test_non_png_is_rejected(self):
        target = self.root / 'ARTIFICIAL_gray.jpg'
        Image.fromarray(np.zeros((3, 7), dtype=np.uint8)).save(target)
        with self.assertRaises(ValueError):
            example.read_gray_png(target)

    def test_invalid_fold_strategy_and_roi_fail_before_model_access(self):
        with patch.object(example, 'read_fusion_config', side_effect=AssertionError('must not read model package')):
            for fold, strategy, roi in [(0.9, 'R11', 'x'), (True, 'R11', 'x'), (0, 'other', 'x'), (0, 'R11', ''), (0, 'R11', ' x')]:
                with self.subTest(fold=fold, strategy=strategy, roi=roi), self.assertRaises(ValueError):
                    example.predict_pair(self.root, 'missing1', 'missing2', fold, strategy, roi)

    def test_existing_report_is_not_overwritten(self):
        target = self.root / 'existing.json'
        target.write_text('existing report', encoding='utf-8')
        argv = ['--models', str(self.root), '--cnn-iq8', 'missing1', '--deit-png', 'missing2',
                '--fold', '0', '--strategy', 'R11', '--roi-id', 'ARTIFICIAL_ROI', '--output', str(target)]
        with contextlib.redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            example.main(argv)
        self.assertEqual(target.read_text(encoding='utf-8'), 'existing report')


if __name__ == '__main__':
    unittest.main()
