import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from paper2_runtime.artifacts import checkpoint_path, sha256
from paper2_runtime.fusion import argmax_grades, common5_fusion, source94_fusion


class ArtifactAndFusionContracts(unittest.TestCase):
    def setUp(self):
        self.config = {
            "common5": {"R10_coral_weights": [0.4]*5,
                        "R11_coral_weights": [0.7,0.6,0.6,0.9,0.9]},
            "source94": {"coral_weights": [1.0,0.9,0.7,0.9,0.9]},
        }

    def test_fixed_and_fold_selected_rules_are_distinct(self):
        coral = [[1.,0.,0.,0.]]
        deit = [[0.,1.,0.,0.]]
        fixed = common5_fusion(coral, deit, 0, "R10", self.config)
        selected = common5_fusion(coral, deit, 0, "R11", self.config)
        np.testing.assert_allclose(fixed, [[.4,.6,0,0]])
        np.testing.assert_allclose(selected, [[.7,.3,0,0]])
        self.assertEqual(argmax_grades(fixed).tolist(), [1])
        self.assertEqual(argmax_grades(selected).tolist(), [0])

    def test_source94_averages_probabilities_before_argmax(self):
        # Three weak class-0 votes lose to two strong class-1 probabilities.
        values = [np.array([[.51,.49,0.,0.]])] * 3 + [np.array([[0.,1.,0.,0.]])] * 2
        result = source94_fusion(values, values, self.config)
        np.testing.assert_allclose(result, [[.306,.694,0.,0.]])
        self.assertEqual(argmax_grades(result).tolist(), [1])
        self.assertEqual(sum(int(v.argmax()) == 0 for v in values), 3)

    def test_source94_requires_same_row_count_in_all_folds(self):
        single = np.array([[1.,0.,0.,0.]])
        values = [single] * 4 + [np.repeat(single, 2, axis=0)]
        with self.assertRaises(ValueError):
            source94_fusion(values, values, self.config)

    def test_checksum_and_path_binding_reject_wrong_artifacts(self):
        # A small generated artifact tests identity handling without model/patient data.
        temp_base = Path(tempfile.gettempdir()).resolve()
        with tempfile.TemporaryDirectory(prefix="paper2_contract_", dir=temp_base) as temp:
            root = Path(temp).resolve()
            if root.parent != temp_base or not root.name.startswith("paper2_contract_"):
                raise RuntimeError("Generated test directory escaped its intended temporary parent")
            file = root / "dummy.pth"
            file.write_bytes(b"generated-test-artifact")
            entry = {"file":"dummy.pth", "bytes":file.stat().st_size, "sha256":sha256(file)}
            self.assertEqual(checkpoint_path(root, entry), file)
            file.write_bytes(b"generated-test-artifacX")
            with self.assertRaises(ValueError):
                checkpoint_path(root, entry)
            with self.assertRaises(ValueError):
                checkpoint_path(root, dict(entry, file="../outside.pth"))


if __name__ == "__main__":
    unittest.main()
