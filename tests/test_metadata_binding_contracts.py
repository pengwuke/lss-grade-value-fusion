import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from paper2_runtime.artifacts import read_fusion_config, read_manifest


class MetadataBindingContracts(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.gettempdir()).resolve()
        self.temp = tempfile.TemporaryDirectory(prefix="paper2_r5_metadata_", dir=self.base)
        self.root = Path(self.temp.name).resolve()
        assert self.root.parent == self.base and self.root.name.startswith("paper2_r5_metadata_")
        self.checkpoint = self.root / "synthetic.pth"
        self.checkpoint.write_bytes(b"generated-byte-fixture-not-a-model")
        self.manifest = self.root / "models_manifest.json"
        self.config = self.root / "fusion_config.json"
        self.record_path = self.root / "CODE_MODEL_COMPATIBILITY.json"
        self.entries = [{"group": "synthetic", "family": "deit_tiny", "fold": 0,
                         "file": "synthetic.pth", "bytes": self.checkpoint.stat().st_size,
                         "sha256": self.hash(self.checkpoint), "tensor_value_sha256": "synthetic"}]
        self.settings = {"common5": {"R11_coral_weights": [.7, .6, .6, .9, .9]}}
        self.manifest.write_text(json.dumps({"model_entries": self.entries}), encoding="utf-8")
        self.config.write_text(json.dumps(self.settings), encoding="utf-8")
        self.record = {"model_manifest_sha256": self.hash(self.manifest),
                       "fusion_config_sha256": self.hash(self.config)}
        self.write_record(self.record)

    @staticmethod
    def hash(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_record(self, record):
        self.record_path.write_text(json.dumps(record), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def cli(self, command):
        # Run the candidate package deliberately; no Torch import is needed for
        # verify or for a metadata failure before load-check begins loading.
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        return subprocess.run([sys.executable, "-m", "paper2_runtime", command,
                               "--models", str(self.root)], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def test_valid_metadata_and_case_insensitive_hex(self):
        self.write_record({key: value.upper() for key, value in self.record.items()})
        self.assertEqual(read_manifest(self.root), self.entries)
        self.assertEqual(read_fusion_config(self.root), self.settings)

    def test_modified_fusion_config_is_rejected(self):
        modified = copy.deepcopy(self.settings)
        modified["common5"]["R11_coral_weights"][0] = 0.
        self.config.write_text(json.dumps(modified), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fusion_config.json SHA256 differs"):
            read_fusion_config(self.root)

    def test_modified_model_role_manifest_is_rejected(self):
        modified = copy.deepcopy(self.entries)
        modified[0]["fold"] = 1
        self.manifest.write_text(json.dumps({"model_entries": modified}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "models_manifest.json SHA256 differs"):
            read_manifest(self.root)

    def test_missing_compatibility_record_is_rejected_by_both_readers(self):
        self.record_path.unlink()
        for reader in [read_manifest, read_fusion_config]:
            with self.subTest(reader=reader.__name__), self.assertRaisesRegex(ValueError, "CODE_MODEL_COMPATIBILITY"):
                reader(self.root)

    def test_missing_malformed_and_wrong_fingerprints_are_rejected(self):
        for key, reader in [("model_manifest_sha256", read_manifest), ("fusion_config_sha256", read_fusion_config)]:
            for invalid in [None, "", "abc", "g" * 64, 12, "0" * 64]:
                record = dict(self.record)
                record[key] = invalid
                self.write_record(record)
                with self.subTest(key=key, value=invalid), self.assertRaises(ValueError):
                    reader(self.root)
            record = dict(self.record)
            del record[key]
            self.write_record(record)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                reader(self.root)

    def test_invalid_json_record_and_bound_json_raise_valueerror(self):
        for data in [b"not-json", b"[]", b"\xff"]:
            self.record_path.write_bytes(data)
            with self.subTest(record=data), self.assertRaises(ValueError):
                read_manifest(self.root)
        self.config.write_bytes(b"not-json")
        record = dict(self.record)
        record["fusion_config_sha256"] = self.hash(self.config)
        self.write_record(record)
        with self.assertRaisesRegex(ValueError, "Invalid JSON in bound model metadata"):
            read_fusion_config(self.root)

    def test_cli_verify_reports_metadata_and_weight_scope_without_forward_claim(self):
        result = self.cli("verify")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["model_manifest_sha256_pass"])
        self.assertTrue(report["fusion_config_sha256_pass"])
        self.assertEqual(report["checkpoint_sha256_pass"], 1)
        self.assertFalse(report["synthetic_forward_performed"])
        self.assertFalse(report["patient_inference_performed"])
        self.assertNotIn("strict_architecture_load_pass", report)

    def test_cli_verify_and_load_check_cannot_pass_edited_config(self):
        modified = copy.deepcopy(self.settings)
        modified["common5"]["R11_coral_weights"][0] = 0.
        self.config.write_text(json.dumps(modified), encoding="utf-8")
        for command in ["verify", "load-check"]:
            result = self.cli(command)
            with self.subTest(command=command):
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("fusion_config.json SHA256 differs", result.stderr)
                self.assertNotIn("checkpoint_sha256_pass", result.stdout)


if __name__ == "__main__":
    unittest.main()
