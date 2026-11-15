"""Synthetic contract tests only; no study rows, models or performance claims."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "evidence_tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("five_fold_report", TOOLS / "five_fold_report.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def artificial_rows(unbalanced=False, probabilities=False):
    rows = []
    for fold in range(5):
        n = 20 if unbalanced and fold == 4 else 4
        for index in range(n):
            truth = index % 4
            correct = not unbalanced or index < (18 if fold == 4 else 2)
            pred = truth if correct else (truth + 1) % 4
            row = {"roi_id": f"artificial_r{fold}_{index}",
                   "patient_id": f"artificial_p{fold}_{index}",
                   "outer_fold": str(fold), "y_true": str(truth), "y_pred": str(pred)}
            if probabilities:
                row.update({f"p{grade}": str(0.8 if grade == pred else 1/15) for grade in range(4)})
            rows.append(row)
    return rows


class ReportingContracts(unittest.TestCase):
    def report(self, rows, **kwargs):
        return module.build_report(rows, expected_patients=len({r['patient_id'] for r in rows}),
                                   expected_rois=len(rows), **kwargs)

    def test_all_folds_and_declared_complete_scale_are_required(self):
        rows = artificial_rows()
        with self.assertRaises(module.InputContractError):
            self.report(rows[:-4])
        with self.assertRaises(module.InputContractError):
            module.build_report(rows, expected_patients=21, expected_rois=20)

    def test_patient_crossing_folds_and_duplicate_roi_are_rejected(self):
        rows = artificial_rows()
        rows[4]['patient_id'] = rows[0]['patient_id']
        with self.assertRaisesRegex(module.InputContractError, "patient occurs"):
            self.report(rows)
        rows = artificial_rows()
        rows[4]['roi_id'] = rows[0]['roi_id']
        with self.assertRaisesRegex(module.InputContractError, "Duplicate ROI"):
            self.report(rows)

    def test_fold_and_grade_values_are_canonical(self):
        rows = artificial_rows()
        rows[0]['y_pred'] = '4'
        with self.assertRaises(module.InputContractError):
            self.report(rows)
        rows[0]['y_pred'] = '0'
        rows[0]['outer_fold'] = '0.0'
        with self.assertRaises(module.InputContractError):
            self.report(rows)

    def test_unequal_folds_distinguish_mean_and_pooled_and_use_sample_sd(self):
        report = self.report(artificial_rows(unbalanced=True))
        values = report['fold_mean_sample_SD']['metrics']['accuracy']
        self.assertAlmostEqual(values['mean'], 0.58)
        self.assertAlmostEqual(values['sample_SD_ddof1'], math.sqrt(0.032))
        self.assertAlmostEqual(report['pooled_OOF']['metrics']['accuracy'], 26/36)
        self.assertEqual(values['n_finite_fold_values'], 5)

    def test_probabilities_require_explicit_origin_declaration(self):
        report = self.report(artificial_rows(probabilities=True))
        self.assertFalse(report['probability_status']['computed'])
        self.assertTrue(report['probability_status']['argmax_parity_verified'])
        self.assertIsNone(report['pooled_OOF']['probability_metrics']['multiclass_log_loss'])
        report = self.report(artificial_rows())
        self.assertFalse(report['probability_status']['argmax_parity_verified'])
        with self.assertRaises(module.InputContractError):
            self.report(artificial_rows(), probabilities_verified_oof=True)

    def test_verified_probability_metrics_have_independent_manual_targets(self):
        report = self.report(artificial_rows(probabilities=True), probabilities_verified_oof=True)
        values = report['pooled_OOF']['probability_metrics']
        self.assertAlmostEqual(values['multiclass_log_loss'], -math.log(0.8))
        self.assertAlmostEqual(values['multiclass_brier_sum_over_4_classes'], 4/75)
        self.assertAlmostEqual(values['top_label_ECE_10_equal_width_bins'], 0.2)
        self.assertFalse(report['probability_status']['origin_independently_certified_by_this_tool'])

    def test_invalid_probability_mass_or_argmax_is_rejected(self):
        rows = artificial_rows(probabilities=True)
        rows[0]['p1'] = '0.8'
        with self.assertRaises(module.InputContractError):
            self.report(rows, probabilities_verified_oof=True)
        rows = artificial_rows(probabilities=True)
        rows[0]['p0'], rows[0]['p1'] = rows[0]['p1'], rows[0]['p0']
        with self.assertRaisesRegex(module.InputContractError, 'argmax'):
            self.report(rows, probabilities_verified_oof=True)

    def test_zero_support_and_undefined_qwk_are_null_without_partial_mean(self):
        rows = artificial_rows()
        for row in rows:
            row['y_true'] = row['y_pred'] = '0'
        report = self.report(rows)
        self.assertIsNone(report['pooled_OOF']['metrics']['qwk'])
        self.assertIsNone(report['fold_mean_sample_SD']['metrics']['qwk']['mean'])
        self.assertIsNone(report['pooled_OOF']['targeted_errors']['severe_underestimated']['rate'])
        json.dumps(report, allow_nan=False)

    def test_cli_never_overwrites_and_exports_no_identifiers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root/'input.csv', root/'report.json'
            rows = artificial_rows()
            import csv
            with source.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=module.REQUIRED_COLUMNS)
                writer.writeheader(); writer.writerows(rows)
            args = ['--predictions', str(source), '--expected-patients', '20',
                    '--expected-rois', '20', '--output', str(output)]
            self.assertEqual(module.main(args), 0)
            self.assertNotIn('artificial_p', output.read_text(encoding='utf-8'))
            self.assertNotIn('artificial_r', output.read_text(encoding='utf-8'))
            before = output.read_bytes()
            with self.assertRaises(SystemExit):
                module.main(args)
            self.assertEqual(output.read_bytes(), before)

    def test_partial_probability_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'bad.csv'
            source.write_text('roi_id,patient_id,outer_fold,y_true,y_pred,p0\nr,p,0,0,0,1\n', encoding='utf-8')
            with self.assertRaises(module.InputContractError):
                module.load_csv(source)


if __name__ == '__main__':
    unittest.main()
