"""Path contracts around unchanged kernels; only artificial IDs and CSV rows."""
import contextlib
import csv
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from paper2_statistics import __main__ as cli
from paper2_statistics.inputs import load_aligned, read_predictions
from paper2_statistics.runner import load_plan


class StatisticsHomePaths(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='paper2_r7_stats_home_')
        self.home = Path(self.temp.name).resolve()/'ARTIFICIAL HOME';self.home.mkdir()
        self.env = patch.dict(os.environ,{'HOME':str(self.home),'USERPROFILE':str(self.home)})
        self.env.start()
        self.csv = self.home/'ARTIFICIAL predictions.csv'
        with self.csv.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream);writer.writerow(['roi_id','patient_id','outer_fold','y_true','y_pred'])
            writer.writerows([[f'ARTIFICIAL_ROI_{i}',f'ARTIFICIAL_PATIENT_{i}',i,i%4,(i+1)%4] for i in range(5)])

    def tearDown(self):
        self.env.stop();self.temp.cleanup()

    def argv(self,output):
        return ['paper2_statistics','--analysis','single','--inputs','~/inputs.json','--output',output,'--iterations','7']

    def test_direct_csv_home_path(self):
        rows=read_predictions('~/ARTIFICIAL predictions.csv')
        self.assertEqual(len(rows),5)
        self.assertEqual(rows['ARTIFICIAL_ROI_3'],('ARTIFICIAL_PATIENT_3',3,3,0))

    def test_home_manifest_keeps_relative_and_home_csv_paths_and_hashes(self):
        systems=load_plan('single')['systems']
        mapping={name:('~/ARTIFICIAL predictions.csv' if i==0 else 'ARTIFICIAL predictions.csv') for i,name in enumerate(systems)}
        (self.home/'inputs.json').write_text(json.dumps({'predictions':mapping}),encoding='utf-8')
        y,p,patients,folds,info=load_aligned('~/inputs.json',systems)
        self.assertEqual(p.shape,(5,4))
        self.assertEqual(folds.tolist(),list(range(5)))
        self.assertEqual({e['sha256'] for e in info['input_files']},{hashlib.sha256(self.csv.read_bytes()).hexdigest()})

    def test_report_home_output_and_existing_destination_guard(self):
        target=self.home/'reports/new.json'
        report={'paper_resampling_configuration':False,'artificial_path_test':True}
        with patch.object(cli,'run',return_value=report),patch.object(sys,'argv',self.argv('~/reports/new.json')),contextlib.redirect_stdout(StringIO()):
            cli.main()
        self.assertTrue(json.loads(target.read_text())['artificial_path_test'])
        protected=target.read_bytes()
        with patch.object(cli,'run',side_effect=AssertionError('resampling must not start')),patch.object(sys,'argv',self.argv('~/reports/new.json')),contextlib.redirect_stderr(StringIO()),self.assertRaises(SystemExit):
            cli.main()
        self.assertEqual(target.read_bytes(),protected)


if __name__=='__main__':unittest.main()
