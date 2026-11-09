import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.metrics import cohen_kappa_score

from paper2_statistics.inputs import read_predictions, load_aligned
from paper2_statistics.runner import run, load_plan, verify_kernels
from paper2_statistics.kernels import fusion as kernel


class StatisticsContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # Artificial identifiers; 5 folds, uneven numbers of ROIs per patient.
        self.rows = [[f"synthetic_roi_{i}_{j}", f"synthetic_patient_{i}", str(i), str((i+j)%4), str((2*i+j)%4)]
                     for i in range(5) for j in range(i%3+1)]

    def tearDown(self):
        self.tmp.cleanup()

    def csv(self, name, rows):
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as stream:
            w = csv.writer(stream);w.writerow(["roi_id", "patient_id", "outer_fold", "y_true", "y_pred"])
            w.writerows(rows)
        return path

    def manifest(self, analysis="single", reverse=False):
        systems = load_plan(analysis)["systems"]
        mapping = {}
        for i, name in enumerate(systems):
            rows = [r[:4]+[str((int(r[4])+i)%4)] for r in self.rows]
            if reverse and i: rows.reverse()
            mapping[name] = self.csv(name+".csv", rows).name
        path=self.root/"inputs.json";path.write_text(json.dumps({"predictions":mapping}),encoding="utf-8")
        return path

    def test_shuffled_rows_align_by_identity(self):
        path=self.manifest(reverse=True)
        y,p,patients,folds,info=load_aligned(path,load_plan("single")["systems"])
        self.assertEqual(p.shape,(len(self.rows),4))
        np.testing.assert_array_equal(p[:,1],(p[:,0]+1)%4)
        self.assertEqual(len(info["systems_reordered_by_roi_id"]),3)

    def test_duplicate_and_patient_fold_leakage_rejected(self):
        for rows in [self.rows+[self.rows[0]], self.rows+[["extra",self.rows[0][1],"4","0","0"]]]:
            with self.subTest(rows=len(rows)),self.assertRaises(ValueError):
                read_predictions(self.csv("bad.csv",rows))

    def test_between_system_identity_mismatches_rejected(self):
        for field,value in [(0,"other_roi"),(1,"other_patient"),(2,"4"),(3,"3")]:
            path=self.manifest();plan=load_plan("single");name=plan["systems"][1]
            target=self.root/(name+".csv")
            rows=[r[:] for r in self.rows];rows[0][field]=value
            self.csv(target.name,rows)
            with self.subTest(field=field),self.assertRaises(ValueError):load_aligned(path,plan["systems"])

    def test_invalid_values_and_schema_rejected(self):
        for field,value in [(0," "),(1,"id "),(2,"0.0"),(3,"-1"),(4,"4"),(4,"1.0"),(4,"01")]:
            rows=[r[:] for r in self.rows];rows[0][field]=value
            with self.subTest(value=value),self.assertRaises(ValueError):read_predictions(self.csv("bad.csv",rows))
        path=self.csv("bad.csv",self.rows)
        path.write_text(path.read_text().replace("y_pred","y_true",1))
        with self.assertRaises(ValueError):read_predictions(path)
        duplicate=self.root/"duplicate.json"
        duplicate.write_text('{"predictions":{},"predictions":{}}',encoding="utf-8")
        with self.assertRaises(ValueError):load_aligned(duplicate,load_plan("single")["systems"])

    def test_active_label_and_undefined_qwk(self):
        y=np.array([0,0,1,3,3]);p=np.array([0,1,3,1,3])
        values=kernel.batch_metrics(kernel.cm(y,p)[None])[0]
        np.testing.assert_allclose(values,kernel.reference(y,p),rtol=0,atol=1e-12,equal_nan=True)
        self.assertNotAlmostEqual(values[4],cohen_kappa_score(y,p,labels=[0,1,2,3],weights="quadratic"))
        zero=np.zeros(4,dtype=np.int64)
        self.assertIsNone(kernel.point(zero,zero)["metrics"]["qwk"])
        bootstrap=kernel.base.paired_patient_bootstrap(zero,np.zeros((4,4),dtype=np.int64),np.array(["a","a","b","b"]),iterations=17)
        for contrast in bootstrap["contrasts"]:
            metric=contrast["metrics"]["qwk"]
            self.assertEqual(metric["valid_replicates"],0)
            self.assertEqual(metric["undefined_replicates"],17)
            self.assertIsNone(metric["percentile_95ci"])

    def test_wrapper_matches_frozen_kernels_all_three_plans(self):
        verify_kernels()
        for analysis,n in [("single",24),("fixed",90),("selected",72)]:
            path=self.manifest(analysis,reverse=True)
            result=run(path,analysis,iterations=37)
            plan=load_plan(analysis)
            y,p,patients,folds,_=load_aligned(path,plan["systems"])
            expected=(kernel.base.paired_patient_bootstrap(y,p,patients,iterations=37,seed=42,chunk=64)
                      if analysis=="single" else kernel.paired_fusion_bootstrap(y,p,patients,plan["systems"],plan["contrasts"],iterations=37,seed=42,chunk=32))
            self.assertEqual(result["paired_bootstrap"],expected)
            self.assertEqual(sum(len(c["metrics"]) for c in expected["contrasts"]),n)
            self.assertFalse(result["paper_resampling_configuration"])
            json.dumps(result,allow_nan=False)

    def test_patient_multiplicities_match_independent_direct_resampling(self):
        path=self.manifest();plan=load_plan("single")
        y,p,patients,folds,_=load_aligned(path,plan["systems"])
        count=29;report=kernel.base.paired_patient_bootstrap(y,p,patients,iterations=count,seed=42)
        unique=np.unique(patients);rng=np.random.default_rng(42)
        draws=rng.integers(0,len(unique),size=(count,len(unique)),dtype=np.int32)
        deltas=[]
        for sample in draws:
            indices=np.concatenate([np.flatnonzero(patients==unique[i]) for i in sample])
            deltas.append(kernel.reference(y[indices],p[indices,3])-kernel.reference(y[indices],p[indices,1]))
        deltas=np.array(deltas)
        for k,name in enumerate(kernel.METRICS):
            finite=deltas[np.isfinite(deltas[:,k]),k]
            expected=np.percentile(finite,[2.5,97.5])
            np.testing.assert_allclose(report["contrasts"][0]["metrics"][name]["percentile_95ci"],expected,rtol=0,atol=1e-12)


if __name__=="__main__":unittest.main()
