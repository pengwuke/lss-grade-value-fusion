import csv,hashlib,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from importlib.resources import files
import numpy as np
from paper2_statistics.robustness import run_robustness
from paper2_statistics.kernels import fusion

class RobustnessContracts(unittest.TestCase):
    def arrays(self):
        y=np.tile(np.arange(4),5);patients=np.repeat(np.array(['p0','p1','p2','p3','p4']),4);folds=np.repeat(np.arange(5),4)
        a=y.copy();b=y.copy();a[4:]=(a[4:]+1)%4;b[:4]=(b[:4]+1)%4
        return y,np.stack([a,b],axis=1),patients,folds
    def test_prior_four_plans_and_legacy_interval_parity(self):
        plans=json.loads(files('paper2_statistics').joinpath('plans.json').read_text());plans.pop('r11_robustness')
        self.assertEqual(hashlib.sha256(json.dumps(plans,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'ab3ed83a2fd08a65e10ac42a3881035752e7cd1e9f941f2136175b58276067c7')
        y,p,patients,folds=self.arrays();r=run_robustness(y,p,patients,folds,37)
        old=fusion.paired_fusion_bootstrap(y,p,patients,['R11','natural'],[{'name':'direct','role':'test','terms':[[0,1],[1,-1]]}],iterations=37,seed=42)
        for k,v in old['contrasts'][0]['metrics'].items():self.assertEqual(r['outcomes'][k]['percentile_95ci'],v['percentile_95ci'])
        self.assertEqual(r['family_percentile_bounds'],[.25,99.75]);self.assertEqual(r['family_size'],10)
    def test_equal_patient_weighting_resists_roi_replication(self):
        y,p,patients,folds=self.arrays();r=run_robustness(y,p,patients,folds,37)
        ii=np.r_[np.arange(20),np.arange(4)];s=run_robustness(y[ii],p[ii],patients[ii],folds[ii],37)
        self.assertAlmostEqual(r['equal_patient_weighted_pooled_metrics']['accuracy']['delta'],-.6)
        self.assertAlmostEqual(s['equal_patient_weighted_pooled_metrics']['accuracy']['delta'],-.6)
        self.assertAlmostEqual(s['outcomes']['accuracy']['point_delta'],-1/3)
        self.assertAlmostEqual(r['outcomes']['accuracy']['point_delta'],-.6)
        self.assertEqual(len(r['delete_one_outer_fold']),5)
    def test_identical_paired_systems_and_real_synthetic_cli(self):
        y,p,patients,folds=self.arrays();p[:,1]=p[:,0]
        r=run_robustness(y,p,patients,folds,37)
        for v in r['outcomes'].values():
            self.assertEqual(v['point_delta'],0);self.assertEqual(v['percentile_95ci'],[0.,0.]);self.assertEqual(v['bonferroni_type_99_5ci'],[0.,0.])
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);mapping={}
            for s,name in enumerate(['R09_SELECTED_ALPHA_VSEL','A0_NATURAL_VSEL']):
                q=d/(name+'.csv')
                with q.open('w',newline='') as f:
                    w=csv.writer(f);w.writerow(['roi_id','patient_id','outer_fold','y_true','y_pred']);w.writerows([str(i),patients[i],str(folds[i]),str(y[i]),str(p[i,s])] for i in range(len(y)))
                mapping[name]=str(q)
            inp=d/'inputs.json';inp.write_text(json.dumps({'predictions':mapping}));output=d/'report.json'
            cmd=[sys.executable,'-m','paper2_statistics','--analysis','r11_robustness','--inputs',str(inp),'--output',str(output),'--iterations','37']
            proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);self.assertEqual(proc.returncode,0,proc.stdout.decode())
            result=json.loads(output.read_text());self.assertEqual(result['robustness']['family_size'],10);self.assertFalse(result['paper_cohort_identity_verified'])
