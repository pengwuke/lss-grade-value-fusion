import csv,hashlib,json,tempfile,unittest,subprocess,sys,os
from pathlib import Path
from paper2_statistics.runner import load_plan,run
from importlib.resources import files

class DirectPlanTests(unittest.TestCase):
    def test_legacy_plans_unchanged(self):
        p=json.loads(files('paper2_statistics').joinpath('plans.json').read_text())
        p.pop('r11_natural');p.pop('r11_robustness')
        self.assertEqual(hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'f9ef98d40c4f48f542b2ef3cd155718078d54029efd6164098e3e1ce6c515022')
    def test_direct_plan_and_identity_pair_with_reordered_rows(self):
        p=load_plan('r11_natural')
        self.assertEqual(p['systems'],['R09_SELECTED_ALPHA_VSEL','A0_NATURAL_VSEL'])
        self.assertEqual(p['contrasts'][0]['terms'],[[0,1],[1,-1]])
        self.assertEqual((p['repeats'],p['seed']),(20000,42))
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);mapping={}
            rows=[[f'roi_{i}',f'patient_{i//4}',(i//4)%5,i%4,i%4] for i in range(40)]
            for i,name in enumerate(p['systems']):
                path=d/(name+'.csv')
                with path.open('w',newline='') as f:
                    w=csv.writer(f);w.writerow(['roi_id','patient_id','outer_fold','y_true','y_pred']);w.writerows(rows if i==0 else rows[::-1])
                mapping[name]=str(path)
            inp=d/'inputs.json';inp.write_text(json.dumps({'predictions':mapping}))
            r=run(inp,'r11_natural',37)
            self.assertIn('Post-hoc',r['analysis_scope'])
            self.assertFalse(r['paper_cohort_identity_verified'])
            self.assertEqual(r['paired_bootstrap']['patients'],10)
            self.assertEqual(r['input_alignment']['systems_reordered_by_roi_id'],[p['systems'][1]])
            for metric in r['paired_bootstrap']['contrasts'][0]['metrics'].values():
                self.assertEqual(metric['point_delta'],0)
                self.assertEqual(metric['percentile_95ci'],[0.,0.])
                self.assertEqual(metric['valid_replicates'],37)
            output=d/'cli.json'
            proc=subprocess.run([sys.executable,'-m','paper2_statistics','--analysis','r11_natural','--inputs',str(inp),'--output',str(output),'--iterations','37'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            self.assertEqual(proc.returncode,0,proc.stdout.decode())
            cli=json.loads(output.read_text())
            self.assertEqual(cli['paired_bootstrap'],r['paired_bootstrap'])
            self.assertIn('Post-hoc',cli['analysis_scope'])

