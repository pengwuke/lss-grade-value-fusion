import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from paper2_runtime.artifacts import find_entry, read_manifest
from paper2_runtime.fusion import common5_fusion, matched_common5_fusion, source94_fusion


class FoldAndPairCountContracts(unittest.TestCase):
    def setUp(self):
        self.config={"common5":{"R10_coral_weights":[.4]*5,"R11_coral_weights":[.7,.6,.6,.9,.9]},
                     "matched_common5":{"fixed_image_weight":.4,"selected_image_weights":{"A0_NATURAL":[.8,.5,.6,.7,.5]}},
                     "source94":{"coral_weights":[1.,.9,.7,.9,.9]}}
        self.coral=np.array([[1.,0.,0.,0.]])
        self.deit=np.array([[0.,1.,0.,0.]])
        self.base=Path(tempfile.gettempdir()).resolve()
        self.temp=tempfile.TemporaryDirectory(prefix='paper2_r4_contract_',dir=self.base)
        self.root=Path(self.temp.name).resolve()
        assert self.root.parent==self.base and self.root.name.startswith('paper2_r4_contract_')
        self.entries=[{"group":"synthetic","family":"deit_tiny","fold":i} for i in range(5)]
        self.manifest=self.root/'models_manifest.json'
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps({"model_entries":self.entries}),encoding='utf-8')
        config=self.root/'fusion_config.json'
        config.write_text(json.dumps(self.config),encoding='utf-8')
        record={'model_manifest_sha256':hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                'fusion_config_sha256':hashlib.sha256(config.read_bytes()).hexdigest()}
        (self.root/'CODE_MODEL_COMPATIBILITY.json').write_text(json.dumps(record),encoding='utf-8')

    def tearDown(self):self.temp.cleanup()

    def test_invalid_fold_cannot_select_any_model_or_fusion(self):
        for value in [True,False,.9,1.0,1.9,np.float64(2),"1",float('nan'),float('inf'),-1,5]:
            for action in [lambda v:find_entry(self.root,'synthetic','deit_tiny',v),
                           lambda v:common5_fusion(self.coral,self.deit,v,'R11',self.config),
                           lambda v:matched_common5_fusion(self.coral,self.deit,v,'A0_NATURAL','selected',self.config)]:
                with self.subTest(value=str(value)),self.assertRaises(ValueError):action(value)

    def test_numpy_integer_fold_keeps_correct_role_and_weight(self):
        self.assertEqual(find_entry(self.root,'synthetic','deit_tiny',np.int64(2))['fold'],2)
        np.testing.assert_allclose(common5_fusion(self.coral,self.deit,np.int64(3),'R11',self.config),[[.9,.1,0.,0.]])
        # At the known 0.5 matched weight the first-index argmax tie remains grade 0.
        result=matched_common5_fusion(self.coral,self.deit,np.int32(1),'A0_NATURAL','selected',self.config)
        np.testing.assert_array_equal(result,[[.5,.5,0.,0.]])
        self.assertEqual(result.argmax(1).tolist(),[0])

    def test_manifest_cannot_assign_boolean_or_fractional_fold(self):
        for value in [True,.9,1.0,'1']:
            self.entries[0]['fold']=value;self.write_manifest()
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,"Fold must"):
                read_manifest(self.root)

    def test_source94_cannot_drop_a_model_pair_silently(self):
        values=[np.array([[.8,.2,0.,0.]])]*3+[np.array([[0.,1.,0.,0.]])]*2
        for weights in [[],[1.]*4,[1.]*6]:
            config=copy.deepcopy(self.config);config['source94']['coral_weights']=weights
            with self.subTest(length=len(weights)),self.assertRaises(ValueError):source94_fusion(values,values,config)
        result=source94_fusion(values,values,self.config)
        np.testing.assert_allclose(result,[[.48,.52,0.,0.]])
        self.assertEqual(result.argmax(1).tolist(),[1])

    def test_source94_weights_must_be_a_finite_five_vector(self):
        values=[self.coral]*5
        for weights in [[[1.]]*5,[1.,1.,1.,1.,float('nan')],[1.,1.,1.,1.,float('inf')],[1.,1.,1.,1.,-0.1],[1.,1.,1.,1.,1.1]]:
            config=copy.deepcopy(self.config);config['source94']['coral_weights']=weights
            with self.subTest(weights=str(weights)),self.assertRaises(ValueError):source94_fusion(values,values,config)


if __name__=='__main__':unittest.main()
