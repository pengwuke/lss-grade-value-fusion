"""Artificial near-tie regressions, derived from the frozen float64 protocol."""
import unittest
import numpy as np
from paper2_runtime.fusion import common5_fusion,matched_common5_fusion,source94_fusion,source94_reference_average

class Common5FusionPrecision(unittest.TestCase):
    def setUp(self):
        self.config={'common5':{'R10_coral_weights':[.4]*5,'R11_coral_weights':[.7,.6,.6,.9,.9]},
                     'matched_common5':{'fixed_image_weight':.4,'selected_image_weights':{'A3_GRADE_FIXED':[.6]*5,'PHADKE_P4C':[.4]*5}},
                     'source94':{'coral_weights':[1.,.9,.7,.9,.9]}}

    def test_known_common5_near_ties_match_historical_grades(self):
        examples=[('R10',0,[.5080505609512329,.4919494390487671,0,0],[.49463292956352234,.5053670406341553,0,0]),
                  ('R11',0,[.32468563318252563,.6753143668174744,0,0],[.9090667963027954,.09093320369720459,0,0]),
                  ('R11',1,[.33943280577659607,.6605671644210815,0,0],[.7408507466316223,.2591492533683777,0,0]),
                  ('R11',3,[.4706898033618927,.5293102264404297,0,0],[.7637917399406433,.2362082600593567,0,0])]
        for strategy,fold,c,d in examples:
            with self.subTest(strategy=strategy,fold=fold):
                c=np.array([c],dtype=np.float32);d=np.array([d],dtype=np.float32)
                result=common5_fusion(c,d,fold,strategy,self.config)
                self.assertEqual(result.dtype,np.dtype('float64'))
                self.assertEqual(result.argmax(1).tolist(),[1])
                self.assertLess(result[0,0],result[0,1])

    def test_matched_common5_preserves_raw_scores_and_float64_inputs(self):
        c=np.array([[.5080505609512329,.4919494390487671,0,0]],dtype=np.float32)
        d=np.array([[.49463292956352234,.5053670406341553,0,0]],dtype=np.float32)
        actual=matched_common5_fusion(c,d,0,'A3_GRADE_FIXED','fixed',self.config)
        self.assertEqual(actual.argmax(1).tolist(),[1])
        for arm,score in [('A3_GRADE_FIXED',c.astype(np.float64)),('PHADKE_P4C',np.array([[1.,1e-6,1e-6,1e-6]],dtype=np.float64))]:
            expected=.4*score+.6*d.astype(np.float64)
            actual=matched_common5_fusion(score,d.astype(np.float64),0,arm,'fixed',self.config)
            np.testing.assert_array_equal(actual,expected)

    def test_separate_source94_and_sr1_paths_preserve_their_dtype(self):
        c=np.array([[.61,.39,0,0]],dtype=np.float32);d=np.array([[.27,.73,0,0]],dtype=np.float32)
        coral=[c]*5;deit=[d]*5
        # Unchanged generic pair rule and averaging for the separate protocol.
        expected=np.stack([float(w)*c+(1.-float(w))*d for w in self.config['source94']['coral_weights']]).mean(0)
        actual=source94_fusion(coral,deit,self.config)
        self.assertEqual(actual.dtype,np.dtype('float32'))
        np.testing.assert_array_equal(actual,expected)
        actual=source94_reference_average(coral)
        self.assertEqual(actual.dtype,np.dtype('float32'))
        np.testing.assert_array_equal(actual,np.stack(coral).mean(0))

if __name__=='__main__':unittest.main()
