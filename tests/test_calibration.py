import unittest
import numpy as np
from wall_tail_selection.calibration import objective,choose,log_counts
class CalibrationTests(unittest.TestCase):
 def test_pairwise_cancels_common_shift(self):
  cfg=dict(shapes=[1,None],sample_sizes=[32],domains=[[.5,4]])
  rows=[dict(domain=0,truth='endpoint_a1',n=32,replicate=0,status='finite',model=m,occam=v) for m,v in [('endpoint_a1',3),('exponential',1)]]
  pen=np.array([[[[3.,1.]]],[[[9.,7.]]],[[[3.,2.]]]])
  curve,_,counts=objective(rows,cfg,pen);np.testing.assert_allclose(curve,[0,0,1]);self.assertEqual(counts['finite_pairs'],1)
 def test_tie_prefers_one(self):self.assertEqual(choose(np.zeros(3),[.5,1,2]),1)
 def test_impossible_omitted(self):
  cfg=dict(shapes=[1,None],sample_sizes=[32],domains=[[.5,4]])
  rows=[dict(domain=0,truth='endpoint_a1',n=32,replicate=0,status='finite',model=m,occam=v) for m,v in [('endpoint_a1',3),('exponential',1)]]
  rows+=[dict(rows[0],replicate=1,status='impossible'),dict(rows[1],replicate=1)]
  _,_,c=objective(rows,cfg,np.array([[[[3.,1.]]]]));self.assertEqual(c['excluded_datasets'],1)
 def test_nonregular_control_is_identifiable(self):
  p=log_counts([1,None],[2048],[[.5,4]],[.5,2],False)
  self.assertGreater(abs((p[0,0,0,0]-p[0,0,0,1])-(p[1,0,0,0]-p[1,0,0,1])),1.)
