import math,unittest
import numpy as np
from scipy.special import gammainc,gammaln
from wall_tail_selection.mechanism import fit
class MechanismTests(unittest.TestCase):
 def test_uniform_analytic_evidence(self):
  for n in [32,512,2048]:
   x=np.linspace(.01,.93,n);L,U=.5,4.;f=fit(x,1,[L,U]);t=.93
   exact=-n*math.log(t)+math.log1p(-math.exp(n*math.log(t/U)))-math.log(n*math.log(U/L))
   self.assertAlmostEqual(f['scale'],t);self.assertAlmostEqual(f['log_evidence'],exact,places=9)
 def test_exponential_analytic_evidence(self):
  x=np.linspace(.1,1.9,128);n=len(x);S=x.sum();L,U=.5,4.;f=fit(x,None,[L,U])
  exact=gammaln(n)-n*math.log(S)+math.log(gammainc(n,S/L)-gammainc(n,S/U))-math.log(math.log(U/L))
  self.assertAlmostEqual(f['scale'],1.);self.assertAlmostEqual(f['log_evidence'],exact,places=9)
 def test_impossible_is_not_numerical_failure(self):
  f=fit(np.array([1.,5.]),2,[.5,4]);self.assertEqual(f['status'],'impossible');self.assertIsNone(f['occam'])
  self.assertEqual(fit(np.array([1.,5.]),None,[.5,4])['status'],'finite')
 def test_tighter_quadrature_and_boundaries(self):
  x=np.linspace(.001,.99,128);ys=np.linspace(0,8,257)
  for shape in [1.5,1.9,2,2.1,3,None]:
   f=fit(x,shape,[.5,4],ys,1e-8);g=fit(x,shape,[.5,4],ys,1e-10)
   self.assertAlmostEqual(f['log_evidence'],g['log_evidence'],places=8)
   np.testing.assert_allclose(f['cdf'],g['cdf'],atol=1e-8)
   self.assertGreaterEqual(f['occam'],-1e-9)
