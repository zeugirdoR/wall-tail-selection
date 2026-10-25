import unittest,math
import numpy as np
from scipy.special import gamma
from scipy.integrate import quad
from wall_tail_selection.singular import *
class SingularTests(unittest.TestCase):
 def test_integral_at_zero(self):self.assertAlmostEqual(local_integral(0),math.log(gamma(.25)/(2*math.sqrt(2))),places=11)
 def test_density_normalization(self):
  for mu in [0,.2,1,2]:
   val=quad(lambda x:math.exp(float(logphi(x))+float(logcosh(mu*x))-.5*mu*mu),-12,12,epsabs=1e-11)[0];self.assertAlmostEqual(val,1.,places=10)
 def test_evidence_and_loo(self):
  x=np.random.default_rng(6).normal(size=32);r=reference(x);peak=mle(x,2)[1]
  exact=peak+math.log(quad(lambda mu:math.exp(float(logratio(x,np.array([mu])).sum())-peak)/2,0,2,epsabs=1e-12)[0]);self.assertAlmostEqual(exact,r['logbf'],places=9)
  loo,_,_=predictive_scores(r)
  for i in [0,5,20]:self.assertAlmostEqual(loo[i],r['logbf']-reference(np.delete(x,i))['logbf'],places=9)
 def test_refinement(self):
  for mu in [0,.2,1]:
   x=sample(np.random.default_rng(3),mu,512);a=reference(x,nodes=256);b=reference(x,nodes=512);self.assertAlmostEqual(a['logbf'],b['logbf'],places=8);np.testing.assert_allclose(predictive_ratio(np.linspace(-5,5,31),a),predictive_ratio(np.linspace(-5,5,31),b),atol=1e-8,rtol=0)
 def test_limiting_quartic(self):
  # Balanced deterministic moments S2=n,S4=3n: x=0 with2/3 mass and±sqrt3 with1/6 each.
  errors=[]
  for n in [600,6000]:
   x=np.r_[np.zeros(2*n//3),np.full(n//6,math.sqrt(3)),np.full(n//6,-math.sqrt(3))];lr=float(logratio(x,np.array([n**(-.25)])).sum());errors.append(abs(lr+.25))
  self.assertLess(errors[1],errors[0]/2)
 def test_stacking(self):
  self.assertEqual(stacking_weight(np.array([-1.,-2.])),0);self.assertEqual(stacking_weight(np.array([1.,2.])),1);self.assertAlmostEqual(stacking_weight(np.array([-2.,2.])),.5,places=8)

 def test_integral_ode(self):
  for z in [-2.,0.,3.]:
   peak=max(z,0)**2/2
   f=lambda t:math.exp(t*t*z/math.sqrt(2)-t**4/4-peak)
   I=quad(f,0,math.inf)[0];Ip=quad(lambda t:t*t/math.sqrt(2)*f(t),0,math.inf)[0];Ipp=quad(lambda t:.5*t**4*f(t),0,math.inf)[0]
   self.assertAlmostEqual(Ipp,z*Ip+I/2,places=8)
 def test_positive_z_asymptotic(self):
  errors=[]
  for z in [10.,30.]:
   leading=z*z/2+math.log(math.pi)/2-math.log(2)/4-math.log(z)/2
   errors.append(abs(local_integral(z)-leading))
  self.assertLess(errors[1],errors[0]);self.assertLess(errors[1],.001)
