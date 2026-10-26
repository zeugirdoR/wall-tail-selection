import unittest,math
import numpy as np
from scipy.integrate import quad
from wall_tail_selection import crossover as c
class CrossoverTests(unittest.TestCase):
 def test_derivatives(self):
  x=np.random.default_rng(9).normal(size=64)
  for fam in ['mixture','variance']:
   for eta in [.01,.3,2.]:
    h=1e-4;f=lambda e:float(c.logratio(x,np.array([math.sqrt(e)]),fam).sum());s,j=c.derivatives(x,eta,fam)
    self.assertAlmostEqual(s,(f(eta+h)-f(eta-h))/(2*h),delta=2e-6);self.assertAlmostEqual(j,(f(eta+h)-2*f(eta)+f(eta-h))/h**2,delta=1e-5)
 def test_zero_and_jacobian(self):
  x=np.linspace(-1,1,64);s,h=c.derivatives(x,0,'mixture');self.assertAlmostEqual(s,float(np.sum(x*x-1)/2));self.assertAlmostEqual(h,-float(np.sum(x**4)/6))
  a,flag=c.approximation(x,2,512,'mixture',-1.,'eta');self.assertEqual(flag,0);v=quad(lambda eta:math.exp(s*eta+.5*h*eta*eta)/4,0,4,weight='alg',wvar=(-.5,0),epsabs=1e-12)[0];self.assertAlmostEqual(a,math.log(v),places=9)
 def test_variance_fallback(self):
  a,flag=c.approximation(np.zeros(32),2,256,'variance',-2.,'eta');self.assertEqual(flag,1);self.assertEqual(a,-2.)
 def test_sbic_root(self):
  for n in [32,128,512]:
   for peak in [0,1,10,1000]:
    r=c.sbic(peak,n);la=peak-.25*math.log(n);lb=peak-.5*math.log(n);self.assertAlmostEqual(r+float(np.logaddexp(r,0)),float(np.logaddexp(lb+r,la)),places=9)
 def test_variance_loo(self):
  cfg=dict(prior_upper=2.,nodes=256);x=np.random.default_rng(8).normal(size=32);scores,ref,*_=c.fit(x,cfg,'variance');loo,_,_=c.sg.predictive_scores(ref)
  for i in [0,7]:self.assertAlmostEqual(loo[i],scores[2]-c.fit(np.delete(x,i),cfg,'variance')[0][2],places=9)
 def test_regular_repair(self):
  x=np.random.default_rng(7).normal(size=4096)*math.sqrt(2);s,_,flags,*_=c.fit(x,dict(prior_upper=2.,nodes=512),'variance');self.assertEqual(flags[0],0);self.assertLess(abs(s[0]-s[2]),.05)

 def test_stacking_stability(self):
  for r in [np.array([-2.,2.]),np.array([-1e-8,1e-8]),np.array([-1e-13,1e-13]),np.array([-30.,.01]),np.array([30.,-.01])]:
   w=c.stacking_weight(r);score=lambda a:float(c.sg.mixture_ratio(r,a).sum())
   self.assertGreaterEqual(w,0);self.assertLessEqual(w,1);self.assertGreaterEqual(score(w)+1e-10,max(score(0),score(1)))
  self.assertAlmostEqual(c.stacking_weight(np.array([-2.,2.])),.5,places=10)
 def test_interval_controls(self):
  from wall_tail_selection.crossover_inference import paired_binary,paired_risk,cp
  mean,lo,hi=paired_binary(np.zeros(4096));self.assertEqual(mean,0);self.assertLess(lo,0);self.assertGreater(hi,0);self.assertLess(hi,.01)
  mean,se,lo,hi=paired_risk(np.ones(32)*-.002);self.assertAlmostEqual(lo,-.002);self.assertAlmostEqual(hi,-.002)
  self.assertEqual(cp(0,32)[0],0);self.assertEqual(cp(32,32)[1],1)
