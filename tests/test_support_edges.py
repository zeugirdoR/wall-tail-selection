import unittest
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import expon
from wall_tail_selection.selection import log_likelihoods
class SupportEdges(unittest.TestCase):
 def test_impossible_endpoints_have_zero_weight(self):
  x=torch.tensor([[5.,6.,7.]],dtype=torch.float64)
  scales=torch.tensor([.5,1.,4.],dtype=torch.float64)
  ll=log_likelihoods(x,scales,[1,2,None]);ev=torch.logsumexp(ll,2)-np.log(3)
  self.assertTrue(torch.isneginf(ll[:,:2,:]).all())
  np.testing.assert_array_equal(torch.softmax(ev,1).numpy(),[[0,0,1]])
 def test_extreme_log_likelihood_stays_finite(self):
  x=torch.full((1,512),100.,dtype=torch.float64);scales=torch.tensor([.5,1.,4.],dtype=torch.float64)
  ll=log_likelihoods(x,scales,[1,None]);ev=torch.logsumexp(ll,2)-np.log(3)
  expected=logsumexp([expon.logpdf(np.full(512,100.),scale=t).sum() for t in [.5,1.,4.]])-np.log(3)
  self.assertAlmostEqual(ev[0,1].item(),expected)
  self.assertTrue(torch.isfinite(ev[0,1]))
 def test_negative_observations_rejected(self):
  with self.assertRaises(ValueError): log_likelihoods(torch.tensor([[-1.]]),torch.tensor([1.]),[1,None])
