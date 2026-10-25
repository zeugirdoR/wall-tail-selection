import unittest

import numpy as np
from scipy.stats import expon

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "install the selection extra to test selection")
class SelectionTests(unittest.TestCase):
    def test_likelihoods_and_support(self):
        from wall_tail_selection.selection import log_likelihoods
        x = torch.tensor([[.1, .3, .7]], dtype=torch.float64)
        scales = torch.tensor([.5, 1., 2.], dtype=torch.float64)
        ll = log_likelihoods(x, scales, [1, 2, None]).numpy()[0]
        self.assertTrue(np.isneginf(ll[:2, 0]).all())
        np.testing.assert_allclose(ll[0, 1:], -3 * np.log([1., 2.]))
        for i, scale in enumerate([.5, 1., 2.]):
            self.assertAlmostEqual(ll[2, i], expon.logpdf(x.numpy()[0], scale=scale).sum())
        expected = np.log(2 * (1 - x.numpy()[0])).sum()
        self.assertAlmostEqual(ll[1, 1], expected)

    def test_cdf_boundaries_and_known_values(self):
        from wall_tail_selection.selection import cdf
        np.testing.assert_allclose(cdf(np.array([-1., 0., .5, 1., 2.]), 1., 1), [0, 0, .5, 1, 1])
        self.assertAlmostEqual(cdf(.5, 1., 2), .75)
        np.testing.assert_allclose(cdf(np.array([0., 1., 2.]), 2., None), expon.cdf([0, 1, 2], scale=2))


if __name__ == "__main__":
    unittest.main()
