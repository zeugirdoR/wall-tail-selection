import math
import unittest

import numpy as np

from wall_tail_selection.geometry import endpoint_h2, resolution_log_step, separated_grid_count


class GeometryTests(unittest.TestCase):
    def test_uniform_against_closed_form(self):
        for d in [1e-7, 1e-4, .1, 1, 4]:
            self.assertAlmostEqual(endpoint_h2(1, d), -math.expm1(-d / 2), delta=1e-13)
        for n in [32, 1024, 32768]:
            self.assertAlmostEqual(resolution_log_step(1, n), -2 * math.log1p(-1 / n), delta=3e-13)

    def test_regular_fisher_constant(self):
        # I_tau = a / ((a-2) tau^2); H^2 ~ I_tau * delta_tau^2 / 8.
        for a in [3, 5]:
            d = 1e-5
            expected = a / (8 * (a - 2))
            self.assertAlmostEqual(endpoint_h2(a, d) / d**2, expected, delta=3e-4)

    def test_distance_monotonicity_symmetry_and_roots(self):
        for a in [1, 1.5, 1.95, 2, 2.05, 3, 5]:
            distances = [endpoint_h2(a, d) for d in np.geomspace(1e-5, 5, 24)]
            self.assertTrue(np.all(np.diff(distances) > 0))
            self.assertEqual(endpoint_h2(a, .3), endpoint_h2(a, -.3))
            for n in [32, 512, 32768]:
                self.assertAlmostEqual(n * endpoint_h2(a, resolution_log_step(a, n)), 1, delta=1e-8)

    def test_count_invariance_to_units(self):
        for a in [1, 2, 5, None]:
            self.assertEqual(separated_grid_count(a, 128, .5, 4), separated_grid_count(a, 128, 5, 40))

    def test_exponential_closed_form(self):
        for n in [32, 512, 32768]:
            d = resolution_log_step(None, n)
            self.assertAlmostEqual(n * (1 - 1 / math.cosh(d / 2)), 1, delta=1e-10)

    def test_invalid_inputs(self):
        for a in [0, .5, float("nan")]:
            with self.assertRaises(ValueError):
                endpoint_h2(a, .1)
        with self.assertRaises(ValueError):
            resolution_log_step(2, 1)


if __name__ == "__main__":
    unittest.main()
