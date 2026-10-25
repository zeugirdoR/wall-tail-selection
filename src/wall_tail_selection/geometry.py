"""Stable Hellinger distances for a moving endpoint with known shape.

p(x | tau, a) = a / tau * (1 - x/tau)**(a-1), 0 < x < tau.
H^2 = 1 - affinity = one half the squared L2 distance of square roots.
All calculations use a >= 1. This does not implement the two-parameter
packing problem or an inverse-ball-volume approximation to that problem.
"""

import math
import warnings

from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq


def endpoint_h2(a: float, log_ratio: float) -> float:
    """H^2(p_tau, p_tau') where log_ratio = log(tau'/tau).

    Scale invariance and symmetry allow tau=1 and tau'>=1. Integrating
    squared differences avoids cancellation from subtracting affinity
    from one. Geometric breakpoints resolve the boundary layer.
    """
    if not math.isfinite(a) or a < 1:
        raise ValueError("shape a must be finite and >= 1")
    if not math.isfinite(log_ratio):
        raise ValueError("log_ratio must be finite")
    d = abs(log_ratio)
    if d == 0:
        return 0.0
    if d > 100:
        return 1.0
    h = math.expm1(d)
    half_log_a = 0.5 * math.log(a)

    def integrand(t):
        lp = half_log_a + 0.5 * (a - 1) * math.log(t)
        lq = half_log_a + 0.5 * ((a - 1) * math.log(t + h) - a * d)
        upper, lower = max(lp, lq), min(lp, lq)
        difference = math.exp(upper) * (-math.expm1(lower - upper))
        return difference * difference

    points = [0.0]
    point = h
    while point < 1.0:
        points.append(point)
        point *= 8
    points.append(1.0)
    # Integration warnings are failures, not suppressed evidence.
    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        overlap = sum(quad(integrand, left, right, epsabs=1e-15,
                           epsrel=2e-10, limit=150)[0]
                      for left, right in zip(points[:-1], points[1:]))
    outside_mass = (-math.expm1(-d)) ** a
    return 0.5 * (overlap + outside_mass)


def resolution_log_step(a: float | None, n: int, c: float = 1.0) -> float:
    """Solve H=c/sqrt(n); a=None denotes the exponential scale family."""
    if n <= 1 or not 0 < c * c / n < 1:
        raise ValueError("require n>1 and 0<c^2/n<1")
    target = c * c / n
    if a is None:
        # Exponential affinity = sech(log_ratio/2).
        return 2 * math.acosh(1 / (1 - target))
    return brentq(lambda d: endpoint_h2(a, d) - target, 1e-14, 100,
                  xtol=2e-14, rtol=2e-12)


def separated_grid_count(a: float | None, n: int, lower: float,
                         upper: float, c: float = 1.0) -> int:
    """Number of equally spaced log-scale points at H separation c/sqrt(n).

    This constructive one-dimensional count is used as the pilot penalty.
    It is not a computation of a multi-parameter packing number. For the
    monotone scale distances here, log-scale spacing is sufficient to
    separate non-neighbors too; tests check this property numerically.
    """
    if not 0 < lower < upper:
        raise ValueError("require 0<lower<upper")
    step = resolution_log_step(a, n, c)
    return math.floor(math.log(upper / lower) / step) + 1
