import numpy as np

from scipy.integrate import quad
from scipy.special import ive


def analytic_torque(γ̇, kT, μ, a=1.0, xmax=1000.0):
    """Mean torque to O(a^2) (Evans, Eq. 84), large N and late time.

    Uses exponentially scaled Bessel functions, e^{-x}(I0(x) + I1(x)) = ive(0, x) + ive(1, x),
    so the integrand stays finite for large x.
    """

    def integrand(x):
        s = ive(0, x) + ive(1, x)
        exponent = -(kT / (2.0 * μ**2)) * x * (1.0 - s)
        return np.cos(γ̇ * x / (2.0 * μ)) * np.exp(exponent)

    integral, _ = quad(integrand, 0.0, xmax, limit=1000)

    return μ * γ̇ + (a**2 * γ̇ / (4.0 * μ * kT)) * integral
