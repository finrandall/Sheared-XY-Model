import numpy as np

from scipy.integrate import quad
from scipy.special import iv


def analytic_torque(γ̇, kT, μ, a=1.0, xmax=1000.0):

    def integrand(x):
        s = iv(0, x) + iv(1, x)
        exponent = -(kT / (2.0 * μ**2)) * x * (1.0 - np.exp(-x) * s)
        return np.cos(γ̇ * x / (2.0 * μ)) * np.exp(exponent)

    integral, _ = quad(integrand, 0.0, xmax, limit=1000)

    return μ * γ̇ + (a**2 * γ̇ / (4.0 * μ * kT)) * integral
