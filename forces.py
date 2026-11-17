import numpy as np

from numba import njit


@njit(cache=True)
def compute_forces(X, V, R, t, ω, μ, a, noise_prefactor, Forces):
    """Bond torques tau_i = a sin(phi_i) + mu dphi_i/dt + eta_i, with phi_i = X[i+1] - X[i].

    The last bond joins rotor N-1 to the image of rotor 0, whose angle is X[0] + ω t
    and angular velocity V[0] + ω (sheared periodic boundary).
    Returns the bond-averaged conservative torque a <sin phi>.
    """
    n = X.shape[0]
    conservative = 0.0

    for i in range(n - 1):
        c = a * np.sin(X[i + 1] - X[i])
        conservative += c
        Forces[i] = c + μ * (V[i + 1] - V[i]) + noise_prefactor * R[i]

    c = a * np.sin(X[0] + t * ω - X[n - 1])
    conservative += c
    Forces[n - 1] = c + μ * (V[0] + ω - V[n - 1]) + noise_prefactor * R[n - 1]

    return conservative / n


@njit(cache=True)
def compute_spin_forces(Forces, SpinForces):
    """Net torque on rotor i: tau_i - tau_{i-1}."""
    n = Forces.shape[0]

    SpinForces[0] = Forces[0] - Forces[n - 1]

    for i in range(1, n):
        SpinForces[i] = Forces[i] - Forces[i - 1]
