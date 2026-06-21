import numpy as np

from numba import njit


@njit
def compute_forces(X, V, R, t, ω, μ, noise_prefactor, Forces):
    n = X.shape[0]

    for i in range(n - 1):
        dX = X[i] - X[i + 1]
        dV = V[i] - V[i + 1]
        Forces[i] = -np.sin(dX) - μ * dV + noise_prefactor * R[i]

    dX = X[n - 1] - X[0] - t * ω
    dV = V[n - 1] - V[0] - ω
    Forces[n - 1] = -np.sin(dX) - μ * dV + noise_prefactor * R[n - 1]


@njit
def compute_spin_forces(Forces, SpinForces):
    n = Forces.shape[0]

    SpinForces[0] = Forces[0] - Forces[n - 1]

    for i in range(1, n):
        SpinForces[i] = Forces[i] - Forces[i - 1]
