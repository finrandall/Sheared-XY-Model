import numpy as np

from numba import njit


@njit
def compute_potential_energy(X, t, ω):
    n = X.shape[0]
    U = 0.0

    for i in range(n - 1):
        U += -np.cos(X[i] - X[i + 1])

    U += -np.cos(X[n - 1] - X[0] - t * ω)

    return U / n


@njit
def compute_local_potential_row(X, t, ω, local_potential_row):
    n = X.shape[0]

    for i in range(n - 1):
        local_potential_row[i] = -np.cos(X[i] - X[i + 1])

    local_potential_row[n - 1] = -np.cos(X[n - 1] - X[0] - t * ω)


@njit
def compute_mean_torque(Forces):
    n = Forces.shape[0]
    total = 0.0

    for i in range(n):
        total += Forces[i]

    return total / n


@njit
def compute_rotor_autocorrelation(X, rotor_autocorrelation):
    n = X.shape[0]

    for r in range(n):
        total = 0.0

        for i in range(n):
            j = (i + r) % n
            total += np.cos(X[i] - X[j])

        rotor_autocorrelation[r] = total / n