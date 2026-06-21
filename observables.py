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
