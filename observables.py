import numpy as np

from numba import njit


@njit(cache=True)
def compute_potential_energy(X, t, ω, a):
    n = X.shape[0]
    U = 0.0

    for i in range(n - 1):
        U -= np.cos(X[i + 1] - X[i])

    U -= np.cos(X[0] + t * ω - X[n - 1])

    return a * U / n


@njit(cache=True)
def compute_local_potential_row(X, t, ω, a, local_potential_row):
    n = X.shape[0]

    for i in range(n - 1):
        local_potential_row[i] = -a * np.cos(X[i + 1] - X[i])

    local_potential_row[n - 1] = -a * np.cos(X[0] + t * ω - X[n - 1])


@njit(cache=True)
def compute_rotor_autocorrelation(X, t, ω, rotor_autocorrelation):
    """C(r) = (1/N) sum_i cos(theta_{i+r} - theta_i), r = 0..N-1.

    Rotors beyond the boundary are periodic images shifted by ω t (sheared boundary),
    i.e. theta_{j+N} = theta_j + ω t. Uses precomputed cos/sin so the O(N^2) loop
    contains only multiply-adds.
    """
    n = X.shape[0]
    c = np.empty(n)
    s = np.empty(n)
    ci = np.empty(n)
    si = np.empty(n)
    shift = t * ω

    for i in range(n):
        c[i] = np.cos(X[i])
        s[i] = np.sin(X[i])
        ci[i] = np.cos(X[i] + shift)
        si[i] = np.sin(X[i] + shift)

    for r in range(n):
        total = 0.0

        for i in range(n - r):
            j = i + r
            total += c[i] * c[j] + s[i] * s[j]

        for i in range(n - r, n):
            j = i + r - n
            total += c[i] * ci[j] + s[i] * si[j]

        rotor_autocorrelation[r] = total / n
