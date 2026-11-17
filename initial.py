import numpy as np

from numba import njit


@njit(cache=True)
def rand_uniform(out, r):
    n = out.shape[0]

    for i in range(n):
        out[i] = np.random.uniform(-r, r)


@njit(cache=True)
def initialise_uniform_velocity(V, γ̇):
    n = V.shape[0]
    ω = n * γ̇

    for i in range(n):
        V[i] = -ω / 2.0 + i * γ̇

    return ω
