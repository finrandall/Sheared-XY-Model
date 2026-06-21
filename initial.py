import numpy as np

from numba import njit


@njit
def rand_uniform(out, r):
    n = out.shape[0]

    for i in range(n):
        out[i] = np.random.uniform(-r, r)
