import numpy as np

from numba import njit

from forces import compute_forces, compute_spin_forces
from initial import rand_uniform
from observables import compute_potential_energy


@njit
def evolve(kT, γ̇, μ, N, EndTime, TimeAverage, Δt):
    TimeThreshold = EndTime - TimeAverage
    Iterations = int(EndTime / Δt)

    ω = N * γ̇
    ν = np.sqrt(2.0 * kT * μ)
    noise_prefactor = ν / np.sqrt(Δt)

    TwoPi = 2.0 * np.pi
    UnitVar = np.sqrt(3.0)
    squareΔt = Δt * Δt

    X = np.empty(N)
    V = np.empty(N)
    R = np.empty(N)

    rand_uniform(X, np.pi)

    for i in range(N):
        V[i] = -ω / 2.0 + i * γ̇

    rand_uniform(R, UnitVar)

    Forces = np.zeros(N)
    SpinForces = np.zeros(N)
    Forces_v = np.zeros(N)
    NewForces = np.zeros(N)
    NewSpinForces = np.zeros(N)

    MeanPotentialEnergy = 0.0
    MeanTorque = 0.0
    Count = 0
    t = 0.0

    for i in range(Iterations):
        t += Δt

        compute_forces(X, V, R, t, ω, μ, noise_prefactor, Forces)
        compute_spin_forces(Forces, SpinForces)

        rand_uniform(R, UnitVar)

        for s in range(N):
            X[s] = X[s] + Δt * V[s] + 0.5 * squareΔt * SpinForces[s]

            if X[s] >= TwoPi:
                X[s] -= TwoPi
            elif X[s] < 0.0:
                X[s] += TwoPi

        for s in range(N):
            V[s] = V[s] + 0.5 * Δt * SpinForces[s]

        compute_forces(X, V, R, t, ω, μ, noise_prefactor, Forces_v)

        for s in range(N):
            NewForces[s] = 0.5 * (Forces[s] + Forces_v[s])

        compute_spin_forces(NewForces, NewSpinForces)

        for s in range(N):
            V[s] = V[s] + Δt * NewSpinForces[s]

        if t >= TimeThreshold:
            tor_sum = 0.0

            for s in range(N):
                tor_sum += Forces_v[s]

            MeanTorque += tor_sum / N
            MeanPotentialEnergy += compute_potential_energy(X, t, ω)
            Count += 1

    return MeanTorque / Count, MeanPotentialEnergy / Count
