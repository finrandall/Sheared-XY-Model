import numpy as np

from numba import njit

from forces import compute_forces, compute_spin_forces

from initial import rand_uniform, initialise_uniform_velocity

from observables import compute_potential_energy, compute_local_potential_row, compute_mean_torque, compute_rotor_autocorrelation


@njit
def evolve(kT, γ̇, μ, N, EndTime, BurnInTime, Δt):
    Iterations = int(EndTime / Δt)

    if BurnInTime < 0.0:
        BurnInTime = 0.0

    if BurnInTime > EndTime:
        BurnInTime = EndTime

    ν = np.sqrt(2.0 * kT * μ)
    noise_prefactor = ν / np.sqrt(Δt)

    TwoPi = 2.0 * np.pi
    UnitVar = np.sqrt(3.0)
    squareΔt = Δt * Δt

    X = np.empty(N)
    V = np.empty(N)
    R = np.empty(N)

    rand_uniform(X, np.pi)
    ω = initialise_uniform_velocity(V, γ̇)
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

        if t >= BurnInTime:
            MeanTorque += compute_mean_torque(NewForces)
            MeanPotentialEnergy += compute_potential_energy(X, t, ω)
            Count += 1

    if Count > 0:
        MeanTorque = MeanTorque / Count
        MeanPotentialEnergy = MeanPotentialEnergy / Count

    return MeanTorque, MeanPotentialEnergy


@njit
def evolve_timeseries(kT, γ̇, μ, N, EndTime, BurnInTime, Δt, sample_interval, potential_window):
    Iterations = int(EndTime / Δt)
    max_samples = Iterations // sample_interval + 1

    if BurnInTime < 0.0:
        BurnInTime = 0.0

    if BurnInTime > EndTime:
        BurnInTime = EndTime

    potential_start_time = EndTime - potential_window

    if potential_start_time < 0.0:
        potential_start_time = 0.0

    potential_rows = int(np.ceil((EndTime - potential_start_time) / Δt)) + 1

    ν = np.sqrt(2.0 * kT * μ)
    noise_prefactor = ν / np.sqrt(Δt)

    TwoPi = 2.0 * np.pi
    UnitVar = np.sqrt(3.0)
    squareΔt = Δt * Δt

    X = np.empty(N)
    V = np.empty(N)
    R = np.empty(N)

    rand_uniform(X, np.pi)
    ω = initialise_uniform_velocity(V, γ̇)
    rand_uniform(R, UnitVar)

    Forces = np.zeros(N)
    SpinForces = np.zeros(N)
    Forces_v = np.zeros(N)
    NewForces = np.zeros(N)
    NewSpinForces = np.zeros(N)

    time = np.empty(max_samples)
    potential_energy = np.empty(max_samples)
    mean_torque = np.empty(max_samples)
    local_potential_history = np.empty((potential_rows, N))

    average_velocity = np.zeros(N)

    rotor_autocorrelation = np.zeros(N)
    rotor_autocorrelation_current = np.empty(N)

    sample = 0
    potential_sample = 0
    measurement_count = 0

    MeanPotentialEnergy = 0.0
    MeanTorque = 0.0

    t = 0.0

    compute_forces(X, V, R, t, ω, μ, noise_prefactor, Forces)

    time[sample] = t
    potential_energy[sample] = compute_potential_energy(X, t, ω)
    mean_torque[sample] = compute_mean_torque(Forces)

    if t >= BurnInTime:
        MeanPotentialEnergy += potential_energy[sample]
        MeanTorque += mean_torque[sample]

        for s in range(N):
            average_velocity[s] += V[s]

        compute_rotor_autocorrelation(X, rotor_autocorrelation_current)

        for s in range(N):
            rotor_autocorrelation[s] += rotor_autocorrelation_current[s]

        measurement_count += 1

    sample += 1

    if t >= potential_start_time and potential_sample < potential_rows:
        compute_local_potential_row(X, t, ω, local_potential_history[potential_sample])
        potential_sample += 1

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

        if (i + 1) % sample_interval == 0:
            time[sample] = t
            potential_energy[sample] = compute_potential_energy(X, t, ω)
            mean_torque[sample] = compute_mean_torque(NewForces)

            if t >= BurnInTime:
                MeanPotentialEnergy += potential_energy[sample]
                MeanTorque += mean_torque[sample]

                for s in range(N):
                    average_velocity[s] += V[s]

                compute_rotor_autocorrelation(X, rotor_autocorrelation_current)

                for s in range(N):
                    rotor_autocorrelation[s] += rotor_autocorrelation_current[s]

                measurement_count += 1

            sample += 1

        if t >= potential_start_time and potential_sample < potential_rows:
            compute_local_potential_row(X, t, ω, local_potential_history[potential_sample])
            potential_sample += 1

    if measurement_count > 0:
        MeanPotentialEnergy = MeanPotentialEnergy / measurement_count
        MeanTorque = MeanTorque / measurement_count

        for s in range(N):
            average_velocity[s] = average_velocity[s] / measurement_count
            rotor_autocorrelation[s] = rotor_autocorrelation[s] / measurement_count

    return time[:sample], potential_energy[:sample], mean_torque[:sample], V, average_velocity, local_potential_history[:potential_sample], rotor_autocorrelation, MeanPotentialEnergy, MeanTorque