"""Time integration of the sheared 1D XY model.

Integrator: Groot-Warren modified velocity-Verlet (lambda = 1/2), as described in the
dissertation (Sec. 4.2):

    1. X(t+dt)  = X(t) + dt V(t) + dt^2/2 F(t)
    2. V~       = V(t) + dt/2 F(t)
    3. F(t+dt)  = F(X(t+dt), V~(t+dt))
    4. V(t+dt)  = V(t) + dt/2 [F(t) + F(t+dt)]

F(t+dt) from step 3 is carried into the next step, so there is one force evaluation per step.
Here F is the net torque on each rotor (difference of neighbouring bond torques), I = 1.

Mean torque samples use  mu * shear_rate + a <sin phi>. This equals the bond-averaged torque
with the zero-mean noise term removed (the bond-averaged friction torque is exactly
mu * shear_rate because the mean shear rate is conserved), so it has the same expectation as the
full torque with far smaller sampling noise.
"""
import numpy as np

from numba import njit

from forces import compute_forces, compute_spin_forces

from initial import rand_uniform, initialise_uniform_velocity

from observables import compute_potential_energy, compute_local_potential_row, compute_rotor_autocorrelation


TwoPi = 2.0 * np.pi
UnitVar = np.sqrt(3.0)


@njit(cache=True)
def setup(kT, γ̇, μ, a, N, Δt):
    noise_prefactor = np.sqrt(2.0 * kT * μ) / np.sqrt(Δt)

    X = np.empty(N)
    V = np.empty(N)
    R = np.empty(N)

    rand_uniform(X, np.pi)
    for i in range(N):
        X[i] += np.pi
    ω = initialise_uniform_velocity(V, γ̇)
    rand_uniform(R, UnitVar)

    Forces = np.empty(N)
    SpinForces = np.empty(N)
    conservative = compute_forces(X, V, R, 0.0, ω, μ, a, noise_prefactor, Forces)
    compute_spin_forces(Forces, SpinForces)

    return X, V, R, Forces, SpinForces, ω, noise_prefactor, conservative


@njit(cache=True)
def step(X, V, R, Forces, SpinForces, PredictedV, NewForces, NewSpinForces, t, Δt, ω, μ, a, noise_prefactor):
    """Advance one step from t to t + Δt. Forces/SpinForces hold F(t) on entry and F(t+Δt) on exit."""
    n = X.shape[0]
    halfΔt = 0.5 * Δt
    halfΔt2 = 0.5 * Δt * Δt

    for s in range(n):
        x = X[s] + Δt * V[s] + halfΔt2 * SpinForces[s]

        if x >= TwoPi:
            x -= TwoPi * np.floor(x / TwoPi)
        elif x < 0.0:
            x -= TwoPi * np.floor(x / TwoPi)

        X[s] = x
        PredictedV[s] = V[s] + halfΔt * SpinForces[s]

    rand_uniform(R, UnitVar)

    conservative = compute_forces(X, PredictedV, R, t + Δt, ω, μ, a, noise_prefactor, NewForces)
    compute_spin_forces(NewForces, NewSpinForces)

    for s in range(n):
        V[s] += halfΔt * (SpinForces[s] + NewSpinForces[s])
        Forces[s] = NewForces[s]
        SpinForces[s] = NewSpinForces[s]

    return conservative


@njit(cache=True)
def evolve(kT, γ̇, μ, N, EndTime, BurnInTime, Δt, a=1.0):
    """Lightweight run: returns (mean torque, mean potential energy) after burn-in."""
    Iterations = int(round(EndTime / Δt))
    BurnInTime = min(max(BurnInTime, 0.0), EndTime)

    X, V, R, Forces, SpinForces, ω, noise_prefactor, conservative = setup(kT, γ̇, μ, a, N, Δt)
    PredictedV = np.empty(N)
    NewForces = np.empty(N)
    NewSpinForces = np.empty(N)

    MeanPotentialEnergy = 0.0
    MeanTorque = 0.0
    Count = 0
    t = 0.0

    for i in range(Iterations):
        conservative = step(X, V, R, Forces, SpinForces, PredictedV, NewForces, NewSpinForces, t, Δt, ω, μ, a, noise_prefactor)
        t = (i + 1) * Δt

        if t >= BurnInTime:
            MeanTorque += μ * γ̇ + conservative
            MeanPotentialEnergy += compute_potential_energy(X, t, ω, a)
            Count += 1

    if Count > 0:
        MeanTorque /= Count
        MeanPotentialEnergy /= Count

    return MeanTorque, MeanPotentialEnergy


@njit(cache=True)
def evolve_timeseries(kT, γ̇, μ, N, EndTime, BurnInTime, Δt, sample_interval, potential_window, block_samples, save_spatial_correlation, save_trajectory, trajectory_stride, a=1.0):
    Iterations = int(round(EndTime / Δt))
    max_samples = Iterations // sample_interval + 1
    max_blocks = (max_samples + block_samples - 1) // block_samples

    if save_trajectory:
        max_trajectory_samples = Iterations // trajectory_stride + 1
    else:
        max_trajectory_samples = 1

    BurnInTime = min(max(BurnInTime, 0.0), EndTime)
    potential_start_time = max(EndTime - potential_window, 0.0)
    potential_rows = int(np.ceil((EndTime - potential_start_time) / Δt)) + 1

    X, V, R, Forces, SpinForces, ω, noise_prefactor, conservative = setup(kT, γ̇, μ, a, N, Δt)
    PredictedV = np.empty(N)
    NewForces = np.empty(N)
    NewSpinForces = np.empty(N)

    time = np.empty(max_samples)
    potential_energy = np.empty(max_samples)
    mean_torque = np.empty(max_samples)
    local_potential_history = np.empty((potential_rows, N))

    average_velocity = np.zeros(N)
    rotor_autocorrelation = np.zeros(N)
    rotor_autocorrelation_current = np.zeros(N)

    block_start_time = np.empty(max_blocks)
    block_end_time = np.empty(max_blocks)
    block_sample_count = np.zeros(max_blocks, dtype=np.int64)
    block_mean_velocity = np.empty((max_blocks, N))
    block_mean_spatial_correlation = np.empty((max_blocks, N))
    block_velocity_sum = np.zeros(N)
    block_spatial_correlation_sum = np.zeros(N)

    trajectory_time = np.empty(max_trajectory_samples)
    rotor_angle_trajectory = np.empty((max_trajectory_samples, N), dtype=np.float32)

    sample = 0
    potential_sample = 0
    measurement_count = 0
    block = 0
    block_count = 0
    trajectory_sample = 0

    MeanPotentialEnergy = 0.0
    MeanTorque = 0.0

    t = 0.0

    for i in range(Iterations + 1):
        if i > 0:
            conservative = step(X, V, R, Forces, SpinForces, PredictedV, NewForces, NewSpinForces, t, Δt, ω, μ, a, noise_prefactor)
            t = i * Δt

        if i % sample_interval == 0:
            time[sample] = t
            potential_energy[sample] = compute_potential_energy(X, t, ω, a)
            mean_torque[sample] = μ * γ̇ + conservative

            if save_spatial_correlation:
                compute_rotor_autocorrelation(X, t, ω, rotor_autocorrelation_current)

            if block_count == 0:
                block_start_time[block] = t

            block_end_time[block] = t

            for s in range(N):
                block_velocity_sum[s] += V[s]

            if save_spatial_correlation:
                for s in range(N):
                    block_spatial_correlation_sum[s] += rotor_autocorrelation_current[s]

            block_count += 1

            if t >= BurnInTime:
                MeanPotentialEnergy += potential_energy[sample]
                MeanTorque += mean_torque[sample]

                for s in range(N):
                    average_velocity[s] += V[s]

                if save_spatial_correlation:
                    for s in range(N):
                        rotor_autocorrelation[s] += rotor_autocorrelation_current[s]

                measurement_count += 1

            sample += 1

            if block_count == block_samples:
                block_sample_count[block] = block_count

                for s in range(N):
                    block_mean_velocity[block, s] = block_velocity_sum[s] / block_count
                    block_velocity_sum[s] = 0.0

                if save_spatial_correlation:
                    for s in range(N):
                        block_mean_spatial_correlation[block, s] = block_spatial_correlation_sum[s] / block_count
                        block_spatial_correlation_sum[s] = 0.0

                block += 1
                block_count = 0

        if save_trajectory and i % trajectory_stride == 0:
            trajectory_time[trajectory_sample] = t

            for s in range(N):
                rotor_angle_trajectory[trajectory_sample, s] = X[s]

            trajectory_sample += 1

        if t >= potential_start_time and potential_sample < potential_rows:
            compute_local_potential_row(X, t, ω, a, local_potential_history[potential_sample])
            potential_sample += 1

    if measurement_count > 0:
        MeanPotentialEnergy /= measurement_count
        MeanTorque /= measurement_count

        for s in range(N):
            average_velocity[s] /= measurement_count

        if save_spatial_correlation:
            for s in range(N):
                rotor_autocorrelation[s] /= measurement_count

    if block_count > 0:
        block_sample_count[block] = block_count

        for s in range(N):
            block_mean_velocity[block, s] = block_velocity_sum[s] / block_count

        if save_spatial_correlation:
            for s in range(N):
                block_mean_spatial_correlation[block, s] = block_spatial_correlation_sum[s] / block_count

        block += 1

    correlation_samples = N if save_spatial_correlation else 0
    correlation_blocks = block if save_spatial_correlation else 0

    return time[:sample], potential_energy[:sample], mean_torque[:sample], V, average_velocity, local_potential_history[:potential_sample], rotor_autocorrelation[:correlation_samples], MeanPotentialEnergy, MeanTorque, block_start_time[:block], block_end_time[:block], block_sample_count[:block], block_mean_velocity[:block], block_mean_spatial_correlation[:correlation_blocks], trajectory_time[:trajectory_sample], rotor_angle_trajectory[:trajectory_sample]
