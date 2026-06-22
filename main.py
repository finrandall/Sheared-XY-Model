import time

import numpy as np

from mpi4py import MPI

from solver import evolve, evolve_timeseries
from interface import get_parameters
from parameters import default_parameters

from plot import plot_single_potential, plot_single_torque, plot_final_velocity, plot_average_velocity, plot_local_potential_history, plot_rotor_autocorrelation, plot_torque, plot_potential

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def run_single(params):
    kT = params["kT"]
    γ̇ = params["γ̇"]
    μ = params["μ"]
    N = params["N"]
    EndTime = params["EndTime"]
    BurnInTime = params.get("BurnInTime", 0.0)
    Δt = params["Δt"]
    sample_interval = params.get("sample_interval", 1)
    potential_window = params.get("potential_window", EndTime)

    return evolve_timeseries(kT, γ̇, μ, N, EndTime, BurnInTime, Δt, sample_interval, potential_window)


def print_single_averages(MeanPotentialEnergy, MeanTorque, measurement_count):
    print("Measured observables after burn-in")
    print("----------------------------------")
    print(f"Average potential energy: {MeanPotentialEnergy:.12g}")
    print(f"Average torque:           {MeanTorque:.12g}")
    print(f"Measurement samples:      {measurement_count}")


def run_sweep(params):
    μ = params["μ"]
    N = params["N"]
    EndTime = params["EndTime"]
    BurnInTime = params.get("BurnInTime", 0.0)
    Δt = params["Δt"]
    γ̇List = params["γ̇List"]
    kTList = params["kTList"]

    MeanTorqueArray = np.zeros((kTList.shape[0], γ̇List.shape[0]))
    MeanPotentialArray = np.zeros((kTList.shape[0], γ̇List.shape[0]))

    for j in range(kTList.shape[0]):
        local_torque = np.zeros(γ̇List.shape[0])
        local_potential = np.zeros(γ̇List.shape[0])

        for i in range(rank, γ̇List.shape[0], size):
            local_torque[i], local_potential[i] = evolve(kTList[j], γ̇List[i], μ, N, EndTime, BurnInTime, Δt)

        comm.Reduce(local_torque, MeanTorqueArray[j] if rank == 0 else None, op=MPI.SUM, root=0)
        comm.Reduce(local_potential, MeanPotentialArray[j] if rank == 0 else None, op=MPI.SUM, root=0)

    return MeanTorqueArray, MeanPotentialArray


def main(use_interface=True):
    if rank == 0:
        if use_interface:
            params = get_parameters()
        else:
            params = default_parameters()
    else:
        params = None

    params = comm.bcast(params, root=0)

    if params is None:
        return None

    if params.get("mode", "sweep") == "single":
        if rank == 0:
            simulation_start = time.time()
            time_series, potential_energy, mean_torque, final_velocity, average_velocity, local_potential_history, rotor_autocorrelation, MeanPotentialEnergy, MeanTorque, measurement_count = run_single(params)
            simulation_elapsed = time.time() - simulation_start

            minutes = int(simulation_elapsed // 60)
            seconds = simulation_elapsed % 60
            print(f"Simulation time: {minutes:02}:{seconds:05.2f}")
            print_single_averages(MeanPotentialEnergy, MeanTorque, measurement_count)

            plot_single_potential(time_series, potential_energy)
            plot_single_torque(time_series, mean_torque)
            plot_final_velocity(final_velocity)
            plot_average_velocity(average_velocity)
            plot_local_potential_history(local_potential_history)
            plot_rotor_autocorrelation(rotor_autocorrelation)

            return time_series, potential_energy, mean_torque, final_velocity, average_velocity, local_potential_history, rotor_autocorrelation

        return None

    simulation_start = time.time()
    MeanTorqueArray, MeanPotentialArray = run_sweep(params)
    simulation_elapsed = time.time() - simulation_start

    if rank == 0:
        minutes = int(simulation_elapsed // 60)
        seconds = simulation_elapsed % 60
        print(f"Simulation time: {minutes:02}:{seconds:05.2f}")

        γ̇List = params["γ̇List"]
        kTList = params["kTList"]

        plot_torque(γ̇List, kTList, MeanTorqueArray, params["μ"], params["show_analytic"])

        plot_potential(γ̇List, kTList, MeanPotentialArray)

        return γ̇List, kTList, MeanTorqueArray, MeanPotentialArray

    return None


if __name__ == "__main__":
    results = main(use_interface=True)