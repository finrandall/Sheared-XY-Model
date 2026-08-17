import csv
import itertools
import time
from pathlib import Path

import numpy as np

from mpi4py import MPI

from solver import evolve_timeseries
from parameters import get_parameters

from plot import plot_single_potential, plot_single_torque, plot_final_velocity, plot_average_velocity, plot_local_potential_history, plot_rotor_autocorrelation

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def parameter_combinations(params):
    names = tuple(params.keys())
    values = tuple(params[name] for name in names)

    return [dict(zip(names, combination)) for combination in itertools.product(*values)]


def run_simulation(params, settings):
    kT = params["kT"]
    γ̇ = params["γ̇"]
    μ = params["μ"]
    N = params["N"]
    EndTime = params["EndTime"]
    BurnInTime = params["BurnInTime"]
    Δt = params["Δt"]
    stride = params["stride"]
    potential_window = params["potential_window"]
    block_samples = settings["block_samples"]
    save_trajectory = settings["save_trajectory"]
    trajectory_stride = settings["trajectory_stride"]

    time_series, potential_energy, mean_torque, final_velocity, average_velocity, local_potential_history, rotor_autocorrelation, MeanPotentialEnergy, MeanTorque, block_start_time, block_end_time, block_sample_count, block_mean_velocity, block_mean_spatial_correlation, trajectory_time, rotor_angle_trajectory = evolve_timeseries(kT, γ̇, μ, N, EndTime, BurnInTime, Δt, stride, potential_window, block_samples, save_trajectory, trajectory_stride)

    return {
        "parameters": params,
        "time_series": time_series,
        "potential_energy": potential_energy,
        "mean_torque": mean_torque,
        "final_velocity": final_velocity,
        "average_velocity": average_velocity,
        "local_potential_history": local_potential_history,
        "rotor_autocorrelation": rotor_autocorrelation,
        "MeanPotentialEnergy": MeanPotentialEnergy,
        "MeanTorque": MeanTorque,
        "block_start_time": block_start_time,
        "block_end_time": block_end_time,
        "block_sample_count": block_sample_count,
        "block_mean_velocity": block_mean_velocity,
        "block_mean_spatial_correlation": block_mean_spatial_correlation,
        "trajectory_time": trajectory_time,
        "rotor_angle_trajectory": rotor_angle_trajectory,
    }


def print_averages(result, number, total):
    print(f"Simulation {number}/{total}")

    for name, value in result["parameters"].items():
        print(f"{name}: {value}")

    print(f"Average potential energy: {result['MeanPotentialEnergy']:.12g}")
    print(f"Average torque:           {result['MeanTorque']:.12g}")
    print()


def plot_result(result):
    plot_single_potential(result["time_series"], result["potential_energy"])
    plot_single_torque(result["time_series"], result["mean_torque"])
    plot_final_velocity(result["final_velocity"])
    plot_average_velocity(result["average_velocity"])
    plot_local_potential_history(result["local_potential_history"])
    plot_rotor_autocorrelation(result["rotor_autocorrelation"])


def save_results(results, settings):
    output_directory = settings["output_directory"]
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    index_rows = []

    for i, result in enumerate(results):
        filename = f"simulation_{i:04}.npz"
        parameter_values = result["parameters"]

        np.savez_compressed(
            output_path / filename,
            kT=parameter_values["kT"],
            shear_rate=parameter_values["γ̇"],
            damping=parameter_values["μ"],
            system_size=parameter_values["N"],
            end_time=parameter_values["EndTime"],
            burn_in_time=parameter_values["BurnInTime"],
            time_step=parameter_values["Δt"],
            stride=parameter_values["stride"],
            potential_window=parameter_values["potential_window"],
            time_series=result["time_series"],
            potential_energy=result["potential_energy"],
            mean_torque=result["mean_torque"],
            final_velocity=result["final_velocity"],
            average_velocity=result["average_velocity"],
            local_potential_history=result["local_potential_history"],
            rotor_autocorrelation=result["rotor_autocorrelation"],
            mean_potential_energy=result["MeanPotentialEnergy"],
            mean_torque_after_burn_in=result["MeanTorque"],
            block_start_time=result["block_start_time"],
            block_end_time=result["block_end_time"],
            block_sample_count=result["block_sample_count"],
            block_mean_velocity=result["block_mean_velocity"],
            block_mean_spatial_correlation=result["block_mean_spatial_correlation"],
            trajectory_time=result["trajectory_time"],
            rotor_angle_trajectory=result["rotor_angle_trajectory"],
            block_samples=settings["block_samples"],
            save_trajectory=settings["save_trajectory"],
            trajectory_stride=settings["trajectory_stride"],
        )

        index_rows.append({
            "file": filename,
            "temperature": parameter_values["kT"],
            "shear_rate": parameter_values["γ̇"],
            "damping": parameter_values["μ"],
            "system_size": parameter_values["N"],
            "end_time": parameter_values["EndTime"],
            "burn_in_time": parameter_values["BurnInTime"],
            "time_step": parameter_values["Δt"],
            "stride": parameter_values["stride"],
            "potential_window": parameter_values["potential_window"],
            "mean_potential_energy": result["MeanPotentialEnergy"],
            "mean_torque": result["MeanTorque"],
        })

    with open(output_path / "index.csv", "w", newline="") as index_file:
        writer = csv.DictWriter(index_file, fieldnames=index_rows[0].keys())
        writer.writeheader()
        writer.writerows(index_rows)

    return output_path


def main(params=None, settings=None):
    if rank == 0:
        if params is None:
            params, settings = get_parameters()
    else:
        params = None
        settings = None

    params = comm.bcast(params, root=0)
    settings = comm.bcast(settings, root=0)
    combinations = parameter_combinations(params)

    simulation_start = time.time()
    local_results = []

    for i in range(rank, len(combinations), size):
        local_results.append((i, run_simulation(combinations[i], settings)))

    gathered_results = comm.gather(local_results, root=0)
    simulation_elapsed = time.time() - simulation_start

    if rank == 0:
        indexed_results = []

        for process_results in gathered_results:
            indexed_results.extend(process_results)

        indexed_results.sort(key=lambda item: item[0])
        results = [result for _, result in indexed_results]

        minutes = int(simulation_elapsed // 60)
        seconds = simulation_elapsed % 60
        print(f"Simulation time: {minutes:02}:{seconds:05.2f}")
        print(f"Completed simulations: {len(results)}")
        print()

        for i, result in enumerate(results):
            print_averages(result, i + 1, len(results))

        output_path = save_results(results, settings)
        print(f"Saved results to: {output_path}")

        if len(results) == 1:
            plot_result(results[0])

        return results

    return None


if __name__ == "__main__":
    results = main()
