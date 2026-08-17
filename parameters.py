import argparse

import numpy as np


def default_parameters():
    params = {}

    params["μ"] = np.array([1.0])
    params["N"] = np.array([2 ** 9])
    params["EndTime"] = np.array([5000.0])
    params["BurnInTime"] = np.array([4000.0])
    params["Δt"] = np.array([0.05])

    params["kT"] = np.array([0.001])
    params["γ̇"] = np.array([0.00585938])
    params["stride"] = np.array([1])
    params["potential_window"] = np.array([100.0])

    settings = {}

    settings["block_samples"] = 100
    settings["save_trajectory"] = False
    settings["trajectory_stride"] = 100
    settings["output_directory"] = "results"

    return params, settings


def get_parameters(args=None):
    params, settings = default_parameters()

    parser = argparse.ArgumentParser(description="Simulate the sheared XY model.")
    parser.add_argument("--damping", type=float, nargs="+", default=params["μ"])
    parser.add_argument("--system-size", type=int, nargs="+", default=params["N"])
    parser.add_argument("--end-time", type=float, nargs="+", default=params["EndTime"])
    parser.add_argument("--burn-in-time", type=float, nargs="+", default=params["BurnInTime"])
    parser.add_argument("--time-step", type=float, nargs="+", default=params["Δt"])

    parser.add_argument("--temperature", type=float, nargs="+", default=params["kT"])
    parser.add_argument("--shear-rate", type=float, nargs="+", default=params["γ̇"])
    parser.add_argument("--stride", type=int, nargs="+", default=params["stride"])
    parser.add_argument("--potential-window", type=float, nargs="+", default=params["potential_window"])
    parser.add_argument("--block-samples", type=int, default=settings["block_samples"])
    parser.add_argument("--save-trajectory", action=argparse.BooleanOptionalAction, default=settings["save_trajectory"])
    parser.add_argument("--trajectory-stride", type=int, default=settings["trajectory_stride"])
    parser.add_argument("--output-directory", default=settings["output_directory"])

    options = parser.parse_args(args)

    params["μ"] = np.array(options.damping)
    params["N"] = np.array(options.system_size)
    params["EndTime"] = np.array(options.end_time)
    params["BurnInTime"] = np.array(options.burn_in_time)
    params["Δt"] = np.array(options.time_step)

    params["kT"] = np.array(options.temperature)
    params["γ̇"] = np.array(options.shear_rate)
    params["stride"] = np.array(options.stride)
    params["potential_window"] = np.array(options.potential_window)

    settings["block_samples"] = options.block_samples
    settings["save_trajectory"] = options.save_trajectory
    settings["trajectory_stride"] = options.trajectory_stride
    settings["output_directory"] = options.output_directory

    if settings["block_samples"] < 1:
        parser.error("--block-samples must be at least 1")

    if settings["trajectory_stride"] < 1:
        parser.error("--trajectory-stride must be at least 1")

    return params, settings


if __name__ == "__main__":
    from main import main

    results = main(*get_parameters())
