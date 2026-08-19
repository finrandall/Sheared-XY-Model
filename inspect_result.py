import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Inspect the time series from one saved XY-model simulation.")
    parser.add_argument("result_file")
    parser.add_argument("--burn-in-time", type=float, required=True)
    args = parser.parse_args()

    with np.load(args.result_file) as data:
        time = data["time_series"]
        mean_torque = data["mean_torque"]
        potential_energy = data["potential_energy"]
        temperature = float(data["kT"])
        shear_rate = float(data["shear_rate"])
        damping = float(data["damping"])

    fig, axes = plt.subplots(2, 1, sharex=True)

    axes[0].plot(time, mean_torque, linewidth=1)
    axes[0].axvline(args.burn_in_time, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel(r"$\bar{\tau}$")

    axes[1].plot(time, potential_energy, linewidth=1)
    axes[1].axvline(args.burn_in_time, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel(r"$U$")

    fig.suptitle(fr"$kT={temperature:g},\;\dot{{\gamma}}={shear_rate:g},\;\mu={damping:g}$")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
