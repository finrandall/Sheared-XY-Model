import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from analytics import analytic_torque


parameter_names = (
    "temperature",
    "shear_rate",
    "damping",
    "coupling",
    "system_size",
    "end_time",
    "time_step",
    "stride",
    "potential_window",
)


parameter_labels = {
    "temperature": r"$kT$",
    "shear_rate": r"$\dot{\gamma}$",
    "damping": r"$\mu$",
    "coupling": r"$a$",
    "system_size": r"$N$",
    "end_time": r"$t_{\mathrm{end}}$",
    "time_step": r"$\Delta t$",
    "stride": "Stride",
    "potential_window": "Potential window",
}


def default_plot_settings():
    settings = {}

    settings["results_directory"] = "results"
    settings["burn_in_time"] = 4000.0
    settings["x_parameter"] = "shear_rate"
    settings["log_x"] = False
    settings["save_prefix"] = None
    settings["show_analytic"] = True
    settings["analytic_points"] = 200
    settings["analytic_xmax"] = 200.0

    return settings


def load_averages(results_directory, burn_in_time):
    results_path = Path(results_directory)
    rows = []

    with open(results_path / "index.csv", newline="") as index_file:
        for index_row in csv.DictReader(index_file):
            result_file = results_path / index_row["file"]
            data = np.load(result_file)
            time = data["time_series"]
            measured = time >= burn_in_time

            if not np.any(measured):
                raise ValueError(f"Burn-in time {burn_in_time} leaves no samples in {result_file}.")

            row = {"file": index_row["file"]}

            for name in parameter_names:
                row[name] = float(index_row.get(name, 1.0 if name == "coupling" else "nan"))

            row["burn_in_time"] = burn_in_time
            row["samples_after_burn_in"] = int(np.count_nonzero(measured))
            row["mean_torque"] = float(np.mean(data["mean_torque"][measured]))
            row["mean_potential_energy"] = float(np.mean(data["potential_energy"][measured]))
            rows.append(row)

    return rows


def varying_parameters(rows):
    return [name for name in parameter_names if len({row[name] for row in rows}) > 1]


def choose_x_parameter(rows, requested_parameter=None):
    if requested_parameter is not None:
        return requested_parameter

    varying = varying_parameters(rows)

    if "shear_rate" in varying:
        return "shear_rate"

    if varying:
        return varying[0]

    return "shear_rate"


def group_rows(rows, x_parameter):
    group_parameters = [name for name in varying_parameters(rows) if name != x_parameter]
    groups = {}

    for row in rows:
        key = tuple(row[name] for name in group_parameters)
        groups.setdefault(key, []).append(row)

    return group_parameters, groups


def group_label(group_parameters, key):
    if not group_parameters:
        return None

    return ", ".join(f"{name}={value:g}" for name, value in zip(group_parameters, key))


def write_processed_index(rows, output_file):
    with open(output_file, "w", newline="") as processed_file:
        writer = csv.DictWriter(processed_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def open_time_series(row, results_directory):
    result_file = (Path(results_directory) / row["file"]).resolve()
    inspector_script = Path(__file__).with_name("inspect_result.py")

    subprocess.Popen([
        sys.executable,
        str(inspector_script),
        str(result_file),
        "--burn-in-time",
        str(row["burn_in_time"]),
    ])


def plot_averages(rows, x_parameter, results_directory, log_x=False, save_prefix=None, show_analytic=True, analytic_points=200, analytic_xmax=200.0):
    import matplotlib.pyplot as plt

    group_parameters, groups = group_rows(rows, x_parameter)
    observables = (
        ("mean_torque", r"$\langle \bar{\tau} \rangle$", "torque"),
    )

    for observable, y_label, filename_suffix in observables:
        fig, ax = plt.subplots()

        for key, group in groups.items():
            group.sort(key=lambda row: row[x_parameter])
            x_values = np.array([row[x_parameter] for row in group])
            y_values = np.array([row[observable] for row in group])
            line, = ax.plot(x_values, y_values, linestyle="None", marker="o", markersize=3, picker=5, label=group_label(group_parameters, key))
            line.result_rows = group

            if observable == "mean_torque" and x_parameter == "shear_rate" and show_analytic:
                if log_x and np.all(x_values > 0.0):
                    analytic_shear_rate = np.logspace(np.log10(np.min(x_values)), np.log10(np.max(x_values)), analytic_points)
                else:
                    analytic_shear_rate = np.linspace(np.min(x_values), np.max(x_values), analytic_points)

                temperature = group[0]["temperature"]
                damping = group[0]["damping"]
                coupling = group[0]["coupling"]
                analytic_values = np.array([analytic_torque(value, temperature, damping, a=coupling, xmax=analytic_xmax) for value in analytic_shear_rate])
                ax.plot(analytic_shear_rate, analytic_values, linewidth=1, alpha=0.7, color=line.get_color())

        ax.set_xlabel(parameter_labels[x_parameter])
        ax.set_ylabel(y_label)

        if log_x:
            ax.set_xscale("log")

        if group_parameters:
            ax.legend()

        def inspect_point(event):
            if not hasattr(event.artist, "result_rows") or len(event.ind) == 0:
                return

            row = event.artist.result_rows[event.ind[0]]
            print(f"Inspecting {row['file']}")
            open_time_series(row, results_directory)

        fig.canvas.mpl_connect("pick_event", inspect_point)
        fig.tight_layout()

        if save_prefix is not None:
            fig.savefig(f"{save_prefix}_{filename_suffix}.png", dpi=300)

    plt.show()


def main():
    settings = default_plot_settings()

    parser = argparse.ArgumentParser(description="Plot saved XY-model results using a chosen burn-in time.")
    parser.add_argument("results_directory", nargs="?", default=settings["results_directory"])
    parser.add_argument("--burn-in-time", type=float, default=settings["burn_in_time"])
    parser.add_argument("--x-parameter", choices=parameter_names, default=settings["x_parameter"])
    parser.add_argument("--log-x", action=argparse.BooleanOptionalAction, default=settings["log_x"])
    parser.add_argument("--save-prefix", default=settings["save_prefix"])
    parser.add_argument("--show-analytic", action=argparse.BooleanOptionalAction, default=settings["show_analytic"])
    parser.add_argument("--analytic-points", type=int, default=settings["analytic_points"])
    parser.add_argument("--analytic-xmax", type=float, default=settings["analytic_xmax"])
    args = parser.parse_args()

    rows = load_averages(args.results_directory, args.burn_in_time)
    x_parameter = choose_x_parameter(rows, args.x_parameter)
    processed_index = Path(args.results_directory) / "processed_index.csv"

    write_processed_index(rows, processed_index)
    plot_averages(rows, x_parameter, args.results_directory, args.log_x, args.save_prefix, args.show_analytic, args.analytic_points, args.analytic_xmax)

    print(f"Processed averages written to: {processed_index}")


if __name__ == "__main__":
    main()
