import time

import matplotlib.pyplot as plt
import numpy as np

from mpi4py import MPI

from analytics import analytic_torque
from solver import evolve
from interface import get_parameters
from parameters import default_parameters


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def run_sweep(params):
    μ = params["μ"]
    N = params["N"]
    EndTime = params["EndTime"]
    TimeAverage = params["TimeAverage"]
    Δt = params["Δt"]
    γ̇List = params["γ̇List"]
    kTList = params["kTList"]

    MeanTorqueArray = np.zeros((kTList.shape[0], γ̇List.shape[0]))
    MeanPotentialArray = np.zeros((kTList.shape[0], γ̇List.shape[0]))

    for j in range(kTList.shape[0]):
        local_torque = np.zeros(γ̇List.shape[0])
        local_potential = np.zeros(γ̇List.shape[0])

        for i in range(rank, γ̇List.shape[0], size):
            local_torque[i], local_potential[i] = evolve(kTList[j], γ̇List[i], μ, N, EndTime, TimeAverage, Δt)

        comm.Reduce(local_torque, MeanTorqueArray[j] if rank == 0 else None, op=MPI.SUM, root=0)
        comm.Reduce(local_potential, MeanPotentialArray[j] if rank == 0 else None, op=MPI.SUM, root=0)

    return MeanTorqueArray, MeanPotentialArray

def plot_torque(γ̇List, gamma_smooth, kTList, MeanTorqueArray, μ, analytic_xmax, show_analytic, save_figures, show_figures):
    fig, ax = plt.subplots()

    torque_colours = plt.cm.Dark2(np.linspace(0, 1, len(kTList)))
    markers = ["o", "s", "^", "D", "v", "*", "P", "X"]

    for j in range(kTList.shape[0]):
        ax.plot(
            γ̇List,
            MeanTorqueArray[j],
            linestyle="None",
            marker=markers[j % len(markers)],
            markersize=3,
            color=torque_colours[j],
            label=fr"Simulation $kT={kTList[j]}$")

        if show_analytic:
            analytic_vals = np.array([analytic_torque(g, kTList[j], μ, xmax=analytic_xmax) for g in gamma_smooth])

            ax.plot(gamma_smooth, analytic_vals, "-", linewidth=1, alpha=0.7, color=torque_colours[j])

    ax.set_xlabel(r"$\dot{\gamma}$")
    ax.set_ylabel(r"$\bar{\tau}$")
    ax.set_xlim(min(γ̇List) - 0.1, max(γ̇List))
    ax.margins(y=0)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_potential(γ̇List, kTList, MeanPotentialArray, save_figures, show_figures):
    fig, ax = plt.subplots()

    potential_colours = plt.cm.OrRd(np.linspace(0.4, 0.9, len(kTList)))
    markers = ["o", "s", "^", "D", "v", "*", "P", "X"]

    for j in range(kTList.shape[0]):
        ax.plot(γ̇List, MeanPotentialArray[j], linestyle="None", marker=markers[j % len(markers)], markersize=3, color=potential_colours[j], label=fr"$kT={kTList[j]}$")

    ax.set_xlabel(r"$\dot{\gamma}$")
    ax.set_ylabel(r"$\langle U \rangle$")
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.show()


def main(use_interface=True):
    if rank == 0:
        if use_interface:
            params = get_parameters()
        else:
            params = default_parameters()
    else:
        params = None

    params = comm.bcast(params, root=0)

    MeanTorqueArray, MeanPotentialArray = run_sweep(params)

    if rank == 0:
        γ̇List = params["γ̇List"]
        kTList = params["kTList"]
        gamma_smooth = np.logspace(np.log10(min(γ̇List)), np.log10(max(γ̇List)), params["analytic_points"])

        plot_torque(
            γ̇List,
            gamma_smooth,
            kTList,
            MeanTorqueArray,
            params["μ"],
            params["analytic_xmax"],
            params["show_analytic"],
            params["save_figures"],
            params["show_figures"])

        plot_potential(
            γ̇List,
            kTList,
            MeanPotentialArray,
            params["save_figures"],
            params["show_figures"])

        return γ̇List, kTList, MeanTorqueArray, MeanPotentialArray

    return None, None, None, None

if __name__ == "__main__":
    start = time.time()
    γ̇List, kTList, MeanTorqueArray, MeanPotentialArray = main(use_interface=True)
    elapsed = time.time() - start

    if rank == 0:
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        print(f"{minutes:02}:{seconds:05.2f}")
