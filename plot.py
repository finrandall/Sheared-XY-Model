import matplotlib.pyplot as plt
import numpy as np

from analytics import analytic_torque


def plot_single_potential(time_series, potential_energy):
    fig, ax = plt.subplots()

    ax.plot(time_series, potential_energy, linewidth=1)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$U(t)$")
    fig.tight_layout()
    plt.show()


def plot_single_torque(time_series, mean_torque):
    fig, ax = plt.subplots()

    ax.plot(time_series, mean_torque, linewidth=1)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\bar{\tau}(t)$")
    fig.tight_layout()
    plt.show()


def plot_final_velocity(final_velocity):
    fig, ax = plt.subplots()

    spin_list = np.arange(final_velocity.shape[0])

    ax.grid()
    ax.scatter(spin_list, final_velocity, color="black", s=1.4, zorder=2)
    ax.set_xlabel("Spin")
    ax.set_ylabel(r"$v$")
    ax.tick_params(axis="both", length=0)

    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    plt.show()

def plot_average_velocity(average_velocity):
    fig, ax = plt.subplots()

    spin_list = np.arange(average_velocity.shape[0])

    ax.grid()
    ax.scatter(spin_list, average_velocity, color="black", s=1.4, zorder=2)
    ax.set_xlabel("Spin")
    ax.set_ylabel(r"$\langle v \rangle$")
    ax.tick_params(axis="both", length=0)

    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    plt.show()


def plot_local_potential_history(local_potential_history):
    N = local_potential_history.shape[1]

    plt.figure(figsize=(7, 4))
    plt.imshow(local_potential_history, aspect="auto", cmap="gray_r", interpolation="nearest")
    plt.xticks([0, N])
    plt.xlabel("Spins")
    plt.ylabel("t")
    plt.yticks([])
    plt.show()


def plot_rotor_autocorrelation(rotor_autocorrelation):
    fig, ax = plt.subplots()

    half_length = rotor_autocorrelation.shape[0] // 2 + 1
    separation = np.arange(half_length)

    ax.plot(separation, rotor_autocorrelation[:half_length], linewidth=1)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$C(r)$")
    fig.tight_layout()
    plt.show()


def plot_torque(γ̇List, kTList, MeanTorqueArray, μ, show_analytic):
    fig, ax = plt.subplots()

    torque_colours = plt.cm.Dark2(np.linspace(0, 1, len(kTList)))
    markers = ["o", "s", "^", "D", "v", "*", "P", "X"]
    gamma_smooth = np.linspace(min(γ̇List), max(γ̇List), 1000)
    analytic_xmax = max(γ̇List)

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
    ax.set_xlim(min(γ̇List), max(γ̇List))
    ax.margins(y=0)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_potential(γ̇List, kTList, MeanPotentialArray):
    fig, ax = plt.subplots()

    potential_colours = plt.cm.OrRd(np.linspace(0.4, 0.9, len(kTList)))
    markers = ["o", "s", "^", "D", "v", "*", "P", "X"]

    for j in range(kTList.shape[0]):
        ax.plot(
            γ̇List,
            MeanPotentialArray[j],
            linestyle="None",
            marker=markers[j % len(markers)],
            markersize=3,
            color=potential_colours[j],
            label=fr"$kT={kTList[j]}$")

    ax.set_xlabel(r"$\dot{\gamma}$")
    ax.set_ylabel(r"$\langle U \rangle$")
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.show()
