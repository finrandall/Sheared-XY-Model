import numpy as np


def default_parameters():
    params = {}

    params["mode"] = "single"

    params["μ"] = 1.0
    params["N"] = 2 ** 9
    params["EndTime"] = 5000.0
    params["BurnInTime"] = 4000.0
    params["Δt"] = 0.05

    params["kT"] = 0.001
    params["γ̇"] = 0.00585938
    params["stride"] = 1
    params["potential_window"] = 100.0

    params["kTList"] = np.array([0.1, 1.0], dtype=np.float64)

    params["γ̇_min_exp"] = -3.0
    params["γ̇_max_exp"] = np.log10(3.0)
    params["γ̇_points"] = 36
    params["γ̇List"] = np.logspace(params["γ̇_min_exp"], params["γ̇_max_exp"], params["γ̇_points"])

    params["analytic_points"] = 1000
    params["analytic_xmax"] = 1000.0
    params["show_analytic"] = True

    params["save_figures"] = True
    params["show_figures"] = False

    return params