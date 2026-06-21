import numpy as np


def default_parameters():
    params = {}

    params["μ"] = 1.0
    params["N"] = 2 ** 11
    params["EndTime"] = 10000.0
    params["TimeAverage"] = 1000.0
    params["Δt"] = 0.03

    params["kTList"] = np.array([0.1, 1.0], dtype=np.float64)
    params["γ̇List"] = np.concatenate((np.logspace(-3, -1, 12), np.logspace(-1, np.log10(3), 24)[1:]))
    params["analytic_points"] = 300
    params["analytic_xmax"] = 200.0
    params["show_analytic"] = True

    params["save_figures"] = True
    params["show_figures"] = False

    return params
