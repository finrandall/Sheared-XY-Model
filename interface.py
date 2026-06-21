import tkinter as tk

from tkinter import ttk

import numpy as np

from parameters import default_parameters


def parse_float_list(text):
    values = []
    for item in text.replace(" ", "").split(","):
        if item:
            values.append(float(item))
    return np.array(values, dtype=np.float64)


def get_parameters():
    params = {}

    root = tk.Tk()
    root.title("Marmalade parameters")

    entries = {}

    base = default_parameters()

    display_to_value = {
        "show_analytic": {
            "True": True,
            "False": False,
        },
        "save_figures": {
            "True": True,
            "False": False,
        },
        "show_figures": {
            "True": True,
            "False": False,
        },
    }

    value_to_display = {}

    for key, mapping in display_to_value.items():
        value_to_display[key] = {value: display for display, value in mapping.items()}

    defaults = {
        "μ": str(base["μ"]),
        "N": str(base["N"]),
        "EndTime": str(base["EndTime"]),
        "TimeAverage": str(base["TimeAverage"]),
        "Δt": str(base["Δt"]),
        "kTList": ", ".join(str(x) for x in base["kTList"]),
        "γ̇_min": "1e-3",
        "γ̇_mid": "1e-1",
        "γ̇_max": "3.0",
        "γ̇_low_points": "12",
        "γ̇_high_points": "24",
        "analytic_points": str(base["analytic_points"]),
        "analytic_xmax": str(base["analytic_xmax"]),
        "show_analytic": base["show_analytic"],
        "save_figures": base["save_figures"],
        "show_figures": base["show_figures"],
    }

    labels = {
        "μ": "Damping μ",
        "N": "System size N",
        "EndTime": "End time",
        "TimeAverage": "Average time",
        "Δt": "Time step Δt",
        "kTList": "Temperatures kT",
        "γ̇_min": "Minimum shear rate γ̇",
        "γ̇_mid": "Middle shear rate γ̇",
        "γ̇_max": "Maximum shear rate γ̇",
        "γ̇_low_points": "Low-rate points",
        "γ̇_high_points": "High-rate points",
        "analytic_points": "Analytic curve points",
        "analytic_xmax": "Analytic integral xmax",
        "show_analytic": "Show analytic curve",
        "save_figures": "Save figures",
        "show_figures": "Show figures",
    }

    row = 0
    hidden_keys = {"analytic_points", "analytic_xmax", "show_analytic", "save_figures", "show_figures"}

    for key, value in defaults.items():
        if key in hidden_keys:
            continue

        label = ttk.Label(root, text=labels[key])
        label.grid(row=row, column=0, padx=8, pady=4, sticky="w")

        if key in display_to_value:
            entry = ttk.Combobox(root, values=tuple(display_to_value[key].keys()), width=18, state="readonly")
            entry.set(value_to_display[key][value])
        else:
            entry = ttk.Entry(root, width=20)
            entry.insert(0, value)

        entry.grid(row=row, column=1, padx=8, pady=4)
        entries[key] = entry
        row += 1

    def open_plot_parameters():
        window = tk.Toplevel(root)
        window.title("Plot parameters")

        plot_entries = {}
        plot_keys = ["analytic_points", "analytic_xmax", "show_analytic", "save_figures", "show_figures"]

        plot_row = 0

        for key in plot_keys:
            label = ttk.Label(window, text=labels[key])
            label.grid(row=plot_row, column=0, padx=8, pady=4, sticky="w")

            if key in display_to_value:
                entry = ttk.Combobox(window, values=tuple(display_to_value[key].keys()), width=18, state="readonly")
                entry.set(value_to_display[key][defaults[key]])
            else:
                entry = ttk.Entry(window, width=20)
                entry.insert(0, str(defaults[key]))

            entry.grid(row=plot_row, column=1, padx=8, pady=4)
            plot_entries[key] = entry
            plot_row += 1

        def save_plot_parameters():
            defaults["analytic_points"] = plot_entries["analytic_points"].get()
            defaults["analytic_xmax"] = plot_entries["analytic_xmax"].get()
            defaults["show_analytic"] = display_to_value["show_analytic"][plot_entries["show_analytic"].get()]
            defaults["save_figures"] = display_to_value["save_figures"][plot_entries["save_figures"].get()]
            defaults["show_figures"] = display_to_value["show_figures"][plot_entries["show_figures"].get()]

            window.destroy()

        button = ttk.Button(window, text="Save", command=save_plot_parameters)
        button.grid(row=plot_row, column=0, columnspan=2, padx=8, pady=10)

    def submit():
        γ̇_min = float(entries["γ̇_min"].get())
        γ̇_mid = float(entries["γ̇_mid"].get())
        γ̇_max = float(entries["γ̇_max"].get())
        γ̇_low_points = int(entries["γ̇_low_points"].get())
        γ̇_high_points = int(entries["γ̇_high_points"].get())

        params["μ"] = float(entries["μ"].get())
        params["N"] = int(entries["N"].get())
        params["EndTime"] = float(entries["EndTime"].get())
        params["TimeAverage"] = float(entries["TimeAverage"].get())
        params["Δt"] = float(entries["Δt"].get())
        params["kTList"] = parse_float_list(entries["kTList"].get())
        params["γ̇List"] = np.concatenate((np.logspace(np.log10(γ̇_min), np.log10(γ̇_mid), γ̇_low_points), np.logspace(np.log10(γ̇_mid), np.log10(γ̇_max), γ̇_high_points)[1:]))
        params["analytic_points"] = int(defaults["analytic_points"])
        params["analytic_xmax"] = float(defaults["analytic_xmax"])
        params["show_analytic"] = defaults["show_analytic"]
        params["save_figures"] = defaults["save_figures"]
        params["show_figures"] = defaults["show_figures"]

        root.withdraw()
        root.update_idletasks()
        root.destroy()

    plot_button = ttk.Button(root, text="Plot parameters", command=open_plot_parameters)
    plot_button.grid(row=row, column=0, columnspan=2, padx=8, pady=4)
    row += 1

    button = ttk.Button(root, text="Run simulation", command=submit)
    button.grid(row=row, column=0, columnspan=2, padx=8, pady=10)

    root.mainloop()

    return params
