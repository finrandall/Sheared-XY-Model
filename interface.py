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
        "shear_mode": {
            "Low shear": "low",
            "Medium shear": "medium",
            "High shear": "high",
            "Full range": "full",
            "Custom": "custom",
        },
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
        "kTList": ", ".join(str(x) for x in base["kTList"]),
        "N": str(base["N"]),
        "Δt": str(base["Δt"]),
        "EndTime": str(base["EndTime"]),
        "TimeAverage": str(base["TimeAverage"]),
        "shear_mode": "full",
        "γ̇_min": "1e-3",
        "γ̇_mid": "1e-1",
        "γ̇_max": "3.0",
        "γ̇_low_points": "12",
        "γ̇_high_points": "24",
        "show_analytic": base["show_analytic"],
        "save_figures": base["save_figures"],
        "show_figures": base["show_figures"],
    }

    labels = {
        "μ": "Damping μ",
        "kTList": "Temperature kT",
        "N": "System size N",
        "Δt": "Time step Δt",
        "EndTime": "End time",
        "TimeAverage": "Final averaging time",
        "shear_mode": "Shear range",
        "γ̇_min": "Minimum shear rate γ̇",
        "γ̇_mid": "Middle shear rate γ̇",
        "γ̇_max": "Maximum shear rate γ̇",
        "γ̇_low_points": "Low-rate points",
        "γ̇_high_points": "High-rate points",
        "show_analytic": "Show analytic curve",
        "save_figures": "Save figures",
        "show_figures": "Show figures",
    }

    row = 0
    hidden_keys = {"γ̇_min", "γ̇_mid", "γ̇_max", "γ̇_low_points", "γ̇_high_points", "show_analytic", "save_figures", "show_figures"}

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

    def open_shear_parameters():
        window = tk.Toplevel(root)
        window.title("Shear parameters")

        shear_entries = {}
        shear_keys = ["γ̇_min", "γ̇_mid", "γ̇_max", "γ̇_low_points", "γ̇_high_points"]

        shear_row = 0

        for key in shear_keys:
            label = ttk.Label(window, text=labels[key])
            label.grid(row=shear_row, column=0, padx=8, pady=4, sticky="w")

            entry = ttk.Entry(window, width=20)
            entry.insert(0, str(defaults[key]))
            entry.grid(row=shear_row, column=1, padx=8, pady=4)

            shear_entries[key] = entry
            shear_row += 1

        def save_shear_parameters():
            for key in shear_keys:
                defaults[key] = shear_entries[key].get()

            window.destroy()

        button = ttk.Button(window, text="Save", command=save_shear_parameters)
        button.grid(row=shear_row, column=0, columnspan=2, padx=8, pady=10)

    def open_plot_parameters():
        window = tk.Toplevel(root)
        window.title("Plot parameters")

        plot_entries = {}
        plot_keys = ["show_analytic", "save_figures", "show_figures"]

        plot_row = 0

        for key in plot_keys:
            label = ttk.Label(window, text=labels[key])
            label.grid(row=plot_row, column=0, padx=8, pady=4, sticky="w")

            entry = ttk.Combobox(window, values=tuple(display_to_value[key].keys()), width=18, state="readonly")
            entry.set(value_to_display[key][defaults[key]])
            entry.grid(row=plot_row, column=1, padx=8, pady=4)

            plot_entries[key] = entry
            plot_row += 1

        def save_plot_parameters():
            defaults["show_analytic"] = display_to_value["show_analytic"][plot_entries["show_analytic"].get()]
            defaults["save_figures"] = display_to_value["save_figures"][plot_entries["save_figures"].get()]
            defaults["show_figures"] = display_to_value["show_figures"][plot_entries["show_figures"].get()]

            window.destroy()

        button = ttk.Button(window, text="Save", command=save_plot_parameters)
        button.grid(row=plot_row, column=0, columnspan=2, padx=8, pady=10)

    def make_shear_list(shear_mode):
        if shear_mode == "low":
            return np.logspace(-3, -1, 16)

        if shear_mode == "medium":
            return np.logspace(-2, 0, 20)

        if shear_mode == "high":
            return np.logspace(-1, np.log10(3.0), 20)

        γ̇_min = float(defaults["γ̇_min"])
        γ̇_mid = float(defaults["γ̇_mid"])
        γ̇_max = float(defaults["γ̇_max"])
        γ̇_low_points = int(defaults["γ̇_low_points"])
        γ̇_high_points = int(defaults["γ̇_high_points"])

        return np.concatenate((
            np.logspace(np.log10(γ̇_min), np.log10(γ̇_mid), γ̇_low_points),
            np.logspace(np.log10(γ̇_mid), np.log10(γ̇_max), γ̇_high_points)[1:]
        ))

    def submit():
        shear_mode = display_to_value["shear_mode"][entries["shear_mode"].get()]

        params["μ"] = float(entries["μ"].get())
        params["kTList"] = parse_float_list(entries["kTList"].get())
        params["N"] = int(entries["N"].get())
        params["Δt"] = float(entries["Δt"].get())
        params["EndTime"] = float(entries["EndTime"].get())
        params["TimeAverage"] = float(entries["TimeAverage"].get())
        params["γ̇List"] = make_shear_list(shear_mode)
        params["analytic_points"] = params["γ̇List"].shape[0] * 5
        params["analytic_xmax"] = base["analytic_xmax"]
        params["show_analytic"] = defaults["show_analytic"]
        params["save_figures"] = defaults["save_figures"]
        params["show_figures"] = defaults["show_figures"]

        root.withdraw()
        root.update_idletasks()
        root.destroy()

    shear_button = ttk.Button(root, text="Shear parameters", command=open_shear_parameters)
    shear_button.grid(row=row, column=0, columnspan=2, padx=8, pady=4)
    row += 1

    plot_button = ttk.Button(root, text="Plot parameters", command=open_plot_parameters)
    plot_button.grid(row=row, column=0, columnspan=2, padx=8, pady=4)
    row += 1

    button = ttk.Button(root, text="Run simulation", command=submit)
    button.grid(row=row, column=0, columnspan=2, padx=8, pady=10)

    root.mainloop()

    return params