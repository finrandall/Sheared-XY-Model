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


def get_mode():
    selected = {"mode": None}

    root = tk.Tk()
    root.title("Marmalade mode")

    label = ttk.Label(root, text="Simulation mode")
    label.grid(row=0, column=0, padx=8, pady=8, sticky="w")

    mode_box = ttk.Combobox(
        root,
        values=("Single realisation", "Parameter sweep"),
        width=20,
        state="readonly")
    mode_box.set("Single realisation")
    mode_box.grid(row=0, column=1, padx=8, pady=8)

    def submit():
        if mode_box.get() == "Single realisation":
            selected["mode"] = "single"
        else:
            selected["mode"] = "sweep"

        root.withdraw()
        root.update_idletasks()
        root.destroy()

    button = ttk.Button(root, text="Next", command=submit)
    button.grid(row=1, column=0, columnspan=2, padx=8, pady=10)

    root.mainloop()

    return selected["mode"]


def get_single_parameters(base):
    params = {"mode": "single"}

    root = tk.Tk()
    root.title("Single realisation parameters")

    entries = {}
    base_γ̇List = base.get("γ̇List", np.linspace(0.001, 3.0, 30))

    defaults = {
        "μ": str(base["μ"]),
        "kT": str(base.get("kT", base["kTList"][0])),
        "γ̇": str(base.get("γ̇", base_γ̇List[0])),
        "N": str(base["N"]),
        "Δt": str(base["Δt"]),
        "EndTime": str(base["EndTime"]),
        "BurnInTime": str(base.get("BurnInTime", 0.0)),
        "sample_interval": str(base.get("sample_interval", 1)),
        "potential_window": str(base.get("potential_window", base["EndTime"])),
    }

    labels = {
        "μ": "Damping μ",
        "kT": "Temperature kT",
        "γ̇": "Shear rate γ̇",
        "N": "System size N",
        "Δt": "Time step Δt",
        "EndTime": "End time",
        "BurnInTime": "Burn-in time",
        "sample_interval": "Sample interval",
        "potential_window": "Final potential window",
    }

    row = 0

    for key, value in defaults.items():
        label = ttk.Label(root, text=labels[key])
        label.grid(row=row, column=0, padx=8, pady=4, sticky="w")

        entry = ttk.Entry(root, width=20)
        entry.insert(0, value)
        entry.grid(row=row, column=1, padx=8, pady=4)

        entries[key] = entry
        row += 1

    def submit():
        params["μ"] = float(entries["μ"].get())
        params["kT"] = float(entries["kT"].get())
        params["γ̇"] = float(entries["γ̇"].get())
        params["N"] = int(entries["N"].get())
        params["Δt"] = float(entries["Δt"].get())
        params["EndTime"] = float(entries["EndTime"].get())
        params["BurnInTime"] = float(entries["BurnInTime"].get())
        params["sample_interval"] = int(entries["sample_interval"].get())
        params["potential_window"] = float(entries["potential_window"].get())

        root.withdraw()
        root.update_idletasks()
        root.destroy()

    button = ttk.Button(root, text="Run simulation", command=submit)
    button.grid(row=row, column=0, columnspan=2, padx=8, pady=10)

    root.mainloop()

    if len(params) == 1:
        return None

    return params


def get_sweep_parameters(base):
    params = {"mode": "sweep"}

    root = tk.Tk()
    root.title("Parameter sweep parameters")

    entries = {}
    base_γ̇List = base.get("γ̇List", np.linspace(0.001, 3.0, 30))

    display_to_value = {
        "show_analytic": {
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
        "γ̇_min": str(np.min(base_γ̇List)),
        "γ̇_max": str(np.max(base_γ̇List)),
        "γ̇_points": str(base_γ̇List.shape[0]),
        "N": str(base["N"]),
        "Δt": str(base["Δt"]),
        "EndTime": str(base["EndTime"]),
        "TimeAverage": str(base["TimeAverage"]),
        "show_analytic": base["show_analytic"],
    }

    labels = {
        "μ": "Damping μ",
        "kTList": "Temperature sweep kT",
        "γ̇_min": "Minimum shear rate γ̇",
        "γ̇_max": "Maximum shear rate γ̇",
        "γ̇_points": "Number of shear points",
        "N": "System size N",
        "Δt": "Time step Δt",
        "EndTime": "End time",
        "TimeAverage": "Final averaging time",
        "show_analytic": "Show analytic curve",
    }

    row = 0

    for key, value in defaults.items():
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

    def submit():
        γ̇_min = float(entries["γ̇_min"].get())
        γ̇_max = float(entries["γ̇_max"].get())
        γ̇_points = int(entries["γ̇_points"].get())

        params["μ"] = float(entries["μ"].get())
        params["kTList"] = parse_float_list(entries["kTList"].get())
        params["γ̇List"] = np.linspace(γ̇_min, γ̇_max, γ̇_points)
        params["N"] = int(entries["N"].get())
        params["Δt"] = float(entries["Δt"].get())
        params["EndTime"] = float(entries["EndTime"].get())
        params["TimeAverage"] = float(entries["TimeAverage"].get())
        params["show_analytic"] = display_to_value["show_analytic"][entries["show_analytic"].get()]

        root.withdraw()
        root.update_idletasks()
        root.destroy()

    button = ttk.Button(root, text="Run simulation", command=submit)
    button.grid(row=row, column=0, columnspan=2, padx=8, pady=10)

    root.mainloop()

    if len(params) == 1:
        return None

    return params


def get_parameters():
    base = default_parameters()
    mode = get_mode()

    if mode is None:
        return None

    if mode == "single":
        return get_single_parameters(base)

    return get_sweep_parameters(base)