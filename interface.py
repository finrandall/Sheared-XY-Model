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
    base = default_parameters()
    params = {"mode": None}

    root = tk.Tk()
    root.title("Marmalade parameters")

    entries = {}

    mode_label = ttk.Label(root, text="Simulation mode")
    mode_label.grid(row=0, column=0, padx=8, pady=8, sticky="w")

    mode_box = ttk.Combobox(
        root,
        values=("Single realisation", "Shear rate sweep"),
        width=20,
        state="readonly")
    mode_box.set("Single realisation")
    mode_box.grid(row=0, column=1, padx=8, pady=8)

    field_frame = ttk.Frame(root)
    field_frame.grid(row=1, column=0, columnspan=2, padx=8, pady=4, sticky="nsew")

    display_to_value = {
        "show_analytic": {
            "True": True,
            "False": False,
        },
    }

    value_to_display = {}

    for key, mapping in display_to_value.items():
        value_to_display[key] = {value: display for display, value in mapping.items()}

    def clear_fields():
        for widget in field_frame.winfo_children():
            widget.destroy()

        entries.clear()

    def add_section_fields(parent, defaults, labels):
        row = 0

        for key, value in defaults.items():
            label = ttk.Label(parent, text=labels[key])
            label.grid(row=row, column=0, padx=8, pady=4, sticky="w")

            if key in display_to_value:
                entry = ttk.Combobox(parent, values=tuple(display_to_value[key].keys()), width=18, state="readonly")
                entry.set(value_to_display[key][value])
            else:
                entry = ttk.Entry(parent, width=20)
                entry.insert(0, value)

            entry.grid(row=row, column=1, padx=8, pady=4)
            entries[key] = entry
            row += 1

    def build_single_fields():
        clear_fields()

        base_γ̇List = base.get("γ̇List", np.logspace(-3.0, np.log10(3.0), 35))

        parameter_defaults = {
            "kT": str(base.get("kT", base["kTList"][0])),
            "μ": str(base["μ"]),
            "γ̇": str(base.get("γ̇", base_γ̇List[0])),
        }

        parameter_labels = {
            "kT": "Temperature kT",
            "μ": "Damping μ",
            "γ̇": "Shear rate γ̇",
        }

        time_defaults = {
            "Δt": str(base["Δt"]),
            "EndTime": str(base["EndTime"]),
            "BurnInTime": str(base.get("BurnInTime", 0.0)),
            "stride": str(base.get("stride", base.get("sample_interval", 1))),
        }

        time_labels = {
            "Δt": "Time step Δt",
            "EndTime": "End time",
            "BurnInTime": "Burn-in time",
            "stride": "Stride",
        }

        system_defaults = {
            "N": str(base["N"]),
            "potential_window": str(base.get("potential_window", base["EndTime"])),
        }

        system_labels = {
            "N": "System size N",
            "potential_window": "Final potential window",
        }

        parameter_frame = ttk.LabelFrame(field_frame, text="Parameters")
        parameter_frame.grid(row=0, column=0, padx=8, pady=6, sticky="ew")

        time_frame = ttk.LabelFrame(field_frame, text="Time constants")
        time_frame.grid(row=1, column=0, padx=8, pady=6, sticky="ew")

        system_frame = ttk.LabelFrame(field_frame, text="System and output")
        system_frame.grid(row=2, column=0, padx=8, pady=6, sticky="ew")

        add_section_fields(parameter_frame, parameter_defaults, parameter_labels)
        add_section_fields(time_frame, time_defaults, time_labels)
        add_section_fields(system_frame, system_defaults, system_labels)

    def build_sweep_fields():
        clear_fields()

        base_γ̇List = base.get("γ̇List", np.logspace(-3.0, np.log10(3.0), 35))

        parameter_defaults = {
            "μ": str(base["μ"]),
            "kTList": ", ".join(str(x) for x in base["kTList"]),
        }

        parameter_labels = {
            "μ": "Damping μ",
            "kTList": "Temperature sweep kT",
        }

        shear_defaults = {
            "γ̇_min": str(base.get("γ̇_min", np.min(base_γ̇List))),
            "γ̇_max": str(base.get("γ̇_max", np.max(base_γ̇List))),
            "γ̇_points": str(base.get("γ̇_points", base_γ̇List.shape[0])),
        }

        shear_labels = {
            "γ̇_min": "Minimum shear rate γ̇",
            "γ̇_max": "Maximum shear rate γ̇",
            "γ̇_points": "Number of shear points",
        }

        time_defaults = {
            "Δt": str(base["Δt"]),
            "EndTime": str(base["EndTime"]),
            "BurnInTime": str(base.get("BurnInTime", 0.0)),
        }

        time_labels = {
            "Δt": "Time step Δt",
            "EndTime": "End time",
            "BurnInTime": "Burn-in time",
        }

        system_defaults = {
            "N": str(base["N"]),
        }

        system_labels = {
            "N": "System size N",
        }

        output_defaults = {
            "show_analytic": base["show_analytic"],
        }

        output_labels = {
            "show_analytic": "Show analytic curve",
        }

        parameter_frame = ttk.LabelFrame(field_frame, text="Parameters")
        parameter_frame.grid(row=0, column=0, padx=8, pady=6, sticky="ew")

        shear_frame = ttk.LabelFrame(field_frame, text="Shear rate sweep")
        shear_frame.grid(row=1, column=0, padx=8, pady=6, sticky="ew")

        time_frame = ttk.LabelFrame(field_frame, text="Time constants")
        time_frame.grid(row=2, column=0, padx=8, pady=6, sticky="ew")

        system_frame = ttk.LabelFrame(field_frame, text="System")
        system_frame.grid(row=3, column=0, padx=8, pady=6, sticky="ew")

        output_frame = ttk.LabelFrame(field_frame, text="Output")
        output_frame.grid(row=4, column=0, padx=8, pady=6, sticky="ew")

        add_section_fields(parameter_frame, parameter_defaults, parameter_labels)
        add_section_fields(shear_frame, shear_defaults, shear_labels)
        add_section_fields(time_frame, time_defaults, time_labels)
        add_section_fields(system_frame, system_defaults, system_labels)
        add_section_fields(output_frame, output_defaults, output_labels)

    def update_mode_fields(event=None):
        if mode_box.get() == "Single realisation":
            build_single_fields()
        else:
            build_sweep_fields()

    def submit():
        if mode_box.get() == "Single realisation":
            params["mode"] = "single"

            params["μ"] = float(entries["μ"].get())
            params["kT"] = float(entries["kT"].get())
            params["γ̇"] = float(entries["γ̇"].get())
            params["N"] = int(entries["N"].get())
            params["Δt"] = float(entries["Δt"].get())
            params["EndTime"] = float(entries["EndTime"].get())
            params["BurnInTime"] = float(entries["BurnInTime"].get())
            params["stride"] = int(entries["stride"].get())
            params["potential_window"] = float(entries["potential_window"].get())

        else:
            params["mode"] = "sweep"

            γ̇_min_exp = float(entries["γ̇_min_exp"].get())
            γ̇_max_exp = float(entries["γ̇_max_exp"].get())
            γ̇_points = int(entries["γ̇_points"].get())

            params["μ"] = float(entries["μ"].get())
            params["kTList"] = parse_float_list(entries["kTList"].get())
            params["γ̇List"] = np.logspace(γ̇_min_exp, γ̇_max_exp, γ̇_points)
            params["N"] = int(entries["N"].get())
            params["Δt"] = float(entries["Δt"].get())
            params["EndTime"] = float(entries["EndTime"].get())
            params["BurnInTime"] = float(entries["BurnInTime"].get())
            params["show_analytic"] = display_to_value["show_analytic"][entries["show_analytic"].get()]

        root.withdraw()
        root.update_idletasks()
        root.destroy()

    mode_box.bind("<<ComboboxSelected>>", update_mode_fields)

    button = ttk.Button(root, text="Run simulation", command=submit)
    button.grid(row=2, column=0, columnspan=2, padx=8, pady=10)

    build_single_fields()

    root.mainloop()

    if params["mode"] is None:
        return None

    return params