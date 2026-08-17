# Parallel shear XY model

Add description.

## PyCharm

Set the project interpreter to the environment containing the packages in `requirements.txt`.

The default parameter arrays are in `parameters.py`. One value fixes a parameter;
multiple values sweep it. Values can be written explicitly or generated with NumPy.
Running `parameters.py` directly uses those arrays:

```bash
python parameters.py
```

`main.py` does the same thing. Command-line flags override the corresponding arrays
from `parameters.py`:

```bash
python main.py --temperature 0.1 0.5 1.0 --shear-rate 0.001 0.01 0.1
```

Every combination of the supplied parameters is simulated. The example above runs
nine simulations. Use `--help` to see all available flags:

```bash
python main.py --help
```

Each simulation retains its full time series. Detailed plots are displayed when
only one parameter combination is run. Results are written to the `results`
directory by default. Use `--output-directory` to choose another location:

```bash
python main.py --temperature 0.1 1.0 --output-directory results/temperature_sweep
```

Each simulation is stored as a compressed `.npz` file containing all parameters,
time series and measured observables. The accompanying `index.csv` lists the
parameters, averages and filename for every simulation. A plotting script can load
a result with:

```python
import numpy as np

data = np.load("results/simulation_0000.npz")
time = data["time_series"]
torque = data["mean_torque"]
```

Velocity profiles and equal-time spatial rotor correlations are stored as block
averages. `block_samples` in `parameters.py` controls the number of sampled times
in each block, or it can be overridden from the terminal:

```bash
python main.py --block-samples 100
```

The archive contains the start time, end time and sample count for each block, so
blocks before a chosen equilibration time can be excluded during later analysis.

Rotor-angle trajectories for time-dependent analysis are optional because they can
be large. Enable them in `parameters.py`, or use:

```bash
python main.py --save-trajectory --trajectory-stride 100
```

The trajectory is stored as `float32`. `correlations.py` calculates a temporal
rotor correlation from a saved trajectory:

```bash
python correlations.py results/simulation_0000.npz --max-lag 1000
```

For MPI, run from the terminal in this folder, for example:

```bash
mpiexec -n 4 python main.py --temperature 0.1 1.0 --shear-rate 0.001 0.01 0.1
```
