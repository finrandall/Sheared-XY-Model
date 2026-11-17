# Parallel shear XY model

Simulates the 1D classical XY model under steady shear with conserved angular momentum
(Evans et al., PRL 114, 138301). Each bond carries a torque
`tau_i = a sin(phi_i) + mu dphi_i/dt + eta_i`, with `phi_i = theta_{i+1} - theta_i` and
`<eta_i(t) eta_j(t')> = 2 mu kT delta_ij delta(t - t')`; rotor `i` feels `tau_i - tau_{i-1}`
(moment of inertia 1). The last bond couples rotor `N-1` to the image of rotor 0, rotated by
`omega t` with `omega = N * shear_rate`.

Integration uses the Groot-Warren modified velocity-Verlet scheme (lambda = 1/2) with one force
evaluation per step; see the docstring in `solver.py`.

`--coupling` sets `a`; `--coupling 0` switches the potential off (the reference system).

The stored `mean_torque` is `mu * shear_rate + a <sin phi>`, the bond-averaged torque without
its zero-mean noise term. It has the same expectation as the full torque but much smaller
sampling noise.

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
