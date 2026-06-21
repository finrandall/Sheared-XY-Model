# Parallel shear XY model

This is a Marmalade-style split of `parallel_shear_sweep.py`.

## Files

- `main.py`: MPI sweep, plotting, and entry point.
- `interface.py`: Tkinter parameter window.
- `parameters.py`: default parameters.
- `initial.py`: random initialisation helper.
- `forces.py`: force kernels.
- `observables.py`: potential energy observable.
- `integrator.py`: DPD/velocity-Verlet evolution kernel.
- `analytics.py`: analytic torque curve.

## PyCharm

Set the project interpreter to the environment containing the packages in `requirements.txt`.

Run `main.py` directly for a normal single-process run with the Tkinter interface.

For MPI, run from the terminal in this folder, for example:

```bash
mpiexec -n 4 python main.py
```

Only rank 0 opens the parameter window. The selected parameters are broadcast to the other ranks.
