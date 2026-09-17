# Dense matrix experiments

This folder contains the dense synthetic GMRES-IR/RL experiments used by the
paper. The reported workflow is CPU-based and uses the standard precision set
`bf16`, `tf32`, `fp32`, and `fp64`.

## Run Without Slurm

From a workstation or an interactive CPU node, first activate an environment
with `numpy`, `scipy`, `pandas`, `torch`, and `pychop` available. Then run from
this directory:

```bash
cd /path/to/rlops/papers/dense_test
```

To regenerate the four dense RL training/testing result directories without
Slurm, run the four solver entry points directly:

```bash
python gmresir_solver1.py
python gmresir_solver2.py
python gmresir_solver3.py
python gmresir_solver4.py
```

These write `results_random_dense1`--`results_random_dense4`. The solver files
use their built-in experiment settings, so no `sbatch` environment variables are
needed for this path.

## Visualize Results

Plotting scripts can be run directly after the result directories are present:

```bash
python plot_err1.py --font-size 15
python plot_err2.py --font-size 15
python plot_err3.py --font-size 15
python plot_err4.py --font-size 15
python vis_types_dist.py
```

The `plot_err*.py` wrappers share `plot_error_iterations.py`; the global
`--font-size` value controls x/y labels, x/y ticks, legends, and titles.
Each wrapper also writes the paper-facing alias such as
`error_iterations_comparison2.pdf`.
