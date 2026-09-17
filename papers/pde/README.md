# PDE-generated matrix experiments

This folder adds PDE-discretized sparse linear systems for the GMRES-IR/RL
experiments, following the matrix families used in "Estimating Condition Number
with Graph Neural Networks" (arXiv:2603.10277).

The default dataset contains four PDE matrix families:

- `poisson_2d`
- `anisotropic_poisson_2d`
- `high_contrast_diffusion_2d`
- `convection_diffusion_2d`

The right-hand side is generated with a manufactured solution by default:
`b = A @ x_true`, where `x_true` is a smooth sine-mode field on the grid. This
keeps the solve target known for forward-error evaluation. Alternative RHS
modes are available through `--rhs-mode random` and `--rhs-mode piecewise`.

## Generate Data Only

For no-Slurm runs, first activate an environment with `numpy`, `scipy`,
`pandas`, `torch`, and `pychop` available.

```bash
cd /path/to/rlops/papers/pde
python pde_matrix_utils.py --output-dir data_pde --num-train 100 --num-test 100
```

By default, both training and test PDE systems are generated with dimensions
between 100 and 500. The training runners use 100 episodes by default, and
cached data are reused only when `dataset_config.json` matches the requested
generation settings.

Useful options:

```bash
python pde_matrix_utils.py \
  --output-dir data_pde \
  --train-size-min 100 \
  --train-size-max 500 \
  --test-size-min 100 \
  --test-size-max 500 \
  --rhs-mode smooth \
  --cond-method auto \
  --seed 3407
```

`--cond-method auto` uses a dense condition-number computation for the default
small matrices and a sparse 1-norm estimator for larger override ranges, with
the switch controlled by `--dense-cond-limit`.

## Run Experiments

The runners reuse the existing `papers/sparse_test/gmresir_solver1.py` solver
implementation and change only the generated dataset and result directories.
Run W1 and W2 as two separate experiment entry points without Slurm. The
default run trains and tests on 100--500 dimensional systems:

```bash
cd /path/to/rlops/papers/pde
python run_pde_w1_experiment.py
python run_pde_w2_experiment.py
```

Together these create:

- `results_pde1`: W1, `tol=1e-6`
- `results_pde2`: W2, `tol=1e-6`
- `results_pde3`: W1, `tol=1e-8`
- `results_pde4`: W2, `tol=1e-8`

`run_pde_experiment.py` remains available as a lower-level runner when you want
to manually select configs, for example `python run_pde_experiment.py --configs
1`.

For a quick local check:

```bash
python run_pde_w1_experiment.py \
  --configs 1 \
  --num-train 2 \
  --num-test 2 \
  --size-min 16 \
  --size-max 25 \
  --episodes 1 \
  --top-k 1 \
  --data-dir /tmp/data_pde_smoke \
  --results-root /tmp/results_pde_smoke \
  --regenerate-data
```

## Visualize Results

Visualization is intentionally separated from experiment execution and can also
be run without Slurm:

```bash
cd /path/to/rlops/papers/pde
python visualize_pde_results.py --font-size 15
```

The global `--font-size` value is applied to x/y labels, x/y tick labels,
legends, and titles. Training-curve legends are read from `run_config.json`, so
they emphasize `$W_1$`/`$W_2$` and `tau` instead of internal directory names.
Figures are written to `figures_pde/` by default; for W2 runs the visualizer
also writes the paper-facing aliases `pde_error_iterations_w2_tol1e6.*` and
`pde_error_iterations_w2_tol1e8.*`.

You can point the visualizer at smoke-test or custom directories:

```bash
python visualize_pde_results.py \
  --results-root /tmp/results_pde_smoke \
  --result-dirs results_pde1 \
  --data-dir /tmp/data_pde_smoke \
  --output-dir /tmp/figures_pde_smoke \
  --font-size 12
```

## Run on Convergence

`run_pde_convergence.sh` is a CPU-only `sbatch` script for the LIP6 Convergence
cluster. It uses the `convergence` partition, requests 12 CPU cores and 64 GB
RAM, runs the PDE experiments, and then runs the separate visualization step.

```bash
cd /path/to/rlops/papers/pde
sbatch run_pde_convergence.sh
```

The job script uses `SLURM_SUBMIT_DIR` as the experiment directory, so submit it
from `rlops/papers/pde`. If you submit from another directory, set
`RLOPS_PDE_DIR=/path/to/rlops/papers/pde`. For quick smoke tests, the legacy
`SIZE_MIN`/`SIZE_MAX` environment variables still set both train and test
ranges at once.

The script loads:

```bash
module load python/anaconda3
```

Useful submit-time overrides:

```bash
CONDA_ENV=rlops RUN_W2=0 sbatch run_pde_convergence.sh
NUM_TRAIN=100 NUM_TEST=100 TRAIN_SIZE_MIN=100 TRAIN_SIZE_MAX=500 TEST_SIZE_MIN=100 TEST_SIZE_MAX=500 EPISODES=100 sbatch run_pde_convergence.sh
NUM_TRAIN=2 NUM_TEST=2 SIZE_MIN=16 SIZE_MAX=25 EPISODES=1 RUN_W2=0 sbatch run_pde_convergence.sh
RUN_VISUALIZATION=0 sbatch run_pde_convergence.sh
```
