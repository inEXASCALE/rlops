# CPU Low-Condition GMRES-IR Experiment

This directory adds a CPU validation path for the low-condition dense regime
motivated by `chenxinye/irp3`.  It tests whether the RL precision policy can
choose mixed-precision actions for the four GMRES-IR stages from `fp16`, `fp32`, and `fp64` and then
validate the chosen precision with actual compiled C++ CPU timings.

## Experiment Target

- Matrix regime: dense low-condition systems with target condition numbers in
  `10^0`--`10^3`.
- Default size range: `1000`--`1500`.
- Default split: 50 training systems and 100 testing systems.
- Actions: monotone four-tuples `[uf, ug, u, ur]`, where each stage chooses only from `fp16`, `fp32`, and `fp64`. The action order is `fp16 < fp32 < fp64`; `bf16` is intentionally excluded from this GMRES-IR action space.
- RL policy: epsilon-greedy contextual bandit over a `10 x 10` discretized state
  table using `log10(state_condition_number)` and `log10(||A||_inf)`, matching
  the dense/sparse/PDE state model.
- State condition feature: the target condition number is used only to generate
  controlled regimes and for target-regime reporting.  By default, the RL state
  uses a Hager--Higham-type one-norm condition estimate computed from each
  generated matrix and cached in `condition_features_hager_higham.csv`.
- Reward: the PDE-style accuracy reward plus condition-scaled precision reward
  minus a logarithmic GMRES-iteration penalty. Native CPU speedup and storage
  ratio are still reported as validation metrics, but they are no longer the
  reward optimized by the CPU policy.
- Four standard configurations: `W1` and `W2` crossed with tolerances `1e-6`
  and `1e-8`.

The C++ path consumes the same four stage names as the Python dense solver: `uf` controls dense LU factor/preconditioner storage, `ug` controls restarted left-preconditioned GMRES arithmetic, `u` controls working/update precision, and `ur` controls residual precision. FP16 data are stored as IEEE binary16; if the host compiler enables F16C, hardware conversion intrinsics are used, otherwise a portable half conversion fallback is used. AVX-512 row kernels are compiled when `-march=native` exposes them on the target CPU. The native runner records `||A||_inf` and GMRES iteration counts so the CPU policy uses the same two-feature state and logarithmic iteration-penalty structure as the PDE experiments.

## Run

From this directory:

```bash
python run_cpu_lowcond_experiment.py
python plot_cpu_lowcond_results.py --font-size 15
```

By default the runner tries to compile `gmres_ir_cpu.cpp` with:

```bash
c++ -O3 -std=c++17 -march=native gmres_ir_cpu.cpp -o build/gmres_ir_cpu
```

If compilation is unavailable in `--mode auto`, it falls back to the proxy
evaluator so training and plotting infrastructure can still be tested. Use
`--mode native` to require actual compiled C++ timing.

Key validation defaults:

- `--repeats 3`: number of repeated timed solves per matrix.
- `--outer-max-iters 10`: maximum outer refinement iterations.
- `--gmres-max-iters 30`: restart dimension / maximum Arnoldi steps per GMRES cycle. Use `0` for full GMRES, matching irp3 `restart=0`.
- `--gmres-cycles 1`: maximum restarted GMRES cycles per outer refinement iteration.
- `--gmres-tol 1e-4`: relative residual threshold for the preconditioned inner GMRES solve.
- `--tolerance 1e-6`: relative forward-error threshold used by a given configuration.
- `--backward-tol 1e-14`: diagnostic normwise backward-error threshold. It is reported as `backward_pass_diagnostic` but is not part of the default validity or reward gate.

The native runner accepts a case when the selected solve completes and the relative forward error is at most `--tolerance`. The independently computed FP64 normwise backward error is retained as the diagnostic column `backward_pass_diagnostic`; it is not used by `valid` or reward. A fixed `1e-14` gate rejects useful low-precision configurations in this low-condition regime even when they satisfy the user-specified accuracy threshold. The first solve for each matrix is a warm-up and is not included in `avg_ms`.

Training and validation use a fixed-budget invariant: within one run, every legal precision action is measured with exactly the same `repeats`, `outer_max_iters`, `gmres_max_iters`, `gmres_cycles`, and `gmres_tol`. The Q-table update then consumes these measured action metrics through an epsilon-greedy contextual-bandit loop. These settings are recorded in `policy.json` and `summary.csv` as `validation_budget`/`validation_budget_id`. This is required for comparable speedup and accuracy.

For irp3-compatible validation, use `--repeats 3 --outer-max-iters 10 --gmres-tol 1e-4 --gmres-max-iters 0 --gmres-cycles 1`. For a cheaper budgeted RL sweep, use the same values except set `--gmres-max-iters 10` or `30`.


## Low/Medium Training / Cross-Regime Validation

Use this variant when you want the RL policy to train on low- and
medium-condition systems, then evaluate on a cross-regime CPU GMRES-IR benchmark
spanning low, medium, and high condition numbers.  The action space remains exactly
`fp16`/`fp32`/`fp64` for `[uf, ug, u, ur]`, and the native C++ validation budget
is shared across every precision action in a run.

Condition regimes are sampled as log10 condition-number ranges:

- `low`: `10^0`--`10^3`
- `medium`: `10^3`--`10^6`
- `high`: `10^6`--`10^9`

The default split uses 50 training systems and 100 testing systems.
Training cases cycle through `low,medium`; test cases cycle through
`low,medium,high`.

Recommended split native commands for the cluster validation budget discussed
with `irp3`:

```bash
COMMON_ARGS=( \
  --mode native \
  --cxx g++ \
  --weights W1,W2 \
  --repeats 3 \
  --outer-max-iters 10 \
  --gmres-max-iters 0 \
  --gmres-cycles 1 \
  --gmres-tol 1e-4 \
  --backward-tol 1e-14 \
  --train-cond-regimes low,medium \
  --test-cond-regimes low,medium,high \
)

python train_cross_regime_cpu_tol1e6.py "${COMMON_ARGS[@]}" \
  --regenerate-data \
  --rebuild
python train_cross_regime_cpu_tol1e8.py "${COMMON_ARGS[@]}"
python validate_cross_regime_cpu_policies.py "${COMMON_ARGS[@]}"
python plot_cpu_lowcond_results.py \
  --input-dir results_cpu_cross_regime \
  --output-dir figures_cpu_cross_regime \
  --font-size 15
```

The original all-in-one `run_cross_regime_cpu_experiment.py` entry point is
still available and writes the same final outputs.  The split path separates the
two tolerance-specific training passes from CPU test validation; within each
tolerance, native action metrics are collected once and reused for both `W1` and
`W2` policy/reward computations.

Outputs are written under `results_cpu_cross_regime/` and
`figures_cpu_cross_regime/`.  `policy.json` and `summary.csv` record
`train_condition_regimes`, `test_condition_regimes`, `validation_budget`, and
`validation_budget_id` so the cross-regime result is auditable against the
low-only run.  For this runner, `train_condition_regimes` defaults to
`low,medium` and `test_condition_regimes` defaults to `low,medium,high`.




### Alternative run

This will generate identical results.

Configuration and setup:

```bash
COMMON_ARGS=( \
  --mode native \
  --data-dir generated_data/cross_regime_dense_train_low_medium_test \
  --output-dir results_cpu_cross_regime \
  --train-count 50 \
  --test-count 100 \
  --size-min 1000 \
  --size-max 1500 \
  --train-cond-regimes low,medium \
  --test-cond-regimes low,medium,high \
  --episodes 100 \
)
```

Run in two nodes:

Node 1:
```bash
python train_cross_regime_cpu_tol1e6.py "${COMMON_ARGS[@]}"
```


Node 2:
```bash
python train_cross_regime_cpu_tol1e8.py "${COMMON_ARGS[@]}"
```

Validate and visualize on any node:
```bash
python validate_cross_regime_cpu_policies.py "${COMMON_ARGS[@]}"
python plot_cpu_lowcond_results.py \
  --input-dir results_cpu_cross_regime \
  --output-dir figures_cpu_cross_regime
```





## Outputs

`run_cpu_lowcond_experiment.py` writes:

- `generated_data/lowcond_dense/manifest.csv`: train/test case manifest.
- `generated_data/lowcond_dense/condition_features_hager_higham.csv`: cached
  state-condition estimates used by the RL policy; `target_cond` remains the
  controlled generator value used for regime reporting.
- `results_cpu_lowcond/<config>/train_curve.csv`: per-episode reward,
  selected mixed-precision action, discretized state, reward components, target
  condition bucket, estimated state-condition bucket, stage precisions, and
  Q-update loss.
- `results_cpu_lowcond/<config>/action_coverage.csv`: number of Q-table updates
  and valid cases per legal action.
- `results_cpu_lowcond/<config>/policy.json`: learned two-feature Q-table policy
  and metadata, including `train_mode`, `episodes`, `rl_updates`,
  `native_train_action_evaluations`, state bounds, reward model, and the
  acceptance rule.
- `results_cpu_lowcond/<config>/test_results.csv`: per-test-case selected
  mixed-precision action, `uf/ug/u/ur`, runtime, FP64 baseline runtime,
  relative error, backward error, speedup (`FP64/mixed`), memory ratio
  (`mixed/FP64`), target condition bucket, and estimated state-condition bucket.
- `results_cpu_lowcond/summary.csv`: one row per `W/tolerance` configuration.

`plot_cpu_lowcond_results.py` writes:

- `figures_cpu_lowcond/cpu_lowcond_summary.{pdf,png}`.
- `figures_cpu_lowcond/cpu_lowcond_training_curves.{pdf,png}`.
- `figures_cpu_lowcond/cpu_lowcond_test_details.{pdf,png}`.

## Fast Smoke Test

Use small matrices and proxy mode when checking the pipeline on a laptop:

```bash
python run_cpu_lowcond_experiment.py \
  --mode proxy \
  --train-count 4 \
  --test-count 3 \
  --size-min 8 \
  --size-max 12 \
  --data-dir /tmp/rlops_cpu_lowcond_data \
  --output-dir /tmp/rlops_cpu_lowcond_results
python plot_cpu_lowcond_results.py \
  --input-dir /tmp/rlops_cpu_lowcond_results \
  --output-dir /tmp/rlops_cpu_lowcond_figures \
  --font-size 12
```

Use small matrices and native mode to check the compiled C++ evaluator:

```bash
python run_cpu_lowcond_experiment.py \
  --mode native \
  --train-count 2 \
  --test-count 2 \
  --size-min 6 \
  --size-max 8 \
  --repeats 1 \
  --outer-max-iters 10 \
  --gmres-max-iters 30 \
  --gmres-cycles 1 \
  --gmres-tol 1e-4 \
  --backward-tol 1e-14 \
  --data-dir /tmp/rlops_cpu_lowcond_native_data \
  --output-dir /tmp/rlops_cpu_lowcond_native_results \
  --rebuild
```

