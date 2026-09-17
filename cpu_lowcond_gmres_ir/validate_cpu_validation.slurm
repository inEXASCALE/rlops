#!/usr/bin/env bash
#SBATCH --job-name=cpuval-validate
#SBATCH --output=logs/cpuval-validate-%j.out
#SBATCH --error=logs/cpuval-validate-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=48:00:00

set -euo pipefail

cd "$HOME/rlops/cpu_lowcond_gmres_ir"

module purge
module load python/anaconda3

PYTHON_BIN="${PYTHON_BIN:-/software/python/anaconda3/bin/python}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

COMMON_ARGS=(
  --mode native
  --binary build/gmres_ir_cpu
  --cxx g++
  --weights W1,W2
  --repeats 3
  --outer-max-iters 10
  --gmres-max-iters 0
  --gmres-cycles 1
  --gmres-tol 1e-4
  --backward-tol 1e-14
  --episodes 100
  --train-cond-regimes low,medium
  --test-cond-regimes low,medium,high
)

"$PYTHON_BIN" validate_cross_regime_cpu_policies.py "${COMMON_ARGS[@]}"

"$PYTHON_BIN" plot_cpu_lowcond_results.py \
  --input-dir results_cpu_cross_regime \
  --output-dir figures_cpu_cross_regime \
  --font-size 15

cat results_cpu_cross_regime/summary.csv
