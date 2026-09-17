#!/usr/bin/env bash
#SBATCH --job-name=cpuval-train
#SBATCH --output=logs/cpuval-train-%x-%j.out
#SBATCH --error=logs/cpuval-train-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=48:00:00

set -euo pipefail

cd "$HOME/rlops/papers/cpu_lowcond_gmres_ir"

module purge
module load python/anaconda3

PYTHON_BIN="${PYTHON_BIN:-/software/python/anaconda3/bin/python}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

TOL="${TOL:?set TOL to 1e6 or 1e8}"
WEIGHT="${WEIGHT:?set WEIGHT to W1 or W2}"

COMMON_ARGS=(
  --mode native
  --binary build/gmres_ir_cpu
  --cxx g++
  --weights "$WEIGHT"
  --repeats 3
  --outer-max-iters 10
  --gmres-max-iters 0
  --gmres-cycles 1
  --gmres-tol 1e-4
  --backward-tol 1e-14
  --episodes 100
)

echo "Running tolerance=$TOL weight=$WEIGHT"

if [[ "$TOL" == "1e6" ]]; then
  "$PYTHON_BIN" train_cross_regime_cpu_tol1e6.py "${COMMON_ARGS[@]}"
elif [[ "$TOL" == "1e8" ]]; then
  "$PYTHON_BIN" train_cross_regime_cpu_tol1e8.py "${COMMON_ARGS[@]}"
else
  echo "Unknown TOL=$TOL; use 1e6 or 1e8" >&2
  exit 1
fi
