#!/bin/bash
#SBATCH --job-name=rlops_pde
#SBATCH --partition=convergence
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=100:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module purge
module load python/anaconda3

eval "$(conda shell.bash hook)"
if [[ -n "${CONDA_ENV:-}" ]]; then
    if ! conda activate "${CONDA_ENV}"; then
        echo "Failed to activate CONDA_ENV=${CONDA_ENV}." >&2
        echo "Unset CONDA_ENV or pass PYTHON_BIN=/path/to/python if using a venv/module Python." >&2
        exit 2
    fi
fi

if [[ -n "${RLOPS_PDE_DIR:-}" ]]; then
    SCRIPT_DIR="${RLOPS_PDE_DIR}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

cd "${SCRIPT_DIR}"

if [[ ! -f "run_pde_experiment.py" ]]; then
    echo "Cannot find run_pde_experiment.py in ${SCRIPT_DIR}" >&2
    echo "Submit this job from rlops/papers/pde or set RLOPS_PDE_DIR=/path/to/rlops/papers/pde." >&2
    exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"

: "${PYTHON_BIN:=python}"
: "${PYTHON_USER_SITE:=}"
: "${RUN_W1:=1}"
: "${RUN_W2:=1}"
: "${NUM_TRAIN:=100}"
: "${NUM_TEST:=100}"
: "${TRAIN_SIZE_MIN:=${SIZE_MIN:-100}}"
: "${TRAIN_SIZE_MAX:=${SIZE_MAX:-500}}"
: "${TEST_SIZE_MIN:=${SIZE_MIN:-100}}"
: "${TEST_SIZE_MAX:=${SIZE_MAX:-500}}"
: "${EPISODES:=100}"
: "${TOP_K:=}"
: "${DATA_DIR:=${SCRIPT_DIR}/data_pde}"
: "${RESULTS_ROOT:=${SCRIPT_DIR}}"
: "${RHS_MODE:=smooth}"
: "${COND_METHOD:=auto}"
: "${SEED:=3407}"
: "${FONT_SIZE:=15}"
: "${RUN_VISUALIZATION:=1}"
: "${REGENERATE_DATA:=0}"

if ! PYTHON_BIN_RESOLVED="$(command -v "${PYTHON_BIN}" 2>/dev/null)"; then
    if [[ -x "${PYTHON_BIN}" ]]; then
        PYTHON_BIN_RESOLVED="${PYTHON_BIN}"
    else
        echo "Cannot find PYTHON_BIN=${PYTHON_BIN}" >&2
        exit 2
    fi
fi
PYTHON_BIN="${PYTHON_BIN_RESOLVED}"

unset PYTHONNOUSERSITE
if [[ -z "${PYTHON_USER_SITE}" ]]; then
    PYTHON_USER_SITE="$("${PYTHON_BIN}" -m site --user-site 2>/dev/null || true)"
fi
if [[ -n "${PYTHON_USER_SITE}" && -d "${PYTHON_USER_SITE}" ]]; then
    export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${PYTHON_USER_SITE}"
fi
export PYTHON_BIN

echo "Job ${SLURM_JOB_ID:-local} running on ${SLURM_JOB_NODELIST:-local}"
echo "Node: $(hostname)"
echo "Working directory: ${SCRIPT_DIR}"
echo "Python executable: ${PYTHON_BIN}"
echo "Python: $(${PYTHON_BIN} --version)"
echo "PYTHONPATH: ${PYTHONPATH:-<unset>}"
echo "Run W1: ${RUN_W1}"
echo "Run W2: ${RUN_W2}"
echo "Train size range: ${TRAIN_SIZE_MIN}-${TRAIN_SIZE_MAX}"
echo "Test size range: ${TEST_SIZE_MIN}-${TEST_SIZE_MAX}"
module list

echo "Running Python smoke test through srun..."
srun --ntasks=1 --cpus-per-task=1 \
    --export=ALL,PYTHON_BIN="${PYTHON_BIN}",PYTHONPATH="${PYTHONPATH:-}",PYTHONNOUSERSITE= \
    "${PYTHON_BIN}" - <<'PY'
import sys
import numpy
import pandas
import scipy
import torch
import pychop

print("Python smoke test: ok")
print("sys.executable:", sys.executable)
print("torch:", getattr(torch, "__version__", None), getattr(torch, "__file__", None))
print("torch.set_printoptions:", hasattr(torch, "set_printoptions"))
print("pychop:", getattr(pychop, "__file__", None))
if not hasattr(torch, "set_printoptions"):
    raise RuntimeError(
        "Imported module named 'torch' is not a usable PyTorch install; "
        f"torch.__file__={getattr(torch, '__file__', None)!r}"
    )
PY

COMMON_ARGS=(
    --num-train "${NUM_TRAIN}"
    --num-test "${NUM_TEST}"
    --train-size-min "${TRAIN_SIZE_MIN}"
    --train-size-max "${TRAIN_SIZE_MAX}"
    --test-size-min "${TEST_SIZE_MIN}"
    --test-size-max "${TEST_SIZE_MAX}"
    --episodes "${EPISODES}"
    --data-dir "${DATA_DIR}"
    --results-root "${RESULTS_ROOT}"
    --rhs-mode "${RHS_MODE}"
    --cond-method "${COND_METHOD}"
    --seed "${SEED}"
)

if [[ -n "${TOP_K}" ]]; then
    COMMON_ARGS+=(--top-k "${TOP_K}")
fi

if [[ "${REGENERATE_DATA}" == "1" ]]; then
    COMMON_ARGS+=(--regenerate-data)
fi

if [[ "${RUN_W1}" == "1" ]]; then
    srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-12}" \
        --export=ALL,PYTHON_BIN="${PYTHON_BIN}",PYTHONPATH="${PYTHONPATH:-}",PYTHONNOUSERSITE= \
        "${PYTHON_BIN}" run_pde_w1_experiment.py "${COMMON_ARGS[@]}"
fi

if [[ "${RUN_W2}" == "1" ]]; then
    srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-12}" \
        --export=ALL,PYTHON_BIN="${PYTHON_BIN}",PYTHONPATH="${PYTHONPATH:-}",PYTHONNOUSERSITE= \
        "${PYTHON_BIN}" run_pde_w2_experiment.py "${COMMON_ARGS[@]}"
fi

if [[ "${RUN_VISUALIZATION}" == "1" ]]; then
    srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-12}" \
        --export=ALL,PYTHON_BIN="${PYTHON_BIN}",PYTHONPATH="${PYTHONPATH:-}",PYTHONNOUSERSITE= \
        "${PYTHON_BIN}" visualize_pde_results.py \
        --results-root "${RESULTS_ROOT}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${SCRIPT_DIR}/figures_pde" \
        --font-size "${FONT_SIZE}"
fi
