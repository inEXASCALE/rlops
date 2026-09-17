#!/usr/bin/env bash
#SBATCH --job-name=cpuval-setup
#SBATCH --output=logs/cpuval-setup-%j.out
#SBATCH --error=logs/cpuval-setup-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

set -euo pipefail

cd "$HOME/rlops/cpu_lowcond_gmres_ir"

module purge
module load python/anaconda3

PYTHON_BIN="${PYTHON_BIN:-/software/python/anaconda3/bin/python}"

echo "python: $PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import sys, numpy
print(sys.executable)
print("numpy", numpy.__version__)
PY

"$PYTHON_BIN" generate_lowcond_dense_systems.py \
  --output-dir generated_data/cross_regime_dense_train_low_medium_test \
  --train-count 50 \
  --test-count 100 \
  --size-min 1000 \
  --size-max 1500 \
  --train-regimes low,medium \
  --test-regimes low,medium,high \
  --overwrite

mkdir -p build
g++ -O3 -std=c++17 -march=native gmres_ir_cpu.cpp -o build/gmres_ir_cpu

echo "Generated manifest:"
wc -l generated_data/cross_regime_dense_train_low_medium_test/manifest.csv
ls -lh build/gmres_ir_cpu
