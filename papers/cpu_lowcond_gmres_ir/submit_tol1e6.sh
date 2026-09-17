#!/bin/bash
#SBATCH --job-name=cpuval-tol1e6
#SBATCH --output=logs/cpuval-tol1e6-%j.out
#SBATCH --error=logs/cpuval-tol1e6-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00

mkdir -p logs

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
)

python train_cross_regime_cpu_tol1e6.py "${COMMON_ARGS[@]}"
