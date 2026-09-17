#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

python run_cpu_lowcond_experiment.py "$@"
python plot_cpu_lowcond_results.py --input-dir "${OUTPUT_DIR:-results_cpu_lowcond}" --font-size "${FONT_SIZE:-15}"
