#!/usr/bin/env python3
"""Train on low/medium dense regimes and validate across low/medium/high CPU GMRES-IR cases.

This entry point keeps the CPU validation semantics identical to
run_cpu_lowcond_experiment.py while changing only the data split: training cases
cycle through low and medium condition-number regimes; testing cases span low,
medium, and high regimes for native GMRES-IR validation.
"""

from __future__ import annotations

from typing import Iterable

from run_cpu_lowcond_experiment import build_arg_parser, run_experiment, validate_args


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    parser.description = __doc__
    parser.set_defaults(
        data_dir="generated_data/cross_regime_dense_train_low_medium_test",
        output_dir="results_cpu_cross_regime",
        train_cond_regimes="low,medium",
        test_cond_regimes="low,medium,high",
    )
    args = parser.parse_args(argv)
    validate_args(args)
    run_experiment(args)


if __name__ == "__main__":
    main()
