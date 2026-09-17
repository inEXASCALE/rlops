#!/usr/bin/env python3
"""Train cross-regime CPU GMRES-IR policies for tolerance 1e-6."""

from __future__ import annotations

from typing import Iterable

from run_cpu_lowcond_experiment import build_arg_parser, run_training, validate_args


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    parser.description = __doc__
    parser.set_defaults(
        data_dir="generated_data/cross_regime_dense_train_low_medium_test",
        output_dir="results_cpu_cross_regime",
        train_cond_regimes="low,medium",
        test_cond_regimes="low,medium,high",
        tolerances="1e-6",
    )
    args = parser.parse_args(argv)
    validate_args(args)
    run_training(args)


if __name__ == "__main__":
    main()
