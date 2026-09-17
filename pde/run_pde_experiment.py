import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
SPARSE_TEST_DIR = CURRENT_DIR.parent / "sparse_test"
sys.path.insert(0, str(SPARSE_TEST_DIR))

import pychop  # noqa: E402

if not hasattr(pychop, "LightChop") and hasattr(pychop, "Chop"):
    pychop.LightChop = pychop.Chop

from gmresir_solver1 import GMRESIR5Solver  # noqa: E402
from pde_matrix_utils import (  # noqa: E402
    PDEDatasetConfig,
    generate_pde_datasets,
    load_pde_datasets,
    parse_families,
    resolve_split_size_ranges,
    save_pde_datasets,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


EXPERIMENT_CONFIGS: Dict[str, Dict[str, float]] = {
    "1": {"name": "results_pde1", "tol": 1e-6, "c1": 1.0, "c2": 0.1, "label": "W1, tol=1e-6"},
    "2": {"name": "results_pde2", "tol": 1e-6, "c1": 1.0, "c2": 1.0, "label": "W2, tol=1e-6"},
    "3": {"name": "results_pde3", "tol": 1e-8, "c1": 1.0, "c2": 0.1, "label": "W1, tol=1e-8"},
    "4": {"name": "results_pde4", "tol": 1e-8, "c1": 1.0, "c2": 1.0, "label": "W2, tol=1e-8"},
}


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def metric_frame(rows: List[Dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if "split" in frame.columns:
        frame = frame.drop(columns=["split"])
    return frame


def resolve_configs(configs: Iterable[str]) -> List[str]:
    names = list(configs)
    if not names or "all" in names:
        return list(EXPERIMENT_CONFIGS.keys())
    unknown = [name for name in names if name not in EXPERIMENT_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown configs {unknown}; choose from {sorted(EXPERIMENT_CONFIGS)} or 'all'.")
    return names


def resolve_size_ranges(args) -> None:
    args.train_size_min, args.train_size_max, args.test_size_min, args.test_size_max = resolve_split_size_ranges(
        args.train_size_min,
        args.train_size_max,
        args.test_size_min,
        args.test_size_max,
        args.size_min,
        args.size_max,
    )


def get_or_generate_dataset(args):
    dataset_config = PDEDatasetConfig(
        num_train=args.num_train,
        num_test=args.num_test,
        train_size_min=args.train_size_min,
        train_size_max=args.train_size_max,
        test_size_min=args.test_size_min,
        test_size_max=args.test_size_max,
        seed=args.seed,
        rhs_mode=args.rhs_mode,
        cond_method=args.cond_method,
        dense_cond_limit=args.dense_cond_limit,
        families=parse_families(args.families),
    )

    if not args.regenerate_data:
        loaded = load_pde_datasets(args.num_train, args.num_test, args.data_dir, dataset_config)
        train_data, test_data, train_metrics, test_metrics = loaded
        if train_data is not None and test_data is not None:
            logging.info("Loaded cached PDE dataset from %s", args.data_dir)
            return train_data, test_data, train_metrics, test_metrics

    logging.info("Generating PDE dataset in %s", args.data_dir)
    train_data, test_data = generate_pde_datasets(dataset_config, verbose=not args.quiet_data)
    save_pde_datasets(train_data, test_data, args.data_dir, dataset_config)
    train_metrics = [{"split": "train", "matrix_id": str(i), **metadata} for i, (_, _, _, metadata) in enumerate(train_data)]
    test_metrics = [{"split": "test", "matrix_id": str(i), **metadata} for i, (_, _, _, metadata) in enumerate(test_data)]
    return train_data, test_data, train_metrics, test_metrics


def run_single_config(config_id: str, args, train_data, test_data, train_metrics, test_metrics) -> None:
    exp_config = EXPERIMENT_CONFIGS[config_id]
    results_dir = Path(args.results_root) / exp_config["name"]

    if args.skip_existing and (results_dir / "test_results.csv").exists():
        logging.info("Skipping config %s because %s already has test_results.csv", config_id, results_dir)
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    solver = GMRESIR5Solver(
        max_iter=args.max_iter,
        tol=exp_config["tol"],
        precisions=args.precisions,
        max_inner_iter=args.max_inner_iter,
        max_stagnation=args.max_stagnation,
        c1=exp_config["c1"],
        c2=exp_config["c2"],
    )

    top_k = max(1, args.top_k if args.top_k is not None else args.episodes // 4)
    logging.info("Running PDE config %s (%s), results=%s", config_id, exp_config["label"], results_dir)
    episode_rewards, episode_losses = solver.train(
        train_data,
        episodes=args.episodes,
        top_k=top_k,
        results_dir=str(results_dir),
    )
    results, precision_logs, averages = solver.test(test_data)

    pd.DataFrame(results).to_csv(results_dir / "test_results.csv", index=False)
    metric_frame(train_metrics).to_csv(results_dir / "train_metrics.csv", index=False)
    metric_frame(test_metrics).to_csv(results_dir / "test_metrics.csv", index=False)

    with open(results_dir / "precision_logs.pkl", "wb") as f:
        pickle.dump(precision_logs, f)
    solver.save_model(str(results_dir / "q_tables.pkl"))

    run_config = {
        "config_id": config_id,
        **exp_config,
        "num_train": args.num_train,
        "num_test": args.num_test,
        "train_size_min": args.train_size_min,
        "train_size_max": args.train_size_max,
        "test_size_min": args.test_size_min,
        "test_size_max": args.test_size_max,
        "episodes": args.episodes,
        "top_k": top_k,
        "precisions": args.precisions,
        "data_dir": args.data_dir,
        "rhs_mode": args.rhs_mode,
        "cond_method": args.cond_method,
        "dense_cond_limit": args.dense_cond_limit,
        "seed": args.seed,
        "families": parse_families(args.families),
        "max_iter": args.max_iter,
        "max_inner_iter": args.max_inner_iter,
        "max_stagnation": args.max_stagnation,
    }
    with open(results_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(run_config), f, indent=2)
    with open(results_dir / "averages.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(averages), f, indent=2)

    logging.info("Saved PDE config %s outputs to %s", config_id, results_dir)
    logging.info("Average RL error: %.4e", averages["avg_rl_error"])
    logging.info("Average FP64 error: %.4e", averages["avg_fp64_error"])
    logging.info("Precision usage: %s", averages["sum_precision_usage"])


def build_arg_parser(description: str = "Run GMRES-IR/RL experiments on PDE-generated sparse matrices.") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--configs", nargs="+", default=["all"], help="Experiment config ids: 1 2 3 4, or all.")
    parser.add_argument("--data-dir", default=str(CURRENT_DIR / "data_pde"))
    parser.add_argument("--results-root", default=str(CURRENT_DIR))
    parser.add_argument("--num-train", type=int, default=100)
    parser.add_argument("--num-test", type=int, default=100)
    parser.add_argument("--train-size-min", type=int, default=None)
    parser.add_argument("--train-size-max", type=int, default=None)
    parser.add_argument("--test-size-min", type=int, default=None)
    parser.add_argument("--test-size-max", type=int, default=None)
    parser.add_argument("--size-min", type=int, default=None, help="Legacy option: sets both train and test minimum sizes when split-specific sizes are omitted.")
    parser.add_argument("--size-max", type=int, default=None, help="Legacy option: sets both train and test maximum sizes when split-specific sizes are omitted.")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--rhs-mode", choices=("smooth", "piecewise", "random"), default="smooth")
    parser.add_argument("--cond-method", choices=("auto", "dense", "onenormest"), default="auto")
    parser.add_argument("--dense-cond-limit", type=int, default=700)
    parser.add_argument("--families", default="all", help="Comma-separated PDE family names, or 'all'.")
    parser.add_argument("--regenerate-data", action="store_true")
    parser.add_argument("--quiet-data", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=9999)
    parser.add_argument("--max-inner-iter", type=int, default=1)
    parser.add_argument("--max-stagnation", type=int, default=3)
    parser.add_argument("--precisions", nargs="+", default=["bf16", "tf32", "fp32", "fp64"])
    return parser


def run_configs(default_configs=None, description=None) -> None:
    parser = build_arg_parser(description or "Run GMRES-IR/RL experiments on PDE-generated sparse matrices.")
    if default_configs is not None:
        parser.set_defaults(configs=list(default_configs))
    args = parser.parse_args()
    resolve_size_ranges(args)
    train_data, test_data, train_metrics, test_metrics = get_or_generate_dataset(args)
    for config_id in resolve_configs(args.configs):
        run_single_config(config_id, args, train_data, test_data, train_metrics, test_metrics)


def main() -> None:
    run_configs()


if __name__ == "__main__":
    main()
