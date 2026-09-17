#!/usr/bin/env python3
"""Train/evaluate mixed-precision RL policies on low-condition CPU GMRES-IR cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import struct
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla

PRECISIONS = ("fp16", "fp32", "fp64")
PRECISION_ORDER = {name: index for index, name in enumerate(PRECISIONS)}
PRECISION_SIG_BITS = {"fp16": 10, "fp32": 23, "fp64": 52}
PRECISION_REWARD_WEIGHTS = {
    name: PRECISION_SIG_BITS["fp64"] / sig_bits for name, sig_bits in PRECISION_SIG_BITS.items()
}
STAGES = ("uf", "ug", "u", "ur")
BASELINE_ACTION = ("fp64", "fp64", "fp64", "fp64")
WEIGHTS = {
    "W1": {"c1": 1.0, "c2": 0.1},
    "W2": {"c1": 1.0, "c2": 1.0},
}
DEFAULT_TOLERANCES = ("1e-6", "1e-8")
BUCKETS = tuple(f"1e{idx}-1e{idx + 1}" for idx in range(9)) + ("1e9+",)
BUCKET_INDEX = {bucket: index for index, bucket in enumerate(BUCKETS)}
COND_BINS = 10
NORM_BINS = 10
STATE_SPACE_SIZE = COND_BINS * NORM_BINS
REWARD_EPS = 1.0e-10
CASE_MAGIC = b"RLCPU1\0\0"
CASE_HEADER = struct.Struct("<8sIId")


def build_actions() -> tuple[tuple[str, str, str, str], ...]:
    actions: list[tuple[str, str, str, str]] = []
    for uf in PRECISIONS:
        for ug in PRECISIONS:
            if PRECISION_ORDER[ug] < PRECISION_ORDER[uf]:
                continue
            for u in PRECISIONS:
                if PRECISION_ORDER[u] < PRECISION_ORDER[ug]:
                    continue
                for ur in PRECISIONS:
                    if PRECISION_ORDER[ur] < PRECISION_ORDER[u]:
                        continue
                    actions.append((uf, ug, u, ur))
    return tuple(actions)


ACTIONS = build_actions()


@dataclass(frozen=True)
class Case:
    file: str
    split: str
    n: int
    target_cond: float
    a_norm_inf: float | None = None
    regime: str = ""
    state_cond: float | None = None


@dataclass(frozen=True)
class StateBounds:
    cond_min: float
    cond_max: float
    norm_min: float
    norm_max: float


@dataclass(frozen=True)
class StateInfo:
    index: int
    cond_bin: int
    norm_bin: int
    cond_log10: float
    norm_log10: float
    a_norm_inf: float


@dataclass(frozen=True)
class RewardPayload:
    reward: float
    speedup: float
    memory_ratio: float
    valid: bool
    accuracy_reward: float
    precision_reward: float
    iteration_penalty: float


def action_id(action: tuple[str, str, str, str]) -> str:
    return "_".join(action)


def parse_action(value: str) -> tuple[str, str, str, str]:
    parts = tuple(value.replace(",", "_").split("_"))
    if len(parts) != 4 or any(part not in PRECISION_ORDER for part in parts):
        raise ValueError(f"invalid action: {value}")
    return parts  # type: ignore[return-value]


def optional_float(row: dict[str, str], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            try:
                parsed = float(value)
            except ValueError:
                continue
            if math.isfinite(parsed):
                return parsed
    return None


def read_manifest(path: Path) -> list[Case]:
    with path.open(newline="") as handle:
        return [
            Case(
                file=row["file"],
                split=row["split"],
                n=int(row["n"]),
                target_cond=float(row["target_cond"]),
                a_norm_inf=optional_float(row, "a_norm_inf", "A_norm_inf", "matrix_norm_inf"),
                regime=row.get("regime", ""),
            )
            for row in csv.DictReader(handle)
        ]


def condition_bucket(target_cond: float) -> str:
    if not math.isfinite(target_cond) or target_cond <= 1.0:
        return "1e0-1e1"
    exponent = int(math.floor(math.log10(target_cond)))
    if exponent < 0:
        return "1e0-1e1"
    if exponent >= 9:
        return "1e9+"
    return f"1e{exponent}-1e{exponent + 1}"


def state_condition(case: Case) -> float:
    if case.state_cond is not None and math.isfinite(case.state_cond) and case.state_cond > 0.0:
        return case.state_cond
    return case.target_cond


def state_bucket(case: Case) -> str:
    return condition_bucket(state_condition(case))


def target_bucket(case: Case) -> str:
    return condition_bucket(case.target_cond)


def state_condition_source(args: argparse.Namespace) -> str:
    if args.state_cond_method == "target":
        return "target condition number stored by the controlled generator"
    return "Hager-Higham-type one-norm condition estimate computed from the generated matrix"


def condition_cache_path(args: argparse.Namespace, data_dir: Path) -> Path:
    if args.state_cond_cache:
        path = Path(args.state_cond_cache)
        return path if path.is_absolute() else data_dir / path
    return data_dir / "condition_features_hager_higham.csv"


def read_case_matrix(data_dir: Path, case: Case) -> np.ndarray:
    path = data_dir / case.file
    with path.open("rb") as handle:
        header = handle.read(CASE_HEADER.size)
        if len(header) != CASE_HEADER.size:
            raise SystemExit(f"truncated CPU case header: {path}")
        magic, n, _split_id, stored_target_cond = CASE_HEADER.unpack(header)
        if magic != CASE_MAGIC:
            raise SystemExit(f"invalid CPU case magic in {path}")
        if n != case.n:
            raise SystemExit(f"manifest n={case.n} disagrees with binary n={n} for {path}")
        if not math.isclose(stored_target_cond, case.target_cond, rel_tol=1e-10, abs_tol=1e-12):
            raise SystemExit(f"manifest target_cond disagrees with binary target_cond for {path}")
        values = np.fromfile(handle, dtype="<f8", count=n * n)
    if values.size != n * n:
        raise SystemExit(f"truncated matrix payload in {path}")
    return values.reshape((n, n))


def hager_higham_condition_estimate(A: np.ndarray) -> float:
    A = np.asarray(A, dtype=np.float64, order="F")
    try:
        lu, piv = la.lu_factor(A, check_finite=False)

        def solve(x: np.ndarray) -> np.ndarray:
            return la.lu_solve((lu, piv), x, check_finite=False)

        def solve_transpose(x: np.ndarray) -> np.ndarray:
            return la.lu_solve((lu, piv), x, trans=1, check_finite=False)

        inv_op = spla.LinearOperator(A.shape, matvec=solve, rmatvec=solve_transpose, dtype=np.float64)
        cond = float(np.linalg.norm(A, 1)) * float(spla.onenormest(inv_op))
        return cond if math.isfinite(cond) else float("inf")
    except Exception:
        return float("inf")


def load_state_condition_cache(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            row["file"]: float(row["state_cond"])
            for row in rows
            if row.get("file") and row.get("state_cond")
        }


def write_state_condition_cache(path: Path, cases: list[Case]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fieldnames = [
        "file", "split", "n", "target_cond", "target_bucket", "regime",
        "state_cond", "state_bucket", "a_norm_inf", "state_cond_method",
    ]
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerow({
                "file": case.file,
                "split": case.split,
                "n": case.n,
                "target_cond": f"{case.target_cond:.12e}",
                "target_bucket": target_bucket(case),
                "regime": case.regime,
                "state_cond": f"{state_condition(case):.12e}",
                "state_bucket": state_bucket(case),
                "a_norm_inf": f"{a_norm_for(case):.12e}",
                "state_cond_method": "hager_higham_onenorm",
            })
    tmp.replace(path)


def attach_state_conditions(cases: list[Case], data_dir: Path, args: argparse.Namespace) -> list[Case]:
    if args.state_cond_method == "target":
        return [replace(case, state_cond=case.target_cond) for case in cases]

    cache_path = condition_cache_path(args, data_dir)
    cache = {} if args.regenerate_data else load_state_condition_cache(cache_path)
    updated: list[Case] = []
    cache_changed = False
    missing_count = sum(
        1 for case in cases
        if not (cache.get(case.file) is not None and math.isfinite(cache[case.file]) and cache[case.file] > 0.0)
    )
    computed_count = 0
    if missing_count:
        print(f"Computing Hager-Higham state condition estimates for {missing_count} CPU case(s); cache: {cache_path}")
    for case in cases:
        cached = cache.get(case.file)
        if cached is not None and math.isfinite(cached) and cached > 0.0:
            updated.append(replace(case, state_cond=cached))
            continue
        estimate = hager_higham_condition_estimate(read_case_matrix(data_dir, case))
        updated.append(replace(case, state_cond=estimate))
        cache_changed = True
        computed_count += 1
        if computed_count == missing_count or computed_count % 10 == 0:
            print(f"  computed {computed_count}/{missing_count} state condition estimates")
    if cache_changed or args.regenerate_data or not cache_path.exists():
        write_state_condition_cache(cache_path, updated)
    return updated


def config_label(weight: str, tolerance: str) -> str:
    return f"{weight}_tol{tolerance.replace('-', 'm').replace('+', '')}"


def policy_action_for_bucket(policy: dict[str, str], bucket: str) -> tuple[str, str, str, str]:
    if bucket in policy:
        return parse_action(policy[bucket])
    if not policy:
        return BASELINE_ACTION
    target = BUCKET_INDEX[bucket]
    nearest = min(policy, key=lambda known: abs(BUCKET_INDEX[known] - target))
    return parse_action(policy[nearest])


def log10_feature(value: float) -> float:
    return math.log10(max(value, 1.0))


def metric_float(metrics: dict[str, float | str], name: str, default: float) -> float:
    value = metrics.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def a_norm_for(case: Case, metrics: dict[str, float | str] | None = None) -> float:
    if metrics is not None:
        value = metric_float(metrics, "a_norm_inf", math.nan)
        if math.isfinite(value) and value > 0.0:
            return value
    if case.a_norm_inf is not None and case.a_norm_inf > 0.0:
        return case.a_norm_inf
    return 1.0


def state_features(case: Case, metrics: dict[str, float | str] | None = None) -> tuple[float, float, float]:
    a_norm_inf = a_norm_for(case, metrics)
    return log10_feature(state_condition(case)), log10_feature(a_norm_inf), a_norm_inf


def compute_state_bounds(cases: list[Case], metrics: dict[tuple[str, tuple[str, str, str, str]], dict[str, float | str]]) -> StateBounds:
    cond_logs: list[float] = []
    norm_logs: list[float] = []
    for case in cases:
        baseline = metrics.get((case.file, BASELINE_ACTION))
        cond_log, norm_log, _ = state_features(case, baseline)
        cond_logs.append(cond_log)
        norm_logs.append(norm_log)
    return StateBounds(
        cond_min=min(cond_logs) if cond_logs else 0.0,
        cond_max=max(cond_logs) if cond_logs else 1.0,
        norm_min=min(norm_logs) if norm_logs else 0.0,
        norm_max=max(norm_logs) if norm_logs else 1.0,
    )


def discretize_feature(value: float, lower: float, upper: float, bins: int) -> int:
    feature_range = upper - lower if upper > lower else 1.0
    raw = int((value - lower) / feature_range * (bins - 1))
    return max(0, min(bins - 1, raw))


def discretize_state(case: Case, metrics: dict[str, float | str] | None, bounds: StateBounds) -> StateInfo:
    cond_log, norm_log, a_norm_inf = state_features(case, metrics)
    cond_bin = discretize_feature(cond_log, bounds.cond_min, bounds.cond_max, COND_BINS)
    norm_bin = discretize_feature(norm_log, bounds.norm_min, bounds.norm_max, NORM_BINS)
    return StateInfo(
        index=cond_bin * NORM_BINS + norm_bin,
        cond_bin=cond_bin,
        norm_bin=norm_bin,
        cond_log10=cond_log,
        norm_log10=norm_log,
        a_norm_inf=a_norm_inf,
    )


def state_bounds_to_json(bounds: StateBounds) -> dict[str, float]:
    return {
        "cond_min": bounds.cond_min,
        "cond_max": bounds.cond_max,
        "norm_min": bounds.norm_min,
        "norm_max": bounds.norm_max,
    }


def state_bounds_from_json(payload: dict[str, object]) -> StateBounds:
    raw = payload.get("state_bounds")
    if not isinstance(raw, dict):
        return StateBounds(cond_min=0.0, cond_max=1.0, norm_min=0.0, norm_max=1.0)
    return StateBounds(
        cond_min=float(raw.get("cond_min", 0.0)),
        cond_max=float(raw.get("cond_max", 1.0)),
        norm_min=float(raw.get("norm_min", 0.0)),
        norm_max=float(raw.get("norm_max", 1.0)),
    )


def greedy_action_index(q_values: list[float]) -> int:
    return max(range(len(q_values)), key=lambda index: q_values[index])


def ensure_data(args: argparse.Namespace) -> Path:
    data_dir = Path(args.data_dir)
    manifest = data_dir / "manifest.csv"
    if manifest.exists() and not args.regenerate_data:
        return manifest

    from generate_lowcond_dense_systems import build_arg_parser as build_generator_parser
    from generate_lowcond_dense_systems import generate as generate_data

    gen_parser = build_generator_parser()
    generator_argv = [
        "--output-dir",
        str(data_dir),
        "--train-count",
        str(args.train_count),
        "--test-count",
        str(args.test_count),
        "--size-min",
        str(args.size_min),
        "--size-max",
        str(args.size_max),
        "--cond-log-min",
        str(args.cond_log_min),
        "--cond-log-max",
        str(args.cond_log_max),
        "--seed",
        str(args.seed),
    ]
    if args.train_cond_regimes:
        generator_argv.extend(["--train-regimes", args.train_cond_regimes])
    if args.test_cond_regimes:
        generator_argv.extend(["--test-regimes", args.test_cond_regimes])
    if args.regenerate_data:
        generator_argv.append("--overwrite")
    gen_args = gen_parser.parse_args(generator_argv)
    generate_data(gen_args)
    return manifest


def build_solver(args: argparse.Namespace, root: Path) -> Path | None:
    if args.mode == "proxy":
        return None
    binary = Path(args.binary)
    if not binary.is_absolute():
        binary = root / binary
    if binary.exists() and not args.rebuild:
        return binary
    compiler = shutil.which(args.cxx)
    if compiler is None:
        if args.mode == "native":
            raise SystemExit(f"C++ compiler not found: {args.cxx}")
        print("C++ compiler not found; falling back to proxy evaluator.")
        return None
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = [compiler, "-O3", "-std=c++17", "-march=native", str(root / "gmres_ir_cpu.cpp"), "-o", str(binary)]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        if args.mode == "native":
            raise
        print("Native CPU solver build failed; falling back to proxy evaluator.")
        return None
    return binary




def validation_budget(args: argparse.Namespace) -> dict[str, object]:
    gmres_restart = "full" if args.gmres_max_iters == 0 else args.gmres_max_iters
    return {
        "repeats": args.repeats,
        "outer_max_iters": args.outer_max_iters,
        "gmres_restart": gmres_restart,
        "gmres_max_iters": args.gmres_max_iters,
        "gmres_cycles": args.gmres_cycles,
        "gmres_tol": args.gmres_tol,
    }


def validation_budget_id(args: argparse.Namespace) -> str:
    restart = "full" if args.gmres_max_iters == 0 else str(args.gmres_max_iters)
    return f"r{args.repeats}_outer{args.outer_max_iters}_gmres{restart}_cycles{args.gmres_cycles}_gtol{args.gmres_tol:g}"


def condition_regime_label(args: argparse.Namespace, split: str) -> str:
    regimes = args.train_cond_regimes if split == "train" else args.test_cond_regimes
    if regimes:
        return regimes
    return f"custom_log10_{args.cond_log_min:g}_{args.cond_log_max:g}"


def read_solver_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        return {row["file"]: row for row in csv.DictReader(handle)}


def precision_storage_bytes(precision: str) -> float:
    return {"fp16": 2.0, "fp32": 4.0, "fp64": 8.0}[precision]


def proxy_metrics(case: Case, action: tuple[str, str, str, str], tolerance: float, backward_tolerance: float) -> dict[str, float | str]:
    uf, ug, u, ur = action
    a_norm_inf = a_norm_for(case)
    matrix_bytes = precision_storage_bytes(u) * case.n * case.n
    if ur != u:
        matrix_bytes += precision_storage_bytes(ur) * case.n * case.n
    factor_bytes = precision_storage_bytes(uf) * case.n
    vector_bytes = (precision_storage_bytes(ug) * 4.0 + precision_storage_bytes(u) * 2.0 + precision_storage_bytes(ur) * 2.0) * case.n
    memory_bytes = matrix_bytes + factor_bytes + vector_bytes

    stage_error = {"fp16": 2.0e-7, "fp32": 5.0e-10, "fp64": 1.0e-15}
    rel_error = max(stage_error[p] for p in action) * max(1.0, math.sqrt(case.target_cond))
    fp64_ms = 1.0e-6 * case.n * case.n * (1.0 + 0.02 * math.log10(case.target_cond))
    stage_cost = {
        "uf": {"fp16": 0.45, "fp32": 0.70, "fp64": 1.00}[uf],
        "ug": {"fp16": 0.45, "fp32": 0.65, "fp64": 1.00}[ug],
        "u": {"fp16": 0.55, "fp32": 0.70, "fp64": 1.00}[u],
        "ur": {"fp16": 0.60, "fp32": 0.75, "fp64": 1.00}[ur],
    }
    avg_ms = fp64_ms * (0.20 * stage_cost["uf"] + 0.50 * stage_cost["ug"] + 0.15 * stage_cost["u"] + 0.15 * stage_cost["ur"])
    backward_error = rel_error / max(case.target_cond, 1.0)
    status = "ok" if rel_error <= tolerance else "not_converged"
    gmres_iterations = max(1, int(math.ceil(math.log2(max(case.n, 2)))))
    return {
        "status": status,
        "avg_ms": avg_ms,
        "rel_error": rel_error,
        "backward_error": backward_error,
        "backward_pass": int(backward_error <= backward_tolerance),
        "memory_bytes": memory_bytes,
        "gmres_iterations": gmres_iterations,
        "a_norm_inf": a_norm_inf,
    }


def run_native_action(binary: Path, manifest: Path, data_dir: Path, output_dir: Path, split: str, action: tuple[str, str, str, str], tolerance: float, args: argparse.Namespace) -> dict[str, dict[str, str]]:
    output = output_dir / f"{split}_{action_id(action)}.csv"
    command = [
        str(binary), "--manifest", str(manifest), "--data-dir", str(data_dir), "--output", str(output),
        "--split", split, "--uf", action[0], "--ug", action[1], "--u", action[2], "--ur", action[3],
        "--repeats", str(args.repeats), "--outer-max-iters", str(args.outer_max_iters),
        "--gmres-max-iters", str(args.gmres_max_iters), "--gmres-cycles", str(args.gmres_cycles),
        "--gmres-tol", str(args.gmres_tol), "--tolerance", str(tolerance),
        "--backward-tol", str(args.backward_tol),
    ]
    started = time.monotonic()
    print(f"    launching native solver for {split} action {action_id(action)} -> {output.name}", flush=True)
    subprocess.run(command, check=True)
    elapsed = time.monotonic() - started
    print(f"    finished native solver for {split} action {action_id(action)} in {elapsed / 60.0:.1f} min", flush=True)
    return read_solver_rows(output)


def collect_metrics(cases: list[Case], split: str, tolerance: float, args: argparse.Namespace, binary: Path | None, manifest: Path, output_dir: Path) -> dict[tuple[str, tuple[str, str, str, str]], dict[str, float | str]]:
    selected = [case for case in cases if case.split == split]
    metrics: dict[tuple[str, tuple[str, str, str, str]], dict[str, float | str]] = {}
    if binary is None:
        print(f"Collecting proxy {split} metrics for {len(selected)} case(s), {len(ACTIONS)} action(s), tolerance={tolerance:g}", flush=True)
        for case_index, case in enumerate(selected, start=1):
            if case_index == 1 or case_index == len(selected) or case_index % 10 == 0:
                print(f"  proxy {split}: case {case_index}/{len(selected)}", flush=True)
            for action in ACTIONS:
                metrics[(case.file, action)] = proxy_metrics(case, action, tolerance, args.backward_tol)
        return metrics

    print(f"Collecting native {split} metrics for {len(selected)} case(s), {len(ACTIONS)} action(s), tolerance={tolerance:g}", flush=True)
    for action_index, action in enumerate(ACTIONS, start=1):
        print(f"  native {split}: action {action_index}/{len(ACTIONS)} {action_id(action)}", flush=True)
        rows = run_native_action(binary, manifest, Path(args.data_dir), output_dir, split, action, tolerance, args)
        for case in selected:
            row = rows[case.file]
            a_norm_value = row.get("a_norm_inf")
            if a_norm_value in (None, ""):
                a_norm_value = str(case.a_norm_inf if case.a_norm_inf is not None else 1.0)
            metrics[(case.file, action)] = {
                "status": row["status"],
                "avg_ms": float(row["avg_ms"]),
                "rel_error": float(row["rel_error"]),
                "backward_error": float(row["backward_error"]),
                "backward_pass": int(float(row["backward_error"]) <= args.backward_tol),
                "memory_bytes": float(row["memory_bytes"]),
                "gmres_iterations": float(row.get("gmres_iterations", row.get("outer_iters", "0"))),
                "a_norm_inf": float(a_norm_value),
            }
    return metrics


def reward_for(
    metrics: dict[str, float | str],
    baseline: dict[str, float | str],
    action: tuple[str, str, str, str],
    tolerance: float,
    backward_tolerance: float,
    weight: str,
    condition_number: float,
    lambda_iter: float,
) -> RewardPayload:
    del backward_tolerance  # kept in the signature for backwards-compatible callers and output metadata.
    avg_ms = float(metrics["avg_ms"])
    baseline_ms = max(float(baseline["avg_ms"]), 1e-12)
    speedup = baseline_ms / max(avg_ms, 1e-12)
    memory_ratio = float(metrics["memory_bytes"]) / max(float(baseline["memory_bytes"]), 1e-12)
    rel_error = float(metrics["rel_error"])
    valid = metrics["status"] == "ok" and rel_error <= tolerance
    w = WEIGHTS[weight]
    scaled_error = metric_float(metrics, "backward_error", rel_error)
    if metrics["status"] != "ok" or rel_error > 1.0 or scaled_error > 1.0:
        accuracy_reward = -5.0 * w["c1"]
    else:
        accuracy_reward = -w["c1"] * (
            math.log10(max(rel_error, REWARD_EPS)) + math.log10(max(scaled_error, REWARD_EPS))
        )
    cond_scale = 1.0 + math.log10(max(condition_number, 1.0))
    precision_reward = w["c2"] * sum(PRECISION_REWARD_WEIGHTS[p] / cond_scale for p in action)
    gmres_iterations = metric_float(metrics, "gmres_iterations", 0.0)
    iteration_penalty = math.log2(max(gmres_iterations, 1.0)) if gmres_iterations > 0.0 else 5.0
    reward = accuracy_reward + precision_reward - lambda_iter * iteration_penalty
    return RewardPayload(
        reward=reward,
        speedup=speedup,
        memory_ratio=memory_ratio,
        valid=valid,
        accuracy_reward=accuracy_reward,
        precision_reward=precision_reward,
        iteration_penalty=iteration_penalty,
    )


def train_policy(train_cases: list[Case], metrics: dict[tuple[str, tuple[str, str, str, str]], dict[str, float | str]], tolerance: float, args: argparse.Namespace, weight: str, output_dir: Path) -> dict[str, object]:
    action_space = tuple(reversed(ACTIONS))
    state_bounds = compute_state_bounds(train_cases, metrics)
    q_table = [[0.0 for _ in action_space] for _ in range(STATE_SPACE_SIZE)]
    rng = random.Random(args.policy_seed)
    action_visits: dict[str, dict[str, float]] = defaultdict(lambda: {"train_updates": 0, "train_valid": 0, "reward_sum": 0.0})
    curve_rows: list[dict[str, object]] = []
    if not train_cases:
        raise SystemExit("no training cases available")

    for step in range(1, args.episodes + 1):
        case = train_cases[(step - 1) % len(train_cases)]
        baseline = metrics[(case.file, BASELINE_ACTION)]
        state = discretize_state(case, baseline, state_bounds)
        epsilon = max(args.epsilon_min, 1.0 - (step - 1) / max(args.episodes, 1))
        if rng.random() < epsilon:
            action_index = rng.randrange(len(action_space))
        else:
            action_index = greedy_action_index(q_table[state.index])
        action = action_space[action_index]
        payload = reward_for(
            metrics[(case.file, action)],
            baseline,
            action,
            tolerance,
            args.backward_tol,
            weight,
            state_condition(case),
            args.lambda_iter,
        )
        previous = q_table[state.index][action_index]
        updated = previous + args.alpha * (payload.reward - previous)
        q_table[state.index][action_index] = updated
        aid = action_id(action)
        action_visits[aid]["train_updates"] += 1
        action_visits[aid]["train_valid"] += int(payload.valid)
        action_visits[aid]["reward_sum"] += payload.reward
        curve_rows.append({
            "step": step,
            "file": case.file,
            "target_cond": case.target_cond,
            "target_bucket": target_bucket(case),
            "state_cond": state_condition(case),
            "state_bucket": state_bucket(case),
            "bucket": target_bucket(case),
            "state_idx": state.index,
            "cond_bin": state.cond_bin,
            "norm_bin": state.norm_bin,
            "condition_log10": state.cond_log10,
            "target_condition_log10": log10_feature(case.target_cond),
            "a_norm_log10": state.norm_log10,
            "a_norm_inf": state.a_norm_inf,
            "epsilon": epsilon,
            "action_index": action_index,
            "selected_action": aid,
            "uf": action[0],
            "ug": action[1],
            "u": action[2],
            "ur": action[3],
            "reward": payload.reward,
            "train_loss": abs(updated - previous),
            "accuracy_reward": payload.accuracy_reward,
            "precision_reward": payload.precision_reward,
            "iteration_penalty": payload.iteration_penalty,
            "speedup": payload.speedup,
            "memory_ratio": payload.memory_ratio,
            "valid": int(payload.valid),
        })

    policy = {
        str(state_idx): action_id(action_space[greedy_action_index(q_values)])
        for state_idx, q_values in enumerate(q_table)
    }

    with (output_dir / "train_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    with (output_dir / "action_coverage.csv").open("w", newline="") as handle:
        fieldnames = ["action", "uf", "ug", "u", "ur", "train_updates", "train_valid", "mean_train_reward"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for action in action_space:
            aid = action_id(action)
            visits = action_visits[aid]
            updates = int(visits["train_updates"])
            writer.writerow({
                "action": aid, "uf": action[0], "ug": action[1], "u": action[2], "ur": action[3],
                "train_updates": updates, "train_valid": int(visits["train_valid"]),
                "mean_train_reward": visits["reward_sum"] / updates if updates else math.nan,
            })
    payload: dict[str, object] = {
        "algorithm": "epsilon_greedy_contextual_bandit_q_table",
        "train_mode": "epsilon_greedy_q_update_over_precomputed_fixed_budget_action_metrics",
        "episodes": args.episodes,
        "rl_updates": args.episodes,
        "native_train_action_evaluations": len(train_cases) * len(ACTIONS),
        "stages": STAGES,
        "actions": [action_id(action) for action in action_space],
        "constraints": "uf <= ug <= u <= ur in fp16 < fp32 < fp64 order",
        "state": "log10(state condition estimate) and log10(infinity-norm(A)) binned into a 10x10 table",
        "state_condition_source": state_condition_source(args),
        "target_condition_usage": "target_cond is used only to generate controlled regimes and for target-regime reporting/statistics",
        "state_bins": {"condition": COND_BINS, "matrix_norm": NORM_BINS},
        "state_index": "cond_bin * norm_bins + norm_bin",
        "state_bounds": state_bounds_to_json(state_bounds),
        "reward": "PDE-style accuracy reward plus condition-scaled precision reward minus logarithmic GMRES-iteration penalty",
        "alpha": args.alpha,
        "epsilon_min": args.epsilon_min,
        "lambda_iter": args.lambda_iter,
        "policy_seed": args.policy_seed,
        "acceptance_rule": "status == ok and rel_error <= tolerance; backward_error is diagnostic only",
        "validation_budget": validation_budget(args),
        "validation_budget_id": validation_budget_id(args),
        "fixed_budget_invariant": "Every legal action is evaluated with this same validation budget within a run before the Q-table updates consume the measured metrics.",
        "weight": weight,
        "tolerance": tolerance,
        "backward_tolerance_diagnostic": args.backward_tol,
        "train_condition_regimes": condition_regime_label(args, "train"),
        "test_condition_regimes": condition_regime_label(args, "test"),
        "policy": policy,
        "q_table": q_table,
    }
    with (output_dir / "policy.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def evaluate_policy(test_cases: list[Case], metrics: dict[tuple[str, tuple[str, str, str, str]], dict[str, float | str]], policy_payload: dict[str, object], tolerance: float, args: argparse.Namespace, weight: str, output_dir: Path) -> dict[str, object]:
    policy = policy_payload.get("policy")
    if not isinstance(policy, dict):
        raise SystemExit("policy payload does not contain a valid policy map")
    state_bounds = state_bounds_from_json(policy_payload)
    rows: list[dict[str, object]] = []
    valid_speedups: list[float] = []
    memory_ratios: list[float] = []
    for case in test_cases:
        baseline = metrics[(case.file, BASELINE_ACTION)]
        state = discretize_state(case, baseline, state_bounds)
        action = parse_action(str(policy.get(str(state.index), action_id(BASELINE_ACTION))))
        selected = metrics[(case.file, action)]
        payload = reward_for(selected, baseline, action, tolerance, args.backward_tol, weight, state_condition(case), args.lambda_iter)
        rows.append({
            "file": case.file, "n": case.n, "target_cond": case.target_cond, "target_bucket": target_bucket(case),
            "state_cond": state_condition(case), "state_bucket": state_bucket(case), "bucket": target_bucket(case),
            "regime": case.regime,
            "state_idx": state.index, "cond_bin": state.cond_bin, "norm_bin": state.norm_bin,
            "condition_log10": state.cond_log10, "target_condition_log10": log10_feature(case.target_cond),
            "a_norm_log10": state.norm_log10, "a_norm_inf": state.a_norm_inf,
            "selected_action": action_id(action), "uf": action[0], "ug": action[1], "u": action[2], "ur": action[3],
            "tolerance": tolerance, "status": selected["status"], "valid": int(payload.valid),
            "rel_error": selected["rel_error"], "backward_error": selected["backward_error"],
            "backward_pass_diagnostic": int(selected.get("backward_pass", int(float(selected["backward_error"]) <= args.backward_tol))),
            "rl_avg_ms": selected["avg_ms"], "fp64_avg_ms": baseline["avg_ms"],
            "speedup_vs_fp64": payload.speedup, "memory_ratio_vs_fp64": payload.memory_ratio,
            "accuracy_reward": payload.accuracy_reward, "precision_reward": payload.precision_reward,
            "iteration_penalty": payload.iteration_penalty, "reward": payload.reward,
        })
        if payload.valid:
            valid_speedups.append(payload.speedup)
        memory_ratios.append(payload.memory_ratio)

    with (output_dir / "test_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return {"test_cases": len(rows), "valid_cases": sum(int(row["valid"]) for row in rows), "mean_valid_speedup": mean(valid_speedups) if valid_speedups else math.nan, "mean_memory_ratio": mean(memory_ratios) if memory_ratios else math.nan}


def validate_args(args: argparse.Namespace) -> None:
    if args.repeats <= 0 or args.outer_max_iters <= 0 or args.gmres_cycles <= 0 or args.gmres_max_iters < 0:
        raise SystemExit("repeats, outer-max-iters, and gmres-cycles must be positive; gmres-max-iters must be non-negative, with 0 meaning full GMRES")
    if args.train_count <= 0 or args.test_count <= 0:
        raise SystemExit("train-count and test-count must be positive")
    if args.episodes <= 0:
        raise SystemExit("episodes must be positive")
    if not (0.0 < args.alpha <= 1.0):
        raise SystemExit("alpha must be in (0, 1]")
    if not (0.0 <= args.epsilon_min <= 1.0):
        raise SystemExit("epsilon-min must be in [0, 1]")
    if args.lambda_iter < 0.0:
        raise SystemExit("lambda-iter must be non-negative")
    if args.size_min <= 0 or args.size_max < args.size_min:
        raise SystemExit("invalid matrix size range")
    if args.cond_log_min < 0 or args.cond_log_max < args.cond_log_min:
        raise SystemExit("invalid condition-number exponent range")


def run_experiment(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    args.data_dir = str(data_dir)
    manifest = ensure_data(args)
    cases = attach_state_conditions(read_manifest(manifest), data_dir, args)
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    binary = build_solver(args, root)
    evaluator = "native" if binary is not None else "proxy"

    tolerances = [float(item) for item in args.tolerances.split(",")]
    summary_rows: list[dict[str, object]] = []
    for weight in args.weights.split(","):
        weight = weight.strip()
        if weight not in WEIGHTS:
            raise SystemExit(f"unknown weight profile: {weight}")
        for tolerance in tolerances:
            label = config_label(weight, f"{tolerance:g}")
            cfg_dir = output_root / label
            cfg_dir.mkdir(parents=True, exist_ok=True)
            for stale in cfg_dir.glob("train_*.csv"):
                if stale.name != "train_curve.csv":
                    stale.unlink()
            for stale in cfg_dir.glob("test_*.csv"):
                if stale.name != "test_results.csv":
                    stale.unlink()
            train_metrics = collect_metrics(cases, "train", tolerance, args, binary, manifest, cfg_dir)
            test_metrics = collect_metrics(cases, "test", tolerance, args, binary, manifest, cfg_dir)
            train_cases = [case for case in cases if case.split == "train"]
            test_cases = [case for case in cases if case.split == "test"]
            policy_payload = train_policy(train_cases, train_metrics, tolerance, args, weight, cfg_dir)
            test_summary = evaluate_policy(test_cases, test_metrics, policy_payload, tolerance, args, weight, cfg_dir)
            summary_rows.append({
                "config": label,
                "weight": weight,
                "tolerance": tolerance,
                "acceptance_rule": "rel_error",
                "backward_tolerance_diagnostic": args.backward_tol,
                "train_condition_regimes": condition_regime_label(args, "train"),
                "test_condition_regimes": condition_regime_label(args, "test"),
                "validation_budget_id": validation_budget_id(args),
                "validation_budget": json.dumps(validation_budget(args), sort_keys=True),
                "evaluator": evaluator,
                "state_condition_source": state_condition_source(args),
                "policy": json.dumps(policy_payload["policy"], sort_keys=True),
                "rl_model": policy_payload["algorithm"],
                "state_model": policy_payload["state"],
                "reward_model": policy_payload["reward"],
                **test_summary,
            })

    with (output_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote CPU low-condition experiment outputs to {output_root} using {evaluator} evaluator")


def parse_weights(value: str) -> list[str]:
    weights = [item.strip() for item in value.split(",") if item.strip()]
    for weight in weights:
        if weight not in WEIGHTS:
            raise SystemExit(f"unknown weight profile: {weight}")
    if not weights:
        raise SystemExit("at least one weight profile is required")
    return weights


def parse_tolerances(value: str) -> list[float]:
    tolerances = [float(item) for item in value.split(",") if item.strip()]
    if not tolerances:
        raise SystemExit("at least one tolerance is required")
    return tolerances


def prepare_run(args: argparse.Namespace) -> tuple[Path, list[Case], Path, Path | None, str]:
    root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    args.data_dir = str(data_dir)
    manifest = ensure_data(args)
    cases = attach_state_conditions(read_manifest(manifest), data_dir, args)
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    binary = build_solver(args, root)
    evaluator = "native" if binary is not None else "proxy"
    return manifest, cases, output_root, binary, evaluator


def clean_split_metrics(cfg_dir: Path, split: str) -> None:
    keep = "train_curve.csv" if split == "train" else "test_results.csv"
    for stale in cfg_dir.glob(f"{split}_*.csv"):
        if stale.name != keep:
            stale.unlink()


def copy_split_metrics(source_dir: Path, target_dir: Path, split: str) -> None:
    if source_dir == target_dir:
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.glob(f"{split}_*.csv"):
        if source.name in {"train_curve.csv", "test_results.csv"}:
            continue
        shutil.copy2(source, target_dir / source.name)


def run_training(args: argparse.Namespace) -> None:
    manifest, cases, output_root, binary, evaluator = prepare_run(args)
    weights = parse_weights(args.weights)
    tolerances = parse_tolerances(args.tolerances)
    train_cases = [case for case in cases if case.split == "train"]

    for tolerance in tolerances:
        cfg_dirs: dict[str, Path] = {}
        for weight in weights:
            cfg_dir = output_root / config_label(weight, f"{tolerance:g}")
            cfg_dir.mkdir(parents=True, exist_ok=True)
            clean_split_metrics(cfg_dir, "train")
            cfg_dirs[weight] = cfg_dir

        metric_dir = cfg_dirs[weights[0]]
        train_metrics = collect_metrics(cases, "train", tolerance, args, binary, manifest, metric_dir)
        for weight in weights[1:]:
            copy_split_metrics(metric_dir, cfg_dirs[weight], "train")
        for weight in weights:
            train_policy(train_cases, train_metrics, tolerance, args, weight, cfg_dirs[weight])

    print(f"Wrote CPU low-condition training outputs to {output_root} using {evaluator} evaluator")


def read_policy(path: Path) -> dict[str, object]:
    with path.open() as handle:
        payload = json.load(handle)
    policy = payload.get("policy")
    if not isinstance(policy, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in policy.items()):
        raise SystemExit(f"invalid or missing policy in {path}")
    if payload.get("algorithm") != "epsilon_greedy_contextual_bandit_q_table" or "state_bounds" not in payload:
        raise SystemExit(f"{path} was produced by the old CPU policy model; rerun training before validation")
    return payload


def run_policy_validation(args: argparse.Namespace) -> None:
    manifest, cases, output_root, binary, evaluator = prepare_run(args)
    weights = parse_weights(args.weights)
    tolerances = parse_tolerances(args.tolerances)
    test_cases = [case for case in cases if case.split == "test"]
    summary_by_config: dict[tuple[str, float], dict[str, object]] = {}

    for tolerance in tolerances:
        cfg_dirs: dict[str, Path] = {}
        policies: dict[str, dict[str, object]] = {}
        for weight in weights:
            cfg_dir = output_root / config_label(weight, f"{tolerance:g}")
            policy_path = cfg_dir / "policy.json"
            if not policy_path.exists():
                raise SystemExit(f"missing trained policy: {policy_path}")
            cfg_dir.mkdir(parents=True, exist_ok=True)
            clean_split_metrics(cfg_dir, "test")
            cfg_dirs[weight] = cfg_dir
            policies[weight] = read_policy(policy_path)

        metric_dir = cfg_dirs[weights[0]]
        test_metrics = collect_metrics(cases, "test", tolerance, args, binary, manifest, metric_dir)
        for weight in weights[1:]:
            copy_split_metrics(metric_dir, cfg_dirs[weight], "test")
        for weight in weights:
            policy_payload = policies[weight]
            test_summary = evaluate_policy(test_cases, test_metrics, policy_payload, tolerance, args, weight, cfg_dirs[weight])
            summary_by_config[(weight, tolerance)] = {
                "config": config_label(weight, f"{tolerance:g}"),
                "weight": weight,
                "tolerance": tolerance,
                "acceptance_rule": "rel_error",
                "backward_tolerance_diagnostic": args.backward_tol,
                "train_condition_regimes": condition_regime_label(args, "train"),
                "test_condition_regimes": condition_regime_label(args, "test"),
                "validation_budget_id": validation_budget_id(args),
                "validation_budget": json.dumps(validation_budget(args), sort_keys=True),
                "evaluator": evaluator,
                "state_condition_source": state_condition_source(args),
                "policy": json.dumps(policy_payload["policy"], sort_keys=True),
                "rl_model": policy_payload.get("algorithm", ""),
                "state_model": policy_payload.get("state", ""),
                "reward_model": policy_payload.get("reward", ""),
                **test_summary,
            }

    summary_rows = [summary_by_config[(weight, tolerance)] for weight in weights for tolerance in tolerances]
    with (output_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote CPU low-condition validation outputs to {output_root} using {evaluator} evaluator")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="generated_data/lowcond_dense")
    parser.add_argument("--output-dir", default="results_cpu_lowcond")
    parser.add_argument("--mode", choices=("auto", "native", "proxy"), default="auto")
    parser.add_argument("--binary", default="build/gmres_ir_cpu")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--regenerate-data", action="store_true")
    parser.add_argument("--weights", default="W1,W2")
    parser.add_argument("--tolerances", default=",".join(DEFAULT_TOLERANCES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--outer-max-iters", type=int, default=10)
    parser.add_argument("--gmres-max-iters", type=int, default=30, help="Restart dimension / maximum Arnoldi steps per GMRES cycle; 0 means full GMRES, matching irp3 restart=0.")
    parser.add_argument("--gmres-cycles", type=int, default=1, help="Maximum restarted GMRES cycles per outer refinement iteration.")
    parser.add_argument("--gmres-tol", type=float, default=1e-4, help="Relative residual threshold for the preconditioned inner GMRES solve.")
    parser.add_argument("--backward-tol", type=float, default=1e-14, help="Diagnostic-only backward-error threshold; validity/reward use the user tolerance on relative forward error.")
    parser.add_argument("--train-count", type=int, default=50)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epsilon-min", type=float, default=0.1)
    parser.add_argument("--lambda-iter", type=float, default=0.5)
    parser.add_argument("--policy-seed", type=int, default=42)
    parser.add_argument("--size-min", type=int, default=1000)
    parser.add_argument("--size-max", type=int, default=1500)
    parser.add_argument("--cond-log-min", type=float, default=0.0)
    parser.add_argument("--cond-log-max", type=float, default=3.0)
    parser.add_argument("--train-cond-regimes", default="", help="Comma-separated training condition regimes passed to the data generator: low,medium,high.")
    parser.add_argument("--test-cond-regimes", default="", help="Comma-separated testing condition regimes passed to the data generator: low,medium,high.")
    parser.add_argument("--state-cond-method", choices=("hager-higham", "target"), default="hager-higham", help="Condition feature used for the RL state. The default estimates a one-norm condition number from each generated matrix; 'target' uses the controlled generator metadata.")
    parser.add_argument("--state-cond-cache", default="", help="CSV cache for Hager-Higham state condition estimates. Relative paths are resolved under --data-dir; the default is condition_features_hager_higham.csv in the data directory.")
    parser.add_argument("--seed", type=int, default=20260912)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    validate_args(args)
    run_experiment(args)


if __name__ == "__main__":
    main()
