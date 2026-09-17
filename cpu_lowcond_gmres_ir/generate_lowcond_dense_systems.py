#!/usr/bin/env python3
"""Generate low-condition dense linear systems for CPU GMRES-IR experiments.

The default regime follows the low-condition setting requested for RLPT/RLOps
experiments: condition numbers sampled in [10^0, 10^3], sizes sampled in
[1000, 1500], 50 train cases, and 100 test cases.  Matrix data are generated on
demand and intentionally ignored by git.

Binary case format, little endian:
  8 bytes  magic = b"RLCPU1\0\0"
  uint32   n
  uint32   split_id: 0=train, 1=test
  float64  target_cond
  float64  A[n,n], row-major
  float64  b[n]
  float64  x_true[n]
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


MAGIC = b"RLCPU1\0\0"
SPLIT_IDS = {"train": 0, "test": 1}
CONDITION_REGIMES = {
    "low": (0.0, 3.0),
    "medium": (3.0, 6.0),
    "high": (6.0, 9.0),
}


@dataclass(frozen=True)
class Case:
    split: str
    index: int
    n: int
    target_cond: float
    seed: int
    regime: str

    @property
    def name(self) -> str:
        return f"{self.split}_{self.index:03d}_n{self.n}_k{self.target_cond:.2e}.bin"


def _write_case(path: Path, split: str, A: np.ndarray, b: np.ndarray, x_true: np.ndarray, target_cond: float) -> None:
    n = A.shape[0]
    with path.open("wb") as handle:
        handle.write(struct.pack("<8sIId", MAGIC, n, SPLIT_IDS[split], target_cond))
        np.asarray(A, dtype="<f8", order="C").tofile(handle)
        np.asarray(b, dtype="<f8").tofile(handle)
        np.asarray(x_true, dtype="<f8").tofile(handle)


def _read_matrix_norm_inf(path: Path) -> float:
    with path.open("rb") as handle:
        magic, n, _, _ = struct.unpack("<8sIId", handle.read(struct.calcsize("<8sIId")))
        if magic != MAGIC:
            raise ValueError(f"invalid case file magic: {path}")
        A = np.fromfile(handle, dtype="<f8", count=n * n).reshape((n, n))
    return float(np.linalg.norm(A, ord=np.inf))


def _dense_diagonal_dominant_system(case: Case, perturbation_rank: int, noise_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a dense, controlled-condition matrix without an O(n^3) QR step.

    The base diagonal has the requested spectral spread.  A small dense low-rank
    perturbation makes every row dense while keeping the low-condition regime
    stable enough for the experiment.  This is deliberately practical for
    n=1000--1500 and 150 cases.
    """

    rng = np.random.default_rng(case.seed)
    n = case.n
    diagonal = np.geomspace(1.0, 1.0 / case.target_cond, n)
    if case.index % 2:
        diagonal = diagonal[::-1].copy()
    A = np.diag(diagonal)

    rank = max(1, min(perturbation_rank, n))
    u = rng.standard_normal((n, rank))
    v = rng.standard_normal((n, rank))
    perturb = (u @ v.T) / math.sqrt(rank * n)
    # Keep the perturbation safely below the smallest diagonal scale.  The
    # matrix is dense, but its condition remains in the intended low regime.
    eps = noise_scale / max(case.target_cond, 1.0)
    A = A + eps * perturb

    x_true = rng.standard_normal(n)
    b = A @ x_true
    return A, b, x_true


def parse_regimes(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    regimes = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    unknown = sorted(set(regimes) - set(CONDITION_REGIMES))
    if unknown:
        raise SystemExit(f"unknown condition regime(s): {', '.join(unknown)}")
    return regimes


def build_cases(args: argparse.Namespace) -> list[Case]:
    rng = np.random.default_rng(args.seed)
    cases: list[Case] = []
    split_regimes = {
        "train": parse_regimes(args.train_regimes),
        "test": parse_regimes(args.test_regimes),
    }
    for split, count in (("train", args.train_count), ("test", args.test_count)):
        for index in range(count):
            n = int(rng.integers(args.size_min, args.size_max + 1))
            regimes = split_regimes[split]
            if regimes:
                regime = regimes[index % len(regimes)]
                lo, hi = CONDITION_REGIMES[regime]
            else:
                regime = "custom"
                lo, hi = args.cond_log_min, args.cond_log_max
            exponent = float(rng.uniform(lo, hi))
            target_cond = 10.0**exponent
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            cases.append(Case(split=split, index=index, n=n, target_cond=target_cond, seed=seed, regime=regime))
    return cases


def generate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for case in build_cases(args):
        path = output_dir / case.name
        if not path.exists() or args.overwrite:
            A, b, x_true = _dense_diagonal_dominant_system(
                case,
                perturbation_rank=args.perturbation_rank,
                noise_scale=args.noise_scale,
            )
            a_norm_inf = float(np.linalg.norm(A, ord=np.inf))
            _write_case(path, case.split, A, b, x_true, case.target_cond)
        else:
            a_norm_inf = _read_matrix_norm_inf(path)

        rows.append(
            {
                "file": case.name,
                "split": case.split,
                "n": case.n,
                "target_cond": f"{case.target_cond:.12e}",
                "a_norm_inf": f"{a_norm_inf:.12e}",
                "regime": case.regime,
                "generator": "diagonal_low_rank_dense",
                "seed": case.seed,
            }
        )

    manifest = output_dir / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} cases and manifest to {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="generated_data/lowcond_dense")
    parser.add_argument("--train-count", type=int, default=50)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--size-min", type=int, default=1000)
    parser.add_argument("--size-max", type=int, default=1500)
    parser.add_argument("--cond-log-min", type=float, default=0.0)
    parser.add_argument("--cond-log-max", type=float, default=3.0)
    parser.add_argument("--train-regimes", default="", help="Comma-separated condition regimes for training cases: low,medium,high. Overrides cond-log range for train split.")
    parser.add_argument("--test-regimes", default="", help="Comma-separated condition regimes for testing cases: low,medium,high. Overrides cond-log range for test split.")
    parser.add_argument("--perturbation-rank", type=int, default=8)
    parser.add_argument("--noise-scale", type=float, default=1.0e-2)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.train_count <= 0 or args.test_count <= 0:
        raise SystemExit("train-count and test-count must be positive")
    if args.size_min <= 0 or args.size_max < args.size_min:
        raise SystemExit("invalid matrix size range")
    if args.cond_log_min < 0 or args.cond_log_max < args.cond_log_min:
        raise SystemExit("invalid condition-number exponent range")
    generate(args)


if __name__ == "__main__":
    main()
