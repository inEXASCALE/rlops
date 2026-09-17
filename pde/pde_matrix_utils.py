import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla


PDE_FAMILIES = (
    "poisson_2d",
    "anisotropic_poisson_2d",
    "high_contrast_diffusion_2d",
    "convection_diffusion_2d",
)


@dataclass(frozen=True)
class PDEDatasetConfig:
    num_train: int = 100
    num_test: int = 100
    train_size_min: int = 100
    train_size_max: int = 500
    test_size_min: int = 100
    test_size_max: int = 500
    seed: int = 3407
    rhs_mode: str = "smooth"
    cond_method: str = "auto"
    dense_cond_limit: int = 700
    families: Tuple[str, ...] = PDE_FAMILIES


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def compute_sparsity(A: sp.spmatrix) -> float:
    return float(A.nnz) / float(A.shape[0] * A.shape[1])


def grid_size_from_target(n: int) -> int:
    return int(math.ceil(math.sqrt(n)))


class PDEMatrixGenerator:
    @staticmethod
    def generate_poisson_2d(n: int) -> Tuple[sp.csr_matrix, Dict[str, float]]:
        grid_size = grid_size_from_target(n)
        N = grid_size * grid_size
        main = 4.0 * np.ones(N)
        off_x = -np.ones(N - 1)
        off_y = -np.ones(N - grid_size)

        for i in range(1, grid_size):
            off_x[i * grid_size - 1] = 0.0

        A = sp.diags(
            [main, off_x, off_x, off_y, off_y],
            [0, 1, -1, grid_size, -grid_size],
            format="csr",
        )
        return A, {}

    @staticmethod
    def generate_anisotropic_poisson_2d(
        n: int, rng: np.random.Generator, eps: Optional[float] = None
    ) -> Tuple[sp.csr_matrix, Dict[str, float]]:
        if eps is None:
            eps = float(10.0 ** rng.uniform(-8.0, -3.0))

        grid_size = grid_size_from_target(n)
        N = grid_size * grid_size
        h2inv = float((grid_size + 1) ** 2)
        main = 2.0 * (eps + 1.0) * h2inv * np.ones(N)
        off_x = -eps * h2inv * np.ones(N - 1)
        off_y = -h2inv * np.ones(N - grid_size)

        for i in range(1, grid_size):
            off_x[i * grid_size - 1] = 0.0

        A = sp.diags(
            [main, off_x, off_x, off_y, off_y],
            [0, 1, -1, grid_size, -grid_size],
            format="csr",
        )
        return A, {"eps": eps}

    @staticmethod
    def generate_high_contrast_diffusion_2d(
        n: int, rng: np.random.Generator, contrast: Optional[float] = None
    ) -> Tuple[sp.csr_matrix, Dict[str, float]]:
        if contrast is None:
            contrast = float(10.0 ** rng.uniform(6.0, 13.0))

        grid_size = grid_size_from_target(n)
        N = grid_size * grid_size
        kappa_cell = 10.0 ** rng.uniform(0.0, math.log10(contrast), (grid_size, grid_size))

        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        def harmonic(a: float, b: float) -> float:
            return 2.0 / (1.0 / a + 1.0 / b)

        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                diag = 0.0

                if j + 1 < grid_size:
                    value = harmonic(kappa_cell[i, j], kappa_cell[i, j + 1])
                    rows.append(idx)
                    cols.append(idx + 1)
                    data.append(-value)
                    diag += value
                else:
                    diag += float(kappa_cell[i, j])

                if j > 0:
                    diag += harmonic(kappa_cell[i, j], kappa_cell[i, j - 1])
                else:
                    diag += float(kappa_cell[i, j])

                if i + 1 < grid_size:
                    value = harmonic(kappa_cell[i, j], kappa_cell[i + 1, j])
                    rows.append(idx)
                    cols.append(idx + grid_size)
                    data.append(-value)
                    diag += value
                else:
                    diag += float(kappa_cell[i, j])

                if i > 0:
                    diag += harmonic(kappa_cell[i, j], kappa_cell[i - 1, j])
                else:
                    diag += float(kappa_cell[i, j])

                rows.append(idx)
                cols.append(idx)
                data.append(diag)

        A_upper = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
        A = A_upper + sp.triu(A_upper, k=1).T
        return A.tocsr(), {"contrast": contrast}

    @staticmethod
    def generate_convection_diffusion_2d(
        n: int,
        rng: np.random.Generator,
        eps: Optional[float] = None,
        beta_x: Optional[float] = None,
        beta_y: Optional[float] = None,
    ) -> Tuple[sp.csr_matrix, Dict[str, float]]:
        if eps is None:
            eps = float(10.0 ** rng.uniform(-3.0, 0.0))
        if beta_x is None:
            beta_x = float(rng.uniform(-10.0, 10.0))
        if beta_y is None:
            beta_y = float(rng.uniform(-10.0, 10.0))
        if abs(beta_x) + abs(beta_y) < 1e-12:
            beta_x = 1.0

        grid_size = grid_size_from_target(n)
        N = grid_size * grid_size
        h = 1.0 / float(grid_size + 1)
        h2 = h * h
        cx = beta_x / h
        cy = beta_y / h

        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        for idx in range(N):
            row = idx // grid_size
            col = idx % grid_size
            diag = 4.0 * eps / h2 + abs(cx) + abs(cy)

            rows.append(idx)
            cols.append(idx)
            data.append(diag)

            if col > 0:
                value = -eps / h2
                if cx >= 0.0:
                    value -= cx
                rows.append(idx)
                cols.append(idx - 1)
                data.append(value)

            if col < grid_size - 1:
                value = -eps / h2
                if cx < 0.0:
                    value += cx
                rows.append(idx)
                cols.append(idx + 1)
                data.append(value)

            if row > 0:
                value = -eps / h2
                if cy >= 0.0:
                    value -= cy
                rows.append(idx)
                cols.append(idx - grid_size)
                data.append(value)

            if row < grid_size - 1:
                value = -eps / h2
                if cy < 0.0:
                    value += cy
                rows.append(idx)
                cols.append(idx + grid_size)
                data.append(value)

        A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
        return A, {"eps": eps, "beta_x": beta_x, "beta_y": beta_y}


def make_exact_solution(grid_size: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    x = np.linspace(1.0 / (grid_size + 1), grid_size / (grid_size + 1), grid_size)
    y = np.linspace(1.0 / (grid_size + 1), grid_size / (grid_size + 1), grid_size)
    X, Y = np.meshgrid(x, y, indexing="ij")

    if mode == "smooth":
        values = np.sin(np.pi * X) * np.sin(np.pi * Y)
        values += 0.25 * np.sin(2.0 * np.pi * X) * np.sin(3.0 * np.pi * Y)
    elif mode == "piecewise":
        values = np.where((X < 0.5) & (Y < 0.5), 1.0, -0.5)
        values += 0.1 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    elif mode == "random":
        values = rng.standard_normal((grid_size, grid_size))
    else:
        raise ValueError(f"Unknown rhs_mode '{mode}'.")

    x_true = np.asarray(values, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(x_true, ord=np.inf)
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError(f"Invalid exact solution generated for rhs_mode '{mode}'.")
    return x_true


def estimate_condition_number(A: sp.spmatrix, method: str = "auto", dense_limit: int = 700) -> float:
    n = A.shape[0]
    if method == "auto":
        method = "dense" if n <= dense_limit else "onenormest"

    try:
        if method == "dense":
            cond = float(np.linalg.cond(A.toarray()))
        elif method == "onenormest":
            A_csc = A.tocsc().astype(np.float64)
            lu = spla.splu(A_csc)
            inv_op = spla.LinearOperator(
                shape=A.shape,
                matvec=lu.solve,
                rmatvec=lambda x: lu.solve(x, trans="T"),
                dtype=np.float64,
            )
            cond = float(abs(A).sum(axis=0).max()) * float(spla.onenormest(inv_op))
        else:
            raise ValueError(f"Unknown cond_method '{method}'.")
    except Exception:
        return float("inf")

    return cond if np.isfinite(cond) and cond > 0.0 else float("inf")


def generate_pde_linear_system(
    family: str,
    target_size: int,
    rhs_mode: str,
    rng: np.random.Generator,
    cond_method: str = "auto",
    dense_cond_limit: int = 700,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray, Dict[str, object]]:
    if family == "poisson_2d":
        A, params = PDEMatrixGenerator.generate_poisson_2d(target_size)
    elif family == "anisotropic_poisson_2d":
        A, params = PDEMatrixGenerator.generate_anisotropic_poisson_2d(target_size, rng)
    elif family == "high_contrast_diffusion_2d":
        A, params = PDEMatrixGenerator.generate_high_contrast_diffusion_2d(target_size, rng)
    elif family == "convection_diffusion_2d":
        A, params = PDEMatrixGenerator.generate_convection_diffusion_2d(target_size, rng)
    else:
        raise ValueError(f"Unsupported PDE family '{family}'.")

    grid_size = int(math.sqrt(A.shape[0]))
    x_true = make_exact_solution(grid_size, rhs_mode, rng)
    b = np.asarray(A @ x_true, dtype=np.float64)
    cond_number = estimate_condition_number(A, method=cond_method, dense_limit=dense_cond_limit)

    metadata: Dict[str, object] = {
        "condition_number": cond_number,
        "kappa": cond_number,
        "matrix_size": A.shape[0],
        "grid_size": grid_size,
        "target_size": target_size,
        "sparsity": compute_sparsity(A),
        "nnz": int(A.nnz),
        "pde_type": family,
        "rhs_mode": rhs_mode,
        "cond_method": cond_method,
        "symmetric": bool((A - A.T).nnz == 0),
    }
    metadata.update(params)
    return A.tocsr(), b, x_true, metadata


def metadata_rows(dataset: Sequence[Tuple[sp.csr_matrix, np.ndarray, np.ndarray, Dict[str, object]]], split: str) -> List[Dict[str, object]]:
    rows = []
    for i, (_, _, _, metadata) in enumerate(dataset):
        row = {"split": split, "matrix_id": str(i)}
        row.update(metadata)
        rows.append(row)
    return rows


def save_pde_datasets(
    train_data: Sequence[Tuple[sp.csr_matrix, np.ndarray, np.ndarray, Dict[str, object]]],
    test_data: Sequence[Tuple[sp.csr_matrix, np.ndarray, np.ndarray, Dict[str, object]]],
    data_dir: str,
    config: PDEDatasetConfig,
) -> None:
    os.makedirs(data_dir, exist_ok=True)
    for split, dataset in (("train", train_data), ("test", test_data)):
        for i, (A, b, x_true, _) in enumerate(dataset):
            sp.save_npz(os.path.join(data_dir, f"{split}_matrix_{i}.npz"), A)
            np.save(os.path.join(data_dir, f"{split}_vector_{i}.npy"), b)
            np.save(os.path.join(data_dir, f"{split}_x_true_{i}.npy"), x_true)

    metadata = metadata_rows(train_data, "train") + metadata_rows(test_data, "test")
    pd.DataFrame(metadata).to_csv(os.path.join(data_dir, "metadata.csv"), index=False)
    with open(os.path.join(data_dir, "dataset_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)


def normalize_dataset_config(config: Dict[str, object]) -> Dict[str, object]:
    normalized = dict(config)
    if "families" in normalized:
        normalized["families"] = list(normalized["families"])
    return normalized


def dataset_config_matches(data_dir: str, expected_config: Optional[PDEDatasetConfig]) -> bool:
    if expected_config is None:
        return True
    config_path = os.path.join(data_dir, "dataset_config.json")
    if not os.path.exists(config_path):
        return False
    with open(config_path, "r", encoding="utf-8") as f:
        cached = normalize_dataset_config(json.load(f))
    expected = normalize_dataset_config(asdict(expected_config))
    return all(cached.get(key) == value for key, value in expected.items())


def load_pde_datasets(
    num_train: int,
    num_test: int,
    data_dir: str,
    expected_config: Optional[PDEDatasetConfig] = None,
) -> Tuple[Optional[List[Tuple[sp.csr_matrix, np.ndarray, np.ndarray, Dict[str, object]]]], Optional[List[Tuple[sp.csr_matrix, np.ndarray, np.ndarray, Dict[str, object]]]], Optional[List[Dict[str, object]]], Optional[List[Dict[str, object]]]]:
    if not dataset_config_matches(data_dir, expected_config):
        return None, None, None, None

    metadata_path = os.path.join(data_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        return None, None, None, None

    metadata = pd.read_csv(metadata_path)
    train_data = []
    test_data = []

    for split, expected, target in (("train", num_train, train_data), ("test", num_test, test_data)):
        rows = metadata[metadata["split"] == split].copy()
        if len(rows) < expected:
            return None, None, None, None
        rows = rows.head(expected)
        for _, row in rows.iterrows():
            matrix_id = int(row["matrix_id"])
            matrix_path = os.path.join(data_dir, f"{split}_matrix_{matrix_id}.npz")
            vector_path = os.path.join(data_dir, f"{split}_vector_{matrix_id}.npy")
            x_true_path = os.path.join(data_dir, f"{split}_x_true_{matrix_id}.npy")
            if not (os.path.exists(matrix_path) and os.path.exists(vector_path) and os.path.exists(x_true_path)):
                return None, None, None, None
            A = sp.load_npz(matrix_path).tocsr()
            b = np.load(vector_path)
            x_true = np.load(x_true_path)
            row_dict = row.drop(labels=["split"]).to_dict()
            row_dict["matrix_id"] = str(row_dict["matrix_id"])
            target.append((A, b, x_true, row_dict))

    train_metrics = metadata_rows(train_data, "train")
    test_metrics = metadata_rows(test_data, "test")
    return train_data, test_data, train_metrics, test_metrics


def generate_pde_datasets(
    config: PDEDatasetConfig,
    verbose: bool = True,
) -> Tuple[List[Tuple[sp.csr_matrix, np.ndarray, np.ndarray, Dict[str, object]]], List[Tuple[sp.csr_matrix, np.ndarray, np.ndarray, Dict[str, object]]]]:
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    families = tuple(config.families)
    if not families or any(family not in PDE_FAMILIES for family in families):
        raise ValueError(f"families must be selected from {PDE_FAMILIES}.")

    def generate_split(
        split: str,
        count: int,
        size_min: int,
        size_max: int,
    ) -> List[Tuple[sp.csr_matrix, np.ndarray, np.ndarray, Dict[str, object]]]:
        dataset = []
        attempts = 0
        max_attempts = max(10 * count, count + 20)

        while len(dataset) < count and attempts < max_attempts:
            attempts += 1
            target_size = int(rng.integers(size_min, size_max + 1))
            family = families[len(dataset) % len(families)]
            try:
                A, b, x_true, metadata = generate_pde_linear_system(
                    family=family,
                    target_size=target_size,
                    rhs_mode=config.rhs_mode,
                    rng=rng,
                    cond_method=config.cond_method,
                    dense_cond_limit=config.dense_cond_limit,
                )
            except Exception as exc:
                if verbose:
                    print(f"[warn] failed to generate {split} {family} target_size={target_size}: {exc}")
                continue

            if not np.isfinite(metadata["condition_number"]):
                if verbose:
                    print(f"[warn] skipped singular/ill-conditioned {split} {family} target_size={target_size}")
                continue
            if not np.all(np.isfinite(b)) or not np.all(np.isfinite(x_true)):
                if verbose:
                    print(f"[warn] skipped non-finite RHS/solution for {split} {family} target_size={target_size}")
                continue

            dataset.append((A, b, x_true, metadata))
            if verbose:
                print(
                    f"[{split} {len(dataset):03d}/{count}] {family}: n={A.shape[0]}, "
                    f"nnz={A.nnz}, cond={metadata['condition_number']:.3e}"
                )

        if len(dataset) < count:
            raise RuntimeError(f"Generated only {len(dataset)} valid {split} PDE systems out of requested {count}.")
        return dataset

    train_data = generate_split("train", config.num_train, config.train_size_min, config.train_size_max)
    test_data = generate_split("test", config.num_test, config.test_size_min, config.test_size_max)
    return train_data, test_data


def parse_families(value: str) -> Tuple[str, ...]:
    if value.strip().lower() == "all":
        return PDE_FAMILIES
    return tuple(part.strip() for part in value.split(",") if part.strip())


def resolve_split_size_ranges(
    train_size_min: Optional[int],
    train_size_max: Optional[int],
    test_size_min: Optional[int],
    test_size_max: Optional[int],
    size_min: Optional[int],
    size_max: Optional[int],
) -> Tuple[int, int, int, int]:
    resolved_train_min = train_size_min if train_size_min is not None else (size_min if size_min is not None else 100)
    resolved_train_max = train_size_max if train_size_max is not None else (size_max if size_max is not None else 500)
    resolved_test_min = test_size_min if test_size_min is not None else (size_min if size_min is not None else 100)
    resolved_test_max = test_size_max if test_size_max is not None else (size_max if size_max is not None else 500)

    if resolved_train_min > resolved_train_max:
        raise ValueError("--train-size-min must be <= --train-size-max")
    if resolved_test_min > resolved_test_max:
        raise ValueError("--test-size-min must be <= --test-size-max")
    return resolved_train_min, resolved_train_max, resolved_test_min, resolved_test_max


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate PDE-discretized sparse linear-system datasets.")
    parser.add_argument("--output-dir", default="data_pde", help="Directory for matrices, RHS vectors, and metadata.")
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
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train_size_min, train_size_max, test_size_min, test_size_max = resolve_split_size_ranges(
        args.train_size_min,
        args.train_size_max,
        args.test_size_min,
        args.test_size_max,
        args.size_min,
        args.size_max,
    )
    config = PDEDatasetConfig(
        num_train=args.num_train,
        num_test=args.num_test,
        train_size_min=train_size_min,
        train_size_max=train_size_max,
        test_size_min=test_size_min,
        test_size_max=test_size_max,
        seed=args.seed,
        rhs_mode=args.rhs_mode,
        cond_method=args.cond_method,
        dense_cond_limit=args.dense_cond_limit,
        families=parse_families(args.families),
    )
    train_data, test_data = generate_pde_datasets(config)
    save_pde_datasets(train_data, test_data, args.output_dir, config)
    print(f"Saved {len(train_data)} train and {len(test_data)} test PDE systems to {args.output_dir}")


if __name__ == "__main__":
    main()
