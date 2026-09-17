import argparse
import ast
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
FONT_SIZE = 14
FAMILY_COLORS = {
    "poisson_2d": "#4C78A8",
    "anisotropic_poisson_2d": "#F58518",
    "high_contrast_diffusion_2d": "#54A24B",
    "convection_diffusion_2d": "#B279A2",
}
SCATTER_ALPHA = 0.75
SCATTER_EDGE_COLOR = "white"
SCATTER_LINEWIDTH = 1.2
SCATTER_SIZE = 720.0
CONFIG_LABELS = {
    "1": r"$W_1$, $\tau=10^{-6}$",
    "2": r"$W_2$, $\tau=10^{-6}$",
    "3": r"$W_1$, $\tau=10^{-8}$",
    "4": r"$W_2$, $\tau=10^{-8}$",
}


def apply_plot_style(font_size: int) -> None:
    global FONT_SIZE
    FONT_SIZE = font_size
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE + 2,
            "xtick.labelsize": FONT_SIZE - 1,
            "ytick.labelsize": FONT_SIZE - 1,
            "legend.fontsize": FONT_SIZE - 1,
            "legend.title_fontsize": FONT_SIZE,
            "figure.titlesize": FONT_SIZE + 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure_stems(fig: plt.Figure, output_dir: Path, stems: List[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", dpi=400)
        fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", dpi=400)
    plt.close(fig)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    save_figure_stems(fig, output_dir, [stem])


def discover_result_dirs(results_root: Path, result_dirs: Iterable[str]) -> List[Path]:
    requested = list(result_dirs)
    if requested:
        return [Path(item) if Path(item).is_absolute() else results_root / item for item in requested]
    return [path for path in sorted(results_root.glob("results_pde*")) if (path / "test_results.csv").exists()]


def load_merged_results(results_dir: Path) -> pd.DataFrame:
    results = pd.read_csv(results_dir / "test_results.csv")
    metrics_path = results_dir / "test_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        results["matrix_id"] = results["matrix_id"].astype(str)
        metrics["matrix_id"] = metrics["matrix_id"].astype(str)
        drop_cols = [col for col in ("matrix_size",) if col in metrics.columns and col in results.columns]
        metrics = metrics.drop(columns=drop_cols)
        results = results.merge(metrics, on="matrix_id", how="left")
    return results


def format_training_label(results_dir: Path) -> str:
    suffix = results_dir.name.removeprefix("results_pde")
    fallback = CONFIG_LABELS.get(suffix, results_dir.name.replace("results_", ""))
    config_path = results_dir / "run_config.json"
    if not config_path.exists():
        return fallback
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return fallback
    config_id = str(config.get("config_id", suffix))
    if config_id in CONFIG_LABELS:
        return CONFIG_LABELS[config_id]
    c2 = float(config.get("c2", 1.0))
    weight = r"$W_1$" if c2 < 1.0 else r"$W_2$"
    tol = float(config.get("tol", 0.0))
    if tol > 0.0:
        exponent = int(np.round(np.log10(tol)))
        return rf"{weight}, $\tau=10^{{{exponent}}}$"
    return str(config.get("label", results_dir.name.replace("results_", "")))


def error_iteration_stems(results_dir: Path) -> List[str]:
    stems = [f"{results_dir.name}_error_iterations"]
    config_path = results_dir / "run_config.json"
    if not config_path.exists():
        return stems
    try:
        config = json.loads(config_path.read_text())
        tol = float(config.get("tol", 0.0))
        c2 = float(config.get("c2", 1.0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return stems
    if tol > 0.0 and c2 >= 1.0:
        exponent = abs(int(np.round(np.log10(tol))))
        stems.append(f"pde_error_iterations_w2_tol1e{exponent}")
    return stems


def positive_finite(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric[np.isfinite(numeric) & (numeric > 0.0)]


def plot_training_curves(result_dirs: List[Path], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=300)
    plotted = False

    for results_dir in result_dirs:
        label = format_training_label(results_dir)
        rewards_path = results_dir / "episode_rewards.npy"
        losses_path = results_dir / "training_losses.npy"

        if rewards_path.exists():
            rewards = np.load(rewards_path)
            axes[0].plot(np.arange(len(rewards)), rewards, linewidth=2.0, label=label)
            plotted = True
        if losses_path.exists():
            losses = np.load(losses_path)
            axes[1].plot(np.arange(len(losses)), losses, linewidth=2.0, label=label)
            plotted = True

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Average reward")
    axes[0].set_title("Training reward")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Average RPE")
    axes[1].set_title("Training loss")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        if ax.has_data():
            ax.legend(frameon=True, framealpha=0.9)

    if plotted:
        fig.tight_layout()
        save_figure(fig, output_dir, "pde_training_curves")
    else:
        plt.close(fig)


def marker_sizes(matrix_size: pd.Series) -> np.ndarray:
    return np.full(len(matrix_size), SCATTER_SIZE)


def plot_error_iteration_comparison(results_dir: Path, output_dir: Path) -> None:
    df = load_merged_results(results_dir)
    required = ["rl_error", "fp64_error", "rl_gmres_iterations", "fp64_gmres_iterations"]
    df = df.dropna(subset=[col for col in required if col in df.columns]).copy()
    if df.empty:
        return

    family_col = "pde_type" if "pde_type" in df.columns else None
    df["plot_marker_size"] = marker_sizes(df["matrix_size"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    rng = np.random.default_rng(42)

    groups = [(None, df)] if family_col is None else list(df.groupby(family_col, dropna=False))
    for family, group in groups:
        label = str(family) if family is not None else "PDE matrices"
        color = FAMILY_COLORS.get(label, None)
        size = group["plot_marker_size"]

        axes[0].scatter(
            group["rl_error"],
            group["fp64_error"],
            s=size,
            alpha=SCATTER_ALPHA,
            label=label,
            color=color,
            edgecolors=SCATTER_EDGE_COLOR,
            linewidths=SCATTER_LINEWIDTH,
        )

        jitter = rng.uniform(-0.12, 0.12, size=len(group))
        axes[1].scatter(
            group["rl_gmres_iterations"] + jitter,
            group["fp64_gmres_iterations"] - jitter,
            s=size,
            alpha=SCATTER_ALPHA,
            label=label,
            color=color,
            edgecolors=SCATTER_EDGE_COLOR,
            linewidths=SCATTER_LINEWIDTH,
        )

    error_values = pd.concat([positive_finite(df["rl_error"]), positive_finite(df["fp64_error"])])
    if not error_values.empty:
        min_err = max(error_values.min() * 0.5, 1e-18)
        max_err = error_values.max() * 2.0
        axes[0].plot([min_err, max_err], [min_err, max_err], color="0.35", linestyle="--", linewidth=1.5)
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlim(min_err, max_err)
        axes[0].set_ylim(min_err, max_err)

    iter_min = float(df[["rl_gmres_iterations", "fp64_gmres_iterations"]].min().min()) - 1.0
    iter_max = float(df[["rl_gmres_iterations", "fp64_gmres_iterations"]].max().max()) + 1.0
    axes[1].plot([iter_min, iter_max], [iter_min, iter_max], color="0.35", linestyle="--", linewidth=1.5)

    axes[0].set_xlabel("RL relative forward error")
    axes[0].set_ylabel("FP64 relative forward error")
    axes[0].set_title("Error comparison")
    axes[1].set_xlabel("RL GMRES iterations")
    axes[1].set_ylabel("FP64 GMRES iterations")
    axes[1].set_title("GMRES iteration comparison")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 2)
        ax.tick_params(axis="both", which="minor", labelsize=FONT_SIZE - 4)
    axes[1].legend(
        title="PDE family",
        loc="best",
        frameon=True,
        fancybox=True,
        framealpha=0.85,
        facecolor="white",
        edgecolor="none",
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_figure_stems(fig, output_dir, error_iteration_stems(results_dir))


def parse_precision_usage(value) -> Dict[str, float]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            return {}
    return {}


def plot_precision_usage(results_dir: Path, output_dir: Path) -> None:
    df = load_merged_results(results_dir)
    if "precision_usage" not in df.columns:
        return
    if "pde_type" not in df.columns:
        df["pde_type"] = "PDE matrices"

    rows = []
    for _, row in df.iterrows():
        for precision, count in parse_precision_usage(row["precision_usage"]).items():
            rows.append({"pde_type": row["pde_type"], "precision": precision, "count": float(count)})
    if not rows:
        return

    usage = pd.DataFrame(rows)
    table = usage.pivot_table(index="pde_type", columns="precision", values="count", aggfunc="sum", fill_value=0.0)
    table = table.reindex(columns=[col for col in ["bf16", "tf32", "fp32", "fp64"] if col in table.columns])
    fractions = table.div(table.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)

    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=300)
    bottom = np.zeros(len(fractions))
    colors = {"bf16": "#4C78A8", "tf32": "#F58518", "fp32": "#54A24B", "fp64": "#E45756"}

    for precision in fractions.columns:
        values = fractions[precision].to_numpy()
        ax.bar(fractions.index, values, bottom=bottom, label=precision, color=colors.get(precision))
        bottom += values

    ax.set_ylabel("Fraction of selected precision slots")
    ax.set_title("Precision usage by PDE family")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Precision", frameon=True, framealpha=0.9)
    fig.tight_layout()
    save_figure(fig, output_dir, f"{results_dir.name}_precision_usage")


def plot_condition_distribution(data_dir: Path, output_dir: Path) -> None:
    metadata_path = data_dir / "metadata.csv"
    if not metadata_path.exists():
        return
    df = pd.read_csv(metadata_path)
    if "condition_number" not in df.columns:
        return
    df["log10_condition"] = np.log10(pd.to_numeric(df["condition_number"], errors="coerce"))
    df = df[np.isfinite(df["log10_condition"])]
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=300)
    for family, group in df.groupby("pde_type"):
        ax.hist(
            group["log10_condition"],
            bins=14,
            alpha=0.48,
            label=str(family),
            color=FAMILY_COLORS.get(str(family)),
        )

    ax.set_xlabel(r"$\log_{10}(\kappa(A))$")
    ax.set_ylabel("Count")
    ax.set_title("PDE dataset condition-number distribution")
    ax.legend(title="PDE family", frameon=True, framealpha=0.9)
    fig.tight_layout()
    save_figure(fig, output_dir, "pde_condition_distribution")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize PDE experiment outputs without running solvers.")
    parser.add_argument("--results-root", default=str(CURRENT_DIR))
    parser.add_argument("--result-dirs", nargs="*", default=[], help="Specific result dirs under --results-root.")
    parser.add_argument("--data-dir", default=str(CURRENT_DIR / "data_pde"))
    parser.add_argument("--output-dir", default=str(CURRENT_DIR / "figures_pde"))
    parser.add_argument("--font-size", type=int, default=14, help="Global fontsize for x/y labels, x/y ticks, legends, and titles.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    apply_plot_style(args.font_size)

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    result_dirs = discover_result_dirs(results_root, args.result_dirs)

    plot_training_curves(result_dirs, output_dir)
    for results_dir in result_dirs:
        if (results_dir / "test_results.csv").exists():
            plot_error_iteration_comparison(results_dir, output_dir)
            plot_precision_usage(results_dir, output_dir)
    plot_condition_distribution(Path(args.data_dir), output_dir)
    print(f"Saved PDE figures to {output_dir}")


if __name__ == "__main__":
    main()
