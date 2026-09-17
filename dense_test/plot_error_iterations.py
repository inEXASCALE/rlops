import argparse
import os
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FONT_SIZE = 14
SIZE_BINS = [101, 201, 301, 401, 501]
SIZE_LABELS = ["101-200", "201-300", "301-400", "401-500"]
POINT_SIZE = 720.0
GROUP_COLORS = {
    "101-200": "#4C78A8",
    "201-300": "#F58518",
    "301-400": "#54A24B",
    "401-500": "#B279A2",
}
SCATTER_ALPHA = 0.75
SCATTER_EDGE_COLOR = "white"
SCATTER_LINEWIDTH = 1.2
JITTER = 0.12


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


def positive_finite(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric[np.isfinite(numeric) & (numeric > 0.0)]


def prepare_dataframe(results_dir: Path, seed: int) -> pd.DataFrame:
    df = pd.read_csv(results_dir / "test_results.csv")
    df["size_group"] = pd.cut(df["matrix_size"], bins=SIZE_BINS, labels=SIZE_LABELS, include_lowest=True)
    df = df.dropna(subset=["size_group"]).copy()
    df["point_size"] = POINT_SIZE
    rng = np.random.default_rng(seed)
    df["rl_gmres_jitter"] = df["rl_gmres_iterations"] + rng.uniform(-JITTER, JITTER, size=len(df))
    df["fp64_gmres_jitter"] = df["fp64_gmres_iterations"] + rng.uniform(-JITTER, JITTER, size=len(df))
    return df


def save_figure(fig: plt.Figure, results_dir: Path, output_stems: List[str]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    for stem in output_stems:
        fig.savefig(results_dir / f"{stem}.pdf", bbox_inches="tight", dpi=400)
        fig.savefig(results_dir / f"{stem}.png", bbox_inches="tight", dpi=400)
    plt.close(fig)


def plot_error_iterations(results_dir: Path, output_stems: List[str], font_size: int, seed: int) -> None:
    apply_plot_style(font_size)
    df = prepare_dataframe(results_dir, seed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

    for size_group, group in df.groupby("size_group", observed=True):
        label = str(size_group)
        color = GROUP_COLORS.get(label, None)
        axes[0].scatter(
            group["rl_error"],
            group["fp64_error"],
            s=group["point_size"],
            alpha=SCATTER_ALPHA,
            label=label,
            color=color,
            edgecolors=SCATTER_EDGE_COLOR,
            linewidths=SCATTER_LINEWIDTH,
        )
        axes[1].scatter(
            group["rl_gmres_jitter"],
            group["fp64_gmres_jitter"],
            s=group["point_size"],
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

    min_iter = float(df[["rl_gmres_iterations", "fp64_gmres_iterations"]].min().min()) - 1.0
    max_iter = float(df[["rl_gmres_iterations", "fp64_gmres_iterations"]].max().max()) + 1.0
    axes[1].plot([min_iter, max_iter], [min_iter, max_iter], color="0.35", linestyle="--", linewidth=1.5)
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
        title="Matrix size",
        loc="best",
        frameon=True,
        fancybox=True,
        framealpha=0.85,
        facecolor="white",
        edgecolor="none",
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_figure(fig, results_dir, output_stems)


def build_arg_parser(default_results_dir: str = "results_random_dense2", default_alias_stem: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot dense RL-vs-FP64 error and GMRES iteration comparisons.")
    parser.add_argument("--results-dir", default=default_results_dir)
    parser.add_argument("--font-size", type=int, default=14, help="Global fontsize for x/y labels, x/y ticks, legends, and titles.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic GMRES-iteration jitter.")
    parser.add_argument("--output-stem", default="error_iterations_comparison")
    parser.add_argument("--alias-stem", default=default_alias_stem, help="Optional additional output stem, e.g. the filename used by the paper.")
    return parser


def main(default_results_dir: str = "results_random_dense2", default_alias_stem: Optional[str] = None) -> None:
    args = build_arg_parser(default_results_dir, default_alias_stem).parse_args()
    stems = [args.output_stem]
    if args.alias_stem and args.alias_stem not in stems:
        stems.append(args.alias_stem)
    plot_error_iterations(Path(args.results_dir), stems, args.font_size, args.seed)
    print(f"Saved dense error/iteration figures to {args.results_dir}")


if __name__ == "__main__":
    main()
