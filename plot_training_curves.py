#!/usr/bin/env python3
"""Plot combined training reward/loss curves for dense, sparse, and PDE tests."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent

CONFIG_LABELS = {
    1: r"$W_1$, $\tau=10^{-6}$",
    2: r"$W_2$, $\tau=10^{-6}$",
    3: r"$W_1$, $\tau=10^{-8}$",
    4: r"$W_2$, $\tau=10^{-8}$",
}

LINE_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
LINE_STYLES = ["-", "--", "-.", ":"]


@dataclass(frozen=True)
class TrainingDataset:
    name: str
    root: Path
    result_template: str
    output_dir: Path
    output_stem: str
    title_prefix: str


DATASETS = {
    "dense": TrainingDataset(
        name="dense",
        root=CURRENT_DIR / "dense_test",
        result_template="results_random_dense{config_id}",
        output_dir=CURRENT_DIR / "dense_test" / "figures_training",
        output_stem="dense_training_curves",
        title_prefix="Dense systems",
    ),
    "sparse": TrainingDataset(
        name="sparse",
        root=CURRENT_DIR / "sparse_test",
        result_template="results_random_sparse{config_id}",
        output_dir=CURRENT_DIR / "sparse_test" / "figures_training",
        output_stem="sparse_training_curves",
        title_prefix="Sparse systems",
    ),
    "pde": TrainingDataset(
        name="pde",
        root=CURRENT_DIR / "pde",
        result_template="results_pde{config_id}",
        output_dir=CURRENT_DIR / "pde" / "figures_pde",
        output_stem="pde_training_curves",
        title_prefix="PDE-generated systems",
    ),
}


def apply_plot_style(font_size: int) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 1,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "legend.title_fontsize": font_size,
            "figure.titlesize": font_size + 2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", dpi=400)
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", dpi=400)
    plt.close(fig)


def load_curve(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    values = np.load(path)
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def plot_dataset(dataset: TrainingDataset) -> bool:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=300)
    plotted = False

    for offset, config_id in enumerate((1, 2, 3, 4)):
        results_dir = dataset.root / dataset.result_template.format(config_id=config_id)
        label = CONFIG_LABELS[config_id]
        color = LINE_COLORS[offset]
        linestyle = LINE_STYLES[offset]

        rewards = load_curve(results_dir / "episode_rewards.npy")
        if rewards is not None and rewards.size:
            axes[0].plot(
                np.arange(1, rewards.size + 1),
                rewards,
                color=color,
                linestyle=linestyle,
                linewidth=2.1,
                label=label,
            )
            plotted = True

        losses = load_curve(results_dir / "training_losses.npy")
        if losses is not None and losses.size:
            axes[1].plot(
                np.arange(1, losses.size + 1),
                losses,
                color=color,
                linestyle=linestyle,
                linewidth=2.1,
                label=label,
            )
            plotted = True

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Average reward")
    axes[0].set_title(f"{dataset.title_prefix}: reward")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Average RPE")
    axes[1].set_title(f"{dataset.title_prefix}: training loss")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        if ax.has_data():
            ax.legend(title="Configuration", frameon=True, framealpha=0.92)

    if not plotted:
        plt.close(fig)
        return False

    fig.tight_layout()
    save_figure(fig, dataset.output_dir, dataset.output_stem)
    print(f"Saved {dataset.name} training curves to {dataset.output_dir}")
    return True


def resolve_dataset_names(names: Sequence[str]) -> list[str]:
    if not names or "all" in names:
        return ["dense", "sparse", "pde"]
    unknown = sorted(set(names) - set(DATASETS))
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {', '.join(unknown)}")
    return list(names)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["all"],
        help="Datasets to plot: dense sparse pde all. Default: all.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=14,
        help="Global font size for labels, ticks, legends, and titles.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    apply_plot_style(args.font_size)
    for name in resolve_dataset_names(args.datasets):
        if not plot_dataset(DATASETS[name]):
            print(f"No training curves found for {name}.")


if __name__ == "__main__":
    main()
