#!/usr/bin/env python3
"""Plot CPU low-condition GMRES-IR experiment outputs."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"fp16": "#4C78A8", "fp32": "#F58518", "fp64": "#54A24B"}


def apply_style(font_size: int) -> None:
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", dpi=400)
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", dpi=400)
    plt.close(fig)


def plot_summary(input_dir: Path, output_dir: Path) -> None:
    rows = read_csv(input_dir / "summary.csv")
    labels = [row["config"] for row in rows]
    speedups = [float(row["mean_valid_speedup"]) for row in rows]
    memory = [float(row["mean_memory_ratio"]) for row in rows]
    valid = [float(row["valid_cases"]) / max(float(row["test_cases"]), 1.0) for row in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=300)
    axes[0].bar(x, speedups, color="#4C78A8")
    axes[0].set_ylabel("Mean valid speedup vs FP64")
    axes[1].bar(x, memory, color="#F58518")
    axes[1].set_ylabel("Mean memory ratio vs FP64")
    axes[2].bar(x, valid, color="#54A24B")
    axes[2].set_ylabel("Valid test fraction")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_xlabel("Configuration")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, output_dir, "cpu_lowcond_summary")


def plot_training_curves(input_dir: Path, output_dir: Path) -> None:
    config_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir() and (path / "train_curve.csv").exists())
    if not config_dirs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=300)
    for cfg in config_dirs:
        rows = read_csv(cfg / "train_curve.csv")
        steps = [int(row["step"]) for row in rows]
        rewards = [float(row["reward"]) for row in rows]
        losses = [float(row["train_loss"]) for row in rows]
        axes[0].plot(steps, rewards, linewidth=1.9, label=cfg.name)
        axes[1].plot(steps, losses, linewidth=1.9, label=cfg.name)
    axes[0].set_xlabel("Training case")
    axes[0].set_ylabel("Selected-action reward")
    axes[1].set_xlabel("Training case")
    axes[1].set_ylabel("Training loss")
    for ax in axes:
        ax.legend(title="Configuration", frameon=True, framealpha=0.92)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, output_dir, "cpu_lowcond_training_curves")


def plot_test_details(input_dir: Path, output_dir: Path) -> None:
    config_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir() and (path / "test_results.csv").exists())
    if not config_dirs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), dpi=300)
    for cfg in config_dirs:
        rows = read_csv(cfg / "test_results.csv")
        cond = [float(row["target_cond"]) for row in rows]
        speedup = [float(row["speedup_vs_fp64"]) for row in rows]
        memory = [float(row["memory_ratio_vs_fp64"]) for row in rows]
        update_precision = [row["u"] for row in rows]
        for p in ("fp16", "fp32", "fp64"):
            xs = [c for c, q in zip(cond, update_precision) if q == p]
            ys_speed = [s for s, q in zip(speedup, update_precision) if q == p]
            ys_mem = [m for m, q in zip(memory, update_precision) if q == p]
            if xs:
                axes[0].scatter(xs, ys_speed, label=f"{cfg.name}:u={p}", s=28, alpha=0.75, color=COLORS[p])
                axes[1].scatter(xs, ys_mem, label=f"{cfg.name}:u={p}", s=28, alpha=0.75, color=COLORS[p])
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel("Target condition number")
    axes[0].set_ylabel("Speedup vs FP64")
    axes[1].set_xlabel("Target condition number")
    axes[1].set_ylabel("Memory ratio vs FP64")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    # Keep the dense legend outside so markers remain readable.
    axes[1].legend(title="Config:update precision", bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=True)
    fig.tight_layout()
    save(fig, output_dir, "cpu_lowcond_test_details")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="results_cpu_lowcond")
    parser.add_argument("--output-dir", default="figures_cpu_lowcond")
    parser.add_argument("--font-size", type=int, default=14)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not (input_dir / "summary.csv").exists():
        raise SystemExit(f"missing summary.csv in {input_dir}")
    apply_style(args.font_size)
    plot_summary(input_dir, output_dir)
    plot_training_curves(input_dir, output_dir)
    plot_test_details(input_dir, output_dir)
    print(f"Wrote CPU low-condition figures to {output_dir}")


if __name__ == "__main__":
    main()
