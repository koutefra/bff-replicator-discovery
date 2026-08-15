#!/usr/bin/env python3
"""Regenerate every external figure included by the manuscript."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_toy_canonical_progression import (
    DEFAULT_FOCAL,
    DEFAULT_PARTNER,
    canonical_snapshots,
    evaluate_modulo,
    render_figure,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "paper"
FIGURES = ROOT / "figures"
N_COLOR = "#1f77b4"
R_COLOR = "#d62728"
OP_COLUMNS = [
    ("op_lt", "<"),
    ("op_gt", ">"),
    ("op_lbrace", "{"),
    ("op_rbrace", "}"),
    ("op_plus", "+"),
    ("op_minus", "−"),
    ("op_dot", "."),
    ("op_comma", ","),
    ("op_lbracket", "["),
    ("op_rbracket", "]"),
]
OP_COLORS = plt.get_cmap("tab10")(np.linspace(0, 1, len(OP_COLUMNS)))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_discovery(output: Path) -> None:
    table = rows(DATA / "discovery_times.csv")
    n = [int(row["first_epoch"]) for row in table if row["regime"] == "N"]
    r = [int(row["first_epoch"]) for row in table if row["regime"] == "R" and row["discovered"] == "1"]
    edges = np.linspace(0, 16_000, 17)
    fig, ax = plt.subplots(figsize=(3.25, 2.35))
    ax.hist(n, bins=edges, color=N_COLOR, alpha=0.48, edgecolor="none", label=f"N (n={len(n)})")
    ax.hist(r, bins=edges, color=R_COLOR, alpha=0.48, edgecolor="none", label=f"R (n={len(r)})")
    ax.axvline(np.median(n), color=N_COLOR, linestyle="--", linewidth=1.2)
    ax.axvline(np.median(r), color=R_COLOR, linestyle="--", linewidth=1.2)
    ax.set(xlabel="First discovery epoch", ylabel="Runs", xlim=(0, 16_000))
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def centered_mean(x: np.ndarray, y: np.ndarray, width: float) -> np.ndarray:
    result = np.empty_like(y, dtype=float)
    radius = width / 2.0
    for index, epoch in enumerate(x):
        mask = np.abs(x - epoch) <= radius
        result[index] = float(np.mean(y[mask]))
    return result


def summary_series(path: Path, origin_column: str, value) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows(path):
        if row[origin_column] != "all":
            continue
        grouped[row["run_regime"]].append((float(row["epoch"]), float(value(row))))
    result = {}
    for regime, points in grouped.items():
        points.sort()
        result[regime] = (
            np.asarray([point[0] for point in points]),
            np.asarray([point[1] for point in points]),
        )
    return result


def plot_mechanism(output: Path) -> None:
    activation = summary_series(
        DATA / "activation.csv",
        "partner_origin",
        lambda row: row["mean_seed_hit_fraction"],
    )
    pre_replicator = summary_series(
        DATA / "pre_replicator_availability.csv",
        "fixed_origin",
        lambda row: int(row["total_hits"]) / int(row["seed_count"]),
    )
    discovery = rows(DATA / "discovery_events.csv")

    # The N pre-replicator term is a time-independent uniform-partner baseline. P(t)
    # is displayed with the manuscript's centered 500-epoch smoother.
    n_mean = float(np.mean(pre_replicator["N"][1]))
    pre_replicator_plot = {
        "N": (
            pre_replicator["N"][0],
            np.full_like(pre_replicator["N"][1], n_mean),
        ),
        "R": (
            pre_replicator["R"][0],
            centered_mean(
                pre_replicator["R"][0], pre_replicator["R"][1], 500.0
            ),
        ),
    }

    fig, axes = plt.subplots(3, 1, figsize=(3.3, 5.8), sharex=True)
    products: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for regime, color in (("N", N_COLOR), ("R", R_COLOR)):
        ax_x, ax_y = activation[regime]
        axes[0].plot(ax_x, ax_y, color=color, linewidth=1.25, label=regime)
        px, py = pre_replicator_plot[regime]
        axes[1].plot(px, py, color=color, linewidth=1.25, label=regime)
        common = np.intersect1d(ax_x, px)
        av = np.interp(common, ax_x, ax_y)
        pv = np.interp(common, px, py)
        products[regime] = (common, 30.0 * av * pv)
        axes[2].plot(
            common,
            products[regime][1],
            color=color,
            linewidth=1.25,
            label=f"{regime}: 30 P(t)A(t)",
        )

    centers = np.asarray([float(row["epoch_mid"]) for row in discovery])
    axes[2].plot(
        centers,
        [float(row["n_events_per_run"]) for row in discovery],
        color=N_COLOR,
        linestyle="--",
        marker="o",
        markersize=2.4,
        linewidth=0.9,
        label="N events",
    )
    axes[2].plot(
        centers,
        [float(row["r_events_per_run"]) for row in discovery],
        color=R_COLOR,
        linestyle="--",
        marker="o",
        markersize=2.4,
        linewidth=0.9,
        label="R conditioned events",
    )
    axes[0].set_ylabel("Activation\nprobability A(t)")
    axes[1].set_ylabel("Mean pre-replicator\navailability P(t)")
    axes[2].set_ylabel("Events per run\nper bin")
    axes[2].set_xlabel("Epoch")
    for label, ax in zip(("A", "B", "C"), axes):
        ax.text(0.01, 0.94, label, transform=ax.transAxes, va="top", fontweight="bold")
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.set_xlim(0, 16_000)
    axes[0].legend(frameon=False, ncol=2, loc="lower right")
    axes[2].legend(frameon=False, ncol=2, fontsize=6, loc="upper right")
    fig.subplots_adjust(hspace=0.12, left=0.22, right=0.98, top=0.99, bottom=0.09)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def operator_data():
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows(DATA / "operator_trajectories.csv"):
        if row["regime"] == "R" and row["eligible"] != "1":
            continue
        grouped[(row["regime"], int(row["seed"]))].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["epoch"]))
    return grouped


def plot_operators(output: Path) -> None:
    grouped = operator_data()
    fig, axes = plt.subplots(1, 2, figsize=(6.65, 2.8), sharex=True, sharey=True)
    for ax, regime, title in zip(axes, ("R", "N"), ("R conditioned", "N")):
        regime_runs = {seed: values for (label, seed), values in grouped.items() if label == regime}
        for op_index, (column, symbol) in enumerate(OP_COLUMNS):
            color = OP_COLORS[op_index]
            epoch_values: dict[int, list[float]] = defaultdict(list)
            for values in regime_runs.values():
                xs = np.asarray([int(row["epoch"]) for row in values])
                ys = np.asarray([100.0 * int(row[column]) / int(row["total_bytes"]) for row in values])
                ax.plot(xs, ys, color=color, alpha=0.055, linewidth=0.38)
                for epoch, value in zip(xs, ys):
                    epoch_values[int(epoch)].append(float(value))
            epochs = sorted(epoch_values)
            medians = [float(np.median(epoch_values[epoch])) for epoch in epochs]
            ax.plot(epochs, medians, color=color, linewidth=1.35, label=symbol)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_xlim(0, 16_000)
        ax.grid(alpha=0.15, linewidth=0.6)
    axes[0].set_ylabel("Operator frequency (% of bytes)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=10, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.21, wspace=0.08)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_style()

    plot_discovery(output_dir / "first_replicator_discovery_hist.pdf")
    snapshots = canonical_snapshots(evaluate_modulo(DEFAULT_PARTNER, DEFAULT_FOCAL))
    render_figure(snapshots, output_dir / "toy_canonical_16_progression")
    (output_dir / "toy_canonical_16_progression.png").unlink(missing_ok=True)
    plot_mechanism(output_dir / "mechanism_summary.pdf")
    plot_operators(output_dir / "operator_enrichment_dist.pdf")
    for name in (
        "first_replicator_discovery_hist.pdf",
        "toy_canonical_16_progression.pdf",
        "mechanism_summary.pdf",
        "operator_enrichment_dist.pdf",
    ):
        path = output_dir / name
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Figure was not created: {path}")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
