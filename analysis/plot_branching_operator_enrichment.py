#!/usr/bin/env python3
"""
Plot branching operator enrichment trajectories from one N source and multiple R forks.

Layout:
- 1 panel for original N trajectory
- 1 panel per fork (default: 5 forks), where each fork panel starts from the
  original N trajectory up to the fork epoch and then continues with the R fork.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np


COUNT_COLS = [
    "count_op_<",
    "count_op_>",
    "count_op_{",
    "count_op_}",
    "count_op_+",
    "count_op_-",
    "count_op_.",
    "count_op_,",
    "count_op_[",
    "count_op_]",
    "count_zero",
]

CATEGORY_SLICES = {
    "movement": [0, 1, 2, 3],
    "copy": [6, 7],
    "arithmetic": [4, 5],
    "loop": [8, 9],
    "zero": [10],
}

CATEGORY_ORDER = ["movement", "copy", "arithmetic", "loop", "zero"]

CATEGORY_LABELS = {
    "movement": "Movement (< > { })",
    "copy": "Copy (. ,)",
    "arithmetic": "Arithmetic (+ -)",
    "loop": "Loop ([ ])",
    "zero": "Zero",
}

CATEGORY_COLORS = {
    "movement": "#2ca02c",
    "copy": "#ff7f0e",
    "arithmetic": "#9467bd",
    "loop": "#8c564b",
    "zero": "#1f77b4",
}

GRID_COLOR = "#d8dde2"
OPERATOR_COUNT_DENOM = float((2**17) * 64)


def parse_fork_spec(spec: str) -> tuple[int, Path]:
    if ":" not in spec:
        raise ValueError(f"invalid --fork-log '{spec}' (expected EPOCH:PATH)")
    epoch_str, path_str = spec.split(":", 1)
    try:
        epoch = int(epoch_str)
    except ValueError:
        raise ValueError(f"invalid fork epoch '{epoch_str}' in --fork-log '{spec}'") from None
    if epoch <= 0:
        raise ValueError(f"fork epoch must be positive in --fork-log '{spec}'")
    path_str = path_str.strip()
    if not path_str:
        raise ValueError(f"missing path in --fork-log '{spec}'")
    return epoch, Path(path_str)


def resolve_fork_logs(args: argparse.Namespace) -> dict[int, Path]:
    fork_logs: dict[int, Path] = {}
    if args.fork_log:
        for spec in args.fork_log:
            epoch, path = parse_fork_spec(spec)
            if epoch in fork_logs:
                raise ValueError(f"duplicate fork epoch {epoch} in --fork-log")
            fork_logs[epoch] = path
        return fork_logs

    fork_epochs = args.fork_epoch if args.fork_epoch else [2000, 8000]
    for epoch in fork_epochs:
        if epoch <= 0:
            raise ValueError(f"fork epoch must be positive: {epoch}")
        if epoch in fork_logs:
            raise ValueError(f"duplicate fork epoch {epoch} in --fork-epoch")
        try:
            path = Path(args.r_log_template.format(epoch=epoch))
        except Exception as exc:
            raise ValueError(f"failed to format --r-log-template for epoch {epoch}: {exc}") from exc
        fork_logs[epoch] = path
    return fork_logs


def load_categories(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            fields = set(reader.fieldnames or [])
            missing = [col for col in COUNT_COLS if col not in fields]
            if "epoch" not in fields:
                raise ValueError("missing required column: epoch")
            if missing:
                raise ValueError(f"missing required count columns: {','.join(missing)}")
            for row in reader:
                try:
                    epoch = int(row["epoch"])
                except (TypeError, ValueError):
                    continue
                counts: list[float] = []
                ok = True
                for col in COUNT_COLS:
                    try:
                        counts.append(float(row[col]))
                    except (TypeError, ValueError):
                        ok = False
                        break
                if not ok:
                    continue
                cat_vals: dict[str, float] = {}
                for cat, idxs in CATEGORY_SLICES.items():
                    vals = [counts[i] for i in idxs]
                    cat_vals[cat] = sum(vals) / float(len(vals))
                out[epoch] = cat_vals
    except FileNotFoundError:
        raise FileNotFoundError(f"missing file: {path}") from None
    return out


def trim_epoch_map(
    data: dict[int, dict[str, float]], max_epoch: int
) -> dict[int, dict[str, float]]:
    return {epoch: vals for epoch, vals in data.items() if epoch <= max_epoch}


def extend_n_baseline(
    base: dict[int, dict[str, float]],
    extension: dict[int, dict[str, float]],
) -> dict[int, dict[str, float]]:
    if not base:
        return dict(extension)
    out = dict(base)
    max_base = max(base)
    for epoch in sorted(extension):
        if epoch > max_base and epoch not in out:
            out[epoch] = extension[epoch]
    return out


def build_fork_panel_map(
    n_map: dict[int, dict[str, float]],
    r_map: dict[int, dict[str, float]],
    fork_epoch: int,
    max_epoch: int,
) -> dict[int, dict[str, float]]:
    panel: dict[int, dict[str, float]] = {}
    for epoch in sorted(n_map):
        if epoch > max_epoch:
            continue
        if epoch <= fork_epoch:
            panel[epoch] = n_map[epoch]
    for epoch in sorted(r_map):
        if epoch > max_epoch:
            continue
        if epoch > fork_epoch:
            panel[epoch] = r_map[epoch]
    return panel


def panel_limits(panels: list[dict[int, dict[str, float]]]) -> tuple[float, float]:
    vals: list[float] = []
    for panel in panels:
        for epoch_vals in panel.values():
            for cat in CATEGORY_ORDER:
                val = epoch_vals.get(cat)
                if val is not None and math.isfinite(val):
                    vals.append(float(val))
    if not vals:
        return 0.0, 1.0
    lo = min(vals)
    hi = max(vals)
    if math.isclose(lo, hi):
        pad = max(1e-3, abs(hi) * 0.05 if hi != 0.0 else 0.1)
    else:
        pad = (hi - lo) * 0.05
    lo2 = max(0.0, lo - pad)
    hi2 = hi + pad
    if math.isclose(lo2, hi2):
        hi2 = lo2 + 1e-3
    return lo2, hi2


def plot_panel(
    ax,
    data: dict[int, dict[str, float]],
    title: str,
    fork_epoch: int | None = None,
    title_loc: str = "upper left",
) -> None:
    epochs = sorted(data)
    for cat in CATEGORY_ORDER:
        xs: list[int] = []
        ys: list[float] = []
        for epoch in epochs:
            val = data[epoch].get(cat)
            if val is None or not math.isfinite(val):
                continue
            xs.append(epoch)
            ys.append(float(val) / OPERATOR_COUNT_DENOM)
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            color=CATEGORY_COLORS[cat],
            linewidth=1.9,
            label=CATEGORY_LABELS[cat],
        )
    if fork_epoch is not None:
        ax.axvline(
            fork_epoch,
            color="#8a8a8a",
            linestyle=(0, (3, 3)),
            linewidth=2.0,
            alpha=0.9,
            zorder=1,
        )
    text_x = 0.98 if title_loc == "upper right" else 0.02
    text_ha = "right" if title_loc == "upper right" else "left"
    ax.text(
        text_x,
        0.98,
        title,
        transform=ax.transAxes,
        ha=text_ha,
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#1f2937",
    )
    ax.set_xlabel("Epoch")
    ax.grid(axis="y", color=GRID_COLOR, linestyle="-", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def main() -> int:
    global OPERATOR_COUNT_DENOM
    parser = argparse.ArgumentParser(
        description=(
            "Plot branching operator enrichment with one N panel and one panel per R fork."
        )
    )
    parser.add_argument(
        "--n-log",
        default="runs/no_mutation/random/run_3_dumps/logs/n.log",
        help="Baseline N log",
    )
    parser.add_argument(
        "--n-extension-log",
        default="runs/no_mutation/random/log_3.log",
        help="Optional full-length N log used to extend baseline beyond n-log",
    )
    parser.add_argument(
        "--fork-log",
        action="append",
        default=None,
        help=(
            "Fork branch as EPOCH:PATH (repeat for multiple branches). "
            "Overrides --fork-epoch/--r-log-template when set."
        ),
    )
    parser.add_argument(
        "--fork-epoch",
        type=int,
        action="append",
        default=None,
        help="Fork epoch (repeat for multiple branches; uses --r-log-template for each path).",
    )
    parser.add_argument(
        "--r-log-template",
        default="runs/no_mutation/random/run_3_r_from_{epoch}/logs/r.log",
        help="Template used with --fork-epoch, must include {epoch}.",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=16000,
        help="Maximum epoch to plot",
    )
    parser.add_argument(
        "--num-programs",
        type=int,
        default=2**17,
        help="Population size used to normalize operator counts.",
    )
    parser.add_argument(
        "--output",
        default="analysis_plots/no_mutation/branching_operator_enrichment_run3_2000_8000.pdf",
        help="Output plot path",
    )
    args = parser.parse_args()
    if args.num_programs <= 0:
        parser.error("--num-programs must be positive")
    OPERATOR_COUNT_DENOM = float(args.num_programs * 64)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.stderr.write("matplotlib not available, skipping plot.\n")
        return 1

    try:
        n_map = load_categories(Path(args.n_log))
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    n_map = trim_epoch_map(n_map, args.max_epoch)

    if args.n_extension_log and (not n_map or max(n_map) < args.max_epoch):
        try:
            ext_map = load_categories(Path(args.n_extension_log))
        except (FileNotFoundError, ValueError) as exc:
            sys.stderr.write(f"{exc}\n")
            return 1
        ext_map = trim_epoch_map(ext_map, args.max_epoch)
        n_map = extend_n_baseline(n_map, ext_map)

    if not n_map:
        sys.stderr.write("No usable N baseline points found.\n")
        return 1

    try:
        fork_logs = resolve_fork_logs(args)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    if not fork_logs:
        sys.stderr.write("No fork branches configured.\n")
        return 1

    panels: list[tuple[str, dict[int, dict[str, float]], int | None]] = []
    for fork_epoch in sorted(fork_logs):
        if fork_epoch not in n_map:
            sys.stderr.write(f"N baseline missing fork epoch {fork_epoch}; cannot attach branch.\n")
            return 1
        log_path = fork_logs[fork_epoch]
        try:
            r_map = load_categories(log_path)
        except (FileNotFoundError, ValueError) as exc:
            sys.stderr.write(f"{exc}\n")
            return 1
        r_map = trim_epoch_map(r_map, args.max_epoch)
        if not r_map:
            sys.stderr.write(f"No usable R branch points found in {log_path}\n")
            return 1
        panel_map = build_fork_panel_map(n_map, r_map, fork_epoch, args.max_epoch)
        panels.append((f"Fork @ {fork_epoch}", panel_map, fork_epoch))

    if not panels:
        sys.stderr.write("No fork panels available.\n")
        return 1

    from matplotlib.ticker import FuncFormatter, MaxNLocator
    from matplotlib.transforms import ScaledTranslation

    with plt.rc_context(
        {
            "figure.figsize": (3.9 * len(panels), 3.6),
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "axes.linewidth": 1.2,
        }
    ):
        fig, axes = plt.subplots(1, len(panels), sharex=True, sharey=True)
        axes_flat = np.atleast_1d(axes).ravel().tolist()
        y_min, y_max = panel_limits([panel_map for _, panel_map, _ in panels])

        for idx, (title, panel_map, fork_epoch) in enumerate(panels):
            ax = axes_flat[idx]
            plot_panel(
                ax,
                panel_map,
                title,
                fork_epoch=fork_epoch,
                title_loc="upper right" if idx == 0 else "upper left",
            )
            ax.set_xlim(1, args.max_epoch)
            ax.set_ylim(y_min / OPERATOR_COUNT_DENOM, y_max / OPERATOR_COUNT_DENOM)

        formatter = FuncFormatter(lambda value, _pos: f"{value * 100:.1f}%")
        for ax in axes_flat:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_formatter(formatter)
            ax.tick_params(axis="y", pad=6)
            for tick in ax.get_yticklabels():
                tick.set_rotation(90)
                tick.set_va("center")
                tick.set_ha("center")
                tick.set_transform(
                    tick.get_transform()
                    + ScaledTranslation(0.0, 0.4 / 72.0, fig.dpi_scale_trans)
                )

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.03)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    print("Plot written to:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
