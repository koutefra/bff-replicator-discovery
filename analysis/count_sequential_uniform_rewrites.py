#!/usr/bin/env python3
"""
Count interactions that produce long contiguous rewrites to a single byte.

Event definition for a persisted tape:
- there exists a contiguous block of at least `threshold` positions
- every position in the block changed between consecutive saved states
- all post-state bytes in the block are identical

For random-partner traces, each program is updated independently once per epoch,
so each qualifying tape counts as one qualifying interaction.

For interaction traces, programs are paired within each epoch. Pairings are
reconstructed exactly from the seed and epoch using the simulator's shuffle
logic. An interaction counts as qualifying if either persisted tape in the pair
contains a qualifying rewrite block.
"""

from __future__ import annotations

import argparse
import csv
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


HEADER_STRUCT = struct.Struct("=QQQ")
TAPE_SIZE = 64
MASK64 = (1 << 64) - 1
SPLITMIX_INC = 0x9E3779B97F4A7C15
SPLITMIX_MUL1 = 0xBF58476D1CE4E5B9
SPLITMIX_MUL2 = 0x94D049BB133111EB


@dataclass
class TraceSpec:
    label: str
    mode: str
    seed: int
    path: Path


@dataclass
class TraceResult:
    label: str
    mode: str
    seed: int
    epochs: list[int]
    event_interactions_by_epoch: list[int]
    event_tapes_by_epoch: list[int]
    both_tapes_by_epoch: list[int]
    max_run_by_epoch: list[int]
    total_interactions: int
    total_event_interactions: int
    total_event_tapes: int
    total_both_tapes: int


def splitmix64(x: int) -> int:
    z = (x + SPLITMIX_INC) & MASK64
    z = ((z ^ (z >> 30)) * SPLITMIX_MUL1) & MASK64
    z = ((z ^ (z >> 27)) * SPLITMIX_MUL2) & MASK64
    return (z ^ (z >> 31)) & MASK64


def seeded_value(seed: int, seed2: int) -> int:
    return splitmix64(splitmix64(seed) ^ splitmix64(seed2))


def interaction_shuffle(num_programs: int, seed: int, epoch: int) -> np.ndarray:
    order = np.arange(num_programs, dtype=np.int32)
    for i in range(num_programs - 1, -1, -1):
        j = seeded_value(seed, epoch * num_programs + i) % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def parse_trace(value: str) -> TraceSpec:
    parts = value.split("=", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Expected LABEL=MODE=SEED=PATH, e.g. N=random=100=runs/..."
        )
    label, mode, seed_raw, path_raw = parts
    mode = mode.strip().lower()
    if mode not in {"random", "interaction"}:
        raise argparse.ArgumentTypeError("MODE must be 'random' or 'interaction'")
    try:
        seed = int(seed_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("SEED must be an integer") from exc
    return TraceSpec(label=label.strip(), mode=mode, seed=seed, path=Path(path_raw.strip()))


def load_dump(path: Path) -> tuple[int, np.ndarray]:
    raw = path.read_bytes()
    _, num_programs, epoch_saved = HEADER_STRUCT.unpack(raw[: HEADER_STRUCT.size])
    soup = np.frombuffer(raw, dtype=np.uint8, offset=HEADER_STRUCT.size).reshape(
        num_programs, TAPE_SIZE
    )
    return epoch_saved, soup


def longest_uniform_changed_run(before: np.ndarray, after: np.ndarray) -> tuple[int, int, int]:
    best_len = 0
    best_start = -1
    best_byte = -1
    i = 0
    while i < TAPE_SIZE:
        if before[i] == after[i]:
            i += 1
            continue
        value = int(after[i])
        j = i + 1
        while j < TAPE_SIZE and before[j] != after[j] and int(after[j]) == value:
            j += 1
        run_len = j - i
        if run_len > best_len:
            best_len = run_len
            best_start = i
            best_byte = value
        i = j
    return best_len, best_start, best_byte


def analyze_trace(spec: TraceSpec, threshold: int) -> TraceResult:
    dump_paths = sorted(spec.path.glob("*.dat"))
    if len(dump_paths) < 2:
        raise FileNotFoundError(f"Need at least two dump files in {spec.path}")

    _, prev_soup = load_dump(dump_paths[0])
    num_programs = prev_soup.shape[0]

    epochs: list[int] = []
    event_interactions_by_epoch: list[int] = []
    event_tapes_by_epoch: list[int] = []
    both_tapes_by_epoch: list[int] = []
    max_run_by_epoch: list[int] = []
    total_interactions = 0
    total_event_interactions = 0
    total_event_tapes = 0
    total_both_tapes = 0

    for transition_idx, path in enumerate(dump_paths[1:], start=1):
        epoch_saved, cur_soup = load_dump(path)
        if cur_soup.shape != prev_soup.shape:
            raise ValueError(f"Shape mismatch in {path}: {cur_soup.shape} vs {prev_soup.shape}")

        tape_hits = np.zeros(num_programs, dtype=bool)
        max_run = 0
        for slot in range(num_programs):
            run_len, _, _ = longest_uniform_changed_run(prev_soup[slot], cur_soup[slot])
            if run_len > max_run:
                max_run = run_len
            if run_len >= threshold:
                tape_hits[slot] = True

        if spec.mode == "random":
            event_interactions = int(tape_hits.sum())
            both_tapes = 0
            total_interactions += num_programs
        else:
            order = interaction_shuffle(num_programs, spec.seed, transition_idx)
            event_interactions = 0
            both_tapes = 0
            for i in range(0, num_programs, 2):
                a = int(order[i])
                b = int(order[i + 1])
                a_hit = bool(tape_hits[a])
                b_hit = bool(tape_hits[b])
                if a_hit or b_hit:
                    event_interactions += 1
                if a_hit and b_hit:
                    both_tapes += 1
            total_interactions += num_programs // 2

        event_tapes = int(tape_hits.sum())
        epochs.append(epoch_saved)
        event_interactions_by_epoch.append(event_interactions)
        event_tapes_by_epoch.append(event_tapes)
        both_tapes_by_epoch.append(both_tapes)
        max_run_by_epoch.append(max_run)

        total_event_interactions += event_interactions
        total_event_tapes += event_tapes
        total_both_tapes += both_tapes
        prev_soup = cur_soup

    return TraceResult(
        label=spec.label,
        mode=spec.mode,
        seed=spec.seed,
        epochs=epochs,
        event_interactions_by_epoch=event_interactions_by_epoch,
        event_tapes_by_epoch=event_tapes_by_epoch,
        both_tapes_by_epoch=both_tapes_by_epoch,
        max_run_by_epoch=max_run_by_epoch,
        total_interactions=total_interactions,
        total_event_interactions=total_event_interactions,
        total_event_tapes=total_event_tapes,
        total_both_tapes=total_both_tapes,
    )


def write_summary(path: Path, results: list[TraceResult], threshold: int) -> None:
    rows: list[dict[str, object]] = []
    for result in results:
        rows.append(
            {
                "label": result.label,
                "mode": result.mode,
                "seed": result.seed,
                "threshold": threshold,
                "total_interactions": result.total_interactions,
                "event_interactions": result.total_event_interactions,
                "event_interaction_fraction": result.total_event_interactions
                / max(result.total_interactions, 1),
                "event_tapes": result.total_event_tapes,
                "event_tapes_per_epoch": result.total_event_tapes / max(len(result.epochs), 1),
                "both_tapes_same_interaction": result.total_both_tapes,
                "mean_max_run_per_epoch": float(np.mean(result.max_run_by_epoch)),
                "max_run_observed": max(result.max_run_by_epoch),
                "epochs_with_any_event": sum(v > 0 for v in result.event_interactions_by_epoch),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_timeseries(path: Path, results: list[TraceResult]) -> None:
    rows: list[dict[str, object]] = []
    for result in results:
        for idx, epoch in enumerate(result.epochs):
            rows.append(
                {
                    "label": result.label,
                    "mode": result.mode,
                    "seed": result.seed,
                    "epoch": epoch,
                    "event_interactions": result.event_interactions_by_epoch[idx],
                    "event_tapes": result.event_tapes_by_epoch[idx],
                    "both_tapes_same_interaction": result.both_tapes_by_epoch[idx],
                    "max_run_observed": result.max_run_by_epoch[idx],
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_timeseries(path: Path, results: list[TraceResult], threshold: int) -> None:
    if plt is None:
        return
    fig, axes = plt.subplots(len(results), 1, figsize=(9.5, 3.0 * len(results)), sharex=True)
    if len(results) == 1:
        axes = [axes]
    for ax, result in zip(axes, results):
        ax.plot(result.epochs, result.event_interactions_by_epoch, lw=1.4, color="#1f77b4")
        ax.set_title(f"{result.label} ({result.mode})")
        ax.set_ylabel(f"Interactions with run >= {threshold}")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace",
        action="append",
        type=parse_trace,
        required=True,
        help="Trace specification as LABEL=MODE=SEED=PATH",
    )
    parser.add_argument("--threshold", type=int, default=30)
    parser.add_argument(
        "--summary-out",
        default="analysis_plots/no_mutation/sequential_uniform_rewrite_summary.csv",
    )
    parser.add_argument(
        "--timeseries-out",
        default="analysis_plots/no_mutation/sequential_uniform_rewrite_timeseries.csv",
    )
    parser.add_argument(
        "--plot-out",
        default="analysis_plots/no_mutation/sequential_uniform_rewrite_timeseries",
    )
    args = parser.parse_args()

    results = [analyze_trace(spec, args.threshold) for spec in args.trace]
    write_summary(Path(args.summary_out), results, args.threshold)
    write_timeseries(Path(args.timeseries_out), results)
    plot_timeseries(Path(args.plot_out), results, args.threshold)

    print("Summary:", args.summary_out)
    print("Timeseries:", args.timeseries_out)
    if plt is not None:
        print("Plot:", str(Path(args.plot_out).with_suffix(".pdf")))
    for result in results:
        print(
            result.label,
            "interactions",
            result.total_event_interactions,
            "/",
            result.total_interactions,
            "tapes",
            result.total_event_tapes,
            "both",
            result.total_both_tapes,
            "max_run",
            max(result.max_run_by_epoch),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
