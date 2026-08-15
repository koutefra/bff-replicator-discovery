#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(x: float) -> str:
    return f"{x:.12g}"


def safe_stats(values: list[float]) -> tuple[float, float, float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ordered = sorted(values)
    mean = statistics.fmean(ordered)
    variance = statistics.variance(ordered) if len(ordered) > 1 else 0.0
    std = math.sqrt(variance)
    if len(ordered) == 1:
        q25 = ordered[0]
        median = ordered[0]
        q75 = ordered[0]
    else:
        quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
        q25 = quartiles[0]
        median = statistics.median(ordered)
        q75 = quartiles[2]
    return mean, variance, std, q25, median, q75


def summarize_epoch_profile(epoch_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in epoch_rows:
        grouped[(str(row["run_regime"]), str(row["partner_origin"]))].append(row)

    out: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        total_hits = sum(int(row["total_hits"]) for row in rows)
        if total_hits:
            weighted_mean_epoch = sum(int(row["epoch"]) * int(row["total_hits"]) for row in rows) / total_hits
            weighted_var = sum(
                int(row["total_hits"]) * (int(row["epoch"]) - weighted_mean_epoch) ** 2
                for row in rows
            ) / total_hits
            weighted_sd_epoch = math.sqrt(weighted_var)
            cumulative_targets = [0.25 * total_hits, 0.5 * total_hits, 0.75 * total_hits]
            cumulative = 0
            q_epochs: list[int] = []
            target_idx = 0
            for row in sorted(rows, key=lambda r: int(r["epoch"])):
                cumulative += int(row["total_hits"])
                while target_idx < len(cumulative_targets) and cumulative >= cumulative_targets[target_idx]:
                    q_epochs.append(int(row["epoch"]))
                    target_idx += 1
            while len(q_epochs) < 3:
                q_epochs.append(int(rows[-1]["epoch"]))
        else:
            weighted_mean_epoch = 0.0
            weighted_sd_epoch = 0.0
            q_epochs = [0, 0, 0]

        peak = max(rows, key=lambda r: float(r["aggregate_hit_fraction"]))
        out.append(
            {
                "run_regime": key[0],
                "partner_origin": key[1],
                "total_hits": total_hits,
                "weighted_hit_epoch_mean": fmt(weighted_mean_epoch),
                "weighted_hit_epoch_sd": fmt(weighted_sd_epoch),
                "hit_q25_epoch": q_epochs[0],
                "hit_median_epoch": q_epochs[1],
                "hit_q75_epoch": q_epochs[2],
                "peak_epoch": peak["epoch"],
                "peak_aggregate_hit_fraction": peak["aggregate_hit_fraction"],
                "peak_mean_seed_hit_fraction": peak["mean_seed_hit_fraction"],
                "peak_seed_hit_fraction_sd": peak["seed_hit_fraction_sd"],
            }
        )
    return out


def write_summary_text(
    path: Path,
    aggregate_rows: list[dict[str, object]],
    profile_rows: list[dict[str, object]],
) -> None:
    agg_index = {(str(r["run_regime"]), str(r["partner_origin"])): r for r in aggregate_rows}
    prof_index = {(str(r["run_regime"]), str(r["partner_origin"])): r for r in profile_rows}
    regimes = sorted({key[0] for key in agg_index})

    lines = []
    lines.append(
        "Activation probability recomputed with union-over-partners semantics: "
        "one program counts once per epoch, and is a hit if any partner in the selected set produces a replicator."
    )
    lines.append("")
    for regime in regimes:
        all_key = (regime, "all")
        if all_key not in agg_index:
            continue
        all_row = agg_index[all_key]
        lines.append(
            f"{regime} all partners: hit_fraction={all_row['aggregate_hit_fraction']} "
            f"mean_seed={all_row['mean_seed_hit_fraction']} sd_seed={all_row['seed_hit_fraction_sd']}"
        )
        for origin in ("N-derived", "R-derived"):
            key = (regime, origin)
            if key not in agg_index or key not in prof_index:
                continue
            row = agg_index[key]
            prof = prof_index[key]
            lines.append(
                f"{regime} conditioned on {origin}: hit_fraction={row['aggregate_hit_fraction']} "
                f"mean_seed={row['mean_seed_hit_fraction']} sd_seed={row['seed_hit_fraction_sd']} "
                f"peak_epoch={prof['peak_epoch']} peak_fraction={prof['peak_aggregate_hit_fraction']}"
            )
        all_prof_key = (regime, "all")
        if all_prof_key in prof_index:
            all_prof = prof_index[all_prof_key]
            lines.append(
                f"{regime} time profile all partners: peak_epoch={all_prof['peak_epoch']} "
                f"peak_fraction={all_prof['peak_aggregate_hit_fraction']} "
                f"weighted_hit_epoch_mean={all_prof['weighted_hit_epoch_mean']}"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_run_dirs(root: Path) -> list[Path]:
    out = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "per_epoch_summary.csv").exists() and (path / "hit_cases.csv").exists():
            out.append(path)
    if not out:
        raise SystemExit(f"No completed run directories found in {root}")
    return out


def load_epoch_denominators(path: Path) -> tuple[str, int, list[int], dict[int, int]]:
    epoch_to_tested: dict[int, int] = {}
    regime = ""
    seed = -1
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not regime:
                regime = row["regime"]
                seed = int(row["seed"])
            epoch = int(row["epoch"])
            tested = int(row["tested_at_epoch"])
            prev = epoch_to_tested.get(epoch)
            if prev is None:
                epoch_to_tested[epoch] = tested
            elif prev != tested:
                raise SystemExit(
                    f"Inconsistent tested_at_epoch within {path} for epoch {epoch}: {prev} vs {tested}"
                )
    epochs = sorted(epoch_to_tested)
    if not epochs:
        raise SystemExit(f"No epochs found in {path}")
    return regime, seed, epochs, epoch_to_tested


def stream_hit_unions(
    hit_cases_path: Path,
    epochs: list[int],
    num_programs: int,
) -> dict[str, np.ndarray]:
    epoch_to_idx = {epoch: idx for idx, epoch in enumerate(epochs)}
    unions = {
        "all": np.zeros((len(epochs), num_programs), dtype=np.bool_),
        "N-derived": np.zeros((len(epochs), num_programs), dtype=np.bool_),
        "R-derived": np.zeros((len(epochs), num_programs), dtype=np.bool_),
    }

    with hit_cases_path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        next(fh, None)
        for line in fh:
            parts = line.rstrip("\n").split(",", 6)
            if len(parts) < 6:
                continue
            epoch = int(parts[2])
            epoch_idx = epoch_to_idx.get(epoch)
            if epoch_idx is None:
                continue
            soup_program_index = int(parts[3])
            if soup_program_index < 0 or soup_program_index >= num_programs:
                continue
            partner_group = parts[5].strip().strip('"')
            unions["all"][epoch_idx, soup_program_index] = True
            if partner_group in unions:
                unions[partner_group][epoch_idx, soup_program_index] = True

    return {origin: arr.sum(axis=1, dtype=np.int64) for origin, arr in unions.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute activation probability using union-over-partners semantics: "
            "one program counts once per epoch, and is a hit if any selected partner activates it."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True, help="Replay run root.")
    parser.add_argument(
        "--out-prefix",
        type=Path,
        required=True,
        help="Output prefix, for example analysis_plots/no_mutation/foo",
    )
    args = parser.parse_args()

    run_dirs = collect_run_dirs(args.run_root)
    partner_origins = ("all", "N-derived", "R-derived")

    run_summary_rows: list[dict[str, object]] = []
    aggregate_acc: dict[tuple[str, str], dict[str, object]] = defaultdict(
        lambda: {"hits": 0, "tested": 0, "fractions": []}
    )
    epoch_run_acc: dict[tuple[str, int, str, int], dict[str, int]] = defaultdict(
        lambda: {"hits": 0, "tested": 0}
    )

    for run_dir in run_dirs:
        regime, seed, epochs, epoch_to_tested = load_epoch_denominators(run_dir / "per_epoch_summary.csv")
        num_programs = max(epoch_to_tested.values())
        union_counts = stream_hit_unions(run_dir / "hit_cases.csv", epochs, num_programs)

        for origin in partner_origins:
            total_hits = int(np.sum(union_counts[origin], dtype=np.int64))
            total_tested = sum(epoch_to_tested[epoch] for epoch in epochs)
            frac = total_hits / total_tested if total_tested else 0.0
            run_summary_rows.append(
                {
                    "run_regime": regime,
                    "seed": seed,
                    "partner_origin": origin,
                    "total_hits": total_hits,
                    "total_tested": total_tested,
                    "hit_fraction": fmt(frac),
                }
            )
            aggregate_acc[(regime, origin)]["hits"] += total_hits
            aggregate_acc[(regime, origin)]["tested"] += total_tested
            aggregate_acc[(regime, origin)]["fractions"].append(frac)

            origin_counts = union_counts[origin]
            for epoch, hits in zip(epochs, origin_counts.tolist()):
                acc = epoch_run_acc[(regime, seed, origin, epoch)]
                acc["hits"] = hits
                acc["tested"] = epoch_to_tested[epoch]

    aggregate_rows: list[dict[str, object]] = []
    for (run_regime, origin), stats in sorted(aggregate_acc.items()):
        mean, variance, std, q25, median, q75 = safe_stats(stats["fractions"])
        aggregate_rows.append(
            {
                "run_regime": run_regime,
                "partner_origin": origin,
                "seed_count": len(stats["fractions"]),
                "total_hits": stats["hits"],
                "total_tested": stats["tested"],
                "aggregate_hit_fraction": fmt(stats["hits"] / stats["tested"] if stats["tested"] else 0.0),
                "mean_seed_hit_fraction": fmt(mean),
                "seed_hit_fraction_variance": fmt(variance),
                "seed_hit_fraction_sd": fmt(std),
                "seed_hit_fraction_q25": fmt(q25),
                "seed_hit_fraction_median": fmt(median),
                "seed_hit_fraction_q75": fmt(q75),
            }
        )

    epoch_seed_map: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    epoch_total_map: dict[tuple[str, str, int], dict[str, int]] = defaultdict(
        lambda: {"hits": 0, "tested": 0}
    )
    for (run_regime, seed, origin, epoch), stats in sorted(epoch_run_acc.items()):
        frac = stats["hits"] / stats["tested"] if stats["tested"] else 0.0
        epoch_seed_map[(run_regime, origin, epoch)].append(frac)
        epoch_total_map[(run_regime, origin, epoch)]["hits"] += stats["hits"]
        epoch_total_map[(run_regime, origin, epoch)]["tested"] += stats["tested"]

    epoch_rows_out: list[dict[str, object]] = []
    epoch_run_rows_out: list[dict[str, object]] = []
    for key in sorted(epoch_seed_map):
        run_regime, origin, epoch = key
        fractions = epoch_seed_map[key]
        totals = epoch_total_map[key]
        mean, variance, std, q25, median, q75 = safe_stats(fractions)
        epoch_rows_out.append(
            {
                "run_regime": run_regime,
                "partner_origin": origin,
                "epoch": epoch,
                "seed_count": len(fractions),
                "total_hits": totals["hits"],
                "total_tested": totals["tested"],
                "aggregate_hit_fraction": fmt(totals["hits"] / totals["tested"] if totals["tested"] else 0.0),
                "mean_seed_hit_fraction": fmt(mean),
                "seed_hit_fraction_variance": fmt(variance),
                "seed_hit_fraction_sd": fmt(std),
                "seed_hit_fraction_q25": fmt(q25),
                "seed_hit_fraction_median": fmt(median),
                "seed_hit_fraction_q75": fmt(q75),
            }
        )

    for (run_regime, seed, origin, epoch), stats in sorted(epoch_run_acc.items()):
        frac = stats["hits"] / stats["tested"] if stats["tested"] else 0.0
        epoch_run_rows_out.append(
            {
                "run_regime": run_regime,
                "seed": seed,
                "partner_origin": origin,
                "epoch": epoch,
                "hits_at_epoch": stats["hits"],
                "tested_at_epoch": stats["tested"],
                "hit_fraction": fmt(frac),
            }
        )

    profile_rows = summarize_epoch_profile(epoch_rows_out)

    write_csv(
        args.out_prefix.parent / f"{args.out_prefix.name}_run_summary.csv",
        ["run_regime", "seed", "partner_origin", "total_hits", "total_tested", "hit_fraction"],
        run_summary_rows,
    )
    write_csv(
        args.out_prefix.parent / f"{args.out_prefix.name}_aggregate_summary.csv",
        [
            "run_regime",
            "partner_origin",
            "seed_count",
            "total_hits",
            "total_tested",
            "aggregate_hit_fraction",
            "mean_seed_hit_fraction",
            "seed_hit_fraction_variance",
            "seed_hit_fraction_sd",
            "seed_hit_fraction_q25",
            "seed_hit_fraction_median",
            "seed_hit_fraction_q75",
        ],
        aggregate_rows,
    )
    write_csv(
        args.out_prefix.parent / f"{args.out_prefix.name}_epoch_summary.csv",
        [
            "run_regime",
            "partner_origin",
            "epoch",
            "seed_count",
            "total_hits",
            "total_tested",
            "aggregate_hit_fraction",
            "mean_seed_hit_fraction",
            "seed_hit_fraction_variance",
            "seed_hit_fraction_sd",
            "seed_hit_fraction_q25",
            "seed_hit_fraction_median",
            "seed_hit_fraction_q75",
        ],
        epoch_rows_out,
    )
    write_csv(
        args.out_prefix.parent / f"{args.out_prefix.name}_epoch_run_rows.csv",
        [
            "run_regime",
            "seed",
            "partner_origin",
            "epoch",
            "hits_at_epoch",
            "tested_at_epoch",
            "hit_fraction",
        ],
        epoch_run_rows_out,
    )
    write_csv(
        args.out_prefix.parent / f"{args.out_prefix.name}_epoch_profile_summary.csv",
        [
            "run_regime",
            "partner_origin",
            "total_hits",
            "weighted_hit_epoch_mean",
            "weighted_hit_epoch_sd",
            "hit_q25_epoch",
            "hit_median_epoch",
            "hit_q75_epoch",
            "peak_epoch",
            "peak_aggregate_hit_fraction",
            "peak_mean_seed_hit_fraction",
            "peak_seed_hit_fraction_sd",
        ],
        profile_rows,
    )
    write_summary_text(
        args.out_prefix.parent / f"{args.out_prefix.name}_summary.txt",
        aggregate_rows,
        profile_rows,
    )


if __name__ == "__main__":
    main()
