#!/usr/bin/env python3
"""Export compact paper data from the full CuBFF raw logs.

The full logs are intentionally not committed: the N/R CSV logs alone are
hundreds of megabytes.  This exporter retains every value needed by the paper
figures and validation checks.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


OPS = {
    "op_lt": "count_op_<",
    "op_gt": "count_op_>",
    "op_lbrace": "count_op_{",
    "op_rbrace": "count_op_}",
    "op_plus": "count_op_+",
    "op_minus": "count_op_-",
    "op_dot": "count_op_.",
    "op_comma": "count_op_,",
    "op_lbracket": "count_op_[",
    "op_rbracket": "count_op_]",
    "zero": "count_zero",
}
TAKEOVER_WINDOW = 500
TAKEOVER_OVERRIDES = {19: 4770, 89: 750, 96: 600}
SCORE_THRESHOLD = 60.0
MAX_EPOCH = 16_000
SOUP_SIZE = 131_072
TAPE_SIZE = 64


def seed_from_path(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot read seed from {path}")
    return int(match.group(1))


def parse_scores(text: str) -> list[float]:
    try:
        value = ast.literal_eval(text or "[]")
    except (SyntaxError, ValueError):
        return []
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            pass
    return result


def parse_tapes(text: str) -> list[str]:
    try:
        value = ast.literal_eval(text or "[]")
    except (SyntaxError, ValueError):
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


def robust(row: dict[str, str]) -> bool:
    return any(score >= SCORE_THRESHOLD for score in parse_scores(row.get("selfrep_scores", "")))


def present_at_threshold(row: dict[str, str], threshold: float) -> bool:
    return any(score >= threshold for score in parse_scores(row.get("selfrep_scores", "")))


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def takeover_epoch(path: Path, threshold: float = 0.0) -> int | None:
    flags: list[tuple[int, bool]] = []
    for row in read_rows(path):
        epoch = int(row["epoch"])
        if epoch > MAX_EPOCH:
            break
        flags.append((epoch, present_at_threshold(row, threshold)))
    for start in range(0, len(flags) - TAKEOVER_WINDOW + 1):
        if flags[start][1] and all(flag for _, flag in flags[start : start + TAKEOVER_WINDOW]):
            return flags[start][0]
    return None


def discovery_and_operator_tables(raw_root: Path, stride: int):
    discovery_rows: list[dict[str, object]] = []
    operator_rows: list[dict[str, object]] = []
    event_epochs: dict[str, list[tuple[int, int]]] = defaultdict(list)
    takeover_raw: dict[int, int] = {}
    takeover_for_operators: dict[int, int] = {}

    for regime, dirname in (("N", "random"), ("R", "interaction")):
        log_dir = raw_root / "runs" / "no_mutation" / dirname
        paths = sorted(log_dir.glob("log_*.log"), key=seed_from_path)
        if len(paths) != 100:
            raise RuntimeError(f"Expected 100 {regime} logs in {log_dir}, found {len(paths)}")
        if regime == "R":
            for path in paths:
                seed = seed_from_path(path)
                # Match the released analyses: takeover uses any reported
                # replicator (threshold 0); discovery itself requires score 60.
                found = takeover_epoch(path, threshold=0.0)
                if found is not None:
                    takeover_raw[seed] = found
                    takeover_for_operators[seed] = TAKEOVER_OVERRIDES.get(seed, found)

        for path in paths:
            seed = seed_from_path(path)
            first_epoch: int | None = None
            previous = False
            event_cutoff = takeover_raw.get(seed) if regime == "R" else None
            operator_cutoff = takeover_for_operators.get(seed) if regime == "R" else None
            for row in read_rows(path):
                epoch = int(row["epoch"])
                if epoch > MAX_EPOCH:
                    break
                present = robust(row)
                if first_epoch is None and present:
                    first_epoch = epoch
                if present and not previous and (event_cutoff is None or epoch <= event_cutoff):
                    event_epochs[regime].append((seed, epoch))
                previous = present

                if epoch == 1 or epoch % stride == 0:
                    item: dict[str, object] = {
                        "regime": regime,
                        "seed": seed,
                        "epoch": epoch,
                        "takeover_epoch": operator_cutoff if operator_cutoff is not None else "",
                        "eligible": int(operator_cutoff is None or epoch <= operator_cutoff),
                        "total_bytes": int(row["soup_size"]) * TAPE_SIZE,
                    }
                    for output_name, raw_name in OPS.items():
                        item[output_name] = int(row[raw_name])
                    operator_rows.append(item)

            discovery_rows.append(
                {
                    "regime": regime,
                    "seed": seed,
                    "discovered": int(first_epoch is not None),
                    "first_epoch": first_epoch if first_epoch is not None else "",
                }
            )
    return discovery_rows, operator_rows, event_epochs, takeover_raw, takeover_for_operators


def uniform_table(raw_root: Path):
    rows: list[dict[str, object]] = []
    samples = 0
    paths = sorted((raw_root / "runs" / "reinit").glob("log_*.log"), key=seed_from_path)
    if len(paths) != 100:
        raise RuntimeError(f"Expected 100 U logs, found {len(paths)}")
    for path in paths:
        seed = seed_from_path(path)
        for row in read_rows(path):
            epoch = int(row["epoch"])
            if epoch > MAX_EPOCH:
                break
            samples += int(row["soup_size"])
            scores = parse_scores(row.get("selfrep_scores", ""))
            tapes = parse_tapes(row.get("selfrep_tapes", ""))
            for index, score in enumerate(scores):
                if score >= SCORE_THRESHOLD:
                    rows.append(
                        {
                            "seed": seed,
                            "epoch": epoch,
                            "score": score,
                            "tape_hex": tapes[index] if index < len(tapes) else "",
                        }
                    )
    return rows, samples


def binned_events(event_epochs: dict[str, list[tuple[int, int]]], takeover: dict[int, int]):
    bins = 12
    edges = [MAX_EPOCH * i / bins for i in range(bins + 1)]
    rows: list[dict[str, object]] = []
    for index in range(bins):
        lo, hi = edges[index], edges[index + 1]
        n_count = sum(lo < epoch <= hi for _, epoch in event_epochs["N"])
        r_count = sum(lo < epoch <= hi for _, epoch in event_epochs["R"])
        r_exposure = sum(1 for seed in range(100) if seed not in takeover or takeover[seed] > lo)
        rows.append(
            {
                "bin": index,
                "epoch_start": lo,
                "epoch_end": hi,
                "epoch_mid": (lo + hi) / 2,
                "n_events": n_count,
                "n_exposure": 100,
                "n_events_per_run": n_count / 100,
                "r_events": r_count,
                "r_exposure": r_exposure,
                "r_events_per_run": r_count / r_exposure,
            }
        )
    return rows


def copy_table(source: Path, destination: Path) -> None:
    destination.write_bytes(source.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True, help="Root of the full source repository")
    parser.add_argument("--output", type=Path, default=Path("data/paper"))
    parser.add_argument("--operator-stride", type=int, default=100)
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    discovery, operators, events, takeover, takeover_for_operators = discovery_and_operator_tables(
        raw_root, args.operator_stride
    )
    uniform, uniform_samples = uniform_table(raw_root)
    event_bins = binned_events(events, takeover)

    write_csv(output / "discovery_times.csv", discovery)
    write_csv(output / "operator_trajectories.csv", operators)
    write_csv(output / "uniform_replicators.csv", uniform)
    write_csv(output / "discovery_events.csv", event_bins)

    n_epochs = [int(row["first_epoch"]) for row in discovery if row["regime"] == "N"]
    r_epochs = [int(row["first_epoch"]) for row in discovery if row["regime"] == "R" and row["discovered"]]
    n_epochs.sort()
    r_epochs.sort()
    p_program = len(uniform) / uniform_samples
    p_epoch = 1.0 - (1.0 - p_program) ** SOUP_SIZE
    uniform_median = math.log(0.5) / math.log1p(-p_epoch)
    rewrite_path = output / "uniform_rewrite_summary.csv"
    rewrite_rows = list(read_rows(rewrite_path)) if rewrite_path.exists() else []
    rewrite_ratio = None
    if len(rewrite_rows) == 2:
        values = {row["label"]: float(row["event_interaction_fraction"]) for row in rewrite_rows}
        rewrite_ratio = values["N"] / values["R"]

    expected = {
        "configuration": {
            "program_length": TAPE_SIZE,
            "soup_size": SOUP_SIZE,
            "epochs": MAX_EPOCH,
            "runs_per_regime": 100,
            "replication_score_threshold": SCORE_THRESHOLD,
            "takeover_window_epochs": TAKEOVER_WINDOW,
        },
        "uniform": {
            "samples": uniform_samples,
            "replicators": len(uniform),
            "per_program_probability": p_program,
            "median_epochs": uniform_median,
        },
        "discovery": {
            "N_discovered": len(n_epochs),
            "N_median_epoch": (n_epochs[49] + n_epochs[50]) / 2,
            "R_discovered": len(r_epochs),
            "R_median_epoch": r_epochs[len(r_epochs) // 2],
        },
        "takeover": {
            "runs": len(takeover),
            "non_takeover_runs": 100 - len(takeover),
            "mean_epoch": sum(takeover.values()) / len(takeover),
            "figure_cutoff_overrides": TAKEOVER_OVERRIDES,
        },
        "discovery_events": {
            "N_total": len(events["N"]),
            "R_conditioned_total": len(events["R"]),
            "N_events_per_run": len(events["N"]) / 100,
            "R_exposure_normalized_sum": sum(float(row["r_events_per_run"]) for row in event_bins),
        },
        "uniform_rewrite_rate_ratio_N_over_R": rewrite_ratio,
    }
    (output / "expected_results.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Wrote compact paper data to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
