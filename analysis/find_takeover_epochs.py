#!/usr/bin/env python3
"""
Scan interaction logs and find the takeover epoch per seed.

Definition:
- The takeover epoch is the earliest epoch that has at least one replicator
  with score >= threshold, and the next 1000 epochs (including itself) each
  also have at least one replicator with score >= threshold.
- Also reports the score(s) of replicators present at that takeover epoch.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
TAKEOVER_WINDOW = 500


@dataclass(frozen=True)
class TakeoverClassification:
    takeover_epochs: dict[int, int]
    takeover_scores: dict[int, list[float]]
    non_takeover: set[int]

    def classify_seed(self, seed: int | None) -> str | None:
        if seed is None:
            return None
        if seed in self.non_takeover:
            return "non_takeover"
        if seed in self.takeover_epochs:
            return "takeover"
        return None


def parse_scores(value: str) -> list[float]:
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = None
    scores: list[float] = []
    if isinstance(parsed, list):
        for item in parsed:
            try:
                scores.append(float(item))
            except (TypeError, ValueError):
                continue
        return scores
    return [float(x) for x in NUMBER_RE.findall(value)]


def find_takeover_epoch(rows: list[tuple[int, bool, list[float]]]) -> int | None:
    if not rows:
        return None
    window = TAKEOVER_WINDOW
    if len(rows) < window:
        return None
    flags = [has_high for _, has_high, _ in rows]
    for start in range(0, len(flags) - window + 1):
        if not flags[start]:
            continue
        if all(flags[start : start + window]):
            return rows[start][0]
    return None


def parse_run_seed(path: Path, fallback: int) -> int:
    match = re.search(r"(\d+)", path.stem)
    if match:
        return int(match.group(1))
    return fallback


def load_epoch_flags(
    log_path: Path, threshold: float
) -> list[tuple[int, bool, list[float]]]:
    rows: list[tuple[int, bool, list[float]]] = []
    try:
        with log_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    epoch = int(row.get("epoch", ""))
                except (TypeError, ValueError):
                    continue
                scores = parse_scores(row.get("selfrep_scores", ""))
                high_scores = [score for score in scores if score >= threshold]
                has_high = bool(high_scores)
                rows.append((epoch, has_high, high_scores))
    except FileNotFoundError:
        return []
    return rows


def format_set(values: set[int]) -> str:
    return "{" + ", ".join(str(v) for v in sorted(values)) + "}"


def ensure_csv_field_size_limit() -> None:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)


def classify_takeover_runs(
    interaction_dir: Path | str,
    pattern: str = "log_*.log",
    threshold: float = 0.0,
) -> TakeoverClassification:
    ensure_csv_field_size_limit()

    root = Path(interaction_dir)
    paths = sorted(root.glob(pattern))
    takeover_epochs: dict[int, int] = {}
    takeover_scores: dict[int, list[float]] = {}
    non_takeover: set[int] = set()

    for idx, path in enumerate(paths):
        seed = parse_run_seed(path, idx)
        rows = load_epoch_flags(path, threshold)
        takeover_epoch = find_takeover_epoch(rows)
        if takeover_epoch is None:
            non_takeover.add(seed)
            continue

        takeover_epochs[seed] = takeover_epoch
        for epoch, _, high_scores in rows:
            if epoch == takeover_epoch:
                takeover_scores[seed] = high_scores
                break

    return TakeoverClassification(
        takeover_epochs=dict(sorted(takeover_epochs.items())),
        takeover_scores=dict(sorted(takeover_scores.items())),
        non_takeover=non_takeover,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find takeover epochs for interaction log runs."
    )
    parser.add_argument(
        "--interaction-dir",
        default="runs/no_mutation/interaction",
        help="Directory containing log_*.log files",
    )
    parser.add_argument(
        "--pattern",
        default="log_*.log",
        help="Glob pattern for logs within interaction-dir",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Replicator score threshold for takeover detection",
    )
    args = parser.parse_args()

    classification = classify_takeover_runs(
        args.interaction_dir,
        pattern=args.pattern,
        threshold=args.threshold,
    )
    r_null_count = len(classification.non_takeover)
    r_post_count = len(classification.takeover_epochs)
    print(f"takeover_epochs = {classification.takeover_epochs}")
    print(f"takeover_scores = {classification.takeover_scores}")
    print(f"non_takeover = {format_set(classification.non_takeover)}")
    print(f"R_null_count = {r_null_count}")
    print(f"R_post_count = {r_post_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
