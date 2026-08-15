#!/usr/bin/env python3
"""Compare bounded CUDA replay outputs with the committed reference data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "activation": ("per_epoch_summary.csv", "hit_cases.csv", "per_partner_summary.csv"),
    "pre-replicator": ("per_epoch_summary.csv", "hit_cases.csv", "per_fixed_summary.csv"),
}


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise AssertionError(f"missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate(experiment: str, run_root: Path) -> None:
    fixture_root = ROOT / "data" / "validation" / "replays" / experiment
    for regime in ("n", "r"):
        actual_root = run_root / f"{regime}_seed0"
        for name in FILES[experiment]:
            expected = csv_rows(fixture_root / regime / name)
            actual = csv_rows(actual_root / name)
            if actual != expected:
                raise AssertionError(
                    f"{experiment} {regime.upper()} {name} differs from the reference output"
                )
        print(f"{experiment} {regime.upper()}: 3/3 CSV outputs match the reference data")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-root", type=Path, required=True)
    parser.add_argument("--pre-replicator-root", type=Path, required=True)
    args = parser.parse_args()
    validate("activation", args.activation_root.resolve())
    validate("pre-replicator", args.pre_replicator_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
