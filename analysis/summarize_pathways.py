#!/usr/bin/env python3
"""Recompute the manuscript's 100-event pathway audit from the curated table."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/pathways/discovery_extension_crossclose_last_open_classification.csv"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100:
        raise SystemExit(f"Expected 100 events, found {len(rows)}")

    result: dict[str, object] = {"events": len(rows), "regimes": {}}
    for regime in ("N", "R"):
        subset = [row for row in rows if row["regime"] == regime]
        changed = []
        for row in subset:
            before = bytes.fromhex(row["focal_before_hex"])
            after = bytes.fromhex(row["focal_after_hex"])
            changed.append(sum(left != right for left, right in zip(before, after)))
        result["regimes"][regime] = {
            "events": len(subset),
            "focal_on_right": sum(row["focal_role"] == "p2" for row in subset),
            "exact_replays": sum(truth(row["replay_exact"]) for row in subset),
            "dynamic_classes": dict(sorted(Counter(row["dynamic_class"] for row in subset).items())),
            "dominant_pathway": sum(
                truth(row["cross_lastopen_copy_only"])
                or row["dynamic_class"] == "PARTNER_COMPLETE_LOOP"
                for row in subset
            ),
            "strict_pre_replicator": sum(truth(row["cross_lastopen_strict_proto"]) for row in subset),
            "copy_no_arithmetic": sum(truth(row["cross_lastopen_copy_nondis"]) for row in subset),
            "copy_only": sum(truth(row["cross_lastopen_copy_only"]) for row in subset),
            "changed_bytes_min": min(changed),
            "changed_bytes_median": sorted(changed)[len(changed) // 2],
            "changed_bytes_max": max(changed),
        }

    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
