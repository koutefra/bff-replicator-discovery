#!/usr/bin/env python3
"""Machine-check the compact paper data and website against reported results."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "paper"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{actual} != {expected}")


def main() -> int:
    expected = json.loads((DATA / "expected_results.json").read_text(encoding="utf-8"))
    discovery = read_csv(DATA / "discovery_times.csv")
    n = [int(row["first_epoch"]) for row in discovery if row["regime"] == "N" and row["discovered"] == "1"]
    r = [int(row["first_epoch"]) for row in discovery if row["regime"] == "R" and row["discovered"] == "1"]
    assert len(discovery) == 200
    assert len(n) == expected["discovery"]["N_discovered"] == 100
    assert len(r) == expected["discovery"]["R_discovered"] == 99
    close(statistics.median(n), 1220.0)
    close(statistics.median(r), 2453.0)

    uniform = read_csv(DATA / "uniform_replicators.csv")
    assert len(uniform) == expected["uniform"]["replicators"] == 15
    assert expected["uniform"]["samples"] == 209_715_200_000
    close(expected["uniform"]["median_epochs"], 73935.71009090493)

    events = read_csv(DATA / "discovery_events.csv")
    assert sum(int(row["n_events"]) for row in events) == 819
    assert sum(int(row["r_events"]) for row in events) == 301
    assert [int(row["r_exposure"]) for row in events] == [100, 91, 81, 79, 78, 75, 73, 69, 65, 61, 59, 56]
    close(sum(float(row["r_events_per_run"]) for row in events), 3.9765831984785227)

    activation = [row for row in read_csv(DATA / "activation.csv") if row["partner_origin"] == "all"]
    pre_replicator = [
        row
        for row in read_csv(DATA / "pre_replicator_availability.csv")
        if row["fixed_origin"] == "all"
    ]
    assert {int(row["seed_count"]) for row in activation} == {4}
    assert {int(row["seed_count"]) for row in pre_replicator} == {8}

    pathways = json.loads((ROOT / "data" / "pathways" / "summary.json").read_text(encoding="utf-8"))
    assert pathways["regimes"]["N"]["dominant_pathway"] == 43
    assert pathways["regimes"]["R"]["dominant_pathway"] == 45
    assert pathways["regimes"]["N"]["focal_on_right"] == 50
    assert pathways["regimes"]["R"]["focal_on_right"] == 48
    assert pathways["regimes"]["N"]["changed_bytes_min"] == 32

    rewrite = {row["label"]: row for row in read_csv(DATA / "uniform_rewrite_summary.csv")}
    ratio = float(rewrite["N"]["event_interaction_fraction"]) / float(rewrite["R"]["event_interaction_fraction"])
    close(ratio, 3.484)

    operators = read_csv(DATA / "operator_trajectories.csv")
    assert len(operators) == 32_200
    assert {row["regime"] for row in operators} == {"N", "R"}

    website = json.loads((ROOT / "web" / "data" / "research-data.json").read_text(encoding="utf-8"))
    close(float(website["hero"]["N"]["medianEpochs"]), 1220.0)
    close(float(website["hero"]["R"]["medianEpochs"]), 2453.0)
    assert website["hero"]["U"]["sampleCount"] == 209_715_200_000
    assert website["hero"]["U"]["discoveries"] == 15
    assert len(website["mechanism"]["N"]["preReplicator"]) == 160
    assert len(website["mechanism"]["R"]["preReplicator"]) == 160
    assert website["mechanism"]["discovery"]["rExposureCounts"] == [100, 91, 81, 79, 78, 75, 73, 69, 65, 61, 59, 56]

    print("Paper-data validation passed.")
    print("N median=1220; R median=2453; U=15/209715200000; takeovers=45; events=819/301")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
