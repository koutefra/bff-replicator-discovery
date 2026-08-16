#!/usr/bin/env python3
"""Export the compact, static data bundle used by the BFF web demo.

The exporter deliberately reads the paper's existing analysis products and raw
simulation logs.  It does not run a new experiment, and the browser does not
need Python at runtime.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import re
import statistics
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


ACTIVATION_REL = Path(
    "analysis_plots/no_mutation/"
    "activation_experiment_first10_seed0_seed3_program_union_epoch_summary.csv"
)
MOTIF_REL = Path(
    "analysis_plots/no_mutation/"
    "motif_availability_focal70_program_union_seed0_seed2_seed3_seed5_seed6_seed22_seed31_seed53_epoch_summary.csv"
)
DISCOVERY_REL = Path(
    "analysis_plots/no_mutation/"
    "discovery_hist_cache_bin1333_max16000_score60_a014775cf2b7.npz"
)
TOY_SCRIPT_REL = Path("scripts/plot_toy_canonical_progression.py")
RECONSTRUCTED_EVENTS_REL = Path(
    "analysis_plots/no_mutation/discovery_extension_reconstructed_events.csv"
)
N_LOG_DIR_REL = Path("runs/no_mutation/random")
R_LOG_DIR_REL = Path("runs/no_mutation/interaction")
U_LOG_DIR_REL = Path("runs/reinit")

MAX_EPOCH = 16_000
SCORE_THRESHOLD = 60.0
TAKEOVER_WINDOW = 500
TARGET_OPERATOR_EPOCHS = (1,) + tuple(range(1_000, MAX_EPOCH + 1, 1_000))
OPERATOR_SYMBOLS = ("<", ">", "{", "}", "+", "-", ".", ",", "[", "]", "0")
OPERATOR_COUNT_DENOMINATOR = (2**17) * 64
R_COND_TAKEOVER_EPOCH_OVERRIDES = {19: 4_770, 89: 750, 96: 600}
NUMBER_RE = re.compile(rb"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class DynamicalRun:
    seed: int
    first_discovery_epoch: int | None
    takeover_epoch: int | None
    operator_counts: dict[int, tuple[float, ...]]


def find_repo_root(start: Path) -> Path:
    """Find the repository independently of the caller's current directory."""

    for candidate in (start, *start.parents):
        if (candidate / TOY_SCRIPT_REL).is_file() and (candidate / "runs").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate the cubff repository above {start}")


def relative_source(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def parse_seed(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Could not parse a seed from {path}")
    return int(match.group(1))


def score_field_from_log_line(line: bytes) -> bytes:
    """Return selfrep_scores without parsing the potentially huge tape field.

    selfrep_scores is the last quoted CSV field in the N/R log schema.  Reading
    it from the right avoids materializing the often very large selfrep_tapes
    field and makes regeneration practical while still using the raw logs.
    """

    end_quote = line.rfind(b'"')
    if end_quote < 0:
        return b""
    start_quote = line.rfind(b'"', 0, end_quote)
    if start_quote < 0:
        return b""
    return line[start_quote + 1 : end_quote]


def has_score_at_least(score_field: bytes, threshold: float) -> bool:
    if score_field in (b"", b"[]"):
        return False
    for match in NUMBER_RE.finditer(score_field):
        if float(match.group(0)) >= threshold:
            return True
    return False


def count_scores_at_least(score_field: bytes, threshold: float) -> int:
    if score_field in (b"", b"[]"):
        return 0
    return sum(float(match.group(0)) >= threshold for match in NUMBER_RE.finditer(score_field))


def operator_counts_from_log_line(line: bytes, path: Path, epoch: int) -> tuple[float, ...]:
    # The final 15 columns are the 11 operator counts followed by four execution
    # metrics.  Splitting from the right is safe even when quoted tape data
    # contains commas.
    fields = line.rstrip(b"\r\n").rsplit(b",", 15)
    if len(fields) != 16:
        raise ValueError(f"Unexpected operator-log schema in {path} at epoch {epoch}")
    try:
        counts = tuple(float(value) for value in fields[1:12])
    except ValueError as exc:
        raise ValueError(f"Invalid operator count in {path} at epoch {epoch}") from exc
    if len(counts) != len(OPERATOR_SYMBOLS):
        raise AssertionError("Operator-column count drifted from the paper analysis")
    return counts


def scan_dynamical_log(path: Path, classify_takeover: bool) -> DynamicalRun:
    first_discovery: int | None = None
    takeover_epoch: int | None = None
    streak_length = 0
    streak_start_epoch: int | None = None
    target_counts: dict[int, tuple[float, ...]] = {}

    with path.open("rb") as fh:
        header = fh.readline()
        if not header.startswith(b"epoch,"):
            raise ValueError(f"Unexpected CSV header in {path}")
        for line in fh:
            comma = line.find(b",")
            if comma <= 0:
                continue
            try:
                epoch = int(line[:comma])
            except ValueError:
                continue

            high_score = has_score_at_least(
                score_field_from_log_line(line), SCORE_THRESHOLD
            )
            if high_score and first_discovery is None and epoch <= MAX_EPOCH:
                first_discovery = epoch

            if classify_takeover and takeover_epoch is None:
                if high_score:
                    if streak_length == 0:
                        streak_start_epoch = epoch
                    streak_length += 1
                    if streak_length >= TAKEOVER_WINDOW:
                        takeover_epoch = streak_start_epoch
                else:
                    streak_length = 0
                    streak_start_epoch = None

            if epoch in TARGET_OPERATOR_EPOCHS:
                target_counts[epoch] = operator_counts_from_log_line(line, path, epoch)

    missing = sorted(set(TARGET_OPERATOR_EPOCHS) - set(target_counts))
    if missing:
        raise ValueError(f"Missing target operator epochs in {path}: {missing}")
    return DynamicalRun(
        seed=parse_seed(path),
        first_discovery_epoch=first_discovery,
        takeover_epoch=takeover_epoch,
        operator_counts=target_counts,
    )


def scan_dynamical_runs(root: Path, classify_takeover: bool, label: str) -> list[DynamicalRun]:
    paths = sorted(root.glob("log_*.log"), key=parse_seed)
    if not paths:
        raise FileNotFoundError(f"No log_*.log files found in {root}")
    print(f"Scanning {len(paths)} raw {label} logs ...", flush=True)
    runs: list[DynamicalRun] = []
    for index, path in enumerate(paths, start=1):
        runs.append(scan_dynamical_log(path, classify_takeover=classify_takeover))
        if index % 20 == 0 or index == len(paths):
            print(f"  {label}: {index}/{len(paths)}", flush=True)
    return runs


def scan_uniform_null(root: Path) -> dict[str, int | float]:
    paths = sorted(root.glob("log_*.log"), key=parse_seed)
    if not paths:
        raise FileNotFoundError(f"No uniform log_*.log files found in {root}")

    robust_discoveries = 0
    total_samples = 0
    soup_sizes: set[int] = set()
    included_rows = 0
    for path in paths:
        with path.open("rb") as fh:
            header = fh.readline()
            if not header.startswith(b"epoch,"):
                raise ValueError(f"Unexpected CSV header in {path}")
            for line in fh:
                first_fields = line.split(b",", 3)
                if len(first_fields) < 4:
                    continue
                try:
                    epoch = int(first_fields[0])
                    soup_size = int(first_fields[2])
                except ValueError:
                    continue
                if epoch < 1 or epoch > MAX_EPOCH:
                    continue
                included_rows += 1
                soup_sizes.add(soup_size)
                total_samples += soup_size
                robust_discoveries += count_scores_at_least(
                    score_field_from_log_line(line), SCORE_THRESHOLD
                )

    if len(soup_sizes) != 1:
        raise ValueError(f"Expected one uniform soup size, found {sorted(soup_sizes)}")
    programs_per_epoch = next(iter(soup_sizes))
    if robust_discoveries <= 0 or total_samples <= 0:
        raise ValueError("Uniform-null logs contain no usable discovery estimate")

    per_program_probability = robust_discoveries / total_samples
    per_epoch_probability = -math.expm1(
        programs_per_epoch * math.log1p(-per_program_probability)
    )
    estimated_median = math.log(0.5) / math.log1p(-per_epoch_probability)
    rounded_median = int(round(estimated_median / 1_000.0) * 1_000)
    return {
        "medianEpochs": rounded_median,
        "estimatedMedianEpochs": estimated_median,
        "totalRuns": len(paths),
        "discoveries": robust_discoveries,
        "sampleCount": total_samples,
        "programsPerEpoch": programs_per_epoch,
        "includedEpochRows": included_rows,
        "scoreThreshold": SCORE_THRESHOLD,
    }


def build_hero(
    repo_root: Path,
    n_runs: list[DynamicalRun],
    r_runs: list[DynamicalRun],
) -> dict[str, dict[str, int | float]]:
    uniform = scan_uniform_null(repo_root / U_LOG_DIR_REL)

    def dynamical_summary(runs: list[DynamicalRun]) -> dict[str, int | float]:
        discoveries = sorted(
            run.first_discovery_epoch
            for run in runs
            if run.first_discovery_epoch is not None
        )
        if not discoveries:
            raise ValueError("No first-discovery epochs found")
        return {
            "medianEpochs": statistics.median(discoveries),
            "totalRuns": len(runs),
            "discoveredRuns": len(discoveries),
            "scoreThreshold": SCORE_THRESHOLD,
            "maxEpoch": MAX_EPOCH,
        }

    hero = {
        "U": uniform,
        "N": dynamical_summary(n_runs),
        "R": dynamical_summary(r_runs),
    }
    # These are the audited headline values in the current manuscript.  Failing
    # loudly is preferable to silently publishing a different data selection.
    if hero["U"]["discoveries"] != 15 or hero["U"]["medianEpochs"] != 74_000:
        raise AssertionError(f"Uniform-null headline drifted: {hero['U']}")
    if hero["N"]["medianEpochs"] != 1_220 or hero["N"]["discoveredRuns"] != 100:
        raise AssertionError(f"N headline drifted: {hero['N']}")
    if hero["R"]["medianEpochs"] != 2_453 or hero["R"]["discoveredRuns"] != 99:
        raise AssertionError(f"R headline drifted: {hero['R']}")
    return hero


def import_toy_source(repo_root: Path):
    # Keep Matplotlib's import-time cache out of a user's home directory.  The
    # imported module is the authoritative source of the evaluator and figure
    # constants; no plotting occurs here.
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cubff-web-demo-mplconfig")
    )
    scripts_dir = repo_root / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import plot_toy_canonical_progression as toy
    finally:
        sys.path.pop(0)
    return toy


def combined_loop_spans(snapshot, half_length: int) -> list[list[int]]:
    spans: list[list[int]] = []
    for half, start, end in snapshot.loop_spans:
        offset = 0 if half == "P" else half_length
        spans.append([int(start + offset), int(end + offset)])
    return spans


def build_trace(repo_root: Path) -> dict[str, object]:
    toy = import_toy_source(repo_root)
    history = toy.evaluate_modulo(toy.DEFAULT_PARTNER, toy.DEFAULT_FOCAL, step_cap=308)
    if [int(row["step"]) for row in history] != list(range(308)):
        raise AssertionError("Canonical toy history is not the expected exact step range 0..307")
    snapshots = {snapshot.step: snapshot for snapshot in toy.canonical_snapshots(history)}
    half_length = len(toy.DEFAULT_PARTNER)

    phase_specs = [
        (
            "loop-completion",
            0,
            0,
            0,
            "The pre-replicator fragment on the partner tape meets the target tape’s closing bracket after concatenation, forming the pre-replicator loop.",
        ),
        (
            "reverse-construction",
            1,
            122,
            122,
            "Repeated loop traversal drives asymmetric head transport and copies selected partner symbols into the focal tape in reverse.",
        ),
        (
            "forward-reconstruction",
            123,
            215,
            215,
            "The heads traverse the newly constructed region and copy it forward, rewriting the focal tape.",
        ),
        (
            "stabilization",
            216,
            307,
            307,
            "The rewritten focal tape now contains a functional reverse-copy replicator.",
        ),
    ]
    phases: list[dict[str, object]] = []
    annotated_cross_boundary_spans = combined_loop_spans(snapshots[122], half_length)
    if annotated_cross_boundary_spans != [[3, 15], [16, 17]]:
        raise AssertionError("Canonical cross-boundary loop annotation changed")
    for phase_id, start, milestone, end, caption in phase_specs:
        snapshot = snapshots[milestone]
        loop_spans = combined_loop_spans(snapshot, half_length)
        if milestone == 0:
            # At concatenation, the two adjacent source spans form one concrete
            # loop from the partner's '[' through the focal tape's ']'.
            loop_spans = [
                [annotated_cross_boundary_spans[0][0], annotated_cross_boundary_spans[-1][1]]
            ]
        phases.append(
            {
                "id": phase_id,
                "label": snapshot.label,
                "startStep": start,
                "milestoneStep": milestone,
                "endStep": end,
                "caption": caption,
                "loopSpans": loop_spans,
                "copyPairs": [
                    [int(source), int(destination)]
                    for source, destination in snapshot.copy_pairs
                ],
            }
        )

    # Ensure the explanatory arrows remain byte-for-byte tied to the source
    # figure's constants rather than being hand-maintained in this exporter.
    assert phases[1]["copyPairs"] == [list(pair) for pair in toy.REVERSE_CONSTRUCTION_PAIRS]
    assert phases[2]["copyPairs"] == [list(pair) for pair in toy.FORWARD_RECONSTRUCTION_PAIRS]
    assert phases[3]["copyPairs"] == [list(pair) for pair in toy.STABILIZATION_PAIRS]

    states = [
        {
            "step": int(row["step"]),
            "pc": int(row["pc"]),
            "h0": int(row["h0"]),
            "h1": int(row["h1"]),
            "partner": str(row["partner"]),
            "focal": str(row["focal"]),
        }
        for row in history
    ]
    return {
        "halfLength": half_length,
        "maxStep": 307,
        "states": states,
        "phases": phases,
    }


def read_activation(path: Path) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {"N": {}, "R": {}}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["partner_origin"] != "all":
                continue
            regime = row["run_regime"]
            if regime not in out:
                continue
            out[regime][int(row["epoch"])] = float(row["mean_seed_hit_fraction"])
    return out


def read_pre_replicator(path: Path) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {"N": {}, "R": {}}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["fixed_origin"] != "all":
                continue
            regime = row["run_regime"]
            if regime not in out:
                continue
            seed_count = int(row["seed_count"])
            if seed_count <= 0:
                raise ValueError(f"Non-positive pre-replicator seed_count at epoch {row['epoch']}")
            out[regime][int(row["epoch"])] = float(row["total_hits"]) / seed_count
    return out


def centered_epoch_mean(
    epochs: list[int], values: list[float], window_epochs: int
) -> list[float]:
    half_width = window_epochs / 2.0
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    smoothed: list[float] = []
    for epoch in epochs:
        left = bisect.bisect_left(epochs, epoch - half_width)
        right = bisect.bisect_right(epochs, epoch + half_width)
        smoothed.append((prefix[right] - prefix[left]) / (right - left))
    return smoothed


def load_discovery(path: Path, r_runs: list[DynamicalRun]) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        bin_edges = np.asarray(data["bin_edges"], dtype=np.float64)
        n_event_epochs = np.asarray(data["n_event_epochs"], dtype=np.int64)
        n_run_count = int(data["n_run_count"])
        r_rates = np.asarray(data["r_event_weights"], dtype=np.float64)
        r_exposure = np.asarray(data["r_exposure_counts"], dtype=np.int64)
        r_all_run_count = int(data["r_all_run_count"])

    n_counts, _ = np.histogram(n_event_epochs, bins=bin_edges)
    n_rates = n_counts.astype(np.float64) / n_run_count
    if len(bin_edges) != 13 or len(n_rates) != 12 or len(r_rates) != 12:
        raise AssertionError("Expected the paper's 12-bin discovery payload")

    r_final_runs = sum(run.takeover_epoch is None for run in r_runs)
    return {
        "binEdges": [float(value) for value in bin_edges],
        "N": [float(value) for value in n_rates],
        "R": [float(value) for value in r_rates],
        "nRuns": n_run_count,
        "rInitialRuns": r_all_run_count,
        "rFinalRuns": r_final_runs,
        "rExposureCounts": [int(value) for value in r_exposure],
        "binWidthApprox": float(bin_edges[1] - bin_edges[0]),
    }


def build_mechanism(repo_root: Path, r_runs: list[DynamicalRun]) -> dict[str, object]:
    activation = read_activation(repo_root / ACTIVATION_REL)
    pre_replicator_raw = read_pre_replicator(repo_root / MOTIF_REL)
    expected_epochs = list(range(100, MAX_EPOCH + 1, 100))
    for regime in ("N", "R"):
        if sorted(activation[regime]) != expected_epochs:
            raise AssertionError(f"Unexpected {regime} activation epochs")
        if sorted(pre_replicator_raw[regime]) != expected_epochs:
            raise AssertionError(f"Unexpected {regime} pre-replicator epochs")

    n_raw = [pre_replicator_raw["N"][epoch] for epoch in expected_epochs]
    r_raw = [pre_replicator_raw["R"][epoch] for epoch in expected_epochs]
    n_mean = math.fsum(n_raw) / len(n_raw)
    pre_replicator = {
        "N": [n_mean] * len(expected_epochs),
        "R": centered_epoch_mean(expected_epochs, r_raw, window_epochs=500),
    }

    regimes: dict[str, dict[str, list[float]]] = {}
    for regime, raw_values in (("N", n_raw), ("R", r_raw)):
        activation_values = [activation[regime][epoch] for epoch in expected_epochs]
        pre_replicator_values = pre_replicator[regime]
        regimes[regime] = {
            "activation": activation_values,
            "preReplicatorRaw": raw_values,
            "preReplicator": pre_replicator_values,
            "model": [
                30.0 * activation_value * pre_replicator_value
                for activation_value, pre_replicator_value in zip(activation_values, pre_replicator_values)
            ],
        }

    return {
        "epochs": expected_epochs,
        "N": regimes["N"],
        "R": regimes["R"],
        "discovery": load_discovery(repo_root / DISCOVERY_REL, r_runs),
    }


def median_operator_percentages(
    runs: Iterable[DynamicalRun], epoch: int
) -> tuple[list[float], int]:
    rows = [run.operator_counts[epoch] for run in runs if epoch in run.operator_counts]
    if not rows:
        raise ValueError(f"No operator rows available at epoch {epoch}")
    values = [
        statistics.median(row[index] for row in rows)
        / OPERATOR_COUNT_DENOMINATOR
        * 100.0
        for index in range(len(OPERATOR_SYMBOLS))
    ]
    return values, len(rows)


def build_operators(
    n_runs: list[DynamicalRun], r_runs: list[DynamicalRun]
) -> dict[str, object]:
    n_values: list[list[float]] = []
    n_counts: list[int] = []
    r_values: list[list[float]] = []
    r_counts: list[int] = []

    for epoch in TARGET_OPERATOR_EPOCHS:
        values, count = median_operator_percentages(n_runs, epoch)
        n_values.append(values)
        n_counts.append(count)

        eligible_r = []
        for run in r_runs:
            if run.takeover_epoch is None:
                eligible_r.append(run)
                continue
            cutoff = R_COND_TAKEOVER_EPOCH_OVERRIDES.get(
                run.seed, run.takeover_epoch
            )
            if epoch <= cutoff:
                eligible_r.append(run)
        values, count = median_operator_percentages(eligible_r, epoch)
        r_values.append(values)
        r_counts.append(count)

    if n_counts != [100] * len(TARGET_OPERATOR_EPOCHS):
        raise AssertionError(f"Unexpected N operator run counts: {n_counts}")
    if r_counts[-1] != 55:
        raise AssertionError(f"Expected 55 non-takeover R runs at epoch 16000, got {r_counts[-1]}")

    return {
        "epochs": list(TARGET_OPERATOR_EPOCHS),
        "symbols": list(OPERATOR_SYMBOLS),
        "N": {"values": n_values, "nRuns": n_counts},
        "Rcond": {"values": r_values, "nRuns": r_counts},
        "unit": "percent of all tape bytes",
        "denominatorBytes": OPERATOR_COUNT_DENOMINATOR,
    }


def build_reference_calculation(hero: dict[str, dict[str, int | float]]) -> dict[str, int | float]:
    """Compute the deliberately restrictive canonical fragment reference."""

    program_length = 64
    fragment_length = 5
    alphabet_size = 256
    instruction_values = 10
    neutral_values = alphabet_size - instruction_values
    probability = (
        math.comb(program_length, fragment_length)
        * (1.0 / alphabet_size) ** fragment_length
        * (neutral_values / alphabet_size) ** (program_length - fragment_length)
    )
    partners_per_epoch = 2**17
    correctly_oriented_per_epoch = partners_per_epoch * probability / 2.0
    n_median = float(hero["N"]["medianEpochs"])
    result: dict[str, int | float] = {
        "programLength": program_length,
        "fragmentLength": fragment_length,
        "alphabetSize": alphabet_size,
        "instructionValues": instruction_values,
        "neutralValues": neutral_values,
        "probability": probability,
        "onePerPrograms": 1.0 / probability,
        "partnersPerEpoch": partners_per_epoch,
        "perEpochBeforeOrientation": partners_per_epoch * probability,
        "correctOrientationPerEpoch": correctly_oriented_per_epoch,
        "nMedianEpoch": n_median,
        "expectedByNMedian": correctly_oriented_per_epoch * n_median,
    }
    if not math.isclose(probability, 6.6073238146e-7, rel_tol=1e-10):
        raise AssertionError(f"Canonical fragment calculation drifted: {probability}")
    return result


def build_reconstruction_observations(path: Path) -> dict[str, object]:
    """Export focal-side counts from the 100 reconstructed discovery events."""

    counts = {"N": {"total": 0, "rightFocal": 0}, "R": {"total": 0, "rightFocal": 0}}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            regime = row.get("regime")
            if regime not in counts:
                continue
            counts[regime]["total"] += 1
            counts[regime]["rightFocal"] += row.get("focal_role") == "p2"

    total = sum(item["total"] for item in counts.values())
    right_focal = sum(item["rightFocal"] for item in counts.values())
    if counts != {
        "N": {"total": 50, "rightFocal": 50},
        "R": {"total": 50, "rightFocal": 48},
    }:
        raise AssertionError(f"Reconstructed focal-side counts drifted: {counts}")
    return {
        "byRegime": counts,
        "total": total,
        "rightFocal": right_focal,
    }


def validate_payload(payload: dict[str, object]) -> None:
    trace = payload["trace"]
    assert isinstance(trace, dict)
    assert len(trace["states"]) == 308
    assert len(trace["phases"]) == 4

    mechanism = payload["mechanism"]
    assert isinstance(mechanism, dict)
    epoch_count = len(mechanism["epochs"])
    assert epoch_count == 160
    for regime in ("N", "R"):
        series = mechanism[regime]
        for key in ("activation", "preReplicatorRaw", "preReplicator", "model"):
            values = series[key]
            assert len(values) == epoch_count
            assert all(math.isfinite(float(value)) for value in values)
    assert len(mechanism["discovery"]["N"]) == 12
    assert len(mechanism["discovery"]["R"]) == 12

    observations = payload["reconstructionObservations"]
    assert observations["rightFocal"] == 98
    assert observations["total"] == 100

    reference = payload["referenceCalculation"]
    assert 52.8 < float(reference["expectedByNMedian"]) < 52.9

    operators = payload["operators"]
    assert isinstance(operators, dict)
    assert len(operators["epochs"]) == 17
    for regime in ("N", "Rcond"):
        assert len(operators[regime]["values"]) == 17
        assert all(len(row) == 11 for row in operators[regime]["values"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: web-demo/data/research-data.json).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    output_path = args.output or repo_root / "web-demo/data/research-data.json"
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    n_runs = scan_dynamical_runs(
        repo_root / N_LOG_DIR_REL, classify_takeover=False, label="N"
    )
    r_runs = scan_dynamical_runs(
        repo_root / R_LOG_DIR_REL, classify_takeover=True, label="R"
    )
    takeover_count = sum(run.takeover_epoch is not None for run in r_runs)
    if takeover_count != 45:
        raise AssertionError(f"Expected 45 score>=60 takeover runs, found {takeover_count}")

    hero = build_hero(repo_root, n_runs, r_runs)
    payload: dict[str, object] = {
        "meta": {
            "generatedAt": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "sources": {
                "toyTrace": relative_source(repo_root / TOY_SCRIPT_REL, repo_root),
                "reconstructedEvents": relative_source(
                    repo_root / RECONSTRUCTED_EVENTS_REL, repo_root
                ),
                "activation": relative_source(repo_root / ACTIVATION_REL, repo_root),
                "preReplicator": relative_source(repo_root / MOTIF_REL, repo_root),
                "discovery": relative_source(repo_root / DISCOVERY_REL, repo_root),
                "uniformNull": "runs/reinit/log_*.log",
                "operatorN": "runs/no_mutation/random/log_*.log",
                "operatorR": "runs/no_mutation/interaction/log_*.log",
            },
            "notes": [
                "The toy states are the exact evaluator history at steps 0 through 307; phase copy pairs are explanatory relations defined by the source figure, not a per-instruction event log. The step-0 loop span [3,17] is the concrete contiguous cross-boundary loop derived from the source tapes and the source figure's step-122 annotation.",
                "Activation uses partner_origin=all and mean_seed_hit_fraction.",
                "Raw pre-replicator availability is total_hits / seed_count with fixed_origin=all. N is flattened to its exact 160-epoch time mean; R uses a centered 500-epoch moving average (observations within plus or minus 250 epochs, with available-point endpoint handling).",
                "The model curve is 30 * A(t) * M(t) and is an approximate first-order account, not an exact mechanistic law or fitted causal model.",
                "Discovery values are events per eligible run in the 12 bins stored in the bin1333 NPZ; R uses that cache's takeover-conditioned, exposure-normalized rates.",
                "Operator values are per-symbol medians from raw logs at epoch 1 and every 1000-epoch interval thereafter, divided by 2^17 * 64 and multiplied by 100. Epoch 1 precedes any execution, so both regimes start from the same uniform byte distribution. Rcond includes non-takeover runs in full and truncates takeover runs at score>=60 takeover, with the three source-figure epoch overrides.",
                "The U headline is recomputed from 15 robust replicators in 209,715,200,000 uniform samples and rounded to the manuscript's approximately 74,000 epochs; N and R medians are recomputed from raw first-discovery epochs.",
            ],
        },
        "hero": hero,
        "trace": build_trace(repo_root),
        "reconstructionObservations": build_reconstruction_observations(
            repo_root / RECONSTRUCTED_EVENTS_REL
        ),
        "mechanism": build_mechanism(repo_root, r_runs),
        "operators": build_operators(n_runs, r_runs),
        "referenceCalculation": build_reference_calculation(hero),
    }
    validate_payload(payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        fh.write("\n")

    try:
        output_label = output_path.relative_to(repo_root)
    except ValueError:
        output_label = output_path
    print(f"Wrote {output_label} ({output_path.stat().st_size:,} bytes)")
    print(
        "Validated: 308 trace states, 160 mechanism epochs, "
        "12 discovery bins, 17 operator epochs x 11 symbols"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
