#!/usr/bin/env python3
"""Fork N populations into R dynamics at selected epochs."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_main(binary: Path, args: list[str], stdout: Path, stderr: Path) -> None:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("wb") as out, stderr.open("wb") as err:
        result = subprocess.run([str(binary), *args], cwd=ROOT, stdout=out, stderr=err)
    if result.returncode:
        raise RuntimeError(f"simulator failed; see {stderr}")


def merge_logs(paths: list[Path], output: Path) -> None:
    by_epoch: dict[int, dict[str, str]] = {}
    fieldnames: list[str] | None = None
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = fieldnames or reader.fieldnames
            for row in reader:
                by_epoch[int(row["epoch"])] = row
    if not fieldnames:
        raise RuntimeError("no N log rows were produced")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in sorted(by_epoch):
            writer.writerow(by_epoch[epoch])


def make_n_baseline(
    binary: Path,
    seed: int,
    programs: int,
    fork_epochs: list[int],
    final_epoch: int,
    root: Path,
) -> tuple[Path, dict[int, Path]]:
    phases = root / "phases"
    phases.mkdir(parents=True)
    logs: list[Path] = []
    checkpoints: dict[int, Path] = {}
    load: Path | None = None
    targets = sorted(set(fork_epochs))
    for index, target in enumerate(targets):
        internal = target - 1
        phase = phases / f"to_{target}"
        checkpoint_dir = phase / "checkpoints"
        log = phase / "n.log"
        args = [
            "--lang", "bff_noheads", "--num", str(programs), "--seed", str(seed),
            "--mutation_prob", "0.0", "--random_partner_interaction",
            "--max_epochs", str(internal), "--log_interval", "1",
            "--print_interval", str(final_epoch + 2), "--disable_output",
            "--checkpoint_dir", str(checkpoint_dir), "--save_interval", str(max(1, internal)),
            "--log", str(log),
        ]
        if load is not None:
            args.extend(["--load", str(load)])
        run_main(binary, args, phase / "stdout.log", phase / "stderr.log")
        source = checkpoint_dir / f"{internal:010d}.dat"
        destination = root / f"epoch_{target}.dat"
        shutil.copy2(source, destination)
        checkpoints[target] = destination
        load = source
        logs.append(log)

    if max(targets) < final_epoch:
        phase = phases / f"to_{final_epoch}"
        log = phase / "n.log"
        args = [
            "--lang", "bff_noheads", "--num", str(programs), "--seed", str(seed),
            "--mutation_prob", "0.0", "--random_partner_interaction",
            "--max_epochs", str(final_epoch - 1), "--log_interval", "1",
            "--print_interval", str(final_epoch + 2), "--disable_output",
            "--load", str(load), "--log", str(log),
        ]
        run_main(binary, args, phase / "stdout.log", phase / "stderr.log")
        logs.append(log)
    baseline = root / "n.log"
    merge_logs(logs, baseline)
    return baseline, checkpoints


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default=None, help="Comma-separated source seeds")
    parser.add_argument("--fork-epochs", default=None, help="Comma-separated epochs")
    parser.add_argument("--programs", type=int, default=None)
    parser.add_argument("--final-epoch", type=int, default=None)
    parser.add_argument("--confirm-full", action="store_true")
    args = parser.parse_args()
    if args.profile == "paper" and not args.confirm_full:
        parser.error("--profile paper requires --confirm-full")

    seeds = [int(item) for item in args.seeds.split(",")] if args.seeds else (
        [0] if args.profile == "smoke" else [0, 1, 2, 3]
    )
    forks = [int(item) for item in args.fork_epochs.split(",")] if args.fork_epochs else (
        [2] if args.profile == "smoke" else [500, 1000, 2000, 4000, 8000]
    )
    programs = args.programs or (128 if args.profile == "smoke" else 131_072)
    final_epoch = args.final_epoch or (6 if args.profile == "smoke" else 16_000)
    if not forks or min(forks) <= 0 or max(forks) >= final_epoch:
        parser.error("fork epochs must be positive and less than --final-epoch")
    binary = ROOT / "bin" / "main"
    if not binary.is_file():
        parser.error("missing bin/main; run make first")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    for seed in seeds:
        seed_root = output / f"seed_{seed}"
        seed_root.mkdir()
        baseline, checkpoints = make_n_baseline(binary, seed, programs, forks, final_epoch, seed_root)
        fork_specs: list[str] = []
        for fork_epoch, checkpoint in checkpoints.items():
            branch = seed_root / f"r_from_{fork_epoch}"
            branch.mkdir()
            log = branch / "r.log"
            run_main(
                binary,
                [
                    "--lang", "bff_noheads", "--num", str(programs), "--seed", str(seed),
                    "--mutation_prob", "0.0", "--max_epochs", str(final_epoch - 1),
                    "--log_interval", "1", "--print_interval", str(final_epoch + 2),
                    "--disable_output", "--load", str(checkpoint), "--log", str(log),
                ],
                branch / "stdout.log",
                branch / "stderr.log",
            )
            fork_specs.extend(["--fork-log", f"{fork_epoch}:{log}"])
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "analysis" / "plot_branching_operator_enrichment.py"),
                "--n-log", str(baseline), "--n-extension-log", "",
                "--max-epoch", str(final_epoch), "--num-programs", str(programs),
                "--output", str(seed_root / "perturbation_operator_trajectories.pdf"),
                *fork_specs,
            ],
            cwd=ROOT,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
