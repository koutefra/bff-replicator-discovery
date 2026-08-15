#!/usr/bin/env python3
"""Run the matched N/R trace used for the long uniform-rewrite comparison."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_trace(binary: Path, mode: str, seed: int, programs: int, transitions: int, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    cmd = [
        str(binary),
        "--lang",
        "bff_noheads",
        "--num",
        str(programs),
        "--seed",
        str(seed),
        "--mutation_prob",
        "0.0",
        "--max_epochs",
        str(transitions),
        "--log_interval",
        "1",
        "--print_interval",
        str(transitions + 2),
        "--save_interval",
        "1",
        "--checkpoint_dir",
        str(output),
        "--disable_output",
    ]
    if mode == "random":
        cmd.append("--random_partner_interaction")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--programs", type=int, default=None)
    parser.add_argument("--transitions", type=int, default=None)
    parser.add_argument("--threshold", type=int, default=30)
    parser.add_argument("--confirm-full", action="store_true")
    args = parser.parse_args()
    if args.profile == "paper" and not args.confirm_full:
        parser.error("--profile paper requires --confirm-full")

    programs = args.programs or (8 if args.profile == "smoke" else 100)
    transitions = args.transitions or (4 if args.profile == "smoke" else 16_000)
    binary = ROOT / "bin" / "main"
    if not binary.is_file():
        parser.error("missing bin/main; run make first")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    n_dir, r_dir = output / "N", output / "R"
    run_trace(binary, "random", args.seed, programs, transitions, n_dir)
    run_trace(binary, "interaction", args.seed, programs, transitions, r_dir)
    analysis_base = output / "analysis"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "analysis" / "count_sequential_uniform_rewrites.py"),
            "--trace",
            f"N=random={args.seed}={n_dir}",
            "--trace",
            f"R=interaction={args.seed}={r_dir}",
            "--threshold",
            str(args.threshold),
            "--summary-out",
            str(analysis_base.with_name("rewrite_summary.csv")),
            "--timeseries-out",
            str(analysis_base.with_name("rewrite_timeseries.csv")),
            "--plot-out",
            str(analysis_base.with_name("rewrite_timeseries")),
        ],
        cwd=ROOT,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
