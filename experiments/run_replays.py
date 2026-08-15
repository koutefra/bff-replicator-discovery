#!/usr/bin/env python3
"""Run controlled activation or pre-replicator availability replays (CUDA)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_PRE_REPLICATOR_SEEDS = [0, 2, 3, 5, 6, 22, 31, 53]


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_one(
    binary: Path,
    manifest: Path,
    experiment: str,
    regime: str,
    seed: int,
    output_root: Path,
    num_programs: int,
    epochs: int,
    stride: int,
    limit: int,
    gpu: str | None,
) -> Path:
    output = output_root / f"{regime.lower()}_seed{seed}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    option = "--partner_manifest" if experiment == "activation" else "--fixed_manifest"
    cmd = [
        str(binary),
        "--regime",
        regime,
        "--seed",
        str(seed),
        "--num_programs",
        str(num_programs),
        "--epochs",
        str(epochs),
        "--epoch_stride",
        str(stride),
        option,
        str(manifest),
        "--output_dir",
        str(output),
    ]
    if limit:
        cmd.extend(["--validation_program_limit", str(limit)])
        cmd.extend(
            [
                "--validation_partner_limit" if experiment == "activation" else "--validation_fixed_limit",
                "2",
            ]
        )
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
        cmd.extend(["--gpu_label", gpu])
    completed = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
    (output_root / "launch_logs").mkdir(parents=True, exist_ok=True)
    stem = output_root / "launch_logs" / f"{regime.lower()}_seed{seed}"
    stem.with_suffix(".stdout.log").write_text(completed.stdout, encoding="utf-8")
    stem.with_suffix(".stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"{experiment} {regime} seed {seed} failed: {completed.stderr[-500:]}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("activation", "pre-replicator"))
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seeds, default=None)
    parser.add_argument("--num-programs", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--epoch-stride", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--confirm-full", action="store_true")
    args = parser.parse_args()

    if args.profile == "paper" and not args.confirm_full:
        parser.error("--profile paper requires --confirm-full")
    default_seeds = [0] if args.profile == "smoke" else (
        list(range(4)) if args.experiment == "activation" else PAPER_PRE_REPLICATOR_SEEDS
    )
    seeds = args.seeds or default_seeds
    num_programs = args.num_programs or (128 if args.profile == "smoke" else 131_072)
    epochs = args.epochs or (3 if args.profile == "smoke" else 16_000)
    stride = args.epoch_stride or (1 if args.profile == "smoke" else 100)
    limit = 128 if args.profile == "smoke" else 0
    binary_name = (
        "fixed_partner_replay"
        if args.experiment == "activation"
        else "pre_replicator_availability_replay"
    )
    manifest_name = (
        "activation_partners.csv"
        if args.experiment == "activation"
        else "pre_replicator_focals.csv"
    )
    binary = ROOT / "bin" / binary_name
    manifest = ROOT / "data" / "manifests" / manifest_name
    if not binary.is_file():
        parser.error(f"missing {binary}; run make CUDA=1 replays")
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    jobs = [(regime, seed) for seed in seeds for regime in ("R", "N")]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_one,
                binary,
                manifest,
                args.experiment,
                regime,
                seed,
                output_root,
                num_programs,
                epochs,
                stride,
                limit,
                gpus[index % len(gpus)] if gpus else None,
            ): (regime, seed)
            for index, (regime, seed) in enumerate(jobs)
        }
        for future in as_completed(futures):
            regime, seed = futures[future]
            print(f"{regime} seed {seed}: {future.result()}")

    summary_dir = output_root / "summary"
    summary_dir.mkdir(exist_ok=True)
    script_name = (
        "analyze_proto_rep_partner_replay_program_union.py"
        if args.experiment == "activation"
        else "analyze_pre_replicator_availability_program_union.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "analysis" / script_name),
            "--run-root",
            str(output_root),
            "--out-prefix",
            str(summary_dir / args.experiment),
        ],
        cwd=ROOT,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
