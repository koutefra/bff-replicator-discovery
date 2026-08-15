#!/usr/bin/env python3
"""Run the U, N, and R discovery regimes with smoke or paper settings."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_seeds(value: str) -> list[int]:
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    if not result:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return sorted(result)


def command(binary: Path, regime: str, seed: int, soup_size: int, epochs: int, log_path: Path):
    result = [
        str(binary),
        "--lang",
        "bff_noheads",
        "--num",
        str(soup_size),
        "--max_epochs",
        str(epochs - 1),
        "--log_interval",
        "1",
        "--print_interval",
        str(max(epochs + 1, 512)),
        "--mutation_prob",
        "0.0",
        "--seed",
        str(seed),
        "--eval_selfrep",
        "--print_selfrep",
        "--disable_output",
        "--log",
        str(log_path),
    ]
    if regime == "U":
        result.append("--reinit_each_epoch")
    elif regime == "N":
        result.append("--random_partner_interaction")
    return result


def run_one(
    binary: Path,
    root: Path,
    regime: str,
    seed: int,
    soup_size: int,
    epochs: int,
    gpu: str | None,
    resume: bool,
) -> Path:
    run_dir = root / regime.lower()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"log_{seed}.log"
    stdout_path = run_dir / f"stdout_{seed}.log"
    stderr_path = run_dir / f"stderr_{seed}.log"
    metadata_path = run_dir / f"metadata_{seed}.json"
    if resume and log_path.exists() and log_path.stat().st_size > 0:
        return log_path
    for path in (log_path, stdout_path, stderr_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; use --resume or another --output")

    cmd = command(binary, regime, seed, soup_size, epochs, log_path)
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    metadata_path.write_text(
        json.dumps(
            {
                "regime": regime,
                "seed": seed,
                "soup_size": soup_size,
                "epochs": epochs,
                "mutation_probability": 0.0,
                "gpu": gpu,
                "command": cmd,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
    if completed.returncode:
        raise RuntimeError(f"{regime} seed {seed} failed; see {stderr_path}")
    return log_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--regimes", default="U,N,R", help="Comma-separated subset of U,N,R")
    parser.add_argument("--seeds", type=parse_seeds, default=None, help="Examples: 0 or 0-99")
    parser.add_argument("--soup-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin" / "main")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpus", default="", help="Comma-separated visible GPU IDs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--confirm-full",
        action="store_true",
        help="Required for the paper profile, which takes hours per dynamical run.",
    )
    args = parser.parse_args()

    if args.profile == "paper" and not args.confirm_full:
        parser.error("--profile paper requires --confirm-full")
    seeds = args.seeds if args.seeds is not None else ([0] if args.profile == "smoke" else list(range(100)))
    soup_size = args.soup_size if args.soup_size is not None else (128 if args.profile == "smoke" else 131_072)
    epochs = args.epochs if args.epochs is not None else (3 if args.profile == "smoke" else 16_000)
    regimes = [item.strip().upper() for item in args.regimes.split(",") if item.strip()]
    if not regimes or any(item not in {"U", "N", "R"} for item in regimes):
        parser.error("--regimes must contain only U, N, and R")
    if soup_size <= 0 or soup_size % 2 or epochs <= 0 or args.workers <= 0:
        parser.error("soup size must be a positive even integer; epochs/workers must be positive")
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"missing simulator: {binary}; run make first")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]

    tasks = [(regime, seed) for regime in regimes for seed in seeds]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_one,
                binary,
                output,
                regime,
                seed,
                soup_size,
                epochs,
                gpus[index % len(gpus)] if gpus else None,
                args.resume,
            ): (regime, seed)
            for index, (regime, seed) in enumerate(tasks)
        }
        for future in as_completed(futures):
            regime, seed = futures[future]
            print(f"{regime} seed {seed}: {future.result()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
