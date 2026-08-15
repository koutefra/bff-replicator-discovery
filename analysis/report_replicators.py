#!/usr/bin/env python3
"""
Scan logs for self-replicators and print decoded tapes + opcode strips,
collapsing consecutive duplicate replicators and sorting by seed, epoch.
"""

import argparse
import csv
import pathlib
import ast


# Mapping from opcode byte to BFF-noheads glyph
COMMAND_BYTES = {
    ord("["): "[",
    ord("]"): "]",
    ord("+"): "+",
    ord("-"): "-",
    ord("."): ".",
    ord(","): ",",
    ord("<"): "<",
    ord(">"): ">",
    ord("{"): "{",
    ord("}"): "}",
}

OP_SYMBOLS = set(COMMAND_BYTES.values()) | {"0"}


def map_byte(byte: int) -> str:
    if byte == 0:
        return "0"
    if byte in COMMAND_BYTES:
        return COMMAND_BYTES[byte]
    return chr(0x0100 + byte)


def decode_hex_tape(hex_string: str) -> list[str]:
    tape_bytes = bytes.fromhex(hex_string.strip())
    return [map_byte(b) for b in tape_bytes]


def matching_bytes(a: str, b: str) -> int:
    """Count identical bytes at identical positions."""
    return sum(x == y for x, y in zip(a, b))


def pick_replicator(
    tapes: list[str], scores: list[object], threshold: float | None
) -> tuple[str, object] | None:
    if threshold is None:
        return tapes[0], scores[0]

    best: tuple[float, int, str, object] | None = None
    for idx, (tape, score) in enumerate(zip(tapes, scores)):
        score_value = float(score)
        if score_value < threshold:
            continue
        candidate = (score_value, -idx, tape, score)
        if best is None or candidate > best:
            best = candidate

    if best is None:
        return None
    return best[2], best[3]


def print_replicator(run, rep_id):
    print("=" * 80)
    print(
        f"(rep {rep_id}) seed={run['seed']}, "
        f"epoch={run['start_epoch']} → {run['end_epoch']} "
        f"(len={run['duration']}), score={run['score']}"
    )
    print(run["full_tape"])
    print("   ".join(run["ops"]))


def main():
    parser = argparse.ArgumentParser(
        description="Find replicators in logs and print decoded tapes."
    )
    parser.add_argument("mode", choices=["reinit", "interaction", "random"])
    parser.add_argument(
        "--group",
        default="no_mutation",
        help=(
            "Run group for interaction/random modes "
            "(runs/<group>/<mode>/). Ignored for reinit."
        ),
    )
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=None,
        help="Explicit run directory override (for example runs/reinit/reinit_N).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Only report epochs where at least one self-replicator score is >= threshold.",
    )
    args = parser.parse_args()

    if args.root is not None:
        root = args.root
    elif args.mode == "reinit":
        root = pathlib.Path("runs") / "reinit"
    else:
        root = pathlib.Path("runs") / args.group / args.mode
    if not root.is_dir():
        raise SystemExit(f"Missing folder: {root}")

    found_any = False
    all_runs = []

    # -------- SCAN + COLLAPSE --------
    for log_path in sorted(root.glob("log_*.log")):
        seed = log_path.stem.split("_")[1]
        active_run = None

        with log_path.open() as fh:
            reader = csv.DictReader(fh)

            for row in reader:
                if int(row.get("number_selfreps", 0)) == 0:
                    continue

                epoch = int(row["epoch"])

                tapes = ast.literal_eval(row["selfrep_tapes"])
                scores = ast.literal_eval(row["selfrep_scores"])
                if len(tapes) != len(scores):
                    raise SystemExit(
                        f"Mismatched selfrep_tapes/selfrep_scores lengths in {log_path} at epoch {epoch}"
                    )
                if not tapes:
                    continue

                selected = pick_replicator(tapes, scores, args.threshold)
                if selected is None:
                    continue

                found_any = True
                hex_tape, score = selected

                decoded = decode_hex_tape(hex_tape)
                full_tape = "".join(decoded)
                ops = [c for c in decoded if c in OP_SYMBOLS]

                if (
                    active_run
                    and epoch == active_run["end_epoch"] + 1
                    and matching_bytes(full_tape, active_run["full_tape"]) >= 50
                ):
                    active_run["end_epoch"] = epoch
                    active_run["duration"] += 1
                else:
                    if active_run:
                        all_runs.append(active_run)

                    active_run = {
                        "seed": seed,
                        "start_epoch": epoch,
                        "end_epoch": epoch,
                        "duration": 1,
                        "score": score,
                        "full_tape": full_tape,
                        "ops": ops,
                    }

        if active_run:
            all_runs.append(active_run)

    if not found_any:
        if args.threshold is None:
            print(f"No self-replicators detected in {root}")
        else:
            print(f"No self-replicators detected in {root} with score >= {args.threshold:g}")
        return

    # -------- SORT (seed, epoch) --------
    all_runs.sort(key=lambda r: (int(r["seed"]), r["start_epoch"]))

    # -------- PRINT (global counter) --------
    global_rep_id = 0
    for run in all_runs:
        global_rep_id += 1
        print_replicator(run, global_rep_id)


if __name__ == "__main__":
    main()
