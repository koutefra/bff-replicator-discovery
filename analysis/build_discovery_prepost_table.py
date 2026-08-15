#!/usr/bin/env python3
"""
Build focal/partner pre/post table from reconstructed discovery events.

This script turns:
  discovery_extension_reconstructed_events.csv
into:
  discovery_extension_all_events_focal_partner_prepost.csv

and optional N/R splits.
"""

from __future__ import annotations

import argparse
import csv
import struct
from pathlib import Path

from report_replicators import OP_SYMBOLS, decode_hex_tape
from extend_discovery_partner_reconstruction import build_partner_n


HEADER_STRUCT = struct.Struct("=QQQ")
K_SINGLE_TAPE_SIZE = 64


def strip_from_hex(hex_tape: str) -> str:
    decoded = decode_hex_tape(hex_tape)
    return "".join(ch for ch in decoded if ch in OP_SYMBOLS)


def load_dump(path: Path) -> tuple[int, memoryview]:
    raw = path.read_bytes()
    _, num_programs, _ = HEADER_STRUCT.unpack_from(raw, 0)
    payload = raw[HEADER_STRUCT.size :]
    expected = num_programs * K_SINGLE_TAPE_SIZE
    if len(payload) != expected:
        raise RuntimeError(f"Corrupt dump size at {path}: got={len(payload)} expected={expected}")
    return num_programs, memoryview(payload)


def tape_hex_at_slot(payload: memoryview, slot: int) -> str:
    start = slot * K_SINGLE_TAPE_SIZE
    end = start + K_SINGLE_TAPE_SIZE
    return bytes(payload[start:end]).hex()


def derive_phase(event_epoch: int, takeover_epoch_text: str) -> str:
    if not takeover_epoch_text.strip():
        return "no_takeover_run"
    takeover_epoch = int(float(takeover_epoch_text))
    if event_epoch < takeover_epoch:
        return "pre_takeover"
    if event_epoch == takeover_epoch:
        return "at_takeover"
    return "post_takeover"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build discovery extension focal/partner pre/post table from reconstructed events."
    )
    parser.add_argument(
        "--reconstructed-csv",
        type=Path,
        default=Path("analysis_plots/no_mutation/discovery_extension_reconstructed_events.csv"),
    )
    parser.add_argument(
        "--dumps-root",
        type=Path,
        required=True,
        help="Root used by extend_discovery_partner_reconstruction.py --dumps-root",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("analysis_plots/no_mutation/discovery_extension_all_events_focal_partner_prepost.csv"),
    )
    parser.add_argument(
        "--write-regime-splits",
        action="store_true",
        help="Also write *_N_* and *_R_* CSV split files beside --out-csv.",
    )
    args = parser.parse_args()

    if not args.reconstructed_csv.exists():
        raise SystemExit(f"Missing reconstructed CSV: {args.reconstructed_csv}")
    if not args.dumps_root.exists():
        raise SystemExit(f"Missing dumps root: {args.dumps_root}")

    with args.reconstructed_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("Reconstructed CSV has no rows.")

    dump_cache: dict[tuple[str, int, int], tuple[int, memoryview]] = {}

    def get_dump(regime: str, seed: int, epoch: int) -> tuple[int, memoryview]:
        key = (regime, seed, epoch)
        if key in dump_cache:
            return dump_cache[key]
        path = args.dumps_root / regime / f"seed_{seed:03d}" / "dumps" / f"epoch_{epoch}.dat"
        if not path.exists():
            raise FileNotFoundError(f"Missing dump file: {path}")
        dump_cache[key] = load_dump(path)
        return dump_cache[key]

    out_rows: list[dict[str, object]] = []
    for row in rows:
        regime = str(row["regime"])
        seed = int(row["seed"])
        event_epoch = int(row["event_epoch"])
        pre_epoch = event_epoch - 1
        post_epoch = event_epoch

        focal_role = str(row["focal_role"])
        concat_order = "partner+focal" if focal_role == "p2" else "focal+partner"
        focal_slot = int(row["focal_slot"])
        partner_slot = int(row["partner_slot"]) if str(row["partner_slot"]).strip() else None

        focal_pre_hex = str(row["focal_before_hex"])
        focal_post_hex = str(row["focal_after_hex"])
        partner_pre_hex = str(row["partner_hex"])

        if regime == "R":
            if partner_slot is None:
                raise RuntimeError(f"R row missing partner_slot: seed={seed} epoch={event_epoch}")
            _, payload = get_dump(regime, seed, post_epoch)
            partner_post_hex = tape_hex_at_slot(payload, partner_slot)
        else:
            partner_post, _ = build_partner_n(seed, post_epoch, focal_slot)
            partner_post_hex = partner_post.hex()

        focal_pre_strip = strip_from_hex(focal_pre_hex)
        focal_post_strip = strip_from_hex(focal_post_hex)
        partner_pre_strip = strip_from_hex(partner_pre_hex)
        partner_post_strip = strip_from_hex(partner_post_hex)

        if focal_role == "p2":
            pre_concat_hex = partner_pre_hex + focal_pre_hex
            post_concat_hex = partner_post_hex + focal_post_hex
            pre_concat_strip = partner_pre_strip + focal_pre_strip
            post_concat_strip = partner_post_strip + focal_post_strip
        else:
            pre_concat_hex = focal_pre_hex + partner_pre_hex
            post_concat_hex = focal_post_hex + partner_post_hex
            pre_concat_strip = focal_pre_strip + partner_pre_strip
            post_concat_strip = focal_post_strip + partner_post_strip

        out_rows.append(
            {
                "regime": regime,
                "seed": seed,
                "event_rank": int(row["event_rank"]),
                "event_epoch": event_epoch,
                "pre_epoch": pre_epoch,
                "post_epoch": post_epoch,
                "takeover_epoch": row["takeover_epoch"],
                "event_takeover_phase": derive_phase(event_epoch, str(row["takeover_epoch"])),
                "focal_role": focal_role,
                "concat_order": concat_order,
                "focal_slot": focal_slot,
                "partner_slot": partner_slot if partner_slot is not None else "",
                "focal_pre_hex": focal_pre_hex,
                "focal_pre_strip": focal_pre_strip,
                "partner_pre_hex": partner_pre_hex,
                "partner_pre_strip": partner_pre_strip,
                "pre_concat_hex": pre_concat_hex,
                "pre_concat_strip": pre_concat_strip,
                "focal_post_hex": focal_post_hex,
                "focal_post_strip": focal_post_strip,
                "partner_post_hex": partner_post_hex,
                "partner_post_strip": partner_post_strip,
                "post_concat_hex": post_concat_hex,
                "post_concat_strip": post_concat_strip,
                "replay_exact": row["replay_exact"],
                "dynamic_class": row["dynamic_class"],
            }
        )

    out_rows.sort(key=lambda r: (str(r["regime"]), int(r["seed"]), int(r["event_rank"])))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(args.out_csv)

    if args.write_regime_splits:
        stem = args.out_csv.stem
        suffix = args.out_csv.suffix
        for regime in ("N", "R"):
            reg_rows = [r for r in out_rows if str(r["regime"]) == regime]
            reg_path = args.out_csv.with_name(f"{stem}_{regime}{suffix}")
            with reg_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
                writer.writeheader()
                writer.writerows(reg_rows)
            print(reg_path)


if __name__ == "__main__":
    main()
