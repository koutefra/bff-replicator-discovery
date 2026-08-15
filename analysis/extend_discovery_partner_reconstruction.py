#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import os
import queue
import re
import shutil
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from find_takeover_epochs import classify_takeover_runs
from report_replicators import OP_SYMBOLS, decode_hex_tape


HEADER_STRUCT = struct.Struct("=QQQ")
K_SINGLE_TAPE_SIZE = 64
K_TAPE_SIZE = 2 * K_SINGLE_TAPE_SIZE
K_STEP_CAP = 8 * 1024
TAKEOVER_THRESHOLD = 0.0
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
OP_BYTES = {ord(c) for c in "[]+-.,<>{}"}
MOVE_OPS = {"<": (-1, 0), ">": (1, 0), "{": (0, -1), "}": (0, 1)}


@dataclass(frozen=True)
class EventSpec:
    regime: str  # "N" or "R"
    seed: int
    event_rank: int  # 1-based rank within seed
    epoch: int
    score: float
    tape_hex: str
    selfrep_index: int
    takeover_epoch: int | None


@dataclass(frozen=True)
class SeedSelection:
    regime: str
    seed: int
    takeover_epoch: int | None
    first_epoch: int
    second_epoch: int
    events: tuple[EventSpec, EventSpec]


@dataclass(frozen=True)
class DumpJob:
    regime: str
    seed: int
    gpu: int
    target_epochs: tuple[int, ...]
    run_dir: Path


@dataclass(frozen=True)
class EvalResult:
    tape: bytes
    executed_steps: int
    termination_reason: str


@dataclass(frozen=True)
class FirstProtoEvent:
    step: int
    open_idx: int
    close_idx: int
    open_side: str
    close_side: str
    loop_ops: str
    delta_h0: int
    delta_h1: int
    span_changed: bool
    loops_before: int
    nonproto_before: int


def parse_csv_int_set(text: str) -> set[int]:
    out: set[int] = set()
    text = text.strip()
    if not text:
        return out
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        out.add(int(p))
    return out


def parse_csv_int_list(text: str) -> list[int]:
    out: list[int] = []
    text = text.strip()
    if not text:
        return out
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def ensure_csv_field_size_limit() -> None:
    try:
        csv.field_size_limit(2**31 - 1)
    except OverflowError:
        csv.field_size_limit(2**30 - 1)


def parse_seed_from_log(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    if not m:
        raise ValueError(f"Unable to parse seed from {path}")
    return int(m.group(1))


def parse_scores_and_tapes(row: dict[str, str]) -> tuple[list[float], list[str]]:
    scores_raw = row.get("selfrep_scores", "")
    tapes_raw = row.get("selfrep_tapes", "")
    try:
        scores_parsed = ast.literal_eval(scores_raw)
    except (SyntaxError, ValueError):
        scores_parsed = []
    try:
        tapes_parsed = ast.literal_eval(tapes_raw)
    except (SyntaxError, ValueError):
        tapes_parsed = []

    scores: list[float] = []
    if isinstance(scores_parsed, list):
        for x in scores_parsed:
            try:
                scores.append(float(x))
            except (TypeError, ValueError):
                scores.append(float("-inf"))
    tapes: list[str] = []
    if isinstance(tapes_parsed, list):
        for x in tapes_parsed:
            if isinstance(x, str):
                tapes.append(x.strip())
            else:
                tapes.append("")
    return scores, tapes


def has_score_at_least(value: str, threshold: float) -> bool:
    for token in NUMBER_RE.findall(value or ""):
        try:
            if float(token) >= threshold:
                return True
        except ValueError:
            continue
    return False


def best_replicator_for_row(
    row: dict[str, str], score_threshold: float
) -> tuple[int, float, str] | None:
    """
    Return the best (index, score, tape_hex) replicator at or above threshold
    for one log row, or None if no qualifying replicator is present.
    """
    scores, tapes = parse_scores_and_tapes(row)
    if not scores or not tapes:
        return None

    best_idx = -1
    best_score = float("-inf")
    for idx, score in enumerate(scores):
        if idx >= len(tapes):
            break
        if score < score_threshold:
            continue
        tape_hex = tapes[idx]
        if len(tape_hex) != 2 * K_SINGLE_TAPE_SIZE:
            continue
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx < 0:
        return None
    return best_idx, best_score, tapes[best_idx]


def first_n_events_for_seed(
    *,
    regime: str,
    seed: int,
    log_path: Path,
    score_threshold: float,
    n_events: int,
    max_epoch: int | None,
    takeover_epoch: int | None,
) -> list[EventSpec]:
    """
    Discovery events are transitions from 0 -> 1+ qualifying replicators
    (thresholded by score_threshold), not every epoch with qualifying hits.
    """
    events: list[EventSpec] = []
    prev_has_replicator = False
    with log_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            epoch = int(row["epoch"])
            if max_epoch is not None and epoch > max_epoch:
                break

            best = best_replicator_for_row(row, score_threshold)
            has_replicator = best is not None
            is_discovery = has_replicator and (not prev_has_replicator)

            if is_discovery:
                best_idx, best_score, tape_hex = best
                events.append(
                    EventSpec(
                        regime=regime,
                        seed=seed,
                        event_rank=len(events) + 1,
                        epoch=epoch,
                        score=best_score,
                        tape_hex=tape_hex,
                        selfrep_index=best_idx,
                        takeover_epoch=takeover_epoch,
                    )
                )
                if len(events) >= n_events:
                    break

            prev_has_replicator = has_replicator
    return events


def select_seeds_and_events(
    *,
    logs_dir: Path,
    regime: str,
    score_threshold: float,
    events_per_seed: int,
    events_total: int,
    excluded_seeds: set[int],
    takeover_epochs: dict[int, int] | None,
) -> list[SeedSelection]:
    if events_total % events_per_seed != 0:
        raise ValueError(
            f"events_total ({events_total}) must be divisible by events_per_seed ({events_per_seed})"
        )
    needed_seeds = events_total // events_per_seed
    candidates: list[SeedSelection] = []

    for log_path in sorted(logs_dir.glob("log_*.log")):
        seed = parse_seed_from_log(log_path)
        if seed in excluded_seeds:
            continue
        takeover_epoch = None
        max_epoch = None
        if takeover_epochs is not None:
            takeover_epoch = takeover_epochs.get(seed)
            max_epoch = takeover_epoch
        events = first_n_events_for_seed(
            regime=regime,
            seed=seed,
            log_path=log_path,
            score_threshold=score_threshold,
            n_events=events_per_seed,
            max_epoch=max_epoch,
            takeover_epoch=takeover_epoch,
        )
        if len(events) != events_per_seed:
            continue
        candidates.append(
            SeedSelection(
                regime=regime,
                seed=seed,
                takeover_epoch=takeover_epoch,
                first_epoch=events[0].epoch,
                second_epoch=events[1].epoch,
                events=(events[0], events[1]),
            )
        )

    candidates.sort(key=lambda c: (c.second_epoch, c.first_epoch, c.seed))
    if len(candidates) < needed_seeds:
        raise RuntimeError(
            f"Not enough seeds for regime {regime}: need {needed_seeds}, found {len(candidates)}"
        )
    return candidates[:needed_seeds]


def splitmix64(seed: int) -> int:
    z = (seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF


def seeded(global_seed: int, seed2: int) -> int:
    return splitmix64(splitmix64(global_seed) ^ splitmix64(seed2))


def shuffled_indices(global_seed: int, internal_epoch: int, num_programs: int) -> list[int]:
    s = list(range(num_programs))
    length = num_programs
    for i in range(length - 1, -1, -1):
        j = splitmix64(seeded(global_seed, internal_epoch * length + i)) % (i + 1)
        s[i], s[j] = s[j], s[i]
    return s


def partner_for_slot_r(
    global_seed: int, logical_epoch: int, focal_slot: int, num_programs: int
) -> tuple[int, str]:
    internal_epoch = logical_epoch - 1
    s = shuffled_indices(global_seed, internal_epoch, num_programs)
    pos = s.index(focal_slot)
    partner = s[pos ^ 1]
    return partner, ("p1" if pos % 2 == 0 else "p2")


def build_partner_n(seed: int, internal_epoch: int, slot: int) -> tuple[bytes, bool]:
    base_seed = splitmix64(splitmix64(seed) ^ splitmix64(internal_epoch))
    partner_seed = splitmix64(base_seed ^ splitmix64(slot))
    partner = bytes(
        splitmix64((partner_seed + i) & 0xFFFFFFFFFFFFFFFF) & 0xFF
        for i in range(K_SINGLE_TAPE_SIZE)
    )
    a_first = (splitmix64(partner_seed ^ 0x9E3779B97F4A7C15) & 1) == 0
    return partner, a_first


def get_op_kind(byte: int) -> str:
    if byte == ord("["):
        return "loop_start"
    if byte == ord("]"):
        return "loop_end"
    if byte == ord("+"):
        return "plus"
    if byte == ord("-"):
        return "minus"
    if byte == ord("."):
        return "copy01"
    if byte == ord(","):
        return "copy10"
    if byte == ord("<"):
        return "dec0"
    if byte == ord(">"):
        return "inc0"
    if byte == ord("{"):
        return "dec1"
    if byte == ord("}"):
        return "inc1"
    if byte == 0:
        return "null"
    return "noop"


def decode_tape(tape: bytes) -> str:
    return "".join(decode_hex_tape(tape.hex()))


def strip_tape(tape: bytes) -> str:
    return "".join(ch for ch in decode_tape(tape) if ch in OP_SYMBOLS)


def load_dump(path: Path) -> tuple[int, int, np.ndarray]:
    raw = path.read_bytes()
    _, num_programs, epoch_saved = HEADER_STRUCT.unpack_from(raw, 0)
    soup = np.frombuffer(raw, dtype=np.uint8, offset=HEADER_STRUCT.size).reshape(
        num_programs, K_SINGLE_TAPE_SIZE
    )
    return epoch_saved, num_programs, soup.copy()


def count_static_proto_loops(strip: str) -> int:
    stack: list[int] = []
    count = 0
    for i, ch in enumerate(strip):
        if ch == "[":
            stack.append(i)
            continue
        if ch != "]" or not stack:
            continue
        j = stack.pop()
        body = strip[j + 1 : i]
        body_ops = "".join(c for c in body if c in "[]+-.,<>{}")
        has_copy = ("." in body_ops) or ("," in body_ops)
        has_disruptive = ("+" in body_ops) or ("-" in body_ops)
        dh0 = 0
        dh1 = 0
        for c in body_ops:
            delta = MOVE_OPS.get(c)
            if delta is None:
                continue
            dh0 += delta[0]
            dh1 += delta[1]
        if has_copy and (not has_disruptive) and {abs(dh0), abs(dh1)} == {1, 2}:
            count += 1
    return count


def compute_loop_metrics(mem: bytearray, open_idx: int, close_idx: int) -> tuple[str, int, int, bool]:
    body_ops = "".join(
        chr(mem[i]) for i in range(open_idx + 1, close_idx) if mem[i] in OP_BYTES
    )
    has_copy = ("." in body_ops) or ("," in body_ops)
    has_disruptive = ("+" in body_ops) or ("-" in body_ops)
    dh0 = 0
    dh1 = 0
    for c in body_ops:
        delta = MOVE_OPS.get(c)
        if delta is None:
            continue
        dh0 += delta[0]
        dh1 += delta[1]
    is_proto = has_copy and (not has_disruptive) and {abs(dh0), abs(dh1)} == {1, 2}
    return body_ops, dh0, dh1, is_proto


def evaluate_with_first_proto(tape: bytes) -> tuple[EvalResult, FirstProtoEvent | None]:
    mem = bytearray(tape)
    mem_initial = bytes(tape)
    pc = 0
    head0 = K_TAPE_SIZE
    head1 = K_TAPE_SIZE
    reason = "step_cap"
    executed = 0
    loops_seen = 0
    proto_seen = 0
    first_proto: FirstProtoEvent | None = None

    for i in range(K_STEP_CAP):
        head0 &= K_TAPE_SIZE - 1
        head1 &= K_TAPE_SIZE - 1
        bracket_mismatch = False
        kind = get_op_kind(mem[pc])

        if kind == "dec0":
            head0 -= 1
        elif kind == "inc0":
            head0 += 1
        elif kind == "dec1":
            head1 -= 1
        elif kind == "inc1":
            head1 += 1
        elif kind == "plus":
            mem[head0] = (mem[head0] + 1) & 0xFF
        elif kind == "minus":
            mem[head0] = (mem[head0] - 1) & 0xFF
        elif kind == "copy01":
            mem[head1] = mem[head0]
        elif kind == "copy10":
            mem[head0] = mem[head1]
        elif kind == "loop_start":
            if get_op_kind(mem[head0]) == "null":
                scan_closed = 1
                pc += 1
                while pc < K_TAPE_SIZE and scan_closed > 0:
                    k = get_op_kind(mem[pc])
                    if k == "loop_end":
                        scan_closed -= 1
                    if k == "loop_start":
                        scan_closed += 1
                    pc += 1
                pc -= 1
                if scan_closed != 0:
                    bracket_mismatch = True
                    pc = K_TAPE_SIZE
        elif kind == "loop_end":
            if get_op_kind(mem[head0]) != "null":
                scan_open = 1
                close_idx = pc
                pc -= 1
                while pc >= 0 and scan_open > 0:
                    k = get_op_kind(mem[pc])
                    if k == "loop_end":
                        scan_open += 1
                    if k == "loop_start":
                        scan_open -= 1
                    pc -= 1
                pc += 1
                if scan_open != 0:
                    bracket_mismatch = True
                    pc = -1
                else:
                    open_idx = pc
                    loops_seen += 1
                    body_ops, dh0, dh1, is_proto = compute_loop_metrics(mem, open_idx, close_idx)
                    if is_proto:
                        proto_seen += 1
                        if first_proto is None:
                            span_changed = any(
                                mem[idx] != mem_initial[idx]
                                for idx in range(open_idx, close_idx + 1)
                            )
                            first_proto = FirstProtoEvent(
                                step=i + 1,
                                open_idx=open_idx,
                                close_idx=close_idx,
                                open_side="partner" if open_idx < K_SINGLE_TAPE_SIZE else "focal",
                                close_side="partner"
                                if close_idx < K_SINGLE_TAPE_SIZE
                                else "focal",
                                loop_ops=body_ops,
                                delta_h0=dh0,
                                delta_h1=dh1,
                                span_changed=span_changed,
                                loops_before=loops_seen - 1,
                                nonproto_before=(loops_seen - 1) - (proto_seen - 1),
                            )

        if pc < 0 or pc >= K_TAPE_SIZE:
            reason = "bracket_mismatch" if bracket_mismatch else "ip_out_of_bounds"
            executed = i + 1
            break
        pc += 1
        if pc >= K_TAPE_SIZE:
            reason = "ip_out_of_bounds"
            executed = i + 1
            break
        executed = i + 1

    return EvalResult(bytes(mem), executed, reason), first_proto


def run_single_dump_job(
    *,
    job: DumpJob,
    bin_main: Path,
    num_programs: int,
    mutation_prob: float,
    keep_checkpoints: bool,
) -> None:
    run_dir = job.run_dir
    dump_dir = run_dir / "dumps"
    logs_dir = run_dir / "logs"
    ckpt_root = run_dir / "checkpoints"
    dump_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    expected = [dump_dir / f"epoch_{ep}.dat" for ep in job.target_epochs]
    if all(path.exists() for path in expected):
        return

    load_from: Path | None = None
    phase_logs: list[Path] = []
    phase_dirs: list[Path] = []

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)

    for idx, target_epoch in enumerate(job.target_epochs, start=1):
        if target_epoch < 2:
            raise RuntimeError(
                f"target epoch must be >=2 for dump extraction, got {target_epoch}"
            )
        internal_epoch = target_epoch - 1
        phase_name = f"{job.regime.lower()}_phase{idx}"
        phase_dir = ckpt_root / phase_name
        phase_log = logs_dir / f"{phase_name}.log"
        phase_out = logs_dir / f"{phase_name}.out.log"
        phase_err = logs_dir / f"{phase_name}.err.log"
        phase_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(bin_main),
            "--lang",
            "bff_noheads",
            "--num",
            str(num_programs),
            "--print_interval",
            "512",
            "--log_interval",
            "1",
            "--mutation_prob",
            str(mutation_prob),
            "--seed",
            str(job.seed),
            "--disable_output",
            "--max_epochs",
            str(internal_epoch),
            "--checkpoint_dir",
            str(phase_dir),
            "--save_interval",
            str(internal_epoch),
            "--log",
            str(phase_log),
        ]
        if job.regime == "N":
            cmd.append("--random_partner_interaction")
        if load_from is not None:
            cmd.extend(["--load", str(load_from)])

        with phase_out.open("w") as out_fh, phase_err.open("w") as err_fh:
            subprocess.run(cmd, check=True, env=env, stdout=out_fh, stderr=err_fh)

        zero_dump = phase_dir / "0000000000.dat"
        if zero_dump.exists():
            zero_dump.unlink()

        dump_src = phase_dir / f"{internal_epoch:010d}.dat"
        if not dump_src.exists():
            raise RuntimeError(f"Expected checkpoint not found: {dump_src}")
        dump_dst = dump_dir / f"epoch_{target_epoch}.dat"
        shutil.copy2(dump_src, dump_dst)

        load_from = dump_src
        phase_logs.append(phase_log)
        phase_dirs.append(phase_dir)

    merged_log = logs_dir / f"{job.regime.lower()}_merged.log"
    if phase_logs:
        with merged_log.open("w") as out_fh:
            with phase_logs[0].open() as first:
                out_fh.write(first.read())
            for path in phase_logs[1:]:
                with path.open() as fh:
                    lines = fh.readlines()
                if not lines:
                    continue
                out_fh.writelines(lines[1:])

    if not keep_checkpoints:
        for phase_dir in phase_dirs:
            shutil.rmtree(phase_dir, ignore_errors=True)


def run_jobs_on_gpus(
    *,
    jobs: list[DumpJob],
    gpus: list[int],
    bin_main: Path,
    num_programs: int,
    mutation_prob: float,
    keep_checkpoints: bool,
) -> None:
    job_queue: queue.Queue[DumpJob] = queue.Queue()
    for job in jobs:
        job_queue.put(job)

    lock = threading.Lock()
    errors: list[str] = []
    completed = 0
    total = len(jobs)

    def worker(gpu: int) -> None:
        nonlocal completed
        while True:
            try:
                job = job_queue.get_nowait()
            except queue.Empty:
                return

            start = time.time()
            with lock:
                print(
                    f"[dump] start regime={job.regime} seed={job.seed} gpu={gpu} epochs={list(job.target_epochs)}",
                    flush=True,
                )
            try:
                run_single_dump_job(
                    job=DumpJob(
                        regime=job.regime,
                        seed=job.seed,
                        gpu=gpu,
                        target_epochs=job.target_epochs,
                        run_dir=job.run_dir,
                    ),
                    bin_main=bin_main,
                    num_programs=num_programs,
                    mutation_prob=mutation_prob,
                    keep_checkpoints=keep_checkpoints,
                )
                elapsed = time.time() - start
                with lock:
                    completed += 1
                    print(
                        f"[dump] done regime={job.regime} seed={job.seed} gpu={gpu} elapsed={elapsed:.1f}s ({completed}/{total})",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(
                        f"regime={job.regime} seed={job.seed} gpu={gpu}: {type(exc).__name__}: {exc}"
                    )
                    print(f"[dump] ERROR {errors[-1]}", flush=True)
            finally:
                job_queue.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        raise RuntimeError("Dump generation failed:\n" + "\n".join(errors))


def classify_dynamic(first_proto: FirstProtoEvent | None) -> str:
    if first_proto is None:
        return "NO_PROTO_FOUND"
    if first_proto.span_changed:
        return "DYNAMIC_ASSEMBLED"
    if first_proto.open_side == "partner" and first_proto.close_side == "focal":
        return "CROSS_TAPE_FOCAL_CLOSE"
    if first_proto.open_side == "partner" and first_proto.close_side == "partner":
        return "PARTNER_COMPLETE_LOOP"
    if first_proto.open_side == "focal" and first_proto.close_side == "focal":
        return "FOCAL_COMPLETE_LOOP"
    return "OTHER"


def reconstruct_all_events(
    *,
    selections: list[SeedSelection],
    dumps_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for sel in selections:
        seed_dir = dumps_root / sel.regime / f"seed_{sel.seed:03d}"
        dump_dir = seed_dir / "dumps"

        needed_epochs = sorted(
            {
                event.epoch
                for event in sel.events
            }
            | {event.epoch - 1 for event in sel.events}
        )

        dump_cache: dict[int, tuple[int, np.ndarray]] = {}
        for epoch in needed_epochs:
            path = dump_dir / f"epoch_{epoch}.dat"
            if not path.exists():
                raise FileNotFoundError(f"Missing dump file: {path}")
            _, num_programs, soup = load_dump(path)
            dump_cache[epoch] = (num_programs, soup)

        for event in sel.events:
            num_programs_after, soup_after = dump_cache[event.epoch]
            num_programs_before, soup_before = dump_cache[event.epoch - 1]
            if num_programs_after != num_programs_before:
                raise RuntimeError(
                    f"num_programs mismatch for regime={sel.regime} seed={sel.seed} epoch={event.epoch}"
                )
            num_programs = num_programs_after

            target = np.frombuffer(bytes.fromhex(event.tape_hex), dtype=np.uint8)
            matches = np.where(np.all(soup_after == target, axis=1))[0]
            if len(matches) == 0:
                raise RuntimeError(
                    f"No slot match for event tape: regime={sel.regime} seed={sel.seed} epoch={event.epoch}"
                )
            focal_slot = int(matches[0])
            slot_match_count = int(len(matches))

            focal_before = bytes(soup_before[focal_slot].tolist())
            focal_after = bytes(soup_after[focal_slot].tolist())

            if sel.regime == "N":
                partner, a_first = build_partner_n(sel.seed, event.epoch - 1, focal_slot)
                focal_role = "p1" if a_first else "p2"
                partner_slot: int | None = None
            else:
                partner_slot, focal_role = partner_for_slot_r(
                    sel.seed, event.epoch, focal_slot, num_programs
                )
                partner = bytes(soup_before[partner_slot].tolist())
                a_first = None

            if focal_role == "p1":
                combined_before = focal_before + partner
                focal_offset = 0
            else:
                combined_before = partner + focal_before
                focal_offset = K_SINGLE_TAPE_SIZE

            eval_result, first_proto = evaluate_with_first_proto(combined_before)
            observed_after = eval_result.tape[focal_offset : focal_offset + K_SINGLE_TAPE_SIZE]
            replay_exact = int(observed_after == focal_after)

            partner_strip = strip_tape(partner)
            partner_static_proto_count = count_static_proto_loops(partner_strip)

            row: dict[str, Any] = {
                "regime": sel.regime,
                "seed": sel.seed,
                "event_rank": event.event_rank,
                "event_epoch": event.epoch,
                "event_score": event.score,
                "takeover_epoch": event.takeover_epoch if event.takeover_epoch is not None else "",
                "event_selfrep_index": event.selfrep_index,
                "event_tape_hex": event.tape_hex,
                "slot_match_count": slot_match_count,
                "focal_slot": focal_slot,
                "partner_slot": partner_slot if partner_slot is not None else "",
                "focal_role": focal_role,
                "a_first": int(a_first) if a_first is not None else "",
                "focal_before_hex": focal_before.hex(),
                "focal_before_strip": strip_tape(focal_before),
                "partner_hex": partner.hex(),
                "partner_strip": partner_strip,
                "focal_after_hex": focal_after.hex(),
                "focal_after_strip": strip_tape(focal_after),
                "replay_termination_reason": eval_result.termination_reason,
                "replay_steps": eval_result.executed_steps,
                "replay_exact": replay_exact,
                "partner_static_proto_loop_count": partner_static_proto_count,
                "partner_static_proto_loop_present": int(partner_static_proto_count > 0),
                "first_proto_found": int(first_proto is not None),
                "first_proto_step": first_proto.step if first_proto is not None else "",
                "first_proto_open_side": first_proto.open_side if first_proto is not None else "",
                "first_proto_close_side": first_proto.close_side if first_proto is not None else "",
                "first_proto_ops": first_proto.loop_ops if first_proto is not None else "",
                "first_proto_dh0": first_proto.delta_h0 if first_proto is not None else "",
                "first_proto_dh1": first_proto.delta_h1 if first_proto is not None else "",
                "first_proto_span_changed": int(first_proto.span_changed)
                if first_proto is not None
                else "",
                "loops_before_first_proto": first_proto.loops_before if first_proto is not None else "",
                "nonproto_before_first_proto": first_proto.nonproto_before
                if first_proto is not None
                else "",
                "dynamic_class": classify_dynamic(first_proto),
            }
            rows.append(row)

    rows.sort(key=lambda r: (r["regime"], int(r["seed"]), int(r["event_rank"])))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_selection_rows(selections: list[SeedSelection]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sel in selections:
        rows.append(
            {
                "regime": sel.regime,
                "seed": sel.seed,
                "takeover_epoch": sel.takeover_epoch if sel.takeover_epoch is not None else "",
                "first_epoch": sel.first_epoch,
                "second_epoch": sel.second_epoch,
            }
        )
    rows.sort(key=lambda r: (r["regime"], int(r["second_epoch"]), int(r["first_epoch"]), int(r["seed"])))
    return rows


def build_event_rows(selections: list[SeedSelection]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sel in selections:
        for event in sel.events:
            rows.append(
                {
                    "regime": sel.regime,
                    "seed": sel.seed,
                    "event_rank": event.event_rank,
                    "event_epoch": event.epoch,
                    "event_score": event.score,
                    "event_selfrep_index": event.selfrep_index,
                    "event_tape_hex": event.tape_hex,
                    "takeover_epoch": event.takeover_epoch if event.takeover_epoch is not None else "",
                }
            )
    rows.sort(key=lambda r: (r["regime"], int(r["seed"]), int(r["event_rank"])))
    return rows


def summarize_classes(reconstructed_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dynamic_counts: dict[tuple[str, str], int] = {}
    static_counts: dict[tuple[str, int], int] = {}
    for row in reconstructed_rows:
        regime = str(row["regime"])
        dynamic_class = str(row["dynamic_class"])
        dynamic_counts[(regime, dynamic_class)] = dynamic_counts.get((regime, dynamic_class), 0) + 1

        static_present = int(row["partner_static_proto_loop_present"])
        static_counts[(regime, static_present)] = static_counts.get((regime, static_present), 0) + 1

    dynamic_rows = [
        {"regime": reg, "dynamic_class": cls, "count": count}
        for (reg, cls), count in sorted(dynamic_counts.items())
    ]
    static_rows = [
        {"regime": reg, "partner_static_proto_loop_present": present, "count": count}
        for (reg, present), count in sorted(static_counts.items())
    ]
    return dynamic_rows, static_rows


def build_jobs(
    *,
    selections: list[SeedSelection],
    dumps_root: Path,
    gpus: list[int],
) -> list[DumpJob]:
    by_regime_seed: dict[tuple[str, int], set[int]] = {}
    for sel in selections:
        key = (sel.regime, sel.seed)
        epochs = by_regime_seed.setdefault(key, set())
        for event in sel.events:
            epochs.add(event.epoch - 1)
            epochs.add(event.epoch)

    jobs: list[DumpJob] = []
    gpu_idx = 0
    for (regime, seed), epochs in sorted(by_regime_seed.items()):
        target_epochs = tuple(sorted(epochs))
        gpu = gpus[gpu_idx % len(gpus)]
        gpu_idx += 1
        run_dir = dumps_root / regime / f"seed_{seed:03d}"
        jobs.append(
            DumpJob(
                regime=regime,
                seed=seed,
                gpu=gpu,
                target_epochs=target_epochs,
                run_dir=run_dir,
            )
        )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extend discovery reconstruction to 50 N and 50 R events (2 per seed), "
            "generate targeted dumps on specified GPUs, reconstruct focal/partner tapes, "
            "and classify partner mechanisms."
        )
    )
    parser.add_argument(
        "--random-logs-dir",
        type=Path,
        default=Path("runs/no_mutation/random"),
    )
    parser.add_argument(
        "--interaction-logs-dir",
        type=Path,
        default=Path("runs/no_mutation/interaction"),
    )
    parser.add_argument(
        "--bin-main",
        type=Path,
        default=Path("bin/main"),
    )
    parser.add_argument("--num-programs", type=int, default=131072)
    parser.add_argument("--mutation-prob", type=float, default=0.0)
    parser.add_argument("--score-threshold", type=float, default=60.0)
    parser.add_argument("--events-per-seed", type=int, default=2)
    parser.add_argument("--events-per-regime", type=int, default=50)
    parser.add_argument("--exclude-n-seeds", default="0,1")
    parser.add_argument("--exclude-r-seeds", default="2,4,7")
    parser.add_argument("--gpus", default="0,1,2,4")
    parser.add_argument(
        "--dumps-root",
        type=Path,
        default=None,
        help="Root for generated dump runs. Default: runs/no_mutation/discovery_extension_<timestamp>",
    )
    parser.add_argument(
        "--analysis-out-dir",
        type=Path,
        default=Path("analysis_plots/no_mutation"),
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep intermediate phase checkpoint directories.",
    )
    parser.add_argument(
        "--skip-dump-generation",
        action="store_true",
        help="Skip running bin/main and only reconstruct from existing dumps.",
    )
    args = parser.parse_args()

    ensure_csv_field_size_limit()

    gpus = parse_csv_int_list(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs specified")
    exclude_n = parse_csv_int_set(args.exclude_n_seeds)
    exclude_r = parse_csv_int_set(args.exclude_r_seeds)

    if args.dumps_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dumps_root = Path(f"runs/no_mutation/discovery_extension_{stamp}")
    else:
        dumps_root = args.dumps_root
    dumps_root.mkdir(parents=True, exist_ok=True)

    bin_main = args.bin_main
    if not bin_main.is_absolute():
        bin_main = Path.cwd() / bin_main
    if not bin_main.exists():
        raise FileNotFoundError(f"Missing binary: {bin_main}")

    print("[select] computing takeover epochs (interaction)...", flush=True)
    takeover = classify_takeover_runs(
        args.interaction_logs_dir,
        threshold=TAKEOVER_THRESHOLD,
    )
    takeover_epochs = takeover.takeover_epochs
    print(f"[select] takeover seeds: {len(takeover_epochs)}", flush=True)

    print("[select] selecting N seeds/events...", flush=True)
    selected_n = select_seeds_and_events(
        logs_dir=args.random_logs_dir,
        regime="N",
        score_threshold=args.score_threshold,
        events_per_seed=args.events_per_seed,
        events_total=args.events_per_regime,
        excluded_seeds=exclude_n,
        takeover_epochs=None,
    )
    print(
        f"[select] selected N seeds={len(selected_n)} events={len(selected_n) * args.events_per_seed}",
        flush=True,
    )

    print("[select] selecting R seeds/events...", flush=True)
    selected_r = select_seeds_and_events(
        logs_dir=args.interaction_logs_dir,
        regime="R",
        score_threshold=args.score_threshold,
        events_per_seed=args.events_per_seed,
        events_total=args.events_per_regime,
        excluded_seeds=exclude_r,
        takeover_epochs=takeover_epochs,
    )
    print(
        f"[select] selected R seeds={len(selected_r)} events={len(selected_r) * args.events_per_seed}",
        flush=True,
    )

    selections = selected_n + selected_r

    selection_rows = build_selection_rows(selections)
    event_rows = build_event_rows(selections)
    selection_csv = args.analysis_out_dir / "discovery_extension_seed_selection.csv"
    event_csv = args.analysis_out_dir / "discovery_extension_event_selection.csv"
    write_csv(selection_csv, selection_rows)
    write_csv(event_csv, event_rows)
    print(f"[write] {selection_csv}", flush=True)
    print(f"[write] {event_csv}", flush=True)

    jobs = build_jobs(selections=selections, dumps_root=dumps_root, gpus=gpus)
    jobs_csv_rows = [
        {
            "regime": job.regime,
            "seed": job.seed,
            "gpu_assigned": job.gpu,
            "target_epochs_csv": ",".join(str(ep) for ep in job.target_epochs),
            "run_dir": str(job.run_dir),
        }
        for job in jobs
    ]
    jobs_csv = args.analysis_out_dir / "discovery_extension_dump_jobs.csv"
    write_csv(jobs_csv, jobs_csv_rows)
    print(f"[write] {jobs_csv}", flush=True)

    if not args.skip_dump_generation:
        print(
            f"[dump] launching {len(jobs)} seed jobs across GPUs={gpus} with num_programs={args.num_programs}",
            flush=True,
        )
        run_jobs_on_gpus(
            jobs=jobs,
            gpus=gpus,
            bin_main=bin_main,
            num_programs=args.num_programs,
            mutation_prob=args.mutation_prob,
            keep_checkpoints=args.keep_checkpoints,
        )
        print("[dump] all jobs complete", flush=True)
    else:
        print("[dump] skipping generation (requested)", flush=True)

    print("[reconstruct] reconstructing focal/partner tapes and classifying...", flush=True)
    reconstructed_rows = reconstruct_all_events(selections=selections, dumps_root=dumps_root)
    recon_csv = args.analysis_out_dir / "discovery_extension_reconstructed_events.csv"
    write_csv(recon_csv, reconstructed_rows)
    print(f"[write] {recon_csv}", flush=True)

    n_rows = [row for row in reconstructed_rows if row["regime"] == "N"]
    r_rows = [row for row in reconstructed_rows if row["regime"] == "R"]
    n_csv = args.analysis_out_dir / "discovery_extension_reconstructed_events_N.csv"
    r_csv = args.analysis_out_dir / "discovery_extension_reconstructed_events_R.csv"
    write_csv(n_csv, n_rows)
    write_csv(r_csv, r_rows)
    print(f"[write] {n_csv}", flush=True)
    print(f"[write] {r_csv}", flush=True)

    dynamic_rows, static_rows = summarize_classes(reconstructed_rows)
    dynamic_csv = args.analysis_out_dir / "discovery_extension_dynamic_class_summary.csv"
    static_csv = args.analysis_out_dir / "discovery_extension_partner_static_summary.csv"
    write_csv(dynamic_csv, dynamic_rows)
    write_csv(static_csv, static_rows)
    print(f"[write] {dynamic_csv}", flush=True)
    print(f"[write] {static_csv}", flush=True)

    exact_replays = sum(int(row["replay_exact"]) for row in reconstructed_rows)
    print(
        "[done] "
        f"events={len(reconstructed_rows)} "
        f"N={len(n_rows)} R={len(r_rows)} "
        f"replay_exact={exact_replays}/{len(reconstructed_rows)} "
        f"dumps_root={dumps_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
