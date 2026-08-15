#!/usr/bin/env python3
"""
Analyze output of report_replicators.py from stdin.

- Counts replicator events (score >= threshold)
- Also reports score split among detected replicator events:
    score >= min-score vs score < min-score
- Classifies each replicator into a reverse-copy body class:
    H0
    H1
    mixed
    Unclassified
- Prints one example per class.
- Also classifies per-run dominant class in an epoch window.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

from find_takeover_epochs import classify_takeover_runs

COPY_OPS = {".", ","}
H0_OPS = {"<", ">"}
H1_OPS = {"{", "}"}
INC_DEC_OPS = {"+", "-"}
OP_SYMBOLS = {"[", "]", "+", "-", ".", ",", "<", ">", "{", "}", "0"}
BRACKET_OPS = {"[", "]"}

MIN_SCORE = 60
SCORE_RE = re.compile(r"score=(\d+)")
HEADER_META_RE = re.compile(r"seed=(\d+),\s*epoch=(\d+)\D+(\d+)")
HEADER_RANGE_RE = re.compile(r"epoch=(\d+)\s*→\s*(\d+)\s*\(len=(\d+)\)")
MECHANISM_LABELS = ("H0R", "H0ZS", "H1R", "Unclassified")
FAMILY_LABELS = ("palindromic", "offset", "nine", "ten", "other")
FAMILY_DISPLAY = {
    "palindromic": "Palindromic family",
    "offset": "Offset family",
    "nine": "9-symbol family",
    "ten": "10-symbol family",
    "other": "Unclassified family",
}
REVERSE_COPY_CLASS_LABELS = ("H0", "H1", "mixed", "Unclassified")
REVERSE_COPY_CLASS_DISPLAY = {
    "H0": "H0 class",
    "H1": "H1 class",
    "mixed": "mixed class",
    "Unclassified": "Unclassified class",
}


def opposite_dirs(h0: str, h1: str) -> bool:
    return (h0 == "<" and h1 == "}") or (h0 == ">" and h1 == "{")


def cleanse(ops: list[str]) -> list[str]:
    cancel_pairs = {(">", "<"), ("}", "{"), ("<", ">"), ("{", "}")}
    res: list[str] = []

    for op in ops:
        if op == '0':
            continue

        if res:
            last = res[-1]
            if op in COPY_OPS and last in COPY_OPS:
                continue
            if (last, op) in cancel_pairs:
                res.pop()
                continue
        res.append(op)

    return res


def is_valid_body_legacy(body: list[str]) -> bool:
    if len(body) not in {5, 6}:
        return False

    loop_start = ["[" for c in body if c == "["]
    copies = [c for c in body if c in COPY_OPS]
    h0s = [c for c in body if c in H0_OPS]
    h1s = [c for c in body if c in H1_OPS]
    loop_end = ["]" for c in body if c == "]"]
    inc_decs = [c for c in body if c in INC_DEC_OPS]

    if not (
        len(loop_start) == len(copies) == len(h0s) == len(h1s) == len(loop_end) == 1
    ):
        return False

    if len(body) == 5:
        return len(inc_decs) == 0 and opposite_dirs(h0s[0], h1s[0])

    if len(inc_decs) != 1 or copies[0] != ",":
        return False
    if not any(
        body[i] == "," and body[i - 1] in INC_DEC_OPS for i in range(1, len(body))
    ):
        return False

    return opposite_dirs(h0s[0], h1s[0])


def find_reverse_replicator_legacy(ops: list[str]) -> tuple[int, int] | None:
    # Legacy fixed-length matcher kept exclusively for palindromic detection.
    n = len(ops)
    for i in range(n):
        if ops[i] != "[":
            continue

        body_start = i
        for body_len in (5, 6):
            body_end = body_start + body_len
            if body_end >= n:
                continue

            if not is_valid_body_legacy(ops[body_start:body_end]):
                continue

            return body_start, body_end

    return None


def net_head_motion(ops: list[str]) -> tuple[int, int]:
    """Compute net head motion after cancellation across the whole segment."""
    net_h0 = ops.count(">") - ops.count("<")
    net_h1 = ops.count("}") - ops.count("{")
    return net_h0, net_h1


def iter_simple_loop_bounds(ops: list[str], start_idx: int = 0):
    """
    Yield [start, end) loop bounds for loops whose interiors contain no brackets.
    """
    n = len(ops)
    for i in range(max(0, start_idx), n):
        if ops[i] != "[":
            continue
        j = i + 1
        while j < n and ops[j] not in BRACKET_OPS:
            j += 1
        if j < n and ops[j] == "]":
            yield i, j + 1


def is_valid_replicator_loop(
    inner_ops: list[str], *, require_same_direction: bool
) -> bool:
    """
    Updated body definition:
    - At least one copy op ('.' or ','); any number is allowed.
    - Net head0 movement is exactly one or two steps after cancellation.
    - Net head1 movement is exactly one or two steps after cancellation.
    - Regular body requires opposite directions; single-direction body requires
      the same direction.
    """
    if not any(op in COPY_OPS for op in inner_ops):
        return False

    net_h0, net_h1 = net_head_motion(inner_ops)
    if abs(net_h0) not in {1, 2} or abs(net_h1) not in {1, 2}:
        return False

    if require_same_direction:
        return net_h0 == net_h1
    return net_h0 == -net_h1


def find_replicator_body(
    ops: list[str], *, require_same_direction: bool, start_idx: int = 0
) -> tuple[int, int] | None:
    """
    Return first [start, end) loop bounds matching the updated body definition.
    """
    for i, end in iter_simple_loop_bounds(ops, start_idx=start_idx):
        inner_ops = ops[i + 1 : end - 1]
        if is_valid_replicator_loop(
            inner_ops, require_same_direction=require_same_direction
        ):
            return i, end
    return None


def is_palindromic(raw_tape: str) -> bool:
    tape = list(raw_tape)
    if len(tape) % 2 != 0:
        return False

    half = len(tape) // 2
    left = tape[:half]
    right = tape[half:]
    if right != list(reversed(left)):
        return False

    left_ops = cleanse([c for c in left if c in OP_SYMBOLS])
    # Keep palindromic detection unchanged.
    return find_reverse_replicator_legacy(left_ops) is not None


def is_valid_offset_copy_loop(inner_ops: list[str]) -> bool:
    """
    Offset pre-loop rule:
    1) net h0 is +/-1 or +/-2 and net h1 is 0.
    2) or net h1 is +/-1 or +/-2 and net h0 is 0, with a comma present and
       no inc-dec ops after the first comma. Additional copy ops are allowed.

    In both cases the loop must include at least one copy op.
    """
    if not any(op in COPY_OPS for op in inner_ops):
        return False

    net_h0, net_h1 = net_head_motion(inner_ops)
    if abs(net_h0) in {1, 2} and net_h1 == 0:
        return True

    if abs(net_h1) in {1, 2} and net_h0 == 0 and "," in inner_ops:
        comma_idx = inner_ops.index(",")
        forbidden_after_comma = INC_DEC_OPS
        if any(op in forbidden_after_comma for op in inner_ops[comma_idx + 1 :]):
            return False
        return True

    return False


def is_offset_replicator(raw_ops: list[str]) -> bool:
    for i, end in iter_simple_loop_bounds(raw_ops):
        loop_inner = raw_ops[i + 1 : end - 1]
        if not is_valid_offset_copy_loop(loop_inner):
            continue

        # Offset requires a later single-direction body.
        if (
            find_replicator_body(
                raw_ops, require_same_direction=True, start_idx=end
            )
            is not None
        ):
            return True

    return False


def classify_nine_ten(ops: list[str]) -> str | None:
    n = len(ops)
    body_1 = find_replicator_body(ops, require_same_direction=False)
    body_2_reversed = find_replicator_body(
        list(reversed(ops)), require_same_direction=False
    )
    body_2 = (
        (n - body_2_reversed[1], n - body_2_reversed[0])
        if body_2_reversed
        else None
    )
    if not body_1 or not body_2:
        return None
    if body_1[1] == body_2[0]:
        return "nine"

    return "ten"


def classify_replicator(raw_ops: list[str], raw_tape: str) -> str | None:
    # Requested precedence: 9/10 and palindromic first, offset last.
    rep_type = classify_nine_ten(raw_ops)
    if rep_type == "ten" and is_palindromic(raw_tape):
        return "palindromic"
    if rep_type is not None:
        return rep_type
    if is_offset_replicator(raw_ops):
        return "offset"
    return None


def parse_header_meta(header: str) -> tuple[int, int, int] | None:
    """
    Parse `(seed, start_epoch, end_epoch)` from a report_replicators header line.
    """
    match = HEADER_META_RE.search(header)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def overlap_epochs(
    start_epoch: int, end_epoch: int, window_start: int, window_end: int
) -> int:
    start = max(start_epoch, window_start)
    end = min(end_epoch, window_end)
    if end < start:
        return 0
    return end - start + 1


def find_first_body(ops: list[str]) -> list[str] | None:
    try:
        start = ops.index("[")
    except ValueError:
        return None
    try:
        end = ops.index("]", start + 1)
    except ValueError:
        return None
    if end <= start + 1:
        return []
    return ops[start + 1 : end]


def classify_execution_mechanism(ops: list[str]) -> str:
    """
    Heuristic execution-mechanism classifier aligned with
    analyze_replicator_body_types:
    - H0R/H0ZS: loop body includes both '<' and ','; order distinguishes robust
      vs zero-susceptible behavior.
    - H1R: loop body includes '>' and no H0R/H0ZS signature.
    - Unclassified: otherwise.
    """
    body = find_first_body(ops)
    if body is None:
        return "Unclassified"

    has_lt = "<" in body
    has_comma = "," in body
    if has_lt and has_comma:
        if body.index("<") < body.index(","):
            return "H0R"
        return "H0ZS"

    if ">" in body:
        return "H1R"

    return "Unclassified"


def classify_body_direction(inner_ops: list[str]) -> str | None:
    net_h0, net_h1 = net_head_motion(inner_ops)
    if net_h0 < 0 and net_h1 > 0:
        return "H0"
    if net_h0 > 0 and net_h1 < 0:
        return "H1"
    return None


def classify_reverse_copy_class(raw_ops: list[str]) -> str:
    body_classes: set[str] = set()

    body_1 = find_replicator_body(raw_ops, require_same_direction=False)
    if body_1 is not None:
        inner_1 = raw_ops[body_1[0] + 1 : body_1[1] - 1]
        body_class_1 = classify_body_direction(inner_1)
        if body_class_1 is not None:
            body_classes.add(body_class_1)

    reversed_ops = list(reversed(raw_ops))
    body_2 = find_replicator_body(reversed_ops, require_same_direction=False)
    if body_2 is not None:
        inner_2 = reversed_ops[body_2[0] + 1 : body_2[1] - 1]
        body_class_2 = classify_body_direction(inner_2)
        if body_class_2 is not None:
            body_classes.add(body_class_2)

    if body_classes == {"H0"}:
        return "H0"
    if body_classes == {"H1"}:
        return "H1"
    if body_classes == {"H0", "H1"}:
        return "mixed"
    return "Unclassified"


def family_from_rep_type(rep_type: str | None) -> str:
    if rep_type in {"palindromic", "offset", "nine", "ten"}:
        return rep_type
    return "other"


def family_example_limit(family: str) -> int:
    return 5 if family == "other" else 1


def maybe_add_example(
    bucket: dict[str, list[tuple[str, str, str]]],
    label: str,
    example: tuple[str, str, str],
    limit: int,
) -> None:
    items = bucket.setdefault(label, [])
    if len(items) < limit:
        items.append(example)


def reverse_copy_example_limit(label: str) -> int:
    return 5 if label == "Unclassified" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify replicator types either from stdin (report_replicators output) "
            "or directly from a run mode."
        )
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["reinit", "interaction", "random"],
        help=(
            "Optional run mode. If provided, this script will invoke "
            "scripts/report_replicators.py <mode> and analyze its output."
        ),
    )
    parser.add_argument(
        "--group",
        default="no_mutation",
        help=(
            "Run group for interaction/random modes when invoking "
            "report_replicators.py (runs/<group>/<mode>/). Ignored for reinit."
        ),
    )
    parser.add_argument(
        "--interaction-scope",
        choices=["all", "cond"],
        default="all",
        help=(
            "For interaction mode only: 'all' uses the full run, "
            "'cond' clips takeover runs at the takeover epoch inclusive."
        ),
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=MIN_SCORE,
        help="Minimum score for counting/classifying replicator events.",
    )
    parser.add_argument(
        "--epoch-start",
        type=int,
        default=15360,
        help="Start epoch (inclusive) for per-run execution-mechanism summary.",
    )
    parser.add_argument(
        "--epoch-end",
        type=int,
        default=15872,
        help="End epoch (inclusive) for per-run execution-mechanism summary.",
    )
    parser.add_argument(
        "--min-mechanism-count",
        type=int,
        default=512,
        help=(
            "Only report runs with at least this many qualifying replicator-epochs "
            "inside the execution-mechanism window."
        ),
    )
    return parser.parse_args()


def rewrite_header_epoch_range(header: str, start_epoch: int, end_epoch: int) -> str:
    duration = end_epoch - start_epoch + 1
    return HEADER_RANGE_RE.sub(
        f"epoch={start_epoch} → {end_epoch} (len={duration})",
        header,
        count=1,
    )


def iter_report_blocks(lines: list[str]):
    for block_start in range(0, len(lines), 4):
        if block_start + 3 >= len(lines):
            break
        yield (
            lines[block_start],
            lines[block_start + 1],
            lines[block_start + 2],
            lines[block_start + 3],
        )


def filter_interaction_lines_to_cond(lines: list[str], group: str) -> list[str]:
    interaction_root = Path("runs") / group / "interaction"
    takeover = classify_takeover_runs(interaction_root)
    takeover_epochs = takeover.takeover_epochs

    filtered: list[str] = []
    for sep, header, full_tape, ops_line in iter_report_blocks(lines):
        meta = parse_header_meta(header)
        if meta is None:
            filtered.extend([sep, header, full_tape, ops_line])
            continue
        seed, start_epoch, end_epoch = meta
        cutoff_epoch = takeover_epochs.get(seed)
        if cutoff_epoch is None:
            filtered.extend([sep, header, full_tape, ops_line])
            continue
        if start_epoch > cutoff_epoch:
            continue
        clipped_end = min(end_epoch, cutoff_epoch)
        if clipped_end < start_epoch:
            continue
        if clipped_end != end_epoch:
            header = rewrite_header_epoch_range(header, start_epoch, clipped_end)
        filtered.extend([sep, header, full_tape, ops_line])
    return filtered


def load_input_lines(mode: str | None, group: str, interaction_scope: str) -> list[str]:
    if mode:
        report_script = Path(__file__).with_name("report_replicators.py")
        proc = subprocess.run(
            [sys.executable, str(report_script), mode, "--group", group],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            err = proc.stderr.strip() or "report_replicators.py failed"
            raise SystemExit(err)
        lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
        if mode == "interaction" and interaction_scope == "cond":
            return filter_interaction_lines_to_cond(lines, group)
        return lines

    if sys.stdin.isatty():
        raise SystemExit(
            "No input provided on stdin.\n"
            "Use:\n"
            "  python3 scripts/report_replicators.py <mode> --group <group> | "
            "python3 scripts/analyze_replicator_types.py\n"
            "or:\n"
            "  python3 scripts/analyze_replicator_types.py <mode> --group <group>"
        )

    return [l.strip() for l in sys.stdin.read().splitlines() if l.strip()]


def main() -> None:
    args = parse_args()
    if args.epoch_end < args.epoch_start:
        raise SystemExit("--epoch-end must be greater than or equal to --epoch-start")
    if args.min_mechanism_count < 0:
        raise SystemExit("--min-mechanism-count must be non-negative")

    lines = load_input_lines(args.mode, args.group, args.interaction_scope)

    total = 0
    score_ge_min = 0
    score_lt_min = 0
    class_counts = {label: 0 for label in REVERSE_COPY_CLASS_LABELS}
    class_examples: dict[str, list[tuple[str, str, str]]] = {}

    class_by_run: dict[int, dict[str, int]] = {}
    class_examples_by_run: dict[int, dict[str, tuple[int, int, str, str]]] = {}
    classes_by_run_set: dict[int, set[str]] = {}

    for block_start in range(0, len(lines), 4):
        if block_start + 3 >= len(lines):
            break

        header = lines[block_start + 1]
        full_tape = lines[block_start + 2]
        ops_line = lines[block_start + 3]
        run_meta = parse_header_meta(header)

        match = SCORE_RE.search(header)
        if not match:
            continue
        score = int(match.group(1))
        if score >= args.min_score:
            score_ge_min += 1
        else:
            score_lt_min += 1
        if score < args.min_score:
            continue

        raw_ops = [op for op in ops_line.split() if op in OP_SYMBOLS]
        reverse_copy_class = classify_reverse_copy_class(raw_ops)

        total += 1
        class_counts[reverse_copy_class] += 1

        maybe_add_example(
            class_examples,
            reverse_copy_class,
            (header, full_tape, ops_line),
            reverse_copy_example_limit(reverse_copy_class),
        )

        if run_meta is not None:
            run_id, start_epoch, end_epoch = run_meta
            classes_by_run_set.setdefault(run_id, set()).add(reverse_copy_class)
            overlap = overlap_epochs(
                start_epoch, end_epoch, args.epoch_start, args.epoch_end
            )
            if overlap > 0:
                row = class_by_run.setdefault(
                    run_id,
                    {"total": 0, "H0": 0, "H1": 0, "mixed": 0, "Unclassified": 0},
                )
                row["total"] += overlap
                row[reverse_copy_class] += overlap
                examples = class_examples_by_run.setdefault(run_id, {})
                if reverse_copy_class not in examples:
                    examples[reverse_copy_class] = (
                        start_epoch,
                        end_epoch,
                        full_tape,
                        ops_line,
                    )

    class_run_counts = {label: 0 for label in REVERSE_COPY_CLASS_LABELS}
    for run_classes in classes_by_run_set.values():
        for label in run_classes:
            class_run_counts[label] += 1

    print("Total replicator events:", total)
    print("H0 class:", class_counts["H0"])
    print("H1 class:", class_counts["H1"])
    print("Mixed class:", class_counts["mixed"])
    print("Unclassified class:", class_counts["Unclassified"])
    print(
        f"Detected replicators with score >= {args.min_score}:",
        score_ge_min,
    )
    print(
        f"Detected replicators with score < {args.min_score}:",
        score_lt_min,
    )

    print("\nNumber of runs in which at least one replicator of class X appears:")
    print(f"H0 {class_run_counts['H0']}")
    print(f"H1 {class_run_counts['H1']}")
    print(f"mixed {class_run_counts['mixed']}")
    print(f"Unclassified {class_run_counts['Unclassified']}")

    print("\n=== Class examples ===")
    for label in REVERSE_COPY_CLASS_LABELS:
        count = class_counts[label]
        if count == 0:
            continue
        examples = class_examples.get(label, [])
        print(f"{REVERSE_COPY_CLASS_DISPLAY[label]} (n={count}):")
        for header, full_tape, ops_line in examples:
            print(header)
            print(full_tape)
            print(ops_line)

    print(
        f"\n=== Class by run (epochs {args.epoch_start}-{args.epoch_end}, "
        f"score>={args.min_score}, min_n={args.min_mechanism_count}) ==="
    )
    printed_any = False
    for run_id in sorted(class_by_run):
        row = class_by_run[run_id]
        total_for_run = row["total"]
        if total_for_run < args.min_mechanism_count:
            continue
        dominant = max(REVERSE_COPY_CLASS_LABELS, key=lambda label: row[label])
        h0_pct = 100.0 * row["H0"] / total_for_run
        h1_pct = 100.0 * row["H1"] / total_for_run
        mixed_pct = 100.0 * row["mixed"] / total_for_run
        unclassified_pct = 100.0 * row["Unclassified"] / total_for_run
        print(
            f"Run {run_id}: class={dominant} "
            f"(H0={h0_pct:.2f}% H1={h1_pct:.2f}% "
            f"mixed={mixed_pct:.2f}% Unclassified={unclassified_pct:.2f}%, n={total_for_run})"
        )
        run_examples = class_examples_by_run.get(run_id, {})
        for label in REVERSE_COPY_CLASS_LABELS:
            if row[label] <= 0:
                continue
            example = run_examples.get(label)
            if example is None:
                continue
            start_epoch, end_epoch, full_tape, ops_line = example
            print(f"    - one {label} example: epoch={start_epoch}->{end_epoch}")
            print(f"      {full_tape}")
            print(f"      {ops_line}")
        printed_any = True
    if not printed_any:
        print("(none)")



if __name__ == "__main__":
    main()
