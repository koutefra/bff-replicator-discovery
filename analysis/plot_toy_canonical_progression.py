#!/usr/bin/env python3
"""
Plot the canonical 16-cell cross-boundary toy replicator progression.
"""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath


INERT = "x"

DEFAULT_PARTNER = "xxx[x<}xx,xxx}xx"
DEFAULT_FOCAL = "x]xxxxxxxxxxxxxx"

SYMBOL_COLORS = {
    "[": "#0f766e",
    "]": "#0f766e",
    "<": "#ea580c",
    ">": "#ea580c",
    "{": "#06b6d4",
    "}": "#06b6d4",
    ".": "#059669",
    ",": "#0f766e",
    "+": "#7c3aed",
    "-": "#7c3aed",
    "0": "#4b5563",
}

FIGURE_SIZE = (18.5, 12.2)
FIGURE_DPI = 220
PHASE_TAPE_ROW_HEIGHT = 1.0
PHASE_TITLE_ONLY_ROW_HEIGHT = 0.62
PHASE_TEXT_BASE_HEIGHT = 0.42
PHASE_TEXT_LINE_HEIGHT = 0.20
PHASE_TEXT_EXTRA_HEIGHT_BY_STEP = {
    0: 0.00,
    122: 0.08,
    215: 0.12,
    307: 0.00,
}
STABILIZATION_COMMENT_BASE_HEIGHT = 0.24
STABILIZATION_COMMENT_LINE_HEIGHT = 0.17

CELL_W = 1.0
CELL_H = 1.0
HALF_GAP = 0.25

CELL_FACE_COLOR = "#fafafa"
CELL_EDGE_COLOR = "#374151"
CELL_EDGE_WIDTH = 2.0

PANEL_RULE_LINEWIDTH = 1.6
PANEL_RULE_COLOR = "#c0c4cc"
PANEL_RULE_X_PAD = 0.25
PANEL_RULE_Y = CELL_H + 2.36

LOOP_BRACE_COLOR = "#6b7280"
LOOP_BRACE_LINEWIDTH = 4.4
LOOP_BRACE_ALPHA = 1.0
LOOP_BRACE_Y = -0.05
LOOP_BRACE_HEIGHT = -0.10
LOOP_BRACE_X_PAD = 0.08
LOOP_BOUNDARY_COLOR = "#6b7280"
LOOP_BOUNDARY_LINEWIDTH = 3.6

COPY_ARROW_START_COLOR = SYMBOL_COLORS["}"]
COPY_ARROW_END_COLOR = SYMBOL_COLORS["<"]
COPY_ARROW_LINEWIDTH = 1.5
COPY_ARROW_ALPHA = 0.35
COPY_ARROW_SEGMENTS = 42
COPY_ARROW_OUTLINE_COLOR = (1.0, 1.0, 1.0, 0.9)
COPY_ARROW_OUTLINE_WIDTH = 0.9
INTERPHASE_ARROW_SOURCE_Y = 0.0
INTERPHASE_ARROW_DEST_Y = CELL_H

TAPE_SYMBOL_FONTSIZE = 24
TAPE_SYMBOL_FONTFAMILY = "DejaVu Sans Mono"
TAPE_SYMBOL_FONTWEIGHT = "bold"
TEXT_PRIMARY_COLOR = "#111827"
TEXT_MUTED_COLOR = "#6b7280"

HEAD_MARKER = "v"
HEAD_MARKER_SIZE = 400
HEAD_MARKER_EDGE_COLOR = "white"
HEAD_MARKER_EDGE_WIDTH = 1.0
HEAD_MARKER_Y = CELL_H + 0.28
HEAD_LABEL_DY = 0.24
HEAD_LABEL_FONTSIZE = 11.0
HEAD_LABEL_FONTWEIGHT = "bold"
HEAD_SHIFT_OVERLAP_LEFT = -0.3
HEAD_SHIFT_OVERLAP_RIGHT = 0.3
HEAD_SHIFT_DEFAULT_LEFT = -0.10
HEAD_SHIFT_DEFAULT_RIGHT = 0.10

IP_MARKER = "^"
IP_MARKER_SIZE = HEAD_MARKER_SIZE
IP_MARKER_COLOR = "#6b7280"
IP_MARKER_EDGE_COLOR = HEAD_MARKER_EDGE_COLOR
IP_MARKER_EDGE_WIDTH = 1.6
IP_MARKER_Y = -0.18
IP_LABEL_DY = 0.30
IP_LABEL_FONTSIZE = HEAD_LABEL_FONTSIZE
IP_LABEL_FONTWEIGHT = HEAD_LABEL_FONTWEIGHT

PANEL_LABEL_FONTSIZE = 15.8
PHASE_TITLE_FONTSIZE = 16.8
PHASE_TIME_FONTSIZE = 11.0
PHASE_TITLE_TIME_SEP = 5
PHASE_COMMENT_FONTSIZE = 15.5
PHASE_COMMENT_WRAP_DEFAULT = 105
PHASE_COMMENT_WRAP_BY_STEP = {
    0: 145,
    122: 75,
    215: 95,
    307: 145,
}
PHASE_COMMENT_LINE_SPACING = 1.10
PHASE_BLOCK_LEFT_PAD = 0.18
PHASE_TEXT_RULE_Y_AXES = 0.96
PHASE_TEXT_LEFT_AXES = 0.0
PHASE_TITLE_TOP_AXES = 0.78
PHASE_COMMENT_TOP_AXES = 0.56
PHASE_COMMENT_TOP_AXES_ADJUST = {
    0: 0.01,
    122: 0.00,
    215: 0.01,
    307: 0.00,
}
STABILIZATION_COMMENT_TOP_AXES = 0.92
PANEL_LABEL_TAPE_PADDING = 0.10
PHASE_TITLE_FONTWEIGHT = "bold"
PHASE_TIME_FONTWEIGHT = "normal"
PANEL_LABEL_TOP_PAD = 0.06
HEAD_TOP_PAD = 0.16

X_LIM_PAD = 0.0
Y_LIM_LOW = -0.52

SUBPLOTS_ADJUST = {
    "left": 0.0,
    "right": 0.997,
    "top": 0.99,
    "bottom": 0.0,
    "wspace": 0.0,
    "hspace": 0.08,
}

HEAD_COLORS = {"H0": SYMBOL_COLORS["<"], "H1": SYMBOL_COLORS["}"]}

REVERSE_CONSTRUCTION_PAIRS = [
    (1, 31),
    (3, 30),
    (5, 29),
    (7, 28),
    (9, 27),
    (11, 26),
    (13, 25),
    (15, 24),
    (17, 23),
]

FORWARD_RECONSTRUCTION_PAIRS = [
    (19, 22),
    (21, 21),
    (23, 20),
    (25, 19),
    (27, 18),
    (29, 17),
    (30, 16),
]

STABILIZATION_PAIRS = [
    (31, 15),
    (0, 14),
    (1, 13),
    (2, 12),
    (3, 11),
    (4, 10),
    (5, 9),
    (6, 8),
    (7, 7),
    (8, 6),
    (9, 5),
    (10, 4),
    (11, 3),
    (12, 2),
    (13, 1),
    (14, 0),
    (15, 31),
    (16, 30),
    (17, 29),
    (18, 28),
    (19, 27),
    (20, 26),
    (21, 25),
]


@dataclass
class Snapshot:
    step: int
    label: str
    comment: str
    partner: str
    focal: str
    pc: int
    h0: int
    h1: int
    loop_spans: list[tuple[str, int, int]]
    copy_pairs: list[tuple[int, int]]
    arrow_mode: str


def evaluate_modulo(partner: str, focal: str, step_cap: int = 8192) -> list[dict]:
    mem = list(partner + focal)
    tape_len = len(mem)
    half = len(partner)
    pc = 0
    h0 = tape_len
    h1 = tape_len
    history: list[dict] = []

    for step in range(step_cap):
        h0 %= tape_len
        h1 %= tape_len
        history.append(
            {
                "step": step,
                "pc": pc,
                "h0": h0,
                "h1": h1,
                "partner": "".join(mem[:half]),
                "focal": "".join(mem[half:]),
            }
        )

        kind = mem[pc]
        if kind == "<":
            h0 -= 1
        elif kind == ">":
            h0 += 1
        elif kind == "{":
            h1 -= 1
        elif kind == "}":
            h1 += 1
        elif kind == "+":
            mem[h0] = chr((ord(mem[h0]) + 1) % 256)
        elif kind == "-":
            mem[h0] = chr((ord(mem[h0]) - 1) % 256)
        elif kind == ".":
            mem[h1] = mem[h0]
        elif kind == ",":
            mem[h0] = mem[h1]
        elif kind == "[":
            if mem[h0] == "\x00":
                depth = 1
                pc += 1
                while pc < tape_len and depth > 0:
                    if mem[pc] == "]":
                        depth -= 1
                    if mem[pc] == "[":
                        depth += 1
                    pc += 1
                pc -= 1
                if depth != 0:
                    break
        elif kind == "]":
            if mem[h0] != "\x00":
                depth = 1
                pc -= 1
                while pc >= 0 and depth > 0:
                    if mem[pc] == "]":
                        depth += 1
                    if mem[pc] == "[":
                        depth -= 1
                    pc -= 1
                pc += 1
                if depth != 0:
                    break

        pc += 1
        if pc < 0 or pc >= tape_len:
            break

    return history


def find_history_step(history: list[dict], step: int) -> dict:
    for item in history:
        if item["step"] == step:
            return item
    raise ValueError(f"step {step} not found")


def canonical_snapshots(history: list[dict]) -> list[Snapshot]:
    selected = [
        Snapshot(
            step=0,
            label="1. Loop completion",
            comment=(
                ""
                # "The interaction begins with the focal "
                # "tape supplying the closing bracket ], thereby completing "
                # "a loop structure that spans both tapes. This interaction-completed "
                # "loop forms a pre-replicator that was not present on either "
                # "tape in isolation."
            ),
            partner="",
            focal="",
            pc=0,
            h0=0,
            h1=0,
            loop_spans=[],
            copy_pairs=[],
            arrow_mode="none",
        ),
        Snapshot(
            step=122,
            label="2. Reverse construction",
            comment=(
                ""
                # "Execution of the proto-replicator "
                # "loop induces asymmetric head transport and inter-head copying. Over "
                # "repeated traversals, this process selectively propagates symbols from "
                # "the partner tape into the focal tape, constructing a reversed copy of "
                # "the loop body in the tail region of the focal tape. The copying is "
                # "sparse and structured, effectively filtering the source sequence."
            ),
            partner="",
            focal="",
            pc=0,
            h0=0,
            h1=0,
            loop_spans=[("P", 3, 15), ("F", 0, 1)],
            copy_pairs=REVERSE_CONSTRUCTION_PAIRS,
            arrow_mode="upper",
        ),
        Snapshot(
            step=215,
            label="3. Forward reconstruction",
            comment=(
                ""
                # "As execution continues, the heads "
                # "traverse into the newly constructed region. The same "
                # "transport-and-copy mechanism is now applied to the reversed "
                # "structure in the focal tape, copying it back toward the beginning of "
                # "the tape and restoring the forward orientation. During this process, "
                # "local modifications (e.g., insertion of control symbols such as <) "
                # "disrupt the original proto-loop, but execution "
                # "transitions into the newly formed loop structure, completing it "
                # "dynamically."
            ),
            partner="",
            focal="",
            pc=0,
            h0=0,
            h1=0,
            loop_spans=[("F", 0, 4)],
            copy_pairs=FORWARD_RECONSTRUCTION_PAIRS,
            arrow_mode="lower",
        ),
        Snapshot(
            step=307,
            label="4. Stabilization",
            comment=(
                ""
                # "Once both halves of the reverse-copy "
                # "scaffold are present, further loop execution enforces a symmetric, "
                # "self-consistent configuration. The tape becomes effectively "
                # "palindromic with respect to the active heads, and subsequent copying "
                # "operations preserve this structure. No further large-scale rewrites "
                # "occur; the system remains in a stable execution cycle until the step "
                # "limit is reached. The focal tape now contains a functional "
                # "reverse-copy replicator."
            ),
            partner="",
            focal="",
            pc=0,
            h0=0,
            h1=0,
            loop_spans=[("F", 0, 4)],
            copy_pairs=STABILIZATION_PAIRS,
            arrow_mode="split",
        ),
    ]

    out: list[Snapshot] = []
    for snapshot in selected:
        state = find_history_step(history, snapshot.step)
        out.append(
            Snapshot(
                step=snapshot.step,
                label=snapshot.label,
                comment=snapshot.comment,
                partner=state["partner"],
                focal=state["focal"],
                pc=state["pc"],
                h0=state["h0"],
                h1=state["h1"],
                loop_spans=snapshot.loop_spans,
                copy_pairs=snapshot.copy_pairs,
                arrow_mode=snapshot.arrow_mode,
            )
        )
    return out


def split_combined_index(index: int, half_len: int) -> tuple[str, int]:
    if index < half_len:
        return "P", index
    return "F", index - half_len


def cell_x(which_half: str, index: int, half_len: int, gap: float) -> float:
    base = 0.0 if which_half == "P" else half_len + gap
    return base + index * CELL_W


def combined_cell_left(index: int, half_len: int, gap: float) -> float:
    which_half, half_idx = split_combined_index(index, half_len)
    return cell_x(which_half, half_idx, half_len, gap)


def combined_cell_center(index: int, half_len: int, gap: float) -> tuple[float, float]:
    x = combined_cell_left(index, half_len, gap) + 0.5
    return x, CELL_H / 2.0


def merged_loop_ranges(
    loop_spans: list[tuple[str, int, int]],
    half_len: int,
) -> list[tuple[int, int]]:
    full_ranges: list[tuple[int, int]] = []
    for half, start, end in loop_spans:
        if half == "P":
            full_ranges.append((start, end))
        else:
            full_ranges.append((half_len + start, half_len + end))
    full_ranges.sort()

    merged: list[tuple[int, int]] = []
    for start, end in full_ranges:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
    return merged


def interpolate_rgba(color_a: str, color_b: str, t: float, alpha: float) -> tuple[float, float, float, float]:
    rgba_a = to_rgba(color_a, alpha)
    rgba_b = to_rgba(color_b, alpha)
    return tuple((1.0 - t) * a + t * b for a, b in zip(rgba_a, rgba_b))


def quadratic_bezier_points(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    steps: int,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for idx in range(steps + 1):
        t = idx / steps
        mt = 1.0 - t
        x = mt * mt * p0[0] + 2.0 * mt * t * p1[0] + t * t * p2[0]
        y = mt * mt * p0[1] + 2.0 * mt * t * p1[1] + t * t * p2[1]
        points.append((x, y))
    return points


def draw_gradient_arrow(
    ax,
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> None:
    points = quadratic_bezier_points(p0, p1, p2, COPY_ARROW_SEGMENTS)
    segments = list(zip(points[:-1], points[1:]))
    colors = [
        interpolate_rgba(
            COPY_ARROW_START_COLOR,
            COPY_ARROW_END_COLOR,
            idx / max(1, len(segments) - 1),
            COPY_ARROW_ALPHA,
        )
        for idx in range(len(segments))
    ]
    collection = LineCollection(
        segments,
        colors=colors,
        linewidths=COPY_ARROW_LINEWIDTH,
        capstyle="butt",
        zorder=1.1,
    )
    collection.set_path_effects(
        [
            pe.Stroke(
                linewidth=COPY_ARROW_LINEWIDTH + COPY_ARROW_OUTLINE_WIDTH,
                foreground=COPY_ARROW_OUTLINE_COLOR,
                capstyle="butt",
            ),
            pe.Normal(),
        ]
    )
    ax.add_collection(collection)


def data_to_figure_coords(
    fig: plt.Figure,
    ax,
    point: tuple[float, float],
) -> tuple[float, float]:
    display = ax.transData.transform(point)
    return tuple(fig.transFigure.inverted().transform(display))


def draw_interphase_copy_arrows(
    fig: plt.Figure,
    overlay_ax,
    axes: list,
    snapshots: list[Snapshot],
    half_len: int,
    gap: float,
) -> None:
    for panel_idx in range(1, len(snapshots)):
        prev_ax = axes[panel_idx - 1]
        curr_ax = axes[panel_idx]
        for source_idx, dest_idx in snapshots[panel_idx].copy_pairs:
            src_x, _ = combined_cell_center(source_idx, half_len, gap)
            dst_x, _ = combined_cell_center(dest_idx, half_len, gap)
            start = data_to_figure_coords(
                fig,
                prev_ax,
                (src_x, INTERPHASE_ARROW_SOURCE_Y),
            )
            end = data_to_figure_coords(
                fig,
                curr_ax,
                (dst_x, INTERPHASE_ARROW_DEST_Y),
            )
            control = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
            draw_gradient_arrow(overlay_ax, start, control, end)


def draw_cells(
    ax,
    which_half: str,
    tape: str,
    half_len: int,
    gap: float,
) -> None:
    x0 = cell_x(which_half, 0, half_len, gap)
    for idx, ch in enumerate(tape):
        x = cell_x(which_half, idx, half_len, gap)
        ax.add_patch(
            Rectangle(
                (x, 0.0),
                CELL_W,
                CELL_H,
                facecolor=CELL_FACE_COLOR,
                edgecolor="none",
                linewidth=0.0,
                zorder=1,
            )
        )
        if ch != INERT:
            ax.text(
                x + 0.5,
                CELL_H / 2.0,
                ch,
                ha="center",
                va="center",
                fontsize=TAPE_SYMBOL_FONTSIZE,
                family=TAPE_SYMBOL_FONTFAMILY,
                fontweight=TAPE_SYMBOL_FONTWEIGHT,
                color=SYMBOL_COLORS.get(ch, TEXT_PRIMARY_COLOR),
                zorder=2.2,
            )
    x1 = x0 + len(tape) * CELL_W
    ax.plot(
        [x0, x1],
        [0.0, 0.0],
        color=CELL_EDGE_COLOR,
        linewidth=CELL_EDGE_WIDTH,
        solid_capstyle="butt",
        zorder=1.4,
    )
    ax.plot(
        [x0, x1],
        [CELL_H, CELL_H],
        color=CELL_EDGE_COLOR,
        linewidth=CELL_EDGE_WIDTH,
        solid_capstyle="butt",
        zorder=1.4,
    )
    for idx in range(len(tape) + 1):
        xv = x0 + idx * CELL_W
        ax.plot(
            [xv, xv],
            [0.0, CELL_H],
            color=CELL_EDGE_COLOR,
            linewidth=CELL_EDGE_WIDTH,
            solid_capstyle="butt",
            zorder=1.4,
        )


def draw_horizontal_brace(
    ax,
    x0: float,
    x1: float,
    y: float,
    height: float,
) -> None:
    width = x1 - x0
    if width <= 0:
        return
    mid = (x0 + x1) / 2.0
    vertices = [
        (x0, y),
        (x0 + 0.14 * width, y),
        (x0 + 0.22 * width, y + height),
        (mid, y),
        (x1 - 0.22 * width, y + height),
        (x1 - 0.14 * width, y),
        (x1, y),
    ]
    codes = [MplPath.MOVETO] + [MplPath.CURVE4] * 6
    path = MplPath(vertices, codes)
    brace = PathPatch(
        path,
        facecolor="none",
        edgecolor=LOOP_BRACE_COLOR,
        linewidth=LOOP_BRACE_LINEWIDTH,
        alpha=LOOP_BRACE_ALPHA,
        capstyle="round",
        joinstyle="round",
        zorder=3.6,
    )
    ax.add_patch(brace)


def draw_loop_braces(
    ax,
    loop_spans: list[tuple[str, int, int]],
    half_len: int,
    gap: float,
) -> None:
    for start_idx, end_idx in merged_loop_ranges(loop_spans, half_len):
        x0 = combined_cell_left(start_idx, half_len, gap) + LOOP_BRACE_X_PAD
        x1 = combined_cell_left(end_idx, half_len, gap) + CELL_W - LOOP_BRACE_X_PAD
        draw_horizontal_brace(ax, x0, x1, LOOP_BRACE_Y, LOOP_BRACE_HEIGHT)


def draw_loop_boundary_cells(
    ax,
    loop_spans: list[tuple[str, int, int]],
    half_len: int,
    gap: float,
) -> None:
    for start_idx, end_idx in merged_loop_ranges(loop_spans, half_len):
        for idx in range(start_idx, end_idx + 1):
            x_top = combined_cell_left(idx, half_len, gap)
            ax.plot(
                [x_top, x_top + CELL_W],
                [CELL_H, CELL_H],
                color=LOOP_BOUNDARY_COLOR,
                linewidth=LOOP_BOUNDARY_LINEWIDTH,
                solid_capstyle="butt",
                zorder=3.5,
            )
        x_left = combined_cell_left(start_idx, half_len, gap)
        ax.plot(
            [x_left, x_left],
            [0.0, CELL_H],
            color=LOOP_BOUNDARY_COLOR,
            linewidth=LOOP_BOUNDARY_LINEWIDTH,
            solid_capstyle="butt",
            zorder=3.5,
        )
        x_right = combined_cell_left(end_idx, half_len, gap) + CELL_W
        ax.plot(
            [x_right, x_right],
            [0.0, CELL_H],
            color=LOOP_BOUNDARY_COLOR,
            linewidth=LOOP_BOUNDARY_LINEWIDTH,
            solid_capstyle="butt",
            zorder=3.5,
        )


def draw_head_marker(
    ax,
    head_idx: int,
    label: str,
    color: str,
    half_len: int,
    gap: float,
    x_shift: float,
) -> None:
    which_half, idx = split_combined_index(head_idx, half_len)
    x = cell_x(which_half, idx, half_len, gap) + 0.5 + x_shift
    y = HEAD_MARKER_Y
    ax.scatter(
        [x],
        [y],
        marker=HEAD_MARKER,
        s=HEAD_MARKER_SIZE,
        color=color,
        edgecolors=HEAD_MARKER_EDGE_COLOR,
        linewidths=HEAD_MARKER_EDGE_WIDTH,
        zorder=5,
    )
    ax.text(
        x,
        y + HEAD_LABEL_DY,
        label,
        ha="center",
        va="bottom",
        fontsize=HEAD_LABEL_FONTSIZE,
        color=color,
        fontweight=HEAD_LABEL_FONTWEIGHT,
        zorder=6,
    )


def draw_ip_marker(
    ax,
    pc_idx: int,
    half_len: int,
    gap: float,
) -> None:
    which_half, idx = split_combined_index(pc_idx, half_len)
    x = cell_x(which_half, idx, half_len, gap) + 0.5
    y = IP_MARKER_Y
    ax.scatter(
        [x],
        [y],
        marker=IP_MARKER,
        s=IP_MARKER_SIZE,
        color=IP_MARKER_COLOR,
        edgecolors=IP_MARKER_EDGE_COLOR,
        linewidths=IP_MARKER_EDGE_WIDTH,
        zorder=5,
    )
    ax.text(
        x,
        y - IP_LABEL_DY,
        "IP",
        ha="center",
        va="top",
        fontsize=IP_LABEL_FONTSIZE,
        color=IP_MARKER_COLOR,
        fontweight=IP_LABEL_FONTWEIGHT,
        zorder=6,
    )


def wrapped_comment(text: str, step: int) -> str:
    width = PHASE_COMMENT_WRAP_BY_STEP.get(step, PHASE_COMMENT_WRAP_DEFAULT)
    return textwrap.fill(text, width=width, break_long_words=False)


def wrapped_line_count(text: str, step: int) -> int:
    wrapped = wrapped_comment(text, step)
    return wrapped.count("\n") + 1 if wrapped else 0


def phase_text_row_height(snapshot: Snapshot, include_comment: bool) -> float:
    if not include_comment:
        return PHASE_TITLE_ONLY_ROW_HEIGHT + PHASE_TEXT_EXTRA_HEIGHT_BY_STEP.get(snapshot.step, 0.0)
    lines = wrapped_line_count(snapshot.comment, snapshot.step)
    return PHASE_TEXT_BASE_HEIGHT + lines * PHASE_TEXT_LINE_HEIGHT + PHASE_TEXT_EXTRA_HEIGHT_BY_STEP.get(snapshot.step, 0.0)


def stabilization_comment_row_height(snapshot: Snapshot) -> float:
    lines = wrapped_line_count(snapshot.comment, snapshot.step)
    return STABILIZATION_COMMENT_BASE_HEIGHT + lines * STABILIZATION_COMMENT_LINE_HEIGHT


def render_figure(snapshots: list[Snapshot], output_base: Path) -> list[Path]:
    half_len = len(snapshots[0].partner)
    gap = HALF_GAP
    nrows = len(snapshots)
    height_ratios: list[float] = []
    for idx, snapshot in enumerate(snapshots):
        include_comment = idx != nrows - 1
        height_ratios.append(phase_text_row_height(snapshot, include_comment))
        height_ratios.append(PHASE_TAPE_ROW_HEIGHT)
    height_ratios.append(stabilization_comment_row_height(snapshots[-1]))

    fig = plt.figure(figsize=FIGURE_SIZE)
    gs = fig.add_gridspec(nrows=2 * nrows + 1, ncols=1, height_ratios=height_ratios)
    text_axes = [fig.add_subplot(gs[2 * idx, 0]) for idx in range(nrows)]
    tape_axes = [fig.add_subplot(gs[2 * idx + 1, 0]) for idx in range(nrows)]
    stabilization_comment_ax = fig.add_subplot(gs[-1, 0])
    last_comment_text = wrapped_comment(snapshots[-1].comment, snapshots[-1].step)

    y_lim_high = max(
        HEAD_MARKER_Y + HEAD_LABEL_DY + HEAD_TOP_PAD,
        CELL_H + PANEL_LABEL_TAPE_PADDING + PANEL_LABEL_TOP_PAD,
    )

    for panel_idx, (text_ax, ax, snapshot) in enumerate(zip(text_axes, tape_axes, snapshots)):
        text_ax.set_zorder(2.0)
        text_ax.patch.set_alpha(0.0)
        text_ax.set_xlim(0.0, 1.0)
        text_ax.set_ylim(0.0, 1.0)
        text_ax.axis("off")
        text_ax.plot(
            [0.0, 0.998],
            [PHASE_TEXT_RULE_Y_AXES, PHASE_TEXT_RULE_Y_AXES],
            color=PANEL_RULE_COLOR,
            linewidth=PANEL_RULE_LINEWIDTH,
            solid_capstyle="round",
            transform=text_ax.transAxes,
            zorder=5.5,
        )

        ax.set_zorder(2.0)
        ax.patch.set_alpha(0.0)
        draw_cells(ax, "P", snapshot.partner, half_len, gap)
        draw_cells(ax, "F", snapshot.focal, half_len, gap)
        draw_loop_boundary_cells(ax, snapshot.loop_spans, half_len, gap)
        draw_loop_braces(ax, snapshot.loop_spans, half_len, gap)

        same_combined_cell = snapshot.h0 == snapshot.h1
        h0_shift = HEAD_SHIFT_OVERLAP_LEFT if same_combined_cell else HEAD_SHIFT_DEFAULT_LEFT
        h1_shift = HEAD_SHIFT_OVERLAP_RIGHT if same_combined_cell else HEAD_SHIFT_DEFAULT_RIGHT
        h1_display_idx = snapshot.h1 + (1 if snapshot.step == 215 else 0)
        draw_head_marker(ax, snapshot.h0, "H0", HEAD_COLORS["H0"], half_len, gap, x_shift=h0_shift)
        draw_head_marker(ax, h1_display_idx, "H1", HEAD_COLORS["H1"], half_len, gap, x_shift=h1_shift)
        draw_ip_marker(ax, snapshot.pc, half_len, gap)

        left_mid = half_len / 2.0
        right_mid = half_len + gap + half_len / 2.0
        if panel_idx == 0:
            ax.text(
                left_mid,
                CELL_H + PANEL_LABEL_TAPE_PADDING,
                "partner",
                ha="center",
                va="bottom",
                fontsize=PANEL_LABEL_FONTSIZE,
                color=TEXT_MUTED_COLOR,
            )
            ax.text(
                right_mid,
                CELL_H + PANEL_LABEL_TAPE_PADDING,
                "focal",
                ha="center",
                va="bottom",
                fontsize=PANEL_LABEL_FONTSIZE,
                color=TEXT_MUTED_COLOR,
            )

        title_area = TextArea(
            snapshot.label,
            textprops={
                "fontsize": PHASE_TITLE_FONTSIZE,
                "fontweight": PHASE_TITLE_FONTWEIGHT,
                "color": TEXT_PRIMARY_COLOR,
            },
        )
        time_area = TextArea(
            f"(t={snapshot.step})",
            textprops={
                "fontsize": PHASE_TIME_FONTSIZE,
                "fontweight": PHASE_TIME_FONTWEIGHT,
                "color": TEXT_MUTED_COLOR,
            },
        )
        title_pack = HPacker(children=[title_area, time_area], align="baseline", pad=0, sep=PHASE_TITLE_TIME_SEP)
        text_ax.add_artist(
            AnnotationBbox(
                title_pack,
                (PHASE_TEXT_LEFT_AXES, PHASE_TITLE_TOP_AXES),
                xycoords=text_ax.transAxes,
                frameon=False,
                box_alignment=(0.0, 1.0),
                zorder=6,
            )
        )
        if panel_idx != nrows - 1:
            comment_area = TextArea(
                wrapped_comment(snapshot.comment, snapshot.step),
                textprops={
                    "fontsize": PHASE_COMMENT_FONTSIZE,
                    "color": TEXT_PRIMARY_COLOR,
                    "linespacing": PHASE_COMMENT_LINE_SPACING,
                    "multialignment": "left",
                },
            )
            text_ax.add_artist(
                AnnotationBbox(
                    comment_area,
                    (
                        PHASE_TEXT_LEFT_AXES,
                        PHASE_COMMENT_TOP_AXES + PHASE_COMMENT_TOP_AXES_ADJUST.get(snapshot.step, 0.0),
                    ),
                    xycoords=text_ax.transAxes,
                    frameon=False,
                    box_alignment=(0.0, 1.0),
                    zorder=6,
                )
            )

        ax.set_xlim(-X_LIM_PAD, 2 * half_len + gap + X_LIM_PAD)
        ax.set_ylim(Y_LIM_LOW, y_lim_high)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.subplots_adjust(**SUBPLOTS_ADJUST)
    fig.canvas.draw()

    overlay_ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], zorder=0.1)
    overlay_ax.set_xlim(0.0, 1.0)
    overlay_ax.set_ylim(0.0, 1.0)
    overlay_ax.axis("off")
    overlay_ax.patch.set_alpha(0.0)
    draw_interphase_copy_arrows(fig, overlay_ax, tape_axes, snapshots, half_len, gap)

    stabilization_comment_ax.set_zorder(2.0)
    stabilization_comment_ax.patch.set_alpha(0.0)
    stabilization_comment_ax.set_xlim(0.0, 1.0)
    stabilization_comment_ax.set_ylim(0.0, 1.0)
    stabilization_comment_ax.axis("off")
    stabilization_comment_area = TextArea(
        last_comment_text,
        textprops={
            "fontsize": PHASE_COMMENT_FONTSIZE,
            "color": TEXT_PRIMARY_COLOR,
            "linespacing": PHASE_COMMENT_LINE_SPACING,
            "multialignment": "left",
        },
    )
    stabilization_comment_ax.add_artist(
        AnnotationBbox(
            stabilization_comment_area,
            (PHASE_TEXT_LEFT_AXES, STABILIZATION_COMMENT_TOP_AXES),
            xycoords=stabilization_comment_ax.transAxes,
            frameon=False,
            box_alignment=(0.0, 1.0),
            zorder=6,
        )
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in (".pdf", ".png"):
        path = output_base.with_suffix(suffix)
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0)
        outputs.append(path)
    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="analysis_plots/no_mutation/toy_canonical_16_progression",
        help="Output path without suffix; both .pdf and .png are written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    history = evaluate_modulo(DEFAULT_PARTNER, DEFAULT_FOCAL)
    snapshots = canonical_snapshots(history)
    outputs = render_figure(snapshots, Path(args.output))
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
