"""Small figures for the final evaluation: distributions, relations, overlays.

The histograms and the relation chart follow ``plot_distribution`` and
``plot_m2n_counts`` of the organizers' self-evaluation notebook (Kaggle:
https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook): 50 bins
over [0, 1] of the scores above 0, bar heights normalized to probabilities
with a dashed mean line, and a split bar chart with ground-truth degrees on
the right and prediction degrees on the left. The code here is our own.

Figures are built on :class:`matplotlib.figure.Figure` directly, so no pyplot
state or display is needed. The frames are GONG data; every figure carries the
attribution :data:`ATTRIBUTION`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from filament.final_eval import Example
from filament.final_metrics import RelationCounts

ATTRIBUTION = "GONG data, NSO/AURA/NSF"
DPI = 100
GT_COLOR = "#009E73"
PRED_COLOR = "#D55E00"
HIST_COLOR = "navy"
MEAN_COLOR = "crimson"
# Colors of plot_m2n_counts.
COLOR_GT_MISS = "crimson"
COLOR_GT_HIT = "forestgreen"
COLOR_PRED_MISS = "darkorange"
COLOR_PRED_HIT = "steelblue"


def _footer(fig: Figure) -> None:
    fig.text(
        0.99, 0.005, ATTRIBUTION, ha="right", va="bottom", fontsize=6, color="0.35"
    )


def distribution_figure(scores: np.ndarray, title: str, x_label: str) -> Figure:
    """Histogram of the scores above 0 with their mean (``plot_distribution``)."""
    fig = Figure(figsize=(4.0, 3.0), dpi=DPI, layout="constrained")
    ax = fig.subplots()
    values = np.asarray(scores, dtype=np.float64)
    values = values[values > 0]
    counts, edges = np.histogram(values, bins=np.linspace(0.0, 1.0, 51))
    probability = counts / counts.sum() if counts.sum() else counts
    centers = (edges[:-1] + edges[1:]) / 2.0
    ax.bar(
        centers,
        probability,
        width=0.02,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.7,
        color=HIST_COLOR,
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    if values.size:
        mean = float(values.mean())
        ax.axvline(
            mean,
            color=MEAN_COLOR,
            linestyle="--",
            linewidth=1.2,
            label=f"Mean = {mean:.3f} (n = {values.size})",
        )
        ax.legend(fontsize=7)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel("Probability", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    _footer(fig)
    return fig


def relations_figure(relations: RelationCounts, title: str) -> Figure:
    """Split bar chart of match degrees (``plot_m2n_counts``).

    Ground-truth masks on the right (``1:n``), predictions on the left (``n:1``).
    Degree 0 is a miss.
    """
    fig = Figure(figsize=(5.0, 3.2), dpi=DPI, layout="constrained")
    ax = fig.subplots()
    offset = 0.5
    ticks: list[float] = []
    labels: list[str] = []
    for degree, count in relations.gt_degrees.items():
        position = degree + offset
        ax.bar(
            position,
            count,
            width=0.8,
            color=COLOR_GT_MISS if degree == 0 else COLOR_GT_HIT,
            edgecolor="black",
            linewidth=0.3,
            alpha=0.7,
        )
        ticks.append(position)
        labels.append(f"1:{degree}")
    for degree, count in relations.pred_degrees.items():
        position = -(degree + offset)
        ax.bar(
            position,
            count,
            width=0.8,
            color=COLOR_PRED_MISS if degree == 0 else COLOR_PRED_HIT,
            edgecolor="black",
            linewidth=0.3,
            alpha=0.7,
        )
        ticks.append(position)
        labels.append(f"{degree}:1")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.margins(y=0.4)  # headroom for the legend
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    ax.set_xticks(ticks, labels, rotation=45, fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("Matching case (any overlap)", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(
        handles=[
            Patch(
                facecolor=COLOR_GT_HIT,
                alpha=0.7,
                label=f"GT 1:1 {relations.gt_one_to_one}, 1:n (n>=2) {relations.one_to_many}",
            ),
            Patch(
                facecolor=COLOR_GT_MISS,
                alpha=0.7,
                label=f"GT misses (1:0): {relations.gt_miss}",
            ),
            Patch(
                facecolor=COLOR_PRED_HIT,
                alpha=0.7,
                label=(
                    f"Pred 1:1 {relations.pred_one_to_one}, "
                    f"n:1 (n>=2) {relations.many_to_one}"
                ),
            ),
            Patch(
                facecolor=COLOR_PRED_MISS,
                alpha=0.7,
                label=f"Pred misses (0:1): {relations.pred_miss}",
            ),
        ],
        fontsize=6,
        loc="upper center",
        ncol=2,
    )
    _footer(fig)
    return fig


def _focus_box(
    example: Example,
    labels: np.ndarray,
    gt_masks: Sequence[np.ndarray],
    margin: int,
    min_side: int,
) -> tuple[int, int, int, int]:
    """Crop ``(y0, y1, x0, x1)`` around the example's objects, at least ``min_side`` wide."""
    focus = np.zeros(labels.shape, dtype=bool)
    for index in example.gt_indices:
        focus |= np.asarray(gt_masks[index]) != 0
    for ident in example.pred_ids:
        focus |= labels == ident
    ys, xs = np.nonzero(focus)
    height, width = labels.shape
    if ys.size == 0:
        return 0, height, 0, width
    box = []
    for low, high, size in (
        (ys.min(), ys.max() + 1, height),
        (xs.min(), xs.max() + 1, width),
    ):
        low, high = int(low) - margin, int(high) + margin
        extra = min_side - (high - low)
        if extra > 0:
            low, high = low - extra // 2, high + extra - extra // 2
        low, high = max(low, 0), min(high, size)
        box.append((low, high))
    (y0, y1), (x0, x1) = box
    return y0, y1, x0, x1


def _outline(ax: Axes, mask: np.ndarray, color: str, width: float) -> None:
    if not mask.any():
        return
    padded = np.pad(mask.astype(np.float32), 1)
    ax.contour(
        np.arange(-1, mask.shape[1] + 1),
        np.arange(-1, mask.shape[0] + 1),
        padded,
        levels=[0.5],
        colors=color,
        linewidths=width,
    )


def draw_example(
    ax: Axes,
    example: Example,
    gray: np.ndarray,
    labels: np.ndarray,
    gt_masks: Sequence[np.ndarray],
    *,
    margin: int = 48,
    min_side: int = 160,
) -> None:
    """Outlines of the view's masks (green) and the prediction (orange) on a crop.

    The objects the example is about are drawn thicker; others in the crop thinner.
    """
    y0, y1, x0, x1 = _focus_box(example, labels, gt_masks, margin, min_side)
    ax.imshow(gray[y0:y1, x0:x1], cmap="gray", interpolation="nearest")
    crop_labels = labels[y0:y1, x0:x1]
    for index, mask in enumerate(gt_masks):
        crop = np.asarray(mask)[y0:y1, x0:x1] != 0
        _outline(ax, crop, GT_COLOR, 1.6 if index in example.gt_indices else 0.6)
    for ident in np.unique(crop_labels):
        if ident == 0:
            continue
        _outline(
            ax,
            crop_labels == ident,
            PRED_COLOR,
            1.6 if int(ident) in example.pred_ids else 0.6,
        )
    ax.set_xlim(-0.5, x1 - x0 - 0.5)
    ax.set_ylim(y1 - y0 - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{example.file_name} v{example.view_index}\n{example.note}", fontsize=6
    )


def overlay_figure(
    kind: str,
    panels: Sequence[tuple[Example, np.ndarray, np.ndarray, Sequence[np.ndarray]]],
) -> Figure:
    """One row of overlay panels. Each panel is ``(example, gray, labels, gt_masks)``."""
    count = max(len(panels), 1)
    fig = Figure(figsize=(2.6 * count, 3.1), dpi=DPI, layout="constrained")
    axes = np.atleast_1d(fig.subplots(1, count))
    for ax in axes:
        ax.set_axis_off()
    for ax, (example, gray, labels, gt_masks) in zip(axes, panels, strict=False):
        ax.set_axis_on()
        draw_example(ax, example, gray, labels, gt_masks)
    fig.suptitle(kind.replace("_", " "), fontsize=9)
    fig.legend(
        handles=[
            Line2D([], [], color=GT_COLOR, linewidth=1.4, label="annotator"),
            Line2D([], [], color=PRED_COLOR, linewidth=1.4, label="prediction"),
        ],
        loc="lower left",
        fontsize=6,
        ncol=2,
    )
    _footer(fig)
    return fig


def save_figure(fig: Figure, path: Path) -> Path:
    """Write ``fig`` as a PNG at ``path`` (parent created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    return path
