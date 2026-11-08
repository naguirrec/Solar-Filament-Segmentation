"""Why the holdout scores above the public leaderboard: two testable hypotheses.

``docs/ROADMAP_V2.md`` section 3 gives three hypotheses for a holdout PQ of about
0.41 against a leaderboard of 0.34. This module measures two of them from cached
probability maps, on the holdout only (H3 is the parity test with the official
notebook):

* **H1, selection.** The settings were chosen on the same files they are scored
  on. A cross-fit chooses on some dates and scores on the others, over several
  partitions. ``sweep maximum minus cross-fit`` estimates what selection adds.
* **H2, annotator mix.** The pooled PQ weighs a file by its number of views, and
  where annotators disagree the reachable PQ is lower. The holdout is scored
  separately on files with one view and with several, and with the file mix
  tilted to the views per file reported for the test set.

These are estimates on 142 files. They say how much of the gap each hypothesis can
account for, and they do not prove a cause. The stratum of files with several views
also differs from the others in ways other than the number of views.
"""

from __future__ import annotations

import csv
import datetime as dt
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from filament.annotations import TrainImage
from filament.metrics import PqCounts
from filament.split import observation_date
from filament.stats import (
    CrossfitResult,
    bootstrap_pq,
    crossfit_select,
    date_folds,
    effective_files,
    tilt_to_mean_views,
)
from filament.sweep import GridPoint, PerFileCounts, format_pq, postproc_grid

# The setting behind the holdout 0.409 and LB 0.34 of docs/PROJECT_REPORT.md, row 12.
CURRENT_POINT = GridPoint(0.45, 11, 400)

# One source, not verified: kaggle.too.foo estimates this many views per test
# image [VERIFY]. It is an input to a what-if, not a measurement of the test set.
TEST_VIEWS_PER_FILE = 2.44

# Roughly the settings compared so far: the grids of sweep_postprocess_r34.ipynb
# and sweep_tta_ensemble.ipynb, as one grid.
GAP_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
GAP_CLOSE_SIZES = (0, 3, 7, 11, 15, 21, 25)
GAP_MIN_AREAS = (240, 320, 400, 500, 600, 700)

CROSSFIT_SEEDS = (0, 1, 2, 3, 4)
CROSSFIT_FOLDS = 2

SUMMARY_COLUMNS = ("quantity", "value", "low", "high", "files", "note")
PER_FILE_COLUMNS = ("file_name", "views", "tp", "fp", "fn", "iou_sum")


def gap_points() -> list[GridPoint]:
    """The grid a cross-fit chooses from."""
    return postproc_grid(GAP_THRESHOLDS, GAP_CLOSE_SIZES, GAP_MIN_AREAS)


def holdout_dates(holdout: Sequence[TrainImage]) -> list[dt.date]:
    """The observation date of each file, by the rule of the date split."""
    return [
        observation_date(
            {"file_name": image.file_name, "date_captured": image.date_captured}
        )
        for image in holdout
    ]


def repeated_crossfit(
    per_file: PerFileCounts,
    dates: Sequence[dt.date],
    *,
    seeds: Sequence[int] = CROSSFIT_SEEDS,
    n_folds: int = CROSSFIT_FOLDS,
) -> list[CrossfitResult]:
    """One cross-fit per seed, each on another partition of the dates."""
    if len(dates) != len(per_file.file_names):
        raise ValueError(
            f"{len(dates)} dates for {len(per_file.file_names)} files must match"
        )
    return [
        crossfit_select(per_file.counts, date_folds(dates, n_folds, seed=seed))
        for seed in seeds
    ]


def _row(
    quantity: str,
    value: float,
    low: float | None = None,
    high: float | None = None,
    files: int | None = None,
    note: str = "",
) -> dict[str, str]:
    return {
        "quantity": quantity,
        "value": format_pq(value),
        "low": "" if low is None else format_pq(low),
        "high": "" if high is None else format_pq(high),
        "files": "" if files is None else str(files),
        "note": note,
    }


def _describe_point(point: object) -> str:
    if isinstance(point, GridPoint):
        return (
            f"threshold {point.threshold:g}, close_size {point.close_size}, "
            f"min_area {point.min_area}"
        )
    return str(point)


def gap_summary_rows(
    per_file: PerFileCounts,
    dates: Sequence[dt.date],
    *,
    point: GridPoint = CURRENT_POINT,
    target_views: float = TEST_VIEWS_PER_FILE,
    seeds: Sequence[int] = CROSSFIT_SEEDS,
    n_folds: int = CROSSFIT_FOLDS,
    resamples: int = 2000,
    seed: int = 0,
) -> list[dict[str, str]]:
    """Every number the analysis reports, as CSV rows. Intervals are 95% bootstrap."""
    if point not in per_file.counts:
        raise ValueError(f"{point} was not scored")
    counts = per_file.counts[point]
    views = np.asarray(per_file.views)
    rows = []

    overall = bootstrap_pq(counts, n=resamples, seed=seed)
    rows.append(
        _row(
            "current_setting",
            overall.point,
            overall.low,
            overall.high,
            len(counts),
            _describe_point(point),
        )
    )

    for name, inside in (
        ("h2_stratum_one_view", views == 1),
        ("h2_stratum_several_views", views >= 2),
    ):
        chosen = [c for c, keep in zip(counts, inside, strict=True) if keep]
        if chosen:
            result = bootstrap_pq(chosen, n=resamples, seed=seed)
            rows.append(_row(name, result.point, result.low, result.high, len(chosen)))
        else:
            rows.append(_row(name, float("nan"), files=0, note="no such files"))

    # One row per exact count too: 2 and 3 views can behave differently.
    for count in sorted(set(views.tolist())):
        if count >= 2:
            chosen = [c for c, v in zip(counts, views, strict=True) if v == count]
            result = bootstrap_pq(chosen, n=resamples, seed=seed)
            rows.append(
                _row(
                    f"h2_stratum_views_{count}",
                    result.point,
                    result.low,
                    result.high,
                    len(chosen),
                )
            )

    try:
        weights = tilt_to_mean_views(views.tolist(), target_views)
    except ValueError as exc:
        # Say so in the table: the rest of the analysis still stands.
        rows.append(
            _row("h2_reweighted_views", float("nan"), files=len(counts), note=str(exc))
        )
    else:
        tilted = bootstrap_pq(counts, n=resamples, seed=seed, weights=weights)
        rows.append(
            _row(
                "h2_reweighted_views",
                tilted.point,
                tilted.low,
                tilted.high,
                len(counts),
                f"file mix tilted from {views.mean():.2f} to {target_views:g} views "
                f"per file; effective files {effective_files(weights):.0f}",
            )
        )

    results = repeated_crossfit(per_file, dates, seeds=seeds, n_folds=n_folds)
    best = results[0]
    rows.append(
        _row(
            "h1_sweep_maximum",
            best.best_pq,
            files=len(counts),
            note=f"{_describe_point(best.best_key)}; {len(per_file.counts)} settings",
        )
    )
    for used_seed, result in zip(seeds, results, strict=True):
        rows.append(
            _row(
                f"h1_crossfit_seed_{used_seed}",
                result.pooled_pq,
                files=len(counts),
                note="; ".join(_describe_point(key) for key in result.selected),
            )
        )
    pooled = [result.pooled_pq for result in results]
    rows.append(
        _row(
            "h1_crossfit_mean",
            float(np.mean(pooled)),
            float(np.min(pooled)),
            float(np.max(pooled)),
            len(counts),
            "low and high are the lowest and highest partition, not an interval",
        )
    )
    rows.append(
        _row(
            "h1_sweep_maximum_minus_crossfit",
            best.best_pq - float(np.mean(pooled)),
            files=len(counts),
            note="estimate of what selection adds; each fold chooses on half the files",
        )
    )
    return rows


def write_summary_rows(path: str | Path, rows: Sequence[dict[str, str]]) -> None:
    """Write :func:`gap_summary_rows` with the header ``SUMMARY_COLUMNS``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def format_summary(rows: Sequence[dict[str, str]]) -> str:
    """The rows as aligned lines, for the notebook to print."""
    lines = []
    for row in rows:
        span = f" [{row['low']}, {row['high']}]" if row["low"] else ""
        files = f" files={row['files']}" if row["files"] else ""
        note = f"  ({row['note']})" if row["note"] else ""
        lines.append(f"{row['quantity']:<34} {row['value']}{span}{files}{note}")
    return "\n".join(lines)


def per_file_rows(per_file: PerFileCounts, point: GridPoint) -> list[dict[str, str]]:
    """One CSV row per file for one setting. Counts only, no image or mask data."""
    if point not in per_file.counts:
        raise ValueError(f"{point} was not scored")
    rows = []
    for name, views, counts in zip(
        per_file.file_names, per_file.views, per_file.counts[point], strict=True
    ):
        rows.append(
            {
                "file_name": name,
                "views": str(views),
                "tp": str(counts.tp),
                "fp": str(counts.fp),
                "fn": str(counts.fn),
                "iou_sum": f"{counts.iou_sum:.6f}",
            }
        )
    return rows


def write_per_file_rows(path: str | Path, rows: Sequence[dict[str, str]]) -> None:
    """Write :func:`per_file_rows` with the header ``PER_FILE_COLUMNS``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_FILE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_per_file_counts(path: str | Path) -> list[PqCounts]:
    """Read per-file counts back, for a paired comparison between two runs."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [
            PqCounts(
                tp=int(row["tp"]),
                fp=int(row["fp"]),
                fn=int(row["fn"]),
                iou_sum=float(row["iou_sum"]),
            )
            for row in csv.DictReader(handle)
        ]
