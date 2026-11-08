"""Compare settings against a base with the adoption rule of the roadmap.

One table for a sweep along an axis: each setting's pooled PQ, and its paired
difference from the base over the same resampled files. A sweep's best row is
biased up by selection, so :func:`crossfit_rows` also scores the whole procedure:
each partition of the dates chooses a setting on some files and scores it on the
others, and those out-of-fold counts are what gets compared with the base.

``docs/ROADMAP_V2.md``: adopt a change only if the paired bootstrap supports it (a
gain of at least 0.005 in at least 90% of resamples), here in at least 4 of 5
partitions of the dates. Pure numpy.
"""

from __future__ import annotations

import csv
import datetime as dt
from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path

import numpy as np

from filament.metrics import PqCounts
from filament.stats import (
    adopts_change,
    crossfit_select,
    date_folds,
    paired_bootstrap,
    weighted_pq,
)
from filament.sweep import format_pq

AXIS_COLUMNS = (
    "axis",
    "setting",
    "pooled_pq",
    "diff_vs_base",
    "diff_low",
    "diff_high",
    "fraction_better",
    "note",
)

MIN_PARTITIONS_ADOPTED = 4


def _describe(setting: Hashable) -> str:
    return str(setting)


def axis_rows(
    axis: str,
    base: Sequence[PqCounts],
    settings: Mapping[Hashable, Sequence[PqCounts]],
    *,
    describe: Callable[[Hashable], str] = _describe,
    resamples: int = 2000,
    seed: int = 0,
) -> list[dict[str, str]]:
    """One row per setting: its pooled PQ and its paired difference from ``base``."""
    rows = []
    for setting, counts in settings.items():
        comparison = paired_bootstrap(base, counts, n=resamples, seed=seed)
        rows.append(
            {
                "axis": axis,
                "setting": describe(setting),
                "pooled_pq": format_pq(weighted_pq(counts)),
                "diff_vs_base": format_pq(comparison.point),
                "diff_low": format_pq(comparison.low),
                "diff_high": format_pq(comparison.high),
                "fraction_better": f"{comparison.fraction_better:.3f}",
                "note": "",
            }
        )
    return rows


def crossfit_rows(
    axis: str,
    base: Sequence[PqCounts],
    settings: Mapping[Hashable, Sequence[PqCounts]],
    dates: Sequence[dt.date],
    *,
    describe: Callable[[Hashable], str] = _describe,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    n_folds: int = 2,
    resamples: int = 2000,
) -> tuple[list[dict[str, str]], bool]:
    """Cross-fit over ``seeds``. Returns one row per partition, a verdict row, and it.

    The verdict adopts the procedure when the paired bootstrap of the out-of-fold
    counts against ``base`` passes the roadmap's rule in at least
    :data:`MIN_PARTITIONS_ADOPTED` partitions. ``settings`` should include the base
    itself, so a partition may keep it.
    """
    rows = []
    adopted = 0
    pooled = []
    for seed in seeds:
        result = crossfit_select(settings, date_folds(dates, n_folds, seed=seed))
        comparison = paired_bootstrap(base, result.out_of_fold, n=resamples, seed=seed)
        passed = adopts_change(comparison)
        adopted += passed
        pooled.append(result.pooled_pq)
        rows.append(
            {
                "axis": axis,
                "setting": f"crossfit_seed_{seed}",
                "pooled_pq": format_pq(result.pooled_pq),
                "diff_vs_base": format_pq(comparison.point),
                "diff_low": format_pq(comparison.low),
                "diff_high": format_pq(comparison.high),
                "fraction_better": f"{comparison.fraction_better:.3f}",
                "note": (
                    ("passes; " if passed else "does not pass; ")
                    + "chose "
                    + " and ".join(describe(key) for key in result.selected)
                ),
            }
        )
    verdict = adopted >= MIN_PARTITIONS_ADOPTED
    rows.append(
        {
            "axis": axis,
            "setting": "crossfit_verdict",
            "pooled_pq": format_pq(float(np.mean(pooled))),
            "diff_vs_base": format_pq(float(np.mean(pooled)) - weighted_pq(base)),
            "diff_low": "",
            "diff_high": "",
            "fraction_better": "",
            "note": (
                f"{'ADOPT' if verdict else 'DO NOT ADOPT'}: {adopted} of {len(seeds)} "
                f"partitions pass, {MIN_PARTITIONS_ADOPTED} needed"
            ),
        }
    )
    return rows, verdict


def write_axis_rows(path: str | Path, rows: Sequence[dict[str, str]]) -> None:
    """Write the rows with the header ``AXIS_COLUMNS``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AXIS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def format_axis_rows(rows: Sequence[dict[str, str]]) -> str:
    """The rows as aligned lines, for a notebook to print."""
    lines = []
    for row in rows:
        span = f" [{row['diff_low']}, {row['diff_high']}]" if row["diff_low"] else ""
        better = f" better={row['fraction_better']}" if row["fraction_better"] else ""
        note = f"  {row['note']}" if row["note"] else ""
        lines.append(
            f"{row['axis']:<10} {row['setting']:<22} pq={row['pooled_pq']} "
            f"diff={row['diff_vs_base']}{span}{better}{note}"
        )
    return "\n".join(lines)
