"""The table and the verdict of an axis sweep, on synthetic per-file counts."""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from filament.axis_report import (
    AXIS_COLUMNS,
    axis_rows,
    crossfit_rows,
    format_axis_rows,
    write_axis_rows,
)
from filament.metrics import PqCounts


def _noisy(rng: np.random.Generator, mean: float, n: int = 142) -> tuple[PqCounts, ...]:
    values = np.clip(rng.normal(mean, 0.15, n), 0.0, 1.0)
    return tuple(PqCounts(tp=10, iou_sum=10 * float(v)) for v in values)


def _shifted(base: tuple[PqCounts, ...], gain: float) -> tuple[PqCounts, ...]:
    return tuple(PqCounts(tp=c.tp, iou_sum=c.iou_sum + 10 * gain) for c in base)


def _dates(n: int = 142) -> list[dt.date]:
    return [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)]


def test_axis_rows_report_the_paired_difference() -> None:
    base = _noisy(np.random.default_rng(0), 0.4)
    rows = axis_rows(
        "t", base, {"same": base, "up": _shifted(base, 0.02)}, resamples=200
    )
    same, up = rows
    assert same["setting"] == "same" and float(same["diff_vs_base"]) == 0.0
    assert float(same["fraction_better"]) == 0.0
    assert float(up["diff_vs_base"]) == pytest.approx(0.02)
    assert float(up["fraction_better"]) == 1.0
    assert float(up["pooled_pq"]) > float(same["pooled_pq"])


def test_axis_rows_use_the_given_description() -> None:
    base = _noisy(np.random.default_rng(0), 0.4, 10)
    rows = axis_rows("t", base, {1: base}, describe=lambda k: f"k={k}", resamples=10)
    assert rows[0]["setting"] == "k=1"


def test_a_real_gain_is_adopted() -> None:
    base = _noisy(np.random.default_rng(0), 0.4)
    settings = {"base": base, "up": _shifted(base, 0.02)}
    rows, verdict = crossfit_rows("t", base, settings, _dates(), resamples=200)
    assert verdict is True
    assert rows[-1]["setting"] == "crossfit_verdict"
    assert rows[-1]["note"].startswith("ADOPT: 5 of 5")
    assert len(rows) == 6
    assert "chose up and up" in rows[0]["note"]


def test_a_gain_below_the_rule_is_not_adopted_even_if_consistent() -> None:
    base = _noisy(np.random.default_rng(0), 0.4)
    settings = {"base": base, "up": _shifted(base, 0.001)}
    rows, verdict = crossfit_rows("t", base, settings, _dates(), resamples=200)
    assert verdict is False
    assert rows[-1]["note"].startswith("DO NOT ADOPT: 0 of 5")


def test_a_peak_from_noise_is_not_adopted() -> None:
    rng = np.random.default_rng(1)
    base = _noisy(rng, 0.4)
    settings = {"base": base, **{i: _noisy(rng, 0.4) for i in range(40)}}
    _rows, verdict = crossfit_rows("t", base, settings, _dates(), resamples=200)
    assert verdict is False


def test_the_csv_has_the_columns_and_the_text_prints(tmp_path: Path) -> None:
    base = _noisy(np.random.default_rng(0), 0.4, 20)
    rows = axis_rows("t", base, {"same": base}, resamples=20)
    more, _ = crossfit_rows("t", base, {"same": base}, _dates(20), resamples=20)
    all_rows = rows + more
    path = tmp_path / "axis.csv"
    write_axis_rows(path, all_rows)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == AXIS_COLUMNS
        assert list(reader) == all_rows
    text = format_axis_rows(all_rows)
    assert "crossfit_verdict" in text and "DO NOT ADOPT" in text
