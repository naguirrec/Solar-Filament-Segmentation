"""The solar-disk gate of the detector fusion, on synthetic frames."""

from __future__ import annotations

import numpy as np
import pytest

from filament import detector_fusion
from filament.detector_fusion import (
    Detection,
    FusionConfig,
    FusionFile,
    claim_pixels,
    per_file_detector_counts,
)

SIDE = 64
WINDOWS = [(0, 0, SIDE, 40), (0, 24, SIDE, SIDE)]


def frame(*boxes: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    for top, left, bottom, right in boxes:
        mask[top:bottom, left:right] = True
    return mask


def det(mask: np.ndarray, score: float, window: int = 0) -> Detection:
    return Detection.from_mask(mask, score, window)


def _disk(columns: int = 30) -> np.ndarray:
    gate = np.zeros((SIDE, SIDE), dtype=bool)
    gate[:, :columns] = True
    return gate


def test_disk_gate_keeps_claims_and_unclaimed_pieces_on_the_disk() -> None:
    unet = frame((10, 5, 20, 45), (40, 5, 50, 20))
    detection = det(frame((10, 5, 20, 45)), 0.9)
    labels, _ = claim_pixels([detection], unet, 0, 1, disk=_disk())
    assert labels[10:20, 5:30].all()
    assert not labels[:, 30:].any()
    # The second U-Net piece lies on the disk, so it is emitted as an unscored id.
    labels, scores = claim_pixels(
        [detection], unet, 0, 1, emit_unclaimed=True, disk=_disk()
    )
    assert labels.max() == 2 and np.isnan(scores[1])
    # Off the disk it is not.
    gate = _disk()
    gate[40:50] = False
    labels, scores = claim_pixels(
        [detection], unet, 0, 1, emit_unclaimed=True, disk=gate
    )
    assert labels.max() == 1 and scores.tolist() == [0.9]


def test_no_disk_is_no_gate_and_a_wrong_shape_is_an_error() -> None:
    unet = frame((10, 5, 20, 45))
    detection = det(unet, 0.9)
    plain, _ = claim_pixels([detection], unet, 0, 1)
    open_gate, _ = claim_pixels([detection], unet, 0, 1, disk=np.ones_like(unet))
    assert np.array_equal(plain, open_gate)
    with pytest.raises(ValueError, match="disk"):
        claim_pixels([detection], unet, 0, 1, disk=np.ones((3, 3), dtype=bool))


def test_the_disk_reaches_fuse_and_the_counts_through_fusion_file() -> None:
    unet = frame((10, 5, 20, 45))
    truth = frame((10, 5, 20, 30))
    detection = det(unet, 0.9)
    config = FusionConfig(dilation=0, min_area=1)
    open_file = FusionFile("a", [detection], unet, [[truth]])
    gated_file = FusionFile("a", [detection], unet, [[truth]], disk=_disk())
    (open_counts,) = per_file_detector_counts([open_file], WINDOWS, config)
    (gated_counts,) = per_file_detector_counts([gated_file], WINDOWS, config)
    assert open_counts.iou_sum == pytest.approx(25 / 40)
    assert gated_counts.iou_sum == pytest.approx(1.0)
    labels, _ = detector_fusion.fuse([detection], unet, WINDOWS, config, _disk())
    assert not labels[:, 30:].any()
