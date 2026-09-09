"""Derive external-view gripper anchors from reversible public RGB articulation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

IMAGE_SIZE = 256
IMAGE_BORDER_PX = 12


@dataclass(frozen=True)
class GripperAnchorEvidence:
    anchor_uv_px: tuple[float, float]
    changed_pixels: int
    component_pixels: int
    return_mae: float


def _rgb(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or image.dtype != np.uint8:
        raise ValueError("gripper articulation requires official 256x256 RGB")
    return image


def locate_gripper_anchor(
    opened_rgb: np.ndarray,
    closed_rgb: np.ndarray,
    returned_rgb: np.ndarray,
    *,
    static_noise_px: float,
) -> GripperAnchorEvidence:
    """Locate the largest reversible finger-motion component in one external view."""
    opened = _rgb(opened_rgb)
    closed = _rgb(closed_rgb)
    returned = _rgb(returned_rgb)
    noise = float(static_noise_px)
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("static noise is invalid")
    return_mae = float(np.mean(np.abs(returned.astype(np.float32) - opened)))
    if return_mae > max(2.0, 4.0 * noise):
        raise ValueError("gripper articulation failed its return gate")
    difference = np.max(
        np.abs(closed.astype(np.int16) - opened.astype(np.int16)), axis=2
    )
    threshold = max(2.0, 4.0 * noise)
    mask = (difference > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    changed = int(np.count_nonzero(mask))
    if changed < 10:
        raise ValueError("gripper articulation is unobservable")
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates: list[tuple[int, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= 8:
            candidates.append((area, label))
    if not candidates:
        raise ValueError("gripper articulation has no stable component")
    area, label = max(candidates)
    x, y = (float(value) for value in centroids[label])
    if min(x, y, IMAGE_SIZE - 1 - x, IMAGE_SIZE - 1 - y) < IMAGE_BORDER_PX:
        raise ValueError("gripper articulation component violates the image border")
    return GripperAnchorEvidence((x, y), changed, area, return_mae)
