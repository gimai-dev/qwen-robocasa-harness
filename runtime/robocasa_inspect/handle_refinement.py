"""Public-RGB refinement of a semantic drawer-handle citation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

IMAGE_SIZE = 256


@dataclass(frozen=True)
class HandleRefinement:
    anchor_uv_px: tuple[float, float]
    bounding_box_xywh: tuple[int, int, int, int]
    area_px: int


def handle_axis_endpoints(refinement: HandleRefinement) -> np.ndarray:
    """Return a textured inset point and centroid along the handle's long axis."""
    left, top, width, height = refinement.bounding_box_xywh
    inset = max(3, min(8, width // 8))
    if width - 2 * inset < 12:
        raise ValueError("refined handle is too short for an axis estimate")
    center_y = top + 0.5 * (height - 1)
    return np.asarray(
        [[left + inset, center_y], list(refinement.anchor_uv_px)],
        dtype=np.float64,
    )


def refine_horizontal_metal_handle(
    rgb: np.ndarray, semantic_uv_px: object
) -> HandleRefinement:
    """Find the nearest neutral, elongated handle component around a VLM citation."""
    image = np.asarray(rgb)
    semantic = np.asarray(semantic_uv_px, dtype=np.float64)
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or image.dtype != np.uint8:
        raise ValueError("handle refinement requires official 256x256 RGB")
    if semantic.shape != (2,) or not np.isfinite(semantic).all():
        raise ValueError("semantic handle citation is invalid")
    x, y = semantic
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    neutral = ((hsv[:, :, 1] < 90) & (hsv[:, :, 2] > 80)).astype(np.uint8)
    roi = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    x0 = max(0, int(np.floor(x - 60)))
    x1 = min(IMAGE_SIZE, int(np.ceil(x + 60)) + 1)
    y0 = max(0, int(np.floor(y - 60)))
    y1 = min(IMAGE_SIZE, int(np.ceil(y + 40)) + 1)
    roi[y0:y1, x0:x1] = 1
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        neutral * roi, connectivity=8
    )
    candidates: list[tuple[float, int]] = []
    for component in range(1, count):
        left, top, width, height, area = (
            int(value) for value in stats[component]
        )
        if not (30 <= area <= 1_000 and width >= 2 * height and height >= 3):
            continue
        center = np.asarray(centroids[component], dtype=np.float64)
        distance = float(np.linalg.norm(center - semantic))
        candidates.append((distance, component))
    if not candidates:
        raise ValueError("no observable horizontal metal handle near citation")
    _, component = min(candidates)
    left, top, width, height, area = (
        int(value) for value in stats[component]
    )
    center = np.asarray(centroids[component], dtype=np.float64)
    if np.any(center < 12.0) or np.any(center > IMAGE_SIZE - 13.0):
        raise ValueError("refined handle violates the 12-pixel border")
    return HandleRefinement(
        anchor_uv_px=(float(center[0]), float(center[1])),
        bounding_box_xywh=(left, top, width, height),
        area_px=area,
    )
