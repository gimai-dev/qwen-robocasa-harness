"""Public-RGB anchor selection for broad movable surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

IMAGE_SIZE_PX = 256
MIN_BORDER_MARGIN_PX = 12
SURFACE_SEARCH_RADIUS_PX = 40


@dataclass(frozen=True)
class SurfaceAnchor:
    anchor_uv_px: tuple[float, float]
    semantic_disagreement_px: float
    candidate_count: int


def refine_trackable_surface_anchor(
    image: object, semantic_uv_px: object
) -> SurfaceAnchor:
    """Select the nearest strong corner inside a cited public-RGB surface region."""

    rgb = np.asarray(image)
    if rgb.shape != (IMAGE_SIZE_PX, IMAGE_SIZE_PX, 3) or rgb.dtype != np.uint8:
        raise ValueError("surface refinement requires one uint8 256x256 RGB image")
    semantic = np.asarray(semantic_uv_px, dtype=np.float64)
    if (
        semantic.shape != (2,)
        or not np.isfinite(semantic).all()
        or np.any(semantic < MIN_BORDER_MARGIN_PX)
        or np.any(semantic > IMAGE_SIZE_PX - 1 - MIN_BORDER_MARGIN_PX)
    ):
        raise ValueError("semantic surface point is outside the certified interior")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mask = np.zeros((IMAGE_SIZE_PX, IMAGE_SIZE_PX), dtype=np.uint8)
    cv2.circle(
        mask,
        tuple(round(value) for value in semantic),
        SURFACE_SEARCH_RADIUS_PX,
        255,
        -1,
    )
    mask[:MIN_BORDER_MARGIN_PX] = 0
    mask[-MIN_BORDER_MARGIN_PX:] = 0
    mask[:, :MIN_BORDER_MARGIN_PX] = 0
    mask[:, -MIN_BORDER_MARGIN_PX:] = 0
    found = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=64,
        qualityLevel=0.01,
        minDistance=5,
        mask=mask,
        blockSize=3,
    )
    if found is None:
        raise ValueError("no trackable surface corner matches the citation")
    candidates = found.reshape(-1, 2).astype(np.float64)
    ranked = sorted(
        (
            float(np.linalg.norm(candidate - semantic)),
            float(candidate[0]),
            float(candidate[1]),
        )
        for candidate in candidates
    )
    distance, u, v = ranked[0]
    if distance > SURFACE_SEARCH_RADIUS_PX:
        raise ValueError("no trackable surface corner matches the citation")
    return SurfaceAnchor((u, v), distance, len(candidates))
