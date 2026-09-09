"""Truthful three-image CURRENT/SOURCE role presentation for Cycle 15."""

from __future__ import annotations

import io
from collections.abc import Mapping

import numpy as np
from PIL import Image, ImageDraw

PANEL_SIZE = (512, 286)
OFFICIAL_VIEW_SIZE = (256, 256)
HEADER_HEIGHT = 30
CURRENT_LABEL = "IMAGE 1 CURRENT | LEFT CAMERA | RIGHT CAMERA"
SOURCE_LABEL = "IMAGE 2 SOURCE REFERENCE | LEFT CAMERA | RIGHT CAMERA"


def _panel_views(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
    if image.shape != (PANEL_SIZE[1], PANEL_SIZE[0], 3):
        raise ValueError("Cycle 15 input panel dimensions drifted")
    official = image[HEADER_HEIGHT:, :, :]
    return official[:, :256, :].copy(), official[:, 256:, :].copy()


def _mosaic(left: np.ndarray, right: np.ndarray, *, label: str) -> bytes:
    if left.shape != (256, 256, 3) or right.shape != (256, 256, 3):
        raise ValueError("Cycle 15 official external view dimensions drifted")
    canvas = Image.new("RGB", PANEL_SIZE, "white")
    canvas.paste(Image.fromarray(left, mode="RGB"), (0, HEADER_HEIGHT))
    canvas.paste(Image.fromarray(right, mode="RGB"), (256, HEADER_HEIGHT))
    ImageDraw.Draw(canvas).text((4, 7), label, fill="red")
    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=False)
    return output.getvalue()


def cycle15_three_images(images: Mapping[str, bytes]) -> dict[str, bytes]:
    """Re-arrange all public external pixels; preserve wrist bytes exactly."""
    if set(images) != {"left", "right", "wrist"}:
        raise ValueError("Cycle 15 requires exactly the three official image slots")
    current_left, source_left = _panel_views(images["left"])
    current_right, source_right = _panel_views(images["right"])
    wrist = np.asarray(Image.open(io.BytesIO(images["wrist"])).convert("RGB"))
    if wrist.shape != (256, 256, 3):
        raise ValueError("Cycle 15 official wrist dimensions drifted")
    return {
        "left": _mosaic(current_left, current_right, label=CURRENT_LABEL),
        "right": _mosaic(source_left, source_right, label=SOURCE_LABEL),
        "wrist": images["wrist"],
    }


def cycle15_official_pixels(images: Mapping[str, bytes]) -> dict[str, np.ndarray]:
    """Expose exact official pixels for authority tests, never model state."""
    current_left, source_left = _panel_views(images["left"])
    current_right, source_right = _panel_views(images["right"])
    wrist = np.asarray(Image.open(io.BytesIO(images["wrist"])).convert("RGB"))
    return {
        "current_left": current_left,
        "current_right": current_right,
        "source_left": source_left,
        "source_right": source_right,
        "current_wrist": wrist,
    }
