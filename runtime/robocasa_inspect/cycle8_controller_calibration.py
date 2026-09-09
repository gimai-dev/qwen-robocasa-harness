"""Fable-reviewed Cycle 8 public-EEF controller calibration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .contracts import OSC_ROTATION_PER_STEP_RAD, OSC_TRANSLATION_PER_STEP_M

TRANSLATION_CONTROLLER_GAIN = 3.5
ROTATION_CONTROLLER_GAIN = 3.5
MAX_REALIZED_TRANSLATION_PER_STEP_M = 0.005
MAX_REALIZED_ROTATION_PER_STEP_RAD = 0.025


def authored_delta_from_controller_action(
    action: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Recover the authored physical delta from calibrated controller units."""
    translation = (
        np.asarray(action["action.end_effector_position"], dtype=np.float64)
        * OSC_TRANSLATION_PER_STEP_M
        / TRANSLATION_CONTROLLER_GAIN
    )
    rotation = (
        np.asarray(action["action.end_effector_rotation"], dtype=np.float64)
        * OSC_ROTATION_PER_STEP_RAD
        / ROTATION_CONTROLLER_GAIN
    )
    return translation, rotation


def realized_step_overshot(
    translation_m: Sequence[float], rotation_rad: float
) -> bool:
    """Fail closed when an immediate public EEF transition exceeds its contract."""
    translation = np.asarray(translation_m, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        return True
    if not np.isfinite(rotation_rad):
        return True
    return bool(
        np.linalg.norm(translation)
        > MAX_REALIZED_TRANSLATION_PER_STEP_M + 1e-12
        or rotation_rad > MAX_REALIZED_ROTATION_PER_STEP_RAD + 1e-12
    )
