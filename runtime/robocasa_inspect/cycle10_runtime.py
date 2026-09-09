"""Runtime-only closed helpers for the Fable-accepted Cycle 10 harness."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .contracts import OSC_ROTATION_PER_STEP_RAD, OSC_TRANSLATION_PER_STEP_M
from .cycle8_controller_calibration import (
    ROTATION_CONTROLLER_GAIN,
    TRANSLATION_CONTROLLER_GAIN,
)

MAX_TRANSLATION_TICK_M = 0.003
MAX_ROTATION_TICK_RAD = 0.020


def resolve_gripper(choice: str, source_value: float) -> str:
    if choice not in {"source", "open", "close"} or not math.isfinite(source_value):
        raise ValueError("Cycle 10 gripper choice is invalid")
    if choice != "source":
        return choice
    return "close" if source_value >= 0.5 else "open"


def material_qwen_contribution(
    residual_id: str, gripper_choice: str, source_value: float
) -> bool:
    source = resolve_gripper("source", source_value)
    resolved = resolve_gripper(gripper_choice, source_value)
    return residual_id != "zero" or resolved != source


def predicted_servo_ticks(translation_error_m: float, rotation_error_rad: float) -> int:
    values = (float(translation_error_m), float(rotation_error_rad))
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("Cycle 10 servo error is invalid")
    return max(
        math.ceil(values[0] / MAX_TRANSLATION_TICK_M - 1e-12),
        math.ceil(values[1] / MAX_ROTATION_TICK_RAD - 1e-12),
    )


def servo_action(
    translation_m: Sequence[float], rotation_rad: Sequence[float], gripper: str
) -> dict[str, np.ndarray]:
    translation = np.asarray(translation_m, dtype=np.float64)
    rotation = np.asarray(rotation_rad, dtype=np.float64)
    if (
        translation.shape != (3,)
        or rotation.shape != (3,)
        or not np.isfinite(translation).all()
        or not np.isfinite(rotation).all()
        or np.linalg.norm(translation) > MAX_TRANSLATION_TICK_M + 1e-12
        or np.linalg.norm(rotation) > MAX_ROTATION_TICK_RAD + 1e-12
        or gripper not in {"open", "close"}
    ):
        raise ValueError("Cycle 10 controller tick is invalid")
    return {
        "action.end_effector_position": (
            translation * TRANSLATION_CONTROLLER_GAIN / OSC_TRANSLATION_PER_STEP_M
        ),
        "action.end_effector_rotation": (
            rotation * ROTATION_CONTROLLER_GAIN / OSC_ROTATION_PER_STEP_RAD
        ),
        "action.gripper_close": np.asarray([1.0 if gripper == "close" else 0.0]),
        "action.base_motion": np.zeros(4),
        "action.control_mode": np.asarray([0.0]),
    }


def calibration_servo_action(
    translation_m: Sequence[float], rotation_rad: Sequence[float], gripper: str
) -> dict[str, np.ndarray]:
    """Emit only the Fable-frozen predicate-free 4 mm/20 mrad probe tick."""
    translation = np.asarray(translation_m, dtype=np.float64)
    rotation = np.asarray(rotation_rad, dtype=np.float64)
    if (
        translation.shape != (3,)
        or rotation.shape != (3,)
        or not np.isfinite(translation).all()
        or not np.isfinite(rotation).all()
        or np.linalg.norm(translation) > 0.004 + 1e-12
        or np.linalg.norm(rotation) > 0.020 + 1e-12
        or gripper not in {"open", "close"}
    ):
        raise ValueError("Cycle 10 calibration tick is invalid")
    return {
        "action.end_effector_position": (
            translation * TRANSLATION_CONTROLLER_GAIN / OSC_TRANSLATION_PER_STEP_M
        ),
        "action.end_effector_rotation": (
            rotation * ROTATION_CONTROLLER_GAIN / OSC_ROTATION_PER_STEP_RAD
        ),
        "action.gripper_close": np.asarray([1.0 if gripper == "close" else 0.0]),
        "action.base_motion": np.zeros(4),
        "action.control_mode": np.asarray([0.0]),
    }
