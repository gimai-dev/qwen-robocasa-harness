"""Official mobile-base active-perception primitives and certification gates."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

BASE_AXES = {"x": 0, "y": 1, "yaw": 2}


def base_pulse_chunk(
    *, axis: str, normalized_velocity: float, steps: int = 5
) -> list[dict[str, np.ndarray]]:
    """Build a pulse using the official action mapping; torso is always zero."""
    if axis not in BASE_AXES:
        raise ValueError("unsupported base axis")
    if (
        isinstance(normalized_velocity, bool)
        or not np.isfinite(normalized_velocity)
        or abs(normalized_velocity) > 1.0
        or abs(normalized_velocity) <= 1e-12
    ):
        raise ValueError("base pulse exceeds its normalized bound")
    if steps != 5:
        raise ValueError("official pulse must contain exactly five steps")
    base = np.zeros(4, dtype=np.float64)
    base[BASE_AXES[axis]] = normalized_velocity
    action = {
        "action.end_effector_position": np.zeros(3, dtype=np.float64),
        "action.end_effector_rotation": np.zeros(3, dtype=np.float64),
        "action.gripper_close": np.zeros(1, dtype=np.float64),
        "action.base_motion": base,
        "action.control_mode": np.zeros(1, dtype=np.float64),
    }
    return [
        {name: value.copy() for name, value in action.items()} for _ in range(steps)
    ]


def torso_pulse_chunk(
    *, normalized_velocity: float, steps: int = 5
) -> list[dict[str, np.ndarray]]:
    """Build one bounded official torso pulse with the mobile base held still."""
    if (
        isinstance(normalized_velocity, bool)
        or not np.isfinite(normalized_velocity)
        or abs(normalized_velocity) > 0.25
        or abs(normalized_velocity) <= 1e-12
    ):
        raise ValueError("torso pulse exceeds its normalized bound")
    if steps != 5:
        raise ValueError("official pulse must contain exactly five steps")
    body = np.zeros(4, dtype=np.float64)
    body[3] = normalized_velocity
    action = {
        "action.end_effector_position": np.zeros(3, dtype=np.float64),
        "action.end_effector_rotation": np.zeros(3, dtype=np.float64),
        "action.gripper_close": np.zeros(1, dtype=np.float64),
        "action.base_motion": body,
        "action.control_mode": np.zeros(1, dtype=np.float64),
    }
    return [
        {name: value.copy() for name, value in action.items()} for _ in range(steps)
    ]


def characterize_axis(
    *,
    axis: str,
    excursions: Sequence[float],
    return_errors: Sequence[float],
    changed_fractions: Sequence[float],
    returned_mae: Sequence[float],
) -> dict[str, object]:
    """Apply the locked two-repeat response/return gates to one base axis."""
    if axis not in BASE_AXES:
        raise ValueError("unsupported base axis")
    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in (excursions, return_errors, changed_fractions, returned_mae)
    ]
    if any(array.shape != (2,) or not np.isfinite(array).all() for array in arrays):
        raise ValueError("base characterization requires two finite repeats")
    excursion, returned, changed, mae = arrays
    if np.any(excursion <= 0) or np.any(returned < 0) or np.any(changed < 0):
        raise ValueError("base characterization evidence is invalid")
    repeat_fraction = float(abs(excursion[1] - excursion[0]) / min(excursion))
    absolute_return = 0.02 if axis == "yaw" else 0.002
    return_limits = np.maximum(0.20 * excursion, absolute_return)
    gates = {
        "return": bool(np.all(returned <= return_limits)),
        "repeat": repeat_fraction <= 0.20,
        "motion_response": bool(np.all(changed >= 0.005)),
        "frame_return": bool(np.all(mae <= 2.0)),
    }
    return {
        "axis": axis,
        "enabled": all(gates.values()),
        "gates": gates,
        "excursions": excursion.tolist(),
        "return_errors": returned.tolist(),
        "return_limits": return_limits.tolist(),
        "changed_fractions": changed.tolist(),
        "returned_mae": mae.tolist(),
        "repeat_fraction": repeat_fraction,
    }
