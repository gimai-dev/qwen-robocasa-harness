"""Predicate-free servo-regime identification for Cycle 10."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def calibration_schedule(
    *, translation_m: float, rotation_rad: float
) -> list[dict[str, object]]:
    if not 0.0 < translation_m <= 0.004 or not 0.0 < rotation_rad <= 0.020:
        raise ValueError("Cycle 10 calibration profile is invalid")
    rows: list[dict[str, object]] = []
    for kind in ("translation", "rotation"):
        for axis_index, axis in enumerate("xyz"):
            for cycle in range(5):
                sign = 1 if cycle % 2 == 0 else -1
                for _ in range(5):
                    rows.append(
                        {
                            "kind": kind,
                            "axis": axis,
                            "axis_index": axis_index,
                            "condition": "settle",
                            "sign": 0,
                            "translation": np.zeros(3),
                            "rotation": np.zeros(3),
                        }
                    )
                for tick in range(10):
                    direction = sign if tick < 5 else -sign
                    translation = np.zeros(3)
                    rotation = np.zeros(3)
                    if kind == "translation":
                        translation[axis_index] = direction * translation_m
                    else:
                        rotation[axis_index] = direction * rotation_rad
                    rows.append(
                        {
                            "kind": kind,
                            "axis": axis,
                            "axis_index": axis_index,
                            "condition": "rest" if tick == 0 else "mid_motion",
                            "sign": direction,
                            "translation": translation,
                            "rotation": rotation,
                        }
                    )
    return rows


def select_green_profile(
    profiles: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    for profile in profiles:
        if (
            float(profile["max_realized_translation_m"]) <= 0.0045 + 1e-12
            and float(profile["max_realized_rotation_rad"]) <= 0.0225 + 1e-12
            and float(profile["max_rest_precondition_translation_m"])
            <= 0.0005 + 1e-12
            and float(profile["max_rest_precondition_rotation_rad"])
            <= 0.003 + 1e-12
        ):
            return profile
    raise RuntimeError("Cycle 10 servo calibration has no green profile")
