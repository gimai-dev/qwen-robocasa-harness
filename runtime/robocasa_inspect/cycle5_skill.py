"""Closed schema and bounded direct-EEF macro execution for Cycle 5."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .contracts import OSC_ROTATION_PER_STEP_RAD, OSC_TRANSLATION_PER_STEP_M
from .cycle8_controller_calibration import (
    ROTATION_CONTROLLER_GAIN,
    TRANSLATION_CONTROLLER_GAIN,
)
from .goal30_cycle7 import ROTATION_COMPONENT_MAX_RAD, TRANSLATION_COMPONENT_MAX_M

PHASES = ("approach", "contact", "actuate", "transport", "release", "verify")
NOTE_PATTERN = re.compile(r"[A-Za-z0-9 .,;:!?()'/_+\-]{1,120}")
_MOVE_FIELDS = {
    "kind",
    "observation_token",
    "phase",
    "candidate_id",
    "translation_m",
    "rotation_axis_angle_rad",
    "gripper",
    "note",
}
_POSE_MOVE_FIELDS = (
    _MOVE_FIELDS
    - {"translation_m", "rotation_axis_angle_rad"}
    | {
        "translation_m_f64_repr",
        "lateral_residual_m_f64_repr",
        "rotation_axis_angle_rad_f64_repr",
        "residual_id",
    }
)
_COMPACT_POSE_MOVE_FIELDS = {
    "kind",
    "observation_token",
    "phase",
    "candidate_id",
    "residual_id",
    "gripper",
    "note",
}
_TERMINAL_FIELDS = {"kind", "observation_token", "note"}


def lateral_residual_bank(source_translation: object) -> tuple[dict[str, object], ...]:
    """Build the deterministic public-source const bank accepted by Fable."""
    source = np.asarray(source_translation, dtype=np.float64)
    if source.shape != (3,) or not np.isfinite(source).all():
        raise ValueError("source tangent must contain three finite values")
    source_norm = float(np.linalg.norm(source))
    if source_norm <= 1e-12:
        bases = (np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0]))
    else:
        tangent = source / source_norm
        axes = np.eye(3, dtype=np.float64)
        axis = axes[int(np.argmin(np.abs(axes @ tangent)))]
        first = axis - float(np.dot(axis, tangent)) * tangent
        first /= np.linalg.norm(first)
        second = np.cross(tangent, first)
        second /= np.linalg.norm(second)
        bases = (first, second)
    options: list[dict[str, object]] = [
        {
            "residual_id": "zero",
            "lateral_residual_m": [0.0, 0.0, 0.0],
            "translation_m": source.tolist(),
        }
    ]
    for direction_index, basis in enumerate(bases):
        for magnitude in (0.003, 0.006, 0.009):
            for sign_name, sign in (("negative", -1.0), ("positive", 1.0)):
                residual = basis * (sign * magnitude)
                final = source + residual
                if np.max(np.abs(final)) > TRANSLATION_COMPONENT_MAX_M + 1e-12:
                    continue
                options.append(
                    {
                        "residual_id": (
                            f"lateral_{direction_index}_{sign_name}_{int(magnitude * 1000):03d}mm"
                        ),
                        "lateral_residual_m": residual.tolist(),
                        "translation_m": final.tolist(),
                    }
                )
    return tuple(options)


def presented_residual_bank(
    source_translation: object,
) -> tuple[dict[str, object], ...]:
    """Describe every deterministic branch for an informed compact selection."""
    return tuple(
        {
            "residual_id": option["residual_id"],
            "lateral_offset_mm": [
                float(value) * 1000.0 for value in option["lateral_residual_m"]
            ],
            "resulting_translation_m_f64_repr": [
                repr(float(value)) for value in option["translation_m"]
            ],
        }
        for option in lateral_residual_bank(source_translation)
    )


def cycle5_response_schema(
    observation_token: str | None = None,
    *,
    source_candidates: Mapping[str, Mapping[str, object]] | None = None,
    compact_pose: bool = False,
) -> dict[str, object]:
    if observation_token is not None and not re.fullmatch(
        r"[0-9a-f]{12}", observation_token
    ):
        raise ValueError("Cycle 5 schema token must be twelve lowercase hex characters")
    token = (
        {"const": observation_token}
        if observation_token is not None
        else {"type": "string", "pattern": "^[0-9a-f]{12}$"}
    )
    note = {
        "type": "string",
        "minLength": 1,
        "maxLength": 120,
        "pattern": "^[A-Za-z0-9 .,;:!?()'/_+\\-]{1,120}$",
    }
    def bounded_vector(limit: float) -> dict[str, object]:
        component = {
            "type": "number",
            "minimum": -limit,
            "maximum": limit,
        }
        return {
        "type": "array",
        "prefixItems": [dict(component) for _ in range(3)],
        "minItems": 3,
        "maxItems": 3,
        }
    def exact_repr_vector(value: object) -> dict[str, object]:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("Cycle 7 source candidate vector is invalid")
        return {
            "type": "array",
            "prefixItems": [{"const": repr(float(item))} for item in vector],
            "minItems": 3,
            "maxItems": 3,
        }

    def custom_move_branch() -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "kind": {"const": "move"},
                "observation_token": token,
                "phase": {"enum": list(PHASES)},
                "candidate_id": {"const": "custom"},
                "translation_m": bounded_vector(TRANSLATION_COMPONENT_MAX_M),
                "rotation_axis_angle_rad": bounded_vector(ROTATION_COMPONENT_MAX_RAD),
                "gripper": {"enum": ["open", "hold", "close"]},
                "note": note,
            },
            "required": sorted(_MOVE_FIELDS),
            "additionalProperties": False,
        }

    def pose_move_branch(
        candidate: Mapping[str, object], option: Mapping[str, object]
    ) -> dict[str, object]:
        source_translation = np.asarray(candidate["translation_m"], dtype=np.float64)
        if source_translation.shape != (3,) or not np.isfinite(source_translation).all():
            raise ValueError("Cycle 7 source translation is invalid")
        return {
            "type": "object",
            "properties": {
                "kind": {"const": "move"},
                "observation_token": token,
                "phase": {"enum": list(PHASES)},
                "candidate_id": {"const": "source_plus_residual"},
                "residual_id": {"const": option["residual_id"]},
                "translation_m_f64_repr": exact_repr_vector(option["translation_m"]),
                "lateral_residual_m_f64_repr": exact_repr_vector(
                    option["lateral_residual_m"]
                ),
                "rotation_axis_angle_rad_f64_repr": exact_repr_vector(
                    candidate["rotation_axis_angle_rad"]
                ),
                "gripper": {"enum": ["open", "hold", "close"]},
                "note": note,
            },
            "required": sorted(_POSE_MOVE_FIELDS),
            "additionalProperties": False,
        }

    def compact_pose_move_branch(option: Mapping[str, object]) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "kind": {"const": "move"},
                "observation_token": token,
                "phase": {"enum": list(PHASES)},
                "candidate_id": {"const": "source_plus_residual"},
                "residual_id": {"const": option["residual_id"]},
                "gripper": {"enum": ["open", "hold", "close"]},
                "note": note,
            },
            "required": sorted(_COMPACT_POSE_MOVE_FIELDS),
            "additionalProperties": False,
        }

    source_candidates = source_candidates or {}
    unknown = set(source_candidates) - {"source_full"}
    if unknown:
        raise ValueError("Cycle 7 source candidate vocabulary drifted")
    move_branches = (
        [
            (
                compact_pose_move_branch(option)
                if compact_pose
                else pose_move_branch(source_candidates["source_full"], option)
            )
            for option in lateral_residual_bank(
                source_candidates["source_full"]["translation_m"]
            )
        ]
        if "source_full" in source_candidates
        else [custom_move_branch()]
    )
    return {
        "type": "object",
        "oneOf": [
            *move_branches,
            *[
                {
                    "type": "object",
                    "properties": {
                        "kind": {"const": kind},
                        "observation_token": token,
                        "note": note,
                    },
                    "required": sorted(_TERMINAL_FIELDS),
                    "additionalProperties": False,
                }
                for kind in ("give_up", "done")
            ],
        ],
    }


def expand_compact_pose_command(
    value: Mapping[str, object], *, source: Mapping[str, object]
) -> dict[str, object]:
    """Reconstruct the exact D1 command selected by one compact branch ID."""
    if set(value) != _COMPACT_POSE_MOVE_FIELDS:
        raise ValueError("compact pose command fields drifted")
    if value.get("kind") != "move" or value.get("candidate_id") != "source_plus_residual":
        raise ValueError("compact pose command is not a source branch")
    residual_id = value.get("residual_id")
    option = next(
        (
            item
            for item in lateral_residual_bank(source.get("translation_m"))
            if item["residual_id"] == residual_id
        ),
        None,
    )
    if option is None:
        raise ValueError("compact pose residual is outside the frozen bank")
    rotation = np.asarray(source.get("rotation_axis_angle_rad"), dtype=np.float64)
    if rotation.shape != (3,) or not np.isfinite(rotation).all():
        raise ValueError("compact pose source rotation is invalid")
    return {
        **dict(value),
        "translation_m_f64_repr": [
            repr(float(item)) for item in option["translation_m"]
        ],
        "lateral_residual_m_f64_repr": [
            repr(float(item)) for item in option["lateral_residual_m"]
        ],
        "rotation_axis_angle_rad_f64_repr": [repr(float(item)) for item in rotation],
    }


def _vector(value: object, *, name: str, component_max: float) -> tuple[float, float, float]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite values")
    if bool(np.any(np.abs(vector) > component_max)):
        raise ValueError(f"{name} exceeds its component bound")
    return tuple(float(item) for item in vector)


def _repr_vector(
    value: object, *, name: str, component_max: float
) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3 or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"{name} must contain three canonical repr strings")
    try:
        decoded = tuple(float(item) for item in value)
    except ValueError as error:
        raise ValueError(f"{name} contains an invalid float repr") from error
    if not all(math.isfinite(item) for item in decoded):
        raise ValueError(f"{name} contains a nonfinite float repr")
    if [repr(item) for item in decoded] != value:
        raise ValueError(f"{name} must use canonical repr strings")
    if any(abs(item) > component_max for item in decoded):
        raise ValueError(f"{name} exceeds its component bound")
    return decoded


@dataclass(frozen=True)
class Cycle5Decision:
    kind: str
    note: str
    phase: str = ""
    candidate_id: str = "custom"
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lateral_residual_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    residual_id: str = ""
    rotation_axis_angle_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper: str = "hold"


def decode_cycle5_decision(
    value: Mapping[str, object], *, observation_token: str
) -> Cycle5Decision:
    kind = value.get("kind")
    note = value.get("note")
    if not isinstance(note, str) or NOTE_PATTERN.fullmatch(note) is None:
        raise ValueError("Cycle 5 note is invalid")
    if value.get("observation_token") != observation_token:
        raise ValueError("Cycle 5 observation token is stale")
    if kind in {"give_up", "done"}:
        if set(value) != _TERMINAL_FIELDS:
            raise ValueError("Cycle 5 terminal fields drifted")
        return Cycle5Decision(kind=str(kind), note=note)
    candidate_id = value.get("candidate_id")
    expected_fields = (
        _POSE_MOVE_FIELDS
        if candidate_id == "source_plus_residual"
        else _MOVE_FIELDS
    )
    if kind != "move" or set(value) != expected_fields:
        raise ValueError("Cycle 5 move fields drifted")
    phase = value.get("phase")
    if phase not in PHASES:
        raise ValueError("Cycle 5 phase is outside the reviewed vocabulary")
    gripper = value.get("gripper")
    if gripper not in {"open", "hold", "close"}:
        raise ValueError("Cycle 5 gripper is invalid")
    if candidate_id not in {"source_plus_residual", "custom"}:
        raise ValueError("Cycle 7 candidate ID is invalid")
    return Cycle5Decision(
        kind="move",
        note=note,
        phase=str(phase),
        candidate_id=str(candidate_id),
        translation_m=(
            _repr_vector(
                value.get("translation_m_f64_repr"),
                name="translation",
                component_max=TRANSLATION_COMPONENT_MAX_M,
            )
            if candidate_id == "source_plus_residual"
            else _vector(
                value.get("translation_m"),
                name="translation",
                component_max=TRANSLATION_COMPONENT_MAX_M,
            )
        ),
        lateral_residual_m=(
            _repr_vector(
                value.get("lateral_residual_m_f64_repr"),
                name="lateral residual",
                component_max=0.010,
            )
            if candidate_id == "source_plus_residual"
            else (0.0, 0.0, 0.0)
        ),
        residual_id=(
            str(value.get("residual_id"))
            if candidate_id == "source_plus_residual"
            else ""
        ),
        rotation_axis_angle_rad=(
            _repr_vector(
                value.get("rotation_axis_angle_rad_f64_repr"),
                name="rotation",
                component_max=ROTATION_COMPONENT_MAX_RAD,
            )
            if candidate_id == "source_plus_residual"
            else _vector(
                value.get("rotation_axis_angle_rad"),
                name="rotation",
                component_max=ROTATION_COMPONENT_MAX_RAD,
            )
        ),
        gripper=str(gripper),
    )


def macro_substeps(
    decision: Cycle5Decision, *, previous_gripper: str
) -> tuple[list[dict[str, np.ndarray]], str]:
    if decision.kind != "move":
        raise ValueError("only Cycle 5 moves have controller substeps")
    translation = np.asarray(decision.translation_m, dtype=np.float64)
    rotation = np.asarray(decision.rotation_axis_angle_rad, dtype=np.float64)
    count = max(
        1,
        math.ceil(float(np.linalg.norm(translation)) / 0.005),
        math.ceil(float(np.linalg.norm(rotation)) / 0.025),
    )
    gripper = previous_gripper if decision.gripper == "hold" else decision.gripper
    close = 1.0 if gripper == "close" else 0.0
    action = {
        "action.end_effector_position": translation
        * TRANSLATION_CONTROLLER_GAIN
        / (count * OSC_TRANSLATION_PER_STEP_M),
        "action.end_effector_rotation": rotation
        * ROTATION_CONTROLLER_GAIN
        / (count * OSC_ROTATION_PER_STEP_RAD),
        "action.gripper_close": np.asarray([close]),
        "action.base_motion": np.zeros(4),
        "action.control_mode": np.asarray([0.0]),
    }
    return (
        [{key: value.copy() for key, value in action.items()} for _ in range(count)],
        gripper,
    )


def validate_source_plus_residual(
    decision: Cycle5Decision,
    *,
    source: Mapping[str, object],
) -> str:
    if decision.candidate_id != "source_plus_residual":
        raise ValueError("decision is not source_plus_residual")
    source_translation = np.asarray(source["translation_m"], dtype=np.float64)
    source_rotation = np.asarray(source["rotation_axis_angle_rad"], dtype=np.float64)
    residual = np.asarray(decision.lateral_residual_m, dtype=np.float64)
    final_translation = np.asarray(decision.translation_m, dtype=np.float64)
    final_rotation = np.asarray(decision.rotation_axis_angle_rad, dtype=np.float64)
    if float(np.linalg.norm(residual)) > 0.010 + 1e-12:
        raise ValueError("source plus residual exceeds residual norm cap")
    if np.max(np.abs(final_translation - source_translation - residual)) > 1e-12:
        raise ValueError("source plus residual binding drifted")
    if np.max(np.abs(final_rotation - source_rotation)) > 1e-12:
        raise ValueError("source plus residual rotation drifted")
    residual_norm = float(np.linalg.norm(residual))
    source_norm = float(np.linalg.norm(source_translation))
    if residual_norm > 1e-12:
        if source_norm <= 1e-12:
            raise ValueError("lateral residual has no source tangent")
        cosine = abs(float(np.dot(residual, source_translation))) / (
            residual_norm * source_norm
        )
        if cosine > 0.25 + 1e-12:
            raise ValueError("lateral residual is not lateral to source")
    for option in lateral_residual_bank(source_translation):
        if max(
            float(
                np.max(
                    np.abs(
                        np.asarray(option["lateral_residual_m"], dtype=np.float64)
                        - residual
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        np.asarray(option["translation_m"], dtype=np.float64)
                        - final_translation
                    )
                )
            ),
        ) <= 1e-12 and decision.residual_id == option["residual_id"]:
            return str(option["residual_id"])
    raise ValueError("source plus residual is outside the frozen const bank")


def materially_contributes(
    decision: Cycle5Decision,
    source: Mapping[str, object] | None,
    *,
    suggested_gripper: str,
    resolved_gripper: str | None = None,
) -> bool:
    if decision.kind != "move":
        return False
    actual_gripper = (
        suggested_gripper
        if resolved_gripper is None and decision.gripper == "hold"
        else decision.gripper if resolved_gripper is None else resolved_gripper
    )
    if actual_gripper != suggested_gripper:
        return True
    if source is None:
        return True
    return bool(np.linalg.norm(decision.lateral_residual_m) >= 0.003 - 1e-12)


def stall_governor_active(consecutive_stalls: int) -> bool:
    """The third consecutive stalled macro terminates before another model call."""
    if consecutive_stalls < 0:
        raise ValueError("consecutive stalls cannot be negative")
    return consecutive_stalls >= 3
