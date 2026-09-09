"""Closed Qwen-selected source-pose skills for Cycle 4."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .contracts import Command
from .visual_waypoint import SourcePoseSuggestion

_ACTION_FIELDS = {
    "kind",
    "observation_token",
    "skill_id",
    "note",
    "translation_m",
    "rotation_axis_angle_rad",
    "lateral_residual_m",
    "gripper",
}
_STOP_FIELDS = {"kind", "observation_token", "note"}


def observation_token(observation_id: str, contract_sha256: str) -> str:
    if len(observation_id) != 64 or len(contract_sha256) != 64:
        raise ValueError("Cycle 4 freshness authorities must be SHA256 values")
    return hashlib.sha256((observation_id + contract_sha256).encode()).hexdigest()[:12]


def _rounded(vector: object) -> tuple[float, float, float]:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("candidate vector must contain three finite values")
    return tuple(float(item) for item in np.round(value, 6))


def exact_source_candidates(
    suggestion: SourcePoseSuggestion,
) -> dict[str, dict[str, tuple[float, float, float]]]:
    translation = np.asarray(suggestion.translation_m)
    rotation = np.asarray(suggestion.rotation_axis_angle_rad)
    return {
        "source_full": {
            "translation_m": _rounded(translation),
            "rotation_axis_angle_rad": _rounded(rotation),
        },
        "source_half": {
            "translation_m": _rounded(0.5 * translation),
            "rotation_axis_angle_rad": _rounded(0.5 * rotation),
        },
        "translation_only": {
            "translation_m": _rounded(translation),
            "rotation_axis_angle_rad": (0.0, 0.0, 0.0),
        },
        "rotation_only": {
            "translation_m": (0.0, 0.0, 0.0),
            "rotation_axis_angle_rad": _rounded(rotation),
        },
    }


@dataclass(frozen=True)
class SkillSelection:
    command: Command
    skill_id: str
    lateral_residual_m: tuple[float, float, float]


def decode_skill_selection(
    value: Mapping[str, object],
    *,
    token: str,
    observation_id: str,
    candidates: Mapping[str, Mapping[str, object]],
    allowed_grippers: set[str],
    lateral_residual_cap_m: float,
    lateral_cosine_abs_max: float,
    source_translation_floor_m: float,
    final_component_cap_m: float,
) -> SkillSelection | None:
    kind = value.get("kind")
    note = value.get("note")
    if not isinstance(note, str) or not note.strip() or len(note) > 320:
        raise ValueError("invalid Cycle 4 command note")
    if value.get("observation_token") != token:
        raise ValueError("stale Cycle 4 observation token")
    if kind == "give_up":
        if set(value) != _STOP_FIELDS:
            raise ValueError("Cycle 4 give_up fields drifted")
        return None
    if kind != "candidate_action" or set(value) != _ACTION_FIELDS:
        raise ValueError("Cycle 4 action fields drifted")
    skill_id = value.get("skill_id")
    if skill_id not in {*candidates, "visual_lateral"}:
        raise ValueError("unsupported Cycle 4 skill candidate")
    gripper = value.get("gripper")
    if gripper not in allowed_grippers:
        raise ValueError("Cycle 4 gripper is outside the current/source binary choices")
    translation = _rounded(value.get("translation_m"))
    rotation = _rounded(value.get("rotation_axis_angle_rad"))
    residual = _rounded(value.get("lateral_residual_m"))
    if skill_id in candidates:
        expected = candidates[str(skill_id)]
        if translation != tuple(expected["translation_m"]) or rotation != tuple(
            expected["rotation_axis_angle_rad"]
        ):
            raise ValueError("Qwen output does not equal the selected exact candidate")
        if residual != (0.0, 0.0, 0.0):
            raise ValueError("exact candidate cannot contain a lateral residual")
    else:
        source_half = candidates["source_half"]
        base_translation = np.asarray(source_half["translation_m"], dtype=np.float64)
        source_translation = np.asarray(
            candidates["source_full"]["translation_m"], dtype=np.float64
        )
        residual_array = np.asarray(residual)
        if np.linalg.norm(residual_array) > lateral_residual_cap_m + 1e-12:
            raise ValueError("visual lateral residual exceeds its norm bound")
        source_norm = float(np.linalg.norm(source_translation))
        residual_norm = float(np.linalg.norm(residual_array))
        if (
            source_norm >= source_translation_floor_m
            and residual_norm > 1e-12
            and abs(float(np.dot(residual_array, source_translation)))
            > lateral_cosine_abs_max * residual_norm * source_norm + 1e-12
        ):
            raise ValueError("visual residual is not lateral to the source tangent")
        expected_translation = _rounded(base_translation + residual_array)
        if translation != expected_translation:
            raise ValueError("visual lateral final action does not bind its residual")
        if rotation not in {
            tuple(source_half["rotation_axis_angle_rad"]),
            (0.0, 0.0, 0.0),
        }:
            raise ValueError("visual lateral rotation is outside its closed choices")
    if max(map(abs, translation)) > final_component_cap_m + 1e-12:
        raise ValueError("Cycle 4 final translation exceeds its component bound")
    return SkillSelection(
        command=Command(
            kind="action",
            observation_id=observation_id,
            note=note,
            translation_m=translation,
            rotation_axis_angle_rad=rotation,
            gripper=str(gripper),
        ),
        skill_id=str(skill_id),
        lateral_residual_m=residual,
    )


def cursor_progressed(
    *,
    before_error_m: float,
    after_error_m: float,
    after_rotation_error_rad: float,
    minimum_progress_m: float,
    arrival_position_max_m: float,
    arrival_rotation_max_rad: float,
) -> bool:
    if not all(
        np.isfinite(value)
        for value in (before_error_m, after_error_m, after_rotation_error_rad)
    ):
        raise ValueError("cursor progress evidence must be finite")
    return (
        after_error_m <= before_error_m - minimum_progress_m
        and after_error_m <= arrival_position_max_m
        and after_rotation_error_rad <= arrival_rotation_max_rad
    )


def expected_contact(
    *,
    affordance: str,
    selected_index: int,
    trajectory_length: int,
    target_closed: bool,
    previous_closed: bool,
) -> bool:
    if not 0 <= selected_index < trajectory_length or trajectory_length <= 0:
        raise ValueError("contact phase cursor is invalid")
    if affordance == "fixture_manipulation_contact":
        return selected_index / trajectory_length >= 0.02
    if affordance in {
        "object_grasp_contact",
        "object_or_fixture_contact_by_instruction",
    }:
        return target_closed or previous_closed
    if affordance == "no_contact_affordance":
        return False
    raise ValueError("unknown full-catalog contact affordance")
