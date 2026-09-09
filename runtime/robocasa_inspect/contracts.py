"""Closed observation and action contracts for the official RoboCasa wrapper."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

CAMERAS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
STATE_KEYS = (
    ("state.end_effector_position_relative", 3),
    ("state.end_effector_rotation_relative", 4),
    ("state.base_position", 3),
    ("state.base_rotation", 4),
    ("state.gripper_qpos", 2),
)
ACTION_FIELDS = {
    "kind",
    "observation_id",
    "note",
    "translation_m",
    "rotation_axis_angle_rad",
    "gripper",
}
NO_ACTION_FIELDS = {"kind", "observation_id", "note"}
SERVO_FIELDS = {
    "kind",
    "observation_id",
    "note",
    "phase",
    "left",
    "right",
    "gripper",
}
FEATURE_FIELDS = {"target_uv", "gripper_uv"}
BASE_ACTION_FIELDS = {
    "kind",
    "observation_id",
    "note",
    "axis",
    "normalized_velocity",
    "gripper",
}
SERVO_PHASES = {"observe", "approach", "contact", "manipulate", "retreat"}
CHUNK_STEPS = 5
OSC_TRANSLATION_PER_STEP_M = 0.05
OSC_ROTATION_PER_STEP_RAD = 0.5


@dataclass(frozen=True)
class PublicObservation:
    observation_id: str
    instruction: str
    images: dict[str, np.ndarray]
    state_groups: dict[str, list[float]]


@dataclass(frozen=True)
class FeaturePixels:
    target_uv: tuple[float, float]
    gripper_uv: tuple[float, float]


@dataclass(frozen=True)
class Command:
    kind: str
    observation_id: str
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_axis_angle_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper: str = "hold"
    note: str = ""
    phase: str = ""
    left: FeaturePixels | None = None
    right: FeaturePixels | None = None
    base_axis: str = ""
    base_velocity: float = 0.0


def project_observation(
    raw: Mapping[str, object], *, episode: str, sequence: int
) -> PublicObservation:
    if sequence < 0 or not episode:
        raise ValueError("invalid observation provenance")
    images: dict[str, np.ndarray] = {}
    image_hashes: dict[str, str] = {}
    for key in CAMERAS:
        image = np.asarray(raw[key])
        if image.shape != (256, 256, 3) or image.dtype != np.uint8:
            raise ValueError(f"camera contract drift: {key}")
        images[key] = image.copy()
        image_hashes[key] = hashlib.sha256(image.tobytes()).hexdigest()
    groups: dict[str, list[float]] = {}
    for key, width in STATE_KEYS:
        value = np.asarray(raw[key], dtype=np.float64).reshape(-1)
        if value.shape != (width,) or not np.isfinite(value).all():
            raise ValueError(f"state contract drift: {key}")
        groups[key] = value.tolist()
    instruction = raw.get("annotation.human.task_description")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("task instruction is missing")
    identity = {
        "episode": episode,
        "sequence": sequence,
        "instruction": instruction,
        "images": image_hashes,
        "state": groups,
    }
    observation_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return PublicObservation(observation_id, instruction, images, groups)


def _vector(value: object, *, name: str, limit: float) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain three finite values")
    if np.any(np.abs(array) > limit + 1e-12):
        raise ValueError(f"{name} exceeds its bound")
    return tuple(float(item) for item in array)


def _normalized_uv(value: object, *, name: str) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain two finite normalized coordinates")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must remain in the normalized image")
    return float(array[0]), float(array[1])


def _feature_pixels(value: object, *, view: str) -> FeaturePixels:
    if not isinstance(value, Mapping) or set(value) != FEATURE_FIELDS:
        raise ValueError(f"{view} feature fields do not match the closed schema")
    return FeaturePixels(
        target_uv=_normalized_uv(value.get("target_uv"), name=f"{view} target_uv"),
        gripper_uv=_normalized_uv(value.get("gripper_uv"), name=f"{view} gripper_uv"),
    )


def decode_command(value: Mapping[str, object], *, observation_id: str) -> Command:
    kind = value.get("kind")
    note = value.get("note")
    if not isinstance(note, str) or not note.strip() or len(note) > 320:
        raise ValueError("invalid command note")
    if kind == "servo_feature":
        if set(value) != SERVO_FIELDS:
            raise ValueError("servo feature fields do not match the closed schema")
        if value.get("observation_id") != observation_id:
            raise ValueError("stale observation citation")
        phase = value.get("phase")
        if phase not in SERVO_PHASES:
            raise ValueError("invalid servo phase")
        gripper = value.get("gripper")
        if gripper not in {"open", "hold", "close"}:
            raise ValueError("invalid gripper command")
        return Command(
            kind="servo_feature",
            observation_id=observation_id,
            note=note,
            phase=str(phase),
            left=_feature_pixels(value.get("left"), view="left"),
            right=_feature_pixels(value.get("right"), view="right"),
            gripper=str(gripper),
        )
    if kind == "action":
        if set(value) != ACTION_FIELDS:
            raise ValueError("action fields do not match the closed schema")
        if value.get("observation_id") != observation_id:
            raise ValueError("stale observation citation")
        gripper = value.get("gripper")
        if gripper not in {"open", "hold", "close"}:
            raise ValueError("invalid gripper command")
        return Command(
            kind="action",
            observation_id=observation_id,
            note=note,
            translation_m=_vector(
                value.get("translation_m"), name="translation", limit=0.02
            ),
            rotation_axis_angle_rad=_vector(
                value.get("rotation_axis_angle_rad"), name="rotation", limit=0.0873
            ),
            gripper=str(gripper),
        )
    if kind == "base_action":
        if set(value) != BASE_ACTION_FIELDS:
            raise ValueError("base action fields do not match the closed schema")
        if value.get("observation_id") != observation_id:
            raise ValueError("stale observation citation")
        axis = value.get("axis")
        if axis not in {"x", "y", "yaw"}:
            raise ValueError("invalid base axis")
        raw_velocity = value.get("normalized_velocity")
        if isinstance(raw_velocity, bool):
            raise ValueError("base velocity exceeds its bound")
        velocity = float(raw_velocity)
        if (
            not np.isfinite(velocity)
            or abs(velocity) <= 1e-12
            or abs(velocity) > 0.25 + 1e-12
        ):
            raise ValueError("base velocity exceeds its bound")
        gripper = value.get("gripper")
        if gripper not in {"open", "hold", "close"}:
            raise ValueError("invalid gripper command")
        return Command(
            kind="base_action",
            observation_id=observation_id,
            note=note,
            gripper=str(gripper),
            base_axis=str(axis),
            base_velocity=velocity,
        )
    if kind in {"finish", "give_up"}:
        if set(value) != NO_ACTION_FIELDS:
            raise ValueError("non-action fields do not match the closed schema")
        if value.get("observation_id") != observation_id:
            raise ValueError("stale observation citation")
        return Command(kind=str(kind), observation_id=observation_id, note=note)
    raise ValueError("unsupported command kind")


def adapt_bounded_action(
    value: Mapping[str, object], *, observation_id: str
) -> tuple[Command, list[str]]:
    """Normalize common action syntax while preserving every safety boundary."""
    changes: list[str] = []
    candidate: dict[str, object]
    if set(value) == {"action"} and isinstance(value.get("action"), Mapping):
        candidate = dict(value["action"])
        changes.append("unwrapped_action_envelope")
    else:
        candidate = dict(value)
    if "kind" not in candidate:
        candidate["kind"] = "action"
        changes.append("inferred_action_kind")
    if set(candidate) != ACTION_FIELDS or candidate.get("kind") != "action":
        raise ValueError("action adapter fields do not match the closed schema")
    if candidate.get("observation_id") != observation_id:
        raise ValueError("stale observation citation")
    note = candidate.get("note")
    if not isinstance(note, str) or not note.strip():
        raise ValueError("invalid command note")
    if len(note) > 320:
        candidate["note"] = note[:320]
        changes.append("truncated_note")
    if candidate.get("gripper") not in {"open", "hold", "close"}:
        raise ValueError("invalid gripper command")
    for field, limit, label in (
        ("translation_m", 0.02, "translation"),
        ("rotation_axis_angle_rad", 0.0873, "rotation"),
    ):
        vector = np.asarray(candidate.get(field), dtype=np.float64)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"{label} must contain three finite values")
        clipped = np.clip(vector, -limit, limit)
        if not np.array_equal(vector, clipped):
            candidate[field] = clipped.tolist()
            changes.append(f"clipped_{label}")
    return decode_command(candidate, observation_id=observation_id), changes


def is_zero_motion(command: Command, *, previous_gripper: str) -> bool:
    return (
        command.kind == "action"
        and max(map(abs, command.translation_m + command.rotation_axis_angle_rad))
        <= 1e-12
        and command.gripper in {"hold", previous_gripper}
    )


def official_action_chunk(
    command: Command, *, previous_gripper: str
) -> tuple[list[dict[str, np.ndarray]], str]:
    if command.kind != "action":
        raise ValueError("only actions produce a simulator chunk")
    gripper = previous_gripper if command.gripper == "hold" else command.gripper
    close = 1.0 if gripper == "close" else 0.0
    translation = np.asarray(command.translation_m, dtype=np.float64) / (
        OSC_TRANSLATION_PER_STEP_M * CHUNK_STEPS
    )
    rotation = np.asarray(command.rotation_axis_angle_rad, dtype=np.float64) / (
        OSC_ROTATION_PER_STEP_RAD * CHUNK_STEPS
    )
    action = {
        "action.end_effector_position": translation,
        "action.end_effector_rotation": rotation,
        "action.gripper_close": np.asarray([close]),
        "action.base_motion": np.zeros(4),
        "action.control_mode": np.asarray([0.0]),
    }
    return (
        [
            {key: value.copy() for key, value in action.items()}
            for _ in range(CHUNK_STEPS)
        ],
        gripper,
    )
