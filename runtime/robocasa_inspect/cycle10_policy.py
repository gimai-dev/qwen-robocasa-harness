"""Closed Cycle 10 Qwen schema, persistent residual, and public source panels."""

from __future__ import annotations

import io
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .goal30_cycle10 import RESIDUAL_IDS

PHASES = ("align", "grasp", "manipulate", "release", "advance", "give_up")
GRIPPERS = ("source", "open", "close")
NOTE_PATTERN = re.compile(r"[A-Za-z0-9 .,;:!?()'/_+\-]{1,120}")

CYCLE10_SYSTEM_PROMPT = r"""You are an Inspect-style visual robot policy for the
official RoboCasa PandaOmron simulation. You receive exactly three public images.
Image 1 is official LEFT with CURRENT on the left half and SOURCE REFERENCE on the
right half. Image 2 has the same layout for official RIGHT. Image 3 is the
unmodified CURRENT official wrist camera. SOURCE pixels are a motion prior from a
retrieved same-task successful human demonstration, not current state, reward,
success, depth, or an action.

Return exactly one schema object. residual_id selects one closed base-frame skill:
tx/ty/tz change persistent X/Y/Z translation; rx/ry/rz change persistent
axis-angle rotation. + and - are robot-base signs and the suffix is millimeters or
milliradians. The residual persists across every later source keyframe and is
shown in public state. zero preserves it. gripper source follows the source
keyframe; open or close deliberately overrides it. phase describes the visible
intent. advance is only valid after the harness reports public EEF error within
5 mm and 25 mrad. give_up is allowed when bounded safe progress is blocked.

Compare CURRENT against SOURCE in both external views and use the wrist view for
near-contact evidence. Prefer zero when aligned. Use one small residual when the
same task object or handle is visibly offset, then re-observe before another.
Never output coordinates, pixels, joint values, done, verify, reward, task
predicate, hidden contact, private simulator state, or free-form tools. The
harness only executes the selected closed skill and bounded servo ticks."""


@dataclass(frozen=True)
class Cycle10Decision:
    observation_token: str
    residual_id: str
    gripper: str
    phase: str
    note: str


def cycle10_response_schema(observation_token: str) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{12}", observation_token) is None:
        raise ValueError("Cycle 10 observation token must be twelve lowercase hex")
    fields = {
        "observation_token": {"const": observation_token},
        "residual_id": {"enum": list(RESIDUAL_IDS)},
        "gripper": {"enum": list(GRIPPERS)},
        "phase": {"enum": list(PHASES)},
        "note": {
            "type": "string",
            "minLength": 1,
            "maxLength": 120,
            "pattern": "^[A-Za-z0-9 .,;:!?()'/_+\\-]{1,120}$",
        },
    }
    return {
        "type": "object",
        "properties": fields,
        "required": sorted(fields),
        "additionalProperties": False,
    }


def decode_cycle10_decision(
    value: Mapping[str, object], *, observation_token: str
) -> Cycle10Decision:
    required = {"observation_token", "residual_id", "gripper", "phase", "note"}
    if set(value) != required:
        raise ValueError("Cycle 10 decision fields drifted")
    if value.get("observation_token") != observation_token:
        raise ValueError("Cycle 10 observation token is stale")
    residual_id = value.get("residual_id")
    gripper = value.get("gripper")
    phase = value.get("phase")
    note = value.get("note")
    if residual_id not in RESIDUAL_IDS:
        raise ValueError("Cycle 10 residual is outside the closed vocabulary")
    if gripper not in GRIPPERS:
        raise ValueError("Cycle 10 gripper is outside the closed vocabulary")
    if phase not in PHASES:
        raise ValueError("Cycle 10 phase is outside the closed vocabulary")
    if not isinstance(note, str) or NOTE_PATTERN.fullmatch(note) is None:
        raise ValueError("Cycle 10 note is invalid")
    return Cycle10Decision(
        observation_token=observation_token,
        residual_id=str(residual_id),
        gripper=str(gripper),
        phase=str(phase),
        note=note,
    )


def residual_increment(residual_id: str) -> tuple[np.ndarray, np.ndarray]:
    if residual_id == "zero":
        return np.zeros(3), np.zeros(3)
    if residual_id not in RESIDUAL_IDS or len(residual_id) < 5:
        raise ValueError("Cycle 10 residual is outside the closed vocabulary")
    kind, axis, sign = residual_id[0], residual_id[1], residual_id[2]
    magnitude = int(residual_id[3:]) / 1000.0
    direction = 1.0 if sign == "+" else -1.0
    index = "xyz".index(axis)
    translation = np.zeros(3)
    rotation = np.zeros(3)
    (translation if kind == "t" else rotation)[index] = direction * magnitude
    return translation, rotation


def accumulate_residual(
    translation: Sequence[float], rotation: Sequence[float], residual_id: str
) -> tuple[np.ndarray, np.ndarray]:
    current_translation = np.asarray(translation, dtype=np.float64)
    current_rotation = np.asarray(rotation, dtype=np.float64)
    if (
        current_translation.shape != (3,)
        or current_rotation.shape != (3,)
        or not np.isfinite(current_translation).all()
        or not np.isfinite(current_rotation).all()
    ):
        raise ValueError("Cycle 10 persistent residual is invalid")
    delta_translation, delta_rotation = residual_increment(residual_id)

    def clamp(value: np.ndarray, maximum: float) -> np.ndarray:
        norm = float(np.linalg.norm(value))
        return value if norm <= maximum else value * (maximum / norm)

    return (
        clamp(current_translation + delta_translation, 0.050),
        clamp(current_rotation + delta_rotation, 0.20),
    )


def _axis_angle_quaternion(rotation: Sequence[float]) -> np.ndarray:
    value = np.asarray(rotation, dtype=np.float64)
    angle = float(np.linalg.norm(value))
    if angle <= 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0])
    return np.concatenate(
        (value * (math.sin(angle / 2.0) / angle), [math.cos(angle / 2.0)])
    )


def _quaternion_multiply(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    x1, y1, z1, w1 = np.asarray(left, dtype=np.float64)
    x2, y2, z2, w2 = np.asarray(right, dtype=np.float64)
    value = np.asarray(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )
    return value / np.linalg.norm(value)


def source_target_with_residual(
    source_state: Sequence[float],
    translation_residual: Sequence[float],
    rotation_residual: Sequence[float],
) -> np.ndarray:
    target = np.asarray(source_state, dtype=np.float64).copy()
    if target.shape != (16,) or not np.isfinite(target).all():
        raise ValueError("Cycle 10 source target is invalid")
    translation = np.asarray(translation_residual, dtype=np.float64)
    rotation = np.asarray(rotation_residual, dtype=np.float64)
    if translation.shape != (3,) or rotation.shape != (3,):
        raise ValueError("Cycle 10 target residual is invalid")
    target[7:10] += translation
    target[10:14] = _quaternion_multiply(
        _axis_angle_quaternion(rotation), target[10:14]
    )
    return target


def current_reference_panel_png(
    current: bytes, reference: object, *, camera: str, frame_index: int
) -> bytes:
    current_image = Image.open(io.BytesIO(current)).convert("RGB")
    reference_image = Image.fromarray(np.asarray(reference, dtype=np.uint8), mode="RGB")
    if current_image.size != (256, 256) or reference_image.size != (256, 256):
        raise ValueError("Cycle 10 panels require official 256x256 RGB")
    canvas = Image.new("RGB", (512, 286), "white")
    canvas.paste(current_image, (0, 30))
    canvas.paste(reference_image, (256, 30))
    ImageDraw.Draw(canvas).text(
        (4, 7),
        f"{camera.upper()} CURRENT | SOURCE REFERENCE frame {frame_index}",
        fill="red",
    )
    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=False)
    return output.getvalue()
