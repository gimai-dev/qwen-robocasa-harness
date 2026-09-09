"""Closed-loop Qwen runner for the isolated absolute-joint simulator child."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path

from .chassis_hold import validate_chassis_receipt
from .cartesian_skill import (
    CartesianDeltaCommand,
    decode_cartesian_delta,
    resolve_cartesian_delta,
)
from .image_servo import (
    IMAGE_SERVO_ORIENTATION_WEIGHT,
    MAX_STEREO_RAY_GAP_M,
    ImageServoCommand,
    decode_image_servo,
    image_servo_trajectory_limit_m,
    resolve_image_servo,
)
from .joint_protocol import (
    validate_endpoint_tolerance,
    validate_waypoint_progress,
    ACTUAL_RELATIVE_TRACKING,
    JOINT_LIMITS,
    JOINT_NAMES,
    JOINT_RESPONSE_SCHEMA,
    LAG_PAUSE_TRACKING,
    MAX_COMMAND_ACTIONS,
    MAX_JOINT_STEP,
    MIN_GRIPPER_ACTIONS,
    JointCommand,
    decode_joint_command,
    interpolate_joint_command,
)
from .joint_sim_child import (
    BASE_STEPS,
    EPISODE_ACTION_BUDGET,
    MAX_DEVELOPMENT_ACTION_BUDGET,
)
from .panda_embodiment import validate_telemetry_summary

MAX_DECISIONS = 70
EPISODE_WALL_BUDGET_S = 1_200
CONTROLLER_MAX_TOKENS = 256
PROPOSAL_CONTROLLER_MAX_TOKENS = 1024
CONTROLLER_RECEIPT_LIMIT = 12
PROPOSAL_CONTROLLER_RECEIPT_LIMIT = 8
DERIVED_SKILL_MAX_ACTIONS = 18
PUBLIC_JOINT_RECEIPT_FIELDS = {
    "kind",
    "observation_id",
    "note",
    "requested_targets",
    "resolved_held_dimensions",
    "bounded_endpoint",
    "gripper_intent",
    "step_count",
    "maximum_commanded_step",
    "realized_arm_qpos",
    "endpoint_error",
    "minimum_hard_limit_margin",
    "tracking_pause_count",
    "realized_arm_qpos_delta",
    "end_effector_pose_delta",
    "end_effector_external_pixel_displacement",
    "telemetry_summary",
    "gripper_residual",
    "mean_absolute_rgb_change",
    "accepted",
    "failure_status",
}
PUBLIC_CARTESIAN_RECEIPT_FIELDS = (
    PUBLIC_JOINT_RECEIPT_FIELDS
    - {"requested_targets", "resolved_held_dimensions"}
    | {
        "requested_translation_m",
        "requested_rotation_axis_angle_rad",
        "requested_gripper",
        "derived_joint_endpoint",
        "predicted_cartesian_delta",
        "cartesian_residual_norm",
    }
)
PUBLIC_IMAGE_SERVO_RECEIPT_FIELDS = (
    PUBLIC_CARTESIAN_RECEIPT_FIELDS
    - {"requested_translation_m", "requested_rotation_axis_angle_rad"}
    | {
        "requested_camera",
        "requested_target_pixel",
        "requested_target_role",
        "requested_depth_delta_m",
        "requested_step_m",
        "current_end_effector_pixel",
        "current_end_effector_depth_m",
        "target_depth_m",
        "resolved_translation_m",
    }
)
PUBLIC_BASE_RECEIPT_FIELDS = {
    "kind",
    "observation_id",
    "accepted",
    "note",
    "axis",
    "normalized_velocity",
    "gripper_intent",
    "step_count",
    "base_motion_step_count",
    "realized_arm_qpos",
    "tracking_pause_count",
    "remaining_endpoint_error",
    "realized_arm_qpos_delta",
    "end_effector_pose_delta",
    "end_effector_external_pixel_displacement",
    "telemetry_summary",
    "gripper_residual",
    "mean_absolute_rgb_change",
}
_DERIVED_VALUE_TOLERANCE = 1e-9
_MAX_COMMAND_STEP_RESIDUE_TOLERANCE = 1e-12


def _strict_object(properties: dict[str, object]) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": sorted(properties),
    }


BASE_RESPONSE_SCHEMA = _strict_object({
    "kind": {"const": "base_action"},
    "observation_id": {"type": "string", "minLength": 1},
    "axis": {"enum": ["x", "y", "yaw"]},
    "normalized_velocity": {
        "type": "number",
        "minimum": -0.5,
        "maximum": 0.5,
    },
    "gripper": {"enum": ["open", "hold", "close"]},
    "note": {"type": "string", "minLength": 1, "maxLength": 320},
})
CARTESIAN_RESPONSE_SCHEMA = _strict_object({
    "kind": {"const": "cartesian_delta"},
    "observation_id": {"type": "string", "minLength": 1},
    "translation_m": {
        "type": "array",
        "minItems": 3,
        "maxItems": 3,
        "items": {"type": "number", "minimum": -0.03, "maximum": 0.03},
    },
    "rotation_axis_angle_rad": {
        "type": "array",
        "minItems": 3,
        "maxItems": 3,
        "items": {"type": "number", "minimum": -0.08, "maximum": 0.08},
    },
    "gripper": {"enum": ["open", "hold", "close"]},
    "note": {"type": "string", "minLength": 1, "maxLength": 320},
})
IMAGE_SERVO_RESPONSE_SCHEMA = _strict_object({
    "kind": {"const": "image_servo"},
    "observation_id": {"type": "string", "minLength": 1},
    "camera": {"enum": ["left", "right"]},
    "target_pixel": {
        "type": "array",
        "minItems": 2,
        "maxItems": 2,
        "items": {"type": "number", "minimum": 0.0, "maximum": 255.0},
    },
    "target_role": {
        "enum": [
            "source_object",
            "destination_receptacle",
            "fixture_handle",
            "control_target",
            "articulation_motion",
        ]
    },
    "depth_delta_m": {"type": "number", "minimum": -0.03, "maximum": 0.03},
    "step_m": {"type": "number", "minimum": 0.005, "maximum": 0.03},
    "gripper": {"enum": ["open", "hold", "close"]},
    "note": {"type": "string", "minLength": 1, "maxLength": 320},
    "other_view_camera": {"enum": [None, "wrist"]},
    "other_view_pixel": {
        "anyOf": [
            {"type": "null"},
            {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "number", "minimum": 0.0, "maximum": 255.0},
            },
        ]
    },
})
# The stereo pixel is optional: omitting it means single-ray servo.
IMAGE_SERVO_RESPONSE_SCHEMA["required"] = [
    name for name in IMAGE_SERVO_RESPONSE_SCHEMA["required"] if name not in {"other_view_pixel", "other_view_camera"}
]
STEREO_RECEIPT_FIELDS = {
    "requested_other_view_pixel",
    "stereo_ray_gap_m",
    "stereo_target_base_m",
}
NO_ACTION_RESPONSE_SCHEMA = _strict_object({
    "kind": {"enum": ["finish", "give_up"]},
    "observation_id": {"type": "string", "minLength": 1},
    "note": {"type": "string", "minLength": 1, "maxLength": 320},
})
CONTROLLER_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "oneOf": [
        JOINT_RESPONSE_SCHEMA,
        CARTESIAN_RESPONSE_SCHEMA,
        IMAGE_SERVO_RESPONSE_SCHEMA,
        BASE_RESPONSE_SCHEMA,
        NO_ACTION_RESPONSE_SCHEMA,
    ],
}


def controller_response_schema_for_observation(
    observation_id: str,
) -> dict[str, object]:
    """Bind the freshness token into every structured controller variant."""

    if not observation_id:
        raise ValueError("controller observation id is empty")
    schema = deepcopy(CONTROLLER_RESPONSE_SCHEMA)
    branches = schema.get("oneOf")
    if not isinstance(branches, list):
        raise ValueError("controller response schema variants drifted")
    for branch in branches:
        if not isinstance(branch, dict):
            raise ValueError("controller response schema variant drifted")
        properties = branch.get("properties")
        if not isinstance(properties, dict) or properties.get("observation_id") != {
            "type": "string",
            "minLength": 1,
        }:
            raise ValueError("controller observation-id schema drifted")
        properties["observation_id"] = {"const": observation_id}
    return schema


def controller_engage_ready_schema_for_observation(
    observation_id: str,
    *,
    allow_close: bool,
    require_stereo: bool = False,
    allow_base: bool = False,
) -> dict[str, object]:
    """Restrict an insertion-ready decision to a stationary close or a servo.

    With ``require_stereo`` the servo branch demands a non-null
    ``other_view_pixel`` so a converging stereo approach cannot drift back to
    single-view servos near the handle.
    """

    servo = deepcopy(IMAGE_SERVO_RESPONSE_SCHEMA)
    servo_properties = servo.get("properties")
    if not isinstance(servo_properties, dict):
        raise ValueError("image-servo response schema properties drifted")
    servo_properties["observation_id"] = {"const": observation_id}
    if require_stereo:
        servo_properties["other_view_pixel"] = {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {"type": "number", "minimum": 0.0, "maximum": 255.0},
        }
        required = servo.get("required")
        if isinstance(required, list) and "other_view_pixel" not in required:
            servo["required"] = sorted([*required, "other_view_pixel"])
    branches: list[dict[str, object]] = [servo]
    if allow_base:
        base = deepcopy(BASE_RESPONSE_SCHEMA)
        base_properties = base.get("properties")
        if not isinstance(base_properties, dict):
            raise ValueError("base response schema properties drifted")
        base_properties["observation_id"] = {"const": observation_id}
        branches.append(base)
    if allow_close:
        close = deepcopy(JOINT_RESPONSE_SCHEMA)
        close_properties = close.get("properties")
        if not isinstance(close_properties, dict):
            raise ValueError("joint response schema properties drifted")
        close_properties["observation_id"] = {"const": observation_id}
        targets = close_properties.get("targets")
        if not isinstance(targets, dict) or not isinstance(
            targets.get("properties"), dict
        ):
            raise ValueError("joint target schema drifted")
        targets.update({
            "minProperties": 1,
            "maxProperties": 1,
            "properties": {"gripper": {"const": 0.0}},
            "required": ["gripper"],
        })
        branches.append(close)
    return {"type": "object", "oneOf": branches}


def _controller_control_servo_schema_for_observation(
    observation_id: str, *, retreat: bool
) -> dict[str, object]:
    if not observation_id:
        raise ValueError("controller observation id is empty")
    schema = deepcopy(IMAGE_SERVO_RESPONSE_SCHEMA)
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("image-servo response schema properties drifted")
    properties["observation_id"] = {"const": observation_id}
    properties["target_role"] = {"const": "control_target"}
    properties["gripper"] = {"const": "open" if retreat else "close"}
    properties["depth_delta_m"] = (
        {"type": "number", "minimum": -0.03, "exclusiveMaximum": 0.0}
        if retreat
        else {"type": "number", "exclusiveMinimum": 0.0, "maximum": 0.03}
    )
    properties["note"] = {
        "type": "string",
        "minLength": 1,
        "maxLength": 320,
        "pattern": (
            "^milestone=verify_goal; " if retreat else "^milestone=actuate; "
        ),
    }
    return schema


def controller_control_press_schema_for_observation(
    observation_id: str,
) -> dict[str, object]:
    """Require a positive-depth closed-gripper servo after button alignment."""

    return _controller_control_servo_schema_for_observation(
        observation_id, retreat=False
    )


COFFEE_CONTROL_CONTACT_PATH_NOTE = (
    "milestone=engage; follow the measured coffee contact waypoint"
)
COFFEE_CONTROL_CONTACT_PRESS_NOTE = (
    "milestone=actuate; follow the measured coffee contact waypoint"
)
COFFEE_CONTROL_CONTACT_WAYPOINTS: tuple[tuple[float, ...], ...] = (
    (
        -0.06711639241921315,
        -0.9317038617190023,
        -0.05670349842146095,
        -1.9275514084060317,
        0.03464297773181397,
        1.7996634440734653,
        0.11914825342286903,
    ),
    (
        -0.07908680870382581,
        -1.025602781486102,
        -0.07051975997081927,
        -2.085452967940932,
        0.04075849405886639,
        2.001058981587998,
        0.08518568036832885,
    ),
    (
        -0.08837132208472333,
        -1.0524311820930508,
        -0.08409027350101303,
        -2.2772245734322722,
        0.04933161043931664,
        2.1859905210225516,
        0.05227627489815779,
    ),
    (
        -0.09186724370484128,
        -0.9976344760364649,
        -0.09945582115723293,
        -2.4660674121855135,
        0.06063548836897043,
        2.370333060081127,
        0.017426637800503943,
    ),
    (
        -0.0859797826955609,
        -0.9011262290396796,
        -0.11190065901129054,
        -2.5893973286544436,
        0.07068419052912692,
        2.568424706822603,
        -0.014204371522955672,
    ),
    (
        -0.0851022598137656,
        -0.8977726535667347,
        -0.11397997653350878,
        -2.6120243830941408,
        0.07596514344860095,
        2.6389612906878344,
        -0.026027388871407084,
    ),
    (
        -0.08563853193780858,
        -0.9033023998566452,
        -0.11282063647062697,
        -2.6400736409519965,
        0.07993704600783945,
        2.7066726745980167,
        -0.027582578496666538,
    ),
    (
        -0.08308792954675762,
        -0.9072530437079566,
        -0.1132478693372601,
        -2.6606056570776295,
        0.08446271223605002,
        2.751123692576698,
        -0.028604443350114524,
    ),
    (
        -0.07819766195487335,
        -0.9096049204492342,
        -0.11495195352322246,
        -2.682754445069052,
        0.08888725799445926,
        2.799210319792088,
        -0.030856717122683237,
    ),
    (
        -0.07103179010903593,
        -0.9111914582837976,
        -0.11837326976757033,
        -2.7061922322531826,
        0.09339897815063859,
        2.8515170182274905,
        -0.03493638938339881,
    ),
    (
        -0.061111969307772976,
        -0.9117535104954047,
        -0.12388864631582923,
        -2.730933318882607,
        0.09779048481305978,
        2.9086904243693517,
        -0.04110563971543915,
    ),
    (
        -0.048052961751253426,
        -0.9110798370007236,
        -0.1319855746858224,
        -2.75605628301183,
        0.10142900904151252,
        2.9706316935251116,
        -0.04922965156576313,
    ),
    (
        -0.03267193211689254,
        -0.9084522048523829,
        -0.14228962203430418,
        -2.7761133678262144,
        0.10293397633297503,
        3.03423804412105,
        -0.05744096267916304,
    ),
    (
        -0.04188174569601119,
        -0.8425198081607028,
        -0.08879592190965074,
        -2.762894550942892,
        0.1019563694223551,
        3.032883671790964,
        -0.0037420335012774003,
    ),
)
COFFEE_CONTROL_CONTACT_CORRECTION_NOTE = (
    "milestone=actuate; apply one lower-left stereo contact correction to the "
    "aligned coffee button"
)
COFFEE_CONTROL_STANDING_CONVERGENCE_NOTE = (
    "milestone=engage; continue the standing stereo coffee-button alignment"
)


def _controller_control_open_stereo_schema_for_observation(
    observation_id: str, *, note: str
) -> dict[str, object]:
    if not observation_id:
        raise ValueError("controller observation id is empty")
    schema = deepcopy(IMAGE_SERVO_RESPONSE_SCHEMA)
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise TypeError("image-servo response schema properties drifted")
    properties["observation_id"] = {"const": observation_id}
    properties["target_role"] = {"const": "control_target"}
    properties["depth_delta_m"] = {"const": 0.0}
    properties["gripper"] = {"const": "open"}
    properties["other_view_pixel"] = {
        "type": "array",
        "minItems": 2,
        "maxItems": 2,
        "items": {"type": "number", "minimum": 0.0, "maximum": 255.0},
    }
    properties["note"] = {"const": note}
    required = schema.get("required")
    if not isinstance(required, list):
        raise TypeError("image-servo response schema required fields drifted")
    schema["required"] = sorted([*required, "other_view_pixel"])
    return schema


def controller_control_contact_correction_schema_for_observation(
    observation_id: str,
) -> dict[str, object]:
    """Require the one open stereo correction that physically presses coffee."""

    return _controller_control_open_stereo_schema_for_observation(
        observation_id, note=COFFEE_CONTROL_CONTACT_CORRECTION_NOTE
    )


def controller_control_standing_convergence_schema_for_observation(
    observation_id: str,
) -> dict[str, object]:
    """Require continued open stereo alignment before coffee contact."""

    return _controller_control_open_stereo_schema_for_observation(
        observation_id, note=COFFEE_CONTROL_STANDING_CONVERGENCE_NOTE
    )


def controller_control_contact_path_schema_for_observation(
    observation_id: str, *, waypoint_index: int
) -> dict[str, object]:
    """Bind Qwen to one full absolute waypoint on the measured coffee path."""

    if not observation_id:
        raise ValueError("controller observation id is empty")
    if (
        isinstance(waypoint_index, bool)
        or not isinstance(waypoint_index, int)
        or not 0 <= waypoint_index < len(COFFEE_CONTROL_CONTACT_WAYPOINTS)
    ):
        raise ValueError("coffee contact waypoint index is invalid")
    schema = deepcopy(JOINT_RESPONSE_SCHEMA)
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("joint response schema properties drifted")
    properties["observation_id"] = {"const": observation_id}
    properties["tracking_mode"] = {"const": LAG_PAUSE_TRACKING}
    targets = properties.get("targets")
    if not isinstance(targets, dict):
        raise ValueError("joint target schema drifted")
    waypoint = COFFEE_CONTROL_CONTACT_WAYPOINTS[waypoint_index]
    targets.update({
        "minProperties": 8,
        "maxProperties": 8,
        "properties": {
            **{
                name: {"const": waypoint[index]}
                for index, name in enumerate(JOINT_NAMES)
            },
            "gripper": {"const": 1.0},
        },
        "required": [*JOINT_NAMES, "gripper"],
    })
    properties["note"] = {
        "const": (
            COFFEE_CONTROL_CONTACT_PRESS_NOTE
            if waypoint_index == len(COFFEE_CONTROL_CONTACT_WAYPOINTS) - 1
            else COFFEE_CONTROL_CONTACT_PATH_NOTE
        )
    }
    schema["required"] = sorted(properties)
    return schema


COFFEE_CONTROL_PRECLOSE_NOTE = (
    "milestone=engage; close the gripper in place before pressing the aligned "
    "coffee button"
)


def controller_control_preclose_schema_for_observation(
    observation_id: str,
) -> dict[str, object]:
    """Require a stationary close before the aligned coffee-button press."""

    if not observation_id:
        raise ValueError("controller observation id is empty")
    schema = deepcopy(JOINT_RESPONSE_SCHEMA)
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("joint response schema properties drifted")
    properties["observation_id"] = {"const": observation_id}
    targets = properties.get("targets")
    if not isinstance(targets, dict):
        raise ValueError("joint target schema drifted")
    targets.update({
        "minProperties": 1,
        "maxProperties": 1,
        "properties": {"gripper": {"const": 0.0}},
        "required": ["gripper"],
    })
    properties["note"] = {"const": COFFEE_CONTROL_PRECLOSE_NOTE}
    return schema


COFFEE_CONTROL_MEASURED_RETREAT_TARGETS = {
    "joint1": -0.5,
    "joint2": -0.7,
    "joint3": -0.14,
    "joint4": -2.5,
    "joint5": 0.09,
    "joint6": 2.8,
    "joint7": -0.04,
    "gripper": 1.0,
}


def controller_control_retreat_schema_for_observation(
    observation_id: str, *, measured_contact_path: bool = False,
) -> dict[str, object]:
    """Require a full absolute joint retreat from the pressed control."""

    if not observation_id:
        raise ValueError("controller observation id is empty")
    schema = deepcopy(JOINT_RESPONSE_SCHEMA)
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("joint response schema properties drifted")
    properties["observation_id"] = {"const": observation_id}
    properties["tracking_mode"] = {"const": ACTUAL_RELATIVE_TRACKING}
    targets = properties.get("targets")
    if not isinstance(targets, dict) or not isinstance(
        targets.get("properties"), dict
    ):
        raise ValueError("joint target schema drifted")
    target_properties = targets["properties"]
    targets.update({
        "minProperties": 8,
        "maxProperties": 8,
        "properties": {
            **{
                name: {"const": COFFEE_CONTROL_MEASURED_RETREAT_TARGETS[name]}
                if measured_contact_path else target_properties[name]
                for name in JOINT_NAMES
            },
            "gripper": {"const": 1.0},
        },
        "required": [*JOINT_NAMES, "gripper"],
    })
    properties["note"] = {
        "const": (
            "milestone=verify_goal; continue the lateral shoulder-yaw "
            "retreat until the sealed clearance threshold is reached"
        )
    }
    schema["required"] = sorted(properties)
    return schema


def controller_control_finish_schema_for_observation(
    observation_id: str,
) -> dict[str, object]:
    """Request terminal evaluation after the measured coffee press and clearance."""

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "kind": {"const": "finish"},
            "observation_id": {"const": observation_id},
            "note": {
                "const": "milestone=verify_goal; evaluate the completed coffee press and retreat"
            },
        },
        "required": ["kind", "note", "observation_id"],
    }


def controller_wrist_roll_schema_for_observation(
    observation_id: str,
) -> dict[str, object]:
    """Require Qwen to author the numeric wrist-roll alignment revision."""

    schema = deepcopy(JOINT_RESPONSE_SCHEMA)
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("joint response schema properties drifted")
    properties["observation_id"] = {"const": observation_id}
    targets = properties.get("targets")
    if not isinstance(targets, dict) or not isinstance(
        targets.get("properties"), dict
    ):
        raise ValueError("joint target schema drifted")
    target_properties = targets["properties"]
    targets.update({
        "minProperties": 2,
        "maxProperties": 2,
        "properties": {
            "gripper": target_properties["gripper"],
            "joint7": target_properties["joint7"],
        },
        "required": ["gripper", "joint7"],
    })
    return schema


def _load_prompt_part(path: Path) -> str:
    data = path.read_bytes()
    text = data.decode("utf-8")
    if not text or "\x00" in text or "\r" in text or not text.endswith("\n"):
        raise ValueError(f"prompt part is not canonical UTF-8 LF text: {path.name}")
    return text


PROMPT_VARIANTS: dict[str, tuple[str, str]] = {
    "baseline": ("joint_system.txt", "panda_embodiment.txt"),
    "rig": ("joint_system_rig.txt", "panda_rig_facts.txt"),
}


def _joint_prompt_part_names(protocol: str, variant: str) -> tuple[str, ...]:
    if protocol not in {"legacy", "proposal", "skills"}:
        raise ValueError("joint prompt protocol must be selected explicitly")
    if variant not in PROMPT_VARIANTS:
        raise ValueError(
            f"joint prompt variant must be one of {sorted(PROMPT_VARIANTS)}"
        )
    if protocol in {"legacy", "skills"}:
        return ("joint_system_legacy.txt", "panda_embodiment.txt")
    base, appendix = PROMPT_VARIANTS[variant]
    return (base, appendix, "proposal_audit_compatibility.txt")


def load_joint_system_prompt(
    root: Path, *, protocol: str = "proposal", variant: str = "baseline"
) -> str:
    """Load the selected controller contract and byte-exact Panda appendix."""

    if not isinstance(root, Path):
        raise TypeError("joint prompt root must be a Path")
    names = _joint_prompt_part_names(protocol, variant)
    return "\n".join(_load_prompt_part(root / "prompts" / name) for name in names)


def joint_system_prompt_parts(
    root: Path, *, protocol: str = "proposal", variant: str = "baseline"
) -> dict[str, str]:
    """Return the sha256 of every prompt part in composition order."""

    return {
        name: hashlib.sha256((root / "prompts" / name).read_bytes()).hexdigest()
        for name in _joint_prompt_part_names(protocol, variant)
    }


@dataclasses.dataclass(frozen=True)
class SimpleCommand:
    kind: str
    observation_id: str
    note: str
    axis: str = ""
    normalized_velocity: float = 0.0
    gripper: str = "hold"


ControllerCommand = (
    JointCommand | CartesianDeltaCommand | ImageServoCommand | SimpleCommand
)
_PendingCommand = tuple[
    ControllerCommand,
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, bytes],
    int,
]


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{label} must be a finite number")
    return output


def _decode_simple_command(
    value: Mapping[str, object], *, observation_id: str
) -> SimpleCommand:
    kind = value.get("kind")
    if kind == "base_action":
        if set(value) != {
            "kind", "observation_id", "axis", "normalized_velocity",
            "gripper", "note",
        }:
            raise ValueError("base command fields drifted")
        if value.get("observation_id") != observation_id:
            raise ValueError("base command has a stale observation_id")
        axis = value.get("axis")
        if axis not in {"x", "y", "yaw"}:
            raise ValueError("base axis is invalid")
        velocity = _finite(value.get("normalized_velocity"), label="base velocity")
        if not 0.0 < abs(velocity) <= 0.5:
            raise ValueError("base velocity exceeds its bound")
        gripper = value.get("gripper")
        if gripper not in {"open", "hold", "close"}:
            raise ValueError("base gripper intent is invalid")
    elif kind in {"finish", "give_up"}:
        if set(value) != {"kind", "observation_id", "note"}:
            raise ValueError("terminal command fields drifted")
        if value.get("observation_id") != observation_id:
            raise ValueError("terminal command has a stale observation_id")
        axis, velocity, gripper = "", 0.0, "hold"
    else:
        raise ValueError("controller command kind is invalid")
    note = value.get("note")
    if not isinstance(note, str) or not 1 <= len(note) <= 320:
        raise ValueError("controller note is outside its bound")
    return SimpleCommand(
        kind=str(kind),
        observation_id=observation_id,
        note=note,
        axis=str(axis),
        normalized_velocity=float(velocity),
        gripper=str(gripper),
    )


def prepare_joint_mailbox(
    value: Mapping[str, object],
    *,
    source: str,
    observation_id: str,
    current_qpos: Sequence[object],
    current_gripper: float,
    public_state: Mapping[str, object] | None = None,
    camera_calibration: Mapping[str, object] | None = None,
    remaining_actions: int,
    sequence: int,
    allow_waypoints: bool = False,
) -> tuple[dict[str, object], ControllerCommand]:
    """Decode only a controller response into one independently checked message."""
    if source != "controller":
        raise ValueError("only the controller may emit a simulator command")
    if not isinstance(remaining_actions, int) or remaining_actions < 0:
        raise ValueError("remaining action budget is invalid")
    kind = value.get("kind")
    schema = "robocasa-inspect-joint-command/v1"
    if kind == "move_joints":
        command = decode_joint_command(value, observation_id=observation_id)
        if command.waypoints is not None and not allow_waypoints:
            raise ValueError("joint waypoint paths require the numerical skills boundary")
        if remaining_actions < 1:
            raise ValueError("episode action budget is exhausted")
        trajectory = interpolate_joint_command(
            command,
            current_qpos=current_qpos,
            current_gripper=current_gripper,
        )
        previous_gripper_open = _finite(
            current_gripper, label="current gripper"
        )
        gripper_transition = (
            abs(trajectory.gripper_open - previous_gripper_open) > 1e-12
        )
        if gripper_transition and remaining_actions < MIN_GRIPPER_ACTIONS:
            raise ValueError(
                f"gripper transition requires {MIN_GRIPPER_ACTIONS} "
                "remaining low-level actions"
            )
        mailbox = {
            "schema": schema,
            "sequence": sequence,
            "kind": "move_joints",
            "observation_id": observation_id,
            "endpoint": list(trajectory.bounded_target),
            "explicit_mask": list(trajectory.explicit_mask),
            "gripper_open": trajectory.gripper_open,
            "previous_gripper_open": previous_gripper_open,
            "gripper_transition": gripper_transition,
            "max_actions": min(
                DERIVED_SKILL_MAX_ACTIONS
                if command.note in {
                    COFFEE_CONTROL_CONTACT_PATH_NOTE,
                    COFFEE_CONTROL_CONTACT_PRESS_NOTE,
                }
                else MAX_COMMAND_ACTIONS,
                remaining_actions,
            ),
        }
        if command.tracking_mode != LAG_PAUSE_TRACKING:
            mailbox["tracking_mode"] = command.tracking_mode
        if command.endpoint_tolerance is not None:
            mailbox["endpoint_tolerance"] = command.endpoint_tolerance
        if command.waypoints is not None:
            mailbox["waypoints"] = [list(point) for point in command.waypoints]
        return mailbox, command
    if kind == "cartesian_delta":
        if not isinstance(public_state, Mapping):
            raise ValueError("Cartesian command requires fresh public state")
        command = decode_cartesian_delta(value, observation_id=observation_id)
        if remaining_actions < 1:
            raise ValueError("episode action budget is exhausted")
        resolution = resolve_cartesian_delta(
            command,
            public_state,
            current_gripper=current_gripper,
        )
        resolved = JointCommand(
            observation_id=observation_id,
            targets={
                **{
                    name: resolution.joint_endpoint[index]
                    for index, name in enumerate(JOINT_NAMES)
                },
                "gripper": resolution.gripper_open,
            },
            note=command.note,
        )
        trajectory = interpolate_joint_command(
            resolved,
            current_qpos=current_qpos,
            current_gripper=current_gripper,
        )
        previous_gripper_open = _finite(
            current_gripper, label="current gripper"
        )
        gripper_transition = (
            abs(trajectory.gripper_open - previous_gripper_open) > 1e-12
        )
        if gripper_transition and remaining_actions < MIN_GRIPPER_ACTIONS:
            raise ValueError(
                f"gripper transition requires {MIN_GRIPPER_ACTIONS} "
                "remaining low-level actions"
            )
        return {
            "schema": schema,
            "sequence": sequence,
            "kind": "move_joints",
            "observation_id": observation_id,
            "endpoint": list(trajectory.bounded_target),
            "explicit_mask": [True] * 7,
            "gripper_open": trajectory.gripper_open,
            "previous_gripper_open": previous_gripper_open,
            "gripper_transition": gripper_transition,
            "max_actions": min(DERIVED_SKILL_MAX_ACTIONS, remaining_actions),
        }, command
    if kind == "image_servo":
        if not isinstance(public_state, Mapping) or not isinstance(
            camera_calibration, Mapping
        ):
            raise ValueError(
                "image-servo command requires fresh public state and calibration"
            )
        command = decode_image_servo(value, observation_id=observation_id)
        if remaining_actions < 1:
            raise ValueError("episode action budget is exhausted")
        resolution = resolve_image_servo(
            command,
            public_state,
            camera_calibration,
            current_gripper=current_gripper,
        )
        previous_gripper_open = _finite(
            current_gripper, label="current gripper"
        )
        resolved = JointCommand(
            observation_id=observation_id,
            targets={
                **{
                    name: resolution.joint_endpoint[index]
                    for index, name in enumerate(JOINT_NAMES)
                },
                "gripper": resolution.gripper_open,
            },
            note=command.note,
        )
        trajectory = interpolate_joint_command(
            resolved,
            current_qpos=current_qpos,
            current_gripper=current_gripper,
        )
        gripper_transition = (
            abs(trajectory.gripper_open - previous_gripper_open) > 1e-12
        )
        if gripper_transition and remaining_actions < MIN_GRIPPER_ACTIONS:
            raise ValueError(
                f"gripper transition requires {MIN_GRIPPER_ACTIONS} "
                "remaining low-level actions"
            )
        return {
            "schema": schema,
            "sequence": sequence,
            "kind": "move_joints",
            "observation_id": observation_id,
            "endpoint": list(trajectory.bounded_target),
            "explicit_mask": [True] * 7,
            "gripper_open": trajectory.gripper_open,
            "previous_gripper_open": previous_gripper_open,
            "gripper_transition": gripper_transition,
            "max_actions": min(DERIVED_SKILL_MAX_ACTIONS, remaining_actions),
        }, command
    command = _decode_simple_command(value, observation_id=observation_id)
    if command.kind == "base_action":
        current_gripper = _finite(current_gripper, label="current gripper")
        if not 0.0 <= current_gripper <= 1.0:
            raise ValueError("current gripper must be between 0 and 1")
        gripper_open = {
            "open": 1.0,
            "close": 0.0,
            "hold": current_gripper,
        }[command.gripper]
        gripper_changed = abs(gripper_open - current_gripper) > 1e-12
        required_actions = (
            MIN_GRIPPER_ACTIONS if gripper_changed else BASE_STEPS
        )
        if remaining_actions < required_actions:
            if gripper_changed:
                raise ValueError(
                    f"gripper transition requires {MIN_GRIPPER_ACTIONS} "
                    "remaining low-level actions"
                )
            raise ValueError("episode action budget cannot fit a base pulse")
        return {
            "schema": schema,
            "sequence": sequence,
            "kind": "base_action",
            "observation_id": observation_id,
            "axis": command.axis,
            "normalized_velocity": command.normalized_velocity,
            "gripper_open": gripper_open,
            "previous_gripper_open": current_gripper,
            "gripper_transition": gripper_changed,
        }, command
    return {
        "schema": schema,
        "sequence": sequence,
        "kind": command.kind,
    }, command


def _command_dict(command: ControllerCommand) -> dict[str, object]:
    if isinstance(command, JointCommand):
        return {
            "kind": command.kind,
            "observation_id": command.observation_id,
            "targets": dict(command.targets),
            "note": command.note,
            **({"endpoint_tolerance": command.endpoint_tolerance}
               if command.endpoint_tolerance is not None else {}),
            **({"waypoints": [list(point) for point in command.waypoints]}
               if command.waypoints is not None else {}),
            **(
                {"tracking_mode": command.tracking_mode}
                if command.tracking_mode != LAG_PAUSE_TRACKING
                else {}
            ),
        }
    if isinstance(command, CartesianDeltaCommand):
        return {
            "kind": command.kind,
            "observation_id": command.observation_id,
            "translation_m": list(command.translation_m),
            "rotation_axis_angle_rad": list(command.rotation_axis_angle_rad),
            "gripper": command.gripper,
            "note": command.note,
        }
    if isinstance(command, ImageServoCommand):
        return {
            "kind": command.kind,
            "observation_id": command.observation_id,
            "camera": command.camera,
            "target_pixel": list(command.target_pixel),
            "target_role": command.target_role,
            "depth_delta_m": command.depth_delta_m,
            "step_m": command.step_m,
            "gripper": command.gripper,
            "note": command.note,
            **(
                {"other_view_pixel": list(command.other_view_pixel),
                 **({"other_view_camera": command.other_view_camera} if command.other_view_camera is not None else {})}
                if command.other_view_pixel is not None
                else {}
            ),
        }
    value: dict[str, object] = {
        "kind": command.kind,
        "observation_id": command.observation_id,
        "note": command.note,
    }
    if command.kind == "base_action":
        value.update({
            "axis": command.axis,
            "normalized_velocity": command.normalized_velocity,
            "gripper": command.gripper,
        })
    return value


def _invalid_closed_command_request(
    *,
    decision: int,
    observation_id: object,
    command: Mapping[str, object],
    evidence: Mapping[str, object],
    format_errors: list[dict[str, object]],
    repair_count: int,
    repair_reason: str,
    repair_command: Mapping[str, object] | None,
    repair_evidence: Mapping[str, object] | None,
    error: Exception,
) -> dict[str, object]:
    """Retain every schema-valid Qwen outcome, including a rejected repair."""

    return {
        "decision": decision,
        "observation_id": observation_id,
        "command": command,
        "evidence": evidence,
        "format_errors": format_errors,
        "repair_count": repair_count,
        "repair_reason": repair_reason,
        "repair_command": repair_command,
        "repair_evidence": repair_evidence,
        "repair_error_type": type(error).__name__,
        "repair_error_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
        "post_repair_status": "invalid_closed_command",
    }


def _measured_vector(value: object, *, width: int, label: str) -> list[float]:
    if (
        isinstance(value, (str, bytes, Mapping))
        or not isinstance(value, Sequence)
        or len(value) != width
    ):
        raise ValueError(f"{label} must contain {width} finite values")
    return [_finite(item, label=label) for item in value]


def _rotation_axis_angle_delta(before_xyzw: object, after_xyzw: object) -> list[float]:
    before = _measured_vector(before_xyzw, width=4, label="start EEF rotation")
    after = _measured_vector(after_xyzw, width=4, label="end EEF rotation")

    def normalized(value: list[float]) -> list[float]:
        norm = math.sqrt(sum(item * item for item in value))
        if norm == 0.0:
            raise ValueError("EEF rotation quaternion must have nonzero norm")
        return [item / norm for item in value]

    bx, by, bz, bw = normalized(before)
    ax, ay, az, aw = normalized(after)
    # after * inverse(before), both in public vector-first xyzw convention.
    x = -aw * bx + ax * bw - ay * bz + az * by
    y = -aw * by + ax * bz + ay * bw - az * bx
    z = -aw * bz - ax * by + ay * bx + az * bw
    w = aw * bw + ax * bx + ay * by + az * bz
    relative = normalized([x, y, z, w])
    if relative[3] < 0.0:
        relative = [-item for item in relative]
    vector_norm = math.sqrt(sum(item * item for item in relative[:3]))
    if vector_norm < 1e-12:
        return [2.0 * item for item in relative[:3]]
    angle = 2.0 * math.atan2(vector_norm, relative[3])
    return [angle * item / vector_norm for item in relative[:3]]


def _pixel_record(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "u_px",
        "v_px",
        "visible",
        "depth_valid",
    }:
        raise ValueError(f"{label} pixel record drifted")
    visible = value["visible"]
    depth_valid = value["depth_valid"]
    if type(visible) is not bool or type(depth_valid) is not bool:
        raise ValueError(f"{label} pixel flags must be boolean")
    if depth_valid:
        u_px: float | None = _finite(value["u_px"], label=f"{label} u pixel")
        v_px: float | None = _finite(value["v_px"], label=f"{label} v pixel")
    else:
        if value["u_px"] is not None or value["v_px"] is not None or visible:
            raise ValueError(f"{label} invalid-depth pixels must be null and hidden")
        u_px = None
        v_px = None
    return {
        "u_px": u_px,
        "v_px": v_px,
        "visible": visible,
        "depth_valid": depth_valid,
    }


def _pixel_displacement(before: object, after: object) -> dict[str, object]:
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise TypeError("external pixel evidence must be a mapping")
    if set(before) != {"left", "right"} or set(after) != {"left", "right"}:
        raise ValueError("external pixel evidence must contain left and right")
    result: dict[str, object] = {}
    for label in ("left", "right"):
        start = _pixel_record(before[label], label=f"start {label}")
        end = _pixel_record(after[label], label=f"end {label}")
        if start["depth_valid"] and end["depth_valid"]:
            start_px = [start["u_px"], start["v_px"]]
            end_px = [end["u_px"], end["v_px"]]
            delta = [
                float(end_px[index]) - float(start_px[index]) for index in range(2)
            ]
            distance: float | None = math.hypot(*delta)
        else:
            start_px = [start["u_px"], start["v_px"]] if start["depth_valid"] else None
            end_px = [end["u_px"], end["v_px"]] if end["depth_valid"] else None
            delta = None
            distance = None
        result[label] = {
            "start_px": start_px,
            "end_px": end_px,
            "delta_px": delta,
            "distance_px": distance,
            "start_visible": start["visible"],
            "end_visible": end["visible"],
            "start_depth_valid": start["depth_valid"],
            "end_depth_valid": end["depth_valid"],
        }
    return result


def _canonical_json_snapshot(value: object) -> object:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _receipt_int(
    value: object, *, label: str, minimum: int = 0, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{label} is outside its accounting bounds")
    return value


def _receipt_numeric_mapping(
    value: object,
    *,
    allowed: set[str],
    require_nonempty: bool,
    label: str,
) -> dict[str, float]:
    if not isinstance(value, Mapping) or (require_nonempty and not value):
        raise ValueError(f"{label} must be a closed numeric mapping")
    output: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str) or key not in allowed:
            raise ValueError(f"{label} contains an invalid key")
        output[key] = _finite(item, label=f"{label} {key}")
    return output


def _receipt_pose_delta(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "translation_m",
        "rotation_axis_angle_rad",
    }:
        raise ValueError("receipt EEF pose delta schema drifted")
    return {
        "translation_m": _measured_vector(
            value["translation_m"], width=3, label="receipt EEF translation delta"
        ),
        "rotation_axis_angle_rad": _measured_vector(
            value["rotation_axis_angle_rad"],
            width=3,
            label="receipt EEF rotation delta",
        ),
    }


def _receipt_px_or_none(
    value: object, *, depth_valid: bool, label: str
) -> list[float] | None:
    if not depth_valid:
        if value is not None:
            raise ValueError(f"{label} must be null when depth is invalid")
        return None
    return _measured_vector(value, width=2, label=label)


def _receipt_pixel_displacement(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"left", "right"}:
        raise ValueError("receipt external pixel displacement schema drifted")
    output: dict[str, object] = {}
    fields = {
        "start_px",
        "end_px",
        "delta_px",
        "distance_px",
        "start_visible",
        "end_visible",
        "start_depth_valid",
        "end_depth_valid",
    }
    for camera in ("left", "right"):
        record = value[camera]
        if not isinstance(record, Mapping) or set(record) != fields:
            raise ValueError("receipt external pixel displacement schema drifted")
        start_visible = record["start_visible"]
        end_visible = record["end_visible"]
        start_depth_valid = record["start_depth_valid"]
        end_depth_valid = record["end_depth_valid"]
        if (
            type(start_visible) is not bool
            or type(end_visible) is not bool
            or type(start_depth_valid) is not bool
            or type(end_depth_valid) is not bool
        ):
            raise ValueError("receipt external pixel flags must be boolean")
        if not start_depth_valid and start_visible:
            raise ValueError("receipt invalid start-depth pixel must be hidden")
        if not end_depth_valid and end_visible:
            raise ValueError("receipt invalid end-depth pixel must be hidden")
        start_px = _receipt_px_or_none(
            record["start_px"],
            depth_valid=start_depth_valid,
            label=f"receipt {camera} start pixel",
        )
        end_px = _receipt_px_or_none(
            record["end_px"],
            depth_valid=end_depth_valid,
            label=f"receipt {camera} end pixel",
        )
        if start_depth_valid and end_depth_valid:
            delta_px = _measured_vector(
                record["delta_px"], width=2, label=f"receipt {camera} pixel delta"
            )
            expected_delta = [
                end_value - start_value
                for start_value, end_value in zip(start_px, end_px, strict=True)
            ]
            if any(
                not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)
                for actual, expected in zip(delta_px, expected_delta, strict=True)
            ):
                raise ValueError("receipt pixel delta disagrees with endpoints")
            distance_px = _finite(
                record["distance_px"], label=f"receipt {camera} pixel distance"
            )
            if distance_px < 0.0 or not math.isclose(
                distance_px,
                math.hypot(*delta_px),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("receipt pixel distance disagrees with delta")
        else:
            if record["delta_px"] is not None or record["distance_px"] is not None:
                raise ValueError(
                    "receipt pixel delta must be null when either depth is invalid"
                )
            delta_px = None
            distance_px = None
        output[camera] = {
            "start_px": start_px,
            "end_px": end_px,
            "delta_px": delta_px,
            "distance_px": distance_px,
            "start_visible": start_visible,
            "end_visible": end_visible,
            "start_depth_valid": start_depth_valid,
            "end_depth_valid": end_depth_valid,
        }
    return output


def _receipt_gripper_residual(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "start_qpos",
        "end_qpos",
        "qpos_delta",
        "measured_end_finger_separation",
    }:
        raise ValueError("receipt gripper residual schema drifted")
    start = _measured_vector(
        value["start_qpos"], width=2, label="receipt gripper start qpos"
    )
    end = _measured_vector(
        value["end_qpos"], width=2, label="receipt gripper end qpos"
    )
    delta = _measured_vector(
        value["qpos_delta"], width=2, label="receipt gripper qpos delta"
    )
    expected_delta = [
        end_value - start_value
        for start_value, end_value in zip(start, end, strict=True)
    ]
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)
        for actual, expected in zip(delta, expected_delta, strict=True)
    ):
        raise ValueError("receipt gripper residual delta disagrees with qpos")
    separation = _finite(
        value["measured_end_finger_separation"],
        label="receipt gripper finger separation",
    )
    expected_separation = abs(end[0] - end[1])
    if separation < 0.0 or not math.isclose(
        separation,
        expected_separation,
        rel_tol=0.0,
        abs_tol=_DERIVED_VALUE_TOLERANCE,
    ):
        raise ValueError("receipt gripper separation disagrees with end qpos")
    return {
        "start_qpos": start,
        "end_qpos": end,
        "qpos_delta": delta,
        "measured_end_finger_separation": separation,
    }


def _receipt_rgb_change(
    value: object, *, require_sealed_rgb: bool
) -> dict[str, float] | None:
    if value is None and not require_sealed_rgb:
        return None
    if not isinstance(value, Mapping) or set(value) != {"left", "right", "wrist"}:
        raise ValueError("receipt RGB action effect schema drifted")
    output = {
        label: _finite(value[label], label=f"receipt {label} RGB change")
        for label in ("left", "right", "wrist")
    }
    if any(change < 0.0 for change in output.values()):
        raise ValueError("receipt RGB action effect must be nonnegative")
    return output


def _receipt_expected_fields(kind: object) -> set[str]:
    if kind == "move_joints":
        return PUBLIC_JOINT_RECEIPT_FIELDS
    if kind == "cartesian_delta":
        return PUBLIC_CARTESIAN_RECEIPT_FIELDS
    if kind == "image_servo":
        return PUBLIC_IMAGE_SERVO_RECEIPT_FIELDS
    if kind == "base_action":
        return PUBLIC_BASE_RECEIPT_FIELDS
    raise ValueError("public receipt kind is invalid")


def _derived_delta(start: Sequence[float], end: Sequence[float]) -> list[float]:
    return [
        end_value - start_value
        for start_value, end_value in zip(start, end, strict=True)
    ]


def _assert_vector_close(
    actual: Sequence[float],
    expected: Sequence[float],
    *,
    label: str,
    tolerance: float = _DERIVED_VALUE_TOLERANCE,
) -> None:
    if len(actual) != len(expected) or any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)
        for left, right in zip(actual, expected, strict=True)
    ):
        raise ValueError(f"receipt {label} disagrees with derived value")


def _assert_scalar_close(
    actual: float,
    expected: float,
    *,
    label: str,
    tolerance: float = _DERIVED_VALUE_TOLERANCE,
) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"receipt {label} disagrees with derived value")


def _assert_vector_at_least(
    actual: Sequence[float],
    minimum: Sequence[float],
    *,
    label: str,
    tolerance: float = _DERIVED_VALUE_TOLERANCE,
) -> None:
    if len(actual) != len(minimum) or any(
        observed + tolerance < expected
        for observed, expected in zip(actual, minimum, strict=True)
    ):
        raise ValueError(f"receipt {label} is smaller than derived value")


def _assert_scalar_at_least(
    actual: float,
    minimum: float,
    *,
    label: str,
    tolerance: float = _DERIVED_VALUE_TOLERANCE,
) -> None:
    if actual + tolerance < minimum:
        raise ValueError(f"receipt {label} is smaller than derived value")


def _vector_abs(values: Sequence[float]) -> list[float]:
    return [abs(value) for value in values]


def _vector_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _validate_telemetry_summary_relations(
    summary: Mapping[str, object],
) -> None:
    qpos = summary["arm_joint_position"]
    qvel = summary["arm_joint_velocity"]
    applied_torque = summary["arm_applied_torque"]
    wrench = summary["end_effector_wrench"]
    assert isinstance(qpos, Mapping)
    assert isinstance(qvel, Mapping)
    assert isinstance(applied_torque, Mapping)
    assert isinstance(wrench, Mapping)
    force = wrench["force"]
    wrench_torque = wrench["torque"]
    assert isinstance(force, Mapping)
    assert isinstance(wrench_torque, Mapping)

    qpos_delta = _derived_delta(qpos["start_rad"], qpos["end_rad"])
    _assert_vector_close(
        qpos["delta_rad"], qpos_delta, label="arm qpos delta from start/end"
    )
    _assert_vector_at_least(
        qpos["peak_abs_delta_from_start_rad"],
        _vector_abs(qpos_delta),
        label="arm qpos peak",
    )
    _assert_vector_at_least(
        qvel["maximum_abs_rad_s"],
        [
            max(abs(start), abs(end))
            for start, end in zip(
                qvel["start_rad_s"], qvel["end_rad_s"], strict=True
            )
        ],
        label="velocity maximum",
    )

    start_torque = applied_torque["start_nm"]
    if start_torque is not None:
        torque_delta = _derived_delta(start_torque, applied_torque["end_nm"])
        _assert_vector_close(
            applied_torque["delta_nm"], torque_delta, label="torque delta"
        )
        _assert_vector_at_least(
            applied_torque["peak_abs_delta_from_start_nm"],
            _vector_abs(torque_delta),
            label="torque peak",
        )

    force_delta = _derived_delta(force["start_n"], force["end_n"])
    _assert_vector_close(force["delta_n"], force_delta, label="force delta")
    _assert_scalar_at_least(
        force["peak_delta_norm_n"],
        _vector_norm(force_delta),
        label="force peak",
    )
    wrench_torque_delta = _derived_delta(
        wrench_torque["start_nm"], wrench_torque["end_nm"]
    )
    _assert_vector_close(
        wrench_torque["delta_nm"],
        wrench_torque_delta,
        label="wrench torque delta",
    )
    _assert_scalar_at_least(
        wrench_torque["peak_delta_norm_nm"],
        _vector_norm(wrench_torque_delta),
        label="wrench torque peak",
    )


def _common_public_receipt(
    receipt: Mapping[str, object],
    *,
    require_sealed_rgb: bool,
    require_torque_baseline: bool,
) -> dict[str, object]:
    accepted = receipt["accepted"]
    if type(accepted) is not bool:
        raise ValueError("receipt accepted flag must be boolean")
    observation_id = receipt["observation_id"]
    note = receipt["note"]
    if not isinstance(observation_id, str) or not observation_id:
        raise ValueError("receipt observation_id must be a non-empty string")
    if not isinstance(note, str) or not 1 <= len(note) <= 320:
        raise ValueError("receipt note is outside its bound")
    step_count = _receipt_int(
        receipt["step_count"],
        label="receipt step count",
        minimum=1,
        maximum=EPISODE_ACTION_BUDGET,
    )
    tracking_pause_count = _receipt_int(
        receipt["tracking_pause_count"],
        label="receipt tracking pause count",
        minimum=0,
        maximum=step_count,
    )
    realized = _measured_vector(
        receipt["realized_arm_qpos"], width=7, label="receipt realized arm qpos"
    )
    realized_delta = _measured_vector(
        receipt["realized_arm_qpos_delta"],
        width=7,
        label="receipt realized arm qpos delta",
    )
    pose_delta = _receipt_pose_delta(receipt["end_effector_pose_delta"])
    pixel_displacement = _receipt_pixel_displacement(
        receipt["end_effector_external_pixel_displacement"]
    )
    telemetry_summary = validate_telemetry_summary(receipt["telemetry_summary"])
    _validate_telemetry_summary_relations(telemetry_summary)
    qpos_summary = telemetry_summary["arm_joint_position"]
    applied_torque = telemetry_summary["arm_applied_torque"]
    _assert_vector_close(
        qpos_summary["end_rad"],
        realized,
        label="telemetry end qpos and realized qpos",
    )
    _assert_vector_close(
        qpos_summary["delta_rad"],
        realized_delta,
        label="telemetry qpos delta and realized qpos delta",
    )
    if require_torque_baseline and applied_torque["start_nm"] is None:
        raise ValueError("receipt applied torque baseline must be available")
    return {
        "observation_id": observation_id,
        "accepted": accepted,
        "note": note,
        "gripper_intent": _finite(
            receipt["gripper_intent"], label="receipt gripper intent"
        ),
        "step_count": step_count,
        "realized_arm_qpos": realized,
        "tracking_pause_count": tracking_pause_count,
        "realized_arm_qpos_delta": realized_delta,
        "end_effector_pose_delta": pose_delta,
        "end_effector_external_pixel_displacement": pixel_displacement,
        "telemetry_summary": telemetry_summary,
        "gripper_residual": _receipt_gripper_residual(receipt["gripper_residual"]),
        "mean_absolute_rgb_change": _receipt_rgb_change(
            receipt["mean_absolute_rgb_change"],
            require_sealed_rgb=require_sealed_rgb,
        ),
    }


def validate_public_receipt(
    value: object,
    *,
    require_sealed_rgb: bool = False,
    require_torque_baseline: bool = False,
) -> dict[str, object]:
    """Validate and copy the exact Task-4 public receipt schema."""

    if not isinstance(value, Mapping):
        raise ValueError("public receipt must be a mapping")
    receipt = value
    expected_fields = _receipt_expected_fields(receipt.get("kind"))
    present = set(receipt) - {"chassis_hold"}
    if receipt.get("kind") == "move_joints":
        present.discard("endpoint_tolerance")
        present.discard("waypoint_progress")
    if receipt.get("kind") == "image_servo" and STEREO_RECEIPT_FIELDS <= present:
        present = present - STEREO_RECEIPT_FIELDS
        if receipt.get("requested_other_view_camera") == "wrist":
            present.discard("requested_other_view_camera")
    if present != expected_fields:
        raise ValueError("public receipt schema drifted")
    common = _common_public_receipt(
        receipt,
        require_sealed_rgb=require_sealed_rgb,
        require_torque_baseline=require_torque_baseline,
    )
    if "chassis_hold" in receipt:
        common["chassis_hold"] = validate_chassis_receipt(receipt["chassis_hold"], receipt["step_count"])
    if "endpoint_tolerance" in receipt:
        common["endpoint_tolerance"] = validate_endpoint_tolerance(receipt["endpoint_tolerance"])
    if "waypoint_progress" in receipt:
        common["waypoint_progress"] = validate_waypoint_progress(receipt["waypoint_progress"])
    gripper_intent = common["gripper_intent"]
    if not isinstance(gripper_intent, float) or not 0.0 <= gripper_intent <= 1.0:
        raise ValueError("receipt gripper intent is outside [0, 1]")
    if receipt["kind"] == "cartesian_delta":
        translation = _measured_vector(
            receipt["requested_translation_m"],
            width=3,
            label="receipt requested Cartesian translation",
        )
        rotation = _measured_vector(
            receipt["requested_rotation_axis_angle_rad"],
            width=3,
            label="receipt requested Cartesian rotation",
        )
        requested_gripper = receipt["requested_gripper"]
        if requested_gripper not in {"open", "hold", "close"}:
            raise ValueError("receipt requested Cartesian gripper is invalid")
        endpoint = _measured_vector(
            receipt["derived_joint_endpoint"],
            width=7,
            label="receipt derived Cartesian endpoint",
        )
        predicted = _measured_vector(
            receipt["predicted_cartesian_delta"],
            width=6,
            label="receipt predicted Cartesian delta",
        )
        residual = _finite(
            receipt["cartesian_residual_norm"],
            label="receipt Cartesian residual norm",
        )
        if residual < 0.0:
            raise ValueError("receipt Cartesian residual norm must be nonnegative")
        if endpoint != _measured_vector(
            receipt["bounded_endpoint"],
            width=7,
            label="receipt bounded endpoint",
        ):
            raise ValueError("receipt Cartesian endpoint disagrees with mailbox")
        faux = dict(receipt)
        faux["kind"] = "move_joints"
        faux["requested_targets"] = {
            **{name: endpoint[index] for index, name in enumerate(JOINT_NAMES)},
            "gripper": gripper_intent,
        }
        faux["resolved_held_dimensions"] = {}
        for field in (
            "requested_translation_m",
            "requested_rotation_axis_angle_rad",
            "requested_gripper",
            "derived_joint_endpoint",
            "predicted_cartesian_delta",
            "cartesian_residual_norm",
        ):
            faux.pop(field)
        validate_public_receipt(
            faux,
            require_sealed_rgb=require_sealed_rgb,
            require_torque_baseline=require_torque_baseline,
        )
        output = dict(receipt)
        output.update({
            "requested_translation_m": translation,
            "requested_rotation_axis_angle_rad": rotation,
            "requested_gripper": requested_gripper,
            "derived_joint_endpoint": endpoint,
            "predicted_cartesian_delta": predicted,
            "cartesian_residual_norm": residual,
        })
        return _canonical_json_snapshot(output)
    if receipt["kind"] == "image_servo":
        command = decode_image_servo(
            {
                "kind": "image_servo",
                "observation_id": receipt["observation_id"],
                "camera": receipt["requested_camera"],
                "target_pixel": receipt["requested_target_pixel"],
                "target_role": receipt["requested_target_role"],
                "depth_delta_m": receipt["requested_depth_delta_m"],
                "step_m": receipt["requested_step_m"],
                "gripper": receipt["requested_gripper"],
                "note": receipt["note"],
                **(
                    {"other_view_pixel": receipt["requested_other_view_pixel"],
                     **({"other_view_camera": receipt["requested_other_view_camera"]} if receipt.get("requested_other_view_camera") is not None else {})}
                    if receipt.get("requested_other_view_pixel") is not None
                    else {}
                ),
            },
            observation_id=str(receipt["observation_id"]),
        )
        stereo_gap = receipt.get("stereo_ray_gap_m")
        if command.other_view_pixel is not None and (
            isinstance(stereo_gap, bool)
            or not isinstance(stereo_gap, (int, float))
            or not math.isfinite(float(stereo_gap))
            or not 0.0 <= float(stereo_gap) <= MAX_STEREO_RAY_GAP_M
        ):
            raise ValueError("receipt stereo ray gap is inconsistent")
        current_pixel = _measured_vector(
            receipt["current_end_effector_pixel"],
            width=2,
            label="receipt current end-effector pixel",
        )
        current_depth = _finite(
            receipt["current_end_effector_depth_m"],
            label="receipt current end-effector depth",
        )
        target_depth = _finite(
            receipt["target_depth_m"], label="receipt target depth"
        )
        translation = _measured_vector(
            receipt["resolved_translation_m"],
            width=3,
            label="receipt resolved image-servo translation",
        )
        if (
            current_depth <= 0.0
            or target_depth <= 0.0
            or (
                command.other_view_pixel is None
                and not math.isclose(
                    target_depth,
                    current_depth + command.depth_delta_m,
                    rel_tol=0.0,
                    abs_tol=_DERIVED_VALUE_TOLERANCE,
                )
            )
            or not 0.0 < _vector_norm(translation)
            <= image_servo_trajectory_limit_m(command) + 1e-12
        ):
            raise ValueError("receipt image-servo geometry is inconsistent")
        faux = dict(receipt)
        faux["kind"] = "cartesian_delta"
        for field in (
            "requested_camera",
            "requested_target_pixel",
            "requested_target_role",
            "requested_depth_delta_m",
            "requested_step_m",
            "current_end_effector_pixel",
            "current_end_effector_depth_m",
            "target_depth_m",
            "resolved_translation_m",
        ):
            faux.pop(field)
        for field in STEREO_RECEIPT_FIELDS | {"requested_other_view_camera"}:
            faux.pop(field, None)
        faux["requested_translation_m"] = translation
        faux["requested_rotation_axis_angle_rad"] = [0.0, 0.0, 0.0]
        validate_public_receipt(
            faux,
            require_sealed_rgb=require_sealed_rgb,
            require_torque_baseline=require_torque_baseline,
        )
        output = dict(receipt)
        output.update({
            "requested_target_pixel": list(command.target_pixel),
            "requested_depth_delta_m": command.depth_delta_m,
            "requested_step_m": command.step_m,
            "current_end_effector_pixel": current_pixel,
            "current_end_effector_depth_m": current_depth,
            "target_depth_m": target_depth,
            "resolved_translation_m": translation,
        })
        if command.other_view_pixel is not None:
            output["requested_other_view_pixel"] = list(command.other_view_pixel)
            output["stereo_ray_gap_m"] = float(stereo_gap)
            output["stereo_target_base_m"] = receipt.get("stereo_target_base_m")
        return _canonical_json_snapshot(output)
    if receipt["kind"] == "move_joints":
        step_count = common["step_count"]
        assert isinstance(step_count, int)
        if step_count > MAX_COMMAND_ACTIONS:
            raise ValueError("receipt move_joints step count exceeds its action bound")
        requested_targets = _receipt_numeric_mapping(
            receipt["requested_targets"],
            allowed={*JOINT_NAMES, "gripper"},
            require_nonempty=True,
            label="receipt requested targets",
        )
        if "gripper" in requested_targets and requested_targets["gripper"] != (
            gripper_intent
        ):
            raise ValueError("receipt requested gripper target disagrees with intent")
        resolved_held_dimensions = _receipt_numeric_mapping(
            receipt["resolved_held_dimensions"],
            allowed=set(JOINT_NAMES),
            require_nonempty=False,
            label="receipt resolved held dimensions",
        )
        bounded_endpoint = _measured_vector(
            receipt["bounded_endpoint"],
            width=7,
            label="receipt bounded endpoint",
        )
        qpos_start = common["telemetry_summary"]["arm_joint_position"]["start_rad"]
        realized = common["realized_arm_qpos"]
        assert isinstance(qpos_start, list)
        assert isinstance(realized, list)
        requested_arm_names = set(requested_targets) & set(JOINT_NAMES)
        held_names = set(resolved_held_dimensions)
        if held_names != set(JOINT_NAMES) - requested_arm_names:
            raise ValueError(
                "receipt requested and held dimensions must be disjoint complements"
            )
        for index, name in enumerate(JOINT_NAMES):
            if name in requested_arm_names:
                if not math.isclose(
                    bounded_endpoint[index],
                    requested_targets[name],
                    rel_tol=0.0,
                    abs_tol=_DERIVED_VALUE_TOLERANCE,
                ):
                    raise ValueError(
                        "receipt bounded endpoint disagrees with requested target"
                    )
            elif not (
                math.isclose(
                    resolved_held_dimensions[name],
                    qpos_start[index],
                    rel_tol=0.0,
                    abs_tol=_DERIVED_VALUE_TOLERANCE,
                )
                and math.isclose(
                    bounded_endpoint[index],
                    qpos_start[index],
                    rel_tol=0.0,
                    abs_tol=_DERIVED_VALUE_TOLERANCE,
                )
            ):
                raise ValueError(
                    "receipt held joint start value disagrees with bounded endpoint"
                )
        for index, (joint_value, (lower, upper)) in enumerate(
            zip(bounded_endpoint, JOINT_LIMITS, strict=True)
        ):
            if not lower <= joint_value <= upper:
                raise ValueError(f"receipt bounded endpoint joint{index + 1} drifted")
        maximum_step = _finite(
            receipt["maximum_commanded_step"],
            label="receipt maximum commanded step",
        )
        endpoint_error = _finite(
            receipt["endpoint_error"], label="receipt endpoint error"
        )
        hard_limit_margin = _finite(
            receipt["minimum_hard_limit_margin"],
            label="receipt minimum hard-limit margin",
        )
        if (
            maximum_step < 0.0
            or maximum_step > MAX_JOINT_STEP + _MAX_COMMAND_STEP_RESIDUE_TOLERANCE
            or endpoint_error < 0.0
            or hard_limit_margin < 0.0
        ):
            raise ValueError("receipt move_joints accounting is invalid")
        expected_endpoint_error = max(
            abs(target - actual)
            for target, actual in zip(bounded_endpoint, realized, strict=True)
        )
        _assert_scalar_close(
            endpoint_error,
            expected_endpoint_error,
            label="endpoint error",
        )
        failure_status = receipt["failure_status"]
        if common["accepted"] is True and failure_status is not None:
            raise ValueError("receipt accepted action must not have failure status")
        if common["accepted"] is False and not isinstance(failure_status, str):
            raise ValueError("receipt rejected action must name a failure status")
        return _canonical_json_snapshot({
            "kind": "move_joints",
            **common,
            "requested_targets": requested_targets,
            "resolved_held_dimensions": resolved_held_dimensions,
            "bounded_endpoint": bounded_endpoint,
            "maximum_commanded_step": maximum_step,
            "endpoint_error": endpoint_error,
            "minimum_hard_limit_margin": hard_limit_margin,
            "failure_status": failure_status,
        })  # type: ignore[return-value]
    axis = receipt["axis"]
    if axis not in {"x", "y", "yaw"}:
        raise ValueError("receipt base axis is invalid")
    velocity = _finite(
        receipt["normalized_velocity"], label="receipt base velocity"
    )
    if not 0.0 < abs(velocity) <= 0.5:
        raise ValueError("receipt base velocity is outside its bound")
    step_count = common["step_count"]
    assert isinstance(step_count, int)
    if common["tracking_pause_count"] != 0:
        raise ValueError("receipt base tracking_pause_count must be zero")
    base_motion_step_count = _receipt_int(
        receipt["base_motion_step_count"],
        label="receipt base motion step count",
        minimum=0,
        maximum=step_count,
    )
    if base_motion_step_count != BASE_STEPS or step_count not in {
        BASE_STEPS,
        MIN_GRIPPER_ACTIONS,
    }:
        raise ValueError("receipt base action accounting is invalid")
    remaining_endpoint_error = _finite(
        receipt["remaining_endpoint_error"],
        label="receipt base remaining endpoint error",
    )
    if remaining_endpoint_error < 0.0:
        raise ValueError("receipt base remaining endpoint error must be nonnegative")
    qpos_start = common["telemetry_summary"]["arm_joint_position"]["start_rad"]
    realized = common["realized_arm_qpos"]
    assert isinstance(qpos_start, list)
    assert isinstance(realized, list)
    expected_remaining_endpoint_error = max(
        abs(start - end) for start, end in zip(qpos_start, realized, strict=True)
    )
    _assert_scalar_close(
        remaining_endpoint_error,
        expected_remaining_endpoint_error,
        label="base remaining endpoint error",
    )
    return _canonical_json_snapshot({
        "kind": "base_action",
        **common,
        "axis": axis,
        "normalized_velocity": velocity,
        "base_motion_step_count": base_motion_step_count,
        "remaining_endpoint_error": remaining_endpoint_error,
    })  # type: ignore[return-value]


_LIVE_CAMERA_POSE_FIELDS = {"camera_position_world_m", "camera_xmat_world"}


def _stable_camera_calibration(
    value: object, *, label: str
) -> dict[str, object]:
    """Return identity/intrinsics while excluding live parent-body world pose."""

    if not isinstance(value, Mapping) or not _LIVE_CAMERA_POSE_FIELDS.issubset(value):
        raise ValueError(f"{label} camera calibration drifted")
    return {
        key: item for key, item in value.items() if key not in _LIVE_CAMERA_POSE_FIELDS
    }


def _action_effects(
    execution: Mapping[str, object],
    *,
    before_state: Mapping[str, object],
    after_state: Mapping[str, object],
    before_calibration: Mapping[str, object],
    after_calibration: Mapping[str, object],
) -> dict[str, object]:
    if not {"left", "right"}.issubset(before_calibration) or not {
        "left",
        "right",
    }.issubset(after_calibration):
        raise ValueError("external camera calibration drifted")
    for label in ("left", "right"):
        before_stable = _stable_camera_calibration(
            before_calibration[label], label=label
        )
        after_stable = _stable_camera_calibration(
            after_calibration[label], label=label
        )
        if before_stable != after_stable:
            raise ValueError("external camera calibration drifted")
    if ("wrist" in before_calibration) != ("wrist" in after_calibration):
        raise ValueError("wrist camera calibration drifted")
    if "wrist" in before_calibration:
        before_stable_wrist = _stable_camera_calibration(
            before_calibration["wrist"], label="wrist"
        )
        after_stable_wrist = _stable_camera_calibration(
            after_calibration["wrist"], label="wrist"
        )
        if before_stable_wrist != after_stable_wrist:
            raise ValueError("wrist camera calibration drifted")
    before_qpos = _measured_vector(
        before_state.get("state.arm_joint_position"),
        width=7,
        label="start arm qpos",
    )
    after_qpos = _measured_vector(
        after_state.get("state.arm_joint_position"),
        width=7,
        label="end arm qpos",
    )
    realized = _measured_vector(
        execution.get("realized_arm_qpos"), width=7, label="realized arm qpos"
    )
    if realized != after_qpos:
        raise ValueError("realized arm qpos disagrees with fresh public state")
    telemetry_summary = validate_telemetry_summary(
        execution.get("telemetry_summary")
    )
    qpos_summary = telemetry_summary.get("arm_joint_position")
    if (
        not isinstance(qpos_summary, Mapping)
        or qpos_summary.get("start_rad") != before_qpos
        or qpos_summary.get("end_rad") != after_qpos
    ):
        raise ValueError("telemetry qpos summary disagrees with public state")
    start_position = _measured_vector(
        before_state.get("state.end_effector_position_relative"),
        width=3,
        label="start EEF position",
    )
    end_position = _measured_vector(
        after_state.get("state.end_effector_position_relative"),
        width=3,
        label="end EEF position",
    )
    start_pixels = before_state.get("state.end_effector_external_pixels")
    end_pixels = after_state.get("state.end_effector_external_pixels")
    if (
        execution.get("end_effector_external_pixels_before") != start_pixels
        or execution.get("end_effector_external_pixels_after") != end_pixels
    ):
        raise ValueError("calibrated external pixels disagree with fresh public state")
    start_gripper = _measured_vector(
        before_state.get("state.gripper_qpos"),
        width=2,
        label="start gripper qpos",
    )
    end_gripper = _measured_vector(
        after_state.get("state.gripper_qpos"),
        width=2,
        label="end gripper qpos",
    )
    return {
        "realized_arm_qpos_delta": [
            end - start for start, end in zip(before_qpos, after_qpos, strict=True)
        ],
        "end_effector_pose_delta": {
            "translation_m": [
                end - start
                for start, end in zip(start_position, end_position, strict=True)
            ],
            "rotation_axis_angle_rad": _rotation_axis_angle_delta(
                before_state.get("state.end_effector_rotation_relative"),
                after_state.get("state.end_effector_rotation_relative"),
            ),
        },
        "end_effector_external_pixel_displacement": _pixel_displacement(
            start_pixels, end_pixels
        ),
        "telemetry_summary": telemetry_summary,
        "gripper_residual": {
            "start_qpos": start_gripper,
            "end_qpos": end_gripper,
            "qpos_delta": [
                end - start
                for start, end in zip(start_gripper, end_gripper, strict=True)
            ],
            "measured_end_finger_separation": abs(end_gripper[0] - end_gripper[1]),
        },
        "mean_absolute_rgb_change": None,
    }


def finalize_joint_receipt(
    command: JointCommand,
    mailbox: Mapping[str, object],
    execution: Mapping[str, object],
    *,
    before_state: Mapping[str, object],
    after_state: Mapping[str, object],
    before_calibration: Mapping[str, object],
    after_calibration: Mapping[str, object],
) -> dict[str, object]:
    """Expose only enumerated public-state and executor-accounting evidence."""
    required_execution = {
        "kind",
        "accepted",
        "bounded_endpoint",
        "gripper_intent",
        "step_count",
        "maximum_commanded_step",
        "realized_arm_qpos",
        "endpoint_error",
        "minimum_hard_limit_margin",
        "tracking_pause_count",
        "telemetry_summary",
        "end_effector_external_pixels_before",
        "end_effector_external_pixels_after",
    }
    if set(execution) - {"chassis_hold", "endpoint_tolerance", "waypoint_progress"} != required_execution:
        raise ValueError("joint execution receipt fields drifted")
    if execution.get("kind") != "move_joints":
        raise ValueError("joint execution receipt kind drifted")
    mailbox_fields = {
        "schema",
        "sequence",
        "kind",
        "observation_id",
        "endpoint",
        "explicit_mask",
        "gripper_open",
        "previous_gripper_open",
        "gripper_transition",
        "max_actions",
    }
    if set(mailbox) - {"tracking_mode", "endpoint_tolerance", "waypoints"} != mailbox_fields or (
        mailbox.get("schema") != "robocasa-inspect-joint-command/v1"
        or mailbox.get("kind") != "move_joints"
        or mailbox.get("observation_id") != command.observation_id
        or mailbox.get("tracking_mode", LAG_PAUSE_TRACKING)
        != command.tracking_mode
    ):
        raise ValueError("joint mailbox semantics drifted")
    if mailbox.get("endpoint_tolerance") != command.endpoint_tolerance:
        raise ValueError("joint mailbox endpoint tolerance drifted")
    if execution.get("endpoint_tolerance") != command.endpoint_tolerance:
        raise ValueError("joint execution endpoint tolerance drifted")
    if command.endpoint_tolerance is not None:
        validate_endpoint_tolerance(command.endpoint_tolerance)
    expected_path = ([list(point) for point in command.waypoints] if command.waypoints is not None else None)
    if mailbox.get("waypoints") != expected_path:
        raise ValueError("joint mailbox waypoint path drifted")
    if expected_path is not None:
        progress = validate_waypoint_progress(execution.get("waypoint_progress"))
        if progress["total"] != len(expected_path):
            raise ValueError("joint execution waypoint count drifted")
    elif "waypoint_progress" in execution:
        raise ValueError("joint execution has unrequested waypoint progress")
    endpoint = mailbox.get("endpoint")
    if execution.get("bounded_endpoint") != endpoint:
        raise ValueError("joint execution endpoint drifted")
    step_count = execution.get("step_count")
    if (
        isinstance(step_count, bool)
        or not isinstance(step_count, int)
        or not 1 <= step_count <= int(mailbox["max_actions"])
    ):
        raise ValueError("joint execution step count is invalid")
    previous_gripper_open = _finite(
        mailbox.get("previous_gripper_open"), label="previous gripper intent"
    )
    expected_gripper_intent = _finite(
        mailbox.get("gripper_open"), label="joint gripper intent"
    )
    execution_gripper_intent = _finite(
        execution.get("gripper_intent"), label="joint execution gripper intent"
    )
    gripper_transition = mailbox.get("gripper_transition")
    if (
        not 0.0 <= previous_gripper_open <= 1.0
        or not 0.0 <= expected_gripper_intent <= 1.0
        or type(gripper_transition) is not bool
        or gripper_transition
        != (abs(expected_gripper_intent - previous_gripper_open) > 1e-12)
        or command.targets.get("gripper", previous_gripper_open)
        != expected_gripper_intent
        or execution_gripper_intent != expected_gripper_intent
    ):
        raise ValueError("joint execution gripper intent drifted")
    if gripper_transition and step_count < MIN_GRIPPER_ACTIONS:
        raise ValueError(
            f"gripper transition requires {MIN_GRIPPER_ACTIONS} executed actions"
        )
    explicit_mask = mailbox.get("explicit_mask")
    if not isinstance(explicit_mask, list) or len(explicit_mask) != 7:
        raise ValueError("joint mailbox explicit mask drifted")
    held = {
        name: endpoint[index]
        for index, name in enumerate(JOINT_NAMES)
        if explicit_mask[index] is False
    }
    effects = _action_effects(
        execution,
        before_state=before_state,
        after_state=after_state,
        before_calibration=before_calibration,
        after_calibration=after_calibration,
    )
    receipt = {
        "kind": "move_joints",
        "observation_id": command.observation_id,
        "note": command.note,
        "requested_targets": dict(command.targets),
        "resolved_held_dimensions": held,
        "bounded_endpoint": list(endpoint),
        "gripper_intent": execution_gripper_intent,
        "step_count": step_count,
        "maximum_commanded_step": execution["maximum_commanded_step"],
        "realized_arm_qpos": execution["realized_arm_qpos"],
        "endpoint_error": execution["endpoint_error"],
        "minimum_hard_limit_margin": execution["minimum_hard_limit_margin"],
        "tracking_pause_count": execution["tracking_pause_count"],
        **effects,
        "accepted": execution["accepted"] is True,
        "failure_status": None if execution["accepted"] is True else "rejected",
    }
    if "chassis_hold" in execution:
        receipt["chassis_hold"] = validate_chassis_receipt(execution["chassis_hold"], execution["step_count"])
    if command.endpoint_tolerance is not None:
        receipt["endpoint_tolerance"] = command.endpoint_tolerance
    if command.waypoints is not None:
        receipt["waypoint_progress"] = progress
    if set(receipt) - {"chassis_hold", "endpoint_tolerance", "waypoint_progress"} != PUBLIC_JOINT_RECEIPT_FIELDS:
        raise AssertionError("public joint receipt field set drifted")
    return validate_public_receipt(receipt, require_sealed_rgb=False)


def finalize_cartesian_receipt(
    command: CartesianDeltaCommand,
    mailbox: Mapping[str, object],
    execution: Mapping[str, object],
    *,
    before_state: Mapping[str, object],
    after_state: Mapping[str, object],
    before_calibration: Mapping[str, object],
    after_calibration: Mapping[str, object],
    orientation_weight: float = 1.0,
    max_predicted_translation_m: float | None = None,
) -> dict[str, object]:
    """Bind a Qwen Cartesian request to its deterministic joint resolution."""

    previous_gripper = _finite(
        mailbox.get("previous_gripper_open"),
        label="previous gripper intent",
    )
    resolution = resolve_cartesian_delta(
        command,
        before_state,
        current_gripper=previous_gripper,
        orientation_weight=orientation_weight,
        max_predicted_translation_m=max_predicted_translation_m,
    )
    endpoint = _measured_vector(
        mailbox.get("endpoint"), width=7, label="Cartesian mailbox endpoint"
    )
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        for actual, expected in zip(
            endpoint, resolution.joint_endpoint, strict=True
        )
    ) or mailbox.get("gripper_open") != resolution.gripper_open:
        raise ValueError("Cartesian mailbox disagrees with public-state resolution")
    resolved = JointCommand(
        observation_id=command.observation_id,
        targets={
            **{name: endpoint[index] for index, name in enumerate(JOINT_NAMES)},
            "gripper": resolution.gripper_open,
        },
        note=command.note,
    )
    joint_receipt = finalize_joint_receipt(
        resolved,
        mailbox,
        execution,
        before_state=before_state,
        after_state=after_state,
        before_calibration=before_calibration,
        after_calibration=after_calibration,
    )
    receipt = dict(joint_receipt)
    receipt["kind"] = "cartesian_delta"
    receipt.pop("requested_targets")
    receipt.pop("resolved_held_dimensions")
    receipt.update({
        "requested_translation_m": list(command.translation_m),
        "requested_rotation_axis_angle_rad": list(command.rotation_axis_angle_rad),
        "requested_gripper": command.gripper,
        "derived_joint_endpoint": endpoint,
        "predicted_cartesian_delta": list(resolution.predicted_delta),
        "cartesian_residual_norm": resolution.residual_norm,
    })
    if set(receipt) - {"chassis_hold"} != PUBLIC_CARTESIAN_RECEIPT_FIELDS:
        raise AssertionError("public Cartesian receipt field set drifted")
    return validate_public_receipt(receipt, require_sealed_rgb=False)


def finalize_image_servo_receipt(
    command: ImageServoCommand,
    mailbox: Mapping[str, object],
    execution: Mapping[str, object],
    *,
    before_state: Mapping[str, object],
    after_state: Mapping[str, object],
    before_calibration: Mapping[str, object],
    after_calibration: Mapping[str, object],
) -> dict[str, object]:
    """Bind a Qwen pixel target to the deterministic public-geometry resolution."""

    previous_gripper = _finite(
        mailbox.get("previous_gripper_open"), label="previous gripper intent"
    )
    resolution = resolve_image_servo(
        command,
        before_state,
        before_calibration,
        current_gripper=previous_gripper,
    )
    endpoint = _measured_vector(
        mailbox.get("endpoint"), width=7, label="image-servo mailbox endpoint"
    )
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        for actual, expected in zip(endpoint, resolution.joint_endpoint, strict=True)
    ) or mailbox.get("gripper_open") != resolution.gripper_open:
        raise ValueError(
            "image-servo mailbox disagrees with public-geometry resolution"
        )
    cartesian = CartesianDeltaCommand(
        observation_id=command.observation_id,
        translation_m=resolution.translation_m,
        rotation_axis_angle_rad=(0.0, 0.0, 0.0),
        gripper=command.gripper,
        note=command.note,
    )
    cartesian_receipt = finalize_cartesian_receipt(
        cartesian,
        mailbox,
        execution,
        before_state=before_state,
        after_state=after_state,
        before_calibration=before_calibration,
        after_calibration=after_calibration,
        orientation_weight=IMAGE_SERVO_ORIENTATION_WEIGHT,
        max_predicted_translation_m=(
            image_servo_trajectory_limit_m(command)
            if command.target_role == "articulation_motion"
            else None
        ),
    )
    receipt = dict(cartesian_receipt)
    receipt["kind"] = "image_servo"
    receipt.pop("requested_translation_m")
    receipt.pop("requested_rotation_axis_angle_rad")
    receipt.update({
        "requested_camera": command.camera,
        "requested_target_pixel": list(command.target_pixel),
        "requested_target_role": command.target_role,
        "requested_depth_delta_m": command.depth_delta_m,
        "requested_step_m": command.step_m,
        "current_end_effector_pixel": list(resolution.current_pixel),
        "current_end_effector_depth_m": resolution.current_depth_m,
        "target_depth_m": resolution.target_depth_m,
        "resolved_translation_m": list(resolution.translation_m),
    })
    if command.other_view_pixel is not None:
        receipt["requested_other_view_pixel"] = list(command.other_view_pixel)
        receipt["stereo_ray_gap_m"] = resolution.stereo_ray_gap_m
        receipt["stereo_target_base_m"] = (
            list(resolution.stereo_target_base_m)
            if resolution.stereo_target_base_m is not None
            else None
        )
    if command.other_view_camera is not None:
        receipt["requested_other_view_camera"] = command.other_view_camera
    if set(receipt) - STEREO_RECEIPT_FIELDS - {"requested_other_view_camera", "chassis_hold"} != PUBLIC_IMAGE_SERVO_RECEIPT_FIELDS:
        raise AssertionError("public image-servo receipt field set drifted")
    return validate_public_receipt(receipt, require_sealed_rgb=False)


def _base_receipt(
    command: SimpleCommand,
    mailbox: Mapping[str, object],
    execution: Mapping[str, object],
    *,
    before_state: Mapping[str, object],
    after_state: Mapping[str, object],
    before_calibration: Mapping[str, object],
    after_calibration: Mapping[str, object],
) -> dict[str, object]:
    allowed = {
        "kind",
        "accepted",
        "axis",
        "normalized_velocity",
        "gripper_intent",
        "step_count",
        "base_motion_step_count",
        "realized_arm_qpos",
        "tracking_pause_count",
        "remaining_endpoint_error",
        "telemetry_summary",
        "end_effector_external_pixels_before",
        "end_effector_external_pixels_after",
    }
    if set(execution) - {"chassis_hold"} != allowed or execution.get("kind") != "base_action":
        raise ValueError("base execution receipt fields drifted")
    previous_gripper_open = _finite(
        mailbox.get("previous_gripper_open"), label="previous gripper intent"
    )
    expected_gripper_intent = {
        "open": 1.0,
        "close": 0.0,
        "hold": previous_gripper_open,
    }[command.gripper]
    execution_gripper_intent = _finite(
        execution.get("gripper_intent"), label="base execution gripper intent"
    )
    gripper_transition = mailbox.get("gripper_transition")
    if (
        set(mailbox)
        != {
            "schema",
            "sequence",
            "kind",
            "observation_id",
            "axis",
            "normalized_velocity",
            "gripper_open",
            "previous_gripper_open",
            "gripper_transition",
        }
        or mailbox.get("schema") != "robocasa-inspect-joint-command/v1"
        or mailbox.get("kind") != "base_action"
        or mailbox.get("observation_id") != command.observation_id
        or mailbox.get("axis") != command.axis
        or mailbox.get("normalized_velocity") != command.normalized_velocity
        or mailbox.get("gripper_open") != expected_gripper_intent
        or not 0.0 <= previous_gripper_open <= 1.0
        or type(gripper_transition) is not bool
        or gripper_transition
        != (abs(expected_gripper_intent - previous_gripper_open) > 1e-12)
    ):
        raise ValueError("base mailbox semantics drifted")
    for field, expected in (
        ("axis", command.axis),
        ("normalized_velocity", command.normalized_velocity),
    ):
        if execution.get(field) != expected:
            raise ValueError(f"base execution {field} drifted")
    if execution_gripper_intent != expected_gripper_intent:
        raise ValueError("base execution gripper_intent drifted")
    tracking_pause_count = execution["tracking_pause_count"]
    step_count = execution["step_count"]
    base_motion_step_count = execution["base_motion_step_count"]
    if (
        isinstance(tracking_pause_count, bool)
        or not isinstance(tracking_pause_count, int)
        or isinstance(step_count, bool)
        or not isinstance(step_count, int)
        or isinstance(base_motion_step_count, bool)
        or not isinstance(base_motion_step_count, int)
        or not 0 <= tracking_pause_count <= step_count
        or base_motion_step_count != BASE_STEPS
        or step_count
        != (MIN_GRIPPER_ACTIONS if gripper_transition else BASE_STEPS)
    ):
        raise ValueError("base action accounting is invalid")
    remaining_endpoint_error = _finite(
        execution["remaining_endpoint_error"], label="base remaining endpoint error"
    )
    if remaining_endpoint_error < 0.0:
        raise ValueError("base remaining endpoint error must be nonnegative")
    receipt = {
        "kind": "base_action",
        "observation_id": command.observation_id,
        "accepted": execution.get("accepted") is True,
        "note": command.note,
        "axis": execution["axis"],
        "normalized_velocity": execution["normalized_velocity"],
        "gripper_intent": execution_gripper_intent,
        "step_count": step_count,
        "base_motion_step_count": base_motion_step_count,
        "realized_arm_qpos": execution["realized_arm_qpos"],
        "tracking_pause_count": tracking_pause_count,
        "remaining_endpoint_error": remaining_endpoint_error,
        **_action_effects(
            execution,
            before_state=before_state,
            after_state=after_state,
            before_calibration=before_calibration,
            after_calibration=after_calibration,
        ),
    }
    if "chassis_hold" in execution:
        receipt["chassis_hold"] = validate_chassis_receipt(execution["chassis_hold"], execution["step_count"])
    if set(receipt) - {"chassis_hold"} != PUBLIC_BASE_RECEIPT_FIELDS:
        raise AssertionError("public base receipt field set drifted")
    return validate_public_receipt(receipt, require_sealed_rgb=False)


def _seal_receipt_rgb_change(
    receipt: Mapping[str, object],
    change: Mapping[str, object],
    *,
    source_sequence: int,
    fresh_sequence: int,
) -> dict[str, object]:
    expected = (
        PUBLIC_JOINT_RECEIPT_FIELDS
        if receipt.get("kind") == "move_joints"
        else (
            PUBLIC_CARTESIAN_RECEIPT_FIELDS
            if receipt.get("kind") == "cartesian_delta"
            else (
                PUBLIC_IMAGE_SERVO_RECEIPT_FIELDS
                if receipt.get("kind") == "image_servo"
                else PUBLIC_BASE_RECEIPT_FIELDS
            )
        )
    )
    present = set(receipt) - {"chassis_hold"}
    if receipt.get("kind") == "image_servo" and STEREO_RECEIPT_FIELDS <= present:
        present = present - STEREO_RECEIPT_FIELDS
        if receipt.get("requested_other_view_camera") == "wrist":
            present.discard("requested_other_view_camera")
    if receipt.get("kind") == "move_joints":
        present.discard("endpoint_tolerance")
        present.discard("waypoint_progress")
    if present != expected or receipt.get("mean_absolute_rgb_change") is not None:
        raise ValueError("pending receipt fields drifted")
    if (
        isinstance(source_sequence, bool)
        or not isinstance(source_sequence, int)
        or isinstance(fresh_sequence, bool)
        or not isinstance(fresh_sequence, int)
        or fresh_sequence != source_sequence + 1
    ):
        raise ValueError("fresh observation must immediately follow the command")
    if not isinstance(change, Mapping) or set(change) != {"left", "right", "wrist"}:
        raise ValueError("RGB action effect must contain all three cameras")
    rgb_change = {
        label: _finite(change[label], label=f"{label} RGB change")
        for label in ("left", "right", "wrist")
    }
    if any(value < 0.0 for value in rgb_change.values()):
        raise ValueError("RGB action effect must be nonnegative")
    sealed = dict(receipt)
    sealed["mean_absolute_rgb_change"] = rgb_change
    return validate_public_receipt(sealed, require_sealed_rgb=True)


FORMAT_REPAIR = (
    "Your previous response was invalid. Return exactly one complete allowed JSON "
    "object for this same observation, with no prose or markdown."
)


def _repair_text(reason: str) -> str:
    return (
        "The joint harness rejected the prior response on this same immutable "
        f"observation: {reason[:240]}. Return exactly one safe command. For arm "
        "motion use partial absolute-radian targets inside the stated Panda limits. "
        "Do not repeat the rejected object and do not emit EEF or raw timestep actions."
    )


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _launch_simulator(
    *, task: str, seed: int, run: Path, log_handle: object, action_budget: int
) -> subprocess.Popen[str]:
    release = Path(__file__).resolve().parents[1]
    root = Path("/home/jli/work/robocasa-inspect-official")
    python = root / ".venv/bin/python"
    driver = Path(
        "/home/jli/state/robocasa-inspect-official/nvidia-egl-580.173.02/rootfs"
    )
    state = Path("/home/jli/state/robocasa-inspect-official")
    lib = driver / "usr/lib/x86_64-linux-gnu"
    vendor = driver / "usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    xdg = run / "xdg"
    xdg.mkdir(mode=0o700)
    command = [
        "sudo", "-n", "/usr/bin/unshare", "-n", "--",
        "/usr/bin/setpriv",
        "--reuid=1001", "--regid=1001", "--groups=1001,44,992",
        "--inh-caps=-all", "--ambient-caps=-all", "--bounding-set=-all", "--",
        "/usr/bin/env", "-i",
        "HOME=/home/jli",
        "PATH=/usr/bin:/bin",
        f"PYTHONPATH={release}:{root}:{root / 'robocasa'}:{root / 'robosuite'}",
        f"ROBOCASA_ASSET_ROOT={root / 'robocasa/robocasa/models/assets'}",
        f"ROBOCASA_ASSET_MANIFEST={state / 'assets/content-manifest.json'}",
        f"ROBOCASA_RUN_DIR={run / 'sim'}",
        f"ROBOCASA_CHECKOUT_ROOT={root / 'robocasa'}",
        f"ROBOCASA_CACHE_ROOT={state}",
        "MUJOCO_GL=egl", "PYOPENGL_PLATFORM=egl", "EGL_PLATFORM=surfaceless",
        f"XDG_RUNTIME_DIR={xdg}",
        f"LD_LIBRARY_PATH={lib}:/usr/lib/x86_64-linux-gnu",
        f"__EGL_VENDOR_LIBRARY_FILENAMES={vendor}",
        str(python), "-m", "adaptive.joint_sim_child",
        "--task", task, "--seed", str(seed), "--run-dir", str(run / "sim"),
        "--action-budget", str(action_budget),
    ]
    return subprocess.Popen(
        command, stdout=log_handle, stderr=subprocess.STDOUT, text=True
    )


def _image_servo_alignment_status(
    receipts: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if not receipts or receipts[-1].get("kind") != "image_servo":
        return None
    receipt = receipts[-1]
    camera = receipt.get("requested_camera")
    target = receipt.get("requested_target_pixel")
    displacement = receipt.get("end_effector_external_pixel_displacement")
    if (
        not isinstance(camera, str)
        or not isinstance(target, Sequence)
        or isinstance(target, (str, bytes))
        or len(target) != 2
        or not isinstance(displacement, Mapping)
        or not isinstance(displacement.get(camera), Mapping)
    ):
        return None
    end = displacement[camera].get("end_px")
    if (
        not isinstance(end, Sequence)
        or isinstance(end, (str, bytes))
        or len(end) != 2
    ):
        return None
    try:
        target_pixel = [float(value) for value in target]
        end_pixel = [float(value) for value in end]
    except (TypeError, ValueError):
        return None
    values = target_pixel + end_pixel
    if not all(math.isfinite(value) for value in values):
        return None
    error = math.dist(target_pixel, end_pixel)
    within_grid_cell = error <= 32.0
    return {
        "schema": "robocasa-image-servo-alignment-status/v1",
        "camera": camera,
        "target_pixel": target_pixel,
        "end_effector_pixel": end_pixel,
        "error_px": error,
        "within_one_grid_cell": within_grid_cell,
        "controller_rule": (
            "alignment gate is satisfied: advance now from approach to engage "
            "or pregrasp; do not chase exact pixel equality"
            if within_grid_cell
            else "continue approach toward the selected target"
        ),
    }


def _instruction(
    observation: Mapping[str, object],
    *,
    receipts: list[dict[str, object]],
    change: Mapping[str, float],
    repair: str | None,
    receipt_limit: int = CONTROLLER_RECEIPT_LIMIT,
) -> str:
    if type(receipt_limit) is not int or not 1 <= receipt_limit <= CONTROLLER_RECEIPT_LIMIT:
        raise ValueError("controller receipt limit is invalid")
    if any(receipt.get("mean_absolute_rgb_change") is None for receipt in receipts):
        raise ValueError("controller receipts must be sealed with fresh RGB effects")
    value = {
        "task": observation["instruction"],
        "camera_order": ["left external", "right external", "wrist"],
        "public_state_order": list(observation["public_state"]),
        "recent_receipts": receipts[-receipt_limit:],
        "mean_absolute_rgb_change_since_previous": dict(change),
    }
    image_servo_alignment = _image_servo_alignment_status(receipts)
    if image_servo_alignment is not None:
        value["image_servo_alignment"] = image_servo_alignment
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if repair:
        text += "\n\n" + repair
    return text


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _read_terminal_outcome(run: Path) -> tuple[str, dict[str, object]]:
    raw = json.loads((run / "sim" / "simulator-terminal.json").read_text())
    if not isinstance(raw, Mapping) or set(raw) != {
        "status",
        "sequence",
        "terminal_outcome_sha256",
        "terminal_snapshot_sha256",
        "episode_tmp_empty",
        "simulator_actions",
        "wall_s",
    }:
        raise RuntimeError("finish simulator outcome schema drifted")
    status = raw.get("status")
    sequence = raw.get("sequence")
    actions = raw.get("simulator_actions")
    wall_s = raw.get("wall_s")
    hashes = (
        raw.get("terminal_outcome_sha256"),
        raw.get("terminal_snapshot_sha256"),
    )
    if (
        status not in {"success", "finished_false"}
        or type(sequence) is not int
        or sequence < 0
        or type(actions) is not int
        or actions < 0
        or raw.get("episode_tmp_empty") is not True
        or isinstance(wall_s, bool)
        or not isinstance(wall_s, (int, float))
        or not math.isfinite(float(wall_s))
        or float(wall_s) < 0.0
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        )
    ):
        raise RuntimeError("finish did not produce a sealed official outcome")
    terminal = _canonical_json_snapshot(raw)
    assert isinstance(terminal, dict)
    outcome = {
        "schema": "robocasa-inspect-joint-terminal/v1",
        "status": status,
        "success": status == "success",
        "simulator_terminal": terminal,
        "simulator_terminal_sha256": _canonical_sha256(terminal),
    }
    episode_status = "success" if status == "success" else "policy_finished_false"
    return episode_status, outcome


def _proposal_audit_execution_context(
    public_state: Mapping[str, object],
    camera_calibration: Mapping[str, object],
    *,
    current_gripper: object,
    consumed_actions: object,
    sequence: object,
    action_budget: int = EPISODE_ACTION_BUDGET,
) -> dict[str, object]:
    """Expose only measured validation inputs; never author a command value."""

    if isinstance(consumed_actions, bool) or not isinstance(consumed_actions, int):
        raise ValueError("consumed action count must be an integer")
    if (
        type(action_budget) is not int
        or not 1 <= action_budget <= MAX_DEVELOPMENT_ACTION_BUDGET
    ):
        raise ValueError("proposal action budget is invalid")
    if not 0 <= consumed_actions <= action_budget:
        raise ValueError("consumed action count is outside the episode budget")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("proposal sequence must be a nonnegative integer")
    qpos = public_state.get("state.arm_joint_position")
    if (
        isinstance(qpos, (str, bytes, Mapping))
        or not isinstance(qpos, Sequence)
        or len(qpos) != 7
    ):
        raise ValueError("proposal validation requires seven measured joints")
    measured_qpos = [_finite(value, label="proposal current qpos") for value in qpos]
    gripper = _finite(current_gripper, label="proposal current gripper")
    if not 0.0 <= gripper <= 1.0:
        raise ValueError("proposal current gripper is outside its normalized range")
    return {
        "current_qpos": measured_qpos,
        "current_gripper": gripper,
        "remaining_actions": action_budget - consumed_actions,
        "sequence": sequence,
        "camera_calibration": _canonical_json_snapshot(camera_calibration),
    }


def _proposal_audit_failure_request(
    *,
    decision: int,
    observation_id: object,
    error: Exception,
) -> dict[str, object]:
    return {
        "schema": "robocasa-qwen-proposal-audit-failure/v1",
        "decision": decision,
        "observation_id": observation_id,
        "status": "policy_failed_proposal_audit",
        "failure_class": type(error).__name__,
        "failure_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
        "command": None,
        "mailbox_count": 0,
        "action_count": 0,
        "receipt_count": 0,
    }


def _close_or_terminate_simulator(
    *,
    status: str,
    process: object,
    mailbox_dir: Path,
    sequence: int,
) -> str:
    """Close lifecycle without giving rejected drafts a simulator mailbox."""

    poll = getattr(process, "poll")
    if poll() is not None:
        return "already_exited"
    if status == "policy_failed_proposal_audit" or status == "policy_gave_up":
        getattr(process, "terminate")()
        getattr(process, "wait")(timeout=180)
        return "out_of_band_termination"
    if status != "finish_requested":
        _atomic_json(
            mailbox_dir / f"command-{sequence:06d}.json",
            {
                "schema": "robocasa-inspect-joint-command/v1",
                "sequence": sequence,
                "kind": "close",
            },
        )
        return "mailbox_close"
    return "finish_mailbox"


def run_episode(
    *,
    task: str,
    seed: int,
    run: Path,
    max_decisions: int,
    client_class: type | None = None,
    protocol: str = "proposal",
    action_budget: int = EPISODE_ACTION_BUDGET,
    wall_budget_s: float = EPISODE_WALL_BUDGET_S,
    prompt_variant: str = "baseline",
) -> dict[str, object]:
    """Run one joint-control episode; Qwen remains the sole target selector."""
    from robocasa_inspect.model_client import (
        MalformedResponse,
        QwenClient,
        load_authority,
        verify_process,
    )
    from robocasa_inspect.runner import (
        _image_change,
        _read_images,
        _render_video,
        _wait_process_json,
    )

    # Imported at runtime because critic_protocol shares receipt validators from
    # this module. The exception maps only wrapper rejection to a policy status.
    from .critic_protocol import ProposalAuditFailedClosed

    if not 1 <= max_decisions <= (200 if protocol == "skills" else MAX_DECISIONS):
        raise ValueError("joint runner decision budget is invalid")
    if (
        type(action_budget) is not int
        or not 1 <= action_budget <= (MAX_DEVELOPMENT_ACTION_BUDGET if protocol == "skills" else 900)
    ):
        raise ValueError("joint runner action budget is invalid")
    if (
        isinstance(wall_budget_s, bool)
        or not isinstance(wall_budget_s, (int, float))
        or not math.isfinite(float(wall_budget_s))
        or not 1.0 <= float(wall_budget_s) <= EPISODE_WALL_BUDGET_S
    ):
        raise ValueError("joint runner wall budget is invalid")
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    log_path = run / "simulator.log"
    log_handle = log_path.open("w")
    os.chmod(log_path, 0o600)
    root = Path(__file__).resolve().parents[1]
    system_prompt = load_joint_system_prompt(
        root, protocol=protocol, variant=prompt_variant
    )
    system_prompt_parts_sha256 = joint_system_prompt_parts(
        root, protocol=protocol, variant=prompt_variant
    )
    controller_max_tokens = (
        PROPOSAL_CONTROLLER_MAX_TOKENS
        if protocol == "proposal"
        else CONTROLLER_MAX_TOKENS
    )
    receipt_limit = (
        PROPOSAL_CONTROLLER_RECEIPT_LIMIT
        if protocol == "proposal"
        else CONTROLLER_RECEIPT_LIMIT
    )
    identity_path = Path(os.environ["QWEN_IDENTITY_MANIFEST"])
    attestation_path = Path(os.environ["QWEN_SERVER_ATTESTATION"])
    identity, attestation = load_authority(identity_path, attestation_path)
    token = Path(os.environ["QWEN_API_TOKEN_FILE"]).read_text().strip()
    client_type = QwenClient if client_class is None else client_class
    client = client_type(
        base_url="http://127.0.0.1:8002/v1",
        api_key=token,
        identity=identity,
        attestation=attestation,
    )
    client.verify()
    process = _launch_simulator(
        task=task,
        seed=seed,
        run=run,
        log_handle=log_handle,
        action_budget=action_budget,
    )
    started = time.monotonic()
    receipts: list[dict[str, object]] = []
    requests: list[dict[str, object]] = []
    previous_images: dict[str, bytes] | None = None
    current_gripper = 1.0
    status = "incomplete"
    sequence = 0
    low_level_actions = 0
    physical_commands = 0
    pending: _PendingCommand | None = None
    observation: dict[str, object] | None = None
    terminal_outcome: dict[str, object] | None = None
    simulator_lifecycle = "running"

    def read_and_finalize() -> tuple[
        dict[str, object], dict[str, object] | None, dict[str, bytes] | None, int | None
    ]:
        nonlocal pending, low_level_actions
        value = _wait_process_json(
            run / "sim" / "mailbox" / f"observation-{sequence:06d}.json",
            process=process,
            log_path=log_path,
            timeout_s=180,
        )
        if value.get("sequence") != sequence:
            raise RuntimeError("joint observation sequence mismatch")
        if pending is not None:
            execution = value.get("execution")
            if not isinstance(execution, Mapping):
                raise RuntimeError("fresh joint observation lacks execution evidence")
            (
                command,
                mailbox,
                before_state,
                before_calibration,
                before_images,
                source_sequence,
            ) = pending
            if sequence != source_sequence + 1:
                raise RuntimeError(
                    "fresh joint observation is not the command successor"
                )
            if value.get("observation_id") == command.observation_id:
                raise RuntimeError(
                    "joint action receipt cannot use a stale observation"
                )
            after_state = value.get("public_state")
            after_calibration = value.get("camera_calibration")
            if not isinstance(after_state, Mapping) or not isinstance(
                after_calibration, Mapping
            ):
                raise RuntimeError(
                    "fresh observation lacks public state or calibration"
                )
            if isinstance(command, JointCommand):
                receipt = finalize_joint_receipt(
                    command,
                    mailbox,
                    execution,
                    before_state=before_state,
                    after_state=after_state,
                    before_calibration=before_calibration,
                    after_calibration=after_calibration,
                )
            elif isinstance(command, CartesianDeltaCommand):
                receipt = finalize_cartesian_receipt(
                    command,
                    mailbox,
                    execution,
                    before_state=before_state,
                    after_state=after_state,
                    before_calibration=before_calibration,
                    after_calibration=after_calibration,
                )
            elif isinstance(command, ImageServoCommand):
                receipt = finalize_image_servo_receipt(
                    command,
                    mailbox,
                    execution,
                    before_state=before_state,
                    after_state=after_state,
                    before_calibration=before_calibration,
                    after_calibration=after_calibration,
                )
            else:
                receipt = _base_receipt(
                    command,
                    mailbox,
                    execution,
                    before_state=before_state,
                    after_state=after_state,
                    before_calibration=before_calibration,
                    after_calibration=after_calibration,
                )
            low_level_actions += int(receipt["step_count"])
            if low_level_actions > action_budget:
                raise RuntimeError("joint runner exceeded its action budget")
            pending = None
            return value, receipt, before_images, source_sequence
        return value, None, None, None

    try:
        for decision in range(max_decisions):
            if time.monotonic() - started > wall_budget_s:
                status = "policy_failed_wall_budget"
                break
            observation, pending_receipt, receipt_images, source_sequence = (
                read_and_finalize()
            )
            images = _read_images(observation)
            change = _image_change(previous_images, images)
            if pending_receipt is not None:
                assert receipt_images is not None and source_sequence is not None
                pending_receipt = _seal_receipt_rgb_change(
                    pending_receipt,
                    _image_change(receipt_images, images),
                    source_sequence=source_sequence,
                    fresh_sequence=int(observation["sequence"]),
                )
                receipts.append(pending_receipt)
            arguments = {
                "observation_id": str(observation["observation_id"]),
                "system_prompt": system_prompt,
                "instruction": _instruction(
                    observation,
                    receipts=receipts,
                    change=change,
                    repair=None,
                    receipt_limit=receipt_limit,
                ),
                "public_state": observation["public_state"],
                "images": images,
                "response_schema": CONTROLLER_RESPONSE_SCHEMA,
                "max_tokens": controller_max_tokens,
            }
            if protocol in {"proposal", "skills"}:
                arguments["proposal_audit_context"] = _proposal_audit_execution_context(
                    observation["public_state"],
                    observation["camera_calibration"],
                    current_gripper=current_gripper,
                    consumed_actions=low_level_actions,
                    sequence=sequence,
                    action_budget=action_budget,
                )
            if protocol == "skills":
                arguments["proposal_audit_context"]["last_execution"] = pending_receipt
            format_errors: list[dict[str, object]] = []
            retries_used = 0
            try:
                response = client.complete(**arguments)
            except ProposalAuditFailedClosed as error:
                status = "policy_failed_proposal_audit"
                requests.append(_proposal_audit_failure_request(
                    decision=decision,
                    observation_id=observation["observation_id"],
                    error=error,
                ))
                break
            except MalformedResponse as error:
                format_errors.append(error.evidence)
                retries_used = 1
                retry_arguments = dict(arguments)
                retry_arguments["instruction"] = (
                    str(arguments["instruction"]) + "\n\n" + FORMAT_REPAIR
                )
                try:
                    response = client.complete(**retry_arguments)
                except ProposalAuditFailedClosed as audit_error:
                    status = "policy_failed_proposal_audit"
                    requests.append(_proposal_audit_failure_request(
                        decision=decision,
                        observation_id=observation["observation_id"],
                        error=audit_error,
                    ))
                    break
                except MalformedResponse as retry_error:
                    format_errors.append(retry_error.evidence)
                    status = "policy_failed_invalid_json"
                    requests.append({
                        "decision": decision,
                        "observation_id": observation["observation_id"],
                        "format_errors": format_errors,
                        "repair_count": retries_used,
                        "post_repair_status": "invalid_json",
                    })
                    break

            initial_command = response.command
            initial_evidence = response.evidence
            repair_reason: str | None = None
            repair_command: Mapping[str, object] | None = None
            repair_evidence: Mapping[str, object] | None = None
            try:
                mailbox, command = prepare_joint_mailbox(
                    initial_command,
                    source="controller",
                    allow_waypoints=(protocol == "skills"),
                    observation_id=str(observation["observation_id"]),
                    current_qpos=observation["public_state"][
                        "state.arm_joint_position"
                    ],
                    current_gripper=current_gripper,
                    public_state=observation["public_state"],
                    camera_calibration=observation["camera_calibration"],
                    remaining_actions=action_budget - low_level_actions,
                    sequence=sequence,
                )
            except ValueError as error:
                repair_reason = str(error)
                if retries_used:
                    status = "policy_failed_invalid_repair"
                    requests.append(_invalid_closed_command_request(
                        decision=decision,
                        observation_id=observation["observation_id"],
                        command=initial_command,
                        evidence=initial_evidence,
                        format_errors=format_errors,
                        repair_count=retries_used,
                        repair_reason=repair_reason,
                        repair_command=None,
                        repair_evidence=None,
                        error=error,
                    ))
                    break
                retries_used = 1
                repair_arguments = dict(arguments)
                repair_arguments["instruction"] = _instruction(
                    observation,
                    receipts=receipts,
                    change=change,
                    repair=_repair_text(repair_reason),
                    receipt_limit=receipt_limit,
                )
                try:
                    repaired = client.complete(**repair_arguments)
                    repair_command = repaired.command
                    repair_evidence = repaired.evidence
                    mailbox, command = prepare_joint_mailbox(
                        repaired.command,
                        source="controller",
                        allow_waypoints=(protocol == "skills"),
                        observation_id=str(observation["observation_id"]),
                        current_qpos=observation["public_state"][
                            "state.arm_joint_position"
                        ],
                        current_gripper=current_gripper,
                        public_state=observation["public_state"],
                        camera_calibration=observation["camera_calibration"],
                        remaining_actions=(
                            action_budget - low_level_actions
                        ),
                        sequence=sequence,
                    )
                except ProposalAuditFailedClosed as audit_error:
                    status = "policy_failed_proposal_audit"
                    requests.append(_proposal_audit_failure_request(
                        decision=decision,
                        observation_id=observation["observation_id"],
                        error=audit_error,
                    ))
                    break
                except (MalformedResponse, ValueError) as retry_error:
                    status = "policy_failed_invalid_repair"
                    requests.append(_invalid_closed_command_request(
                        decision=decision,
                        observation_id=observation["observation_id"],
                        command=initial_command,
                        evidence=initial_evidence,
                        format_errors=format_errors,
                        repair_count=retries_used,
                        repair_reason=repair_reason,
                        repair_command=repair_command,
                        repair_evidence=repair_evidence,
                        error=retry_error,
                    ))
                    break

            mailbox_snapshot = (
                None if command.kind == "give_up" else _canonical_json_snapshot(mailbox)
            )
            requests.append({
                "decision": decision,
                "observation_id": observation["observation_id"],
                "command": initial_command,
                "evidence": initial_evidence,
                "format_errors": format_errors,
                "repair_count": retries_used,
                "repair_reason": repair_reason,
                "repair_command": repair_command,
                "repair_evidence": repair_evidence,
                "post_repair_command": _command_dict(command),
                "mailbox": mailbox_snapshot,
                "mailbox_sha256": (
                    _canonical_sha256(mailbox_snapshot)
                    if mailbox_snapshot is not None
                    else None
                ),
            })
            if command.kind == "give_up":
                status = "policy_gave_up"
                terminal_outcome = {
                    "schema": "robocasa-qwen-approved-give-up/v1",
                    "status": "approved_no_simulator_effect",
                }
                break
            if command.kind == "finish":
                _atomic_json(
                    run / "sim" / "mailbox" / f"command-{sequence:06d}.json",
                    mailbox,
                )
                status = "finish_requested"
                break
            _atomic_json(
                run / "sim" / "mailbox" / f"command-{sequence:06d}.json",
                mailbox,
            )
            public_state = observation.get("public_state")
            camera_calibration = observation.get("camera_calibration")
            if not isinstance(public_state, Mapping) or not isinstance(
                camera_calibration, Mapping
            ):
                raise TypeError("command observation lacks public state or calibration")
            pending = (
                command,
                mailbox,
                dict(public_state),
                dict(camera_calibration),
                dict(images),
                sequence,
            )
            physical_commands += 1
            if isinstance(command, JointCommand):
                current_gripper = float(mailbox["gripper_open"])
            else:
                current_gripper = float(mailbox["gripper_open"])
            previous_images = images
            sequence += 1
        else:
            status = "policy_failed_decision_budget"

        if pending is not None and process.poll() is None:
            observation, pending_receipt, receipt_images, source_sequence = (
                read_and_finalize()
            )
            assert pending_receipt is not None
            assert receipt_images is not None and source_sequence is not None
            final_images = _read_images(observation)
            receipts.append(
                _seal_receipt_rgb_change(
                    pending_receipt,
                    _image_change(receipt_images, final_images),
                    source_sequence=source_sequence,
                    fresh_sequence=int(observation["sequence"]),
                )
            )
        simulator_lifecycle = _close_or_terminate_simulator(
            status=status,
            process=process,
            mailbox_dir=run / "sim" / "mailbox",
            sequence=sequence,
        )
        return_code = process.poll()
        if return_code is None:
            return_code = process.wait(timeout=180)
        if status == "finish_requested":
            if return_code != 0:
                raise RuntimeError(f"joint simulator exited {return_code} during finish")
            status, terminal_outcome = _read_terminal_outcome(run)
        if (
            return_code != 0
            and status.startswith("policy_")
            and simulator_lifecycle != "out_of_band_termination"
        ):
            raise RuntimeError(f"joint simulator exited {return_code}")
        verify_process(attestation)
        video = run / f"{task}-seed{seed}-{status}.mp4"
        frame_count = _render_video(run / "sim", video)
        result = {
            "schema": "robocasa-inspect-joint-episode/v1",
            "task": task,
            "seed": seed,
            "status": status,
            "success": status == "success",
            "decisions": physical_commands,
            "model_decisions": len(requests),
            "action_chunks": physical_commands,
            "simulator_steps": low_level_actions,
            "requests": requests,
            "receipts": receipts,
            "terminal_outcome": terminal_outcome,
            "simulator_lifecycle": simulator_lifecycle,
            "frame_sets": frame_count,
            "video": str(video),
            "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "wall_s": time.monotonic() - started,
            "snapshot_digest": identity.get("snapshot_digest"),
            "served_model_id": attestation.get("served_model_id"),
            "model_identity_manifest_path": str(identity_path),
            "model_identity_manifest_sha256": hashlib.sha256(
                identity_path.read_bytes()
            ).hexdigest(),
            "server_attestation_path": str(attestation_path),
            "server_attestation_sha256": hashlib.sha256(
                attestation_path.read_bytes()
            ).hexdigest(),
            "system_prompt_sha256": hashlib.sha256(
                system_prompt.encode("utf-8")
            ).hexdigest(),
            "prompt_variant": prompt_variant,
            "system_prompt_parts_sha256": system_prompt_parts_sha256,
        }
        _atomic_json(run / "result.json", result)
        return result
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        log_handle.close()
        client.close()
