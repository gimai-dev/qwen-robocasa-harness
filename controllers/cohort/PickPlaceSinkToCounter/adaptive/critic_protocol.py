"""Frozen advisory-critic contract for direct RoboCasa Inspect control."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import cast

from .joint_protocol import JOINT_NAMES
from .joint_runner import (
    BASE_STEPS,
    COFFEE_CONTROL_CONTACT_WAYPOINTS,
    PROPOSAL_CONTROLLER_MAX_TOKENS,
    PUBLIC_BASE_RECEIPT_FIELDS,
    PUBLIC_CARTESIAN_RECEIPT_FIELDS,
    PUBLIC_IMAGE_SERVO_RECEIPT_FIELDS,
    PUBLIC_JOINT_RECEIPT_FIELDS,
    controller_control_contact_correction_schema_for_observation,
    controller_control_contact_path_schema_for_observation,
    controller_control_finish_schema_for_observation,
    controller_control_preclose_schema_for_observation,
    controller_control_press_schema_for_observation,
    controller_control_retreat_schema_for_observation,
    controller_control_standing_convergence_schema_for_observation,
    controller_engage_ready_schema_for_observation,
    controller_response_schema_for_observation,
    controller_wrist_roll_schema_for_observation,
    validate_public_receipt,
)
from .panda_embodiment import validate_public_state

CRITIC_TIMEOUT_S = 30
CRITIC_MAX_TOKENS = 384
CRITIC_MAX_ATTEMPTS_PER_TASK = 3
CRITIC_COOLDOWN_COMMANDS = 2
STALL_TRANSLATION_M = 0.005
STALL_ROTATION_DEG = 2.0
STALL_GRIPPER_QPOS = 0.002
# Absolute separation after an empty source close is about 0.0024 m; this
# contact threshold is distinct from the per-finger motion threshold above.
SOURCE_CONTACT_MIN_SEPARATION = 0.0032
# The seed-7 toaster pull closed at 0.0038 m with both views aligned: thin bars
# sit only ~1.5 mm above the empty value (0.0024), so contact starts at 0.0032.
ARTICULATED_CONTACT_MIN_SEPARATION = 0.0032
ARTICULATED_WRIST_ROLL_SETTLE_TOLERANCE_RAD = 0.05
IMAGE_SERVO_GRID_CELL_PX = 32.0
IMAGE_SERVO_CLOSE_TOLERANCE_PX = 8.0
STALL_ARM_JOINT_RAD = 0.01
PRIOR_FULL_RUN_WALL_S = 7_849.0
OPTIMIZATION_STARTED_AT = "2026-09-03T21:00:00Z"
OPTIMIZATION_MAX_HOURS = 72.0
PRIOR_MEAN_DECISIONS_PER_TASK = 12

DIRECT_COMMAND_KINDS = {
    "move_joints", "cartesian_delta", "image_servo", "base_action", "finish", "give_up",
}
PHYSICAL_COMMAND_KINDS = {
    "move_joints", "cartesian_delta", "image_servo", "base_action",
}
ADVISORY_ENUMS = {
    "diagnosis": (
        "tracking_lag",
        "tracking_stall",
        "visual_misalignment",
        "possible_load_or_contact",
        "gripper_uncertainty",
        "progress",
    ),
    "affected_region": (
        "shoulder",
        "elbow",
        "wrist",
        "gripper",
        "base",
        "camera",
    ),
    "evidence": (
        "joint_tracking",
        "joint_velocity",
        "applied_torque",
        "end_effector_wrench",
        "gripper_state",
        "external_rgb",
        "action_receipt",
    ),
    "suggested_correction": (
        "observe",
        "retreat",
        "retry_intent",
        "reduce_motion",
        "relocalize",
        "verify_gripper",
        "continue",
        "change_approach",
    ),
    "confidence": ("low", "medium", "high"),
}


def receipt_requested_gripper_intent(
    receipt: Mapping[str, object],
) -> str | None:
    """Normalize the explicit public gripper intent across receipt kinds."""

    requested = receipt.get("requested_gripper")
    if requested in {"open", "hold", "close"}:
        return str(requested)
    if receipt.get("kind") != "move_joints":
        return None
    targets = receipt.get("requested_targets")
    target = targets.get("gripper") if isinstance(targets, Mapping) else None
    if isinstance(target, bool) or not isinstance(target, (int, float)):
        return None
    numeric = float(target)
    if not math.isfinite(numeric):
        return None
    if numeric == 0.0:
        return "close"
    if numeric == 1.0:
        return "open"
    return None


def _receipt_gripper_motion_is_settled(receipt: Mapping[str, object]) -> bool:
    residual = receipt.get("gripper_residual")
    delta = residual.get("qpos_delta") if isinstance(residual, Mapping) else None
    return bool(
        isinstance(delta, Sequence)
        and not isinstance(delta, (str, bytes))
        and len(delta) == 2
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and abs(float(value)) <= STALL_GRIPPER_QPOS
            for value in delta
        )
    )


def articulated_contact_is_confirmed(receipt: Mapping[str, object]) -> bool:
    """Return whether contact survived a post-close hold or actuation receipt."""

    residual = receipt.get("gripper_residual")
    separation = (
        residual.get("measured_end_finger_separation")
        if isinstance(residual, Mapping)
        else None
    )
    if (
        receipt_requested_gripper_intent(receipt) not in {"hold", "close"}
        or isinstance(separation, bool)
        or not isinstance(separation, (int, float))
        or not math.isfinite(float(separation))
        or float(separation) <= ARTICULATED_CONTACT_MIN_SEPARATION
    ):
        return False
    if receipt.get("kind") == "move_joints":
        return (
            receipt_requested_gripper_intent(receipt) == "close"
            and _receipt_gripper_motion_is_settled(receipt)
        )
    return not (
        receipt.get("kind") == "image_servo"
        and receipt.get("requested_gripper") == "close"
        and receipt.get("requested_target_role") in {None, "fixture_handle"}
    )


def articulated_contact_is_provisional(receipt: Mapping[str, object]) -> bool:
    """Return whether only the initial close exposed a handle obstruction."""

    residual = receipt.get("gripper_residual")
    separation = (
        residual.get("measured_end_finger_separation")
        if isinstance(residual, Mapping)
        else None
    )
    initial_close = bool(
        (
            receipt.get("kind") == "image_servo"
            and receipt.get("requested_gripper") == "close"
            and receipt.get("requested_target_role") in {None, "fixture_handle"}
        )
        or (
            receipt.get("kind") == "move_joints"
            and receipt_requested_gripper_intent(receipt) == "close"
            and not _receipt_gripper_motion_is_settled(receipt)
        )
    )
    return bool(
        initial_close
        and not isinstance(separation, bool)
        and isinstance(separation, (int, float))
        and math.isfinite(float(separation))
        and float(separation) > ARTICULATED_CONTACT_MIN_SEPARATION
    )


def articulated_handle_insertion_receipt(receipt: Mapping[str, object]) -> bool:
    """Return whether a receipt is an open positive servo toward a handle."""

    depth = receipt.get("requested_depth_delta_m")
    return bool(
        receipt.get("kind") == "image_servo"
        and receipt.get("requested_target_role") == "fixture_handle"
        and receipt.get("requested_gripper") == "open"
        and not isinstance(depth, bool)
        and isinstance(depth, (int, float))
        and math.isfinite(float(depth))
        and float(depth) > 0.0
    )


OPEN_FINGER_OBSTRUCTION_MAX_SEPARATION_M = 0.075


def articulated_open_finger_obstruction(receipt: Mapping[str, object]) -> bool:
    """Return whether an open insertion ended with the fingers pushed together."""

    if not articulated_handle_insertion_receipt(receipt):
        return False
    residual = receipt.get("gripper_residual")
    separation = (
        residual.get("measured_end_finger_separation")
        if isinstance(residual, Mapping)
        else None
    )
    return bool(
        not isinstance(separation, bool)
        and isinstance(separation, (int, float))
        and math.isfinite(float(separation))
        and ARTICULATED_CONTACT_MIN_SEPARATION
        < float(separation)
        < OPEN_FINGER_OBSTRUCTION_MAX_SEPARATION_M
    )


APPROACH_STALL_MIN_PROGRESS_M = 0.006
APPROACH_STALL_RECEIPTS = 2
TOASTER_STALL_CLOSE_MAX_PIXEL_ERROR_PX = 20.0


def _receipt_translation_norm(receipt: Mapping[str, object]) -> float | None:
    delta = receipt.get("end_effector_pose_delta")
    translation = delta.get("translation_m") if isinstance(delta, Mapping) else None
    if (
        not isinstance(translation, Sequence)
        or isinstance(translation, (str, bytes))
        or len(translation) != 3
    ):
        return None
    try:
        values = [float(item) for item in translation]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in values):
        return None
    return math.sqrt(sum(item * item for item in values))


def _servo_pixel_error(receipt: Mapping[str, object]) -> float | None:
    camera = receipt.get("requested_camera")
    target = receipt.get("requested_target_pixel")
    displacements = receipt.get("end_effector_external_pixel_displacement")
    record = (
        displacements.get(camera)
        if isinstance(displacements, Mapping) and isinstance(camera, str)
        else None
    )
    end_pixel = record.get("end_px") if isinstance(record, Mapping) else None
    if any(
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        for value in (target, end_pixel)
    ):
        return None
    try:
        error = math.dist(
            [float(v) for v in cast(Sequence[object], target)],
            [float(v) for v in cast(Sequence[object], end_pixel)],
        )
    except (TypeError, ValueError):
        return None
    return error if math.isfinite(error) else None


def articulated_approach_stall_status(
    recent_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Detect an open handle servo that no longer moves the grip site.

    The last ``APPROACH_STALL_RECEIPTS`` receipts must be open ``image_servo``
    commands toward the same camera and (within the close tolerance) the same
    pixel, each realizing less than ``APPROACH_STALL_MIN_PROGRESS_M`` of
    grip-site translation while the grip-site pixel is still outside the
    close tolerance. That is a physically blocked approach, not a pixel error.
    """

    window = [
        item for item in recent_receipts[-APPROACH_STALL_RECEIPTS:]
        if isinstance(item, Mapping)
    ]
    if len(window) < APPROACH_STALL_RECEIPTS:
        return None
    camera = window[-1].get("requested_camera")
    target = window[-1].get("requested_target_pixel")
    if (
        not isinstance(camera, str)
        or not isinstance(target, Sequence)
        or isinstance(target, (str, bytes))
        or len(target) != 2
    ):
        return None
    norms: list[float] = []
    for receipt in window:
        other_target = receipt.get("requested_target_pixel")
        if (
            receipt.get("kind") != "image_servo"
            or receipt.get("requested_gripper") != "open"
            or receipt.get("requested_camera") != camera
            or not isinstance(other_target, Sequence)
            or isinstance(other_target, (str, bytes))
            or len(other_target) != 2
        ):
            return None
        try:
            shift = math.dist(
                [float(v) for v in target], [float(v) for v in other_target]
            )
        except (TypeError, ValueError):
            return None
        norm = _receipt_translation_norm(receipt)
        if shift > IMAGE_SERVO_CLOSE_TOLERANCE_PX or norm is None:
            return None
        norms.append(norm)
    pixel_error = _servo_pixel_error(window[-1])
    if pixel_error is None or pixel_error <= IMAGE_SERVO_CLOSE_TOLERANCE_PX:
        return None
    if any(norm >= APPROACH_STALL_MIN_PROGRESS_M for norm in norms):
        return None
    return {
        "stalled": True,
        "camera": camera,
        "target_pixel": [float(target[0]), float(target[1])],
        "realized_translation_m": [round(norm, 4) for norm in norms],
        "pixel_error_px": round(pixel_error, 1),
        "min_progress_m": APPROACH_STALL_MIN_PROGRESS_M,
    }


def toaster_stall_close_probe_due(
    task: str | None,
    approach_stall: Mapping[str, object] | None,
    immediate_prior_receipt: Mapping[str, object] | None,
) -> bool:
    """Allow one cheap close where a near-pull toaster stall can catch the edge."""

    pixel_error = (
        approach_stall.get("pixel_error_px")
        if isinstance(approach_stall, Mapping)
        else None
    )
    return bool(
        task == "OpenToasterOvenDoor"
        and isinstance(approach_stall, Mapping)
        and approach_stall.get("stalled") is True
        and not isinstance(pixel_error, bool)
        and isinstance(pixel_error, (int, float))
        and math.isfinite(float(pixel_error))
        and float(pixel_error) <= TOASTER_STALL_CLOSE_MAX_PIXEL_ERROR_PX
        and isinstance(immediate_prior_receipt, Mapping)
        and articulated_handle_insertion_receipt(immediate_prior_receipt)
    )


def articulated_handle_insertion_is_ready(receipt: Mapping[str, object]) -> bool:
    """Return whether an open insertion ended within the handle close tolerance."""

    if not articulated_handle_insertion_receipt(receipt):
        return False
    if articulated_open_finger_obstruction(receipt):
        return True
    camera = receipt.get("requested_camera")
    target = receipt.get("requested_target_pixel")
    displacements = receipt.get("end_effector_external_pixel_displacement")
    camera_displacement = (
        displacements.get(camera)
        if isinstance(displacements, Mapping) and isinstance(camera, str)
        else None
    )
    end_pixel = (
        camera_displacement.get("end_px")
        if isinstance(camera_displacement, Mapping)
        else None
    )
    if any(
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        for value in (target, end_pixel)
    ):
        return False
    try:
        error = math.dist(
            [float(value) for value in cast(Sequence[object], target)],
            [float(value) for value in cast(Sequence[object], end_pixel)],
        )
    except (TypeError, ValueError):
        return False
    return math.isfinite(error) and error <= IMAGE_SERVO_CLOSE_TOLERANCE_PX


ADVISORY_EVIDENCE_MAX_ITEMS = 3

PROPOSAL_AUDIT_ENUMS = {
    "verdict": ("approve", "revise"),
    "contradiction": (
        "none",
        "stale_or_missing_evidence",
        "visual_alignment_unverified",
        "motion_effect_mismatch",
        "tracking_not_settled",
        "grasp_unverified",
        "transport_unverified",
        "release_unverified",
        "actuation_unverified",
        "terminal_unverified",
        "stagnation",
        "safety_bound_risk",
        "milestone_order_violation",
    ),
    "evidence": (
        "external_rgb",
        "wrist_rgb",
        "joint_tracking",
        "joint_velocity",
        "applied_torque",
        "end_effector_wrench",
        "gripper_state",
        "jacobian_projection",
        "action_receipt",
        "milestone_history",
    ),
    "suggested_correction": (
        "observe",
        "wait_for_settle",
        "revise_alignment",
        "revise_approach",
        "verify_gripper",
        "verify_actuation",
        "verify_transport",
        "verify_release",
        "revise_within_bounds",
        "continue_milestone",
        "verify_goal",
    ),
    "confidence": ("low", "medium", "high"),
}
PROPOSAL_AUDIT_EVIDENCE_MAX_ITEMS = 4
PROPOSAL_MAX_REVISIONS_PER_OBSERVATION = 9
MILESTONE_CONTEXT_MARKER = "\n\nCONTROLLER_MILESTONE_CONTEXT_V1:\n"
MILESTONE_GRAPHS = {
    "grasp_place": (
        "observe",
        "approach",
        "pregrasp",
        "grasp",
        "transport",
        "release",
        "verify_goal",
    ),
    "articulated": ("observe", "approach", "engage", "actuate", "verify_goal"),
    "control": ("observe", "approach", "engage", "actuate", "verify_goal"),
}
_MILESTONE_NOTE = re.compile(r"^milestone=([a-z_]+);(?:\s|$)")


def milestone_context_packet(
    family: str,
    milestone_history: Sequence[str],
) -> dict[str, object]:
    """Expose the closed graph state without adding motor authority."""

    graph = MILESTONE_GRAPHS.get(family)
    if graph is None:
        raise ValueError("unknown task family")
    if isinstance(milestone_history, (str, bytes)):
        raise ValueError("milestone history must be a sequence")
    closed: list[str] = []
    for milestone in milestone_history:
        if not isinstance(milestone, str):
            raise ValueError("milestone history contains a non-string value")
        validate_milestone_transition(family, milestone, closed)
        closed.append(milestone)
    if not closed:
        current: str | None = None
        allowed = [graph[0]]
    else:
        current = closed[-1]
        index = graph.index(current)
        allowed = [current]
        if index + 1 < len(graph):
            allowed.append(graph[index + 1])
    packet: dict[str, object] = {
        "allowed_next_milestones": allowed,
        "current_closed_milestone": current,
        "family": family,
    }
    if family == "articulated" and current == "approach":
        packet["edge_action_contract"] = {
            "engage": {
                "depth_delta_m": "nonzero",
                "gripper": "close",
                "kind": "image_servo",
            },
            "open_image_servo_claims": "approach",
        }
    return packet


def wrist_roll_revision_required(
    previous_record: Mapping[str, object] | None,
    observation_id: str,
) -> bool:
    """Identify the qualitative handle-alignment correction schema."""

    if not isinstance(previous_record, Mapping):
        return False
    previous_draft = previous_record.get("draft")
    audit = previous_record.get("audit")
    evidence = audit.get("evidence") if isinstance(audit, Mapping) else None
    return bool(
        previous_record.get("status") == "rejected_by_critic"
        and previous_record.get("observation_id") == observation_id
        and previous_record.get("contradiction")
        == "visual_alignment_unverified"
        and isinstance(evidence, list)
        and "wrist_rgb" in evidence
        and isinstance(previous_draft, Mapping)
        and previous_draft.get("kind") == "image_servo"
        and previous_draft.get("target_role") == "fixture_handle"
        and previous_draft.get("gripper") == "close"
    )


class ProposalAuditFailedClosed(RuntimeError):
    """A draft cannot cross the execution boundary without a valid audit."""


class ProposalAuditExhausted(ProposalAuditFailedClosed):
    """The initial draft and all allowed same-observation revisions were rejected."""


def _strict_object(properties: dict[str, object]) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": sorted(properties),
    }


ADVISORY_SCHEMA: dict[str, object] = _strict_object({
    name: {"enum": list(values)}
    for name, values in ADVISORY_ENUMS.items()
    if name != "evidence"
} | {
    "evidence": {
        "type": "array",
        "items": {"enum": list(ADVISORY_ENUMS["evidence"])},
        "maxItems": ADVISORY_EVIDENCE_MAX_ITEMS,
    },
})

PROPOSAL_AUDIT_SCHEMA: dict[str, object] = _strict_object({
    name: {"enum": list(values)}
    for name, values in PROPOSAL_AUDIT_ENUMS.items()
    if name != "evidence"
} | {
    "evidence": {
        "type": "array",
        "items": {"enum": list(PROPOSAL_AUDIT_ENUMS["evidence"])},
        "maxItems": PROPOSAL_AUDIT_EVIDENCE_MAX_ITEMS,
    },
})

PROPOSAL_AUDIT_CONFIG = {
    "schema": "robocasa-qwen-proposal-audit-config/v1",
    "timeout_s": CRITIC_TIMEOUT_S,
    "max_tokens": CRITIC_MAX_TOKENS,
    "max_revisions_per_observation": PROPOSAL_MAX_REVISIONS_PER_OBSERVATION,
    "critic_schema": PROPOSAL_AUDIT_SCHEMA,
    "milestone_graphs": MILESTONE_GRAPHS,
    "same_observation_revisions": True,
    "critic_failure": "fail_closed",
    "numeric_command_authority": "qwen_controller_only",
}
PROPOSAL_AUDIT_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(
        PROPOSAL_AUDIT_CONFIG,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


CRITIC_CONFIG = {
    "schema": "robocasa-qwen-advisory-config/v1",
    "timeout_s": CRITIC_TIMEOUT_S,
    "max_tokens": CRITIC_MAX_TOKENS,
    "max_attempts_per_task": CRITIC_MAX_ATTEMPTS_PER_TASK,
    "cooldown_commands": CRITIC_COOLDOWN_COMMANDS,
    "first_action_review": True,
    "first_repeat_or_stall_exempt_from_cooldown": True,
    "late_episode_fraction": 0.5,
    "stall_translation_m": STALL_TRANSLATION_M,
    "stall_rotation_deg": STALL_ROTATION_DEG,
    "stall_gripper_qpos": STALL_GRIPPER_QPOS,
    "stall_arm_joint_rad": STALL_ARM_JOINT_RAD,
    "advisory_enums": ADVISORY_ENUMS,
    "advisory_evidence_max_items": ADVISORY_EVIDENCE_MAX_ITEMS,
    "advisory_consumption": "exactly_one_controller_request_before_repair_retry",
    "failed_attempts_consume_budget": True,
}
CRITIC_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(CRITIC_CONFIG, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def canonical_json_copy(value: object) -> object:
    """Return a mutation-safe JSON snapshot, rejecting NaN/Inf values."""

    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def strict_canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def image_hashes(images: Mapping[str, bytes]) -> dict[str, str]:
    return {name: hashlib.sha256(value).hexdigest() for name, value in images.items()}


_RGB_LABELS = {"left", "right", "wrist"}


def validate_advisory(value: Mapping[str, object]) -> dict[str, object]:
    if set(value) != set(ADVISORY_SCHEMA["properties"]):
        raise ValueError("advisory fields drifted")
    output: dict[str, object] = {}
    for name in ("diagnosis", "affected_region", "suggested_correction", "confidence"):
        label = value.get(name)
        if not isinstance(label, str) or label not in ADVISORY_ENUMS[name]:
            raise ValueError(f"advisory {name} is outside its closed vocabulary")
        output[name] = label
    evidence = value.get("evidence")
    if (
        not isinstance(evidence, list)
        or len(evidence) > ADVISORY_EVIDENCE_MAX_ITEMS
        or any(
            not isinstance(label, str) or label not in ADVISORY_ENUMS["evidence"]
            for label in evidence
        )
        or len(set(evidence)) != len(evidence)
    ):
        raise ValueError("advisory evidence is outside its closed vocabulary")
    output["evidence"] = list(evidence)
    return output


def validate_proposal_audit(value: Mapping[str, object]) -> dict[str, object]:
    """Validate the critic's non-executable, closed proposal verdict."""

    if set(value) != set(PROPOSAL_AUDIT_SCHEMA["properties"]):
        raise ValueError("proposal audit fields drifted")
    output: dict[str, object] = {}
    for name in ("verdict", "contradiction", "suggested_correction", "confidence"):
        label = value.get(name)
        if not isinstance(label, str) or label not in PROPOSAL_AUDIT_ENUMS[name]:
            raise ValueError(f"proposal audit {name} is outside its closed vocabulary")
        output[name] = label
    evidence = value.get("evidence")
    if (
        not isinstance(evidence, list)
        or len(evidence) > PROPOSAL_AUDIT_EVIDENCE_MAX_ITEMS
        or any(
            not isinstance(label, str)
            or label not in PROPOSAL_AUDIT_ENUMS["evidence"]
            for label in evidence
        )
    ):
        raise ValueError("proposal audit evidence must be a closed list")
    # Evidence is semantically a set. Structured decoding can still emit the
    # same closed label twice, so canonicalize that harmless representation
    # drift before sealing and validating the audit trail.
    output["evidence"] = list(dict.fromkeys(evidence))
    verdict = output["verdict"]
    contradiction = output["contradiction"]
    if verdict == "approve" and contradiction != "none":
        raise ValueError("proposal audit approve requires contradiction none")
    if verdict == "revise" and contradiction == "none":
        raise ValueError("proposal audit revise requires a non-none contradiction")
    return output


def milestone_from_note(note: object) -> str:
    if not isinstance(note, str):
        raise ValueError("controller note lacks a milestone declaration")
    match = _MILESTONE_NOTE.match(note)
    if match is None:
        raise ValueError("controller note lacks a closed milestone declaration")
    return match.group(1)


def validate_milestone_transition(
    family: str,
    claimed_milestone: str,
    history: Sequence[object],
) -> str:
    """Accept only the current milestone or the next edge in a closed graph."""

    graph = MILESTONE_GRAPHS.get(family)
    if graph is None:
        raise ValueError("controller family has no closed milestone graph")
    if not isinstance(claimed_milestone, str) or claimed_milestone not in graph:
        raise ValueError("claimed milestone is outside the family graph")
    validated_history: list[str] = []
    for raw in history:
        if not isinstance(raw, str) or raw not in graph:
            raise ValueError("milestone history is outside the family graph")
        index = graph.index(raw)
        if validated_history:
            previous_index = graph.index(validated_history[-1])
            if index < previous_index:
                raise ValueError("milestone history regressed")
            if index > previous_index + 1:
                raise ValueError("milestone history skipped an edge")
        elif index != 0:
            raise ValueError("milestone history skipped its initial edge")
        validated_history.append(raw)
    claimed_index = graph.index(claimed_milestone)
    if not validated_history:
        if claimed_index != 0:
            raise ValueError("claimed milestone would skip the initial edge")
        return claimed_milestone
    current_index = graph.index(validated_history[-1])
    if claimed_index < current_index:
        raise ValueError("claimed milestone would regress")
    if claimed_index > current_index + 1:
        raise ValueError("claimed milestone would skip an edge")
    return claimed_milestone


def articulated_contact_recovery_status(
    claimed_milestone: str,
    draft: Mapping[str, object],
    immediate_prior_receipt: Mapping[str, object] | None,
    recent_receipts: Sequence[Mapping[str, object]] = (),
) -> dict[str, bool] | None:
    """Compare an articulated empty-handle retry with its public receipt."""

    if claimed_milestone not in {"engage", "actuate"} or not isinstance(
        immediate_prior_receipt, Mapping
    ):
        return None
    residual = immediate_prior_receipt.get("gripper_residual")
    separation = (
        residual.get("measured_end_finger_separation")
        if isinstance(residual, Mapping)
        else None
    )
    comparable = (
        receipt_requested_gripper_intent(immediate_prior_receipt)
        in {"hold", "close"}
        and not isinstance(separation, bool)
        and isinstance(separation, (int, float))
        and math.isfinite(float(separation))
    )
    recovery_required = bool(
        comparable and float(separation) <= ARTICULATED_CONTACT_MIN_SEPARATION
    )

    def direction(value: object) -> int | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return 1 if numeric > 0.0 else -1 if numeric < 0.0 else 0

    image_retry = draft.get("kind") == "image_servo" and draft.get(
        "gripper"
    ) == "close"
    insertion_retry = bool(
        draft.get("kind") == "image_servo"
        and draft.get("target_role") == "fixture_handle"
        and draft.get("gripper") == "open"
        and direction(draft.get("depth_delta_m")) == 1
    )
    command_kind_changed = bool(
        image_retry and immediate_prior_receipt.get("kind") != "image_servo"
    )
    target_changed = False
    if image_retry and immediate_prior_receipt.get("kind") == "image_servo":
        current_role = draft.get("target_role")
        previous_role = immediate_prior_receipt.get("requested_target_role")
        target_changed = bool(
            (
                isinstance(current_role, str)
                and isinstance(previous_role, str)
                and current_role != previous_role
            )
            or draft.get("camera")
            != immediate_prior_receipt.get("requested_camera")
        )
        current_target = draft.get("target_pixel")
        previous_target = immediate_prior_receipt.get("requested_target_pixel")
        if (
            not target_changed
            and isinstance(current_target, Sequence)
            and not isinstance(current_target, (str, bytes))
            and len(current_target) == 2
            and isinstance(previous_target, Sequence)
            and not isinstance(previous_target, (str, bytes))
            and len(previous_target) == 2
        ):
            try:
                target_shift = abs(
                    float(current_target[0]) - float(previous_target[0])
                ) + abs(float(current_target[1]) - float(previous_target[1]))
            except (TypeError, ValueError):
                target_shift = 0.0
            target_changed = bool(
                math.isfinite(target_shift)
                and target_shift >= IMAGE_SERVO_GRID_CELL_PX
            )
    previous_depth_direction = direction(
        immediate_prior_receipt.get("requested_depth_delta_m")
    )
    current_depth_direction = direction(draft.get("depth_delta_m"))
    depth_direction_changed = bool(
        image_retry
        and previous_depth_direction is not None
        and current_depth_direction is not None
        and previous_depth_direction != current_depth_direction
    )
    prior_was_servo = immediate_prior_receipt.get("kind") == "image_servo"
    candidate_camera = (
        immediate_prior_receipt.get("requested_camera")
        if prior_was_servo
        else draft.get("camera")
    )
    candidate_target = (
        immediate_prior_receipt.get("requested_target_pixel")
        if prior_was_servo
        else draft.get("target_pixel")
    )
    history = list(recent_receipts) or [immediate_prior_receipt]
    failed_depth_directions: set[int] = set()
    for receipt in reversed(history):
        receipt_residual = receipt.get("gripper_residual")
        receipt_separation = (
            receipt_residual.get("measured_end_finger_separation")
            if isinstance(receipt_residual, Mapping)
            else None
        )
        receipt_direction = direction(receipt.get("requested_depth_delta_m"))
        if (
            receipt.get("kind") == "image_servo"
            and receipt.get("requested_gripper") == "close"
            and receipt.get("requested_camera") == candidate_camera
            and receipt.get("requested_target_pixel") == candidate_target
            and not isinstance(receipt_separation, bool)
            and isinstance(receipt_separation, (int, float))
            and math.isfinite(float(receipt_separation))
            and float(receipt_separation) <= ARTICULATED_CONTACT_MIN_SEPARATION
            and receipt_direction in {-1, 1}
        ):
            failed_depth_directions.add(receipt_direction)
    positive_depth_failed = 1 in failed_depth_directions
    negative_depth_failed = -1 in failed_depth_directions
    target_exhausted = positive_depth_failed and negative_depth_failed
    motion: list[object] = []
    if draft.get("kind") == "cartesian_delta":
        for key in ("translation_m", "rotation_axis_angle_rad"):
            raw = draft.get(key)
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                motion.extend(raw)
    retreat_selected = (
        draft.get("kind") == "cartesian_delta"
        and draft.get("gripper") == "open"
        and any(direction(value) not in {None, 0} for value in motion)
    )
    reengagement_after_retreat = (
        image_retry
        and current_depth_direction in {-1, 1}
        and immediate_prior_receipt.get("kind") == "cartesian_delta"
        and immediate_prior_receipt.get("requested_gripper") == "open"
    )
    return {
        "required": recovery_required,
        "command_kind_changed": command_kind_changed,
        "target_changed": bool(target_changed),
        "depth_direction_changed": depth_direction_changed,
        "retreat_selected": retreat_selected,
        "reengagement_after_retreat": reengagement_after_retreat,
        "positive_depth_failed": positive_depth_failed,
        "negative_depth_failed": negative_depth_failed,
        "target_exhausted": target_exhausted,
        "strategy_changed": bool(
            target_changed
            or insertion_retry
            or retreat_selected
            or reengagement_after_retreat
            or (
                not target_exhausted
                and (command_kind_changed or depth_direction_changed)
            )
        ),
    }


def articulated_wrist_roll_tracking_status(
    receipt: Mapping[str, object] | None,
) -> dict[str, float | bool] | None:
    """Compare a Qwen-authored joint7 recovery target with its realized endpoint."""

    if not isinstance(receipt, Mapping) or receipt.get("kind") != "move_joints":
        return None
    targets = receipt.get("requested_targets")
    realized = receipt.get("realized_arm_qpos")
    if (
        not isinstance(targets, Mapping)
        or set(targets) != {"gripper", "joint7"}
        or targets.get("gripper") != 1.0
        or not isinstance(realized, Sequence)
        or isinstance(realized, (str, bytes))
        or len(realized) != 7
    ):
        return None
    requested_joint7 = targets.get("joint7")
    realized_joint7 = realized[6]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in (requested_joint7, realized_joint7)
    ):
        return None
    requested_value = float(cast(float, requested_joint7))
    realized_value = float(cast(float, realized_joint7))
    error = abs(requested_value - realized_value)
    status: dict[str, float | bool] = {
        "requested_joint7_rad": requested_value,
        "realized_joint7_rad": realized_value,
        "absolute_error_rad": error,
        "settled": error <= ARTICULATED_WRIST_ROLL_SETTLE_TOLERANCE_RAD,
    }
    summary = receipt.get("telemetry_summary")
    velocity = summary.get("arm_joint_velocity") if isinstance(summary, Mapping) else None
    end_velocity = velocity.get("end_rad_s") if isinstance(velocity, Mapping) else None
    if (
        isinstance(end_velocity, Sequence)
        and not isinstance(end_velocity, (str, bytes))
        and len(end_velocity) == 7
        and not isinstance(end_velocity[6], bool)
        and isinstance(end_velocity[6], (int, float))
        and math.isfinite(float(end_velocity[6]))
    ):
        absolute_velocity = abs(float(end_velocity[6]))
        status["absolute_joint7_velocity_rad_s"] = absolute_velocity
        status["settled"] = bool(
            status["settled"] and absolute_velocity < STALL_ARM_JOINT_RAD
        )
    return status


def articulated_failed_contact_targets(
    recent_receipts: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return failed handle pixels from the current wrist-orientation epoch."""

    directions_by_target: dict[tuple[str, tuple[float, float]], set[int]] = {}
    latest_open_target: tuple[tuple[str, tuple[float, float]], int] | None = None
    epoch_start = 0
    for index, receipt in enumerate(recent_receipts):
        roll = articulated_wrist_roll_tracking_status(receipt)
        if isinstance(roll, Mapping) and roll.get("settled") is True:
            epoch_start = index + 1
    for receipt in recent_receipts[epoch_start:]:
        residual = receipt.get("gripper_residual")
        separation = (
            residual.get("measured_end_finger_separation")
            if isinstance(residual, Mapping)
            else None
        )
        camera = receipt.get("requested_camera")
        pixel = receipt.get("requested_target_pixel")
        depth = receipt.get("requested_depth_delta_m")
        valid_pixel_target = bool(
            camera in {"left", "right"}
            and isinstance(pixel, Sequence)
            and not isinstance(pixel, (str, bytes))
            and len(pixel) == 2
            and all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                for value in pixel
            )
            and not isinstance(depth, bool)
            and isinstance(depth, (int, float))
            and math.isfinite(float(depth))
            and float(depth) != 0.0
        )
        if (
            receipt.get("kind") == "image_servo"
            and receipt.get("requested_target_role") == "fixture_handle"
            and receipt.get("requested_gripper") == "open"
            and valid_pixel_target
        ):
            latest_open_target = (
                (str(camera), (float(pixel[0]), float(pixel[1]))),
                1 if float(depth) > 0.0 else -1,
            )
        if (
            receipt.get("kind") == "move_joints"
            and receipt_requested_gripper_intent(receipt) == "close"
            and latest_open_target is not None
            and not isinstance(separation, bool)
            and isinstance(separation, (int, float))
            and math.isfinite(float(separation))
            and float(separation) <= ARTICULATED_CONTACT_MIN_SEPARATION
        ):
            key, direction = latest_open_target
            directions_by_target.setdefault(key, set()).add(direction)
        if (
            receipt.get("kind") != "image_servo"
            or receipt.get("requested_gripper") != "close"
            or not valid_pixel_target
            or isinstance(separation, bool)
            or not isinstance(separation, (int, float))
            or not math.isfinite(float(separation))
            or float(separation) > ARTICULATED_CONTACT_MIN_SEPARATION
        ):
            continue
        key = (str(camera), (float(pixel[0]), float(pixel[1])))
        directions_by_target.setdefault(key, set()).add(
            1 if float(depth) > 0.0 else -1
        )
    return [
        {"camera": camera, "target_pixel": [pixel[0], pixel[1]]}
        for (camera, pixel), directions in sorted(directions_by_target.items())
        if directions
    ]


def cartesian_motion_axis_set(command: Mapping[str, object]) -> list[str]:
    """Return the active Cartesian axes of a draft or public receipt."""

    active: list[str] = []
    for field, requested_field, labels in (
        (
            "translation_m",
            "requested_translation_m",
            ("translation_x", "translation_y", "translation_z"),
        ),
        (
            "rotation_axis_angle_rad",
            "requested_rotation_axis_angle_rad",
            ("rotation_x", "rotation_y", "rotation_z"),
        ),
    ):
        raw = command.get(requested_field, command.get(field))
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes))
            or len(raw) != 3
        ):
            return []
        for label, item in zip(labels, raw, strict=True):
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
            ):
                return []
            if float(item) != 0.0:
                active.append(label)
    return active


def articulated_failed_actuation_axis_sets(
    recent_receipts: Sequence[Mapping[str, object]],
) -> list[list[str]]:
    """Return motion-axis sets that lost articulated-handle contact."""

    failed: list[list[str]] = []
    for receipt in recent_receipts:
        residual = receipt.get("gripper_residual")
        separation = (
            residual.get("measured_end_finger_separation")
            if isinstance(residual, Mapping)
            else None
        )
        axes = cartesian_motion_axis_set(receipt)
        if (
            receipt.get("kind") == "cartesian_delta"
            and receipt.get("requested_gripper") in {"hold", "close"}
            and axes
            and not isinstance(separation, bool)
            and isinstance(separation, (int, float))
            and math.isfinite(float(separation))
            and float(separation) <= ARTICULATED_CONTACT_MIN_SEPARATION
            and axes not in failed
        ):
            failed.append(axes)
    return failed


def source_grasp_approach_target(
    recent_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Find the executed source approach without crossing intervening arm motion."""

    for receipt in reversed(recent_receipts):
        if receipt.get("kind") == "image_servo":
            if receipt.get("requested_target_role") not in {None, "source_object"}:
                return None
            return {
                "camera": receipt.get("requested_camera"),
                "target_pixel": receipt.get("requested_target_pixel"),
                "other_view_pixel": receipt.get("requested_other_view_pixel"),
                **({"other_view_camera": receipt["requested_other_view_camera"]} if receipt.get("requested_other_view_camera") is not None else {}),
                "depth_delta_m": receipt.get("requested_depth_delta_m"),
            }
        targets = receipt.get("requested_targets")
        if not (
            receipt.get("kind") == "move_joints"
            and isinstance(targets, Mapping)
            and set(targets) == {"gripper"}
        ):
            return None
    return None


def source_grasp_recovery_target(
    recent_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Retain the latest empty source contact through open recovery motions."""

    for index in range(len(recent_receipts) - 1, -1, -1):
        receipt = recent_receipts[index]
        if receipt_requested_gripper_intent(receipt) != "close":
            continue
        residual = receipt.get("gripper_residual")
        separation = (
            residual.get("measured_end_finger_separation")
            if isinstance(residual, Mapping) else None
        )
        if (
            isinstance(separation, bool)
            or not isinstance(separation, (int, float))
            or not math.isfinite(float(separation))
        ):
            continue
        if float(separation) > SOURCE_CONTACT_MIN_SEPARATION:
            return None
        target = source_grasp_approach_target(recent_receipts[:index + 1])
        if target is not None:
            return {
                **target,
                "measured_end_finger_separation": float(separation),
            }
    return None


def validate_milestone_action_semantics(
    family: str,
    claimed_milestone: str,
    draft: Mapping[str, object],
    *,
    task: str | None = None,
    milestone_history: Sequence[object] = (),
    immediate_prior_receipt: Mapping[str, object] | None = None,
    recent_receipts: Sequence[Mapping[str, object]] = (),
    articulated_forbidden_targets: Sequence[Mapping[str, object]] = (),
    articulated_failed_axes: Sequence[Sequence[str]] = (),
    cross_view_depth_mismatch: bool = False,
    approach_stall: Mapping[str, object] | None = None,
) -> None:
    """Require the controller to author the physical edge it claims."""

    if (
        family == "articulated"
        and claimed_milestone == "actuate"
        and isinstance(immediate_prior_receipt, Mapping)
        and articulated_contact_is_provisional(immediate_prior_receipt)
    ):
        targets = draft.get("targets")
        confirms_without_arm_motion = bool(
            draft.get("kind") == "move_joints"
            and isinstance(targets, Mapping)
            and set(targets) == {"gripper"}
            and targets.get("gripper") == 0.0
        )
        if not confirms_without_arm_motion:
            raise ValueError(
                "first articulated obstruction requires stationary contact "
                "confirmation"
            )

    if draft.get("target_role") == "articulation_motion":
        contact_preserved = bool(
            family == "articulated"
            and claimed_milestone == "actuate"
            and isinstance(immediate_prior_receipt, Mapping)
            and articulated_contact_is_confirmed(immediate_prior_receipt)
        )
        if not contact_preserved:
            raise ValueError(
                "articulation-motion image servo requires sealed articulated contact"
            )

    if family == "articulated" and claimed_milestone in {"engage", "actuate"}:
        incomplete_roll = articulated_wrist_roll_tracking_status(
            immediate_prior_receipt
        )
        if (
            claimed_milestone == "engage"
            and isinstance(incomplete_roll, Mapping)
            and incomplete_roll.get("settled") is False
        ):
            continuation_targets = draft.get("targets")
            continues_roll = bool(
                draft.get("kind") == "move_joints"
                and isinstance(continuation_targets, Mapping)
                and set(continuation_targets) == {"gripper", "joint7"}
                and continuation_targets.get("gripper") == 1.0
                and continuation_targets.get("joint7")
                == incomplete_roll.get("requested_joint7_rad")
            )
            if not continues_roll:
                raise ValueError(
                    "articulated engage must continue incomplete joint7 "
                    "orientation target before reinsertion"
                )
        prior_residual = (
            immediate_prior_receipt.get("gripper_residual")
            if isinstance(immediate_prior_receipt, Mapping)
            else None
        )
        prior_separation = (
            prior_residual.get("measured_end_finger_separation")
            if isinstance(prior_residual, Mapping)
            else None
        )
        entering_actuate = bool(
            claimed_milestone == "actuate"
            and milestone_history
            and milestone_history[-1] == "engage"
        )
        sealed_contact_before_actuate = bool(
            isinstance(immediate_prior_receipt, Mapping)
            and receipt_requested_gripper_intent(immediate_prior_receipt)
            in {"hold", "close"}
            and not isinstance(prior_separation, bool)
            and isinstance(prior_separation, (int, float))
            and math.isfinite(float(prior_separation))
            and float(prior_separation) > ARTICULATED_CONTACT_MIN_SEPARATION
        )
        if entering_actuate and not sealed_contact_before_actuate:
            raise ValueError(
                "articulated actuation requires sealed contact before leaving engage"
            )
        prior_close_contact = bool(
            isinstance(immediate_prior_receipt, Mapping)
            and immediate_prior_receipt.get("kind") == "image_servo"
            and immediate_prior_receipt.get("requested_gripper") == "close"
            and not isinstance(prior_separation, bool)
            and isinstance(prior_separation, (int, float))
            and math.isfinite(float(prior_separation))
            and float(prior_separation) > ARTICULATED_CONTACT_MIN_SEPARATION
        )
        if claimed_milestone == "engage" and prior_close_contact:
            raise ValueError(
                "articulated handle contact already established; advance to actuation"
            )
        recovery = articulated_contact_recovery_status(
            claimed_milestone,
            draft,
            immediate_prior_receipt,
            recent_receipts,
        )
        engage_recovery_retreat = bool(
            claimed_milestone == "engage"
            and recovery is not None
            and recovery["required"]
            and recovery["retreat_selected"]
        )
        engage_recovery_view_move = bool(
            claimed_milestone == "engage"
            and articulated_forbidden_targets
            and isinstance(immediate_prior_receipt, Mapping)
            and immediate_prior_receipt.get("requested_gripper") == "open"
            and draft.get("kind") == "cartesian_delta"
            and draft.get("gripper") == "open"
        )
        orientation_targets = draft.get("targets")
        orientation_failure_count = 1 if task == "OpenToasterOvenDoor" else 2
        orientation_recovery_due = bool(
            claimed_milestone == "engage"
            and len(articulated_forbidden_targets) >= orientation_failure_count
            and isinstance(immediate_prior_receipt, Mapping)
            and immediate_prior_receipt.get("kind") == "cartesian_delta"
            and immediate_prior_receipt.get("requested_gripper") == "open"
        )
        orientation_recovery = bool(
            draft.get("kind") == "move_joints"
            and isinstance(orientation_targets, Mapping)
            and set(orientation_targets) == {"gripper", "joint7"}
            and orientation_targets.get("gripper") == 1.0
            and not isinstance(orientation_targets.get("joint7"), bool)
            and isinstance(orientation_targets.get("joint7"), (int, float))
            and math.isfinite(float(cast(float, orientation_targets["joint7"])))
        )
        if orientation_recovery_due and not orientation_recovery:
            raise ValueError(
                "articulated engage requires open joint7 orientation recovery "
                f"after {orientation_failure_count} empty handle target(s)"
            )
        insertion_ready = bool(
            isinstance(immediate_prior_receipt, Mapping)
            and articulated_handle_insertion_is_ready(immediate_prior_receipt)
        )
        if claimed_milestone == "engage" and insertion_ready:
            targets = draft.get("targets")
            stationary_close = bool(
                draft.get("kind") == "move_joints"
                and isinstance(targets, Mapping)
                and set(targets) == {"gripper"}
                and targets.get("gripper") == 0.0
            )
            if cross_view_depth_mismatch:
                depth = draft.get("depth_delta_m")
                same_camera = draft.get("camera") == cast(
                    Mapping[str, object], immediate_prior_receipt
                ).get("requested_camera")
                depth_servo = bool(
                    draft.get("kind") == "image_servo"
                    and draft.get("target_role") == "fixture_handle"
                    and draft.get("gripper") == "open"
                    and (
                        not same_camera
                        or (
                            not isinstance(depth, bool)
                            and isinstance(depth, (int, float))
                            and math.isfinite(float(depth))
                            and float(depth) != 0.0
                        )
                    )
                )
                if not depth_servo:
                    raise ValueError(
                        "articulated insertion is off the handle in the other "
                        "external view; author an open image_servo that corrects "
                        "depth before any stationary close"
                    )
            elif not stationary_close:
                raise ValueError(
                    "articulated open insertion requires stationary gripper close"
                )
        elif (
            claimed_milestone == "engage"
            and not engage_recovery_retreat
            and not engage_recovery_view_move
            and not orientation_recovery
        ):
            stalled = bool(
                isinstance(approach_stall, Mapping)
                and approach_stall.get("stalled") is True
            )
            if stalled:
                stall_targets = draft.get("targets")
                stall_camera = approach_stall.get("camera")
                stall_target = approach_stall.get("target_pixel")
                repeated_servo = False
                draft_target = draft.get("target_pixel")
                if (
                    draft.get("kind") == "image_servo"
                    and draft.get("camera") == stall_camera
                    and isinstance(draft_target, Sequence)
                    and not isinstance(draft_target, (str, bytes))
                    and isinstance(stall_target, Sequence)
                    and len(draft_target) == 2
                ):
                    try:
                        repeated_servo = math.dist(
                            [float(v) for v in draft_target],
                            [float(v) for v in cast(Sequence[object], stall_target)],
                        ) <= IMAGE_SERVO_CLOSE_TOLERANCE_PX
                    except (TypeError, ValueError):
                        repeated_servo = False
                if repeated_servo:
                    raise ValueError(
                        "approach stalled: the grip site no longer moves toward "
                        "this target; change the approach geometry instead of "
                        "repeating the servo"
                    )
                reorientation = bool(
                    draft.get("kind") == "move_joints"
                    and isinstance(stall_targets, Mapping)
                    and any(key.startswith("joint") for key in stall_targets)
                    and stall_targets.get("gripper", 1.0) == 1.0
                )
                open_retreat = bool(
                    draft.get("kind") == "cartesian_delta"
                    and draft.get("gripper") == "open"
                )
                toaster_close_probe = bool(
                    toaster_stall_close_probe_due(
                        task, approach_stall, immediate_prior_receipt
                    )
                    and draft.get("kind") == "move_joints"
                    and isinstance(stall_targets, Mapping)
                    and set(stall_targets) == {"gripper"}
                    and stall_targets.get("gripper") == 0.0
                )
                if reorientation or open_retreat or toaster_close_probe:
                    return
            depth = draft.get("depth_delta_m")
            open_insertion = bool(
                draft.get("kind") == "image_servo"
                and draft.get("target_role") == "fixture_handle"
                and draft.get("gripper") == "open"
                and not isinstance(depth, bool)
                and isinstance(depth, (int, float))
                and math.isfinite(float(depth))
                and float(depth) > 0.0
            )
            if not open_insertion:
                raise ValueError(
                    "articulated engage requires open insertion before stationary close"
                )
        if (
            claimed_milestone in {"engage", "actuate"}
            and draft.get("kind") == "image_servo"
            and draft.get("target_role") == "fixture_handle"
            and draft.get("gripper") in {"open", "close"}
        ):
            camera = draft.get("camera")
            target = draft.get("target_pixel")
            if (
                isinstance(camera, str)
                and isinstance(target, Sequence)
                and not isinstance(target, (str, bytes))
                and len(target) == 2
            ):
                for exhausted in articulated_forbidden_targets:
                    old_target = exhausted.get("target_pixel")
                    if (
                        exhausted.get("camera") != camera
                        or not isinstance(old_target, Sequence)
                        or isinstance(old_target, (str, bytes))
                        or len(old_target) != 2
                    ):
                        continue
                    try:
                        target_shift = abs(
                            float(target[0]) - float(old_target[0])
                        ) + abs(float(target[1]) - float(old_target[1]))
                    except (TypeError, ValueError):
                        continue
                    if (
                        math.isfinite(target_shift)
                        and target_shift < IMAGE_SERVO_GRID_CELL_PX
                    ):
                        prefix = (
                            "empty handle engagement"
                            if claimed_milestone == "engage"
                            else "actuation contact recovery"
                        )
                        raise ValueError(
                            f"{prefix} target must move at least one grid cell "
                            "from an exhausted target"
                        )
        contact_established = (
            claimed_milestone == "actuate"
            and isinstance(immediate_prior_receipt, Mapping)
            and receipt_requested_gripper_intent(immediate_prior_receipt)
            in {"hold", "close"}
            and not isinstance(prior_separation, bool)
            and isinstance(prior_separation, (int, float))
            and math.isfinite(float(prior_separation))
            and float(prior_separation) > ARTICULATED_CONTACT_MIN_SEPARATION
        )
        failed_axis_sets = [list(item) for item in articulated_failed_axes]
        for item in articulated_failed_actuation_axis_sets(recent_receipts):
            if item not in failed_axis_sets:
                failed_axis_sets.append(item)
        if (
            contact_established
            and draft.get("kind") == "cartesian_delta"
            and cartesian_motion_axis_set(draft) in failed_axis_sets
        ):
            raise ValueError(
                "articulated actuation must change motion axes after contact loss"
            )
        if (
            contact_established
            and draft.get("kind") == "image_servo"
            and draft.get("gripper") == "close"
            and (
                articulated_contact_is_confirmed(immediate_prior_receipt)
                or (
                    draft.get("camera")
                    == immediate_prior_receipt.get("requested_camera")
                    and draft.get("target_pixel")
                    == immediate_prior_receipt.get("requested_target_pixel")
                )
            )
        ):
            raise ValueError(
                "articulated handle contact already established; actuate before "
                "another close servo"
            )
        if (
            claimed_milestone == "actuate"
            and isinstance(immediate_prior_receipt, Mapping)
            and immediate_prior_receipt.get("requested_gripper") == "open"
            and draft.get("kind") == "cartesian_delta"
            and draft.get("gripper") in {"hold", "close"}
        ):
            raise ValueError(
                "articulated actuation requires re-engagement after open retreat"
            )
        if (
            recovery is not None
            and recovery["target_exhausted"]
            and not recovery["target_changed"]
            and not recovery["retreat_selected"]
            and not recovery["reengagement_after_retreat"]
        ):
            if claimed_milestone == "engage":
                raise ValueError(
                    "empty handle engagement target exhausted; change target "
                    "pixel or retreat"
                )
            raise ValueError(
                "actuation contact target exhausted; change target pixel or retreat"
            )
        if recovery is not None and recovery["required"] and not recovery[
            "strategy_changed"
        ]:
            if claimed_milestone == "engage":
                raise ValueError(
                    "empty handle engagement retry must change the image-servo "
                    "target or depth direction, or retreat"
                )
            raise ValueError(
                "actuation lost contact retry must change the image-servo "
                "target or depth direction, or retreat"
            )

    if family != "grasp_place":
        return
    if claimed_milestone == "release":
        kind = draft.get("kind")
        if kind == "move_joints":
            targets = draft.get("targets")
            raw_gripper = (
                targets.get("gripper") if isinstance(targets, Mapping) else None
            )
            opens_gripper = (
                not isinstance(raw_gripper, bool)
                and isinstance(raw_gripper, (int, float))
                and math.isfinite(float(raw_gripper))
                and float(raw_gripper) == 1.0
            )
        else:
            opens_gripper = draft.get("gripper") == "open"
        if not opens_gripper:
            raise ValueError(
                "release milestone requires a controller-authored open gripper"
            )
    receipt = immediate_prior_receipt
    kind = receipt.get("kind") if isinstance(receipt, Mapping) else None
    if kind == "move_joints" and isinstance(receipt, Mapping):
        targets = receipt.get("requested_targets")
        raw_gripper = targets.get("gripper") if isinstance(targets, Mapping) else None
        requested_close = (
            not isinstance(raw_gripper, bool)
            and isinstance(raw_gripper, (int, float))
            and math.isfinite(float(raw_gripper))
            and float(raw_gripper) == 0.0
        )
    else:
        requested_close = (
            isinstance(receipt, Mapping)
            and receipt.get("requested_gripper") == "close"
        )
    residual = receipt.get("gripper_residual") if isinstance(receipt, Mapping) else None
    separation = (
        residual.get("measured_end_finger_separation")
        if isinstance(residual, Mapping)
        else None
    )
    source_receipts = (
        [*recent_receipts, receipt] if isinstance(receipt, Mapping) else recent_receipts
    )
    recovery_target = source_grasp_recovery_target(source_receipts)
    stationary_source_close = (
        draft.get("kind") == "move_joints"
        and draft.get("targets") == {"gripper": 0.0}
    )
    contact_target = (
        source_grasp_approach_target(source_receipts)
        if stationary_source_close else draft
    )
    if (
        claimed_milestone == "grasp"
        and recovery_target is not None
        and stationary_source_close
        and contact_target is None
    ):
        raise ValueError(
            "empty-grasp retry must change the executed source approach before "
            "a stationary re-close; arm motion after the last source servo "
            "requires a new source image-servo receipt"
        )
    if (
        claimed_milestone == "grasp"
        and recovery_target is not None
        and (
            stationary_source_close
            or (draft.get("kind") == "image_servo" and draft.get("gripper") == "close")
        )
        and contact_target is not None
        and contact_target.get("camera") == recovery_target["camera"]
        and contact_target.get("target_pixel") == recovery_target["target_pixel"]
    ):
        old_depth = recovery_target["depth_delta_m"]
        new_depth = contact_target.get("depth_delta_m")
        depth_changed = (
            isinstance(old_depth, (int, float))
            and isinstance(new_depth, (int, float))
            and not isinstance(new_depth, bool)
            and math.isfinite(float(new_depth))
            and not math.isclose(float(old_depth), float(new_depth), abs_tol=1e-6)
        )
        # The existing image-servo resolver still validates any new stereo pair
        # against public calibration before a command can reach the mailbox.
        stereo_changed = (
            contact_target.get("other_view_pixel") is not None
            and (contact_target.get("other_view_pixel") != recovery_target["other_view_pixel"]
                 or contact_target.get("other_view_camera") != recovery_target.get("other_view_camera"))
        )
        if not depth_changed and not stereo_changed:
            raise ValueError(
                "empty-grasp retry must change the source point, depth, or stereo "
                "grounding before re-closing; an open retreat or reapproach alone "
                "does not change the failed contact strategy"
            )
    if (
        claimed_milestone != "transport"
        or not milestone_history
        or milestone_history[-1] != "grasp"
    ):
        return
    if (
        not requested_close
        or isinstance(separation, bool)
        or not isinstance(separation, (int, float))
        or not math.isfinite(float(separation))
        or float(separation) <= SOURCE_CONTACT_MIN_SEPARATION
    ):
        raise ValueError("transport milestone requires a sealed nonempty grasp")


def command_signature(command: Mapping[str, object]) -> str:
    value = {
        key: item
        for key, item in command.items()
        if key not in {"observation_id", "note"}
    }
    return canonical_sha256(value)


def _vector(value: object) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _distance(left: object, right: object) -> float:
    a, b = _vector(left), _vector(right)
    if a is None or b is None or len(a) != len(b):
        return math.inf
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def _max_delta(left: object, right: object) -> float:
    a, b = _vector(left), _vector(right)
    if a is None or b is None or len(a) != len(b):
        return math.inf
    return max((abs(x - y) for x, y in zip(a, b, strict=True)), default=0.0)


def _quaternion_delta_deg(left: object, right: object) -> float:
    a, b = _vector(left), _vector(right)
    if a is None or b is None or len(a) != 4 or len(b) != 4:
        return math.inf
    an = math.sqrt(sum(value * value for value in a))
    bn = math.sqrt(sum(value * value for value in b))
    if an < 1e-12 or bn < 1e-12:
        return math.inf
    dot = abs(sum(x * y for x, y in zip(a, b, strict=True)) / (an * bn))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def _strict_vector(value: object, *, width: int) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != width:
        return None
    output: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if not math.isfinite(number):
            return None
        output.append(number)
    return output


def _strict_number(value: object, *, nonnegative: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        return None
    return number


def _same_number(left: object, right: object) -> bool:
    left_number = _strict_number(left)
    right_number = _strict_number(right)
    return (
        left_number is not None
        and right_number is not None
        and left_number == right_number
    )


def _rgb_change(value: object) -> dict[str, float] | None:
    if not isinstance(value, Mapping) or set(value) != _RGB_LABELS:
        return None
    output: dict[str, float] = {}
    for label, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if not math.isfinite(number) or number < 0.0:
            return None
        output[str(label)] = number
    return output


def _public_state_snapshot(
    public_state: Mapping[str, object],
    *,
    require_torque_available: bool = False,
) -> tuple[str | None, dict[str, object] | None]:
    try:
        return None, validate_public_state(
            public_state,
            require_torque_available=require_torque_available,
        )
    except (TypeError, ValueError) as error:
        if "torque" in str(error):
            return "partial_initial_torque", None
        return "malformed_public_state", None


def _receipt_expected_fields(kind: object) -> set[str] | None:
    if kind == "move_joints":
        return PUBLIC_JOINT_RECEIPT_FIELDS
    if kind == "cartesian_delta":
        return PUBLIC_CARTESIAN_RECEIPT_FIELDS
    if kind == "image_servo":
        return PUBLIC_IMAGE_SERVO_RECEIPT_FIELDS
    if kind == "base_action":
        return PUBLIC_BASE_RECEIPT_FIELDS
    return None


def _numeric_mapping(
    value: object,
    *,
    allowed: set[str],
    require_nonempty: bool,
) -> dict[str, float] | None:
    if not isinstance(value, Mapping) or (require_nonempty and not value):
        return None
    output: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str) or key not in allowed:
            return None
        number = _strict_number(item)
        if number is None:
            return None
        output[key] = number
    return output


def _numeric_mappings_match(
    left: object, right: object, *, allowed: set[str], require_nonempty: bool
) -> bool:
    left_values = _numeric_mapping(
        left, allowed=allowed, require_nonempty=require_nonempty
    )
    right_values = _numeric_mapping(
        right, allowed=allowed, require_nonempty=require_nonempty
    )
    if left_values is None or right_values is None or set(left_values) != set(right_values):
        return False
    return all(_same_number(left[name], right[name]) for name in left_values)


def _receipt_command_mismatch(
    receipt: Mapping[str, object], command: Mapping[str, object]
) -> str | None:
    if receipt.get("observation_id") != command.get("observation_id"):
        return "stale_receipt"
    if receipt.get("note") != command.get("note"):
        return "command_mismatch"
    if receipt.get("kind") == "move_joints":
        if not _numeric_mappings_match(
            receipt.get("requested_targets"),
            command.get("targets"),
            allowed={*JOINT_NAMES, "gripper"},
            require_nonempty=True,
        ):
            return "command_mismatch"
        gripper_intent = _strict_number(receipt.get("gripper_intent"))
        if gripper_intent is None or not 0.0 <= gripper_intent <= 1.0:
            return "command_mismatch"
        command_targets = command.get("targets")
        if (
            isinstance(command_targets, Mapping)
            and "gripper" in command_targets
            and not _same_number(gripper_intent, command_targets["gripper"])
        ):
            return "command_mismatch"
    elif receipt.get("kind") == "cartesian_delta":
        translation = _strict_vector(command.get("translation_m"), width=3)
        rotation = _strict_vector(
            command.get("rotation_axis_angle_rad"), width=3
        )
        if (
            translation is None
            or rotation is None
            or receipt.get("requested_translation_m") != translation
            or receipt.get("requested_rotation_axis_angle_rad") != rotation
            or receipt.get("requested_gripper") != command.get("gripper")
        ):
            return "command_mismatch"
    elif receipt.get("kind") == "image_servo":
        target_pixel = _strict_vector(command.get("target_pixel"), width=2)
        if (
            target_pixel is None
            or receipt.get("requested_camera") != command.get("camera")
            or receipt.get("requested_target_pixel") != target_pixel
            or receipt.get("requested_target_role") != command.get("target_role")
            or not _same_number(
                receipt.get("requested_depth_delta_m"),
                command.get("depth_delta_m"),
            )
            or not _same_number(
                receipt.get("requested_step_m"), command.get("step_m")
            )
            or receipt.get("requested_gripper") != command.get("gripper")
            or receipt.get("requested_other_view_camera") != command.get("other_view_camera")
            or (
                command.get("other_view_pixel") is not None
                and receipt.get("requested_other_view_pixel")
                != _strict_vector(command.get("other_view_pixel"), width=2)
            )
            or (
                command.get("other_view_pixel") is None
                and receipt.get("requested_other_view_pixel") is not None
            )
        ):
            return "command_mismatch"
        gripper_intent = _strict_number(receipt.get("gripper_intent"))
        if gripper_intent is None or not 0.0 <= gripper_intent <= 1.0:
            return "command_mismatch"
        if (
            command.get("gripper") == "open"
            and not _same_number(gripper_intent, 1.0)
        ) or (
            command.get("gripper") == "close"
            and not _same_number(gripper_intent, 0.0)
        ):
            return "command_mismatch"
        gripper_intent = _strict_number(receipt.get("gripper_intent"))
        if gripper_intent is None or not 0.0 <= gripper_intent <= 1.0:
            return "command_mismatch"
        if (
            command.get("gripper") == "open"
            and not _same_number(gripper_intent, 1.0)
        ) or (
            command.get("gripper") == "close"
            and not _same_number(gripper_intent, 0.0)
        ):
            return "command_mismatch"
    elif receipt.get("kind") == "base_action":
        if (
            receipt.get("axis") != command.get("axis")
            or not _same_number(
                receipt.get("normalized_velocity"),
                command.get("normalized_velocity"),
            )
        ):
            return "command_mismatch"
        gripper_intent = _strict_number(receipt.get("gripper_intent"))
        if gripper_intent is None or not 0.0 <= gripper_intent <= 1.0:
            return "command_mismatch"
        gripper_command = command.get("gripper")
        if (gripper_command == "open" and not _same_number(gripper_intent, 1.0)) or (
            gripper_command == "close" and not _same_number(gripper_intent, 0.0)
        ):
            return "command_mismatch"
        if gripper_command not in {"open", "hold", "close"}:
            return "command_mismatch"
        if gripper_command == "hold" and receipt.get("step_count") != BASE_STEPS:
            return "command_mismatch"
    else:
        return "command_mismatch"
    return None


def _receipt_telemetry_ineligibility(
    receipt: Mapping[str, object], public_state: Mapping[str, object]
) -> str | None:
    current_qpos = _strict_vector(
        public_state.get("state.arm_joint_position"), width=7
    )
    current_qvel = _strict_vector(
        public_state.get("state.arm_joint_velocity"), width=7
    )
    current_gripper = _strict_vector(
        public_state.get("state.gripper_qpos"), width=2
    )
    torque = public_state.get("state.arm_applied_torque")
    wrench = public_state.get("state.end_effector_wrench")
    if (
        current_qpos is None
        or current_qvel is None
        or current_gripper is None
        or not isinstance(torque, Mapping)
        or not isinstance(wrench, Mapping)
    ):
        return "missing_public_telemetry"
    current_torque = _strict_vector(torque.get("values_nm"), width=7)
    current_force = _strict_vector(wrench.get("force_n"), width=3)
    current_wrench_torque = _strict_vector(wrench.get("torque_nm"), width=3)
    if current_torque is None or current_force is None or current_wrench_torque is None:
        return "partial_initial_torque"
    if _strict_vector(receipt.get("realized_arm_qpos"), width=7) != current_qpos:
        return "fresh_state_mismatch"
    gripper_residual = receipt["gripper_residual"]
    if (
        not isinstance(gripper_residual, Mapping)
        or gripper_residual["end_qpos"] != current_gripper
    ):
        return "fresh_state_mismatch"
    summary = receipt["telemetry_summary"]
    if not isinstance(summary, Mapping):
        return "incomplete_receipt"
    qpos = summary["arm_joint_position"]
    qvel = summary["arm_joint_velocity"]
    applied_torque = summary["arm_applied_torque"]
    summary_wrench = summary["end_effector_wrench"]
    force = summary_wrench["force"]
    wrench_torque = summary_wrench["torque"]
    if applied_torque["start_nm"] is None:
        return "partial_initial_torque"
    if receipt.get("realized_arm_qpos_delta") != qpos["delta_rad"]:
        return "fresh_state_mismatch"
    if (
        qpos["end_rad"] != current_qpos
        or qvel["end_rad_s"] != current_qvel
        or applied_torque["end_nm"] != current_torque
        or force["end_n"] != current_force
        or wrench_torque["end_nm"] != current_wrench_torque
    ):
        return "fresh_state_mismatch"
    return None


def select_sealed_public_receipt(
    command: Mapping[str, object] | None,
    receipts: object,
    public_state: Mapping[str, object],
    rgb_change: object = None,
) -> tuple[str, dict[str, object] | None]:
    """Return the immediately preceding sealed public receipt, or fail closed."""

    if command is None or command.get("kind") not in PHYSICAL_COMMAND_KINDS:
        return "no_physical_command", None
    state_reason, _state_snapshot = _public_state_snapshot(
        public_state,
        require_torque_available=True,
    )
    if state_reason is not None:
        return state_reason, None
    if not isinstance(receipts, list) or not receipts:
        return "missing_receipt", None
    latest = receipts[-1]
    if not isinstance(latest, Mapping):
        return "malformed_receipt", None
    receipt = dict(latest)
    expected_fields = _receipt_expected_fields(receipt.get("kind"))
    if expected_fields is None:
        return "wrong_kind", None
    present_fields = set(receipt)
    stereo_fields = {
        "requested_other_view_pixel",
        "stereo_ray_gap_m",
        "stereo_target_base_m",
    }
    if receipt.get("kind") == "image_servo" and stereo_fields <= present_fields:
        present_fields = present_fields - stereo_fields
        if receipt.get("requested_other_view_camera") == "wrist":
            present_fields.discard("requested_other_view_camera")
    if present_fields != expected_fields:
        return "incomplete_receipt", None
    if receipt.get("kind") != command.get("kind"):
        return "wrong_kind", None
    if receipt.get("accepted") is not True:
        return "rejected_receipt", None
    mismatch = _receipt_command_mismatch(receipt, command)
    if mismatch is not None:
        return mismatch, None
    receipt_rgb = _rgb_change(receipt.get("mean_absolute_rgb_change"))
    if receipt_rgb is None:
        return "unsealed_receipt", None
    if rgb_change is not None:
        current_rgb = _rgb_change(rgb_change)
        if current_rgb is None or receipt_rgb != current_rgb:
            return "rgb_mismatch", None
    try:
        receipt = validate_public_receipt(
            receipt,
            require_sealed_rgb=True,
            require_torque_baseline=True,
        )
    except (TypeError, ValueError) as error:
        if "torque" in str(error):
            return "partial_initial_torque", None
        return "incomplete_receipt", None
    telemetry_reason = _receipt_telemetry_ineligibility(receipt, public_state)
    if telemetry_reason is not None:
        return telemetry_reason, None
    return "eligible", cast(dict[str, object], canonical_json_copy(receipt))


def _significant_receipts(receipts: object) -> list[dict[str, object]]:
    if not isinstance(receipts, list):
        return []
    output = []
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            continue
        if (
            receipt.get("source") == "public_rgb_harness"
            or receipt.get("accepted") is False
            or str(receipt.get("kind", "")).startswith("servo_")
            or receipt.get("kind") in {"aligned", "official_terminated"}
        ):
            output.append(dict(receipt))
    return output


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _receipt_outcome(receipts: object) -> dict[str, object]:
    if not isinstance(receipts, list) or not receipts:
        return {
            "receipt_tracking_pause_count": 0,
            "receipt_endpoint_error_rad": 0.0,
            "receipt_arm_joint_max_delta_rad": 0.0,
            "receipt_joint_velocity_max_abs_rad_s": 0.0,
        }
    latest = receipts[-1]
    if not isinstance(latest, Mapping):
        return {
            "receipt_tracking_pause_count": 0,
            "receipt_endpoint_error_rad": 0.0,
            "receipt_arm_joint_max_delta_rad": 0.0,
            "receipt_joint_velocity_max_abs_rad_s": 0.0,
        }
    qpos_delta = _strict_vector(latest.get("realized_arm_qpos_delta"), width=7)
    summary = latest.get("telemetry_summary")
    velocity_max = 0.0
    summary_delta = None
    if isinstance(summary, Mapping):
        qpos = summary.get("arm_joint_position")
        qvel = summary.get("arm_joint_velocity")
        if isinstance(qpos, Mapping):
            summary_delta = _strict_vector(qpos.get("delta_rad"), width=7)
        if isinstance(qvel, Mapping):
            maxima = _strict_vector(qvel.get("maximum_abs_rad_s"), width=7)
            if maxima is not None:
                velocity_max = max(maxima, default=0.0)
    joint_delta = qpos_delta if qpos_delta is not None else summary_delta
    endpoint_error = latest.get(
        "endpoint_error",
        latest.get("remaining_endpoint_error", 0.0),
    )
    if isinstance(endpoint_error, bool) or not isinstance(endpoint_error, (int, float)):
        endpoint_error_rad = 0.0
    else:
        endpoint_error_rad = float(endpoint_error)
        if not math.isfinite(endpoint_error_rad) or endpoint_error_rad < 0.0:
            endpoint_error_rad = 0.0
    return {
        "receipt_tracking_pause_count": _nonnegative_int(
            latest.get("tracking_pause_count")
        ),
        "receipt_endpoint_error_rad": endpoint_error_rad,
        "receipt_arm_joint_max_delta_rad": (
            max((abs(item) for item in joint_delta), default=0.0)
            if joint_delta is not None
            else 0.0
        ),
        "receipt_joint_velocity_max_abs_rad_s": velocity_max,
    }


def public_outcome_delta(
    previous: Mapping[str, object],
    current: Mapping[str, object],
    previous_receipts: object,
    current_receipts: object,
) -> dict[str, object]:
    eef_translation = _distance(
        previous.get("state.end_effector_position_relative"),
        current.get("state.end_effector_position_relative"),
    )
    eef_rotation = _quaternion_delta_deg(
        previous.get("state.end_effector_rotation_relative"),
        current.get("state.end_effector_rotation_relative"),
    )
    base_translation = _distance(
        previous.get("state.base_position"), current.get("state.base_position")
    )
    base_rotation = _quaternion_delta_deg(
        previous.get("state.base_rotation"), current.get("state.base_rotation")
    )
    gripper = _max_delta(
        previous.get("state.gripper_qpos"), current.get("state.gripper_qpos")
    )
    arm_joint = _max_delta(
        previous.get("state.arm_joint_position"),
        current.get("state.arm_joint_position"),
    )
    current_arm_qvel = _strict_vector(
        current.get("state.arm_joint_velocity"), width=7
    )
    current_arm_qvel_max = (
        max((abs(item) for item in current_arm_qvel), default=0.0)
        if current_arm_qvel is not None
        else math.inf
    )
    receipt_outcome = _receipt_outcome(current_receipts)
    receipts_changed = canonical_sha256(_significant_receipts(previous_receipts)) != (
        canonical_sha256(_significant_receipts(current_receipts))
    )
    stalled = (
        eef_translation < STALL_TRANSLATION_M
        and eef_rotation < STALL_ROTATION_DEG
        and base_translation < STALL_TRANSLATION_M
        and base_rotation < STALL_ROTATION_DEG
        and gripper < STALL_GRIPPER_QPOS
        and arm_joint < STALL_ARM_JOINT_RAD
        and current_arm_qvel_max < STALL_ARM_JOINT_RAD
        and receipt_outcome["receipt_arm_joint_max_delta_rad"] < STALL_ARM_JOINT_RAD
        and receipt_outcome["receipt_joint_velocity_max_abs_rad_s"]
        < STALL_ARM_JOINT_RAD
        and not receipts_changed
    )
    return {
        "eef_translation_m": eef_translation,
        "eef_rotation_deg": eef_rotation,
        "base_translation_m": base_translation,
        "base_rotation_deg": base_rotation,
        "gripper_qpos_max_delta": gripper,
        "arm_joint_max_delta_rad": arm_joint,
        "current_arm_joint_velocity_max_abs_rad_s": current_arm_qvel_max,
        **receipt_outcome,
        "significant_receipts_changed": receipts_changed,
        "stalled": stalled,
    }


@dataclasses.dataclass
class CriticContext:
    task: str
    family: str
    run: object
    critic_prompt: str
    max_decisions: int
    model_calls: int = 0
    controller_calls: int = 0
    critic_attempts: int = 0
    critic_successes: int = 0
    critic_unavailable: int = 0
    executed_command_count: int = 0
    last_attempt_command_count: int | None = None
    early_failure_exemption_used: bool = False
    first_action_review_pending: bool = True
    late_trigger_used: bool = False
    active_observation_id: str | None = None
    active_public_state: dict[str, object] | None = None
    active_receipts: object = dataclasses.field(default_factory=list)
    pending_command: dict[str, object] | None = None
    executed_commands: list[dict[str, object]] = dataclasses.field(default_factory=list)
    trigger_evaluations: list[dict[str, object]] = dataclasses.field(default_factory=list)
    critic_records: list[dict[str, object]] = dataclasses.field(default_factory=list)
    controller_records: list[dict[str, object]] = dataclasses.field(default_factory=list)
    attempt_records: list[dict[str, object]] = dataclasses.field(default_factory=list)
    decisions_by_observation: dict[str, dict[str, object]] = dataclasses.field(
        default_factory=dict
    )
    advice_by_observation: dict[str, dict[str, object] | None] = dataclasses.field(
        default_factory=dict
    )
    consumed_advice_observations: set[str] = dataclasses.field(default_factory=set)
    proposal_observation_id: str | None = None
    proposal_public_state_sha256: str | None = None
    proposal_image_sha256: dict[str, str] | None = None
    proposal_prior_receipt_sha256: str | None = None
    proposal_revisions_used: int = 0
    proposal_pending_record_index: int | None = None
    proposal_pending_command: dict[str, object] | None = None
    proposal_pending_milestone: str | None = None
    proposal_records: list[dict[str, object]] = dataclasses.field(default_factory=list)
    milestone_history: list[str] = dataclasses.field(default_factory=list)
    source_perception_records: list[dict[str, object]] = dataclasses.field(default_factory=list)
    source_perception_packet: dict[str, object] | None = None
    source_perception_observation_id: str | None = None
    source_perception_deadline: float | None = None
    failure_skill_memory: list[dict[str, object]] = dataclasses.field(default_factory=list)
    failure_skill_uses: list[dict[str, object]] = dataclasses.field(default_factory=list)
    active_source_failure: dict[str, object] | None = None
    proposal_failed_observations: set[str] = dataclasses.field(default_factory=set)
    articulated_forbidden_targets: list[dict[str, object]] = dataclasses.field(
        default_factory=list
    )
    articulated_failed_actuation_axis_sets: list[list[str]] = dataclasses.field(
        default_factory=list
    )
    articulated_wrist_roll_origin_joint7: float | None = None
    coffee_press_world_eef_m: list[float] | None = None
    coffee_retreat_targets: dict[str, float] | None = None

    def begin_observation(
        self,
        observation_id: str,
        public_state: Mapping[str, object],
        receipts: object,
        rgb_change: object = None,
    ) -> dict[str, object]:
        cached = self.decisions_by_observation.get(observation_id)
        if cached is not None:
            return cached

        previous_state = self.active_public_state
        previous_receipts = self.active_receipts
        _state_reason, current_public_state_snapshot = _public_state_snapshot(
            public_state
        )
        if (
            self.active_observation_id is not None
            and self.pending_command is not None
            and self.pending_command.get("kind") in PHYSICAL_COMMAND_KINDS
        ):
            self.executed_commands.append(
                cast(dict[str, object], canonical_json_copy(self.pending_command))
            )
            self.executed_command_count += 1

        self.active_observation_id = observation_id
        self.active_public_state = current_public_state_snapshot
        self.active_receipts = deepcopy(receipts)
        self.pending_command = None

        outcome = (
            public_outcome_delta(
                previous_state,
                current_public_state_snapshot
                if current_public_state_snapshot is not None
                else public_state,
                previous_receipts,
                receipts,
            )
            if previous_state is not None and self.executed_commands
            else None
        )
        repeated = (
            len(self.executed_commands) >= 2
            and command_signature(self.executed_commands[-1])
            == command_signature(self.executed_commands[-2])
        )
        trigger: str | None = None
        if self.executed_command_count >= 1 and self.first_action_review_pending:
            trigger = "first_action"
        elif repeated:
            trigger = "repeat"
        elif outcome is not None and outcome["stalled"] is True:
            trigger = "stall"
        elif (
            not self.late_trigger_used
            and self.executed_command_count >= math.ceil(self.max_decisions / 2)
        ):
            trigger = "late_episode"

        suppressed: str | None = None
        receipt_eligibility = "not_evaluated"
        sealed_receipt: dict[str, object] | None = None
        fresh_public_state: dict[str, object] | None = None
        if current_public_state_snapshot is not None:
            fresh_public_state = current_public_state_snapshot
        previous_command = (
            cast(dict[str, object], canonical_json_copy(self.executed_commands[-1]))
            if self.executed_commands
            else None
        )
        if trigger is not None:
            receipt_eligibility, sealed_receipt = select_sealed_public_receipt(
                previous_command,
                receipts,
                public_state,
                rgb_change=rgb_change,
            )
            if sealed_receipt is None:
                suppressed = "receipt_ineligible"
        if trigger is not None and self.critic_attempts >= CRITIC_MAX_ATTEMPTS_PER_TASK:
            suppressed = "cap"
        elif trigger in {"repeat", "stall"} and self.early_failure_exemption_used:
            since = (
                math.inf
                if self.last_attempt_command_count is None
                else self.executed_command_count - self.last_attempt_command_count
            )
            if since < CRITIC_COOLDOWN_COMMANDS:
                suppressed = "cooldown"

        fire = trigger is not None and suppressed is None and sealed_receipt is not None
        advisory_id: str | None = None
        attempt_index: int | None = None
        if fire:
            # Reservation happens before transport. Every failure consumes the slot.
            self.critic_attempts += 1
            self.model_calls += 1
            attempt_index = self.critic_attempts
            self.last_attempt_command_count = self.executed_command_count
            if trigger in {"repeat", "stall"}:
                self.early_failure_exemption_used = True
            if trigger == "first_action":
                self.first_action_review_pending = False
            if trigger == "late_episode":
                self.late_trigger_used = True
            advisory_id = hashlib.sha256(
                (
                    f"{self.task}:{observation_id}:{attempt_index}:{trigger}:"
                    f"{CRITIC_CONFIG_SHA256}"
                ).encode()
            ).hexdigest()[:20]

        decision: dict[str, object] = {
            "observation_id": observation_id,
            "executed_command_count": self.executed_command_count,
            "eligible_trigger": trigger,
            "fired_trigger": trigger if fire else None,
            "suppressed": suppressed,
            "critic_attempt_index": attempt_index,
            "advisory_id": advisory_id,
            "previous_executed_command": previous_command,
            "previous_executed_command_sha256": (
                strict_canonical_sha256(previous_command)
                if previous_command is not None
                else None
            ),
            "fresh_public_state": fresh_public_state,
            "fresh_public_state_sha256": (
                strict_canonical_sha256(fresh_public_state)
                if fresh_public_state is not None
                else None
            ),
            "sealed_public_receipt": sealed_receipt,
            "sealed_public_receipt_sha256": (
                strict_canonical_sha256(sealed_receipt)
                if sealed_receipt is not None
                else None
            ),
            "receipt_eligibility": receipt_eligibility,
            "outcome_delta": outcome,
        }
        self.decisions_by_observation[observation_id] = decision
        self.trigger_evaluations.append(decision)
        return decision

    def take_advisory(
        self, observation_id: str
    ) -> tuple[str | None, dict[str, object] | None]:
        if observation_id in self.consumed_advice_observations:
            return None, None
        advisory = self.advice_by_observation.get(observation_id)
        if advisory is None:
            return None, None
        decision = self.decisions_by_observation.get(observation_id, {})
        advisory_id = decision.get("advisory_id")
        if not isinstance(advisory_id, str):
            return None, None
        self.consumed_advice_observations.add(observation_id)
        return advisory_id, cast(dict[str, object], canonical_json_copy(advisory))

    def _mark_consumed_critic(
        self,
        consumed_advisory_id: str | None,
        *,
        controller_request_sha256: str | None,
        controller_call_manifest_sha256: str | None,
        controller_request_linkage_status: str,
        command_sha256: str | None,
    ) -> None:
        if consumed_advisory_id is None:
            return
        for record in reversed(self.critic_records):
            if record.get("advisory_id") == consumed_advisory_id:
                record["consumed_by_controller_request_sha256"] = (
                    controller_request_sha256
                )
                record["consumed_by_controller_call_manifest_sha256"] = (
                    controller_call_manifest_sha256
                )
                record["consumed_by_controller_linkage_status"] = (
                    controller_request_linkage_status
                )
                record["qwen_command_sha256"] = command_sha256
                break

    def stage_controller_return(
        self,
        observation_id: str,
        command: Mapping[str, object],
        consumed_advisory_id: str | None,
        command_evidence_sha256: str,
        *,
        controller_request_sha256: str | None = None,
        critic_request_sha256: str | None = None,
        controller_call_manifest_sha256: str | None = None,
        critic_call_manifest_sha256: str | None = None,
        advisory_sha256: str | None = None,
        controller_request_linkage_status: str = "complete",
        qwen_attempt_index: int = 0,
    ) -> None:
        if observation_id != self.active_observation_id:
            raise RuntimeError("controller return observation drift")
        command_snapshot = cast(dict[str, object], canonical_json_copy(command))
        self.pending_command = command_snapshot
        command_sha256 = strict_canonical_sha256(command_snapshot)
        self.controller_records.append({
            "observation_id": observation_id,
            "controller_call_index": self.controller_calls,
            "qwen_attempt_index": qwen_attempt_index,
            "command": command_snapshot,
            "command_sha256": command_sha256,
            "command_evidence_sha256": command_evidence_sha256,
            "consumed_advisory_id": consumed_advisory_id,
            "controller_request_sha256": controller_request_sha256,
            "controller_request_linkage_status": controller_request_linkage_status,
            "critic_request_sha256": critic_request_sha256,
            "controller_call_manifest_sha256": controller_call_manifest_sha256,
            "critic_call_manifest_sha256": critic_call_manifest_sha256,
            "advisory_sha256": advisory_sha256,
        })
        self._mark_consumed_critic(
            consumed_advisory_id,
            controller_request_sha256=controller_request_sha256,
            controller_call_manifest_sha256=controller_call_manifest_sha256,
            controller_request_linkage_status=controller_request_linkage_status,
            command_sha256=command_sha256,
        )

    def stage_controller_failure(
        self,
        observation_id: str,
        consumed_advisory_id: str | None,
        command_evidence_sha256: str,
        *,
        controller_request_sha256: str | None = None,
        critic_request_sha256: str | None = None,
        controller_call_manifest_sha256: str | None = None,
        critic_call_manifest_sha256: str | None = None,
        advisory_sha256: str | None = None,
        controller_request_linkage_status: str = "incomplete",
        qwen_attempt_index: int = 0,
        failure_class: str,
        failure_sha256: str,
    ) -> None:
        if observation_id != self.active_observation_id:
            raise RuntimeError("controller failure observation drift")
        self.controller_records.append({
            "observation_id": observation_id,
            "controller_call_index": self.controller_calls,
            "qwen_attempt_index": qwen_attempt_index,
            "status": "controller_unavailable",
            "command": None,
            "command_sha256": None,
            "command_evidence_sha256": command_evidence_sha256,
            "consumed_advisory_id": consumed_advisory_id,
            "controller_request_sha256": controller_request_sha256,
            "controller_request_linkage_status": controller_request_linkage_status,
            "critic_request_sha256": critic_request_sha256,
            "controller_call_manifest_sha256": controller_call_manifest_sha256,
            "critic_call_manifest_sha256": critic_call_manifest_sha256,
            "advisory_sha256": advisory_sha256,
            "failure_class": failure_class,
            "failure_sha256": failure_sha256,
        })
        self._mark_consumed_critic(
            consumed_advisory_id,
            controller_request_sha256=controller_request_sha256,
            controller_call_manifest_sha256=controller_call_manifest_sha256,
            controller_request_linkage_status=controller_request_linkage_status,
            command_sha256=None,
        )


_PROPOSAL_EVIDENCE_FIELDS = {
    "schema",
    "task",
    "family",
    "observation_id",
    "proposal_index",
    "revision_index",
    "fresh_public_state_sha256",
    "fresh_public_rgb_sha256",
    "immediate_prior_receipt_sha256",
    "milestone_history_before",
    "milestone_history_sha256",
    "claimed_milestone",
    "command_kind",
    "draft",
    "draft_sha256",
    "controller_call_index",
    "controller_request_sha256",
    "controller_call_manifest_sha256",
    "controller_attempt_record_sha256",
    "critic_call_index",
    "critic_request_sha256",
    "critic_call_manifest_sha256",
    "critic_attempt_record_sha256",
    "audit",
    "audit_sha256",
    "status",
    "contradiction",
    "suggested_correction",
    "returned_command_sha256",
    "executed",
    "mailbox_count",
    "action_count",
    "receipt_count",
    "execution_receipt_sha256",
    "mailbox_sha256",
    "terminal_outcome",
    "terminal_outcome_sha256",
    "milestone_history_after",
    "milestone_history_after_sha256",
    "milestone_closed",
    "critic_origin_execution",
    "failure_class",
    "failure_sha256",
    "record_sha256",
}
_PROPOSAL_CONTROLLER_FIELDS = {
    "observation_id",
    "controller_call_index",
    "qwen_attempt_index",
    "command",
    "command_sha256",
    "controller_request_sha256",
    "controller_request_linkage_status",
    "controller_call_manifest_sha256",
    "status",
    "record_sha256",
}
_PROPOSAL_ATTEMPT_FIELDS = {
    "schema",
    "role",
    "call_index",
    "record",
    "record_sha256",
}
_PROPOSAL_ATTEMPT_RECORD_FIELDS = {
    "request_sha256",
    "observation_id",
    "attempt_index",
    "raw_body_sha256",
    "sanitized_raw_command",
    "http_status",
    "finish_reason",
    "completion_tokens",
    "latency_s",
    "response_schema_sha256",
    "served_model_id",
}
_PROPOSAL_STATUSES = {
    "approved_for_execution",
    "rejected_by_protocol",
    "rejected_by_critic",
    "critic_unavailable",
    "controller_linkage_incomplete",
}
_PROPOSAL_EFFECT_FIELDS = {
    "executed",
    "mailbox_count",
    "action_count",
    "receipt_count",
    "execution_receipt_sha256",
    "mailbox_sha256",
    "terminal_outcome",
    "terminal_outcome_sha256",
    "milestone_history_after",
    "milestone_history_after_sha256",
    "milestone_closed",
    "record_sha256",
}


def _closure_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be a strict integer")
    return value


def _closure_sha256(
    value: object,
    *,
    label: str,
    optional: bool = False,
) -> str | None:
    if value is None and optional:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a canonical SHA-256")
    return value


def _closure_sealed_record(
    value: object,
    *,
    fields: set[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} schema drifted")
    record = dict(value)
    seal = _closure_sha256(record.get("record_sha256"), label=f"{label} seal")
    canonical = {key: item for key, item in record.items() if key != "record_sha256"}
    if seal != strict_canonical_sha256(canonical):
        raise ValueError(f"{label} seal drifted")
    return record


def _closure_attempt(
    value: object,
    *,
    role: str,
    call_index: int,
    response_schema_sha256: str,
    allow_truncated: bool = False,
) -> tuple[dict[str, object], str]:
    if not isinstance(value, Mapping) or set(value) != _PROPOSAL_ATTEMPT_FIELDS:
        raise ValueError(f"{role} AttemptEvidenceLog envelope schema drifted")
    envelope = dict(value)
    if (
        envelope.get("schema") != "robocasa-qwen-attempt-evidence/v1"
        or envelope.get("role") != role
        or _closure_int(
            envelope.get("call_index"), label=f"{role} call index", minimum=1
        )
        != call_index
    ):
        raise ValueError(f"{role} AttemptEvidenceLog envelope identity drifted")
    raw_record = envelope.get("record")
    if (
        not isinstance(raw_record, Mapping)
        or set(raw_record) != _PROPOSAL_ATTEMPT_RECORD_FIELDS
    ):
        raise ValueError(f"{role} AttemptEvidenceLog record schema drifted")
    record = dict(raw_record)
    seal = _closure_sha256(
        envelope.get("record_sha256"), label=f"{role} attempt record seal"
    )
    if seal != strict_canonical_sha256(record):
        raise ValueError(f"{role} AttemptEvidenceLog record seal drifted")
    _closure_sha256(record.get("request_sha256"), label=f"{role} request")
    _closure_sha256(record.get("raw_body_sha256"), label=f"{role} raw body")
    _closure_int(record.get("attempt_index"), label=f"{role} attempt index")
    http_status = _closure_int(
        record.get("http_status"), label=f"{role} HTTP status"
    )
    completion_tokens = _closure_int(
        record.get("completion_tokens"),
        label=f"{role} completion tokens",
        minimum=1,
    )
    latency = record.get("latency_s")
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(float(latency))
        or float(latency) < 0.0
    ):
        raise ValueError(f"{role} attempt latency drifted")
    if (
        record.get("response_schema_sha256") != response_schema_sha256
        or not isinstance(record.get("observation_id"), str)
        or not isinstance(record.get("sanitized_raw_command"), str)
        or not isinstance(record.get("finish_reason"), str)
        or not isinstance(record.get("served_model_id"), str)
        or not record.get("served_model_id")
    ):
        raise ValueError(f"{role} AttemptEvidenceLog semantics drifted")
    completion_limit = (
        PROPOSAL_CONTROLLER_MAX_TOKENS
        if role == "controller"
        else CRITIC_MAX_TOKENS
    )
    finish_reason = record.get("finish_reason")
    truncated = finish_reason == "length" and completion_tokens >= completion_limit
    if http_status != 200 or not (
        (finish_reason == "stop" and completion_tokens < completion_limit)
        or (allow_truncated and truncated)
    ):
        # A truncated draft is tolerated only when the proposal it produced was
        # rejected by protocol as malformed: it never executed, so it cannot
        # hide a command, and treating it as corruption voided whole episodes.
        raise ValueError(f"{role} completion integrity drifted")
    return record, cast(str, seal)


def canonical_sanitized_output(value: object) -> object:
    if not isinstance(value, str) or not value:
        raise ValueError("Qwen sanitized output is missing")
    candidate = value.rsplit("</think>", 1)[-1].strip()
    try:
        return canonical_json_copy(json.loads(candidate))
    except (json.JSONDecodeError, TypeError, ValueError):
        return value


def _closure_zero_effects(record: Mapping[str, object], *, label: str) -> None:
    for field in ("mailbox_count", "action_count", "receipt_count"):
        if _closure_int(record.get(field), label=f"{label} {field}") != 0:
            raise ValueError(f"{label} has execution effects")
    if record.get("executed") is not False:
        raise ValueError(f"{label} executed flag drifted")
    if any(
        record.get(field) is not None
        for field in (
            "execution_receipt_sha256",
            "mailbox_sha256",
            "terminal_outcome",
            "terminal_outcome_sha256",
        )
    ):
        raise ValueError(f"{label} effect hashes drifted")
    if record.get("milestone_closed") is not False:
        raise ValueError(f"{label} milestone closure drifted")
    if record.get("milestone_history_after") != record.get(
        "milestone_history_before"
    ):
        raise ValueError(f"{label} advanced milestone history")


def _closure_terminal_outcome(
    value: object,
    *,
    command_kind: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("terminal outcome is missing")
    outcome = dict(value)
    if command_kind == "give_up":
        expected = {
            "schema": "robocasa-qwen-approved-give-up/v1",
            "status": "approved_no_simulator_effect",
        }
        if outcome != expected:
            raise ValueError("approved give_up disposition drifted")
        return outcome
    if set(outcome) != {
        "schema",
        "status",
        "success",
        "simulator_terminal",
        "simulator_terminal_sha256",
    }:
        raise ValueError("approved finish outcome schema drifted")
    status = outcome.get("status")
    if (
        outcome.get("schema") != "robocasa-inspect-joint-terminal/v1"
        or status not in {"success", "finished_false"}
        or outcome.get("success") is not (status == "success")
    ):
        raise ValueError("approved finish outcome semantics drifted")
    simulator_terminal = outcome.get("simulator_terminal")
    if not isinstance(simulator_terminal, Mapping) or set(simulator_terminal) != {
        "status",
        "sequence",
        "terminal_outcome_sha256",
        "terminal_snapshot_sha256",
        "episode_tmp_empty",
        "simulator_actions",
        "wall_s",
    }:
        raise ValueError("official simulator terminal schema drifted")
    _closure_int(
        simulator_terminal.get("sequence"), label="terminal sequence"
    )
    _closure_int(
        simulator_terminal.get("simulator_actions"), label="terminal actions"
    )
    _closure_sha256(
        simulator_terminal.get("terminal_outcome_sha256"),
        label="official terminal outcome",
    )
    _closure_sha256(
        simulator_terminal.get("terminal_snapshot_sha256"),
        label="official terminal snapshot",
    )
    wall_s = simulator_terminal.get("wall_s")
    if (
        simulator_terminal.get("status") != status
        or simulator_terminal.get("episode_tmp_empty") is not True
        or isinstance(wall_s, bool)
        or not isinstance(wall_s, (int, float))
        or not math.isfinite(float(wall_s))
        or float(wall_s) < 0.0
        or outcome.get("simulator_terminal_sha256")
        != strict_canonical_sha256(simulator_terminal)
    ):
        raise ValueError("official simulator terminal linkage drifted")
    return outcome


def _closure_execution_effects(
    record: Mapping[str, object],
    *,
    command_kind: str,
) -> None:
    if record.get("executed") is not True:
        raise ValueError("approved proposal execution flag drifted")
    mailbox_count = _closure_int(
        record.get("mailbox_count"), label="approved mailbox count"
    )
    action_count = _closure_int(
        record.get("action_count"), label="approved action count"
    )
    receipt_count = _closure_int(
        record.get("receipt_count"), label="approved receipt count"
    )
    if command_kind in PHYSICAL_COMMAND_KINDS:
        if (mailbox_count, action_count, receipt_count) != (1, 1, 1):
            raise ValueError("approved physical effect cardinality drifted")
        _closure_sha256(record.get("mailbox_sha256"), label="physical mailbox")
        _closure_sha256(
            record.get("execution_receipt_sha256"), label="physical receipt"
        )
        if record.get("terminal_outcome") is not None or record.get(
            "terminal_outcome_sha256"
        ) is not None:
            raise ValueError("physical proposal has terminal evidence")
    elif command_kind == "finish":
        if (mailbox_count, action_count, receipt_count) != (1, 0, 0):
            raise ValueError("approved finish effect cardinality drifted")
        _closure_sha256(record.get("mailbox_sha256"), label="finish mailbox")
        if record.get("execution_receipt_sha256") is not None:
            raise ValueError("approved finish has a physical receipt")
        outcome = _closure_terminal_outcome(
            record.get("terminal_outcome"), command_kind=command_kind
        )
        if record.get("terminal_outcome_sha256") != strict_canonical_sha256(outcome):
            raise ValueError("approved finish outcome hash drifted")
    elif command_kind == "give_up":
        if (mailbox_count, action_count, receipt_count) != (0, 0, 0):
            raise ValueError("approved give_up effect cardinality drifted")
        if record.get("mailbox_sha256") is not None or record.get(
            "execution_receipt_sha256"
        ) is not None:
            raise ValueError("approved give_up has simulator effects")
        outcome = _closure_terminal_outcome(
            record.get("terminal_outcome"), command_kind=command_kind
        )
        if record.get("terminal_outcome_sha256") != strict_canonical_sha256(outcome):
            raise ValueError("approved give_up outcome hash drifted")
    else:
        raise ValueError("approved proposal command kind drifted")


def validate_proposal_audit_closure(
    context: CriticContext,
    *,
    require_execution_closure: bool = False,
    expected_served_model_id: str | None = None,
) -> dict[str, int]:
    """Prove exact proposal, attempt, approval, and kind-specific closure."""

    controller_calls = _closure_int(
        context.controller_calls, label="controller call count"
    )
    critic_calls = _closure_int(context.critic_attempts, label="critic call count")
    controller_attempts = [
        record
        for record in context.attempt_records
        if isinstance(record, Mapping) and record.get("role") == "controller"
    ]
    critic_attempts = [
        record
        for record in context.attempt_records
        if isinstance(record, Mapping) and record.get("role") == "critic"
    ]
    if (
        controller_calls != len(controller_attempts)
        or controller_calls != len(context.proposal_records)
        or controller_calls != len(context.controller_records)
    ):
        raise ValueError("controller proposal/attempt cardinality drifted")
    if critic_calls != len(critic_attempts):
        raise ValueError("critic call/AttemptEvidenceLog cardinality drifted")
    critic_schema_sha256 = strict_canonical_sha256(PROPOSAL_AUDIT_SCHEMA)
    controller_attempt_by_call: dict[int, tuple[dict[str, object], str]] = {}
    critic_attempt_by_call: dict[int, tuple[dict[str, object], str]] = {}
    served_model_id: str | None = None
    for call_index, raw in enumerate(controller_attempts, start=1):
        proposal = context.proposal_records[call_index - 1]
        observation_id = proposal.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError("controller proposal observation id drifted")
        default_schema = controller_response_schema_for_observation(observation_id)
        wrist_roll_schema = controller_wrist_roll_schema_for_observation(
            observation_id
        )
        raw_attempt = raw.get("record") if isinstance(raw, Mapping) else None
        recorded_schema_sha256 = (
            raw_attempt.get("response_schema_sha256")
            if isinstance(raw_attempt, Mapping)
            else None
        )
        engage_close_schema = controller_engage_ready_schema_for_observation(
            observation_id, allow_close=True
        )
        engage_servo_schema = controller_engage_ready_schema_for_observation(
            observation_id, allow_close=False
        )
        stereo_servo_schema = controller_engage_ready_schema_for_observation(
            observation_id, allow_close=False, require_stereo=True
        )
        stereo_close_schema = controller_engage_ready_schema_for_observation(
            observation_id, allow_close=True, require_stereo=True
        )
        stereo_base_schema = controller_engage_ready_schema_for_observation(
            observation_id, allow_close=False, require_stereo=True, allow_base=True
        )
        servo_base_schema = controller_engage_ready_schema_for_observation(
            observation_id, allow_close=False, require_stereo=False, allow_base=True
        )
        control_press_schema = controller_control_press_schema_for_observation(
            observation_id
        )
        control_contact_correction_schema = (
            controller_control_contact_correction_schema_for_observation(
                observation_id
            )
        )
        control_contact_path_schemas = [
            controller_control_contact_path_schema_for_observation(
                observation_id, waypoint_index=index
            )
            for index in range(len(COFFEE_CONTROL_CONTACT_WAYPOINTS))
        ]
        control_standing_convergence_schema = (
            controller_control_standing_convergence_schema_for_observation(
                observation_id
            )
        )
        control_preclose_schema = (
            controller_control_preclose_schema_for_observation(observation_id)
        )
        control_retreat_schema = controller_control_retreat_schema_for_observation(
            observation_id
        )
        measured_retreat_schema = controller_control_retreat_schema_for_observation(
            observation_id, measured_contact_path=True
        )
        control_finish_schema = controller_control_finish_schema_for_observation(
            observation_id
        )
        allowed_schemas = {
            strict_canonical_sha256(default_schema): default_schema,
            strict_canonical_sha256(wrist_roll_schema): wrist_roll_schema,
            strict_canonical_sha256(engage_close_schema): engage_close_schema,
            strict_canonical_sha256(engage_servo_schema): engage_servo_schema,
            strict_canonical_sha256(stereo_servo_schema): stereo_servo_schema,
            strict_canonical_sha256(stereo_close_schema): stereo_close_schema,
            strict_canonical_sha256(stereo_base_schema): stereo_base_schema,
            strict_canonical_sha256(servo_base_schema): servo_base_schema,
            strict_canonical_sha256(control_contact_correction_schema): (
                control_contact_correction_schema
            ),
            strict_canonical_sha256(control_standing_convergence_schema): (
                control_standing_convergence_schema
            ),
            strict_canonical_sha256(control_press_schema): control_press_schema,
            strict_canonical_sha256(control_preclose_schema): control_preclose_schema,
            strict_canonical_sha256(control_retreat_schema): control_retreat_schema,
            strict_canonical_sha256(measured_retreat_schema): measured_retreat_schema,
            strict_canonical_sha256(control_finish_schema): control_finish_schema,
        }
        allowed_schemas.update({
            strict_canonical_sha256(schema): schema
            for schema in control_contact_path_schemas
        })
        from .workflow import approach_schema, ready_schema, alternative_schema
        workflow_schemas = [ready_schema(observation_id)] + [
            alternative_schema(observation_id, kind) for kind in ['image_servo','move_joints','cartesian_delta','base_action']
        ] + [
            approach_schema(schema) for schema in list(allowed_schemas.values())
        ]
        allowed_schemas.update({strict_canonical_sha256(schema): schema for schema in workflow_schemas})
        response_schema = allowed_schemas.get(recorded_schema_sha256)
        if response_schema is None:
            raise ValueError("controller response schema is not protocol-authorized")
        attempt, seal = _closure_attempt(
            raw,
            role="controller",
            call_index=call_index,
            response_schema_sha256=strict_canonical_sha256(response_schema),
            allow_truncated=bool(
                proposal.get("status") == "rejected_by_protocol"
                and proposal.get("failure_class") == "MalformedResponse"
            ),
        )
        controller_attempt_by_call[call_index] = (attempt, seal)
        current_model = cast(str, attempt["served_model_id"])
        served_model_id = served_model_id or current_model
        if current_model != served_model_id:
            raise ValueError("controller served-model identity drifted")
    for call_index, raw in enumerate(critic_attempts, start=1):
        attempt, seal = _closure_attempt(
            raw,
            role="critic",
            call_index=call_index,
            response_schema_sha256=critic_schema_sha256,
        )
        critic_attempt_by_call[call_index] = (attempt, seal)
        current_model = cast(str, attempt["served_model_id"])
        served_model_id = served_model_id or current_model
        if current_model != served_model_id:
            raise ValueError("controller/critic served-model identity drifted")
    if expected_served_model_id is not None and (
        not expected_served_model_id
        or served_model_id != expected_served_model_id
    ):
        raise ValueError("attempts do not match the attested served-model identity")

    if any(
        not isinstance(record, Mapping)
        or record.get("role") not in {"controller", "critic"}
        for record in context.attempt_records
    ):
        raise ValueError("AttemptEvidenceLog contains an unknown role")
    expected_attempt_order: list[tuple[str, int]] = []
    for raw_proposal in context.proposal_records:
        if not isinstance(raw_proposal, Mapping):
            raise ValueError("proposal evidence is not a record")
        expected_attempt_order.append((
            "controller",
            _closure_int(
                raw_proposal.get("controller_call_index"),
                label="proposal controller call index",
                minimum=1,
            ),
        ))
        raw_critic_call_index = raw_proposal.get("critic_call_index")
        if raw_critic_call_index is not None:
            expected_attempt_order.append((
                "critic",
                _closure_int(
                    raw_critic_call_index,
                    label="proposal critic call index",
                    minimum=1,
                ),
            ))
    actual_attempt_order = [
        (
            cast(str, record["role"]),
            _closure_int(
                record.get("call_index"),
                label="AttemptEvidenceLog call index",
                minimum=1,
            ),
        )
        for record in context.attempt_records
        if isinstance(record, Mapping)
    ]
    if actual_attempt_order != expected_attempt_order:
        raise ValueError("AttemptEvidenceLog global order drifted")
    critic_records_by_call: dict[int, dict[str, object]] = {}
    for raw in context.critic_records:
        critic_record = _closure_sealed_record(
            raw,
            fields=_PROPOSAL_EVIDENCE_FIELDS,
            label="critic proposal snapshot",
        )
        call_index = _closure_int(
            critic_record.get("critic_call_index"),
            label="critic snapshot call index",
            minimum=1,
        )
        if call_index in critic_records_by_call:
            raise ValueError("critic proposal snapshot replayed")
        _closure_zero_effects(critic_record, label="critic proposal snapshot")
        critic_records_by_call[call_index] = critic_record
    if len(critic_records_by_call) != critic_calls:
        raise ValueError("critic proposal snapshot cardinality drifted")

    approved = 0
    rejected = 0
    executed = 0
    critic_successes = 0
    closed_history: list[str] = []
    if context.proposal_records:
        initial_history = context.proposal_records[0].get(
            "milestone_history_before"
        )
        if not isinstance(initial_history, list):
            raise ValueError("initial milestone history schema drifted")
        for milestone in initial_history:
            if not isinstance(milestone, str):
                raise ValueError("initial milestone history type drifted")
            validate_milestone_transition(context.family, milestone, closed_history)
            closed_history.append(milestone)
    revision_by_observation: dict[str, int] = {}
    immutable_observation: dict[str, tuple[object, ...]] = {}
    for proposal_index, raw_proposal in enumerate(context.proposal_records, start=1):
        proposal = _closure_sealed_record(
            raw_proposal,
            fields=_PROPOSAL_EVIDENCE_FIELDS,
            label="proposal evidence",
        )
        if (
            proposal.get("schema") != "robocasa-qwen-proposal-evidence/v1"
            or proposal.get("task") != context.task
            or proposal.get("family") != context.family
            or _closure_int(
                proposal.get("proposal_index"), label="proposal index", minimum=1
            )
            != proposal_index
        ):
            raise ValueError("proposal identity drifted")
        observation_id = proposal.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError("proposal observation identity drifted")
        revision_index = _closure_int(
            proposal.get("revision_index"), label="proposal revision index"
        )
        expected_revision = revision_by_observation.get(observation_id, 0)
        if (
            revision_index != expected_revision
            or revision_index > PROPOSAL_MAX_REVISIONS_PER_OBSERVATION
        ):
            raise ValueError("proposal revision order drifted")
        revision_by_observation[observation_id] = expected_revision + 1
        state_sha256 = _closure_sha256(
            proposal.get("fresh_public_state_sha256"), label="proposal public state"
        )
        rgb_sha256 = proposal.get("fresh_public_rgb_sha256")
        if not isinstance(rgb_sha256, Mapping) or set(rgb_sha256) != _RGB_LABELS:
            raise ValueError("proposal RGB identity drifted")
        normalized_rgb = tuple(
            _closure_sha256(rgb_sha256[label], label=f"proposal {label} RGB")
            for label in sorted(_RGB_LABELS)
        )
        prior_receipt_sha256 = _closure_sha256(
            proposal.get("immediate_prior_receipt_sha256"),
            label="proposal immediate prior receipt",
            optional=True,
        )
        history_before = proposal.get("milestone_history_before")
        if not isinstance(history_before, list) or history_before != closed_history:
            raise ValueError("proposal milestone history ordering drifted")
        if proposal.get("milestone_history_sha256") != strict_canonical_sha256(
            history_before
        ):
            raise ValueError("proposal milestone history hash drifted")
        immutable = (
            state_sha256,
            normalized_rgb,
            prior_receipt_sha256,
            strict_canonical_sha256(history_before),
        )
        prior_immutable = immutable_observation.setdefault(observation_id, immutable)
        if immutable != prior_immutable:
            raise ValueError("same-observation proposal evidence refreshed")

        draft = proposal.get("draft")
        draft_sha256 = _closure_sha256(
            proposal.get("draft_sha256"), label="proposal draft", optional=True
        )
        if (draft is None) is not (draft_sha256 is None):
            raise ValueError("proposal draft/hash presence drifted")
        if draft is not None and draft_sha256 != strict_canonical_sha256(draft):
            raise ValueError("proposal draft hash drifted")
        expected_kind = (
            draft.get("kind") if isinstance(draft, Mapping) else "malformed"
        )
        if proposal.get("command_kind") != expected_kind:
            raise ValueError("proposal command kind drifted")

        controller_call_index = _closure_int(
            proposal.get("controller_call_index"),
            label="proposal controller call index",
            minimum=1,
        )
        if controller_call_index != proposal_index:
            raise ValueError("proposal controller call order drifted")
        controller_record = _closure_sealed_record(
            context.controller_records[proposal_index - 1],
            fields=_PROPOSAL_CONTROLLER_FIELDS,
            label="proposal controller record",
        )
        if (
            controller_record.get("controller_call_index") != controller_call_index
            or controller_record.get("observation_id") != observation_id
            or controller_record.get("command") != draft
            or controller_record.get("command_sha256") != draft_sha256
            or controller_record.get("status") != proposal.get("status")
            or controller_record.get("controller_request_linkage_status")
            != "complete"
        ):
            raise ValueError("proposal/controller record linkage drifted")
        qwen_attempt_index = _closure_int(
            controller_record.get("qwen_attempt_index"),
            label="controller Qwen attempt index",
        )
        if qwen_attempt_index > 2:
            raise ValueError("controller Qwen attempt index drifted")
        controller_request_sha256 = _closure_sha256(
            proposal.get("controller_request_sha256"), label="controller request"
        )
        controller_manifest_sha256 = _closure_sha256(
            proposal.get("controller_call_manifest_sha256"),
            label="controller call manifest",
        )
        if (
            controller_record.get("controller_request_sha256")
            != controller_request_sha256
            or controller_record.get("controller_call_manifest_sha256")
            != controller_manifest_sha256
        ):
            raise ValueError("controller request/manifest linkage drifted")
        controller_attempt, controller_attempt_seal = controller_attempt_by_call[
            controller_call_index
        ]
        if (
            proposal.get("controller_attempt_record_sha256")
            != controller_attempt_seal
            or controller_attempt.get("request_sha256")
            != controller_request_sha256
            or controller_attempt.get("observation_id") != observation_id
            or controller_attempt.get("attempt_index") != qwen_attempt_index
            or canonical_sanitized_output(
                controller_attempt.get("sanitized_raw_command")
            )
            != draft
        ):
            raise ValueError("controller AttemptEvidenceLog linkage drifted")

        status = proposal.get("status")
        if status not in _PROPOSAL_STATUSES:
            raise ValueError("proposal status drifted")
        audit = proposal.get("audit")
        critic_call_index_raw = proposal.get("critic_call_index")
        if critic_call_index_raw is None:
            if any(
                proposal.get(field) is not None
                for field in (
                    "critic_request_sha256",
                    "critic_call_manifest_sha256",
                    "critic_attempt_record_sha256",
                    "audit",
                    "audit_sha256",
                )
            ):
                raise ValueError("proposal has orphan critic fields")
        else:
            critic_call_index = _closure_int(
                critic_call_index_raw, label="proposal critic call index", minimum=1
            )
            critic_request_sha256 = _closure_sha256(
                proposal.get("critic_request_sha256"), label="critic request"
            )
            _closure_sha256(
                proposal.get("critic_call_manifest_sha256"),
                label="critic call manifest",
            )
            critic_attempt, critic_attempt_seal = critic_attempt_by_call[
                critic_call_index
            ]
            if (
                proposal.get("critic_attempt_record_sha256") != critic_attempt_seal
                or critic_attempt.get("request_sha256") != critic_request_sha256
                or critic_attempt.get("observation_id") != observation_id
            ):
                raise ValueError("critic AttemptEvidenceLog linkage drifted")
            critic_snapshot = critic_records_by_call[critic_call_index]
            for field in _PROPOSAL_EVIDENCE_FIELDS - _PROPOSAL_EFFECT_FIELDS:
                if critic_snapshot.get(field) != proposal.get(field):
                    raise ValueError("critic proposal snapshot linkage drifted")
            if not isinstance(audit, Mapping):
                raise ValueError("audited proposal lacks an audit object")
            validated_audit = validate_proposal_audit(audit)
            attempted_audit = validate_proposal_audit(
                canonical_sanitized_output(
                    critic_attempt.get("sanitized_raw_command")
                )
            )
            if (
                proposal.get("audit_sha256")
                != strict_canonical_sha256(validated_audit)
                or attempted_audit != validated_audit
            ):
                raise ValueError("proposal audit hash/output linkage drifted")
            critic_successes += 1

        if proposal.get("critic_origin_execution") is not False:
            raise ValueError("critic-origin execution drifted")
        claimed_milestone = proposal.get("claimed_milestone")
        if claimed_milestone is not None:
            if not isinstance(claimed_milestone, str):
                raise ValueError("claimed milestone type drifted")
            validate_milestone_transition(
                context.family, claimed_milestone, history_before
            )

        if status == "approved_for_execution":
            approved += 1
            if (
                not isinstance(audit, Mapping)
                or audit.get("verdict") != "approve"
                or proposal.get("contradiction") != "none"
                or proposal.get("suggested_correction")
                != audit.get("suggested_correction")
                or proposal.get("returned_command_sha256") != draft_sha256
                or proposal.get("failure_class") is not None
                or proposal.get("failure_sha256") is not None
                or not isinstance(claimed_milestone, str)
            ):
                raise ValueError("approved proposal semantics drifted")
            if proposal.get("executed") is True:
                executed += 1
                _closure_execution_effects(
                    proposal, command_kind=cast(str, expected_kind)
                )
                expected_after = [*history_before, claimed_milestone]
                if (
                    proposal.get("milestone_closed") is not True
                    or proposal.get("milestone_history_after") != expected_after
                ):
                    raise ValueError("approved milestone closure drifted")
                closed_history = expected_after
            else:
                _closure_zero_effects(proposal, label="unexecuted approved proposal")
                if require_execution_closure:
                    raise ValueError("approved proposal lacks execution closure")
        else:
            rejected += 1
            if proposal.get("returned_command_sha256") is not None:
                raise ValueError("rejected proposal returned a command")
            _closure_zero_effects(proposal, label="rejected proposal")
            if status == "rejected_by_critic":
                if (
                    not isinstance(audit, Mapping)
                    or audit.get("verdict") != "revise"
                    or proposal.get("contradiction") != audit.get("contradiction")
                    or proposal.get("suggested_correction")
                    != audit.get("suggested_correction")
                    or proposal.get("failure_class") is not None
                    or proposal.get("failure_sha256") is not None
                ):
                    raise ValueError("critic rejection semantics drifted")
            else:
                if audit is not None:
                    raise ValueError("protocol failure unexpectedly contains an audit")
                if (
                    not isinstance(proposal.get("failure_class"), str)
                    or not proposal.get("failure_class")
                ):
                    raise ValueError("protocol rejection failure class drifted")
                _closure_sha256(
                    proposal.get("failure_sha256"), label="protocol failure"
                )
        after_history = proposal.get("milestone_history_after")
        if proposal.get("milestone_history_after_sha256") != strict_canonical_sha256(
            after_history
        ):
            raise ValueError("proposal post-milestone hash drifted")

    if critic_successes != _closure_int(
        context.critic_successes, label="critic success count"
    ):
        raise ValueError("critic success count drifted")
    critic_unavailable = _closure_int(
        context.critic_unavailable, label="critic unavailable count"
    )
    if critic_calls != critic_successes + critic_unavailable:
        raise ValueError("critic success/unavailable accounting drifted")
    if closed_history != context.milestone_history:
        raise ValueError("final milestone history drifted")
    return {
        "proposal_count": len(context.proposal_records),
        "controller_attempt_count": len(controller_attempts),
        "critic_attempt_count": len(critic_attempts),
        "rejected_count": rejected,
        "approved_count": approved,
        "executed_count": executed,
        "critic_origin_executions": 0,
    }
