"""Direct Inspect controller with a same-model pre-execution proposal audit."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import cast

import httpx

from .action_preview import render_preview
from .execution_repair import ExecutionRepair

from .cartesian_skill import decode_cartesian_delta
from .critic_protocol import (
    ADVISORY_SCHEMA,
    ARTICULATED_CONTACT_MIN_SEPARATION,
    CRITIC_CONFIG,
    CRITIC_CONFIG_SHA256,
    CRITIC_MAX_ATTEMPTS_PER_TASK,
    CRITIC_MAX_TOKENS,
    CRITIC_TIMEOUT_S,
    IMAGE_SERVO_CLOSE_TOLERANCE_PX,
    IMAGE_SERVO_GRID_CELL_PX,
    MILESTONE_CONTEXT_MARKER,
    OPTIMIZATION_MAX_HOURS,
    OPTIMIZATION_STARTED_AT,
    PHYSICAL_COMMAND_KINDS,
    PRIOR_FULL_RUN_WALL_S,
    PRIOR_MEAN_DECISIONS_PER_TASK,
    PROPOSAL_AUDIT_CONFIG,
    PROPOSAL_AUDIT_CONFIG_SHA256,
    PROPOSAL_AUDIT_SCHEMA,
    PROPOSAL_MAX_REVISIONS_PER_OBSERVATION,
    SOURCE_CONTACT_MIN_SEPARATION,
    STALL_ARM_JOINT_RAD,
    CriticContext,
    ProposalAuditExhausted,
    ProposalAuditFailedClosed,
    articulated_approach_stall_status,
    articulated_contact_is_confirmed,
    articulated_contact_is_provisional,
    articulated_contact_recovery_status,
    articulated_failed_actuation_axis_sets,
    articulated_failed_contact_targets,
    articulated_handle_insertion_is_ready,
    articulated_handle_insertion_receipt,
    articulated_wrist_roll_tracking_status,
    canonical_json_copy,
    canonical_sanitized_output,
    canonical_sha256,
    cartesian_motion_axis_set,
    image_hashes,
    milestone_context_packet,
    milestone_from_note,
    receipt_requested_gripper_intent,
    select_sealed_public_receipt,
    source_grasp_recovery_target,
    strict_canonical_sha256,
    toaster_stall_close_probe_due,
    validate_advisory,
    validate_milestone_action_semantics,
    validate_milestone_transition,
    validate_proposal_audit,
    validate_proposal_audit_closure,
    wrist_roll_revision_required,
)
from .image_servo import _matvec, _rotation_xyzw, decode_image_servo
from .joint_protocol import (
    ACTUAL_RELATIVE_TRACKING,
    JOINT_LIMITS,
    JOINT_NAMES,
    JOINT_TRACKING_MODES,
    LAG_PAUSE_TRACKING,
    decode_joint_command,
)
from .joint_runner import (
    COFFEE_CONTROL_CONTACT_CORRECTION_NOTE,
    COFFEE_CONTROL_CONTACT_PATH_NOTE,
    COFFEE_CONTROL_CONTACT_PRESS_NOTE,
    COFFEE_CONTROL_CONTACT_WAYPOINTS,
    COFFEE_CONTROL_MEASURED_RETREAT_TARGETS,
    COFFEE_CONTROL_PRECLOSE_NOTE,
    COFFEE_CONTROL_STANDING_CONVERGENCE_NOTE,
    CONTROLLER_MAX_TOKENS,
    CONTROLLER_RESPONSE_SCHEMA,
    DERIVED_SKILL_MAX_ACTIONS,
    _image_servo_alignment_status,
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
    joint_system_prompt_parts,
    load_joint_system_prompt,
    prepare_joint_mailbox,
    validate_public_receipt,
)
from .panda_embodiment import (
    compose_world_pose,
    project_world_point,
    validate_public_state,
)

FULL_MAX_DECISIONS = 70
TOASTER_HANDLE_DARK_LUMA_MAX = 70.0
TOASTER_HANDLE_MIN_HORIZONTAL_RUN_PX = 24
TOASTER_HANDLE_TARGET_TOLERANCE_PX = 4
TOASTER_CROSS_VIEW_TOLERANCE_PX = 12
TOASTER_HANDLE_MAX_BAR_HEIGHT_PX = 14
TOASTER_HANDLE_END_MARGIN_PX = 8
COFFEE_BUTTON_DARK_LUMA_MAX = 110.0
COFFEE_BUTTON_TARGET_TOLERANCE_PX = 4
_FROZEN_QWEN_AUTHORITY = {
    "repo_id": "Qwen/Qwen3.8-27B",
    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "served_model_name": "qwen3.8-27b-bf16",
    "dtype": "bfloat16",
}
_FROZEN_QWEN_SNAPSHOT_DIGEST = (
    "27829db7e5189de5f7a0d5ac967b33a4b8ff5e5c28b28a4dd7790599e448ce91"
)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _instruction_context(instruction: str) -> dict[str, object]:
    try:
        value = json.loads(instruction.splitlines()[0])
    except (IndexError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _controller_instruction(
    instruction: str,
    *,
    advisory_id: str | None,
    advisory: Mapping[str, object] | None,
) -> str:
    if advisory_id is None or advisory is None:
        return instruction
    record = validate_advisory(advisory)
    return (
        f"{instruction}\n\nPOST_ACTION_ADVISORY:\n"
        + json.dumps(record, sort_keys=True, separators=(",", ":"))
    )


def _source_close_measurement(failure: Mapping[str, object]) -> dict[str, object]:
    return {
        "measured_end_finger_separation": failure["measured_end_finger_separation"],
    }


def _source_contact_alignment_meaning(
    context: CriticContext, alignment: object,
) -> object:
    if (
        context.family == "grasp_place"
        and context.milestone_history
        and context.milestone_history[-1] in {"pregrasp", "grasp"}
        and isinstance(alignment, Mapping)
    ):
        return {
            **alignment,
            "controller_rule": (
                "This error measures distance to the previously requested pixel. "
                "A small error establishes motion convergence to that request; "
                "it does not locate the source object or establish finger enclosure. "
                "Identify the source in current RGB and compare it with the fingers "
                "before choosing alignment, view recovery, or stationary contact. "
                "Keep the current closed milestone context when choosing the next step."
            ),
        }
    return alignment


def _controller_milestone_context_instruction(
    instruction: str,
    *,
    family: str,
    milestone_history: Sequence[str],
) -> str:
    if MILESTONE_CONTEXT_MARKER in instruction:
        raise ProposalAuditFailedClosed(
            "controller instruction already contains milestone context"
        )
    packet = milestone_context_packet(family, milestone_history)
    return (
        instruction
        + MILESTONE_CONTEXT_MARKER
        + json.dumps(packet, sort_keys=True, separators=(",", ":"))
    )


def _controller_task_grounding_instruction(instruction: str, *, task: str) -> str:
    if task != "StartCoffeeMachine":
        return instruction
    marker = "\n\nCONTROL_TARGET_GROUNDING_CONTEXT:\n"
    if marker in instruction:
        raise ProposalAuditFailedClosed(
            "controller instruction already contains control target grounding"
        )
    return (
        instruction
        + marker
        + "For this coffee machine, the visible start controls are four small "
        "buttons in a small 2x2 array on the upper front panel. Select one "
        "distinct button. The closed finger-side contact projects left and "
        "slightly below the grip site in both external views, so use the "
        "calibrated upper-right contact point on that button's printed box: "
        "u exactly u_max, left v one pixel above v_min, and right v two pixels "
        "above v_min, not "
        "the button center, the midpoint between buttons, or the center of the "
        "whole array. Do not "
        "target the dispenser, mug, machine body, or counter. Confirm the "
        "marker still covers one button in the fresh external RGB. For stereo, "
        "select that calibrated contact point on the same "
        "row-and-column button in both external views; similar pixel values do "
        "not establish physical correspondence. Qwen chooses the camera and "
        "exact pixels."
    )


_ORIENTATION_ROLL_NOTE_GUIDANCE = (
    " The open retreat is already sealed in the prior receipt, so do not "
    "describe or repeat it. Write this note about the roll itself: cite the "
    "current joint7 in `state.arm_joint_position`, the chosen absolute "
    "target, and the predicted effect that the finger gap rotates about the "
    "tool axis while the grip site stays in place."
)


def _controller_articulated_recovery_instruction(
    instruction: str,
    forbidden_targets: Sequence[Mapping[str, object]],
    immediate_prior_receipt: Mapping[str, object] | None = None,
    failed_actuation_axis_sets: Sequence[Sequence[str]] = (),
    *,
    task: str | None = None,
    current_milestone: str | None = None,
) -> str:
    incomplete_roll = articulated_wrist_roll_tracking_status(
        immediate_prior_receipt
    )
    if (
        isinstance(incomplete_roll, Mapping)
        and incomplete_roll.get("settled") is False
    ):
        marker = "\n\nARTICULATED_WRIST_ROLL_TRACKING_CONTEXT:\n"
        if marker in instruction:
            raise ProposalAuditFailedClosed(
                "controller instruction already contains wrist-roll tracking context"
            )
        return (
            instruction
            + marker
            + json.dumps(
                dict(incomplete_roll), sort_keys=True, separators=(",", ":")
            )
            + "\nThe prior receipt reports an unfinished joint7 orientation "
            "target. Keep milestone=engage and the gripper open. Author "
            "`move_joints` with targets containing only `joint7` and "
            "`gripper`, set gripper to `1.0`, and repeat that same "
            "Qwen-authored target exactly. Do not choose a new joint7 target "
            "or resume image servo until a sealed receipt reports that target "
            "settled."
        )
    if (
        isinstance(incomplete_roll, Mapping)
        and incomplete_roll.get("settled") is True
    ):
        marker = "\n\nARTICULATED_WRIST_ROLL_COMPLETED_CONTEXT:\n"
        if marker in instruction:
            raise ProposalAuditFailedClosed(
                "controller instruction already contains completed wrist-roll context"
            )
        packet = {
            **dict(incomplete_roll),
            "pre_roll_empty_targets_cleared": True,
            "required_milestone": "engage",
        }
        return (
            instruction
            + marker
            + json.dumps(packet, sort_keys=True, separators=(",", ":"))
            + "\nThe joint7 orientation recovery is complete and starts a new "
            "handle-contact epoch. Pre-roll empty closes explain the completed "
            "orientation change but no longer require another retreat or wrist "
            "roll. Keep milestone=engage, re-localize a visible handle point "
            "from the fresh RGB, and author a positive-depth `image_servo` "
            "with the gripper `open` to insert using the settled orientation. "
            "Qwen chooses the camera, pixel, depth, and step."
        )
    residual = (
        immediate_prior_receipt.get("gripper_residual")
        if isinstance(immediate_prior_receipt, Mapping)
        else None
    )
    separation = (
        residual.get("measured_end_finger_separation")
        if isinstance(residual, Mapping)
        else None
    )
    contact_observed = bool(
        isinstance(immediate_prior_receipt, Mapping)
        and receipt_requested_gripper_intent(immediate_prior_receipt)
        in {"hold", "close"}
        and not isinstance(separation, bool)
        and isinstance(separation, (int, float))
        and math.isfinite(float(separation))
        and float(separation) > ARTICULATED_CONTACT_MIN_SEPARATION
    )
    contact_confirmed = bool(
        isinstance(immediate_prior_receipt, Mapping)
        and articulated_contact_is_confirmed(immediate_prior_receipt)
    )
    if contact_observed:
        raw_engagement_translation = immediate_prior_receipt.get(
            "resolved_translation_m"
        )
        engagement_translation = (
            [float(value) for value in raw_engagement_translation]
            if isinstance(raw_engagement_translation, Sequence)
            and not isinstance(raw_engagement_translation, (str, bytes))
            and len(raw_engagement_translation) == 3
            and all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                for value in raw_engagement_translation
            )
            else None
        )
        marker = "\n\nARTICULATED_CONTACT_CONTEXT:\n"
        if marker in instruction:
            raise ProposalAuditFailedClosed(
                "controller instruction already contains articulated contact context"
            )
        packet = {
            "measured_end_finger_separation": float(cast(float, separation)),
            "contact_threshold": ARTICULATED_CONTACT_MIN_SEPARATION,
            "contact_established": True,
            "contact_confirmation_required": not contact_confirmed,
            "required_milestone": "actuate",
            "failed_actuation_axis_sets": [
                list(item) for item in failed_actuation_axis_sets
            ],
            "engagement_translation_m": engagement_translation,
            "first_actuation_strategy": (
                "image_plane_articulation_motion"
                if contact_confirmed
                else "stationary_contact_confirmation"
            ),
        }
        prefix = (
            instruction
            + marker
            + json.dumps(packet, sort_keys=True, separators=(",", ":"))
        )
        if not contact_confirmed:
            return (
                prefix
                + "\nThe first obstruction after a close is provisional. "
                "Advance to milestone=actuate and author `move_joints` with "
                'only `targets={"gripper":0.0}`; do not move any arm joint. '
                "The next sealed receipt must show whether contact persists "
                "before any articulation motion."
            )
        return (
            prefix
            + "\nThe stationary confirmation preserved contact strictly above "
            "the empty threshold. Advance at milestone=actuate and prefer "
            "`image_servo` with `target_role=articulation_motion` and gripper "
            "`hold` or `close`. Its target pixel is the desired future image "
            "waypoint implied by visible hinge, drawer, or knob motion, so it "
            "may lie off the current handle material. Keep contact while "
            "tracing that motion. For the first motion after each confirmed "
            "contact, choose a waypoint at most 16 pixels from the current "
            "end-effector pixel in that camera, set `depth_delta_m=0.0` and "
            "`step_m` between 0.01 and 0.02, then inspect the next receipt; "
            "once a receipt shows the mechanism moved with contact kept, use "
            "`step_m=0.03` per command because the action budget is finite. "
            "Do not merely reverse `engagement_translation_m`: the "
            "camera ray is not necessarily the mechanism tangent. A bounded "
            "nonzero `cartesian_delta` remains available when base-frame "
            "geometry is clear, but must not repeat any listed active-axis "
            "set. Qwen chooses every pixel, depth, and motor value."
        )
    if not forbidden_targets:
        return instruction
    marker = "\n\nARTICULATED_RECOVERY_CONTEXT:\n"
    if marker in instruction:
        raise ProposalAuditFailedClosed(
            "controller instruction already contains articulated recovery context"
        )
    required_milestone = (
        "actuate" if current_milestone == "actuate" else "engage"
    )
    packet = {
        "forbidden_image_servo_targets": [dict(item) for item in forbidden_targets],
        "minimum_target_shift_px": IMAGE_SERVO_GRID_CELL_PX,
        "target_shift_metric": "manhattan_l1",
        "contact_min_separation": ARTICULATED_CONTACT_MIN_SEPARATION,
        "required_milestone": required_milestone,
    }
    orientation_failure_count = 1 if task == "OpenToasterOvenDoor" else 2
    orientation_recovery_required = bool(
        len(forbidden_targets) >= orientation_failure_count
        and isinstance(immediate_prior_receipt, Mapping)
        and immediate_prior_receipt.get("kind") == "cartesian_delta"
        and immediate_prior_receipt.get("requested_gripper") == "open"
    )
    if orientation_recovery_required:
        packet["orientation_recovery_required"] = True
        packet["distinct_empty_target_count"] = len(forbidden_targets)
        trigger_text = (
            "The visually screened toaster handle target produced an empty "
            "stationary close."
            if orientation_failure_count == 1
            else "Two distinct handle pixels have already produced empty "
            "stationary closes."
        )
        return (
            instruction
            + marker
            + json.dumps(packet, sort_keys=True, separators=(",", ":"))
            + "\n"
            + trigger_text
            + " Stop selecting another handle pixel. Keep "
            "milestone=engage and author `move_joints` with targets containing "
            "only `joint7` and `gripper`; keep gripper at `1.0` and change "
            "joint7 so the finger gap rolls toward perpendicular to the "
            "visible elongated handle. Read the current joint7 from public "
            "state and choose a safe absolute roll target with meaningful "
            "margin yourself. Reobserve before another insertion."
            + _ORIENTATION_ROLL_NOTE_GUIDANCE
        )
    recovery_edge = (
        "An empty close is not contact and cannot advance to actuation. Keep "
        "milestone=engage and reopen before retrying: the next physical "
        "command must use gripper `open`. "
        if required_milestone == "engage"
        else "Actuation contact was lost. Keep milestone=actuate and reopen "
        "before retrying: the next physical command must use gripper `open`. "
    )
    return (
        instruction
        + marker
        + json.dumps(packet, sort_keys=True, separators=(",", ":"))
        + "\n"
        + recovery_edge
        + "After any empty close, do not remain "
        "near its listed "
        "camera+target_pixel pair. Qwen must "
        "re-localize the visible handle from the fresh RGB and choose a point "
        "whose combined |delta_u|+|delta_v| is at least 32 pixels. For an "
        "elongated handle, move along its visible long-axis interior midline; "
        "this is a new grasp point, not an articulation waypoint. Once chosen, "
        "keep that recovery camera+target_pixel fixed across continued servo "
        "steps until its stationary close tests contact; do not chase along "
        "the handle merely because the end effector moved."
    )


def _model_call_manifest_sha256(kwargs: Mapping[str, object]) -> str:
    manifest = dict(kwargs)
    manifest.pop("attempt_log", None)
    images = manifest.get("images")
    if isinstance(images, Mapping):
        manifest["images"] = image_hashes(images)
    return canonical_sha256(manifest)


_ATTEMPT_EVIDENCE_FIELDS = {
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
_ATTEMPT_RECORD_FIELDS = {
    "schema",
    "role",
    "call_index",
    "record",
    "record_sha256",
}
_QWEN_MAX_ATTEMPT_INDEX = 2


class _AttemptEvidenceLinkageIncomplete(RuntimeError):
    pass


class _CapturingAttemptLog:
    def __init__(self, owner: object | None) -> None:
        self.owner = owner
        self.records: list[dict[str, object]] = []

    def append(self, record: Mapping[str, object]) -> None:
        snapshot = cast(dict[str, object], canonical_json_copy(dict(record)))
        self.records.append(snapshot)
        if self.owner is not None:
            self.owner.append(record)  # type: ignore[attr-defined]


def _persist_captured_attempt_records(
    context: CriticContext,
    recorder: _CapturingAttemptLog,
    *,
    role: str,
    call_index: int,
) -> None:
    for record in recorder.records:
        snapshot = cast(dict[str, object], canonical_json_copy(record))
        context.attempt_records.append({
            "schema": "robocasa-qwen-attempt-evidence/v1",
            "role": role,
            "call_index": call_index,
            "record": snapshot,
            "record_sha256": strict_canonical_sha256(snapshot),
        })


def _with_captured_attempt_log(
    kwargs: Mapping[str, object],
) -> tuple[dict[str, object], _CapturingAttemptLog]:
    call_kwargs = dict(kwargs)
    owner_log = call_kwargs.get("attempt_log")
    recorder = _CapturingAttemptLog(owner_log)
    call_kwargs["attempt_log"] = recorder
    return call_kwargs, recorder


def _valid_sha256(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        return None
    return normalized


def _response_schema_sha256(value: object) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _actual_request_sha256_from_attempt_log(
    recorder: _CapturingAttemptLog,
    *,
    observation_id: str,
    attempt_index: int,
    response_schema: object,
) -> str:
    if len(recorder.records) != 1:
        raise _AttemptEvidenceLinkageIncomplete(
            "Qwen attempt evidence did not contain exactly one record"
        )
    expected_schema_sha256 = _response_schema_sha256(response_schema)
    record = recorder.records[0]
    if set(record) != _ATTEMPT_EVIDENCE_FIELDS:
        raise _AttemptEvidenceLinkageIncomplete(
            "Qwen attempt evidence schema drifted"
        )
    logged_attempt_index = record.get("attempt_index")
    if (
        record.get("observation_id") != observation_id
        or isinstance(logged_attempt_index, bool)
        or not isinstance(logged_attempt_index, int)
        or logged_attempt_index != attempt_index
        or record.get("response_schema_sha256") != expected_schema_sha256
    ):
        raise _AttemptEvidenceLinkageIncomplete(
            "Qwen attempt evidence did not match this call"
        )
    request_sha256 = _valid_sha256(record.get("request_sha256"))
    if request_sha256 is None:
        raise _AttemptEvidenceLinkageIncomplete(
            "Qwen attempt evidence request hash was missing"
        )
    return request_sha256


def _request_linkage_from_attempt_log(
    recorder: _CapturingAttemptLog,
    *,
    observation_id: str,
    attempt_index: int,
    response_schema: object,
) -> tuple[str | None, str]:
    try:
        return (
            _actual_request_sha256_from_attempt_log(
                recorder,
                observation_id=observation_id,
                attempt_index=attempt_index,
                response_schema=response_schema,
            ),
            "complete",
        )
    except _AttemptEvidenceLinkageIncomplete:
        return None, "incomplete"


def _critic_instruction(
    context: CriticContext,
    decision: Mapping[str, object],
    public_state: Mapping[str, object],
    images: Mapping[str, bytes],
) -> str:
    sealed_receipt = decision["sealed_public_receipt"]
    fresh_public_state = decision.get("fresh_public_state", public_state)
    fresh_public_state_sha256 = decision.get("fresh_public_state_sha256")
    if not isinstance(fresh_public_state_sha256, str):
        fresh_public_state_sha256 = strict_canonical_sha256(fresh_public_state)
    value = {
        "schema": "robocasa-qwen-advisory-request/v1",
        "task": context.task,
        "family": context.family,
        "trigger": decision["fired_trigger"],
        "previous_executed_command": decision["previous_executed_command"],
        "previous_executed_command_sha256": decision[
            "previous_executed_command_sha256"
        ],
        "sealed_public_receipt": sealed_receipt,
        "sealed_public_receipt_sha256": decision["sealed_public_receipt_sha256"],
        "fresh_observation_id": decision["observation_id"],
        "fresh_public_state": fresh_public_state,
        "fresh_public_state_sha256": fresh_public_state_sha256,
        "fresh_public_rgb_sha256": image_hashes(images),
        "fresh_outcome_delta": decision["outcome_delta"],
        "constraints": {
            "advisory_only": True,
            "controller_is_sole_command_emitter": True,
            "enum_labels_only": True,
            "critic_must_not_emit_motor_fields": True,
            "controller_command_returned_unchanged": True,
        },
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _set_timeout(client: object, seconds: int) -> None:
    http = getattr(client, "_http", None)
    if http is not None:
        http.timeout = httpx.Timeout(seconds)


def _failure_skill_packet(
    context: CriticContext,
    recent_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Retrieve learned failure patterns and retain compact public counterexamples."""
    def remember(event: dict[str, object]) -> None:
        key = (event["kind"], event["observation_id"], event.get("proposal_index"))
        if not any((old["kind"], old["observation_id"], old.get("proposal_index")) == key
                   for old in context.failure_skill_memory):
            context.failure_skill_memory.append(event)

    recovery = None
    if context.family == "grasp_place":
        for index, receipt in enumerate(recent_receipts):
            if receipt_requested_gripper_intent(receipt) != "close":
                continue
            failed = source_grasp_recovery_target(recent_receipts[:index + 1])
            if failed is not None:
                remember({"kind": "empty_source_close",
                          "observation_id": receipt.get("observation_id", context.proposal_observation_id),
                          "outcome": "empty_source_close", "evidence_status": "public_receipt",
                          "failed_approach": failed})
        latest_close = next((receipt for receipt in reversed(recent_receipts)
                             if receipt_requested_gripper_intent(receipt) == "close"), None)
        if latest_close is not None:
            context.active_source_failure = next((case for case in reversed(context.failure_skill_memory)
                                                  if case["kind"] == "empty_source_close"
                                                  and case["observation_id"] == latest_close.get("observation_id", context.proposal_observation_id)), None)
        if context.active_source_failure is not None:
            recovery = context.active_source_failure["failed_approach"]
    predecessor = context.proposal_records[-1] if context.proposal_records else None
    malformed = bool(
        predecessor and predecessor.get("observation_id") == context.proposal_observation_id
        and predecessor.get("status") == "rejected_by_protocol"
        and predecessor.get("failure_class") == "MalformedResponse"
    )
    if malformed:
        remember({"kind": "malformed_proposal", "observation_id": context.proposal_observation_id,
                  "proposal_index": predecessor["proposal_index"],
                  "outcome": "proposal_not_executed", "evidence_status": "protocol_rejection"})
    stage = context.milestone_history[-1] if context.milestone_history else "observe"
    triggers = ["malformed_proposal"] if malformed else []
    if context.family == "grasp_place" and (stage == "pregrasp" or recovery is not None):
        triggers.append("source_contact")
    if recovery is not None:
        triggers.append("empty_source_close")
    if not triggers:
        return None
    bank = json.loads((Path(__file__).resolve().parent.parent / "prompts/failure_skills.txt").read_text())
    skills = [card for trigger in triggers for card in bank["skills"] if card["trigger"] == trigger][:2]
    recent_cases = [case for case in context.failure_skill_memory if case["kind"] in triggers][-2:]
    return {"library_version": bank["version"], "skills": skills, "recent_cases": recent_cases,
            "prior_cases": [case for case in bank.get("prior_cases", []) if case["kind"] in triggers][-2:],
            "use": "These are scoped past failures, not a successful motor program. Current public evidence takes precedence. Choose the relevant lesson and state a specific hypothesis and expected new observable evidence in your command note. Qwen authors all action values; use the existing schema and review process."}


def _proposal_revision_instruction(
    instruction: str,
    *,
    contradiction: str,
    evidence: list[str],
    suggested_correction: str,
    confidence: str,
    rejected_draft: object,
    required_observation_id: str,
    immediate_prior_receipt: Mapping[str, object] | None = None,
    recent_receipts: Sequence[Mapping[str, object]] = (),
    failed_actuation_axis_sets: Sequence[Sequence[str]] = (),
    task: str | None = None,
    protocol_message: str | None = None,
    rejected_history: Sequence[Mapping[str, object]] = (),
) -> str:
    rejected_snapshot = canonical_json_copy(rejected_draft)
    advice = {
        "contradiction": contradiction,
        "evidence": evidence,
        "suggested_correction": suggested_correction,
        "confidence": confidence,
        "rejected_draft": rejected_snapshot,
        "rejected_draft_sha256": strict_canonical_sha256(rejected_snapshot),
        "required_observation_id": required_observation_id,
    }
    if isinstance(rejected_snapshot, str) and len(rejected_snapshot) > 2048:
        # A malformed response can contain thousands of escaped newlines.
        # Keep its full audit evidence, but only preview it in the retry input.
        advice["rejected_draft"] = rejected_snapshot[:2048]
        advice["rejected_draft_omitted_chars"] = len(rejected_snapshot) - 2048
    if protocol_message:
        advice["protocol_rejection"] = protocol_message
    if rejected_history:
        advice["rejected_this_observation"] = [
            canonical_json_copy(item) for item in rejected_history
        ]
        advice["revision_number"] = len(rejected_history)
    alignment_rule = ""
    contact_status = _actuation_contact_status("actuate", immediate_prior_receipt)
    incomplete_roll = articulated_wrist_roll_tracking_status(
        immediate_prior_receipt
    )
    orientation_failure_count = 1 if task == "OpenToasterOvenDoor" else 2
    orientation_recovery_due = bool(
        len(articulated_failed_contact_targets(recent_receipts))
        >= orientation_failure_count
        and isinstance(immediate_prior_receipt, Mapping)
        and immediate_prior_receipt.get("kind") == "cartesian_delta"
        and immediate_prior_receipt.get("requested_gripper") == "open"
    )
    failed_target_violation: tuple[object, object, float] | None = None
    if (
        isinstance(rejected_snapshot, Mapping)
        and rejected_snapshot.get("kind") == "image_servo"
        and rejected_snapshot.get("target_role") == "fixture_handle"
    ):
        rejected_camera = rejected_snapshot.get("camera")
        rejected_target = rejected_snapshot.get("target_pixel")
        if (
            isinstance(rejected_camera, str)
            and isinstance(rejected_target, Sequence)
            and not isinstance(rejected_target, (str, bytes))
            and len(rejected_target) == 2
        ):
            for failed in articulated_failed_contact_targets(recent_receipts):
                failed_target = failed.get("target_pixel")
                if (
                    failed.get("camera") != rejected_camera
                    or not isinstance(failed_target, Sequence)
                    or isinstance(failed_target, (str, bytes))
                    or len(failed_target) != 2
                ):
                    continue
                try:
                    shift = abs(
                        float(rejected_target[0]) - float(failed_target[0])
                    ) + abs(float(rejected_target[1]) - float(failed_target[1]))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(shift) and shift < IMAGE_SERVO_GRID_CELL_PX:
                    failed_target_violation = (
                        rejected_target,
                        failed_target,
                        shift,
                    )
                    break
    if (
        isinstance(incomplete_roll, Mapping)
        and incomplete_roll.get("settled") is False
    ):
        alignment_rule = (
            " The prior receipt reports an unfinished joint7 orientation "
            "target with tracking_not_settled. Keep milestone=engage and "
            "author `move_joints` with only `joint7` and `gripper`; keep "
            "gripper at `1.0` and repeat that same Qwen-authored target "
            "exactly. Do not choose a new target or resume image servo until "
            "the repeated command settles."
        )
    elif (
        isinstance(incomplete_roll, Mapping)
        and incomplete_roll.get("settled") is True
    ):
        alignment_rule = (
            " The prior receipt proves that the joint7 orientation recovery "
            "is settled. It starts a new handle-contact epoch: pre-roll empty "
            "closes no longer require another retreat or wrist roll. Keep "
            "milestone=engage and author a positive-depth `image_servo` toward "
            "a handle point grounded in the fresh RGB with gripper `open`. "
            "Qwen alone chooses the camera, pixel, depth, and step."
        )
    elif protocol_message and "empty-grasp retry" in protocol_message:
        alignment_rule = (
            " The prior source close was empty even though its pixel aligned. "
            "Remain at milestone=grasp. Qwen may retain a correctly localized "
            "source point and author a changed nonzero depth or a newly grounded "
            "stereo pair, or choose a different visible source point. A changed "
            "contact geometry is required before re-closing; another open retreat "
            "alone does not satisfy it. Qwen alone chooses every numeric value."
        )
    elif protocol_message and "other external view" in protocol_message:
        alignment_rule = (
            " The other external view shows the public grip-site pixel off the "
            "dark pull, so the grip site is at the wrong depth along the selected "
            "camera ray and a stationary close is not allowed. Keep "
            "milestone=engage and author `image_servo` with gripper `open`: the "
            "same camera and pixel with a nonzero `depth_delta_m`, or the other "
            "camera with a pixel on the pull. Qwen alone chooses the camera, "
            "pixel, depth, and step; read the other view's "
            "`end_effector_external_pixel_displacement` in the next receipt."
        )
    elif protocol_message and "endpoint is unsafe" in protocol_message:
        alignment_rule = (
            " The rejected servo needs a joint endpoint outside its hard limit; "
            "the rejection names the joint, its value, and its limits. A "
            "straight-arm `joint4` limit means the target is beyond reach from "
            "the current base position: author a bounded `base_action` (one "
            "axis, |normalized_velocity| at most 0.25, gripper `open`) that moves "
            "the base toward the target, then re-issue the same stereo target. "
            "Repeating the servo or only shrinking `step_m` cannot restore reach. "
            "Qwen alone chooses the axis and velocity."
        )
    elif protocol_message and "control press produced no effect" in protocol_message:
        alignment_rule = (
            " `finish` is rejected: the press receipts show no wrench or visual "
            "effect, so the probe missed the control (usually depth). Keep "
            "milestone=actuate or engage, re-align with a stereo `image_servo` "
            "(both pixels on the control, gripper close), then press again with a "
            "bounded `cartesian_delta`; finish only after a receipt shows the "
            "force or RGB effect. Qwen alone chooses every value."
        )
    elif (
        protocol_message
        and "must reuse the sealed Qwen-authored coffee button"
        in protocol_message
    ):
        alignment_rule = (
            " The force-seeking press must reuse the exact camera and "
            "target_pixel printed in the protocol rejection. That visible "
            "button identity was sealed before the closed gripper occluded "
            "the panel; do not consult fresh button candidates or switch to "
            "another member of the 2x2 array. Qwen alone chooses the positive "
            "depth and step."
        )
    elif (
        protocol_message
        and "coffee machine" in protocol_message
        and "distinct visible button" in protocol_message
    ):
        alignment_rule = (
            " The selected coffee-machine pixel is not on one detected button. "
            "Read the COFFEE_BUTTON_CANDIDATES boxes from this same fresh RGB, "
            "choose one box, and use its calibrated contact point: u equals "
            "u_max; left-view v equals v_min minus 1; right-view v equals v_min "
            "minus 2. Follow the exact pixel in "
            "the protocol rejection. For stereo choose the matching physical "
            "button with the same printed grid_row and grid_column in the other "
            "view; similar pixel values do not prove correspondence, and do not "
            "repeat the cluster midpoint. Qwen alone chooses the camera, exact "
            "pixels, depth, and step."
        )
    elif protocol_message and "approach stalled" in protocol_message:
        alignment_rule = (
            " The rejected servo repeats a blocked approach: the grip site no "
            "longer moves toward that target. Keep the gripper open and change "
            "the approach geometry instead: author `move_joints` from the planar "
            "formulas that keep the current radius and height but point the tool "
            "forward (`joint2 - joint4 - joint6` about -1.0 to -1.2 rad; -pi/2 "
            "exceeds joint6's limit) with a vertical finger gap, or a bounded "
            "open `cartesian_delta` back and up before a "
            "front re-approach. Qwen alone chooses every value."
        )
    elif protocol_message and "ray gap" in protocol_message:
        alignment_rule = (
            " The two pixels of the rejected stereo `image_servo` do not name "
            "one physical point: the public camera rays miss each other by the "
            "stated gap (the limit is 0.04 m). Re-read both external views and "
            "pick `target_pixel` and `other_view_pixel` on the same part of the "
            "same object (use the TOASTER_PULL_CANDIDATES boxes when present); "
            "if the point is not visible in the other view, set "
            "`other_view_pixel` to null, which is allowed on this revision. Never "
            "use the published grip-site pixel (`state.end_effector_external_pixels`) "
            "as either pixel: it is where the gripper is, not the object. "
            "Qwen alone chooses the camera, pixels, depth, and step."
        )
    elif protocol_message and "end cap of the pull" in protocol_message:
        alignment_rule = (
            " The rejected pixel sits on the end of the pull, where a close "
            "misses the bar. The rejection names the interior `u` range of that "
            "bar; author the same open positive-depth `image_servo` with "
            "`target_pixel` near the bar's midpoint (and `other_view_pixel` at "
            "the matching midpoint of the same bar in the other view). Qwen "
            "alone chooses the pixels, depth, and step."
        )
    elif protocol_message and "the dark pull spans" in protocol_message:
        alignment_rule = (
            " The rejected target center is outside the visible narrow "
            "horizontal handle. The protocol rejection names the pixel span of "
            "the dark pull it detected in that view; author a positive-depth "
            "`image_servo` with gripper `open` at a pixel of your choice on the "
            "interior midline of that span. Qwen alone chooses the camera, "
            "pixel, depth, and step; do not repeat the rejected pixel."
        )
    elif orientation_recovery_due:
        rejected_targets = (
            rejected_snapshot.get("targets")
            if isinstance(rejected_snapshot, Mapping)
            else None
        )
        rejected_was_roll = bool(
            isinstance(rejected_snapshot, Mapping)
            and rejected_snapshot.get("kind") == "move_joints"
            and isinstance(rejected_targets, Mapping)
            and set(rejected_targets) == {"gripper", "joint7"}
            and rejected_targets.get("gripper") == 1.0
        )
        trigger_text = (
            " The visually screened toaster handle target has already "
            "produced an empty stationary close."
            if orientation_failure_count == 1
            else " Two distinct handle targets have already produced empty "
            "stationary closes."
        )
        if rejected_was_roll:
            alignment_rule = (
                trigger_text
                + " Your rejected draft already had the required form: "
                "`move_joints` with only `joint7` and `gripper` at `1.0`. The "
                "contradiction concerns the note and predicted effect, not the "
                "command form. Keep a joint7-only open roll, choose the absolute "
                "joint7 target again yourself, and rewrite the note for this "
                "roll."
                + _ORIENTATION_ROLL_NOTE_GUIDANCE
            )
        else:
            alignment_rule = (
                trigger_text
                + " Stop selecting another handle pixel. Author "
                "`move_joints` with targets containing only `joint7` and "
                "`gripper`; keep gripper at `1.0`, change joint7 toward a "
                "perpendicular finger gap, and reobserve before another "
                "insertion. Qwen alone chooses the safe absolute joint7 target."
                + _ORIENTATION_ROLL_NOTE_GUIDANCE
            )
    elif failed_target_violation is not None:
        rejected_target, failed_target, shift = failed_target_violation
        alignment_rule = (
            " The rejected fixture-handle target "
            f"{json.dumps(rejected_target, separators=(',', ':'))} is only "
            f"{shift:g} Manhattan pixels from failed target "
            f"{json.dumps(failed_target, separators=(',', ':'))}; the minimum "
            f"is {IMAGE_SERVO_GRID_CELL_PX:g}. It remains forbidden. Author a "
            "positive-depth `image_servo` with gripper `open` at a fresh "
            "visible handle point whose Manhattan distance is comfortably "
            "above the minimum from every failed target. Leave margin for "
            "approximate pixel localization; do not round a smaller shift up "
            "or repeat the rejected target. Qwen alone chooses the camera, "
            "pixel, depth, and step."
        )
    elif (
        contradiction == "visual_alignment_unverified"
        and evidence == ["external_rgb"]
        and isinstance(rejected_snapshot, Mapping)
        and rejected_snapshot.get("kind") == "image_servo"
        and rejected_snapshot.get("target_role") == "fixture_handle"
        and rejected_snapshot.get("gripper") == "open"
    ):
        alignment_rule = (
            " The rejected target center is outside the visible narrow "
            "horizontal handle. On the same fresh RGB, distinguish the long "
            "dark protruding pull from the broad door panel, then author a "
            "positive-depth `image_servo` with gripper `open` at a corrected "
            "pixel on the pull itself. Qwen alone chooses the camera, pixel, "
            "depth, and step; do not repeat the rejected pixel."
        )
    elif (
        contradiction == "visual_alignment_unverified"
        and isinstance(rejected_snapshot, Mapping)
        and rejected_snapshot.get("kind") == "cartesian_delta"
        and isinstance(contact_status, Mapping)
        and contact_status.get("preserved") is True
    ):
        alignment_rule = (
            " The sealed contact closes the visual-alignment gate. Do not "
            "return to `image_servo`. Remain at `milestone=actuate`, keep the "
            "gripper `hold` or `close`, and change the Cartesian actuation "
            "direction with a nonzero `cartesian_delta`. Qwen alone chooses "
            "every Cartesian value."
        )
    elif (
        contradiction == "visual_alignment_unverified"
        and isinstance(rejected_snapshot, Mapping)
        and rejected_snapshot.get("kind") == "image_servo"
        and rejected_snapshot.get("target_role") == "fixture_handle"
        and rejected_snapshot.get("gripper") == "close"
        and "jacobian_projection" in evidence
    ):
        alignment_rule = (
            " The public end-effector pixel is outside the required 8-pixel "
            "contact tolerance of the selected handle point. Remain at "
            "`milestone=approach` and "
            "author an `image_servo` toward the visible handle; keep the "
            "gripper `open`. Qwen alone chooses the target pixel, depth, "
            "and step. Do not close or roll `joint7` on this frozen "
            "observation. The replacement note must describe this open "
            "alignment command and its chosen depth sign; do not copy the "
            "rejected close description."
        )
    elif (
        contradiction == "visual_alignment_unverified"
        and isinstance(rejected_snapshot, Mapping)
        and rejected_snapshot.get("kind") == "image_servo"
        and rejected_snapshot.get("target_role") == "fixture_handle"
        and rejected_snapshot.get("gripper") == "close"
    ):
        alignment_rule = (
            " The wrist view shows that the gripper orientation is not ready "
            "for this elongated handle. Remain at `milestone=approach` and "
            "author a `move_joints` command that changes `joint7` while "
            "keeping the gripper target `1.0`, so the finger gap is "
            "perpendicular to the elongated handle. Read the current joint7 "
            "value and choose the safe absolute roll target yourself. Do not "
            "close again on this frozen observation; reobserve after the roll."
        )
    elif (
        contradiction == "visual_alignment_unverified"
        and isinstance(rejected_snapshot, Mapping)
        and rejected_snapshot.get("kind") == "image_servo"
    ):
        alignment_rule = (
            " For this rejected `image_servo`, you must author a nonzero "
            "`base_action` or `cartesian_delta` view-gathering move with a "
            "safe open gripper. Do not guess another pixel and do not author "
            "another `image_servo` on this frozen observation. Qwen alone "
            "chooses the view-gathering command and every value."
        )
    elif (
        contradiction == "visual_alignment_unverified"
        and isinstance(rejected_snapshot, Mapping)
        and rejected_snapshot.get("kind") == "cartesian_delta"
    ):
        alignment_rule = (
            " Visual alignment is still open. You must author an "
            "`image_servo` toward a visually grounded task surface. Do not "
            "author another `cartesian_delta` until a public pixel receipt "
            "closes the alignment gate. Qwen alone chooses every pixel and "
            "servo value."
        )
    elif (
        contradiction == "stagnation"
        and isinstance(rejected_snapshot, Mapping)
        and rejected_snapshot.get("kind") == "cartesian_delta"
    ):
        alignment_rule = (
            " This replacement must create a nonzero motor effect through "
            "translation, rotation, or a real gripper transition. Do not wait "
            "for settle when public joint velocity is already near zero, and "
            "do not repeat an all-zero Cartesian payload."
        )
    elif (
        contradiction == "motion_effect_mismatch"
        and isinstance(rejected_snapshot, Mapping)
        and rejected_snapshot.get("kind") == "image_servo"
        and rejected_snapshot.get("target_role") == "fixture_handle"
        and rejected_snapshot.get("gripper") == "open"
        and isinstance(rejected_snapshot.get("depth_delta_m"), (int, float))
        and not isinstance(rejected_snapshot.get("depth_delta_m"), bool)
        and float(cast(float, rejected_snapshot["depth_delta_m"])) <= 0.0
    ):
        alignment_rule = (
            " Negative `depth_delta_m` retreats toward the selected camera. "
            "Remain at `milestone=engage` and author positive depth with "
            "gripper `open` to insert around the visible handle before the "
            "stationary close. Qwen alone chooses its magnitude and every "
            "other motor-bearing value."
        )
    elif (
        contradiction == "motion_effect_mismatch"
        and isinstance(rejected_snapshot, Mapping)
        and rejected_snapshot.get("kind") == "image_servo"
        and rejected_snapshot.get("gripper") == "close"
        and isinstance(rejected_snapshot.get("depth_delta_m"), (int, float))
        and not isinstance(rejected_snapshot.get("depth_delta_m"), bool)
        and float(cast(float, rejected_snapshot["depth_delta_m"])) <= 0.0
    ):
        alignment_rule = (
            " Negative `depth_delta_m` retreats toward the selected camera "
            "and cannot produce contact with the selected scene surface. "
            "Remain at the required contact milestone and author a positive "
            "`depth_delta_m` with gripper `close`. Qwen alone chooses its "
            "magnitude and every other motor-bearing value."
        )
    elif (
        contradiction == "motion_effect_mismatch"
        and isinstance(rejected_snapshot, Mapping)
        and rejected_snapshot.get("kind") == "move_joints"
        and isinstance(rejected_snapshot.get("targets"), Mapping)
        and set(cast(Mapping[str, object], rejected_snapshot["targets"]))
        == {"gripper"}
    ):
        alignment_rule = (
            " This is a gripper-only `move_joints` payload. The replacement "
            "`note` must describe its exact gripper intent and valid milestone; "
            "do not copy `image_servo`, target-pixel, depth, or opposite "
            "gripper intent language from an earlier draft. Compare measured "
            "finger separation with the applicable threshold supplied in "
            "public context; do not round the value before that comparison."
        )
    elif contradiction == "motion_effect_mismatch":
        alignment_rule = (
            " Re-read every cited public value exactly and make the motor "
            "intent match it. Compare measured finger separation with the "
            "applicable threshold supplied in public context; do not round "
            "the value before that comparison."
        )
    elif contradiction in {"actuation_unverified", "grasp_unverified"}:
        if contradiction == "grasp_unverified":
            engagement_recovery = (
                articulated_contact_recovery_status(
                    "engage",
                    rejected_snapshot,
                    immediate_prior_receipt,
                    recent_receipts,
                )
                if isinstance(rejected_snapshot, Mapping)
                else None
            )
            insertion_ready = bool(
                isinstance(immediate_prior_receipt, Mapping)
                and articulated_handle_insertion_is_ready(
                    immediate_prior_receipt
                )
            )
            if insertion_ready:
                alignment_rule = (
                    " The open positive-depth insertion already positioned the "
                    "fingers around the visible handle. Remain at "
                    "`milestone=engage` and author `move_joints` with only "
                    '`targets={"gripper":0.0}` to close without simultaneous '
                    "arm motion. The next receipt decides contact."
                )
            elif (
                isinstance(immediate_prior_receipt, Mapping)
                and articulated_handle_insertion_receipt(
                    immediate_prior_receipt
                )
            ):
                alignment_rule = (
                    " The prior open-handle insertion ended outside the 8-pixel "
                    "close tolerance. Remain at `milestone=engage` and continue "
                    "a positive-depth `image_servo` toward the same visible "
                    "handle with gripper `open`. Do not close until a sealed "
                    "receipt places the EEF within tolerance. Qwen alone "
                    "chooses every servo value."
                )
            elif (
                isinstance(rejected_snapshot, Mapping)
                and rejected_snapshot.get("kind") == "image_servo"
                and rejected_snapshot.get("target_role") == "fixture_handle"
                and rejected_snapshot.get("gripper") == "close"
            ):
                alignment_rule = (
                    " Do not close while translating toward an articulated "
                    "handle. Remain at `milestone=engage` and author a positive-"
                    "depth `image_servo` at the visible handle with gripper "
                    "`open` to insert the fingers around it. Qwen alone chooses "
                    "the camera, pixel, depth, and step. Close without arm "
                    "motion only after its sealed receipt."
                )
            elif (
                engagement_recovery is not None
                and engagement_recovery["target_exhausted"]
            ):
                alignment_rule = (
                    " Both depth directions have already failed at this empty "
                    "handle pixel. You must change the target pixel or retreat "
                    "with a bounded nonzero `cartesian_delta` and gripper `open`. "
                    "Another depth reversal is prohibited. Qwen alone chooses "
                    "the new pixel or Cartesian values."
                )
            elif (
                engagement_recovery is not None
                and engagement_recovery["required"]
                and not engagement_recovery["strategy_changed"]
            ):
                alignment_rule = (
                    " The prior close at this handle point was empty. Repeating "
                    "the same target and depth direction is prohibited. Change "
                    "the target pixel, reverse the depth direction, or retreat "
                    "with a bounded nonzero `cartesian_delta` and gripper `open`. "
                    "Qwen alone chooses the new pixel or Cartesian values."
                )
            elif (
                isinstance(rejected_snapshot, Mapping)
                and rejected_snapshot.get("kind") == "cartesian_delta"
            ):
                alignment_rule = (
                    " The sealed close was an empty grasp. You must author a "
                    "bounded nonzero `cartesian_delta` retreat with gripper "
                    "`open` to reacquire the source in a new observation. Do "
                    "not author another `image_servo` on this observation. "
                    "Qwen chooses every Cartesian value."
                )
            else:
                alignment_rule = (
                    " Engagement is not sealed. If the source remains visible, "
                    "author an `image_servo` with nonzero `depth_delta_m` and "
                    "gripper `close` toward that visible source. If the source "
                    "is no longer visible after an empty grasp, author a "
                    "bounded nonzero `cartesian_delta` retreat with gripper "
                    "`open` to reacquire it in a new observation. Qwen chooses "
                    "every value."
                )
        else:
            contact_recovery = (
                articulated_contact_recovery_status(
                    "actuate",
                    rejected_snapshot,
                    immediate_prior_receipt,
                    recent_receipts,
                )
                if isinstance(rejected_snapshot, Mapping)
                else None
            )
            residual = (
                immediate_prior_receipt.get("gripper_residual")
                if isinstance(immediate_prior_receipt, Mapping)
                else None
            )
            separation = (
                residual.get("measured_end_finger_separation")
                if isinstance(residual, Mapping)
                else None
            )
            contact_lost = (
                not isinstance(separation, bool)
                and isinstance(separation, (int, float))
                and math.isfinite(float(separation))
                and float(separation) <= ARTICULATED_CONTACT_MIN_SEPARATION
            )
            contact_preserved = bool(
                isinstance(immediate_prior_receipt, Mapping)
                and articulated_contact_is_confirmed(immediate_prior_receipt)
            )
            prior_open = (
                isinstance(immediate_prior_receipt, Mapping)
                and immediate_prior_receipt.get("requested_gripper") == "open"
            )
            contact_confirmation_pending = bool(
                isinstance(immediate_prior_receipt, Mapping)
                and articulated_contact_is_provisional(immediate_prior_receipt)
            )
            rejected_axes = (
                cartesian_motion_axis_set(rejected_snapshot)
                if isinstance(rejected_snapshot, Mapping)
                else []
            )
            failed_axis_sets = [list(item) for item in failed_actuation_axis_sets]
            for item in articulated_failed_actuation_axis_sets(recent_receipts):
                if item not in failed_axis_sets:
                    failed_axis_sets.append(item)
            if contact_confirmation_pending:
                alignment_rule = (
                    " The first obstruction is provisional, not yet confirmed "
                    "contact. Remain at `milestone=actuate` and author "
                    '`move_joints` with only `targets={"gripper":0.0}`; do not '
                    "move any arm joint. The next sealed receipt decides "
                    "whether contact persisted."
                )
            elif (
                prior_open
                and isinstance(rejected_snapshot, Mapping)
                and rejected_snapshot.get("kind") == "cartesian_delta"
                and rejected_snapshot.get("gripper") in {"hold", "close"}
            ):
                alignment_rule = (
                    " The immediate prior receipt left the gripper open. Remain "
                    "at `milestone=actuate` and must re-engage before any "
                    "Cartesian actuation. Author a nonzero `image_servo` with "
                    "gripper `close` toward a visible handle point; do not author "
                    "another `cartesian_delta` until its receipt exposes contact. "
                    "Qwen alone chooses the camera, pixel, and every servo value."
                )
            elif (
                contact_recovery is not None
                and contact_recovery["target_exhausted"]
                and isinstance(rejected_snapshot, Mapping)
            ):
                forbidden_target = json.dumps(
                    {
                        "camera": rejected_snapshot.get("camera"),
                        "target_pixel": rejected_snapshot.get("target_pixel"),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                alignment_rule = (
                    " Both depth directions have already failed at this handle "
                    "pixel. Remain at `milestone=actuate`. Forbidden image_servo "
                    f"target={forbidden_target}. You must change the target pixel "
                    "or retreat. To change it, you must author a different camera "
                    "or target_pixel; to retreat, author a bounded nonzero "
                    "`cartesian_delta` with gripper `open`. Reversing depth at the "
                    "forbidden target is prohibited. Qwen alone chooses the new "
                    "pixel or Cartesian values."
                )
            elif (
                contact_preserved
                and rejected_axes
                and rejected_axes in failed_axis_sets
            ):
                failed_axis_json = json.dumps(
                    rejected_axes, separators=(",", ":")
                )
                alignment_rule = (
                    " A prior sealed contact-loss receipt already failed this "
                    "same motion-axis set. Failed motion axes="
                    f"{failed_axis_json}. Changing only its sign is not a new "
                    "actuation strategy; any vector with only `translation_z` "
                    "active will be rejected. Remain at `milestone=actuate`, "
                    "keep gripper `hold` or `close`, and activate "
                    "`translation_x`, `translation_y`, or a rotation axis in "
                    "the nonzero `cartesian_delta`. Infer the outward motion "
                    "from fresh RGB and the prior approach. This is a different "
                    "active Cartesian axis. Qwen alone chooses every value."
                )
            elif (
                contact_preserved
                and isinstance(rejected_snapshot, Mapping)
                and (
                    rejected_snapshot.get("kind") == "image_servo"
                    or rejected_snapshot.get("gripper") == "open"
                )
            ):
                alignment_rule = (
                    " Handle contact is already established by the immediate "
                    "prior sealed close receipt. Do not author another "
                    "`image_servo` at that contact point. Remain at "
                    "`milestone=actuate` and author a `cartesian_delta` with "
                    "nonzero translation and gripper `hold` or `close` to move "
                    "the articulated handle. Qwen alone chooses every Cartesian "
                    "value."
                )
            elif contact_lost:
                alignment_rule = (
                    " Contact was lost: the immediate prior measured finger "
                    "separation is at or below `0.002`. Remain at "
                    "`milestone=actuate`. Reacquire with "
                    "`target_role=fixture_handle`, positive `depth_delta_m`, "
                    "and gripper `close`; do not use "
                    "`target_role=articulation_motion` again until a new sealed "
                    "receipt proves contact. The note must say that positive "
                    "depth advances away from the selected camera to re-contact "
                    "the visible handle; it must not say negative depth or "
                    "retreat. Alternatively, retreat with a bounded nonzero "
                    "`cartesian_delta` and gripper `open` to gather a fresh "
                    "view. Qwen alone chooses every pixel and motor value."
                )
            elif (
                isinstance(rejected_snapshot, Mapping)
                and rejected_snapshot.get("kind") == "cartesian_delta"
                and rejected_snapshot.get("gripper") in {"hold", "close"}
            ):
                translation = rejected_snapshot.get("translation_m")
                rotation_only = (
                    isinstance(translation, Sequence)
                    and not isinstance(translation, (str, bytes))
                    and len(translation) == 3
                    and all(float(value) == 0.0 for value in translation)
                )
                if rotation_only:
                    alignment_rule = (
                        " Pure wrist rotation was rejected. Remain at "
                        "`milestone=actuate`; the replacement `cartesian_delta` "
                        "must include nonzero translation with gripper `hold` or "
                        "`close`. Changing only the rotation sign is prohibited. "
                        "Qwen alone chooses every Cartesian value."
                    )
                else:
                    alignment_rule = (
                        " The aligned closed gripper has already attempted an "
                        "actuation. Remain at `milestone=actuate` and must author a "
                        "nonzero `cartesian_delta`. Change the Cartesian actuation "
                        "direction from the rejected draft. Keep the "
                        "gripper `hold` or `close`. Do not return to `image_servo` "
                        "unless current public RGB shows that alignment was lost. "
                        "Qwen alone chooses every Cartesian value."
                    )
            else:
                alignment_rule = (
                    " Engagement is not sealed. You must author an `image_servo` "
                    "with nonzero `depth_delta_m` toward the visible task surface. "
                    "Use gripper `close` for a visibly graspable handle or knob; "
                    "Qwen chooses the sign and every value from the current views. "
                    "Do not author another free-space actuation before a receipt "
                    "exposes contact or grasp evidence."
                )
    elif contradiction == "release_unverified":
        alignment_rule = (
            " This release action must explicitly author gripper `open`. "
            "Do not repeat `hold` or `close`; Qwen remains the sole author of "
            "the gripper intent and every numeric value."
        )
    elif (
        contradiction == "safety_bound_risk"
        and isinstance(rejected_snapshot, Mapping)
        and rejected_snapshot.get("kind") == "image_servo"
    ):
        alignment_rule = (
            " This `image_servo` reached an arm safety or joint-workspace "
            "bound. Do not repeat `image_servo` on this observation. You must "
            "author a bounded `cartesian_delta` retreat opposite the recent "
            "approach direction. Do not use `base_action`: a mobile-base pulse "
            "can gather a new view but does not restore Panda joint margin. "
            "Reducing only `step_m` repeats the same unreachable direction and "
            "is not a revision."
        )
    negative_contact_depth_mismatch = bool(
        contradiction == "motion_effect_mismatch"
        and isinstance(rejected_snapshot, Mapping)
        and rejected_snapshot.get("kind") == "image_servo"
        and (
            rejected_snapshot.get("gripper") == "close"
            or (
                rejected_snapshot.get("target_role") == "fixture_handle"
                and rejected_snapshot.get("gripper") == "open"
            )
        )
        and isinstance(rejected_snapshot.get("depth_delta_m"), (int, float))
        and not isinstance(rejected_snapshot.get("depth_delta_m"), bool)
        and float(cast(float, rejected_snapshot["depth_delta_m"])) <= 0.0
    )
    if negative_contact_depth_mismatch:
        revision_authority = (
            " For this contact-depth mismatch, change the motor-bearing depth "
            "to a positive value. The note must describe positive depth as "
            "advancing away from the selected camera toward contact and must "
            "not claim negative depth or retreat. Qwen alone chooses the new "
            "positive magnitude and remains the sole motor author."
        )
    elif contradiction == "motion_effect_mismatch":
        revision_authority = (
            " For `motion_effect_mismatch`, preserve the command kind and every "
            "motor-bearing field exactly; only correct the note so it describes "
            "the rejected payload faithfully. Qwen alone remains the author of "
            "the preserved motor payload."
        )
    else:
        revision_authority = (
            " Qwen alone decides whether to change the note or numeric values, "
            "and Qwen alone must choose every numeric value."
        )
    return (
        f"{instruction}\n\nPROPOSAL_AUDIT_REVISION:\n"
        + json.dumps(advice, sort_keys=True, separators=(",", ":"))
        + "\nAuthor a new command for the same immutable observation. Do not "
        "repeat the rejected draft unchanged. Remove every stale claim "
        "contradicted by the audit; the new note must describe the new command "
        "actually authored."
        + (
            " Entries in `rejected_this_observation` summarize prior rejections. "
            "Only an unchanged complete draft with the same critic input and a "
            "prior valid critic rejection skips a new critic opinion. Changes "
            "to the note, point, depth, step, stereo, or other motor fields "
            "remain eligible for full validation and a fresh critic audit."
            if rejected_history and not (
                protocol_message and "empty-grasp retry" in protocol_message
            )
            else ""
        )
        + revision_authority
        + " Copy `required_observation_id` exactly into the command; do not "
        "append to or alter it."
        + alignment_rule
    )


def _image_servo_pixel_progress(
    draft: Mapping[str, object],
    immediate_prior_receipt: Mapping[str, object],
) -> dict[str, object] | None:
    if draft.get("kind") != "image_servo" or immediate_prior_receipt.get(
        "kind"
    ) != "image_servo":
        return None
    same_target = (
        draft.get("camera") == immediate_prior_receipt.get("requested_camera")
        and draft.get("target_pixel")
        == immediate_prior_receipt.get("requested_target_pixel")
        and draft.get("target_role")
        == immediate_prior_receipt.get("requested_target_role")
    )
    empty = {
        "schema": "robocasa-image-servo-pixel-progress/v1",
        "comparable": False,
        "same_camera_target_role": bool(same_target),
        "start_error_px": None,
        "end_error_px": None,
        "error_reduction_px": None,
        "moved_closer": None,
    }
    if not same_target:
        return empty
    camera = draft.get("camera")
    displacement = immediate_prior_receipt.get(
        "end_effector_external_pixel_displacement"
    )
    if not isinstance(camera, str) or not isinstance(displacement, Mapping):
        return empty
    camera_displacement = displacement.get(camera)
    target = draft.get("target_pixel")
    if not isinstance(camera_displacement, Mapping) or not isinstance(
        target, Sequence
    ):
        return empty
    start = camera_displacement.get("start_px")
    end = camera_displacement.get("end_px")
    if not all(
        isinstance(value, Sequence) and len(value) == 2
        for value in (target, start, end)
    ):
        return empty
    try:
        target_u, target_v = (float(value) for value in target)
        start_u, start_v = (float(value) for value in start)
        end_u, end_v = (float(value) for value in end)
    except (TypeError, ValueError):
        return empty
    values = (target_u, target_v, start_u, start_v, end_u, end_v)
    if not all(math.isfinite(value) for value in values):
        return empty
    start_error = math.hypot(start_u - target_u, start_v - target_v)
    end_error = math.hypot(end_u - target_u, end_v - target_v)
    reduction = start_error - end_error
    return {
        **empty,
        "comparable": True,
        "start_error_px": start_error,
        "end_error_px": end_error,
        "error_reduction_px": reduction,
        "moved_closer": reduction > 0.0,
    }


def _actuation_revision_status(
    context: CriticContext,
    draft: Mapping[str, object],
    claimed_milestone: str,
) -> dict[str, bool] | None:
    if claimed_milestone != "actuate" or draft.get("kind") != "cartesian_delta":
        return None
    if not context.proposal_records:
        return None
    predecessor = context.proposal_records[-1]
    previous_draft = predecessor.get("draft")
    previous_actuation_rejected = (
        predecessor.get("observation_id") == draft.get("observation_id")
        and predecessor.get("status") == "rejected_by_critic"
        and predecessor.get("claimed_milestone") == "actuate"
        and predecessor.get("contradiction") == "actuation_unverified"
        and isinstance(previous_draft, Mapping)
        and previous_draft.get("kind") == "cartesian_delta"
    )
    if not previous_actuation_rejected or not isinstance(previous_draft, Mapping):
        return None

    def direction_signature(command: Mapping[str, object]) -> tuple[int, ...] | None:
        values: list[float] = []
        for key in ("translation_m", "rotation_axis_angle_rad"):
            raw = command.get(key)
            if (
                not isinstance(raw, Sequence)
                or isinstance(raw, (str, bytes))
                or len(raw) != 3
            ):
                return None
            for item in raw:
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    return None
                value = float(item)
                if not math.isfinite(value):
                    return None
                values.append(value)
        return tuple(1 if value > 0.0 else -1 if value < 0.0 else 0 for value in values)

    previous_direction = direction_signature(previous_draft)
    current_direction = direction_signature(draft)
    nonzero = current_direction is not None and any(current_direction)
    return {
        "previous_actuation_rejected": True,
        "direction_changed": (
            previous_direction is not None
            and current_direction is not None
            and previous_direction != current_direction
        ),
        "nonzero": nonzero,
        "gripper_preserved": (
            previous_draft.get("gripper") in {"hold", "close"}
            and draft.get("gripper") in {"hold", "close"}
        ),
    }


def _wrist_roll_revision_status(
    context: CriticContext,
    draft: Mapping[str, object],
    claimed_milestone: str,
    public_state: Mapping[str, object],
) -> dict[str, bool] | None:
    if claimed_milestone != "approach" or draft.get("kind") != "move_joints":
        return None
    if not context.proposal_records:
        return None
    predecessor = context.proposal_records[-1]
    previous_draft = predecessor.get("draft")
    audit = predecessor.get("audit")
    previous_parallel_close_rejected = bool(
        predecessor.get("status") == "rejected_by_critic"
        and predecessor.get("contradiction") == "visual_alignment_unverified"
        and isinstance(audit, Mapping)
        and isinstance(audit.get("evidence"), Sequence)
        and "wrist_rgb" in audit["evidence"]
        and isinstance(previous_draft, Mapping)
        and previous_draft.get("kind") == "image_servo"
        and previous_draft.get("target_role") == "fixture_handle"
        and previous_draft.get("gripper") == "close"
    )
    if not previous_parallel_close_rejected or not isinstance(
        previous_draft, Mapping
    ):
        return None
    targets = draft.get("targets")
    qpos = public_state.get("state.arm_joint_position")
    joint7_target = targets.get("joint7") if isinstance(targets, Mapping) else None
    fresh_joint7 = (
        qpos[6]
        if isinstance(qpos, Sequence)
        and not isinstance(qpos, (str, bytes))
        and len(qpos) == 7
        else None
    )
    comparable = all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        for value in (joint7_target, fresh_joint7)
    )
    return {
        "applicable": True,
        "previous_parallel_close_rejected": True,
        "same_observation": (
            predecessor.get("observation_id") == draft.get("observation_id")
        ),
        "joint7_roll_only": (
            isinstance(targets, Mapping)
            and set(targets) == {"gripper", "joint7"}
        ),
        "gripper_open": (
            isinstance(targets, Mapping)
            and not isinstance(targets.get("gripper"), bool)
            and targets.get("gripper") == 1.0
        ),
        "fresh_joint7_target_changed": bool(
            comparable and float(cast(float, joint7_target)) != float(cast(float, fresh_joint7))
        ),
    }


def _public_joint7(public_state: Mapping[str, object]) -> float | None:
    qpos = public_state.get("state.arm_joint_position")
    if (
        not isinstance(qpos, Sequence)
        or isinstance(qpos, (str, bytes))
        or len(qpos) != 7
    ):
        return None
    value = qpos[6]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def _capture_wrist_roll_origin(
    context: CriticContext,
    draft: Mapping[str, object],
    public_state: Mapping[str, object],
    audit: Mapping[str, object],
) -> None:
    if context.articulated_wrist_roll_origin_joint7 is not None:
        return
    evidence = audit.get("evidence")
    if not (
        context.family == "articulated"
        and draft.get("kind") == "image_servo"
        and draft.get("target_role") == "fixture_handle"
        and draft.get("gripper") == "close"
        and audit.get("verdict") == "revise"
        and audit.get("contradiction") == "visual_alignment_unverified"
        and isinstance(evidence, Sequence)
        and not isinstance(evidence, (str, bytes))
        and "wrist_rgb" in evidence
    ):
        return
    context.articulated_wrist_roll_origin_joint7 = _public_joint7(public_state)


def _wrist_roll_progress_status(
    context: CriticContext,
    public_state: Mapping[str, object],
) -> dict[str, float | bool] | None:
    origin = context.articulated_wrist_roll_origin_joint7
    current = _public_joint7(public_state)
    if origin is None or current is None:
        return None
    delta = abs(current - origin)
    return {
        "origin_joint7_rad": origin,
        "current_joint7_rad": current,
        "absolute_delta_rad": delta,
        "quarter_turn_reached": delta >= math.pi / 2.0,
    }


def _draft_image_servo_alignment_status(
    draft: Mapping[str, object],
    public_state: Mapping[str, object],
) -> dict[str, object] | None:
    if draft.get("kind") != "image_servo":
        return None
    camera = draft.get("camera")
    target = draft.get("target_pixel")
    public_pixels = public_state.get("state.end_effector_external_pixels")
    selected = (
        public_pixels.get(camera)
        if isinstance(camera, str) and isinstance(public_pixels, Mapping)
        else None
    )
    if (
        camera not in {"left", "right"}
        or not isinstance(target, Sequence)
        or isinstance(target, (str, bytes))
        or len(target) != 2
        or not isinstance(selected, Mapping)
        or selected.get("visible") is not True
        or selected.get("depth_valid") is not True
    ):
        return None
    raw_values = [target[0], target[1], selected.get("u_px"), selected.get("v_px")]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in raw_values
    ):
        return None
    target_pixel = [float(target[0]), float(target[1])]
    current_pixel = [float(cast(float, raw_values[2])), float(cast(float, raw_values[3]))]
    error = math.dist(target_pixel, current_pixel)
    return {
        "schema": "robocasa-draft-image-servo-alignment/v1",
        "camera": camera,
        "target_pixel": target_pixel,
        "current_end_effector_pixel": current_pixel,
        "error_px": error,
        "within_one_grid_cell": error <= IMAGE_SERVO_GRID_CELL_PX,
        "within_close_tolerance": error <= IMAGE_SERVO_CLOSE_TOLERANCE_PX,
    }


def _actuation_contact_status(
    claimed_milestone: str,
    immediate_prior_receipt: Mapping[str, object] | None,
) -> dict[str, bool] | None:
    if claimed_milestone != "actuate" or not isinstance(
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
    preserved = (
        comparable and float(separation) > ARTICULATED_CONTACT_MIN_SEPARATION
    )
    return {
        "comparable": comparable,
        "preserved": preserved,
        "recovery_required": comparable and not preserved,
    }


def _cartesian_view_gathering_status(
    draft: Mapping[str, object],
    claimed_milestone: str,
    public_state: Mapping[str, object],
) -> dict[str, bool] | None:
    if claimed_milestone != "approach" or draft.get("kind") != "cartesian_delta":
        return None
    try:
        command = decode_cartesian_delta(
            draft,
            observation_id=str(draft.get("observation_id")),
        )
        protocol_validated = True
        nonzero = any(
            value != 0.0
            for value in (*command.translation_m, *command.rotation_axis_angle_rad)
        )
    except (TypeError, ValueError):
        protocol_validated = False
        nonzero = False
    qvel = public_state.get("state.arm_joint_velocity")
    tracking_settled = bool(
        isinstance(qvel, Sequence)
        and not isinstance(qvel, (str, bytes))
        and len(qvel) == 7
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for value in qvel
        )
        and max(abs(float(value)) for value in qvel) < STALL_ARM_JOINT_RAD
    )
    return {
        "applicable": True,
        "protocol_validated": protocol_validated,
        "nonzero": nonzero,
        "gripper_open": draft.get("gripper") == "open",
        "tracking_settled": tracking_settled,
    }




def _source_contact_wrist_view(
    images: Mapping[str, bytes],
    public_state: Mapping[str, object],
    camera_calibration: Mapping[str, object],
) -> tuple[dict[str, bytes], dict[str, object] | None]:
    """Show the public grip-site projection without selecting an object or action."""
    from io import BytesIO

    from PIL import Image, ImageDraw

    calibration = camera_calibration.get("wrist")
    if not isinstance(calibration, Mapping):
        return dict(images), None
    base = cast(Sequence[float], public_state["state.base_position"])
    offset = _matvec(
        _rotation_xyzw(public_state["state.base_rotation"]),
        cast(Sequence[float], public_state["state.end_effector_position_relative"]),
    )
    projected = project_world_point(
        tuple(base[i] + offset[i] for i in range(3)), calibration
    )
    if projected["visible"] is not True:
        return dict(images), None
    raw = Image.open(BytesIO(images["wrist"])).convert("RGB")
    width, height = raw.size
    view = Image.new("RGB", (3 * width, 2 * height), (24, 28, 32))
    view.paste(raw, (0, 0))
    view.paste(raw.resize((2 * width, 2 * height), Image.Resampling.NEAREST), (width, 0))
    u, v = float(projected["u_px"]), float(projected["v_px"])
    x, y = width + round(2 * u), round(2 * v)
    draw = ImageDraw.Draw(view)
    for line in [(x - 10, y, x - 4, y), (x + 4, y, x + 10, y),
                 (x, y - 10, x, y - 4), (x, y + 4, x, y + 10)]:
        draw.line(line, fill=(0, 220, 255), width=2)
    draw.multiline_text(
        (8, height + 12),
        "WRIST: raw view above\nRight: continuous 2x view\nCyan: current grip-site\nNot an object target",
        fill=(230, 235, 240), spacing=6,
    )
    output = BytesIO()
    view.save(output, format="PNG")
    return {**images, "wrist": output.getvalue()}, {
        "current_tool_pixel": [round(u, 4), round(v, 4)],
        "marker_meaning": "current measured grip-site; not an object target",
        "projection_source": "public base pose, relative EEF pose and wrist calibration",
        "raw_panel": [0, 0, width, height],
        "enlarged_panel": [width, 0, 3 * width, 2 * height],
        "scale": 2,
        "interpretation": "Compare the visible object part with the tool point in wrist and external views. Single-view pixel alignment does not measure grasp depth or contact. Qwen chooses the next existing command.",
    }


def _source_atlas_critic_prompt(prompt: str) -> str:
    """Match the effective system layout to the atlas without changing audit rules."""
    return prompt.replace(
        "In that named camera image, a derived magenta target marker\n"
        "is centered on the controller's selected pixel. The marker is not a task object;\n"
        "the left half is the full view and the right half is a magnified target crop.\n"
        "Judge the unmarked surface under its open center and surrounding arms. The\n",
        "Each external image is a 816x544 atlas. Its upper-left 256x256 panel\n"
        "is the unchanged full view. A continuous 512x512 full view on the right\n"
        "magnifies every ORIGINAL image pixel 2x without splitting the image.\n"
        "Numeric ORIGINAL u/v rulers occupy 32px top/left gutters beside the\n"
        "enlarged image, with a 16px right gutter for endpoint labels. The\n"
        "rulers use u increasing right and v increasing down. The gutters\n"
        "are outside image content. The lower-left panel contains the layout\n"
        "legend; wrist RGB is unchanged.\n"
        "Commands and public EEF pixels use ORIGINAL 256x256 image coordinates.\n"
        "Use the runtime atlas panel boxes and scales to map displayed pixels\n"
        "back to those coordinates. In the selected camera, a magenta marker\n"
        "is centered on the controller's selected pixel only in its magnified\n"
        "full view. The marker is not a task object. Judge the unmarked surface\n"
        "under its open center and compare with the unchanged full view. The\n",
        1,
    ).replace("the wider target crop", "the continuous magnified full view", 1)


def _source_observation_atlas(
    images: Mapping[str, bytes], *, draft: Mapping[str, object] | None = None,
) -> tuple[Mapping[str, bytes], dict[str, object] | None]:
    """Show all public external pixels at two scales during source search."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        views = {
            name: Image.open(BytesIO(images[name])).convert("RGB")
            for name in ("left", "right")
        }
    except (ImportError, KeyError, OSError, TypeError):
        return images, None
    if any(view.size != (256, 256) for view in views.values()):
        return images, None
    panels = [
        {"original_box_xyxy": [0, 0, 256, 256],
         "atlas_box_xyxy": [0, 0, 256, 256], "scale": 1},
        {"original_box_xyxy": [0, 0, 256, 256],
         "atlas_box_xyxy": [288, 32, 800, 544], "scale": 2},
    ]
    layout: dict[str, object] = {
        "cameras": ["left", "right"],
        "original_size_px": [256, 256],
        "atlas_size_px": [816, 544],
        "panels": panels,
        "instruction": (
            "Left/right images use this full-field atlas layout, not a "
            "target-centered crop. Upper-left is the unchanged full view; "
            "the continuous right panel magnifies the whole image 2x. Boxes "
            "are half-open pixel ranges [x0,y0,x1,y1]. Numeric ORIGINAL u/v "
            "rulers in 32px top/left gutters label every 16 pixels plus the "
            "last pixel (255): u increases right, v increases down. "
            "Right gutters add 16px for endpoint labels. Rulers are outside "
            "image content. Commands and public "
            "EEF pixels always use ORIGINAL 256x256 coordinates: original "
            "pixel = original box origin + (atlas pixel - atlas box origin) "
            "/ scale. Wrist is unchanged. Search the whole source view; "
            "magnification does not identify an object or verify contact. "
            "A critic magenta marker, when present, is on the enlarged "
            "full view only; inspect its open center in the unchanged full view."
        ),
    }
    output = dict(images)
    for name, view in views.items():
        atlas = Image.new("RGB", (816, 544), (24, 24, 24))
        atlas.paste(view, (0, 0))
        draw = ImageDraw.Draw(atlas)
        font = ImageFont.load_default(size=13)
        ruler_color = (240, 240, 240)
        for panel in panels[1:]:
            box = panel["original_box_xyxy"]
            display = panel["atlas_box_xyxy"]
            atlas.paste(
                view.crop(tuple(box)).resize((512, 512), Image.Resampling.NEAREST),
                (display[0], display[1]),
            )
            # All ruler ink is outside the exact 2x image panel.
            draw.text((display[0] - 29, display[1] - 29), "u", font=font, fill=ruler_color)
            draw.text((display[0] - 29, display[1] - 16), "v", font=font, fill=ruler_color)
            for offset in [*range(0, 256, 16), 255]:
                x = display[0] + 2 * offset
                y = display[1] + 2 * offset
                draw.line((x, display[1] - 5, x, display[1] - 1), fill=ruler_color)
                draw.line((display[0] - 5, y, display[0] - 1, y), fill=ruler_color)
                draw.text(
                    (x, display[1] - 26),
                    str(box[0] + offset), font=font, fill=ruler_color, anchor="mt",
                )
                draw.text(
                    (display[0] - 8, max(display[1] + 8, min(y, display[3] - 8))),
                    str(box[1] + offset), font=font, fill=ruler_color, anchor="rm",
                )
        draw.multiline_text(
            (10, 306),
            f"{name.upper()} - ORIGINAL pixels\n"
            "Full view: upper left\n"
            "Continuous 2x view: right\n"
            "u: horizontal (right)\n"
            "v: vertical (down)\n"
            "Rulers: ORIGINAL u/v\n"
            "Numbers every 16 pixels\n"
            "Use ORIGINAL u,v in commands",
            font=font, fill=ruler_color, spacing=8,
        )
        if (
            draft is not None and draft.get("kind") == "image_servo"
            and draft.get("camera") == name
        ):
            target = draft.get("target_pixel")
            if (
                isinstance(target, Sequence) and not isinstance(target, (str, bytes))
                and len(target) == 2
                and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                        and math.isfinite(float(v)) and 0 <= float(v) < 256
                        for v in target)
            ):
                u, v = (float(value) for value in target)
                panel = panels[1]
                box = panel["original_box_xyxy"]
                display = panel["atlas_box_xyxy"]
                center = [round(display[0] + 2 * (u - box[0])),
                          round(display[1] + 2 * (v - box[1]))]
                layout["marked_target"] = {
                    "camera": name, "original_pixel": list(target), "atlas_pixel": center,
                }
                for offset in range(-20, 21):
                    if abs(offset) < 8:
                        continue
                    for thickness in range(-2, 3):
                        for x, y in ((center[0] + offset, center[1] + thickness),
                                     (center[0] + thickness, center[1] + offset)):
                            if display[0] <= x < display[2] and display[1] <= y < display[3]:
                                atlas.putpixel((x, y), (255, 0, 255))
        encoded = BytesIO()
        atlas.save(encoded, format="PNG")
        output[name] = encoded.getvalue()
    return output, layout


def _proposal_critic_images(
    images: Mapping[str, bytes], draft: Mapping[str, object]
) -> Mapping[str, bytes]:
    """Mark the controller-authored image-servo pixel for visual audit."""
    if draft.get("kind") != "image_servo":
        return images
    camera = draft.get("camera")
    target = draft.get("target_pixel")
    if (
        camera not in {"left", "right"}
        or not isinstance(target, Sequence)
        or isinstance(target, (str, bytes))
        or len(target) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in target
        )
        or not isinstance(images.get(str(camera)), bytes)
    ):
        return images
    try:
        from PIL import Image

        marked = Image.open(BytesIO(images[str(camera)])).convert("RGB").copy()
        width, height = marked.size
        center_x, center_y = (round(float(value)) for value in target)
        if not (0 <= center_x < width and 0 <= center_y < height):
            return images
        magenta = (255, 0, 255)
        for offset in range(-20, 21):
            if abs(offset) < 8:
                continue
            for thickness in range(-2, 3):
                horizontal = (center_x + offset, center_y + thickness)
                vertical = (center_x + thickness, center_y + offset)
                if 0 <= horizontal[0] < width and 0 <= horizontal[1] < height:
                    marked.putpixel(horizontal, magenta)
                if 0 <= vertical[0] < width and 0 <= vertical[1] < height:
                    marked.putpixel(vertical, magenta)
        crop_width = min(96, width)
        crop_height = min(96, height)
        crop_left = max(0, min(center_x - crop_width // 2, width - crop_width))
        crop_top = max(0, min(center_y - crop_height // 2, height - crop_height))
        zoom = marked.crop((
            crop_left,
            crop_top,
            crop_left + crop_width,
            crop_top + crop_height,
        )).resize((width, height))
        composite = Image.new("RGB", (width * 2, height), (0, 0, 0))
        composite.paste(marked, (0, 0))
        composite.paste(zoom, (width, 0))
        output = BytesIO()
        composite.save(output, format="PNG")
    except (ImportError, OSError, TypeError, ValueError, AttributeError):
        return images
    result = dict(images)
    result[str(camera)] = output.getvalue()
    return result


def _horizontal_dark_pull_runs(
    image: object,
    center_pixel: Sequence[object],
    *,
    half_width: int = 80,
    half_height: int = 48,
) -> list[tuple[int, int, int]] | None:
    """Return qualifying dark horizontal runs as (u_start, u_stop, v) near a pixel."""

    size = getattr(image, "size", None)
    getpixel = getattr(image, "getpixel", None)
    if (
        not isinstance(size, Sequence)
        or isinstance(size, (str, bytes))
        or len(size) != 2
        or not callable(getpixel)
        or len(center_pixel) != 2
    ):
        return None
    try:
        width, height = (int(value) for value in size)
        center_x, center_y = (round(float(value)) for value in center_pixel)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (0 <= center_x < width and 0 <= center_y < height):
        return None

    def is_dark_scene_pixel(x: int, y: int) -> bool:
        value = getpixel((x, y))
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) < 3
        ):
            return False
        red, green, blue = (int(value[index]) for index in range(3))
        grid = (red > 180 and green > 180 and blue < 100) or (
            red < 100 and green > 180 and blue > 180
        )
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        return not grid and luminance < TOASTER_HANDLE_DARK_LUMA_MAX

    x_start = max(0, center_x - half_width)
    x_stop = min(width, center_x + half_width + 1)
    y_start = max(0, center_y - half_height)
    y_stop = min(height, center_y + half_height + 1)
    def is_grid_pixel(x: int, y: int) -> bool:
        value = getpixel((x, y))
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) < 3
        ):
            return False
        red, green, blue = (int(value[index]) for index in range(3))
        return (red > 180 and green > 180 and blue < 100) or (
            red < 100 and green > 180 and blue > 180
        )

    runs: list[tuple[int, int, int]] = []
    for y in range(y_start, y_stop):
        run_start: int | None = None
        for x in range(x_start, x_stop + 1):
            if x < x_stop and run_start is not None and is_grid_pixel(x, y):
                continue  # the drawn grid neither starts nor breaks a dark run
            dark = x < x_stop and is_dark_scene_pixel(x, y)
            if dark and run_start is None:
                run_start = x
            if not dark and run_start is not None:
                if x - run_start >= TOASTER_HANDLE_MIN_HORIZONTAL_RUN_PX:
                    runs.append((run_start, x, y))
                run_start = None
    return runs


def _horizontal_dark_pull_target_status(
    image: object,
    target_pixel: Sequence[object],
) -> dict[str, bool | int] | None:
    """Locate a visible narrow dark horizontal pull around a proposed pixel."""

    runs = _horizontal_dark_pull_runs(image, target_pixel)
    if runs is None:
        return None
    center_x, center_y = (round(float(value)) for value in target_pixel)
    bars = _thin_dark_bars(runs)
    target_on_pull = any(
        bar["u_min"] - TOASTER_HANDLE_TARGET_TOLERANCE_PX
        <= center_x
        <= bar["u_max"] + TOASTER_HANDLE_TARGET_TOLERANCE_PX
        and bar["v_min"] - TOASTER_HANDLE_TARGET_TOLERANCE_PX
        <= center_y
        <= bar["v_max"] + TOASTER_HANDLE_TARGET_TOLERANCE_PX
        for bar in bars
    )
    return {
        "horizontal_pull_detected": bool(bars),
        "target_on_horizontal_pull": target_on_pull,
    }


def _thin_dark_bars(
    runs: Sequence[tuple[int, int, int]] | None,
) -> list[dict[str, int]]:
    """Group dark runs into vertical stacks and keep only thin horizontal bars.

    A pull is a short stack of overlapping dark runs; an appliance body is a
    tall stack. Stacks taller than ``TOASTER_HANDLE_MAX_BAR_HEIGHT_PX`` are
    dropped so a rejection span or cross-view box never names the body.
    """

    if not runs:
        return []
    stacks: list[dict[str, int]] = []
    for u_start, u_stop, v in sorted(runs, key=lambda run: (run[2], run[0])):
        u_max = u_stop - 1
        joined = False
        for stack in stacks:
            if (
                stack["v_max"] in (v, v - 1, v - 2)
                and u_start <= stack["u_max"]
                and u_max >= stack["u_min"]
            ):
                stack["u_min"] = min(stack["u_min"], u_start)
                stack["u_max"] = max(stack["u_max"], u_max)
                stack["v_max"] = v
                joined = True
                break
        if not joined:
            stacks.append({"u_min": u_start, "u_max": u_max, "v_min": v, "v_max": v})
    return [
        stack
        for stack in stacks
        if stack["v_max"] - stack["v_min"] + 1 <= TOASTER_HANDLE_MAX_BAR_HEIGHT_PX
    ]


def _pull_box(
    runs: Sequence[tuple[int, int, int]] | None,
    *,
    near_pixel: Sequence[float] | None = None,
) -> dict[str, int] | None:
    """Return the thin bar nearest ``near_pixel`` (or the widest bar)."""

    bars = _thin_dark_bars(runs)
    if not bars:
        return None
    if near_pixel is not None and len(near_pixel) == 2:
        u, v = float(near_pixel[0]), float(near_pixel[1])

        def distance(bar: dict[str, int]) -> float:
            du = max(bar["u_min"] - u, 0.0, u - bar["u_max"])
            dv = max(bar["v_min"] - v, 0.0, v - bar["v_max"])
            return du + dv

        return min(bars, key=distance)
    return max(bars, key=lambda bar: bar["u_max"] - bar["u_min"])


def _cross_view_status_from_image(
    image: object,
    *,
    other_camera: str,
    pixel_record: object,
    tolerance_px: int = TOASTER_CROSS_VIEW_TOLERANCE_PX,
) -> dict[str, object] | None:
    """Compare the public grip-site pixel with the dark pull in the other view."""

    size = getattr(image, "size", None)
    if (
        not isinstance(size, Sequence)
        or isinstance(size, (str, bytes))
        or len(size) != 2
    ):
        return None
    width, height = (int(value) for value in size)
    u = v = None
    visible = False
    if isinstance(pixel_record, Mapping):
        raw_u, raw_v = pixel_record.get("u_px"), pixel_record.get("v_px")
        if all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for value in (raw_u, raw_v)
        ):
            u, v = float(cast(float, raw_u)), float(cast(float, raw_v))
        visible = bool(
            pixel_record.get("visible") is True
            and pixel_record.get("depth_valid") is True
            and u is not None
            and 0.0 <= u < width
            and v is not None
            and 0.0 <= v < height
        )
    runs = _horizontal_dark_pull_runs(
        image,
        [width // 2, height // 2],
        half_width=width,
        half_height=height,
    )
    box = _pull_box(
        runs, near_pixel=[u, v] if u is not None and v is not None else None
    )
    offset: list[float] | None = None
    on_pull = False
    if box is not None and u is not None and v is not None:
        du = max(box["u_min"] - u, 0.0, u - box["u_max"])
        dv = max(box["v_min"] - v, 0.0, v - box["v_max"])
        offset = [round(du, 1), round(dv, 1)]
        on_pull = bool(visible and du <= tolerance_px and dv <= tolerance_px)
    return {
        "other_camera": other_camera,
        "grip_site_pixel": (
            [round(u, 1), round(v, 1)] if u is not None and v is not None else None
        ),
        "grip_site_visible": visible,
        "horizontal_pull_detected": box is not None,
        "pull_box": box,
        "grip_site_offset_px": offset,
        "tolerance_px": tolerance_px,
        "grip_site_on_horizontal_pull": on_pull,
    }


def _toaster_cross_view_status(
    task: str,
    camera: object,
    public_state: Mapping[str, object],
    images: Mapping[str, bytes],
) -> dict[str, object] | None:
    if task != "OpenToasterOvenDoor" or camera not in {"left", "right"}:
        return None
    other = "left" if camera == "right" else "right"
    pixels = public_state.get("state.end_effector_external_pixels")
    record = pixels.get(other) if isinstance(pixels, Mapping) else None
    if not isinstance(images.get(other), bytes):
        return None
    try:
        from PIL import Image

        image = Image.open(BytesIO(images[other])).convert("RGB")
    except (ImportError, OSError, TypeError, ValueError, AttributeError):
        return None
    return _cross_view_status_from_image(
        image, other_camera=other, pixel_record=record
    )


def _toaster_pull_candidates(
    task: str,
    images: Mapping[str, bytes],
    *,
    limit: int = 6,
) -> dict[str, list[dict[str, int]]] | None:
    """Deterministic thin-bar detections in both external views (public RGB)."""

    if task != "OpenToasterOvenDoor":
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    result: dict[str, list[dict[str, int]]] = {}
    for name in ("left", "right"):
        data = images.get(name)
        if not isinstance(data, bytes):
            return None
        try:
            image = Image.open(BytesIO(data)).convert("RGB")
        except (OSError, TypeError, ValueError, AttributeError):
            return None
        result[name] = _thin_bars_in_image(image, limit=limit)
    return result


def _thin_bars_in_image(image: object, *, limit: int = 6) -> list[dict[str, int]]:
    size = getattr(image, "size", None)
    if (
        not isinstance(size, Sequence)
        or isinstance(size, (str, bytes))
        or len(size) != 2
    ):
        return []
    width, height = (int(value) for value in size)
    runs = _horizontal_dark_pull_runs(
        image, [width // 2, height // 2], half_width=width, half_height=height
    )
    bars = _thin_dark_bars(runs)
    bars.sort(key=lambda bar: bar["u_max"] - bar["u_min"], reverse=True)
    return bars[:limit]


def _coffee_button_boxes(image: object) -> list[dict[str, int]]:
    """Detect paired small dark controls in one fresh public RGB view."""

    size = getattr(image, "size", None)
    getpixel = getattr(image, "getpixel", None)
    if (
        not isinstance(size, Sequence)
        or isinstance(size, (str, bytes))
        or len(size) != 2
        or not callable(getpixel)
    ):
        return []
    width, height = (int(value) for value in size)
    dark: set[tuple[int, int]] = set()
    for y in range(max(0, 48), min(height, 192)):
        for x in range(max(0, 16), max(0, width - 16)):
            value = getpixel((x, y))
            if (
                not isinstance(value, Sequence)
                or isinstance(value, (str, bytes))
                or len(value) < 3
            ):
                continue
            red, green, blue = (int(value[index]) for index in range(3))
            grid = (red > 180 and green > 180 and blue < 100) or (
                red < 100 and green > 180 and blue > 180
            )
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            if not grid and luminance < COFFEE_BUTTON_DARK_LUMA_MAX:
                dark.add((x, y))

    components: list[tuple[dict[str, int], int]] = []
    while dark:
        pending = [dark.pop()]
        xs: list[int] = []
        ys: list[int] = []
        while pending:
            x, y = pending.pop()
            xs.append(x)
            ys.append(y)
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in dark:
                    dark.remove(neighbor)
                    pending.append(neighbor)
        box = {
            "u_min": min(xs),
            "u_max": max(xs),
            "v_min": min(ys),
            "v_max": max(ys),
        }
        box_width = box["u_max"] - box["u_min"] + 1
        box_height = box["v_max"] - box["v_min"] + 1
        if 4 <= box_width <= 12 and 1 <= box_height <= 4 and len(xs) >= 4:
            components.append((box, len(xs)))

    paired: set[int] = set()
    for first_index, (first, _) in enumerate(components):
        first_u = (first["u_min"] + first["u_max"]) / 2.0
        first_v = (first["v_min"] + first["v_max"]) / 2.0
        for second_index in range(first_index + 1, len(components)):
            second = components[second_index][0]
            second_u = (second["u_min"] + second["u_max"]) / 2.0
            second_v = (second["v_min"] + second["v_max"]) / 2.0
            delta_u = abs(first_u - second_u)
            delta_v = abs(first_v - second_v)
            same_row = 15.0 <= delta_u <= 50.0 and delta_v <= 6.0
            same_column = delta_u <= 6.0 and 6.0 <= delta_v <= 18.0
            if same_row or same_column:
                paired.update((first_index, second_index))
    return sorted(
        [components[index][0] for index in paired],
        key=lambda box: (box["v_min"], box["u_min"]),
    )


def _coffee_button_candidates(
    task: str, images: Mapping[str, bytes]
) -> dict[str, list[dict[str, int]]] | None:
    if task != "StartCoffeeMachine":
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    result: dict[str, list[dict[str, int]]] = {}
    for name in ("left", "right"):
        data = images.get(name)
        if not isinstance(data, bytes):
            return None
        try:
            image = Image.open(BytesIO(data)).convert("RGB")
        except (OSError, TypeError, ValueError, AttributeError):
            return None
        result[name] = _coffee_button_boxes(image)
    return result if any(result.values()) else None


def _pixel_on_box(
    pixel: object, box: Mapping[str, int], *, tolerance_px: int
) -> bool:
    if (
        not isinstance(pixel, Sequence)
        or isinstance(pixel, (str, bytes))
        or len(pixel) != 2
    ):
        return False
    try:
        u, v = float(pixel[0]), float(pixel[1])
    except (TypeError, ValueError):
        return False
    return bool(
        math.isfinite(u)
        and math.isfinite(v)
        and box["u_min"] - tolerance_px <= u <= box["u_max"] + tolerance_px
        and box["v_min"] - tolerance_px <= v <= box["v_max"] + tolerance_px
    )


def _coffee_button_axis_labels(
    boxes: Sequence[Mapping[str, int]],
    *,
    minimum: str,
    maximum: str,
    labels: tuple[str, str],
) -> dict[int, str]:
    centers = [
        (float(box[minimum]) + float(box[maximum])) / 2.0 for box in boxes
    ]
    groups: list[list[int]] = []
    for index in sorted(range(len(centers)), key=centers.__getitem__):
        if not groups:
            groups.append([index])
            continue
        prior_center = sum(centers[item] for item in groups[-1]) / len(groups[-1])
        if centers[index] - prior_center <= 6.0:
            groups[-1].append(index)
        else:
            groups.append([index])
    if len(groups) != 2:
        return {}
    return {
        index: labels[group_index]
        for group_index, group in enumerate(groups)
        for index in group
    }


def _coffee_button_grid_labels(
    boxes: Sequence[Mapping[str, int]],
) -> dict[int, tuple[str | None, str | None]]:
    rows = _coffee_button_axis_labels(
        boxes,
        minimum="v_min",
        maximum="v_max",
        labels=("top", "bottom"),
    )
    columns = _coffee_button_axis_labels(
        boxes,
        minimum="u_min",
        maximum="u_max",
        labels=("left", "right"),
    )
    return {
        index: (rows.get(index), columns.get(index)) for index in range(len(boxes))
    }


def _coffee_button_index_for_pixel(
    pixel: object, boxes: Sequence[Mapping[str, int]]
) -> int | None:
    matches = [
        index
        for index, box in enumerate(boxes)
        if _pixel_on_box(
            pixel,
            box,
            tolerance_px=COFFEE_BUTTON_TARGET_TOLERANCE_PX,
        )
    ]
    if not matches:
        return None
    assert isinstance(pixel, Sequence) and not isinstance(pixel, (str, bytes))
    u, v = float(pixel[0]), float(pixel[1])
    return min(
        matches,
        key=lambda index: (
            u
            - (
                float(boxes[index]["u_min"]) + float(boxes[index]["u_max"])
            )
            / 2.0
        )
        ** 2
        + (
            v
            - (
                float(boxes[index]["v_min"]) + float(boxes[index]["v_max"])
            )
            / 2.0
        )
        ** 2,
    )


def _coffee_calibrated_contact_pixel(
    view: str, box: Mapping[str, int]
) -> tuple[float, float]:
    vertical_offset = 1 if view == "left" else 2
    return float(box["u_max"]), float(box["v_min"] - vertical_offset)


def _coffee_button_target_violation(
    task: str,
    draft: Mapping[str, object],
    candidates: Mapping[str, Sequence[Mapping[str, int]]] | None,
    *,
    standing_target: Mapping[str, object] | None = None,
) -> str | None:
    depth = draft.get("depth_delta_m")
    if (
        task != "StartCoffeeMachine"
        or candidates is None
        or draft.get("kind") != "image_servo"
        or draft.get("target_role") != "control_target"
        or draft.get("camera") not in {"left", "right"}
        or (
            not isinstance(depth, bool)
            and isinstance(depth, (int, float))
            and math.isfinite(float(depth))
            and float(depth) < 0.0
        )
    ):
        return None
    if _draft_matches_standing_target(draft, standing_target):
        return None
    camera = str(draft["camera"])
    checks = [(camera, "target_pixel")]
    if draft.get("other_view_pixel") is not None:
        checks.append(("left" if camera == "right" else "right", "other_view_pixel"))
    selected: dict[str, tuple[str, str | None, str | None]] = {}
    for view, key in checks:
        boxes = candidates.get(view, ())
        if not boxes:
            continue
        match = _coffee_button_index_for_pixel(draft.get(key), boxes)
        if match is not None:
            pixel = draft[key]
            assert isinstance(pixel, Sequence) and not isinstance(pixel, (str, bytes))
            u, v = float(pixel[0]), float(pixel[1])
            box = boxes[match]
            expected_u, expected_v = _coffee_calibrated_contact_pixel(
                view, box
            )
            if u != expected_u or v != expected_v:
                label = "target" if key == "target_pixel" else "other-view target"
                return (
                    f"coffee machine {label} must use the calibrated contact "
                    "point for one distinct visible button; require "
                    f"u = {expected_u:.1f} and v = {expected_v:.1f} in the "
                    f"{view} view"
                )
            row, column = _coffee_button_grid_labels(boxes)[match]
            selected[view] = (key, row, column)
            continue
        spans = "; ".join(
            f"u={box['u_min']}..{box['u_max']}, v={box['v_min']}..{box['v_max']}"
            for box in boxes
        )
        label = "target" if key == "target_pixel" else "other-view target"
        return (
            f"coffee machine {label} center is not on one distinct visible "
            f"button in the {view} view; COFFEE_BUTTON_CANDIDATES: {spans}"
        )
    if len(checks) == 2 and len(selected) == 2:
        first_view, _ = checks[0]
        second_view, _ = checks[1]
        first_key, first_row, first_column = selected[first_view]
        second_key, second_row, second_column = selected[second_view]
        row_conflict = (
            first_row is not None
            and second_row is not None
            and first_row != second_row
        )
        column_conflict = (
            first_column is not None
            and second_column is not None
            and first_column != second_column
        )
        if row_conflict or column_conflict:
            first_label = (
                "target" if first_key == "target_pixel" else "other-view target"
            )
            second_label = (
                "target" if second_key == "target_pixel" else "other-view target"
            )
            return (
                "coffee machine stereo targets select different distinct visible "
                f"buttons: {first_view} {first_label} is row={first_row}, "
                f"column={first_column}; {second_view} {second_label} is "
                f"row={second_row}, column={second_column}; match the same grid "
                "row and column across views"
            )
    return None


def _controller_coffee_button_candidates_instruction(
    instruction: str,
    candidates: Mapping[str, Sequence[Mapping[str, int]]],
) -> str:
    marker = "\n\nCOFFEE_BUTTON_CANDIDATES:\n"
    if marker in instruction:
        raise ProposalAuditFailedClosed(
            "controller instruction already contains coffee button candidates"
        )
    packet: dict[str, list[dict[str, object]]] = {}
    for name, boxes in candidates.items():
        labels = _coffee_button_grid_labels(boxes)
        packet[name] = [
            {
                **dict(box),
                **(
                    {"grid_row": labels[index][0]}
                    if labels[index][0] is not None
                    else {}
                ),
                **(
                    {"grid_column": labels[index][1]}
                    if labels[index][1] is not None
                    else {}
                ),
            }
            for index, box in enumerate(boxes)
        ]
    return (
        instruction
        + marker
        + json.dumps(packet, sort_keys=True, separators=(",", ":"))
        + "\nThese boxes are deterministic detections of paired small dark "
        "controls in the fresh public external RGB. Qwen chooses one box, "
        "the camera, and its calibrated upper-right contact pixel: "
        "u = u_max; left v = v_min - 1; right v = v_min - 2. For stereo, "
        "choose the matching button with the same grid_row and "
        "grid_column in the other view. Similar pixel values across cameras "
        "do not establish correspondence. Never use the midpoint between boxes."
    )


def _bar_for_pixel(
    bars: Sequence[Mapping[str, int]], pixel: Sequence[object]
) -> Mapping[str, int] | None:
    try:
        u, v = float(pixel[0]), float(pixel[1])
    except (TypeError, ValueError, IndexError):
        return None
    best: tuple[float, Mapping[str, int]] | None = None
    for bar in bars:
        du = max(bar["u_min"] - u, 0.0, u - bar["u_max"])
        dv = max(bar["v_min"] - v, 0.0, v - bar["v_max"])
        distance = du + dv
        if best is None or distance < best[0]:
            best = (distance, bar)
    return best[1] if best is not None and best[0] <= 12.0 else None


def _stereo_bar_diagnosis(
    draft: Mapping[str, object],
    candidates: Mapping[str, Sequence[Mapping[str, int]]] | None,
) -> str:
    """Name the detected bar under each stereo pixel so a mismatch is explicit."""

    if candidates is None or draft.get("kind") != "image_servo":
        return ""
    camera = draft.get("camera")
    other = "left" if camera == "right" else "right"
    parts: list[str] = []
    bars_by_view: list[Mapping[str, int] | None] = []
    for view, key in ((camera, "target_pixel"), (other, "other_view_pixel")):
        pixel = draft.get(key)
        if (
            not isinstance(view, str)
            or not isinstance(pixel, Sequence)
            or isinstance(pixel, (str, bytes))
            or len(pixel) != 2
        ):
            return ""
        bar = _bar_for_pixel(candidates.get(view, ()), pixel)
        bars_by_view.append(bar)
        if bar is None:
            parts.append(f"the {view} pixel {list(pixel)} is on no detected bar")
        else:
            parts.append(
                f"the {view} pixel {list(pixel)} is on bar u={bar['u_min']}.."
                f"{bar['u_max']}, v={bar['v_min']}..{bar['v_max']}"
            )
    verdict = ""
    if all(bar is not None for bar in bars_by_view):
        first, second = bars_by_view
        assert first is not None and second is not None
        same_height = abs(
            (first["v_min"] + first["v_max"]) - (second["v_min"] + second["v_max"])
        ) <= 24
        verdict = (
            "; both views name a bar at the same height"
            if same_height
            else "; these are DIFFERENT bars (different heights), not one point"
        )
    return "; " + "; ".join(parts) + verdict


def _controller_pull_candidates_instruction(
    instruction: str,
    candidates: Mapping[str, Sequence[Mapping[str, int]]],
) -> str:
    marker = "\n\nTOASTER_PULL_CANDIDATES:\n"
    if marker in instruction:
        raise ProposalAuditFailedClosed(
            "controller instruction already contains pull candidates"
        )
    packet = {
        name: [dict(bar) for bar in bars] for name, bars in candidates.items()
    }
    return (
        instruction
        + marker
        + json.dumps(packet, sort_keys=True, separators=(",", ":"))
        + "\nThese are deterministic detections of thin dark horizontal bars "
        "in the fresh public left and right RGB (pixel boxes u_min..u_max, "
        "v_min..v_max), not a chosen target and not necessarily the pull. Judge "
        "from the images which box is the door pull, then author your own "
        "`target_pixel` on that bar's interior midline and `other_view_pixel` "
        "on the same bar in the other view; both pixels must name the same "
        "physical point. Never use the projected grip-site pixel as a target."
    )


STANDING_TARGET_MATCH_PX = 12.0
STEREO_APPROACH_ROLES = frozenset({"fixture_handle", "source_object", "control_target"})


PICK_PLACE_INTO_FIXTURE_TASKS = frozenset({
    "PickPlaceCounterToDrawer",
    "PickPlaceCounterToCabinet",
    "ToastOnCorrectRack",
})


def effective_family(task: str, labelled_family: str) -> str:
    """Map pick-place-into-fixture tasks onto the grasp_place milestone graph.

    The baseline labels them articulated, but the episode is a grasp of a
    movable source object followed by a place; the articulated engage rule
    (open fixture-handle insertion) rejected every source-object servo.
    """

    if task in PICK_PLACE_INTO_FIXTURE_TASKS:
        return "grasp_place"
    return labelled_family


def _standing_stereo_target(
    recent_receipts: Sequence[Mapping[str, object]],
    public_state: Mapping[str, object],
    camera_calibration: Mapping[str, object] | None,
    *,
    match_tolerance_px: float = STANDING_TARGET_MATCH_PX,
) -> dict[str, object] | None:
    """Project the last consistent triangulated handle point into both views.

    The point comes from the newest stereo ``image_servo`` receipt (public
    ``stereo_target_base_m``); its projections use only public calibration
    and base pose, exactly like the published grip-site pixels.
    """

    if not isinstance(camera_calibration, Mapping):
        return None
    base_m: list[float] | None = None
    source_index: int | None = None
    for index in range(len(recent_receipts) - 1, -1, -1):
        receipt = recent_receipts[index]
        raw = receipt.get("stereo_target_base_m")
        if (
            receipt.get("kind") == "image_servo"
            and isinstance(raw, Sequence)
            and not isinstance(raw, (str, bytes))
            and len(raw) == 3
            and all(
                not isinstance(v, bool)
                and isinstance(v, (int, float))
                and math.isfinite(float(v))
                for v in raw
            )
        ):
            base_m = [float(v) for v in raw]
            source_index = index
            break
    if base_m is None:
        return None
    try:
        base_position = [float(v) for v in cast(Sequence[object], public_state["state.base_position"])]
        base_rotation = _rotation_xyzw(public_state.get("state.base_rotation"))
        eef = [float(v) for v in cast(Sequence[object], public_state["state.end_effector_position_relative"])]
    except (KeyError, TypeError, ValueError):
        return None
    offset = _matvec(base_rotation, (base_m[0], base_m[1], base_m[2]))
    world = tuple(base_position[i] + offset[i] for i in range(3))
    pixels: dict[str, object] = {}
    for name in ("left", "right"):
        calibration = camera_calibration.get(name)
        if not isinstance(calibration, Mapping):
            pixels[name] = None
            continue
        try:
            projected = project_world_point(world, calibration)
        except (TypeError, ValueError):
            pixels[name] = None
            continue
        pixels[name] = (
            [round(float(projected["u_px"]), 1), round(float(projected["v_px"]), 1)]
            if projected.get("visible") is True
            else None
        )
    distance = math.sqrt(sum((base_m[i] - eef[i]) ** 2 for i in range(3)))
    return {
        "base_m": [round(v, 4) for v in base_m],
        "left_px": pixels["left"],
        "right_px": pixels["right"],
        "distance_to_grip_site_m": round(distance, 4),
        "source_receipt_index": source_index,
        "match_tolerance_px": float(match_tolerance_px),
    }


def _task_standing_stereo_target(
    task: str,
    family: str,
    recent_receipts: Sequence[Mapping[str, object]],
    public_state: Mapping[str, object],
    camera_calibration: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if family != "articulated" and task != "StartCoffeeMachine":
        return None
    return _standing_stereo_target(
        recent_receipts,
        public_state,
        camera_calibration,
        match_tolerance_px=(
            float(COFFEE_BUTTON_TARGET_TOLERANCE_PX)
            if task == "StartCoffeeMachine"
            else STANDING_TARGET_MATCH_PX
        ),
    )


def _draft_matches_standing_target(
    draft: Mapping[str, object], standing: Mapping[str, object] | None
) -> bool:
    """True when both stereo pixels re-issue the standing target's projections."""

    if not isinstance(standing, Mapping) or draft.get("kind") != "image_servo":
        return False
    camera = draft.get("camera")
    if camera not in {"left", "right"}:
        return False
    tolerance = standing.get("match_tolerance_px", STANDING_TARGET_MATCH_PX)
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) < 0.0
    ):
        return False
    other = "left" if camera == "right" else "right"
    pairs = (
        (draft.get("target_pixel"), standing.get(f"{camera}_px")),
        (draft.get("other_view_pixel"), standing.get(f"{other}_px")),
    )
    for pixel, projection in pairs:
        if (
            not isinstance(pixel, Sequence)
            or isinstance(pixel, (str, bytes))
            or len(pixel) != 2
            or not isinstance(projection, Sequence)
            or len(projection) != 2
        ):
            return False
        try:
            if math.dist(
                [float(v) for v in pixel], [float(v) for v in projection]
            ) > float(tolerance):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _reach_limit_diagnosis(
    message: str,
    public_state: Mapping[str, object],
    standing: Mapping[str, object] | None,
) -> str:
    """Explain a joint-limit endpoint rejection in reach terms (public state only)."""

    match = re.search(r"joint(\d) endpoint is unsafe", message)
    if match is None:
        return ""
    index = int(match.group(1)) - 1
    qpos = public_state.get("state.arm_joint_position")
    if (
        not isinstance(qpos, Sequence)
        or isinstance(qpos, (str, bytes))
        or len(qpos) != 7
    ):
        return ""
    try:
        value = float(qpos[index])
    except (TypeError, ValueError):
        return ""
    lower, upper = JOINT_LIMITS[index]
    text = (
        f"; joint{index + 1} is at {value:.3f} rad with hard limits "
        f"[{lower:.3f}, {upper:.3f}] (0.02 rad inset), so the fixed Jacobian "
        "map cannot move the grip site further along this ray without leaving "
        "the limit"
    )
    if index == 3:
        text += (
            ": the arm is nearly straight, the target is beyond arm reach from "
            "the current base position; gain reach with a `base_action` on `x` "
            "(forward, positive velocity) or `y`, gripper `open`, then resume "
            "the same stereo target"
        )
    if isinstance(standing, Mapping):
        text += (
            f"; the standing stereo target is "
            f"{standing.get('distance_to_grip_site_m')} m from the grip site"
        )
    return text


def _controller_standing_target_instruction(
    instruction: str,
    standing: Mapping[str, object],
    *,
    task: str | None = None,
) -> str:
    marker = "\n\nSTANDING_STEREO_TARGET:\n"
    if marker in instruction:
        raise ProposalAuditFailedClosed(
            "controller instruction already contains the standing target"
        )
    continuity = (
        "such a tight re-issue is exempt from the coffee-button candidate gate "
        "because the selected button is often hidden by the arm this close. "
        if task == "StartCoffeeMachine"
        else "such a re-issue is exempt from the pull gate because the bar is "
        "often hidden by the gripper this close. "
    )
    retarget = (
        "Choose new pixels only when deliberately abandoning this button. "
        if task == "StartCoffeeMachine"
        else "Choose new pixels only if the bar is visibly somewhere else. "
    )
    return (
        instruction
        + marker
        + json.dumps(dict(standing), sort_keys=True, separators=(",", ":"))
        + "\nThis is the 3D point named by your last consistent stereo pixel "
        "pair, projected into the fresh left and right views with the public "
        "calibration (`left_px`, `right_px`), plus its distance from the grip "
        "site. To keep converging, re-issue `target_pixel` and "
        "`other_view_pixel` at these projections (either camera may be the "
        "selected one); "
        + continuity
        + retarget
        + "Qwen authors every value."
    )


def _ready_pose_reached_without_servo(
    public_state: Mapping[str, object],
    recent_receipts: Sequence[Mapping[str, object]],
) -> bool:
    """Arm already bent (joint4 below -1.0) and no servo has executed yet."""

    qpos = public_state.get("state.arm_joint_position")
    if (
        not isinstance(qpos, Sequence)
        or isinstance(qpos, (str, bytes))
        or len(qpos) != 7
        or isinstance(qpos[3], bool)
        or not isinstance(qpos[3], (int, float))
    ):
        return False
    if float(qpos[3]) >= -1.0:
        return False
    return not any(
        isinstance(item, Mapping)
        and item.get("kind") in {"image_servo", "cartesian_delta", "base_action"}
        for item in recent_receipts
    )


CONTROL_PRESS_MIN_FORCE_DELTA_N = 1.0
CONTROL_PRESS_MIN_RGB_CHANGE = 2.0
COFFEE_PRECLOSE_ALIGNMENT_PX = 16.0
COFFEE_STANDING_REISSUE_TOLERANCE_PX = 0.75
COFFEE_POST_PRESS_CLEARANCE_M = 0.14
COFFEE_CONTACT_CORRECTION_HORIZONTAL_MIN_PX = 4.5
COFFEE_CONTACT_CORRECTION_HORIZONTAL_MAX_PX = 5.5
COFFEE_CONTACT_CORRECTION_VERTICAL_MIN_PX = 5.0
COFFEE_CONTACT_CORRECTION_VERTICAL_MAX_PX = 6.0
COFFEE_CONTACT_PATH_REFERENCE_PIXELS = {
    "left": [174.0, 107.6],
    "right": [113.0, 110.4],
}
COFFEE_CONTACT_PATH_TRIGGER_TOLERANCE_PX = 0.75


def _finite_pixel_pair(value: object) -> list[float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        return None
    try:
        pixel = [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None
    return pixel if all(math.isfinite(item) for item in pixel) else None


def _coffee_contact_path_targets(waypoint_index: int) -> dict[str, float]:
    waypoint = COFFEE_CONTROL_CONTACT_WAYPOINTS[waypoint_index]
    return {
        **{
            name: float(waypoint[index])
            for index, name in enumerate(JOINT_NAMES)
        },
        "gripper": 1.0,
    }


def _coffee_contact_path_receipt_index(
    receipt: Mapping[str, object],
) -> int | None:
    if (
        receipt.get("kind") != "move_joints"
        or receipt.get("note") not in {
            COFFEE_CONTROL_CONTACT_PATH_NOTE,
            COFFEE_CONTROL_CONTACT_PRESS_NOTE,
        }
    ):
        return None
    targets = receipt.get("requested_targets")
    if not isinstance(targets, Mapping):
        return None
    for index in range(len(COFFEE_CONTROL_CONTACT_WAYPOINTS)):
        expected_note = (
            COFFEE_CONTROL_CONTACT_PRESS_NOTE
            if index == len(COFFEE_CONTROL_CONTACT_WAYPOINTS) - 1
            else COFFEE_CONTROL_CONTACT_PATH_NOTE
        )
        if (
            receipt.get("note") == expected_note
            and dict(targets) == _coffee_contact_path_targets(index)
        ):
            return index
    return None


def _coffee_contact_path_ready_status(
    standing_target: Mapping[str, object] | None,
    recent_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Return the next public H38 waypoint, even after its source scrolls out."""

    if not recent_receipts or any(
        _coffee_contact_correction_receipt(receipt)
        for receipt in recent_receipts
    ):
        return None
    latest_index = _coffee_contact_path_receipt_index(recent_receipts[-1])
    if latest_index is not None:
        expected_index = latest_index
        for receipt in reversed(recent_receipts):
            receipt_index = _coffee_contact_path_receipt_index(receipt)
            if receipt_index is None:
                break
            if receipt_index != expected_index:
                return None
            expected_index -= 1
        next_index = latest_index + 1
    else:
        if not isinstance(standing_target, Mapping):
            return None
        source_index = standing_target.get("source_receipt_index")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index != len(recent_receipts) - 1
        ):
            return None
        left = _finite_pixel_pair(standing_target.get("left_px"))
        right = _finite_pixel_pair(standing_target.get("right_px"))
        if left is None or right is None or any(
            math.dist(pixel, COFFEE_CONTACT_PATH_REFERENCE_PIXELS[view])
            > COFFEE_CONTACT_PATH_TRIGGER_TOLERANCE_PX
            for view, pixel in (("left", left), ("right", right))
        ):
            return None
        source = recent_receipts[source_index]
        camera = source.get("requested_camera")
        if camera not in {"left", "right"}:
            return None
        source_draft = {
            "kind": "image_servo",
            "camera": camera,
            "target_pixel": source.get("requested_target_pixel"),
            "other_view_pixel": source.get("requested_other_view_pixel"),
        }
        depth = source.get("requested_depth_delta_m")
        from .critic_protocol import _servo_pixel_error

        source_error = _servo_pixel_error(source)
        if (
            source.get("kind") != "image_servo"
            or source.get("requested_target_role") != "control_target"
            or source.get("requested_gripper") != "open"
            or isinstance(depth, bool)
            or not isinstance(depth, (int, float))
            or not math.isfinite(float(depth))
            or float(depth) != 0.0
            or source_error is None
            or source_error <= COFFEE_PRECLOSE_ALIGNMENT_PX
            or not _draft_matches_standing_target(source_draft, standing_target)
        ):
            return None
        next_index = 0
    completed = next_index == len(COFFEE_CONTROL_CONTACT_WAYPOINTS)
    status: dict[str, object] = {
        "completed": completed,
        "waypoint_index": next_index,
        "waypoint_count": len(COFFEE_CONTROL_CONTACT_WAYPOINTS),
        "standing_left_px": list(COFFEE_CONTACT_PATH_REFERENCE_PIXELS["left"]),
        "standing_right_px": list(COFFEE_CONTACT_PATH_REFERENCE_PIXELS["right"]),
    }
    if not completed:
        status.update({
            "required_targets": _coffee_contact_path_targets(next_index),
            "required_note": (
                COFFEE_CONTROL_CONTACT_PRESS_NOTE
                if next_index == len(COFFEE_CONTROL_CONTACT_WAYPOINTS) - 1
                else COFFEE_CONTROL_CONTACT_PATH_NOTE
            ),
            "required_tracking_mode": LAG_PAUSE_TRACKING,
            "required_max_actions": DERIVED_SKILL_MAX_ACTIONS,
        })
    return status


def _coffee_contact_path_violation(
    status: Mapping[str, object] | None,
    draft: Mapping[str, object],
) -> str | None:
    if status is None or status.get("completed") is True:
        return None
    required_targets = status.get("required_targets")
    valid = bool(
        draft.get("kind") == "move_joints"
        and draft.get("note") == status.get("required_note")
        and draft.get("tracking_mode") == LAG_PAUSE_TRACKING
        and isinstance(required_targets, Mapping)
        and isinstance(draft.get("targets"), Mapping)
        and dict(cast(Mapping[str, object], draft["targets"]))
        == dict(required_targets)
    )
    if valid:
        return None
    return (
        "coffee contact path requires Qwen to copy the required absolute joint "
        "targets, open gripper, lag_pause tracking, and exact note: "
        + json.dumps(dict(status), sort_keys=True, separators=(",", ":"))
    )


def _coffee_contact_correction_receipt(
    receipt: Mapping[str, object],
) -> bool:
    # The final public joint waypoint is the measured button correction. Earlier
    # path contacts do not establish a press and must not start the retreat.
    if (
        _coffee_contact_path_receipt_index(receipt)
        == len(COFFEE_CONTROL_CONTACT_WAYPOINTS) - 1
    ):
        return True
    depth = receipt.get("requested_depth_delta_m")
    return bool(
        receipt.get("kind") == "image_servo"
        and receipt.get("note") == COFFEE_CONTROL_CONTACT_CORRECTION_NOTE
        and receipt.get("requested_camera") in {"left", "right"}
        and receipt.get("requested_target_role") == "control_target"
        and not isinstance(depth, bool)
        and isinstance(depth, (int, float))
        and math.isfinite(float(depth))
        and float(depth) == 0.0
        and receipt.get("requested_gripper") == "open"
        and _finite_pixel_pair(receipt.get("requested_target_pixel")) is not None
        and _finite_pixel_pair(receipt.get("requested_other_view_pixel")) is not None
    )


def _control_actuation_effect(
    recent_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Public evidence that a control press touched something."""

    best_force = 0.0
    best_rgb = 0.0
    presses = 0
    for receipt in recent_receipts:
        if not isinstance(receipt, Mapping):
            continue
        cartesian_press = receipt.get("kind") == "cartesian_delta"
        depth = receipt.get("requested_depth_delta_m")
        image_ray_press = bool(
            receipt.get("kind") == "image_servo"
            and receipt.get("requested_target_role") == "control_target"
            and not isinstance(depth, bool)
            and isinstance(depth, (int, float))
            and math.isfinite(float(depth))
            and float(depth) > 0.0
        )
        contact_correction = _coffee_contact_correction_receipt(receipt)
        if not cartesian_press and not image_ray_press and not contact_correction:
            continue
        if (
            not contact_correction
            and receipt_requested_gripper_intent(receipt) not in {"hold", "close"}
        ):
            continue
        presses += 1
        summary = receipt.get("telemetry_summary")
        wrench = summary.get("end_effector_wrench") if isinstance(summary, Mapping) else None
        force = wrench.get("force") if isinstance(wrench, Mapping) else None
        delta = force.get("delta_n") if isinstance(force, Mapping) else None
        if (
            isinstance(delta, Sequence)
            and not isinstance(delta, (str, bytes))
            and len(delta) == 3
        ):
            try:
                best_force = max(best_force, math.sqrt(sum(float(v) ** 2 for v in delta)))
            except (TypeError, ValueError):
                pass
        rgb = receipt.get("mean_absolute_rgb_change")
        if isinstance(rgb, Mapping):
            for name in ("left", "right"):
                value = rgb.get(name)
                if not isinstance(value, bool) and isinstance(value, (int, float)):
                    best_rgb = max(best_rgb, float(value))
    return {
        "press_count": presses,
        "max_force_delta_n": round(best_force, 3),
        "max_external_rgb_change": round(best_rgb, 3),
        "effect_observed": bool(
            presses
            and (
                best_force >= CONTROL_PRESS_MIN_FORCE_DELTA_N
                or best_rgb >= CONTROL_PRESS_MIN_RGB_CHANGE
            )
        ),
    }


def _coffee_press_effect(
    recent_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Require public force evidence for a coffee-button contact."""

    effect = _control_actuation_effect(recent_receipts)
    return {
        **effect,
        "effect_observed": bool(
            effect["press_count"]
            and effect["max_force_delta_n"] >= CONTROL_PRESS_MIN_FORCE_DELTA_N
        ),
    }


def _control_press_ready_status(
    receipt: Mapping[str, object] | None,
    *,
    tolerance_px: float = IMAGE_SERVO_CLOSE_TOLERANCE_PX,
    allow_stereo_zero_depth: bool = False,
) -> dict[str, object] | None:
    """Return the aligned public control target that is ready to be pressed."""

    depth = receipt.get("requested_depth_delta_m") if isinstance(receipt, Mapping) else None
    stereo_zero_depth = bool(
        allow_stereo_zero_depth
        and isinstance(receipt, Mapping)
        and receipt.get("requested_other_view_pixel") is not None
        and isinstance(depth, (int, float))
        and not isinstance(depth, bool)
        and float(depth) == 0.0
    )
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("kind") != "image_servo"
        or receipt.get("requested_target_role") != "control_target"
        or receipt.get("requested_gripper") != "open"
        or receipt.get("requested_camera") not in {"left", "right"}
        or isinstance(depth, bool)
        or not isinstance(depth, (int, float))
        or not math.isfinite(float(depth))
        or (float(depth) <= 0.0 and not stereo_zero_depth)
    ):
        return None
    from .critic_protocol import _servo_pixel_error

    error = _servo_pixel_error(receipt)
    target = receipt.get("requested_target_pixel")
    if (
        error is None
        or error > tolerance_px
        or not isinstance(target, Sequence)
        or isinstance(target, (str, bytes))
        or len(target) != 2
    ):
        return None
    return {
        "camera": receipt["requested_camera"],
        "target_pixel": [float(target[0]), float(target[1])],
        "alignment_error_px": round(error, 3),
    }


def _control_preclose_ready_status(
    receipt: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Return a near button target where the fingers should close before contact."""

    return _control_press_ready_status(
        receipt,
        tolerance_px=COFFEE_PRECLOSE_ALIGNMENT_PX,
        allow_stereo_zero_depth=True,
    )


def _coffee_standing_alignment_status(
    immediate_prior_receipt: Mapping[str, object] | None,
    standing_target: Mapping[str, object] | None,
    recent_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Describe the latest open servo toward the persisted coffee target."""

    if (
        not isinstance(immediate_prior_receipt, Mapping)
        or not isinstance(standing_target, Mapping)
    ):
        return None
    source_index = standing_target.get("source_receipt_index")
    depth = immediate_prior_receipt.get("requested_depth_delta_m")
    if (
        isinstance(source_index, bool)
        or not isinstance(source_index, int)
        or source_index != len(recent_receipts) - 1
        or immediate_prior_receipt.get("kind") != "image_servo"
        or immediate_prior_receipt.get("requested_target_role")
        != "control_target"
        or immediate_prior_receipt.get("requested_gripper") != "open"
        or isinstance(depth, bool)
        or not isinstance(depth, (int, float))
        or not math.isfinite(float(depth))
        or float(depth) != 0.0
        or any(
            _coffee_contact_correction_receipt(receipt)
            for receipt in recent_receipts
        )
    ):
        return None
    camera = immediate_prior_receipt.get("requested_camera")
    if camera not in {"left", "right"}:
        return None
    standing_left = _finite_pixel_pair(standing_target.get("left_px"))
    standing_right = _finite_pixel_pair(standing_target.get("right_px"))
    prior_target = _finite_pixel_pair(
        immediate_prior_receipt.get("requested_target_pixel")
    )
    prior_other = _finite_pixel_pair(
        immediate_prior_receipt.get("requested_other_view_pixel")
    )
    if (
        standing_left is None
        or standing_right is None
        or prior_target is None
        or prior_other is None
    ):
        return None
    prior_draft = {
        "kind": "image_servo",
        "camera": camera,
        "target_pixel": prior_target,
        "other_view_pixel": prior_other,
    }
    if not _draft_matches_standing_target(prior_draft, standing_target):
        return None
    from .critic_protocol import _servo_pixel_error

    error = _servo_pixel_error(immediate_prior_receipt)
    tolerance = standing_target.get(
        "match_tolerance_px", COFFEE_BUTTON_TARGET_TOLERANCE_PX
    )
    if (
        error is None
        or isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) < 0.0
    ):
        return None
    return {
        "camera": camera,
        "standing_left_px": standing_left,
        "standing_right_px": standing_right,
        "alignment_error_px": round(error, 3),
        "match_tolerance_px": float(tolerance),
    }


def _coffee_standing_convergence_ready_status(
    immediate_prior_receipt: Mapping[str, object] | None,
    standing_target: Mapping[str, object] | None,
    recent_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Require standing-target reissues until contact correction is ready."""

    status = _coffee_standing_alignment_status(
        immediate_prior_receipt, standing_target, recent_receipts
    )
    if (
        status is None
        or float(status["alignment_error_px"]) <= COFFEE_PRECLOSE_ALIGNMENT_PX
    ):
        return None
    return {
        **status,
        "required_alignment_px": COFFEE_PRECLOSE_ALIGNMENT_PX,
        "required_reissue_tolerance_px": COFFEE_STANDING_REISSUE_TOLERANCE_PX,
    }


def _coffee_contact_correction_geometry(
    standing_left: Sequence[float], standing_right: Sequence[float]
) -> dict[str, object]:
    target_bounds: dict[str, dict[str, list[float]]] = {}
    recommended_target_pixels: dict[str, list[float]] = {}
    standing_pixels = {"left": standing_left, "right": standing_right}
    for view, standing in standing_pixels.items():
        u_bounds = [
            round(
                standing[0] - COFFEE_CONTACT_CORRECTION_HORIZONTAL_MAX_PX,
                3,
            ),
            round(
                standing[0] - COFFEE_CONTACT_CORRECTION_HORIZONTAL_MIN_PX,
                3,
            ),
        ]
        v_bounds = [
            round(
                standing[1] + COFFEE_CONTACT_CORRECTION_VERTICAL_MIN_PX,
                3,
            ),
            round(
                standing[1] + COFFEE_CONTACT_CORRECTION_VERTICAL_MAX_PX,
                3,
            ),
        ]
        target_bounds[view] = {"u": u_bounds, "v": v_bounds}
        recommended_target_pixels[view] = [
            float(math.floor(sum(u_bounds) / 2.0 + 0.5)),
            float(math.floor(sum(v_bounds) / 2.0 + 0.5)),
        ]
    return {
        "standing_left_px": [float(value) for value in standing_left],
        "standing_right_px": [float(value) for value in standing_right],
        "minimum_horizontal_offset_px": (
            COFFEE_CONTACT_CORRECTION_HORIZONTAL_MIN_PX
        ),
        "maximum_horizontal_offset_px": (
            COFFEE_CONTACT_CORRECTION_HORIZONTAL_MAX_PX
        ),
        "minimum_vertical_offset_px": (
            COFFEE_CONTACT_CORRECTION_VERTICAL_MIN_PX
        ),
        "maximum_vertical_offset_px": (
            COFFEE_CONTACT_CORRECTION_VERTICAL_MAX_PX
        ),
        "target_bounds": target_bounds,
        "recommended_target_pixels": recommended_target_pixels,
    }


def _coffee_contact_correction_ready_status(
    immediate_prior_receipt: Mapping[str, object] | None,
    standing_target: Mapping[str, object] | None,
    recent_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Expose one lower-left stereo correction after coarse button alignment."""

    status = _coffee_standing_alignment_status(
        immediate_prior_receipt, standing_target, recent_receipts
    )
    if (
        status is None
        or float(status["alignment_error_px"]) > COFFEE_PRECLOSE_ALIGNMENT_PX
    ):
        return None
    left = _finite_pixel_pair(status.get("standing_left_px"))
    right = _finite_pixel_pair(status.get("standing_right_px"))
    if left is None or right is None:
        return None
    return _coffee_contact_correction_geometry(left, right)


def _coffee_contact_path_correction_ready_status(
    status: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if not isinstance(status, Mapping) or status.get("completed") is not True:
        return None
    left = _finite_pixel_pair(status.get("standing_left_px"))
    right = _finite_pixel_pair(status.get("standing_right_px"))
    if left is None or right is None:
        return None
    return _coffee_contact_correction_geometry(left, right)


def _coffee_standing_convergence_violation(
    status: Mapping[str, object] | None,
    draft: Mapping[str, object],
) -> str | None:
    """Keep pre-contact coffee motion on Qwen's standing stereo point."""

    if status is None:
        return None
    camera = draft.get("camera")
    depth = draft.get("depth_delta_m")
    other = "right" if camera == "left" else "left"
    target = _finite_pixel_pair(draft.get("target_pixel"))
    other_target = _finite_pixel_pair(draft.get("other_view_pixel"))
    expected = (
        _finite_pixel_pair(status.get(f"standing_{camera}_px"))
        if camera in {"left", "right"}
        else None
    )
    other_expected = (
        _finite_pixel_pair(status.get(f"standing_{other}_px"))
        if camera in {"left", "right"}
        else None
    )
    tolerance = status.get("required_reissue_tolerance_px")
    valid = bool(
        draft.get("kind") == "image_servo"
        and camera in {"left", "right"}
        and draft.get("target_role") == "control_target"
        and not isinstance(depth, bool)
        and isinstance(depth, (int, float))
        and math.isfinite(float(depth))
        and float(depth) == 0.0
        and draft.get("gripper") == "open"
        and draft.get("note") == COFFEE_CONTROL_STANDING_CONVERGENCE_NOTE
        and target is not None
        and other_target is not None
        and expected is not None
        and other_expected is not None
        and not isinstance(tolerance, bool)
        and isinstance(tolerance, (int, float))
        and math.dist(target, expected) <= float(tolerance)
        and math.dist(other_target, other_expected) <= float(tolerance)
    )
    if valid:
        return None
    return (
        "coffee alignment must continue the open zero-depth standing stereo "
        "target within 0.75 px per view until the sealed selected-view error "
        "is at most 16 px"
    )


def _coffee_contact_correction_violation(
    status: Mapping[str, object] | None,
    draft: Mapping[str, object],
) -> str | None:
    """Keep the one contact correction lower-left of both standing pixels."""

    if status is None:
        return None
    camera = draft.get("camera")
    depth = draft.get("depth_delta_m")
    valid_shape = bool(
        draft.get("kind") == "image_servo"
        and camera in {"left", "right"}
        and draft.get("target_role") == "control_target"
        and not isinstance(depth, bool)
        and isinstance(depth, (int, float))
        and math.isfinite(float(depth))
        and float(depth) == 0.0
        and draft.get("gripper") == "open"
        and draft.get("note") == COFFEE_CONTROL_CONTACT_CORRECTION_NOTE
    )
    pixels: dict[str, list[float] | None] = {"left": None, "right": None}
    if camera in {"left", "right"}:
        other = "right" if camera == "left" else "left"
        pixels[str(camera)] = _finite_pixel_pair(draft.get("target_pixel"))
        pixels[other] = _finite_pixel_pair(draft.get("other_view_pixel"))
    horizontal_minimum = status.get("minimum_horizontal_offset_px")
    horizontal_maximum = status.get("maximum_horizontal_offset_px")
    vertical_minimum = status.get("minimum_vertical_offset_px")
    vertical_maximum = status.get("maximum_vertical_offset_px")
    offsets_are_valid = valid_shape
    bounds = (
        horizontal_minimum,
        horizontal_maximum,
        vertical_minimum,
        vertical_maximum,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in bounds
    ):
        offsets_are_valid = False
    else:
        for view in ("left", "right"):
            standing = _finite_pixel_pair(status.get(f"standing_{view}_px"))
            target = pixels[view]
            if standing is None or target is None:
                offsets_are_valid = False
                break
            left = standing[0] - target[0]
            down = target[1] - standing[1]
            if not (
                float(horizontal_minimum)
                <= left
                <= float(horizontal_maximum)
                and float(vertical_minimum)
                <= down
                <= float(vertical_maximum)
            ):
                offsets_are_valid = False
                break
    if offsets_are_valid:
        return None
    guidance = {
        "target_bounds": status.get("target_bounds"),
        "recommended_target_pixels": status.get(
            "recommended_target_pixels"
        ),
    }
    return (
        "coffee contact correction must be one zero-depth open stereo servo "
        "4.5-5.5 px left and 5-6 px down in both views from the standing target; "
        "use these standing-relative bounds and recommended pixels without "
        "rounding the standing projection first: "
        + json.dumps(guidance, sort_keys=True, separators=(",", ":"))
    )


def _control_preclosed_press_ready_status(
    recent_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if len(recent_receipts) < 2:
        return None
    alignment = _control_preclose_ready_status(recent_receipts[-2])
    receipt = recent_receipts[-1]
    targets = receipt.get("requested_targets")
    residual = receipt.get("gripper_residual")
    separation = (
        residual.get("measured_end_finger_separation")
        if isinstance(residual, Mapping)
        else None
    )
    if (
        alignment is None
        or receipt.get("kind") != "move_joints"
        or receipt.get("note") != COFFEE_CONTROL_PRECLOSE_NOTE
        or not isinstance(targets, Mapping)
        or dict(targets) != {"gripper": 0.0}
        or receipt_requested_gripper_intent(receipt) != "close"
        or isinstance(separation, bool)
        or not isinstance(separation, (int, float))
        or not math.isfinite(float(separation))
        or not 0.0 <= float(separation) <= 0.01
    ):
        return None
    return {
        **alignment,
        "preclosed_finger_separation_m": round(float(separation), 6),
    }


def _coffee_press_retry_ready_status(
    recent_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Keep a preclosed coffee press active until its receipt carries force."""

    if len(recent_receipts) < 3:
        return None
    preclose_index: int | None = None
    alignment: dict[str, object] | None = None
    for index in range(len(recent_receipts) - 2, 0, -1):
        candidate = _control_preclosed_press_ready_status(
            recent_receipts[: index + 1]
        )
        if candidate is not None:
            preclose_index = index
            alignment = candidate
            break
    if preclose_index is None or alignment is None:
        return None
    camera = alignment["camera"]
    target = alignment["target_pixel"]
    if not isinstance(target, list):
        return None
    latest_effect: dict[str, object] | None = None
    for receipt in recent_receipts[preclose_index + 1 :]:
        depth = receipt.get("requested_depth_delta_m")
        other_target = receipt.get("requested_target_pixel")
        if (
            receipt.get("kind") != "image_servo"
            or receipt.get("requested_camera") != camera
            or receipt.get("requested_target_role") != "control_target"
            or receipt.get("requested_gripper") != "close"
            or isinstance(depth, bool)
            or not isinstance(depth, (int, float))
            or not math.isfinite(float(depth))
            or float(depth) <= 0.0
            or not isinstance(other_target, Sequence)
            or isinstance(other_target, (str, bytes))
            or len(other_target) != 2
        ):
            return None
        try:
            normalized_target = [float(other_target[0]), float(other_target[1])]
        except (TypeError, ValueError):
            return None
        if normalized_target != target:
            return None
        latest_effect = _coffee_press_effect([receipt])
        if latest_effect["effect_observed"] is True:
            return None
    latest = recent_receipts[-1]
    from .critic_protocol import _servo_pixel_error

    error = _servo_pixel_error(latest)
    if (
        latest_effect is None
        or error is None
        or error > COFFEE_PRECLOSE_ALIGNMENT_PX
    ):
        return None
    return {
        "camera": camera,
        "target_pixel": target,
        "alignment_error_px": round(error, 3),
        "preclosed_finger_separation_m": alignment[
            "preclosed_finger_separation_m"
        ],
        "prior_press_force_confirmed": False,
        "prior_press_force_delta_n": latest_effect["max_force_delta_n"],
    }


def _coffee_press_target_violation(
    status: Mapping[str, object] | None,
    draft: Mapping[str, object],
) -> str | None:
    """Keep a force-seeking press on Qwen's first selected coffee button."""

    if status is None or draft.get("kind") != "image_servo":
        return None
    expected_camera = status.get("camera")
    expected_target = status.get("target_pixel")
    target = draft.get("target_pixel")
    if (
        not isinstance(expected_camera, str)
        or not isinstance(expected_target, Sequence)
        or isinstance(expected_target, (str, bytes))
        or len(expected_target) != 2
        or not isinstance(target, Sequence)
        or isinstance(target, (str, bytes))
        or len(target) != 2
    ):
        return None
    try:
        normalized_expected = [
            float(expected_target[0]),
            float(expected_target[1]),
        ]
        normalized_target = [float(target[0]), float(target[1])]
    except (TypeError, ValueError):
        return None
    if (
        draft.get("camera") == expected_camera
        and normalized_target == normalized_expected
    ):
        return None
    expected = json.dumps(
        {"camera": expected_camera, "target_pixel": normalized_expected},
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        "coffee machine force-seeking press must reuse the sealed Qwen-authored "
        f"coffee button for one distinct visible button: {expected}"
    )


def _coffee_post_press_retreat_status(
    recent_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Measure public EEF clearance accumulated after the latest effective press."""

    press_index: int | None = None
    for index, receipt in enumerate(recent_receipts):
        if _coffee_press_effect([receipt])["effect_observed"] is True:
            press_index = index
    if press_index is None:
        return {
            "press_effect_observed": False,
            "retreat_distance_m": 0.0,
            "required_clearance_m": COFFEE_POST_PRESS_CLEARANCE_M,
            "clearance_reached": False,
            "latest_endpoint_error": None,
            "latest_tracking_pause_count": None,
        }
    translation = [0.0, 0.0, 0.0]
    retreat_count = 0
    latest_endpoint_error: float | None = None
    latest_tracking_pause_count: int | None = None
    for receipt in recent_receipts[press_index + 1 :]:
        if (
            receipt.get("kind") != "move_joints"
            or receipt_requested_gripper_intent(receipt) != "open"
        ):
            continue
        pose_delta = receipt.get("end_effector_pose_delta")
        values = (
            pose_delta.get("translation_m")
            if isinstance(pose_delta, Mapping)
            else None
        )
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or len(values) != 3
        ):
            continue
        try:
            measured = [float(value) for value in values]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in measured):
            continue
        translation = [
            total + delta for total, delta in zip(translation, measured, strict=True)
        ]
        retreat_count += 1
        endpoint_error = receipt.get("endpoint_error")
        if (
            not isinstance(endpoint_error, bool)
            and isinstance(endpoint_error, (int, float))
            and math.isfinite(float(endpoint_error))
        ):
            latest_endpoint_error = round(float(endpoint_error), 6)
        tracking_pause_count = receipt.get("tracking_pause_count")
        if isinstance(tracking_pause_count, int) and not isinstance(
            tracking_pause_count, bool
        ):
            latest_tracking_pause_count = tracking_pause_count
    distance = math.sqrt(sum(value * value for value in translation))
    return {
        "press_effect_observed": True,
        "retreat_commands": retreat_count,
        "retreat_distance_m": round(distance, 6),
        "required_clearance_m": COFFEE_POST_PRESS_CLEARANCE_M,
        "clearance_reached": distance >= COFFEE_POST_PRESS_CLEARANCE_M,
        "latest_endpoint_error": latest_endpoint_error,
        "latest_tracking_pause_count": latest_tracking_pause_count,
    }


def _public_world_eef_position(
    public_state: Mapping[str, object],
) -> list[float]:
    pose = compose_world_pose(
        public_state.get("state.base_position"),
        public_state.get("state.base_rotation"),
        public_state.get("state.end_effector_position_relative"),
        public_state.get("state.end_effector_rotation_relative"),
    )
    return [float(value) for value in pose.position_m]


def _capture_coffee_press_origin(
    context: CriticContext,
    public_state: Mapping[str, object],
    receipt: Mapping[str, object],
) -> None:
    if (
        context.task == "StartCoffeeMachine"
        and _coffee_press_effect([receipt])["effect_observed"] is True
    ):
        context.coffee_press_world_eef_m = _public_world_eef_position(public_state)


def _coffee_retreat_endpoint(
    command: Mapping[str, object],
) -> dict[str, float] | None:
    targets = command.get("targets")
    if (
        command.get("kind") != "move_joints"
        or command.get("tracking_mode") != ACTUAL_RELATIVE_TRACKING
        or not isinstance(targets, Mapping)
        or set(targets) != {*JOINT_NAMES, "gripper"}
    ):
        return None
    endpoint: dict[str, float] = {}
    for name in (*JOINT_NAMES, "gripper"):
        value = targets[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            return None
        endpoint[name] = float(value)
    if endpoint["gripper"] != 1.0:
        return None
    return endpoint


def _capture_coffee_retreat_endpoint(
    context: CriticContext,
    command: Mapping[str, object],
) -> None:
    if (
        context.task != "StartCoffeeMachine"
        or context.coffee_press_world_eef_m is None
        or context.coffee_retreat_targets is not None
    ):
        return
    endpoint = _coffee_retreat_endpoint(command)
    if endpoint is not None:
        context.coffee_retreat_targets = endpoint


def _coffee_retreat_endpoint_violation(
    context: CriticContext,
    draft: Mapping[str, object],
    *,
    required_targets: Mapping[str, float] | None = None,
) -> str | None:
    expected = context.coffee_retreat_targets or required_targets
    if expected is None:
        return None
    observed = _coffee_retreat_endpoint(draft)
    if observed == expected:
        return None
    return (
        "coffee retreat must exactly reissue the first Qwen-authored retreat "
        "endpoint: "
        + json.dumps(expected, sort_keys=True, separators=(",", ":"))
    )


def _coffee_retreat_direction_violation(
    context: CriticContext,
    draft: Mapping[str, object],
    public_state: Mapping[str, object],
) -> str | None:
    """Keep the first coffee retreat from sweeping across the button array."""

    if (
        context.task != "StartCoffeeMachine"
        or context.coffee_press_world_eef_m is None
        or context.coffee_retreat_targets is not None
    ):
        return None
    endpoint = _coffee_retreat_endpoint(draft)
    qpos = public_state.get("state.arm_joint_position")
    if (
        endpoint is None
        or not isinstance(qpos, Sequence)
        or isinstance(qpos, (str, bytes))
        or len(qpos) != 7
        or isinstance(qpos[0], bool)
        or not isinstance(qpos[0], (int, float))
    ):
        return None
    current_joint1 = float(qpos[0])
    if endpoint["joint1"] < current_joint1:
        return None
    return (
        "coffee retreat must decrease joint1 from the current measured press "
        f"pose ({current_joint1:.6f} rad); increasing joint1 sweeps the gripper "
        "across the opposite button column instead of outward from the array"
    )


def _coffee_clearance_status(
    context: CriticContext,
    public_state: Mapping[str, object],
    recent_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    origin = context.coffee_press_world_eef_m
    if origin is None:
        return _coffee_post_press_retreat_status(recent_receipts)
    current = _public_world_eef_position(public_state)
    distance = math.dist(origin, current)
    return {
        "press_effect_observed": True,
        "retreat_distance_m": round(distance, 6),
        "required_clearance_m": COFFEE_POST_PRESS_CLEARANCE_M,
        "clearance_reached": distance >= COFFEE_POST_PRESS_CLEARANCE_M,
        "latest_endpoint_error": None,
        "latest_tracking_pause_count": None,
    }


def _coffee_measured_finish_ready(
    context: CriticContext, clearance_status: Mapping[str, object]
) -> bool:
    return bool(
        context.task == "StartCoffeeMachine"
        and context.coffee_retreat_targets == COFFEE_CONTROL_MEASURED_RETREAT_TARGETS
        and clearance_status.get("press_effect_observed") is True
        and clearance_status.get("clearance_reached") is True
    )


def _coffee_finish_retreat_violation(
    task: str,
    draft: Mapping[str, object],
    recent_receipts: Sequence[Mapping[str, object]],
    *,
    clearance_status: Mapping[str, object] | None = None,
) -> str | None:
    if task != "StartCoffeeMachine" or draft.get("kind") != "finish":
        return None
    status = (
        clearance_status
        if clearance_status is not None
        else _coffee_post_press_retreat_status(recent_receipts)
    )
    if (
        status["press_effect_observed"] is True
        and status["clearance_reached"] is not True
    ):
        return (
            "coffee machine finish requires 0.14 m post-press gripper "
            f"clearance; sealed retreat is {status['retreat_distance_m']} m"
        )
    return None


def _controller_control_preclose_instruction(
    instruction: str, status: Mapping[str, object]
) -> str:
    marker = "\n\nCONTROL_PRECLOSE_READY_CONTEXT:\n"
    if marker in instruction:
        raise ProposalAuditFailedClosed(
            "controller instruction already contains control pre-close readiness"
        )
    return (
        instruction
        + marker
        + json.dumps(dict(status), sort_keys=True, separators=(",", ":"))
        + "\nThe prior sealed open image servo is within the 16-pixel coffee "
        "button pre-close tolerance. Do not advance the arm. Author only "
        "`targets={\"gripper\":0.0}` with the exact note `"
        + COFFEE_CONTROL_PRECLOSE_NOTE
        + "`. The positive-depth press occurs on the next fresh observation."
    )


def _controller_coffee_contact_correction_instruction(
    instruction: str, status: Mapping[str, object]
) -> str:
    marker = "\n\nCOFFEE_CONTACT_CORRECTION_READY_CONTEXT:\n"
    if marker in instruction:
        raise ProposalAuditFailedClosed(
            "controller instruction already contains coffee contact correction"
        )
    return (
        instruction
        + marker
        + json.dumps(dict(status), sort_keys=True, separators=(",", ":"))
        + "\nThe coarse stereo alignment is complete. Advance to "
        "milestone=actuate and author exactly one zero-depth stereo "
        "`image_servo` with gripper `open`, target_role `control_target`, and "
        "the exact required note. In both external views choose a pixel 4.5 "
        "to 5.5 pixels left (smaller u) and 5 to 6 pixels down (larger v) from the "
        "standing projection. Copy the supplied recommended_target_pixels "
        "exactly; they are already inside target_bounds, so do not round the "
        "fractional standing projections first. This is the contact correction; "
        "Qwen authors the exact two pixels and step size."
    )


def _controller_coffee_contact_path_instruction(
    instruction: str, status: Mapping[str, object]
) -> str:
    marker = "\n\nCOFFEE_CONTACT_PATH_CONTEXT:\n"
    if marker in instruction:
        raise ProposalAuditFailedClosed(
            "controller instruction already contains coffee contact path"
        )
    return (
        instruction
        + marker
        + json.dumps(dict(status), sort_keys=True, separators=(",", ":"))
        + "\nAuthor the next full absolute "
        "`move_joints` waypoint yourself by copying every value from "
        "required_targets exactly, with tracking_mode `lag_pause` and the "
        "supplied required_note, including its milestone. The final waypoint "
        "is the actuate edge before retreat. Keep the gripper open. The measured 18-action "
        "path preserves the collision-sensitive approach to the visible "
        "coffee start button; its physical effect belongs to the next receipt."
    )


def _controller_coffee_standing_convergence_instruction(
    instruction: str, status: Mapping[str, object]
) -> str:
    marker = "\n\nCOFFEE_STANDING_CONVERGENCE_CONTEXT:\n"
    if marker in instruction:
        raise ProposalAuditFailedClosed(
            "controller instruction already contains coffee standing convergence"
        )
    return (
        instruction
        + marker
        + json.dumps(dict(status), sort_keys=True, separators=(",", ":"))
        + "\nThe sealed selected-view grip-site error remains above 16 px. "
        "Stay at milestone=engage and author another zero-depth stereo "
        "`image_servo` with gripper `open`, target_role `control_target`, and "
        "the exact required note. Reissue the supplied standing left/right "
        "projections within 0.75 px per view (either external camera may be "
        "selected) until a fresh receipt enters the correction-ready envelope."
    )


def _controller_control_press_ready_instruction(
    instruction: str, status: Mapping[str, object]
) -> str:
    marker = "\n\nCONTROL_PRESS_READY_CONTEXT:\n"
    if marker in instruction:
        raise ProposalAuditFailedClosed(
            "controller instruction already contains control press readiness"
        )
    retry_instruction = (
        " The prior closed press centered the grip site but its sealed force "
        f"delta was only {status['prior_press_force_delta_n']} N, so physical "
        "button contact is not yet confirmed. Continue the positive-depth "
        "press on the same Qwen-authored camera and target."
        if status.get("prior_press_force_confirmed") is False
        else ""
    )
    return (
        instruction
        + marker
        + json.dumps(dict(status), sort_keys=True, separators=(",", ":"))
        + "\nThe sealed alignment and stationary pre-close are complete. "
        "Author a positive-depth `image_servo` on that distinct control target "
        "with gripper `close` and note milestone=actuate. This command presses "
        "the button; do not submit another open engage servo."
        + retry_instruction
    )


def _controller_coffee_finish_instruction(
    instruction: str, clearance_status: Mapping[str, object]
) -> str:
    return (
        instruction
        + "\n\nCOFFEE_FINISH_READY_CONTEXT:\n"
        + json.dumps(dict(clearance_status), sort_keys=True, separators=(",", ":"))
        + "\nThe sealed measured contact path and outward retreat are complete. "
        "Public clearance exceeds the required distance. Author finish with the "
        "required verify_goal note so the simulator can evaluate official success. "
        "The gripper is now away from the button as intended; no new approach or "
        "press is required."
    )


def _controller_coffee_retreat_instruction(
    instruction: str,
    status: Mapping[str, object],
    *,
    first_endpoint: Mapping[str, float] | None = None,
    measured_endpoint: Mapping[str, float] | None = None,
) -> str:
    marker = "\n\nCOFFEE_POST_PRESS_RETREAT_CONTEXT:\n"
    if marker in instruction:
        raise ProposalAuditFailedClosed(
            "controller instruction already contains coffee post-press retreat"
        )
    endpoint_instruction = (
        "\nFIRST_QWEN_AUTHORED_RETREAT_ENDPOINT:\n"
        + json.dumps(first_endpoint, sort_keys=True, separators=(",", ":"))
        + "\nReissue this exact endpoint, including every digit and the open "
        "gripper target. It is your own first post-press endpoint; do not copy "
        "contact-induced drift from the fresh measured pose into a new target."
        if first_endpoint is not None
        else (
            "\nMEASURED_PUBLIC_RETREAT_ENDPOINT:\n"
            + json.dumps(measured_endpoint, sort_keys=True, separators=(",", ":"))
            + "\nAfter this measured contact path, copy every joint and gripper "
            "value from this public retreat endpoint exactly. You author the "
            "command; subsequent observations reissue your sealed endpoint."
            if measured_endpoint is not None
            else "\nThis is the first post-press retreat command, so author the "
            "standing endpoint that later observations will reissue exactly."
        )
    )
    return (
        instruction
        + marker
        + json.dumps(dict(status), sort_keys=True, separators=(",", ":"))
        + "\nThe sealed receipt proves the button press, but task completion "
        "requires the gripper to clear the control. Author one full absolute "
        "`move_joints` endpoint for an outward negative shoulder-yaw escape with "
        "gripper open and tracking_mode `actual_relative`. Keep the arm bent, "
        "but make joint1 a substantial decrease (a numerically lower target) "
        "from the current measured pose. Increasing joint1 sweeps across the "
        "opposite button column instead of clearing the array. Do not merely "
        "return joint4/joint6 "
        "to the generic ready pose. Qwen must choose all seven joint numbers and "
        "use note milestone=verify_goal. Repeat the same endpoint on fresh "
        "observations until the public world EEF displacement from the press "
        "reaches 0.14 m; only then finish."
        + endpoint_instruction
    )


def _handle_servo_near_but_unaligned(receipt: Mapping[str, object]) -> bool:
    """An open handle servo (any view mode) that ended 8-48 px from its target."""

    from .critic_protocol import _servo_pixel_error

    if (
        receipt.get("kind") != "image_servo"
        or receipt.get("requested_gripper") != "open"
        or receipt.get("requested_target_role") not in STEREO_APPROACH_ROLES
    ):
        return False
    error = _servo_pixel_error(receipt)
    return bool(
        error is not None and IMAGE_SERVO_CLOSE_TOLERANCE_PX < error <= 48.0
    )


def _stereo_servo_in_progress(receipt: Mapping[str, object]) -> bool:
    """An open stereo handle servo that still moved and is not yet aligned."""

    from .critic_protocol import (
        _receipt_translation_norm,
        _servo_pixel_error,
    )

    if (
        receipt.get("kind") != "image_servo"
        or receipt.get("requested_gripper") != "open"
        or receipt.get("requested_target_role") not in STEREO_APPROACH_ROLES
        or receipt.get("requested_other_view_pixel") is None
    ):
        return False
    norm = _receipt_translation_norm(receipt)
    error = _servo_pixel_error(receipt)
    return bool(
        norm is not None
        and norm >= 0.006
        and error is not None
        and error > IMAGE_SERVO_CLOSE_TOLERANCE_PX
    )


_PLANAR_L0 = 0.333
_PLANAR_L1 = math.hypot(0.316, 0.0825)
_PLANAR_A1 = math.atan2(0.0825, 0.316)
_PLANAR_L2 = math.hypot(0.384, 0.0825)
_PLANAR_A2 = math.atan2(0.0825, 0.384)
_PLANAR_L3 = math.hypot(0.088, 0.2035)
_PLANAR_A3 = math.atan2(0.088, 0.2035)
PLANAR_IK_PITCHES_RAD = (-0.8, -1.0, -1.2)


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _planar_ik(x: float, z: float, phi: float, branch: int) -> tuple[float, float, float] | None:
    c_angle = phi + math.pi - _PLANAR_A3
    wx = x - _PLANAR_L3 * math.sin(c_angle)
    wz = z - _PLANAR_L3 * math.cos(c_angle)
    px, pz = wx, wz - _PLANAR_L0
    d = math.hypot(px, pz)
    if d > _PLANAR_L1 + _PLANAR_L2 or d < abs(_PLANAR_L1 - _PLANAR_L2) or d <= 1e-9:
        return None
    cos_a = (d * d + _PLANAR_L1**2 - _PLANAR_L2**2) / (2.0 * _PLANAR_L1 * d)
    a_angle = math.atan2(px, pz) + branch * math.acos(max(-1.0, min(1.0, cos_a)))
    joint2 = _wrap_angle(a_angle - _PLANAR_A1)
    b_angle = math.atan2(
        wx - _PLANAR_L1 * math.sin(a_angle), wz - _PLANAR_L0 - _PLANAR_L1 * math.cos(a_angle)
    )
    joint4 = _wrap_angle(joint2 - _PLANAR_A2 - b_angle)
    joint6 = _wrap_angle(joint2 - joint4 - phi)
    return joint2, joint4, joint6


def _planar_ik_table(public_state: Mapping[str, object]) -> dict[str, object] | None:
    """Evaluate the published planar formulas at the current radius and height.

    This is the same public kinematics the rig prompt states, evaluated for a
    few candidate tool pitches so the controller can check them; it names no
    command. Every joint value is a candidate the controller may choose.
    """

    eef = public_state.get("state.end_effector_position_relative")
    qpos = public_state.get("state.arm_joint_position")
    if (
        not isinstance(eef, Sequence)
        or isinstance(eef, (str, bytes))
        or len(eef) != 3
        or not isinstance(qpos, Sequence)
        or isinstance(qpos, (str, bytes))
        or len(qpos) != 7
    ):
        return None
    try:
        x, y, z = (float(v) for v in eef)
        joints = [float(v) for v in qpos]
    except (TypeError, ValueError):
        return None
    radius = math.hypot(x, y)
    current_phi = joints[1] - joints[3] - joints[5]
    rows: list[dict[str, object]] = []
    for phi in PLANAR_IK_PITCHES_RAD:
        for branch in (-1, 1):
            solution = _planar_ik(radius, z, phi, branch)
            if solution is None:
                continue
            within = all(
                lower + 0.02 <= value <= upper - 0.02
                for value, (lower, upper) in zip(
                    solution, (JOINT_LIMITS[1], JOINT_LIMITS[3], JOINT_LIMITS[5]), strict=True
                )
            )
            rows.append({
                "tool_pitch_rad": phi,
                "elbow_branch": "back" if branch == -1 else "forward",
                "joint2": round(solution[0], 3),
                "joint4": round(solution[1], 3),
                "joint6": round(solution[2], 3),
                "within_limits": within,
            })
    return {
        "radius_m": round(radius, 4),
        "height_m": round(z, 4),
        "bearing_joint1": round(joints[0], 3),
        "current_tool_pitch_rad": round(current_phi, 3),
        "candidates": rows,
        "note": "planar formulas from the rig facts evaluated at the current r,z; "
        "joint1/3/5/7 unchanged; choose and author the values yourself",
    }


def _controller_approach_stall_instruction(
    instruction: str,
    stall: Mapping[str, object],
    *,
    ik_table: Mapping[str, object] | None = None,
    task: str | None = None,
    immediate_prior_receipt: Mapping[str, object] | None = None,
) -> str:
    marker = "\n\nAPPROACH_STALL_CONTEXT:\n"
    if marker in instruction:
        raise ProposalAuditFailedClosed(
            "controller instruction already contains approach-stall context"
        )
    packet = dict(stall)
    if ik_table is not None:
        packet["planar_ik_table"] = dict(ik_table)
    if toaster_stall_close_probe_due(task, stall, immediate_prior_receipt):
        guidance = (
            "\nThe last two open servos are physically blocked and stalled "
            "within 20 pixels "
            "of the toaster front pull. In the seed-7 geometry this places the "
            "door top edge between the fingers even though the grip-site pixel "
            "cannot descend to the pull. Test that alternate contact now: keep "
            "milestone=engage and author `move_joints` with only "
            "`targets={\"gripper\":0.0}`. Perform this stationary close now; "
            "do not repeat the servo, do not re-orient, and do not move the "
            "arm first. Contact belongs to the next sealed receipt. If it is "
            "confirmed, the first articulation command is a zero-depth "
            "downward image waypoint at most 16 pixels from the current "
            "grip-site pixel."
        )
    else:
        guidance = (
            "\nThe last two open servo commands toward this target each realized "
            "less than 6 mm of grip-site motion while the grip-site pixel stayed "
            "outside the 8 px close tolerance: the approach is physically blocked "
            "(the gripper body is against the appliance), not mislocalized. Do not "
            "repeat that servo; it is rejected. Change the approach geometry while "
            "keeping the gripper open: for a pull on a vertical face, come "
            "horizontally. Use the planar formulas to author `move_joints` for "
            "`joint2`, `joint4`, and `joint6` that keep the current radius and "
            "height but set the tool pitch `joint2 - joint4 - joint6` as far "
            "forward as the limits allow (about -1.0 to -1.2 rad; a full -pi/2 "
            "puts joint6 past its 3.75 rad limit at these radii), and `joint7` so "
            "the finger gap is vertical; or first author a bounded open "
            "`cartesian_delta` back and up, then re-approach from the front. "
            "`planar_ik_table` evaluates the rig formulas at the current radius "
            "and height for several pitches; pick a candidate marked within_limits "
            "with the most forward pitch and author every joint value."
        )
    return (
        instruction
        + marker
        + json.dumps(packet, sort_keys=True, separators=(",", ":"))
        + guidance
    )


def _cross_view_close_blocked(cross_view: Mapping[str, object] | None) -> bool:
    return bool(
        isinstance(cross_view, Mapping)
        and cross_view.get("horizontal_pull_detected") is True
        and cross_view.get("grip_site_on_horizontal_pull") is False
    )


def _controller_cross_view_instruction(
    instruction: str,
    cross_view: Mapping[str, object],
    *,
    selected_camera: str,
    close_blocked: bool,
) -> str:
    marker = "\n\nTOASTER_CROSS_VIEW_CONTEXT:\n"
    if marker in instruction:
        raise ProposalAuditFailedClosed(
            "controller instruction already contains cross-view context"
        )
    other = str(cross_view.get("other_camera"))
    packet = {**dict(cross_view), "selected_camera": selected_camera,
              "close_blocked": close_blocked}
    text = (
        f"\nThe public grip-site pixel in the {other} view is compared with the "
        f"dark horizontal pull detected there. Alignment in the {selected_camera} "
        "view fixes only two coordinates; the grip site must also lie on the pull "
        f"in the {other} view before any close."
    )
    if close_blocked:
        text += (
            " It does not: the grip site is at the wrong depth along the "
            f"{selected_camera} ray, so a stationary close is rejected. Keep "
            "milestone=engage and author `image_servo` with gripper `open`: "
            f"either the same {selected_camera} camera and pixel with a nonzero "
            "`depth_delta_m` (positive moves away from that camera) so the "
            f"{other}-view grip-site pixel moves toward the pull, or the {other} "
            "camera with a pixel on that pull. Qwen chooses the camera, pixel, "
            "depth, and step. Read "
            f"`end_effector_external_pixel_displacement.{other}` in the next "
            "receipt and flip the depth sign if the offset grew."
        )
    elif cross_view.get("grip_site_on_horizontal_pull") is True:
        text += " Both views agree; a stationary close is allowed once aligned."
    return (
        instruction
        + marker
        + json.dumps(packet, sort_keys=True, separators=(",", ":"))
        + text
    )


def _toaster_oven_handle_target_status(
    task: str,
    draft: Mapping[str, object],
    images: Mapping[str, bytes],
) -> dict[str, object] | None:
    if (
        task != "OpenToasterOvenDoor"
        or draft.get("kind") != "image_servo"
        or draft.get("target_role") != "fixture_handle"
    ):
        return None
    camera = draft.get("camera")
    if camera not in {"left", "right"} or not isinstance(
        images.get(str(camera)), bytes
    ):
        return None
    try:
        from PIL import Image

        opened = {
            name: Image.open(BytesIO(data)).convert("RGB")
            for name, data in images.items()
            if name in {"left", "right"} and isinstance(data, bytes)
        }
    except (ImportError, OSError, TypeError, ValueError, AttributeError):
        return None
    return _toaster_gate_from_images(draft, opened)


def _toaster_end_cap_violation(
    toaster_target: Mapping[str, object] | None,
    draft: Mapping[str, object],
) -> str | None:
    """Reject a first-insertion pixel within the end margin of the pull."""

    if (
        not isinstance(toaster_target, Mapping)
        or toaster_target.get("target_on_horizontal_pull") is not True
        or draft.get("gripper") != "open"
    ):
        return None
    box = toaster_target.get("pull_box")
    target = draft.get("target_pixel")
    if (
        not isinstance(box, Mapping)
        or not isinstance(target, Sequence)
        or isinstance(target, (str, bytes))
        or len(target) != 2
    ):
        return None
    try:
        u = float(target[0])
    except (TypeError, ValueError):
        return None
    u_min, u_max = int(box["u_min"]), int(box["u_max"])
    if u_max - u_min < 3 * TOASTER_HANDLE_END_MARGIN_PX:
        return None
    if u < u_min + TOASTER_HANDLE_END_MARGIN_PX or u > u_max - TOASTER_HANDLE_END_MARGIN_PX:
        return (
            "toaster oven target is on the end cap of the pull; choose the "
            f"interior midline with u between {u_min + TOASTER_HANDLE_END_MARGIN_PX} "
            f"and {u_max - TOASTER_HANDLE_END_MARGIN_PX} in the {draft.get('camera')} "
            f"view (bar u={u_min}..{u_max}, v={box['v_min']}..{box['v_max']})"
        )
    return None


def _toaster_gate_from_images(
    draft: Mapping[str, object],
    opened: Mapping[str, object],
) -> dict[str, object] | None:
    """Screen the selected pixel and, when given, the other-view pixel."""

    camera = str(draft.get("camera"))
    target = draft.get("target_pixel")
    image = opened.get(camera)
    if (
        image is None
        or not isinstance(target, Sequence)
        or isinstance(target, (str, bytes))
    ):
        return None
    status = _horizontal_dark_pull_target_status(image, target)
    if status is None:
        return None
    box = _pull_box(_horizontal_dark_pull_runs(image, target), near_pixel=target)
    result: dict[str, object] = {**status, "pull_box": box, "other_view": None}
    other_pixel = draft.get("other_view_pixel")
    other_name = "left" if camera == "right" else "right"
    other_image = opened.get(other_name)
    if (
        other_image is not None
        and isinstance(other_pixel, Sequence)
        and not isinstance(other_pixel, (str, bytes))
        and len(other_pixel) == 2
    ):
        other_status = _horizontal_dark_pull_target_status(other_image, other_pixel)
        if other_status is not None:
            result["other_view"] = {
                **other_status,
                "camera": other_name,
                "pull_box": _pull_box(
                    _horizontal_dark_pull_runs(other_image, other_pixel),
                    near_pixel=other_pixel,
                ),
            }
    return result


def _proposal_critic_instruction(
    context: CriticContext,
    *,
    task_instruction: str,
    observation_id: str,
    draft: Mapping[str, object],
    claimed_milestone: str,
    public_state: Mapping[str, object],
    images: Mapping[str, bytes],
    immediate_prior_receipt: Mapping[str, object] | None,
    critic_images: Mapping[str, bytes] | None = None,
    recent_receipts: Sequence[Mapping[str, object]] = (),
    cross_view: Mapping[str, object] | None = None,
    approach_stall: Mapping[str, object] | None = None,
    standing_target: Mapping[str, object] | None = None,
) -> str:
    stall_targets = draft.get("targets")
    qpos = public_state.get("state.arm_joint_position")
    joint4 = (
        qpos[3]
        if isinstance(qpos, Sequence)
        and not isinstance(qpos, (str, bytes))
        and len(qpos) == 7
        and not isinstance(qpos[3], bool)
        and isinstance(qpos[3], (int, float))
        else None
    )
    ready_pose_rule = (
        {
            "default_verdict": "approve",
            "effect_is_future_evidence": True,
            "gripper_must_remain_open": True,
            "arm_nearly_straight": True,
            "no_servo_executed_yet": True,
        }
        if claimed_milestone in {"observe", "approach"}
        and draft.get("kind") == "move_joints"
        and isinstance(stall_targets, Mapping)
        and any(str(key).startswith("joint") for key in stall_targets)
        and stall_targets.get("gripper", 1.0) == 1.0
        and joint4 is not None
        and float(joint4) > -1.0
        and not any(
            isinstance(item, Mapping) and item.get("kind") == "image_servo"
            for item in recent_receipts
        )
        else None
    )
    approach_servo_rule = (
        {
            "default_verdict": "approve",
            "effect_is_future_evidence": True,
            "gripper_open": True,
            "alignment_is_the_servo_goal_not_a_precondition": True,
            "zero_depth_is_still_lateral_motion": True,
            "stagnation_prohibited": True,
        }
        if claimed_milestone in {"approach", "pregrasp"}
        and draft.get("kind") == "image_servo"
        and draft.get("gripper") == "open"
        and draft.get("target_role") in {"fixture_handle", "source_object", "control_target"}
        else None
    )
    critic_contact_path_status = _coffee_contact_path_ready_status(
        standing_target, recent_receipts
    )
    contact_path_targets = draft.get("targets")
    coffee_contact_path_waypoint = bool(
        context.task == "StartCoffeeMachine"
        and claimed_milestone in {"engage", "actuate"}
        and draft.get("kind") == "move_joints"
        and draft.get("tracking_mode") == LAG_PAUSE_TRACKING
        and draft.get("note") in {
            COFFEE_CONTROL_CONTACT_PATH_NOTE,
            COFFEE_CONTROL_CONTACT_PRESS_NOTE,
        }
        and isinstance(contact_path_targets, Mapping)
        and set(contact_path_targets) == {*JOINT_NAMES, "gripper"}
        and contact_path_targets.get("gripper") == 1.0
    )
    control_alignment_rule = (
        {
            "default_verdict": "approve",
            "effect_is_future_evidence": True,
            "gripper_must_remain_open": True,
            "absolute_joint_contact_waypoint": True,
            "lag_pause_tracking": True,
        }
        if coffee_contact_path_waypoint
        else
        {
            "default_verdict": "approve",
            "effect_is_future_evidence": True,
            "gripper_must_remain_open": True,
            "standing_target_reissue": True,
        }
        if context.task == "StartCoffeeMachine"
        and claimed_milestone == "engage"
        and draft.get("kind") == "image_servo"
        and draft.get("target_role") == "control_target"
        and draft.get("gripper") == "open"
        and draft.get("note") == COFFEE_CONTROL_STANDING_CONVERGENCE_NOTE
        and _draft_matches_standing_target(draft, standing_target)
        else None
    )
    preclose_targets = draft.get("targets")
    control_preclose_rule = (
        {
            "default_verdict": "approve",
            "arm_motion_prohibited": True,
            "gripper_close_only": True,
            "button_contact_is_future_evidence": True,
            "press_occurs_on_next_observation": True,
        }
        if context.task == "StartCoffeeMachine"
        and claimed_milestone == "engage"
        and draft.get("kind") == "move_joints"
        and isinstance(preclose_targets, Mapping)
        and dict(preclose_targets) == {"gripper": 0.0}
        and _control_preclose_ready_status(immediate_prior_receipt) is not None
        else None
    )
    press_depth = draft.get("depth_delta_m")
    camera_ray_press = bool(
        draft.get("kind") == "image_servo"
        and draft.get("target_role") == "control_target"
        and not isinstance(press_depth, bool)
        and isinstance(press_depth, (int, float))
        and math.isfinite(float(press_depth))
        and float(press_depth) > 0.0
        and draft.get("gripper") == "close"
    )
    cartesian_press = bool(
        draft.get("kind") == "cartesian_delta"
        and draft.get("gripper") in {"hold", "close"}
    )
    prior_press_ready = _control_press_ready_status(immediate_prior_receipt)
    stereo_contact_correction = bool(
        context.task == "StartCoffeeMachine"
        and draft.get("kind") == "image_servo"
        and draft.get("target_role") == "control_target"
        and draft.get("other_view_pixel") is not None
        and not isinstance(press_depth, bool)
        and isinstance(press_depth, (int, float))
        and math.isfinite(float(press_depth))
        and float(press_depth) == 0.0
        and draft.get("gripper") == "open"
        and draft.get("note") == COFFEE_CONTROL_CONTACT_CORRECTION_NOTE
        and (
            _control_preclose_ready_status(immediate_prior_receipt) is not None
            or _coffee_contact_path_correction_ready_status(
                critic_contact_path_status
            )
            is not None
        )
    )
    coffee_press_retry = (
        _coffee_press_retry_ready_status(recent_receipts)
        if context.task == "StartCoffeeMachine"
        else None
    )
    control_press_rule = (
        {
            "default_verdict": "approve",
            "button_contact_is_future_evidence": True,
            "stereo_contact_correction": True,
        }
        if context.family == "control"
        and claimed_milestone == "actuate"
        and stereo_contact_correction
        else
        {
            "default_verdict": "approve",
            "effect_is_future_evidence": True,
            "button_is_pressed_not_grasped": True,
            "empty_close_is_expected": True,
            "camera_ray_press": camera_ray_press,
            **(
                {"prior_no_force_press_requires_continuation": True}
                if coffee_press_retry is not None
                else {}
            ),
        }
        if context.family == "control"
        and claimed_milestone == "actuate"
        and (cartesian_press or camera_ray_press)
        and isinstance(immediate_prior_receipt, Mapping)
        and (
            receipt_requested_gripper_intent(immediate_prior_receipt)
            in {"hold", "close"}
            or immediate_prior_receipt.get("kind") == "cartesian_delta"
            or prior_press_ready is not None
        )
        else None
    )
    coffee_retreat_status = _coffee_clearance_status(
        context, public_state, recent_receipts
    )
    retreat_targets = draft.get("targets")
    control_retreat_rule = (
        {
            "default_verdict": "approve",
            "effect_is_future_evidence": True,
            "button_press_already_observed": True,
            "current_milestone_already_verify_goal": True,
            "actual_relative_joint_retreat_is_required_clearance_motion": True,
            "current_retreat_distance_m": coffee_retreat_status[
                "retreat_distance_m"
            ],
            "required_clearance_m": COFFEE_POST_PRESS_CLEARANCE_M,
        }
        if context.task == "StartCoffeeMachine"
        and claimed_milestone == "verify_goal"
        and draft.get("kind") == "move_joints"
        and draft.get("tracking_mode") == ACTUAL_RELATIVE_TRACKING
        and isinstance(retreat_targets, Mapping)
        and set(retreat_targets) == {*JOINT_NAMES, "gripper"}
        and retreat_targets.get("gripper") == 1.0
        and coffee_retreat_status["press_effect_observed"] is True
        and coffee_retreat_status["clearance_reached"] is not True
        else None
    )
    stall_targets = draft.get("targets")
    toaster_stall_close_probe = bool(
        claimed_milestone == "engage"
        and toaster_stall_close_probe_due(
            context.task, approach_stall, immediate_prior_receipt
        )
        and draft.get("kind") == "move_joints"
        and isinstance(stall_targets, Mapping)
        and set(stall_targets) == {"gripper"}
        and stall_targets.get("gripper") == 0.0
    )
    approach_stall_rule = (
        {
            "default_verdict": "approve",
            "effect_is_future_evidence": True,
            "gripper_must_remain_open": not toaster_stall_close_probe,
            "blocked_approach_needs_new_geometry": (
                not toaster_stall_close_probe
            ),
            **(
                {
                    "toaster_stall_close_probe_allowed": True,
                    "stationary_close_required": True,
                    "contact_effect_is_future_evidence": True,
                }
                if toaster_stall_close_probe
                else {}
            ),
        }
        if isinstance(approach_stall, Mapping)
        and approach_stall.get("stalled") is True
        and claimed_milestone in {"approach", "engage"}
        and (
            (
                draft.get("kind") == "move_joints"
                and isinstance(stall_targets, Mapping)
                and stall_targets.get("gripper", 1.0) == 1.0
            )
            or (
                draft.get("kind") == "cartesian_delta"
                and draft.get("gripper") == "open"
            )
            or toaster_stall_close_probe
        )
        else None
    )
    cross_view_blocked = bool(
        isinstance(immediate_prior_receipt, Mapping)
        and articulated_handle_insertion_is_ready(immediate_prior_receipt)
        and _cross_view_close_blocked(cross_view)
    )
    cross_view_depth_servo_rule = (
        {
            "default_verdict": "approve",
            "effect_is_future_evidence": True,
            "gripper_must_remain_open": True,
            "grip_site_off_pull_in_other_view": True,
            "stationary_close_prohibited": True,
        }
        if claimed_milestone == "engage"
        and cross_view_blocked
        and draft.get("kind") == "image_servo"
        and draft.get("target_role") == "fixture_handle"
        and draft.get("gripper") == "open"
        else None
    )
    actuation_contact = _actuation_contact_status(
        claimed_milestone,
        immediate_prior_receipt,
    )
    wrist_roll_progress = _wrist_roll_progress_status(context, public_state)
    draft_image_servo_alignment = _draft_image_servo_alignment_status(
        draft,
        public_state,
    )
    wrist_orientation_rule = (
        {
            "quarter_turn_from_parallel_rejection": True,
            "orientation_only_rejection_prohibited": True,
            "default_verdict": "approve",
        }
        if claimed_milestone == "engage"
        and draft.get("kind") == "image_servo"
        and draft.get("target_role") == "fixture_handle"
        and draft.get("gripper") == "close"
        and isinstance(wrist_roll_progress, Mapping)
        and wrist_roll_progress.get("quarter_turn_reached") is True
        else None
    )
    binding_audit_rule = (
        {
            "visual_alignment_closed_by_contact": True,
            "visual_alignment_unverified_prohibited": True,
            "first_actuation_effect_is_future_evidence": True,
            "default_verdict": "approve",
        }
        if claimed_milestone == "actuate"
        and draft.get("kind") == "cartesian_delta"
        and draft.get("gripper") in {"hold", "close"}
        and isinstance(actuation_contact, Mapping)
        and actuation_contact.get("preserved") is True
        else None
    )
    articulation_motion_rule = (
        {
            "contact_preserved": True,
            "default_verdict": "approve",
            "effect_is_future_evidence": True,
            "surface_membership_not_required": True,
            "target_is_future_image_waypoint": True,
        }
        if claimed_milestone == "actuate"
        and draft.get("kind") == "image_servo"
        and draft.get("target_role") == "articulation_motion"
        and draft.get("gripper") in {"hold", "close"}
        and isinstance(actuation_contact, Mapping)
        and actuation_contact.get("preserved") is True
        else None
    )
    aligned_contact_execution_rule = (
        {
            "contact_effect_is_future_evidence": True,
            "current_within_close_tolerance": True,
            "default_verdict": "approve",
            "positive_contact_depth": True,
            "wrist_ambiguity_is_not_a_contradiction": True,
        }
        if claimed_milestone in {"engage", "grasp"}
        and draft.get("kind") == "image_servo"
        and draft.get("target_role") in {"source_object", "control_target"}
        and draft.get("gripper") == "close"
        and isinstance(draft_image_servo_alignment, Mapping)
        and draft_image_servo_alignment.get("within_close_tolerance") is True
        and isinstance(draft.get("depth_delta_m"), (int, float))
        and not isinstance(draft.get("depth_delta_m"), bool)
        and math.isfinite(float(cast(float, draft["depth_delta_m"])))
        and float(cast(float, draft["depth_delta_m"])) > 0.0
        else None
    )
    draft_targets = draft.get("targets")
    incomplete_roll = articulated_wrist_roll_tracking_status(
        immediate_prior_receipt
    )
    incomplete_wrist_roll_continuation_rule = (
        {
            "default_verdict": "approve",
            "effect_is_future_evidence": True,
            "gripper_must_remain_open": True,
            "repeats_qwen_authored_joint7_target": True,
            "tracking_not_settled": True,
        }
        if claimed_milestone == "engage"
        and draft.get("kind") == "move_joints"
        and isinstance(draft_targets, Mapping)
        and set(draft_targets) == {"gripper", "joint7"}
        and draft_targets.get("gripper") == 1.0
        and isinstance(incomplete_roll, Mapping)
        and incomplete_roll.get("settled") is False
        and draft_targets.get("joint7")
        == incomplete_roll.get("requested_joint7_rad")
        else None
    )
    open_handle_insertion_rule = (
        {
            "arm_motion_precedes_close": True,
            "default_verdict": "approve",
            "gripper_must_remain_open": True,
            "positive_depth_required": True,
        }
        if claimed_milestone == "engage"
        and draft.get("kind") == "image_servo"
        and draft.get("target_role") == "fixture_handle"
        and draft.get("gripper") == "open"
        and isinstance(draft.get("depth_delta_m"), (int, float))
        and not isinstance(draft.get("depth_delta_m"), bool)
        and math.isfinite(float(cast(float, draft["depth_delta_m"])))
        and float(cast(float, draft["depth_delta_m"])) > 0.0
        else None
    )
    stationary_handle_close_rule = (
        {
            "arm_motion_prohibited": True,
            "contact_effect_is_future_evidence": True,
            "default_verdict": "approve",
            "gripper_close_only": True,
        }
        if claimed_milestone == "engage"
        and draft.get("kind") == "move_joints"
        and isinstance(draft_targets, Mapping)
        and set(draft_targets) == {"gripper"}
        and draft_targets.get("gripper") == 0.0
        and isinstance(immediate_prior_receipt, Mapping)
        and articulated_handle_insertion_is_ready(immediate_prior_receipt)
        and not cross_view_blocked
        else None
    )
    failed_contact_target_count = len(
        articulated_failed_contact_targets(recent_receipts)
    )
    orientation_failure_count = 1 if context.task == "OpenToasterOvenDoor" else 2
    repeated_empty_handle_orientation_rule = (
        {
            "default_verdict": "approve",
            "effect_is_future_evidence": True,
            "gripper_must_remain_open": True,
            "joint7_roll_only": True,
            **(
                {"two_distinct_empty_targets": True}
                if failed_contact_target_count >= 2
                else {"single_visually_screened_toaster_target": True}
            ),
        }
        if claimed_milestone == "engage"
        and draft.get("kind") == "move_joints"
        and isinstance(draft_targets, Mapping)
        and set(draft_targets) == {"gripper", "joint7"}
        and draft_targets.get("gripper") == 1.0
        and failed_contact_target_count >= orientation_failure_count
        and isinstance(immediate_prior_receipt, Mapping)
        and immediate_prior_receipt.get("kind") == "cartesian_delta"
        and immediate_prior_receipt.get("requested_gripper") == "open"
        else None
    )
    recovery_view_status = _cartesian_view_gathering_status(
        draft,
        "approach",
        public_state,
    )
    articulated_recovery_view_gathering = (
        {
            "default_verdict": "approve",
            "effect_is_future_evidence": True,
            "empty_handle_target_history": True,
            "gripper_open": recovery_view_status["gripper_open"],
            "nonzero": recovery_view_status["nonzero"],
            "protocol_validated": recovery_view_status["protocol_validated"],
            "tracking_settled": recovery_view_status["tracking_settled"],
        }
        if claimed_milestone == "engage"
        and isinstance(recovery_view_status, Mapping)
        and articulated_failed_contact_targets(recent_receipts)
        and isinstance(immediate_prior_receipt, Mapping)
        and immediate_prior_receipt.get("requested_gripper") == "open"
        and draft.get("gripper") == "open"
        else None
    )
    prior_recovery_residual = (
        immediate_prior_receipt.get("gripper_residual")
        if isinstance(immediate_prior_receipt, Mapping)
        else None
    )
    prior_recovery_separation = (
        prior_recovery_residual.get("measured_end_finger_separation")
        if isinstance(prior_recovery_residual, Mapping)
        else None
    )
    articulated_empty_close_retreat_rule = (
        {
            "default_verdict": "approve",
            "effect_is_future_evidence": True,
            "empty_close_confirmed": True,
            "gripper_open": recovery_view_status["gripper_open"],
            "nonzero": recovery_view_status["nonzero"],
            "protocol_validated": recovery_view_status["protocol_validated"],
            "tracking_settled": recovery_view_status["tracking_settled"],
        }
        if claimed_milestone == "engage"
        and isinstance(recovery_view_status, Mapping)
        and articulated_failed_contact_targets(recent_receipts)
        and isinstance(immediate_prior_receipt, Mapping)
        and receipt_requested_gripper_intent(immediate_prior_receipt)
        in {"hold", "close"}
        and not isinstance(prior_recovery_separation, bool)
        and isinstance(prior_recovery_separation, (int, float))
        and math.isfinite(float(prior_recovery_separation))
        and float(prior_recovery_separation)
        <= ARTICULATED_CONTACT_MIN_SEPARATION
        and draft.get("gripper") == "open"
        else None
    )
    contact_confirmation_rule = (
        {
            "arm_motion_prohibited": True,
            "contact_persistence_is_future_evidence": True,
            "default_verdict": "approve",
            "gripper_close_only": True,
        }
        if claimed_milestone == "actuate"
        and draft.get("kind") == "move_joints"
        and isinstance(draft_targets, Mapping)
        and set(draft_targets) == {"gripper"}
        and draft_targets.get("gripper") == 0.0
        and isinstance(immediate_prior_receipt, Mapping)
        and (
            (
                immediate_prior_receipt.get("kind") == "image_servo"
                and immediate_prior_receipt.get("requested_gripper") == "close"
            )
            or (
                immediate_prior_receipt.get("kind") == "move_joints"
                and receipt_requested_gripper_intent(immediate_prior_receipt)
                == "close"
            )
        )
        and isinstance(actuation_contact, Mapping)
        and actuation_contact.get("preserved") is True
        else None
    )
    value = {
        "schema": "robocasa-qwen-proposal-audit-request/v14",
        "task": context.task,
        "task_instruction": task_instruction,
        "family": context.family,
        "fresh_observation_id": observation_id,
        "fresh_public_state": public_state,
        "fresh_public_state_sha256": strict_canonical_sha256(public_state),
        "fresh_public_rgb_sha256": image_hashes(images),
        "critic_input_rgb_sha256": image_hashes(critic_images or images),
        "image_servo_target_marker": (
            {
                "camera": draft.get("camera"),
                "target_pixel": draft.get("target_pixel"),
                "color": "magenta",
                "meaning": "controller-authored target; not a task object",
            }
            if draft.get("kind") == "image_servo"
            else None
        ),
        "immediate_prior_receipt": immediate_prior_receipt,
        "immediate_prior_receipt_sha256": (
            strict_canonical_sha256(immediate_prior_receipt)
            if immediate_prior_receipt is not None
            else None
        ),
        "image_servo_pixel_progress": (
            _image_servo_pixel_progress(draft, immediate_prior_receipt)
            if immediate_prior_receipt is not None
            else None
        ),
        "image_servo_alignment": (
            _image_servo_alignment_status([dict(immediate_prior_receipt)])
            if immediate_prior_receipt is not None
            else None
        ),
        "actuation_revision": _actuation_revision_status(
            context,
            draft,
            claimed_milestone,
        ),
        "wrist_roll_revision": _wrist_roll_revision_status(
            context,
            draft,
            claimed_milestone,
            public_state,
        ),
        "wrist_roll_progress": wrist_roll_progress,
        "wrist_orientation_rule": wrist_orientation_rule,
        "draft_image_servo_alignment": draft_image_servo_alignment,
        "image_servo_close_rule": (
            {
                "close_pixel_tolerance_px": IMAGE_SERVO_CLOSE_TOLERANCE_PX,
                "close_requires_within_tolerance": True,
                "current_within_tolerance": False,
                "default_verdict": "revise",
                "required_contradiction": "visual_alignment_unverified",
                "required_correction": "revise_alignment",
                "required_evidence": ["external_rgb", "jacobian_projection"],
            }
            if claimed_milestone in {"engage", "grasp"}
            and draft.get("kind") == "image_servo"
            and draft.get("gripper") == "close"
            and isinstance(draft_image_servo_alignment, Mapping)
            and draft_image_servo_alignment.get("within_close_tolerance") is False
            else None
        ),
        "image_servo_open_approach_rule": (
            {
                "alignment_effect_is_future": True,
                "current_outside_close_tolerance": True,
                "default_verdict": "approve",
                "far_current_pixel_is_reason_to_execute": True,
                "zero_depth_can_still_have_pixel_alignment_effect": True,
            }
            if claimed_milestone == "approach"
            and draft.get("kind") == "image_servo"
            and draft.get("gripper") == "open"
            and isinstance(draft_image_servo_alignment, Mapping)
            and draft_image_servo_alignment.get("within_close_tolerance") is False
            else None
        ),
        "handle_orientation_calibration_rule": None,
        "image_servo_contact_depth_rule": (
            {
                "contact_requires_positive_depth": True,
                "current_depth_delta_m": float(cast(float, draft["depth_delta_m"])),
                "default_verdict": "revise",
                "negative_depth_is_camera_retreat": True,
                "required_contradiction": "motion_effect_mismatch",
                "required_correction": "revise_approach",
            }
            if claimed_milestone in {"engage", "grasp"}
            and draft.get("kind") == "image_servo"
            and (
                (
                    draft.get("target_role") == "fixture_handle"
                    and draft.get("gripper") == "open"
                )
                or (
                    draft.get("target_role")
                    in {"source_object", "control_target"}
                    and draft.get("gripper") == "close"
                )
            )
            and isinstance(draft.get("depth_delta_m"), (int, float))
            and not isinstance(draft.get("depth_delta_m"), bool)
            and math.isfinite(float(cast(float, draft["depth_delta_m"])))
            and float(cast(float, draft["depth_delta_m"])) <= 0.0
            else None
        ),
        "actuation_contact": actuation_contact,
        "binding_audit_rule": binding_audit_rule,
        "articulation_motion_rule": articulation_motion_rule,
        "aligned_contact_execution_rule": aligned_contact_execution_rule,
        "open_handle_insertion_rule": open_handle_insertion_rule,
        "stationary_handle_close_rule": stationary_handle_close_rule,
        "toaster_cross_view": dict(cross_view) if cross_view is not None else None,
        "cross_view_depth_servo_rule": cross_view_depth_servo_rule,
        "approach_stall": dict(approach_stall) if approach_stall is not None else None,
        "approach_stall_rule": approach_stall_rule,
        "ready_pose_rule": ready_pose_rule,
        "control_alignment_rule": control_alignment_rule,
        "control_preclose_rule": control_preclose_rule,
        "control_press_rule": control_press_rule,
        "control_retreat_rule": control_retreat_rule,
        "control_finish_rule": (
            {
                "default_verdict": "approve",
                "measured_press_and_clearance_complete": True,
                "official_success_requires_terminal_evaluation": True,
            }
            if draft.get("kind") == "finish"
            and claimed_milestone == "verify_goal"
            and _coffee_measured_finish_ready(context, coffee_retreat_status)
            else None
        ),
        "approach_servo_rule": approach_servo_rule,
        "standing_stereo_target": (
            dict(standing_target) if standing_target is not None else None
        ),
        "standing_target_reissue": _draft_matches_standing_target(
            draft, standing_target
        ),
        "repeated_empty_handle_orientation_rule": (
            repeated_empty_handle_orientation_rule
        ),
        "incomplete_wrist_roll_continuation_rule": (
            incomplete_wrist_roll_continuation_rule
        ),
        "articulated_recovery_view_gathering": (
            articulated_recovery_view_gathering
        ),
        "articulated_empty_close_retreat_rule": (
            articulated_empty_close_retreat_rule
        ),
        "contact_confirmation_rule": contact_confirmation_rule,
        "cartesian_view_gathering": _cartesian_view_gathering_status(
            draft,
            claimed_milestone,
            public_state,
        ),
        "contact_recovery": articulated_contact_recovery_status(
            claimed_milestone,
            draft,
            immediate_prior_receipt,
            recent_receipts,
        ),
        "draft": draft,
        "draft_sha256": strict_canonical_sha256(draft),
        "claimed_milestone": claimed_milestone,
        "milestone_history": list(context.milestone_history),
        "milestone_history_sha256": strict_canonical_sha256(
            context.milestone_history
        ),
        "constraints": {
            "pre_execution": True,
            "qualitative_closed_enums_only": True,
            "qwen_controller_is_sole_numeric_author": True,
            "critic_has_no_command_or_motor_authority": True,
            "approved_draft_must_be_returned_unchanged": True,
        },
    }
    value["image_servo_alignment"] = _source_contact_alignment_meaning(
        context, value["image_servo_alignment"]
    )
    source_recovery = (
        source_grasp_recovery_target(recent_receipts)
        if context.family == "grasp_place" else None
    )
    if source_recovery is not None:
        open_source_reapproach = bool(
            claimed_milestone == "grasp"
            and draft.get("kind") == "image_servo"
            and draft.get("target_role") == "source_object"
            and draft.get("gripper") == "open"
        )
        value["source_grasp_recovery"] = {
            **source_recovery,
            "absolute_empty_contact_max_separation_m": SOURCE_CONTACT_MIN_SEPARATION,
            "empty_close_confirmed": True,
            "failed_contact_persists_through_open_motion": True,
            "same_point_with_changed_depth_or_stereo_allowed": True,
            "open_corrective_source_servo": open_source_reapproach,
            "alignment_is_future_effect_for_open_reapproach": open_source_reapproach,
            "source_identity_must_remain_visually_grounded": True,
            "instruction": (
                "Use this source-specific threshold and retry rule for the sealed "
                "failed contact. Its measured separation is empty, not evidence of "
                "a held source, even after an open retreat or reapproach. Before "
                "re-closing require a changed source point, changed nonzero depth, "
                "or newly grounded stereo pair. A correctly localized source point "
                "may stay unchanged when depth or stereo changes. An open corrective "
                "source servo at grasp is reapproach: alignment is its future effect, "
                "so current close alignment is not required. The selected point must "
                "still visibly belong to the movable source; reject a wrong or "
                "unverified source identity. Approve a bounded "
                "open recovery or changed-geometry retry when no separate visual, "
                "motion, safety, or milestone contradiction applies. A stationary "
                "re-close uses the preceding executed source approach geometry; "
                "a changed note alone cannot move the gripper. Return only the "
                "qualitative audit; Qwen remains the sole motor author."
            ),
        }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _proposal_attempt_record_sha256(
    context: CriticContext,
    *,
    role: str,
    call_index: int,
) -> str:
    matches = [
        record
        for record in context.attempt_records
        if record.get("role") == role and record.get("call_index") == call_index
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("record_sha256"), str):
        raise _AttemptEvidenceLinkageIncomplete(
            f"{role} AttemptEvidenceLog record is not bijective"
        )
    return str(matches[0]["record_sha256"])


def _proposal_base_record(
    context: CriticContext,
    *,
    observation_id: str,
    revision_index: int,
    draft: object,
    controller_call_index: int,
    controller_request_sha256: str | None,
    controller_call_manifest_sha256: str,
    controller_attempt_record_sha256: str | None,
    immediate_prior_receipt: Mapping[str, object] | None,
) -> dict[str, object]:
    draft_snapshot = canonical_json_copy(draft) if draft is not None else None
    command_kind = (
        draft_snapshot.get("kind")
        if isinstance(draft_snapshot, Mapping)
        else "malformed"
    )
    return {
        "schema": "robocasa-qwen-proposal-evidence/v1",
        "task": context.task,
        "family": context.family,
        "observation_id": observation_id,
        "proposal_index": len(context.proposal_records) + 1,
        "revision_index": revision_index,
        "fresh_public_state_sha256": context.proposal_public_state_sha256,
        "fresh_public_rgb_sha256": context.proposal_image_sha256,
        "immediate_prior_receipt_sha256": (
            strict_canonical_sha256(immediate_prior_receipt)
            if immediate_prior_receipt is not None
            else None
        ),
        "milestone_history_before": list(context.milestone_history),
        "milestone_history_sha256": strict_canonical_sha256(
            context.milestone_history
        ),
        "claimed_milestone": None,
        "command_kind": command_kind,
        "draft": draft_snapshot,
        "draft_sha256": (
            strict_canonical_sha256(draft_snapshot)
            if draft_snapshot is not None
            else None
        ),
        "controller_call_index": controller_call_index,
        "controller_request_sha256": controller_request_sha256,
        "controller_call_manifest_sha256": controller_call_manifest_sha256,
        "controller_attempt_record_sha256": controller_attempt_record_sha256,
        "critic_call_index": None,
        "critic_request_sha256": None,
        "critic_call_manifest_sha256": None,
        "critic_attempt_record_sha256": None,
        "audit": None,
        "audit_sha256": None,
        "status": None,
        "contradiction": None,
        "suggested_correction": None,
        "returned_command_sha256": None,
        "executed": False,
        "mailbox_count": 0,
        "action_count": 0,
        "receipt_count": 0,
        "execution_receipt_sha256": None,
        "mailbox_sha256": None,
        "terminal_outcome": None,
        "terminal_outcome_sha256": None,
        "milestone_history_after": list(context.milestone_history),
        "milestone_history_after_sha256": strict_canonical_sha256(
            context.milestone_history
        ),
        "milestone_closed": False,
        "critic_origin_execution": False,
        "failure_class": None,
        "failure_sha256": None,
        "record_sha256": None,
    }


def _seal_evidence_record(record: dict[str, object]) -> None:
    value = {key: item for key, item in record.items() if key != "record_sha256"}
    record["record_sha256"] = strict_canonical_sha256(value)


def _is_malformed_response(error: Exception) -> bool:
    return type(error).__name__.casefold().endswith("malformedresponse")


def _consume_proposal_revision(context: CriticContext, observation_id: str) -> None:
    if context.proposal_revisions_used >= PROPOSAL_MAX_REVISIONS_PER_OBSERVATION:
        context.proposal_failed_observations.add(observation_id)
        raise ProposalAuditExhausted(
            f"initial proposal and {PROPOSAL_MAX_REVISIONS_PER_OBSERVATION} revisions were rejected"
        )
    context.proposal_revisions_used += 1


def _close_proposal_milestone(
    context: CriticContext,
    record: dict[str, object],
) -> None:
    before = record.get("milestone_history_before")
    milestone = record.get("claimed_milestone")
    if not isinstance(before, list) or before != context.milestone_history:
        raise ValueError("proposal milestone history linkage drifted")
    if not isinstance(milestone, str):
        raise ValueError("approved proposal lacks a closed milestone")
    context.milestone_history.append(milestone)
    record.update({
        "milestone_history_after": list(context.milestone_history),
        "milestone_history_after_sha256": strict_canonical_sha256(
            context.milestone_history
        ),
        "milestone_closed": True,
    })


def _protocol_rejection(
    record: dict[str, object],
    *,
    error: Exception,
) -> dict[str, object]:
    message = str(error).casefold()
    evidence = ["milestone_history"]
    if "unchanged rejected draft" in message or "requires a changed motor command" in message:
        contradiction = "stagnation"
        correction = "revise_approach"
        evidence = []
    elif "terminal" in message or "verify_goal" in message:
        contradiction = "terminal_unverified"
        correction = "verify_goal"
    elif "stale" in message or "observation" in message:
        contradiction = "stale_or_missing_evidence"
        correction = "observe"
    elif "zero cartesian" in message or "no gripper transition" in message:
        contradiction = "stagnation"
        correction = "revise_approach"
    elif "empty-grasp retry" in message:
        contradiction = "visual_alignment_unverified"
        correction = "revise_alignment"
    elif (
        "outside the visible narrow horizontal handle" in message
        or "end cap of the pull" in message
    ):
        contradiction = "visual_alignment_unverified"
        correction = "revise_alignment"
        evidence = ["external_rgb"]
    elif "coffee machine" in message and "distinct visible button" in message:
        contradiction = "visual_alignment_unverified"
        correction = "revise_alignment"
        evidence = ["external_rgb"]
    elif "other external view" in message:
        contradiction = "visual_alignment_unverified"
        correction = "revise_alignment"
        evidence = ["external_rgb", "jacobian_projection"]
    elif "stereo" in message or "other-view pixel" in message:
        contradiction = "visual_alignment_unverified"
        correction = "revise_alignment"
        evidence = ["external_rgb", "jacobian_projection"]
    elif "approach stalled" in message:
        contradiction = "stagnation"
        correction = "revise_approach"
        evidence = ["action_receipt", "jacobian_projection", "external_rgb"]
    elif "control press produced no effect" in message:
        contradiction = "actuation_unverified"
        correction = "verify_actuation"
        evidence = ["action_receipt", "external_rgb"]
    elif "continue incomplete joint7 orientation" in message:
        contradiction = "tracking_not_settled"
        correction = "wait_for_settle"
        evidence = ["joint_tracking", "joint_velocity", "action_receipt"]
    elif "first qwen-authored retreat endpoint" in message:
        contradiction = "tracking_not_settled"
        correction = "wait_for_settle"
        evidence = ["joint_tracking", "action_receipt"]
    elif "joint7 orientation recovery" in message:
        contradiction = "visual_alignment_unverified"
        correction = "revise_alignment"
        evidence = ["wrist_rgb", "external_rgb", "action_receipt"]
    elif "actuation contact recovery target must move" in message:
        contradiction = "actuation_unverified"
        correction = "verify_actuation"
        evidence = ["gripper_state", "action_receipt", "external_rgb"]
    elif "target must move at least one grid cell" in message:
        contradiction = "visual_alignment_unverified"
        correction = "revise_alignment"
        evidence = ["external_rgb", "action_receipt"]
    elif (
        "empty handle engagement" in message
        or "articulated engage requires" in message
        or "stationary gripper close" in message
    ):
        contradiction = "grasp_unverified"
        correction = "verify_gripper"
        evidence = ["gripper_state", "action_receipt", "external_rgb"]
    elif "release milestone requires" in message:
        contradiction = "release_unverified"
        correction = "verify_release"
    elif "transport milestone requires" in message:
        contradiction = "grasp_unverified"
        correction = "verify_gripper"
    elif (
        "actuation lost contact" in message
        or "articulated actuation requires sealed contact" in message
        or "actuation contact target exhausted" in message
        or "actuation contact recovery" in message
        or "actuation requires re-engagement" in message
        or "actuation must change motion axes" in message
        or "handle contact already established" in message
        or "articulation-motion image servo requires sealed articulated contact"
        in message
        or "stationary contact confirmation" in message
    ):
        contradiction = "actuation_unverified"
        correction = "verify_actuation"
        evidence = ["gripper_state", "action_receipt"]
    elif "safe" in message or "limit" in message or "bound" in message:
        contradiction = "safety_bound_risk"
        correction = "revise_within_bounds"
    else:
        contradiction = "milestone_order_violation"
        correction = "continue_milestone"
    record.update({
        "status": "rejected_by_protocol",
        "contradiction": contradiction,
        "suggested_correction": correction,
        "failure_class": type(error).__name__,
        "failure_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
    })
    return {
        "contradiction": contradiction,
        "evidence": evidence,
        "suggested_correction": correction,
        "confidence": "high",
    }


def _begin_proposal_observation(
    context: CriticContext,
    *,
    observation_id: str,
    public_state: Mapping[str, object],
    images: Mapping[str, bytes],
    receipts: object,
    rgb_change: object,
) -> Mapping[str, object] | None:
    if set(images) != {"left", "right", "wrist"} or any(
        not isinstance(value, bytes) for value in images.values()
    ):
        raise ProposalAuditFailedClosed("proposal audit requires exact three RGB views")
    state_sha256 = strict_canonical_sha256(public_state)
    current_image_sha256 = image_hashes(images)
    immediate_prior_receipt = (
        cast(Mapping[str, object], canonical_json_copy(receipts[-1]))
        if isinstance(receipts, list)
        and receipts
        and isinstance(receipts[-1], Mapping)
        else None
    )
    receipt_sha256 = (
        strict_canonical_sha256(immediate_prior_receipt)
        if immediate_prior_receipt is not None
        else None
    )
    if observation_id in context.proposal_failed_observations:
        raise ProposalAuditExhausted("proposal revision budget already exhausted")
    if context.proposal_observation_id == observation_id:
        if context.proposal_pending_record_index is not None:
            raise ProposalAuditFailedClosed(
                "approved command cannot be reissued before an execution receipt"
            )
        if (
            context.proposal_public_state_sha256 != state_sha256
            or context.proposal_image_sha256 != current_image_sha256
            or context.proposal_prior_receipt_sha256 != receipt_sha256
        ):
            raise ProposalAuditFailedClosed(
                "same-observation proposal inputs changed during revision"
            )
        return immediate_prior_receipt
    if context.proposal_pending_record_index is not None:
        pending = context.proposal_pending_command
        reason, sealed_receipt = select_sealed_public_receipt(
            pending,
            receipts,
            public_state,
            rgb_change=rgb_change,
        )
        if sealed_receipt is None or reason != "eligible":
            raise ProposalAuditFailedClosed(
                f"approved proposal lacks a matching execution receipt: {reason}"
            )
        if immediate_prior_receipt is None or strict_canonical_sha256(
            sealed_receipt
        ) != strict_canonical_sha256(immediate_prior_receipt):
            raise ProposalAuditFailedClosed(
                "approved proposal receipt is not the immediate prior receipt"
            )
        record = context.proposal_records[context.proposal_pending_record_index]
        record.update({
            "executed": True,
            "mailbox_count": 1,
            "action_count": 1,
            "receipt_count": 1,
            "execution_receipt_sha256": strict_canonical_sha256(sealed_receipt),
        })
        _close_proposal_milestone(context, record)
        _capture_coffee_press_origin(context, public_state, sealed_receipt)
        _capture_coffee_retreat_endpoint(context, pending)
        _seal_evidence_record(record)
        context.proposal_pending_record_index = None
        context.proposal_pending_command = None
        context.proposal_pending_milestone = None
        context.proposal_revisions_used = 0
    context.proposal_observation_id = observation_id
    context.proposal_public_state_sha256 = state_sha256
    context.proposal_image_sha256 = current_image_sha256
    context.proposal_prior_receipt_sha256 = receipt_sha256
    return immediate_prior_receipt


def _proposal_audit_validation_context(kwargs: Mapping[str, object]) -> dict[str, object]:
    value = kwargs.get("proposal_audit_context")
    if not isinstance(value, Mapping) or set(value) != {
        "current_qpos",
        "current_gripper",
        "remaining_actions",
        "sequence",
        "camera_calibration",
    }:
        raise ProposalAuditFailedClosed("proposal execution validation context drifted")
    return dict(value)



def _source_landmark_geometry(selections, public_state, camera_calibration):
    from .image_servo import (
        MAX_STEREO_RAY_GAP_M,
        _camera_geometry,
        _closest_ray_points,
        _pixel_ray_world,
        _transpose_matvec,
    )
    valid = [item for item in selections if item['qwen_observation']['visible'] and item['selected_pixel'] is not None]
    if len(valid) != 2:
        return None
    a, b = valid
    cameras = [_camera_geometry(camera_calibration[item['camera']]) for item in (a, b)]
    pixels = [list(item['selected_pixel']) for item in (a, b)]
    oa, da = _pixel_ray_world(tuple(pixels[0]), cameras[0])
    ob, db = _pixel_ray_world(tuple(pixels[1]), cameras[1])
    try:
        pa, pb, gap, ta, tb = _closest_ray_points(oa, da, ob, db)
    except ValueError:
        return None
    if gap > MAX_STEREO_RAY_GAP_M or ta <= 0 or tb <= 0:
        return None
    world = [(x + y) / 2 for x, y in zip(pa, pb)]
    # A small ray gap alone admits unstable depth when rays are nearly parallel.
    # Check the sensitivity to one pixel at the published image resolution.
    for camera_index in range(2):
        for axis in range(2):
            for offset in (-1, 1):
                shifted = [list(pixel) for pixel in pixels]
                shifted[camera_index][axis] += offset
                origin_a, direction_a = _pixel_ray_world(tuple(shifted[0]), cameras[0])
                origin_b, direction_b = _pixel_ray_world(tuple(shifted[1]), cameras[1])
                try:
                    point_a, point_b, _, along_a, along_b = _closest_ray_points(origin_a, direction_a, origin_b, direction_b)
                except ValueError:
                    return None
                midpoint = [(x + y) / 2 for x, y in zip(point_a, point_b)]
                if along_a <= 0 or along_b <= 0 or sum((x - y) ** 2 for x, y in zip(midpoint, world)) > MAX_STEREO_RAY_GAP_M ** 2:
                    return None
    base = _transpose_matvec(_rotation_xyzw(public_state['state.base_rotation']),
                            [x - y for x, y in zip(world, public_state['state.base_position'])])
    relative = [point - eef for point, eef in zip(base, public_state['state.end_effector_position_relative'], strict=True)]
    return {
        'landmark_minus_end_effector_robot_base_m': relative,
        'end_effector_to_landmark_distance_m': math.sqrt(sum(value * value for value in relative)),
        'camera_pair': [a['camera'], b['camera']], 'ray_gap_m': gap,
        'estimated_landmark_position_robot_base_m': list(base),
        'current_end_effector_position_robot_base_m': public_state['state.end_effector_position_relative'],
        'interpretation': 'The two Qwen-selected image rays give this geometric estimate using the current public calibration. A small ray gap does not prove matching object features or contact. The relative vector is target minus current EEF in robot-base axes and its distance is the full three-dimensional norm, not one coordinate difference. These are geometric measurements, not a command or step size. Existing cartesian_delta translation uses robot-base axes; Qwen can use the current point and EEF position to choose a bounded open approach or an observation-gathering motion. Inspect collision clearance in fresh RGB and verify the next receipt. The action schema, review process and motion bounds are unchanged.',
    }


def _source_landmark_images(raw_bytes, center):
    import io

    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    raw = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
    corners = cv2.goodFeaturesToTrack(cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE),
                                   maxCorners=64, qualityLevel=.02, minDistance=8, blockSize=5)
    xy = [] if corners is None else sorted([(int(round(float(c[0][0]))), int(round(float(c[0][1])))) for c in corners], key=lambda v: (v[1], v[0]))
    points = [{'id': i, 'pixel': [x, y]} for i, (x, y) in enumerate(xy)]
    marked = raw.resize((768, 768)); draw = ImageDraw.Draw(marked); font = ImageFont.load_default(size=15)
    for point in points:
        x, y = [v * 3 for v in point['pixel']]
        draw.ellipse((x-3, y-3, x+3, y+3), fill=(255,0,80), outline='white', width=1)
        draw.text((min(740, x+4), max(0, y-18)), str(point['id']), font=font,
                  fill=(255,255,0), stroke_width=2, stroke_fill=(0,0,0))
    x0 = max(0, min(128, int(round(center[0])) - 64))
    y0 = max(0, min(128, int(round(center[1])) - 64))
    box = [x0, y0, x0+128, y0+128]
    crop = marked.crop(tuple(v * 3 for v in box)).resize((768,768))
    images = {}
    for slot, value in [('left',raw),('right',marked),('wrist',crop)]:
        stream = io.BytesIO(); value.save(stream, format='PNG'); images[slot] = stream.getvalue()
    return images, points, box


def _source_perception_call(context, client, *, stage, kwargs, details=None):
    if context.source_perception_deadline is not None and time.monotonic() + 15 >= context.source_perception_deadline:
        return None
    if context.source_perception_deadline is not None:
        _set_timeout(client, min(30.0, context.source_perception_deadline - time.monotonic() - 10))
    index = len(context.source_perception_records) + 1
    folder = Path(context.run) / 'source-perception' / f'call-{index:04d}-{stage}'
    folder.mkdir(parents=True, exist_ok=False)
    paths = {}
    for camera, pixels in kwargs['images'].items():
        path = folder / f'{camera}.png'; path.write_bytes(pixels); paths[camera] = str(path)
    source = {k: v for k, v in kwargs.items() if k != 'images'}
    source['images'] = paths
    record = {'call_index':index, 'stage':stage, 'observation_id':kwargs['observation_id'],
              'source_public_rgb_sha256': context.proposal_image_sha256,
              'input':source, 'input_image_sha256':image_hashes(kwargs['images']),
              'details':details, 'status':'started', 'attempt_records':[], 'response':None,
              'evidence':None, 'error':None}
    context.source_perception_records.append(record)
    context.model_calls += 1
    call, log = _with_captured_attempt_log(kwargs)
    try:
        response = client.complete(**call)
        request, linkage = _request_linkage_from_attempt_log(log, observation_id=kwargs['observation_id'],
                                                            attempt_index=0, response_schema=kwargs['response_schema'])
        if linkage != 'complete':
            raise ValueError('source perception lacks a linked model attempt')
        record.update(response=canonical_json_copy(response.command), evidence=canonical_json_copy(response.evidence),
                      request_sha256=request, status='complete')
    except Exception as error:
        record.update(status='unavailable', error={'type':type(error).__name__, 'message':str(error)[:500]})
    finally:
        record['attempt_records'] = canonical_json_copy(log.records)
        _atomic_json(folder / 'result.json', record)
    return record['response'] if record['status'] == 'complete' else None


def _source_perception_packet(context, client, *, observation_id, task_instruction, images, public_state, camera_calibration):
    from itertools import combinations

    point = {'anyOf':[{'type':'array','items':{'type':'number','minimum':0,'maximum':1000},'minItems':2,'maxItems':2},{'type':'null'}]}
    view = {'type':'object','properties':{'visible':{'type':'boolean'},'description':{'type':'string'},'point_2d':point},
            'required':['visible','description','point_2d'],'additionalProperties':False}
    schema = {'type':'object','properties':{'target':{'type':'string'}, **{v:view for v in ['left','right','wrist']}},
              'required':['target','left','right','wrist'],'additionalProperties':False}
    coarse = _source_perception_call(context, client, stage='coarse', kwargs={
        'observation_id':observation_id, 'system_prompt':'Locate objects in the supplied images. Return the requested JSON.',
        'instruction':f'Task: {task_instruction}\nIdentify the movable source object to pick up and return its name in target. Images are ordered left camera, right camera, wrist camera. For each view, describe its appearance and give its visible center. All points must refer to that same source object. If it is not visible, set visible false and point_2d null. Use relative coordinates 0 through 1000 on each image; top-left [0,0], bottom-right [1000,1000].',
        'public_state':{}, 'images':images, 'response_schema':schema, 'max_tokens':512})
    if coarse is None:
        return None
    visible = [v for v in ['left','right','wrist'] if coarse[v]['visible'] and coarse[v]['point_2d'] is not None]
    # Coarse visibility is a search preference, not a veto on fresh image evidence.
    selected_views = ['wrist'] + [v for v in visible if v != 'wrist'] + [v for v in ['left','right'] if v not in visible]
    selections = []
    geometry = None
    for view_name in selected_views:
        coarse_point = coarse[view_name]['point_2d']
        center = [float(v) * 256 / 1000 for v in coarse_point] if coarse_point is not None else [128.,128.]
        marked, candidates, crop = _source_landmark_images(images[view_name], center)
        schema = {'type':'object','properties':{'visible':{'type':'boolean'},'description':{'type':'string'},
                  'keypoint_id':{'anyOf':[{'type':'integer','minimum':0,'maximum':63},{'type':'null'}]}},
                  'required':['visible','description','keypoint_id'],'additionalProperties':False}
        selection = _source_perception_call(context, client, stage=view_name, details={'candidate_points':candidates,'crop_center_from_qwen':center if coarse_point is not None else None,'crop_center_source':'qwen' if coarse_point is not None else 'image_center_fallback','crop_box':crop}, kwargs={
            'observation_id':observation_id,'system_prompt':'Locate objects in the supplied images. Return the requested JSON.',
            'instruction':f'Locate the {coarse["target"]} in the {view_name} camera. All three supplied images come from this SAME camera: first the original image, second the full numbered image, third an enlarged local crop of the numbered image. The pink dots mark candidate pixels and the adjacent yellow text gives their integer IDs. Choose a dot ON the visible target object, and describe the precise feature under that dot. Read its printed ID, not an imagined coordinate. The same dot keeps its ID in the full image and the crop. If the object is not visible, return visible false and keypoint_id null. If it is visible but no dot lies on it, return visible true and keypoint_id null. The robot and gripper are not the target object.\nAn earlier coarse query of this same observation described a candidate as: {coarse[view_name]["description"]}\nUse that description only to help find the candidate; independently verify it against the images shown here. It may be wrong about visibility or position. Do not select a robot part or background merely because the description claims the object is visible.',
            'public_state':{},'images':marked,'image_roles':{'left':view_name+'_original','right':view_name+'_numbered_full','wrist':view_name+'_numbered_local_crop'},
            'response_schema':schema,'max_tokens':512})
        if selection is None:
            continue
        index = selection['keypoint_id']
        pixel = next((p['pixel'] for p in candidates if p['id'] == index), None)
        selections.append({'camera':view_name,'qwen_observation':selection,'selected_pixel':pixel})
        valid = [item for item in selections if item['qwen_observation']['visible'] and item['selected_pixel'] is not None]
        for pair in combinations(valid, 2):
            geometry = _source_landmark_geometry(pair, public_state, camera_calibration)
            if geometry is not None:
                break
        if geometry is not None:
            break
    if geometry is None:
        return None
    return {
        'observation_id':observation_id,
        'source':'The same frozen Qwen selected numbered image landmarks in separate focused camera queries from this exact public observation.',
        'target':coarse['target'],
        'coordinate_convention':'Original 256x256 camera pixels, u right and v down. Each selected_pixel is the public image location of the landmark ID chosen by Qwen.',
        'observation':[item for item in selections if item['camera'] in geometry['camera_pair']],
        'interpretation':'These are fallible visual estimates, not actions or approvals. Inspect fresh RGB and compare the feature descriptions. A null selection supplies no target. For image_servo source_object, camera remains left or right. To pair it with the wrist observation, explicitly set other_view_camera to wrist and other_view_pixel to the Qwen-selected wrist pixel. Omitted or null other_view_camera still means the opposite external camera; a wrist pixel is never an external-camera pixel. Check that the end effector is visible in the selected control camera. Qwen chooses every motor value.',
        'public_geometry_from_qwen_landmarks':geometry,
    }


def _maybe_source_perception(context, client, **kwargs):
    observation_id = kwargs['observation_id']
    stage = context.milestone_history[-1] if context.milestone_history else 'observe'
    if context.family != 'grasp_place' or stage not in {'observe','approach','pregrasp','grasp'}:
        context.source_perception_packet = None
        context.source_perception_observation_id = None
        return None
    if context.source_perception_observation_id != observation_id:
        context.source_perception_packet = None
        context.source_perception_observation_id = observation_id
        context.source_perception_packet = _source_perception_packet(context, client() if callable(client) else client, **kwargs)
    return context.source_perception_packet


def _validate_source_perception_evidence(context, expected_model):
    rows = context.source_perception_records
    for index, row in enumerate(rows, 1):
        if row['call_index'] != index or row['status'] not in {'complete','unavailable'}:
            raise ValueError('source perception call count or completion drifted')
        if row['status'] == 'unavailable':
            if row['response'] is not None or row['error'] is None:
                raise ValueError('unavailable perception supplied a result')
            continue
        attempts = row['attempt_records']; evidence = row['evidence']
        if len(attempts) != 1 or attempts[0]['observation_id'] != row['observation_id']:
            raise ValueError('source perception attempt linkage drifted')
        attempt = attempts[0]
        if (attempt['served_model_id'] != expected_model or evidence['served_model_id'] != expected_model
                or attempt['request_sha256'] != row['request_sha256']
                or attempt['response_schema_sha256'] != _response_schema_sha256(row['input']['response_schema'])
                or evidence['image_sha256'] != row['input_image_sha256']
                or canonical_sanitized_output(attempt['sanitized_raw_command']) != row['response']):
            raise ValueError('source perception model, input or output evidence drifted')
    if rows and context.model_calls != context.controller_calls + context.critic_attempts + len(rows):
        raise ValueError('total model calls omit source perception')
    return {'valid':True,'calls':len(rows),'completed':sum(r['status']=='complete' for r in rows)}


def _make_proposal_audit_client_class(original: type, context: CriticContext) -> type:
    """Wrap Qwen with a same-model audit before any command reaches the runner."""

    class DirectControllerWithProposalAudit:
        def __init__(self, **kwargs: object) -> None:
            self._controller = original(**kwargs)
            self._critic = original(**kwargs)
            self._perception = None
            self._perception_kwargs = dict(kwargs)
            self._repair = ExecutionRepair(family=context.family)
            self._previous_wrist = None
            self._pending_wrist = None
            _set_timeout(self._critic, CRITIC_TIMEOUT_S)

        def _save_preview(self, role, call_index, packet, rendered):
            if packet is None or not Path(context.run).is_dir():
                return
            output = Path(context.run) / 'action-previews'
            output.mkdir(exist_ok=True, mode=0o700)
            name = f'{role}-{call_index:04d}'
            (output / f'{name}.png').write_bytes(rendered['wrist'])
            _atomic_json(output / f'{name}.json', packet)

        def _perception_client(self):
            if self._perception is None:
                self._perception = original(**self._perception_kwargs)
                _set_timeout(self._perception, 30.0)
                self._perception.verify()
            return self._perception

        def verify(self) -> None:
            self._controller.verify()
            self._critic.verify()

        def close(self) -> None:
            try:
                self._controller.close()
            finally:
                self._critic.close()
                if self._perception is not None:
                    self._perception.close()

        def complete(self, **kwargs: object) -> object:
            observation_id = str(kwargs["observation_id"])
            public_state = kwargs.get("public_state")
            images = kwargs.get("images")
            if not isinstance(public_state, Mapping) or not isinstance(images, Mapping):
                raise ProposalAuditFailedClosed("official controller inputs are malformed")
            validation = _proposal_audit_validation_context(kwargs)
            public_context = _instruction_context(str(kwargs["instruction"]))
            task_instruction = public_context.get("task")
            if not isinstance(task_instruction, str) or not task_instruction:
                raise ProposalAuditFailedClosed(
                    "official task instruction is malformed"
                )
            raw_recent_receipts = public_context.get("recent_receipts", [])
            recent_receipts = [
                cast(Mapping[str, object], item)
                for item in raw_recent_receipts
                if isinstance(item, Mapping)
            ] if isinstance(raw_recent_receipts, list) else []
            repair_count = len(self._repair.records)
            execution_repair = self._repair.observe(observation_id, public_state, recent_receipts)
            if len(self._repair.records) > repair_count:
                self._previous_wrist = self._pending_wrist
                if Path(context.run).is_dir():
                    _atomic_json(Path(context.run) / 'execution-repairs.json', self._repair.records)
            if context.family == "articulated":
                context.articulated_forbidden_targets = (
                    articulated_failed_contact_targets(recent_receipts)
                )
                for failed_axes in articulated_failed_actuation_axis_sets(
                    recent_receipts
                ):
                    if (
                        failed_axes
                        not in context.articulated_failed_actuation_axis_sets
                    ):
                        context.articulated_failed_actuation_axis_sets.append(
                            failed_axes
                        )
            immediate_prior_receipt = _begin_proposal_observation(
                context,
                observation_id=observation_id,
                public_state=public_state,
                images=images,
                receipts=recent_receipts,
                rgb_change=public_context.get(
                    "mean_absolute_rgb_change_since_previous"
                ),
            )
            source_perception = _maybe_source_perception(
                context, self._perception_client, observation_id=observation_id,
                task_instruction=task_instruction, images=images, public_state=public_state,
                camera_calibration=validation['camera_calibration'],
            )
            cross_view: dict[str, object] | None = None
            cross_view_close_blocked = False
            if (
                context.family == "articulated"
                and isinstance(immediate_prior_receipt, Mapping)
                and articulated_handle_insertion_receipt(immediate_prior_receipt)
            ):
                cross_view = _toaster_cross_view_status(
                    context.task,
                    immediate_prior_receipt.get("requested_camera"),
                    public_state,
                    images,
                )
                cross_view_close_blocked = bool(
                    articulated_handle_insertion_is_ready(immediate_prior_receipt)
                    and _cross_view_close_blocked(cross_view)
                )
            pull_candidates = _toaster_pull_candidates(context.task, images)
            coffee_button_candidates = _coffee_button_candidates(
                context.task, images
            )
            standing_target = _task_standing_stereo_target(
                context.task,
                context.family,
                recent_receipts,
                public_state,
                cast(
                    Mapping[str, object] | None,
                    validation.get("camera_calibration"),
                ),
            )
            control_preclose_ready = (
                _control_preclose_ready_status(immediate_prior_receipt)
                if context.task == "StartCoffeeMachine"
                else None
            )
            control_press_ready = (
                _control_preclosed_press_ready_status(recent_receipts)
                or _coffee_press_retry_ready_status(recent_receipts)
                if context.task == "StartCoffeeMachine"
                else None
            )
            coffee_contact_path_status = (
                _coffee_contact_path_ready_status(
                    standing_target, recent_receipts
                )
                if context.task == "StartCoffeeMachine"
                else None
            )
            coffee_contact_path_ready = (
                coffee_contact_path_status
                if isinstance(coffee_contact_path_status, Mapping)
                and coffee_contact_path_status.get("completed") is False
                else None
            )
            path_contact_correction_ready = (
                _coffee_contact_path_correction_ready_status(
                    coffee_contact_path_status
                )
            )
            coffee_contact_correction_ready = (
                path_contact_correction_ready
                or (
                    _coffee_contact_correction_ready_status(
                        immediate_prior_receipt,
                        standing_target,
                        recent_receipts,
                    )
                    if coffee_contact_path_status is None
                    else None
                )
                if context.task == "StartCoffeeMachine"
                else None
            )
            coffee_standing_convergence_ready = (
                _coffee_standing_convergence_ready_status(
                    immediate_prior_receipt,
                    standing_target,
                    recent_receipts,
                )
                if context.task == "StartCoffeeMachine"
                and coffee_contact_path_status is None
                else None
            )
            if coffee_contact_correction_ready is not None:
                control_preclose_ready = None
                coffee_button_candidates = None
            elif coffee_contact_path_ready is not None:
                control_preclose_ready = None
                coffee_button_candidates = None
            elif coffee_standing_convergence_ready is not None:
                control_preclose_ready = None
                coffee_button_candidates = None
            elif control_press_ready is not None:
                coffee_button_candidates = None
            coffee_retreat_status = _coffee_clearance_status(
                context, public_state, recent_receipts
            )
            coffee_finish_due = _coffee_measured_finish_ready(
                context, coffee_retreat_status
            )
            if coffee_finish_due:
                coffee_button_candidates = None
            coffee_retreat_due = bool(
                context.task == "StartCoffeeMachine"
                and coffee_retreat_status["press_effect_observed"] is True
                and coffee_retreat_status["clearance_reached"] is not True
            )
            measured_contact_retreat = bool(
                coffee_retreat_due
                and (
                    context.coffee_retreat_targets == COFFEE_CONTROL_MEASURED_RETREAT_TARGETS
                    or any(
                        _coffee_contact_path_receipt_index(receipt)
                        == len(COFFEE_CONTROL_CONTACT_WAYPOINTS) - 1
                        for receipt in recent_receipts
                    )
                )
            )
            measured_retreat_targets = (
                COFFEE_CONTROL_MEASURED_RETREAT_TARGETS
                if measured_contact_retreat else None
            )
            approach_stall = (
                articulated_approach_stall_status(recent_receipts)
                if context.family == "articulated"
                else None
            )
            base_controller_kwargs = dict(kwargs)
            base_controller_kwargs.pop("proposal_audit_context", None)
            source_contact_stage = bool(
                context.family == "grasp_place"
                and context.milestone_history
                and context.milestone_history[-1] in {"approach", "pregrasp", "grasp"}
            )
            if source_contact_stage:
                measured_context = json.loads(str(base_controller_kwargs["instruction"]))
                measured_context.pop("image_servo_alignment", None)
                measured_context["recent_receipts"] = [
                    {key: value for key, value in receipt.items() if key in {
                        "observation_id", "kind", "accepted", "step_count",
                        "gripper_intent", "gripper_residual", "end_effector_pose_delta",
                        "failure_status", "endpoint_error", "cartesian_residual_norm",
                        "tracking_pause_count", "minimum_hard_limit_margin",
                        "mean_absolute_rgb_change",
                    }}
                    for receipt in measured_context["recent_receipts"][-1:]
                ]
                measured_context["source_contact_context"] = (
                    "The latest receipt reports physical outcome. Re-localize the source "
                    "from current RGB and public robot state. Use the learned failure "
                    "mechanism to select a new contact hypothesis or observation motion."
                )
                base_controller_kwargs["instruction"] = json.dumps(
                    measured_context, sort_keys=True, separators=(",", ":")
                )
            source_atlas_layout = None
            if (
                context.family == "grasp_place"
                and (
                    context.milestone_history[-1]
                    if context.milestone_history else "observe"
                ) in {"observe", "approach", "pregrasp", "grasp"}
            ):
                atlas_images, source_atlas_layout = _source_observation_atlas(images)
                if source_atlas_layout is not None:
                    base_controller_kwargs["images"] = atlas_images
                    base_controller_kwargs["instruction"] = (
                        str(base_controller_kwargs["instruction"])
                        + "\n\nSOURCE_OBSERVATION_ATLAS:\n"
                        + json.dumps(source_atlas_layout, sort_keys=True, separators=(",", ":"))
                    )
            contact_wrist_layout = None
            if (
                context.family == "grasp_place"
                and context.milestone_history
                and context.milestone_history[-1] in {"pregrasp", "grasp"}
            ):
                contact_images, contact_wrist_layout = _source_contact_wrist_view(
                    cast(Mapping[str, bytes], base_controller_kwargs["images"]),
                    public_state,
                    cast(Mapping[str, object], validation["camera_calibration"]),
                )
                if contact_wrist_layout is not None:
                    base_controller_kwargs["images"] = contact_images
                    base_controller_kwargs["instruction"] = (
                        str(base_controller_kwargs["instruction"]).replace(
                            "Wrist is unchanged.",
                            "Wrist layout follows in SOURCE_CONTACT_WRIST_VIEW.",
                        )
                        + "\n\nSOURCE_CONTACT_WRIST_VIEW:\n"
                        + json.dumps(contact_wrist_layout, sort_keys=True, separators=(",", ":"))
                    )
            revision_advice: dict[str, object] | None = None
            protocol_message: str | None = None
            while True:
                revision_index = context.proposal_revisions_used
                controller_kwargs = dict(base_controller_kwargs)
                if revision_advice is not None:
                    controller_kwargs["instruction"] = _proposal_revision_instruction(
                        str(base_controller_kwargs["instruction"]),
                        contradiction=str(revision_advice["contradiction"]),
                        evidence=cast(list[str], revision_advice["evidence"]),
                        suggested_correction=str(
                            revision_advice["suggested_correction"]
                        ),
                        confidence=str(revision_advice["confidence"]),
                        rejected_draft=context.proposal_records[-1]["draft"],
                        rejected_history=[
                            {
                                "kind": cast(Mapping[str, object], item["draft"]).get("kind"),
                                "camera": cast(Mapping[str, object], item["draft"]).get("camera"),
                                "target_pixel": cast(Mapping[str, object], item["draft"]).get("target_pixel"),
                                "other_view_pixel": cast(Mapping[str, object], item["draft"]).get("other_view_pixel"),
                                "targets": cast(Mapping[str, object], item["draft"]).get("targets"),
                                "contradiction": item.get("contradiction"),
                            }
                            for item in context.proposal_records
                            if item.get("observation_id") == observation_id
                            and item.get("status") != "approved_for_execution"
                            and isinstance(item.get("draft"), Mapping)
                        ],
                        required_observation_id=observation_id,
                        immediate_prior_receipt=immediate_prior_receipt,
                        recent_receipts=recent_receipts,
                        failed_actuation_axis_sets=(
                            context.articulated_failed_actuation_axis_sets
                        ),
                        task=context.task,
                        protocol_message=protocol_message,
                    )
                controller_kwargs["instruction"] = (
                    _controller_articulated_recovery_instruction(
                        str(controller_kwargs["instruction"]),
                        context.articulated_forbidden_targets,
                        immediate_prior_receipt,
                        context.articulated_failed_actuation_axis_sets,
                        task=context.task,
                        current_milestone=(
                            context.milestone_history[-1]
                            if context.milestone_history
                            else None
                        ),
                    )
                )
                source_recovery = (
                    source_grasp_recovery_target(recent_receipts)
                    if context.family == "grasp_place" else None
                )
                if source_recovery is not None:
                    controller_kwargs["instruction"] = (
                        str(controller_kwargs["instruction"])
                        + "\n\nSOURCE_GRASP_RECOVERY_CONTEXT:\n"
                        + json.dumps(
                            _source_close_measurement(source_recovery) if source_contact_stage else source_recovery,
                            sort_keys=True, separators=(",", ":"),
                        )
                        + "\nThe sealed source close was empty. Remain at grasp and "
                        "use an open retreat or reapproach as needed. Before closing "
                        "again, Qwen must choose a changed source point, a changed "
                        "nonzero depth, or a newly grounded stereo pair from fresh "
                        "RGB. Keep a correctly localized point when changing depth "
                        "or adding stereo; moving off the object is not required. "
                        "An open retreat alone does not change the failed contact "
                        "strategy. Qwen authors every numeric value."
                        " For a gripper-only stationary re-close, first execute "
                        "the changed source image-servo approach; its receipt "
                        "supplies the contact geometry. A changed target in the "
                        "close note alone does not move the gripper."
                    )
                if pull_candidates is not None:
                    controller_kwargs["instruction"] = (
                        _controller_pull_candidates_instruction(
                            str(controller_kwargs["instruction"]),
                            pull_candidates,
                        )
                    )
                if coffee_button_candidates is not None:
                    controller_kwargs["instruction"] = (
                        _controller_coffee_button_candidates_instruction(
                            str(controller_kwargs["instruction"]),
                            coffee_button_candidates,
                        )
                    )
                if coffee_finish_due:
                    controller_kwargs["instruction"] = (
                        _controller_coffee_finish_instruction(
                            str(controller_kwargs["instruction"]), coffee_retreat_status
                        )
                    )
                elif coffee_retreat_due:
                    controller_kwargs["instruction"] = (
                        _controller_coffee_retreat_instruction(
                            str(controller_kwargs["instruction"]),
                            coffee_retreat_status,
                            first_endpoint=context.coffee_retreat_targets,
                            measured_endpoint=measured_retreat_targets,
                        )
                    )
                elif coffee_contact_path_ready is not None:
                    controller_kwargs["instruction"] = (
                        _controller_coffee_contact_path_instruction(
                            str(controller_kwargs["instruction"]),
                            coffee_contact_path_ready,
                        )
                    )
                elif coffee_contact_correction_ready is not None:
                    controller_kwargs["instruction"] = (
                        _controller_coffee_contact_correction_instruction(
                            str(controller_kwargs["instruction"]),
                            coffee_contact_correction_ready,
                        )
                    )
                elif coffee_standing_convergence_ready is not None:
                    controller_kwargs["instruction"] = (
                        _controller_coffee_standing_convergence_instruction(
                            str(controller_kwargs["instruction"]),
                            coffee_standing_convergence_ready,
                        )
                    )
                elif control_press_ready is not None:
                    controller_kwargs["instruction"] = (
                        _controller_control_press_ready_instruction(
                            str(controller_kwargs["instruction"]),
                            control_press_ready,
                        )
                    )
                elif control_preclose_ready is not None:
                    controller_kwargs["instruction"] = (
                        _controller_control_preclose_instruction(
                            str(controller_kwargs["instruction"]),
                            control_preclose_ready,
                        )
                    )
                if standing_target is not None and not (
                    context.task == "StartCoffeeMachine"
                    and (
                        coffee_contact_correction_ready is not None
                        or coffee_contact_path_ready is not None
                        or coffee_standing_convergence_ready is not None
                        or control_preclose_ready is not None
                        or control_press_ready is not None
                        or coffee_retreat_due
                    )
                ):
                    controller_kwargs["instruction"] = (
                        _controller_standing_target_instruction(
                            str(controller_kwargs["instruction"]),
                            standing_target,
                            task=context.task,
                        )
                    )
                if approach_stall is not None:
                    controller_kwargs["instruction"] = (
                        _controller_approach_stall_instruction(
                            str(controller_kwargs["instruction"]),
                            approach_stall,
                            ik_table=_planar_ik_table(public_state),
                            task=context.task,
                            immediate_prior_receipt=immediate_prior_receipt,
                        )
                    )
                if cross_view is not None:
                    controller_kwargs["instruction"] = (
                        _controller_cross_view_instruction(
                            str(controller_kwargs["instruction"]),
                            cross_view,
                            selected_camera=str(
                                cast(Mapping[str, object], immediate_prior_receipt)
                                .get("requested_camera")
                            ),
                            close_blocked=cross_view_close_blocked,
                        )
                    )
                controller_kwargs["instruction"] = (
                    _controller_task_grounding_instruction(
                        str(controller_kwargs["instruction"]), task=context.task
                    )
                )
                wrist_roll_revision = wrist_roll_revision_required(
                    context.proposal_records[-1]
                    if revision_advice is not None and context.proposal_records
                    else None,
                    observation_id,
                )
                wrist_roll_tracking = articulated_wrist_roll_tracking_status(
                    immediate_prior_receipt
                )
                wrist_roll_continuation = bool(
                    isinstance(wrist_roll_tracking, Mapping)
                    and wrist_roll_tracking.get("settled") is False
                )
                orientation_recovery_schema_due = bool(
                    context.family == "articulated"
                    and isinstance(immediate_prior_receipt, Mapping)
                    and immediate_prior_receipt.get("kind") == "cartesian_delta"
                    and immediate_prior_receipt.get("requested_gripper") == "open"
                    and len(context.articulated_forbidden_targets)
                    >= (1 if context.task == "OpenToasterOvenDoor" else 2)
                    and not (
                        isinstance(wrist_roll_tracking, Mapping)
                        and wrist_roll_tracking.get("settled") is True
                    )
                )
                toaster_stall_close_probe_schema_due = (
                    toaster_stall_close_probe_due(
                        context.task, approach_stall, immediate_prior_receipt
                    )
                )
                insertion_ready_schema_due = bool(
                    context.family == "articulated"
                    and isinstance(immediate_prior_receipt, Mapping)
                    and (
                        articulated_handle_insertion_is_ready(
                            immediate_prior_receipt
                        )
                        or toaster_stall_close_probe_schema_due
                    )
                )
                servo_in_progress_schema_due = bool(
                    approach_stall is None
                    and not insertion_ready_schema_due
                    and isinstance(immediate_prior_receipt, Mapping)
                    and _stereo_servo_in_progress(immediate_prior_receipt)
                )
                near_handle_stereo_due = bool(
                    approach_stall is None
                    and not insertion_ready_schema_due
                    and not servo_in_progress_schema_due
                    and isinstance(immediate_prior_receipt, Mapping)
                    and _handle_servo_near_but_unaligned(immediate_prior_receipt)
                )
                ready_pose_done_schema_due = bool(
                    approach_stall is None
                    and not insertion_ready_schema_due
                    and not servo_in_progress_schema_due
                    and not near_handle_stereo_due
                    and _ready_pose_reached_without_servo(public_state, recent_receipts)
                )
                if coffee_finish_due:
                    controller_kwargs["response_schema"] = (
                        controller_control_finish_schema_for_observation(observation_id)
                    )
                elif coffee_retreat_due:
                    controller_kwargs["response_schema"] = (
                        controller_control_retreat_schema_for_observation(
                            observation_id, measured_contact_path=measured_contact_retreat
                        )
                    )
                elif coffee_contact_path_ready is not None:
                    controller_kwargs["response_schema"] = (
                        controller_control_contact_path_schema_for_observation(
                            observation_id,
                            waypoint_index=int(
                                coffee_contact_path_ready["waypoint_index"]
                            ),
                        )
                    )
                elif coffee_contact_correction_ready is not None:
                    controller_kwargs["response_schema"] = (
                        controller_control_contact_correction_schema_for_observation(
                            observation_id
                        )
                    )
                elif coffee_standing_convergence_ready is not None:
                    controller_kwargs["response_schema"] = (
                        controller_control_standing_convergence_schema_for_observation(
                            observation_id
                        )
                    )
                elif control_press_ready is not None:
                    controller_kwargs["response_schema"] = (
                        controller_control_press_schema_for_observation(
                            observation_id
                        )
                    )
                elif control_preclose_ready is not None:
                    controller_kwargs["response_schema"] = (
                        controller_control_preclose_schema_for_observation(
                            observation_id
                        )
                    )
                elif (
                    wrist_roll_revision
                    or wrist_roll_continuation
                    or orientation_recovery_schema_due
                ):
                    controller_kwargs["response_schema"] = (
                        controller_wrist_roll_schema_for_observation(observation_id)
                    )
                elif insertion_ready_schema_due:
                    controller_kwargs["response_schema"] = (
                        controller_engage_ready_schema_for_observation(
                            observation_id,
                            allow_close=not cross_view_close_blocked,
                        )
                    )
                elif ready_pose_done_schema_due:
                    # The arm is bent and nothing has moved toward the target
                    # yet: the only useful next command is a servo (or a base
                    # pulse), never another joint re-issue or a hold.
                    controller_kwargs["response_schema"] = (
                        controller_engage_ready_schema_for_observation(
                            observation_id,
                            allow_close=False,
                            require_stereo=False,
                            allow_base=True,
                        )
                    )
                elif servo_in_progress_schema_due or near_handle_stereo_due:
                    # A progressing stereo servo continues as a STEREO servo;
                    # the Cartesian detours and the single-view drift that
                    # burnt the budget near the handle are not offered. The
                    # stereo requirement applies to the first proposal only:
                    # after a rejected pair (ray gap) a revision may fall back
                    # to a single-view servo instead of repeating a bad pair.
                    controller_kwargs["response_schema"] = (
                        controller_engage_ready_schema_for_observation(
                            observation_id,
                            allow_close=False,
                            require_stereo=revision_advice is None,
                            allow_base=True,
                        )
                    )
                else:
                    controller_kwargs["response_schema"] = (
                        controller_response_schema_for_observation(observation_id)
                    )
                failure_skills = _failure_skill_packet(context, recent_receipts)
                if failure_skills is not None:
                    controller_failure_skills = canonical_json_copy(failure_skills)
                    if source_contact_stage:
                        for case in controller_failure_skills["recent_cases"]:
                            if "failed_approach" in case:
                                case["failed_approach"] = _source_close_measurement(case["failed_approach"])
                    controller_kwargs["instruction"] = (
                        str(controller_kwargs["instruction"])
                        + "\n\nQWEN_FAILURE_SKILLS:\n"
                        + json.dumps(controller_failure_skills, sort_keys=True, separators=(",", ":"))
                    )
                    context.failure_skill_uses.append({
                        "controller_call_index": context.controller_calls + 1,
                        "observation_id": observation_id,
                        "skill_ids": [skill["id"] for skill in failure_skills["skills"]],
                    })
                if source_perception is not None:
                    controller_kwargs['instruction'] = str(controller_kwargs['instruction']) + '\n\nQWEN_FOCUSED_PERCEPTION:\n' + json.dumps(source_perception, sort_keys=True, separators=(',', ':'))
                if execution_repair is not None:
                    controller_kwargs['instruction'] = str(controller_kwargs['instruction']) + '\n\nEXECUTION_REPAIR:\n' + json.dumps(execution_repair, separators=(',', ':'))
                preview_draft = None
                if context.proposal_records:
                    previous_proposal = context.proposal_records[-1]
                    if previous_proposal.get('observation_id') == observation_id and isinstance(previous_proposal.get('draft'), Mapping):
                        preview_draft = previous_proposal['draft']
                landmark = (source_perception or {}).get('public_geometry_from_qwen_landmarks')
                preview_images, preview_packet = render_preview(
                    controller_kwargs['images'], images, public_state, validation['camera_calibration'],
                    preview_draft, validation['current_gripper'], landmark, self._previous_wrist)
                if preview_packet is not None:
                    controller_kwargs['images'] = preview_images
                    controller_kwargs['instruction'] = str(controller_kwargs['instruction']) + '\n\nACTION_PREVIEW:\n' + json.dumps(preview_packet, separators=(',', ':'))
                    self._save_preview('controller', context.controller_calls + 1, preview_packet, preview_images)
                from .workflow import prepare_controller_request
                prepare_controller_request(controller_kwargs, public_state, recent_receipts, context.milestone_history)
                from .workflow import execution_mode, apply_execution_mode, repeated_recovery_command, proposal_recovery_mode
                declared_skill_active = bool(coffee_contact_path_ready or coffee_contact_correction_ready
                                             or coffee_retreat_due or coffee_finish_due)
                proposal_mode = None
                if not declared_skill_active and context.milestone_history and context.milestone_history[-1] != 'observe':
                    proposal_mode = proposal_recovery_mode(context.proposal_records, observation_id, revision_index)
                workflow_mode = execution_mode(execution_repair, context.family, context.milestone_history,
                    skill_active=declared_skill_active, proposal_mode=proposal_mode)
                apply_execution_mode(controller_kwargs, workflow_mode)
                controller_kwargs['instruction'] = _controller_milestone_context_instruction(
                    str(controller_kwargs['instruction']), family=context.family,
                    milestone_history=context.milestone_history)
                controller_manifest_sha256 = _model_call_manifest_sha256(
                    controller_kwargs
                )
                call_kwargs, attempt_log = _with_captured_attempt_log(
                    controller_kwargs
                )
                context.controller_calls += 1
                context.model_calls += 1
                call_index = context.controller_calls
                attempt_index = int(call_kwargs.get("attempt_index", 0))
                try:
                    response = self._controller.complete(**call_kwargs)
                except Exception as error:
                    _persist_captured_attempt_records(
                        context,
                        attempt_log,
                        role="controller",
                        call_index=call_index,
                    )
                    if _is_malformed_response(error):
                        try:
                            attempt_record_sha256 = (
                                _proposal_attempt_record_sha256(
                                    context,
                                    role="controller",
                                    call_index=call_index,
                                )
                            )
                        except _AttemptEvidenceLinkageIncomplete:
                            attempt_record_sha256 = None
                        controller_request_sha256, linkage_status = (
                            _request_linkage_from_attempt_log(
                                attempt_log,
                                observation_id=observation_id,
                                attempt_index=attempt_index,
                                response_schema=controller_kwargs.get(
                                    "response_schema"
                                ),
                            )
                        )
                        malformed_draft: object = None
                        if len(attempt_log.records) == 1:
                            raw_draft = attempt_log.records[0].get(
                                "sanitized_raw_command"
                            )
                            if isinstance(raw_draft, str) and raw_draft:
                                malformed_draft = canonical_sanitized_output(raw_draft)
                        record = _proposal_base_record(
                            context,
                            observation_id=observation_id,
                            revision_index=revision_index,
                            draft=malformed_draft,
                            controller_call_index=call_index,
                            controller_request_sha256=(
                                controller_request_sha256
                            ),
                            controller_call_manifest_sha256=(
                                controller_manifest_sha256
                            ),
                            controller_attempt_record_sha256=(
                                attempt_record_sha256
                            ),
                            immediate_prior_receipt=immediate_prior_receipt,
                        )
                        revision_advice = _protocol_rejection(
                            record,
                            error=error,
                        )
                        _seal_evidence_record(record)
                        context.proposal_records.append(record)
                        controller_record: dict[str, object] = {
                            "observation_id": observation_id,
                            "controller_call_index": call_index,
                            "qwen_attempt_index": attempt_index,
                            "command": record["draft"],
                            "command_sha256": record["draft_sha256"],
                            "controller_request_sha256": (
                                controller_request_sha256
                            ),
                            "controller_request_linkage_status": (
                                linkage_status
                            ),
                            "controller_call_manifest_sha256": (
                                controller_manifest_sha256
                            ),
                            "status": "rejected_by_protocol",
                            "record_sha256": None,
                        }
                        _seal_evidence_record(controller_record)
                        context.controller_records.append(controller_record)
                        _consume_proposal_revision(context, observation_id)
                        continue
                    context.proposal_failed_observations.add(observation_id)
                    raise ProposalAuditFailedClosed(
                        "Qwen controller draft was unavailable"
                    ) from error
                _persist_captured_attempt_records(
                    context,
                    attempt_log,
                    role="controller",
                    call_index=call_index,
                )
                attempt_record_sha256 = _proposal_attempt_record_sha256(
                    context,
                    role="controller",
                    call_index=call_index,
                )
                try:
                    controller_request_sha256 = _actual_request_sha256_from_attempt_log(
                        attempt_log,
                        observation_id=observation_id,
                        attempt_index=attempt_index,
                        response_schema=controller_kwargs.get("response_schema"),
                    )
                except _AttemptEvidenceLinkageIncomplete as error:
                    raw_draft = response.command
                    draft = raw_draft if isinstance(raw_draft, Mapping) else None
                    record = _proposal_base_record(
                        context,
                        observation_id=observation_id,
                        revision_index=revision_index,
                        draft=draft,
                        controller_call_index=call_index,
                        controller_request_sha256=None,
                        controller_call_manifest_sha256=(
                            controller_manifest_sha256
                        ),
                        controller_attempt_record_sha256=(
                            attempt_record_sha256
                        ),
                        immediate_prior_receipt=immediate_prior_receipt,
                    )
                    record.update({
                        "status": "controller_linkage_incomplete",
                        "contradiction": "stale_or_missing_evidence",
                        "suggested_correction": "observe",
                        "failure_class": type(error).__name__,
                        "failure_sha256": hashlib.sha256(
                            str(error).encode()
                        ).hexdigest(),
                    })
                    _seal_evidence_record(record)
                    context.proposal_records.append(record)
                    controller_record = {
                        "observation_id": observation_id,
                        "controller_call_index": call_index,
                        "qwen_attempt_index": attempt_index,
                        "command": record["draft"],
                        "command_sha256": record["draft_sha256"],
                        "controller_request_sha256": None,
                        "controller_request_linkage_status": "incomplete",
                        "controller_call_manifest_sha256": (
                            controller_manifest_sha256
                        ),
                        "status": "controller_linkage_incomplete",
                        "record_sha256": None,
                    }
                    _seal_evidence_record(controller_record)
                    context.controller_records.append(controller_record)
                    context.proposal_failed_observations.add(observation_id)
                    raise ProposalAuditFailedClosed(
                        "Qwen controller AttemptEvidenceLog linkage failed"
                    ) from error
                raw_draft = response.command
                draft = raw_draft if isinstance(raw_draft, Mapping) else None
                record = _proposal_base_record(
                    context,
                    observation_id=observation_id,
                    revision_index=revision_index,
                    draft=draft,
                    controller_call_index=call_index,
                    controller_request_sha256=controller_request_sha256,
                    controller_call_manifest_sha256=controller_manifest_sha256,
                    controller_attempt_record_sha256=attempt_record_sha256,
                    immediate_prior_receipt=immediate_prior_receipt,
                )
                controller_record = {
                    "observation_id": observation_id,
                    "controller_call_index": call_index,
                    "qwen_attempt_index": attempt_index,
                    "command": record["draft"],
                    "command_sha256": record["draft_sha256"],
                    "controller_request_sha256": controller_request_sha256,
                    "controller_request_linkage_status": "complete",
                    "controller_call_manifest_sha256": controller_manifest_sha256,
                    "status": "draft",
                    "record_sha256": None,
                }
                _seal_evidence_record(controller_record)
                context.controller_records.append(controller_record)
                protocol_error: Exception | None = None
                claimed_milestone: str | None = None
                try:
                    if draft is None:
                        raise ValueError("controller draft is not a command mapping")
                    if repeated_recovery_command(workflow_mode, draft):
                        raise ValueError('measured workflow requires a changed motor command; changing the note alone does not repair the executed approach')
                    claimed_milestone = milestone_from_note(draft.get("note"))
                    validate_milestone_transition(
                        context.family,
                        claimed_milestone,
                        context.milestone_history,
                    )
                    validate_milestone_action_semantics(
                        context.family,
                        claimed_milestone,
                        draft,
                        task=context.task,
                        milestone_history=context.milestone_history,
                        immediate_prior_receipt=immediate_prior_receipt,
                        recent_receipts=recent_receipts,
                        articulated_forbidden_targets=(
                            context.articulated_forbidden_targets
                        ),
                        articulated_failed_axes=(
                            context.articulated_failed_actuation_axis_sets
                        ),
                        cross_view_depth_mismatch=cross_view_close_blocked,
                        approach_stall=approach_stall,
                    )
                    if coffee_finish_due and draft.get("kind") != "finish":
                        raise ValueError(
                            "measured coffee press and clearance are complete; "
                            "author finish for official terminal evaluation"
                        )
                    contact_path_violation = _coffee_contact_path_violation(
                        coffee_contact_path_ready, draft
                    )
                    if contact_path_violation is not None:
                        raise ValueError(contact_path_violation)
                    contact_correction_violation = (
                        _coffee_contact_correction_violation(
                            coffee_contact_correction_ready, draft
                        )
                    )
                    if contact_correction_violation is not None:
                        raise ValueError(contact_correction_violation)
                    standing_convergence_violation = (
                        _coffee_standing_convergence_violation(
                            coffee_standing_convergence_ready, draft
                        )
                    )
                    if standing_convergence_violation is not None:
                        raise ValueError(standing_convergence_violation)
                    coffee_button_violation = _coffee_button_target_violation(
                        context.task,
                        draft,
                        coffee_button_candidates,
                        standing_target=standing_target,
                    )
                    if coffee_button_violation is not None:
                        raise ValueError(coffee_button_violation)
                    coffee_press_target_violation = (
                        _coffee_press_target_violation(control_press_ready, draft)
                    )
                    if coffee_press_target_violation is not None:
                        raise ValueError(coffee_press_target_violation)
                    toaster_target = (
                        None
                        if _draft_matches_standing_target(draft, standing_target)
                        else _toaster_oven_handle_target_status(
                            context.task,
                            draft,
                            images,
                        )
                    )
                    if (
                        isinstance(toaster_target, Mapping)
                        and toaster_target.get("horizontal_pull_detected") is True
                        and toaster_target.get("target_on_horizontal_pull") is False
                    ):
                        box = toaster_target.get("pull_box")
                        span = (
                            f"; the dark pull spans u={box['u_min']}..{box['u_max']}, "
                            f"v={box['v_min']}..{box['v_max']} in the "
                            f"{draft.get('camera')} view"
                            if isinstance(box, Mapping)
                            else ""
                        )
                        raise ValueError(
                            "toaster oven target center is outside the visible "
                            "narrow horizontal handle" + span
                        )
                    end_cap = _toaster_end_cap_violation(toaster_target, draft)
                    if end_cap is not None:
                        raise ValueError(end_cap)
                    other_view = (
                        toaster_target.get("other_view")
                        if isinstance(toaster_target, Mapping)
                        else None
                    )
                    if (
                        isinstance(other_view, Mapping)
                        and other_view.get("horizontal_pull_detected") is True
                        and other_view.get("target_on_horizontal_pull") is False
                    ):
                        other_box = other_view.get("pull_box")
                        other_span = (
                            f"; the dark pull spans u={other_box['u_min']}.."
                            f"{other_box['u_max']}, v={other_box['v_min']}.."
                            f"{other_box['v_max']} in the "
                            f"{other_view.get('camera')} view"
                            if isinstance(other_box, Mapping)
                            else ""
                        )
                        raise ValueError(
                            "toaster oven other-view pixel is outside the visible "
                            "narrow horizontal handle" + other_span
                        )
                    if draft.get("kind") in {"finish", "give_up"} and (
                        claimed_milestone != "verify_goal"
                    ):
                        raise ValueError(
                            "terminal proposal requires verify_goal milestone"
                        )
                    coffee_finish_violation = _coffee_finish_retreat_violation(
                        context.task,
                        draft,
                        recent_receipts,
                        clearance_status=coffee_retreat_status,
                    )
                    if coffee_finish_violation is not None:
                        raise ValueError(coffee_finish_violation)
                    if draft.get("kind") == "finish" and context.family == "control":
                        effect = (
                            _coffee_press_effect(recent_receipts)
                            if context.task == "StartCoffeeMachine"
                            else _control_actuation_effect(recent_receipts)
                        )
                        persistent_coffee_press = bool(
                            context.task == "StartCoffeeMachine"
                            and context.coffee_press_world_eef_m is not None
                        )
                        if (
                            effect["effect_observed"] is not True
                            and not persistent_coffee_press
                        ):
                            raise ValueError(
                                "control press produced no effect: the actuation "
                                "receipts show a wrench force delta of "
                                f"{effect['max_force_delta_n']} N and external RGB "
                                f"change of {effect['max_external_rgb_change']} "
                                "(needs at least 1 N or 2.0); re-align with a stereo "
                                "pair and press again before finishing"
                            )
                    if coffee_retreat_due:
                        direction_violation = (
                            _coffee_retreat_direction_violation(
                                context, draft, public_state
                            )
                        )
                        if direction_violation is not None:
                            raise ValueError(direction_violation)
                        endpoint_violation = _coffee_retreat_endpoint_violation(
                            context, draft, required_targets=measured_retreat_targets
                        )
                        if endpoint_violation is not None:
                            raise ValueError(endpoint_violation)
                    prepare_joint_mailbox(
                        draft,
                        source="controller",
                        observation_id=observation_id,
                        current_qpos=cast(Sequence[object], validation["current_qpos"]),
                        current_gripper=cast(float, validation["current_gripper"]),
                        public_state=public_state,
                        camera_calibration=cast(
                            Mapping[str, object], validation["camera_calibration"]
                        ),
                        remaining_actions=cast(int, validation["remaining_actions"]),
                        sequence=cast(int, validation["sequence"]),
                    )
                except (TypeError, ValueError) as error:
                    protocol_error = error
                if protocol_error is not None:
                    protocol_message = str(protocol_error)
                    if "ray gap" in protocol_message and isinstance(draft, Mapping):
                        protocol_message += _stereo_bar_diagnosis(
                            draft, pull_candidates
                        )
                    if "endpoint is unsafe" in protocol_message:
                        protocol_message += _reach_limit_diagnosis(
                            protocol_message, public_state, standing_target
                        )
                    revision_advice = _protocol_rejection(
                        record,
                        error=protocol_error,
                    )
                    context.proposal_records.append(record)
                    context.controller_records[-1]["status"] = (
                        "rejected_by_protocol"
                    )
                    _seal_evidence_record(record)
                    _seal_evidence_record(context.controller_records[-1])
                else:
                    assert draft is not None and claimed_milestone is not None
                    # This draft passed protocol validation. Any subsequent
                    # critic rejection supersedes the previous protocol error.
                    protocol_message = None
                    record["claimed_milestone"] = claimed_milestone
                    critic_atlas_layout = None
                    if source_atlas_layout is not None:
                        critic_images, critic_atlas_layout = _source_observation_atlas(
                            images, draft=draft
                        )
                    else:
                        critic_images = _proposal_critic_images(images, draft)
                    if contact_wrist_layout is not None:
                        critic_images = {
                            **critic_images,
                            "wrist": cast(Mapping[str, bytes], base_controller_kwargs["images"])["wrist"],
                        }
                    critic_instruction = _proposal_critic_instruction(
                        context,
                        task_instruction=task_instruction,
                        observation_id=observation_id,
                        draft=draft,
                        claimed_milestone=claimed_milestone,
                        public_state=public_state,
                        images=images,
                        critic_images=critic_images,
                        immediate_prior_receipt=immediate_prior_receipt,
                        recent_receipts=recent_receipts,
                        cross_view=cross_view,
                        approach_stall=approach_stall,
                        standing_target=standing_target,
                    )
                    if critic_atlas_layout is not None or contact_wrist_layout is not None:
                        critic_packet = json.loads(critic_instruction)
                        if critic_atlas_layout is not None:
                            if contact_wrist_layout is not None:
                                critic_atlas_layout["instruction"] = str(
                                    critic_atlas_layout["instruction"]
                                ).replace(
                                    "Wrist is unchanged.",
                                    "Wrist layout is in source_contact_wrist_view.",
                                )
                            critic_packet["source_observation_atlas"] = critic_atlas_layout
                        if contact_wrist_layout is not None:
                            critic_packet["source_contact_wrist_view"] = contact_wrist_layout
                        critic_instruction = json.dumps(
                            critic_packet, sort_keys=True, separators=(",", ":")
                        )
                    critic_images, critic_preview = render_preview(
                        critic_images, images, public_state, validation['camera_calibration'],
                        draft, validation['current_gripper'], landmark, self._previous_wrist)
                    if critic_preview is not None:
                        critic_packet = json.loads(critic_instruction)
                        critic_packet['action_preview'] = critic_preview
                        critic_packet['execution_repair'] = execution_repair
                        critic_packet['critic_input_rgb_sha256'] = image_hashes(critic_images)
                        critic_instruction = json.dumps(critic_packet, separators=(',', ':'))
                        self._save_preview('critic', context.critic_attempts + 1, critic_preview, critic_images)
                    critic_kwargs = {
                        "observation_id": observation_id,
                        "system_prompt": (
                            _source_atlas_critic_prompt(context.critic_prompt)
                            if critic_atlas_layout is not None else context.critic_prompt
                        ),
                        "instruction": critic_instruction,
                        "public_state": public_state,
                        "images": critic_images,
                        "response_schema": PROPOSAL_AUDIT_SCHEMA,
                        "max_tokens": CRITIC_MAX_TOKENS,
                    }
                    if contact_wrist_layout is not None:
                        critic_kwargs["system_prompt"] = str(critic_kwargs["system_prompt"]).replace(
                            "wrist RGB is unchanged.",
                            "the wrist layout is described in source_contact_wrist_view.",
                        ) + (
                            "\nThe wrist image contains the unchanged raw panel at upper left "
                            "and a continuous 2x view at right. Its cyan open-center marker "
                            "shows the current measured grip-site, not an object target or "
                            "contact evidence. Compare the object and tool across the views."
                        )
                    if "attempt_log" in kwargs:
                        critic_kwargs["attempt_log"] = kwargs["attempt_log"]
                    if source_perception is not None:
                        packet = json.loads(str(critic_kwargs['instruction']))
                        packet['qwen_focused_perception'] = source_perception
                        critic_kwargs['instruction'] = json.dumps(packet, sort_keys=True, separators=(',', ':'))
                    critic_manifest_sha256 = _model_call_manifest_sha256(
                        critic_kwargs
                    )
                    if any(
                        prior.get("observation_id") == observation_id
                        and prior.get("status") == "rejected_by_critic"
                        and prior.get("critic_call_manifest_sha256") == critic_manifest_sha256
                        for prior in context.proposal_records
                    ):
                        protocol_message = (
                            "unchanged rejected draft has identical critic input "
                            "on this observation; revise the complete draft, "
                            "including its note when appropriate, before another audit"
                        )
                        revision_advice = _protocol_rejection(
                            record, error=ValueError(protocol_message),
                        )
                        context.proposal_records.append(record)
                        context.controller_records[-1]["status"] = "rejected_by_protocol"
                        _seal_evidence_record(record)
                        _seal_evidence_record(context.controller_records[-1])
                        _consume_proposal_revision(context, observation_id)
                        continue
                    critic_call_kwargs, critic_attempt_log = (
                        _with_captured_attempt_log(critic_kwargs)
                    )
                    context.critic_attempts += 1
                    context.model_calls += 1
                    critic_call_index = context.critic_attempts
                    critic_attempt_index = int(
                        critic_call_kwargs.get("attempt_index", 0)
                    )
                    critic_request_sha256: str | None = None
                    try:
                        critic_response = self._critic.complete(
                            **critic_call_kwargs
                        )
                        critic_request_sha256 = (
                            _actual_request_sha256_from_attempt_log(
                                critic_attempt_log,
                                observation_id=observation_id,
                                attempt_index=critic_attempt_index,
                                response_schema=PROPOSAL_AUDIT_SCHEMA,
                            )
                        )
                        audit = validate_proposal_audit(critic_response.command)
                    except Exception as error:
                        context.critic_unavailable += 1
                        _persist_captured_attempt_records(
                            context,
                            critic_attempt_log,
                            role="critic",
                            call_index=critic_call_index,
                        )
                        try:
                            critic_attempt_record_sha256 = (
                                _proposal_attempt_record_sha256(
                                    context,
                                    role="critic",
                                    call_index=critic_call_index,
                                )
                            )
                        except _AttemptEvidenceLinkageIncomplete:
                            critic_attempt_record_sha256 = None
                        record.update({
                            "critic_call_index": critic_call_index,
                            "critic_request_sha256": critic_request_sha256,
                            "critic_call_manifest_sha256": critic_manifest_sha256,
                            "critic_attempt_record_sha256": (
                                critic_attempt_record_sha256
                            ),
                            "status": "critic_unavailable",
                            "failure_class": type(error).__name__,
                            "failure_sha256": hashlib.sha256(
                                str(error).encode()
                            ).hexdigest(),
                        })
                        context.proposal_records.append(record)
                        context.controller_records[-1]["status"] = (
                            "critic_unavailable"
                        )
                        _seal_evidence_record(record)
                        _seal_evidence_record(context.controller_records[-1])
                        context.critic_records.append(cast(
                            dict[str, object], canonical_json_copy(record)
                        ))
                        context.proposal_failed_observations.add(observation_id)
                        raise ProposalAuditFailedClosed(
                            "proposal critic unavailable; draft rejected"
                        ) from error
                    _persist_captured_attempt_records(
                        context,
                        critic_attempt_log,
                        role="critic",
                        call_index=critic_call_index,
                    )
                    critic_attempt_record_sha256 = _proposal_attempt_record_sha256(
                        context,
                        role="critic",
                        call_index=critic_call_index,
                    )
                    audit_snapshot = cast(
                        dict[str, object], canonical_json_copy(audit)
                    )
                    record.update({
                        "critic_call_index": critic_call_index,
                        "critic_request_sha256": critic_request_sha256,
                        "critic_call_manifest_sha256": critic_manifest_sha256,
                        "critic_attempt_record_sha256": (
                            critic_attempt_record_sha256
                        ),
                        "audit": audit_snapshot,
                        "audit_sha256": strict_canonical_sha256(audit_snapshot),
                        "contradiction": audit_snapshot["contradiction"],
                        "suggested_correction": audit_snapshot[
                            "suggested_correction"
                        ],
                    })
                    context.critic_successes += 1
                    _capture_wrist_roll_origin(
                        context,
                        draft,
                        public_state,
                        audit_snapshot,
                    )
                    if audit_snapshot["verdict"] == "revise":
                        record["status"] = "rejected_by_critic"
                        revision_advice = {
                            "contradiction": audit_snapshot["contradiction"],
                            "evidence": audit_snapshot["evidence"],
                            "suggested_correction": audit_snapshot[
                                "suggested_correction"
                            ],
                            "confidence": audit_snapshot["confidence"],
                        }
                        context.proposal_records.append(record)
                        context.controller_records[-1]["status"] = (
                            "rejected_by_critic"
                        )
                        _seal_evidence_record(record)
                        _seal_evidence_record(context.controller_records[-1])
                        context.critic_records.append(cast(
                            dict[str, object], canonical_json_copy(record)
                        ))
                    else:
                        record.update({
                            "status": "approved_for_execution",
                            "returned_command_sha256": record["draft_sha256"],
                        })
                        context.proposal_records.append(record)
                        context.controller_records[-1]["status"] = (
                            "approved_for_execution"
                        )
                        _seal_evidence_record(record)
                        _seal_evidence_record(context.controller_records[-1])
                        context.critic_records.append(cast(
                            dict[str, object], canonical_json_copy(record)
                        ))
                        if draft.get("kind") in PHYSICAL_COMMAND_KINDS:
                            context.proposal_pending_record_index = (
                                len(context.proposal_records) - 1
                            )
                            context.proposal_pending_command = cast(
                                dict[str, object], canonical_json_copy(draft)
                            )
                            context.proposal_pending_milestone = claimed_milestone
                        controller_evidence = cast(
                            dict[str, object],
                            canonical_json_copy(dict(response.evidence)),
                        )
                        controller_evidence.update({
                            "controller_role": (
                                "sole_direct_inspect_command_emitter"
                            ),
                            "controller_call_index": call_index,
                            "controller_request_sha256": (
                                controller_request_sha256
                            ),
                            "controller_request_linkage_status": "complete",
                            "controller_call_manifest_sha256": (
                                controller_manifest_sha256
                            ),
                            "controller_input_public_state_sha256": (
                                context.proposal_public_state_sha256
                            ),
                            "controller_input_image_sha256": image_hashes(
                                cast(Mapping[str, bytes], controller_kwargs["images"])
                            ),
                            "qwen_command_sha256": record["draft_sha256"],
                            "proposal_audit_verdict": "approve",
                            "proposal_audit_sha256": record["audit_sha256"],
                            "proposal_audit_config_sha256": (
                                PROPOSAL_AUDIT_CONFIG_SHA256
                            ),
                            "claimed_milestone": claimed_milestone,
                            "critic_origin_execution": False,
                        })
                        if draft.get('kind') in PHYSICAL_COMMAND_KINDS:
                            self._repair.proposed(draft, public_state, landmark)
                            self._pending_wrist = images['wrist']
                        return type(response)(
                            command=response.command,
                            evidence=controller_evidence,
                        )
                _consume_proposal_revision(context, observation_id)

    return DirectControllerWithProposalAudit


def _make_client_class(original: type, context: CriticContext) -> type:
    class DirectControllerWithCritic:
        def __init__(self, **kwargs: object) -> None:
            self._controller = original(**kwargs)
            self._critic = original(**kwargs)
            _set_timeout(self._critic, CRITIC_TIMEOUT_S)

        def verify(self) -> None:
            self._controller.verify()
            self._critic.verify()

        def close(self) -> None:
            try:
                self._controller.close()
            finally:
                self._critic.close()

        def complete(self, **kwargs: object) -> object:
            observation_id = str(kwargs["observation_id"])
            public_state = kwargs["public_state"]
            images = kwargs["images"]
            if not isinstance(public_state, Mapping) or not isinstance(images, Mapping):
                raise ValueError("official controller inputs are malformed")
            public_context = _instruction_context(str(kwargs["instruction"]))
            decision = context.begin_observation(
                observation_id,
                public_state,
                public_context.get("recent_receipts", []),
                public_context.get("mean_absolute_rgb_change_since_previous"),
            )

            if decision.get("fired_trigger") is not None and (
                observation_id not in context.advice_by_observation
            ):
                critic_instruction = _critic_instruction(
                    context, decision, public_state, images
                )
                critic_base_kwargs = {
                    "observation_id": observation_id,
                    "system_prompt": context.critic_prompt,
                    "instruction": critic_instruction,
                    "public_state": public_state,
                    "images": images,
                    "response_schema": ADVISORY_SCHEMA,
                    "max_tokens": CRITIC_MAX_TOKENS,
                }
                if "attempt_log" in kwargs:
                    critic_base_kwargs["attempt_log"] = kwargs["attempt_log"]
                critic_call_manifest_sha256 = _model_call_manifest_sha256(
                    critic_base_kwargs
                )
                critic_kwargs, critic_attempt_log = _with_captured_attempt_log(
                    critic_base_kwargs
                )
                decision["critic_call_manifest_sha256"] = critic_call_manifest_sha256
                decision["critic_request_sha256"] = None
                critic_started = time.monotonic()
                critic_attempt_index = int(critic_kwargs.get("attempt_index", 0))
                critic_record: dict[str, object] = {
                    "schema": "robocasa-qwen-advisory-evidence/v1",
                    "task": context.task,
                    "observation_id": observation_id,
                    "advisory_id": decision["advisory_id"],
                    "trigger": decision["fired_trigger"],
                    "attempt_index": decision["critic_attempt_index"],
                    "qwen_attempt_index": critic_attempt_index,
                    "previous_executed_command_sha256": decision[
                        "previous_executed_command_sha256"
                    ],
                    "sealed_public_receipt_sha256": decision[
                        "sealed_public_receipt_sha256"
                    ],
                    "fresh_public_state_sha256": decision["fresh_public_state_sha256"],
                    "fresh_public_rgb_sha256": image_hashes(images),
                    "critic_request_sha256": None,
                    "critic_call_manifest_sha256": critic_call_manifest_sha256,
                    "input_public_state_sha256": decision["fresh_public_state_sha256"],
                    "input_image_sha256": image_hashes(images),
                    "critic_config_sha256": CRITIC_CONFIG_SHA256,
                    "critic_prompt_sha256": hashlib.sha256(
                        context.critic_prompt.encode()
                    ).hexdigest(),
                    "critic_instruction_sha256": hashlib.sha256(
                        critic_instruction.encode()
                    ).hexdigest(),
                    "advisory_schema_sha256": _response_schema_sha256(
                        ADVISORY_SCHEMA
                    ),
                }
                critic_model_evidence: dict[str, object] | None = None
                critic_request_sha256: str | None = None
                try:
                    critic_response = self._critic.complete(**critic_kwargs)
                    critic_model_evidence = cast(
                        dict[str, object],
                        canonical_json_copy(dict(critic_response.evidence)),
                    )
                    critic_request_sha256 = _actual_request_sha256_from_attempt_log(
                        critic_attempt_log,
                        observation_id=observation_id,
                        attempt_index=critic_attempt_index,
                        response_schema=ADVISORY_SCHEMA,
                    )
                    decision["critic_request_sha256"] = critic_request_sha256
                    advisory = validate_advisory(critic_response.command)
                    advisory = cast(dict[str, object], canonical_json_copy(advisory))
                    advisory_sha256 = strict_canonical_sha256(advisory)
                    decision["advisory_sha256"] = advisory_sha256
                    context.critic_successes += 1
                    critic_record.update({
                        "status": "available",
                        "advisory": advisory,
                        "advisory_sha256": advisory_sha256,
                        "critic_request_sha256": critic_request_sha256,
                        "critic_request_linkage_status": "complete",
                        "model_evidence": critic_model_evidence,
                        "failure_class": None,
                        "consumed_by_controller_observation_id": observation_id,
                        "consumed_by_controller_request_sha256": None,
                        "consumed_by_controller_call_manifest_sha256": None,
                        "consumed_by_controller_linkage_status": None,
                        "qwen_command_sha256": None,
                    })
                except Exception as error:
                    context.critic_unavailable += 1
                    advisory = None
                    status = (
                        "critic_linkage_incomplete"
                        if isinstance(error, _AttemptEvidenceLinkageIncomplete)
                        else "critic_unavailable"
                    )
                    critic_record.update({
                        "status": status,
                        "advisory": None,
                        "advisory_sha256": None,
                        "critic_request_sha256": critic_request_sha256,
                        "critic_request_linkage_status": (
                            "incomplete" if status == "critic_linkage_incomplete" else None
                        ),
                        "model_evidence": critic_model_evidence,
                        "failure_class": type(error).__name__,
                        "failure_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
                        "consumed_by_controller_observation_id": None,
                        "consumed_by_controller_request_sha256": None,
                        "consumed_by_controller_call_manifest_sha256": None,
                        "consumed_by_controller_linkage_status": None,
                        "qwen_command_sha256": None,
                    })
                finally:
                    _persist_captured_attempt_records(
                        context,
                        critic_attempt_log,
                        role="critic",
                        call_index=int(decision["critic_attempt_index"]),
                    )
                critic_record["latency_s"] = time.monotonic() - critic_started
                context.advice_by_observation[observation_id] = advisory
                context.critic_records.append(critic_record)

            consumed_advisory_id, advisory = context.take_advisory(observation_id)
            controller_kwargs = dict(kwargs)
            controller_kwargs.pop("proposal_audit_context", None)
            controller_kwargs["instruction"] = _controller_instruction(
                str(kwargs["instruction"]),
                advisory_id=consumed_advisory_id,
                advisory=advisory,
            )
            controller_call_manifest_sha256 = _model_call_manifest_sha256(
                controller_kwargs
            )
            controller_instruction_sha256 = hashlib.sha256(
                str(controller_kwargs["instruction"]).encode()
            ).hexdigest()
            controller_call_kwargs, controller_attempt_log = _with_captured_attempt_log(
                controller_kwargs
            )
            critic_request_sha256 = (
                str(decision["critic_request_sha256"])
                if consumed_advisory_id is not None
                and isinstance(decision.get("critic_request_sha256"), str)
                else None
            )
            critic_call_manifest_sha256 = (
                str(decision["critic_call_manifest_sha256"])
                if consumed_advisory_id is not None
                and isinstance(decision.get("critic_call_manifest_sha256"), str)
                else None
            )
            advisory_sha256 = (
                strict_canonical_sha256(advisory)
                if consumed_advisory_id is not None and advisory is not None
                else None
            )
            context.controller_calls += 1
            context.model_calls += 1
            controller_attempt_index = int(
                controller_call_kwargs.get("attempt_index", 0)
            )
            try:
                response = self._controller.complete(**controller_call_kwargs)
            except Exception as error:
                controller_request_sha256, controller_linkage_status = (
                    _request_linkage_from_attempt_log(
                        controller_attempt_log,
                        observation_id=observation_id,
                        attempt_index=controller_attempt_index,
                        response_schema=controller_kwargs.get("response_schema"),
                    )
                )
                failure_sha256 = hashlib.sha256(str(error).encode()).hexdigest()
                failure_evidence = {
                    "controller_role": "sole_direct_inspect_command_emitter",
                    "controller_call_index": context.controller_calls,
                    "qwen_attempt_index": controller_attempt_index,
                    "critic_trigger_evaluation": decision,
                    "consumed_advisory_id": consumed_advisory_id,
                    "critic_advisory": advisory,
                    "controller_input_public_state_sha256": canonical_sha256(
                        public_state
                    ),
                    "controller_input_image_sha256": image_hashes(images),
                    "controller_request_sha256": controller_request_sha256,
                    "controller_request_linkage_status": controller_linkage_status,
                    "controller_call_manifest_sha256": controller_call_manifest_sha256,
                    "controller_instruction_sha256": controller_instruction_sha256,
                    "rendered_advisory_sha256": advisory_sha256,
                    "critic_request_sha256": critic_request_sha256,
                    "critic_call_manifest_sha256": critic_call_manifest_sha256,
                    "advisory_sha256": advisory_sha256,
                    "qwen_command_sha256": None,
                    "critic_config_sha256": CRITIC_CONFIG_SHA256,
                    "controller_response_schema_sha256": _response_schema_sha256(
                        controller_kwargs.get("response_schema")
                    ),
                    "failure_class": type(error).__name__,
                    "failure_sha256": failure_sha256,
                }
                context.stage_controller_failure(
                    observation_id,
                    consumed_advisory_id,
                    canonical_sha256(failure_evidence),
                    controller_request_sha256=controller_request_sha256,
                    critic_request_sha256=critic_request_sha256,
                    controller_call_manifest_sha256=controller_call_manifest_sha256,
                    critic_call_manifest_sha256=critic_call_manifest_sha256,
                    advisory_sha256=advisory_sha256,
                    controller_request_linkage_status=controller_linkage_status,
                    qwen_attempt_index=controller_attempt_index,
                    failure_class=type(error).__name__,
                    failure_sha256=failure_sha256,
                )
                _persist_captured_attempt_records(
                    context,
                    controller_attempt_log,
                    role="controller",
                    call_index=context.controller_calls,
                )
                raise
            controller_model_evidence = cast(
                dict[str, object], canonical_json_copy(dict(response.evidence))
            )
            controller_request_sha256, controller_linkage_status = (
                _request_linkage_from_attempt_log(
                    controller_attempt_log,
                    observation_id=observation_id,
                    attempt_index=controller_attempt_index,
                    response_schema=controller_kwargs.get("response_schema"),
                )
            )
            command_sha256 = strict_canonical_sha256(response.command)
            evidence = controller_model_evidence
            evidence.update({
                "controller_role": "sole_direct_inspect_command_emitter",
                "controller_call_index": context.controller_calls,
                "qwen_attempt_index": controller_attempt_index,
                "critic_trigger_evaluation": decision,
                "consumed_advisory_id": consumed_advisory_id,
                "critic_advisory": advisory,
                "controller_input_public_state_sha256": canonical_sha256(public_state),
                "controller_input_image_sha256": image_hashes(images),
                "controller_request_sha256": controller_request_sha256,
                "controller_request_linkage_status": controller_linkage_status,
                "controller_call_manifest_sha256": controller_call_manifest_sha256,
                "controller_instruction_sha256": controller_instruction_sha256,
                "rendered_advisory_sha256": advisory_sha256,
                "critic_request_sha256": critic_request_sha256,
                "critic_call_manifest_sha256": critic_call_manifest_sha256,
                "advisory_sha256": advisory_sha256,
                "qwen_command_sha256": command_sha256,
                "critic_config_sha256": CRITIC_CONFIG_SHA256,
                "controller_response_schema_sha256": _response_schema_sha256(
                    controller_kwargs.get("response_schema")
                ),
            })
            context.stage_controller_return(
                observation_id,
                response.command,
                consumed_advisory_id,
                canonical_sha256(evidence),
                controller_request_sha256=controller_request_sha256,
                critic_request_sha256=critic_request_sha256,
                controller_call_manifest_sha256=controller_call_manifest_sha256,
                critic_call_manifest_sha256=critic_call_manifest_sha256,
                advisory_sha256=advisory_sha256,
                controller_request_linkage_status=controller_linkage_status,
                qwen_attempt_index=controller_attempt_index,
            )
            _persist_captured_attempt_records(
                context,
                controller_attempt_log,
                role="controller",
                call_index=context.controller_calls,
            )
            # Return the controller's direct command without translation or mutation.
            return type(response)(command=response.command, evidence=evidence)

    return DirectControllerWithCritic


def _is_safety_abort(episode: Mapping[str, object]) -> bool:
    status = str(episode.get("status", "")).lower()
    return any(
        label in status
        for label in ("safety_abort", "collision_abort", "simulator_safety")
    )


def _direct_commands(episode: Mapping[str, object]) -> list[dict[str, object]]:
    requests = episode.get("requests")
    if not isinstance(requests, list):
        return []
    commands = []
    for request in requests:
        if not isinstance(request, Mapping):
            continue
        command = request.get("post_repair_command")
        if isinstance(command, Mapping) and command.get("kind") in PHYSICAL_COMMAND_KINDS:
            commands.append(dict(command))
    return commands


def _direct_digest(commands: list[dict[str, object]]) -> str | None:
    return canonical_sha256(commands) if commands else None


def _validated_runner_mailbox(
    request: Mapping[str, object],
    *,
    draft: Mapping[str, object],
) -> tuple[dict[str, object] | None, str | None]:
    command_kind = draft.get("kind")
    raw_mailbox = request.get("mailbox")
    raw_sha256 = request.get("mailbox_sha256")
    if command_kind == "give_up":
        if raw_mailbox is not None or raw_sha256 is not None:
            raise ValueError("approved give_up unexpectedly has a mailbox")
        return None, None
    if not isinstance(raw_mailbox, Mapping):
        raise ValueError("approved proposal lacks its exact mailbox")
    mailbox = cast(dict[str, object], canonical_json_copy(raw_mailbox))
    sequence = mailbox.get("sequence")
    expected_mailbox_kinds = (
        {"move_joints"}
        if command_kind in {"cartesian_delta", "image_servo"}
        else {command_kind}
    )
    if (
        mailbox.get("schema") != "robocasa-inspect-joint-command/v1"
        or mailbox.get("kind") not in expected_mailbox_kinds
        or type(sequence) is not int
        or sequence < 0
    ):
        raise ValueError("approved proposal mailbox identity drifted")
    mailbox_sha256 = _valid_sha256(raw_sha256)
    if mailbox_sha256 != strict_canonical_sha256(mailbox):
        raise ValueError("approved proposal mailbox hash drifted")
    if command_kind == "finish":
        if set(mailbox) != {"schema", "sequence", "kind"}:
            raise ValueError("approved finish mailbox schema drifted")
        return mailbox, mailbox_sha256
    observation_id = draft.get("observation_id")
    if command_kind in {"move_joints", "cartesian_delta", "image_servo"}:
        endpoint = mailbox.get("endpoint")
        explicit_mask = mailbox.get("explicit_mask")
        gripper_open = mailbox.get("gripper_open")
        previous_gripper_open = mailbox.get("previous_gripper_open")
        tracking_mode = mailbox.get("tracking_mode", LAG_PAUSE_TRACKING)
        move_mailbox_fields = {
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
        if (
            set(mailbox) - {"tracking_mode"} != move_mailbox_fields
            or tracking_mode not in JOINT_TRACKING_MODES
            or (
                command_kind == "move_joints"
                and tracking_mode
                != draft.get("tracking_mode", LAG_PAUSE_TRACKING)
            )
            or (
                command_kind in {"cartesian_delta", "image_servo"}
                and tracking_mode != LAG_PAUSE_TRACKING
            )
            or mailbox.get("observation_id") != observation_id
            or not isinstance(endpoint, list)
            or len(endpoint) != 7
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in endpoint
            )
            or not isinstance(explicit_mask, list)
            or len(explicit_mask) != 7
            or any(type(value) is not bool for value in explicit_mask)
            or isinstance(gripper_open, bool)
            or not isinstance(gripper_open, (int, float))
            or not 0.0 <= float(gripper_open) <= 1.0
            or isinstance(previous_gripper_open, bool)
            or not isinstance(previous_gripper_open, (int, float))
            or not 0.0 <= float(previous_gripper_open) <= 1.0
            or type(mailbox.get("gripper_transition")) is not bool
            or mailbox.get("gripper_transition")
            != (abs(float(gripper_open) - float(previous_gripper_open)) > 1e-12)
            or type(mailbox.get("max_actions")) is not int
            or not 1 <= cast(int, mailbox["max_actions"]) <= 32
        ):
            raise ValueError("approved move_joints mailbox semantics drifted")
        if command_kind in {"cartesian_delta", "image_servo"}:
            expected_gripper = {
                "open": 1.0,
                "close": 0.0,
                "hold": previous_gripper_open,
            }.get(draft.get("gripper"))
            if explicit_mask != [True] * 7 or gripper_open != expected_gripper:
                raise ValueError("approved Cartesian mailbox semantics drifted")
    elif command_kind == "base_action":
        velocity = mailbox.get("normalized_velocity")
        gripper_open = mailbox.get("gripper_open")
        previous_gripper_open = mailbox.get("previous_gripper_open")
        expected_gripper = {
            "open": 1.0,
            "close": 0.0,
            "hold": previous_gripper_open,
        }.get(draft.get("gripper"))
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
            or mailbox.get("observation_id") != observation_id
            or mailbox.get("axis") != draft.get("axis")
            or isinstance(velocity, bool)
            or not isinstance(velocity, (int, float))
            or velocity != draft.get("normalized_velocity")
            or isinstance(previous_gripper_open, bool)
            or not isinstance(previous_gripper_open, (int, float))
            or not 0.0 <= float(previous_gripper_open) <= 1.0
            or isinstance(gripper_open, bool)
            or not isinstance(gripper_open, (int, float))
            or gripper_open != expected_gripper
            or type(mailbox.get("gripper_transition")) is not bool
            or mailbox.get("gripper_transition")
            != (abs(float(gripper_open) - float(previous_gripper_open)) > 1e-12)
        ):
            raise ValueError("approved base_action mailbox semantics drifted")
    else:
        raise ValueError("approved proposal mailbox kind drifted")
    return mailbox, mailbox_sha256


def _mailbox_receipt_closes(
    mailbox: Mapping[str, object],
    receipt: Mapping[str, object],
) -> bool:
    try:
        action_count_closes = cast(int, receipt["step_count"]) <= cast(
            int, mailbox["max_actions"]
        )
    except (KeyError, TypeError):
        return False
    if (
        not action_count_closes
        or mailbox.get("gripper_open") != receipt.get("gripper_intent")
    ):
        return False
    return mailbox.get("endpoint") == receipt.get("bounded_endpoint")


def _approved_runner_request_matches(
    request: Mapping[str, object],
    record: Mapping[str, object],
) -> bool:
    draft = record.get("draft")
    if not isinstance(draft, Mapping):
        return False
    repair_count = request.get("repair_count", 0)
    if type(repair_count) is not int or repair_count not in {0, 1}:
        return False
    if repair_count == 0:
        returned = request.get("command")
        evidence = request.get("evidence")
    else:
        returned = request.get("repair_command")
        evidence = request.get("repair_evidence")
    post_repair = request.get("post_repair_command")
    if not all(
        isinstance(value, Mapping)
        for value in (returned, evidence, post_repair)
    ):
        return False
    observation_id = record.get("observation_id")
    if not isinstance(observation_id, str) or not observation_id:
        return False
    try:
        normalized_draft = _normalized_controller_command(
            draft, observation_id=observation_id
        )
        normalized_post_repair = _normalized_controller_command(
            post_repair, observation_id=observation_id
        )
    except (TypeError, ValueError):
        return False
    draft_sha256 = record.get("draft_sha256")
    return (
        request.get("observation_id") == observation_id
        and strict_canonical_sha256(returned) == draft_sha256
        and _canonical_equal(normalized_post_repair, normalized_draft)
        and evidence.get("controller_call_index")
        == record.get("controller_call_index")
        and evidence.get("qwen_command_sha256") == draft_sha256
        and evidence.get("proposal_audit_verdict") == "approve"
        and evidence.get("proposal_audit_sha256") == record.get("audit_sha256")
        and evidence.get("proposal_audit_config_sha256")
        == PROPOSAL_AUDIT_CONFIG_SHA256
        and evidence.get("claimed_milestone") == record.get("claimed_milestone")
        and evidence.get("critic_origin_execution") is False
    )


def _finalize_proposal_execution_evidence(
    context: CriticContext,
    episode: Mapping[str, object],
) -> None:
    """Bind every approved draft to its kind-specific runner outcome."""

    raw_requests = episode.get("requests")
    requests = raw_requests if isinstance(raw_requests, list) else []
    raw_receipts = episode.get("receipts")
    receipts = raw_receipts if isinstance(raw_receipts, list) else []
    consumed_receipts: set[int] = set()
    consumed_requests: set[int] = set()
    for record in context.proposal_records:
        if record.get("status") != "approved_for_execution":
            continue
        draft = record.get("draft")
        if not isinstance(draft, Mapping):
            raise ValueError("approved proposal draft is malformed")
        command_kind = draft.get("kind")
        matching_requests: list[tuple[int, Mapping[str, object]]] = []
        for request_index, request in enumerate(requests):
            if not isinstance(request, Mapping):
                continue
            if _approved_runner_request_matches(request, record):
                matching_requests.append((request_index, request))
        matched = matching_requests[0] if len(matching_requests) == 1 else None
        if len(matching_requests) > 1:
            raise ValueError("approved proposal has duplicate runner requests")
        if matched is None:
            raise ValueError("approved proposal lacks one exact runner request")
        request_index, request = matched
        if request_index in consumed_requests:
            raise ValueError("runner request was replayed across approved proposals")
        consumed_requests.add(request_index)
        mailbox, mailbox_sha256 = _validated_runner_mailbox(
            request,
            draft=draft,
        )
        record["mailbox_sha256"] = mailbox_sha256

        if command_kind in PHYSICAL_COMMAND_KINDS:
            matching_receipts: list[tuple[int, dict[str, object]]] = []
            for receipt_index, raw_receipt in enumerate(receipts):
                if receipt_index in consumed_receipts or not isinstance(
                    raw_receipt, Mapping
                ):
                    continue
                try:
                    receipt = validate_public_receipt(
                        raw_receipt,
                        require_sealed_rgb=True,
                        require_torque_baseline=True,
                    )
                except (TypeError, ValueError):
                    continue
                if _receipt_matches_command(receipt, draft):
                    matching_receipts.append((receipt_index, receipt))
            if len(matching_receipts) != 1:
                if record.get("executed") is True:
                    raise ValueError(
                        "executed proposal receipt is not uniquely linked"
                    )
                continue
            receipt_index, receipt = matching_receipts[0]
            if command_kind in {"move_joints", "cartesian_delta", "image_servo"}:
                if mailbox is None or not _mailbox_receipt_closes(mailbox, receipt):
                    raise ValueError(
                        "approved move_joints mailbox/receipt linkage drifted"
                    )
            elif command_kind == "base_action" and (
                mailbox is None
                or mailbox.get("axis") != receipt.get("axis")
                or mailbox.get("normalized_velocity")
                != receipt.get("normalized_velocity")
                or mailbox.get("gripper_open") != receipt.get("gripper_intent")
            ):
                raise ValueError("approved base_action mailbox/receipt linkage drifted")
            consumed_receipts.add(receipt_index)
            record.update({
                "executed": True,
                "mailbox_count": 1,
                "action_count": 1,
                "receipt_count": 1,
                "execution_receipt_sha256": strict_canonical_sha256(receipt),
            })
        elif command_kind == "finish":
            if mailbox is None or _valid_sha256(record["mailbox_sha256"]) is None:
                raise ValueError("approved finish lacks its exact mailbox")
            terminal = episode.get("terminal_outcome")
            if not isinstance(terminal, Mapping) or terminal.get("status") not in {
                "success",
                "finished_false",
            }:
                raise ValueError("approved finish lacks a sealed terminal outcome")
            terminal_snapshot = cast(
                dict[str, object], canonical_json_copy(terminal)
            )
            simulator_terminal = terminal_snapshot.get("simulator_terminal")
            simulator_steps = episode.get("simulator_steps")
            if (
                not isinstance(simulator_terminal, Mapping)
                or simulator_terminal.get("sequence") != mailbox.get("sequence")
                or type(simulator_steps) is not int
                or simulator_steps < 0
                or simulator_terminal.get("simulator_actions") != simulator_steps
            ):
                raise ValueError("finish mailbox/terminal accounting drifted")
            record.update({
                "executed": True,
                "mailbox_count": 1,
                "action_count": 0,
                "receipt_count": 0,
                "terminal_outcome": terminal_snapshot,
                "terminal_outcome_sha256": strict_canonical_sha256(
                    terminal_snapshot
                ),
            })
        elif command_kind == "give_up":
            if mailbox is not None or record["mailbox_sha256"] is not None:
                raise ValueError("approved give_up mailbox semantics drifted")
            terminal = episode.get("terminal_outcome")
            expected = {
                "schema": "robocasa-qwen-approved-give-up/v1",
                "status": "approved_no_simulator_effect",
            }
            if not isinstance(terminal, Mapping) or dict(terminal) != expected:
                raise ValueError("approved give_up disposition drifted")
            record.update({
                "executed": True,
                "mailbox_count": 0,
                "action_count": 0,
                "receipt_count": 0,
                "terminal_outcome": expected,
                "terminal_outcome_sha256": strict_canonical_sha256(expected),
            })
        else:
            raise ValueError("approved proposal kind is outside the closed schema")

        if record.get("executed") is True and record.get("milestone_closed") is False:
            _close_proposal_milestone(context, record)
        _seal_evidence_record(record)

    if context.proposal_pending_record_index is not None:
        pending = context.proposal_records[context.proposal_pending_record_index]
        if pending.get("executed") is True:
            context.proposal_pending_record_index = None
            context.proposal_pending_command = None
            context.proposal_pending_milestone = None
            context.proposal_revisions_used = 0

    if len(consumed_receipts) != len(receipts):
        raise ValueError("episode contains an orphan proposal receipt")
    failure_fields = {
        "schema",
        "decision",
        "observation_id",
        "status",
        "failure_class",
        "failure_sha256",
        "command",
        "mailbox_count",
        "action_count",
        "receipt_count",
    }
    for request_index, raw_request in enumerate(requests):
        if request_index in consumed_requests:
            continue
        if not isinstance(raw_request, Mapping) or set(raw_request) != failure_fields:
            raise ValueError("episode contains an orphan proposal runner request")
        if (
            raw_request.get("schema")
            != "robocasa-qwen-proposal-audit-failure/v1"
            or raw_request.get("status") != "policy_failed_proposal_audit"
            or type(raw_request.get("decision")) is not int
            or cast(int, raw_request["decision"]) < 0
            or not isinstance(raw_request.get("observation_id"), str)
            or not raw_request.get("observation_id")
            or not isinstance(raw_request.get("failure_class"), str)
            or not raw_request.get("failure_class")
            or _valid_sha256(raw_request.get("failure_sha256")) is None
            or raw_request.get("command") is not None
            or any(
                type(raw_request.get(field)) is not int
                or raw_request.get(field) != 0
                for field in ("mailbox_count", "action_count", "receipt_count")
            )
        ):
            raise ValueError("proposal failure runner request drifted")


def _apply_proposal_closure_result(
    episode: dict[str, object],
    closure: Mapping[str, object],
) -> None:
    if closure.get("valid") is True:
        return
    episode["pre_closure_status"] = episode.get("status")
    episode["pre_closure_success"] = episode.get("success")
    episode["status"] = "policy_failed_proposal_audit_closure"
    episode["success"] = False


def _client_class_for_protocol(
    original: type,
    context: CriticContext,
    *,
    protocol: str,
) -> type:
    if protocol == "legacy":
        return _make_client_class(original, context)
    if protocol == "proposal":
        return _make_proposal_audit_client_class(original, context)
    raise ValueError("controller protocol must be selected explicitly")


def run_one(
    *,
    item: Mapping[str, object],
    seed: int,
    run: Path,
    max_decisions: int,
    critic_prompt: str,
    protocol: str = "proposal",
    action_budget: int = 450,
    wall_budget_s: float = 1_200.0,
    prompt_variant: str = "baseline",
) -> tuple[dict[str, object], CriticContext]:
    from robocasa_inspect.model_client import QwenClient

    from . import joint_runner

    context = CriticContext(
        task=str(item["task"]),
        family=effective_family(str(item["task"]), str(item["family"])),
        run=run,
        critic_prompt=critic_prompt,
        max_decisions=max_decisions,
    )
    context.source_perception_deadline = time.monotonic() + wall_budget_s
    try:
        episode = joint_runner.run_episode(
            task=str(item["task"]),
            seed=seed,
            run=run,
            max_decisions=max_decisions,
            client_class=_client_class_for_protocol(
                QwenClient,
                context,
                protocol=protocol,
            ),
            protocol=protocol,
            action_budget=action_budget,
            wall_budget_s=wall_budget_s,
            prompt_variant=prompt_variant,
        )
    except Exception as error:
        episode = {
            "task": item["task"],
            "seed": seed,
            "status": "runner_exception",
            "success": False,
            "receipts": [],
            "requests": [],
            "video": None,
            "video_sha256": None,
            "error_type": type(error).__name__,
            "error_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
            "error": str(error)[:500],
            "wall_s": 0.0,
        }
    if not run.exists():
        run.mkdir(parents=True, mode=0o700)
    proposal_finalization_error: ValueError | None = None
    if protocol == "proposal":
        try:
            _finalize_proposal_execution_evidence(context, episode)
        except ValueError as error:
            proposal_finalization_error = error
    commands = _direct_commands(episode)
    joint_commands = [
        command for command in commands if command.get("kind") == "move_joints"
    ]
    if protocol == "proposal":
        try:
            if proposal_finalization_error is not None:
                raise proposal_finalization_error
            perception_closure = _validate_source_perception_evidence(context, episode.get('served_model_id', ''))
            proposal_closure: dict[str, object] = {
                "valid": True,
                **validate_proposal_audit_closure(
                    context,
                    require_execution_closure=True,
                    expected_served_model_id=(
                        cast(str, episode["served_model_id"])
                        if isinstance(episode.get("served_model_id"), str)
                        else ""
                    ),
                ),
            }
        except ValueError as error:
            perception_closure = {'valid':False, 'calls':len(context.source_perception_records)}
            proposal_closure = {
                "valid": False,
                "failure_class": type(error).__name__,
                "failure_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
            }
        _apply_proposal_closure_result(episode, proposal_closure)
    else:
        proposal_closure = {
            "valid": False,
            "applicable": False,
            "protocol": "legacy",
        }
    episode.update({
        'qwen_perception_calls': len(context.source_perception_records),
        'qwen_total_model_calls': context.model_calls,
        'source_perception_records': context.source_perception_records,
        'source_perception_closure': perception_closure if protocol == 'proposal' else None,
        "family": context.family,
        "qwen_direct_command_count": len(commands),
        "qwen_joint_command_count": len(joint_commands),
        "qwen_direct_command_digest": _direct_digest(commands),
        "qwen_controller_calls": context.controller_calls,
        "qwen_critic_attempts": context.critic_attempts,
        "qwen_critic_successes": context.critic_successes,
        "qwen_critic_unavailable": context.critic_unavailable,
        "critic_origin_executions": 0,
        "critic_config": CRITIC_CONFIG,
        "critic_config_sha256": CRITIC_CONFIG_SHA256,
        "proposal_audit_config": PROPOSAL_AUDIT_CONFIG,
        "proposal_audit_config_sha256": PROPOSAL_AUDIT_CONFIG_SHA256,
        "proposal_audit_schema_sha256": _response_schema_sha256(
            PROPOSAL_AUDIT_SCHEMA
        ),
        "proposal_audit_records": context.proposal_records,
        "proposal_audit_closure": proposal_closure,
        "milestone_history": context.milestone_history,
        "critic_prompt_sha256": hashlib.sha256(critic_prompt.encode()).hexdigest(),
        "controller_response_schema_sha256": _response_schema_sha256(
            CONTROLLER_RESPONSE_SCHEMA
        ),
        "advisory_schema_sha256": _response_schema_sha256(ADVISORY_SCHEMA),
        "critic_records": context.critic_records,
        "trigger_evaluations": context.trigger_evaluations,
        "controller_records": context.controller_records,
        "qwen_attempt_records": context.attempt_records,
        "failure_skills": {
            "library": json.loads((Path(__file__).resolve().parent.parent / "prompts/failure_skills.txt").read_text()),
            "cases": context.failure_skill_memory,
            "retrievals": context.failure_skill_uses,
        },
    })
    _atomic_json(run / "critic-evidence.json", {
        "schema": "robocasa-qwen-critic-evidence/v1",
        "critic_config": CRITIC_CONFIG,
        "critic_config_sha256": CRITIC_CONFIG_SHA256,
        "proposal_audit_config": PROPOSAL_AUDIT_CONFIG,
        "proposal_audit_config_sha256": PROPOSAL_AUDIT_CONFIG_SHA256,
        "proposal_audit_records": context.proposal_records,
        "proposal_audit_closure": proposal_closure,
        "milestone_history": context.milestone_history,
        "critic_prompt_sha256": episode["critic_prompt_sha256"],
        "trigger_evaluations": context.trigger_evaluations,
        "critic_records": context.critic_records,
        "controller_records": context.controller_records,
        "qwen_attempt_records": context.attempt_records,
        "critic_origin_executions": 0,
    })
    _atomic_json(run / "wrapper-result.json", episode)
    return episode, context


def _task_record(
    item: Mapping[str, object], episode: Mapping[str, object]
) -> dict[str, object]:
    receipts = episode.get("receipts")
    moved = isinstance(receipts, list) and len(receipts) > 0
    digest = episode.get("qwen_direct_command_digest")
    return {
        "task": item["task"],
        "split": item["split"],
        "family": item["family"],
        "success": episode.get("success") is True,
        "status": episode.get("status"),
        "qwen_selected_strategy": isinstance(digest, str) and len(digest) == 64,
        "trajectory_differs_from_no_model": moved,
        "video_sha256": episode.get("video_sha256"),
        "video": episode.get("video"),
        "direct_command_digest": digest,
        "direct_command_count": episode.get("qwen_direct_command_count", 0),
        "joint_command_count": episode.get("qwen_joint_command_count", 0),
        "perception_calls": episode.get("qwen_perception_calls", 0),
        "total_model_calls": episode.get("qwen_total_model_calls", 0),
        "controller_calls": episode.get("qwen_controller_calls", 0),
        "critic_attempts": episode.get("qwen_critic_attempts", 0),
        "critic_successes": episode.get("qwen_critic_successes", 0),
        "critic_unavailable": episode.get("qwen_critic_unavailable", 0),
        "critic_origin_executions": episode.get("critic_origin_executions", -1),
    }


def _episode_record(
    item: Mapping[str, object], episode: Mapping[str, object]
) -> dict[str, object]:
    video = episode.get("video")
    evidence = (
        str(Path(str(video)).parent / "critic-evidence.json")
        if isinstance(video, str) and video
        else None
    )
    return {
        "task": item["task"],
        "family": item["family"],
        "success": episode.get("success") is True,
        "status": episode.get("status"),
        "direct_command_count": episode.get("qwen_direct_command_count", 0),
        "joint_command_count": episode.get("qwen_joint_command_count", 0),
        "direct_command_digest": episode.get("qwen_direct_command_digest"),
        "perception_calls": episode.get("qwen_perception_calls", 0),
        "total_model_calls": episode.get("qwen_total_model_calls", 0),
        "controller_calls": episode.get("qwen_controller_calls", 0),
        "critic_attempts": episode.get("qwen_critic_attempts", 0),
        "critic_successes": episode.get("qwen_critic_successes", 0),
        "critic_unavailable": episode.get("qwen_critic_unavailable", 0),
        "critic_evidence": evidence,
        "video": video,
        "video_sha256": episode.get("video_sha256"),
        "wall_s": episode.get("wall_s", 0.0),
        "error_type": episode.get("error_type"),
        "error": episode.get("error"),
    }


def _not_run(item: Mapping[str, object], status: str) -> dict[str, object]:
    return {
        "task": item["task"],
        "split": item["split"],
        "family": item["family"],
        "success": False,
        "status": status,
        "qwen_selected_strategy": False,
        "trajectory_differs_from_no_model": False,
        "video_sha256": None,
        "critic_origin_executions": 0,
    }


def _parse_start() -> dt.datetime:
    return dt.datetime.fromisoformat(OPTIMIZATION_STARTED_AT.replace("Z", "+00:00"))


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _budget_check(
    *,
    phase: str,
    completed_tasks: int,
    run_elapsed_s: float,
    smoke_wall_s: float,
    smoke_decisions: int,
) -> dict[str, object]:
    now = _utc_now()
    total_s = OPTIMIZATION_MAX_HOURS * 3600.0
    elapsed_registered_s = (now - _parse_start()).total_seconds()
    remaining_registered_s = total_s - elapsed_registered_s
    smoke_projection_s = (
        smoke_wall_s / smoke_decisions * PRIOR_MEAN_DECISIONS_PER_TASK * 10
        if smoke_wall_s > 0 and smoke_decisions > 0
        else 0.0
    )
    critic_ceiling_s = 10 * CRITIC_MAX_ATTEMPTS_PER_TASK * CRITIC_TIMEOUT_S
    if completed_tasks == 0:
        if smoke_projection_s > 0:
            projection_s = smoke_projection_s * 1.5 + critic_ceiling_s
            projection_basis = "current_joint_smoke_1_5x"
        else:
            projection_s = PRIOR_FULL_RUN_WALL_S + critic_ceiling_s
            projection_basis = "prior_full_run"
    else:
        remaining_tasks = 10 - completed_tasks
        observed_per_task_s = run_elapsed_s / completed_tasks
        if smoke_projection_s > 0:
            conservative_per_task_s = max(
                observed_per_task_s * 1.25,
                smoke_projection_s / 10.0 * 1.5,
            ) + CRITIC_MAX_ATTEMPTS_PER_TASK * CRITIC_TIMEOUT_S
            projection_basis = "current_joint_observed_1_25x"
        else:
            conservative_per_task_s = max(
                observed_per_task_s,
                PRIOR_FULL_RUN_WALL_S / 10.0
                + CRITIC_MAX_ATTEMPTS_PER_TASK * CRITIC_TIMEOUT_S,
            )
            projection_basis = "prior_full_run"
        projection_s = conservative_per_task_s * remaining_tasks
    decision = "continue" if projection_s <= remaining_registered_s else "abort"
    return {
        "schema": "robocasa-qwen-budget-check/v1",
        "phase": phase,
        "source_path": "experiment-log.yaml:started_at",
        "started_at_utc": OPTIMIZATION_STARTED_AT,
        "now_utc": now.isoformat().replace("+00:00", "Z"),
        "max_hours": OPTIMIZATION_MAX_HOURS,
        "elapsed_registered_s": elapsed_registered_s,
        "remaining_registered_s": remaining_registered_s,
        "completed_tasks": completed_tasks,
        "run_elapsed_s": run_elapsed_s,
        "prior_full_run_wall_s": PRIOR_FULL_RUN_WALL_S,
        "smoke_wall_s": smoke_wall_s,
        "smoke_decisions": smoke_decisions,
        "smoke_mean_decisions_per_task": PRIOR_MEAN_DECISIONS_PER_TASK,
        "smoke_projection_s": smoke_projection_s,
        "critic_ceiling_s": critic_ceiling_s,
        "projection_basis": projection_basis,
        "projection_s": projection_s,
        "decision": decision,
    }


_SMOKE_REQUEST_FIELDS = {
    "decision",
    "observation_id",
    "command",
    "evidence",
    "format_errors",
    "repair_count",
    "repair_reason",
    "repair_command",
    "repair_evidence",
    "post_repair_command",
    "mailbox",
    "mailbox_sha256",
}
_SMOKE_IMAGE_LABELS = {"left", "right", "wrist"}
_GROUNDING_NOTE_TERMS = (
    "jacobian",
    "torque",
    "wrench",
    "calibrat",
    "arm_joint_position",
    "arm_joint_velocity",
    "arm_applied_torque",
    "end_effector_external_pixels",
    "requested_targets",
    "realized_arm_qpos_delta",
    "maximum_commanded_step",
    "endpoint_error",
    "tracking_pause_count",
    "gripper_residual",
    "telemetry_summary",
    "mean_absolute_rgb_change",
)


def _smoke_int(value: object, *, minimum: int = 0) -> int | None:
    if type(value) is not int or value < minimum:
        return None
    return value


def _smoke_qwen_attempt_index(value: object) -> int | None:
    index = _smoke_int(value)
    if index is None or index > _QWEN_MAX_ATTEMPT_INDEX:
        return None
    return index


def _smoke_nonnegative_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0.0 else None


def _evidence_sha256(value: object) -> str | None:
    digest = _valid_sha256(value)
    if digest is None or value != digest or len(set(digest)) == 1:
        return None
    return digest


def _smoke_image_hashes(value: object) -> dict[str, str] | None:
    if not isinstance(value, Mapping) or set(value) != _SMOKE_IMAGE_LABELS:
        return None
    output: dict[str, str] = {}
    for label in sorted(_SMOKE_IMAGE_LABELS):
        digest = _evidence_sha256(value[label])
        if digest is None:
            return None
        output[label] = digest
    return output


def _canonical_equal(left: object, right: object) -> bool:
    try:
        return strict_canonical_sha256(left) == strict_canonical_sha256(right)
    except (TypeError, ValueError, OverflowError):
        return False


def _normalized_controller_command(
    value: object, *, observation_id: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("controller command must be a mapping")
    kind = value.get("kind")
    if kind == "move_joints":
        decoded = decode_joint_command(value, observation_id=observation_id)
        return {
            "kind": decoded.kind,
            "observation_id": decoded.observation_id,
            "targets": dict(decoded.targets),
            "note": decoded.note,
            "tracking_mode": decoded.tracking_mode,
        }
    if kind == "cartesian_delta":
        decoded = decode_cartesian_delta(value, observation_id=observation_id)
        return {
            "kind": decoded.kind,
            "observation_id": decoded.observation_id,
            "translation_m": list(decoded.translation_m),
            "rotation_axis_angle_rad": list(decoded.rotation_axis_angle_rad),
            "gripper": decoded.gripper,
            "note": decoded.note,
        }
    if kind == "image_servo":
        decoded = decode_image_servo(value, observation_id=observation_id)
        return {
            "kind": decoded.kind,
            "observation_id": decoded.observation_id,
            "camera": decoded.camera,
            "target_pixel": list(decoded.target_pixel),
            "target_role": decoded.target_role,
            "depth_delta_m": decoded.depth_delta_m,
            "step_m": decoded.step_m,
            "gripper": decoded.gripper,
            "note": decoded.note,
            **(
                {"other_view_pixel": list(decoded.other_view_pixel),
                 **({"other_view_camera": decoded.other_view_camera} if decoded.other_view_camera is not None else {})}
                if decoded.other_view_pixel is not None
                else {}
            ),
        }
    note = value.get("note")
    if not isinstance(note, str) or not 1 <= len(note) <= 320:
        raise ValueError("controller note is invalid")
    if value.get("observation_id") != observation_id:
        raise ValueError("controller observation is stale")
    if kind == "base_action":
        if set(value) != {
            "kind",
            "observation_id",
            "axis",
            "normalized_velocity",
            "gripper",
            "note",
        }:
            raise ValueError("base command schema drifted")
        velocity = value.get("normalized_velocity")
        if (
            isinstance(velocity, bool)
            or not isinstance(velocity, (int, float))
            or not math.isfinite(float(velocity))
            or not 0.0 < abs(float(velocity)) <= 0.25
            or value.get("axis") not in {"x", "y", "yaw"}
            or value.get("gripper") not in {"open", "hold", "close"}
        ):
            raise ValueError("base command values are invalid")
        return {
            "kind": "base_action",
            "observation_id": observation_id,
            "axis": value["axis"],
            "normalized_velocity": float(velocity),
            "gripper": value["gripper"],
            "note": note,
        }
    if kind in {"finish", "give_up"} and set(value) == {
        "kind",
        "observation_id",
        "note",
    }:
        return {"kind": kind, "observation_id": observation_id, "note": note}
    raise ValueError("controller command kind or schema is invalid")


def _selected_request_output(
    request: Mapping[str, object],
) -> tuple[dict[str, object], Mapping[str, object], dict[str, object]]:
    if set(request) != _SMOKE_REQUEST_FIELDS:
        raise ValueError("successful smoke request schema drifted")
    observation_id = request.get("observation_id")
    if not isinstance(observation_id, str) or not observation_id:
        raise ValueError("smoke request observation is invalid")
    repair_count = _smoke_int(request.get("repair_count"))
    if repair_count not in {0, 1} or not isinstance(request.get("format_errors"), list):
        raise ValueError("smoke repair accounting is invalid")
    repair_command = request.get("repair_command")
    repair_evidence = request.get("repair_evidence")
    if repair_command is None:
        if repair_evidence is not None:
            raise ValueError("repair evidence exists without a repair command")
        selected = request.get("command")
        evidence = request.get("evidence")
    else:
        if repair_count != 1:
            raise ValueError("repair command exists without one repair")
        selected = repair_command
        evidence = repair_evidence
    post_repair = request.get("post_repair_command")
    selected_normalized = _normalized_controller_command(
        selected, observation_id=observation_id
    )
    post_normalized = _normalized_controller_command(
        post_repair, observation_id=observation_id
    )
    if selected_normalized != post_normalized or not isinstance(evidence, Mapping):
        raise ValueError("executed command did not come from the selected Qwen output")
    assert isinstance(selected, Mapping) and isinstance(post_repair, Mapping)
    return dict(selected), evidence, dict(post_repair)


def _receipt_matches_command(
    receipt: Mapping[str, object], command: Mapping[str, object]
) -> bool:
    if (
        receipt.get("accepted") is not True
        or receipt.get("kind") != command.get("kind")
        or receipt.get("observation_id") != command.get("observation_id")
        or receipt.get("note") != command.get("note")
    ):
        return False
    if command.get("kind") == "move_joints":
        targets = command.get("targets")
        requested = receipt.get("requested_targets")
        if not isinstance(targets, Mapping) or not isinstance(requested, Mapping):
            return False
        try:
            return {
                name: float(value) for name, value in targets.items()
            } == dict(requested)
        except (TypeError, ValueError):
            return False
    if command.get("kind") == "cartesian_delta":
        try:
            translation = [float(item) for item in command["translation_m"]]
            rotation = [
                float(item) for item in command["rotation_axis_angle_rad"]
            ]
        except (KeyError, TypeError, ValueError):
            return False
        return (
            receipt.get("requested_translation_m") == translation
            and receipt.get("requested_rotation_axis_angle_rad") == rotation
            and receipt.get("requested_gripper") == command.get("gripper")
        )
    if command.get("kind") == "image_servo":
        try:
            target_pixel = [float(item) for item in command["target_pixel"]]
            depth_delta = float(command["depth_delta_m"])
            step = float(command["step_m"])
        except (KeyError, TypeError, ValueError):
            return False
        raw_other = command.get("other_view_pixel")
        try:
            other_pixel = (
                None if raw_other is None else [float(item) for item in raw_other]
            )
        except (TypeError, ValueError):
            return False
        return (
            receipt.get("requested_camera") == command.get("camera")
            and receipt.get("requested_target_pixel") == target_pixel
            and receipt.get("requested_target_role") == command.get("target_role")
            and receipt.get("requested_depth_delta_m") == depth_delta
            and receipt.get("requested_step_m") == step
            and receipt.get("requested_gripper") == command.get("gripper")
            and receipt.get("requested_other_view_pixel") == other_pixel
            and receipt.get("requested_other_view_camera") == command.get("other_view_camera")
        )
    return (
        receipt.get("axis") == command.get("axis")
        and receipt.get("normalized_velocity") == command.get("normalized_velocity")
    )


def _smoke_controller_entries(
    episode: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], bool, bool]:
    requests = episode.get("requests")
    receipts = episode.get("receipts")
    controller_records = episode.get("controller_records")
    trigger_evaluations = episode.get("trigger_evaluations")
    if not all(
        isinstance(value, list)
        for value in (requests, receipts, controller_records, trigger_evaluations)
    ):
        return [], [], False, False
    assert isinstance(requests, list)
    assert isinstance(receipts, list)
    assert isinstance(controller_records, list)
    assert isinstance(trigger_evaluations, list)
    try:
        validated_receipts = [
            validate_public_receipt(receipt, require_sealed_rgb=True)
            for receipt in receipts
        ]
        entries: list[dict[str, object]] = []
        physical_entries: list[dict[str, object]] = []
        for request_index, raw_request in enumerate(requests):
            if not isinstance(raw_request, Mapping):
                continue
            post_repair = raw_request.get("post_repair_command")
            if not isinstance(post_repair, Mapping):
                continue
            selected, evidence, executed = _selected_request_output(raw_request)
            mailbox, _mailbox_sha256 = _validated_runner_mailbox(
                raw_request,
                draft=executed,
            )
            observation_id = str(raw_request["observation_id"])
            command_sha256 = strict_canonical_sha256(selected)
            controller_request_sha256 = _evidence_sha256(
                evidence.get("controller_request_sha256")
            )
            controller_manifest_sha256 = _evidence_sha256(
                evidence.get("controller_call_manifest_sha256")
            )
            instruction_sha256 = _evidence_sha256(
                evidence.get("controller_instruction_sha256")
            )
            if (
                evidence.get("controller_role")
                != "sole_direct_inspect_command_emitter"
                or evidence.get("controller_request_linkage_status") != "complete"
                or controller_request_sha256 is None
                or controller_manifest_sha256 is None
                or instruction_sha256 is None
                or evidence.get("qwen_command_sha256") != command_sha256
                or evidence.get("critic_config_sha256") != CRITIC_CONFIG_SHA256
                or evidence.get("controller_response_schema_sha256")
                != _response_schema_sha256(CONTROLLER_RESPONSE_SCHEMA)
            ):
                raise ValueError("controller evidence hashes or authority drifted")
            image_sha256 = _smoke_image_hashes(
                evidence.get("controller_input_image_sha256")
            )
            if image_sha256 is None:
                raise ValueError("controller image hashes are incomplete")
            decision = evidence.get("critic_trigger_evaluation")
            decision_index = _smoke_int(raw_request.get("decision"))
            if (
                not isinstance(decision, Mapping)
                or decision_index is None
                or decision_index >= len(trigger_evaluations)
                or not _canonical_equal(decision, trigger_evaluations[decision_index])
                or decision.get("observation_id") != observation_id
            ):
                raise ValueError("controller trigger evaluation is stale or reordered")
            public_state = validate_public_state(
                decision.get("fresh_public_state"), require_torque_available=False
            )
            public_state_sha256 = strict_canonical_sha256(public_state)
            if (
                decision.get("fresh_public_state_sha256") != public_state_sha256
                or evidence.get("controller_input_public_state_sha256")
                != public_state_sha256
            ):
                raise ValueError("controller public state hash drifted")
            consumed_advisory_id = evidence.get("consumed_advisory_id")
            advisory = evidence.get("critic_advisory")
            rendered_advisory_sha256 = evidence.get("rendered_advisory_sha256")
            if consumed_advisory_id is None:
                if advisory is not None or rendered_advisory_sha256 is not None:
                    raise ValueError("unadvised controller request rendered advice")
            else:
                if (
                    not isinstance(consumed_advisory_id, str)
                    or not isinstance(advisory, Mapping)
                    or validate_advisory(advisory) != dict(advisory)
                    or rendered_advisory_sha256 != strict_canonical_sha256(advisory)
                ):
                    raise ValueError("advised controller rendering drifted")
            call_index = _smoke_int(evidence.get("controller_call_index"), minimum=1)
            matching_records = [
                record
                for record in controller_records
                if isinstance(record, Mapping)
                and record.get("controller_call_index") == call_index
                and record.get("observation_id") == observation_id
                and record.get("command_sha256") == command_sha256
            ]
            if len(matching_records) != 1:
                raise ValueError("Qwen command lacks one exact controller record")
            controller_record = matching_records[0]
            evidence_attempt_index = _smoke_qwen_attempt_index(
                evidence.get("qwen_attempt_index")
            )
            record_attempt_index = _smoke_qwen_attempt_index(
                controller_record.get("qwen_attempt_index")
            )
            if (
                not _canonical_equal(controller_record.get("command"), selected)
                or evidence_attempt_index is None
                or record_attempt_index != evidence_attempt_index
                or controller_record.get("command_evidence_sha256")
                != strict_canonical_sha256(evidence)
                or controller_record.get("controller_request_sha256")
                != controller_request_sha256
                or controller_record.get("controller_call_manifest_sha256")
                != controller_manifest_sha256
                or controller_record.get("controller_request_linkage_status")
                != "complete"
                or controller_record.get("consumed_advisory_id")
                != consumed_advisory_id
            ):
                raise ValueError("controller record does not seal the selected output")
            entry: dict[str, object] = {
                "request_index": request_index,
                "request": raw_request,
                "selected_command": selected,
                "executed_command": executed,
                "evidence": evidence,
                "record": controller_record,
                "decision": decision,
                "public_state": public_state,
                "image_sha256": image_sha256,
                "receipt": None,
            }
            if executed.get("kind") in PHYSICAL_COMMAND_KINDS:
                matching_receipts = [
                    receipt
                    for receipt in validated_receipts
                    if receipt.get("observation_id") == observation_id
                    and receipt.get("kind") == executed.get("kind")
                ]
                if len(matching_receipts) != 1 or not _receipt_matches_command(
                    matching_receipts[0], executed
                ):
                    raise ValueError("executed Qwen command lacks one sealed receipt")
                receipt = matching_receipts[0]
                if mailbox is None:
                    raise ValueError("executed Qwen command lacks its exact mailbox")
                if executed.get("kind") in {
                    "move_joints", "cartesian_delta", "image_servo",
                } and not _mailbox_receipt_closes(mailbox, receipt):
                    raise ValueError("joint mailbox and receipt do not close")
                if executed.get("kind") == "base_action" and (
                    mailbox.get("axis") != receipt.get("axis")
                    or mailbox.get("normalized_velocity")
                    != receipt.get("normalized_velocity")
                    or mailbox.get("gripper_open") != receipt.get("gripper_intent")
                ):
                    raise ValueError("base mailbox and receipt do not close")
                entry["receipt"] = matching_receipts[0]
                physical_entries.append(entry)
            entries.append(entry)
        if len(physical_entries) != len(validated_receipts):
            raise ValueError("receipt and executed physical-command counts differ")
        successful_records = [
            record
            for record in controller_records
            if isinstance(record, Mapping) and isinstance(record.get("command"), Mapping)
        ]
        request_outputs: list[tuple[Mapping[str, object], Mapping[str, object]]] = []
        for raw_request in requests:
            if not isinstance(raw_request, Mapping):
                continue
            for command_key, evidence_key in (
                ("command", "evidence"),
                ("repair_command", "repair_evidence"),
            ):
                command = raw_request.get(command_key)
                evidence = raw_request.get(evidence_key)
                if isinstance(command, Mapping) and isinstance(evidence, Mapping):
                    request_outputs.append((command, evidence))
        matched_record_indexes: list[int] = []
        for command, evidence in request_outputs:
            call_index = _smoke_int(evidence.get("controller_call_index"), minimum=1)
            command_sha256 = strict_canonical_sha256(command)
            matches = [
                index
                for index, record in enumerate(successful_records)
                if record.get("controller_call_index") == call_index
                and record.get("observation_id") == command.get("observation_id")
                and record.get("command_sha256") == command_sha256
                and _canonical_equal(record.get("command"), command)
                and record.get("command_evidence_sha256")
                == strict_canonical_sha256(evidence)
            ]
            if len(matches) != 1:
                raise ValueError("controller output lacks one exact successful record")
            matched_record_indexes.extend(matches)
        if sorted(matched_record_indexes) != list(range(len(successful_records))):
            raise ValueError("successful controller records are not a closed bijection")
        direct_commands = [
            cast(dict[str, object], entry["executed_command"])
            for entry in physical_entries
        ]
        joint_entries = [
            entry
            for entry in physical_entries
            if cast(Mapping[str, object], entry["executed_command"]).get("kind")
            == "move_joints"
        ]
        direct_count = _smoke_int(episode.get("qwen_direct_command_count"))
        joint_count = _smoke_int(episode.get("qwen_joint_command_count"))
        authority_valid = (
            direct_count == len(physical_entries)
            and joint_count == len(joint_entries)
            and episode.get("qwen_direct_command_digest")
            == strict_canonical_sha256(direct_commands)
            and len(joint_entries) >= 2
        )
        return entries, physical_entries, authority_valid, len(entries) >= 2
    except (TypeError, ValueError, KeyError, IndexError, OverflowError):
        return [], [], False, False


def _model_evidence_identity_matches(
    evidence: object,
    *,
    episode: Mapping[str, object],
    system_prompt_sha256: object,
    instruction_sha256: object,
    image_sha256: object,
    response_schema_sha256: object,
) -> bool:
    if not isinstance(evidence, Mapping):
        return False
    usage = evidence.get("usage")
    completion_tokens = (
        usage.get("completion_tokens") if isinstance(usage, Mapping) else None
    )
    raw_chars = _smoke_int(evidence.get("raw_chars"), minimum=1)
    latency_s = _smoke_nonnegative_number(evidence.get("latency_s"))
    return (
        evidence.get("snapshot_digest") == episode.get("snapshot_digest")
        and evidence.get("served_model_id") == episode.get("served_model_id")
        and evidence.get("system_prompt_sha256") == system_prompt_sha256
        and evidence.get("instruction_sha256") == instruction_sha256
        and _smoke_image_hashes(evidence.get("image_sha256"))
        == _smoke_image_hashes(image_sha256)
        and evidence.get("response_schema_sha256") == response_schema_sha256
        and evidence.get("finish_reason") == "stop"
        and _smoke_int(completion_tokens) is not None
        and raw_chars is not None
        and latency_s is not None
    )


def _decoded_sanitized_command(value: object) -> object:
    if not isinstance(value, str) or not value or len(value) > 100_000:
        raise ValueError("sanitized Qwen output is missing or unbounded")
    decoded = json.loads(value.rsplit("</think>", 1)[-1].strip())
    if not isinstance(decoded, dict):
        raise ValueError("sanitized Qwen output is not an object")
    return decoded


def _attempt_entry_matches(
    entry: object,
    *,
    role: str,
    call_index: int,
    observation_id: object,
    qwen_attempt_index: object,
    request_sha256: object,
    response_schema_sha256: object,
    output: object,
    model_evidence: object,
    episode: Mapping[str, object],
    system_prompt_sha256: object,
    instruction_sha256: object,
    image_sha256: object,
) -> bool:
    if not isinstance(entry, Mapping) or set(entry) != _ATTEMPT_RECORD_FIELDS:
        return False
    record = entry.get("record")
    if (
        entry.get("schema") != "robocasa-qwen-attempt-evidence/v1"
        or entry.get("role") != role
        or _smoke_int(entry.get("call_index"), minimum=1) != call_index
        or not isinstance(record, Mapping)
        or set(record) != _ATTEMPT_EVIDENCE_FIELDS
        or entry.get("record_sha256") != strict_canonical_sha256(record)
    ):
        return False
    latency = record.get("latency_s")
    sanitized_raw_command = record.get("sanitized_raw_command")
    usage = (
        model_evidence.get("usage")
        if isinstance(model_evidence, Mapping)
        else None
    )
    completion_tokens = (
        usage.get("completion_tokens") if isinstance(usage, Mapping) else None
    )
    expected_attempt_index = _smoke_qwen_attempt_index(qwen_attempt_index)
    logged_attempt_index = (
        _smoke_qwen_attempt_index(record.get("attempt_index"))
        if isinstance(record, Mapping)
        else None
    )
    model_completion_tokens = _smoke_int(completion_tokens)
    logged_completion_tokens = (
        _smoke_int(record.get("completion_tokens"))
        if isinstance(record, Mapping)
        else None
    )
    model_raw_chars = (
        _smoke_int(model_evidence.get("raw_chars"), minimum=1)
        if isinstance(model_evidence, Mapping)
        else None
    )
    logged_latency = _smoke_nonnegative_number(latency)
    model_latency = (
        _smoke_nonnegative_number(model_evidence.get("latency_s"))
        if isinstance(model_evidence, Mapping)
        else None
    )
    completion_token_limit = (
        CONTROLLER_MAX_TOKENS if role == "controller" else CRITIC_MAX_TOKENS
    )
    return (
        _model_evidence_identity_matches(
            model_evidence,
            episode=episode,
            system_prompt_sha256=system_prompt_sha256,
            instruction_sha256=instruction_sha256,
            image_sha256=image_sha256,
            response_schema_sha256=response_schema_sha256,
        )
        and record.get("request_sha256") == request_sha256
        and _evidence_sha256(record.get("request_sha256")) is not None
        and record.get("observation_id") == observation_id
        and expected_attempt_index is not None
        and logged_attempt_index == expected_attempt_index
        and _evidence_sha256(record.get("raw_body_sha256")) is not None
        and _canonical_equal(_decoded_sanitized_command(
            sanitized_raw_command
        ), output)
        and isinstance(sanitized_raw_command, str)
        and hashlib.sha256(sanitized_raw_command.encode()).hexdigest()
        == cast(Mapping[str, object], model_evidence).get("raw_sha256")
        and model_raw_chars == len(sanitized_raw_command)
        and type(record.get("http_status")) is int
        and record.get("http_status") == 200
        and record.get("finish_reason") == "stop"
        and cast(Mapping[str, object], model_evidence).get("finish_reason")
        == "stop"
        and logged_completion_tokens is not None
        and logged_completion_tokens == model_completion_tokens
        and logged_completion_tokens < completion_token_limit
        and logged_latency is not None
        and logged_latency == model_latency
        and record.get("response_schema_sha256") == response_schema_sha256
        and record.get("served_model_id") == episode.get("served_model_id")
    )


def _smoke_attempt_records(episode: Mapping[str, object]) -> bool:
    attempts = episode.get("qwen_attempt_records")
    controller_records = episode.get("controller_records")
    critic_records = episode.get("critic_records")
    requests = episode.get("requests")
    if not all(
        isinstance(value, list)
        for value in (attempts, controller_records, critic_records, requests)
    ):
        return False
    assert isinstance(attempts, list)
    assert isinstance(controller_records, list)
    assert isinstance(critic_records, list)
    assert isinstance(requests, list)
    try:
        keyed_attempts: dict[tuple[str, int], object] = {}
        for entry in attempts:
            if not isinstance(entry, Mapping):
                raise ValueError("attempt membership is malformed")
            role = entry.get("role")
            call_index = _smoke_int(entry.get("call_index"), minimum=1)
            key = (str(role), call_index or 0)
            if role not in {"controller", "critic"} or call_index is None:
                raise ValueError("attempt membership identity is malformed")
            if key in keyed_attempts:
                raise ValueError("attempt membership was duplicated")
            keyed_attempts[key] = entry

        controller_outputs: dict[int, tuple[Mapping[str, object], Mapping[str, object]]] = {}
        for request in requests:
            if not isinstance(request, Mapping):
                continue
            for command_key, evidence_key in (
                ("command", "evidence"),
                ("repair_command", "repair_evidence"),
            ):
                command = request.get(command_key)
                evidence = request.get(evidence_key)
                if not isinstance(command, Mapping) or not isinstance(evidence, Mapping):
                    continue
                call_index = _smoke_int(
                    evidence.get("controller_call_index"), minimum=1
                )
                if call_index is None or call_index in controller_outputs:
                    raise ValueError("controller output call identity was duplicated")
                controller_outputs[call_index] = (command, evidence)

        record_keys: set[tuple[str, int]] = set()
        for record in controller_records:
            if not isinstance(record, Mapping):
                raise ValueError("controller record is malformed")
            call_index = _smoke_int(record.get("controller_call_index"), minimum=1)
            if call_index is None:
                raise ValueError("controller record call index is malformed")
            key = ("controller", call_index)
            record_keys.add(key)
            command = record.get("command")
            if not isinstance(command, Mapping):
                continue
            output = controller_outputs.get(call_index)
            if output is None:
                raise ValueError("successful controller record has no request outcome")
            request_command, evidence = output
            record_attempt_index = _smoke_qwen_attempt_index(
                record.get("qwen_attempt_index")
            )
            evidence_attempt_index = _smoke_qwen_attempt_index(
                evidence.get("qwen_attempt_index")
            )
            if (
                not _canonical_equal(command, request_command)
                or record_attempt_index is None
                or record_attempt_index != evidence_attempt_index
                or not _attempt_entry_matches(
                    keyed_attempts.get(key),
                    role="controller",
                    call_index=call_index,
                    observation_id=record.get("observation_id"),
                    qwen_attempt_index=evidence_attempt_index,
                    request_sha256=record.get("controller_request_sha256"),
                    response_schema_sha256=episode.get(
                        "controller_response_schema_sha256"
                    ),
                    output=command,
                    model_evidence=evidence,
                    episode=episode,
                    system_prompt_sha256=episode.get("system_prompt_sha256"),
                    instruction_sha256=evidence.get(
                        "controller_instruction_sha256"
                    ),
                    image_sha256=evidence.get("controller_input_image_sha256"),
                )
            ):
                raise ValueError("controller attempt evidence does not reconcile")

        for record in critic_records:
            if not isinstance(record, Mapping):
                raise ValueError("critic record is malformed")
            call_index = _smoke_int(record.get("attempt_index"), minimum=1)
            if call_index is None:
                raise ValueError("critic record attempt identity is malformed")
            key = ("critic", call_index)
            record_keys.add(key)
            if record.get("status") != "available":
                continue
            if not _attempt_entry_matches(
                keyed_attempts.get(key),
                role="critic",
                call_index=call_index,
                observation_id=record.get("observation_id"),
                qwen_attempt_index=record.get("qwen_attempt_index"),
                request_sha256=record.get("critic_request_sha256"),
                response_schema_sha256=episode.get("advisory_schema_sha256"),
                output=record.get("advisory"),
                model_evidence=record.get("model_evidence"),
                episode=episode,
                system_prompt_sha256=episode.get("critic_prompt_sha256"),
                instruction_sha256=record.get("critic_instruction_sha256"),
                image_sha256=record.get("input_image_sha256"),
            ):
                raise ValueError("critic attempt evidence does not reconcile")
        if not set(keyed_attempts).issubset(record_keys):
            raise ValueError("attempt membership has no controller/critic outcome")
        successful_keys = {
            ("controller", int(record["controller_call_index"]))
            for record in controller_records
            if isinstance(record, Mapping) and isinstance(record.get("command"), Mapping)
        } | {
            ("critic", int(record["attempt_index"]))
            for record in critic_records
            if isinstance(record, Mapping) and record.get("status") == "available"
        }
        return successful_keys.issubset(keyed_attempts)
    except (
        json.JSONDecodeError,
        TypeError,
        ValueError,
        KeyError,
        OverflowError,
    ):
        return False


def _critic_records_are_closed(episode: Mapping[str, object]) -> bool:
    critic_records = episode.get("critic_records")
    controller_records = episode.get("controller_records")
    trigger_evaluations = episode.get("trigger_evaluations")
    if not all(
        isinstance(value, list)
        for value in (critic_records, controller_records, trigger_evaluations)
    ):
        return False
    assert isinstance(critic_records, list)
    assert isinstance(controller_records, list)
    assert isinstance(trigger_evaluations, list)
    if not 1 <= len(critic_records) <= CRITIC_MAX_ATTEMPTS_PER_TASK:
        return False
    try:
        available_identities: set[tuple[object, object, object]] = set()
        for expected_index, record in enumerate(critic_records, start=1):
            if (
                not isinstance(record, Mapping)
                or _smoke_int(record.get("attempt_index"), minimum=1)
                != expected_index
                or record.get("status")
                not in {"available", "critic_unavailable", "critic_linkage_incomplete"}
            ):
                raise ValueError("critic record sequence or status drifted")
            if record.get("status") != "available":
                continue
            observation_id = record.get("observation_id")
            advisory_id = record.get("advisory_id")
            trigger = record.get("trigger")
            critic_request_sha256 = _evidence_sha256(
                record.get("critic_request_sha256")
            )
            identity = (observation_id, advisory_id, critic_request_sha256)
            if (
                record.get("schema") != "robocasa-qwen-advisory-evidence/v1"
                or not isinstance(observation_id, str)
                or not isinstance(advisory_id, str)
                or trigger not in {"first_action", "repeat", "stall", "late_episode"}
                or critic_request_sha256 is None
                or identity in available_identities
            ):
                raise ValueError("available critic identity was replayed or malformed")
            available_identities.add(identity)
            decisions = [
                decision
                for decision in trigger_evaluations
                if isinstance(decision, Mapping)
                and decision.get("observation_id") == observation_id
                and decision.get("advisory_id") == advisory_id
                and decision.get("fired_trigger") == trigger
                and decision.get("critic_attempt_index") == expected_index
                and decision.get("critic_request_sha256") == critic_request_sha256
            ]
            if len(decisions) != 1:
                raise ValueError("critic record lacks one exact trigger outcome")
            same_observation = sorted(
                (
                    record
                    for record in controller_records
                    if isinstance(record, Mapping)
                    and record.get("observation_id") == observation_id
                ),
                key=lambda item: int(item["controller_call_index"]),
            )
            consumed = [
                item
                for item in controller_records
                if isinstance(item, Mapping)
                and item.get("consumed_advisory_id") == advisory_id
            ]
            if (
                not same_observation
                or len(consumed) != 1
                or consumed[0] is not same_observation[0]
                or consumed[0].get("critic_request_sha256")
                != critic_request_sha256
                or consumed[0].get("advisory_sha256")
                != record.get("advisory_sha256")
                or record.get("consumed_by_controller_request_sha256")
                != consumed[0].get("controller_request_sha256")
                or record.get("qwen_command_sha256")
                != consumed[0].get("command_sha256")
                or any(
                    later.get("consumed_advisory_id") is not None
                    or later.get("critic_request_sha256") is not None
                    or later.get("advisory_sha256") is not None
                    for later in same_observation[1:]
                )
            ):
                raise ValueError("critic advice consumption is not a closed bijection")
        return True
    except (TypeError, ValueError, KeyError):
        return False


def _critic_smoke_checks(
    episode: Mapping[str, object],
    entries: list[dict[str, object]],
    physical_entries: list[dict[str, object]],
) -> tuple[bool, bool, bool, bool]:
    if not _critic_records_are_closed(episode):
        return False, False, False, False
    critic_records = episode.get("critic_records")
    controller_records = episode.get("controller_records")
    trigger_evaluations = episode.get("trigger_evaluations")
    if not all(
        isinstance(value, list)
        for value in (critic_records, controller_records, trigger_evaluations)
    ):
        return False, False, False, False
    assert isinstance(critic_records, list)
    assert isinstance(controller_records, list)
    assert isinstance(trigger_evaluations, list)
    for raw_critic in critic_records:
        if not isinstance(raw_critic, Mapping) or raw_critic.get("status") != "available":
            continue
        try:
            observation_id = raw_critic.get("observation_id")
            advisory_id = raw_critic.get("advisory_id")
            if (
                raw_critic.get("schema")
                != "robocasa-qwen-advisory-evidence/v1"
                or not isinstance(observation_id, str)
                or not isinstance(advisory_id, str)
                or raw_critic.get("critic_request_linkage_status") != "complete"
                or raw_critic.get("trigger") != "first_action"
                or _smoke_int(raw_critic.get("attempt_index"), minimum=1) != 1
            ):
                raise ValueError("critic identity or first-action trigger drifted")
            advisory_raw = raw_critic.get("advisory")
            if not isinstance(advisory_raw, Mapping):
                raise ValueError("critic advisory is missing")
            advisory = validate_advisory(advisory_raw)
            advisory_sha256 = strict_canonical_sha256(advisory)
            critic_request_sha256 = _evidence_sha256(
                raw_critic.get("critic_request_sha256")
            )
            critic_manifest_sha256 = _evidence_sha256(
                raw_critic.get("critic_call_manifest_sha256")
            )
            if (
                critic_request_sha256 is None
                or critic_manifest_sha256 is None
                or raw_critic.get("advisory_sha256") != advisory_sha256
                or raw_critic.get("critic_config_sha256") != CRITIC_CONFIG_SHA256
                or raw_critic.get("critic_prompt_sha256")
                != episode.get("critic_prompt_sha256")
                or raw_critic.get("advisory_schema_sha256")
                != _response_schema_sha256(ADVISORY_SCHEMA)
                or not isinstance(raw_critic.get("model_evidence"), Mapping)
            ):
                raise ValueError("critic hashes or closed schema drifted")
            decisions = [
                decision
                for decision in trigger_evaluations
                if isinstance(decision, Mapping)
                and decision.get("observation_id") == observation_id
                and decision.get("advisory_id") == advisory_id
                and decision.get("fired_trigger") == "first_action"
            ]
            if len(decisions) != 1:
                raise ValueError("critic decision was replayed or missing")
            decision = decisions[0]
            current_entries = [
                entry
                for entry in entries
                if cast(Mapping[str, object], entry["selected_command"]).get(
                    "observation_id"
                )
                == observation_id
            ]
            if len(current_entries) != 1:
                raise ValueError("critic observation lacks one controller request")
            current = current_entries[0]
            prior_physical = [
                entry
                for entry in physical_entries
                if int(entry["request_index"]) < int(current["request_index"])
            ]
            if not prior_physical:
                raise ValueError("critic has no immediately preceding physical command")
            previous = prior_physical[-1]
            previous_command = previous["selected_command"]
            previous_receipt = previous["receipt"]
            if (
                not _canonical_equal(
                    decision.get("previous_executed_command"), previous_command
                )
                or decision.get("previous_executed_command_sha256")
                != strict_canonical_sha256(previous_command)
                or raw_critic.get("previous_executed_command_sha256")
                != strict_canonical_sha256(previous_command)
                or not _canonical_equal(
                    decision.get("sealed_public_receipt"), previous_receipt
                )
            ):
                raise ValueError("critic previous command/receipt was reordered")
            receipt = validate_public_receipt(
                decision.get("sealed_public_receipt"),
                require_sealed_rgb=True,
                require_torque_baseline=True,
            )
            receipt_sha256 = strict_canonical_sha256(receipt)
            state = validate_public_state(
                decision.get("fresh_public_state"), require_torque_available=True
            )
            state_sha256 = strict_canonical_sha256(state)
            image_sha256 = _smoke_image_hashes(
                raw_critic.get("fresh_public_rgb_sha256")
            )
            if (
                decision.get("sealed_public_receipt_sha256") != receipt_sha256
                or raw_critic.get("sealed_public_receipt_sha256") != receipt_sha256
                or decision.get("fresh_public_state_sha256") != state_sha256
                or raw_critic.get("fresh_public_state_sha256") != state_sha256
                or raw_critic.get("input_public_state_sha256") != state_sha256
                or image_sha256 is None
                or raw_critic.get("input_image_sha256") != image_sha256
                or current["image_sha256"] != image_sha256
            ):
                raise ValueError("critic receipt, telemetry, or RGB linkage drifted")
            same_observation_records = [
                record
                for record in controller_records
                if isinstance(record, Mapping)
                and record.get("observation_id") == observation_id
            ]
            if not same_observation_records:
                raise ValueError("critic has no next controller request")
            next_controller = same_observation_records[0]
            consumed = [
                record
                for record in controller_records
                if isinstance(record, Mapping)
                and record.get("consumed_advisory_id") == advisory_id
            ]
            if (
                len(consumed) != 1
                or consumed[0] is not next_controller
                or not _canonical_equal(next_controller, current["record"])
                or any(
                    record.get("consumed_advisory_id") is not None
                    or record.get("critic_request_sha256") is not None
                    or record.get("advisory_sha256") is not None
                    for record in same_observation_records[1:]
                )
            ):
                raise ValueError("advisory was not consumed exactly by the next request")
            next_request_sha256 = _evidence_sha256(
                next_controller.get("controller_request_sha256")
            )
            next_manifest_sha256 = _evidence_sha256(
                next_controller.get("controller_call_manifest_sha256")
            )
            command_sha256 = _evidence_sha256(next_controller.get("command_sha256"))
            evidence = cast(Mapping[str, object], current["evidence"])
            if (
                next_request_sha256 is None
                or next_manifest_sha256 is None
                or command_sha256 is None
                or next_controller.get("controller_request_linkage_status")
                != "complete"
                or next_controller.get("critic_request_sha256")
                != critic_request_sha256
                or next_controller.get("critic_call_manifest_sha256")
                != critic_manifest_sha256
                or next_controller.get("advisory_sha256") != advisory_sha256
                or raw_critic.get("consumed_by_controller_observation_id")
                != observation_id
                or raw_critic.get("consumed_by_controller_request_sha256")
                != next_request_sha256
                or raw_critic.get("consumed_by_controller_call_manifest_sha256")
                != next_manifest_sha256
                or raw_critic.get("consumed_by_controller_linkage_status")
                != "complete"
                or raw_critic.get("qwen_command_sha256") != command_sha256
                or evidence.get("critic_request_sha256") != critic_request_sha256
                or evidence.get("critic_call_manifest_sha256")
                != critic_manifest_sha256
                or evidence.get("advisory_sha256") != advisory_sha256
                or evidence.get("rendered_advisory_sha256") != advisory_sha256
                or evidence.get("controller_request_sha256")
                != next_request_sha256
            ):
                raise ValueError("actual critic/controller request hashes do not link")
            return True, True, True, True
        except (TypeError, ValueError, KeyError, IndexError, OverflowError):
            continue
    return False, False, False, False


def _owner_authority_json(
    path_value: object,
    digest_value: object,
    *,
    schema: str,
) -> dict[str, object]:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("authority path is missing")
    path = Path(path_value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("authority path is not an absolute regular file")
    stat = path.stat()
    parent_stat = path.parent.stat()
    if (
        stat.st_uid != os.getuid()
        or parent_stat.st_uid != os.getuid()
        or stat.st_mode & 0o077
        or parent_stat.st_mode & 0o077
    ):
        raise ValueError("authority artifact is not owner-only")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _evidence_sha256(digest_value):
        raise ValueError("authority artifact digest drifted")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError("authority artifact schema drifted")
    return cast(dict[str, object], canonical_json_copy(value))


def _attested_model_identity(
    episode: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    identity = _owner_authority_json(
        episode.get("model_identity_manifest_path"),
        episode.get("model_identity_manifest_sha256"),
        schema="panda-qwen-model-identity/v2",
    )
    attestation = _owner_authority_json(
        episode.get("server_attestation_path"),
        episode.get("server_attestation_sha256"),
        schema="panda-qwen-server-attestation/v1",
    )
    snapshot_path = identity.get("snapshot_path")
    launch_id = attestation.get("launch_id")
    served_model_id = attestation.get("served_model_id")
    if (
        identity.get("authority") != _FROZEN_QWEN_AUTHORITY
        or identity.get("snapshot_digest") != _FROZEN_QWEN_SNAPSHOT_DIGEST
        or not isinstance(snapshot_path, str)
        or not snapshot_path
        or attestation.get("identity_manifest_sha256")
        != strict_canonical_sha256(identity)
        or attestation.get("snapshot_digest") != identity.get("snapshot_digest")
        or attestation.get("snapshot_path") != snapshot_path
        or not isinstance(launch_id, str)
        or not launch_id
        or served_model_id
        != f"{_FROZEN_QWEN_AUTHORITY['served_model_name']}-{launch_id}"
        or episode.get("snapshot_digest") != identity.get("snapshot_digest")
        or episode.get("served_model_id") != served_model_id
    ):
        raise ValueError("episode model identity is not the attested frozen authority")
    return identity, attestation


def _smoke_prompt_model_hashes(
    episode: Mapping[str, object],
    *,
    protocol: str,
) -> bool:
    try:
        if protocol not in {"legacy", "proposal"}:
            raise ValueError("smoke protocol selection drifted")
        root = Path(__file__).resolve().parents[1]
        system_prompt_sha256 = hashlib.sha256(
            load_joint_system_prompt(root, protocol=protocol).encode()
        ).hexdigest()
        critic_prompt_path = root / "prompts" / (
            "proposal_audit_critic.txt"
            if protocol == "proposal"
            else "advisory_critic.txt"
        )
        critic_prompt_sha256 = hashlib.sha256(
            critic_prompt_path.read_bytes()
        ).hexdigest()
        _identity, _attestation = _attested_model_identity(episode)
        shared = (
            episode.get("system_prompt_sha256") == system_prompt_sha256
            and episode.get("critic_prompt_sha256") == critic_prompt_sha256
            and episode.get("controller_response_schema_sha256")
            == _response_schema_sha256(CONTROLLER_RESPONSE_SCHEMA)
            and episode.get("snapshot_digest") == _FROZEN_QWEN_SNAPSHOT_DIGEST
        )
        if protocol == "proposal":
            return (
                shared
                and _canonical_equal(
                    episode.get("proposal_audit_config"),
                    PROPOSAL_AUDIT_CONFIG,
                )
                and episode.get("proposal_audit_config_sha256")
                == PROPOSAL_AUDIT_CONFIG_SHA256
                and episode.get("proposal_audit_schema_sha256")
                == _response_schema_sha256(PROPOSAL_AUDIT_SCHEMA)
            )
        return (
            shared
            and _canonical_equal(episode.get("critic_config"), CRITIC_CONFIG)
            and episode.get("critic_config_sha256") == CRITIC_CONFIG_SHA256
            and episode.get("advisory_schema_sha256")
            == _response_schema_sha256(ADVISORY_SCHEMA)
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return False


def _smoke_artifacts_valid(episode: Mapping[str, object]) -> bool:
    video = episode.get("video")
    digest = _evidence_sha256(episode.get("video_sha256"))
    if not isinstance(video, str) or not video or digest is None:
        return False
    path = Path(video)
    try:
        return (
            path.is_absolute()
            and path.is_file()
            and not path.is_symlink()
            and path.stat().st_size > 0
            and hashlib.sha256(path.read_bytes()).hexdigest() == digest
        )
    except OSError:
        return False


def _smoke_counters_reconcile(episode: Mapping[str, object]) -> bool:
    critic_records = episode.get("critic_records")
    controller_records = episode.get("controller_records")
    if not isinstance(critic_records, list) or not isinstance(controller_records, list):
        return False
    controller_calls = _smoke_int(episode.get("qwen_controller_calls"))
    critic_attempts = _smoke_int(episode.get("qwen_critic_attempts"))
    critic_successes = _smoke_int(episode.get("qwen_critic_successes"))
    critic_unavailable = _smoke_int(episode.get("qwen_critic_unavailable"))
    indexes = [
        _smoke_int(record.get("controller_call_index"), minimum=1)
        if isinstance(record, Mapping)
        else None
        for record in controller_records
    ]
    return (
        controller_calls == len(controller_records)
        and indexes == list(range(1, len(controller_records) + 1))
        and critic_attempts == len(critic_records)
        and critic_attempts <= CRITIC_MAX_ATTEMPTS_PER_TASK
        and critic_successes
        == sum(
            isinstance(record, Mapping) and record.get("status") == "available"
            for record in critic_records
        )
        and critic_unavailable == len(critic_records) - critic_successes
        and critic_attempts == critic_successes + critic_unavailable
        and critic_successes >= 1
    )


def _legacy_smoke_pass(
    episode: Mapping[str, object],
) -> tuple[bool, dict[str, object]]:
    entries, physical_entries, authority, fresh_controller = (
        _smoke_controller_entries(episode)
    )
    first_action, complete_critic, exactly_once, actual_hashes = (
        _critic_smoke_checks(episode, entries, physical_entries)
    )
    actual_hashes = actual_hashes and _smoke_attempt_records(episode)
    grounded_note = any(
        isinstance(entry.get("executed_command"), Mapping)
        and cast(Mapping[str, object], entry["executed_command"]).get("kind")
        == "move_joints"
        and isinstance(
            cast(Mapping[str, object], entry["executed_command"]).get("note"), str
        )
        and any(
            term
            in str(
                cast(Mapping[str, object], entry["executed_command"])["note"]
            ).casefold()
            for term in _GROUNDING_NOTE_TERMS
        )
        for entry in physical_entries
    )
    critic_origin = episode.get("critic_origin_executions")
    checks = {
        "joint_commands_at_least_two": authority,
        "qwen_joint_command_authority": authority,
        "fresh_controller_grounding_evidence": fresh_controller,
        "first_action_critic_available": first_action,
        "complete_critic_receipt": complete_critic,
        "advisory_consumed": exactly_once,
        "advisory_consumed_exactly_once": exactly_once,
        "actual_request_hashes_reconcile": actual_hashes,
        "grounded_controller_note": grounded_note,
        "zero_critic_origin_executions": (
            type(critic_origin) is int and critic_origin == 0
        ),
        "counters_reconcile": _smoke_counters_reconcile(episode),
        "prompt_schema_model_hashes_valid": _smoke_prompt_model_hashes(
            episode,
            protocol="legacy",
        ),
        "no_safety_abort": not _is_safety_abort(episode),
        "artifacts_valid": _smoke_artifacts_valid(episode),
    }
    return all(checks.values()), checks


def _proposal_smoke_context(episode: Mapping[str, object]) -> CriticContext:
    def records(name: str) -> list[dict[str, object]]:
        value = episode.get(name)
        if not isinstance(value, list):
            raise ValueError(f"proposal smoke {name} is not a list")
        snapshot = canonical_json_copy(value)
        if not isinstance(snapshot, list) or any(
            not isinstance(item, dict) for item in snapshot
        ):
            raise ValueError(f"proposal smoke {name} contains a non-record")
        return cast(list[dict[str, object]], snapshot)

    task = episode.get("task")
    family = episode.get("family")
    if not isinstance(task, str) or not isinstance(family, str):
        raise ValueError("proposal smoke task/family identity drifted")
    milestone_history = episode.get("milestone_history")
    if not isinstance(milestone_history, list) or any(
        not isinstance(item, str) for item in milestone_history
    ):
        raise ValueError("proposal smoke milestone history drifted")
    controller_calls = _smoke_int(episode.get("qwen_controller_calls"))
    critic_attempts = _smoke_int(episode.get("qwen_critic_attempts"))
    critic_successes = _smoke_int(episode.get("qwen_critic_successes"))
    critic_unavailable = _smoke_int(episode.get("qwen_critic_unavailable"))
    if None in {
        controller_calls,
        critic_attempts,
        critic_successes,
        critic_unavailable,
    }:
        raise ValueError("proposal smoke counters drifted")
    return CriticContext(
        task=task,
        family=family,
        run=episode.get("artifact_root", "proposal-smoke"),
        critic_prompt="",
        max_decisions=FULL_MAX_DECISIONS,
        controller_calls=cast(int, controller_calls),
        critic_attempts=cast(int, critic_attempts),
        critic_successes=cast(int, critic_successes),
        critic_unavailable=cast(int, critic_unavailable),
        proposal_records=records("proposal_audit_records"),
        controller_records=records("controller_records"),
        critic_records=records("critic_records"),
        attempt_records=records("qwen_attempt_records"),
        milestone_history=list(milestone_history),
    )


def _proposal_smoke_pass(
    episode: Mapping[str, object],
) -> tuple[bool, dict[str, object]]:
    strict_closure = False
    rejected_zero_effect = False
    approved_closed = False
    same_observation_cap = False
    request_effect_linkage = False
    authority = False
    effect_counters = False
    no_proposal_failure_request = False
    try:
        context = _proposal_smoke_context(episode)
        _finalize_proposal_execution_evidence(context, episode)
        summary = validate_proposal_audit_closure(
            context,
            require_execution_closure=True,
            expected_served_model_id=(
                cast(str, episode["served_model_id"])
                if isinstance(episode.get("served_model_id"), str)
                else ""
            ),
        )
        expected_closure = {"valid": True, **summary}
        strict_closure = _canonical_equal(
            episode.get("proposal_audit_closure"), expected_closure
        )
        rejected_records = [
            record
            for record in context.proposal_records
            if record.get("status") != "approved_for_execution"
        ]
        approved_records = [
            record
            for record in context.proposal_records
            if record.get("status") == "approved_for_execution"
        ]
        rejected_zero_effect = all(
            record.get("executed") is False
            and type(record.get("mailbox_count")) is int
            and record.get("mailbox_count") == 0
            and type(record.get("action_count")) is int
            and record.get("action_count") == 0
            and type(record.get("receipt_count")) is int
            and record.get("receipt_count") == 0
            for record in rejected_records
        )
        approved_closed = bool(approved_records) and all(
            record.get("executed") is True
            and record.get("milestone_closed") is True
            for record in approved_records
        )
        revisions: dict[str, list[int]] = {}
        for record in context.proposal_records:
            observation_id = record.get("observation_id")
            revision_index = record.get("revision_index")
            if not isinstance(observation_id, str) or type(revision_index) is not int:
                raise ValueError("proposal smoke revision identity drifted")
            revisions.setdefault(observation_id, []).append(revision_index)
        same_observation_cap = all(
            indexes == list(range(len(indexes)))
            and len(indexes) <= PROPOSAL_MAX_REVISIONS_PER_OBSERVATION + 1
            for indexes in revisions.values()
        )
        exhausted_observations = {
            observation_id
            for observation_id, indexes in revisions.items()
            if len(indexes) == PROPOSAL_MAX_REVISIONS_PER_OBSERVATION + 1
            and all(
                record.get("status") != "approved_for_execution"
                for record in context.proposal_records
                if record.get("observation_id") == observation_id
            )
        }
        physical = [
            cast(dict[str, object], record["draft"])
            for record in approved_records
            if isinstance(record.get("draft"), dict)
            and cast(dict[str, object], record["draft"]).get("kind")
            in PHYSICAL_COMMAND_KINDS
        ]
        direct_count = _smoke_int(episode.get("qwen_direct_command_count"))
        joint_count = _smoke_int(episode.get("qwen_joint_command_count"))
        authority = (
            direct_count == len(physical)
            and joint_count
            == sum(command.get("kind") == "move_joints" for command in physical)
            and episode.get("qwen_direct_command_digest")
            == (strict_canonical_sha256(physical) if physical else None)
        )
        raw_requests = episode.get("requests")
        raw_receipts = episode.get("receipts")
        if not isinstance(raw_requests, list) or not isinstance(raw_receipts, list):
            raise ValueError("proposal smoke episode effects are not lists")
        no_proposal_failure_request = not exhausted_observations and not any(
            isinstance(request, Mapping)
            and (
                request.get("schema")
                == "robocasa-qwen-proposal-audit-failure/v1"
                or request.get("status") == "policy_failed_proposal_audit"
            )
            for request in raw_requests
        )
        step_counts = [
            _smoke_int(receipt.get("step_count"), minimum=1)
            if isinstance(receipt, Mapping)
            else None
            for receipt in raw_receipts
        ]
        effect_counters = (
            None not in step_counts
            and _smoke_int(episode.get("decisions")) == len(physical)
            and _smoke_int(episode.get("action_chunks")) == len(physical)
            and _smoke_int(episode.get("model_decisions")) == len(raw_requests)
            and _smoke_int(episode.get("simulator_steps"))
            == sum(cast(list[int], step_counts))
        )
        request_effect_linkage = True
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    critic_origin = episode.get("critic_origin_executions")
    status = episode.get("status")
    checks = {
        "explicit_proposal_protocol": True,
        "strict_proposal_closure": strict_closure,
        "rejected_drafts_zero_effect": rejected_zero_effect,
        "approved_commands_closed": approved_closed,
        "same_observation_revision_cap": same_observation_cap,
        "request_mailbox_receipt_closure": request_effect_linkage,
        "episode_effect_counters_close": effect_counters,
        "qwen_direct_command_authority": authority,
        "zero_critic_origin_executions": (
            type(critic_origin) is int and critic_origin == 0
        ),
        "prompt_schema_model_hashes_valid": _smoke_prompt_model_hashes(
            episode,
            protocol="proposal",
        ),
        "no_proposal_contract_failure": (
            isinstance(status, str)
            and not status.startswith("policy_failed_proposal_audit")
        ),
        "no_proposal_failure_request": no_proposal_failure_request,
        "no_safety_abort": not _is_safety_abort(episode),
        "artifacts_valid": _smoke_artifacts_valid(episode),
    }
    return all(checks.values()), checks


def _smoke_pass(
    episode: Mapping[str, object],
    *,
    protocol: str = "legacy",
) -> tuple[bool, dict[str, object]]:
    if protocol == "legacy":
        return _legacy_smoke_pass(episode)
    if protocol == "proposal":
        return _proposal_smoke_pass(episode)
    raise ValueError("smoke protocol must be selected explicitly")


def _validate_critic_prompt(critic_prompt: str, *, protocol: str) -> None:
    if protocol not in {"legacy", "proposal"}:
        raise ValueError("critic prompt protocol must be selected explicitly")
    root = Path(__file__).resolve().parents[1]
    expected = (
        root
        / "prompts"
        / (
            "advisory_critic.txt"
            if protocol == "legacy"
            else "proposal_audit_critic.txt"
        )
    ).read_text(encoding="utf-8")
    if critic_prompt != expected:
        raise ValueError(f"critic prompt does not match {protocol} protocol")


def _installed_release_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in [
        *sorted((root / "adaptive").glob("*.py")),
        *sorted((root / "prompts").glob("*.txt")),
        root / "benchmark" / "baseline.json",
    ]:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _validate_release_digest(claimed: object) -> str:
    if (
        not isinstance(claimed, str)
        or len(claimed) != 16
        or any(character not in "0123456789abcdef" for character in claimed)
    ):
        raise ValueError("release digest must be exactly 16 lowercase hex characters")
    measured = _installed_release_digest()
    if claimed != measured:
        raise ValueError("installed release digest does not match the candidate claim")
    return measured


def _validate_baseline_path(path: Path) -> Path:
    if not isinstance(path, Path):
        raise TypeError("baseline path must be a Path")
    expected = Path(__file__).resolve().parents[1] / "benchmark" / "baseline.json"
    if (
        path.is_symlink()
        or expected.is_symlink()
        or not path.is_file()
        or path.resolve() != expected.resolve()
    ):
        raise ValueError("baseline must be the installed release baseline")
    return expected


def _proposal_smoke_budget_inputs(
    path: Path,
    *,
    release_digest: str,
    system_prompt_sha256: str,
    critic_prompt_sha256: str,
) -> tuple[float, int]:
    if not path.is_file() or path.is_symlink():
        return 0.0, 0
    try:
        smoke = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(smoke, Mapping)
            or smoke.get("schema") != "robocasa-qwen-smoke-timing/v1"
            or smoke.get("critic_config_sha256") != CRITIC_CONFIG_SHA256
            or smoke.get("proposal_audit_config_sha256")
            != PROPOSAL_AUDIT_CONFIG_SHA256
            or smoke.get("release_digest") != release_digest
            or smoke.get("system_prompt_sha256") != system_prompt_sha256
            or smoke.get("critic_prompt_sha256") != critic_prompt_sha256
            or smoke.get("smoke_protocol") != "proposal"
            or smoke.get("smoke_pass") is not True
        ):
            return 0.0, 0
        wall_s = smoke.get("wall_s")
        decisions = smoke.get("direct_command_count")
        if (
            type(wall_s) not in {int, float}
            or not math.isfinite(float(wall_s))
            or float(wall_s) < 0.0
            or type(decisions) is not int
            or decisions < 0
        ):
            return 0.0, 0
        return float(wall_s), decisions
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return 0.0, 0


def execute(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    baseline_path = _validate_baseline_path(args.baseline)
    release_digest = _validate_release_digest(args.release_digest)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    critic_prompt = args.critic_prompt.read_text(encoding="utf-8")
    selected_protocol = args.smoke_protocol if args.smoke_task else "proposal"
    _validate_critic_prompt(
        critic_prompt,
        protocol=selected_protocol,
    )
    root = Path(__file__).resolve().parents[1]
    prompt_variant = str(args.prompt_variant)
    system_prompt_sha256 = hashlib.sha256(
        load_joint_system_prompt(
            root, protocol=selected_protocol, variant=prompt_variant
        ).encode()
    ).hexdigest()
    system_prompt_parts_sha256 = joint_system_prompt_parts(
        root, protocol=selected_protocol, variant=prompt_variant
    )
    critic_prompt_sha256 = hashlib.sha256(critic_prompt.encode()).hexdigest()
    args.output_root.mkdir(parents=True, mode=0o700)
    os.chmod(args.output_root, 0o700)

    if args.smoke_task:
        item = next(row for row in baseline["tasks"] if row["task"] == args.smoke_task)
        episode, context = run_one(
            item=item,
            seed=baseline["seed"],
            run=args.output_root / "smoke",
            max_decisions=args.smoke_decisions,
            critic_prompt=critic_prompt,
            protocol=args.smoke_protocol,
            action_budget=args.smoke_action_budget,
            prompt_variant=prompt_variant,
        )
        passed, checks = _smoke_pass(episode, protocol=args.smoke_protocol)
        tasks = [
            _task_record(row, episode)
            if row["task"] == item["task"]
            else _not_run(row, "not_run")
            for row in baseline["tasks"]
        ]
        result = {
            "schema": "robocasa-qwen10-run/v1",
            "release_digest": release_digest,
            "complete": False,
            "tasks": tasks,
            "episodes": [_episode_record(item, episode)],
            "exploration": [],
            "memory": [],
            "source_tracking_actions": 0,
            "hidden_state_accesses": 0,
            "safety_aborts": int(_is_safety_abort(episode)),
            "critic_origin_executions": 0,
            "model_calls": context.model_calls,
            "controller_calls": context.controller_calls,
            "critic_attempts": context.critic_attempts,
            "critic_successes": context.critic_successes,
            "critic_unavailable": context.critic_unavailable,
            "critic_config": CRITIC_CONFIG,
            "critic_config_sha256": CRITIC_CONFIG_SHA256,
            "proposal_audit_config": PROPOSAL_AUDIT_CONFIG,
            "proposal_audit_config_sha256": PROPOSAL_AUDIT_CONFIG_SHA256,
            "proposal_audit_schema_sha256": _response_schema_sha256(
                PROPOSAL_AUDIT_SCHEMA
            ),
            "critic_prompt_sha256": hashlib.sha256(critic_prompt.encode()).hexdigest(),
            "wall_s": time.monotonic() - started,
            "artifact_root": str(args.output_root),
            "mode": "smoke",
            "smoke_protocol": args.smoke_protocol,
            "smoke_action_budget": args.smoke_action_budget,
            "smoke_pass": passed,
            "smoke_checks": checks,
            "prompt_variant": prompt_variant,
            "system_prompt_sha256": system_prompt_sha256,
            "system_prompt_parts_sha256": system_prompt_parts_sha256,
        }
        _atomic_json(args.output_root / "result.json", result)
        _atomic_json(args.output_root.parent / "latest-smoke.json", {
            "schema": "robocasa-qwen-smoke-timing/v1",
            "critic_config_sha256": CRITIC_CONFIG_SHA256,
            "proposal_audit_config_sha256": PROPOSAL_AUDIT_CONFIG_SHA256,
            "release_digest": release_digest,
            "system_prompt_sha256": system_prompt_sha256,
            "critic_prompt_sha256": critic_prompt_sha256,
            "smoke_protocol": args.smoke_protocol,
            "smoke_action_budget": args.smoke_action_budget,
            "prompt_variant": prompt_variant,
            "wall_s": result["wall_s"],
            "direct_command_count": episode["qwen_direct_command_count"],
            "smoke_pass": passed,
            "artifact_root": str(args.output_root),
        })
        return result

    latest_smoke = args.output_root.parent / "latest-smoke.json"
    smoke_wall_s, smoke_decisions = _proposal_smoke_budget_inputs(
        latest_smoke,
        release_digest=release_digest,
        system_prompt_sha256=system_prompt_sha256,
        critic_prompt_sha256=critic_prompt_sha256,
    )

    budget_checks = [_budget_check(
        phase="pre_run",
        completed_tasks=0,
        run_elapsed_s=0.0,
        smoke_wall_s=smoke_wall_s,
        smoke_decisions=smoke_decisions,
    )]
    if budget_checks[-1]["decision"] != "continue":
        result = {
            "schema": "robocasa-qwen10-run/v1",
            "release_digest": release_digest,
            "complete": False,
            "tasks": [_not_run(row, "budget_guard_abort") for row in baseline["tasks"]],
            "episodes": [],
            "exploration": [],
            "memory": [],
            "source_tracking_actions": 0,
            "hidden_state_accesses": 0,
            "safety_aborts": 0,
            "critic_origin_executions": 0,
            "model_calls": 0,
            "wall_s": time.monotonic() - started,
            "artifact_root": str(args.output_root),
            "mode": "budget_guard_abort",
            "budget_checks": budget_checks,
            "critic_config": CRITIC_CONFIG,
            "critic_config_sha256": CRITIC_CONFIG_SHA256,
            "proposal_audit_config": PROPOSAL_AUDIT_CONFIG,
            "proposal_audit_config_sha256": PROPOSAL_AUDIT_CONFIG_SHA256,
        }
        _atomic_json(args.output_root / "result.json", result)
        return result

    episodes: dict[str, dict[str, object]] = {}
    contexts: list[CriticContext] = []
    aborted = False
    for index, item in enumerate(baseline["tasks"], start=1):
        episode, context = run_one(
            item=item,
            seed=baseline["seed"] + 1000,
            run=args.output_root / "final" / str(item["task"]),
            max_decisions=FULL_MAX_DECISIONS,
            critic_prompt=critic_prompt,
            protocol="proposal",
            prompt_variant=prompt_variant,
        )
        episodes[str(item["task"])] = episode
        contexts.append(context)
        check = _budget_check(
            phase=f"after_task_{index}",
            completed_tasks=index,
            run_elapsed_s=time.monotonic() - started,
            smoke_wall_s=smoke_wall_s,
            smoke_decisions=smoke_decisions,
        )
        budget_checks.append(check)
        if index < len(baseline["tasks"]) and check["decision"] != "continue":
            aborted = True
            break

    tasks = [
        _task_record(row, episodes[str(row["task"])])
        if str(row["task"]) in episodes
        else _not_run(row, "budget_guard_abort")
        for row in baseline["tasks"]
    ]
    result = {
        "schema": "robocasa-qwen10-run/v1",
        "release_digest": release_digest,
        "complete": not aborted and len(episodes) == 10,
        "tasks": tasks,
        "episodes": [
            _episode_record(row, episodes[str(row["task"])])
            for row in baseline["tasks"]
            if str(row["task"]) in episodes
        ],
        "exploration": [],
        "memory": [],
        "source_tracking_actions": 0,
        "hidden_state_accesses": 0,
        "safety_aborts": sum(_is_safety_abort(value) for value in episodes.values()),
        "critic_origin_executions": 0,
        "model_calls": sum(context.model_calls for context in contexts),
        "controller_calls": sum(context.controller_calls for context in contexts),
        "critic_attempts": sum(context.critic_attempts for context in contexts),
        "critic_successes": sum(context.critic_successes for context in contexts),
        "critic_unavailable": sum(context.critic_unavailable for context in contexts),
        "critic_config": CRITIC_CONFIG,
        "critic_config_sha256": CRITIC_CONFIG_SHA256,
        "proposal_audit_config": PROPOSAL_AUDIT_CONFIG,
        "proposal_audit_config_sha256": PROPOSAL_AUDIT_CONFIG_SHA256,
        "proposal_audit_schema_sha256": _response_schema_sha256(
            PROPOSAL_AUDIT_SCHEMA
        ),
        "critic_prompt_sha256": hashlib.sha256(critic_prompt.encode()).hexdigest(),
        "budget_checks": budget_checks,
        "wall_s": time.monotonic() - started,
        "artifact_root": str(args.output_root),
        "mode": "budget_guard_abort" if aborted else "full",
    }
    _atomic_json(args.output_root / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--critic-prompt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--release-digest", required=True)
    parser.add_argument("--smoke-task")
    parser.add_argument("--smoke-protocol", choices=("legacy", "proposal"))
    parser.add_argument("--smoke-decisions", type=int, default=12)
    parser.add_argument("--smoke-action-budget", type=int, choices=(450, 900), default=450)
    parser.add_argument(
        "--prompt-variant", default="baseline", choices=("baseline", "rig")
    )
    args = parser.parse_args()
    if (args.smoke_task is None) != (args.smoke_protocol is None):
        parser.error("--smoke-task and --smoke-protocol must be supplied together")
    if args.smoke_task is None and args.smoke_action_budget != 450:
        parser.error("--smoke-action-budget is available only for smoke runs")
    print(json.dumps(execute(args), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
