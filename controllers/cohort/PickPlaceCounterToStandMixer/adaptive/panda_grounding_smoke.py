"""Deterministic, gate-only Panda geometry and telemetry smoke.

The isolated child is the only process that sees exact simulator ``grip_site``
poses, oracle pixels, the virtual witness point, or numeric probe targets.  Its
persisted result is deliberately aggregate-only.  Evaluated task episodes do
not import this module and retain Qwen-only numeric target authority.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .joint_protocol import (
    JOINT_LIMIT_MARGIN,
    JOINT_LIMITS,
    JOINT_STEP_TOLERANCE,
    MAX_COMMAND_ACTIONS,
    MAX_JOINT_STEP,
    MIN_GRIPPER_ACTIONS,
    TRACKING_LAG_PAUSE,
)
from .joint_sim_child import EPISODE_ACTION_BUDGET
from .panda_embodiment import (
    GRIPPER_MODEL_SHA256,
    GRIPPER_MODEL_SOURCE_PATH,
    PUBLIC_TELEMETRY_KEYS,
    ROBOT_MODEL_SHA256,
    ROBOT_MODEL_SOURCE_PATH,
    Pose,
    panda_fk,
    project_world_point,
    summarize_telemetry_samples,
    validate_telemetry_summary,
)

GROUNDING_EVIDENCE_SCHEMA = "robocasa-inspect-panda-grounding-evidence/v1"
GROUNDING_RESULT_SCHEMA = "robocasa-inspect-panda-grounding-smoke/v1"
OFFICIAL_CALIBRATION_SOURCE = (
    "robocasa_inspect.camera_geometry.official_camera_calibration"
)

STATIC_POSITION_ERROR_MAX_M = 0.001
STATIC_ROTATION_ERROR_MAX_RAD = 0.01
EXTERNAL_PIXEL_ERROR_MAX_PX = 5.0
PROBE_LIMIT_CLEARANCE_MIN_RAD = 0.08
PROBE_COMMAND_RAD = 0.04
REALIZED_DISPLACEMENT_MIN_RAD = 0.02
DYNAMIC_TRANSLATION_COSINE_MIN = 0.95
DYNAMIC_TRANSLATION_RELATIVE_ERROR_MAX = 0.30
DYNAMIC_VECTOR_NORM_MIN_M = 1e-6
RETURN_ERROR_MAX_RAD = 0.005
REST_SAMPLE_COUNT_MIN = 3
REST_QVEL_MAX_ABS_RAD_S = 0.05
REST_QPOS_SPAN_MAX_RAD = 0.001
REST_TORQUE_SPAN_MAX_NM = 0.25
REST_FORCE_SPAN_MAX_N = 0.10
REST_WRENCH_TORQUE_SPAN_MAX_NM = 0.05
GROUNDING_AUDIT_COUNT = 18
GROUNDING_WALL_TIME_MAX_S = 1_300.0

# Joint 7 rotates about the literal grip-site origin, so origin translation is
# structurally zero.  This fixed point in the exact grip-site frame makes all
# seven dynamic FK comparisons observable without changing the static target.
DYNAMIC_WITNESS_OFFSET_GRIP_SITE_M = (0.02, 0.0, 0.0)
DYNAMIC_WITNESS_SHA256 = hashlib.sha256(
    json.dumps(
        DYNAMIC_WITNESS_OFFSET_GRIP_SITE_M,
        separators=(",", ":"),
    ).encode()
).hexdigest()

GROUNDING_CHECKS = frozenset(
    {
        "closed_evidence_schema",
        "installed_model_sources",
        "official_camera_calibration",
        "qwen_only_target_authority",
        "raw_oracle_process_local",
        "stable_rest_baseline",
        "static_fk_position",
        "static_fk_rotation",
        "external_pixel_agreement",
        "all_seven_joints_probed",
        "joint_limit_clearance",
        "safe_probe_command",
        "realized_joint_displacement",
        "dynamic_translation_nondegenerate",
        "dynamic_translation_cosine",
        "dynamic_translation_relative_error",
        "return_to_start",
        "complete_telemetry_sampling",
        "finite_telemetry",
        "telemetry_peaks_reconcile",
        "joint_limit_inset",
        "first_step_tolerance",
        "later_step_bound",
        "tracking_pauses_reconcile",
        "command_action_cap",
        "gripper_transition_actions",
        "base_and_torso_stationary",
        "gripper_close_direction",
        "gripper_reopen_direction",
        "action_accounting_exact",
        "episode_action_budget",
        "network_disabled",
        "simulator_clean",
    }
)

_ROOT_FIELDS = {
    "schema",
    "task",
    "seed",
    "authority",
    "source_hashes",
    "camera_calibration",
    "baseline",
    "rest_baseline",
    "static",
    "probes",
    "gripper",
    "journal_terminal_sha256",
    "action_count",
    "terminal_action_count",
    "network_probe",
    "simulator_error",
    "episode_tmp_empty",
}
_AUTHORITY_FIELDS = {
    "probe_target_authority",
    "normal_episode_target_authority",
    "controller_model_payload_count",
    "critic_model_payload_count",
    "raw_oracle_in_model_payload",
    "raw_oracle_persisted",
    "probe_targets_persisted",
}
_SOURCE_FIELDS = {
    "robot_model_path",
    "robot_model_sha256",
    "gripper_model_path",
    "gripper_model_sha256",
    "dynamic_witness_sha256",
}
_STATIC_FIELDS = {
    "qpos",
    "link0_world_pose",
    "oracle_grip_site_world_pose",
    "public_external_pixels",
}
_PROBE_FIELDS = {
    "joint_index",
    "direction",
    "start_qpos",
    "target_qpos",
    "outbound_qpos",
    "return_qpos",
    "oracle_start_pose",
    "oracle_outbound_pose",
    "oracle_return_pose",
    "outbound",
    "return",
}
_AUDIT_FIELDS = {"receipt", "telemetry", "actions", "journal_range"}
_TELEMETRY_TRACE_FIELDS = {"before", "samples", "completion"}
_JOURNAL_RANGE_FIELDS = {"start_event_index", "end_event_index"}
_ACTION_FIELDS = {
    "realized_before",
    "commanded_qpos",
    "realized_after",
    "gripper_open",
    "base_motion",
    "torso",
}
_MOVE_RECEIPT_FIELDS = {
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
_GRIPPER_FIELDS = {
    "reset_qpos",
    "closed_qpos",
    "reopened_qpos",
    "close",
    "reopen",
}
_CALIBRATION_FIELDS = {
    "camera_name",
    "mujoco_camera_name",
    "image_width_px",
    "image_height_px",
    "fx_px",
    "fy_px",
    "cx_px",
    "cy_px",
    "camera_position_world_m",
    "camera_xmat_world",
    "projection",
}
PUBLIC_RESULT_FIELDS = frozenset(
    {
        "schema",
        "task",
        "seed",
        "passed",
        "checks",
        "metrics",
        "provenance",
        "artifact_root",
        "release_digest",
        "wall_s",
    }
)
GROUNDING_METRIC_FIELDS = frozenset(
    {
        "static_position_error_m",
        "static_rotation_error_rad",
        "max_external_pixel_error_px",
        "min_joint_limit_clearance_rad",
        "max_probe_command_rad",
        "min_realized_joint_displacement_rad",
        "min_dynamic_translation_cosine",
        "max_dynamic_translation_relative_error",
        "min_dynamic_vector_norm_m",
        "max_return_error_rad",
        "max_first_step_rad",
        "max_later_step_rad",
        "telemetry_sample_count",
        "probe_count",
        "audit_count",
        "nonempty_audit_count",
        "action_count",
        "terminal_action_count",
        "audited_action_count",
        "min_actions_per_command",
        "max_actions_per_command",
        "gripper_transition_audit_count",
        "min_gripper_transition_actions",
        "min_joint_limit_inset_rad",
        "max_rest_qpos_span_rad",
        "max_rest_qvel_abs_rad_s",
        "max_rest_torque_span_nm",
        "max_rest_force_span_n",
        "max_rest_wrench_torque_span_nm",
        "max_rest_endpoint_error_rad",
        "tracking_required_pause_count",
        "tracking_observed_pause_count",
        "tracking_violation_count",
        "tracking_receipt_mismatch_count",
        "sampling_reconciliation_error_count",
        "peak_reconciliation_error_count",
        "nonfinite_telemetry_value_count",
        "max_base_motion_abs",
        "max_torso_abs",
        "gripper_reset_separation_m",
        "gripper_closed_separation_m",
        "gripper_reopened_separation_m",
        "gripper_close_separation_delta_m",
        "gripper_reopen_separation_delta_m",
        "controller_model_payload_count",
        "critic_model_payload_count",
        "authority_label_error_count",
        "raw_oracle_model_payload_count",
        "raw_oracle_persisted_count",
        "probe_targets_persisted_count",
        "evidence_schema_error_count",
        "installed_source_mismatch_count",
        "calibration_error_count",
        "network_error_count",
        "simulator_error_count",
        "episode_tmp_file_count",
        "stable_rest_reconciliation_error_count",
        "static_binding_error_count",
        "pixel_binding_error_count",
        "probe_binding_error_count",
        "joint_direction_error_count",
        "probe_command_shape_error_count",
        "realized_direction_error_count",
        "gripper_close_binding_error_count",
        "gripper_reopen_binding_error_count",
        "gripper_close_direction_error_count",
        "gripper_reopen_direction_error_count",
        "command_cap_error_count",
    }
)
GROUNDING_PROVENANCE_FIELDS = frozenset(
    {
        "probe_target_authority",
        "normal_episode_target_authority",
        "raw_oracle_scope",
        "persisted_value_policy",
        "camera_calibration_source",
        "robot_model_sha256",
        "gripper_model_sha256",
        "dynamic_witness_sha256",
        "evidence_schema",
        "journal_trust_boundary",
        "journal_terminal_sha256",
        "network_policy",
        "simulator_status",
        "evidence_sha256",
        "error_sha256",
    }
)
GROUNDING_COUNT_FIELDS = frozenset(
    {
        name for name in GROUNDING_METRIC_FIELDS if name.endswith("_count")
    }
    | {
        "min_actions_per_command",
        "max_actions_per_command",
        "min_gripper_transition_actions",
    }
)


@dataclass(frozen=True)
class _Evaluation:
    passed: bool
    checks: dict[str, bool]
    metrics: dict[str, object]


def _empty_metrics() -> dict[str, object]:
    return {name: None for name in GROUNDING_METRIC_FIELDS}


@dataclass(frozen=True)
class _JournalEvent:
    """One immutable event captured at the actual simulator call boundary."""

    index: int
    audit_label: str
    kind: str
    snapshot_json: str
    previous_sha256: str
    sha256: str


@dataclass(frozen=True)
class _JournalSnapshot:
    events: tuple[_JournalEvent, ...]
    terminal_sha256: str


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _journal_envelope(event: _JournalEvent) -> bytes:
    return _canonical_json(
        {
            "index": event.index,
            "audit_label": event.audit_label,
            "kind": event.kind,
            "snapshot_json": event.snapshot_json,
            "previous_sha256": event.previous_sha256,
        }
    ).encode()


@dataclass(frozen=True)
class _JournalVerifier:
    """Process-local HMAC verifier; its key must never enter raw evidence."""

    snapshot: _JournalSnapshot
    _key: bytes = field(repr=False)

    def verify(self) -> bool:
        prior = "0" * 64
        for index, event in enumerate(self.snapshot.events):
            if (
                event.index != index
                or event.previous_sha256 != prior
                or _canonical_json(json.loads(event.snapshot_json))
                != event.snapshot_json
            ):
                return False
            expected = hmac.new(
                self._key,
                _journal_envelope(event),
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(event.sha256, expected):
                return False
            prior = event.sha256
        return bool(self.snapshot.events) and hmac.compare_digest(
            self.snapshot.terminal_sha256,
            prior,
        )


@dataclass(frozen=True)
class _ProcessLocalGroundingEvidence(Mapping[str, object]):
    """Raw evidence plus its non-serializable, process-local source verifier."""

    raw: Mapping[str, object]
    journal_verifier: _JournalVerifier

    def __getitem__(self, key: str) -> object:
        return self.raw[key]

    def __iter__(self):
        return iter(self.raw)

    def __len__(self) -> int:
        return len(self.raw)


class _JournalRecorder:
    """Streaming gate-local recorder called synchronously by traced boundaries."""

    def __init__(self) -> None:
        self._key = secrets.token_bytes(32)
        self._events: list[_JournalEvent] = []
        self._prior_sha256 = "0" * 64
        self._active_label: str | None = None
        self._active_start: int | None = None

    @property
    def terminal_sha256(self) -> str:
        return self._prior_sha256

    def begin_audit(self, label: str) -> None:
        if self._active_label is not None or not label:
            raise RuntimeError("journal audit lifecycle drifted")
        self._active_label = label
        self._active_start = len(self._events)

    def _record(self, kind: str, snapshot: Mapping[str, object]) -> None:
        if self._active_label is None:
            raise RuntimeError("journal event escaped an audit")
        event = _JournalEvent(
            index=len(self._events),
            audit_label=self._active_label,
            kind=kind,
            snapshot_json=_canonical_json(snapshot),
            previous_sha256=self._prior_sha256,
            sha256="",
        )
        digest = hmac.new(
            self._key,
            _journal_envelope(event),
            hashlib.sha256,
        ).hexdigest()
        sealed = _JournalEvent(
            index=event.index,
            audit_label=event.audit_label,
            kind=event.kind,
            snapshot_json=event.snapshot_json,
            previous_sha256=event.previous_sha256,
            sha256=digest,
        )
        self._events.append(sealed)
        self._prior_sha256 = digest

    def record_step(self, action: Mapping[str, object]) -> None:
        self._record("step", {"action": action})

    def record_read(
        self,
        kind: str,
        *,
        telemetry: Mapping[str, object],
        gripper_qpos: Sequence[float],
    ) -> None:
        if kind not in {"read_before", "read_step", "read_completion"}:
            raise RuntimeError("journal read kind drifted")
        qpos = [float(value) for value in gripper_qpos]
        self._record(
            kind,
            {
                "telemetry": telemetry,
                "gripper_qpos": qpos,
                "finger_separation_m": _finger_separation(qpos),
            },
        )

    def end_audit(self) -> dict[str, int]:
        if self._active_label is None or self._active_start is None:
            raise RuntimeError("journal audit lifecycle drifted")
        value = {
            "start_event_index": self._active_start,
            "end_event_index": len(self._events),
        }
        self._active_label = None
        self._active_start = None
        return value

    def freeze(self) -> _JournalVerifier:
        if self._active_label is not None:
            raise RuntimeError("journal froze during an audit")
        snapshot = _JournalSnapshot(
            tuple(self._events),
            self._prior_sha256,
        )
        return _JournalVerifier(snapshot=snapshot, _key=self._key)


def _mapping(value: object, fields: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} schema drifted")
    return value


def _sequence(value: object, label: str) -> list[object]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence")
    return list(value)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _vector(value: object, width: int, label: str) -> list[float]:
    items = _sequence(value, label)
    if len(items) != width:
        raise ValueError(f"{label} must contain {width} values")
    return [_number(item, label) for item in items]


def _matrix3(value: object, label: str) -> list[list[float]]:
    rows = _sequence(value, label)
    if len(rows) != 3:
        raise ValueError(f"{label} must be 3x3")
    matrix = [_vector(row, 3, label) for row in rows]
    for first in range(3):
        for second in range(3):
            dot = sum(matrix[row][first] * matrix[row][second] for row in range(3))
            expected = 1.0 if first == second else 0.0
            if not math.isclose(dot, expected, abs_tol=1e-6, rel_tol=0.0):
                raise ValueError(f"{label} must be orthonormal")
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1]
        * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2]
        * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if not math.isclose(determinant, 1.0, abs_tol=1e-6, rel_tol=0.0):
        raise ValueError(f"{label} must be a proper rotation")
    return matrix


def _pose_record(value: object, label: str) -> Pose:
    record = _mapping(value, {"position_m", "rotation_matrix"}, label)
    position = _vector(record["position_m"], 3, f"{label} position")
    rotation = _matrix3(record["rotation_matrix"], f"{label} rotation")
    return Pose(tuple(position), tuple(tuple(row) for row in rotation))  # type: ignore[arg-type]


def _compose_pose(parent: Pose, child: Pose) -> Pose:
    rotation = tuple(
        tuple(
            sum(parent.rotation_matrix[row][inner] * child.rotation_matrix[inner][column] for inner in range(3))
            for column in range(3)
        )
        for row in range(3)
    )
    position = tuple(
        parent.position_m[row]
        + sum(parent.rotation_matrix[row][inner] * child.position_m[inner] for inner in range(3))
        for row in range(3)
    )
    return Pose(position, rotation)  # type: ignore[arg-type]


def _transform_point(pose: Pose, point: Sequence[float]) -> tuple[float, ...]:
    return tuple(
        pose.position_m[row]
        + sum(pose.rotation_matrix[row][column] * point[column] for column in range(3))
        for row in range(3)
    )


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _delta(end: Sequence[float], start: Sequence[float]) -> list[float]:
    return [end_value - start_value for start_value, end_value in zip(start, end, strict=True)]


def _rotation_error(left: Pose, right: Pose) -> float:
    trace = sum(
        left.rotation_matrix[row][column] * right.rotation_matrix[row][column]
        for row in range(3)
        for column in range(3)
    )
    return math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))


def _pose_within(
    left: Pose,
    right: Pose,
    *,
    position_m: float,
    rotation_rad: float,
) -> bool:
    return (
        _norm(_delta(left.position_m, right.position_m)) <= position_m
        and _rotation_error(left, right) <= rotation_rad
    )


def _pixel_record(value: object, label: str) -> dict[str, object]:
    record = _mapping(
        value,
        {"u_px", "v_px", "visible", "depth_valid"},
        label,
    )
    visible = record["visible"]
    depth_valid = record["depth_valid"]
    if type(visible) is not bool or type(depth_valid) is not bool:
        raise TypeError(f"{label} flags must be booleans")
    if depth_valid:
        u_px: float | None = _number(record["u_px"], f"{label} u")
        v_px: float | None = _number(record["v_px"], f"{label} v")
    else:
        if record["u_px"] is not None or record["v_px"] is not None or visible:
            raise ValueError(f"{label} conditional nulls drifted")
        u_px = None
        v_px = None
    return {
        "u_px": u_px,
        "v_px": v_px,
        "visible": visible,
        "depth_valid": depth_valid,
    }


def _external_pixels(value: object, label: str) -> dict[str, dict[str, object]]:
    root = _mapping(value, {"left", "right"}, label)
    return {
        camera: _pixel_record(root[camera], f"{label} {camera}")
        for camera in ("left", "right")
    }


def _calibration(value: object) -> dict[str, Mapping[str, object]]:
    root = _mapping(value, {"source", "left", "right"}, "camera calibration")
    if root["source"] != OFFICIAL_CALIBRATION_SOURCE:
        raise ValueError("camera calibration source drifted")
    output: dict[str, Mapping[str, object]] = {}
    for camera in ("left", "right"):
        record = _mapping(root[camera], _CALIBRATION_FIELDS, f"{camera} calibration")
        expected_name = f"robot0_agentview_{camera}"
        if (
            record["camera_name"] != f"video.{expected_name}"
            or record["mujoco_camera_name"] != expected_name
        ):
            raise ValueError("official camera identity drifted")
        # The production projector performs the complete intrinsic/extrinsic and
        # convention validation.  A point in front of each camera is checked later.
        output[camera] = record
    return output


def _telemetry(value: object, label: str) -> dict[str, object]:
    record = _mapping(value, set(PUBLIC_TELEMETRY_KEYS), label)
    qpos = _vector(record["state.arm_joint_position"], 7, f"{label} qpos")
    qvel = _vector(record["state.arm_joint_velocity"], 7, f"{label} qvel")
    torque = _mapping(
        record["state.arm_applied_torque"],
        {"available", "values_nm"},
        f"{label} torque",
    )
    if torque["available"] is not True:
        raise ValueError(f"{label} torque is incomplete")
    torque_values = _vector(torque["values_nm"], 7, f"{label} torque")
    wrench = _mapping(
        record["state.end_effector_wrench"],
        {"force_n", "torque_nm"},
        f"{label} wrench",
    )
    force = _vector(wrench["force_n"], 3, f"{label} force")
    wrench_torque = _vector(wrench["torque_nm"], 3, f"{label} wrench torque")
    for key in ("state.arm_translation_jacobian", "state.arm_rotation_jacobian"):
        rows = _sequence(record[key], f"{label} {key}")
        if len(rows) != 3:
            raise ValueError(f"{label} {key} must be 3x7")
        for row in rows:
            _vector(row, 7, f"{label} {key}")
    pixels = _external_pixels(record["state.end_effector_external_pixels"], label)
    return {
        **dict(record),
        "state.arm_joint_position": qpos,
        "state.arm_joint_velocity": qvel,
        "state.arm_applied_torque": {"available": True, "values_nm": torque_values},
        "state.end_effector_wrench": {"force_n": force, "torque_nm": wrench_torque},
        "state.end_effector_external_pixels": pixels,
    }


def _json_equal(left: object, right: object) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha256_string(value: object, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _audit(value: object, label: str) -> dict[str, object]:
    audit = _mapping(value, _AUDIT_FIELDS, label)
    receipt = _mapping(audit["receipt"], _MOVE_RECEIPT_FIELDS, f"{label} receipt")
    if receipt["kind"] != "move_joints" or receipt["accepted"] is not True:
        raise ValueError(f"{label} receipt was not accepted")
    bounded_endpoint = _vector(receipt["bounded_endpoint"], 7, f"{label} endpoint")
    realized = _vector(receipt["realized_arm_qpos"], 7, f"{label} realized qpos")
    step_count = _integer(receipt["step_count"], f"{label} step count")
    maximum_step = _number(receipt["maximum_commanded_step"], f"{label} maximum step")
    minimum_margin = _number(receipt["minimum_hard_limit_margin"], f"{label} minimum margin")
    tracking_pauses = _integer(receipt["tracking_pause_count"], f"{label} pauses")
    gripper_intent = _number(receipt["gripper_intent"], f"{label} gripper intent")
    endpoint_error = _number(receipt["endpoint_error"], f"{label} endpoint error")
    before_pixels = _external_pixels(
        receipt["end_effector_external_pixels_before"], f"{label} before pixels"
    )
    after_pixels = _external_pixels(
        receipt["end_effector_external_pixels_after"], f"{label} after pixels"
    )
    trace = _mapping(audit["telemetry"], _TELEMETRY_TRACE_FIELDS, f"{label} telemetry")
    before = _telemetry(trace["before"], f"{label} before")
    sample_values = _sequence(trace["samples"], f"{label} samples")
    samples = [
        _telemetry(sample, f"{label} sample {index}")
        for index, sample in enumerate(sample_values)
    ]
    completion = _telemetry(trace["completion"], f"{label} completion")
    actions = _sequence(audit["actions"], f"{label} actions")
    normalized_actions: list[dict[str, object]] = []
    for index, value in enumerate(actions):
        action = _mapping(value, _ACTION_FIELDS, f"{label} action {index}")
        normalized_actions.append(
            {
                "realized_before": _vector(action["realized_before"], 7, f"{label} realized before"),
                "commanded_qpos": _vector(action["commanded_qpos"], 7, f"{label} command"),
                "realized_after": _vector(action["realized_after"], 7, f"{label} realized after"),
                "gripper_open": _number(action["gripper_open"], f"{label} gripper"),
                "base_motion": _vector(action["base_motion"], 3, f"{label} base"),
                "torso": _number(action["torso"], f"{label} torso"),
            }
        )
    journal_range = _mapping(
        audit["journal_range"],
        _JOURNAL_RANGE_FIELDS,
        f"{label} journal range",
    )
    journal_start = _integer(
        journal_range["start_event_index"],
        f"{label} journal start",
    )
    journal_end = _integer(
        journal_range["end_event_index"],
        f"{label} journal end",
    )
    if journal_start < 0 or journal_end <= journal_start:
        raise ValueError(f"{label} journal range drifted")
    summary = validate_telemetry_summary(receipt["telemetry_summary"])
    recomputed = summarize_telemetry_samples(before, samples, completion)
    trace_reconciles = (
        len(normalized_actions) == len(samples)
        and bool(normalized_actions)
        and _json_equal(
            normalized_actions[0]["realized_before"],
            before["state.arm_joint_position"],
        )
        and _json_equal(
            normalized_actions[-1]["commanded_qpos"], bounded_endpoint
        )
        and _json_equal(
            normalized_actions[-1]["realized_after"],
            completion["state.arm_joint_position"],
        )
        and _json_equal(realized, completion["state.arm_joint_position"])
        and all(
            _json_equal(action["realized_after"], sample["state.arm_joint_position"])
            and action["gripper_open"] == gripper_intent
            and (
                index == 0
                or _json_equal(
                    action["realized_before"],
                    normalized_actions[index - 1]["realized_after"],
                )
            )
            for index, (action, sample) in enumerate(
                zip(normalized_actions, samples, strict=True)
            )
        )
        and math.isclose(
            endpoint_error,
            max(
                abs(expected - actual)
                for expected, actual in zip(
                    bounded_endpoint, realized, strict=True
                )
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    return {
        "receipt": {
            **dict(receipt),
            "bounded_endpoint": bounded_endpoint,
            "realized_arm_qpos": realized,
            "step_count": step_count,
            "maximum_commanded_step": maximum_step,
            "minimum_hard_limit_margin": minimum_margin,
            "tracking_pause_count": tracking_pauses,
            "gripper_intent": gripper_intent,
            "endpoint_error": endpoint_error,
            "telemetry_summary": summary,
            "end_effector_external_pixels_before": before_pixels,
            "end_effector_external_pixels_after": after_pixels,
        },
        "before": before,
        "samples": samples,
        "completion": completion,
        "actions": normalized_actions,
        "journal_range": {
            "start_event_index": journal_start,
            "end_event_index": journal_end,
        },
        "trace_reconciles": trace_reconciles,
        "summary_reconciles": (
            _json_equal(summary, recomputed)
            and _json_equal(
                before_pixels,
                before["state.end_effector_external_pixels"],
            )
            and _json_equal(
                after_pixels,
                completion["state.end_effector_external_pixels"],
            )
        ),
    }


def _journal_read_snapshot(value: str) -> dict[str, object]:
    snapshot = _mapping(
        json.loads(value),
        {"telemetry", "gripper_qpos", "finger_separation_m"},
        "journal read snapshot",
    )
    gripper_qpos = _vector(snapshot["gripper_qpos"], 2, "journal gripper qpos")
    separation = _number(
        snapshot["finger_separation_m"],
        "journal finger separation",
    )
    if not math.isclose(
        separation,
        abs(gripper_qpos[0] - gripper_qpos[1]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("journal finger separation drifted")
    return {
        "telemetry": snapshot["telemetry"],
        "gripper_qpos": gripper_qpos,
        "finger_separation_m": separation,
    }


def _reconcile_source_journal(
    evidence: Mapping[str, object],
    audits: Sequence[tuple[str, dict[str, object]]],
) -> tuple[bool, dict[str, dict[str, object]]]:
    """Bind mutable audit views to the immutable process-local source journal."""

    if not isinstance(evidence, _ProcessLocalGroundingEvidence):
        return False, {}
    verifier = evidence.journal_verifier
    if not verifier.verify():
        return False, {}
    snapshot = verifier.snapshot
    terminal = _sha256_string(
        evidence["journal_terminal_sha256"],
        "evidence journal terminal",
    )
    if not hmac.compare_digest(terminal, snapshot.terminal_sha256):
        return False, {}
    cursor = 0
    views: dict[str, dict[str, object]] = {}
    for label, audit in audits:
        event_range = audit["journal_range"]
        start = event_range["start_event_index"]
        end = event_range["end_event_index"]
        actions = audit["actions"]
        samples = audit["samples"]
        if (
            start != cursor
            or end != start + 2 * len(actions) + 2
            or end > len(snapshot.events)
        ):
            return False, {}
        events = snapshot.events[start:end]
        if any(event.audit_label != label for event in events):
            return False, {}
        before_event = events[0]
        completion_event = events[-1]
        if (
            before_event.kind != "read_before"
            or completion_event.kind != "read_completion"
        ):
            return False, {}
        before = _journal_read_snapshot(before_event.snapshot_json)
        completion = _journal_read_snapshot(completion_event.snapshot_json)
        if not _json_equal(before["telemetry"], audit["before"]):
            return False, {}
        if not _json_equal(completion["telemetry"], audit["completion"]):
            return False, {}
        step_reads: list[dict[str, object]] = []
        for index, (action, sample) in enumerate(
            zip(actions, samples, strict=True)
        ):
            step_event = events[1 + 2 * index]
            read_event = events[2 + 2 * index]
            if step_event.kind != "step" or read_event.kind != "read_step":
                return False, {}
            step_snapshot = _mapping(
                json.loads(step_event.snapshot_json),
                {"action"},
                "journal step snapshot",
            )
            read_snapshot = _journal_read_snapshot(read_event.snapshot_json)
            if not _json_equal(step_snapshot["action"], action) or not _json_equal(
                read_snapshot["telemetry"], sample
            ):
                return False, {}
            step_reads.append(read_snapshot)
        views[label] = {
            "before": before,
            "steps": step_reads,
            "completion": completion,
        }
        cursor = end
    return cursor == len(snapshot.events), views


def _span(vectors: Sequence[Sequence[float]]) -> float:
    return max(
        max(vector[index] for vector in vectors)
        - min(vector[index] for vector in vectors)
        for index in range(len(vectors[0]))
    )


def _action_checks(
    audits: Sequence[dict[str, object]],
) -> tuple[dict[str, bool], dict[str, object]]:
    sampling = True
    peaks = True
    inset = True
    first_bound = True
    later_bound = True
    pauses = True
    cap = True
    stationary = True
    maximum_first_step = 0.0
    maximum_later_step = 0.0
    telemetry_samples = 0
    action_counts: list[int] = []
    inset_margins: list[float] = []
    tracking_required = 0
    tracking_observed = 0
    tracking_violations = 0
    tracking_receipt_mismatches = 0
    sampling_reconciliation_errors = 0
    peak_reconciliation_errors = 0
    maximum_base_motion = 0.0
    maximum_torso = 0.0
    for audit in audits:
        receipt = audit["receipt"]
        actions = audit["actions"]
        samples = audit["samples"]
        step_count = receipt["step_count"]
        audit_sampling = (
            step_count == len(actions) == len(samples)
            and step_count > 0
            and audit["trace_reconciles"]
            and _json_equal(
                receipt["realized_arm_qpos"],
                audit["completion"]["state.arm_joint_position"],
            )
        )
        sampling &= audit_sampling
        sampling_reconciliation_errors += int(not audit_sampling)
        audit_peaks = bool(audit["summary_reconciles"])
        cap &= 1 <= step_count <= MAX_COMMAND_ACTIONS
        action_counts.append(len(actions))
        inset &= receipt["minimum_hard_limit_margin"] >= 0.0
        endpoint = receipt["bounded_endpoint"]
        endpoint_insets = [
            min(
                value - (lower + JOINT_LIMIT_MARGIN),
                (upper - JOINT_LIMIT_MARGIN) - value,
            )
            for value, (lower, upper) in zip(endpoint, JOINT_LIMITS, strict=True)
        ]
        inset_margins.extend(
            [receipt["minimum_hard_limit_margin"], *endpoint_insets]
        )
        inset &= min(endpoint_insets) >= 0.0
        telemetry_samples += len(samples) + 2
        computed_maximum = 0.0
        receipt_expected_pauses = 0
        previous_command: list[float] | None = None
        for index, action in enumerate(actions):
            command = action["commanded_qpos"]
            reference = action["realized_before"] if index == 0 else previous_command
            assert reference is not None
            step = max(abs(value - prior) for value, prior in zip(command, reference, strict=True))
            computed_maximum = max(computed_maximum, step)
            if index == 0:
                maximum_first_step = max(maximum_first_step, step)
                first_bound &= step <= MAX_JOINT_STEP + JOINT_STEP_TOLERANCE + 1e-12
            else:
                maximum_later_step = max(maximum_later_step, step)
                later_bound &= step <= MAX_JOINT_STEP + 1e-12
                assert previous_command is not None
                tracking_gap = max(
                    abs(commanded - realized)
                    for commanded, realized in zip(
                        previous_command, action["realized_before"], strict=True
                    )
                )
                if tracking_gap >= TRACKING_LAG_PAUSE:
                    tracking_required += 1
                    if command == previous_command:
                        tracking_observed += 1
                        if previous_command != endpoint:
                            receipt_expected_pauses += 1
                    else:
                        tracking_violations += 1
                        pauses = False
            action_base_max = max(abs(value) for value in action["base_motion"])
            maximum_base_motion = max(maximum_base_motion, action_base_max)
            maximum_torso = max(maximum_torso, abs(action["torso"]))
            stationary &= action_base_max <= 1e-12
            stationary &= abs(action["torso"]) <= 1e-12
            previous_command = command
        audit_peaks &= math.isclose(
            receipt["maximum_commanded_step"],
            computed_maximum,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        peaks &= audit_peaks
        peak_reconciliation_errors += int(not audit_peaks)
        if receipt["tracking_pause_count"] != receipt_expected_pauses:
            tracking_receipt_mismatches += 1
            pauses = False
    return (
        {
            "complete_telemetry_sampling": sampling,
            "finite_telemetry": True,
            "telemetry_peaks_reconcile": peaks,
            "joint_limit_inset": inset,
            "first_step_tolerance": first_bound,
            "later_step_bound": later_bound,
            "tracking_pauses_reconcile": pauses,
            "command_action_cap": cap,
            "base_and_torso_stationary": stationary,
        },
        {
            "max_first_step_rad": maximum_first_step,
            "max_later_step_rad": maximum_later_step,
            "telemetry_sample_count": telemetry_samples,
            "audit_count": len(audits),
            "nonempty_audit_count": sum(bool(audit["actions"]) for audit in audits),
            "min_actions_per_command": min(action_counts, default=0),
            "max_actions_per_command": max(action_counts, default=0),
            "min_joint_limit_inset_rad": min(inset_margins, default=None),
            "tracking_required_pause_count": tracking_required,
            "tracking_observed_pause_count": tracking_observed,
            "tracking_violation_count": tracking_violations,
            "tracking_receipt_mismatch_count": tracking_receipt_mismatches,
            "sampling_reconciliation_error_count": (
                sampling_reconciliation_errors
            ),
            "peak_reconciliation_error_count": peak_reconciliation_errors,
            "nonfinite_telemetry_value_count": 0,
            "max_base_motion_abs": maximum_base_motion,
            "max_torso_abs": maximum_torso,
            "command_cap_error_count": int(not cap),
        },
    )


def _evaluate_grounding(evidence: Mapping[str, object]) -> _Evaluation:
    checks = {name: False for name in GROUNDING_CHECKS}
    metrics = _empty_metrics()
    try:
        if set(evidence) != _ROOT_FIELDS or evidence.get("schema") != GROUNDING_EVIDENCE_SCHEMA:
            return _Evaluation(False, checks, metrics)
        if not isinstance(evidence.get("task"), str) or not evidence["task"]:
            return _Evaluation(False, checks, metrics)
        _integer(evidence.get("seed"), "seed")
        checks["closed_evidence_schema"] = True
        metrics["evidence_schema_error_count"] = 0

        sources = _mapping(evidence["source_hashes"], _SOURCE_FIELDS, "source hashes")
        checks["installed_model_sources"] = sources == {
            "robot_model_path": ROBOT_MODEL_SOURCE_PATH,
            "robot_model_sha256": ROBOT_MODEL_SHA256,
            "gripper_model_path": GRIPPER_MODEL_SOURCE_PATH,
            "gripper_model_sha256": GRIPPER_MODEL_SHA256,
            "dynamic_witness_sha256": DYNAMIC_WITNESS_SHA256,
        }
        metrics["installed_source_mismatch_count"] = int(
            not checks["installed_model_sources"]
        )

        calibration = _calibration(evidence["camera_calibration"])
        checks["official_camera_calibration"] = True
        metrics["calibration_error_count"] = 0

        authority = _mapping(evidence["authority"], _AUTHORITY_FIELDS, "authority")
        controller_payload_count = _integer(
            authority["controller_model_payload_count"],
            "controller model payload count",
        )
        critic_payload_count = _integer(
            authority["critic_model_payload_count"],
            "critic model payload count",
        )
        metrics["controller_model_payload_count"] = controller_payload_count
        metrics["critic_model_payload_count"] = critic_payload_count
        authority_labels_match = (
            authority["probe_target_authority"] == "gate_only_deterministic_probe"
            and authority["normal_episode_target_authority"] == "qwen_only"
        )
        metrics["authority_label_error_count"] = int(
            not authority_labels_match
        )
        checks["qwen_only_target_authority"] = (
            authority_labels_match
            and controller_payload_count == 0
            and critic_payload_count == 0
        )
        checks["raw_oracle_process_local"] = (
            authority["raw_oracle_in_model_payload"] is False
            and authority["raw_oracle_persisted"] is False
            and authority["probe_targets_persisted"] is False
        )
        metrics["raw_oracle_model_payload_count"] = int(
            authority["raw_oracle_in_model_payload"] is not False
        )
        metrics["raw_oracle_persisted_count"] = int(
            authority["raw_oracle_persisted"] is not False
        )
        metrics["probe_targets_persisted_count"] = int(
            authority["probe_targets_persisted"] is not False
        )

        baseline = _audit(evidence["baseline"], "baseline settle")
        rest_baseline = _audit(evidence["rest_baseline"], "rest baseline")
        rest = rest_baseline["samples"]
        rest_endpoint = rest_baseline["receipt"]["bounded_endpoint"]
        rest_actions = rest_baseline["actions"]
        rest_structure_reconciles = (
            len(rest) >= REST_SAMPLE_COUNT_MIN
            and rest_baseline["trace_reconciles"]
            and _json_equal(
                baseline["completion"]["state.arm_joint_position"],
                rest_baseline["before"]["state.arm_joint_position"],
            )
            and _json_equal(
                rest_endpoint,
                rest_baseline["before"]["state.arm_joint_position"],
            )
            and baseline["receipt"]["gripper_intent"] == 1.0
            and all(
                action["gripper_open"] == 1.0
                for action in baseline["actions"]
            )
            and rest_baseline["receipt"]["gripper_intent"] == 1.0
            and rest_baseline["receipt"]["endpoint_error"]
            <= JOINT_STEP_TOLERANCE
            and all(
                _json_equal(action["commanded_qpos"], rest_endpoint)
                and action["gripper_open"] == 1.0
                and all(abs(value) <= 1e-12 for value in action["base_motion"])
                and abs(action["torso"]) <= 1e-12
                for action in rest_actions
            )
        )
        checks["stable_rest_baseline"] = rest_structure_reconciles
        metrics["stable_rest_reconciliation_error_count"] = int(
            not rest_structure_reconciles
        )
        if rest:
            rest_span_samples = [
                rest_baseline["before"],
                *rest,
                rest_baseline["completion"],
            ]
            rest_qvel_max = max(
                abs(component)
                for value in rest_span_samples
                for component in value["state.arm_joint_velocity"]
            )
            rest_qpos_span = _span(
                [
                    value["state.arm_joint_position"]
                    for value in rest_span_samples
                ]
            )
            rest_torque_span = _span(
                [
                    value["state.arm_applied_torque"]["values_nm"]
                    for value in rest_span_samples
                ]
            )
            rest_force_span = _span(
                [
                    value["state.end_effector_wrench"]["force_n"]
                    for value in rest_span_samples
                ]
            )
            rest_wrench_torque_span = _span(
                [
                    value["state.end_effector_wrench"]["torque_nm"]
                    for value in rest_span_samples
                ]
            )
            metrics.update(
                {
                    "max_rest_qpos_span_rad": rest_qpos_span,
                    "max_rest_qvel_abs_rad_s": rest_qvel_max,
                    "max_rest_torque_span_nm": rest_torque_span,
                    "max_rest_force_span_n": rest_force_span,
                    "max_rest_wrench_torque_span_nm": rest_wrench_torque_span,
                    "max_rest_endpoint_error_rad": rest_baseline["receipt"][
                        "endpoint_error"
                    ],
                }
            )
            checks["stable_rest_baseline"] &= (
                rest_qvel_max <= REST_QVEL_MAX_ABS_RAD_S
                and rest_qpos_span <= REST_QPOS_SPAN_MAX_RAD
                and rest_torque_span <= REST_TORQUE_SPAN_MAX_NM
                and rest_force_span <= REST_FORCE_SPAN_MAX_N
                and rest_wrench_torque_span <= REST_WRENCH_TORQUE_SPAN_MAX_NM
            )

        static = _mapping(evidence["static"], _STATIC_FIELDS, "static evidence")
        static_qpos = _vector(static["qpos"], 7, "static qpos")
        link0_world = _pose_record(static["link0_world_pose"], "link0 world pose")
        oracle_static = _pose_record(
            static["oracle_grip_site_world_pose"], "oracle grip-site pose"
        )
        model_static = _compose_pose(link0_world, panda_fk(static_qpos))
        static_bound_to_rest = _json_equal(
            static_qpos,
            rest_baseline["completion"]["state.arm_joint_position"],
        )
        metrics["static_binding_error_count"] = int(not static_bound_to_rest)
        static_position_error = _norm(_delta(model_static.position_m, oracle_static.position_m))
        static_rotation_error = _rotation_error(model_static, oracle_static)
        metrics["static_position_error_m"] = static_position_error
        metrics["static_rotation_error_rad"] = static_rotation_error
        checks["static_fk_position"] = (
            static_bound_to_rest
            and static_position_error <= STATIC_POSITION_ERROR_MAX_M
        )
        checks["static_fk_rotation"] = (
            static_bound_to_rest
            and static_rotation_error <= STATIC_ROTATION_ERROR_MAX_RAD
        )
        public_pixels = _external_pixels(static["public_external_pixels"], "static pixels")
        static_pixels_bound = _json_equal(
            public_pixels,
            rest_baseline["completion"]["state.end_effector_external_pixels"],
        )
        metrics["pixel_binding_error_count"] = int(not static_pixels_bound)
        pixel_errors: list[float] = []
        for camera in ("left", "right"):
            model_px = project_world_point(model_static.position_m, calibration[camera])
            oracle_px = project_world_point(oracle_static.position_m, calibration[camera])
            record = public_pixels[camera]
            if (
                record["depth_valid"] is not True
                or record["visible"] is not oracle_px["visible"]
            ):
                raise ValueError("external pixel conditional state drifted")
            assert record["u_px"] is not None and record["v_px"] is not None
            pixel_errors.extend(
                (
                    math.hypot(
                        model_px["u_px"] - oracle_px["u_px"],
                        model_px["v_px"] - oracle_px["v_px"],
                    ),
                    math.hypot(
                        record["u_px"] - oracle_px["u_px"],
                        record["v_px"] - oracle_px["v_px"],
                    ),
                )
            )
        maximum_pixel_error = max(pixel_errors)
        metrics["max_external_pixel_error_px"] = maximum_pixel_error
        checks["external_pixel_agreement"] = (
            static_pixels_bound
            and maximum_pixel_error <= EXTERNAL_PIXEL_ERROR_MAX_PX
        )

        probe_values = _sequence(evidence["probes"], "probes")
        probes: list[dict[str, object]] = []
        audits = [baseline, rest_baseline]
        labeled_audits = [
            ("baseline settle", baseline),
            ("rest baseline", rest_baseline),
        ]
        joint_indexes: list[int] = []
        clearances: list[float] = []
        commands: list[float] = []
        displacements: list[float] = []
        cosines: list[float] = []
        relative_errors: list[float] = []
        dynamic_vector_norms: list[float] = []
        return_errors: list[float] = []
        safe_directions = True
        safe_commands = True
        realized_motion = True
        nondegenerate = True
        cosine_ok = True
        relative_ok = True
        returns_ok = True
        probe_bindings = True
        previous_return_qpos = list(static_qpos)
        previous_oracle_return = oracle_static
        for number, raw_probe in enumerate(probe_values):
            probe = _mapping(raw_probe, _PROBE_FIELDS, f"probe {number}")
            joint_index = _integer(probe["joint_index"], f"probe {number} joint")
            direction = _integer(probe["direction"], f"probe {number} direction")
            joint_indexes.append(joint_index)
            start = _vector(probe["start_qpos"], 7, f"probe {number} start")
            target = _vector(probe["target_qpos"], 7, f"probe {number} target")
            outbound_qpos = _vector(
                probe["outbound_qpos"], 7, f"probe {number} outbound"
            )
            returned_qpos = _vector(
                probe["return_qpos"], 7, f"probe {number} return"
            )
            outbound = _audit(probe["outbound"], f"probe {number} outbound")
            returned = _audit(probe["return"], f"probe {number} return")
            audits.extend((outbound, returned))
            labeled_audits.extend(
                (
                    (f"probe {number} outbound", outbound),
                    (f"probe {number} return", returned),
                )
            )
            if not 0 <= joint_index < 7 or direction not in {-1, 1}:
                safe_directions = False
                joint_index = min(6, max(0, joint_index))
            lower, upper = JOINT_LIMITS[joint_index]
            clearance = upper - start[joint_index] if direction == 1 else start[joint_index] - lower
            clearances.append(clearance)
            safe_directions &= clearance >= PROBE_LIMIT_CLEARANCE_MIN_RAD
            changes = _delta(target, start)
            command = abs(changes[joint_index])
            commands.append(command)
            safe_commands &= (
                0.0 < command <= PROBE_COMMAND_RAD + 1e-12
                and changes[joint_index] * direction > 0.0
                and all(abs(value) <= 1e-12 for index, value in enumerate(changes) if index != joint_index)
            )
            displacement = abs(outbound_qpos[joint_index] - start[joint_index])
            displacements.append(displacement)
            realized_motion &= (
                displacement >= REALIZED_DISPLACEMENT_MIN_RAD
                and (outbound_qpos[joint_index] - start[joint_index]) * direction > 0.0
            )
            return_error = max(
                abs(value - initial)
                for value, initial in zip(returned_qpos, start, strict=True)
            )
            return_errors.append(return_error)
            returns_ok &= return_error <= RETURN_ERROR_MAX_RAD
            oracle_start = _pose_record(
                probe["oracle_start_pose"], f"probe {number} oracle start"
            )
            oracle_outbound = _pose_record(
                probe["oracle_outbound_pose"], f"probe {number} oracle outbound"
            )
            oracle_return = _pose_record(
                probe["oracle_return_pose"], f"probe {number} oracle return"
            )
            model_start = _compose_pose(link0_world, panda_fk(start))
            model_outbound = _compose_pose(link0_world, panda_fk(outbound_qpos))
            model_return = _compose_pose(link0_world, panda_fk(returned_qpos))
            expected_return_endpoint = list(outbound_qpos)
            expected_return_endpoint[joint_index] = start[joint_index]
            probe_bindings &= (
                _json_equal(start, previous_return_qpos)
                and oracle_start == previous_oracle_return
                and _json_equal(
                    outbound["before"]["state.arm_joint_position"], start
                )
                and _json_equal(outbound["receipt"]["bounded_endpoint"], target)
                and outbound["receipt"]["gripper_intent"] == 1.0
                and _json_equal(
                    outbound["receipt"]["realized_arm_qpos"], outbound_qpos
                )
                and _json_equal(
                    returned["before"]["state.arm_joint_position"], outbound_qpos
                )
                and _json_equal(
                    returned["receipt"]["bounded_endpoint"],
                    expected_return_endpoint,
                )
                and returned["receipt"]["gripper_intent"] == 1.0
                and _json_equal(
                    returned["receipt"]["realized_arm_qpos"], returned_qpos
                )
                and _pose_within(
                    model_start,
                    oracle_start,
                    position_m=STATIC_POSITION_ERROR_MAX_M,
                    rotation_rad=STATIC_ROTATION_ERROR_MAX_RAD,
                )
                and _pose_within(
                    model_outbound,
                    oracle_outbound,
                    position_m=STATIC_POSITION_ERROR_MAX_M,
                    rotation_rad=STATIC_ROTATION_ERROR_MAX_RAD,
                )
                and _pose_within(
                    model_return,
                    oracle_return,
                    position_m=STATIC_POSITION_ERROR_MAX_M,
                    rotation_rad=STATIC_ROTATION_ERROR_MAX_RAD,
                )
            )
            predicted = _delta(
                _transform_point(model_outbound, DYNAMIC_WITNESS_OFFSET_GRIP_SITE_M),
                _transform_point(model_start, DYNAMIC_WITNESS_OFFSET_GRIP_SITE_M),
            )
            realized_vector = _delta(
                _transform_point(oracle_outbound, DYNAMIC_WITNESS_OFFSET_GRIP_SITE_M),
                _transform_point(oracle_start, DYNAMIC_WITNESS_OFFSET_GRIP_SITE_M),
            )
            predicted_norm = _norm(predicted)
            realized_norm = _norm(realized_vector)
            dynamic_vector_norms.extend((predicted_norm, realized_norm))
            vector_ok = (
                predicted_norm >= DYNAMIC_VECTOR_NORM_MIN_M
                and realized_norm >= DYNAMIC_VECTOR_NORM_MIN_M
            )
            nondegenerate &= vector_ok
            if vector_ok:
                cosine = sum(
                    predicted[index] * realized_vector[index] for index in range(3)
                ) / (predicted_norm * realized_norm)
                relative_error = _norm(_delta(predicted, realized_vector)) / realized_norm
            else:
                cosine = -1.0
                relative_error = math.inf
            cosines.append(cosine)
            relative_errors.append(relative_error)
            cosine_ok &= cosine >= DYNAMIC_TRANSLATION_COSINE_MIN
            relative_ok &= relative_error <= DYNAMIC_TRANSLATION_RELATIVE_ERROR_MAX
            probes.append(dict(probe))
            previous_return_qpos = returned_qpos
            previous_oracle_return = oracle_return
        checks["all_seven_joints_probed"] = (
            len(probes) == 7
            and sorted(joint_indexes) == list(range(7))
            and probe_bindings
        )
        metrics["probe_binding_error_count"] = int(
            not checks["all_seven_joints_probed"]
        )
        metrics["joint_direction_error_count"] = int(not safe_directions)
        metrics["probe_command_shape_error_count"] = int(not safe_commands)
        metrics["realized_direction_error_count"] = int(not realized_motion)
        checks["joint_limit_clearance"] = safe_directions
        checks["safe_probe_command"] = safe_commands
        checks["realized_joint_displacement"] = realized_motion
        checks["dynamic_translation_nondegenerate"] = nondegenerate
        checks["dynamic_translation_cosine"] = cosine_ok
        checks["dynamic_translation_relative_error"] = relative_ok
        checks["return_to_start"] = returns_ok
        metrics["probe_count"] = len(probes)
        metrics["min_joint_limit_clearance_rad"] = min(clearances, default=None)
        metrics["max_probe_command_rad"] = max(commands, default=None)
        metrics["min_realized_joint_displacement_rad"] = min(displacements, default=None)
        metrics["min_dynamic_translation_cosine"] = min(cosines, default=None)
        metrics["max_dynamic_translation_relative_error"] = max(relative_errors, default=None)
        metrics["min_dynamic_vector_norm_m"] = min(
            dynamic_vector_norms,
            default=None,
        )
        metrics["max_return_error_rad"] = max(return_errors, default=None)

        gripper = _mapping(evidence["gripper"], _GRIPPER_FIELDS, "gripper")
        reset_gripper = _vector(gripper["reset_qpos"], 2, "reset gripper")
        closed_gripper = _vector(gripper["closed_qpos"], 2, "closed gripper")
        reopened_gripper = _vector(gripper["reopened_qpos"], 2, "reopened gripper")
        close_audit = _audit(gripper["close"], "gripper close")
        reopen_audit = _audit(gripper["reopen"], "gripper reopen")
        audits.extend((close_audit, reopen_audit))
        labeled_audits.extend(
            (("gripper close", close_audit), ("gripper reopen", reopen_audit))
        )
        journal_reconciles, journal_views = _reconcile_source_journal(
            evidence,
            labeled_audits,
        )
        close_journal = journal_views.get("gripper close", {})
        reopen_journal = journal_views.get("gripper reopen", {})
        reset_separation = _finger_separation(reset_gripper)
        closed_separation = _finger_separation(closed_gripper)
        reopened_separation = _finger_separation(reopened_gripper)
        metrics.update(
            {
                "gripper_transition_audit_count": 2,
                "min_gripper_transition_actions": min(
                    len(close_audit["actions"]),
                    len(reopen_audit["actions"]),
                ),
                "gripper_reset_separation_m": reset_separation,
                "gripper_closed_separation_m": closed_separation,
                "gripper_reopened_separation_m": reopened_separation,
                "gripper_close_separation_delta_m": (
                    reset_separation - closed_separation
                ),
                "gripper_reopen_separation_delta_m": (
                    reopened_separation - closed_separation
                ),
            }
        )
        close_journal_binding = bool(close_journal) and (
            _json_equal(
                close_journal["before"]["gripper_qpos"], reset_gripper
            )
            and _json_equal(
                close_journal["completion"]["gripper_qpos"], closed_gripper
            )
            and close_journal["completion"]["finger_separation_m"]
            < close_journal["before"]["finger_separation_m"] - 1e-4
        )
        reopen_journal_binding = bool(reopen_journal) and (
            _json_equal(
                reopen_journal["before"]["gripper_qpos"], closed_gripper
            )
            and _json_equal(
                reopen_journal["completion"]["gripper_qpos"],
                reopened_gripper,
            )
            and reopen_journal["completion"]["finger_separation_m"]
            > reopen_journal["before"]["finger_separation_m"] + 1e-4
        )
        close_binding = (
            _json_equal(
                close_audit["before"]["state.arm_joint_position"],
                previous_return_qpos,
            )
            and _json_equal(
                close_audit["receipt"]["bounded_endpoint"],
                previous_return_qpos,
            )
            and close_audit["receipt"]["gripper_intent"] == 0.0
            and all(
                action["gripper_open"] == 0.0
                for action in close_audit["actions"]
            )
            and close_journal_binding
        )
        reopen_binding = (
            _json_equal(
                reopen_audit["before"]["state.arm_joint_position"],
                close_audit["receipt"]["realized_arm_qpos"],
            )
            and _json_equal(
                reopen_audit["receipt"]["bounded_endpoint"],
                close_audit["receipt"]["realized_arm_qpos"],
            )
            and reopen_audit["receipt"]["gripper_intent"] == 1.0
            and all(
                action["gripper_open"] == 1.0
                for action in reopen_audit["actions"]
            )
            and reopen_journal_binding
        )
        metrics["gripper_close_binding_error_count"] = int(not close_binding)
        metrics["gripper_reopen_binding_error_count"] = int(not reopen_binding)
        checks["gripper_transition_actions"] = (
            len(close_audit["actions"]) >= MIN_GRIPPER_ACTIONS
            and len(reopen_audit["actions"]) >= MIN_GRIPPER_ACTIONS
            and close_binding
            and reopen_binding
        )
        close_direction = all(
            abs(closed - target) < abs(reset - target) - 1e-4
            for reset, closed, target in zip(
                reset_gripper, closed_gripper, (0.0, 0.0), strict=True
            )
        )
        reopen_direction = all(
            abs(reopened - target) < abs(closed - target) - 1e-4
            for closed, reopened, target in zip(
                closed_gripper, reopened_gripper, (0.04, -0.04), strict=True
            )
        )
        metrics["gripper_close_direction_error_count"] = int(
            not close_direction
        )
        metrics["gripper_reopen_direction_error_count"] = int(
            not reopen_direction
        )
        checks["gripper_close_direction"] = close_binding and close_direction
        checks["gripper_reopen_direction"] = reopen_binding and reopen_direction

        action_checks, action_metrics = _action_checks(audits)
        checks.update(action_checks)
        metrics.update(action_metrics)
        metrics["sampling_reconciliation_error_count"] = int(
            metrics["sampling_reconciliation_error_count"]
        ) + int(not journal_reconciles)
        checks["complete_telemetry_sampling"] &= journal_reconciles
        action_count = _integer(evidence["action_count"], "action count")
        terminal_action_count = _integer(
            evidence["terminal_action_count"], "terminal action count"
        )
        audited_action_count = sum(len(audit["actions"]) for audit in audits)
        checks["action_accounting_exact"] = (
            action_count == terminal_action_count == audited_action_count
        )
        realistic_action_minimum = (
            (GROUNDING_AUDIT_COUNT - 2) + 2 * MIN_GRIPPER_ACTIONS
        )
        checks["episode_action_budget"] = (
            realistic_action_minimum
            <= action_count
            <= EPISODE_ACTION_BUDGET
        )
        metrics["action_count"] = action_count
        metrics["terminal_action_count"] = terminal_action_count
        metrics["audited_action_count"] = audited_action_count
        checks["network_disabled"] = evidence["network_probe"] == "network namespace denied"
        metrics["network_error_count"] = int(not checks["network_disabled"])
        checks["simulator_clean"] = (
            evidence["simulator_error"] is None
            and evidence["episode_tmp_empty"] is True
        )
        metrics["simulator_error_count"] = int(
            evidence["simulator_error"] is not None
        )
        metrics["episode_tmp_file_count"] = int(
            evidence["episode_tmp_empty"] is not True
        )
    except (AssertionError, KeyError, TypeError, ValueError, ZeroDivisionError):
        pass
    return _Evaluation(all(checks.values()), checks, metrics)


def evaluate_grounding_smoke(
    evidence: Mapping[str, object],
) -> tuple[bool, dict[str, bool]]:
    """Evaluate the exact closed gate evidence; malformed input fails closed."""

    if not isinstance(evidence, Mapping):
        return False, {name: False for name in GROUNDING_CHECKS}
    result = _evaluate_grounding(evidence)
    return result.passed, result.checks


def _derive_public_grounding_checks(
    metrics: Mapping[str, object],
    provenance: Mapping[str, object],
) -> dict[str, bool]:
    """Recompute every public check from aggregate values and provenance."""

    def number(name: str) -> float | None:
        value = metrics.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            return None
        return float(value)

    def integer(name: str) -> int | None:
        value = metrics.get(name)
        return value if type(value) is int and value >= 0 else None

    def zero(name: str) -> bool:
        return integer(name) == 0

    def bounded(name: str, *, minimum: float, maximum: float) -> bool:
        value = number(name)
        return value is not None and minimum <= value <= maximum

    audit_count = integer("audit_count")
    nonempty_count = integer("nonempty_audit_count")
    action_count = integer("action_count")
    terminal_action_count = integer("terminal_action_count")
    audited_action_count = integer("audited_action_count")
    telemetry_count = integer("telemetry_sample_count")
    min_actions = integer("min_actions_per_command")
    max_actions = integer("max_actions_per_command")
    gripper_audits = integer("gripper_transition_audit_count")
    min_gripper_actions = integer("min_gripper_transition_actions")
    reset_separation = number("gripper_reset_separation_m")
    closed_separation = number("gripper_closed_separation_m")
    reopened_separation = number("gripper_reopened_separation_m")
    close_delta = number("gripper_close_separation_delta_m")
    reopen_delta = number("gripper_reopen_separation_delta_m")
    close_relation = (
        None not in (reset_separation, closed_separation, close_delta)
        and close_delta is not None
        and reset_separation is not None
        and closed_separation is not None
        and reset_separation >= 0.0
        and closed_separation >= 0.0
        and close_delta > 1e-4
        and math.isclose(
            reset_separation - closed_separation,
            close_delta,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    reopen_relation = (
        None not in (closed_separation, reopened_separation, reopen_delta)
        and reopen_delta is not None
        and reopened_separation is not None
        and closed_separation is not None
        and closed_separation >= 0.0
        and reopened_separation >= 0.0
        and reopen_delta > 1e-4
        and math.isclose(
            reopened_separation - closed_separation,
            reopen_delta,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    exact_audits = (
        audit_count == GROUNDING_AUDIT_COUNT
        and nonempty_count == GROUNDING_AUDIT_COUNT
    )
    telemetry_relation = (
        exact_audits
        and audited_action_count is not None
        and telemetry_count
        == audited_action_count + 2 * GROUNDING_AUDIT_COUNT
        and zero("sampling_reconciliation_error_count")
        and type(provenance.get("journal_terminal_sha256")) is str
        and re.fullmatch(
            r"[0-9a-f]{64}",
            str(provenance.get("journal_terminal_sha256")),
        )
        is not None
    )
    command_cap = (
        min_actions is not None
        and max_actions is not None
        and 1 <= min_actions <= max_actions <= MAX_COMMAND_ACTIONS
        and zero("command_cap_error_count")
    )
    gripper_actions = (
        gripper_audits == 2
        and min_gripper_actions is not None
        and min_gripper_actions >= MIN_GRIPPER_ACTIONS
        and zero("gripper_close_binding_error_count")
        and zero("gripper_reopen_binding_error_count")
    )
    required_pauses = integer("tracking_required_pause_count")
    observed_pauses = integer("tracking_observed_pause_count")
    checks = {
        "closed_evidence_schema": (
            provenance.get("evidence_schema") == GROUNDING_EVIDENCE_SCHEMA
            and zero("evidence_schema_error_count")
        ),
        "installed_model_sources": (
            provenance.get("robot_model_sha256") == ROBOT_MODEL_SHA256
            and provenance.get("gripper_model_sha256") == GRIPPER_MODEL_SHA256
            and provenance.get("dynamic_witness_sha256")
            == DYNAMIC_WITNESS_SHA256
            and zero("installed_source_mismatch_count")
        ),
        "official_camera_calibration": (
            provenance.get("camera_calibration_source")
            == OFFICIAL_CALIBRATION_SOURCE
            and zero("calibration_error_count")
        ),
        "qwen_only_target_authority": (
            provenance.get("probe_target_authority")
            == "gate_only_deterministic_probe"
            and provenance.get("normal_episode_target_authority") == "qwen_only"
            and zero("authority_label_error_count")
            and zero("controller_model_payload_count")
            and zero("critic_model_payload_count")
        ),
        "raw_oracle_process_local": (
            provenance.get("raw_oracle_scope")
            == "isolated_process_memory_only"
            and provenance.get("persisted_value_policy")
            == "aggregate_errors_checks_hashes_counts_labels_only"
            and zero("raw_oracle_model_payload_count")
            and zero("raw_oracle_persisted_count")
            and zero("probe_targets_persisted_count")
        ),
        "stable_rest_baseline": (
            zero("stable_rest_reconciliation_error_count")
            and bounded(
                "max_rest_qpos_span_rad",
                minimum=0.0,
                maximum=REST_QPOS_SPAN_MAX_RAD,
            )
            and bounded(
                "max_rest_qvel_abs_rad_s",
                minimum=0.0,
                maximum=REST_QVEL_MAX_ABS_RAD_S,
            )
            and bounded(
                "max_rest_torque_span_nm",
                minimum=0.0,
                maximum=REST_TORQUE_SPAN_MAX_NM,
            )
            and bounded(
                "max_rest_force_span_n",
                minimum=0.0,
                maximum=REST_FORCE_SPAN_MAX_N,
            )
            and bounded(
                "max_rest_wrench_torque_span_nm",
                minimum=0.0,
                maximum=REST_WRENCH_TORQUE_SPAN_MAX_NM,
            )
            and bounded(
                "max_rest_endpoint_error_rad",
                minimum=0.0,
                maximum=JOINT_STEP_TOLERANCE,
            )
        ),
        "static_fk_position": (
            zero("static_binding_error_count")
            and bounded(
                "static_position_error_m",
                minimum=0.0,
                maximum=STATIC_POSITION_ERROR_MAX_M,
            )
        ),
        "static_fk_rotation": (
            zero("static_binding_error_count")
            and bounded(
                "static_rotation_error_rad",
                minimum=0.0,
                maximum=STATIC_ROTATION_ERROR_MAX_RAD,
            )
        ),
        "external_pixel_agreement": (
            zero("pixel_binding_error_count")
            and bounded(
                "max_external_pixel_error_px",
                minimum=0.0,
                maximum=EXTERNAL_PIXEL_ERROR_MAX_PX,
            )
        ),
        "all_seven_joints_probed": (
            integer("probe_count") == 7
            and exact_audits
            and zero("probe_binding_error_count")
        ),
        "joint_limit_clearance": (
            zero("joint_direction_error_count")
            and (
                (value := number("min_joint_limit_clearance_rad")) is not None
                and value >= PROBE_LIMIT_CLEARANCE_MIN_RAD
            )
        ),
        "safe_probe_command": (
            zero("probe_command_shape_error_count")
            and bounded(
                "max_probe_command_rad",
                minimum=1e-15,
                maximum=PROBE_COMMAND_RAD + 1e-12,
            )
        ),
        "realized_joint_displacement": (
            zero("realized_direction_error_count")
            and (
                (value := number("min_realized_joint_displacement_rad"))
                is not None
                and value >= REALIZED_DISPLACEMENT_MIN_RAD
            )
        ),
        "dynamic_translation_nondegenerate": (
            (value := number("min_dynamic_vector_norm_m")) is not None
            and value >= DYNAMIC_VECTOR_NORM_MIN_M
        ),
        "dynamic_translation_cosine": bounded(
            "min_dynamic_translation_cosine",
            minimum=DYNAMIC_TRANSLATION_COSINE_MIN,
            maximum=1.0,
        ),
        "dynamic_translation_relative_error": bounded(
            "max_dynamic_translation_relative_error",
            minimum=0.0,
            maximum=DYNAMIC_TRANSLATION_RELATIVE_ERROR_MAX,
        ),
        "return_to_start": bounded(
            "max_return_error_rad",
            minimum=0.0,
            maximum=RETURN_ERROR_MAX_RAD,
        ),
        "complete_telemetry_sampling": telemetry_relation,
        "finite_telemetry": zero("nonfinite_telemetry_value_count"),
        "telemetry_peaks_reconcile": zero("peak_reconciliation_error_count"),
        "joint_limit_inset": (
            (value := number("min_joint_limit_inset_rad")) is not None
            and value >= 0.0
        ),
        "first_step_tolerance": bounded(
            "max_first_step_rad",
            minimum=0.0,
            maximum=MAX_JOINT_STEP + JOINT_STEP_TOLERANCE + 1e-12,
        ),
        "later_step_bound": bounded(
            "max_later_step_rad",
            minimum=0.0,
            maximum=MAX_JOINT_STEP + 1e-12,
        ),
        "tracking_pauses_reconcile": (
            required_pauses is not None
            and required_pauses == observed_pauses
            and zero("tracking_violation_count")
            and zero("tracking_receipt_mismatch_count")
        ),
        "command_action_cap": command_cap,
        "gripper_transition_actions": gripper_actions,
        "base_and_torso_stationary": (
            bounded("max_base_motion_abs", minimum=0.0, maximum=1e-12)
            and bounded("max_torso_abs", minimum=0.0, maximum=1e-12)
        ),
        "gripper_close_direction": (
            close_relation
            and zero("gripper_close_binding_error_count")
            and zero("gripper_close_direction_error_count")
        ),
        "gripper_reopen_direction": (
            reopen_relation
            and zero("gripper_reopen_binding_error_count")
            and zero("gripper_reopen_direction_error_count")
        ),
        "action_accounting_exact": (
            action_count is not None
            and action_count == terminal_action_count
            and action_count == audited_action_count
        ),
        "episode_action_budget": (
            action_count is not None
            and (GROUNDING_AUDIT_COUNT - 2) + 2 * MIN_GRIPPER_ACTIONS
            <= action_count
            <= EPISODE_ACTION_BUDGET
        ),
        "network_disabled": (
            provenance.get("network_policy") == "network_namespace_denied"
            and zero("network_error_count")
        ),
        "simulator_clean": (
            provenance.get("simulator_status") == "clean"
            and zero("simulator_error_count")
            and zero("episode_tmp_file_count")
        ),
    }
    if set(checks) != GROUNDING_CHECKS:
        raise RuntimeError("grounding derived check contract drifted")
    return checks


def validate_public_grounding_result(
    value: object,
    *,
    expected_task: str,
    expected_seed: int,
    expected_release_digest: str,
    expected_artifact_root: str | Path,
) -> dict[str, object]:
    """Validate the aggregate-only result that may cross the child boundary."""

    if (
        type(expected_task) is not str
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", expected_task) is None
        or type(expected_seed) is not int
        or expected_seed < 0
        or type(expected_release_digest) is not str
        or re.fullmatch(r"[0-9a-f]{16}", expected_release_digest) is None
    ):
        raise ValueError("grounding expected identity drifted")
    expected_artifact = str(expected_artifact_root)
    expected_artifact_path = PurePosixPath(expected_artifact)
    if (
        not expected_artifact_path.is_absolute()
        or str(expected_artifact_path) != expected_artifact
        or any(part in {".", ".."} for part in expected_artifact_path.parts)
    ):
        raise ValueError("grounding expected artifact root drifted")
    result = _mapping(value, set(PUBLIC_RESULT_FIELDS), "grounding result")
    if result["schema"] != GROUNDING_RESULT_SCHEMA:
        raise ValueError("grounding result schema drifted")
    task = result["task"]
    if (
        type(task) is not str
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", task) is None
        or task != expected_task
    ):
        raise ValueError("grounding result task drifted")
    seed = result["seed"]
    if type(seed) is not int or seed < 0 or seed != expected_seed:
        raise ValueError("grounding result seed drifted")
    wall_s = result["wall_s"]
    if (
        isinstance(wall_s, bool)
        or not isinstance(wall_s, (int, float))
        or not math.isfinite(float(wall_s))
        or float(wall_s) <= 0.0
        or float(wall_s) > GROUNDING_WALL_TIME_MAX_S
    ):
        raise ValueError("grounding result wall time drifted")
    artifact_root = result["artifact_root"]
    if type(artifact_root) is not str:
        raise TypeError("grounding result artifact root must be a string")
    artifact_path = PurePosixPath(artifact_root)
    if (
        not artifact_path.is_absolute()
        or str(artifact_path) != artifact_root
        or any(part in {".", ".."} for part in artifact_path.parts)
        or artifact_root != expected_artifact
    ):
        raise ValueError("grounding result artifact root drifted")
    release_digest = result["release_digest"]
    if (
        type(release_digest) is not str
        or re.fullmatch(r"[0-9a-f]{16}", release_digest) is None
        or release_digest != expected_release_digest
    ):
        raise ValueError("grounding result release digest drifted")
    if type(result["passed"]) is not bool:
        raise TypeError("grounding result passed flag must be boolean")
    if len(GROUNDING_CHECKS) != 33:
        raise RuntimeError("grounding result check contract drifted")
    if not isinstance(result["checks"], Mapping) or set(result["checks"]) != GROUNDING_CHECKS:
        raise ValueError("grounding result checks drifted")
    if any(type(item) is not bool for item in result["checks"].values()):
        raise TypeError("grounding result checks must be booleans")
    if result["passed"] is not all(result["checks"].values()):
        raise ValueError("grounding result passed flag does not match checks")
    metrics = result["metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != GROUNDING_METRIC_FIELDS:
        raise ValueError("grounding result metrics drifted")
    for name, metric in metrics.items():
        if metric is None:
            continue
        if name in GROUNDING_COUNT_FIELDS:
            if type(metric) is not int or metric < 0:
                raise ValueError("grounding result counts must be nonnegative integers")
        elif (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
        ):
            raise ValueError("grounding result metrics must be finite aggregates")
    if any(
        metrics[name] is not None and float(metrics[name]) < 0.0
        for name in (
            "gripper_reset_separation_m",
            "gripper_closed_separation_m",
            "gripper_reopened_separation_m",
        )
    ):
        raise ValueError("grounding result physical separation drifted")
    if any(
        metrics[name] is None for name in GROUNDING_METRIC_FIELDS
    ) and result["passed"]:
        raise ValueError("grounding result metrics must be finite aggregates")
    provenance = result["provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != GROUNDING_PROVENANCE_FIELDS:
        raise TypeError("grounding result provenance must be a mapping")
    expected_provenance = {
        "probe_target_authority": "gate_only_deterministic_probe",
        "normal_episode_target_authority": "qwen_only",
        "raw_oracle_scope": "isolated_process_memory_only",
        "persisted_value_policy": "aggregate_errors_checks_hashes_counts_labels_only",
        "camera_calibration_source": OFFICIAL_CALIBRATION_SOURCE,
        "robot_model_sha256": ROBOT_MODEL_SHA256,
        "gripper_model_sha256": GRIPPER_MODEL_SHA256,
        "dynamic_witness_sha256": DYNAMIC_WITNESS_SHA256,
        "evidence_schema": GROUNDING_EVIDENCE_SCHEMA,
        "journal_trust_boundary": "process_local_hmac_sha256_snapshot",
        "network_policy": "network_namespace_denied",
        "simulator_status": "clean",
    }
    if any(provenance[name] != expected for name, expected in expected_provenance.items()):
        raise ValueError("grounding result provenance identity drifted")

    def validate_hash(name: str) -> str | None:
        digest = provenance[name]
        if digest is not None and (
            type(digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError(f"grounding result {name} drifted")
        return digest

    journal_terminal_sha256 = validate_hash("journal_terminal_sha256")
    evidence_sha256 = validate_hash("evidence_sha256")
    error_sha256 = validate_hash("error_sha256")
    if result["passed"] and (
        journal_terminal_sha256 is None
        or evidence_sha256 is None
        or error_sha256 is not None
    ):
        raise ValueError("grounding result passed hash relationship drifted")
    derived_checks = _derive_public_grounding_checks(metrics, provenance)
    if dict(result["checks"]) != derived_checks:
        raise ValueError("grounding result checks are not aggregate-derived")
    serialized = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    forbidden = (
        "oracle_grip_site_world_pose",
        "oracle_start_pose",
        "oracle_outbound_pose",
        "oracle_return_pose",
        "start_qpos",
        "outbound_qpos",
        "return_qpos",
        "target_qpos",
        "commanded_qpos",
        "gripper_qpos",
        "finger_separation_m",
        "source_step_index",
        "snapshot_json",
        "journal_range",
        "journal_events",
        "hmac_key",
        "public_external_pixels",
        "camera_position_world_m",
    )
    if any(field in serialized for field in forbidden):
        raise ValueError("grounding result leaks gate-only values")
    return json.loads(serialized)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _safe_failure_label(stage: str, error: BaseException) -> str:
    """Return a diagnostic label that cannot contain private exception values."""

    safe_stage = "".join(
        character for character in stage if character.isalnum() or character == "_"
    )
    return f"grounding-stage={safe_stage or 'unknown'} error={type(error).__name__}"


def _safe_failure_trace(error: BaseException) -> str:
    """Return traceback function labels without lines, values, or messages."""

    names = []
    for frame in traceback.extract_tb(error.__traceback__)[-12:]:
        name = "".join(
            character
            for character in frame.name
            if character.isalpha() or character == "_"
        )
        names.append(name or "unknown")
    return "grounding-trace=" + ">".join(names)


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


def _public_result(
    *,
    task: str,
    seed: int,
    evaluation: _Evaluation,
    artifact_root: Path,
    wall_s: float,
    journal_terminal_sha256: str | None,
    evidence_sha256: str | None,
    error_sha256: str | None,
) -> dict[str, object]:
    release_digest = _installed_release_digest()
    aggregate_metrics = {
        key: value
        if value is None
        or (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        )
        else None
        for key, value in evaluation.metrics.items()
    }
    result = {
        "schema": GROUNDING_RESULT_SCHEMA,
        "task": task,
        "seed": seed,
        "passed": evaluation.passed,
        "checks": evaluation.checks,
        "metrics": aggregate_metrics,
        "provenance": {
            "probe_target_authority": "gate_only_deterministic_probe",
            "normal_episode_target_authority": "qwen_only",
            "raw_oracle_scope": "isolated_process_memory_only",
            "persisted_value_policy": "aggregate_errors_checks_hashes_counts_labels_only",
            "camera_calibration_source": OFFICIAL_CALIBRATION_SOURCE,
            "robot_model_sha256": ROBOT_MODEL_SHA256,
            "gripper_model_sha256": GRIPPER_MODEL_SHA256,
            "dynamic_witness_sha256": DYNAMIC_WITNESS_SHA256,
            "evidence_schema": GROUNDING_EVIDENCE_SCHEMA,
            "journal_trust_boundary": "process_local_hmac_sha256_snapshot",
            "journal_terminal_sha256": journal_terminal_sha256,
            "network_policy": "network_namespace_denied",
            "simulator_status": "clean",
            "evidence_sha256": evidence_sha256,
            "error_sha256": error_sha256,
        },
        "artifact_root": str(artifact_root),
        "release_digest": release_digest,
        "wall_s": wall_s,
    }
    return validate_public_grounding_result(
        result,
        expected_task=task,
        expected_seed=seed,
        expected_release_digest=release_digest,
        expected_artifact_root=artifact_root,
    )


def _raw_pose(environment: object, *, site: bool) -> dict[str, object]:
    import numpy as np

    source = environment.unwrapped
    robot = source.robots[0]
    if site:
        identifier = robot.eef_site_id["right"]
        position = np.asarray(source.sim.data.site_xpos[identifier], dtype=np.float64)
        rotation = np.asarray(source.sim.data.site_xmat[identifier], dtype=np.float64).reshape(3, 3)
    else:
        name = robot.robot_model.correct_naming("link0")
        position = np.asarray(source.sim.data.get_body_xpos(name), dtype=np.float64)
        rotation = np.asarray(source.sim.data.get_body_xmat(name), dtype=np.float64).reshape(3, 3)
    if position.shape != (3,) or rotation.shape != (3, 3) or not np.isfinite(position).all() or not np.isfinite(rotation).all():
        raise RuntimeError("gate pose geometry is invalid")
    return {"position_m": position.tolist(), "rotation_matrix": rotation.tolist()}


def _gripper_qpos(environment: object) -> list[float]:
    import numpy as np

    source = environment.unwrapped
    robot = source.robots[0]
    indexes = robot._ref_gripper_joint_pos_indexes["right"]
    values = np.asarray([source.sim.data.qpos[index] for index in indexes], dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise RuntimeError("gate gripper qpos is invalid")
    return values.tolist()


def _finger_separation(gripper_qpos: Sequence[float]) -> float:
    if len(gripper_qpos) != 2:
        raise ValueError("gate gripper qpos width drifted")
    return abs(float(gripper_qpos[0]) - float(gripper_qpos[1]))


def _run_isolated_child(*, task: str, seed: int, run: Path) -> dict[str, object]:
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa_inspect.camera_geometry import official_camera_calibration

    from . import joint_sim_child as child

    started = time.monotonic()
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    episode_tmp = run / "episode-tmp"
    episode_tmp.mkdir(mode=0o700)
    os.environ["ROBOCASA_EPISODE_TMPDIR"] = str(episode_tmp)
    environment: object | None = None
    raw_evidence: dict[str, object] | None = None
    journal_verifier: _JournalVerifier | None = None
    error_sha256: str | None = None
    evaluation = _Evaluation(
        False,
        {name: False for name in GROUNDING_CHECKS},
        _empty_metrics(),
    )
    simulator_error: str | None = None
    stage = "setup"
    try:
        stage = "controller_install"
        child._install_joint_controller()
        stage = "environment_create"
        environment = gym.make(f"robocasa/{task}", split="pretrain", seed=seed)
        stage = "environment_reset"
        environment.reset()
        stage = "network_probe"
        network_error = child._network_denied()
        stage = "camera_calibration"
        calibration_all = official_camera_calibration(environment)
        calibration = {
            "source": OFFICIAL_CALIBRATION_SOURCE,
            "left": calibration_all["left"],
            "right": calibration_all["right"],
        }
        original_step = child._step
        original_read = child.read_public_telemetry
        active_actions: list[dict[str, object]] | None = None
        active_telemetry: list[dict[str, object]] | None = None
        active_read_kind: str | None = None
        operation = "idle"
        journal = _JournalRecorder()

        def traced_step(env: object, action: Mapping[str, object]) -> Mapping[str, object]:
            nonlocal active_read_kind, stage
            assert active_actions is not None
            stage = f"{operation}_step_before"
            before = child._arm_qpos(env)
            stage = f"{operation}_simulator_step"
            value = original_step(env, action)
            stage = f"{operation}_step_after"
            after = child._arm_qpos(env)
            action_record = {
                "realized_before": before,
                "commanded_qpos": list(action["joint_position"]),
                "realized_after": after,
                "gripper_open": action["gripper_open"],
                "base_motion": list(action["base_motion"]),
                "torso": action["torso"],
            }
            active_actions.append(action_record)
            journal.record_step(action_record)
            active_read_kind = "read_step"
            return value

        def traced_read(env: object, camera: Mapping[str, object]) -> dict[str, object]:
            nonlocal active_read_kind, stage
            stage = f"{operation}_telemetry_read"
            value = original_read(env, camera)
            if active_telemetry is not None:
                active_telemetry.append(value)
            if active_read_kind is not None:
                measured_gripper = _gripper_qpos(env)
                journal.record_read(
                    active_read_kind,
                    telemetry=value,
                    gripper_qpos=measured_gripper,
                )
                active_read_kind = "read_completion"
            return value

        child._step = traced_step
        child.read_public_telemetry = traced_read
        total_actions = 0
        gripper_open = 1.0
        sequence = 0

        def execute(
            endpoint: list[float],
            explicit_mask: list[bool],
            requested_gripper: float,
            *,
            label: str,
        ) -> dict[str, object]:
            nonlocal active_actions, active_read_kind, active_telemetry, total_actions, gripper_open, sequence, operation, stage
            operation = label.replace(" ", "_")
            stage = f"{operation}_prepare"
            active_actions = []
            active_telemetry = []
            active_read_kind = "read_before"
            journal.begin_audit(label)
            command = {
                "schema": "robocasa-inspect-joint-command/v1",
                "sequence": sequence,
                "kind": "move_joints",
                "observation_id": "gate-only-no-model",
                "endpoint": endpoint,
                "explicit_mask": explicit_mask,
                "gripper_open": requested_gripper,
                "previous_gripper_open": gripper_open,
                "gripper_transition": abs(requested_gripper - gripper_open) > 1e-12,
                "max_actions": MAX_COMMAND_ACTIONS,
            }
            stage = f"{operation}_execute_move"
            _, receipt, gripper_open, total_actions = child._execute_move(
                environment,
                command,
                current_gripper=gripper_open,
                total_actions=total_actions,
                started=started,
            )
            sequence += 1
            stage = f"{operation}_trace_finalize"
            calls = active_telemetry
            actions = active_actions
            active_telemetry = None
            active_actions = None
            active_read_kind = None
            if len(calls) != receipt["step_count"] + 2:
                raise RuntimeError("gate telemetry call accounting drifted")
            journal_range = journal.end_audit()
            return {
                "receipt": receipt,
                "telemetry": {
                    "before": calls[0],
                    "samples": calls[1:-1],
                    "completion": calls[-1],
                },
                "actions": actions,
                "journal_range": journal_range,
            }

        reset = child._arm_qpos(environment)
        stage = "baseline"
        baseline = execute(
            list(reset), [False] * 7, 1.0, label="baseline settle"
        )
        stage = "rest_baseline"
        rest_qpos = child._arm_qpos(environment)
        rest_baseline = execute(
            list(rest_qpos),
            [False] * 7,
            1.0,
            label="rest baseline",
        )
        reset = child._arm_qpos(environment)
        stage = "static_geometry"
        static = {
            "qpos": reset,
            "link0_world_pose": _raw_pose(environment, site=False),
            "oracle_grip_site_world_pose": _raw_pose(environment, site=True),
            "public_external_pixels": rest_baseline["telemetry"]["completion"][
                "state.end_effector_external_pixels"
            ],
        }
        probes: list[dict[str, object]] = []
        for joint_index, limits in enumerate(JOINT_LIMITS):
            start_qpos = child._arm_qpos(environment)
            lower, upper = limits
            positive = upper - start_qpos[joint_index]
            negative = start_qpos[joint_index] - lower
            if max(positive, negative) < PROBE_LIMIT_CLEARANCE_MIN_RAD:
                raise RuntimeError("gate probe has no safe direction")
            direction = 1 if positive >= negative else -1
            target = list(start_qpos)
            target[joint_index] += direction * PROBE_COMMAND_RAD
            oracle_start = _raw_pose(environment, site=True)
            stage = f"probe_joint_{joint_index + 1}_outbound"
            outbound = execute(
                target,
                [index == joint_index for index in range(7)],
                1.0,
                label=f"probe {joint_index} outbound",
            )
            outbound_qpos = child._arm_qpos(environment)
            oracle_outbound = _raw_pose(environment, site=True)
            return_target = list(outbound_qpos)
            return_target[joint_index] = start_qpos[joint_index]
            stage = f"probe_joint_{joint_index + 1}_return"
            returned = execute(
                return_target,
                [index == joint_index for index in range(7)],
                1.0,
                label=f"probe {joint_index} return",
            )
            returned_qpos = child._arm_qpos(environment)
            oracle_return = _raw_pose(environment, site=True)
            probes.append(
                {
                    "joint_index": joint_index,
                    "direction": direction,
                    "start_qpos": start_qpos,
                    "target_qpos": target,
                    "outbound_qpos": outbound_qpos,
                    "return_qpos": returned_qpos,
                    "oracle_start_pose": oracle_start,
                    "oracle_outbound_pose": oracle_outbound,
                    "oracle_return_pose": oracle_return,
                    "outbound": outbound,
                    "return": returned,
                }
            )
        reset_gripper = _gripper_qpos(environment)
        arm_hold = child._arm_qpos(environment)
        stage = "gripper_close"
        close = execute(
            arm_hold,
            [False] * 7,
            0.0,
            label="gripper close",
        )
        closed_gripper = _gripper_qpos(environment)
        arm_hold = child._arm_qpos(environment)
        stage = "gripper_reopen"
        reopen = execute(
            arm_hold,
            [False] * 7,
            1.0,
            label="gripper reopen",
        )
        reopened_gripper = _gripper_qpos(environment)
        journal_verifier = journal.freeze()
        raw_evidence = {
            "schema": GROUNDING_EVIDENCE_SCHEMA,
            "task": task,
            "seed": seed,
            "authority": {
                "probe_target_authority": "gate_only_deterministic_probe",
                "normal_episode_target_authority": "qwen_only",
                "controller_model_payload_count": 0,
                "critic_model_payload_count": 0,
                "raw_oracle_in_model_payload": False,
                "raw_oracle_persisted": False,
                "probe_targets_persisted": False,
            },
            "source_hashes": {
                "robot_model_path": ROBOT_MODEL_SOURCE_PATH,
                "robot_model_sha256": hashlib.sha256(Path(ROBOT_MODEL_SOURCE_PATH).read_bytes()).hexdigest(),
                "gripper_model_path": GRIPPER_MODEL_SOURCE_PATH,
                "gripper_model_sha256": hashlib.sha256(Path(GRIPPER_MODEL_SOURCE_PATH).read_bytes()).hexdigest(),
                "dynamic_witness_sha256": DYNAMIC_WITNESS_SHA256,
            },
            "camera_calibration": calibration,
            "baseline": baseline,
            "rest_baseline": rest_baseline,
            "static": static,
            "probes": probes,
            "gripper": {
                "reset_qpos": reset_gripper,
                "closed_qpos": closed_gripper,
                "reopened_qpos": reopened_gripper,
                "close": close,
                "reopen": reopen,
            },
            "journal_terminal_sha256": journal_verifier.snapshot.terminal_sha256,
            "action_count": total_actions,
            "terminal_action_count": total_actions,
            "network_probe": "network namespace denied" if network_error else "",
            "simulator_error": None,
            "episode_tmp_empty": False,
        }
    except Exception as error:
        simulator_error = f"{type(error).__name__}: {error}"[:500]
        error_sha256 = hashlib.sha256(repr(error).encode()).hexdigest()
        print(_safe_failure_label(stage, error), file=sys.stderr, flush=True)
        print(_safe_failure_trace(error), file=sys.stderr, flush=True)
    finally:
        if environment is not None:
            try:
                environment.close()
            except Exception as close_error:
                simulator_error = "simulator close failed"
                error_sha256 = hashlib.sha256(repr(close_error).encode()).hexdigest()
    if raw_evidence is not None and journal_verifier is not None:
        raw_evidence["simulator_error"] = simulator_error
        raw_evidence["episode_tmp_empty"] = not any(episode_tmp.iterdir())
        process_evidence = _ProcessLocalGroundingEvidence(
            raw=raw_evidence,
            journal_verifier=journal_verifier,
        )
        evaluation = _evaluate_grounding(process_evidence)
        evidence_sha256 = _canonical_sha256(raw_evidence)
        journal_terminal_sha256 = str(raw_evidence["journal_terminal_sha256"])
    else:
        evidence_sha256 = None
        journal_terminal_sha256 = None
    result = _public_result(
        task=task,
        seed=seed,
        evaluation=evaluation,
        artifact_root=run,
        wall_s=time.monotonic() - started,
        journal_terminal_sha256=journal_terminal_sha256,
        evidence_sha256=evidence_sha256,
        error_sha256=error_sha256,
    )
    from .joint_runner import _atomic_json

    _atomic_json(run / "panda-grounding-child.json", result)
    return result


def run_grounding_smoke(*, task: str, seed: int, run: Path) -> dict[str, object]:
    """Launch the gate in the same reduced, network-disabled smoke envelope."""

    from .joint_runner import _atomic_json
    from .joint_safety_smoke import _isolated_child_command

    started = time.monotonic()
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    child_run = run / "gate"
    log_path = run / "simulator.log"
    with log_path.open("w") as log_handle:
        os.chmod(log_path, 0o600)
        process = subprocess.Popen(
            _isolated_child_command(
                module="adaptive.panda_grounding_smoke",
                task=task,
                seed=seed,
                run=run,
                child_run=child_run,
                extra_args=("--isolated-child",),
            ),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = process.wait(timeout=1_300)
    target = child_run / "panda-grounding-child.json"
    if return_code != 0 or not target.is_file():
        error_sha256 = hashlib.sha256(
            f"child-exit:{return_code}".encode()
        ).hexdigest()
        result = _public_result(
            task=task,
            seed=seed,
            evaluation=_Evaluation(
                False,
                {name: False for name in GROUNDING_CHECKS},
                _empty_metrics(),
            ),
            artifact_root=run,
            wall_s=time.monotonic() - started,
            journal_terminal_sha256=None,
            evidence_sha256=None,
            error_sha256=error_sha256,
        )
    else:
        installed_release_digest = _installed_release_digest()
        result = validate_public_grounding_result(
            json.loads(target.read_text()),
            expected_task=task,
            expected_seed=seed,
            expected_release_digest=installed_release_digest,
            expected_artifact_root=child_run,
        )
        result["artifact_root"] = str(run)
        result["wall_s"] = time.monotonic() - started
        result = validate_public_grounding_result(
            result,
            expected_task=task,
            expected_seed=seed,
            expected_release_digest=installed_release_digest,
            expected_artifact_root=run,
        )
    _atomic_json(run / "panda-grounding-smoke.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="OpenToasterOvenDoor")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--isolated-child", action="store_true")
    args = parser.parse_args()
    if args.isolated_child:
        _run_isolated_child(
            task=args.task,
            seed=args.seed,
            run=args.run_dir.resolve(),
        )
        return 0
    result = run_grounding_smoke(
        task=args.task,
        seed=args.seed,
        run=args.run_dir.resolve(),
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
