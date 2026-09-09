from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import hmac
import importlib
import json
import math
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from adaptive.critic_protocol import (
    ADVISORY_ENUMS,
    ADVISORY_SCHEMA,
    CRITIC_CONFIG,
    CRITIC_CONFIG_SHA256,
    CRITIC_MAX_ATTEMPTS_PER_TASK,
    CRITIC_MAX_TOKENS,
    CRITIC_TIMEOUT_S,
    OPTIMIZATION_MAX_HOURS,
    CriticContext,
    command_signature,
    public_outcome_delta,
    validate_advisory,
)
from adaptive.joint_protocol import (
    JOINT_LIMITS,
    JOINT_NAMES,
    JOINT_RESPONSE_SCHEMA,
    MAX_JOINT_STEP,
    MIN_GRIPPER_ACTIONS,
    decode_joint_command,
    interpolate_joint_command,
    next_joint_waypoint,
)
from adaptive.joint_runner import (
    CONTROLLER_MAX_TOKENS,
    CONTROLLER_RESPONSE_SCHEMA,
    PROPOSAL_CONTROLLER_MAX_TOKENS,
    PUBLIC_BASE_RECEIPT_FIELDS,
    PUBLIC_JOINT_RECEIPT_FIELDS,
    finalize_joint_receipt,
    load_joint_system_prompt,
    prepare_joint_mailbox,
)
from adaptive.joint_safety_smoke import evaluate_safety_smoke
from adaptive.joint_sim_child import (
    _execute_base,
    _execute_move,
    _publish_observation,
    _select_arm_qpos,
    joint_controller_config,
    joint_unmap_action,
    validate_joint_action_sequence,
)
from adaptive.panda_embodiment import (
    FLANGE_TO_GRIP_SITE,
    Pose,
    compose_world_pose,
    panda_fk,
    panda_local_jacobian,
    project_world_point,
    read_public_telemetry,
    select_named_arm_vector,
    summarize_telemetry_samples,
)
from adaptive.remote_driver import (
    _budget_check,
    _controller_instruction,
    _direct_commands,
    _make_client_class,
    _smoke_pass,
)

PANDA_PROMPT_PARITY_CONTRACT = {
    "P01": (
        "fresh rgb",
        "one small deliberate command",
        "reobserve",
        "exact public field name",
        "state.arm_joint_velocity",
        "endpoint_error",
    ),
    "P02": (
        "qwen alone",
        "every numeric value",
        "cartesian_delta",
        "move_joints.targets",
        "damped least-squares",
    ),
    "P03": ("mobile base", "robot-base", "right-handed", "grip_site"),
    "P04": tuple(f"joint{index}" for index in range(1, 8))
    + (
        "local +z",
        "hard limits",
        "current measured",
        "grip-site origin translation jacobian column is zero",
        "off-axis finger geometry rotates",
    ),
    "P05": ("robot.xml", "panda_gripper.xml", "flange", "0.1065", "0.097"),
    "P06": (
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "base-relative",
        "predict",
    ),
    "P07": (
        "state.arm_translation_jacobian",
        "state.arm_rotation_jacobian",
        "3x7",
        "base-frame translation",
        "axis-angle rotation",
        "fixed damped",
    ),
    "P08": ("singular", "joint margin", "fixture", "retreat"),
    "P09": (
        "left external",
        "right external",
        "+u right",
        "+v down",
        "behind",
    ),
    "P10": ("before/after", "pixel", "reversible", "realized motion"),
    "P11": (
        "absolute",
        "0.025",
        "32-action",
        "undershoot",
        "state.arm_joint_velocity",
    ),
    "P12": (
        "requested_targets",
        "realized_arm_qpos_delta",
        "end_effector_pose_delta",
        "telemetry_summary",
        "mean_absolute_rgb_change",
    ),
    "P13": ("moving baseline", "gravity", "no single", "proves contact"),
    "P14": (
        "wrench change",
        "torque change",
        "tracking residual",
        "gripper residual",
        "fresh rgb",
    ),
    "P15": ("1.0 open", "0.0 closed", "state.gripper_qpos", "no grasp oracle"),
    "P16": ("visibility", "safe reversible probes", "return near", "coordinated"),
    "P17": ("disappearing", "stale", "stop blind", "relocalize"),
    "P18": ("pregrasp", "align", "close", "transport", "release"),
    "P19": ("milestone", "multi-view", "finish", "task progress"),
    "P20": ("closed qualitative enum", "accept, adapt, or reject", "qwen"),
    "P21": (
        "one command per turn",
        "no raw timestep",
        "no arm-joint velocity",
        "base normalized_velocity",
        "five base-motion steps",
        "stationary base-zero gripper-settling",
        "450",
        "70",
        "1200",
        "public",
    ),
    "P22": ("observation_id", "receipt", "fresh images", "episode-local"),
}


def _panda_prompt_sections(prompt: str) -> dict[str, str]:
    headings = list(re.finditer(r"(?m)^## (P\d{2}) — ([^\n]+)$", prompt))
    return {
        match.group(1): prompt[
            match.start() : headings[index + 1].start()
            if index + 1 < len(headings)
            else len(prompt)
        ].casefold()
        for index, match in enumerate(headings)
    }


def test_composed_joint_prompt_covers_every_inspect_to_panda_parity_section() -> None:
    root = Path(__file__).resolve().parents[1]
    prompt = load_joint_system_prompt(root)
    sections = _panda_prompt_sections(prompt)

    assert tuple(sections) == tuple(PANDA_PROMPT_PARITY_CONTRACT)
    for parity_id, required_phrases in PANDA_PROMPT_PARITY_CONTRACT.items():
        missing = [
            phrase for phrase in required_phrases if phrase.casefold() not in sections[parity_id]
        ]
        assert not missing, f"{parity_id} missing {missing}"


def test_joint_prompt_composition_and_exact_hash_are_deterministic() -> None:
    root = Path(__file__).resolve().parents[1]
    prompt = load_joint_system_prompt(root)
    expected = (
        (root / "prompts" / "joint_system.txt").read_bytes()
        + b"\n"
        + (root / "prompts" / "panda_embodiment.txt").read_bytes()
        + b"\n"
        + (root / "prompts" / "proposal_audit_compatibility.txt").read_bytes()
    )

    assert prompt.encode("utf-8") == expected
    assert load_joint_system_prompt(root) == prompt
    assert hashlib.sha256(load_joint_system_prompt(root).encode()).hexdigest() == (
        hashlib.sha256(expected).hexdigest()
    )
    # The locked proposal-audit rules intentionally extend the controller part;
    # P01-P22 retention is checked semantically and section-by-section below.
    assert len(expected) > 21_489
    assert hashlib.sha256(expected).hexdigest() == (
            "ef23b7cb4035bf305ebb7e632b3eb9237cffc5dbfc37471a6e50aff1e6a5e19d"
    )
    assert 12_000 <= len(expected) <= 33_000
    # The controller prompt now advertises nine revisions; the critic prompt
    # and its authority remain unchanged.
    assert hashlib.sha256(load_joint_system_prompt(root, variant="rig").encode()).hexdigest() == (
        "88a7f00bc80e35423b43e58ecbb7b7a933fc08cabd6475287c8d9791550c42f5"
    )
    assert hashlib.sha256((root / "prompts" / "proposal_audit_critic.txt").read_bytes()).hexdigest() == (
        "47c5940f427054c7e4af68d2df4227b0b9886c2d21f6b8ea56828856393cd032"
    )


def test_joint_prompt_composition_is_protocol_specific() -> None:
    root = Path(__file__).resolve().parents[1]
    legacy = load_joint_system_prompt(root, protocol="legacy")
    proposal = load_joint_system_prompt(root, protocol="proposal")
    expected_legacy = (
        (root / "prompts" / "joint_system_legacy.txt").read_bytes()
        + b"\n"
        + (root / "prompts" / "panda_embodiment.txt").read_bytes()
    ).decode()

    assert legacy == expected_legacy
    assert "Exp-005 proposal-audit compatibility" not in legacy
    assert "Every response is a pre-execution draft" not in legacy
    assert "Every response is a pre-execution draft" in proposal
    assert "Exp-005 proposal-audit compatibility" in proposal
    with pytest.raises(ValueError, match="protocol"):
        load_joint_system_prompt(root, protocol="implicit")


def test_joint_prompt_requires_public_grounding_names_in_physical_command_notes() -> None:
    root = Path(__file__).resolve().parents[1]
    prompt = load_joint_system_prompt(root).casefold()

    assert "every physical-command `note`" in prompt
    assert "exact public field name" in prompt
    assert "`state.arm_joint_velocity`" in prompt
    assert "`endpoint_error`" in prompt
    assert "do not replace those exact names with shorthand" in prompt


@pytest.mark.parametrize("invalid_bytes", [b"", b"bad\r\n", b"bad\x00\n", b"\xff\n"])
def test_joint_prompt_loader_rejects_unvalidated_text_parts(
    tmp_path: Path, invalid_bytes: bytes
) -> None:
    prompt_root = tmp_path / "prompts"
    prompt_root.mkdir()
    (prompt_root / "joint_system.txt").write_bytes(b"base\n")
    (prompt_root / "panda_embodiment.txt").write_bytes(invalid_bytes)

    with pytest.raises((UnicodeDecodeError, ValueError)):
        load_joint_system_prompt(tmp_path)


def test_joint_prompt_excludes_stale_foreign_private_and_false_contact_guidance() -> None:
    root = Path(__file__).resolve().parents[1]
    prompt = load_joint_system_prompt(root).casefold()
    excluded = {
        "typical reset",
        "-1.0249",
        "yam",
        "left_j0",
        "right_j0",
        "l1 =",
        "a = π",
        "sim.data.contact",
        "check_contact",
        "object_pose",
        "gate-only exact",
        "torque alone proves contact",
        "wrench alone proves contact",
        "tracking error proves contact",
        "force means contact",
    }

    present = {term for term in excluded if term in prompt}
    assert not present, sorted(present)


def test_qwen_keeps_all_numeric_joint_or_cartesian_request_authority() -> None:
    root = Path(__file__).resolve().parents[1]
    controller_prompt = " ".join(load_joint_system_prompt(root).casefold().split())
    critic_prompt = (root / "prompts" / "advisory_critic.txt").read_text().casefold()

    assert "move_joints.targets" in controller_prompt
    assert "partial map of absolute" in controller_prompt
    assert "cartesian_delta" in controller_prompt
    assert "qwen alone chooses and emits every numeric value" in controller_prompt
    assert "omitted joints hold their fresh measured values" in controller_prompt
    assert "fixed damped least-squares mapping" in controller_prompt
    assert "never edits qwen's six values" in controller_prompt
    assert "closed qualitative enum" in critic_prompt
    assert "no motor authority" in critic_prompt
    assert "qwen alone" in critic_prompt
    assert not re.search(r"(?<![a-z])[-+]?\d+(?:\.\d+)?", critic_prompt)


def test_critic_prompt_names_public_receipt_evidence_without_command_authority() -> None:
    prompt = (
        Path(__file__).resolve().parents[1] / "prompts" / "advisory_critic.txt"
    ).read_text().casefold()
    for field in (
        "diagnosis",
        "affected_region",
        "evidence",
        "suggested_correction",
        "confidence",
        "state.arm_joint_velocity",
        "telemetry_summary",
        "tracking_pause_count",
        "gripper_residual",
        "mean_absolute_rgb_change",
    ):
        assert field in prompt
    assert "qualitative labels" in prompt
    assert "must not emit" in prompt
    for labels in ADVISORY_ENUMS.values():
        for label in labels:
            assert f"`{label}`" in prompt
    assert "`overall`" not in prompt


def test_panda_fk_matches_literal_robosuite_grip_site_reference() -> None:
    pose = panda_fk(
        [
            0.0,
            0.0,
            0.0,
            -math.pi / 2.0,
            0.0,
            math.pi / 2.0,
            math.pi / 4.0,
        ]
    )

    assert isinstance(pose, Pose)
    assert pose.position_m == pytest.approx((0.5545, 0.0, 0.528), abs=1e-12)
    expected_rotation = (
        (-0.000492624356161, 0.999999878660614, 0.0),
        (0.999999878660613, 0.000492624356161, 0.0),
        (0.0, 0.0, -1.0),
    )
    for actual_row, expected_row in zip(
        pose.rotation_matrix, expected_rotation, strict=True
    ):
        assert actual_row == pytest.approx(expected_row, abs=1e-12)
    assert len(pose.position_m) == 3
    assert len(pose.rotation_matrix) == 3
    assert all(
        math.isfinite(value)
        for value in (*pose.position_m, *sum(pose.rotation_matrix, ()))
    )


def test_panda_fk_pins_explicit_flange_to_grip_site_transform() -> None:
    assert FLANGE_TO_GRIP_SITE.position_m == pytest.approx(
        (0.0, 0.0, 0.2035), abs=1e-12
    )
    expected_rotation = (
        (-0.707455033409464, 0.706758357363826, 0.0),
        (-0.706758357363826, -0.707455033409464, 0.0),
        (0.0, 0.0, 1.0),
    )
    for actual_row, expected_row in zip(
        FLANGE_TO_GRIP_SITE.rotation_matrix, expected_rotation, strict=True
    ):
        assert actual_row == pytest.approx(expected_row, abs=1e-12)


@pytest.mark.parametrize(
    "qpos",
    (
        [],
        [0.0] * 6,
        [0.0] * 8,
        [0.0, 0.0, 0.0, float("nan"), 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, float("inf"), 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, True, 0.0, 0.0, 0.0],
    ),
)
def test_panda_fk_rejects_invalid_joint_vectors(qpos: list[float]) -> None:
    with pytest.raises((TypeError, ValueError)):
        panda_fk(qpos)


@pytest.mark.parametrize(
    "qpos",
    (
        [0.2, -0.4, 0.3, -1.8, 0.5, 1.2, -0.6],
        [-0.6, 0.35, -0.2, -2.1, -0.45, 2.0, 0.25],
        [0.45, -0.7, 0.8, -1.25, 0.2, 2.4, -0.9],
    ),
)
def test_panda_local_jacobian_matches_fk_translation_differences(
    qpos: list[float],
) -> None:
    epsilon = 1e-5
    jacobian = panda_local_jacobian(qpos, epsilon=epsilon)

    assert set(jacobian) == {"translation", "rotation"}
    for name in ("translation", "rotation"):
        assert len(jacobian[name]) == 3
        assert all(len(row) == 7 for row in jacobian[name])
        assert all(math.isfinite(value) for row in jacobian[name] for value in row)
    for joint_index in range(7):
        plus = list(qpos)
        minus = list(qpos)
        plus[joint_index] += epsilon
        minus[joint_index] -= epsilon
        plus_pose = panda_fk(plus)
        minus_pose = panda_fk(minus)
        expected_column = [
            (plus_pose.position_m[axis] - minus_pose.position_m[axis]) / (2.0 * epsilon)
            for axis in range(3)
        ]
        actual_column = [
            jacobian["translation"][axis][joint_index] for axis in range(3)
        ]
        assert actual_column == pytest.approx(expected_column, abs=1e-5)


def test_panda_local_jacobian_rotation_log_uses_base_frame_positive_sign() -> None:
    jacobian = panda_local_jacobian(
        [0.0, 0.0, 0.0, -math.pi / 2.0, 0.0, math.pi / 2.0, math.pi / 4.0]
    )

    first_joint_axis = [jacobian["rotation"][axis][0] for axis in range(3)]
    second_joint_axis = [jacobian["rotation"][axis][1] for axis in range(3)]
    assert first_joint_axis == pytest.approx((0.0, 0.0, 1.0), abs=1e-8)
    assert second_joint_axis == pytest.approx((0.0, 1.0, 0.0), abs=1e-8)


@pytest.mark.parametrize(
    "epsilon",
    (0.0, 1e-8, 0.0100001, float("nan"), float("inf"), True),
)
def test_panda_local_jacobian_rejects_invalid_epsilon(epsilon: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        panda_local_jacobian([0.0] * 7, epsilon=epsilon)


def _identity_camera_calibration() -> dict[str, object]:
    return {
        "camera_name": "left",
        "mujoco_camera_name": "robot0_agentview_left",
        "image_width_px": 640,
        "image_height_px": 480,
        "fx_px": 400.0,
        "fy_px": 500.0,
        "cx_px": 319.5,
        "cy_px": 239.5,
        "camera_position_world_m": [0.0, 0.0, 0.0],
        "camera_xmat_world": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "projection": "mujoco-camera-x-right-y-up-minus-z-forward",
    }


def test_project_world_point_uses_mujoco_camera_axes_and_pinhole_intrinsics() -> None:
    projected = project_world_point([0.2, 0.1, -2.0], _identity_camera_calibration())

    assert set(projected) == {"u_px", "v_px", "depth_m", "visible"}
    assert projected["u_px"] == pytest.approx(359.5)
    assert projected["v_px"] == pytest.approx(214.5)
    assert projected["depth_m"] == pytest.approx(2.0)
    assert projected["visible"] is True


def test_project_world_point_marks_off_image_points_without_clamping() -> None:
    projected = project_world_point([2.0, 0.0, -1.0], _identity_camera_calibration())

    assert projected["u_px"] == pytest.approx(1119.5)
    assert projected["v_px"] == pytest.approx(239.5)
    assert projected["depth_m"] == pytest.approx(1.0)
    assert projected["visible"] is False


def test_project_world_point_rejects_points_behind_camera() -> None:
    with pytest.raises(ValueError, match="front"):
        project_world_point([0.0, 0.0, 1.0], _identity_camera_calibration())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("projection", "z-forward"),
        ("image_width_px", 0),
        ("image_height_px", True),
        ("fx_px", 0.0),
        ("fy_px", float("inf")),
        ("cx_px", float("nan")),
        ("camera_position_world_m", [0.0, 0.0]),
        (
            "camera_xmat_world",
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        ),
    ),
)
def test_project_world_point_rejects_malformed_calibration(
    field: str, value: object
) -> None:
    calibration = _identity_camera_calibration()
    calibration[field] = value
    with pytest.raises((TypeError, ValueError)):
        project_world_point([0.0, 0.0, -1.0], calibration)


def test_project_world_point_requires_all_official_calibration_fields() -> None:
    calibration = _identity_camera_calibration()
    del calibration["camera_xmat_world"]
    with pytest.raises(ValueError, match="calibration"):
        project_world_point([0.0, 0.0, -1.0], calibration)


def test_compose_world_pose_handles_identity_public_base_pose() -> None:
    pose = compose_world_pose(
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.1, 0.2, 0.3],
        [0.0, 0.0, 0.0, 1.0],
    )

    assert pose.position_m == pytest.approx((1.1, 2.2, 3.3), abs=1e-12)
    expected_rotation = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    for actual_row, expected_row in zip(
        pose.rotation_matrix, expected_rotation, strict=True
    ):
        assert actual_row == pytest.approx(expected_row, abs=1e-12)


def test_compose_world_pose_rotates_relative_eef_by_public_xyzw_base_quaternion() -> (
    None
):
    half_sqrt_two = math.sqrt(0.5)
    pose = compose_world_pose(
        [0.5, -0.25, 0.75],
        [0.0, 0.0, half_sqrt_two, half_sqrt_two],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    )

    assert pose.position_m == pytest.approx((0.5, 0.75, 0.75), abs=1e-12)
    expected_rotation = (
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    for actual_row, expected_row in zip(
        pose.rotation_matrix, expected_rotation, strict=True
    ):
        assert actual_row == pytest.approx(expected_row, abs=1e-12)


@dataclasses.dataclass(frozen=True)
class _FakeResponse:
    command: dict[str, object]
    evidence: dict[str, object]


class _FakeMalformedResponse(RuntimeError):
    pass


class _FakeHTTP:
    timeout: object = None


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


class _OwnerAttemptLog:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def append(self, record: dict[str, object]) -> None:
        assert set(record) == _ATTEMPT_EVIDENCE_FIELDS
        self.records.append(dict(record))


def _fake_response_schema_sha256(value: object) -> str | None:
    return None if value is None else _json_sha256(value)


def _fake_actual_request_sha256(
    kwargs: dict[str, object], *, served_model_id: str
) -> str:
    images = kwargs["images"]
    assert isinstance(images, Mapping)
    image_sha256 = {
        label: hashlib.sha256(content).hexdigest()
        for label, content in images.items()
        if isinstance(label, str) and isinstance(content, bytes)
    }
    actual_payload = {
        "model": served_model_id,
        "messages": [
            {"role": "system", "content": kwargs["system_prompt"]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "observation_id": kwargs["observation_id"],
                                "instruction": kwargs["instruction"],
                                "public_state": kwargs["public_state"],
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    },
                    {"type": "image_sha256", "image_sha256": image_sha256},
                ],
            },
        ],
        "seed": 3074294,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": kwargs.get("max_tokens", 1024),
        "chat_template_kwargs": {"enable_thinking": False},
        "response_schema_sha256": _fake_response_schema_sha256(
            kwargs.get("response_schema")
        ),
    }
    return _json_sha256(actual_payload)


def _attempt_record(
    request_sha256: str,
    *,
    observation_id: str,
    attempt_index: int = 0,
    response_schema_sha256: str | None = None,
    served_model_id: str = "qwen-fake-served-model",
) -> dict[str, object]:
    return {
        "request_sha256": request_sha256,
        "observation_id": observation_id,
        "attempt_index": attempt_index,
        "raw_body_sha256": "b" * 64,
        "sanitized_raw_command": '{"kind":"fake"}',
        "http_status": 200,
        "finish_reason": "stop",
        "completion_tokens": 7,
        "latency_s": 0.01,
        "response_schema_sha256": response_schema_sha256,
        "served_model_id": served_model_id,
    }


def _default_attempt_record(
    role: str,
    kwargs: dict[str, object],
    *,
    sanitized_raw_command: str,
    served_model_id: str = "qwen-fake-served-model",
) -> dict[str, object]:
    return _attempt_record(
        _fake_actual_request_sha256(kwargs, served_model_id=served_model_id),
        observation_id=str(kwargs["observation_id"]),
        attempt_index=int(kwargs.get("attempt_index", 0)),
        response_schema_sha256=_fake_response_schema_sha256(
            kwargs.get("response_schema")
        ),
        served_model_id=served_model_id,
    ) | {"sanitized_raw_command": sanitized_raw_command}


class _FakeClient:
    instances: list["_FakeClient"] = []
    calls: list[tuple[str, dict[str, object]]] = []
    controller_outputs: list[dict[str, object]] = []
    critic_outputs: list[dict[str, object]] = []
    controller_evidence: list[dict[str, object]] = []
    critic_evidence: list[dict[str, object]] = []
    controller_attempt_records: list[list[dict[str, object]] | None] = []
    critic_attempt_records: list[list[dict[str, object]] | None] = []
    critic_error: Exception | None = None
    controller_errors_after_log: list[Exception | None] = []
    served_model_id = "qwen-fake-served-model"

    def __init__(self, **_kwargs: object) -> None:
        self.role = "controller" if not type(self).instances else "critic"
        self._http = _FakeHTTP()
        self.verified = False
        self.closed = False
        type(self).instances.append(self)

    def verify(self) -> None:
        self.verified = True

    def close(self) -> None:
        self.closed = True

    def complete(self, **kwargs: object) -> _FakeResponse:
        type(self).calls.append((self.role, kwargs))
        if self.role == "critic" and type(self).critic_error is not None:
            raise type(self).critic_error
        pending_error: Exception | None = None
        if self.role == "critic":
            command = type(self).critic_outputs.pop(0)
            evidence = {"role": self.role}
            if type(self).critic_evidence:
                evidence.update(type(self).critic_evidence.pop(0))
        else:
            if type(self).controller_errors_after_log:
                pending_error = type(self).controller_errors_after_log.pop(0)
            command = (
                type(self).controller_outputs.pop(0)
                if pending_error is None
                else None
            )
            evidence = {"role": self.role}
            if type(self).controller_evidence:
                evidence.update(type(self).controller_evidence.pop(0))
        attempt_log = kwargs.get("attempt_log")
        record_queue = (
            type(self).critic_attempt_records
            if self.role == "critic"
            else type(self).controller_attempt_records
        )
        configured_records = record_queue.pop(0) if record_queue else None
        records = (
            [_default_attempt_record(
                self.role,
                kwargs,
                sanitized_raw_command=(
                    json.dumps(command, sort_keys=True, separators=(",", ":"))
                    if command is not None
                    else str(pending_error)
                ),
                served_model_id=type(self).served_model_id,
            )]
            if configured_records is None
            else configured_records
        )
        if attempt_log is not None:
            for record in records:
                attempt_log.append(record)  # type: ignore[attr-defined]
        if pending_error is not None:
            raise pending_error
        assert command is not None
        return _FakeResponse(command=command, evidence=evidence)


@pytest.fixture(autouse=True)
def _reset_fake() -> None:
    _FakeClient.instances = []
    _FakeClient.calls = []
    _FakeClient.controller_outputs = []
    _FakeClient.critic_outputs = []
    _FakeClient.controller_evidence = []
    _FakeClient.critic_evidence = []
    _FakeClient.controller_attempt_records = []
    _FakeClient.critic_attempt_records = []
    _FakeClient.critic_error = None
    _FakeClient.controller_errors_after_log = []
    _FakeClient.served_model_id = "qwen-fake-served-model"


def _state(
    x: float = 0.0,
    *,
    qvel: float = 0.0,
    torque_available: bool = True,
) -> dict[str, object]:
    torque = [2.0] * 7 if torque_available else None
    return {
        "state.base_position": [0.0, 0.0, 0.0],
        "state.base_rotation": [0.0, 0.0, 0.0, 1.0],
        "state.end_effector_position_relative": [x, 0.0, 0.5],
        "state.end_effector_rotation_relative": [0.0, 0.0, 0.0, 1.0],
        "state.gripper_qpos": [0.04, -0.04],
        "state.arm_joint_position": [x, -1.0, 0.0, -2.2, 0.0, 1.5, 0.7],
        "state.arm_joint_velocity": [qvel] * 7,
        "state.arm_applied_torque": {
            "available": torque_available,
            "values_nm": torque,
        },
        "state.end_effector_wrench": {
            "force_n": [1.0, 0.0, 0.0],
            "torque_nm": [0.0, 1.0, 0.0],
        },
        "state.arm_translation_jacobian": [[0.0] * 7 for _ in range(3)],
        "state.arm_rotation_jacobian": [[0.0] * 7 for _ in range(3)],
        "state.end_effector_external_pixels": {
            "left": {
                "u_px": 100.0 + x,
                "v_px": 200.0,
                "visible": True,
                "depth_valid": True,
            },
            "right": {
                "u_px": 300.0 + x,
                "v_px": 400.0,
                "visible": True,
                "depth_valid": True,
            },
        },
    }


def _images() -> dict[str, bytes]:
    return {"left": b"left", "right": b"right", "wrist": b"wrist"}


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _json_copy(value: object) -> object:
    return json.loads(json.dumps(value))


def _call_manifest_sha256(call: dict[str, object]) -> str:
    manifest = dict(call)
    manifest["images"] = {
        label: hashlib.sha256(content).hexdigest()
        for label, content in call["images"].items()
    }
    manifest.pop("attempt_log", None)
    return _json_sha256(manifest)


def _instruction(
    receipts: list[dict[str, object]] | None = None,
    *,
    rgb_change: dict[str, float] | None = None,
) -> str:
    return json.dumps({
        "task": "open the toaster",
        "recent_receipts": receipts or [],
        "mean_absolute_rgb_change_since_previous": rgb_change
        or {"left": 1.0, "right": 1.0, "wrist": 1.0},
    }, sort_keys=True, separators=(",", ":"))


def _action(observation_id: str, x: float = 0.01) -> dict[str, object]:
    return {
        "kind": "action",
        "observation_id": observation_id,
        "note": "controller direct action",
        "translation_m": [x, 0.0, 0.0],
        "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        "gripper": "open",
    }


def _controller_command(
    observation_id: str,
    x: float = 0.01,
    *,
    kind: str = "move_joints",
) -> dict[str, object]:
    if kind == "base_action":
        return {
            "kind": "base_action",
            "observation_id": observation_id,
            "axis": "x",
            "normalized_velocity": 0.1,
            "gripper": "hold",
            "note": "Qwen selected a bounded base pulse",
        }
    return {
        "kind": "move_joints",
        "observation_id": observation_id,
        "targets": {"joint1": x},
        "note": "Qwen selected an absolute joint target",
    }


def _sealed_critic_receipt(
    command: dict[str, object],
    current_state: dict[str, object],
    *,
    start_torque_available: bool = True,
) -> dict[str, object]:
    end_qpos = list(current_state["state.arm_joint_position"])
    start_qpos = list(end_qpos)
    targets = command["targets"] if command["kind"] == "move_joints" else {}
    arm_target_names = {
        name for name in JOINT_NAMES if isinstance(targets, Mapping) and name in targets
    }
    for index, name in enumerate(JOINT_NAMES):
        if name in arm_target_names:
            start_qpos[index] = end_qpos[index] - 0.02
        else:
            start_qpos[index] = end_qpos[index]
    start = _telemetry_sample(
        qpos=start_qpos,
        qvel=[0.0] * 7,
        torque=[1.0] * 7 if start_torque_available else None,
        force=[0.0, 0.0, 0.0],
        wrench_torque=[0.0, 0.0, 0.0],
    )
    end = _telemetry_sample(
        qpos=end_qpos,
        qvel=list(current_state["state.arm_joint_velocity"]),
        torque=list(current_state["state.arm_applied_torque"]["values_nm"]),
        force=list(current_state["state.end_effector_wrench"]["force_n"]),
        wrench_torque=list(
            current_state["state.end_effector_wrench"]["torque_nm"]
        ),
    )
    shared = {
        "observation_id": command["observation_id"],
        "accepted": True,
        "note": command["note"],
        "gripper_intent": 1.0,
        "step_count": 5,
        "realized_arm_qpos": end_qpos,
        "tracking_pause_count": 1,
        "realized_arm_qpos_delta": [
            end - start for start, end in zip(start_qpos, end_qpos, strict=True)
        ],
        "end_effector_pose_delta": {
            "translation_m": [0.01, 0.0, 0.0],
            "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        },
        "end_effector_external_pixel_displacement": {
            "left": {
                "start_px": [100.0, 200.0],
                "end_px": [100.0 + end_qpos[0], 200.0],
                "delta_px": [end_qpos[0], 0.0],
                "distance_px": abs(end_qpos[0]),
                "start_visible": True,
                "end_visible": True,
                "start_depth_valid": True,
                "end_depth_valid": True,
            },
            "right": {
                "start_px": [300.0, 400.0],
                "end_px": [300.0 + end_qpos[0], 400.0],
                "delta_px": [end_qpos[0], 0.0],
                "distance_px": abs(end_qpos[0]),
                "start_visible": True,
                "end_visible": True,
                "start_depth_valid": True,
                "end_depth_valid": True,
            },
        },
        "telemetry_summary": summarize_telemetry_samples(start, [end], end),
        "gripper_residual": {
            "start_qpos": [0.04, -0.04],
            "end_qpos": [0.04, -0.04],
            "qpos_delta": [0.0, 0.0],
            "measured_end_finger_separation": 0.08,
        },
        "mean_absolute_rgb_change": {
            "left": 1.0,
            "right": 1.0,
            "wrist": 1.0,
        },
    }
    if command["kind"] == "base_action":
        return {
            "kind": "base_action",
            **shared,
            "axis": command["axis"],
            "normalized_velocity": command["normalized_velocity"],
            "tracking_pause_count": 0,
            "base_motion_step_count": 5,
            "remaining_endpoint_error": max(
                abs(start - end) for start, end in zip(start_qpos, end_qpos, strict=True)
            ),
        }
    bounded_endpoint = [
        float(targets[name]) if name in arm_target_names else start_qpos[index]
        for index, name in enumerate(JOINT_NAMES)
    ]
    return {
        "kind": "move_joints",
        **shared,
        "requested_targets": dict(command["targets"]),
        "resolved_held_dimensions": {
            name: start_qpos[index]
            for index, name in enumerate(JOINT_NAMES)
            if name not in arm_target_names
        },
        "bounded_endpoint": bounded_endpoint,
        "maximum_commanded_step": max(
            abs(target - start)
            for target, start in zip(bounded_endpoint, start_qpos, strict=True)
        ),
        "endpoint_error": max(
            abs(target - realized)
            for target, realized in zip(bounded_endpoint, end_qpos, strict=True)
        ),
        "minimum_hard_limit_margin": 0.5,
        "failure_status": None,
    }


def _advisory() -> dict[str, object]:
    return {
        "diagnosis": "visual_misalignment",
        "affected_region": "camera",
        "evidence": ["external_rgb", "action_receipt"],
        "suggested_correction": "relocalize",
        "confidence": "high",
    }


def _context(max_decisions: int = 12) -> CriticContext:
    return CriticContext(
        task="OpenToasterOvenDoor",
        family="articulated",
        run=Path("/nonexistent"),
        critic_prompt="ADVISORY ONLY",
        max_decisions=max_decisions,
    )


def _complete(
    wrapped: object,
    observation_id: str,
    state: dict[str, object],
    images: dict[str, bytes],
    *,
    receipts: list[dict[str, object]] | None = None,
    rgb_change: dict[str, float] | None = None,
    attempt_log: _OwnerAttemptLog | None = None,
) -> _FakeResponse:
    kwargs: dict[str, object] = {
        "observation_id": observation_id,
        "system_prompt": "OFFICIAL SYSTEM PROMPT",
        "instruction": _instruction(receipts, rgb_change=rgb_change),
        "public_state": state,
        "images": images,
        "response_schema": CONTROLLER_RESPONSE_SCHEMA,
        "max_tokens": 768,
    }
    if attempt_log is not None:
        kwargs["attempt_log"] = attempt_log
    return wrapped.complete(**kwargs)


def test_advisory_schema_is_enum_only_bounded_and_non_executable() -> None:
    assert set(ADVISORY_SCHEMA["properties"]) == {
        "diagnosis",
        "affected_region",
        "evidence",
        "suggested_correction",
        "confidence",
    }
    assert set(ADVISORY_ENUMS) == set(ADVISORY_SCHEMA["properties"])
    for name in ("diagnosis", "affected_region", "suggested_correction", "confidence"):
        assert ADVISORY_SCHEMA["properties"][name] == {
            "enum": list(ADVISORY_ENUMS[name])
        }
    assert ADVISORY_SCHEMA["properties"]["evidence"] == {
        "type": "array",
        "items": {"enum": list(ADVISORY_ENUMS["evidence"])},
        "maxItems": 3,
    }
    # Installed vLLM 0.27.1 rejects uniqueItems before generation. The prompt
    # requires uniqueness and validate_advisory still rejects duplicates.
    assert "uniqueItems" not in json.dumps(ADVISORY_SCHEMA)
    assert "kind" not in ADVISORY_SCHEMA["properties"]
    assert "translation_m" not in json.dumps(ADVISORY_SCHEMA)
    assert CRITIC_CONFIG["failed_attempts_consume_budget"] is True
    assert len(CRITIC_CONFIG_SHA256) == 64
    assert validate_advisory(_advisory()) == _advisory()
    assert "targets" not in json.dumps(ADVISORY_SCHEMA).lower()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"diagnosis": "The gripper moved beside the handle."}, "diagnosis"),
        (
            {"evidence": "Both external views show the handle still offset."},
            "evidence",
        ),
        (
            {
                "suggested_correction": (
                    "Reacquire the handle and approach from the right."
                )
            },
            "suggested_correction",
        ),
        ({"suggested_correction": "retreat two radians"}, "suggested_correction"),
        ({"affected_region": "joint1"}, "affected_region"),
        ({"evidence": ["external_rgb", "external_rgb"]}, "evidence"),
        (
            {
                "evidence": [
                    "external_rgb",
                    "joint_tracking",
                    "action_receipt",
                    "gripper_state",
                ]
            },
            "evidence",
        ),
        ({"evidence": [{"targets": {"joint1": 0.25}}]}, "evidence"),
        ({"suggested_correction": {"translation_m": [0.1, 0.0, 0.0]}}, "suggested_correction"),
        ({"targets": {"joint1": 0.25}}, "fields"),
        ({"command": "move_joints"}, "fields"),
    ),
)
def test_advisory_rejects_free_text_magnitudes_maps_and_commands(
    mutation: dict[str, object], message: str
) -> None:
    invalid = _advisory()
    invalid.update(mutation)
    with pytest.raises(ValueError, match=message):
        validate_advisory(invalid)


def test_every_valid_advisory_enum_renders_as_labels_without_motor_authority() -> None:
    advisories: list[dict[str, object]] = []
    baseline = _advisory()
    for field, labels in ADVISORY_ENUMS.items():
        for label in labels:
            advisory = dict(baseline)
            advisory[field] = [label] if field == "evidence" else label
            advisories.append(advisory)

    for advisory in advisories:
        rendered = _controller_instruction(
            "CONTROLLER_INPUT",
            advisory_id="opaque-123-target-0.25",
            advisory=advisory,
        )
        prefix, payload = rendered.split("\n\nPOST_ACTION_ADVISORY:\n", maxsplit=1)
        assert prefix == "CONTROLLER_INPUT"
        assert json.loads(payload) == advisory
        assert set(json.loads(payload)) == set(ADVISORY_ENUMS)
        assert "opaque" not in payload
        assert not re.search(r"\d", payload)
        assert not re.search(
            r"(?<![a-z])(zero|one|two|three|four|five|six|seven|eight|nine)(?![a-z])",
            payload,
        )
        assert not re.search(
            r"radian|degree|metre|meter|newton|translation_m|"
            r"rotation_axis_angle_rad|normalized_velocity|move_joints|base_action|"
            r"joint[1-7]|targets",
            payload,
        )


def test_first_observation_is_controller_only_then_critic_precedes_controller() -> None:
    context = _context()
    wrapped = _make_client_class(_FakeClient, context)()
    wrapped.verify()
    assert all(instance.verified for instance in _FakeClient.instances)
    assert str(_FakeClient.instances[1]._http.timeout) == (
        f"Timeout(timeout={CRITIC_TIMEOUT_S})"
    )

    first = _controller_command("obs-0", 0.01)
    second = _controller_command("obs-1", -0.01)
    retry = _controller_command("obs-1", -0.02)
    _FakeClient.controller_outputs = [first, second, retry]
    _FakeClient.critic_outputs = [_advisory()]
    actual_critic_request_sha256 = "c" * 64
    actual_controller_request_sha256 = "d" * 64
    _FakeClient.critic_evidence = [{"model": "qwen-fake-critic-varying-payload"}]
    _FakeClient.critic_attempt_records = [[
        _attempt_record(
            actual_critic_request_sha256,
            observation_id="obs-1",
            response_schema_sha256=_json_sha256(ADVISORY_SCHEMA),
        )
    ]]
    _FakeClient.controller_evidence = [
        {},
        {"model": "qwen-fake-controller-varying-payload"},
        {},
    ]
    _FakeClient.controller_attempt_records = [
        [
            _attempt_record(
                "a" * 64,
                observation_id="obs-0",
                response_schema_sha256=_json_sha256(CONTROLLER_RESPONSE_SCHEMA),
            )
        ],
        [
            _attempt_record(
                actual_controller_request_sha256,
                observation_id="obs-1",
                response_schema_sha256=_json_sha256(CONTROLLER_RESPONSE_SCHEMA),
            )
        ],
        [
            _attempt_record(
                "e" * 64,
                observation_id="obs-1",
                response_schema_sha256=_json_sha256(CONTROLLER_RESPONSE_SCHEMA),
            )
        ],
    ]
    owner_attempt_log = _OwnerAttemptLog()
    images0, images1 = _images(), _images()
    state0, state1 = _state(0.0), _state(0.01)
    receipt = _sealed_critic_receipt(first, state1)

    response0 = _complete(wrapped, "obs-0", state0, images0, attempt_log=owner_attempt_log)
    assert response0.command is first
    assert [role for role, _ in _FakeClient.calls] == ["controller"]
    assert context.critic_attempts == 0

    response1 = _complete(
        wrapped,
        "obs-1",
        state1,
        images1,
        receipts=[receipt],
        attempt_log=owner_attempt_log,
    )
    assert response1.command is second
    assert [role for role, _ in _FakeClient.calls] == [
        "controller", "critic", "controller",
    ]
    critic_call = _FakeClient.calls[1][1]
    controller_call = _FakeClient.calls[2][1]
    assert critic_call["public_state"] is state1
    assert controller_call["public_state"] is state1
    assert critic_call["images"] is images1
    assert controller_call["images"] is images1
    assert critic_call["system_prompt"] == "ADVISORY ONLY"
    assert controller_call["system_prompt"] == "OFFICIAL SYSTEM PROMPT"
    assert "POST_ACTION_ADVISORY" in str(controller_call["instruction"])
    assert response1.evidence["consumed_advisory_id"]
    assert context.trigger_evaluations[1]["fired_trigger"] == "first_action"
    critic_payload = json.loads(str(critic_call["instruction"]))
    assert set(critic_payload) == {
        "schema",
        "task",
        "family",
        "trigger",
        "previous_executed_command",
        "previous_executed_command_sha256",
        "sealed_public_receipt",
        "sealed_public_receipt_sha256",
        "fresh_observation_id",
        "fresh_public_state",
        "fresh_public_state_sha256",
        "fresh_public_rgb_sha256",
        "fresh_outcome_delta",
        "constraints",
    }
    assert critic_payload["previous_executed_command"] == first
    assert critic_payload["sealed_public_receipt"] == receipt
    assert critic_payload["fresh_public_state"] == state1
    assert critic_payload["fresh_public_rgb_sha256"] == {
        "left": hashlib.sha256(b"left").hexdigest(),
        "right": hashlib.sha256(b"right").hexdigest(),
        "wrist": hashlib.sha256(b"wrist").hexdigest(),
    }
    encoded_critic = json.dumps(critic_payload, sort_keys=True)
    for forbidden in (
        "object_pose",
        '"success"',
        '"reward"',
        '"contact"',
        '"depth"',
        '"segmentation"',
        "gate_only",
    ):
        assert forbidden not in encoded_critic

    critic_call_manifest_sha256 = _call_manifest_sha256(critic_call)
    controller_call_manifest_sha256 = _call_manifest_sha256(controller_call)
    assert actual_critic_request_sha256 != critic_call_manifest_sha256
    assert actual_controller_request_sha256 != controller_call_manifest_sha256
    receipt_sha256 = _json_sha256(receipt)
    advisory_sha256 = _json_sha256(_advisory())
    command_sha256 = _json_sha256(second)
    critic_record = context.critic_records[0]
    controller_record = context.controller_records[1]
    assert critic_record["sealed_public_receipt_sha256"] == receipt_sha256
    assert critic_record["critic_request_sha256"] == actual_critic_request_sha256
    assert critic_record["critic_call_manifest_sha256"] == critic_call_manifest_sha256
    assert "request_sha256" not in critic_record["model_evidence"]
    assert critic_record["advisory_sha256"] == advisory_sha256
    assert critic_record["consumed_by_controller_request_sha256"] == (
        actual_controller_request_sha256
    )
    assert critic_record["qwen_command_sha256"] == command_sha256
    assert controller_record["critic_request_sha256"] == actual_critic_request_sha256
    assert controller_record["advisory_sha256"] == advisory_sha256
    assert controller_record["controller_request_sha256"] == (
        actual_controller_request_sha256
    )
    assert (
        controller_record["controller_call_manifest_sha256"]
        == controller_call_manifest_sha256
    )
    assert controller_record["command_sha256"] == command_sha256
    assert response1.evidence["critic_request_sha256"] == actual_critic_request_sha256
    assert (
        response1.evidence["controller_request_sha256"]
        == actual_controller_request_sha256
    )
    assert [record["request_sha256"] for record in owner_attempt_log.records] == [
        "a" * 64,
        actual_critic_request_sha256,
        actual_controller_request_sha256,
    ]
    assert [
        (entry["role"], entry["call_index"])
        for entry in context.attempt_records
    ] == [("controller", 1), ("critic", 1), ("controller", 2)]
    for entry, owner_record in zip(
        context.attempt_records, owner_attempt_log.records, strict=True
    ):
        assert set(entry) == {
            "schema", "role", "call_index", "record", "record_sha256",
        }
        assert set(entry["record"]) == _ATTEMPT_EVIDENCE_FIELDS
        assert entry["record"] == owner_record
        assert entry["record_sha256"] == _json_sha256(owner_record)

    response_retry = _complete(
        wrapped,
        "obs-1",
        state1,
        images1,
        receipts=[receipt],
        attempt_log=owner_attempt_log,
    )
    assert response_retry.command is retry
    assert [role for role, _ in _FakeClient.calls].count("critic") == 1
    assert response_retry.evidence["consumed_advisory_id"] is None
    assert "POST_ACTION_ADVISORY" not in str(_FakeClient.calls[-1][1]["instruction"])
    assert owner_attempt_log.records[-1]["request_sha256"] == "e" * 64
    assert context.attempt_records[-1]["role"] == "controller"
    assert context.attempt_records[-1]["call_index"] == 3
    assert context.attempt_records[-1]["record"] == owner_attempt_log.records[-1]


@pytest.mark.parametrize(
    "attempt_records",
    (
        [],
        [
            _attempt_record(
                "1" * 64,
                observation_id="obs-1",
                response_schema_sha256=_json_sha256(ADVISORY_SCHEMA),
            ),
            _attempt_record(
                "2" * 64,
                observation_id="obs-1",
                response_schema_sha256=_json_sha256(ADVISORY_SCHEMA),
            ),
        ],
        [
            _attempt_record(
                "3" * 64,
                observation_id="other-observation",
                response_schema_sha256=_json_sha256(ADVISORY_SCHEMA),
            )
        ],
        [
            _attempt_record(
                "4" * 64,
                observation_id="obs-1",
                attempt_index=7,
                response_schema_sha256=_json_sha256(ADVISORY_SCHEMA),
            )
        ],
        [
            _attempt_record(
                "5" * 64,
                observation_id="obs-1",
                response_schema_sha256=_json_sha256(ADVISORY_SCHEMA),
            )
            | {"attempt_index": False}
        ],
        [
            _attempt_record(
                "6" * 64,
                observation_id="obs-1",
                response_schema_sha256=_json_sha256(ADVISORY_SCHEMA),
            ),
            _attempt_record(
                "7" * 64,
                observation_id="other-observation",
                response_schema_sha256=_json_sha256(ADVISORY_SCHEMA),
            ),
        ],
    ),
    ids=(
        "missing",
        "multiple",
        "wrong-observation",
        "wrong-attempt",
        "boolean-attempt",
        "extra-mismatched",
    ),
)
def test_critic_requires_one_matching_attempt_log_record_before_advisory_delivery(
    attempt_records: list[dict[str, object]],
) -> None:
    context = _context()
    wrapped = _make_client_class(_FakeClient, context)()
    first = _controller_command("obs-0")
    second = _controller_command("obs-1")
    state1 = _state(0.01)
    _FakeClient.controller_outputs = [first, second]
    _FakeClient.critic_outputs = [_advisory()]
    _FakeClient.critic_attempt_records = [attempt_records]

    _complete(wrapped, "obs-0", _state(), _images())
    response = _complete(
        wrapped,
        "obs-1",
        state1,
        _images(),
        receipts=[_sealed_critic_receipt(first, state1)],
    )

    assert response.command is second
    assert context.critic_attempts == 1
    assert context.critic_successes == 0
    assert context.critic_unavailable == 1
    assert context.critic_records[0]["status"] == "critic_linkage_incomplete"
    assert context.critic_records[0]["critic_request_sha256"] is None
    assert response.evidence["consumed_advisory_id"] is None
    assert "POST_ACTION_ADVISORY" not in str(_FakeClient.calls[-1][1]["instruction"])


@pytest.mark.parametrize(
    "attempt_records",
    (
        [],
        [
            _attempt_record(
                "1" * 64,
                observation_id="obs-1",
                response_schema_sha256=_json_sha256(CONTROLLER_RESPONSE_SCHEMA),
            ),
            _attempt_record(
                "2" * 64,
                observation_id="obs-1",
                response_schema_sha256=_json_sha256(CONTROLLER_RESPONSE_SCHEMA),
            ),
        ],
        [
            _attempt_record(
                "3" * 64,
                observation_id="other-observation",
                response_schema_sha256=_json_sha256(CONTROLLER_RESPONSE_SCHEMA),
            )
        ],
        [
            _attempt_record(
                "4" * 64,
                observation_id="obs-1",
                attempt_index=7,
                response_schema_sha256=_json_sha256(CONTROLLER_RESPONSE_SCHEMA),
            )
        ],
        [
            _attempt_record(
                "5" * 64,
                observation_id="obs-1",
                response_schema_sha256=_json_sha256(CONTROLLER_RESPONSE_SCHEMA),
            )
            | {"attempt_index": False}
        ],
        [
            _attempt_record(
                "6" * 64,
                observation_id="obs-1",
                response_schema_sha256=_json_sha256(CONTROLLER_RESPONSE_SCHEMA),
            ),
            _attempt_record(
                "7" * 64,
                observation_id="other-observation",
                response_schema_sha256=_json_sha256(CONTROLLER_RESPONSE_SCHEMA),
            ),
        ],
    ),
    ids=(
        "missing",
        "multiple",
        "wrong-observation",
        "wrong-attempt",
        "boolean-attempt",
        "extra-mismatched",
    ),
)
def test_controller_linkage_is_incomplete_without_one_matching_attempt_record(
    attempt_records: list[dict[str, object]],
) -> None:
    context = _context()
    wrapped = _make_client_class(_FakeClient, context)()
    first = _controller_command("obs-0")
    second = _controller_command("obs-1")
    state1 = _state(0.01)
    _FakeClient.controller_outputs = [first, second]
    _FakeClient.critic_outputs = [_advisory()]
    _FakeClient.critic_attempt_records = [[
        _attempt_record(
            "c" * 64,
            observation_id="obs-1",
            response_schema_sha256=_json_sha256(ADVISORY_SCHEMA),
        )
    ]]
    _FakeClient.controller_attempt_records = [None, attempt_records]

    _complete(wrapped, "obs-0", _state(), _images())
    response = _complete(
        wrapped,
        "obs-1",
        state1,
        _images(),
        receipts=[_sealed_critic_receipt(first, state1)],
    )

    assert response.command is second
    assert "POST_ACTION_ADVISORY" in str(_FakeClient.calls[-1][1]["instruction"])
    assert response.evidence["controller_request_sha256"] is None
    assert response.evidence["controller_request_linkage_status"] == "incomplete"
    assert context.controller_records[1]["controller_request_sha256"] is None
    assert (
        context.controller_records[1]["controller_request_linkage_status"]
        == "incomplete"
    )
    assert context.critic_records[0]["consumed_by_controller_request_sha256"] is None
    assert (
        context.critic_records[0]["consumed_by_controller_linkage_status"]
        == "incomplete"
    )
    assert context.critic_records[0]["qwen_command_sha256"] == _json_sha256(second)


def test_controller_post_log_exception_persists_advised_attempt_before_repair() -> None:
    context = _context()
    wrapped = _make_client_class(_FakeClient, context)()
    first = _controller_command("obs-0")
    repair = _controller_command("obs-1", 0.02)
    state1 = _state(0.01)
    receipt = _sealed_critic_receipt(first, state1)
    _FakeClient.controller_outputs = [first, repair]
    _FakeClient.critic_outputs = [_advisory()]
    _FakeClient.critic_attempt_records = [[
        _attempt_record(
            "c" * 64,
            observation_id="obs-1",
            response_schema_sha256=_json_sha256(ADVISORY_SCHEMA),
        )
    ]]
    _FakeClient.controller_attempt_records = [
        [
            _attempt_record(
                "a" * 64,
                observation_id="obs-0",
                response_schema_sha256=_json_sha256(CONTROLLER_RESPONSE_SCHEMA),
            )
        ],
        [
            _attempt_record(
                "d" * 64,
                observation_id="obs-1",
                response_schema_sha256=_json_sha256(CONTROLLER_RESPONSE_SCHEMA),
            )
        ],
        [
            _attempt_record(
                "e" * 64,
                observation_id="obs-1",
                response_schema_sha256=_json_sha256(CONTROLLER_RESPONSE_SCHEMA),
            )
        ],
    ]
    _FakeClient.controller_errors_after_log = [
        None,
        _FakeMalformedResponse("malformed controller json after log"),
        None,
    ]
    owner_attempt_log = _OwnerAttemptLog()

    _complete(wrapped, "obs-0", _state(), _images(), attempt_log=owner_attempt_log)
    with pytest.raises(_FakeMalformedResponse, match="malformed controller json"):
        _complete(
            wrapped,
            "obs-1",
            state1,
            _images(),
            receipts=[receipt],
            attempt_log=owner_attempt_log,
        )

    failed_record = context.controller_records[1]
    advisory_id = context.critic_records[0]["advisory_id"]
    assert failed_record["status"] == "controller_unavailable"
    assert failed_record["consumed_advisory_id"] == advisory_id
    assert failed_record["controller_request_sha256"] == "d" * 64
    assert failed_record["controller_request_linkage_status"] == "complete"
    assert failed_record["command"] is None
    assert failed_record["command_sha256"] is None
    assert context.critic_records[0]["consumed_by_controller_request_sha256"] == (
        "d" * 64
    )
    assert (
        context.critic_records[0]["consumed_by_controller_linkage_status"]
        == "complete"
    )
    assert context.critic_records[0]["qwen_command_sha256"] is None
    assert "POST_ACTION_ADVISORY" in str(_FakeClient.calls[-1][1]["instruction"])

    response_retry = _complete(
        wrapped,
        "obs-1",
        state1,
        _images(),
        receipts=[receipt],
        attempt_log=owner_attempt_log,
    )

    assert response_retry.command is repair
    assert context.controller_records[2]["consumed_advisory_id"] is None
    assert "POST_ACTION_ADVISORY" not in str(_FakeClient.calls[-1][1]["instruction"])
    assert [record["request_sha256"] for record in owner_attempt_log.records] == [
        "a" * 64,
        "c" * 64,
        "d" * 64,
        "e" * 64,
    ]


@pytest.mark.parametrize(
    "attempt_records",
    (
        [],
        [
            _attempt_record(
                "d" * 64,
                observation_id="obs-1",
                response_schema_sha256=_json_sha256(CONTROLLER_RESPONSE_SCHEMA),
            ),
            _attempt_record(
                "e" * 64,
                observation_id="obs-1",
                response_schema_sha256=_json_sha256(CONTROLLER_RESPONSE_SCHEMA),
            ),
        ],
    ),
    ids=("missing", "ambiguous"),
)
def test_controller_post_log_exception_without_exact_record_fails_closed(
    attempt_records: list[dict[str, object]],
) -> None:
    context = _context()
    wrapped = _make_client_class(_FakeClient, context)()
    first = _controller_command("obs-0")
    state1 = _state(0.01)
    receipt = _sealed_critic_receipt(first, state1)
    _FakeClient.controller_outputs = [first]
    _FakeClient.critic_outputs = [_advisory()]
    _FakeClient.critic_attempt_records = [[
        _attempt_record(
            "c" * 64,
            observation_id="obs-1",
            response_schema_sha256=_json_sha256(ADVISORY_SCHEMA),
        )
    ]]
    _FakeClient.controller_attempt_records = [
        [
            _attempt_record(
                "a" * 64,
                observation_id="obs-0",
                response_schema_sha256=_json_sha256(CONTROLLER_RESPONSE_SCHEMA),
            )
        ],
        attempt_records,
    ]
    _FakeClient.controller_errors_after_log = [
        None,
        _FakeMalformedResponse("malformed controller json after log"),
    ]

    _complete(wrapped, "obs-0", _state(), _images())
    with pytest.raises(_FakeMalformedResponse, match="malformed controller json"):
        _complete(wrapped, "obs-1", state1, _images(), receipts=[receipt])

    failed_record = context.controller_records[1]
    assert failed_record["status"] == "controller_unavailable"
    assert failed_record["consumed_advisory_id"] == context.critic_records[0][
        "advisory_id"
    ]
    assert failed_record["controller_request_sha256"] is None
    assert failed_record["controller_request_linkage_status"] == "incomplete"
    assert failed_record["command_sha256"] is None
    assert context.critic_records[0]["consumed_by_controller_request_sha256"] is None
    assert (
        context.critic_records[0]["consumed_by_controller_linkage_status"]
        == "incomplete"
    )
    assert context.critic_records[0]["qwen_command_sha256"] is None


def test_first_action_review_remains_pending_after_partial_first_receipt() -> None:
    context = _context()
    wrapped = _make_client_class(_FakeClient, context)()
    first = _controller_command("obs-0", 0.01)
    second = _controller_command("obs-1", 0.03)
    third = _controller_command("obs-2", -0.02)
    state0, state1, state2 = _state(0.0), _state(0.01), _state(0.03)
    partial_receipt = _sealed_critic_receipt(
        first, state1, start_torque_available=False
    )
    complete_receipt = _sealed_critic_receipt(second, state2)
    _FakeClient.controller_outputs = [first, second, third]
    _FakeClient.critic_outputs = [_advisory()]

    _complete(wrapped, "obs-0", state0, _images())
    response1 = _complete(
        wrapped, "obs-1", state1, _images(), receipts=[partial_receipt]
    )
    assert response1.command is second
    assert context.critic_attempts == 0
    assert context.trigger_evaluations[-1]["eligible_trigger"] == "first_action"
    assert context.trigger_evaluations[-1]["suppressed"] == "receipt_ineligible"

    response2 = _complete(
        wrapped, "obs-2", state2, _images(), receipts=[partial_receipt, complete_receipt]
    )

    assert response2.command is third
    assert context.critic_attempts == 1
    assert context.critic_successes == 1
    assert context.trigger_evaluations[-1]["fired_trigger"] == "first_action"
    assert [role for role, _ in _FakeClient.calls].count("critic") == 1


def test_critic_linkage_records_are_deep_canonical_snapshots() -> None:
    context = _context()
    wrapped = _make_client_class(_FakeClient, context)()
    first = _controller_command("obs-0", 0.01)
    second = _controller_command("obs-1", -0.01)
    advisory = _advisory()
    state1 = _state(0.01)
    receipt = _sealed_critic_receipt(first, state1)
    _FakeClient.controller_outputs = [
        first,
        second,
    ]
    _FakeClient.critic_outputs = [advisory]
    _FakeClient.critic_attempt_records = [[
        _attempt_record(
            "c" * 64,
            observation_id="obs-1",
            response_schema_sha256=_json_sha256(ADVISORY_SCHEMA),
        )
    ]]
    _FakeClient.controller_attempt_records = [
        [
            _attempt_record(
                "a" * 64,
                observation_id="obs-0",
                response_schema_sha256=_json_sha256(CONTROLLER_RESPONSE_SCHEMA),
            )
        ],
        [
            _attempt_record(
                "d" * 64,
                observation_id="obs-1",
                response_schema_sha256=_json_sha256(CONTROLLER_RESPONSE_SCHEMA),
            )
        ],
    ]

    _complete(wrapped, "obs-0", _state(), _images())
    response = _complete(
        wrapped, "obs-1", state1, _images(), receipts=[receipt]
    )
    assert response.command is second
    critic_record = context.critic_records[0]
    controller_record = context.controller_records[1]
    decision = context.trigger_evaluations[1]
    receipt_sha256 = critic_record["sealed_public_receipt_sha256"]
    command_sha256 = controller_record["command_sha256"]

    first["targets"]["joint1"] = 99.0
    second["targets"]["joint1"] = -99.0
    receipt["telemetry_summary"]["arm_joint_position"]["end_rad"][0] = 99.0
    state1["state.arm_joint_position"][0] = 99.0
    advisory["evidence"].append("joint_tracking")

    assert response.command["targets"]["joint1"] == -99.0
    assert decision["previous_executed_command"]["targets"]["joint1"] == 0.01
    assert (
        decision["sealed_public_receipt"]["telemetry_summary"]["arm_joint_position"][
            "end_rad"
        ][0]
        == 0.01
    )
    assert critic_record["sealed_public_receipt_sha256"] == receipt_sha256
    assert critic_record["advisory"] == _advisory()
    assert controller_record["command"]["targets"]["joint1"] == -0.01
    assert controller_record["command_sha256"] == command_sha256


@pytest.mark.parametrize("kind", ("move_joints", "base_action"))
def test_complete_arm_and_base_receipts_are_eligible_for_one_critic_call(
    kind: str,
) -> None:
    context = _context()
    wrapped = _make_client_class(_FakeClient, context)()
    first = _controller_command("obs-0", kind=kind)
    second = _controller_command("obs-1")
    state0, state1 = _state(), _state(0.01)
    receipt = _sealed_critic_receipt(first, state1)
    _FakeClient.controller_outputs = [first, second]
    _FakeClient.critic_outputs = [_advisory()]

    _complete(wrapped, "obs-0", state0, _images())
    response = _complete(
        wrapped, "obs-1", state1, _images(), receipts=[receipt]
    )

    assert response.command is second
    assert context.critic_attempts == 1
    assert context.critic_successes == 1
    assert context.trigger_evaluations[-1]["sealed_public_receipt"] == receipt


@pytest.mark.parametrize(
    "failure",
    (
        "partial_initial_torque",
        "stale_receipt",
        "matching_receipt_is_older",
        "unsealed_receipt",
        "incomplete_receipt",
        "rejected_receipt",
        "wrong_kind",
        "fresh_state_mismatch",
        "rgb_mismatch",
        "private_state",
    ),
)
def test_critic_receipt_eligibility_fails_closed_without_consuming_attempt(
    failure: str,
) -> None:
    context = _context()
    wrapped = _make_client_class(_FakeClient, context)()
    first = _controller_command("obs-0")
    second = _controller_command("obs-1")
    state0, state1 = _state(), _state(0.01)
    receipt = _sealed_critic_receipt(
        first,
        state1,
        start_torque_available=failure != "partial_initial_torque",
    )
    receipts = [receipt]
    rgb_change = None
    if failure in {"stale_receipt", "matching_receipt_is_older"}:
        stale_command = _controller_command("older-observation")
        stale = _sealed_critic_receipt(stale_command, state1)
        receipts = [stale] if failure == "stale_receipt" else [receipt, stale]
    elif failure == "unsealed_receipt":
        receipt["mean_absolute_rgb_change"] = None
    elif failure == "incomplete_receipt":
        del receipt["telemetry_summary"]
    elif failure == "rejected_receipt":
        receipt["accepted"] = False
        receipt["failure_status"] = "rejected"
    elif failure == "wrong_kind":
        receipt["kind"] = "servo_failed"
    elif failure == "fresh_state_mismatch":
        state1["state.arm_joint_position"] = [0.4] * 7
    elif failure == "rgb_mismatch":
        rgb_change = {"left": 9.0, "right": 1.0, "wrist": 1.0}
    elif failure == "private_state":
        state1["state.object_pose"] = [1.0, 2.0, 3.0]
    _FakeClient.controller_outputs = [first, second]
    _FakeClient.critic_outputs = [_advisory()]

    _complete(wrapped, "obs-0", state0, _images())
    response = _complete(
        wrapped,
        "obs-1",
        state1,
        _images(),
        receipts=receipts,
        rgb_change=rgb_change,
    )

    assert response.command is second
    assert context.critic_attempts == 0
    assert context.critic_successes == 0
    assert [role for role, _ in _FakeClient.calls].count("critic") == 0
    assert response.evidence["consumed_advisory_id"] is None
    assert context.trigger_evaluations[-1]["receipt_eligibility"] != "eligible"


@pytest.mark.parametrize(
    "mutation",
    (
        "nan_pose_delta",
        "string_torque_baseline",
        "boolean_gripper_residual",
    ),
)
def test_critic_deep_receipt_validation_rejects_nonfinite_and_bool_aliases(
    mutation: str,
) -> None:
    context = _context()
    wrapped = _make_client_class(_FakeClient, context)()
    first = _controller_command("obs-0")
    second = _controller_command("obs-1")
    state1 = _state(0.01)
    receipt = _sealed_critic_receipt(first, state1)
    if mutation == "nan_pose_delta":
        receipt["end_effector_pose_delta"]["translation_m"][0] = math.nan
    elif mutation == "string_torque_baseline":
        receipt["telemetry_summary"]["arm_applied_torque"]["start_nm"][0] = "1.0"
    elif mutation == "boolean_gripper_residual":
        receipt["gripper_residual"]["qpos_delta"][0] = False
    _FakeClient.controller_outputs = [first, second]
    _FakeClient.critic_outputs = [_advisory()]

    _complete(wrapped, "obs-0", _state(), _images())
    response = _complete(
        wrapped, "obs-1", state1, _images(), receipts=[receipt]
    )

    assert response.command is second
    assert context.critic_attempts == 0
    assert [role for role, _ in _FakeClient.calls].count("critic") == 0
    assert context.trigger_evaluations[-1]["receipt_eligibility"] != "eligible"


@pytest.mark.parametrize(
    ("receipt_target", "receipt_intent"),
    (
        (False, 0.0),
        (0.0, False),
    ),
    ids=("boolean-target-alias", "boolean-intent-alias"),
)
def test_critic_command_match_rejects_move_gripper_boolean_numeric_aliases(
    receipt_target: object,
    receipt_intent: object,
) -> None:
    context = _context()
    wrapped = _make_client_class(_FakeClient, context)()
    first = _controller_command("obs-0")
    first["targets"] = {"joint1": 0.01, "gripper": 0.0}
    second = _controller_command("obs-1")
    state1 = _state(0.01)
    receipt = _sealed_critic_receipt(first, state1)
    receipt["requested_targets"]["gripper"] = receipt_target
    receipt["gripper_intent"] = receipt_intent
    _FakeClient.controller_outputs = [first, second]
    _FakeClient.critic_outputs = [_advisory()]

    _complete(wrapped, "obs-0", _state(), _images())
    response = _complete(
        wrapped, "obs-1", state1, _images(), receipts=[receipt]
    )

    assert response.command is second
    assert context.critic_attempts == 0
    assert [role for role, _ in _FakeClient.calls].count("critic") == 0
    assert context.trigger_evaluations[-1]["receipt_eligibility"] != "eligible"


@pytest.mark.parametrize(
    "mutation",
    (
        "state.grasped",
        "is_success",
        "contact_force",
        "task_progress",
        "reward_signal",
        "gate_truth",
        "state.extra_public",
        "missing_eef_position",
        "missing_base_position",
        "missing_gripper_qpos",
        "pixel_extra_field",
        "pixel_boolean_numeric",
    ),
)
def test_critic_requires_exact_task3_public_state_schema(mutation: str) -> None:
    context = _context()
    wrapped = _make_client_class(_FakeClient, context)()
    first = _controller_command("obs-0")
    second = _controller_command("obs-1")
    state1 = _state(0.01)
    receipt = _sealed_critic_receipt(first, state1)
    if mutation == "missing_eef_position":
        del state1["state.end_effector_position_relative"]
    elif mutation == "missing_base_position":
        del state1["state.base_position"]
    elif mutation == "missing_gripper_qpos":
        del state1["state.gripper_qpos"]
    elif mutation == "pixel_extra_field":
        state1["state.end_effector_external_pixels"]["left"]["depth_m"] = 1.0
    elif mutation == "pixel_boolean_numeric":
        state1["state.end_effector_external_pixels"]["left"]["u_px"] = False
    else:
        state1[mutation] = True
    _FakeClient.controller_outputs = [first, second]
    _FakeClient.critic_outputs = [_advisory()]

    _complete(wrapped, "obs-0", _state(), _images())
    response = _complete(
        wrapped, "obs-1", state1, _images(), receipts=[receipt]
    )

    assert response.command is second
    assert context.critic_attempts == 0
    assert [role for role, _ in _FakeClient.calls].count("critic") == 0
    assert context.trigger_evaluations[-1]["receipt_eligibility"] != "eligible"


def test_shared_public_state_validator_accepts_initial_torque_and_null_depth_pixels() -> None:
    from adaptive.panda_embodiment import validate_public_state

    state = _state(torque_available=False)
    state["state.end_effector_external_pixels"]["left"] = {
        "u_px": None,
        "v_px": None,
        "visible": False,
        "depth_valid": False,
    }

    snapshot = validate_public_state(state)

    assert snapshot == state
    snapshot["state.base_position"][0] = 99.0
    assert state["state.base_position"][0] == 0.0
    with pytest.raises(ValueError, match="torque"):
        validate_public_state(state, require_torque_available=True)


def test_critic_accepts_legal_null_depth_public_pixels_after_complete_receipt() -> None:
    context = _context()
    wrapped = _make_client_class(_FakeClient, context)()
    first = _controller_command("obs-0")
    second = _controller_command("obs-1")
    state1 = _state(0.01)
    state1["state.end_effector_external_pixels"]["right"] = {
        "u_px": None,
        "v_px": None,
        "visible": False,
        "depth_valid": False,
    }
    receipt = _sealed_critic_receipt(first, state1)
    _FakeClient.controller_outputs = [first, second]
    _FakeClient.critic_outputs = [_advisory()]

    _complete(wrapped, "obs-0", _state(), _images())
    response = _complete(wrapped, "obs-1", state1, _images(), receipts=[receipt])

    assert response.command is second
    assert context.critic_attempts == 1
    assert context.critic_successes == 1
    assert context.trigger_evaluations[-1]["receipt_eligibility"] == "eligible"


def test_shared_public_receipt_validator_accepts_task4_null_pixel_displacement() -> None:
    from adaptive.joint_runner import validate_public_receipt

    receipt = _sealed_critic_receipt(_controller_command("obs-0"), _state(0.01))
    receipt["end_effector_external_pixel_displacement"]["left"] = {
        "start_px": None,
        "end_px": [100.01, 200.0],
        "delta_px": None,
        "distance_px": None,
        "start_visible": False,
        "end_visible": True,
        "start_depth_valid": False,
        "end_depth_valid": True,
    }

    snapshot = validate_public_receipt(receipt, require_sealed_rgb=True)

    assert snapshot == receipt
    snapshot["end_effector_external_pixel_displacement"]["left"]["end_px"][0] = 99.0
    assert (
        receipt["end_effector_external_pixel_displacement"]["left"]["end_px"][0]
        == 100.01
    )


def test_public_receipt_validator_accepts_production_maximum_step_float_residue() -> (
    None
):
    from adaptive.joint_runner import validate_public_receipt

    receipt = _sealed_critic_receipt(
        _controller_command("obs-0", -0.975),
        _state(-0.975),
    )
    production_step = -0.975 - -1.0
    qpos = receipt["telemetry_summary"]["arm_joint_position"]
    qpos["start_rad"][0] = -1.0
    qpos["delta_rad"][0] = production_step
    qpos["peak_abs_delta_from_start_rad"][0] = production_step
    receipt["realized_arm_qpos_delta"][0] = production_step
    receipt["maximum_commanded_step"] = production_step

    snapshot = validate_public_receipt(receipt, require_sealed_rgb=True)

    assert production_step > MAX_JOINT_STEP
    assert snapshot["maximum_commanded_step"] == pytest.approx(MAX_JOINT_STEP)
    overstep = _json_copy(receipt)
    assert isinstance(overstep, dict)
    overstep["maximum_commanded_step"] = MAX_JOINT_STEP + 2e-12
    with pytest.raises(ValueError, match="maximum|accounting"):
        validate_public_receipt(overstep, require_sealed_rgb=True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda receipt: receipt["resolved_held_dimensions"].update(
                {"joint1": -0.01}
            ),
            "complement|held|requested",
        ),
        (
            lambda receipt: receipt["resolved_held_dimensions"].pop("joint2"),
            "complement|held|requested",
        ),
        (
            lambda receipt: receipt["bounded_endpoint"].__setitem__(0, 0.02),
            "bounded endpoint|requested|endpoint",
        ),
        (
            lambda receipt: receipt["resolved_held_dimensions"].__setitem__(
                "joint2", -0.5
            ),
            "held|start|endpoint",
        ),
        (
            lambda receipt: receipt["telemetry_summary"]["arm_joint_position"][
                "start_rad"
            ].__setitem__(0, -0.02),
            "delta|start|end",
        ),
        (lambda receipt: receipt.__setitem__("endpoint_error", 0.01), "endpoint"),
        (
            lambda receipt: receipt["gripper_residual"].__setitem__(
                "measured_end_finger_separation", 9.0
            ),
            "gripper|separation",
        ),
        (
            lambda receipt: receipt["telemetry_summary"]["arm_applied_torque"][
                "delta_nm"
            ].__setitem__(0, 9.0),
            "torque|delta",
        ),
        (
            lambda receipt: receipt["telemetry_summary"]["arm_applied_torque"][
                "peak_abs_delta_from_start_nm"
            ].__setitem__(0, 0.5),
            "torque|peak",
        ),
        (
            lambda receipt: receipt["telemetry_summary"]["arm_joint_velocity"][
                "start_rad_s"
            ].__setitem__(0, 0.5),
            "velocity|maximum",
        ),
        (
            lambda receipt: receipt["telemetry_summary"]["end_effector_wrench"][
                "force"
            ]["delta_n"].__setitem__(0, 9.0),
            "force|delta",
        ),
        (
            lambda receipt: receipt["telemetry_summary"]["end_effector_wrench"][
                "force"
            ].__setitem__("peak_delta_norm_n", 0.5),
            "force|peak",
        ),
        (
            lambda receipt: receipt["telemetry_summary"]["end_effector_wrench"][
                "torque"
            ]["delta_nm"].__setitem__(1, 9.0),
            "wrench|torque|delta",
        ),
        (
            lambda receipt: receipt["telemetry_summary"]["end_effector_wrench"][
                "torque"
            ].__setitem__("peak_delta_norm_nm", 0.5),
            "wrench|torque|peak",
        ),
    ),
    ids=(
        "requested-held-overlap",
        "missing-held-complement",
        "requested-endpoint-mismatch",
        "held-start-mismatch",
        "qpos-delta-not-derived",
        "endpoint-error-not-derived",
        "gripper-separation-not-derived",
        "torque-delta-not-derived",
        "torque-peak-too-small",
        "qvel-maximum-too-small",
        "force-delta-not-derived",
        "force-peak-too-small",
        "wrench-torque-delta-not-derived",
        "wrench-torque-peak-too-small",
    ),
)
def test_shared_public_move_receipt_validator_rejects_semantic_contradictions(
    mutation: object, message: str
) -> None:
    from adaptive.joint_runner import validate_public_receipt

    receipt = _sealed_critic_receipt(_controller_command("obs-0"), _state(0.01))
    mutation(receipt)

    with pytest.raises(ValueError, match=message):
        validate_public_receipt(
            receipt,
            require_sealed_rgb=True,
            require_torque_baseline=True,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda receipt: receipt.__setitem__("remaining_endpoint_error", 0.03),
            "remaining endpoint|base",
        ),
        (
            lambda receipt: receipt["telemetry_summary"]["arm_joint_position"][
                "start_rad"
            ].__setitem__(0, -0.02),
            "delta|start|end",
        ),
    ),
    ids=("remaining-error-not-derived", "base-qpos-delta-not-derived"),
)
def test_shared_public_base_receipt_validator_rejects_semantic_contradictions(
    mutation: object, message: str
) -> None:
    from adaptive.joint_runner import validate_public_receipt

    receipt = _sealed_critic_receipt(
        _controller_command("obs-0", kind="base_action"),
        _state(0.01),
    )
    mutation(receipt)

    with pytest.raises(ValueError, match=message):
        validate_public_receipt(
            receipt,
            require_sealed_rgb=True,
            require_torque_baseline=True,
        )


@pytest.mark.parametrize("tracking_pause_count", (1, 5))
def test_shared_public_base_receipt_validator_rejects_any_tracking_pause_count(
    tracking_pause_count: int,
) -> None:
    from adaptive.joint_runner import validate_public_receipt

    receipt = _sealed_critic_receipt(
        _controller_command("obs-0", kind="base_action"),
        _state(0.01),
    )
    receipt["tracking_pause_count"] = tracking_pause_count

    with pytest.raises(ValueError, match="base|tracking_pause"):
        validate_public_receipt(
            receipt,
            require_sealed_rgb=True,
            require_torque_baseline=True,
        )


def test_critic_rejects_base_hold_receipt_with_transition_step_count() -> None:
    context = _context()
    wrapped = _make_client_class(_FakeClient, context)()
    first = _controller_command("obs-0", kind="base_action")
    second = _controller_command("obs-1")
    state1 = _state(0.01)
    receipt = _sealed_critic_receipt(first, state1)
    receipt["tracking_pause_count"] = 0
    receipt["step_count"] = MIN_GRIPPER_ACTIONS
    receipt["base_motion_step_count"] = 5
    _FakeClient.controller_outputs = [first, second]
    _FakeClient.critic_outputs = [_advisory()]

    _complete(wrapped, "obs-0", _state(), _images())
    response = _complete(wrapped, "obs-1", state1, _images(), receipts=[receipt])

    assert response.command is second
    assert context.critic_attempts == 0
    assert [role for role, _ in _FakeClient.calls].count("critic") == 0
    assert context.trigger_evaluations[-1]["receipt_eligibility"] == (
        "command_mismatch"
    )


def test_critic_freshness_rejects_receipt_gripper_end_that_differs_from_state() -> None:
    context = _context()
    wrapped = _make_client_class(_FakeClient, context)()
    first = _controller_command("obs-0")
    second = _controller_command("obs-1")
    state1 = _state(0.01)
    receipt = _sealed_critic_receipt(first, state1)
    receipt["gripper_residual"] = {
        "start_qpos": [0.04, -0.04],
        "end_qpos": [0.0, 0.0],
        "qpos_delta": [-0.04, 0.04],
        "measured_end_finger_separation": 0.0,
    }
    _FakeClient.controller_outputs = [first, second]
    _FakeClient.critic_outputs = [_advisory()]

    _complete(wrapped, "obs-0", _state(), _images())
    response = _complete(wrapped, "obs-1", state1, _images(), receipts=[receipt])

    assert response.command is second
    assert context.critic_attempts == 0
    assert [role for role, _ in _FakeClient.calls].count("critic") == 0
    assert context.trigger_evaluations[-1]["receipt_eligibility"] == (
        "fresh_state_mismatch"
    )


def _proposal_audit(
    verdict: str = "approve",
    *,
    contradiction: str | None = None,
) -> dict[str, object]:
    if verdict == "approve":
        return {
            "verdict": "approve",
            "contradiction": "none",
            "evidence": ["external_rgb", "milestone_history"],
            "suggested_correction": "continue_milestone",
            "confidence": "high",
        }
    return {
        "verdict": "revise",
        "contradiction": contradiction or "visual_alignment_unverified",
        "evidence": ["external_rgb"],
        "suggested_correction": "revise_alignment",
        "confidence": "medium",
    }


def _milestone_command(
    observation_id: str,
    milestone: str,
    target: float = 0.01,
    *,
    kind: str = "move_joints",
) -> dict[str, object]:
    note = f"milestone={milestone}; fresh public evidence supports this proposal"
    if kind in {"finish", "give_up"}:
        return {"kind": kind, "observation_id": observation_id, "note": note}
    return {
        "kind": "move_joints",
        "observation_id": observation_id,
        "targets": {"joint1": target},
        "note": note,
    }


def _cartesian_milestone_command(
    observation_id: str,
    milestone: str,
    translation_x: float = 0.01,
) -> dict[str, object]:
    return {
        "kind": "cartesian_delta",
        "observation_id": observation_id,
        "translation_m": [translation_x, 0.0, 0.0],
        "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        "gripper": "hold",
        "note": (
            f"milestone={milestone}; evidence=external RGB; "
            "expect=end effector translates"
        ),
    }


def _proposal_complete(
    wrapped: object,
    observation_id: str,
    state: dict[str, object],
    images: dict[str, bytes],
    *,
    receipts: list[dict[str, object]] | None = None,
    attempt_log: _OwnerAttemptLog | None = None,
    camera_calibration: dict[str, object] | None = None,
) -> _FakeResponse:
    kwargs: dict[str, object] = {
        "observation_id": observation_id,
        "system_prompt": "OFFICIAL SYSTEM PROMPT",
        "instruction": _instruction(receipts),
        "public_state": state,
        "images": images,
        "response_schema": CONTROLLER_RESPONSE_SCHEMA,
        "max_tokens": CONTROLLER_MAX_TOKENS,
        "proposal_audit_context": {
            "current_qpos": state["state.arm_joint_position"],
            "current_gripper": 1.0,
            "remaining_actions": 450,
            "sequence": 0,
            "camera_calibration": camera_calibration or {},
        },
    }
    if attempt_log is not None:
        kwargs["attempt_log"] = attempt_log
    return wrapped.complete(**kwargs)


def test_proposal_audit_can_approve_image_servo_using_fresh_calibration() -> None:
    from adaptive import remote_driver

    context = _context()
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    command = {
        **_image_servo_payload(),
        "observation_id": "obs-0",
        "target_role": "fixture_handle",
        "note": "milestone=observe; toaster handle center in left RGB",
    }
    _FakeClient.controller_outputs = [command]
    _FakeClient.critic_outputs = [_proposal_audit()]
    state = _state()
    state.update(_image_servo_public_state())

    response = _proposal_complete(
        wrapped,
        "obs-0",
        state,
        _images(),
        camera_calibration=_image_servo_calibration(),
    )

    assert response.command == command
    assert context.proposal_records[-1]["status"] == "approved_for_execution"


def _milestone_context_from_instruction(instruction: object) -> dict[str, object]:
    from adaptive import critic_protocol as protocol

    assert isinstance(instruction, str)
    assert instruction.count(protocol.MILESTONE_CONTEXT_MARKER) == 1
    prefix, payload = instruction.rsplit(protocol.MILESTONE_CONTEXT_MARKER, 1)
    assert prefix
    assert payload and "\n" not in payload
    return json.loads(payload)


@pytest.mark.parametrize(
    ("family", "history", "current", "allowed"),
    (
        ("grasp_place", [], None, ["observe"]),
        ("grasp_place", ["observe"], "observe", ["observe", "approach"]),
        (
            "grasp_place",
            ["observe", "approach"],
            "approach",
            ["approach", "pregrasp"],
        ),
        (
            "articulated",
            ["observe", "approach"],
            "approach",
            ["approach", "engage"],
        ),
        (
            "control",
            ["observe", "approach", "engage", "actuate", "verify_goal"],
            "verify_goal",
            ["verify_goal"],
        ),
    ),
)
def test_grounded_controller_context_is_derived_from_closed_history(
    family: str,
    history: list[str],
    current: str | None,
    allowed: list[str],
) -> None:
    from adaptive import critic_protocol as protocol

    expected: dict[str, object] = {
        "allowed_next_milestones": allowed,
        "current_closed_milestone": current,
        "family": family,
    }
    if family == "articulated" and current == "approach":
        expected["edge_action_contract"] = {
            "engage": {
                "depth_delta_m": "nonzero",
                "gripper": "close",
                "kind": "image_servo",
            },
            "open_image_servo_claims": "approach",
        }
    assert protocol.milestone_context_packet(family, history) == expected


def test_grounded_initial_and_revision_calls_end_with_progress_context() -> None:
    from adaptive import critic_protocol as protocol
    from adaptive import remote_driver

    context = _context()
    context.milestone_history = ["observe", "approach"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    first = _milestone_command("obs-2", "approach", 0.01)
    revised = _milestone_command("obs-2", "approach", 0.02)
    _FakeClient.controller_outputs = [first, revised]
    _FakeClient.critic_outputs = [_proposal_audit("revise"), _proposal_audit()]

    response = _proposal_complete(wrapped, "obs-2", _state(), _images())

    assert response.command is revised
    controller_calls = [call for role, call in _FakeClient.calls if role == "controller"]
    assert len(controller_calls) == 2
    expected = protocol.milestone_context_packet(
        "articulated", ["observe", "approach"]
    )
    for call in controller_calls:
        assert _milestone_context_from_instruction(call["instruction"]) == expected
    assert "PROPOSAL_AUDIT_REVISION" in str(controller_calls[1]["instruction"])
    assert all(
        protocol.MILESTONE_CONTEXT_MARKER not in str(call["instruction"])
        for role, call in _FakeClient.calls
        if role == "critic"
    )


def test_proposal_audit_schema_is_exact_closed_and_semantically_paired() -> None:
    from adaptive import critic_protocol as protocol

    assert protocol.PROPOSAL_AUDIT_ENUMS == {
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
    schema = protocol.PROPOSAL_AUDIT_SCHEMA
    assert set(schema["properties"]) == set(protocol.PROPOSAL_AUDIT_ENUMS)
    assert schema["additionalProperties"] is False
    assert "number" not in json.dumps(schema, sort_keys=True)
    assert protocol.validate_proposal_audit(_proposal_audit()) == _proposal_audit()
    with pytest.raises(ValueError, match="approve.*none"):
        protocol.validate_proposal_audit(
            _proposal_audit() | {"contradiction": "stagnation"}
        )
    with pytest.raises(ValueError, match="revise.*non-none"):
        protocol.validate_proposal_audit(
            _proposal_audit("revise") | {"contradiction": "none"}
        )
    assert protocol.validate_proposal_audit(
        _proposal_audit()
        | {
            "evidence": [
                "external_rgb",
                "wrist_rgb",
                "external_rgb",
            ]
        }
    )["evidence"] == ["external_rgb", "wrist_rgb"]
    with pytest.raises(ValueError, match="fields"):
        protocol.validate_proposal_audit(_proposal_audit() | {"targets": {}})


def test_proposal_critic_prompt_forbids_numeric_or_command_authority() -> None:
    prompt = (
        Path(__file__).resolve().parents[1]
        / "prompts"
        / "proposal_audit_critic.txt"
    ).read_text(encoding="utf-8")

    assert "pre-execution" in prompt.casefold()
    assert "PandaOmron" in prompt
    assert "same frozen qwen" in prompt.casefold()
    assert "must not emit" in prompt.casefold()
    assert "numbers" in prompt.casefold()
    assert "commands" in prompt.casefold()
    assert "target maps" in prompt.casefold()
    assert "free text" in prompt.casefold()
    assert "qwen controller alone" in prompt.casefold()


def test_pick_place_prompts_keep_source_object_distinct_from_destination() -> None:
    root = Path(__file__).resolve().parents[1] / "prompts"
    controller = " ".join(
        (root / "joint_system.txt").read_text(encoding="utf-8").casefold().split()
    )
    critic = " ".join(
        (root / "proposal_audit_critic.txt")
        .read_text(encoding="utf-8")
        .casefold()
        .split()
    )

    for prompt in (controller, critic):
        assert "movable source object" in prompt
        assert "destination receptacle" in prompt
        assert "do not close" in prompt
    assert "until a sealed receipt verifies a grasp" in controller
    assert "revise_alignment" in critic
    assert "approve this bounded retry" in critic
    assert "future grasp effect" in critic
    for prompt in (controller, critic):
        assert "source-holding fixture" in prompt
        assert "view-gathering" in prompt
        assert "settled no-op" in prompt
        assert "measured_end_finger_separation" in prompt
        assert "empty grasp" in prompt
        assert "0.002" in prompt
        assert "empty-grasp recovery" in prompt
        assert "empty-grasp recovery starts" in prompt
        assert "different visible source point" in prompt
        assert "strictly greater than `0.002`" in prompt
        assert "never round" in prompt


def test_proposal_critic_prompt_treats_first_decision_hold_as_current_evidence() -> None:
    prompt = (
        Path(__file__).resolve().parents[1]
        / "prompts"
        / "proposal_audit_critic.txt"
    ).read_text(encoding="utf-8").casefold()
    prompt = " ".join(prompt.split())

    assert "null immediate prior receipt is expected on the first decision" in prompt
    assert "not missing evidence" in prompt
    assert "bounded hold/open draft" in prompt
    assert "current joint velocity" in prompt
    assert "gripper state" in prompt
    assert "no other contradiction" in prompt


def test_proposal_critic_audits_gripper_intent_without_treating_open_as_grasp() -> None:
    prompt = (
        Path(__file__).resolve().parents[1]
        / "prompts"
        / "proposal_audit_critic.txt"
    ).read_text(encoding="utf-8").casefold()
    prompt = " ".join(prompt.split())

    assert "`1.0` means open and `0.0` means closed" in prompt
    assert "note and numeric gripper intent disagree" in prompt
    assert "`motion_effect_mismatch`" in prompt
    assert "does not claim a grasp or release" in prompt
    assert "do not use `grasp_unverified` or `release_unverified`" in prompt
    assert "does not authorize you to output either value" in prompt
    assert "rounded magnitude such as `0.02` for a `0.025` payload" in prompt
    assert "is not a motion-effect contradiction" in prompt
    assert "direction, command kind, and gripper intent agree" in prompt


def test_joint_prompt_retains_p01_p22_and_adds_closed_proposal_rules() -> None:
    root = Path(__file__).resolve().parents[1]
    prompt = load_joint_system_prompt(root)
    sections = _panda_prompt_sections(prompt)

    assert set(sections) == set(PANDA_PROMPT_PARITY_CONTRACT)
    assert "milestone=observe" in prompt
    assert "initial draft plus at most nine revisions" in prompt.casefold()
    assert "same observation" in prompt.casefold()
    assert "proposal audit" in prompt.casefold()
    assert "qwen alone authors" in prompt.casefold()
    assert "verify_goal" in prompt
    assert "omit every arm joint you intend to hold" in prompt.casefold()
    assert "never restate all seven" in prompt.casefold()


def test_grounded_cartesian_prompt_contract_is_six_axis_and_public_only() -> None:
    root = Path(__file__).resolve().parents[1]
    controller = load_joint_system_prompt(root)
    critic = (root / "prompts" / "proposal_audit_critic.txt").read_text(
        encoding="utf-8"
    )
    composed = " ".join(controller.casefold().split())
    critic_normalized = " ".join(critic.casefold().split())

    assert "`cartesian_delta`" in composed
    assert "`translation_m`" in composed
    assert "`rotation_axis_angle_rad`" in composed
    assert "robot-base frame" in composed
    assert "zero rotation" in composed
    assert "constrains orientation" in composed
    assert "0.03" in composed and "0.04" in composed
    assert "0.08" in composed and "0.10" in composed
    assert "damped least-squares" in composed
    assert "fresh public" in composed
    assert "qwen alone chooses" in composed
    assert "must not emit numbers" in critic_normalized
    assert "original `cartesian_delta` draft" in critic_normalized
    assert "derived joint endpoint" in critic_normalized
    assert "open-gripper cartesian approach" in critic_normalized
    assert "does not require verified pixel alignment" in critic_normalized
    assert "approve it as view gathering" in critic_normalized
    assert "`cartesian_view_gathering`" in critic_normalized
    assert "all four booleans are true" in critic_normalized
    assert "approve the original approach draft" in critic_normalized
    for forbidden in (
        "opentoasterovendoor",
        "adjustwatertemperature",
        "pickplacetoastertocounter",
    ):
        assert forbidden not in composed
        assert forbidden not in critic_normalized


def test_critic_allows_first_bounded_actuation_after_public_engagement() -> None:
    root = Path(__file__).resolve().parents[1]
    critic = " ".join(
        (root / "prompts" / "proposal_audit_critic.txt")
        .read_text(encoding="utf-8")
        .casefold()
        .split()
    )

    assert "contact-producing `image_servo` receipt" in critic
    assert "strictly greater than `0.0032`" in critic
    assert "approve the first bounded nonzero `cartesian_delta`" in critic
    assert "does not require the actuation effect before execution" in critic
    assert "only a later repeated actuation" in critic
    assert "changes the cartesian actuation direction" in critic
    assert "approve that changed-direction exploration" in critic
    assert "`actuation_revision`" in critic
    assert "all four booleans are true" in critic
    assert "`actuation_contact.preserved=true`" in critic


def test_critic_allows_contact_recovery_without_regressing_milestone() -> None:
    root = Path(__file__).resolve().parents[1]
    critic = " ".join(
        (root / "prompts" / "proposal_audit_critic.txt")
        .read_text(encoding="utf-8")
        .casefold()
        .split()
    )

    assert "`actuation_contact`" in critic
    assert "`recovery_required=true`" in critic
    assert "at `milestone=actuate`" in critic
    assert "nonzero depth and gripper close" in critic
    assert "approve that bounded contact recovery" in critic


def test_critic_allows_changed_empty_handle_recovery() -> None:
    root = Path(__file__).resolve().parents[1]
    critic = " ".join(
        (root / "prompts" / "proposal_audit_critic.txt")
        .read_text(encoding="utf-8")
        .casefold()
        .split()
    )

    assert "`contact_recovery`" in critic
    assert "`required=true` and `strategy_changed=true`" in critic
    assert "approve that bounded recovery attempt" in critic
    assert "do not approve an unchanged empty-handle retry" in critic
    assert "`target_exhausted=true`" in critic
    assert "combined `|delta_u|+|delta_v|` must be at least 32 pixels" in critic
    assert "`retreat_selected=true` at `milestone=actuate`" in critic
    assert "gripper open is required for that retreat" in critic
    assert "do not return `actuation_unverified` or `revise_alignment`" in critic
    assert "open-gripper engagement rejection does not apply" in critic
    assert "`reengagement_after_retreat=true`" in critic
    assert "approve it unchanged" in critic


def test_image_servo_prompt_assigns_visual_target_to_qwen_and_geometry_to_harness() -> None:
    root = Path(__file__).resolve().parents[1]
    controller = " ".join(load_joint_system_prompt(root).casefold().split())
    critic = " ".join(
        (root / "prompts" / "proposal_audit_critic.txt")
        .read_text(encoding="utf-8")
        .casefold()
        .split()
    )

    for phrase in (
        "`image_servo`",
        "`target_pixel`",
        "`target_role`",
        "`depth_delta_m`",
        "`step_m`",
        "left or right external rgb",
        "qwen chooses",
        "sparse grid every 32 pixels",
        "origin `[0,0]` at top-left",
    ):
        assert phrase in controller
    assert "public camera calibration" in controller
    assert "up to four" in controller
    assert "`articulation_motion` uses one `step_m` increment" in controller
    assert "source_object" in controller
    assert "movable or graspable surface" in controller
    assert "positive `depth_delta_m` moves farther from the selected camera" in (
        controller
    )
    assert "insert around the handle with the gripper open" in controller
    assert "nonzero depth" in controller
    assert "gripper `close`" in controller
    assert "one bounded `cartesian_delta` lift" in controller
    assert "visible destination must use `image_servo`" in controller
    assert "`destination_receptacle`" in controller
    assert "spout, stem, basin" in critic
    assert '"target_pixel":[128.0,128.0]' not in controller
    assert "no default target pixel exists" in controller
    assert "never default to the image center or current eef pixel" in controller
    assert "original `image_servo` draft" in critic
    assert "`articulation_motion` applies one `step_m` increment" in critic
    assert "audit the selected pixel" in critic
    assert "magenta target marker" in critic
    assert "not a task object" in critic
    assert "magnified target crop" in critic
    assert "evidence-producing action" in critic
    assert "open-gripper approach" in critic
    assert "positive visual contradiction" in critic
    assert "mere uncertainty about the exact center" in critic
    assert "one 32-pixel grid cell" in critic
    assert "moved closer to the same target" in critic
    assert "sparse grid every 32 pixels" in critic
    assert "approve a bounded `base_action`" in critic
    assert "does not restore panda joint margin" in critic
    assert "do not approve another base pulse" in critic
    assert "pixel alignment is not physical engagement" in critic
    assert "open-handle insertion" in critic
    assert "stationary handle close" in critic
    assert "contact effect belongs to the next sealed receipt" in critic
    assert "one bounded vertical `cartesian_delta` lift" in critic
    assert "repeated open-loop cartesian transport" in critic
    assert "`destination_receptacle`" in critic
    assert "`actuation_unverified`" in critic
    assert "must not emit numbers" in critic


def test_proposal_critic_marks_controller_selected_pixel_without_mutating_raw_rgb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive import remote_driver

    class OverlayImage:
        def __init__(self, content: bytes, size: tuple[int, int] = (256, 256)) -> None:
            self.content = content
            self.size = size
            self.pixels: dict[tuple[int, int], tuple[int, int, int]] = {}
            self.pastes: list[tuple[int, int]] = []

        def convert(self, mode: str) -> OverlayImage:
            assert mode == "RGB"
            return self

        def copy(self) -> OverlayImage:
            return OverlayImage(self.content, self.size)

        def crop(self, box: tuple[int, int, int, int]) -> OverlayImage:
            return OverlayImage(self.content, (box[2] - box[0], box[3] - box[1]))

        def resize(self, size: tuple[int, int]) -> OverlayImage:
            return OverlayImage(self.content, size)

        def paste(self, image: OverlayImage, point: tuple[int, int]) -> None:
            self.pastes.append(point)
            if point == (0, 0):
                self.pixels.update(image.pixels)

        def putpixel(
            self, point: tuple[int, int], color: tuple[int, int, int]
        ) -> None:
            self.pixels[point] = color

        def save(self, destination: object, *, format: str) -> None:
            assert format == "PNG"
            payload = json.dumps({
                "size": self.size,
                "pixels": [
                    (*point, *color) for point, color in sorted(self.pixels.items())
                ],
                "pastes": self.pastes,
            },
                separators=(",", ":"),
            ).encode()
            destination.write(self.content + b"|" + payload)  # type: ignore[attr-defined]

    class ImageFactory:
        @staticmethod
        def open(source: object) -> OverlayImage:
            return OverlayImage(source.read())  # type: ignore[attr-defined]

        @staticmethod
        def new(
            _mode: str, size: tuple[int, int], _color: tuple[int, int, int]
        ) -> OverlayImage:
            return OverlayImage(b"composite", size)

    pil = ModuleType("PIL")
    pil.Image = ImageFactory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", pil)
    images = {
        "left": b"left-png",
        "right": b"right-png",
        "wrist": b"wrist-png",
    }
    original = dict(images)
    draft = {
        "kind": "image_servo",
        "camera": "right",
        "target_pixel": [100, 110],
    }

    first = remote_driver._proposal_critic_images(images, draft)
    second = remote_driver._proposal_critic_images(images, draft)

    assert images == original
    assert first == second
    assert first["left"] == images["left"]
    assert first["wrist"] == images["wrist"]
    assert first["right"] != images["right"]
    marked = json.loads(first["right"].split(b"|", 1)[1])
    assert marked["size"] == [512, 256]
    assert marked["pastes"] == [[0, 0], [256, 0]]
    for point in ((90, 110), (110, 110), (100, 100), (100, 120)):
        assert [*point, 255, 0, 255] in marked["pixels"]
    assert [90, 109, 255, 0, 255] in marked["pixels"]


def test_toaster_handle_gate_rejects_panel_pixel_without_authoring_replacement() -> None:
    from adaptive import remote_driver

    class SyntheticImage:
        size = (256, 256)

        @staticmethod
        def getpixel(point: tuple[int, int]) -> tuple[int, int, int]:
            x, y = point
            if x % 32 == 0:
                return (255, 255, 0)
            if y % 32 == 0:
                return (0, 255, 255)
            if 70 <= x <= 120 and 120 <= y <= 124:
                return (20, 20, 20)
            return (110, 110, 110)

    panel = remote_driver._horizontal_dark_pull_target_status(
        SyntheticImage(),
        [132.0, 100.0],
    )
    handle = remote_driver._horizontal_dark_pull_target_status(
        SyntheticImage(),
        [95.0, 122.0],
    )

    assert panel == {
        "horizontal_pull_detected": True,
        "target_on_horizontal_pull": False,
    }
    assert handle == {
        "horizontal_pull_detected": True,
        "target_on_horizontal_pull": True,
    }
    record: dict[str, object] = {}
    advice = remote_driver._protocol_rejection(
        record,
        error=ValueError(
            "toaster oven target center is outside the visible narrow "
            "horizontal handle"
        ),
    )
    assert advice == {
        "contradiction": "visual_alignment_unverified",
        "evidence": ["external_rgb"],
        "suggested_correction": "revise_alignment",
        "confidence": "high",
    }
    revision = remote_driver._proposal_revision_instruction(
        "base",
        contradiction=str(advice["contradiction"]),
        evidence=advice["evidence"],
        suggested_correction=str(advice["suggested_correction"]),
        confidence=str(advice["confidence"]),
        rejected_draft={
            "kind": "image_servo",
            "observation_id": "obs-panel",
            "camera": "right",
            "target_pixel": [132.0, 100.0],
            "target_role": "fixture_handle",
            "depth_delta_m": 0.01,
            "step_m": 0.01,
            "gripper": "open",
            "note": "milestone=engage; insert at the proposed point",
        },
        required_observation_id="obs-panel",
    ).casefold()
    assert "outside the visible narrow horizontal handle" in revision
    assert "qwen alone chooses the camera, pixel, depth, and step" in revision
    assert "do not repeat the rejected pixel" in revision


def test_proposal_critic_receives_marked_image_while_controller_keeps_raw_rgb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive import remote_driver

    class OverlayImage:
        def __init__(self, content: bytes, size: tuple[int, int] = (100, 100)) -> None:
            self.content = content
            self.size = size
            self.marked = False

        def convert(self, _mode: str) -> OverlayImage:
            return self

        def copy(self) -> OverlayImage:
            return OverlayImage(self.content, self.size)

        def crop(self, box: tuple[int, int, int, int]) -> OverlayImage:
            return OverlayImage(self.content, (box[2] - box[0], box[3] - box[1]))

        def resize(self, size: tuple[int, int]) -> OverlayImage:
            copy = OverlayImage(self.content, size)
            copy.marked = self.marked
            return copy

        def paste(self, image: OverlayImage, _point: tuple[int, int]) -> None:
            self.marked = self.marked or image.marked

        def putpixel(
            self, _point: tuple[int, int], _color: tuple[int, int, int]
        ) -> None:
            self.marked = True

        def save(self, destination: object, *, format: str) -> None:
            assert format == "PNG"
            destination.write(  # type: ignore[attr-defined]
                self.content + (b"|marked" if self.marked else b"")
            )

    class ImageFactory:
        @staticmethod
        def open(source: object) -> OverlayImage:
            return OverlayImage(source.read())  # type: ignore[attr-defined]

        @staticmethod
        def new(
            _mode: str, size: tuple[int, int], _color: tuple[int, int, int]
        ) -> OverlayImage:
            return OverlayImage(b"composite", size)

    pil = ModuleType("PIL")
    pil.Image = ImageFactory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", pil)
    context = _context()
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    command = {
        **_image_servo_payload(),
        "observation_id": "obs-0",
        "target_role": "fixture_handle",
        "note": "milestone=observe; toaster handle center in left RGB",
    }
    _FakeClient.controller_outputs = [command]
    _FakeClient.critic_outputs = [_proposal_audit()]
    state = _state()
    state.update(_image_servo_public_state())
    images = {"left": b"left", "right": b"right", "wrist": b"wrist"}

    _proposal_complete(
        wrapped,
        "obs-0",
        state,
        images,
        camera_calibration=_image_servo_calibration(),
    )

    controller_call = next(call for role, call in _FakeClient.calls if role == "controller")
    critic_call = next(call for role, call in _FakeClient.calls if role == "critic")
    assert controller_call["images"] == images
    assert critic_call["images"]["left"] == b"composite|marked"
    assert critic_call["images"]["right"] == b"right"
    assert critic_call["images"]["wrist"] == b"wrist"


def test_public_rgb_coordinate_grid_is_deterministic_and_sparse() -> None:
    from adaptive.joint_sim_child import _coordinate_grid_image

    class FakeImage:
        mode = "RGB"

        def __init__(self, pixels=None):
            self.size = (64, 64)
            self.pixels = dict(pixels or {})

        def convert(self, mode):
            assert mode == "RGB"
            return self

        def copy(self):
            return FakeImage(self.pixels)

        def putpixel(self, point, color):
            self.pixels[point] = color

        def getpixel(self, point):
            return self.pixels.get(point, (0, 0, 0))

        def tobytes(self):
            return repr(sorted(self.pixels.items())).encode()

    source = FakeImage()
    first = _coordinate_grid_image(source)
    second = _coordinate_grid_image(source)

    assert first.size == source.size
    assert first.mode == "RGB"
    assert first.tobytes() == second.tobytes()
    assert first.getpixel((32, 20)) != (0, 0, 0)
    assert first.getpixel((20, 32)) != (0, 0, 0)
    assert first.getpixel((20, 20)) == (0, 0, 0)


def test_image_servo_pixel_progress_exposes_public_receipt_math() -> None:
    from adaptive.remote_driver import _image_servo_pixel_progress

    draft = {
        "kind": "image_servo",
        "camera": "left",
        "target_pixel": [150.0, 140.0],
        "target_role": "control_target",
    }
    receipt = {
        "kind": "image_servo",
        "requested_camera": "left",
        "requested_target_pixel": [150.0, 140.0],
        "requested_target_role": "control_target",
        "current_end_effector_pixel": [196.8904, 59.9454],
        "end_effector_external_pixel_displacement": {
            "left": {
                "start_px": [196.8904, 59.9454],
                "end_px": [193.0059, 63.0644],
            }
        },
    }

    progress = _image_servo_pixel_progress(draft, receipt)

    assert progress["comparable"] is True
    assert progress["same_camera_target_role"] is True
    assert progress["moved_closer"] is True
    assert progress["start_error_px"] == pytest.approx(92.7763, abs=1e-3)
    assert progress["end_error_px"] == pytest.approx(88.1396, abs=1e-3)
    assert progress["error_reduction_px"] == pytest.approx(4.6367, abs=1e-3)

    changed = dict(draft, target_pixel=[160.0, 140.0])
    incomparable = _image_servo_pixel_progress(changed, receipt)
    assert incomparable == {
        "schema": "robocasa-image-servo-pixel-progress/v1",
        "comparable": False,
        "same_camera_target_role": False,
        "start_error_px": None,
        "end_error_px": None,
        "error_reduction_px": None,
        "moved_closer": None,
    }


def test_proposal_critic_receives_closed_public_alignment_gate() -> None:
    from adaptive.remote_driver import _proposal_critic_instruction

    receipt = {
        "kind": "image_servo",
        "requested_camera": "left",
        "requested_target_pixel": [128.0, 128.0],
        "end_effector_external_pixel_displacement": {
            "left": {"end_px": [127.1774, 125.3149]}
        },
    }
    payload = json.loads(
        _proposal_critic_instruction(
            _context(),
            task_instruction="adjust the water temperature",
            observation_id="obs-14",
            draft=_cartesian_milestone_command("obs-14", "actuate", 0.0),
            claimed_milestone="actuate",
            public_state=_state(),
            images=_images(),
            immediate_prior_receipt=receipt,
        )
    )

    assert payload["image_servo_alignment"] == {
        "schema": "robocasa-image-servo-alignment-status/v1",
        "camera": "left",
        "target_pixel": [128.0, 128.0],
        "end_effector_pixel": [127.1774, 125.3149],
        "error_px": pytest.approx(2.8082, abs=1e-3),
        "within_one_grid_cell": True,
        "controller_rule": (
            "alignment gate is satisfied: advance now from approach to engage "
            "or pregrasp; do not chase exact pixel equality"
        ),
    }


def test_safe_open_cartesian_approach_is_explicit_in_critic_request() -> None:
    from adaptive.remote_driver import _proposal_critic_instruction

    state = _state()
    state.update(_cartesian_identity_public_state())
    state["state.arm_joint_velocity"] = [0.0] * 7
    draft = _cartesian_milestone_command("obs-view", "approach", 0.025)
    draft["gripper"] = "open"
    payload = json.loads(
        _proposal_critic_instruction(
            _context(),
            task_instruction="open the toaster oven door",
            observation_id="obs-view",
            draft=draft,
            claimed_milestone="approach",
            public_state=state,
            images=_images(),
            immediate_prior_receipt=None,
        )
    )

    assert payload["cartesian_view_gathering"] == {
        "applicable": True,
        "protocol_validated": True,
        "nonzero": True,
        "gripper_open": True,
        "tracking_settled": True,
    }


def test_controller_instruction_exposes_image_servo_alignment_gate() -> None:
    from adaptive.joint_runner import _image_servo_alignment_status, _instruction

    receipt = {
        "kind": "image_servo",
        "requested_camera": "left",
        "requested_target_pixel": [150.0, 140.0],
        "end_effector_external_pixel_displacement": {
            "left": {"end_px": [155.0, 150.0]}
        },
    }

    assert _image_servo_alignment_status([receipt]) == {
        "schema": "robocasa-image-servo-alignment-status/v1",
        "camera": "left",
        "target_pixel": [150.0, 140.0],
        "end_effector_pixel": [155.0, 150.0],
        "error_px": pytest.approx(11.1803, abs=1e-3),
        "within_one_grid_cell": True,
        "controller_rule": (
            "alignment gate is satisfied: advance now from approach to engage "
            "or pregrasp; do not chase exact pixel equality"
        ),
    }
    assert _image_servo_alignment_status([]) is None

    initial = json.loads(
        _instruction(
            {"instruction": "open the toaster", "public_state": {}},
            receipts=[],
            change={"left": 0.0, "right": 0.0, "wrist": 0.0},
            repair=None,
        )
    )
    assert "image_servo_alignment" not in initial

    receipt["mean_absolute_rgb_change"] = {
        "left": 1.0,
        "right": 1.0,
        "wrist": 1.0,
    }
    after_servo = json.loads(
        _instruction(
            {"instruction": "open the toaster", "public_state": {}},
            receipts=[receipt],
            change={"left": 1.0, "right": 1.0, "wrist": 1.0},
            repair=None,
        )
    )
    assert after_servo["image_servo_alignment"]["within_one_grid_cell"] is True


def test_proposal_approval_audits_same_observation_and_returns_draft_unchanged() -> None:
    from adaptive import remote_driver

    context = _context()
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    draft = _milestone_command("obs-0", "observe")
    state = _state()
    images = _images()
    _FakeClient.controller_outputs = [draft]
    _FakeClient.critic_outputs = [_proposal_audit()]

    response = _proposal_complete(wrapped, "obs-0", state, images)

    assert response.command is draft
    assert [role for role, _ in _FakeClient.calls] == ["controller", "critic"]
    critic_call = _FakeClient.calls[1][1]
    controller_schema = _FakeClient.calls[0][1]["response_schema"]
    assert all(
        branch["properties"]["observation_id"] == {"const": "obs-0"}
        for branch in controller_schema["oneOf"]
    )
    assert critic_call["observation_id"] == "obs-0"
    assert critic_call["public_state"] is state
    assert critic_call["images"] is images
    payload = json.loads(str(critic_call["instruction"]))
    assert payload["draft"] == draft
    assert payload["task_instruction"] == "open the toaster"
    assert payload["fresh_observation_id"] == "obs-0"
    assert payload["immediate_prior_receipt"] is None
    assert payload["image_servo_pixel_progress"] is None
    assert payload["milestone_history"] == []
    assert payload["fresh_public_rgb_sha256"] == {
        name: hashlib.sha256(value).hexdigest() for name, value in images.items()
    }
    record = context.proposal_records[0]
    assert record["status"] == "approved_for_execution"
    assert record["draft_sha256"] == _json_sha256(draft)
    assert record["returned_command_sha256"] == _json_sha256(draft)
    assert record["executed"] is False
    assert record["mailbox_count"] == 0
    assert record["action_count"] == 0
    assert record["receipt_count"] == 0
    assert response.evidence["controller_role"] == (
        "sole_direct_inspect_command_emitter"
    )
    assert response.evidence["proposal_audit_verdict"] == "approve"


def test_proposal_approval_returns_cartesian_draft_unchanged() -> None:
    from adaptive import remote_driver

    context = _context()
    context.milestone_history = ["observe"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    draft = _cartesian_milestone_command("obs-0", "approach")
    state = _state()
    state.update(_cartesian_identity_public_state())
    _FakeClient.controller_outputs = [draft]
    _FakeClient.critic_outputs = [_proposal_audit()]

    response = _proposal_complete(wrapped, "obs-0", state, _images())

    assert response.command is draft
    assert [role for role, _ in _FakeClient.calls] == ["controller", "critic"]
    assert json.loads(str(_FakeClient.calls[1][1]["instruction"]))["draft"] == draft
    record = context.proposal_records[0]
    assert record["status"] == "approved_for_execution"
    assert record["draft_sha256"] == _json_sha256(draft)
    assert record["returned_command_sha256"] == _json_sha256(draft)


def test_rejected_cartesian_proposal_has_zero_effect_before_changed_revision() -> None:
    from adaptive import remote_driver

    context = _context()
    context.milestone_history = ["observe"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    rejected = _cartesian_milestone_command("obs-0", "approach", 0.01)
    revised = _cartesian_milestone_command("obs-0", "approach", 0.02)
    state = _state()
    state.update(_cartesian_identity_public_state())
    _FakeClient.controller_outputs = [rejected, revised]
    _FakeClient.critic_outputs = [_proposal_audit("revise"), _proposal_audit()]

    response = _proposal_complete(wrapped, "obs-0", state, _images())

    assert response.command is revised
    assert [role for role, _ in _FakeClient.calls] == [
        "controller",
        "critic",
        "controller",
        "critic",
    ]
    first, second = context.proposal_records
    assert first["status"] == "rejected_by_critic"
    assert first["executed"] is False
    assert (first["mailbox_count"], first["action_count"], first["receipt_count"]) == (
        0,
        0,
        0,
    )
    assert second["status"] == "approved_for_execution"


def test_rejected_proposal_has_zero_effect_then_same_observation_revision_is_audited() -> None:
    from adaptive import remote_driver

    context = _context()
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    first = _milestone_command("obs-0", "observe", 0.01)
    revised = _milestone_command("obs-0", "observe", 0.02)
    state = _state()
    images = _images()
    _FakeClient.controller_outputs = [first, revised]
    _FakeClient.critic_outputs = [_proposal_audit("revise"), _proposal_audit()]

    response = _proposal_complete(wrapped, "obs-0", state, images)

    assert response.command is revised
    assert [role for role, _ in _FakeClient.calls] == [
        "controller",
        "critic",
        "controller",
        "critic",
    ]
    assert all(call["observation_id"] == "obs-0" for _, call in _FakeClient.calls)
    assert all(
        all(
            branch["properties"]["observation_id"] == {"const": "obs-0"}
            for branch in call["response_schema"]["oneOf"]
        )
        for role, call in _FakeClient.calls
        if role == "controller"
    )
    assert all(call["public_state"] is state for _, call in _FakeClient.calls)
    assert all(call["images"] is images for _, call in _FakeClient.calls)
    assert "PROPOSAL_AUDIT_REVISION" in str(_FakeClient.calls[2][1]["instruction"])
    rejected, approved = context.proposal_records
    assert rejected["status"] == "rejected_by_critic"
    assert rejected["executed"] is False
    assert (rejected["mailbox_count"], rejected["action_count"], rejected["receipt_count"]) == (0, 0, 0)
    assert approved["status"] == "approved_for_execution"
    assert context.proposal_revisions_used == 1


def test_motion_mismatch_revision_preserves_motor_payload_and_corrects_note() -> None:
    from adaptive import remote_driver

    rejected_draft = _milestone_command("obs-0", "approach", 0.01)
    instruction = remote_driver._proposal_revision_instruction(
        "base instruction",
        contradiction="motion_effect_mismatch",
        evidence=["joint_velocity", "gripper_state"],
        suggested_correction="revise_approach",
        confidence="high",
        rejected_draft=rejected_draft,
        required_observation_id="obs-0",
    )
    normalized = instruction.casefold()
    revision_payload = json.loads(
        instruction.split("PROPOSAL_AUDIT_REVISION:\n", 1)[1].splitlines()[0]
    )

    assert revision_payload["rejected_draft"] == rejected_draft
    assert revision_payload["rejected_draft_sha256"] == _json_sha256(rejected_draft)
    assert revision_payload["required_observation_id"] == "obs-0"
    assert "copy `required_observation_id` exactly" in normalized
    assert "do not append to or alter it" in normalized
    assert "preserve the command kind and every motor-bearing field exactly" in (
        normalized
    )
    assert "only correct the note" in normalized
    assert "remove every stale claim contradicted by the audit" in normalized
    assert "note must describe the new command actually authored" in normalized
    assert "qwen alone remains the author of the preserved motor payload" in normalized
    assert "qwen alone decides whether to change the note or numeric values" not in (
        normalized
    )


def test_visual_alignment_revision_gathers_a_new_view() -> None:
    from adaptive import remote_driver

    instruction = remote_driver._proposal_revision_instruction(
        "base instruction",
        contradiction="visual_alignment_unverified",
        evidence=["external_rgb"],
        suggested_correction="revise_alignment",
        confidence="high",
        rejected_draft={
            "kind": "image_servo",
            "camera": "left",
            "target_pixel": [128.0, 128.0],
            "target_role": "control_target",
        },
        required_observation_id="obs-0",
    ).casefold()

    assert "must author a nonzero `base_action` or `cartesian_delta`" in instruction
    assert "view-gathering move" in instruction
    assert "do not guess another pixel" in instruction
    assert "do not author another `image_servo`" in instruction


def test_visual_alignment_cartesian_revision_returns_to_image_servo() -> None:
    from adaptive import remote_driver

    instruction = remote_driver._proposal_revision_instruction(
        "base instruction",
        contradiction="visual_alignment_unverified",
        evidence=["external_rgb"],
        suggested_correction="revise_alignment",
        confidence="high",
        rejected_draft={
            "kind": "cartesian_delta",
            "translation_m": [0.02, -0.015, 0.01],
            "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
            "gripper": "open",
        },
        required_observation_id="obs-0",
    ).casefold()

    assert "must author an `image_servo`" in instruction
    assert "do not author another `cartesian_delta`" in instruction


def test_parallel_handle_gripper_revision_requires_open_wrist_roll() -> None:
    from adaptive import remote_driver

    instruction = remote_driver._proposal_revision_instruction(
        "base instruction",
        contradiction="visual_alignment_unverified",
        evidence=["wrist_rgb", "external_rgb"],
        suggested_correction="revise_alignment",
        confidence="high",
        rejected_draft={
            "kind": "image_servo",
            "camera": "right",
            "target_pixel": [128.0, 128.0],
            "target_role": "fixture_handle",
            "depth_delta_m": 0.01,
            "step_m": 0.01,
            "gripper": "close",
            "note": "milestone=engage; close on the horizontal handle",
        },
        required_observation_id="obs-parallel-handle",
    ).casefold()

    assert "remain at `milestone=approach`" in instruction
    assert "author a `move_joints` command" in instruction
    assert "changes `joint7`" in instruction
    assert "gripper target `1.0`" in instruction
    assert "finger gap is perpendicular to the elongated handle" in instruction
    assert "do not close again on this frozen observation" in instruction


def test_simultaneous_handle_close_is_revised_before_critic() -> None:
    from adaptive import remote_driver

    context = _context()
    context.milestone_history = ["observe", "approach"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    rejected_close = {
        **_image_servo_payload(),
        "observation_id": "obs-parallel-handle",
        "target_role": "fixture_handle",
        "gripper": "close",
        "note": "milestone=engage; close on the horizontal toaster handle",
    }
    open_insertion = {
        **_image_servo_payload(),
        "observation_id": "obs-parallel-handle",
        "target_role": "fixture_handle",
        "depth_delta_m": 0.01,
        "gripper": "open",
        "note": "milestone=engage; insert open fingers around the handle",
    }
    _FakeClient.controller_outputs = [rejected_close, open_insertion]
    _FakeClient.critic_outputs = [_proposal_audit()]
    state = _state()
    state.update(_image_servo_public_state())

    response = _proposal_complete(
        wrapped,
        "obs-parallel-handle",
        state,
        _images(),
        camera_calibration=_image_servo_calibration(),
    )

    assert response.command is open_insertion
    assert [role for role, _ in _FakeClient.calls] == [
        "controller",
        "controller",
        "critic",
    ]
    assert context.proposal_records[0]["status"] == "rejected_by_protocol"
    payload = json.loads(
        str([call for role, call in _FakeClient.calls if role == "critic"][0]["instruction"])
    )
    assert payload["open_handle_insertion_rule"]["default_verdict"] == "approve"
    second_controller_call = [
        call for role, call in _FakeClient.calls if role == "controller"
    ][1]
    revision_schema = second_controller_call["response_schema"]
    assert "oneOf" in revision_schema
    revision_instruction = str(second_controller_call["instruction"]).casefold()
    assert "do not close while translating" in revision_instruction
    assert "gripper `open`" in revision_instruction


def test_quarter_turn_closes_only_the_handle_orientation_subgate() -> None:
    from adaptive import remote_driver

    context = _context()
    context.articulated_wrist_roll_origin_joint7 = 0.2
    draft = {
        **_image_servo_payload(),
        "observation_id": "obs-quarter-turn",
        "target_role": "fixture_handle",
        "gripper": "close",
        "note": "milestone=engage; close around the visible handle",
    }
    state = _state()
    state["state.arm_joint_position"] = [
        0.0,
        -1.0,
        0.0,
        -2.2,
        0.0,
        1.5,
        0.2 + math.pi / 2.0,
    ]

    payload = json.loads(
        remote_driver._proposal_critic_instruction(
            context,
            task_instruction="open the toaster oven door",
            observation_id="obs-quarter-turn",
            draft=draft,
            claimed_milestone="engage",
            public_state=state,
            images=_images(),
            immediate_prior_receipt=None,
        )
    )

    assert payload["schema"] == "robocasa-qwen-proposal-audit-request/v14"
    assert payload["wrist_roll_progress"] == {
        "absolute_delta_rad": pytest.approx(math.pi / 2.0),
        "current_joint7_rad": pytest.approx(0.2 + math.pi / 2.0),
        "origin_joint7_rad": 0.2,
        "quarter_turn_reached": True,
    }
    assert payload["wrist_orientation_rule"] == {
        "default_verdict": "approve",
        "orientation_only_rejection_prohibited": True,
        "quarter_turn_from_parallel_rejection": True,
    }


def test_close_waits_until_current_eef_pixel_is_near_selected_handle() -> None:
    from adaptive import remote_driver

    state = _state()
    state.update(_image_servo_public_state())
    draft = {
        **_image_servo_payload(),
        "observation_id": "obs-far-handle",
        "camera": "right",
        "target_pixel": [90.0, 90.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.015,
        "gripper": "close",
        "note": "milestone=engage; close around the selected handle",
    }

    payload = json.loads(
        remote_driver._proposal_critic_instruction(
            _context(),
            task_instruction="open the toaster oven door",
            observation_id="obs-far-handle",
            draft=draft,
            claimed_milestone="engage",
            public_state=state,
            images=_images(),
            immediate_prior_receipt=None,
        )
    )

    assert payload["schema"] == "robocasa-qwen-proposal-audit-request/v14"
    assert payload["draft_image_servo_alignment"] == {
        "camera": "right",
        "current_end_effector_pixel": [50.0, 50.0],
        "error_px": pytest.approx(math.sqrt(3200.0)),
        "schema": "robocasa-draft-image-servo-alignment/v1",
        "target_pixel": [90.0, 90.0],
        "within_close_tolerance": False,
        "within_one_grid_cell": False,
    }
    assert payload["image_servo_close_rule"] == {
        "close_pixel_tolerance_px": 8.0,
        "close_requires_within_tolerance": True,
        "current_within_tolerance": False,
        "default_verdict": "revise",
        "required_contradiction": "visual_alignment_unverified",
        "required_correction": "revise_alignment",
        "required_evidence": ["external_rgb", "jacobian_projection"],
    }
    revision = remote_driver._proposal_revision_instruction(
        "base instruction",
        contradiction="visual_alignment_unverified",
        evidence=["external_rgb", "jacobian_projection"],
        suggested_correction="revise_alignment",
        confidence="high",
        rejected_draft=draft,
        required_observation_id="obs-far-handle",
    ).casefold()
    assert "remain at `milestone=approach`" in revision
    assert "author an `image_servo`" in revision
    assert "keep the gripper `open`" in revision
    assert "outside the required 8-pixel contact tolerance" in revision
    assert "do not close or roll `joint7`" in revision
    assert "replacement note must describe this open alignment command" in revision
    assert "chosen depth sign" in revision

    open_draft = {
        **draft,
        "gripper": "open",
        "depth_delta_m": 0.0,
        "note": "milestone=approach; align the open gripper laterally to the handle",
    }
    open_payload = json.loads(
        remote_driver._proposal_critic_instruction(
            _context(),
            task_instruction="open the toaster oven door",
            observation_id="obs-far-handle",
            draft=open_draft,
            claimed_milestone="approach",
            public_state=state,
            images=_images(),
            immediate_prior_receipt=None,
        )
    )
    assert open_payload["image_servo_close_rule"] is None
    assert open_payload["image_servo_open_approach_rule"] == {
        "alignment_effect_is_future": True,
        "current_outside_close_tolerance": True,
        "default_verdict": "approve",
        "far_current_pixel_is_reason_to_execute": True,
        "zero_depth_can_still_have_pixel_alignment_effect": True,
    }


def test_close_requires_subgrid_pixel_precision_before_handle_contact() -> None:
    from adaptive import remote_driver

    state = _state()
    state.update(_image_servo_public_state())
    draft = {
        **_image_servo_payload(),
        "observation_id": "obs-subgrid-handle",
        "camera": "right",
        "target_pixel": [60.0, 60.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.01,
        "gripper": "close",
        "note": "milestone=engage; close around the selected handle",
    }

    payload = json.loads(
        remote_driver._proposal_critic_instruction(
            _context(),
            task_instruction="open the toaster oven door",
            observation_id="obs-subgrid-handle",
            draft=draft,
            claimed_milestone="engage",
            public_state=state,
            images=_images(),
            immediate_prior_receipt=None,
        )
    )

    assert payload["draft_image_servo_alignment"]["within_one_grid_cell"] is True
    assert payload["draft_image_servo_alignment"]["within_close_tolerance"] is False
    assert payload["image_servo_close_rule"] == {
        "close_pixel_tolerance_px": 8.0,
        "close_requires_within_tolerance": True,
        "current_within_tolerance": False,
        "default_verdict": "revise",
        "required_contradiction": "visual_alignment_unverified",
        "required_correction": "revise_alignment",
        "required_evidence": ["external_rgb", "jacobian_projection"],
    }
    assert payload["handle_orientation_calibration_rule"] is None

    open_payload = json.loads(
        remote_driver._proposal_critic_instruction(
            _context(),
            task_instruction="open the toaster oven door",
            observation_id="obs-subgrid-handle",
            draft={
                **draft,
                "gripper": "open",
                "depth_delta_m": 0.0,
                "note": "milestone=approach; align open gripper to the handle",
            },
            claimed_milestone="approach",
            public_state=state,
            images=_images(),
            immediate_prior_receipt=None,
        )
    )
    assert open_payload["image_servo_open_approach_rule"] == {
        "alignment_effect_is_future": True,
        "current_outside_close_tolerance": True,
        "default_verdict": "approve",
        "far_current_pixel_is_reason_to_execute": True,
        "zero_depth_can_still_have_pixel_alignment_effect": True,
    }

def test_handle_open_insertion_rejects_camera_retreat_depth() -> None:
    from adaptive import remote_driver

    state = _state()
    state.update(_image_servo_public_state())
    draft = {
        **_image_servo_payload(),
        "observation_id": "obs-near-handle",
        "target_pixel": [56.0, 50.0],
        "target_role": "fixture_handle",
        "depth_delta_m": -0.01,
        "gripper": "open",
        "note": "milestone=engage; insert while retreating toward the camera",
    }
    payload = json.loads(
        remote_driver._proposal_critic_instruction(
            _context(),
            task_instruction="open the toaster oven door",
            observation_id="obs-near-handle",
            draft=draft,
            claimed_milestone="engage",
            public_state=state,
            images=_images(),
            immediate_prior_receipt=None,
        )
    )

    assert payload["image_servo_close_rule"] is None
    assert payload["image_servo_contact_depth_rule"] == {
        "contact_requires_positive_depth": True,
        "current_depth_delta_m": -0.01,
        "default_verdict": "revise",
        "negative_depth_is_camera_retreat": True,
        "required_contradiction": "motion_effect_mismatch",
        "required_correction": "revise_approach",
    }

    revision = remote_driver._proposal_revision_instruction(
        "base instruction",
        contradiction="motion_effect_mismatch",
        evidence=["external_rgb", "jacobian_projection"],
        suggested_correction="revise_approach",
        confidence="high",
        rejected_draft=draft,
        required_observation_id="obs-near-handle",
    ).casefold()
    assert "negative `depth_delta_m` retreats toward the selected camera" in revision
    assert "author positive depth with gripper `open`" in revision
    assert "qwen alone chooses its magnitude" in revision
    assert "preserve the command kind and every motor-bearing field exactly" not in (
        revision
    )
    assert "before the stationary close" in revision


def test_aligned_handle_close_does_not_force_refuted_quarter_turn() -> None:
    from adaptive import remote_driver

    state = _state()
    state.update(_image_servo_public_state())
    draft = {
        **_image_servo_payload(),
        "observation_id": "obs-aligned-handle",
        "target_pixel": [56.0, 50.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.01,
        "gripper": "close",
        "note": "milestone=engage; insert open fingers around the aligned handle",
    }
    context = _context()

    initial = json.loads(
        remote_driver._proposal_critic_instruction(
            context,
            task_instruction="open the toaster oven door",
            observation_id="obs-aligned-handle",
            draft=draft,
            claimed_milestone="engage",
            public_state=state,
            images=_images(),
            immediate_prior_receipt=None,
        )
    )
    assert initial["handle_orientation_calibration_rule"] is None

    context.articulated_wrist_roll_origin_joint7 = 0.2
    state["state.arm_joint_position"][6] = 0.7
    partial = json.loads(
        remote_driver._proposal_critic_instruction(
            context,
            task_instruction="open the toaster oven door",
            observation_id="obs-partial-roll",
            draft={**draft, "observation_id": "obs-partial-roll"},
            claimed_milestone="engage",
            public_state=state,
            images=_images(),
            immediate_prior_receipt=None,
        )
    )
    assert partial["handle_orientation_calibration_rule"] is None


def test_prompts_insert_open_gripper_before_stationary_handle_close() -> None:
    root = Path(__file__).resolve().parents[1]
    controller = " ".join(
        (root / "prompts" / "joint_system.txt").read_text().split()
    ).casefold()
    critic = " ".join(
        (root / "prompts" / "proposal_audit_critic.txt").read_text().split()
    ).casefold()

    assert "insert around the handle with the gripper open" in controller
    assert "close without simultaneous arm motion" in controller
    assert "repeat close-only once to test persistence" in controller
    assert "open-handle insertion" in critic
    assert "stationary handle close" in critic
    assert "repeat the stationary close once" in critic
    assert "interior midline of the handle material" in controller
    assert "not its top or bottom boundary" in controller
    assert "inside the handle material near its mid-thickness" in critic
    assert "door panel, gap, outline, or end cap" in critic
    assert "wider target crop" in critic
    assert "reject a marker on the broad flat door slab" in critic
    assert "image_servo_close_rule" in critic
    assert "more than the stated 8-pixel close tolerance" in critic
    assert "image_servo_open_approach_rule" in critic
    assert "outside the 8-pixel close tolerance is the reason" in critic
    assert "image_servo_contact_depth_rule" in critic
    assert "open insertion requires positive depth" in critic


def test_articulated_engage_inserts_open_then_closes_stationary() -> None:
    from adaptive.critic_protocol import validate_milestone_action_semantics

    aligned_approach = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [112.0, 128.0],
        "requested_target_role": "fixture_handle",
        "requested_depth_delta_m": -0.01,
        "requested_gripper": "open",
    }
    moving_close = {
        "kind": "image_servo",
        "camera": "right",
        "target_pixel": [112.0, 128.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.01,
        "step_m": 0.01,
        "gripper": "close",
        "note": "milestone=engage; close while advancing",
    }
    with pytest.raises(ValueError, match="open insertion before stationary close"):
        validate_milestone_action_semantics(
            "articulated",
            "engage",
            moving_close,
            immediate_prior_receipt=aligned_approach,
            recent_receipts=[aligned_approach],
        )

    insertion = {**moving_close, "gripper": "open"}
    validate_milestone_action_semantics(
        "articulated",
        "engage",
        insertion,
        immediate_prior_receipt=aligned_approach,
        recent_receipts=[aligned_approach],
    )
    insertion_receipt = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [112.0, 128.0],
        "requested_target_role": "fixture_handle",
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "open",
        "end_effector_external_pixel_displacement": {
            "right": {"end_px": [112.0, 128.0]}
        },
    }
    with pytest.raises(ValueError, match="stationary gripper close"):
        validate_milestone_action_semantics(
            "articulated",
            "engage",
            insertion,
            immediate_prior_receipt=insertion_receipt,
            recent_receipts=[aligned_approach, insertion_receipt],
        )
    validate_milestone_action_semantics(
        "articulated",
        "engage",
        {
            "kind": "move_joints",
            "targets": {"gripper": 0.0},
            "note": "milestone=engage; close without moving the arm",
        },
        immediate_prior_receipt=insertion_receipt,
        recent_receipts=[aligned_approach, insertion_receipt],
    )

    far_insertion_receipt = {
        **insertion_receipt,
        "end_effector_external_pixel_displacement": {
            "right": {"end_px": [70.0, 60.0]}
        },
    }
    validate_milestone_action_semantics(
        "articulated",
        "engage",
        insertion,
        immediate_prior_receipt=far_insertion_receipt,
        recent_receipts=[aligned_approach, far_insertion_receipt],
    )
    with pytest.raises(ValueError, match="open insertion before stationary close"):
        validate_milestone_action_semantics(
            "articulated",
            "engage",
            {
                "kind": "move_joints",
                "targets": {"gripper": 0.0},
                "note": "milestone=engage; close too far from the handle",
            },
            immediate_prior_receipt=far_insertion_receipt,
            recent_receipts=[aligned_approach, far_insertion_receipt],
        )


def test_critic_marks_open_insertion_and_stationary_close_as_executable() -> None:
    from adaptive import remote_driver

    state = _state()
    state.update(_image_servo_public_state())
    state["state.end_effector_external_pixels"]["right"].update(
        {"u_px": 110.0, "v_px": 126.0}
    )
    insertion = {
        **_image_servo_payload(),
        "observation_id": "obs-insert",
        "camera": "right",
        "target_pixel": [112.0, 128.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.01,
        "step_m": 0.01,
        "gripper": "open",
        "note": "milestone=engage; insert open fingers around the handle",
    }
    insertion_payload = json.loads(
        remote_driver._proposal_critic_instruction(
            _context(),
            task_instruction="open the toaster oven door",
            observation_id="obs-insert",
            draft=insertion,
            claimed_milestone="engage",
            public_state=state,
            images=_images(),
            immediate_prior_receipt=None,
        )
    )
    assert insertion_payload["open_handle_insertion_rule"] == {
        "arm_motion_precedes_close": True,
        "default_verdict": "approve",
        "gripper_must_remain_open": True,
        "positive_depth_required": True,
    }

    insertion_receipt = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [112.0, 128.0],
        "requested_target_role": "fixture_handle",
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "open",
        "end_effector_external_pixel_displacement": {
            "right": {"end_px": [112.0, 128.0]}
        },
    }
    close_payload = json.loads(
        remote_driver._proposal_critic_instruction(
            _context(),
            task_instruction="open the toaster oven door",
            observation_id="obs-close",
            draft={
                "kind": "move_joints",
                "observation_id": "obs-close",
                "targets": {"gripper": 0.0},
                "note": "milestone=engage; close around the inserted handle",
            },
            claimed_milestone="engage",
            public_state=state,
            images=_images(),
            immediate_prior_receipt=insertion_receipt,
        )
    )
    assert close_payload["stationary_handle_close_rule"] == {
        "arm_motion_prohibited": True,
        "contact_effect_is_future_evidence": True,
        "default_verdict": "approve",
        "gripper_close_only": True,
    }


def test_aligned_positive_source_close_has_mandatory_execution_rule() -> None:
    from adaptive import remote_driver

    state = _state()
    state.update(_image_servo_public_state())
    state["state.end_effector_external_pixels"]["left"].update(
        {"u_px": 124.0, "v_px": 123.0}
    )
    payload = json.loads(
        remote_driver._proposal_critic_instruction(
            _context(),
            task_instruction="pick up the mug",
            observation_id="obs-aligned-positive-close",
            draft={
                **_image_servo_payload(),
                "observation_id": "obs-aligned-positive-close",
                "target_pixel": [128.0, 128.0],
                "target_role": "source_object",
                "depth_delta_m": 0.01,
                "gripper": "close",
                "note": "milestone=grasp; execute one bounded close on the mug",
            },
            claimed_milestone="grasp",
            public_state=state,
            images=_images(),
            immediate_prior_receipt=None,
        )
    )

    assert payload["image_servo_close_rule"] is None
    assert payload["image_servo_contact_depth_rule"] is None
    assert payload["aligned_contact_execution_rule"] == {
        "contact_effect_is_future_evidence": True,
        "current_within_close_tolerance": True,
        "default_verdict": "approve",
        "positive_contact_depth": True,
        "wrist_ambiguity_is_not_a_contradiction": True,
    }


def test_visual_alignment_cannot_undo_sealed_actuation_contact() -> None:
    from adaptive import remote_driver

    receipt = {
        "kind": "image_servo",
        "requested_gripper": "close",
        "gripper_residual": {"measured_end_finger_separation": 0.006},
    }
    draft = {
        "kind": "cartesian_delta",
        "observation_id": "obs-contact-actuation",
        "translation_m": [0.0, 0.0, -0.02],
        "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        "gripper": "hold",
        "note": "milestone=actuate; pull the contacted handle",
    }
    instruction = remote_driver._proposal_revision_instruction(
        "base instruction",
        contradiction="visual_alignment_unverified",
        evidence=["external_rgb", "gripper_state"],
        suggested_correction="revise_alignment",
        confidence="high",
        rejected_draft=draft,
        required_observation_id="obs-contact-actuation",
        immediate_prior_receipt=receipt,
    ).casefold()

    assert "sealed contact closes the visual-alignment gate" in instruction
    assert "do not return to `image_servo`" in instruction
    assert "change the cartesian actuation direction" in instruction
    assert "must author an `image_servo`" not in instruction

    payload = json.loads(
        remote_driver._proposal_critic_instruction(
            _context(),
            task_instruction="open the toaster oven door",
            observation_id="obs-contact-actuation",
            draft=draft,
            claimed_milestone="actuate",
            public_state=_state(),
            images=_images(),
            immediate_prior_receipt=receipt,
        )
    )
    assert payload["binding_audit_rule"] == {
        "visual_alignment_closed_by_contact": True,
        "visual_alignment_unverified_prohibited": True,
        "first_actuation_effect_is_future_evidence": True,
        "default_verdict": "approve",
    }


def test_safety_bound_revision_retreats_arm_instead_of_looping_base() -> None:
    from adaptive import remote_driver

    instruction = remote_driver._proposal_revision_instruction(
        "base instruction",
        contradiction="safety_bound_risk",
        evidence=["joint_tracking"],
        suggested_correction="revise_within_bounds",
        confidence="high",
        rejected_draft={
            "kind": "image_servo",
            "camera": "left",
            "target_pixel": [128.0, 160.0],
            "target_role": "control_target",
        },
        required_observation_id="obs-0",
    ).casefold()

    assert "do not repeat `image_servo`" in instruction
    assert "must author a bounded `cartesian_delta` retreat" in instruction
    assert "do not use `base_action`" in instruction
    assert "reducing only `step_m`" in instruction


def test_zero_cartesian_protocol_rejection_requests_nonzero_progress() -> None:
    from adaptive import remote_driver

    advice = remote_driver._protocol_rejection(
        {}, error=ValueError("zero Cartesian delta has no gripper transition")
    )
    assert advice == {
        "contradiction": "stagnation",
        "evidence": ["milestone_history"],
        "suggested_correction": "revise_approach",
        "confidence": "high",
    }

    instruction = remote_driver._proposal_revision_instruction(
        "base instruction",
        contradiction="stagnation",
        evidence=["milestone_history"],
        suggested_correction="revise_approach",
        confidence="high",
        rejected_draft={
            "kind": "cartesian_delta",
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
            "gripper": "open",
        },
        required_observation_id="obs-0",
    ).casefold()
    assert "must create a nonzero motor effect" in instruction
    assert "do not wait for settle when public joint velocity is already near zero" in (
        instruction
    )


def test_unverified_actuation_revision_requires_contact_producing_servo() -> None:
    from adaptive import remote_driver

    instruction = remote_driver._proposal_revision_instruction(
        "base instruction",
        contradiction="actuation_unverified",
        evidence=["action_receipt", "end_effector_wrench"],
        suggested_correction="verify_actuation",
        confidence="high",
        rejected_draft={
            "kind": "cartesian_delta",
            "translation_m": [0.03, 0.0, 0.0],
            "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
            "gripper": "open",
        },
        required_observation_id="obs-0",
    ).casefold()

    assert "must author an `image_servo` with nonzero `depth_delta_m`" in instruction
    assert "use gripper `close` for a visibly graspable handle or knob" in instruction
    assert "do not author another free-space actuation" in instruction


def test_unverified_closed_handle_actuation_changes_cartesian_direction() -> None:
    from adaptive import remote_driver

    instruction = remote_driver._proposal_revision_instruction(
        "base instruction",
        contradiction="actuation_unverified",
        evidence=["action_receipt", "external_rgb"],
        suggested_correction="verify_actuation",
        confidence="high",
        rejected_draft={
            "kind": "cartesian_delta",
            "translation_m": [0.0, 0.0, -0.02],
            "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
            "gripper": "hold",
            "note": (
                "milestone=actuate; the aligned closed gripper pulled down but "
                "the next public receipt did not show the door opening"
            ),
        },
        required_observation_id="obs-0",
    ).casefold()

    assert "remain at `milestone=actuate`" in instruction
    assert "must author a nonzero `cartesian_delta`" in instruction
    assert "change the cartesian actuation direction" in instruction
    assert "keep the gripper `hold` or `close`" in instruction
    assert "do not return to `image_servo`" in instruction


def test_changed_actuation_revision_is_explicit_in_critic_request() -> None:
    from adaptive import remote_driver

    context = _context()
    context.milestone_history = ["observe", "approach", "engage"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    first = {
        **_cartesian_milestone_command("obs-0", "actuate"),
        "translation_m": [0.0, 0.0, 0.03],
        "gripper": "hold",
    }
    revised = {
        **_cartesian_milestone_command("obs-0", "actuate"),
        "translation_m": [0.0, 0.0, -0.02],
        "gripper": "hold",
    }
    _FakeClient.controller_outputs = [first, revised]
    _FakeClient.critic_outputs = [
        {
            "verdict": "revise",
            "contradiction": "actuation_unverified",
            "evidence": ["action_receipt"],
            "suggested_correction": "verify_actuation",
            "confidence": "high",
        },
        _proposal_audit(),
    ]
    state = _state()
    state.update(_cartesian_identity_public_state())
    sealed_contact = {
        "kind": "move_joints",
        "requested_targets": {"gripper": 0.0},
        "gripper_residual": {
            "measured_end_finger_separation": 0.006,
            "qpos_delta": [0.0, 0.0],
        },
    }

    response = _proposal_complete(
        wrapped, "obs-0", state, _images(), receipts=[sealed_contact]
    )

    assert response.command is revised
    payload = json.loads(str(_FakeClient.calls[3][1]["instruction"]))
    assert payload["actuation_revision"] == {
        "previous_actuation_rejected": True,
        "direction_changed": True,
        "nonzero": True,
        "gripper_preserved": True,
    }


def test_lost_actuation_contact_is_explicit_in_critic_request() -> None:
    from adaptive import remote_driver

    context = _context()
    context.milestone_history = ["observe", "approach", "engage", "actuate"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    recovery = {
        "kind": "image_servo",
        "observation_id": "obs-0",
        "camera": "left",
        "target_pixel": [60.0, 50.0],
        "target_role": "fixture_handle",
        "depth_delta_m": -0.01,
        "step_m": 0.02,
        "gripper": "close",
        "note": "milestone=actuate; reacquire the visible handle after contact loss",
    }
    lost_receipt = {
        "kind": "cartesian_delta",
        "requested_gripper": "hold",
        "gripper_residual": {"measured_end_finger_separation": 0.001},
    }
    _FakeClient.controller_outputs = [recovery]
    _FakeClient.critic_outputs = [_proposal_audit()]
    state = _state()
    state.update(_image_servo_public_state())

    response = _proposal_complete(
        wrapped,
        "obs-0",
        state,
        _images(),
        receipts=[lost_receipt],
        camera_calibration=_image_servo_calibration(),
    )

    assert response.command is recovery
    payload = json.loads(str(_FakeClient.calls[1][1]["instruction"]))
    assert payload["actuation_contact"] == {
        "comparable": True,
        "preserved": False,
        "recovery_required": True,
    }
    assert payload["contact_recovery"] == {
        "required": True,
        "command_kind_changed": True,
        "target_changed": False,
        "depth_direction_changed": False,
        "retreat_selected": False,
        "reengagement_after_retreat": False,
        "positive_depth_failed": False,
        "negative_depth_failed": False,
        "target_exhausted": False,
        "strategy_changed": True,
    }


def test_articulated_empty_handle_retry_changes_strategy_before_critic() -> None:
    from adaptive import remote_driver

    context = _context()
    context.milestone_history = ["observe", "approach", "engage"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    repeated = {
        "kind": "image_servo",
        "observation_id": "obs-empty-handle",
        "camera": "right",
        "target_pixel": [60.0, 50.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.01,
        "step_m": 0.01,
        "gripper": "close",
        "note": "milestone=engage; retrying the same empty handle point",
    }
    changed = {
        **repeated,
        "target_pixel": [92.0, 50.0],
        "gripper": "open",
        "note": "milestone=engage; insert open fingers at a new handle point",
    }
    _FakeClient.controller_outputs = [repeated, changed]
    _FakeClient.critic_outputs = [_proposal_audit()]
    empty_handle_receipt = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [60.0, 50.0],
        "requested_depth_delta_m": 0.015,
        "requested_gripper": "close",
        "gripper_residual": {"measured_end_finger_separation": 0.001},
    }
    state = _state()
    state.update(_image_servo_public_state())

    response = _proposal_complete(
        wrapped,
        "obs-empty-handle",
        state,
        _images(),
        receipts=[empty_handle_receipt],
        camera_calibration=_image_servo_calibration(),
    )

    assert response.command is changed
    assert [role for role, _ in _FakeClient.calls] == [
        "controller",
        "controller",
        "critic",
    ]
    rejected = context.proposal_records[0]
    assert rejected["status"] == "rejected_by_protocol"
    assert rejected["contradiction"] == "grasp_unverified"
    assert rejected["suggested_correction"] == "verify_gripper"
    assert rejected["mailbox_count"] == rejected["action_count"] == 0
    revision = _FakeClient.calls[1][1]["instruction"].casefold()
    assert "only 0 manhattan pixels" in revision
    assert "positive-depth `image_servo` with gripper `open`" in revision
    critic_payload = json.loads(str(_FakeClient.calls[2][1]["instruction"]))
    assert critic_payload["open_handle_insertion_rule"]["default_verdict"] == (
        "approve"
    )


def test_actuation_empty_handle_retry_changes_strategy_before_critic() -> None:
    from adaptive import remote_driver

    context = _context()
    context.milestone_history = ["observe", "approach", "engage", "actuate"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    repeated = {
        "kind": "image_servo",
        "observation_id": "obs-actuation-recovery",
        "camera": "right",
        "target_pixel": [60.0, 50.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.01,
        "step_m": 0.01,
        "gripper": "close",
        "note": "milestone=actuate; retrying the same empty handle point",
    }
    changed = {**repeated, "target_pixel": [92.0, 50.0]}
    _FakeClient.controller_outputs = [repeated, changed]
    _FakeClient.critic_outputs = [_proposal_audit()]
    positive_empty_receipt = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [60.0, 50.0],
        "requested_depth_delta_m": 0.015,
        "requested_gripper": "close",
        "gripper_residual": {"measured_end_finger_separation": 0.001},
    }
    negative_empty_receipt = {
        **positive_empty_receipt,
        "requested_depth_delta_m": -0.01,
    }
    state = _state()
    state.update(_image_servo_public_state())

    response = _proposal_complete(
        wrapped,
        "obs-actuation-recovery",
        state,
        _images(),
        receipts=[positive_empty_receipt, negative_empty_receipt],
        camera_calibration=_image_servo_calibration(),
    )

    assert response.command is changed
    assert [role for role, _ in _FakeClient.calls] == [
        "controller",
        "controller",
        "critic",
    ]
    assert context.proposal_records[0]["status"] == "rejected_by_protocol"
    assert context.proposal_records[0]["contradiction"] == "actuation_unverified"
    revision = _FakeClient.calls[1][1]["instruction"].casefold()
    assert "only 0 manhattan pixels" in revision
    assert "positive-depth `image_servo` with gripper `open`" in revision
    payload = json.loads(str(_FakeClient.calls[2][1]["instruction"]))
    assert payload["contact_recovery"] == {
        "required": True,
        "command_kind_changed": False,
        "target_changed": True,
        "depth_direction_changed": True,
        "retreat_selected": False,
        "reengagement_after_retreat": False,
        "positive_depth_failed": True,
        "negative_depth_failed": True,
        "target_exhausted": True,
        "strategy_changed": True,
    }


def test_open_retreat_preserves_failed_handle_target_memory() -> None:
    from adaptive.critic_protocol import (
        articulated_failed_contact_targets,
        validate_milestone_action_semantics,
    )

    target = [60.0, 50.0]
    positive_empty = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": target,
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "close",
        "gripper_residual": {"measured_end_finger_separation": 0.001},
    }
    negative_empty = {
        **positive_empty,
        "requested_depth_delta_m": -0.01,
    }
    retreat = {
        "kind": "cartesian_delta",
        "requested_gripper": "open",
        "gripper_residual": {"measured_end_finger_separation": 0.078},
    }
    same_target_recovery = {
        "kind": "image_servo",
        "observation_id": "obs-after-second-loss",
        "camera": "right",
        "target_pixel": target,
        "target_role": "fixture_handle",
        "depth_delta_m": 0.01,
        "step_m": 0.01,
        "gripper": "close",
        "note": "milestone=actuate; reacquire the same exhausted handle point",
    }

    receipts = [positive_empty, negative_empty, retreat]
    assert articulated_failed_contact_targets(receipts) == [
        {"camera": "right", "target_pixel": target}
    ]
    with pytest.raises(ValueError, match="target must move at least one grid cell"):
        validate_milestone_action_semantics(
            "articulated",
            "actuate",
            same_target_recovery,
            milestone_history=["observe", "approach", "engage", "actuate"],
            immediate_prior_receipt=retreat,
            recent_receipts=receipts,
            articulated_forbidden_targets=articulated_failed_contact_targets(
                receipts
            ),
        )


def test_engage_allows_open_view_move_after_empty_close_retreat() -> None:
    from adaptive.critic_protocol import (
        articulated_failed_contact_targets,
        validate_milestone_action_semantics,
    )

    empty_close = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [100.0, 110.0],
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "close",
        "gripper_residual": {"measured_end_finger_separation": 0.001},
    }
    retreat = {
        "kind": "cartesian_delta",
        "requested_gripper": "open",
        "gripper_residual": {"measured_end_finger_separation": 0.078},
    }
    view_move = {
        "kind": "cartesian_delta",
        "observation_id": "obs-occluded-handle",
        "translation_m": [0.0, 0.03, 0.0],
        "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        "gripper": "open",
        "note": "milestone=engage; move laterally to recover the handle view",
    }
    receipts = [empty_close, retreat]

    validate_milestone_action_semantics(
        "articulated",
        "engage",
        view_move,
        milestone_history=["observe", "approach", "engage"],
        immediate_prior_receipt=retreat,
        recent_receipts=receipts,
        articulated_forbidden_targets=articulated_failed_contact_targets(receipts),
    )


def test_engage_allows_open_view_move_after_recovery_servo() -> None:
    from adaptive import remote_driver
    from adaptive.critic_protocol import (
        articulated_failed_contact_targets,
        validate_milestone_action_semantics,
    )

    empty_close = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [100.0, 110.0],
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "close",
        "gripper_residual": {"measured_end_finger_separation": 0.001},
    }
    recovery_servo = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [160.0, 130.0],
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "open",
        "gripper_residual": {"measured_end_finger_separation": 0.078},
    }
    view_move = {
        "kind": "cartesian_delta",
        "observation_id": "obs-recovery-target-rejected",
        "translation_m": [0.0, 0.0, 0.03],
        "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        "gripper": "open",
        "note": "milestone=engage; recover a clear handle view",
    }
    receipts = [empty_close, recovery_servo]

    validate_milestone_action_semantics(
        "articulated",
        "engage",
        view_move,
        milestone_history=["observe", "approach", "engage"],
        immediate_prior_receipt=recovery_servo,
        recent_receipts=receipts,
        articulated_forbidden_targets=articulated_failed_contact_targets(receipts),
    )
    payload = json.loads(
        remote_driver._proposal_critic_instruction(
            _context(),
            task_instruction="open the toaster oven door",
            observation_id="obs-recovery-target-rejected",
            draft=view_move,
            claimed_milestone="engage",
            public_state=_state(),
            images=_images(),
            immediate_prior_receipt=recovery_servo,
            recent_receipts=receipts,
        )
    )
    assert payload["articulated_recovery_view_gathering"] == {
        "default_verdict": "approve",
        "effect_is_future_evidence": True,
        "empty_handle_target_history": True,
        "gripper_open": True,
        "nonzero": True,
        "protocol_validated": True,
        "tracking_settled": True,
    }


def test_two_empty_handle_targets_require_open_joint7_reorientation() -> None:
    from adaptive import remote_driver

    context = _context()
    context.milestone_history = ["observe", "approach", "engage"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    failures = [
        {
            "kind": "image_servo",
            "requested_camera": "right",
            "requested_target_pixel": target,
            "requested_depth_delta_m": 0.01,
            "requested_gripper": "close",
            "gripper_residual": {"measured_end_finger_separation": 0.001},
        }
        for target in ([100.0, 110.0], [132.0, 110.0])
    ]
    retreat = {
        "kind": "cartesian_delta",
        "requested_gripper": "open",
        "gripper_residual": {"measured_end_finger_separation": 0.078},
    }
    another_pixel = {
        "kind": "image_servo",
        "observation_id": "obs-reorient",
        "camera": "right",
        "target_pixel": [120.0, 150.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.01,
        "step_m": 0.01,
        "gripper": "open",
        "note": "milestone=engage; try a third handle point",
    }
    wrist_roll = {
        "kind": "move_joints",
        "observation_id": "obs-reorient",
        "targets": {"joint7": 1.5, "gripper": 1.0},
        "note": "milestone=engage; roll the open gripper across the handle",
    }
    _FakeClient.controller_outputs = [another_pixel, wrist_roll]
    _FakeClient.critic_outputs = [_proposal_audit()]
    state = _state()
    state.update(_image_servo_public_state())

    response = _proposal_complete(
        wrapped,
        "obs-reorient",
        state,
        _images(),
        receipts=[*failures, retreat],
        camera_calibration=_image_servo_calibration(),
    )

    assert response.command is wrist_roll
    assert [role for role, _ in _FakeClient.calls] == [
        "controller",
        "controller",
        "critic",
    ]
    assert context.proposal_records[0]["status"] == "rejected_by_protocol"
    revision = str(_FakeClient.calls[1][1]["instruction"]).casefold()
    assert "stop selecting another handle pixel" in revision
    assert "only `joint7` and `gripper`" in revision
    critic_payload = json.loads(str(_FakeClient.calls[2][1]["instruction"]))
    assert critic_payload["repeated_empty_handle_orientation_rule"] == {
        "default_verdict": "approve",
        "effect_is_future_evidence": True,
        "gripper_must_remain_open": True,
        "joint7_roll_only": True,
        "two_distinct_empty_targets": True,
    }


def test_incomplete_joint7_reorientation_must_settle_before_reinsertion() -> None:
    from adaptive import remote_driver
    from adaptive.critic_protocol import (
        articulated_failed_contact_targets,
        articulated_wrist_roll_tracking_status,
        validate_milestone_action_semantics,
    )

    failures = [
        {
            "kind": "image_servo",
            "requested_camera": "right",
            "requested_target_pixel": target,
            "requested_depth_delta_m": 0.01,
            "requested_gripper": "close",
            "gripper_residual": {"measured_end_finger_separation": 0.001},
        }
        for target in ([100.0, 110.0], [132.0, 110.0])
    ]
    incomplete_roll = {
        "kind": "move_joints",
        "requested_targets": {"joint7": -0.86, "gripper": 1.0},
        "realized_arm_qpos": [0.0, -0.8, 0.0, -1.5, 0.0, 1.1, -0.53],
        "gripper_residual": {"measured_end_finger_separation": 0.079},
    }
    retry_servo = {
        "kind": "image_servo",
        "observation_id": "obs-incomplete-roll",
        "camera": "right",
        "target_pixel": [100.0, 110.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.01,
        "step_m": 0.01,
        "gripper": "open",
        "note": "milestone=engage; reinsert after the requested wrist roll",
    }
    continue_roll = {
        "kind": "move_joints",
        "observation_id": "obs-incomplete-roll",
        "targets": {"joint7": -0.86, "gripper": 1.0},
        "note": "milestone=engage; continue the unfinished open wrist roll",
    }

    assert articulated_wrist_roll_tracking_status(incomplete_roll) == {
        "absolute_error_rad": pytest.approx(0.33),
        "realized_joint7_rad": -0.53,
        "requested_joint7_rad": -0.86,
        "settled": False,
    }
    with pytest.raises(ValueError, match="continue incomplete joint7 orientation"):
        validate_milestone_action_semantics(
            "articulated",
            "engage",
            retry_servo,
            milestone_history=["observe", "approach", "engage"],
            immediate_prior_receipt=incomplete_roll,
            recent_receipts=[*failures, incomplete_roll],
            articulated_forbidden_targets=failures,
        )
    validate_milestone_action_semantics(
        "articulated",
        "engage",
        continue_roll,
        milestone_history=["observe", "approach", "engage"],
        immediate_prior_receipt=incomplete_roll,
        recent_receipts=[*failures, incomplete_roll],
        articulated_forbidden_targets=failures,
    )
    instruction = remote_driver._controller_articulated_recovery_instruction(
        "base",
        failures,
        incomplete_roll,
    ).casefold()
    assert "unfinished joint7 orientation target" in instruction
    assert "absolute_error_rad" in instruction
    assert "repeat that same qwen-authored target" in instruction

    critic_payload = json.loads(
        remote_driver._proposal_critic_instruction(
            _context(),
            task_instruction="open the toaster oven door",
            observation_id="obs-incomplete-roll",
            draft=continue_roll,
            claimed_milestone="engage",
            public_state=_state(),
            images=_images(),
            immediate_prior_receipt=incomplete_roll,
            recent_receipts=[*failures, incomplete_roll],
        )
    )
    assert critic_payload["incomplete_wrist_roll_continuation_rule"] == {
        "default_verdict": "approve",
        "effect_is_future_evidence": True,
        "gripper_must_remain_open": True,
        "repeats_qwen_authored_joint7_target": True,
        "tracking_not_settled": True,
    }

    completed_roll = {
        **incomplete_roll,
        "realized_arm_qpos": [0.0, -0.8, 0.0, -1.5, 0.0, 1.1, -0.855],
    }
    assert articulated_wrist_roll_tracking_status(completed_roll)["settled"] is True
    assert articulated_failed_contact_targets(
        [*failures, incomplete_roll, completed_roll]
    ) == []
    completed_instruction = remote_driver._controller_articulated_recovery_instruction(
        "base",
        [],
        completed_roll,
    ).casefold()
    assert "joint7 orientation recovery is complete" in completed_instruction
    assert "pre-roll empty closes" in completed_instruction
    assert "positive-depth `image_servo`" in completed_instruction
    completed_revision = remote_driver._proposal_revision_instruction(
        "base",
        contradiction="grasp_unverified",
        evidence=["gripper_state", "action_receipt"],
        suggested_correction="verify_gripper",
        confidence="high",
        rejected_draft={
            "kind": "cartesian_delta",
            "observation_id": "obs-completed-roll",
            "translation_m": [0.02, 0.0, 0.01],
            "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
            "gripper": "open",
            "note": "milestone=engage; retreat after the completed wrist roll",
        },
        required_observation_id="obs-completed-roll",
        immediate_prior_receipt=completed_roll,
        recent_receipts=[*failures, incomplete_roll, completed_roll],
    ).casefold()
    assert "starts a new handle-contact epoch" in completed_revision
    assert "positive-depth `image_servo`" in completed_revision

    moving_near_target_roll = {
        **completed_roll,
        "telemetry_summary": {
            "arm_joint_velocity": {
                "end_rad_s": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.172]
            }
        },
    }
    moving_status = articulated_wrist_roll_tracking_status(
        moving_near_target_roll
    )
    assert moving_status["absolute_joint7_velocity_rad_s"] == pytest.approx(0.172)
    assert moving_status["settled"] is False


def test_incomplete_joint7_reorientation_uses_required_continuation_schema() -> None:
    from adaptive import remote_driver
    from adaptive.joint_runner import (
        controller_wrist_roll_schema_for_observation,
    )

    prior_roll = {
        "kind": "move_joints",
        "observation_id": "obs-roll",
        "targets": {"joint7": -0.85, "gripper": 1.0},
        "note": "milestone=engage; roll the open wrist for handle recovery",
    }
    state = _state(qvel=-0.19)
    state["state.arm_joint_position"][6] = -0.524
    incomplete_receipt = _sealed_critic_receipt(prior_roll, state)
    continuation = {
        "kind": "move_joints",
        "observation_id": "obs-continue-roll",
        "targets": {"joint7": -0.85, "gripper": 1.0},
        "note": "milestone=engage; continue the unfinished open wrist roll",
    }
    context = _context()
    context.milestone_history = ["observe", "approach", "engage"]
    wrapped = remote_driver._make_proposal_audit_client_class(
        _FakeClient, context
    )()
    _FakeClient.controller_outputs = [continuation]
    _FakeClient.critic_outputs = [_proposal_audit()]

    response = _proposal_complete(
        wrapped,
        "obs-continue-roll",
        state,
        _images(),
        receipts=[incomplete_receipt],
    )

    assert response.command is continuation
    controller_call = next(
        call for role, call in _FakeClient.calls if role == "controller"
    )
    assert controller_call["response_schema"] == (
        controller_wrist_roll_schema_for_observation("obs-continue-roll")
    )


def test_new_target_reengagement_after_retreat_is_explicit_for_critic() -> None:
    from adaptive import remote_driver

    context = _context()
    context.milestone_history = ["observe", "approach", "engage", "actuate"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    new_target = {
        "kind": "image_servo",
        "observation_id": "obs-after-retreat",
        "camera": "right",
        "target_pixel": [92.0, 50.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.01,
        "step_m": 0.01,
        "gripper": "close",
        "note": "milestone=actuate; re-engage a new visible handle point",
    }
    failed = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [60.0, 50.0],
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "close",
        "gripper_residual": {"measured_end_finger_separation": 0.001},
    }
    retreat = {
        "kind": "cartesian_delta",
        "requested_gripper": "open",
        "gripper_residual": {"measured_end_finger_separation": 0.078},
    }
    _FakeClient.controller_outputs = [new_target]
    _FakeClient.critic_outputs = [_proposal_audit()]
    state = _state()
    state.update(_image_servo_public_state())

    response = _proposal_complete(
        wrapped,
        "obs-after-retreat",
        state,
        _images(),
        receipts=[failed, retreat],
        camera_calibration=_image_servo_calibration(),
    )

    assert response.command is new_target
    payload = json.loads(str(_FakeClient.calls[1][1]["instruction"]))
    assert payload["contact_recovery"]["reengagement_after_retreat"] is True
    assert payload["contact_recovery"]["strategy_changed"] is True


def test_retreat_keeps_a_failed_handle_target_forbidden() -> None:
    from adaptive import remote_driver

    # Generic articulated rule: the toaster task now rolls after one screened
    # failure, so exercise the two-target memory on another articulated task.
    context = dataclasses.replace(_context(), task="PickPlaceCounterToDrawer")
    context.milestone_history = ["observe", "approach", "engage"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    near_target = {
        "kind": "image_servo",
        "observation_id": "obs-after-retreat",
        "camera": "right",
        "target_pixel": [90.0, 50.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.01,
        "step_m": 0.01,
        "gripper": "open",
        "note": "milestone=engage; insert near the failed handle point",
    }
    shifted_target = {
        **near_target,
        "target_pixel": [92.0, 50.0],
        "note": "milestone=engage; insert at a distinct handle point",
    }
    weak = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [60.0, 50.0],
        "requested_depth_delta_m": -0.01,
        "requested_gripper": "close",
        "gripper_residual": {"measured_end_finger_separation": 0.001},
    }
    retreat = {
        "kind": "cartesian_delta",
        "requested_gripper": "open",
        "gripper_residual": {"measured_end_finger_separation": 0.078},
    }
    _FakeClient.controller_outputs = [near_target, shifted_target]
    _FakeClient.critic_outputs = [_proposal_audit()]
    state = _state()
    state.update(_image_servo_public_state())

    response = _proposal_complete(
        wrapped,
        "obs-after-retreat",
        state,
        _images(),
        receipts=[weak, retreat],
        camera_calibration=_image_servo_calibration(),
    )

    assert response.command is shifted_target
    assert [role for role, _ in _FakeClient.calls] == [
        "controller",
        "controller",
        "critic",
    ]
    assert context.proposal_records[0]["status"] == "rejected_by_protocol"
    assert context.proposal_records[0]["contradiction"] == (
        "visual_alignment_unverified"
    )
    assert "ARTICULATED_RECOVERY_CONTEXT:" in str(
        _FakeClient.calls[0][1]["instruction"]
    )
    assert "combined |delta_u|+|delta_v| is at least 32 pixels" in str(
        _FakeClient.calls[1][1]["instruction"]
    ).casefold()
    assert "only 30 manhattan pixels" in str(
        _FakeClient.calls[1][1]["instruction"]
    ).casefold()
    assert "leave margin" in str(
        _FakeClient.calls[1][1]["instruction"]
    ).casefold()
    assert context.articulated_forbidden_targets == [
        {"camera": "right", "target_pixel": [60.0, 50.0]}
    ]


def test_open_gripper_cannot_actuate_before_reengagement() -> None:
    from adaptive import remote_driver
    from adaptive.critic_protocol import validate_milestone_action_semantics

    open_retreat = {
        "kind": "cartesian_delta",
        "requested_gripper": "open",
        "gripper_residual": {"measured_end_finger_separation": 0.078},
    }
    open_gripper_rotation = {
        "kind": "cartesian_delta",
        "observation_id": "obs-open",
        "translation_m": [0.0, 0.0, 0.0],
        "rotation_axis_angle_rad": [0.0, 0.08, 0.0],
        "gripper": "hold",
        "note": "milestone=actuate; rotate after an open retreat",
    }

    with pytest.raises(ValueError, match="requires re-engagement after open retreat"):
        validate_milestone_action_semantics(
            "articulated",
            "actuate",
            open_gripper_rotation,
            milestone_history=["observe", "approach", "engage", "actuate"],
            immediate_prior_receipt=open_retreat,
            recent_receipts=[open_retreat],
        )

    revision = remote_driver._proposal_revision_instruction(
        "base instruction",
        contradiction="actuation_unverified",
        evidence=["gripper_state", "action_receipt"],
        suggested_correction="verify_actuation",
        confidence="high",
        rejected_draft=open_gripper_rotation,
        required_observation_id="obs-open",
        immediate_prior_receipt=open_retreat,
        recent_receipts=[open_retreat],
    ).casefold()
    assert "the immediate prior receipt left the gripper open" in revision
    assert "must re-engage before any cartesian actuation" in revision
    assert "nonzero `image_servo` with gripper `close`" in revision

    record: dict[str, object] = {}
    advice = remote_driver._protocol_rejection(
        record,
        error=ValueError(
            "articulated actuation requires re-engagement after open retreat"
        ),
    )
    assert advice["contradiction"] == "actuation_unverified"
    assert advice["suggested_correction"] == "verify_actuation"


def test_established_handle_contact_advances_to_cartesian_actuation() -> None:
    from adaptive import remote_driver
    from adaptive.critic_protocol import validate_milestone_action_semantics

    contact_receipt = {
        "kind": "move_joints",
        "requested_targets": {"gripper": 0.0},
        "gripper_residual": {
            "measured_end_finger_separation": 0.006,
            "qpos_delta": [0.0, 0.0],
        },
    }
    repeated_close = {
        "kind": "image_servo",
        "observation_id": "obs-contact",
        "camera": "right",
        "target_pixel": [106.0, 128.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.01,
        "step_m": 0.01,
        "gripper": "close",
        "note": "milestone=actuate; repeat the close servo after contact",
    }

    with pytest.raises(ValueError, match="contact already established"):
        validate_milestone_action_semantics(
            "articulated",
            "actuate",
            repeated_close,
            milestone_history=["observe", "approach", "engage", "actuate"],
            immediate_prior_receipt=contact_receipt,
            recent_receipts=[contact_receipt],
        )

    revision = remote_driver._proposal_revision_instruction(
        "base instruction",
        contradiction="actuation_unverified",
        evidence=["gripper_state", "action_receipt"],
        suggested_correction="verify_actuation",
        confidence="high",
        rejected_draft=repeated_close,
        required_observation_id="obs-contact",
        immediate_prior_receipt=contact_receipt,
        recent_receipts=[contact_receipt],
    ).casefold()
    assert "handle contact is already established" in revision
    assert "do not author another `image_servo`" in revision
    assert "nonzero translation" in revision
    assert "gripper `hold` or `close`" in revision

    record: dict[str, object] = {}
    advice = remote_driver._protocol_rejection(
        record,
        error=ValueError(
            "articulated handle contact already established; actuate before "
            "another close servo"
        ),
    )
    assert advice["contradiction"] == "actuation_unverified"


def test_first_handle_obstruction_forces_confirmation_instead_of_retreat() -> None:
    from adaptive import remote_driver

    context = _context()
    context.milestone_history = ["observe", "approach", "engage"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    wrong_retreat = {
        "kind": "cartesian_delta",
        "observation_id": "obs-thin-contact",
        "translation_m": [-0.02, 0.0, 0.0],
        "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        "gripper": "open",
        "note": "milestone=engage; incorrectly retreat from the thin handle",
    }
    confirmation = {
        "kind": "move_joints",
        "observation_id": "obs-thin-contact",
        "targets": {"gripper": 0.0},
        "note": "milestone=actuate; hold the arm still and confirm contact",
    }
    contact = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [100.0, 120.0],
        "requested_depth_delta_m": -0.015,
        "requested_gripper": "close",
        "resolved_translation_m": [-0.01, -0.02, -0.03],
        "gripper_residual": {"measured_end_finger_separation": 0.006},
    }
    _FakeClient.controller_outputs = [wrong_retreat, confirmation]
    _FakeClient.critic_outputs = [_proposal_audit()]

    response = _proposal_complete(
        wrapped,
        "obs-thin-contact",
        _state(),
        _images(),
        receipts=[contact],
    )

    assert response.command is confirmation
    assert [role for role, _ in _FakeClient.calls] == [
        "controller",
        "controller",
        "critic",
    ]
    assert context.proposal_records[0]["contradiction"] == "actuation_unverified"
    initial_instruction = str(_FakeClient.calls[0][1]["instruction"])
    assert "ARTICULATED_CONTACT_CONTEXT:" in initial_instruction
    assert '"contact_established":true' in initial_instruction
    revision_instruction = str(_FakeClient.calls[1][1]["instruction"]).casefold()
    assert "first obstruction is provisional" in revision_instruction
    assert 'only `targets={"gripper":0.0}`' in revision_instruction
    assert "do not move any arm joint" in revision_instruction


def test_recontact_prompt_carries_episode_failed_actuation_axes() -> None:
    from adaptive import remote_driver

    context = _context()
    context.milestone_history = ["observe", "approach", "engage", "actuate"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    failed_vertical = {
        "kind": "cartesian_delta",
        "requested_translation_m": [0.0, 0.0, -0.02],
        "requested_rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        "requested_gripper": "hold",
        "gripper_residual": {"measured_end_finger_separation": 0.001},
    }
    contact = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [128.0, 128.0],
        "requested_depth_delta_m": -0.01,
        "requested_gripper": "close",
        "resolved_translation_m": [-0.01, -0.02, -0.03],
        "gripper_residual": {"measured_end_finger_separation": 0.006},
    }
    confirmed = {
        "kind": "move_joints",
        "requested_targets": {"gripper": 0.0},
        "gripper_residual": {
            "measured_end_finger_separation": 0.006,
            "qpos_delta": [0.0, 0.0],
        },
    }
    horizontal = {
        "kind": "cartesian_delta",
        "observation_id": "obs-axis-memory",
        "translation_m": [-0.02, 0.0, 0.0],
        "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        "gripper": "hold",
        "note": "milestone=actuate; use an untried horizontal pull axis",
    }
    _FakeClient.controller_outputs = [horizontal]
    _FakeClient.critic_outputs = [_proposal_audit()]

    response = _proposal_complete(
        wrapped,
        "obs-axis-memory",
        _state(),
        _images(),
        receipts=[failed_vertical, contact, confirmed],
    )

    assert response.command is horizontal
    initial_instruction = str(_FakeClient.calls[0][1]["instruction"])
    contact_json = initial_instruction.split("ARTICULATED_CONTACT_CONTEXT:\n", 1)[1]
    contact_packet = json.loads(contact_json.splitlines()[0])
    assert contact_packet["failed_actuation_axis_sets"] == [["translation_z"]]
    assert contact_packet["engagement_translation_m"] is None
    assert contact_packet["first_actuation_strategy"] == (
        "image_plane_articulation_motion"
    )
    assert "must not repeat any listed active-axis set" in initial_instruction
    normalized_instruction = initial_instruction.casefold()
    assert "target_role=articulation_motion" in normalized_instruction
    assert "do not merely reverse `engagement_translation_m`" in (
        normalized_instruction
    )
    assert context.articulated_failed_actuation_axis_sets == [["translation_z"]]


def test_articulation_motion_role_is_contact_gated() -> None:
    from adaptive.critic_protocol import validate_milestone_action_semantics
    from adaptive.image_servo import decode_image_servo
    from adaptive.joint_runner import IMAGE_SERVO_RESPONSE_SCHEMA

    draft = {
        "kind": "image_servo",
        "observation_id": "obs-drag",
        "camera": "right",
        "target_pixel": [112.0, 160.0],
        "target_role": "articulation_motion",
        "depth_delta_m": -0.015,
        "step_m": 0.01,
        "gripper": "hold",
        "note": "milestone=actuate; drag the contacted door handle down and outward",
    }
    command = decode_image_servo(draft, observation_id="obs-drag")
    assert command.target_role == "articulation_motion"
    assert "articulation_motion" in (
        IMAGE_SERVO_RESPONSE_SCHEMA["properties"]["target_role"]["enum"]
    )

    contact = {
        "kind": "move_joints",
        "requested_targets": {"gripper": 0.0},
        "gripper_residual": {
            "measured_end_finger_separation": 0.006,
            "qpos_delta": [0.0, 0.0],
        },
    }
    validate_milestone_action_semantics(
        "articulated",
        "actuate",
        draft,
        immediate_prior_receipt=contact,
        recent_receipts=[contact],
    )
    with pytest.raises(ValueError, match="requires stationary contact confirmation"):
        validate_milestone_action_semantics(
            "articulated",
            "actuate",
            draft,
            immediate_prior_receipt={
                "kind": "image_servo",
                "requested_gripper": "close",
                "gripper_residual": {"measured_end_finger_separation": 0.006},
            },
        )
    with pytest.raises(ValueError, match="requires sealed articulated contact"):
        validate_milestone_action_semantics(
            "articulated",
            "approach",
            {**draft, "note": "milestone=approach; premature drag"},
        )


def test_first_articulated_obstruction_requires_stationary_confirmation() -> None:
    from adaptive import remote_driver

    initial_contact = {
        "kind": "image_servo",
        "requested_gripper": "close",
        "resolved_translation_m": [0.0006, -0.0041, -0.0191],
        "gripper_residual": {"measured_end_finger_separation": 0.006},
    }
    instruction = remote_driver._controller_articulated_recovery_instruction(
        "base instruction",
        [],
        immediate_prior_receipt=initial_contact,
    )
    packet = json.loads(
        instruction.split("ARTICULATED_CONTACT_CONTEXT:\n", 1)[1].splitlines()[0]
    )
    normalized = instruction.casefold()

    assert packet["contact_confirmation_required"] is True
    assert packet["first_actuation_strategy"] == "stationary_contact_confirmation"
    assert "author `move_joints` with only `targets={\"gripper\":0.0}`" in normalized
    assert "do not move any arm joint" in normalized
    assert "target_role=articulation_motion" not in normalized

    critic_payload = json.loads(
        remote_driver._proposal_critic_instruction(
            _context(),
            task_instruction="open the toaster oven door",
            observation_id="obs-confirm",
            draft={
                "kind": "move_joints",
                "observation_id": "obs-confirm",
                "targets": {"gripper": 0.0},
                "note": "milestone=actuate; hold arm still and confirm contact",
            },
            claimed_milestone="actuate",
            public_state=_state(),
            images=_images(),
            immediate_prior_receipt=initial_contact,
        )
    )
    assert critic_payload["contact_confirmation_rule"] == {
        "arm_motion_prohibited": True,
        "contact_persistence_is_future_evidence": True,
        "default_verdict": "approve",
        "gripper_close_only": True,
    }
    critic_prompt = " ".join(
        (Path(__file__).resolve().parents[1] / "prompts" / "proposal_audit_critic.txt")
        .read_text(encoding="utf-8")
        .casefold()
        .split()
    )
    assert "contact_confirmation_rule" in critic_prompt
    assert "approve the stationary confirmation" in critic_prompt
    assert "stale explanatory words about the preceding" in critic_prompt
    assert "are non-executable" in critic_prompt
    assert "not `motion_effect_mismatch`" in critic_prompt
    assert critic_prompt.index("if the note and numeric gripper intent disagree") < (
        critic_prompt.index("stale explanatory words about the preceding")
    )


def test_stationary_close_is_provisional_until_gripper_motion_settles() -> None:
    from adaptive.critic_protocol import (
        articulated_contact_is_confirmed,
        articulated_contact_is_provisional,
    )

    first_close = {
        "kind": "move_joints",
        "requested_targets": {"gripper": 0.0},
        "gripper_residual": {
            "measured_end_finger_separation": 0.006,
            "qpos_delta": [-0.03, 0.03],
        },
    }
    settled_close = {
        **first_close,
        "gripper_residual": {
            "measured_end_finger_separation": 0.006,
            "qpos_delta": [-0.0001, 0.0001],
        },
    }

    assert articulated_contact_is_provisional(first_close) is True
    assert articulated_contact_is_confirmed(first_close) is False
    assert articulated_contact_is_provisional(settled_close) is False
    assert articulated_contact_is_confirmed(settled_close) is True


def test_confirmed_articulation_prefers_image_plane_motion_over_ray_reversal() -> None:
    from adaptive import remote_driver

    instruction = remote_driver._controller_articulated_recovery_instruction(
        "base instruction",
        [],
        immediate_prior_receipt={
            "kind": "move_joints",
            "requested_targets": {"gripper": 0.0},
            "gripper_residual": {
                "measured_end_finger_separation": 0.006,
                "qpos_delta": [0.0, 0.0],
            },
        },
    )
    packet = json.loads(
        instruction.split("ARTICULATED_CONTACT_CONTEXT:\n", 1)[1].splitlines()[0]
    )
    assert packet["contact_confirmation_required"] is False
    assert packet["first_actuation_strategy"] == "image_plane_articulation_motion"
    normalized = instruction.casefold()
    assert "target_role=articulation_motion" in normalized
    assert "visible hinge, drawer, or knob motion" in normalized
    assert "do not merely reverse `engagement_translation_m`" in normalized
    assert "at most 16 pixels from the current end-effector pixel" in normalized
    assert "`depth_delta_m=0.0`" in normalized
    assert "`step_m` between 0.01 and 0.02" in normalized
    assert "`step_m=0.03` per command" in normalized
    assert "pull back along the reverse of `engagement_translation_m`" not in normalized


def test_critic_treats_articulation_motion_pixel_as_a_future_waypoint() -> None:
    from adaptive import remote_driver

    root = Path(__file__).resolve().parents[1]
    controller = " ".join(
        (root / "prompts" / "joint_system.txt").read_text().split()
    ).casefold()
    critic = " ".join(
        (root / "prompts" / "proposal_audit_critic.txt").read_text().split()
    ).casefold()

    assert "target_role=articulation_motion" in controller
    assert "desired future image waypoint" in controller
    assert "target_role=articulation_motion" in critic
    assert "need not lie on the current handle material" in critic
    assert "sealed contact" in critic

    payload = json.loads(
        remote_driver._proposal_critic_instruction(
            _context(),
            task_instruction="open the toaster oven door",
            observation_id="obs-drag-audit",
            draft={
                "kind": "image_servo",
                "observation_id": "obs-drag-audit",
                "camera": "right",
                "target_pixel": [112.0, 160.0],
                "target_role": "articulation_motion",
                "depth_delta_m": -0.015,
                "step_m": 0.01,
                "gripper": "hold",
                "note": "milestone=actuate; drag the contacted handle down",
            },
            claimed_milestone="actuate",
            public_state=_state(),
            images=_images(),
            immediate_prior_receipt={
                "kind": "image_servo",
                "requested_gripper": "close",
                "gripper_residual": {"measured_end_finger_separation": 0.006},
            },
        )
    )
    assert payload["articulation_motion_rule"] == {
        "contact_preserved": True,
        "default_verdict": "approve",
        "effect_is_future_evidence": True,
        "surface_membership_not_required": True,
        "target_is_future_image_waypoint": True,
    }


def test_rejected_rotation_only_actuation_requires_translation() -> None:
    from adaptive import remote_driver

    instruction = remote_driver._proposal_revision_instruction(
        "base instruction",
        contradiction="actuation_unverified",
        evidence=["action_receipt", "external_rgb"],
        suggested_correction="verify_actuation",
        confidence="high",
        rejected_draft={
            "kind": "cartesian_delta",
            "observation_id": "obs-rotation",
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_axis_angle_rad": [0.0, 0.08, 0.0],
            "gripper": "hold",
            "note": "milestone=actuate; rotate the closed handle",
        },
        required_observation_id="obs-rotation",
        immediate_prior_receipt={
            "kind": "move_joints",
            "requested_targets": {"gripper": 0.0},
            "gripper_residual": {
                "measured_end_finger_separation": 0.006,
                "qpos_delta": [0.0, 0.0],
            },
        },
    ).casefold()

    assert "pure wrist rotation was rejected" in instruction
    assert "must include nonzero translation" in instruction
    assert "changing only the rotation sign is prohibited" in instruction


def test_near_empty_articulated_close_is_not_durable_contact() -> None:
    from adaptive import remote_driver
    from adaptive.critic_protocol import (
        ARTICULATED_CONTACT_MIN_SEPARATION,
        articulated_contact_recovery_status,
    )

    thin_handle_contact = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [106.0, 128.0],
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "close",
        "gripper_residual": {"measured_end_finger_separation": 0.002105},
    }
    retry = {
        "kind": "image_servo",
        "camera": "right",
        "target_pixel": [106.0, 128.0],
        "depth_delta_m": 0.01,
        "gripper": "close",
    }

    assert ARTICULATED_CONTACT_MIN_SEPARATION == 0.0032
    assert remote_driver._actuation_contact_status("actuate", thin_handle_contact) == {
        "comparable": True,
        "preserved": False,
        "recovery_required": True,
    }
    recovery = articulated_contact_recovery_status(
        "actuate", retry, thin_handle_contact, [thin_handle_contact]
    )
    assert recovery is not None
    assert recovery["required"] is True
    assert recovery["positive_depth_failed"] is True

    root = Path(__file__).resolve().parents[1]
    controller = " ".join(
        (root / "prompts" / "joint_system.txt")
        .read_text(encoding="utf-8")
        .casefold()
        .split()
    )
    critic = " ".join(
        (root / "prompts" / "proposal_audit_critic.txt")
        .read_text(encoding="utf-8")
        .casefold()
        .split()
    )
    assert "settled close above `0.0032`" in controller
    assert "settled separation strictly greater than `0.0032`" in critic


def test_empty_stationary_close_exhausts_its_open_insertion_target() -> None:
    from adaptive.critic_protocol import articulated_failed_contact_targets

    insertion = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [112.0, 128.0],
        "requested_target_role": "fixture_handle",
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "open",
        "gripper_residual": {"measured_end_finger_separation": 0.079},
    }
    empty_close = {
        "kind": "move_joints",
        "requested_targets": {"gripper": 0.0},
        "gripper_residual": {
            "measured_end_finger_separation": 0.0021,
            "qpos_delta": [-0.0001, 0.0001],
        },
    }

    assert articulated_failed_contact_targets([insertion, empty_close]) == [
        {"camera": "right", "target_pixel": [112.0, 128.0]}
    ]


def test_articulated_cannot_leave_engage_after_an_empty_stationary_close() -> None:
    from adaptive import remote_driver
    from adaptive.critic_protocol import validate_milestone_action_semantics

    insertion = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [112.0, 128.0],
        "requested_target_role": "fixture_handle",
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "open",
    }
    empty_close = {
        "kind": "move_joints",
        "requested_targets": {"gripper": 0.0},
        "gripper_residual": {"measured_end_finger_separation": 0.0024},
    }
    actuate = {
        "kind": "cartesian_delta",
        "observation_id": "obs-empty-actuate",
        "translation_m": [0.0, 0.0, -0.02],
        "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        "gripper": "close",
        "note": "milestone=actuate; pull after the empty close",
    }
    with pytest.raises(ValueError, match="sealed contact before leaving engage"):
        validate_milestone_action_semantics(
            "articulated",
            "actuate",
            actuate,
            task="OpenToasterOvenDoor",
            milestone_history=["observe", "approach", "engage"],
            immediate_prior_receipt=empty_close,
            recent_receipts=[insertion, empty_close],
        )

    text = remote_driver._controller_articulated_recovery_instruction(
        "base",
        [{"camera": "right", "target_pixel": [112.0, 128.0]}],
        empty_close,
        task="OpenToasterOvenDoor",
    ).casefold()
    assert '"required_milestone":"engage"' in text
    assert "reopen before retrying" in text
    assert "gripper `open`" in text


def test_empty_articulated_close_revises_actuate_back_to_open_engage() -> None:
    from adaptive import remote_driver

    observation_id = "obs-empty-recovery"
    invalid_actuate = {
        "kind": "cartesian_delta",
        "observation_id": observation_id,
        "translation_m": [0.0, 0.0, -0.02],
        "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        "gripper": "close",
        "note": "milestone=actuate; pull after an empty close",
    }
    open_reengage = {
        **_image_servo_payload(),
        "observation_id": observation_id,
        "camera": "right",
        "target_pixel": [92.0, 50.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.01,
        "step_m": 0.01,
        "gripper": "open",
        "note": "milestone=engage; reopen at a shifted visible handle point",
    }
    insertion = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [60.0, 50.0],
        "requested_target_role": "fixture_handle",
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "open",
    }
    empty_close = {
        "kind": "move_joints",
        "requested_targets": {"gripper": 0.0},
        "gripper_residual": {"measured_end_finger_separation": 0.0024},
    }
    context = _context()
    context.milestone_history = ["observe", "approach", "engage"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    context.task = "OpenCabinetDoor"
    _FakeClient.controller_outputs = [invalid_actuate, open_reengage]
    _FakeClient.critic_outputs = [_proposal_audit()]
    state = _state()
    state.update(_image_servo_public_state())

    response = _proposal_complete(
        wrapped,
        observation_id,
        state,
        _images(),
        receipts=[insertion, empty_close],
        camera_calibration=_image_servo_calibration(),
    )

    assert response.command is open_reengage
    assert context.proposal_records[0]["status"] == "rejected_by_protocol"
    assert context.proposal_records[0]["contradiction"] == "actuation_unverified"
    assert context.proposal_records[1]["status"] == "approved_for_execution"
    revision = str(_FakeClient.calls[1][1]["instruction"]).casefold()
    assert "sealed contact before leaving engage" in revision
    assert "reopen before retrying" in revision


def test_articulated_engage_requires_open_insertion_before_critic() -> None:
    from adaptive import remote_driver
    from adaptive.critic_protocol import validate_milestone_action_semantics

    open_engage = {
        "kind": "image_servo",
        "observation_id": "obs-engage",
        "camera": "right",
        "target_pixel": [128.0, 128.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.01,
        "step_m": 0.02,
        "gripper": "open",
        "note": "milestone=engage; close on the aligned handle",
    }

    validate_milestone_action_semantics(
        "articulated",
        "engage",
        open_engage,
        milestone_history=["observe", "approach"],
    )

    record: dict[str, object] = {}
    advice = remote_driver._protocol_rejection(
        record,
        error=ValueError(
            "articulated engage requires open insertion before stationary close"
        ),
    )
    assert advice["contradiction"] == "grasp_unverified"
    assert advice["suggested_correction"] == "verify_gripper"


def test_weak_articulated_engage_can_retreat_open_before_critic() -> None:
    from adaptive import remote_driver

    context = _context()
    context.milestone_history = ["observe", "approach", "engage"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    retreat = {
        "kind": "cartesian_delta",
        "observation_id": "obs-engage-retreat",
        "translation_m": [-0.02, 0.0, 0.0],
        "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        "gripper": "open",
        "note": "milestone=engage; retreat after weak handle contact",
    }
    weak_close = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [100.0, 110.0],
        "requested_depth_delta_m": -0.01,
        "requested_gripper": "close",
        "gripper_residual": {"measured_end_finger_separation": 0.001},
    }
    _FakeClient.controller_outputs = [retreat]
    _FakeClient.critic_outputs = [_proposal_audit()]

    response = _proposal_complete(
        wrapped,
        "obs-engage-retreat",
        _state(),
        _images(),
        receipts=[weak_close],
    )

    assert response.command is retreat
    assert [role for role, _ in _FakeClient.calls] == ["controller", "critic"]
    critic_payload = json.loads(str(_FakeClient.calls[1][1]["instruction"]))
    assert critic_payload["contact_recovery"]["required"] is True
    assert critic_payload["contact_recovery"]["retreat_selected"] is True
    assert critic_payload["contact_recovery"]["strategy_changed"] is True
    assert critic_payload["articulated_empty_close_retreat_rule"] == {
        "default_verdict": "approve",
        "effect_is_future_evidence": True,
        "empty_close_confirmed": True,
        "gripper_open": True,
        "nonzero": True,
        "protocol_validated": True,
        "tracking_settled": True,
    }


def test_weak_handle_targets_are_machine_visible_to_controller() -> None:
    from adaptive import remote_driver

    context = _context()
    context.milestone_history = ["observe", "approach", "engage", "actuate"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    old_target = [60.0, 50.0]
    near_target = {
        "kind": "image_servo",
        "observation_id": "obs-new-target",
        "camera": "right",
        "target_pixel": [64.0, 54.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.01,
        "step_m": 0.01,
        "gripper": "close",
        "note": "milestone=actuate; make a tiny local target adjustment",
    }
    new_target = {
        **near_target,
        "target_pixel": [92.0, 50.0],
        "note": "milestone=actuate; re-localize a visibly different handle point",
    }
    weak = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": old_target,
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "close",
        "gripper_residual": {"measured_end_finger_separation": 0.001},
    }
    _FakeClient.controller_outputs = [near_target, new_target]
    _FakeClient.critic_outputs = [_proposal_audit()]
    state = _state()
    state.update(_image_servo_public_state())

    response = _proposal_complete(
        wrapped,
        "obs-new-target",
        state,
        _images(),
        receipts=[weak],
        camera_calibration=_image_servo_calibration(),
    )

    assert response.command is new_target
    assert [role for role, _ in _FakeClient.calls] == [
        "controller",
        "controller",
        "critic",
    ]
    assert context.proposal_records[0]["status"] == "rejected_by_protocol"
    assert "combined |delta_u|+|delta_v| is at least 32 pixels" in str(
        _FakeClient.calls[1][1]["instruction"]
    ).casefold()
    instruction = str(_FakeClient.calls[0][1]["instruction"])
    assert "ARTICULATED_RECOVERY_CONTEXT:" in instruction
    recovery_json = instruction.split("ARTICULATED_RECOVERY_CONTEXT:\n", 1)[1]
    recovery_json = recovery_json.splitlines()[0]
    assert json.loads(recovery_json) == {
        "forbidden_image_servo_targets": [
            {"camera": "right", "target_pixel": old_target}
        ],
        "minimum_target_shift_px": 32.0,
        "target_shift_metric": "manhattan_l1",
        "contact_min_separation": 0.0032,
        "required_milestone": "actuate",
    }
    assert "combined |delta_u|+|delta_v| is at least 32 pixels" in (
        instruction.casefold()
    )
    assert "visible long-axis interior midline" in instruction.casefold()
    assert "not an articulation waypoint" in instruction.casefold()
    assert "keep that recovery camera+target_pixel fixed" in instruction.casefold()
    assert context.articulated_forbidden_targets == [
        {"camera": "right", "target_pixel": old_target}
    ]


def test_open_handle_reinsertion_must_leave_each_empty_close_target() -> None:
    from adaptive.critic_protocol import validate_milestone_action_semantics

    retry = {
        "kind": "image_servo",
        "camera": "right",
        "target_pixel": [110.0, 115.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.01,
        "step_m": 0.01,
        "gripper": "open",
    }
    forbidden = [{"camera": "right", "target_pixel": [100.0, 110.0]}]

    with pytest.raises(ValueError, match="target must move at least one grid cell"):
        validate_milestone_action_semantics(
            "articulated",
            "engage",
            retry,
            milestone_history=["observe", "approach", "engage"],
            articulated_forbidden_targets=forbidden,
        )

    validate_milestone_action_semantics(
        "articulated",
        "engage",
        {**retry, "target_pixel": [132.0, 110.0]},
        milestone_history=["observe", "approach", "engage"],
        articulated_forbidden_targets=forbidden,
    )


def test_unverified_grasp_revision_can_reacquire_a_lost_source_view() -> None:
    from adaptive import remote_driver

    instruction = remote_driver._proposal_revision_instruction(
        "base instruction",
        contradiction="grasp_unverified",
        evidence=["gripper_state", "external_rgb"],
        suggested_correction="verify_gripper",
        confidence="high",
        rejected_draft={
            "kind": "cartesian_delta",
            "translation_m": [0.0, 0.0, 0.02],
            "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
            "gripper": "hold",
            "note": "milestone=transport; empty grasp cannot be lifted",
        },
        required_observation_id="obs-0",
    ).casefold()

    assert "must author a bounded nonzero `cartesian_delta` retreat" in instruction
    assert "gripper `open`" in instruction
    assert "new observation" in instruction
    assert "do not author another `image_servo` on this observation" in instruction


def test_motion_mismatch_revision_preserves_strict_gripper_threshold() -> None:
    from adaptive import remote_driver

    instruction = remote_driver._proposal_revision_instruction(
        "base instruction with measured_end_finger_separation 0.002105",
        contradiction="motion_effect_mismatch",
        evidence=["gripper_state", "action_receipt"],
        suggested_correction="verify_gripper",
        confidence="high",
        rejected_draft={
            "kind": "move_joints",
            "targets": {"gripper": 0.0},
            "note": "milestone=engage; use image_servo with gripper open",
        },
        required_observation_id="obs-0",
    ).casefold()

    assert "applicable threshold supplied in public context" in instruction
    assert "gripper-only `move_joints`" in instruction
    assert "replacement `note`" in instruction
    assert "do not copy `image_servo`" in instruction
    assert "opposite gripper intent" in instruction
    assert "do not round" in instruction


def test_ten_proposals_exhaust_without_executing_rejections() -> None:
    from adaptive import critic_protocol as protocol
    from adaptive import remote_driver

    context = _context()
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    _FakeClient.controller_outputs = [
        _milestone_command("obs-0", "observe", value)
        for value in (0.001 * (i + 1) for i in range(10))
    ]
    _FakeClient.critic_outputs = [_proposal_audit("revise") for _ in range(10)]

    with pytest.raises(protocol.ProposalAuditExhausted):
        _proposal_complete(wrapped, "obs-0", _state(), _images())

    assert context.proposal_revisions_used == 9
    assert len(context.proposal_records) == 10
    assert all(record["executed"] is False for record in context.proposal_records)
    assert all(record["mailbox_count"] == 0 for record in context.proposal_records)
    with pytest.raises(protocol.ProposalAuditExhausted):
        _proposal_complete(wrapped, "obs-0", _state(), _images())
    assert len(_FakeClient.calls) == 20


def test_critic_failure_fails_closed_without_returning_controller_draft() -> None:
    from adaptive import critic_protocol as protocol
    from adaptive import remote_driver

    context = _context()
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    _FakeClient.controller_outputs = [_milestone_command("obs-0", "observe")]
    _FakeClient.critic_error = TimeoutError("critic unavailable")

    with pytest.raises(protocol.ProposalAuditFailedClosed):
        _proposal_complete(wrapped, "obs-0", _state(), _images())

    assert context.critic_unavailable == 1
    assert len(context.proposal_records) == 1
    assert context.proposal_records[0]["status"] == "critic_unavailable"
    assert context.proposal_records[0]["executed"] is False
    assert context.proposal_records[0]["mailbox_count"] == 0
    with pytest.raises(ValueError, match="critic.*AttemptEvidenceLog"):
        protocol.validate_proposal_audit_closure(context)


def test_critic_attempt_linkage_failure_persists_rejected_draft_and_attempt() -> None:
    from adaptive import critic_protocol as protocol
    from adaptive import remote_driver

    context = _context()
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    draft = _milestone_command("obs-0", "observe")
    bad_attempt = _attempt_record(
        "a" * 64,
        observation_id="wrong-observation",
        response_schema_sha256=_fake_response_schema_sha256(
            protocol.PROPOSAL_AUDIT_SCHEMA
        ),
    )
    _FakeClient.controller_outputs = [draft]
    _FakeClient.critic_outputs = [_proposal_audit()]
    _FakeClient.critic_attempt_records = [[bad_attempt]]

    with pytest.raises(protocol.ProposalAuditFailedClosed):
        _proposal_complete(wrapped, "obs-0", _state(), _images())

    record = context.proposal_records[0]
    assert record["draft"] == draft
    assert record["status"] == "critic_unavailable"
    assert record["critic_attempt_record_sha256"] == _json_sha256(bad_attempt)
    assert record["executed"] is False
    assert (record["mailbox_count"], record["action_count"], record["receipt_count"]) == (0, 0, 0)


def test_controller_attempt_linkage_failure_persists_nonexecuted_draft() -> None:
    from adaptive import critic_protocol as protocol
    from adaptive import remote_driver

    context = _context()
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    draft = _milestone_command("obs-0", "observe")
    bad_attempt = _attempt_record(
        "a" * 64,
        observation_id="wrong-observation",
        response_schema_sha256=_fake_response_schema_sha256(
            CONTROLLER_RESPONSE_SCHEMA
        ),
    )
    _FakeClient.controller_outputs = [draft]
    _FakeClient.controller_attempt_records = [[bad_attempt]]

    with pytest.raises(protocol.ProposalAuditFailedClosed):
        _proposal_complete(wrapped, "obs-0", _state(), _images())

    assert len(context.proposal_records) == 1
    record = context.proposal_records[0]
    assert record["draft"] == draft
    assert record["status"] == "controller_linkage_incomplete"
    assert record["controller_attempt_record_sha256"] == _json_sha256(bad_attempt)
    assert record["executed"] is False
    assert record["mailbox_count"] == 0


@pytest.mark.parametrize(
    ("family", "graph"),
    (
        (
            "grasp_place",
            ("observe", "approach", "pregrasp", "grasp", "transport", "release", "verify_goal"),
        ),
        ("articulated", ("observe", "approach", "engage", "actuate", "verify_goal")),
        ("control", ("observe", "approach", "engage", "actuate", "verify_goal")),
    ),
)
def test_milestone_graph_allows_only_remain_or_one_edge(
    family: str, graph: tuple[str, ...]
) -> None:
    from adaptive import critic_protocol as protocol

    assert protocol.MILESTONE_GRAPHS[family] == graph
    assert protocol.validate_milestone_transition(family, graph[0], []) == graph[0]
    for index, milestone in enumerate(graph):
        history = list(graph[: index + 1])
        assert protocol.validate_milestone_transition(
            family, milestone, history
        ) == milestone
        if index + 1 < len(graph):
            assert protocol.validate_milestone_transition(
                family, graph[index + 1], history
            ) == graph[index + 1]
        if index + 2 < len(graph):
            with pytest.raises(ValueError, match="skip"):
                protocol.validate_milestone_transition(
                    family, graph[index + 2], history
                )
        if index > 0:
            with pytest.raises(ValueError, match="regress"):
                protocol.validate_milestone_transition(
                    family, graph[index - 1], history
                )


@pytest.mark.parametrize(
    "draft",
    (
        {"kind": "image_servo", "gripper": "hold"},
        {"kind": "cartesian_delta", "gripper": "close"},
        {"kind": "move_joints", "targets": {"gripper": 0.0}},
    ),
)
def test_release_milestone_requires_controller_authored_open_gripper(
    draft: dict[str, object],
) -> None:
    from adaptive import critic_protocol as protocol

    with pytest.raises(ValueError, match="release milestone requires.*open"):
        protocol.validate_milestone_action_semantics(
            "grasp_place", "release", draft
        )


@pytest.mark.parametrize(
    "draft",
    (
        {"kind": "image_servo", "gripper": "open"},
        {"kind": "cartesian_delta", "gripper": "open"},
        {"kind": "move_joints", "targets": {"gripper": 1.0}},
    ),
)
def test_release_milestone_accepts_each_controller_open_form(
    draft: dict[str, object],
) -> None:
    from adaptive import critic_protocol as protocol

    assert (
        protocol.validate_milestone_action_semantics(
            "grasp_place", "release", draft
        )
        is None
    )


def test_nonrelease_milestone_does_not_impose_gripper_semantics() -> None:
    from adaptive import critic_protocol as protocol

    assert (
        protocol.validate_milestone_action_semantics(
            "grasp_place",
            "transport",
            {"kind": "image_servo", "gripper": "hold"},
        )
        is None
    )


def test_articulated_actuation_lost_contact_requires_reengagement() -> None:
    from adaptive import critic_protocol as protocol
    from adaptive import remote_driver

    receipt = {
        "kind": "cartesian_delta",
        "requested_gripper": "hold",
        "gripper_residual": {"measured_end_finger_separation": 0.001},
    }
    actuation = {
        "kind": "cartesian_delta",
        "translation_m": [0.0, 0.0, -0.03],
        "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        "gripper": "hold",
    }
    recovery = {
        "kind": "image_servo",
        "camera": "right",
        "target_pixel": [128.0, 160.0],
        "target_role": "fixture_handle",
        "depth_delta_m": -0.01,
        "step_m": 0.01,
        "gripper": "close",
    }
    retreat = {
        "kind": "cartesian_delta",
        "translation_m": [0.03, 0.0, 0.02],
        "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        "gripper": "open",
    }

    with pytest.raises(ValueError, match="actuation lost contact retry must") as caught:
        protocol.validate_milestone_action_semantics(
            "articulated",
            "actuate",
            actuation,
            milestone_history=["observe", "approach", "engage", "actuate"],
            immediate_prior_receipt=receipt,
        )
    assert (
        protocol.validate_milestone_action_semantics(
            "articulated",
            "actuate",
            recovery,
            milestone_history=["observe", "approach", "engage", "actuate"],
            immediate_prior_receipt=receipt,
        )
        is None
    )
    assert (
        protocol.validate_milestone_action_semantics(
            "articulated",
            "actuate",
            retreat,
            milestone_history=["observe", "approach", "engage", "actuate"],
            immediate_prior_receipt=receipt,
        )
        is None
    )
    assert remote_driver._protocol_rejection({}, error=caught.value) == {
        "contradiction": "actuation_unverified",
        "evidence": ["gripper_state", "action_receipt"],
        "suggested_correction": "verify_actuation",
        "confidence": "high",
    }


def test_reestablished_contact_requires_new_actuation_axes() -> None:
    from adaptive import critic_protocol as protocol
    from adaptive import remote_driver

    failed_vertical = {
        "kind": "cartesian_delta",
        "requested_translation_m": [0.0, 0.0, 0.02],
        "requested_rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        "requested_gripper": "hold",
        "gripper_residual": {"measured_end_finger_separation": 0.001},
    }
    contact = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [128.0, 128.0],
        "requested_depth_delta_m": -0.01,
        "requested_gripper": "close",
        "gripper_residual": {"measured_end_finger_separation": 0.006},
    }
    confirmed = {
        "kind": "move_joints",
        "requested_targets": {"gripper": 0.0},
        "gripper_residual": {
            "measured_end_finger_separation": 0.006,
            "qpos_delta": [0.0, 0.0],
        },
    }
    reversed_vertical = {
        "kind": "cartesian_delta",
        "translation_m": [0.0, 0.0, -0.02],
        "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        "gripper": "hold",
    }
    horizontal_pull = {
        **reversed_vertical,
        "translation_m": [-0.02, 0.0, 0.0],
    }
    receipts = [failed_vertical, contact, confirmed]

    assert protocol.articulated_failed_actuation_axis_sets(receipts) == [
        ["translation_z"]
    ]
    with pytest.raises(
        ValueError, match="change motion axes after contact loss"
    ) as caught:
        protocol.validate_milestone_action_semantics(
            "articulated",
            "actuate",
            reversed_vertical,
            milestone_history=["observe", "approach", "engage", "actuate"],
            immediate_prior_receipt=confirmed,
            recent_receipts=receipts,
        )
    assert remote_driver._protocol_rejection({}, error=caught.value) == {
        "contradiction": "actuation_unverified",
        "evidence": ["gripper_state", "action_receipt"],
        "suggested_correction": "verify_actuation",
        "confidence": "high",
    }
    assert protocol.validate_milestone_action_semantics(
        "articulated",
        "actuate",
        horizontal_pull,
        milestone_history=["observe", "approach", "engage", "actuate"],
        immediate_prior_receipt=confirmed,
        recent_receipts=receipts,
    ) is None


def test_failed_actuation_axis_revision_requires_a_new_component() -> None:
    from adaptive import remote_driver

    failed_vertical = {
        "kind": "cartesian_delta",
        "requested_translation_m": [0.0, 0.0, 0.02],
        "requested_rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        "requested_gripper": "hold",
        "gripper_residual": {"measured_end_finger_separation": 0.001},
    }
    contact = {
        "kind": "move_joints",
        "requested_targets": {"gripper": 0.0},
        "gripper_residual": {
            "measured_end_finger_separation": 0.006,
            "qpos_delta": [0.0, 0.0],
        },
    }
    instruction = remote_driver._proposal_revision_instruction(
        "base instruction",
        contradiction="actuation_unverified",
        evidence=["gripper_state", "action_receipt"],
        suggested_correction="verify_actuation",
        confidence="high",
        rejected_draft={
            "kind": "cartesian_delta",
            "translation_m": [0.0, 0.0, -0.02],
            "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
            "gripper": "hold",
            "note": "milestone=actuate; reverse the same vertical axis",
        },
        required_observation_id="obs-recontact",
        immediate_prior_receipt=contact,
        recent_receipts=[failed_vertical, contact],
    ).casefold()

    assert "same motion-axis set" in instruction
    assert "changing only its sign" in instruction
    assert "different active cartesian axis" in instruction
    assert 'failed motion axes=["translation_z"]' in instruction
    assert "any vector with only `translation_z` active will be rejected" in (
        instruction
    )
    assert "activate `translation_x`, `translation_y`, or a rotation axis" in (
        instruction
    )


def test_lost_contact_actuation_revision_requires_contact_servo() -> None:
    from adaptive import remote_driver

    receipt = {
        "kind": "cartesian_delta",
        "requested_gripper": "hold",
        "gripper_residual": {"measured_end_finger_separation": 0.001},
    }
    instruction = remote_driver._proposal_revision_instruction(
        "base instruction",
        contradiction="actuation_unverified",
        evidence=["gripper_state", "action_receipt"],
        suggested_correction="verify_actuation",
        confidence="high",
        rejected_draft={
            "kind": "cartesian_delta",
            "translation_m": [0.0, 0.0, -0.03],
            "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
            "gripper": "hold",
            "note": "milestone=actuate; prior actuation lost contact",
        },
        required_observation_id="obs-0",
        immediate_prior_receipt=receipt,
    ).casefold()

    assert "contact was lost" in instruction
    assert "reacquire with `target_role=fixture_handle`" in instruction
    assert "positive `depth_delta_m`" in instruction
    assert "must say that positive depth advances" in instruction
    assert "must not say negative depth or retreat" in instruction
    assert "retreat with a bounded nonzero `cartesian_delta`" in instruction
    assert "do not use `target_role=articulation_motion` again" in instruction


def test_articulation_motion_without_current_contact_requests_recovery() -> None:
    from adaptive import remote_driver

    record: dict[str, object] = {}
    advice = remote_driver._protocol_rejection(
        record,
        error=ValueError(
            "articulation-motion image servo requires sealed articulated contact"
        ),
    )

    assert advice == {
        "contradiction": "actuation_unverified",
        "evidence": ["gripper_state", "action_receipt"],
        "suggested_correction": "verify_actuation",
        "confidence": "high",
    }
    assert record["contradiction"] == "actuation_unverified"
    assert record["suggested_correction"] == "verify_actuation"


def test_contact_recovery_counts_articulation_waypoint_to_handle_role_change() -> None:
    from adaptive.critic_protocol import articulated_contact_recovery_status

    lost_after_articulation = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [100.0, 130.0],
        "requested_target_role": "articulation_motion",
        "requested_depth_delta_m": 0.005,
        "requested_gripper": "hold",
        "gripper_residual": {"measured_end_finger_separation": 0.001},
    }
    reacquire_handle = {
        "kind": "image_servo",
        "camera": "right",
        "target_pixel": [100.0, 110.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.01,
        "step_m": 0.01,
        "gripper": "close",
    }

    recovery = articulated_contact_recovery_status(
        "actuate",
        reacquire_handle,
        lost_after_articulation,
        [lost_after_articulation],
    )

    assert recovery is not None
    assert recovery["required"] is True
    assert recovery["target_changed"] is True
    assert recovery["strategy_changed"] is True


def test_first_transport_rejects_sealed_empty_grasp() -> None:
    from adaptive import critic_protocol as protocol

    receipt = {
        "kind": "image_servo",
        "requested_gripper": "close",
        "gripper_residual": {"measured_end_finger_separation": 0.0016},
    }
    with pytest.raises(ValueError, match="transport milestone requires.*grasp"):
        protocol.validate_milestone_action_semantics(
            "grasp_place",
            "transport",
            {"kind": "cartesian_delta", "gripper": "hold"},
            milestone_history=["observe", "approach", "pregrasp", "grasp"],
            immediate_prior_receipt=receipt,
        )


@pytest.mark.parametrize("separation", [0.0023665, 0.0024, 0.0032])
def test_source_transport_rejects_measured_empty_panda_separation(separation: float) -> None:
    from adaptive.critic_protocol import validate_milestone_action_semantics

    with pytest.raises(ValueError, match="sealed nonempty grasp"):
        validate_milestone_action_semantics(
            "grasp_place", "transport",
            {"kind": "cartesian_delta", "gripper": "hold"},
            milestone_history=["observe", "approach", "pregrasp", "grasp"],
            immediate_prior_receipt={
                "kind": "image_servo", "requested_gripper": "close",
                "requested_target_role": "source_object",
                "gripper_residual": {"measured_end_finger_separation": separation},
            },
        )


def _source_empty_retreat_reapproach_receipts() -> list[dict[str, object]]:
    empty = {
        "kind": "image_servo", "requested_camera": "right",
        "requested_target_pixel": [192.0, 112.0],
        "requested_target_role": "source_object",
        "requested_depth_delta_m": 0.01, "requested_gripper": "close",
        "gripper_residual": {"measured_end_finger_separation": 0.0023665},
    }
    retreat = {
        "kind": "cartesian_delta", "requested_gripper": "open",
        "resolved_translation_m": [0.0, 0.0, 0.03],
        "gripper_residual": {"measured_end_finger_separation": 0.0776336},
    }
    reapproach = {
        **empty, "requested_gripper": "open",
        "gripper_residual": {"measured_end_finger_separation": 0.0792767},
    }
    return [empty, retreat, reapproach]


def test_source_empty_contact_survives_retreat_and_open_reapproach() -> None:
    from adaptive.critic_protocol import validate_milestone_action_semantics

    receipts = _source_empty_retreat_reapproach_receipts()
    repeated = {
        "kind": "image_servo", "camera": "right", "target_pixel": [192, 112],
        "target_role": "source_object", "depth_delta_m": 0.01, "gripper": "close",
    }
    with pytest.raises(ValueError, match="empty-grasp retry must change"):
        validate_milestone_action_semantics(
            "grasp_place", "grasp", repeated, recent_receipts=receipts,
            immediate_prior_receipt=receipts[-1],
        )


def _source_stationary_empty_recovery_receipts() -> list[dict[str, object]]:
    approach = {
        "kind": "image_servo", "requested_camera": "right",
        "requested_target_pixel": [100.0, 128.0],
        "requested_target_role": "source_object", "requested_depth_delta_m": 0.01,
        "requested_gripper": "open",
        "gripper_residual": {"measured_end_finger_separation": 0.07964681833982468},
    }
    close = {
        "kind": "move_joints", "requested_targets": {"gripper": 0.0},
        "gripper_intent": 0.0,
        "gripper_residual": {"measured_end_finger_separation": 0.002366592059843242},
    }
    retreat = {
        "kind": "cartesian_delta", "requested_gripper": "open",
        "resolved_translation_m": [0.0, 0.0, 0.03],
        "gripper_residual": {"measured_end_finger_separation": 0.07763298228383064},
    }
    return [approach, close, retreat, dict(approach)]


@pytest.mark.parametrize("draft", [
    {"kind": "move_joints", "targets": {"gripper": 0.0}},
    {"kind": "image_servo", "camera": "right", "target_pixel": [100, 128],
     "target_role": "source_object", "depth_delta_m": 0.01, "gripper": "close"},
])
def test_stationary_source_empty_close_blocks_unchanged_contact_retry(draft: dict) -> None:
    from adaptive.critic_protocol import validate_milestone_action_semantics

    receipts = _source_stationary_empty_recovery_receipts()
    with pytest.raises(ValueError, match="empty-grasp retry must change"):
        validate_milestone_action_semantics(
            "grasp_place", "grasp", draft,
            recent_receipts=receipts, immediate_prior_receipt=receipts[-1],
        )


def test_stationary_source_empty_close_retains_the_executed_source_target() -> None:
    from adaptive.critic_protocol import source_grasp_recovery_target

    target = source_grasp_recovery_target(_source_stationary_empty_recovery_receipts())
    assert target == {
        "camera": "right", "target_pixel": [100.0, 128.0], "other_view_pixel": None,
        "depth_delta_m": 0.01,
        "measured_end_finger_separation": 0.002366592059843242,
    }


@pytest.mark.parametrize("revision", [
    {"requested_target_pixel": [102.0, 130.0]},
    {"requested_depth_delta_m": 0.02},
    {"requested_other_view_pixel": [195.0, 110.0], "stereo_ray_gap_m": 0.02},
])
def test_stationary_source_retry_accepts_executed_geometry_revision(revision: dict) -> None:
    from adaptive.critic_protocol import validate_milestone_action_semantics

    receipts = _source_stationary_empty_recovery_receipts()
    receipts[-1] = {**receipts[-1], **revision}
    validate_milestone_action_semantics(
        "grasp_place", "grasp", {"kind": "move_joints", "targets": {"gripper": 0.0}},
        recent_receipts=receipts, immediate_prior_receipt=receipts[-1],
    )


def test_stationary_source_contact_does_not_inherit_target_across_arm_motion() -> None:
    from adaptive.critic_protocol import source_grasp_recovery_target

    approach, close, retreat, _ = _source_stationary_empty_recovery_receipts()
    assert source_grasp_recovery_target([approach, retreat, close]) is None
    assert source_grasp_recovery_target([
        approach, {"kind": "move_joints", "requested_targets": {"joint2": -0.7}}, close,
    ]) is None


def test_stationary_source_reclose_requires_reapproach_after_retreat() -> None:
    from adaptive.critic_protocol import validate_milestone_action_semantics

    receipts = _source_stationary_empty_recovery_receipts()[:-1]
    with pytest.raises(ValueError, match="empty-grasp retry must change"):
        validate_milestone_action_semantics(
            "grasp_place", "grasp", {"kind": "move_joints", "targets": {"gripper": 0.0}},
            recent_receipts=receipts, immediate_prior_receipt=receipts[-1],
        )


def test_stationary_nonempty_source_close_supersedes_older_empty_contact() -> None:
    from adaptive.critic_protocol import source_grasp_recovery_target

    receipts = _source_stationary_empty_recovery_receipts()
    receipts.append({
        **receipts[1], "gripper_residual": {"measured_end_finger_separation": 0.0066},
    })
    assert source_grasp_recovery_target(receipts) is None


def test_stationary_source_close_emits_recovery_facts_before_controller_retry() -> None:
    from adaptive import remote_driver

    context = _context()
    context.task = "PickPlaceCounterToStandMixer"
    context.family = "grasp_place"
    context.milestone_history = ["observe", "approach", "pregrasp", "grasp"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    receipts = _source_stationary_empty_recovery_receipts()
    for index in (0, -1):
        receipts[index]["requested_target_pixel"] = [60.0, 50.0]
    repeated = {
        "kind": "move_joints", "observation_id": "obs-stationary-source-retry",
        "targets": {"gripper": 0.0}, "note": "milestone=grasp; test source contact",
    }
    revised = {
        **_image_servo_payload(), "observation_id": "obs-stationary-source-retry",
        "camera": "right", "target_pixel": [60, 50], "depth_delta_m": 0.02,
        "gripper": "open", "note": "milestone=grasp; change source approach depth",
    }
    _FakeClient.controller_outputs = [repeated, revised]
    _FakeClient.critic_outputs = [_proposal_audit()]
    state = _state()
    state.update(_image_servo_public_state())
    response = _proposal_complete(
        wrapped, "obs-stationary-source-retry", state, _images(), receipts=receipts,
        camera_calibration=_image_servo_calibration(),
    )
    assert response.command is revised
    assert context.proposal_records[0]["status"] == "rejected_by_protocol"
    assert context.proposal_records[0]["action_count"] == 0
    instruction = _FakeClient.calls[0][1]["instruction"]
    packet = json.loads(instruction.split("SOURCE_GRASP_RECOVERY_CONTEXT:\n", 1)[1].split("\n", 1)[0])
    assert "target_pixel" not in packet and "depth_delta_m" not in packet
    assert context.active_source_failure["failed_approach"]["target_pixel"] == [60.0, 50.0]
    assert context.active_source_failure["failed_approach"]["depth_delta_m"] == 0.01
    skills = json.loads(instruction.split("QWEN_FAILURE_SKILLS:\n", 1)[1].split("\n", 1)[0])
    assert "target_pixel" not in skills["recent_cases"][-1]["failed_approach"]
    assert packet["measured_end_finger_separation"] == 0.002366592059843242
    critic_call = next(kwargs for role, kwargs in _FakeClient.calls if role == "critic")
    critic_packet = json.loads(critic_call["instruction"])["source_grasp_recovery"]
    assert all(critic_packet[key] == value for key, value in packet.items())
    assert critic_packet["same_point_with_changed_depth_or_stereo_allowed"] is True


@pytest.mark.parametrize("receipt_count", [2, 3, 4])
@pytest.mark.parametrize("target_pixel", [[100, 128], [100, 135]])
def test_critic_source_recovery_retains_empty_close_through_open_motion(
    receipt_count: int, target_pixel: list[int],
) -> None:
    from adaptive import remote_driver

    context = _context()
    context.task = "PickPlaceCounterToStandMixer"
    context.family = "grasp_place"
    receipts = _source_stationary_empty_recovery_receipts()[:receipt_count]
    draft = {
        **_image_servo_payload(), "target_role": "source_object",
        "camera": "right", "target_pixel": target_pixel, "depth_delta_m": 0.02,
        "gripper": "open", "note": "milestone=grasp; change depth at the same source point",
    }
    instruction = remote_driver._proposal_critic_instruction(
        context, task_instruction="Move the cake to the mixer", observation_id="obs-recovery",
        draft=draft, claimed_milestone="grasp", public_state=_state(), images=_images(),
        immediate_prior_receipt=receipts[-1], recent_receipts=receipts,
    )
    facts = json.loads(instruction)["source_grasp_recovery"]
    assert facts["target_pixel"] == [100.0, 128.0]
    assert facts["depth_delta_m"] == 0.01
    assert facts["measured_end_finger_separation"] == 0.002366592059843242
    assert facts["empty_close_confirmed"] is True
    assert facts["absolute_empty_contact_max_separation_m"] == 0.0032
    assert facts["same_point_with_changed_depth_or_stereo_allowed"] is True
    assert facts["failed_contact_persists_through_open_motion"] is True
    assert facts["open_corrective_source_servo"] is True
    assert facts["alignment_is_future_effect_for_open_reapproach"] is True
    assert facts["source_identity_must_remain_visually_grounded"] is True
    assert "source-specific threshold and retry rule" in facts["instruction"]


@pytest.mark.parametrize("family,task,contact", [
    ("control", "StartCoffeeMachine", "empty"),
    ("grasp_place", "PickPlaceCounterToStandMixer", "none"),
    ("grasp_place", "PickPlaceCounterToStandMixer", "nonempty"),
])
def test_source_recovery_instructions_require_applicable_failed_contact(
    family: str, task: str, contact: str,
) -> None:
    from adaptive import remote_driver

    context = _context()
    context.family = family
    context.task = task
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    receipts = _source_stationary_empty_recovery_receipts() if contact != "none" else []
    if contact == "nonempty":
        receipts[1]["gripper_residual"]["measured_end_finger_separation"] = 0.0066
    command = {
        "kind": "move_joints", "observation_id": "obs-no-source-recovery",
        "targets": {"gripper": 1.0}, "note": "milestone=observe; open gripper",
    }
    _FakeClient.controller_outputs = [command]
    _FakeClient.critic_outputs = [_proposal_audit()]
    _proposal_complete(wrapped, "obs-no-source-recovery", _state(), _images(), receipts=receipts)
    assert "SOURCE_GRASP_RECOVERY_CONTEXT" not in _FakeClient.calls[0][1]["instruction"]
    critic_instruction = remote_driver._proposal_critic_instruction(
        context, task_instruction="Task", observation_id="obs-no-source-recovery",
        draft=command, claimed_milestone="observe", public_state=_state(), images=_images(),
        immediate_prior_receipt=receipts[-1] if receipts else None, recent_receipts=receipts,
    )
    assert "source_grasp_recovery" not in json.loads(critic_instruction)


@pytest.mark.parametrize("revision", [
    {"target_pixel": [188, 118]},
    {"depth_delta_m": 0.02},
    {"other_view_pixel": [130, 110]},
    {"gripper": "open"},
])
def test_source_recovery_allows_controller_contact_geometry_revision(revision: dict) -> None:
    from adaptive.critic_protocol import validate_milestone_action_semantics

    receipts = _source_empty_retreat_reapproach_receipts()
    draft = {
        "kind": "image_servo", "camera": "right", "target_pixel": [192, 112],
        "target_role": "source_object", "depth_delta_m": 0.01, "gripper": "close",
        **revision,
    }
    validate_milestone_action_semantics(
        "grasp_place", "grasp", draft, recent_receipts=receipts,
        immediate_prior_receipt=receipts[-1],
    )


def test_source_nonempty_close_clears_prior_empty_contact() -> None:
    from adaptive.critic_protocol import validate_milestone_action_semantics

    receipts = _source_empty_retreat_reapproach_receipts()
    held = {
        **receipts[0],
        "gripper_residual": {"measured_end_finger_separation": 0.0066},
    }
    validate_milestone_action_semantics(
        "grasp_place", "transport", {"kind": "cartesian_delta", "gripper": "hold"},
        milestone_history=["observe", "approach", "pregrasp", "grasp"],
        recent_receipts=[*receipts, held], immediate_prior_receipt=held,
    )
    validate_milestone_action_semantics(
        "grasp_place", "grasp", {
            "kind": "image_servo", "camera": "right", "target_pixel": [192, 112],
            "target_role": "source_object", "depth_delta_m": 0.01, "gripper": "close",
        }, recent_receipts=[*receipts, held], immediate_prior_receipt=held,
    )


def test_source_recovery_does_not_change_control_actions() -> None:
    from adaptive.critic_protocol import validate_milestone_action_semantics

    receipts = _source_empty_retreat_reapproach_receipts()
    validate_milestone_action_semantics(
        "control", "actuate", {
            "kind": "image_servo", "camera": "right", "target_pixel": [192, 112],
            "target_role": "control_target", "depth_delta_m": 0.01, "gripper": "close",
        }, recent_receipts=receipts, immediate_prior_receipt=receipts[-1],
    )


def test_source_recovery_packet_and_rejection_preserve_qwen_authored_depth() -> None:
    from adaptive import remote_driver

    context = _context()
    context.task = "PickPlaceCounterToStandMixer"
    context.family = "grasp_place"
    context.milestone_history = ["observe", "approach", "pregrasp", "grasp"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    repeated = {
        **_image_servo_payload(), "observation_id": "obs-source-recovery",
        "camera": "right", "target_pixel": [60, 50], "target_role": "source_object",
        "depth_delta_m": 0.01, "gripper": "close", "note": "milestone=grasp; retry contact",
    }
    revised = {**repeated, "depth_delta_m": 0.02}
    receipts = _source_empty_retreat_reapproach_receipts()
    receipts[0]["requested_target_pixel"] = [60.0, 50.0]
    receipts[-1]["requested_target_pixel"] = [60.0, 50.0]
    _FakeClient.controller_outputs = [repeated, revised]
    _FakeClient.critic_outputs = [_proposal_audit()]
    state = _state()
    state.update(_image_servo_public_state())

    response = _proposal_complete(
        wrapped, "obs-source-recovery", state, _images(), receipts=receipts,
        camera_calibration=_image_servo_calibration(),
    )

    assert response.command is revised
    assert context.proposal_records[0]["status"] == "rejected_by_protocol"
    assert context.proposal_records[0]["action_count"] == 0
    instruction = _FakeClient.calls[0][1]["instruction"]
    packet = json.loads(instruction.split("SOURCE_GRASP_RECOVERY_CONTEXT:\n", 1)[1].split("\n", 1)[0])
    assert "target_pixel" not in packet and "depth_delta_m" not in packet
    assert context.active_source_failure["failed_approach"]["target_pixel"] == [60.0, 50.0]
    assert context.active_source_failure["failed_approach"]["depth_delta_m"] == 0.01
    skills = json.loads(instruction.split("QWEN_FAILURE_SKILLS:\n", 1)[1].split("\n", 1)[0])
    assert "target_pixel" not in skills["recent_cases"][-1]["failed_approach"]
    assert packet["measured_end_finger_separation"] == 0.0023665


def test_source_recovery_new_stereo_still_requires_public_ray_agreement() -> None:
    from adaptive import remote_driver

    context = _context()
    context.task = "PickPlaceCounterToStandMixer"
    context.family = "grasp_place"
    context.milestone_history = ["observe", "approach", "pregrasp", "grasp"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    revised = {
        **_image_servo_payload(), "observation_id": "obs-source-stereo",
        "camera": "left", "target_pixel": [60, 45], "other_view_pixel": [10, 45],
        "depth_delta_m": 0.01, "gripper": "close",
        "note": "milestone=grasp; add matched stereo grounding for the same source",
    }
    invalid = {**revised, "other_view_pixel": [10, 80]}
    receipts = _source_empty_retreat_reapproach_receipts()
    for index in (0, -1):
        receipts[index]["requested_camera"] = "left"
        receipts[index]["requested_target_pixel"] = [60.0, 45.0]
    _FakeClient.controller_outputs = [invalid, revised]
    _FakeClient.critic_outputs = [_proposal_audit()]
    state = _state()
    state.update(_image_servo_public_state())
    calibration = _stereo_calibration()
    state["state.end_effector_external_pixels"]["right"]["u_px"] = 0.0

    response = _proposal_complete(
        wrapped, "obs-source-stereo", state, _images(), receipts=receipts,
        camera_calibration=calibration,
    )

    assert response.command is revised
    assert context.proposal_records[0]["status"] == "rejected_by_protocol"
    assert context.proposal_records[0]["action_count"] == 0
    assert context.proposal_records[1]["status"] == "approved_for_execution"


def test_first_transport_accepts_obstructed_gripper_and_later_transport() -> None:
    from adaptive import critic_protocol as protocol

    held_receipt = {
        "kind": "image_servo",
        "requested_gripper": "close",
        "gripper_residual": {"measured_end_finger_separation": 0.0066},
    }
    empty_receipt = {
        "kind": "cartesian_delta",
        "requested_gripper": "hold",
        "gripper_residual": {"measured_end_finger_separation": 0.001},
    }
    draft = {"kind": "cartesian_delta", "gripper": "hold"}

    assert (
        protocol.validate_milestone_action_semantics(
            "grasp_place",
            "transport",
            draft,
            milestone_history=["observe", "approach", "pregrasp", "grasp"],
            immediate_prior_receipt=held_receipt,
        )
        is None
    )
    assert (
        protocol.validate_milestone_action_semantics(
            "grasp_place",
            "transport",
            draft,
            milestone_history=[
                "observe",
                "approach",
                "pregrasp",
                "grasp",
                "transport",
            ],
            immediate_prior_receipt=empty_receipt,
        )
        is None
    )


def test_empty_grasp_retry_must_change_pixel_or_retreat() -> None:
    from adaptive import critic_protocol as protocol

    receipt = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [128.0, 160.0],
        "requested_gripper": "close",
        "gripper_residual": {"measured_end_finger_separation": 0.001},
    }
    repeated = {
        "kind": "image_servo",
        "camera": "right",
        "target_pixel": [128.0, 160.0],
        "gripper": "close",
    }
    changed = {**repeated, "target_pixel": [160.0, 160.0]}

    with pytest.raises(ValueError, match="empty-grasp retry must change"):
        protocol.validate_milestone_action_semantics(
            "grasp_place",
            "grasp",
            repeated,
            milestone_history=["observe", "approach", "pregrasp", "grasp"],
            immediate_prior_receipt=receipt,
        )
    assert (
        protocol.validate_milestone_action_semantics(
            "grasp_place",
            "grasp",
            changed,
            milestone_history=["observe", "approach", "pregrasp", "grasp"],
            immediate_prior_receipt=receipt,
        )
        is None
    )
    assert (
        protocol.validate_milestone_action_semantics(
            "grasp_place",
            "grasp",
            {"kind": "cartesian_delta", "gripper": "open"},
            milestone_history=["observe", "approach", "pregrasp", "grasp"],
            immediate_prior_receipt=receipt,
        )
        is None
    )


def test_repeated_empty_grasp_pixel_is_rejected_before_critic() -> None:
    from adaptive import remote_driver

    context = _context()
    context.task = "PickPlaceToasterToCounter"
    context.family = "grasp_place"
    context.milestone_history = ["observe", "approach", "pregrasp", "grasp"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    invalid = {
        **_image_servo_payload(),
        "observation_id": "obs-empty-retry",
        "camera": "right",
        "target_pixel": [60.0, 50.0],
        "target_role": "source_object",
        "depth_delta_m": -0.01,
        "gripper": "close",
        "note": "milestone=grasp; retrying the same visible source point",
    }
    valid = {**invalid, "target_pixel": [70.0, 50.0]}
    _FakeClient.controller_outputs = [invalid, valid]
    _FakeClient.critic_outputs = [_proposal_audit()]
    empty_grasp_receipt = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [60.0, 50.0],
        "requested_gripper": "close",
        "gripper_residual": {"measured_end_finger_separation": 0.001},
    }
    state = _state()
    state.update(_image_servo_public_state())

    response = _proposal_complete(
        wrapped,
        "obs-empty-retry",
        state,
        _images(),
        receipts=[empty_grasp_receipt],
        camera_calibration=_image_servo_calibration(),
    )

    assert response.command is valid
    assert [role for role, _ in _FakeClient.calls] == [
        "controller",
        "controller",
        "critic",
    ]
    rejected = context.proposal_records[0]
    assert rejected["status"] == "rejected_by_protocol"
    assert rejected["contradiction"] == "visual_alignment_unverified"
    assert rejected["suggested_correction"] == "revise_alignment"
    assert rejected["mailbox_count"] == rejected["action_count"] == 0
    revision_instruction = _FakeClient.calls[1][1]["instruction"].casefold()
    assert "changed nonzero depth" in revision_instruction
    assert "newly grounded stereo pair" in revision_instruction


def test_release_hold_is_rejected_before_critic_and_revision_opens_gripper() -> None:
    from adaptive import remote_driver

    context = _context()
    context.task = "PickPlaceToasterToCounter"
    context.family = "grasp_place"
    context.milestone_history = [
        "observe",
        "approach",
        "pregrasp",
        "grasp",
        "transport",
    ]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    invalid = {
        **_cartesian_milestone_command("obs-release", "release"),
        "gripper": "hold",
    }
    valid = {**invalid, "gripper": "open"}
    _FakeClient.controller_outputs = [invalid, valid]
    _FakeClient.critic_outputs = [_proposal_audit()]

    response = _proposal_complete(
        wrapped,
        "obs-release",
        _state(),
        _images(),
    )

    assert response.command is valid
    assert [role for role, _ in _FakeClient.calls] == [
        "controller",
        "controller",
        "critic",
    ]
    rejected = context.proposal_records[0]
    assert rejected["status"] == "rejected_by_protocol"
    assert rejected["contradiction"] == "release_unverified"
    assert rejected["suggested_correction"] == "verify_release"
    assert rejected["mailbox_count"] == rejected["action_count"] == 0
    revision_instruction = _FakeClient.calls[1][1]["instruction"]
    assert "must explicitly author gripper `open`" in revision_instruction.casefold()


def test_empty_grasp_transport_is_rejected_before_critic_and_retried() -> None:
    from adaptive import remote_driver

    context = _context()
    context.task = "PickPlaceToasterToCounter"
    context.family = "grasp_place"
    context.milestone_history = [
        "observe",
        "approach",
        "pregrasp",
        "grasp",
    ]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    invalid = _cartesian_milestone_command("obs-empty-grasp", "transport")
    retry = {
        **_cartesian_milestone_command(
            "obs-empty-grasp", "grasp", translation_x=0.005
        ),
        "gripper": "close",
    }
    _FakeClient.controller_outputs = [invalid, retry]
    _FakeClient.critic_outputs = [_proposal_audit()]
    empty_grasp_receipt = {
        "kind": "image_servo",
        "requested_gripper": "close",
        "gripper_residual": {"measured_end_finger_separation": 0.0016},
    }

    response = _proposal_complete(
        wrapped,
        "obs-empty-grasp",
        _state(),
        _images(),
        receipts=[empty_grasp_receipt],
    )

    assert response.command is retry
    assert [role for role, _ in _FakeClient.calls] == [
        "controller",
        "controller",
        "critic",
    ]
    rejected = context.proposal_records[0]
    assert rejected["status"] == "rejected_by_protocol"
    assert rejected["contradiction"] == "grasp_unverified"
    assert rejected["suggested_correction"] == "verify_gripper"
    assert rejected["mailbox_count"] == rejected["action_count"] == 0
    revision_instruction = _FakeClient.calls[1][1]["instruction"]
    assert "prior close at this handle point was empty" in (
        revision_instruction.casefold()
    )
    assert "reverse the depth direction, or retreat" in revision_instruction.casefold()


@pytest.mark.parametrize("terminal_kind", ("finish", "give_up"))
def test_terminal_proposal_must_revise_to_verify_goal_before_audit(
    terminal_kind: str,
) -> None:
    from adaptive import remote_driver

    context = _context()
    context.milestone_history = ["observe", "approach", "engage", "actuate"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    invalid = _milestone_command("obs-4", "actuate", kind=terminal_kind)
    valid = _milestone_command("obs-4", "verify_goal", kind=terminal_kind)
    _FakeClient.controller_outputs = [invalid, valid]
    _FakeClient.critic_outputs = [_proposal_audit()]

    response = _proposal_complete(wrapped, "obs-4", _state(), _images())

    assert response.command is valid
    assert [role for role, _ in _FakeClient.calls] == [
        "controller",
        "controller",
        "critic",
    ]
    assert context.proposal_records[0]["status"] == "rejected_by_protocol"
    assert context.proposal_records[0]["contradiction"] == (
        "terminal_unverified"
    )
    assert context.proposal_records[0]["executed"] is False
    assert context.proposal_records[1]["status"] == "approved_for_execution"


def test_runtime_invalid_draft_uses_recorded_same_observation_revision_path() -> None:
    from adaptive import remote_driver

    context = _context()
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    stale = _milestone_command("stale", "observe")
    valid = _milestone_command("obs-0", "observe", 0.02)
    _FakeClient.controller_outputs = [stale, valid]
    _FakeClient.critic_outputs = [_proposal_audit()]

    response = _proposal_complete(wrapped, "obs-0", _state(), _images())

    assert response.command is valid
    assert [role for role, _ in _FakeClient.calls] == [
        "controller",
        "controller",
        "critic",
    ]
    rejected = context.proposal_records[0]
    assert rejected["status"] == "rejected_by_protocol"
    assert rejected["contradiction"] == "stale_or_missing_evidence"
    assert rejected["executed"] is False
    assert (rejected["mailbox_count"], rejected["action_count"], rejected["receipt_count"]) == (0, 0, 0)


def test_revision_counter_resets_only_after_matching_execution_receipt() -> None:
    from adaptive import critic_protocol as protocol
    from adaptive import remote_driver

    context = _context()
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    rejected = _milestone_command("obs-0", "observe", 0.01)
    approved = _milestone_command("obs-0", "observe", 0.02)
    _FakeClient.controller_outputs = [rejected, approved]
    _FakeClient.critic_outputs = [_proposal_audit("revise"), _proposal_audit()]
    _proposal_complete(wrapped, "obs-0", _state(), _images())
    assert context.proposal_revisions_used == 1

    with pytest.raises(protocol.ProposalAuditFailedClosed, match="receipt"):
        _proposal_complete(wrapped, "obs-1", _state(0.02), _images())
    assert context.proposal_revisions_used == 1

    receipt = _sealed_critic_receipt(approved, _state(0.02))
    next_command = _milestone_command("obs-1", "approach", 0.03)
    _FakeClient.controller_outputs = [next_command]
    _FakeClient.critic_outputs = [_proposal_audit()]
    response = _proposal_complete(
        wrapped,
        "obs-1",
        _state(0.02),
        _images(),
        receipts=[receipt],
    )

    assert response.command is next_command
    assert context.proposal_revisions_used == 0
    first_approved_record = context.proposal_records[1]
    assert first_approved_record["executed"] is True
    assert first_approved_record["mailbox_count"] == 1
    assert first_approved_record["action_count"] == 1
    assert first_approved_record["receipt_count"] == 1
    assert context.milestone_history == ["observe"]


def test_proposal_attempt_evidence_and_hashes_form_exact_bijection() -> None:
    from adaptive import critic_protocol as protocol
    from adaptive import remote_driver

    context = _context()
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    first = _milestone_command("obs-0", "observe", 0.01)
    revised = _milestone_command("obs-0", "observe", 0.02)
    _FakeClient.controller_outputs = [first, revised]
    _FakeClient.critic_outputs = [_proposal_audit("revise"), _proposal_audit()]
    owner = _OwnerAttemptLog()

    _proposal_complete(
        wrapped, "obs-0", _state(), _images(), attempt_log=owner
    )
    summary = protocol.validate_proposal_audit_closure(context)

    assert summary == {
        "proposal_count": 2,
        "controller_attempt_count": 2,
        "critic_attempt_count": 2,
        "rejected_count": 1,
        "approved_count": 1,
        "executed_count": 0,
        "critic_origin_executions": 0,
    }
    assert len(owner.records) == 4
    assert len(context.attempt_records) == 4
    assert [record["role"] for record in context.attempt_records] == [
        "controller",
        "critic",
        "controller",
        "critic",
    ]
    assert all(set(record["record"]) == _ATTEMPT_EVIDENCE_FIELDS for record in context.attempt_records)
    assert all(record["record_sha256"] == _json_sha256(record["record"]) for record in context.attempt_records)


def test_runner_builds_exact_non_authoring_proposal_validation_context() -> None:
    from adaptive import joint_runner

    state = _state()
    assert joint_runner._proposal_audit_execution_context(
        state,
        _image_servo_calibration(),
        current_gripper=0.25,
        consumed_actions=17,
        sequence=3,
    ) == {
        "current_qpos": state["state.arm_joint_position"],
        "current_gripper": 0.25,
        "remaining_actions": 433,
        "sequence": 3,
        "camera_calibration": _image_servo_calibration(),
    }
    with pytest.raises(ValueError, match="consumed action"):
        joint_runner._proposal_audit_execution_context(
            state,
            _image_servo_calibration(),
            current_gripper=0.25,
            consumed_actions=True,
            sequence=3,
        )
    assert joint_runner._proposal_audit_execution_context(
        state,
        _image_servo_calibration(),
        current_gripper=0.25,
        consumed_actions=17,
        sequence=3,
        action_budget=160,
    )["remaining_actions"] == 143
    assert joint_runner._proposal_audit_execution_context(
        state,
        _image_servo_calibration(),
        current_gripper=0.25,
        consumed_actions=450,
        sequence=3,
        action_budget=900,
    )["remaining_actions"] == 450
    with pytest.raises(ValueError, match="episode budget"):
        joint_runner._proposal_audit_execution_context(
            state,
            _image_servo_calibration(),
            current_gripper=0.25,
            consumed_actions=161,
            sequence=3,
            action_budget=160,
        )


def test_runner_maps_proposal_failure_to_exact_nonexecution_record() -> None:
    from adaptive import critic_protocol as protocol
    from adaptive import joint_runner

    error = protocol.ProposalAuditExhausted("drafts rejected")
    record = joint_runner._proposal_audit_failure_request(
        decision=4,
        observation_id="obs-4",
        error=error,
    )

    assert record == {
        "schema": "robocasa-qwen-proposal-audit-failure/v1",
        "decision": 4,
        "observation_id": "obs-4",
        "status": "policy_failed_proposal_audit",
        "failure_class": "ProposalAuditExhausted",
        "failure_sha256": hashlib.sha256(b"drafts rejected").hexdigest(),
        "command": None,
        "mailbox_count": 0,
        "action_count": 0,
        "receipt_count": 0,
    }


def test_final_episode_receipt_closes_last_approved_proposal_without_new_decision() -> None:
    from adaptive import remote_driver

    context = _context()
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    command = _milestone_command("obs-0", "observe", 0.02)
    state1 = _state(0.02)
    _FakeClient.controller_outputs = [command]
    _FakeClient.critic_outputs = [_proposal_audit()]
    response = _proposal_complete(wrapped, "obs-0", _state(), _images())

    receipt = _sealed_critic_receipt(command, state1)
    mailbox, _decoded = prepare_joint_mailbox(
        command,
        source="controller",
        observation_id="obs-0",
        current_qpos=_state()["state.arm_joint_position"],
        current_gripper=1.0,
        remaining_actions=450,
        sequence=0,
    )
    remote_driver._finalize_proposal_execution_evidence(
        context,
        {
            "requests": [{
                "observation_id": "obs-0",
                "command": command,
                "post_repair_command": command,
                "evidence": response.evidence,
                "mailbox": mailbox,
                "mailbox_sha256": _json_sha256(mailbox),
            }],
            "receipts": [receipt],
        },
    )

    record = context.proposal_records[0]
    assert record["executed"] is True
    assert (record["mailbox_count"], record["action_count"], record["receipt_count"]) == (1, 1, 1)
    assert record["execution_receipt_sha256"] == _json_sha256(receipt)
    assert context.milestone_history == ["observe"]
    assert context.proposal_pending_record_index is None


def test_approved_image_servo_request_accepts_numeric_type_normalization_only() -> None:
    from adaptive import remote_driver

    draft = {
        "kind": "image_servo",
        "observation_id": "obs-pixel-types",
        "camera": "right",
        "target_pixel": [100, 110],
        "target_role": "fixture_handle",
        "depth_delta_m": -0.01,
        "step_m": 0.015,
        "gripper": "close",
        "note": "milestone=engage; target the visible handle",
    }
    audit_sha = _named_digest("audit")
    record = {
        "draft": draft,
        "draft_sha256": _json_sha256(draft),
        "observation_id": "obs-pixel-types",
        "controller_call_index": 4,
        "audit_sha256": audit_sha,
        "claimed_milestone": "engage",
    }
    normalized = {**draft, "target_pixel": [100.0, 110.0]}
    request = {
        "observation_id": "obs-pixel-types",
        "command": draft,
        "post_repair_command": normalized,
        "repair_count": 0,
        "evidence": {
            "controller_call_index": 4,
            "qwen_command_sha256": _json_sha256(draft),
            "proposal_audit_verdict": "approve",
            "proposal_audit_sha256": audit_sha,
            "proposal_audit_config_sha256": (
                remote_driver.PROPOSAL_AUDIT_CONFIG_SHA256
            ),
            "claimed_milestone": "engage",
            "critic_origin_execution": False,
        },
    }

    assert remote_driver._approved_runner_request_matches(request, record)
    changed = {**normalized, "target_pixel": [100.0, 111.0]}
    assert not remote_driver._approved_runner_request_matches(
        {**request, "post_repair_command": changed}, record
    )


def _review_closed_physical_context(
    *,
    reject_first: bool = True,
    served_model_id: str | None = None,
) -> CriticContext:
    from adaptive import remote_driver

    context = _context()
    if served_model_id is not None:
        _FakeClient.served_model_id = served_model_id
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    approved = _milestone_command("obs-0", "observe", 0.02)
    if reject_first:
        rejected = _milestone_command("obs-0", "observe", 0.01)
        _FakeClient.controller_outputs = [rejected, approved]
        _FakeClient.critic_outputs = [_proposal_audit("revise"), _proposal_audit()]
    else:
        _FakeClient.controller_outputs = [approved]
        _FakeClient.critic_outputs = [_proposal_audit()]
    response = _proposal_complete(wrapped, "obs-0", _state(), _images())
    receipt = _sealed_critic_receipt(approved, _state(0.02))
    mailbox, _decoded = prepare_joint_mailbox(
        approved,
        source="controller",
        observation_id="obs-0",
        current_qpos=_state()["state.arm_joint_position"],
        current_gripper=1.0,
        remaining_actions=450,
        sequence=0,
    )
    remote_driver._finalize_proposal_execution_evidence(
        context,
        {
            "requests": [{
                "observation_id": "obs-0",
                "command": approved,
                "post_repair_command": approved,
                "evidence": response.evidence,
                "mailbox": mailbox,
                "mailbox_sha256": _json_sha256(mailbox),
            }],
            "receipts": [receipt],
            "terminal_outcome": None,
        },
    )
    return context


def test_proposal_closure_accepts_runtime_normalized_duplicate_evidence() -> None:
    from adaptive import critic_protocol as protocol
    from adaptive import remote_driver

    context = _review_closed_physical_context(reject_first=False)
    attempt = next(
        record for record in context.attempt_records if record["role"] == "critic"
    )
    normalized = context.proposal_records[0]["audit"]
    assert isinstance(normalized, dict)
    duplicated = dict(normalized)
    evidence = list(normalized["evidence"])
    duplicated["evidence"] = [evidence[0], evidence[0], evidence[0]]
    stored = dict(normalized)
    stored["evidence"] = [evidence[0]]
    for record in (context.proposal_records[0], context.critic_records[0]):
        record["audit"] = stored
        record["audit_sha256"] = _json_sha256(stored)
    attempt["record"]["sanitized_raw_command"] = json.dumps(duplicated)
    attempt["record_sha256"] = _json_sha256(attempt["record"])
    context.proposal_records[0]["critic_attempt_record_sha256"] = attempt[
        "record_sha256"
    ]
    context.critic_records[0]["critic_attempt_record_sha256"] = attempt[
        "record_sha256"
    ]
    remote_driver._seal_evidence_record(context.proposal_records[0])
    remote_driver._seal_evidence_record(context.critic_records[0])

    summary = protocol.validate_proposal_audit_closure(context)

    assert summary["approved_count"] == 1
    assert context.proposal_records[0]["audit"]["evidence"] == [evidence[0]]


def test_proposal_closure_binds_attempts_to_attested_episode_model() -> None:
    from adaptive import critic_protocol as protocol

    context = _review_closed_physical_context()

    with pytest.raises(ValueError, match="attested served-model"):
        protocol.validate_proposal_audit_closure(
            context,
            require_execution_closure=True,
            expected_served_model_id="different-attested-model",
        )


def _finish_terminal_outcome(status: str) -> dict[str, object]:
    simulator_terminal = {
        "status": status,
        "sequence": 0,
        "terminal_outcome_sha256": _named_digest("official terminal outcome"),
        "terminal_snapshot_sha256": _named_digest("terminal snapshot"),
        "episode_tmp_empty": True,
        "simulator_actions": 0,
        "wall_s": 1.25,
    }
    return {
        "schema": "robocasa-inspect-joint-terminal/v1",
        "status": status,
        "success": status == "success",
        "simulator_terminal": simulator_terminal,
        "simulator_terminal_sha256": _json_sha256(simulator_terminal),
    }


@pytest.mark.parametrize(
    "mutation",
    (
        lambda context: (
            context.attempt_records[0]["record"].__setitem__(
                "sanitized_raw_command", "coherently tampered"
            ),
            context.attempt_records[0].__setitem__(
                "record_sha256", _json_sha256(context.attempt_records[0]["record"])
            ),
        ),
        lambda context: context.proposal_records[1].__setitem__(
            "returned_command_sha256", _named_digest("different-return")
        ),
        lambda context: context.proposal_records[1].__setitem__(
            "mailbox_count", True
        ),
        lambda context: context.proposal_records[1].__setitem__(
            "action_count", 1.0
        ),
        lambda context: context.proposal_records[1].__setitem__(
            "audit_sha256", _named_digest("different-audit")
        ),
        lambda context: context.proposal_records[1].__setitem__(
            "revision_index", False
        ),
        lambda context: context.proposal_records[1].__setitem__(
            "milestone_history_sha256", _named_digest("different-history")
        ),
        lambda context: context.controller_records[1].__setitem__(
            "controller_call_index", 1
        ),
    ),
    ids=(
        "coherent-attempt-record",
        "returned-differs-from-draft",
        "bool-counter-alias",
        "float-counter-alias",
        "audit-hash",
        "bool-revision-index",
        "milestone-history-hash",
        "controller-call-order",
    ),
)
def test_strict_proposal_closure_rejects_coherent_mutations(mutation: object) -> None:
    from adaptive import critic_protocol as protocol

    context = _review_closed_physical_context()
    mutation(context)

    with pytest.raises(ValueError):
        protocol.validate_proposal_audit_closure(context)


def test_strict_proposal_closure_rejects_globally_reordered_attempt_log() -> None:
    from adaptive import critic_protocol as protocol

    context = _review_closed_physical_context()
    context.attempt_records[0], context.attempt_records[1] = (
        context.attempt_records[1],
        context.attempt_records[0],
    )

    with pytest.raises(ValueError, match="order"):
        protocol.validate_proposal_audit_closure(context)


@pytest.mark.parametrize(
    ("role", "field", "value"),
    (
        ("controller", "finish_reason", "length"),
        ("controller", "completion_tokens", PROPOSAL_CONTROLLER_MAX_TOKENS),
        ("critic", "finish_reason", "length"),
        ("critic", "completion_tokens", CRITIC_MAX_TOKENS),
    ),
)
def test_strict_proposal_closure_rejects_coherently_resealed_truncation(
    role: str,
    field: str,
    value: object,
) -> None:
    from adaptive import critic_protocol as protocol

    context = _review_closed_physical_context()
    attempt = next(
        record
        for record in context.attempt_records
        if record["role"] == role and record["call_index"] == 1
    )
    attempt["record"][field] = value
    attempt["record_sha256"] = _json_sha256(attempt["record"])
    reference_field = f"{role}_attempt_record_sha256"
    context.proposal_records[0][reference_field] = attempt["record_sha256"]
    context.proposal_records[0]["record_sha256"] = _json_sha256({
        key: item
        for key, item in context.proposal_records[0].items()
        if key != "record_sha256"
    })
    context.critic_records[0][reference_field] = attempt["record_sha256"]
    context.critic_records[0]["record_sha256"] = _json_sha256({
        key: item
        for key, item in context.critic_records[0].items()
        if key != "record_sha256"
    })

    with pytest.raises(ValueError, match="completion"):
        protocol.validate_proposal_audit_closure(context)


def test_invalid_proposal_closure_forces_non_success_episode() -> None:
    from adaptive import remote_driver

    episode = {"status": "success", "success": True}
    remote_driver._apply_proposal_closure_result(
        episode,
        {"valid": False, "failure_sha256": _named_digest("closure failure")},
    )

    assert episode["status"] == "policy_failed_proposal_audit_closure"
    assert episode["success"] is False
    assert episode["pre_closure_status"] == "success"


def test_malformed_draft_consumes_revision_then_valid_draft_is_audited() -> None:
    from adaptive import remote_driver

    context = _context()
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    valid = _milestone_command("obs-0", "observe", 0.02)
    _FakeClient.controller_errors_after_log = [
        _FakeMalformedResponse("malformed json"),
        None,
    ]
    _FakeClient.controller_outputs = [valid]
    _FakeClient.critic_outputs = [_proposal_audit()]

    response = _proposal_complete(wrapped, "obs-0", _state(), _images())

    assert response.command is valid
    assert context.proposal_revisions_used == 1
    malformed, approved = context.proposal_records
    assert malformed["status"] == "rejected_by_protocol"
    assert malformed["executed"] is False
    assert (malformed["mailbox_count"], malformed["action_count"], malformed["receipt_count"]) == (0, 0, 0)
    assert approved["status"] == "approved_for_execution"


def test_malformed_valid_json_is_recorded_as_its_canonical_value() -> None:
    from adaptive import remote_driver

    context = _context()
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    valid = _milestone_command("obs-0", "observe", 0.02)
    _FakeClient.controller_errors_after_log = [
        _FakeMalformedResponse("schema-invalid json"),
        None,
    ]
    _FakeClient.controller_attempt_records = [[
        _attempt_record(
            "a" * 64,
            observation_id="obs-0",
            response_schema_sha256=_json_sha256(CONTROLLER_RESPONSE_SCHEMA),
        )
        | {"sanitized_raw_command": "{}"}
    ]]
    _FakeClient.controller_outputs = [valid]
    _FakeClient.critic_outputs = [_proposal_audit()]

    response = _proposal_complete(wrapped, "obs-0", _state(), _images())

    assert response.command is valid
    assert context.proposal_records[0]["draft"] == {}


def test_ten_malformed_drafts_exhaust_and_later_call_invokes_no_model() -> None:
    from adaptive import critic_protocol as protocol
    from adaptive import remote_driver

    context = _context()
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    _FakeClient.controller_errors_after_log = [
        _FakeMalformedResponse(f"bad {index}") for index in range(10)
    ]

    with pytest.raises(protocol.ProposalAuditExhausted):
        _proposal_complete(wrapped, "obs-0", _state(), _images())
    assert len(context.proposal_records) == 10
    assert context.proposal_revisions_used == 9
    calls = len(_FakeClient.calls)
    with pytest.raises(protocol.ProposalAuditExhausted):
        _proposal_complete(wrapped, "obs-0", _state(), _images())
    assert len(_FakeClient.calls) == calls


def test_proposal_failure_terminates_out_of_band_without_mailbox(tmp_path: Path) -> None:
    from adaptive import joint_runner

    class Process:
        terminated = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, *, timeout: int) -> int:
            assert timeout == 180
            return 0

    mailbox = tmp_path / "sim" / "mailbox"
    mailbox.mkdir(parents=True)
    process = Process()
    result = joint_runner._close_or_terminate_simulator(
        status="policy_failed_proposal_audit",
        process=process,
        mailbox_dir=mailbox,
        sequence=0,
    )

    assert result == "out_of_band_termination"
    assert process.terminated is True
    assert list(mailbox.iterdir()) == []


def test_proposal_failure_real_runner_path_has_zero_simulator_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive import critic_protocol as protocol
    from adaptive import joint_runner

    run = tmp_path / "episode"
    identity_path = tmp_path / "identity.json"
    attestation_path = tmp_path / "attestation.json"
    token_path = tmp_path / "token"
    identity_path.write_text("{}")
    attestation_path.write_text("{}")
    token_path.write_text("token")
    monkeypatch.setenv("QWEN_IDENTITY_MANIFEST", str(identity_path))
    monkeypatch.setenv("QWEN_SERVER_ATTESTATION", str(attestation_path))
    monkeypatch.setenv("QWEN_API_TOKEN_FILE", str(token_path))

    class FailingClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def verify(self) -> None:
            pass

        def complete(self, **_kwargs: object) -> object:
            raise protocol.ProposalAuditFailedClosed("proposal rejected")

        def close(self) -> None:
            pass

    class Process:
        return_code: int | None = None

        def poll(self) -> int | None:
            return self.return_code

        def terminate(self) -> None:
            self.return_code = -15

        def wait(self, *, timeout: int) -> int:
            assert timeout == 180
            assert self.return_code is not None
            return self.return_code

    process = Process()
    package = ModuleType("robocasa_inspect")
    model_client = ModuleType("robocasa_inspect.model_client")
    model_client.MalformedResponse = type("MalformedResponse", (Exception,), {})
    model_client.QwenClient = FailingClient
    model_client.load_authority = lambda *_args: (
        {"snapshot_digest": "snapshot"},
        {"served_model_id": "served-model"},
    )
    model_client.verify_process = lambda _attestation: None
    runner = ModuleType("robocasa_inspect.runner")
    runner._image_change = lambda *_args: {
        "left": 0.0,
        "right": 0.0,
        "wrist": 0.0,
    }
    runner._read_images = lambda _observation: _images()

    def render_video(_sim: Path, video: Path) -> int:
        video.write_bytes(b"proposal failure video")
        return 1

    runner._render_video = render_video
    runner._wait_process_json = lambda *_args, **_kwargs: {
        "sequence": 0,
        "observation_id": "obs-0",
        "instruction": "open the toaster",
        "public_state": _state(),
        "camera_calibration": {},
    }
    monkeypatch.setitem(sys.modules, "robocasa_inspect", package)
    monkeypatch.setitem(sys.modules, "robocasa_inspect.model_client", model_client)
    monkeypatch.setitem(sys.modules, "robocasa_inspect.runner", runner)

    def launch(**kwargs: object) -> Process:
        launched_run = kwargs["run"]
        assert isinstance(launched_run, Path)
        (launched_run / "sim" / "mailbox").mkdir(parents=True)
        return process

    monkeypatch.setattr(joint_runner, "_launch_simulator", launch)

    result = joint_runner.run_episode(
        task="OpenToasterOvenDoor",
        seed=7,
        run=run,
        max_decisions=12,
        client_class=FailingClient,
    )

    assert result["status"] == "policy_failed_proposal_audit"
    assert result["simulator_lifecycle"] == "out_of_band_termination"
    assert result["decisions"] == 0
    assert result["action_chunks"] == 0
    assert result["simulator_steps"] == 0
    assert result["receipts"] == []
    assert len(result["requests"]) == 1
    assert result["requests"][0]["mailbox_count"] == 0
    assert result["requests"][0]["action_count"] == 0
    assert result["requests"][0]["receipt_count"] == 0
    assert list((run / "sim" / "mailbox").iterdir()) == []


def test_runner_seals_the_exact_official_finish_outcome(tmp_path: Path) -> None:
    from adaptive import joint_runner

    simulator_terminal = _finish_terminal_outcome("success")[
        "simulator_terminal"
    ]
    sim = tmp_path / "sim"
    sim.mkdir()
    (sim / "simulator-terminal.json").write_text(
        json.dumps(simulator_terminal, sort_keys=True, separators=(",", ":"))
        + "\n"
    )

    status, outcome = joint_runner._read_terminal_outcome(tmp_path)

    assert status == "success"
    assert outcome == _finish_terminal_outcome("success")


def test_runner_launches_child_with_development_action_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from adaptive import joint_runner

    run = tmp_path / "run"
    run.mkdir()
    launched: dict[str, object] = {}

    def fake_popen(command: list[str], **kwargs: object) -> object:
        launched["command"] = command
        launched["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(joint_runner.subprocess, "Popen", fake_popen)
    with (tmp_path / "simulator.log").open("w") as log_handle:
        joint_runner._launch_simulator(
            task="OpenToasterOvenDoor",
            seed=7,
            run=run,
            log_handle=log_handle,
            action_budget=900,
        )

    command = launched["command"]
    assert isinstance(command, list)
    assert command[command.index("--action-budget") + 1] == "900"


def test_joint_child_cli_forwards_development_action_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from adaptive import joint_sim_child

    recorded: dict[str, object] = {}
    monkeypatch.setattr(
        joint_sim_child,
        "run",
        lambda task, seed, run, *, action_budget: recorded.update(
            task=task, seed=seed, run=run, action_budget=action_budget
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "joint_sim_child.py",
            "--task",
            "OpenToasterOvenDoor",
            "--seed",
            "7",
            "--run-dir",
            str(tmp_path / "sim"),
            "--action-budget",
            "900",
        ],
    )

    joint_sim_child.main()

    assert recorded == {
        "task": "OpenToasterOvenDoor",
        "seed": 7,
        "run": (tmp_path / "sim").resolve(),
        "action_budget": 900,
    }


@pytest.mark.parametrize("terminal_status", ("success", "finished_false"))
def test_finish_closure_binds_mailbox_terminal_outcome_and_verify_goal(
    terminal_status: str,
) -> None:
    from adaptive import critic_protocol as protocol
    from adaptive import remote_driver

    context = _context()
    context.milestone_history = ["observe", "approach", "engage", "actuate"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    command = _milestone_command("obs-4", "verify_goal", kind="finish")
    _FakeClient.controller_outputs = [command]
    _FakeClient.critic_outputs = [_proposal_audit()]
    response = _proposal_complete(wrapped, "obs-4", _state(), _images())
    terminal = _finish_terminal_outcome(terminal_status)
    mailbox = {
        "schema": "robocasa-inspect-joint-command/v1",
        "sequence": 0,
        "kind": "finish",
    }
    remote_driver._finalize_proposal_execution_evidence(
        context,
        {
            "requests": [{
                "observation_id": "obs-4",
                "command": command,
                "post_repair_command": command,
                "evidence": response.evidence,
                "mailbox": mailbox,
                "mailbox_sha256": _json_sha256(mailbox),
            }],
            "receipts": [],
            "terminal_outcome": terminal,
            "simulator_steps": 0,
        },
    )

    record = context.proposal_records[0]
    assert record["executed"] is True
    assert (record["mailbox_count"], record["action_count"], record["receipt_count"]) == (1, 0, 0)
    assert record["terminal_outcome_sha256"] == _json_sha256(terminal)
    assert context.milestone_history[-1] == "verify_goal"
    protocol.validate_proposal_audit_closure(context)


def test_finish_closure_fails_without_official_terminal_outcome() -> None:
    from adaptive import remote_driver

    context = _context()
    context.milestone_history = ["observe", "approach", "engage", "actuate"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    command = _milestone_command("obs-4", "verify_goal", kind="finish")
    _FakeClient.controller_outputs = [command]
    _FakeClient.critic_outputs = [_proposal_audit()]
    response = _proposal_complete(wrapped, "obs-4", _state(), _images())
    mailbox = {
        "schema": "robocasa-inspect-joint-command/v1",
        "sequence": 0,
        "kind": "finish",
    }

    with pytest.raises(ValueError, match="terminal"):
        remote_driver._finalize_proposal_execution_evidence(
            context,
            {
                "requests": [{
                    "observation_id": "obs-4",
                    "command": command,
                    "post_repair_command": command,
                    "evidence": response.evidence,
                    "mailbox": mailbox,
                    "mailbox_sha256": _json_sha256(mailbox),
                }],
                "receipts": [],
                "terminal_outcome": None,
            },
        )


def test_finish_closure_rejects_mailbox_hash_not_matching_mailbox() -> None:
    from adaptive import remote_driver

    context = _context()
    context.milestone_history = ["observe", "approach", "engage", "actuate"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    command = _milestone_command("obs-4", "verify_goal", kind="finish")
    _FakeClient.controller_outputs = [command]
    _FakeClient.critic_outputs = [_proposal_audit()]
    response = _proposal_complete(wrapped, "obs-4", _state(), _images())
    mailbox = {
        "schema": "robocasa-inspect-joint-command/v1",
        "sequence": 0,
        "kind": "finish",
    }

    with pytest.raises(ValueError, match="mailbox"):
        remote_driver._finalize_proposal_execution_evidence(
            context,
            {
                "requests": [{
                    "observation_id": "obs-4",
                    "command": command,
                    "post_repair_command": command,
                    "evidence": response.evidence,
                    "mailbox": mailbox,
                    "mailbox_sha256": _named_digest("different mailbox"),
                }],
                "receipts": [],
                "terminal_outcome": _finish_terminal_outcome("success"),
            },
        )


def test_give_up_closure_is_approved_zero_simulator_effect_terminal() -> None:
    from adaptive import critic_protocol as protocol
    from adaptive import remote_driver

    context = _context()
    context.milestone_history = ["observe", "approach", "engage", "actuate"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    command = _milestone_command("obs-4", "verify_goal", kind="give_up")
    _FakeClient.controller_outputs = [command]
    _FakeClient.critic_outputs = [_proposal_audit()]
    response = _proposal_complete(wrapped, "obs-4", _state(), _images())
    disposition = {
        "schema": "robocasa-qwen-approved-give-up/v1",
        "status": "approved_no_simulator_effect",
    }
    remote_driver._finalize_proposal_execution_evidence(
        context,
        {
            "requests": [{
                "observation_id": "obs-4",
                "command": command,
                "post_repair_command": command,
                "evidence": response.evidence,
                "mailbox": None,
                "mailbox_sha256": None,
            }],
            "receipts": [],
            "terminal_outcome": disposition,
        },
    )

    record = context.proposal_records[0]
    assert record["executed"] is True
    assert (record["mailbox_count"], record["action_count"], record["receipt_count"]) == (0, 0, 0)
    assert record["terminal_outcome"] == disposition
    assert context.milestone_history[-1] == "verify_goal"
    protocol.validate_proposal_audit_closure(context)


def test_prompt_appends_exp005_compatibility_after_byte_exact_p01_p22() -> None:
    root = Path(__file__).resolve().parents[1]
    panda = (root / "prompts" / "panda_embodiment.txt").read_text()
    prompt = load_joint_system_prompt(root)
    marker = "## Exp-005 proposal-audit compatibility"

    assert prompt.index(panda) < prompt.index(marker)
    compatibility = prompt[prompt.index(marker):].casefold()
    assert "supersedes p20" in compatibility
    assert "same-decision pre-execution proposal audit" in compatibility
    assert "immediately prior sealed action receipt" in compatibility
    assert "qwen remains the sole author" in compatibility


def test_composed_prompt_names_pandaomron_without_expanding_control_authority() -> None:
    root = Path(__file__).resolve().parents[1]
    compatibility = load_joint_system_prompt(root).split(
        "## Exp-005 proposal-audit compatibility", maxsplit=1
    )[1]

    assert "PandaOmron" in compatibility
    assert "Franka Emika Panda arm" in compatibility
    assert "Omron mobile base" in compatibility
    assert "bounded `base_action`" in compatibility
    assert "does not expose any other base motor interface" in compatibility


@pytest.mark.parametrize(
    ("kind", "mutation"),
    (
        ("move_joints", {"step_count": 999}),
        ("move_joints", {"tracking_pause_count": 6}),
        ("move_joints", {"maximum_commanded_step": 999.0}),
        ("base_action", {"step_count": 999}),
        ("base_action", {"tracking_pause_count": 6}),
        ("base_action", {"base_motion_step_count": 4}),
        ("base_action", {"base_motion_step_count": 999, "step_count": 999}),
    ),
    ids=(
        "move-step-count",
        "move-pause-over-steps",
        "move-step-bound",
        "base-step-count",
        "base-pause-over-steps",
        "base-motion-count-low",
        "base-motion-count-high",
    ),
)
def test_shared_public_receipt_validator_rejects_impossible_accounting_for_critic(
    kind: str, mutation: dict[str, object]
) -> None:
    from adaptive.joint_runner import validate_public_receipt

    context = _context()
    command = _controller_command("obs-0", kind=kind)
    state = _state(0.01)
    receipt = _sealed_critic_receipt(command, state)
    receipt.update(mutation)

    with pytest.raises(ValueError, match="accounting|step|maximum|tracking_pause"):
        validate_public_receipt(receipt, require_sealed_rgb=True)

    assert context.begin_observation("obs-0", _state(), [])["fired_trigger"] is None
    context.stage_controller_return("obs-0", command, None, "e0")
    decision = context.begin_observation("obs-1", state, [receipt])

    assert context.critic_attempts == 0
    assert decision["receipt_eligibility"] != "eligible"


def test_critic_failure_fails_open_and_consumes_attempt_and_observation_slot() -> None:
    context = _context()
    wrapped = _make_client_class(_FakeClient, context)()
    first = _controller_command("obs-0")
    _FakeClient.controller_outputs = [
        first, _controller_command("obs-1"), _controller_command("obs-1", 0.02),
    ]
    _FakeClient.critic_error = httpx.ReadTimeout("critic timeout")
    images = _images()
    state1 = _state(0.01)
    receipt = _sealed_critic_receipt(first, state1)

    _complete(wrapped, "obs-0", _state(), images)
    response = _complete(
        wrapped, "obs-1", state1, images, receipts=[receipt]
    )
    assert response.command["kind"] == "move_joints"
    assert response.evidence["consumed_advisory_id"] is None
    assert context.critic_attempts == 1
    assert context.critic_unavailable == 1
    assert context.critic_records[0]["status"] == "critic_unavailable"
    assert context.critic_records[0]["trigger"] == "first_action"
    assert context.critic_records[0]["critic_request_sha256"] is None
    assert len(context.critic_records[0]["critic_call_manifest_sha256"]) == 64

    _complete(wrapped, "obs-1", state1, images, receipts=[receipt])
    assert context.critic_attempts == 1
    assert [role for role, _ in _FakeClient.calls].count("critic") == 1


def test_repeat_trigger_bypasses_first_cooldown_then_cap_suppresses() -> None:
    context = _context(max_decisions=20)
    initial_state = _state(-0.01)
    state = _state(0.01)
    receipts: list[dict[str, object]] = []

    assert context.begin_observation("obs-0", initial_state, [])["fired_trigger"] is None
    command = _controller_command("obs-0")
    context.stage_controller_return("obs-0", command, None, "e0")

    receipts.append(_sealed_critic_receipt(command, state))
    first = context.begin_observation("obs-1", state, receipts)
    assert first["fired_trigger"] == "first_action"
    command = _controller_command("obs-1")
    context.stage_controller_return("obs-1", command, None, "e1")

    receipts.append(_sealed_critic_receipt(command, state))
    repeated = context.begin_observation("obs-2", state, receipts)
    assert repeated["fired_trigger"] == "repeat"
    assert context.critic_attempts == 2
    command = _controller_command("obs-2")
    context.stage_controller_return("obs-2", command, None, "e2")

    receipts.append(_sealed_critic_receipt(command, state))
    cooldown = context.begin_observation("obs-3", state, receipts)
    assert cooldown["eligible_trigger"] == "repeat"
    assert cooldown["suppressed"] == "cooldown"
    command = _controller_command("obs-3")
    context.stage_controller_return("obs-3", command, None, "e3")

    receipts.append(_sealed_critic_receipt(command, state))
    third = context.begin_observation("obs-4", state, receipts)
    assert third["fired_trigger"] == "repeat"
    assert context.critic_attempts == CRITIC_MAX_ATTEMPTS_PER_TASK
    command = _controller_command("obs-4")
    context.stage_controller_return("obs-4", command, None, "e4")

    receipts.append(_sealed_critic_receipt(command, state))
    capped = context.begin_observation("obs-5", state, receipts)
    assert capped["eligible_trigger"] == "repeat"
    assert capped["suppressed"] == "cap"


def test_stall_uses_only_public_proprioception_and_significant_receipts() -> None:
    stalled = public_outcome_delta(_state(), _state(), [], [])
    assert stalled["stalled"] is True
    moved = public_outcome_delta(_state(), _state(0.006), [], [])
    assert moved["stalled"] is False
    servo_receipt = [{
        "source": "public_rgb_harness", "kind": "servo_engaged_failed",
        "accepted": False,
    }]
    changed = public_outcome_delta(_state(), _state(), [], servo_receipt)
    assert changed["significant_receipts_changed"] is True
    assert changed["stalled"] is False
    joint_moved = public_outcome_delta(_state(), _state(0.03), [], [])
    assert joint_moved["arm_joint_max_delta_rad"] == pytest.approx(0.03)
    assert joint_moved["stalled"] is False

    command = _controller_command("obs-0")
    receipt = _sealed_critic_receipt(command, _state())
    receipt["realized_arm_qpos_delta"] = [0.0] * 7
    receipt["telemetry_summary"]["arm_joint_position"]["delta_rad"] = [0.0] * 7
    receipt["endpoint_error"] = 0.03
    telemetry_stall = public_outcome_delta(_state(), _state(), [], [receipt])
    assert telemetry_stall["current_arm_joint_velocity_max_abs_rad_s"] == 0.0
    assert telemetry_stall["receipt_tracking_pause_count"] == 1
    assert telemetry_stall["receipt_endpoint_error_rad"] == pytest.approx(0.03)
    assert telemetry_stall["receipt_arm_joint_max_delta_rad"] == 0.0
    assert telemetry_stall["stalled"] is True

    moving_velocity = public_outcome_delta(
        _state(), _state(qvel=0.2), [], [receipt]
    )
    assert moving_velocity["current_arm_joint_velocity_max_abs_rad_s"] == (
        pytest.approx(0.2)
    )
    assert moving_velocity["stalled"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        {"unexpected": "observe"},
        {"confidence": 1},
        {"diagnosis": "move around the handle"},
        {"suggested_correction": "retreat two radians"},
        {"suggested_correction": {"joint1": 0.2}},
        {"targets": {"joint1": 0.2}},
        {"command": "move_joints"},
    ),
)
def test_invalid_critic_output_consumes_attempt_and_fails_open_without_advice(
    mutation: dict[str, object],
) -> None:
    context = _context()
    wrapped = _make_client_class(_FakeClient, context)()
    first = _controller_command("obs-0")
    second = _controller_command("obs-1")
    state1 = _state(0.01)
    invalid = _advisory()
    invalid.update(mutation)
    _FakeClient.controller_outputs = [first, second]
    _FakeClient.critic_outputs = [invalid]

    _complete(wrapped, "obs-0", _state(), _images())
    response = _complete(
        wrapped,
        "obs-1",
        state1,
        _images(),
        receipts=[_sealed_critic_receipt(first, state1)],
    )

    assert response.command is second
    assert context.critic_attempts == 1
    assert context.critic_unavailable == 1
    assert response.evidence["consumed_advisory_id"] is None
    assert "POST_ACTION_ADVISORY" not in str(_FakeClient.calls[-1][1]["instruction"])


def test_command_signature_ignores_note_and_observation_but_not_motor_arguments() -> None:
    first = _action("obs-a", 0.01)
    second = _action("obs-b", 0.01)
    second["note"] = "different prose"
    assert command_signature(first) == command_signature(second)
    assert command_signature(first) != command_signature(_action("obs-c", -0.01))


def test_direct_command_materiality_uses_only_official_physical_commands() -> None:
    joint = _joint_command()
    episode = {
        "requests": [
            {"post_repair_command": joint},
            {"post_repair_command": _action("obs")},
            {"post_repair_command": {
                "kind": "finish", "observation_id": "done", "note": "done",
            }},
        ]
    }
    commands = _direct_commands(episode)
    assert commands == [joint]


def test_budget_check_uses_pinned_utc_clock_and_critic_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive import remote_driver

    start = dt.datetime(2026, 9, 3, 21, 0, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(
        remote_driver, "_utc_now", lambda: start + dt.timedelta(hours=1)
    )
    check = _budget_check(
        phase="pre_run",
        completed_tasks=0,
        run_elapsed_s=0.0,
        smoke_wall_s=60.0,
        smoke_decisions=12,
    )
    assert check["started_at_utc"] == "2026-09-03T21:00:00Z"
    assert OPTIMIZATION_MAX_HOURS == 72.0
    assert check["max_hours"] == 72.0
    assert check["remaining_registered_s"] == 71 * 3600
    assert check["critic_ceiling_s"] == 900
    assert check["decision"] == "continue"


def test_budget_check_allows_authorized_continuation_after_original_eight_hours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive import remote_driver

    start = dt.datetime(2026, 9, 3, 21, 0, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(
        remote_driver,
        "_utc_now",
        lambda: start + dt.timedelta(hours=23, minutes=7),
    )
    check = _budget_check(
        phase="pre_run",
        completed_tasks=0,
        run_elapsed_s=0.0,
        smoke_wall_s=80.9044951479882,
        smoke_decisions=8,
    )
    assert check["started_at_utc"] == "2026-09-03T21:00:00Z"
    assert check["max_hours"] == 72.0
    assert check["elapsed_registered_s"] == 23 * 3600 + 7 * 60
    assert check["projection_s"] <= check["remaining_registered_s"]
    assert check["decision"] == "continue"


def test_budget_check_uses_current_joint_smoke_near_registered_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive import remote_driver

    start = dt.datetime(2026, 9, 3, 21, 0, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(
        remote_driver,
        "_utc_now",
        lambda: start + dt.timedelta(hours=71, minutes=13),
    )
    current = _budget_check(
        phase="pre_run",
        completed_tasks=0,
        run_elapsed_s=0.0,
        smoke_wall_s=46.1,
        smoke_decisions=7,
    )
    assert current["projection_basis"] == "current_joint_smoke_1_5x"
    assert current["projection_s"] < current["remaining_registered_s"]
    assert current["decision"] == "continue"

    without_smoke = _budget_check(
        phase="pre_run",
        completed_tasks=0,
        run_elapsed_s=0.0,
        smoke_wall_s=0.0,
        smoke_decisions=0,
    )
    assert without_smoke["projection_basis"] == "prior_full_run"
    assert without_smoke["decision"] == "abort"


def _named_digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _smoke_decision(
    *,
    observation_id: str,
    state: dict[str, object],
    executed_command_count: int,
    previous_command: dict[str, object] | None = None,
    receipt: dict[str, object] | None = None,
    advisory_id: str | None = None,
) -> dict[str, object]:
    fired = "first_action" if advisory_id is not None else None
    return {
        "observation_id": observation_id,
        "executed_command_count": executed_command_count,
        "eligible_trigger": fired,
        "fired_trigger": fired,
        "suppressed": None,
        "critic_attempt_index": 1 if fired else None,
        "advisory_id": advisory_id,
        "previous_executed_command": previous_command,
        "previous_executed_command_sha256": (
            _json_sha256(previous_command) if previous_command is not None else None
        ),
        "fresh_public_state": state,
        "fresh_public_state_sha256": _json_sha256(state),
        "sealed_public_receipt": receipt,
        "sealed_public_receipt_sha256": (
            _json_sha256(receipt) if receipt is not None else None
        ),
        "receipt_eligibility": "eligible" if receipt is not None else "not_evaluated",
        "outcome_delta": {"stalled": False} if receipt is not None else None,
    }


def _controller_evidence(
    *,
    decision: dict[str, object],
    command: dict[str, object],
    controller_request_sha256: str,
    controller_call_manifest_sha256: str,
    image_sha256: dict[str, str],
    system_prompt_sha256: str,
    snapshot_digest: str,
    served_model_id: str,
    advisory: dict[str, object] | None = None,
    critic_request_sha256: str | None = None,
    critic_call_manifest_sha256: str | None = None,
) -> dict[str, object]:
    instruction_sha256 = _named_digest(
        f"controller-instruction-{decision['observation_id']}"
    )
    raw_command = json.dumps(command, sort_keys=True, separators=(",", ":"))
    return {
        "role": "controller",
        "latency_s": 0.01,
        "system_prompt_sha256": system_prompt_sha256,
        "instruction_sha256": instruction_sha256,
        "image_sha256": image_sha256,
        "image_roles": {
            "left": "official_left",
            "right": "official_right",
            "wrist": "official_wrist",
        },
        "usage": {"completion_tokens": 7},
        "finish_reason": "stop",
        "raw_sha256": hashlib.sha256(raw_command.encode()).hexdigest(),
        "raw_chars": len(raw_command),
        "snapshot_digest": snapshot_digest,
        "served_model_id": served_model_id,
        "decoding": {"max_tokens": 256},
        "response_schema_sha256": _json_sha256(CONTROLLER_RESPONSE_SCHEMA),
        "controller_role": "sole_direct_inspect_command_emitter",
        "controller_call_index": int(decision["executed_command_count"]) + 1,
        "critic_trigger_evaluation": decision,
        "consumed_advisory_id": decision["advisory_id"],
        "critic_advisory": advisory,
        "controller_input_public_state_sha256": decision[
            "fresh_public_state_sha256"
        ],
        "controller_input_image_sha256": image_sha256,
        "controller_request_sha256": controller_request_sha256,
        "controller_request_linkage_status": "complete",
        "controller_call_manifest_sha256": controller_call_manifest_sha256,
        "controller_instruction_sha256": instruction_sha256,
        "qwen_attempt_index": 0,
        "rendered_advisory_sha256": (
            _json_sha256(advisory) if advisory is not None else None
        ),
        "critic_request_sha256": critic_request_sha256,
        "critic_call_manifest_sha256": critic_call_manifest_sha256,
        "advisory_sha256": _json_sha256(advisory) if advisory is not None else None,
        "qwen_command_sha256": _json_sha256(command),
        "critic_config_sha256": CRITIC_CONFIG_SHA256,
        "controller_response_schema_sha256": _json_sha256(
            CONTROLLER_RESPONSE_SCHEMA
        ),
    }


def _controller_record(
    *,
    command: dict[str, object],
    evidence: dict[str, object],
) -> dict[str, object]:
    return {
        "observation_id": command["observation_id"],
        "controller_call_index": evidence["controller_call_index"],
        "qwen_attempt_index": evidence["qwen_attempt_index"],
        "command": command,
        "command_sha256": _json_sha256(command),
        "command_evidence_sha256": _json_sha256(evidence),
        "consumed_advisory_id": evidence["consumed_advisory_id"],
        "controller_request_sha256": evidence["controller_request_sha256"],
        "controller_request_linkage_status": "complete",
        "critic_request_sha256": evidence["critic_request_sha256"],
        "controller_call_manifest_sha256": evidence[
            "controller_call_manifest_sha256"
        ],
        "critic_call_manifest_sha256": evidence["critic_call_manifest_sha256"],
        "advisory_sha256": evidence["advisory_sha256"],
    }


def _grounded_smoke_episode(tmp_path: Path) -> dict[str, object]:
    snapshot_digest = (
        "27829db7e5189de5f7a0d5ac967b33a4b8ff5e5c28b28a4dd7790599e448ce91"
    )
    served_model_id = "qwen3.8-27b-bf16-test-launch"
    system_prompt = load_joint_system_prompt(
        Path(__file__).resolve().parents[1],
        protocol="legacy",
    )
    system_prompt_sha256 = hashlib.sha256(system_prompt.encode()).hexdigest()
    command0 = _controller_command("obs-0", 0.01)
    command0["note"] = "Use the translation jacobian for a small joint probe"
    command1 = _controller_command("obs-1", 0.03)
    command1["note"] = "The torque and wrench receipt support another joint probe"
    state0 = _state(-0.01, torque_available=False)
    state1 = _state(0.01)
    state2 = _state(0.03)
    receipt0 = _sealed_critic_receipt(command0, state1)
    receipt1 = _sealed_critic_receipt(command1, state2)
    mailbox0, _decoded0 = prepare_joint_mailbox(
        command0,
        source="controller",
        observation_id="obs-0",
        current_qpos=state0["state.arm_joint_position"],
        current_gripper=1.0,
        remaining_actions=450,
        sequence=0,
    )
    mailbox1, _decoded1 = prepare_joint_mailbox(
        command1,
        source="controller",
        observation_id="obs-1",
        current_qpos=state1["state.arm_joint_position"],
        current_gripper=1.0,
        remaining_actions=445,
        sequence=1,
    )
    advisory = _advisory()
    advisory_id = "advisory-first-action"
    decision0 = _smoke_decision(
        observation_id="obs-0",
        state=state0,
        executed_command_count=0,
    )
    decision1 = _smoke_decision(
        observation_id="obs-1",
        state=state1,
        executed_command_count=1,
        previous_command=command0,
        receipt=receipt0,
        advisory_id=advisory_id,
    )
    image0 = {
        label: _named_digest(f"obs-0-{label}")
        for label in ("left", "right", "wrist")
    }
    image1 = {
        label: _named_digest(f"obs-1-{label}")
        for label in ("left", "right", "wrist")
    }
    controller_request0 = _named_digest("controller-request-0")
    controller_request1 = _named_digest("controller-request-1")
    controller_manifest0 = _named_digest("controller-manifest-0")
    controller_manifest1 = _named_digest("controller-manifest-1")
    critic_request = _named_digest("critic-request-1")
    critic_manifest = _named_digest("critic-manifest-1")
    decision1["critic_request_sha256"] = critic_request
    decision1["critic_call_manifest_sha256"] = critic_manifest
    decision1["advisory_sha256"] = _json_sha256(advisory)
    evidence0 = _controller_evidence(
        decision=decision0,
        command=command0,
        controller_request_sha256=controller_request0,
        controller_call_manifest_sha256=controller_manifest0,
        image_sha256=image0,
        system_prompt_sha256=system_prompt_sha256,
        snapshot_digest=snapshot_digest,
        served_model_id=served_model_id,
    )
    evidence1 = _controller_evidence(
        decision=decision1,
        command=command1,
        controller_request_sha256=controller_request1,
        controller_call_manifest_sha256=controller_manifest1,
        image_sha256=image1,
        system_prompt_sha256=system_prompt_sha256,
        snapshot_digest=snapshot_digest,
        served_model_id=served_model_id,
        advisory=advisory,
        critic_request_sha256=critic_request,
        critic_call_manifest_sha256=critic_manifest,
    )
    controller_records = [
        _controller_record(command=command0, evidence=evidence0),
        _controller_record(command=command1, evidence=evidence1),
    ]
    critic_prompt_sha256 = hashlib.sha256(
        (Path(__file__).resolve().parents[1] / "prompts" / "advisory_critic.txt")
        .read_text(encoding="utf-8")
        .encode()
    ).hexdigest()
    critic_record = {
        "schema": "robocasa-qwen-advisory-evidence/v1",
        "task": "OpenToasterOvenDoor",
        "observation_id": "obs-1",
        "advisory_id": advisory_id,
        "trigger": "first_action",
        "attempt_index": 1,
        "qwen_attempt_index": 0,
        "previous_executed_command_sha256": _json_sha256(command0),
        "sealed_public_receipt_sha256": _json_sha256(receipt0),
        "fresh_public_state_sha256": _json_sha256(state1),
        "fresh_public_rgb_sha256": image1,
        "critic_request_sha256": critic_request,
        "critic_call_manifest_sha256": critic_manifest,
        "input_public_state_sha256": _json_sha256(state1),
        "input_image_sha256": image1,
        "critic_config_sha256": CRITIC_CONFIG_SHA256,
        "critic_prompt_sha256": critic_prompt_sha256,
        "advisory_schema_sha256": _json_sha256(ADVISORY_SCHEMA),
        "critic_instruction_sha256": _named_digest("critic-instruction-obs-1"),
        "status": "available",
        "advisory": advisory,
        "advisory_sha256": _json_sha256(advisory),
        "critic_request_linkage_status": "complete",
        "model_evidence": {
            "role": "critic",
            "latency_s": 0.01,
            "system_prompt_sha256": critic_prompt_sha256,
            "instruction_sha256": _named_digest("critic-instruction-obs-1"),
            "image_sha256": image1,
            "image_roles": {
                "left": "official_left",
                "right": "official_right",
                "wrist": "official_wrist",
            },
            "usage": {"completion_tokens": 7},
            "finish_reason": "stop",
            "raw_sha256": hashlib.sha256(
                json.dumps(advisory, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "raw_chars": len(
                json.dumps(advisory, sort_keys=True, separators=(",", ":"))
            ),
            "snapshot_digest": snapshot_digest,
            "served_model_id": served_model_id,
            "decoding": {"max_tokens": 384},
            "response_schema_sha256": _json_sha256(ADVISORY_SCHEMA),
        },
        "failure_class": None,
        "consumed_by_controller_observation_id": "obs-1",
        "consumed_by_controller_request_sha256": controller_request1,
        "consumed_by_controller_call_manifest_sha256": controller_manifest1,
        "consumed_by_controller_linkage_status": "complete",
        "qwen_command_sha256": _json_sha256(command1),
        "latency_s": 0.25,
    }
    video = tmp_path / "OpenToasterOvenDoor-seed7-policy_finished_false.mp4"
    video.write_bytes(b"real smoke video bytes")
    authority_dir = tmp_path / "authority"
    authority_dir.mkdir(mode=0o700)
    identity = {
        "authority": {
            "dtype": "bfloat16",
            "repo_id": "Qwen/Qwen3.8-27B",
            "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "served_model_name": "qwen3.8-27b-bf16",
        },
        "runtime": {"transformers": "5.15.1", "vllm": "0.27.1"},
        "schema": "panda-qwen-model-identity/v2",
        "snapshot_digest": snapshot_digest,
        "snapshot_path": "/owner/frozen/qwen-snapshot",
        "snapshot_stat_digest": _named_digest("snapshot stat"),
    }
    identity_path = authority_dir / "identity.json"
    identity_path.write_text(
        json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    identity_path.chmod(0o600)
    attestation = {
        "identity_manifest_sha256": _json_sha256(identity),
        "launch_id": "test-launch",
        "pid": 123,
        "port": 8002,
        "process_start_ticks": "456",
        "schema": "panda-qwen-server-attestation/v1",
        "served_model_id": served_model_id,
        "snapshot_digest": snapshot_digest,
        "snapshot_path": identity["snapshot_path"],
    }
    attestation_path = authority_dir / "server-attestation.json"
    attestation_path.write_text(
        json.dumps(attestation, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    attestation_path.chmod(0o600)
    controller_raw0 = json.dumps(command0, sort_keys=True, separators=(",", ":"))
    controller_raw1 = json.dumps(command1, sort_keys=True, separators=(",", ":"))
    critic_raw = json.dumps(advisory, sort_keys=True, separators=(",", ":"))
    qwen_attempt_records = []
    for role, call_index, observation_id, request_sha256, response_schema, raw in (
        (
            "controller",
            1,
            "obs-0",
            controller_request0,
            CONTROLLER_RESPONSE_SCHEMA,
            controller_raw0,
        ),
        (
            "critic",
            1,
            "obs-1",
            critic_request,
            ADVISORY_SCHEMA,
            critic_raw,
        ),
        (
            "controller",
            2,
            "obs-1",
            controller_request1,
            CONTROLLER_RESPONSE_SCHEMA,
            controller_raw1,
        ),
    ):
        record = _attempt_record(
            request_sha256,
            observation_id=observation_id,
            attempt_index=0,
            response_schema_sha256=_json_sha256(response_schema),
            served_model_id=served_model_id,
        ) | {
            "raw_body_sha256": _named_digest(
                f"{role}-{call_index}-raw-http-body"
            ),
            "sanitized_raw_command": raw,
        }
        qwen_attempt_records.append({
            "schema": "robocasa-qwen-attempt-evidence/v1",
            "role": role,
            "call_index": call_index,
            "record": record,
            "record_sha256": _json_sha256(record),
        })
    physical_commands = [command0, command1]
    episode = {
        "schema": "robocasa-inspect-joint-episode/v1",
        "task": "OpenToasterOvenDoor",
        "seed": 7,
        "status": "policy_finished_false",
        "success": False,
        "requests": [
            {
                "decision": 0,
                "observation_id": "obs-0",
                "command": command0,
                "evidence": evidence0,
                "format_errors": [],
                "repair_count": 0,
                "repair_reason": None,
                "repair_command": None,
                "repair_evidence": None,
                "post_repair_command": command0,
                "mailbox": mailbox0,
                "mailbox_sha256": _json_sha256(mailbox0),
            },
            {
                "decision": 1,
                "observation_id": "obs-1",
                "command": command1,
                "evidence": evidence1,
                "format_errors": [],
                "repair_count": 0,
                "repair_reason": None,
                "repair_command": None,
                "repair_evidence": None,
                "post_repair_command": command1,
                "mailbox": mailbox1,
                "mailbox_sha256": _json_sha256(mailbox1),
            },
        ],
        "receipts": [receipt0, receipt1],
        "qwen_direct_command_count": 2,
        "qwen_joint_command_count": 2,
        "qwen_direct_command_digest": _json_sha256(physical_commands),
        "qwen_controller_calls": 2,
        "qwen_critic_attempts": 1,
        "qwen_critic_successes": 1,
        "qwen_critic_unavailable": 0,
        "critic_origin_executions": 0,
        "critic_config": CRITIC_CONFIG,
        "critic_config_sha256": CRITIC_CONFIG_SHA256,
        "critic_prompt_sha256": critic_prompt_sha256,
        "controller_response_schema_sha256": _json_sha256(
            CONTROLLER_RESPONSE_SCHEMA
        ),
        "advisory_schema_sha256": _json_sha256(ADVISORY_SCHEMA),
        "critic_records": [critic_record],
        "trigger_evaluations": [decision0, decision1],
        "controller_records": controller_records,
        "qwen_attempt_records": qwen_attempt_records,
        "video": str(video),
        "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "system_prompt_sha256": system_prompt_sha256,
        "snapshot_digest": snapshot_digest,
        "served_model_id": served_model_id,
        "model_identity_manifest_path": str(identity_path),
        "model_identity_manifest_sha256": hashlib.sha256(
            identity_path.read_bytes()
        ).hexdigest(),
        "server_attestation_path": str(attestation_path),
        "server_attestation_sha256": hashlib.sha256(
            attestation_path.read_bytes()
        ).hexdigest(),
    }
    snapshot = _json_copy(episode)
    assert isinstance(snapshot, dict)
    return snapshot


def test_smoke_gate_is_mechanical(tmp_path: Path) -> None:
    passed, checks = _smoke_pass(_grounded_smoke_episode(tmp_path))

    assert passed is True, json.dumps(checks, sort_keys=True)
    assert all(checks.values())
    assert checks["qwen_joint_command_authority"] is True
    assert checks["fresh_controller_grounding_evidence"] is True
    assert checks["complete_critic_receipt"] is True
    assert checks["advisory_consumed_exactly_once"] is True
    assert checks["grounded_controller_note"] is True
    assert checks["prompt_schema_model_hashes_valid"] is True


def _proposal_smoke_episode(
    tmp_path: Path,
    *,
    reject_first: bool = True,
) -> dict[str, object]:
    from adaptive import critic_protocol as protocol

    legacy = _grounded_smoke_episode(tmp_path)
    context = _review_closed_physical_context(
        reject_first=reject_first,
        served_model_id=str(legacy["served_model_id"]),
    )
    approved_record = context.proposal_records[-1]
    approved = approved_record["draft"]
    assert isinstance(approved, dict)
    receipt = _sealed_critic_receipt(approved, _state(0.02))
    mailbox, _decoded = prepare_joint_mailbox(
        approved,
        source="controller",
        observation_id="obs-0",
        current_qpos=_state()["state.arm_joint_position"],
        current_gripper=1.0,
        remaining_actions=450,
        sequence=0,
    )
    episode = {
        **legacy,
        "family": context.family,
        "requests": [{
            "observation_id": approved["observation_id"],
            "command": approved,
            "post_repair_command": approved,
            "evidence": {
                "controller_call_index": approved_record["controller_call_index"],
                "qwen_command_sha256": _json_sha256(approved),
                "proposal_audit_verdict": "approve",
                "proposal_audit_sha256": approved_record["audit_sha256"],
                "proposal_audit_config_sha256": (
                    protocol.PROPOSAL_AUDIT_CONFIG_SHA256
                ),
                "claimed_milestone": approved_record["claimed_milestone"],
                "critic_origin_execution": False,
            },
            "mailbox": mailbox,
            "mailbox_sha256": _json_sha256(mailbox),
        }],
        "receipts": [receipt],
        "terminal_outcome": None,
        "decisions": 1,
        "model_decisions": 1,
        "action_chunks": 1,
        "simulator_steps": receipt["step_count"],
        "qwen_direct_command_count": 1,
        "qwen_joint_command_count": 1,
        "qwen_direct_command_digest": _json_sha256([approved]),
        "qwen_controller_calls": context.controller_calls,
        "qwen_critic_attempts": context.critic_attempts,
        "qwen_critic_successes": context.critic_successes,
        "qwen_critic_unavailable": context.critic_unavailable,
        "critic_origin_executions": 0,
        "proposal_audit_config": protocol.PROPOSAL_AUDIT_CONFIG,
        "proposal_audit_config_sha256": protocol.PROPOSAL_AUDIT_CONFIG_SHA256,
        "proposal_audit_schema_sha256": _json_sha256(
            protocol.PROPOSAL_AUDIT_SCHEMA
        ),
        "proposal_audit_records": context.proposal_records,
        "controller_records": context.controller_records,
        "critic_records": context.critic_records,
        "qwen_attempt_records": context.attempt_records,
        "milestone_history": context.milestone_history,
        "proposal_audit_closure": {
            "valid": True,
            **protocol.validate_proposal_audit_closure(context),
        },
        "critic_prompt_sha256": hashlib.sha256(
            (
                Path(__file__).resolve().parents[1]
                / "prompts"
                / "proposal_audit_critic.txt"
            ).read_bytes()
        ).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(
            load_joint_system_prompt(
                Path(__file__).resolve().parents[1],
                protocol="proposal",
            ).encode()
        ).hexdigest(),
    }
    snapshot = _json_copy(episode)
    assert isinstance(snapshot, dict)
    return snapshot


def test_proposal_smoke_gate_is_selected_explicitly_and_mechanical(
    tmp_path: Path,
) -> None:
    episode = _proposal_smoke_episode(tmp_path)

    passed, checks = _smoke_pass(episode, protocol="proposal")

    assert passed is True, json.dumps(checks, sort_keys=True)
    assert all(checks.values())
    assert checks["strict_proposal_closure"] is True
    assert checks["rejected_drafts_zero_effect"] is True
    assert checks["approved_commands_closed"] is True
    assert checks["same_observation_revision_cap"] is True


def test_proposal_smoke_gate_accepts_a_clean_first_pass_approval(
    tmp_path: Path,
) -> None:
    episode = _proposal_smoke_episode(tmp_path, reject_first=False)

    passed, checks = _smoke_pass(episode, protocol="proposal")

    assert passed is True, json.dumps(checks, sort_keys=True)
    assert checks["rejected_drafts_zero_effect"] is True


def test_proposal_smoke_gate_rejects_proposal_audit_policy_failure(
    tmp_path: Path,
) -> None:
    episode = _proposal_smoke_episode(tmp_path)
    episode["status"] = "policy_failed_proposal_audit"
    episode["success"] = False

    passed, checks = _smoke_pass(episode, protocol="proposal")

    assert passed is False
    assert checks["no_proposal_contract_failure"] is False


def test_proposal_smoke_gate_rejects_relabelled_audit_failure_request(
    tmp_path: Path,
) -> None:
    episode = _proposal_smoke_episode(tmp_path)
    requests = episode["requests"]
    assert isinstance(requests, list)
    requests.append({
        "schema": "robocasa-qwen-proposal-audit-failure/v1",
        "decision": len(requests),
        "observation_id": "obs-exhausted",
        "status": "policy_failed_proposal_audit",
        "failure_class": "ProposalAuditExhausted",
        "failure_sha256": hashlib.sha256(b"audit exhausted").hexdigest(),
        "command": None,
        "mailbox_count": 0,
        "action_count": 0,
        "receipt_count": 0,
    })
    episode["model_decisions"] = len(requests)
    episode["status"] = "success"
    episode["success"] = True

    passed, checks = _smoke_pass(episode, protocol="proposal")

    assert passed is False
    assert checks["no_proposal_failure_request"] is False


def test_proposal_smoke_gate_rejects_mutated_returned_command_hash(
    tmp_path: Path,
) -> None:
    episode = _proposal_smoke_episode(tmp_path)
    episode["proposal_audit_records"][1]["returned_command_sha256"] = (
        _named_digest("tampered return")
    )

    passed, checks = _smoke_pass(episode, protocol="proposal")

    assert passed is False
    assert checks["strict_proposal_closure"] is False


def test_proposal_smoke_gate_rejects_mutated_config_hash(tmp_path: Path) -> None:
    episode = _proposal_smoke_episode(tmp_path)
    episode["proposal_audit_config_sha256"] = _named_digest(
        "different proposal config"
    )

    passed, checks = _smoke_pass(episode, protocol="proposal")

    assert passed is False
    assert checks["prompt_schema_model_hashes_valid"] is False


def _proposal_finish_smoke_episode(tmp_path: Path) -> dict[str, object]:
    from adaptive import critic_protocol as protocol
    from adaptive import remote_driver

    legacy = _grounded_smoke_episode(tmp_path)
    _FakeClient.served_model_id = str(legacy["served_model_id"])
    context = _context()
    context.milestone_history = ["observe", "approach", "engage", "actuate"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    rejected = _milestone_command("obs-4", "verify_goal", kind="finish")
    approved = _milestone_command("obs-4", "verify_goal", kind="finish")
    approved["note"] = "milestone=verify_goal; corrected description of the visible goal state"
    _FakeClient.controller_outputs = [rejected, approved]
    _FakeClient.critic_outputs = [_proposal_audit("revise"), _proposal_audit()]
    response = _proposal_complete(wrapped, "obs-4", _state(), _images())
    terminal = _finish_terminal_outcome("success")
    mailbox = {
        "schema": "robocasa-inspect-joint-command/v1",
        "sequence": 0,
        "kind": "finish",
    }
    episode = {
        **legacy,
        "family": context.family,
        "requests": [{
            "observation_id": approved["observation_id"],
            "command": approved,
            "post_repair_command": approved,
            "evidence": response.evidence,
            "mailbox": mailbox,
            "mailbox_sha256": _json_sha256(mailbox),
        }],
        "receipts": [],
        "terminal_outcome": terminal,
        "decisions": 0,
        "model_decisions": 1,
        "action_chunks": 0,
        "simulator_steps": 0,
        "qwen_direct_command_count": 0,
        "qwen_joint_command_count": 0,
        "qwen_direct_command_digest": None,
        "qwen_controller_calls": context.controller_calls,
        "qwen_critic_attempts": context.critic_attempts,
        "qwen_critic_successes": context.critic_successes,
        "qwen_critic_unavailable": context.critic_unavailable,
        "critic_origin_executions": 0,
        "proposal_audit_config": protocol.PROPOSAL_AUDIT_CONFIG,
        "proposal_audit_config_sha256": protocol.PROPOSAL_AUDIT_CONFIG_SHA256,
        "proposal_audit_schema_sha256": _json_sha256(
            protocol.PROPOSAL_AUDIT_SCHEMA
        ),
        "proposal_audit_records": context.proposal_records,
        "controller_records": context.controller_records,
        "critic_records": context.critic_records,
        "qwen_attempt_records": context.attempt_records,
        "milestone_history": context.milestone_history,
        "critic_prompt_sha256": hashlib.sha256(
            (
                Path(__file__).resolve().parents[1]
                / "prompts"
                / "proposal_audit_critic.txt"
            ).read_bytes()
        ).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(
            load_joint_system_prompt(
                Path(__file__).resolve().parents[1],
                protocol="proposal",
            ).encode()
        ).hexdigest(),
    }
    remote_driver._finalize_proposal_execution_evidence(context, episode)
    episode["milestone_history"] = context.milestone_history
    episode["proposal_audit_closure"] = {
        "valid": True,
        **protocol.validate_proposal_audit_closure(
            context,
            require_execution_closure=True,
        ),
    }
    snapshot = _json_copy(episode)
    assert isinstance(snapshot, dict)
    return snapshot


def test_proposal_smoke_gate_closes_approved_finish_terminal(
    tmp_path: Path,
) -> None:
    passed, checks = _smoke_pass(
        _proposal_finish_smoke_episode(tmp_path),
        protocol="proposal",
    )

    assert passed is True, json.dumps(checks, sort_keys=True)
    assert checks["strict_proposal_closure"] is True
    assert checks["approved_commands_closed"] is True
    assert checks["request_mailbox_receipt_closure"] is True


def test_proposal_smoke_gate_rejects_terminal_sequence_drift(tmp_path: Path) -> None:
    episode = _proposal_finish_smoke_episode(tmp_path)
    episode["terminal_outcome"]["simulator_terminal"]["sequence"] = 1

    passed, checks = _smoke_pass(episode, protocol="proposal")

    assert passed is False
    assert checks["strict_proposal_closure"] is False


def test_proposal_smoke_gate_rejects_terminal_action_count_drift(
    tmp_path: Path,
) -> None:
    episode = _proposal_finish_smoke_episode(tmp_path)
    simulator_terminal = episode["terminal_outcome"]["simulator_terminal"]
    simulator_terminal["simulator_actions"] = 1
    episode["terminal_outcome"]["simulator_terminal_sha256"] = _json_sha256(
        simulator_terminal
    )

    passed, checks = _smoke_pass(episode, protocol="proposal")

    assert passed is False
    assert checks["strict_proposal_closure"] is False


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    (
        ("persisted_closure", "strict_proposal_closure"),
        ("attempt_order", "strict_proposal_closure"),
        ("controller_record", "strict_proposal_closure"),
        ("critic_record", "strict_proposal_closure"),
        ("returned_command", "strict_proposal_closure"),
        ("return_evidence", "strict_proposal_closure"),
        ("revision_index", "strict_proposal_closure"),
        ("rejected_effect", "strict_proposal_closure"),
        ("missing_receipt", "strict_proposal_closure"),
        ("orphan_receipt", "strict_proposal_closure"),
        ("extra_request", "strict_proposal_closure"),
        ("action_counters", "episode_effect_counters_close"),
        ("milestone_history", "strict_proposal_closure"),
        ("prompt_hash", "prompt_schema_model_hashes_valid"),
        ("model_identity", "prompt_schema_model_hashes_valid"),
        ("schema_hash", "prompt_schema_model_hashes_valid"),
        ("mailbox", "strict_proposal_closure"),
        ("mailbox_payload", "strict_proposal_closure"),
        ("direct_digest", "qwen_direct_command_authority"),
        ("critic_origin", "zero_critic_origin_executions"),
        ("artifact", "artifacts_valid"),
    ),
)
def test_proposal_smoke_gate_rejects_locked_evidence_mutations(
    mutation: str,
    failed_check: str,
    tmp_path: Path,
) -> None:
    episode = _proposal_smoke_episode(tmp_path)
    if mutation == "persisted_closure":
        episode["proposal_audit_closure"]["approved_count"] = 99
    elif mutation == "attempt_order":
        attempts = episode["qwen_attempt_records"]
        attempts[0], attempts[1] = attempts[1], attempts[0]
    elif mutation == "controller_record":
        episode["controller_records"][1]["command_sha256"] = _named_digest(
            "different controller command"
        )
    elif mutation == "critic_record":
        episode["critic_records"][0]["audit_sha256"] = _named_digest(
            "different critic audit"
        )
    elif mutation == "returned_command":
        episode["requests"][0]["command"]["targets"]["joint1"] = 0.03
    elif mutation == "return_evidence":
        episode["requests"][0]["evidence"]["qwen_command_sha256"] = (
            _named_digest("different returned command evidence")
        )
    elif mutation == "revision_index":
        episode["proposal_audit_records"][1]["revision_index"] = 2
    elif mutation == "rejected_effect":
        episode["proposal_audit_records"][0]["mailbox_count"] = 1
    elif mutation == "missing_receipt":
        episode["receipts"] = []
    elif mutation == "orphan_receipt":
        orphan = _json_copy(episode["receipts"][0])
        orphan["note"] = "orphan receipt"
        episode["receipts"].append(orphan)
    elif mutation == "extra_request":
        extra = _json_copy(episode["requests"][0])
        extra["evidence"]["controller_call_index"] = 99
        episode["requests"].append(extra)
    elif mutation == "action_counters":
        episode["action_chunks"] = 2
    elif mutation == "milestone_history":
        episode["milestone_history"] = ["observe", "approach"]
    elif mutation == "prompt_hash":
        episode["critic_prompt_sha256"] = _named_digest("different prompt")
    elif mutation == "model_identity":
        episode["served_model_id"] = "different-model"
    elif mutation == "schema_hash":
        episode["proposal_audit_schema_sha256"] = _named_digest(
            "different schema"
        )
    elif mutation == "mailbox":
        episode["requests"][0]["mailbox"]["kind"] = "finish"
    elif mutation == "mailbox_payload":
        episode["requests"][0]["mailbox"]["endpoint"][0] = 0.03
        episode["requests"][0]["mailbox_sha256"] = _json_sha256(
            episode["requests"][0]["mailbox"]
        )
    elif mutation == "direct_digest":
        episode["qwen_direct_command_digest"] = _named_digest(
            "different direct digest"
        )
    elif mutation == "critic_origin":
        episode["critic_origin_executions"] = 1
    elif mutation == "artifact":
        episode["video_sha256"] = _named_digest("different artifact")
    else:
        raise AssertionError(mutation)

    passed, checks = _smoke_pass(episode, protocol="proposal")

    assert passed is False
    assert checks[failed_check] is False


def test_legacy_smoke_gate_does_not_self_select_from_episode_stamp(
    tmp_path: Path,
) -> None:
    episode = _grounded_smoke_episode(tmp_path)
    episode["proposal_audit_config_sha256"] = "f" * 64

    passed, checks = _smoke_pass(episode, protocol="legacy")

    assert passed is True
    assert "strict_proposal_closure" not in checks


def test_smoke_gate_rejects_a_fourth_duplicate_available_critic(
    tmp_path: Path,
) -> None:
    episode = _grounded_smoke_episode(tmp_path)
    critics = episode["critic_records"]
    assert isinstance(critics, list)
    for _ in range(3):
        critics.append(_json_copy(critics[0]))
    episode["qwen_critic_attempts"] = 4
    episode["qwen_critic_successes"] = 4

    passed, checks = _smoke_pass(episode)

    assert passed is False
    assert checks["counters_reconcile"] is False


def test_smoke_gate_rejects_an_unmatched_successful_controller_record(
    tmp_path: Path,
) -> None:
    episode = _grounded_smoke_episode(tmp_path)
    controllers = episode["controller_records"]
    assert isinstance(controllers, list)
    duplicate = _json_copy(controllers[-1])
    assert isinstance(duplicate, dict)
    duplicate["controller_call_index"] = 3
    controllers.append(duplicate)
    episode["qwen_controller_calls"] = 3

    passed, checks = _smoke_pass(episode)

    assert passed is False
    assert checks["fresh_controller_grounding_evidence"] is False


def test_smoke_gate_rejects_unattested_valid_looking_model_identity(
    tmp_path: Path,
) -> None:
    episode = _grounded_smoke_episode(tmp_path)
    episode["snapshot_digest"] = _named_digest("unattested snapshot")
    episode["served_model_id"] = "unattested-but-valid-looking-model"

    passed, checks = _smoke_pass(episode)

    assert passed is False
    assert checks["prompt_schema_model_hashes_valid"] is False


def test_smoke_gate_rejects_a_controller_hash_without_an_actual_attempt_record(
    tmp_path: Path,
) -> None:
    episode = _grounded_smoke_episode(tmp_path)
    requests = episode["requests"]
    controllers = episode["controller_records"]
    assert isinstance(requests, list)
    assert isinstance(controllers, list)
    evidence = requests[0]["evidence"]
    assert isinstance(evidence, dict)
    replacement = _named_digest("non-attempt content")
    evidence["controller_request_sha256"] = replacement
    controllers[0]["controller_request_sha256"] = replacement
    controllers[0]["command_evidence_sha256"] = _json_sha256(evidence)

    passed, checks = _smoke_pass(episode)

    assert passed is False
    assert checks["actual_request_hashes_reconcile"] is False


def test_invalid_closed_repair_persists_both_successful_qwen_outcomes() -> None:
    from adaptive import joint_runner

    initial_command = _controller_command("obs-repair", -9.0)
    initial_evidence = {"controller_call_index": 9}
    repair_command = _controller_command("obs-repair", 9.0)
    repair_evidence = {"controller_call_index": 10}

    request = joint_runner._invalid_closed_command_request(
        decision=8,
        observation_id="obs-repair",
        command=initial_command,
        evidence=initial_evidence,
        format_errors=[],
        repair_count=1,
        repair_reason="joint4 target is outside the safe joint-limit inset",
        repair_command=repair_command,
        repair_evidence=repair_evidence,
        error=ValueError("joint4 repair is still outside the inset"),
    )

    assert request["command"] is initial_command
    assert request["evidence"] is initial_evidence
    assert request["repair_command"] is repair_command
    assert request["repair_evidence"] is repair_evidence
    assert request["post_repair_status"] == "invalid_closed_command"
    assert request["repair_error_type"] == "ValueError"


@pytest.mark.parametrize(
    "mutation",
    (
        "attempt_record_digest",
        "attempt_served_model",
        "attempt_response_schema",
        "attempt_finish_reason",
        "attempt_completion_tokens",
        "controller_model_snapshot",
        "controller_model_system_prompt",
        "controller_model_instruction",
        "controller_model_images",
        "critic_model_schema",
        "duplicate_attempt_membership",
    ),
)
def test_smoke_gate_rejects_attempt_or_model_evidence_identity_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    episode = _grounded_smoke_episode(tmp_path)
    attempts = episode["qwen_attempt_records"]
    requests = episode["requests"]
    critics = episode["critic_records"]
    assert isinstance(attempts, list)
    assert isinstance(requests, list)
    assert isinstance(critics, list)
    controller_evidence = requests[0]["evidence"]
    assert isinstance(controller_evidence, dict)
    critic_model_evidence = critics[0]["model_evidence"]
    assert isinstance(critic_model_evidence, dict)

    if mutation == "attempt_record_digest":
        attempts[0]["record_sha256"] = _named_digest("forged attempt record")
    elif mutation == "attempt_served_model":
        attempts[0]["record"]["served_model_id"] = "other-model"
        attempts[0]["record_sha256"] = _json_sha256(attempts[0]["record"])
    elif mutation == "attempt_response_schema":
        attempts[0]["record"]["response_schema_sha256"] = _named_digest(
            "other schema"
        )
        attempts[0]["record_sha256"] = _json_sha256(attempts[0]["record"])
    elif mutation == "attempt_finish_reason":
        attempts[0]["record"]["finish_reason"] = "length"
        attempts[0]["record_sha256"] = _json_sha256(attempts[0]["record"])
    elif mutation == "attempt_completion_tokens":
        attempts[0]["record"]["completion_tokens"] = 8
        attempts[0]["record_sha256"] = _json_sha256(attempts[0]["record"])
    elif mutation == "controller_model_snapshot":
        controller_evidence["snapshot_digest"] = _named_digest("other snapshot")
    elif mutation == "controller_model_system_prompt":
        controller_evidence["system_prompt_sha256"] = _named_digest("other prompt")
    elif mutation == "controller_model_instruction":
        controller_evidence["instruction_sha256"] = _named_digest(
            "other instruction"
        )
    elif mutation == "controller_model_images":
        controller_evidence["image_sha256"]["left"] = _named_digest("other image")
    elif mutation == "critic_model_schema":
        critic_model_evidence["response_schema_sha256"] = _named_digest(
            "other critic schema"
        )
    elif mutation == "duplicate_attempt_membership":
        attempts.append(_json_copy(attempts[0]))

    passed, checks = _smoke_pass(episode)

    assert passed is False
    assert checks["actual_request_hashes_reconcile"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        "controller_boolean",
        "controller_float",
        "controller_negative",
        "controller_out_of_range",
        "controller_evidence_boolean",
        "critic_boolean",
        "critic_float",
        "critic_negative",
        "critic_out_of_range",
        "model_raw_chars_float",
        "model_latency_boolean",
    ),
)
def test_smoke_gate_rejects_non_integral_qwen_attempt_identities(
    tmp_path: Path,
    mutation: str,
) -> None:
    episode = _grounded_smoke_episode(tmp_path)
    attempts = episode["qwen_attempt_records"]
    requests = episode["requests"]
    controllers = episode["controller_records"]
    critics = episode["critic_records"]
    assert isinstance(attempts, list)
    assert isinstance(requests, list)
    assert isinstance(controllers, list)
    assert isinstance(critics, list)
    controller_evidence = requests[0]["evidence"]
    assert isinstance(controller_evidence, dict)

    if mutation == "controller_boolean":
        controllers[0]["qwen_attempt_index"] = False
    elif mutation == "controller_float":
        controllers[0]["qwen_attempt_index"] = 0.0
    elif mutation == "controller_negative":
        controllers[0]["qwen_attempt_index"] = -1
    elif mutation == "controller_out_of_range":
        controllers[0]["qwen_attempt_index"] = 3
        attempts[0]["record"]["attempt_index"] = 3
        attempts[0]["record_sha256"] = _json_sha256(attempts[0]["record"])
    elif mutation == "controller_evidence_boolean":
        controller_evidence["qwen_attempt_index"] = False
        controllers[0]["qwen_attempt_index"] = False
        controllers[0]["command_evidence_sha256"] = _json_sha256(
            controller_evidence
        )
    elif mutation == "critic_boolean":
        critics[0]["qwen_attempt_index"] = False
    elif mutation == "critic_float":
        critics[0]["qwen_attempt_index"] = 0.0
    elif mutation == "critic_negative":
        critics[0]["qwen_attempt_index"] = -1
    elif mutation == "critic_out_of_range":
        critics[0]["qwen_attempt_index"] = 3
        attempts[1]["record"]["attempt_index"] = 3
        attempts[1]["record_sha256"] = _json_sha256(attempts[1]["record"])
    elif mutation == "model_raw_chars_float":
        controller_evidence["raw_chars"] = float(controller_evidence["raw_chars"])
        controllers[0]["command_evidence_sha256"] = _json_sha256(
            controller_evidence
        )
    elif mutation == "model_latency_boolean":
        controller_evidence["latency_s"] = False
        controllers[0]["command_evidence_sha256"] = _json_sha256(
            controller_evidence
        )
        attempts[0]["record"]["latency_s"] = 0.0
        attempts[0]["record_sha256"] = _json_sha256(attempts[0]["record"])

    passed, checks = _smoke_pass(episode)

    assert passed is False
    assert checks["actual_request_hashes_reconcile"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        "controller_length_finish",
        "controller_completion_at_cap",
        "critic_length_finish",
        "critic_completion_at_cap",
    ),
)
def test_smoke_gate_rejects_noncanonical_success_completion_evidence(
    tmp_path: Path,
    mutation: str,
) -> None:
    episode = _grounded_smoke_episode(tmp_path)
    attempts = episode["qwen_attempt_records"]
    requests = episode["requests"]
    controllers = episode["controller_records"]
    critics = episode["critic_records"]
    assert isinstance(attempts, list)
    assert isinstance(requests, list)
    assert isinstance(controllers, list)
    assert isinstance(critics, list)
    controller_evidence = requests[0]["evidence"]
    critic_evidence = critics[0]["model_evidence"]
    assert isinstance(controller_evidence, dict)
    assert isinstance(critic_evidence, dict)

    if mutation == "controller_length_finish":
        controller_evidence["finish_reason"] = "length"
        attempts[0]["record"]["finish_reason"] = "length"
    elif mutation == "controller_completion_at_cap":
        controller_evidence["usage"]["completion_tokens"] = CONTROLLER_MAX_TOKENS
        attempts[0]["record"]["completion_tokens"] = CONTROLLER_MAX_TOKENS
    elif mutation == "critic_length_finish":
        critic_evidence["finish_reason"] = "length"
        attempts[1]["record"]["finish_reason"] = "length"
    elif mutation == "critic_completion_at_cap":
        critic_evidence["usage"]["completion_tokens"] = CRITIC_MAX_TOKENS
        attempts[1]["record"]["completion_tokens"] = CRITIC_MAX_TOKENS

    for index in (0, 1):
        attempts[index]["record_sha256"] = _json_sha256(attempts[index]["record"])
    controllers[0]["command_evidence_sha256"] = _json_sha256(
        controller_evidence
    )

    passed, checks = _smoke_pass(episode)

    assert passed is False
    assert checks["actual_request_hashes_reconcile"] is False


def test_smoke_gate_rejects_legacy_counter_only_fixture() -> None:
    legacy = {
        "qwen_direct_command_count": 2,
        "qwen_joint_command_count": 2,
        "qwen_critic_attempts": 1,
        "critic_records": [{
            "trigger": "first_action",
            "status": "available",
            "critic_request_sha256": "b" * 64,
            "advisory_sha256": "c" * 64,
            "consumed_by_controller_request_sha256": "d" * 64,
            "qwen_command_sha256": "e" * 64,
        }],
        "controller_records": [{
            "consumed_advisory_id": "a1",
            "critic_request_sha256": "b" * 64,
            "advisory_sha256": "c" * 64,
            "controller_request_sha256": "d" * 64,
            "command_sha256": "e" * 64,
        }],
        "critic_origin_executions": 0,
        "status": "policy_finished_false",
        "video_sha256": "a" * 64,
    }

    passed, checks = _smoke_pass(legacy)

    assert passed is False
    assert checks["qwen_joint_command_authority"] is False
    assert checks["fresh_controller_grounding_evidence"] is False
    assert checks["complete_critic_receipt"] is False
    assert checks["grounded_controller_note"] is False


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    (
        ("boolean_joint_count", "qwen_joint_command_authority"),
        ("target_not_from_qwen", "qwen_joint_command_authority"),
        ("boolean_target", "qwen_joint_command_authority"),
        ("stale_command_observation", "qwen_joint_command_authority"),
        ("placeholder_request_hash", "fresh_controller_grounding_evidence"),
        ("changed_public_state", "fresh_controller_grounding_evidence"),
        ("partial_critic_torque", "complete_critic_receipt"),
        ("changed_sealed_receipt", "complete_critic_receipt"),
        ("invalid_enum_advisory", "complete_critic_receipt"),
        ("critic_hash_reordered", "actual_request_hashes_reconcile"),
        ("rendered_advisory_missing", "actual_request_hashes_reconcile"),
        ("advisory_reused", "advisory_consumed_exactly_once"),
        ("unadvised_inserted_before", "advisory_consumed_exactly_once"),
        ("ungrounded_notes", "grounded_controller_note"),
        ("wrong_system_prompt", "prompt_schema_model_hashes_valid"),
        ("wrong_schema_hash", "prompt_schema_model_hashes_valid"),
        ("missing_model_identity", "prompt_schema_model_hashes_valid"),
        ("critic_origin_boolean", "zero_critic_origin_executions"),
        ("video_content_drift", "artifacts_valid"),
        ("counter_drift", "counters_reconcile"),
    ),
)
def test_smoke_gate_rejects_replayed_placeholder_or_unlinked_evidence(
    tmp_path: Path,
    mutation: str,
    failed_check: str,
) -> None:
    episode = _grounded_smoke_episode(tmp_path)
    requests = episode["requests"]
    controllers = episode["controller_records"]
    critics = episode["critic_records"]
    triggers = episode["trigger_evaluations"]
    assert isinstance(requests, list)
    assert isinstance(controllers, list)
    assert isinstance(critics, list)
    assert isinstance(triggers, list)

    if mutation == "boolean_joint_count":
        episode["qwen_joint_command_count"] = True
    elif mutation == "target_not_from_qwen":
        requests[1]["post_repair_command"]["targets"]["joint1"] = 0.04
    elif mutation == "boolean_target":
        requests[1]["command"]["targets"]["joint1"] = True
    elif mutation == "stale_command_observation":
        requests[1]["post_repair_command"]["observation_id"] = "obs-stale"
    elif mutation == "placeholder_request_hash":
        requests[0]["evidence"]["controller_request_sha256"] = "a" * 64
    elif mutation == "changed_public_state":
        requests[1]["evidence"]["critic_trigger_evaluation"][
            "fresh_public_state"
        ]["state.arm_joint_velocity"][0] = 9.0
    elif mutation == "partial_critic_torque":
        triggers[1]["fresh_public_state"]["state.arm_applied_torque"] = {
            "available": False,
            "values_nm": None,
        }
    elif mutation == "changed_sealed_receipt":
        triggers[1]["sealed_public_receipt"]["note"] = "replayed receipt"
    elif mutation == "invalid_enum_advisory":
        critics[0]["advisory"]["suggested_correction"] = "move joint one"
    elif mutation == "critic_hash_reordered":
        critics[0]["consumed_by_controller_request_sha256"] = controllers[0][
            "controller_request_sha256"
        ]
    elif mutation == "rendered_advisory_missing":
        requests[1]["evidence"]["rendered_advisory_sha256"] = None
    elif mutation == "advisory_reused":
        duplicate = _json_copy(controllers[1])
        assert isinstance(duplicate, dict)
        duplicate["controller_call_index"] = 3
        controllers.append(duplicate)
        episode["qwen_controller_calls"] = 3
    elif mutation == "unadvised_inserted_before":
        inserted = _json_copy(controllers[1])
        assert isinstance(inserted, dict)
        inserted["consumed_advisory_id"] = None
        controllers.insert(1, inserted)
        episode["qwen_controller_calls"] = 3
    elif mutation == "ungrounded_notes":
        for request in requests:
            request["command"]["note"] = "make another small safe move"
            request["post_repair_command"]["note"] = "make another small safe move"
    elif mutation == "wrong_system_prompt":
        episode["system_prompt_sha256"] = _named_digest("old prompt")
    elif mutation == "wrong_schema_hash":
        episode["controller_response_schema_sha256"] = _named_digest("old schema")
    elif mutation == "missing_model_identity":
        episode["snapshot_digest"] = None
    elif mutation == "critic_origin_boolean":
        episode["critic_origin_executions"] = False
    elif mutation == "video_content_drift":
        Path(str(episode["video"])).write_bytes(b"changed")
    elif mutation == "counter_drift":
        episode["qwen_critic_successes"] = 2

    passed, checks = _smoke_pass(episode)

    assert passed is False
    assert checks[failed_check] is False


def test_smoke_gate_allows_a_later_same_observation_retry_only_when_unadvised(
    tmp_path: Path,
) -> None:
    episode = _grounded_smoke_episode(tmp_path)
    controllers = episode["controller_records"]
    assert isinstance(controllers, list)
    retry = _json_copy(controllers[1])
    assert isinstance(retry, dict)
    retry.update({
        "controller_call_index": 3,
        "command": None,
        "command_sha256": None,
        "consumed_advisory_id": None,
        "controller_request_sha256": _named_digest("controller retry request"),
        "controller_call_manifest_sha256": _named_digest(
            "controller retry manifest"
        ),
        "critic_request_sha256": None,
        "critic_call_manifest_sha256": None,
        "advisory_sha256": None,
        "status": "controller_unavailable",
        "failure_class": "MalformedResponse",
        "failure_sha256": _named_digest("retry failure"),
    })
    controllers.append(retry)
    episode["qwen_controller_calls"] = 3

    passed, checks = _smoke_pass(episode)

    assert passed is True
    assert checks["advisory_consumed_exactly_once"] is True


def test_smoke_gate_rejects_incomplete_actual_hash_reconciliation() -> None:
    episode = {
        "qwen_direct_command_count": 2,
        "qwen_joint_command_count": 2,
        "qwen_critic_attempts": 1,
        "critic_records": [
            {
                "trigger": "first_action",
                "status": "available",
                "critic_request_sha256": None,
                "advisory_sha256": "c" * 64,
                "consumed_by_controller_request_sha256": None,
                "qwen_command_sha256": "e" * 64,
            }
        ],
        "controller_records": [
            {
                "consumed_advisory_id": "a1",
                "critic_request_sha256": None,
                "advisory_sha256": "c" * 64,
                "controller_request_sha256": None,
                "command_sha256": "e" * 64,
            }
        ],
        "critic_origin_executions": 0,
        "status": "policy_finished_false",
        "video_sha256": "a" * 64,
    }

    passed, checks = _smoke_pass(episode)

    assert passed is False
    assert checks["actual_request_hashes_reconcile"] is False


def _joint_command(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "kind": "move_joints",
        "observation_id": "obs",
        "targets": {"joint1": 0.2, "joint4": -2.1, "gripper": 1.0},
        "note": "move to a visible pregrasp",
    }
    value.update(overrides)
    return value


def _reset_qpos() -> list[float]:
    return [0.0, -1.0, 0.0, -2.2, 0.0, 1.5, 0.7]


def test_move_joints_is_partial_absolute_and_holds_omitted_dimensions() -> None:
    command = decode_joint_command(_joint_command(), observation_id="obs")
    trajectory = interpolate_joint_command(
        command,
        current_qpos=_reset_qpos(),
        current_gripper=0.0,
    )
    assert trajectory.bounded_target[0] == pytest.approx(0.2)
    assert trajectory.bounded_target[1] == pytest.approx(-1.0)
    assert trajectory.bounded_target[3] == pytest.approx(-2.1)
    assert trajectory.gripper_open == pytest.approx(1.0)
    assert trajectory.explicit_mask == (
        True, False, False, True, False, False, False,
    )
    assert set(JOINT_RESPONSE_SCHEMA["properties"]) == {
        "kind", "observation_id", "targets", "note",
    }
    assert JOINT_RESPONSE_SCHEMA["additionalProperties"] is False


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (_joint_command(targets={}), "non-empty"),
        (_joint_command(observation_id="old"), "stale"),
        (_joint_command(targets={"joint8": 0.0}), "unknown"),
        (_joint_command(targets={"joint1": True}), "finite number"),
        (_joint_command(targets={"joint1": float("nan")}), "finite number"),
        (_joint_command(targets={"joint1": JOINT_LIMITS[0][1] - 0.01}), "inset"),
        (_joint_command(targets={"gripper": 1.1}), "gripper"),
        (_joint_command(extra="no"), "fields"),
    ],
)
def test_joint_command_rejects_malformed_or_unsafe_values(
    value: dict[str, object], message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        decode_joint_command(value, observation_id="obs")


def test_joint_interpolation_allows_held_values_outside_only_the_inset() -> None:
    current = _reset_qpos()
    current[1] = JOINT_LIMITS[1][1] - 0.005
    command = decode_joint_command(
        _joint_command(targets={"joint1": 0.1}), observation_id="obs"
    )
    trajectory = interpolate_joint_command(
        command, current_qpos=current, current_gripper=1.0,
    )
    assert trajectory.bounded_target[1] == pytest.approx(current[1])

    current[1] = JOINT_LIMITS[1][1] + 0.001
    with pytest.raises(ValueError, match="hard limit"):
        interpolate_joint_command(
            command, current_qpos=current, current_gripper=1.0,
        )


def test_next_joint_waypoint_is_execution_relative_and_lag_pauses() -> None:
    actual = tuple(_reset_qpos())
    endpoint = tuple([0.2, *actual[1:]])
    first = next_joint_waypoint(actual, None, endpoint)
    assert max(abs(a - b) for a, b in zip(first, actual, strict=True)) <= 0.027
    assert first[0] == pytest.approx(0.025)

    second = next_joint_waypoint(first, first, endpoint)
    assert max(abs(a - b) for a, b in zip(second, first, strict=True)) <= 0.025
    assert second[0] == pytest.approx(0.05)

    lagged_actual = list(second)
    lagged_actual[0] -= 0.05
    paused = next_joint_waypoint(lagged_actual, second, endpoint)
    assert paused == second


def test_next_joint_waypoint_actual_relative_tracking_advances_through_lag() -> None:
    actual = tuple(_reset_qpos())
    endpoint = tuple([0.2, *actual[1:]])
    prior = tuple([0.075, *actual[1:]])

    waypoint = next_joint_waypoint(
        actual,
        prior,
        endpoint,
        tracking_mode="actual_relative",
    )

    assert waypoint[0] == pytest.approx(0.025)
    assert max(
        abs(commanded - realized)
        for commanded, realized in zip(waypoint, actual, strict=True)
    ) <= 0.025


def test_joint_interpolation_is_bounded_and_reports_remaining_error() -> None:
    far = decode_joint_command(
        _joint_command(targets={"joint1": 1.5}), observation_id="obs"
    )
    trajectory = interpolate_joint_command(
        far, current_qpos=_reset_qpos(), current_gripper=1.0,
    )
    assert len(trajectory.waypoints) == 32
    assert trajectory.remaining_error > 0.6
    previous = tuple(_reset_qpos())
    for waypoint in trajectory.waypoints:
        assert max(
            abs(a - b) for a, b in zip(waypoint, previous, strict=True)
        ) <= 0.025 + 1e-12
        previous = waypoint


def test_near_joint_target_reaches_then_settles_twice() -> None:
    near = decode_joint_command(
        _joint_command(targets={"joint1": 0.04}), observation_id="obs"
    )
    trajectory = interpolate_joint_command(
        near, current_qpos=_reset_qpos(), current_gripper=1.0,
    )
    assert trajectory.remaining_error == pytest.approx(0.0)
    assert trajectory.waypoints[-1][0] == pytest.approx(0.04)
    assert trajectory.waypoints[-2:] == (trajectory.bounded_target,) * 2


def test_gripper_only_transition_reserves_h200_verified_hold_actions() -> None:
    command = decode_joint_command(
        _joint_command(targets={"gripper": 0.0}), observation_id="obs"
    )
    trajectory = interpolate_joint_command(
        command, current_qpos=_reset_qpos(), current_gripper=1.0,
    )
    assert len(trajectory.waypoints) >= 16
    assert set(trajectory.waypoints) == {tuple(_reset_qpos())}


def _composite_config() -> dict[str, object]:
    return {
        "type": "HYBRID_MOBILE_BASE",
        "body_parts": {
            "right": {
                "type": "OSC_POSE",
                "input_type": "delta",
                "kp": 150,
                "gripper": {"type": "GRIP"},
            },
            "base": {"type": "JOINT_VELOCITY", "interpolation": "null"},
            "torso": {"type": "JOINT_POSITION", "kp": 2000},
        },
    }


def _child_action(
    qpos: list[float] | None = None,
    *,
    gripper_open: float = 1.0,
    base_motion: list[float] | None = None,
    torso: float = 0.0,
) -> dict[str, object]:
    return {
        "joint_position": qpos or _reset_qpos(),
        "gripper_open": gripper_open,
        "base_motion": base_motion or [0.0, 0.0, 0.0],
        "torso": torso,
    }


@pytest.mark.parametrize("torso_input_type", [None, "delta"])
def test_joint_child_controller_uses_absolute_arm_and_zero_torso_hold(torso_input_type) -> None:
    original = _composite_config()
    if torso_input_type is not None:
        original["body_parts"]["torso"]["input_type"] = torso_input_type
    config = joint_controller_config(original)
    assert original["body_parts"]["right"]["type"] == "OSC_POSE"
    assert config["body_parts"]["right"]["type"] == "JOINT_POSITION"
    assert config["body_parts"]["right"]["input_type"] == "absolute"
    assert len(config["body_parts"]["right"]["input_min"]) == 7
    assert config["body_parts"]["right"]["gripper"] == {"type": "GRIP"}
    assert config["body_parts"]["base"] == original["body_parts"]["base"]
    assert original["body_parts"]["torso"].get("input_type") == torso_input_type
    assert config["body_parts"]["torso"] == {
        **original["body_parts"]["torso"], "input_type": "absolute"
    }
    # With absolute input the existing zero command refers to kitchen reset
    # height, not a zero increment from a displaced torso position.
    assert joint_unmap_action(_child_action())["robot0_torso"] == [0.0]


def test_joint_child_action_mapping_keeps_gripper_base_and_torso_separate() -> None:
    mapped = joint_unmap_action(_child_action(gripper_open=0.0))
    assert mapped["robot0_right"] == _reset_qpos()
    assert mapped["robot0_right_gripper"] == pytest.approx(1.0)
    assert mapped["robot0_base"] == [0.0, 0.0, 0.0]
    assert mapped["robot0_torso"] == [0.0]
    assert mapped["robot0_base_mode"] == pytest.approx(-1.0)
    assert joint_unmap_action(_child_action(gripper_open=1.0))[
        "robot0_right_gripper"
    ] == pytest.approx(-1.0)


def test_joint_child_selects_seven_named_arm_values_from_mobile_robot_qpos() -> None:
    names = ["base_x", "robot0_joint3", "torso", "robot0_joint1",
             "robot0_joint2", "robot0_joint4", "robot0_joint5",
             "robot0_joint6", "robot0_joint7"]
    values = [9.0, 0.3, 8.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7]
    arm_names = [f"robot0_joint{index}" for index in range(1, 8)]
    assert _select_arm_qpos(values, names, arm_names) == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    )


def test_select_named_arm_vector_orders_exact_seven_arm_names() -> None:
    names = [
        "base_x", "robot0_joint3", "torso", "robot0_joint1",
        "robot0_joint2", "robot0_joint4", "robot0_joint5",
        "robot0_joint6", "robot0_joint7",
    ]
    values = [9.0, 0.3, 8.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7]
    arm_names = [f"robot0_joint{index}" for index in range(1, 8)]

    assert select_named_arm_vector(values, names, arm_names, "arm velocity") == [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
    ]


@pytest.mark.parametrize(
    ("values", "names", "arm_names"),
    (
        ([0.0] * 8, [f"joint{index}" for index in range(9)], [f"joint{index}" for index in range(7)]),
        ([0.0] * 7, ["joint1"] * 7, [f"joint{index}" for index in range(1, 8)]),
        ([0.0] * 7, [f"joint{index}" for index in range(1, 8)], ["joint1"] * 7),
        ([0.0] * 7, [f"joint{index}" for index in range(1, 8)], [f"joint{index}" for index in range(2, 9)]),
        ([0.0] * 6, [f"joint{index}" for index in range(1, 7)], [f"joint{index}" for index in range(1, 7)]),
        ([0.0, 0.0, 0.0, True, 0.0, 0.0, 0.0], [f"joint{index}" for index in range(1, 8)], [f"joint{index}" for index in range(1, 8)]),
        ([0.0, 0.0, 0.0, float("nan"), 0.0, 0.0, 0.0], [f"joint{index}" for index in range(1, 8)], [f"joint{index}" for index in range(1, 8)]),
        ([0.0, 0.0, 0.0, float("inf"), 0.0, 0.0, 0.0], [f"joint{index}" for index in range(1, 8)], [f"joint{index}" for index in range(1, 8)]),
    ),
)
def test_select_named_arm_vector_rejects_metadata_and_numeric_drift(
    values: object, names: object, arm_names: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        select_named_arm_vector(values, names, arm_names, "arm velocity")


class _TelemetryController:
    def __init__(self, torques: object) -> None:
        self.torques = torques


class _TelemetryRobot:
    def __init__(self, *, torques: object) -> None:
        self.robot_joints = [
            "base_x", "robot0_joint3", "torso", "robot0_joint1",
            "robot0_joint2", "robot0_joint4", "robot0_joint5",
            "robot0_joint6", "robot0_joint7",
        ]
        self.robot_arm_joints = [f"robot0_joint{index}" for index in range(1, 8)]
        self.part_controllers = {"right": _TelemetryController(torques)}
        self.ee_force = {"right": [1.0, 2.0, 3.0]}
        self.ee_torque = {"right": [0.1, 0.2, 0.3]}

    def get_robot_joint_positions(self) -> list[float]:
        return [9.0, 0.3, 8.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7]


class _TelemetryEnvironment:
    def __init__(self, *, torques: object = (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7), timestep: int = 1) -> None:
        self.unwrapped = self
        self.robots = [_TelemetryRobot(torques=torques)]
        self.timestep = timestep
        half_sqrt_two = math.sqrt(0.5)
        self.raw_observation: dict[str, object] = {
            "robot0_joint_vel": [9.0, 0.5, 8.0, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1],
            "robot0_base_pos": [0.3, -0.1, 0.5],
            "robot0_base_quat": [0.0, 0.0, half_sqrt_two, half_sqrt_two],
            "robot0_base_to_eef_pos": [0.2, 0.1, -2.0],
            "robot0_base_to_eef_quat": [0.0, 0.0, 0.0, 1.0],
            "robot0_object_pose": [99.0, 98.0, 97.0],
            "sim.data.contact": [{"geom1": 1, "geom2": 2}],
            "check_contact": True,
            "reward": 1.0,
            "success": True,
            "depth": [[1.0]],
            "segmentation": [[7]],
        }

    def _get_observations(self, force_update: bool = False) -> dict[str, object]:
        assert force_update is False
        return dict(self.raw_observation)


def _external_camera_calibration() -> dict[str, dict[str, object]]:
    left = _identity_camera_calibration()
    right = dict(left)
    left["camera_name"] = "left"
    left["mujoco_camera_name"] = "robot0_agentview_left"
    right["camera_name"] = "right"
    right["mujoco_camera_name"] = "robot0_agentview_right"
    wrist = dict(left)
    wrist["camera_name"] = "wrist"
    wrist["mujoco_camera_name"] = "robot0_eye_in_hand"
    return {"left": left, "right": right, "wrist": wrist}


def test_read_public_telemetry_publishes_measured_proprioception_and_external_pixels() -> None:
    telemetry = read_public_telemetry(
        _TelemetryEnvironment(), _external_camera_calibration()
    )

    assert telemetry["state.arm_joint_position"] == [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
    ]
    assert telemetry["state.arm_joint_velocity"] == [
        0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
    ]
    assert telemetry["state.arm_applied_torque"] == {
        "available": True,
        "values_nm": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
    }
    assert telemetry["state.end_effector_wrench"] == {
        "force_n": [1.0, 2.0, 3.0],
        "torque_nm": [0.1, 0.2, 0.3],
    }
    assert telemetry["state.end_effector_external_pixels"] == {
        "left": {
            "u_px": pytest.approx(372.8333333333333),
            "v_px": pytest.approx(206.16666666666666),
            "visible": True,
            "depth_valid": True,
        },
        "right": {
            "u_px": pytest.approx(372.8333333333333),
            "v_px": pytest.approx(206.16666666666666),
            "visible": True,
            "depth_valid": True,
        },
    }
    assert len(telemetry["state.arm_translation_jacobian"]) == 3
    assert len(telemetry["state.arm_rotation_jacobian"]) == 3
    assert all(
        len(row) == 7
        for key in ("state.arm_translation_jacobian", "state.arm_rotation_jacobian")
        for row in telemetry[key]
    )


def test_read_public_telemetry_allows_only_initial_torque_unavailable() -> None:
    initial = read_public_telemetry(
        _TelemetryEnvironment(torques=None, timestep=0),
        _external_camera_calibration(),
    )
    assert initial["state.arm_applied_torque"] == {
        "available": False,
        "values_nm": None,
    }

    with pytest.raises(ValueError, match="applied torque"):
        read_public_telemetry(
            _TelemetryEnvironment(torques=None, timestep=1),
            _external_camera_calibration(),
        )


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("raw_observation", "robot0_joint_vel"), [0.0] * 8),
        (("raw_observation", "robot0_joint_vel"), [0.0, 0.0, 0.0, True, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (("raw_observation", "robot0_joint_vel"), [0.0, 0.0, 0.0, float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0]),
        (("raw_observation", "robot0_joint_vel"), [0.0, 0.0, 0.0, float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0]),
        (("robot", "ee_force"), {"right": [0.0, 0.0]}),
        (("robot", "ee_torque"), {"right": [0.0, True, 0.0]}),
        (("controller", "torques"), [0.0] * 8),
    ),
)
def test_read_public_telemetry_rejects_malformed_or_nonfinite_measurements(
    path: tuple[str, str], value: object
) -> None:
    environment = _TelemetryEnvironment()
    owner, attribute = path
    if owner == "raw_observation":
        environment.raw_observation[attribute] = value
    elif owner == "robot":
        setattr(environment.robots[0], attribute, value)
    else:
        setattr(environment.robots[0].part_controllers["right"], attribute, value)

    with pytest.raises((TypeError, ValueError)):
        read_public_telemetry(environment, _external_camera_calibration())


def _telemetry_sample(
    *,
    qpos: list[float],
    qvel: list[float],
    torque: list[float] | None,
    force: list[float],
    wrench_torque: list[float],
) -> dict[str, object]:
    return {
        "state.arm_joint_position": qpos,
        "state.arm_joint_velocity": qvel,
        "state.arm_applied_torque": {
            "available": torque is not None,
            "values_nm": torque,
        },
        "state.end_effector_wrench": {
            "force_n": force,
            "torque_nm": wrench_torque,
        },
        "state.end_effector_external_pixels": {
            "left": {
                "u_px": 100.0 + qpos[0],
                "v_px": 200.0 + qpos[1],
                "visible": True,
                "depth_valid": True,
            },
            "right": {
                "u_px": 300.0 + qpos[0],
                "v_px": 400.0 + qpos[1],
                "visible": True,
                "depth_valid": True,
            },
        },
    }


def test_summarize_telemetry_samples_reports_literal_raw_deltas_and_peaks() -> None:
    before = _telemetry_sample(
        qpos=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        qvel=[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
        torque=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        force=[1.0, 2.0, 3.0],
        wrench_torque=[1.0, 1.0, 1.0],
    )
    first = _telemetry_sample(
        qpos=[1.0, 0.0, 4.0, 1.0, 7.0, 2.0, 10.0],
        qvel=[-3.0, 2.0, -1.0, 4.0, -5.0, 6.0, -7.0],
        torque=[12.0, 17.0, 34.0, 35.0, 56.0, 53.0, 78.0],
        force=[4.0, 6.0, 3.0],
        wrench_torque=[2.0, 3.0, 3.0],
    )
    second = _telemetry_sample(
        qpos=[-2.0, 4.0, -2.0, 8.0, -2.0, 12.0, -2.0],
        qvel=[2.0, -4.0, 6.0, -8.0, 10.0, -12.0, 14.0],
        torque=[6.0, 25.0, 24.0, 47.0, 42.0, 69.0, 60.0],
        force=[7.0, 10.0, 3.0],
        wrench_torque=[0.0, -1.0, -1.0],
    )
    after = _telemetry_sample(
        qpos=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
        qvel=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        torque=[11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0],
        force=[2.0, 3.0, 4.0],
        wrench_torque=[1.5, 1.5, 1.5],
    )

    summary = summarize_telemetry_samples(before, [first, second], after)

    assert summary["arm_joint_position"] == {
        "start_rad": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "end_rad": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
        "delta_rad": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "peak_abs_delta_from_start_rad": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    }
    assert summary["arm_joint_velocity"] == {
        "start_rad_s": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
        "end_rad_s": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "maximum_abs_rad_s": [3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
    }
    assert summary["arm_applied_torque"] == {
        "start_nm": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        "end_nm": [11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0],
        "delta_nm": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "peak_abs_delta_from_start_nm": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    }
    assert summary["end_effector_wrench"]["force"] == {
        "start_n": [1.0, 2.0, 3.0],
        "end_n": [2.0, 3.0, 4.0],
        "delta_n": [1.0, 1.0, 1.0],
        "peak_delta_norm_n": 10.0,
    }
    assert summary["end_effector_wrench"]["torque"] == {
        "start_nm": [1.0, 1.0, 1.0],
        "end_nm": [1.5, 1.5, 1.5],
        "delta_nm": [0.5, 0.5, 0.5],
        "peak_delta_norm_nm": 3.0,
    }
    encoded = json.dumps(summary, sort_keys=True)
    assert not any(
        forbidden in encoded for forbidden in ('"contact"', '"grasped"', '"success"')
    )


@pytest.mark.parametrize("failure", ("empty", "shape", "post_torque"))
def test_summarize_telemetry_samples_rejects_missing_or_drifted_samples(
    failure: str,
) -> None:
    valid = _telemetry_sample(
        qpos=[0.0] * 7,
        qvel=[0.0] * 7,
        torque=[0.0] * 7,
        force=[0.0] * 3,
        wrench_torque=[0.0] * 3,
    )
    samples = [valid]
    after = valid
    if failure == "empty":
        samples = []
    elif failure == "shape":
        samples = [dict(valid)]
        samples[0]["state.arm_joint_velocity"] = [0.0] * 6
    else:
        after = dict(valid)
        after["state.arm_applied_torque"] = {
            "available": False,
            "values_nm": None,
        }

    with pytest.raises((TypeError, ValueError)):
        summarize_telemetry_samples(valid, samples, after)


def test_execute_move_samples_before_each_real_step_and_after_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive import joint_sim_child

    class Environment:
        qpos = _reset_qpos()

    environment = Environment()
    events: list[str] = []
    calibration_reads: list[dict[str, object]] = []
    telemetry_calibration_sequences: list[int] = []

    def arm_qpos(candidate: Environment) -> list[float]:
        return list(candidate.qpos)

    def step(candidate: Environment, action: dict[str, object]) -> dict[str, object]:
        events.append("step")
        candidate.qpos = list(action["joint_position"])
        return {"state.gripper_qpos": [0.04, -0.04]}

    def telemetry(candidate: Environment, calibration: object) -> dict[str, object]:
        events.append("telemetry")
        assert isinstance(calibration, dict)
        telemetry_calibration_sequences.append(calibration["sequence"])
        value = float(events.count("step"))
        return _telemetry_sample(
            qpos=list(candidate.qpos),
            qvel=[value] * 7,
            torque=[10.0 + value] * 7,
            force=[value, 0.0, 0.0],
            wrench_torque=[0.0, value, 0.0],
        )

    monkeypatch.setattr(joint_sim_child, "_arm_qpos", arm_qpos)
    monkeypatch.setattr(joint_sim_child, "_step", step)
    monkeypatch.setattr(joint_sim_child, "read_public_telemetry", telemetry)
    camera_geometry = ModuleType("robocasa_inspect.camera_geometry")

    def official_camera_calibration(candidate: Environment) -> dict[str, object]:
        calibration = {"sequence": len(calibration_reads)}
        calibration_reads.append(calibration)
        return calibration

    camera_geometry.official_camera_calibration = (  # type: ignore[attr-defined]
        official_camera_calibration
    )
    monkeypatch.setitem(
        sys.modules, "robocasa_inspect.camera_geometry", camera_geometry
    )
    endpoint = _reset_qpos()
    endpoint[0] += 0.05
    command = {
        "schema": "robocasa-inspect-joint-command/v1",
        "sequence": 0,
        "kind": "move_joints",
        "observation_id": "obs",
        "endpoint": endpoint,
        "explicit_mask": [True, False, False, False, False, False, False],
        "gripper_open": 1.0,
        "previous_gripper_open": 1.0,
        "gripper_transition": False,
        "max_actions": 4,
    }

    _, execution, _, total_actions = _execute_move(
        environment,
        command,
        current_gripper=1.0,
        total_actions=0,
        started=joint_sim_child.time.monotonic(),
    )

    assert execution["step_count"] == total_actions == 4
    assert events == [
        "telemetry",
        "step",
        "telemetry",
        "step",
        "telemetry",
        "step",
        "telemetry",
        "step",
        "telemetry",
        "telemetry",
    ]
    assert telemetry_calibration_sequences == list(range(len(events) // 2 + 1))
    assert len(calibration_reads) == len(telemetry_calibration_sequences)
    assert (
        execution["telemetry_summary"]["arm_joint_velocity"]["maximum_abs_rad_s"]
        == [4.0] * 7
    )
    assert execution["end_effector_external_pixels_before"]["left"]["u_px"] == 100.0
    assert execution["end_effector_external_pixels_after"]["left"]["u_px"] == 100.05


def test_execute_move_can_continue_after_formal_budget_in_development_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive import joint_sim_child

    class Environment:
        qpos = _reset_qpos()

    environment = Environment()
    monkeypatch.setattr(
        joint_sim_child, "_arm_qpos", lambda candidate: list(candidate.qpos)
    )

    def step(candidate: Environment, action: dict[str, object]) -> dict[str, object]:
        candidate.qpos = list(action["joint_position"])
        return {"state.gripper_qpos": [0.04, -0.04]}

    monkeypatch.setattr(joint_sim_child, "_step", step)
    monkeypatch.setattr(
        joint_sim_child,
        "read_public_telemetry",
        lambda candidate, _calibration: _telemetry_sample(
            qpos=list(candidate.qpos),
            qvel=[0.0] * 7,
            torque=[0.0] * 7,
            force=[0.0] * 3,
            wrench_torque=[0.0] * 3,
        ),
    )
    camera_geometry = ModuleType("robocasa_inspect.camera_geometry")
    camera_geometry.official_camera_calibration = lambda _candidate: {}  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "robocasa_inspect.camera_geometry", camera_geometry
    )
    endpoint = _reset_qpos()
    endpoint[0] += 0.01

    _, execution, _, total_actions = _execute_move(
        environment,
        {
            "schema": "robocasa-inspect-joint-command/v1",
            "sequence": 0,
            "kind": "move_joints",
            "observation_id": "obs",
            "endpoint": endpoint,
            "explicit_mask": [True, False, False, False, False, False, False],
            "gripper_open": 1.0,
            "previous_gripper_open": 1.0,
            "gripper_transition": False,
            "max_actions": 1,
        },
        current_gripper=1.0,
        total_actions=450,
        started=joint_sim_child.time.monotonic(),
        action_budget=900,
    )

    assert execution["step_count"] == 1
    assert total_actions == 451


def test_execute_move_settles_only_after_measured_endpoint_tracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive import joint_sim_child

    class Environment:
        qpos = _reset_qpos()

    environment = Environment()
    monkeypatch.setattr(
        joint_sim_child, "_arm_qpos", lambda candidate: list(candidate.qpos)
    )

    def step(candidate: Environment, action: dict[str, object]) -> dict[str, object]:
        candidate.qpos = [
            current + 0.5 * (commanded - current)
            for current, commanded in zip(
                candidate.qpos, action["joint_position"], strict=True
            )
        ]
        return {"state.gripper_qpos": [0.04, -0.04]}

    monkeypatch.setattr(joint_sim_child, "_step", step)
    monkeypatch.setattr(
        joint_sim_child,
        "read_public_telemetry",
        lambda candidate, _calibration: _telemetry_sample(
            qpos=list(candidate.qpos),
            qvel=[0.0] * 7,
            torque=[0.0] * 7,
            force=[0.0] * 3,
            wrench_torque=[0.0] * 3,
        ),
    )
    camera_geometry = ModuleType("robocasa_inspect.camera_geometry")
    camera_geometry.official_camera_calibration = lambda candidate: {  # type: ignore[attr-defined]
        "left": {},
        "right": {},
        "wrist": {},
    }
    monkeypatch.setitem(
        sys.modules, "robocasa_inspect.camera_geometry", camera_geometry
    )
    endpoint = _reset_qpos()
    endpoint[0] += 0.04

    _, execution, _, total_actions = _execute_move(
        environment,
        {
            "schema": "robocasa-inspect-joint-command/v1",
            "sequence": 0,
            "kind": "move_joints",
            "observation_id": "obs",
            "endpoint": endpoint,
            "explicit_mask": [True, False, False, False, False, False, False],
            "gripper_open": 1.0,
            "previous_gripper_open": 1.0,
            "gripper_transition": False,
            "max_actions": 10,
        },
        current_gripper=1.0,
        total_actions=0,
        started=joint_sim_child.time.monotonic(),
    )

    assert execution["step_count"] == total_actions == 6
    assert execution["endpoint_error"] <= 0.002


def test_execute_move_actual_relative_tracking_uses_each_measured_qpos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive import joint_sim_child

    class Environment:
        qpos = _reset_qpos()

    environment = Environment()
    commanded_from: list[tuple[list[float], list[float]]] = []
    monkeypatch.setattr(
        joint_sim_child, "_arm_qpos", lambda candidate: list(candidate.qpos)
    )

    def step(candidate: Environment, action: dict[str, object]) -> dict[str, object]:
        before = list(candidate.qpos)
        commanded = list(action["joint_position"])
        commanded_from.append((before, commanded))
        candidate.qpos = [
            realized + 0.1 * (target - realized)
            for realized, target in zip(before, commanded, strict=True)
        ]
        return {"state.gripper_qpos": [0.04, -0.04]}

    monkeypatch.setattr(joint_sim_child, "_step", step)
    monkeypatch.setattr(
        joint_sim_child,
        "_read_fresh_public_telemetry",
        lambda candidate: _telemetry_sample(
            qpos=list(candidate.qpos),
            qvel=[0.0] * 7,
            torque=[0.0] * 7,
            force=[0.0] * 3,
            wrench_torque=[0.0] * 3,
        ),
    )
    endpoint = _reset_qpos()
    endpoint[0] += 0.2

    _, execution, _, total_actions = _execute_move(
        environment,
        {
            "schema": "robocasa-inspect-joint-command/v1",
            "sequence": 0,
            "kind": "move_joints",
            "observation_id": "obs",
            "endpoint": endpoint,
            "explicit_mask": [True, False, False, False, False, False, False],
            "gripper_open": 1.0,
            "previous_gripper_open": 1.0,
            "gripper_transition": False,
            "max_actions": 4,
            "tracking_mode": "actual_relative",
        },
        current_gripper=1.0,
        total_actions=0,
        started=joint_sim_child.time.monotonic(),
    )

    assert execution["step_count"] == total_actions == 4
    assert execution["tracking_pause_count"] == 0
    assert [commanded[0] for _, commanded in commanded_from] == sorted(
        commanded[0] for _, commanded in commanded_from
    )
    for realized, commanded in commanded_from:
        assert max(
            abs(target - actual)
            for target, actual in zip(commanded, realized, strict=True)
        ) <= 0.025 + 1e-12


def test_execute_move_fails_closed_when_wall_timeout_truncates_gripper_settling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive import joint_sim_child

    class Environment:
        qpos = _reset_qpos()

    environment = Environment()
    actions: list[dict[str, object]] = []
    monkeypatch.setattr(
        joint_sim_child, "_arm_qpos", lambda candidate: list(candidate.qpos)
    )

    def step(candidate: Environment, action: dict[str, object]) -> dict[str, object]:
        actions.append(action)
        candidate.qpos = list(action["joint_position"])
        return {"state.gripper_qpos": [0.0, 0.0]}

    def telemetry(candidate: Environment, calibration: object) -> dict[str, object]:
        return _telemetry_sample(
            qpos=list(candidate.qpos),
            qvel=[0.0] * 7,
            torque=[0.0] * 7,
            force=[0.0] * 3,
            wrench_torque=[0.0] * 3,
        )

    monkeypatch.setattr(joint_sim_child, "_step", step)
    monkeypatch.setattr(joint_sim_child, "read_public_telemetry", telemetry)
    ticks = iter((0.0, 1_201.0))
    monkeypatch.setattr(joint_sim_child.time, "monotonic", lambda: next(ticks))
    camera_geometry = ModuleType("robocasa_inspect.camera_geometry")
    camera_geometry.official_camera_calibration = lambda candidate: {}  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "robocasa_inspect.camera_geometry", camera_geometry
    )

    with pytest.raises(RuntimeError, match="gripper transition|settling"):
        _execute_move(
            environment,
            {
                "schema": "robocasa-inspect-joint-command/v1",
                "sequence": 0,
                "kind": "move_joints",
                "observation_id": "obs",
                "endpoint": _reset_qpos(),
                "explicit_mask": [False] * 7,
                "gripper_open": 0.0,
                "previous_gripper_open": 1.0,
                "gripper_transition": True,
                "max_actions": MIN_GRIPPER_ACTIONS,
            },
            current_gripper=1.0,
            total_actions=0,
            started=0.0,
        )
    assert len(actions) == 1


def test_execute_base_samples_before_each_real_step_and_after_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive import joint_sim_child

    class Environment:
        qpos = _reset_qpos()

    environment = Environment()
    events: list[str] = []
    applied_actions: list[dict[str, object]] = []
    calibration_reads: list[dict[str, object]] = []
    telemetry_calibration_sequences: list[int] = []

    monkeypatch.setattr(
        joint_sim_child, "_arm_qpos", lambda candidate: list(candidate.qpos)
    )

    def step(candidate: Environment, action: dict[str, object]) -> dict[str, object]:
        events.append("step")
        applied_actions.append(action)
        if any(action["base_motion"]):
            candidate.qpos = list(candidate.qpos)
            candidate.qpos[0] += 0.001
        return {"state.gripper_qpos": [0.04, -0.04]}

    def telemetry(candidate: Environment, calibration: object) -> dict[str, object]:
        events.append("telemetry")
        assert isinstance(calibration, dict)
        telemetry_calibration_sequences.append(calibration["sequence"])
        step_count = float(events.count("step"))
        return _telemetry_sample(
            qpos=list(candidate.qpos),
            qvel=[step_count] * 7,
            torque=[20.0 + step_count] * 7,
            force=[0.0, step_count, 0.0],
            wrench_torque=[0.0, 0.0, step_count],
        )

    monkeypatch.setattr(joint_sim_child, "_step", step)
    monkeypatch.setattr(joint_sim_child, "read_public_telemetry", telemetry)
    camera_geometry = ModuleType("robocasa_inspect.camera_geometry")

    def official_camera_calibration(candidate: Environment) -> dict[str, object]:
        calibration = {"sequence": len(calibration_reads)}
        calibration_reads.append(calibration)
        return calibration

    camera_geometry.official_camera_calibration = (  # type: ignore[attr-defined]
        official_camera_calibration
    )
    monkeypatch.setitem(
        sys.modules, "robocasa_inspect.camera_geometry", camera_geometry
    )

    _, execution, _, total_actions = _execute_base(
        environment,
        {
            "schema": "robocasa-inspect-joint-command/v1",
            "sequence": 0,
            "kind": "base_action",
            "observation_id": "obs",
            "axis": "yaw",
            "normalized_velocity": -0.2,
            "gripper_open": 0.0,
            "previous_gripper_open": 1.0,
            "gripper_transition": True,
        },
        current_gripper=1.0,
        total_actions=450 - MIN_GRIPPER_ACTIONS,
        started=joint_sim_child.time.monotonic(),
    )

    assert execution["step_count"] == MIN_GRIPPER_ACTIONS
    assert total_actions == 450
    assert execution["base_motion_step_count"] == 5
    assert events == (
        ["telemetry"]
        + [event for _ in range(MIN_GRIPPER_ACTIONS) for event in ("step", "telemetry")]
        + ["telemetry"]
    )
    assert telemetry_calibration_sequences == list(
        range(MIN_GRIPPER_ACTIONS + 2)
    )
    assert len(calibration_reads) == len(telemetry_calibration_sequences)
    assert [action["base_motion"] for action in applied_actions[:5]] == [
        [0.0, 0.0, -0.2]
    ] * 5
    assert [action["base_motion"] for action in applied_actions[5:]] == [
        [0.0, 0.0, 0.0]
    ] * (MIN_GRIPPER_ACTIONS - 5)
    assert {action["gripper_open"] for action in applied_actions} == {0.0}
    assert 450 - total_actions == 0
    assert (
        execution["telemetry_summary"]["arm_joint_velocity"]["maximum_abs_rad_s"]
        == [float(MIN_GRIPPER_ACTIONS)] * 7
    )
    assert execution["tracking_pause_count"] == 0
    assert execution["remaining_endpoint_error"] == pytest.approx(0.005)


def test_execute_base_hold_remains_exactly_five_moving_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive import joint_sim_child

    class Environment:
        qpos = _reset_qpos()

    environment = Environment()
    applied_actions: list[dict[str, object]] = []
    monkeypatch.setattr(
        joint_sim_child, "_arm_qpos", lambda candidate: list(candidate.qpos)
    )

    def step(candidate: Environment, action: dict[str, object]) -> dict[str, object]:
        applied_actions.append(action)
        return {"state.gripper_qpos": [0.04, -0.04]}

    def telemetry(candidate: Environment, calibration: object) -> dict[str, object]:
        return _telemetry_sample(
            qpos=list(candidate.qpos),
            qvel=[0.0] * 7,
            torque=[0.0] * 7,
            force=[0.0] * 3,
            wrench_torque=[0.0] * 3,
        )

    monkeypatch.setattr(joint_sim_child, "_step", step)
    monkeypatch.setattr(joint_sim_child, "read_public_telemetry", telemetry)
    camera_geometry = ModuleType("robocasa_inspect.camera_geometry")
    camera_geometry.official_camera_calibration = lambda candidate: {}  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "robocasa_inspect.camera_geometry", camera_geometry
    )

    _, execution, _, total_actions = _execute_base(
        environment,
        {
            "schema": "robocasa-inspect-joint-command/v1",
            "sequence": 0,
            "kind": "base_action",
            "observation_id": "obs",
            "axis": "x",
            "normalized_velocity": 0.2,
            "gripper_open": 1.0,
            "previous_gripper_open": 1.0,
            "gripper_transition": False,
        },
        current_gripper=1.0,
        total_actions=0,
        started=joint_sim_child.time.monotonic(),
    )

    assert execution["step_count"] == total_actions == 5
    assert execution["base_motion_step_count"] == 5
    assert [action["base_motion"] for action in applied_actions] == [
        [0.2, 0.0, 0.0]
    ] * 5


def test_child_rejects_gripper_transitions_that_cannot_complete_sixteen_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive import joint_sim_child

    monkeypatch.setattr(
        joint_sim_child, "_arm_qpos", lambda candidate: _reset_qpos()
    )
    move = {
        "schema": "robocasa-inspect-joint-command/v1",
        "sequence": 0,
        "kind": "move_joints",
        "observation_id": "obs",
        "endpoint": _reset_qpos(),
        "explicit_mask": [False] * 7,
        "gripper_open": 0.0,
        "previous_gripper_open": 1.0,
        "gripper_transition": True,
        "max_actions": MIN_GRIPPER_ACTIONS - 1,
    }
    with pytest.raises((ValueError, RuntimeError), match="16|sixteen"):
        _execute_move(
            object(),
            move,
            current_gripper=1.0,
            total_actions=0,
            started=joint_sim_child.time.monotonic(),
        )

    base = {
        "schema": "robocasa-inspect-joint-command/v1",
        "sequence": 0,
        "kind": "base_action",
        "observation_id": "obs",
        "axis": "x",
        "normalized_velocity": 0.2,
        "gripper_open": 0.0,
        "previous_gripper_open": 1.0,
        "gripper_transition": True,
    }
    with pytest.raises(RuntimeError, match="16|sixteen"):
        _execute_base(
            object(),
            base,
            current_gripper=1.0,
            total_actions=450 - MIN_GRIPPER_ACTIONS + 1,
            started=joint_sim_child.time.monotonic(),
        )


_OFFICIAL_CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
_OFFICIAL_PUBLIC_STATE_KEYS = (
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.base_position",
    "state.base_rotation",
    "state.gripper_qpos",
)


class _TestImage:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.mode = "RGB"
        self.size = (64, 64)

    def convert(self, mode: str) -> "_TestImage":
        assert mode == "RGB"
        return self

    def copy(self) -> "_TestImage":
        return _TestImage(self.content)

    def putpixel(
        self, _point: tuple[int, int], _color: tuple[int, int, int]
    ) -> None:
        return None

    def save(self, path: Path, *, format: str) -> None:
        assert format == "PNG"
        path.write_bytes(self.content)


class _TestImageFactory:
    @staticmethod
    def fromarray(value: bytes) -> _TestImage:
        return _TestImage(value)


def _install_publish_dependencies(
    monkeypatch: pytest.MonkeyPatch, *, inject_hidden_state: bool = False
) -> None:
    numpy = ModuleType("numpy")
    numpy.uint8 = object()  # type: ignore[attr-defined]
    numpy.asarray = lambda value, dtype: value  # type: ignore[attr-defined]
    pil = ModuleType("PIL")
    pil.Image = _TestImageFactory  # type: ignore[attr-defined]
    contracts = ModuleType("robocasa_inspect.contracts")
    contracts.CAMERAS = _OFFICIAL_CAMERA_KEYS  # type: ignore[attr-defined]

    def project_observation(
        raw: dict[str, object], *, episode: str, sequence: int
    ) -> SimpleNamespace:
        assert episode == "episode-1"
        assert sequence == 3
        state = {key: raw[key] for key in _OFFICIAL_PUBLIC_STATE_KEYS}
        if inject_hidden_state:
            state["state.object_pose"] = [4.0, 5.0, 6.0]
        return SimpleNamespace(
            observation_id="official-observation-id",
            instruction=raw["annotation.human.task_description"],
            images={key: raw[key] for key in _OFFICIAL_CAMERA_KEYS},
            state_groups=state,
        )

    contracts.project_observation = project_observation  # type: ignore[attr-defined]
    camera_geometry = ModuleType("robocasa_inspect.camera_geometry")
    camera_geometry.official_camera_calibration = (  # type: ignore[attr-defined]
        lambda environment: _external_camera_calibration()
    )
    package = ModuleType("robocasa_inspect")
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setitem(sys.modules, "robocasa_inspect", package)
    monkeypatch.setitem(
        sys.modules, "robocasa_inspect.camera_geometry", camera_geometry
    )
    monkeypatch.setitem(sys.modules, "robocasa_inspect.contracts", contracts)


def _publish_raw_observation(environment: _TelemetryEnvironment) -> dict[str, object]:
    raw = {
        "annotation.human.task_description": "open the toaster",
        "state.end_effector_position_relative": [0.2, 0.1, -2.0],
        "state.end_effector_rotation_relative": [0.0, 0.0, 0.0, 1.0],
        "state.base_position": [0.3, -0.1, 0.5],
        "state.base_rotation": list(environment.raw_observation["robot0_base_quat"]),
        "state.gripper_qpos": [0.04, -0.04],
    }
    raw.update(
        {
            camera: label.encode()
            for camera, label in zip(
                _OFFICIAL_CAMERA_KEYS, ("left", "right", "wrist"), strict=True
            )
        }
    )
    return raw


def _safe_execution() -> dict[str, object]:
    telemetry = _telemetry_sample(
        qpos=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        qvel=[0.0] * 7,
        torque=[1.0] * 7,
        force=[0.0] * 3,
        wrench_torque=[0.0] * 3,
    )
    return {
        "kind": "base_action",
        "accepted": True,
        "axis": "x",
        "normalized_velocity": 0.1,
        "gripper_intent": 1.0,
        "step_count": 5,
        "base_motion_step_count": 5,
        "realized_arm_qpos": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "tracking_pause_count": 0,
        "remaining_endpoint_error": 0.03,
        "telemetry_summary": summarize_telemetry_samples(
            telemetry, [telemetry], telemetry
        ),
        "end_effector_external_pixels_before": telemetry[
            "state.end_effector_external_pixels"
        ],
        "end_effector_external_pixels_after": telemetry[
            "state.end_effector_external_pixels"
        ],
    }


def test_publish_observation_serializes_only_closed_public_schema_and_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_publish_dependencies(monkeypatch)
    environment = _TelemetryEnvironment()
    (tmp_path / "frames").mkdir()
    (tmp_path / "mailbox").mkdir()

    record = _publish_observation(
        _publish_raw_observation(environment),
        environment=environment,
        run=tmp_path,
        episode="episode-1",
        sequence=3,
        execution=_safe_execution(),
    )

    assert set(record) == {
        "schema",
        "episode",
        "sequence",
        "observation_id",
        "instruction",
        "images",
        "public_state",
        "camera_calibration",
        "execution",
    }
    assert set(record["public_state"]) == {
        *_OFFICIAL_PUBLIC_STATE_KEYS,
        "state.arm_joint_position",
        "state.arm_joint_velocity",
        "state.arm_applied_torque",
        "state.end_effector_wrench",
        "state.arm_translation_jacobian",
        "state.arm_rotation_jacobian",
        "state.end_effector_external_pixels",
    }
    assert set(record["execution"]) == {
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
    assert set(record["images"]) == {"left", "right", "wrist"}
    telemetry = {
        key: record["public_state"][key]
        for key in (
            "state.arm_joint_position",
            "state.arm_joint_velocity",
            "state.arm_applied_torque",
            "state.end_effector_wrench",
            "state.arm_translation_jacobian",
            "state.arm_rotation_jacobian",
            "state.end_effector_external_pixels",
        )
    }
    expected_observation_id = hashlib.sha256(
        json.dumps(
            {
                "official_observation_id": "official-observation-id",
                "telemetry": telemetry,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert record["observation_id"] == expected_observation_id
    persisted = json.loads((tmp_path / "mailbox/observation-000003.json").read_text())
    assert persisted == record
    encoded = json.dumps(record, sort_keys=True)
    for forbidden in (
        "object_pose",
        "reward",
        '"success"',
        '"depth"',
        '"depth_m"',
        "segmentation",
        '"contact"',
    ):
        assert forbidden not in encoded
    assert "robot0_object_pose" not in encoded
    assert "wrist" not in record["public_state"][
        "state.end_effector_external_pixels"
    ]


def test_publish_observation_accepts_initial_torque_unavailable_public_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_publish_dependencies(monkeypatch)
    environment = _TelemetryEnvironment(torques=None, timestep=0)
    (tmp_path / "frames").mkdir()
    (tmp_path / "mailbox").mkdir()

    record = _publish_observation(
        _publish_raw_observation(environment),
        environment=environment,
        run=tmp_path,
        episode="episode-1",
        sequence=3,
        execution=None,
    )

    assert record["public_state"]["state.arm_applied_torque"] == {
        "available": False,
        "values_nm": None,
    }


def test_publish_observation_validates_the_exact_merged_public_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from adaptive import joint_sim_child

    _install_publish_dependencies(monkeypatch)
    environment = _TelemetryEnvironment()
    (tmp_path / "frames").mkdir()
    (tmp_path / "mailbox").mkdir()

    def malformed_telemetry(
        _environment: object, _camera_calibration: object
    ) -> dict[str, object]:
        telemetry = read_public_telemetry(
            _TelemetryEnvironment(), _external_camera_calibration()
        )
        telemetry["state.end_effector_external_pixels"]["left"] = {
            "u_px": None,
            "v_px": None,
            "visible": True,
            "depth_valid": False,
        }
        return telemetry

    monkeypatch.setattr(joint_sim_child, "read_public_telemetry", malformed_telemetry)

    with pytest.raises(ValueError, match="public state"):
        joint_sim_child._publish_observation(
            _publish_raw_observation(environment),
            environment=environment,
            run=tmp_path,
            episode="episode-1",
            sequence=3,
            execution=None,
        )


@pytest.mark.parametrize("hidden_source", ("public_state", "execution"))
def test_publish_observation_rejects_hidden_fields_from_every_record_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hidden_source: str
) -> None:
    _install_publish_dependencies(
        monkeypatch, inject_hidden_state=hidden_source == "public_state"
    )
    environment = _TelemetryEnvironment()
    (tmp_path / "frames").mkdir()
    (tmp_path / "mailbox").mkdir()
    execution = _safe_execution()
    if hidden_source == "execution":
        execution["success"] = True

    with pytest.raises(ValueError, match="public observation schema"):
        _publish_observation(
            _publish_raw_observation(environment),
            environment=environment,
            run=tmp_path,
            episode="episode-1",
            sequence=3,
            execution=execution,
        )


@pytest.mark.parametrize(
    ("container_path", "hidden_field"),
    (
        (("telemetry_summary",), "contact"),
        (("telemetry_summary", "arm_applied_torque"), "grasped"),
        (("telemetry_summary", "end_effector_wrench", "force"), "success"),
    ),
)
def test_publish_observation_rejects_hidden_nested_summary_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    container_path: tuple[str, ...],
    hidden_field: str,
) -> None:
    _install_publish_dependencies(monkeypatch)
    environment = _TelemetryEnvironment()
    (tmp_path / "frames").mkdir()
    (tmp_path / "mailbox").mkdir()
    execution = json.loads(json.dumps(_safe_execution()))
    container = execution
    for key in container_path:
        container = container[key]
    container[hidden_field] = True

    with pytest.raises(ValueError, match="telemetry summary"):
        _publish_observation(
            _publish_raw_observation(environment),
            environment=environment,
            run=tmp_path,
            episode="episode-1",
            sequence=3,
            execution=execution,
        )


def test_joint_child_validates_first_and_later_waypoint_bounds() -> None:
    endpoint = _reset_qpos()
    endpoint[0] = 0.05
    first = _reset_qpos()
    first[0] = 0.025
    second = _reset_qpos()
    second[0] = 0.05
    validated = validate_joint_action_sequence(
        [_child_action(first), _child_action(second)],
        current_qpos=_reset_qpos(),
        endpoint=endpoint,
        explicit_mask=[True, False, False, False, False, False, False],
    )
    assert len(validated) == 2

    bad_first = _reset_qpos()
    bad_first[0] = 0.028
    with pytest.raises(ValueError, match="first waypoint"):
        validate_joint_action_sequence(
            [_child_action(bad_first)],
            current_qpos=_reset_qpos(),
            endpoint=endpoint,
            explicit_mask=[True, False, False, False, False, False, False],
        )
    bad_second = list(first)
    bad_second[0] += 0.026
    with pytest.raises(ValueError, match="later waypoint"):
        validate_joint_action_sequence(
            [_child_action(first), _child_action(bad_second)],
            current_qpos=_reset_qpos(),
            endpoint=endpoint,
            explicit_mask=[True, False, False, False, False, False, False],
        )


def test_joint_child_rejects_malformed_limits_and_unbounded_body_actions() -> None:
    endpoint = _reset_qpos()
    mask = [False] * 7
    valid = [_child_action()]
    cases: list[tuple[list[dict[str, object]], list[float], list[bool], str]] = [
        (valid * 33, endpoint, mask, "32"),
        ([_child_action([0.0] * 6)], endpoint, mask, "seven"),
        ([_child_action([float("nan"), *_reset_qpos()[1:]])], endpoint, mask, "finite"),
        (valid, [JOINT_LIMITS[0][1] - 0.01, *endpoint[1:]], [True, *mask[1:]], "inset"),
        (valid, [endpoint[0], JOINT_LIMITS[1][1] + 0.01, *endpoint[2:]], mask, "hard limit"),
        ([_child_action(base_motion=[0.26, 0.0, 0.0])], endpoint, mask, "base"),
        ([_child_action(base_motion=[0.1, 0.1, 0.0])], endpoint, mask, "one axis"),
        ([_child_action(torso=0.1)], endpoint, mask, "torso"),
    ]
    for actions, candidate_endpoint, candidate_mask, message in cases:
        with pytest.raises(ValueError, match=message):
            validate_joint_action_sequence(
                actions,
                current_qpos=_reset_qpos(),
                endpoint=candidate_endpoint,
                explicit_mask=candidate_mask,
            )

    assert validate_joint_action_sequence(
        [_child_action(base_motion=[0.25, 0.0, 0.0])],
        current_qpos=_reset_qpos(),
        endpoint=endpoint,
        explicit_mask=mask,
    )


def test_joint_runner_accepts_only_controller_authority() -> None:
    command = _joint_command(targets={"joint1": 0.2, "gripper": 0.0})
    mailbox, decoded = prepare_joint_mailbox(
        command,
        source="controller",
        observation_id="obs",
        current_qpos=_reset_qpos(),
        current_gripper=1.0,
        remaining_actions=450,
        sequence=0,
    )
    assert mailbox["kind"] == "move_joints"
    assert mailbox["endpoint"][0] == pytest.approx(0.2)
    assert mailbox["endpoint"][1] == pytest.approx(-1.0)
    assert mailbox["explicit_mask"] == [
        True, False, False, False, False, False, False,
    ]
    assert mailbox["gripper_open"] == pytest.approx(0.0)
    assert mailbox["previous_gripper_open"] == pytest.approx(1.0)
    assert mailbox["gripper_transition"] is True
    assert mailbox["max_actions"] == 32
    assert decoded.kind == "move_joints"

    with pytest.raises(ValueError, match="controller"):
        prepare_joint_mailbox(
            command,
            source="critic",
            observation_id="obs",
            current_qpos=_reset_qpos(),
            current_gripper=1.0,
            remaining_actions=450,
            sequence=0,
        )


def test_joint_runner_forwards_qwen_actual_relative_tracking_mode() -> None:
    command = _joint_command(targets={"joint1": 0.2, "gripper": 1.0})
    command["tracking_mode"] = "actual_relative"

    mailbox, decoded = prepare_joint_mailbox(
        command,
        source="controller",
        observation_id="obs",
        current_qpos=_reset_qpos(),
        current_gripper=1.0,
        remaining_actions=450,
        sequence=0,
    )

    assert decoded.tracking_mode == "actual_relative"
    assert mailbox["tracking_mode"] == "actual_relative"
    assert mailbox["endpoint"][0] == pytest.approx(0.2)


def test_joint_runner_reserves_sixteen_actions_for_every_gripper_transition() -> None:
    move_transition = _joint_command(targets={"gripper": 0.0})
    with pytest.raises(ValueError, match="16|sixteen"):
        prepare_joint_mailbox(
            move_transition,
            source="controller",
            observation_id="obs",
            current_qpos=_reset_qpos(),
            current_gripper=1.0,
            remaining_actions=MIN_GRIPPER_ACTIONS - 1,
            sequence=0,
        )
    mailbox, _ = prepare_joint_mailbox(
        move_transition,
        source="controller",
        observation_id="obs",
        current_qpos=_reset_qpos(),
        current_gripper=1.0,
        remaining_actions=MIN_GRIPPER_ACTIONS,
        sequence=0,
    )
    assert mailbox["max_actions"] == MIN_GRIPPER_ACTIONS

    base_transition = {
        "kind": "base_action",
        "observation_id": "obs",
        "axis": "x",
        "normalized_velocity": 0.2,
        "gripper": "close",
        "note": "move base while closing",
    }
    with pytest.raises(ValueError, match="16|sixteen"):
        prepare_joint_mailbox(
            base_transition,
            source="controller",
            observation_id="obs",
            current_qpos=_reset_qpos(),
            current_gripper=1.0,
            remaining_actions=MIN_GRIPPER_ACTIONS - 1,
            sequence=0,
        )
    base_mailbox, _ = prepare_joint_mailbox(
        base_transition,
        source="controller",
        observation_id="obs",
        current_qpos=_reset_qpos(),
        current_gripper=1.0,
        remaining_actions=MIN_GRIPPER_ACTIONS,
        sequence=0,
    )
    assert base_mailbox["gripper_open"] == 0.0
    assert base_mailbox["previous_gripper_open"] == 1.0
    assert base_mailbox["gripper_transition"] is True


def test_joint_runner_keeps_nontransition_action_minima_unchanged() -> None:
    move_mailbox, _ = prepare_joint_mailbox(
        _joint_command(targets={"joint1": 0.01}),
        source="controller",
        observation_id="obs",
        current_qpos=_reset_qpos(),
        current_gripper=1.0,
        remaining_actions=1,
        sequence=0,
    )
    assert move_mailbox["max_actions"] == 1
    assert move_mailbox["gripper_transition"] is False

    base_mailbox, _ = prepare_joint_mailbox(
        {
            "kind": "base_action",
            "observation_id": "obs",
            "axis": "x",
            "normalized_velocity": 0.2,
            "gripper": "hold",
            "note": "base-only pulse",
        },
        source="controller",
        observation_id="obs",
        current_qpos=_reset_qpos(),
        current_gripper=1.0,
        remaining_actions=5,
        sequence=0,
    )
    assert base_mailbox["gripper_open"] == 1.0
    assert base_mailbox["gripper_transition"] is False


def _stationary_receipt_boundary_inputs() -> tuple[
    dict[str, object], dict[str, object], dict[str, dict[str, object]]
]:
    pixels = {
        "left": {"u_px": 10.0, "v_px": 20.0, "visible": True, "depth_valid": True},
        "right": {"u_px": 30.0, "v_px": 40.0, "visible": True, "depth_valid": True},
    }
    state = {
        "state.arm_joint_position": _reset_qpos(),
        "state.end_effector_position_relative": [0.2, 0.1, 0.5],
        "state.end_effector_rotation_relative": [0.0, 0.0, 0.0, 1.0],
        "state.gripper_qpos": [0.0, 0.0],
        "state.end_effector_external_pixels": pixels,
    }
    telemetry = _telemetry_sample(
        qpos=_reset_qpos(),
        qvel=[0.0] * 7,
        torque=[0.0] * 7,
        force=[0.0] * 3,
        wrench_torque=[0.0] * 3,
    )
    execution_evidence = {
        "realized_arm_qpos": _reset_qpos(),
        "telemetry_summary": summarize_telemetry_samples(
            telemetry, [telemetry], telemetry
        ),
        "end_effector_external_pixels_before": pixels,
        "end_effector_external_pixels_after": pixels,
    }
    return state, execution_evidence, _external_camera_calibration()


def test_receipt_accepts_live_world_extrinsics_for_base_parented_external_cameras(
) -> None:
    """The external cameras are rigid to mobilebase0_support, not world-fixed."""

    mailbox, command = prepare_joint_mailbox(
        _joint_command(targets={"joint1": 0.0}),
        source="controller",
        observation_id="obs",
        current_qpos=_reset_qpos(),
        current_gripper=0.0,
        remaining_actions=32,
        sequence=0,
    )
    state, evidence, before_calibration = _stationary_receipt_boundary_inputs()
    after_calibration = _json_copy(before_calibration)
    assert isinstance(after_calibration, dict)
    for label, translation in (
        ("left", [6.658e-6, -4.124e-6, -4.430e-6]),
        ("right", [6.679e-6, 18.594e-6, -4.430e-6]),
    ):
        camera = after_calibration[label]
        camera["camera_position_world_m"] = translation
        camera["camera_xmat_world"] = [
            [0.999999999, -0.00003, 0.00002],
            [0.00003, 0.999999999, -0.00001],
            [-0.00002, 0.00001, 0.999999999],
        ]
    execution = {
        "kind": "move_joints",
        "accepted": True,
        "bounded_endpoint": mailbox["endpoint"],
        "gripper_intent": 0.0,
        "step_count": 3,
        "maximum_commanded_step": 0.0,
        "endpoint_error": 0.0,
        "minimum_hard_limit_margin": 0.5,
        "tracking_pause_count": 0,
        **evidence,
    }

    receipt = finalize_joint_receipt(
        command,
        mailbox,
        execution,
        before_state=state,
        after_state=state,
        before_calibration=before_calibration,
        after_calibration=after_calibration,
    )

    assert receipt["accepted"] is True
    assert receipt["end_effector_external_pixel_displacement"]["left"][
        "delta_px"
    ] == [0.0, 0.0]


def test_joint_receipt_binds_actual_relative_tracking_mailbox() -> None:
    command_value = _joint_command(targets={"joint1": 0.2, "gripper": 1.0})
    command_value["tracking_mode"] = "actual_relative"
    mailbox, command = prepare_joint_mailbox(
        command_value,
        source="controller",
        observation_id="obs",
        current_qpos=_reset_qpos(),
        current_gripper=1.0,
        remaining_actions=32,
        sequence=0,
    )
    state, evidence, calibration = _stationary_receipt_boundary_inputs()
    execution = {
        "kind": "move_joints",
        "accepted": True,
        "bounded_endpoint": mailbox["endpoint"],
        "gripper_intent": 1.0,
        "step_count": 4,
        "maximum_commanded_step": 0.025,
        "endpoint_error": 0.2,
        "minimum_hard_limit_margin": 0.5,
        "tracking_pause_count": 0,
        **evidence,
    }

    receipt = finalize_joint_receipt(
        command,
        mailbox,
        execution,
        before_state=state,
        after_state=state,
        before_calibration=calibration,
        after_calibration=calibration,
    )

    assert receipt["requested_targets"] == {"joint1": 0.2, "gripper": 1.0}


def test_command_dict_preserves_actual_relative_tracking_mode() -> None:
    from adaptive.joint_protocol import decode_joint_command
    from adaptive.joint_runner import _command_dict

    value = _joint_command(targets={"joint1": 0.2, "gripper": 1.0})
    value["tracking_mode"] = "actual_relative"
    command = decode_joint_command(value, observation_id="obs")

    assert _command_dict(command) == value


@pytest.mark.parametrize(
    ("previous_gripper", "target_gripper", "boolean_intent"),
    ((1.0, 0.0, False), (0.0, 1.0, True)),
    ids=("close-false", "open-true"),
)
def test_move_receipt_rejects_boolean_gripper_intent(
    previous_gripper: float, target_gripper: float, boolean_intent: bool
) -> None:
    mailbox, command = prepare_joint_mailbox(
        _joint_command(targets={"gripper": target_gripper}),
        source="controller",
        observation_id="obs",
        current_qpos=_reset_qpos(),
        current_gripper=previous_gripper,
        remaining_actions=MIN_GRIPPER_ACTIONS,
        sequence=0,
    )
    state, evidence, calibration = _stationary_receipt_boundary_inputs()
    execution = {
        "kind": "move_joints",
        "accepted": True,
        "bounded_endpoint": mailbox["endpoint"],
        "gripper_intent": boolean_intent,
        "step_count": MIN_GRIPPER_ACTIONS,
        "maximum_commanded_step": 0.0,
        "endpoint_error": 0.0,
        "minimum_hard_limit_margin": 0.5,
        "tracking_pause_count": 0,
        **evidence,
    }

    with pytest.raises(ValueError, match="finite number"):
        finalize_joint_receipt(
            command,
            mailbox,
            execution,
            before_state=state,
            after_state=state,
            before_calibration=calibration,
            after_calibration=calibration,
        )


@pytest.mark.parametrize(
    ("previous_gripper", "gripper_command", "boolean_intent"),
    ((1.0, "close", False), (0.0, "open", True)),
    ids=("close-false", "open-true"),
)
def test_base_receipt_rejects_boolean_gripper_intent(
    previous_gripper: float, gripper_command: str, boolean_intent: bool
) -> None:
    from adaptive import joint_runner

    mailbox, command = prepare_joint_mailbox(
        {
            "kind": "base_action",
            "observation_id": "obs",
            "axis": "x",
            "normalized_velocity": 0.2,
            "gripper": gripper_command,
            "note": "boolean mutation probe",
        },
        source="controller",
        observation_id="obs",
        current_qpos=_reset_qpos(),
        current_gripper=previous_gripper,
        remaining_actions=MIN_GRIPPER_ACTIONS,
        sequence=0,
    )
    state, evidence, calibration = _stationary_receipt_boundary_inputs()
    execution = {
        "kind": "base_action",
        "accepted": True,
        "axis": "x",
        "normalized_velocity": 0.2,
        "gripper_intent": boolean_intent,
        "step_count": MIN_GRIPPER_ACTIONS,
        "base_motion_step_count": 5,
        "tracking_pause_count": 0,
        "remaining_endpoint_error": 0.0,
        **evidence,
    }

    with pytest.raises(ValueError, match="finite number"):
        joint_runner._base_receipt(
            command,
            mailbox,
            execution,
            before_state=state,
            after_state=state,
            before_calibration=calibration,
            after_calibration=calibration,
        )


def test_joint_runner_receipt_is_closed_and_realized_before_next_call() -> None:
    command = _joint_command(targets={"joint1": 0.2, "gripper": 0.0})
    mailbox, decoded = prepare_joint_mailbox(
        command,
        source="controller",
        observation_id="obs",
        current_qpos=_reset_qpos(),
        current_gripper=1.0,
        remaining_actions=20,
        sequence=4,
    )
    assert mailbox["max_actions"] == 20
    before_state = {
        "state.arm_joint_position": _reset_qpos(),
        "state.end_effector_position_relative": [0.10, 0.20, 0.30],
        "state.end_effector_rotation_relative": [0.0, 0.0, 0.0, 1.0],
        "state.gripper_qpos": [0.04, -0.04],
        "state.end_effector_external_pixels": {
            "left": {
                "u_px": 100.0,
                "v_px": 200.0,
                "visible": True,
                "depth_valid": True,
            },
            "right": {
                "u_px": 300.0,
                "v_px": 400.0,
                "visible": True,
                "depth_valid": True,
            },
        },
    }
    after_qpos = [0.19, *_reset_qpos()[1:]]
    after_state = {
        "state.arm_joint_position": after_qpos,
        "state.end_effector_position_relative": [0.11, 0.18, 0.33],
        "state.end_effector_rotation_relative": [
            0.0,
            0.0,
            math.sqrt(0.5),
            math.sqrt(0.5),
        ],
        "state.gripper_qpos": [0.01, -0.01],
        "state.end_effector_external_pixels": {
            "left": {
                "u_px": 103.0,
                "v_px": 204.0,
                "visible": True,
                "depth_valid": True,
            },
            "right": {
                "u_px": 294.0,
                "v_px": 408.0,
                "visible": True,
                "depth_valid": True,
            },
        },
    }
    before_telemetry = _telemetry_sample(
        qpos=list(before_state["state.arm_joint_position"]),
        qvel=[0.0] * 7,
        torque=[1.0] * 7,
        force=[1.0, 2.0, 3.0],
        wrench_torque=[0.1, 0.2, 0.3],
    )
    after_telemetry = _telemetry_sample(
        qpos=after_qpos,
        qvel=[0.1] * 7,
        torque=[2.0] * 7,
        force=[4.0, 6.0, 3.0],
        wrench_torque=[1.1, 2.2, 2.3],
    )
    execution = {
        "kind": "move_joints",
        "accepted": True,
        "bounded_endpoint": mailbox["endpoint"],
        "gripper_intent": 0,
        "step_count": MIN_GRIPPER_ACTIONS,
        "maximum_commanded_step": 0.025,
        "realized_arm_qpos": after_qpos,
        "endpoint_error": 0.01,
        "minimum_hard_limit_margin": 0.5,
        "tracking_pause_count": 1,
        "telemetry_summary": summarize_telemetry_samples(
            before_telemetry, [after_telemetry], after_telemetry
        ),
        "end_effector_external_pixels_before": before_state[
            "state.end_effector_external_pixels"
        ],
        "end_effector_external_pixels_after": after_state[
            "state.end_effector_external_pixels"
        ],
    }
    calibration = _external_camera_calibration()
    moved_wrist_calibration = {
        label: dict(camera) for label, camera in calibration.items()
    }
    moved_wrist_calibration["wrist"]["camera_position_world_m"] = [0.4, -0.2, 1.1]
    moved_wrist_calibration["wrist"]["camera_xmat_world"] = [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    short_transition = dict(execution, step_count=MIN_GRIPPER_ACTIONS - 1)
    with pytest.raises(ValueError, match="gripper transition|16|sixteen"):
        finalize_joint_receipt(
            decoded,
            mailbox,
            short_transition,
            before_state=before_state,
            after_state=after_state,
            before_calibration=calibration,
            after_calibration=moved_wrist_calibration,
        )
    receipt = finalize_joint_receipt(
        decoded,
        mailbox,
        execution,
        before_state=before_state,
        after_state=after_state,
        before_calibration=calibration,
        after_calibration=moved_wrist_calibration,
    )
    assert set(receipt) == PUBLIC_JOINT_RECEIPT_FIELDS
    assert receipt["requested_targets"] == {"joint1": 0.2, "gripper": 0.0}
    assert receipt["gripper_intent"] == 0.0
    assert type(receipt["gripper_intent"]) is float
    assert receipt["resolved_held_dimensions"]["joint2"] == pytest.approx(-1.0)
    assert receipt["realized_arm_qpos"][0] == pytest.approx(0.19)
    assert receipt["realized_arm_qpos_delta"] == pytest.approx(
        [0.19, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    assert receipt["end_effector_pose_delta"]["translation_m"] == pytest.approx(
        [0.01, -0.02, 0.03]
    )
    assert receipt["end_effector_pose_delta"][
        "rotation_axis_angle_rad"
    ] == pytest.approx([0.0, 0.0, math.pi / 2.0])
    assert receipt["end_effector_external_pixel_displacement"]["left"] == {
        "start_px": [100.0, 200.0],
        "end_px": [103.0, 204.0],
        "delta_px": [3.0, 4.0],
        "distance_px": 5.0,
        "start_visible": True,
        "end_visible": True,
        "start_depth_valid": True,
        "end_depth_valid": True,
    }
    assert (
        receipt["end_effector_external_pixel_displacement"]["right"]["distance_px"]
        == 10.0
    )
    assert (
        receipt["telemetry_summary"]["arm_applied_torque"]["delta_nm"]
        == [
            1.0,
        ]
        * 7
    )
    assert (
        receipt["telemetry_summary"]["end_effector_wrench"]["force"][
            "peak_delta_norm_n"
        ]
        == 5.0
    )
    assert receipt["gripper_residual"] == {
        "start_qpos": [0.04, -0.04],
        "end_qpos": [0.01, -0.01],
        "qpos_delta": [-0.03, 0.03],
        "measured_end_finger_separation": 0.02,
    }
    assert receipt["mean_absolute_rgb_change"] is None
    assert receipt["tracking_pause_count"] == 1
    assert receipt["accepted"] is True

    leaked_execution = json.loads(json.dumps(execution))
    leaked_execution["telemetry_summary"]["end_effector_wrench"]["torque"][
        "grasped"
    ] = True
    with pytest.raises(ValueError, match="telemetry summary"):
        finalize_joint_receipt(
            decoded,
            mailbox,
            leaked_execution,
            before_state=before_state,
            after_state=after_state,
            before_calibration=calibration,
            after_calibration=moved_wrist_calibration,
        )

    drifted_calibration = dict(calibration)
    drifted_calibration["left"] = dict(calibration["left"], fx_px=401.0)
    with pytest.raises(ValueError, match="calibration drifted"):
        finalize_joint_receipt(
            decoded,
            mailbox,
            execution,
            before_state=before_state,
            after_state=after_state,
            before_calibration=calibration,
            after_calibration=drifted_calibration,
        )

    drifted_wrist_intrinsics = {
        label: dict(camera) for label, camera in moved_wrist_calibration.items()
    }
    drifted_wrist_intrinsics["wrist"]["fx_px"] = 401.0
    with pytest.raises(ValueError, match="calibration drifted"):
        finalize_joint_receipt(
            decoded,
            mailbox,
            execution,
            before_state=before_state,
            after_state=after_state,
            before_calibration=calibration,
            after_calibration=drifted_wrist_intrinsics,
        )


def test_fresh_images_seal_only_the_immediately_preceding_receipt() -> None:
    from adaptive import joint_runner

    pending_receipt = _sealed_critic_receipt(
        _controller_command("obs-4"),
        _state(0.01),
    )
    pending_receipt["mean_absolute_rgb_change"] = None
    sealed = joint_runner._seal_receipt_rgb_change(
        pending_receipt,
        {"left": 1.25, "right": 2.5, "wrist": 3.75},
        source_sequence=4,
        fresh_sequence=5,
    )
    receipts = [
        {
            "kind": "base_action",
            "ordinal": index,
            "mean_absolute_rgb_change": {"left": 0.0, "right": 0.0, "wrist": 0.0},
        }
        for index in range(12)
    ] + [sealed]
    instruction = json.loads(
        joint_runner._instruction(
            {"instruction": "open the toaster", "public_state": {}},
            receipts=receipts,
            change={"left": 1.25, "right": 2.5, "wrist": 3.75},
            repair=None,
        )
    )

    assert len(instruction["recent_receipts"]) == 12
    assert instruction["recent_receipts"][-1]["mean_absolute_rgb_change"] == {
        "left": 1.25,
        "right": 2.5,
        "wrist": 3.75,
    }
    assert instruction["recent_receipts"][0]["ordinal"] == 1
    with pytest.raises(ValueError, match="fresh observation"):
        joint_runner._seal_receipt_rgb_change(
            pending_receipt,
            {"left": 1.25, "right": 2.5, "wrist": 3.75},
            source_sequence=4,
            fresh_sequence=4,
        )


def test_controller_output_reserve_fits_complete_seven_receipt_h200_prompt() -> None:
    # Exact /tokenize and successful production-client replay evidence from the
    # Task-8 OpenToasterOvenDoor observation that previously failed HTTP 400.
    prompt_tokens = 32_474
    observed_complete_response_tokens = 176
    served_context_tokens = 32_768

    assert CONTROLLER_MAX_TOKENS == 256
    assert observed_complete_response_tokens < CONTROLLER_MAX_TOKENS
    assert prompt_tokens + CONTROLLER_MAX_TOKENS <= served_context_tokens


def test_proposal_controller_uses_larger_output_reserve_and_shorter_history() -> None:
    from adaptive import joint_runner

    receipts = [
        {
            "ordinal": index,
            "mean_absolute_rgb_change": {
                "left": 0.0,
                "right": 0.0,
                "wrist": 0.0,
            },
        }
        for index in range(12)
    ]
    instruction = json.loads(
        joint_runner._instruction(
            {"instruction": "generic task", "public_state": {}},
            receipts=receipts,
            change={"left": 0.0, "right": 0.0, "wrist": 0.0},
            repair=None,
            receipt_limit=8,
        )
    )

    assert joint_runner.PROPOSAL_CONTROLLER_MAX_TOKENS == 1024
    assert len(instruction["recent_receipts"]) == 8
    assert instruction["recent_receipts"][0]["ordinal"] == 4


def test_base_receipt_closes_the_same_public_action_effects() -> None:
    from adaptive import joint_runner

    mailbox, command = prepare_joint_mailbox(
        {
            "kind": "base_action",
            "observation_id": "base-obs",
            "axis": "x",
            "normalized_velocity": 0.2,
            "gripper": "close",
            "note": "bounded base calibration pulse",
        },
        source="controller",
        observation_id="base-obs",
        current_qpos=_reset_qpos(),
        current_gripper=1.0,
        remaining_actions=30,
        sequence=7,
    )
    before_pixels = {
        "left": {"u_px": 10.0, "v_px": 20.0, "visible": True, "depth_valid": True},
        "right": {"u_px": 30.0, "v_px": 40.0, "visible": True, "depth_valid": True},
    }
    after_pixels = {
        "left": {"u_px": 11.0, "v_px": 22.0, "visible": True, "depth_valid": True},
        "right": {"u_px": 33.0, "v_px": 44.0, "visible": True, "depth_valid": True},
    }
    before_state = {
        "state.arm_joint_position": _reset_qpos(),
        "state.end_effector_position_relative": [0.2, 0.1, 0.5],
        "state.end_effector_rotation_relative": [0.0, 0.0, 0.0, 1.0],
        "state.gripper_qpos": [0.04, -0.04],
        "state.end_effector_external_pixels": before_pixels,
    }
    after_state = {
        "state.arm_joint_position": _reset_qpos(),
        "state.end_effector_position_relative": [0.2, 0.1, 0.5],
        "state.end_effector_rotation_relative": [0.0, 0.0, 0.0, 1.0],
        "state.gripper_qpos": [0.0, 0.0],
        "state.end_effector_external_pixels": after_pixels,
    }
    before_telemetry = _telemetry_sample(
        qpos=_reset_qpos(),
        qvel=[0.0] * 7,
        torque=[1.0] * 7,
        force=[0.0] * 3,
        wrench_torque=[0.0] * 3,
    )
    after_telemetry = _telemetry_sample(
        qpos=_reset_qpos(),
        qvel=[0.2] * 7,
        torque=[1.5] * 7,
        force=[1.0, 0.0, 0.0],
        wrench_torque=[0.0, 1.0, 0.0],
    )
    execution = {
        "kind": "base_action",
        "accepted": True,
        "axis": "x",
        "normalized_velocity": 0.2,
        "gripper_intent": 0,
        "step_count": MIN_GRIPPER_ACTIONS,
        "base_motion_step_count": 5,
        "realized_arm_qpos": _reset_qpos(),
        "tracking_pause_count": 0,
        "remaining_endpoint_error": 0.0,
        "telemetry_summary": summarize_telemetry_samples(
            before_telemetry, [after_telemetry], after_telemetry
        ),
        "end_effector_external_pixels_before": before_pixels,
        "end_effector_external_pixels_after": after_pixels,
    }
    calibration = _external_camera_calibration()

    receipt = joint_runner._base_receipt(
        command,
        mailbox,
        execution,
        before_state=before_state,
        after_state=after_state,
        before_calibration=calibration,
        after_calibration=dict(calibration),
    )

    assert set(receipt) == PUBLIC_BASE_RECEIPT_FIELDS
    assert receipt["base_motion_step_count"] == 5
    assert receipt["gripper_intent"] == 0.0
    assert type(receipt["gripper_intent"]) is float
    assert receipt["tracking_pause_count"] == 0
    assert receipt["remaining_endpoint_error"] == 0.0
    assert receipt["realized_arm_qpos_delta"] == [0.0] * 7
    assert receipt["end_effector_external_pixel_displacement"]["left"]["delta_px"] == [
        1.0,
        2.0,
    ]
    assert (
        receipt["telemetry_summary"]["arm_applied_torque"]["delta_nm"]
        == [
            0.5,
        ]
        * 7
    )
    assert receipt["gripper_residual"]["measured_end_finger_separation"] == 0.0
    assert receipt["mean_absolute_rgb_change"] is None

    short_transition = dict(execution, step_count=5)
    with pytest.raises(ValueError, match="base action accounting"):
        joint_runner._base_receipt(
            command,
            mailbox,
            short_transition,
            before_state=before_state,
            after_state=after_state,
            before_calibration=calibration,
            after_calibration=dict(calibration),
        )

    hold_mailbox, hold_command = prepare_joint_mailbox(
        {
            "kind": "base_action",
            "observation_id": "base-obs",
            "axis": "x",
            "normalized_velocity": 0.2,
            "gripper": "hold",
            "note": "bounded base hold pulse",
        },
        source="controller",
        observation_id="base-obs",
        current_qpos=_reset_qpos(),
        current_gripper=1.0,
        remaining_actions=30,
        sequence=7,
    )
    oversized_hold = dict(execution, gripper_intent=1.0)
    with pytest.raises(ValueError, match="base action accounting"):
        joint_runner._base_receipt(
            hold_command,
            hold_mailbox,
            oversized_hold,
            before_state=before_state,
            after_state=after_state,
            before_calibration=calibration,
            after_calibration=dict(calibration),
        )

    for field, value in (
        ("axis", "y"),
        ("normalized_velocity", 0.1),
        ("gripper_intent", 1.0),
    ):
        drifted = dict(execution)
        drifted[field] = value
        with pytest.raises(ValueError, match="base execution .* drifted"):
            joint_runner._base_receipt(
                command,
                mailbox,
                drifted,
                before_state=before_state,
                after_state=after_state,
                before_calibration=calibration,
                after_calibration=dict(calibration),
            )


def test_joint_runner_schema_exposes_bounded_cartesian_but_no_critic_fields() -> None:
    encoded = json.dumps(CONTROLLER_RESPONSE_SCHEMA, sort_keys=True)
    assert "move_joints" in encoded
    assert "base_action" in encoded
    assert "finish" in encoded
    assert "give_up" in encoded
    assert "cartesian_delta" in encoded
    assert "translation_m" in encoded
    assert "rotation_axis_angle_rad" in encoded
    assert "diagnosis" not in encoded


def _safety_evidence() -> dict[str, object]:
    reset = _reset_qpos()
    outbound = list(reset)
    outbound[0] += 0.045
    returned = list(reset)
    returned[0] += 0.005
    return {
        "reset_qpos": reset,
        "outbound_qpos": outbound,
        "return_qpos": returned,
        "reset_gripper_qpos": [0.02, -0.02],
        "closed_gripper_qpos": [0.001, -0.001],
        "reopened_gripper_qpos": [0.038, -0.038],
        "maximum_commanded_steps": [0.025, 0.025, 0.0, 0.0],
        "minimum_hard_limit_margin": 0.4,
        "action_count": 18,
        "terminal_action_count": 18,
        "simulator_error": None,
    }


def test_safety_smoke_requires_outbound_return_bounds_and_gripper_direction() -> None:
    passed, checks = evaluate_safety_smoke(_safety_evidence())
    assert passed is True
    assert all(checks.values())

    inverted = _safety_evidence()
    inverted["closed_gripper_qpos"] = [0.038, -0.038]
    inverted["reopened_gripper_qpos"] = [0.001, -0.001]
    passed, checks = evaluate_safety_smoke(inverted)
    assert passed is False
    assert checks["gripper_close_direction"] is False
    assert checks["gripper_reopen_direction"] is False


def test_safety_smoke_rejects_accounting_or_joint_bound_drift() -> None:
    overstep = _safety_evidence()
    overstep["maximum_commanded_steps"] = [0.028]
    assert evaluate_safety_smoke(overstep)[0] is False
    accounting = _safety_evidence()
    accounting["terminal_action_count"] = 19
    assert evaluate_safety_smoke(accounting)[0] is False
    failed = _safety_evidence()
    failed["simulator_error"] = "controller failure"
    assert evaluate_safety_smoke(failed)[0] is False


def _grounding_pose_record(qpos: list[float]) -> dict[str, object]:
    pose = panda_fk(qpos)
    return {
        "position_m": list(pose.position_m),
        "rotation_matrix": [list(row) for row in pose.rotation_matrix],
    }


def _grounding_calibration() -> dict[str, object]:
    left = _identity_camera_calibration()
    left["camera_name"] = "video.robot0_agentview_left"
    left["camera_position_world_m"] = [0.0, 0.0, 3.0]
    right = json.loads(json.dumps(left))
    right["camera_name"] = "video.robot0_agentview_right"
    right["mujoco_camera_name"] = "robot0_agentview_right"
    return {
        "source": "robocasa_inspect.camera_geometry.official_camera_calibration",
        "left": left,
        "right": right,
    }


def _grounding_pixels(
    qpos: list[float], calibration: Mapping[str, object]
) -> dict[str, object]:
    point = panda_fk(qpos).position_m
    output: dict[str, object] = {}
    for camera in ("left", "right"):
        projected = project_world_point(point, calibration[camera])
        output[camera] = {
            "u_px": projected["u_px"],
            "v_px": projected["v_px"],
            "visible": projected["visible"],
            "depth_valid": True,
        }
    return output


def _grounding_telemetry(
    qpos: list[float], calibration: Mapping[str, object]
) -> dict[str, object]:
    jacobian = panda_local_jacobian(qpos)
    return {
        "state.arm_joint_position": list(qpos),
        "state.arm_joint_velocity": [0.0] * 7,
        "state.arm_applied_torque": {
            "available": True,
            "values_nm": [1.0] * 7,
        },
        "state.end_effector_wrench": {
            "force_n": [0.1, 0.2, 0.3],
            "torque_nm": [0.01, 0.02, 0.03],
        },
        "state.arm_translation_jacobian": jacobian["translation"],
        "state.arm_rotation_jacobian": jacobian["rotation"],
        "state.end_effector_external_pixels": _grounding_pixels(
            qpos, calibration
        ),
    }


def _grounding_command_audit(
    start: list[float],
    end: list[float],
    *,
    calibration: Mapping[str, object],
    gripper_open: float = 1.0,
    action_count: int = 4,
    first_step_rad: float | None = None,
) -> dict[str, object]:
    if action_count == 4:
        if first_step_rad is None:
            midpoint = [
                start_value + (end_value - start_value) / 2.0
                for start_value, end_value in zip(start, end, strict=True)
            ]
        else:
            changed = next(
                index
                for index, (start_value, end_value) in enumerate(
                    zip(start, end, strict=True)
                )
                if not math.isclose(start_value, end_value, abs_tol=1e-12)
            )
            midpoint = list(start)
            direction = 1.0 if end[changed] > start[changed] else -1.0
            midpoint[changed] += direction * first_step_rad
        commanded = [midpoint, list(end), list(end), list(end)]
    else:
        assert start == end
        commanded = [list(end) for _ in range(action_count)]
    actions: list[dict[str, object]] = []
    telemetry_samples: list[dict[str, object]] = []
    realized = list(start)
    for index, target in enumerate(commanded):
        actions.append(
            {
                "realized_before": list(realized),
                "commanded_qpos": list(target),
                "realized_after": list(target),
                "gripper_open": gripper_open,
                "base_motion": [0.0, 0.0, 0.0],
                "torso": 0.0,
            }
        )
        realized = list(target)
        telemetry = _grounding_telemetry(realized, calibration)
        telemetry["state.arm_joint_velocity"][0] = index * 0.001
        telemetry["state.arm_applied_torque"]["values_nm"][0] = 1.0 + index * 0.01
        telemetry["state.end_effector_wrench"]["force_n"][0] = 0.1 + index * 0.005
        telemetry["state.end_effector_wrench"]["torque_nm"][0] = 0.01 + index * 0.001
        telemetry_samples.append(telemetry)
    before = _grounding_telemetry(start, calibration)
    completion = _grounding_telemetry(end, calibration)
    summary = summarize_telemetry_samples(before, telemetry_samples, completion)
    maximum_step = 0.0
    previous = list(start)
    for target in commanded:
        maximum_step = max(
            maximum_step,
            max(
                abs(value - prior)
                for value, prior in zip(target, previous, strict=True)
            ),
        )
        previous = target
    minimum_margin = min(
        min(value - lower, upper - value)
        for target in [start, *commanded]
        for value, (lower, upper) in zip(target, JOINT_LIMITS, strict=True)
    )
    return {
        "receipt": {
            "kind": "move_joints",
            "accepted": True,
            "bounded_endpoint": list(end),
            "gripper_intent": gripper_open,
            "step_count": action_count,
            "maximum_commanded_step": maximum_step,
            "realized_arm_qpos": list(end),
            "endpoint_error": 0.0,
            "minimum_hard_limit_margin": minimum_margin,
            "tracking_pause_count": 0,
            "telemetry_summary": summary,
            "end_effector_external_pixels_before": before[
                "state.end_effector_external_pixels"
            ],
            "end_effector_external_pixels_after": completion[
                "state.end_effector_external_pixels"
            ],
        },
        "telemetry": {
            "before": before,
            "samples": telemetry_samples,
            "completion": completion,
        },
        "actions": actions,
    }


def _grounding_audits_in_source_order(
    evidence: Mapping[str, object],
) -> list[tuple[str, dict[str, object]]]:
    audits = [
        ("baseline settle", evidence["baseline"]),
        ("rest baseline", evidence["rest_baseline"]),
    ]
    for index, probe in enumerate(evidence["probes"]):
        audits.extend(
            (
                (f"probe {index} outbound", probe["outbound"]),
                (f"probe {index} return", probe["return"]),
            )
        )
    audits.extend(
        (
            ("gripper close", evidence["gripper"]["close"]),
            ("gripper reopen", evidence["gripper"]["reopen"]),
        )
    )
    return audits


def _test_canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _bind_test_grounding_journal(evidence: dict[str, object]) -> Mapping[str, object]:
    """Build the independent source journal used by synthetic evaluator evidence."""

    grounding = importlib.import_module("adaptive.panda_grounding_smoke")
    key = b"round-three-test-journal-key"
    zero_hash = "0" * 64
    prior_hash = zero_hash
    event_index = 0
    events = []
    open_qpos = list(evidence["gripper"]["reset_qpos"])
    closed_qpos = list(evidence["gripper"]["closed_qpos"])
    reopened_qpos = list(evidence["gripper"]["reopened_qpos"])
    audits = _grounding_audits_in_source_order(evidence)

    def record(
        *,
        label: str,
        kind: str,
        snapshot: Mapping[str, object],
    ) -> None:
        nonlocal event_index, prior_hash
        snapshot_json = _test_canonical_json(snapshot)
        envelope = _test_canonical_json(
            {
                "index": event_index,
                "audit_label": label,
                "kind": kind,
                "snapshot_json": snapshot_json,
                "previous_sha256": prior_hash,
            }
        ).encode()
        digest = hmac.new(key, envelope, hashlib.sha256).hexdigest()
        events.append(
            grounding._JournalEvent(
                index=event_index,
                audit_label=label,
                kind=kind,
                snapshot_json=snapshot_json,
                previous_sha256=prior_hash,
                sha256=digest,
            )
        )
        event_index += 1
        prior_hash = digest

    for position, (label, audit) in enumerate(audits):
        if position < len(audits) - 2:
            before_qpos = after_qpos = open_qpos
        elif position == len(audits) - 2:
            before_qpos, after_qpos = open_qpos, closed_qpos
        else:
            before_qpos, after_qpos = closed_qpos, reopened_qpos
        telemetry = audit["telemetry"]
        actions = audit["actions"]
        start_index = event_index
        record(
            label=label,
            kind="read_before",
            snapshot={
                "telemetry": telemetry["before"],
                "gripper_qpos": list(before_qpos),
                "finger_separation_m": abs(before_qpos[0] - before_qpos[1]),
            },
        )
        action_count = len(actions)
        for index, (action, sample) in enumerate(
            zip(actions, telemetry["samples"], strict=True)
        ):
            fraction = (index + 1) / action_count
            qpos = [
                start + fraction * (end - start)
                for start, end in zip(before_qpos, after_qpos, strict=True)
            ]
            record(label=label, kind="step", snapshot={"action": action})
            record(
                label=label,
                kind="read_step",
                snapshot={
                    "telemetry": sample,
                    "gripper_qpos": qpos,
                    "finger_separation_m": abs(qpos[0] - qpos[1]),
                },
            )
        record(
            label=label,
            kind="read_completion",
            snapshot={
                "telemetry": telemetry["completion"],
                "gripper_qpos": list(after_qpos),
                "finger_separation_m": abs(after_qpos[0] - after_qpos[1]),
            },
        )
        audit["journal_range"] = {
            "start_event_index": start_index,
            "end_event_index": event_index,
        }
    evidence["journal_terminal_sha256"] = prior_hash
    snapshot = grounding._JournalSnapshot(tuple(events), prior_hash)
    verifier = grounding._JournalVerifier(snapshot=snapshot, _key=key)
    return grounding._ProcessLocalGroundingEvidence(
        raw=evidence,
        journal_verifier=verifier,
    )


def _grounding_evidence(
    *, first_probe_first_step_rad: float | None = None
) -> Mapping[str, object]:
    grounding = importlib.import_module("adaptive.panda_grounding_smoke")
    calibration = _grounding_calibration()
    reset = _reset_qpos()
    probes: list[dict[str, object]] = []
    baseline = _grounding_command_audit(
        reset, reset, calibration=calibration
    )
    rest_baseline = _grounding_command_audit(
        reset, reset, calibration=calibration
    )
    action_count = int(baseline["receipt"]["step_count"]) + int(
        rest_baseline["receipt"]["step_count"]
    )
    for joint_index in range(7):
        target = list(reset)
        target[joint_index] += grounding.PROBE_COMMAND_RAD
        outbound = _grounding_command_audit(
            reset,
            target,
            calibration=calibration,
            first_step_rad=(
                first_probe_first_step_rad if joint_index == 0 else None
            ),
        )
        returned = _grounding_command_audit(
            target, reset, calibration=calibration
        )
        action_count += int(outbound["receipt"]["step_count"])
        action_count += int(returned["receipt"]["step_count"])
        probes.append(
            {
                "joint_index": joint_index,
                "direction": 1,
                "start_qpos": list(reset),
                "target_qpos": target,
                "outbound_qpos": target,
                "return_qpos": list(reset),
                "oracle_start_pose": _grounding_pose_record(reset),
                "oracle_outbound_pose": _grounding_pose_record(target),
                "oracle_return_pose": _grounding_pose_record(reset),
                "outbound": outbound,
                "return": returned,
            }
        )
    close = _grounding_command_audit(
        reset,
        reset,
        calibration=calibration,
        gripper_open=0.0,
        action_count=MIN_GRIPPER_ACTIONS,
    )
    reopen = _grounding_command_audit(
        reset,
        reset,
        calibration=calibration,
        gripper_open=1.0,
        action_count=MIN_GRIPPER_ACTIONS,
    )
    action_count += 2 * MIN_GRIPPER_ACTIONS
    evidence = {
        "schema": "robocasa-inspect-panda-grounding-evidence/v1",
        "task": "OpenToasterOvenDoor",
        "seed": 7,
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
            "robot_model_path": grounding.ROBOT_MODEL_SOURCE_PATH,
            "robot_model_sha256": grounding.ROBOT_MODEL_SHA256,
            "gripper_model_path": grounding.GRIPPER_MODEL_SOURCE_PATH,
            "gripper_model_sha256": grounding.GRIPPER_MODEL_SHA256,
            "dynamic_witness_sha256": grounding.DYNAMIC_WITNESS_SHA256,
        },
        "camera_calibration": calibration,
        "baseline": baseline,
        "rest_baseline": rest_baseline,
        "static": {
            "qpos": list(reset),
            "link0_world_pose": {
                "position_m": [0.0, 0.0, 0.0],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            "oracle_grip_site_world_pose": _grounding_pose_record(reset),
            "public_external_pixels": _grounding_pixels(reset, calibration),
        },
        "probes": probes,
        "gripper": {
            "reset_qpos": [0.02, -0.02],
            "closed_qpos": [0.001, -0.001],
            "reopened_qpos": [0.038, -0.038],
            "close": close,
            "reopen": reopen,
        },
        "action_count": action_count,
        "terminal_action_count": action_count,
        "network_probe": "network namespace denied",
        "simulator_error": None,
        "episode_tmp_empty": True,
    }
    return _bind_test_grounding_journal(evidence)


def _mutated_grounding_evidence() -> Mapping[str, object]:
    grounding = importlib.import_module("adaptive.panda_grounding_smoke")
    original = _grounding_evidence()
    raw = json.loads(json.dumps(dict(original)))
    return grounding._ProcessLocalGroundingEvidence(
        raw=raw,
        journal_verifier=original.journal_verifier,
    )


def test_grounding_evaluator_accepts_exact_seven_joint_gate() -> None:
    grounding = importlib.import_module("adaptive.panda_grounding_smoke")

    passed, checks = grounding.evaluate_grounding_smoke(_grounding_evidence())

    assert passed is True
    assert set(checks) == grounding.GROUNDING_CHECKS
    assert all(checks.values())


def test_grounding_evaluator_accepts_legal_first_step_tolerance() -> None:
    grounding = importlib.import_module("adaptive.panda_grounding_smoke")

    evaluation = grounding._evaluate_grounding(
        _grounding_evidence(first_probe_first_step_rad=0.026)
    )

    assert evaluation.passed is True
    assert evaluation.metrics["max_first_step_rad"] == pytest.approx(0.026)
    assert evaluation.metrics["max_later_step_rad"] <= MAX_JOINT_STEP + 1e-12


def test_grounding_source_journal_is_immutable_keyed_and_not_persisted() -> None:
    evidence = _grounding_evidence()
    verifier = evidence.journal_verifier

    assert isinstance(verifier.snapshot.events, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        verifier.snapshot.terminal_sha256 = "0" * 64
    raw = json.dumps(dict(evidence), sort_keys=True)
    assert "round-three-test-journal-key" not in raw
    assert "snapshot_json" not in raw
    assert "journal_verifier" not in raw


def test_grounding_evaluator_rejects_plain_resealable_mapping() -> None:
    grounding = importlib.import_module("adaptive.panda_grounding_smoke")
    evidence = _grounding_evidence()

    passed, checks = grounding.evaluate_grounding_smoke(
        json.loads(json.dumps(dict(evidence)))
    )

    assert passed is False
    assert checks["complete_telemetry_sampling"] is False


def test_grounding_evaluator_rejects_journal_verified_with_wrong_key() -> None:
    grounding = importlib.import_module("adaptive.panda_grounding_smoke")
    evidence = _grounding_evidence()
    wrong_verifier = grounding._JournalVerifier(
        snapshot=evidence.journal_verifier.snapshot,
        _key=b"different-gate-local-key",
    )
    wrong_evidence = grounding._ProcessLocalGroundingEvidence(
        raw=evidence.raw,
        journal_verifier=wrong_verifier,
    )

    passed, checks = grounding.evaluate_grounding_smoke(wrong_evidence)

    assert passed is False
    assert checks["complete_telemetry_sampling"] is False


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    (
        ("extra_root", "closed_evidence_schema"),
        ("source_hash", "installed_model_sources"),
        ("camera_source", "official_camera_calibration"),
        ("authority", "qwen_only_target_authority"),
        ("authority_count_false", "qwen_only_target_authority"),
        ("authority_count_true", "qwen_only_target_authority"),
        ("oracle_payload", "raw_oracle_process_local"),
        ("rest_torque", "stable_rest_baseline"),
        ("rest_qvel", "stable_rest_baseline"),
        ("rest_action", "stable_rest_baseline"),
        ("rest_first_step_jump", "stable_rest_baseline"),
        ("static_position", "static_fk_position"),
        ("static_rotation", "static_fk_rotation"),
        ("pixel", "external_pixel_agreement"),
        ("skip_joint", "all_seven_joints_probed"),
        ("duplicate_joint", "all_seven_joints_probed"),
        ("unsafe_direction", "joint_limit_clearance"),
        ("oversized_probe", "safe_probe_command"),
        ("subthreshold_motion", "realized_joint_displacement"),
        ("degenerate_vector", "dynamic_translation_nondegenerate"),
        ("wrong_vector_direction", "dynamic_translation_cosine"),
        ("wrong_vector_scale", "dynamic_translation_relative_error"),
        ("bad_return", "return_to_start"),
        ("start_audit_mismatch", "all_seven_joints_probed"),
        ("target_receipt_mismatch", "all_seven_joints_probed"),
        ("outbound_receipt_mismatch", "all_seven_joints_probed"),
        ("outbound_intent_mismatch", "all_seven_joints_probed"),
        ("return_endpoint_mismatch", "all_seven_joints_probed"),
        ("return_receipt_mismatch", "all_seven_joints_probed"),
        ("return_intent_mismatch", "all_seven_joints_probed"),
        ("oracle_absolute_mismatch", "all_seven_joints_probed"),
        ("oracle_return_mismatch", "all_seven_joints_probed"),
        ("probe_continuity_mismatch", "all_seven_joints_probed"),
        ("missing_step_sample", "complete_telemetry_sampling"),
        ("action_chain_mismatch", "complete_telemetry_sampling"),
        ("final_command_mismatch", "complete_telemetry_sampling"),
        ("sample_action_mismatch", "complete_telemetry_sampling"),
        ("completion_action_mismatch", "complete_telemetry_sampling"),
        ("telemetry_before_mismatch", "complete_telemetry_sampling"),
        ("journal_range_mismatch", "complete_telemetry_sampling"),
        ("journal_root_mismatch", "complete_telemetry_sampling"),
        ("journal_range_replay", "complete_telemetry_sampling"),
        ("reverse_telemetry_order", "complete_telemetry_sampling"),
        ("coherent_relabel_reorder", "complete_telemetry_sampling"),
        ("nonfinite_qvel", "finite_telemetry"),
        ("peak_drift", "telemetry_peaks_reconcile"),
        ("receipt_pixel_drift", "telemetry_peaks_reconcile"),
        ("limit_inset", "joint_limit_inset"),
        ("first_overstep", "first_step_tolerance"),
        ("later_overstep", "later_step_bound"),
        ("pause_drift", "tracking_pauses_reconcile"),
        ("advance_while_lagging", "tracking_pauses_reconcile"),
        ("too_many_command_actions", "command_action_cap"),
        ("short_gripper", "gripper_transition_actions"),
        ("base_motion", "base_and_torso_stationary"),
        ("close_direction", "gripper_close_direction"),
        ("reopen_direction", "gripper_reopen_direction"),
        ("close_intent", "gripper_close_direction"),
        ("reopen_intent", "gripper_reopen_direction"),
        ("swap_gripper_audits", "gripper_close_direction"),
        ("closed_top_level_exact", "gripper_close_direction"),
        ("reopened_top_level_exact", "gripper_reopen_direction"),
        ("accounting", "action_accounting_exact"),
        ("episode_budget", "episode_action_budget"),
        ("network", "network_disabled"),
        ("simulator", "simulator_clean"),
    ),
)
def test_grounding_evaluator_mutations_fail_closed(
    mutation: str, failed_check: str
) -> None:
    grounding = importlib.import_module("adaptive.panda_grounding_smoke")
    evidence = _mutated_grounding_evidence()
    raw = evidence.raw
    probe = evidence["probes"][0]
    outbound = probe["outbound"]
    if mutation == "extra_root":
        raw["private_oracle"] = [1.0, 2.0, 3.0]
    elif mutation == "source_hash":
        evidence["source_hashes"]["robot_model_sha256"] = "0" * 64
    elif mutation == "camera_source":
        evidence["camera_calibration"]["source"] = "approximate"
    elif mutation == "authority":
        evidence["authority"]["normal_episode_target_authority"] = "harness"
    elif mutation == "authority_count_false":
        evidence["authority"]["controller_model_payload_count"] = False
    elif mutation == "authority_count_true":
        evidence["authority"]["controller_model_payload_count"] = True
    elif mutation == "oracle_payload":
        evidence["authority"]["raw_oracle_in_model_payload"] = True
    elif mutation == "rest_torque":
        evidence["rest_baseline"]["telemetry"]["samples"][2][
            "state.arm_applied_torque"
        ]["values_nm"][0] += 1.0
    elif mutation == "rest_qvel":
        evidence["rest_baseline"]["telemetry"]["samples"][1][
            "state.arm_joint_velocity"
        ][0] = 0.051
    elif mutation == "rest_action":
        evidence["rest_baseline"]["actions"][1]["commanded_qpos"][0] += 0.001
    elif mutation == "rest_first_step_jump":
        rest_audit = evidence["rest_baseline"]
        for sample in rest_audit["telemetry"]["samples"]:
            sample["state.arm_joint_position"][0] += 0.01
        rest_audit["telemetry"]["completion"]["state.arm_joint_position"][0] += 0.01
        for index, action in enumerate(rest_audit["actions"]):
            action["realized_after"][0] += 0.01
            if index:
                action["realized_before"][0] += 0.01
        rest_audit["receipt"]["realized_arm_qpos"][0] += 0.01
        rest_audit["receipt"]["endpoint_error"] = 0.01
        rest_audit["receipt"]["telemetry_summary"] = summarize_telemetry_samples(
            rest_audit["telemetry"]["before"],
            rest_audit["telemetry"]["samples"],
            rest_audit["telemetry"]["completion"],
        )
    elif mutation == "static_position":
        evidence["static"]["oracle_grip_site_world_pose"]["position_m"][0] += 0.0011
    elif mutation == "static_rotation":
        angle = 0.011
        evidence["static"]["oracle_grip_site_world_pose"]["rotation_matrix"] = [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    elif mutation == "pixel":
        evidence["static"]["public_external_pixels"]["left"]["u_px"] += 5.1
    elif mutation == "skip_joint":
        evidence["probes"].pop()
    elif mutation == "duplicate_joint":
        evidence["probes"][6]["joint_index"] = 5
    elif mutation == "unsafe_direction":
        probe["direction"] = 0
    elif mutation == "oversized_probe":
        probe["target_qpos"][0] += 0.001
    elif mutation == "subthreshold_motion":
        probe["outbound_qpos"][0] = probe["start_qpos"][0] + 0.019
    elif mutation == "degenerate_vector":
        probe["oracle_outbound_pose"] = probe["oracle_start_pose"]
    elif mutation == "wrong_vector_direction":
        start = probe["oracle_start_pose"]["position_m"]
        end = probe["oracle_outbound_pose"]["position_m"]
        probe["oracle_outbound_pose"]["position_m"] = [
            2.0 * first - second
            for first, second in zip(start, end, strict=True)
        ]
    elif mutation == "wrong_vector_scale":
        start = probe["oracle_start_pose"]["position_m"]
        end = probe["oracle_outbound_pose"]["position_m"]
        probe["oracle_outbound_pose"]["position_m"] = [
            first + 2.0 * (second - first)
            for first, second in zip(start, end, strict=True)
        ]
    elif mutation == "bad_return":
        probe["return_qpos"][1] += 0.006
    elif mutation == "start_audit_mismatch":
        probe["outbound"]["telemetry"]["before"]["state.arm_joint_position"][
            1
        ] += 0.001
    elif mutation == "target_receipt_mismatch":
        probe["outbound"]["receipt"]["bounded_endpoint"][1] += 0.001
    elif mutation == "outbound_receipt_mismatch":
        probe["outbound"]["receipt"]["realized_arm_qpos"][1] += 0.001
    elif mutation == "outbound_intent_mismatch":
        probe["outbound"]["receipt"]["gripper_intent"] = 0.0
        for action in probe["outbound"]["actions"]:
            action["gripper_open"] = 0.0
    elif mutation == "return_endpoint_mismatch":
        probe["return"]["receipt"]["bounded_endpoint"][1] += 0.001
    elif mutation == "return_receipt_mismatch":
        probe["return"]["receipt"]["realized_arm_qpos"][1] += 0.001
    elif mutation == "return_intent_mismatch":
        probe["return"]["receipt"]["gripper_intent"] = 0.0
        for action in probe["return"]["actions"]:
            action["gripper_open"] = 0.0
    elif mutation == "oracle_absolute_mismatch":
        for field in (
            "oracle_start_pose",
            "oracle_outbound_pose",
            "oracle_return_pose",
        ):
            probe[field]["position_m"][0] += 0.002
    elif mutation == "oracle_return_mismatch":
        probe["oracle_return_pose"]["position_m"][0] += 0.002
    elif mutation == "probe_continuity_mismatch":
        evidence["probes"][1]["start_qpos"][1] += 0.001
    elif mutation == "missing_step_sample":
        outbound["telemetry"]["samples"].pop()
    elif mutation == "action_chain_mismatch":
        outbound["actions"][1]["realized_before"][1] += 0.001
    elif mutation == "final_command_mismatch":
        outbound["actions"][-1]["commanded_qpos"][1] += 0.001
    elif mutation == "sample_action_mismatch":
        outbound["telemetry"]["samples"][1]["state.arm_joint_position"][
            1
        ] += 0.001
    elif mutation == "completion_action_mismatch":
        outbound["telemetry"]["completion"]["state.arm_joint_position"][1] += 0.001
        outbound["receipt"]["realized_arm_qpos"][1] += 0.001
        outbound["receipt"]["telemetry_summary"] = summarize_telemetry_samples(
            outbound["telemetry"]["before"],
            outbound["telemetry"]["samples"],
            outbound["telemetry"]["completion"],
        )
    elif mutation == "telemetry_before_mismatch":
        outbound["actions"][0]["realized_before"][1] += 0.001
    elif mutation == "journal_range_mismatch":
        outbound["journal_range"]["start_event_index"] += 1
    elif mutation == "journal_root_mismatch":
        raw["journal_terminal_sha256"] = "0" * 64
    elif mutation == "journal_range_replay":
        outbound["journal_range"] = json.loads(
            json.dumps(probe["return"]["journal_range"])
        )
    elif mutation == "reverse_telemetry_order":
        samples = outbound["telemetry"]["samples"]
        for key, nested in (
            ("state.arm_joint_velocity", None),
            ("state.arm_applied_torque", "values_nm"),
            ("state.end_effector_wrench", "force_n"),
            ("state.end_effector_wrench", "torque_nm"),
        ):
            values = [
                json.loads(json.dumps(sample[key] if nested is None else sample[key][nested]))
                for sample in samples
            ]
            for sample, value in zip(samples, reversed(values), strict=True):
                if nested is None:
                    sample[key] = value
                else:
                    sample[key][nested] = value
        outbound["receipt"]["telemetry_summary"] = summarize_telemetry_samples(
            outbound["telemetry"]["before"],
            samples,
            outbound["telemetry"]["completion"],
        )
    elif mutation == "coherent_relabel_reorder":
        close = evidence["gripper"]["close"]
        reopen = evidence["gripper"]["reopen"]
        close["actions"], reopen["actions"] = reopen["actions"], close["actions"]
        close["telemetry"], reopen["telemetry"] = (
            reopen["telemetry"],
            close["telemetry"],
        )
        close["receipt"], reopen["receipt"] = reopen["receipt"], close["receipt"]
    elif mutation == "nonfinite_qvel":
        outbound["telemetry"]["samples"][0]["state.arm_joint_velocity"][0] = float("nan")
    elif mutation == "peak_drift":
        outbound["receipt"]["telemetry_summary"]["arm_joint_position"][
            "peak_abs_delta_from_start_rad"
        ][0] += 0.01
    elif mutation == "receipt_pixel_drift":
        outbound["receipt"]["end_effector_external_pixels_after"]["left"][
            "u_px"
        ] += 1.0
    elif mutation == "limit_inset":
        outbound["receipt"]["minimum_hard_limit_margin"] = -0.001
    elif mutation == "first_overstep":
        outbound["actions"][0]["commanded_qpos"][0] = (
            outbound["actions"][0]["realized_before"][0]
            + MAX_JOINT_STEP
            + 0.0021
        )
    elif mutation == "later_overstep":
        outbound["actions"][1]["commanded_qpos"][0] = (
            outbound["actions"][0]["commanded_qpos"][0]
            + MAX_JOINT_STEP
            + 0.0001
        )
    elif mutation == "pause_drift":
        outbound["receipt"]["tracking_pause_count"] = 1
    elif mutation == "advance_while_lagging":
        lagged = outbound["actions"][0]["realized_after"]
        lagged[0] = outbound["actions"][0]["commanded_qpos"][0] - 0.051
        outbound["actions"][1]["realized_before"] = list(lagged)
        outbound["telemetry"]["samples"][0]["state.arm_joint_position"] = list(
            lagged
        )
        outbound["receipt"]["telemetry_summary"] = summarize_telemetry_samples(
            outbound["telemetry"]["before"],
            outbound["telemetry"]["samples"],
            outbound["telemetry"]["completion"],
        )
    elif mutation == "too_many_command_actions":
        outbound["actions"] = outbound["actions"] * 9
        outbound["receipt"]["step_count"] = len(outbound["actions"])
        outbound["telemetry"]["samples"] = outbound["telemetry"]["samples"] * 9
    elif mutation == "short_gripper":
        evidence["gripper"]["close"]["actions"] = evidence["gripper"]["close"]["actions"][:-1]
        evidence["gripper"]["close"]["telemetry"]["samples"] = evidence["gripper"]["close"]["telemetry"]["samples"][:-1]
        evidence["gripper"]["close"]["receipt"]["step_count"] -= 1
    elif mutation == "base_motion":
        outbound["actions"][0]["base_motion"] = [0.1, 0.0, 0.0]
    elif mutation == "close_direction":
        evidence["gripper"]["closed_qpos"] = [0.038, -0.038]
    elif mutation == "reopen_direction":
        evidence["gripper"]["reopened_qpos"] = [0.001, -0.001]
    elif mutation == "close_intent":
        evidence["gripper"]["close"]["receipt"]["gripper_intent"] = 1.0
        for action in evidence["gripper"]["close"]["actions"]:
            action["gripper_open"] = 1.0
    elif mutation == "reopen_intent":
        evidence["gripper"]["reopen"]["receipt"]["gripper_intent"] = 0.0
        for action in evidence["gripper"]["reopen"]["actions"]:
            action["gripper_open"] = 0.0
    elif mutation == "swap_gripper_audits":
        evidence["gripper"]["close"], evidence["gripper"]["reopen"] = (
            evidence["gripper"]["reopen"],
            evidence["gripper"]["close"],
        )
    elif mutation == "closed_top_level_exact":
        evidence["gripper"]["closed_qpos"] = [0.002, -0.002]
    elif mutation == "reopened_top_level_exact":
        evidence["gripper"]["reopened_qpos"] = [0.037, -0.037]
    elif mutation == "accounting":
        raw["terminal_action_count"] += 1
    elif mutation == "episode_budget":
        raw["action_count"] = 451
        raw["terminal_action_count"] = 451
    elif mutation == "network":
        raw["network_probe"] = "network reachable"
    elif mutation == "simulator":
        raw["simulator_error"] = "close failed"
    else:
        raise AssertionError(mutation)

    passed, checks = grounding.evaluate_grounding_smoke(evidence)

    assert passed is False
    assert checks[failed_check] is False


def _grounding_public_result() -> dict[str, object]:
    grounding = importlib.import_module("adaptive.panda_grounding_smoke")
    result = {
        "schema": "robocasa-inspect-panda-grounding-smoke/v1",
        "task": "OpenToasterOvenDoor",
        "seed": 7,
        "passed": True,
        "checks": {name: True for name in grounding.GROUNDING_CHECKS},
        "metrics": {
            "static_position_error_m": 0.0001,
            "static_rotation_error_rad": 0.001,
            "max_external_pixel_error_px": 1.0,
            "min_joint_limit_clearance_rad": 0.10,
            "max_probe_command_rad": 0.04,
            "min_realized_joint_displacement_rad": 0.03,
            "min_dynamic_translation_cosine": 0.99,
            "max_dynamic_translation_relative_error": 0.10,
            "min_dynamic_vector_norm_m": 0.001,
            "max_return_error_rad": 0.001,
            "max_first_step_rad": 0.026,
            "max_later_step_rad": 0.025,
            "telemetry_sample_count": 132,
            "probe_count": 7,
            "audit_count": 18,
            "nonempty_audit_count": 18,
            "action_count": 96,
            "terminal_action_count": 96,
            "audited_action_count": 96,
            "min_actions_per_command": 4,
            "max_actions_per_command": 16,
            "gripper_transition_audit_count": 2,
            "min_gripper_transition_actions": 16,
            "min_joint_limit_inset_rad": 0.01,
            "max_rest_qpos_span_rad": 0.0001,
            "max_rest_qvel_abs_rad_s": 0.01,
            "max_rest_torque_span_nm": 0.10,
            "max_rest_force_span_n": 0.05,
            "max_rest_wrench_torque_span_nm": 0.01,
            "max_rest_endpoint_error_rad": 0.001,
            "tracking_required_pause_count": 0,
            "tracking_observed_pause_count": 0,
            "tracking_violation_count": 0,
            "tracking_receipt_mismatch_count": 0,
            "sampling_reconciliation_error_count": 0,
            "peak_reconciliation_error_count": 0,
            "nonfinite_telemetry_value_count": 0,
            "max_base_motion_abs": 0.0,
            "max_torso_abs": 0.0,
            "gripper_reset_separation_m": 0.04,
            "gripper_closed_separation_m": 0.002,
            "gripper_reopened_separation_m": 0.076,
            "gripper_close_separation_delta_m": 0.038,
            "gripper_reopen_separation_delta_m": 0.074,
            "controller_model_payload_count": 0,
            "critic_model_payload_count": 0,
            "authority_label_error_count": 0,
            "raw_oracle_model_payload_count": 0,
            "raw_oracle_persisted_count": 0,
            "probe_targets_persisted_count": 0,
            "evidence_schema_error_count": 0,
            "installed_source_mismatch_count": 0,
            "calibration_error_count": 0,
            "network_error_count": 0,
            "simulator_error_count": 0,
            "episode_tmp_file_count": 0,
            "stable_rest_reconciliation_error_count": 0,
            "static_binding_error_count": 0,
            "pixel_binding_error_count": 0,
            "probe_binding_error_count": 0,
            "joint_direction_error_count": 0,
            "probe_command_shape_error_count": 0,
            "realized_direction_error_count": 0,
            "gripper_close_binding_error_count": 0,
            "gripper_reopen_binding_error_count": 0,
            "gripper_close_direction_error_count": 0,
            "gripper_reopen_direction_error_count": 0,
            "command_cap_error_count": 0,
        },
        "provenance": {
            "probe_target_authority": "gate_only_deterministic_probe",
            "normal_episode_target_authority": "qwen_only",
            "raw_oracle_scope": "isolated_process_memory_only",
            "persisted_value_policy": "aggregate_errors_checks_hashes_counts_labels_only",
            "camera_calibration_source": grounding.OFFICIAL_CALIBRATION_SOURCE,
            "robot_model_sha256": grounding.ROBOT_MODEL_SHA256,
            "gripper_model_sha256": grounding.GRIPPER_MODEL_SHA256,
            "dynamic_witness_sha256": grounding.DYNAMIC_WITNESS_SHA256,
            "evidence_schema": grounding.GROUNDING_EVIDENCE_SCHEMA,
            "journal_trust_boundary": "process_local_hmac_sha256_snapshot",
            "journal_terminal_sha256": "c" * 64,
            "network_policy": "network_namespace_denied",
            "simulator_status": "clean",
            "evidence_sha256": "a" * 64,
            "error_sha256": None,
        },
        "artifact_root": (
            "/home/jli/state/qwen-workflow-recovery/"
            "run-100-123/panda-grounding"
        ),
        "release_digest": "abcd1234abcd1234",
        "wall_s": 12.0,
    }
    assert set(result["metrics"]) == grounding.GROUNDING_METRIC_FIELDS
    return result


@pytest.mark.parametrize(
    "mutation",
    (
        "passed_relation",
        "task_type",
        "task_empty",
        "seed_type",
        "seed_negative",
        "wall_type",
        "wall_zero",
        "artifact_relative",
        "artifact_traversal",
        "release_case",
        "release_width",
        "probe_authority",
        "normal_authority",
        "raw_scope",
        "persisted_policy",
        "camera_source",
        "robot_hash",
        "evidence_hash",
        "journal_hash",
        "journal_boundary",
        "network_policy",
        "simulator_status",
        "error_hash_on_pass",
        "fractional_count",
        "negative_count",
        "action_count_mismatch",
        "wall_too_large",
        "static_position_metric",
        "static_rotation_metric",
        "pixel_metric",
        "clearance_metric",
        "probe_command_metric",
        "displacement_metric",
        "cosine_metric",
        "relative_metric",
        "return_metric",
        "first_step_metric",
        "later_step_metric",
        "audit_count",
        "nonempty_audit_count",
        "minimum_command_actions",
        "maximum_command_actions",
        "gripper_audit_count",
        "minimum_gripper_actions",
        "joint_inset_metric",
        "dynamic_vector_norm",
        "rest_qpos",
        "rest_qvel",
        "rest_torque",
        "rest_force",
        "rest_wrench_torque",
        "rest_endpoint",
        "tracking_observed",
        "tracking_violation",
        "tracking_receipt_mismatch",
        "sampling_reconciliation",
        "peak_reconciliation",
        "nonfinite_telemetry",
        "base_motion_metric",
        "torso_metric",
        "close_delta",
        "reopen_delta",
        "gripper_endpoint_relation",
        "negative_gripper_separations",
        "close_direction_error",
        "reopen_direction_error",
        "controller_payload",
        "critic_payload",
        "authority_status",
        "oracle_payload_count",
        "oracle_persisted_count",
        "probe_persisted_count",
        "schema_error_count",
        "source_mismatch_count",
        "calibration_error_count",
        "network_error_count",
        "simulator_error_count",
        "episode_tmp_count",
        "stable_rest_reconciliation",
        "static_binding",
        "pixel_binding",
        "probe_binding",
        "joint_direction",
        "probe_command_shape",
        "realized_direction",
        "gripper_close_binding",
        "gripper_reopen_binding",
        "unrealistic_action_count",
        "audited_action_count",
        "command_cap_status",
        "telemetry_relation",
        "failed_metric_check_consistency",
        "failed_count_check_consistency",
    ),
)
def test_public_grounding_result_rejects_identity_type_and_relation_drift(
    mutation: str,
) -> None:
    grounding = importlib.import_module("adaptive.panda_grounding_smoke")
    result = _grounding_public_result()
    if mutation == "passed_relation":
        result["checks"]["network_disabled"] = False
    elif mutation == "task_type":
        result["task"] = True
    elif mutation == "task_empty":
        result["task"] = ""
    elif mutation == "seed_type":
        result["seed"] = True
    elif mutation == "seed_negative":
        result["seed"] = -1
    elif mutation == "wall_type":
        result["wall_s"] = True
    elif mutation == "wall_zero":
        result["wall_s"] = 0.0
    elif mutation == "artifact_relative":
        result["artifact_root"] = "relative/gate"
    elif mutation == "artifact_traversal":
        result["artifact_root"] = "/remote/../private/gate"
    elif mutation == "release_case":
        result["release_digest"] = "ABCD1234ABCD1234"
    elif mutation == "release_width":
        result["release_digest"] = "abcd1234"
    elif mutation == "probe_authority":
        result["provenance"]["probe_target_authority"] = "controller"
    elif mutation == "normal_authority":
        result["provenance"]["normal_episode_target_authority"] = "harness"
    elif mutation == "raw_scope":
        result["provenance"]["raw_oracle_scope"] = "artifact"
    elif mutation == "persisted_policy":
        result["provenance"]["persisted_value_policy"] = "raw"
    elif mutation == "camera_source":
        result["provenance"]["camera_calibration_source"] = "approximate"
    elif mutation == "robot_hash":
        result["provenance"]["robot_model_sha256"] = "0" * 64
    elif mutation == "evidence_hash":
        result["provenance"]["evidence_sha256"] = "not-a-hash"
    elif mutation == "journal_hash":
        result["provenance"]["journal_terminal_sha256"] = "not-a-hash"
    elif mutation == "journal_boundary":
        result["provenance"]["journal_trust_boundary"] = "post_hoc_reseal"
    elif mutation == "network_policy":
        result["provenance"]["network_policy"] = "reachable"
    elif mutation == "simulator_status":
        result["provenance"]["simulator_status"] = "dirty"
    elif mutation == "error_hash_on_pass":
        result["provenance"]["error_sha256"] = "0" * 64
    elif mutation == "fractional_count":
        result["metrics"]["probe_count"] = 7.5
    elif mutation == "negative_count":
        result["metrics"]["telemetry_sample_count"] = -1
    elif mutation == "action_count_mismatch":
        result["metrics"]["terminal_action_count"] = 87
    elif mutation == "wall_too_large":
        result["wall_s"] = 1_301.0
    elif mutation == "static_position_metric":
        result["metrics"]["static_position_error_m"] = 99.0
    elif mutation == "static_rotation_metric":
        result["metrics"]["static_rotation_error_rad"] = 1.0
    elif mutation == "pixel_metric":
        result["metrics"]["max_external_pixel_error_px"] = 99.0
    elif mutation == "clearance_metric":
        result["metrics"]["min_joint_limit_clearance_rad"] = 0.0
    elif mutation == "probe_command_metric":
        result["metrics"]["max_probe_command_rad"] = 1.0
    elif mutation == "displacement_metric":
        result["metrics"]["min_realized_joint_displacement_rad"] = 0.0
    elif mutation == "cosine_metric":
        result["metrics"]["min_dynamic_translation_cosine"] = -1.0
    elif mutation == "relative_metric":
        result["metrics"]["max_dynamic_translation_relative_error"] = 1.0
    elif mutation == "return_metric":
        result["metrics"]["max_return_error_rad"] = 1.0
    elif mutation == "first_step_metric":
        result["metrics"]["max_first_step_rad"] = 0.0270001
    elif mutation == "later_step_metric":
        result["metrics"]["max_later_step_rad"] = 0.0250001
    elif mutation == "audit_count":
        result["metrics"]["audit_count"] = 17
    elif mutation == "nonempty_audit_count":
        result["metrics"]["nonempty_audit_count"] = 17
    elif mutation == "minimum_command_actions":
        result["metrics"]["min_actions_per_command"] = 0
    elif mutation == "maximum_command_actions":
        result["metrics"]["max_actions_per_command"] = 33
    elif mutation == "gripper_audit_count":
        result["metrics"]["gripper_transition_audit_count"] = 1
    elif mutation == "minimum_gripper_actions":
        result["metrics"]["min_gripper_transition_actions"] = 15
    elif mutation == "joint_inset_metric":
        result["metrics"]["min_joint_limit_inset_rad"] = -0.0001
    elif mutation == "dynamic_vector_norm":
        result["metrics"]["min_dynamic_vector_norm_m"] = 0.0
    elif mutation == "rest_qpos":
        result["metrics"]["max_rest_qpos_span_rad"] = 0.0011
    elif mutation == "rest_qvel":
        result["metrics"]["max_rest_qvel_abs_rad_s"] = 0.051
    elif mutation == "rest_torque":
        result["metrics"]["max_rest_torque_span_nm"] = 0.251
    elif mutation == "rest_force":
        result["metrics"]["max_rest_force_span_n"] = 0.101
    elif mutation == "rest_wrench_torque":
        result["metrics"]["max_rest_wrench_torque_span_nm"] = 0.051
    elif mutation == "rest_endpoint":
        result["metrics"]["max_rest_endpoint_error_rad"] = 0.0021
    elif mutation == "tracking_observed":
        result["metrics"]["tracking_required_pause_count"] = 1
    elif mutation == "tracking_violation":
        result["metrics"]["tracking_violation_count"] = 1
    elif mutation == "tracking_receipt_mismatch":
        result["metrics"]["tracking_receipt_mismatch_count"] = 1
    elif mutation == "sampling_reconciliation":
        result["metrics"]["sampling_reconciliation_error_count"] = 1
    elif mutation == "peak_reconciliation":
        result["metrics"]["peak_reconciliation_error_count"] = 1
    elif mutation == "nonfinite_telemetry":
        result["metrics"]["nonfinite_telemetry_value_count"] = 1
    elif mutation == "base_motion_metric":
        result["metrics"]["max_base_motion_abs"] = 0.001
    elif mutation == "torso_metric":
        result["metrics"]["max_torso_abs"] = 0.001
    elif mutation == "close_delta":
        result["metrics"]["gripper_close_separation_delta_m"] = 0.0
    elif mutation == "reopen_delta":
        result["metrics"]["gripper_reopen_separation_delta_m"] = 0.0
    elif mutation == "gripper_endpoint_relation":
        result["metrics"]["gripper_closed_separation_m"] = 0.039
    elif mutation == "negative_gripper_separations":
        result["metrics"]["gripper_reset_separation_m"] = -0.04
        result["metrics"]["gripper_closed_separation_m"] = -0.078
        result["metrics"]["gripper_reopened_separation_m"] = -0.004
    elif mutation == "close_direction_error":
        result["metrics"]["gripper_close_direction_error_count"] = 1
    elif mutation == "reopen_direction_error":
        result["metrics"]["gripper_reopen_direction_error_count"] = 1
    elif mutation == "controller_payload":
        result["metrics"]["controller_model_payload_count"] = 1
    elif mutation == "critic_payload":
        result["metrics"]["critic_model_payload_count"] = 1
    elif mutation == "authority_status":
        result["metrics"]["authority_label_error_count"] = 1
    elif mutation == "oracle_payload_count":
        result["metrics"]["raw_oracle_model_payload_count"] = 1
    elif mutation == "oracle_persisted_count":
        result["metrics"]["raw_oracle_persisted_count"] = 1
    elif mutation == "probe_persisted_count":
        result["metrics"]["probe_targets_persisted_count"] = 1
    elif mutation == "schema_error_count":
        result["metrics"]["evidence_schema_error_count"] = 1
    elif mutation == "source_mismatch_count":
        result["metrics"]["installed_source_mismatch_count"] = 1
    elif mutation == "calibration_error_count":
        result["metrics"]["calibration_error_count"] = 1
    elif mutation == "network_error_count":
        result["metrics"]["network_error_count"] = 1
    elif mutation == "simulator_error_count":
        result["metrics"]["simulator_error_count"] = 1
    elif mutation == "episode_tmp_count":
        result["metrics"]["episode_tmp_file_count"] = 1
    elif mutation == "stable_rest_reconciliation":
        result["metrics"]["stable_rest_reconciliation_error_count"] = 1
    elif mutation == "static_binding":
        result["metrics"]["static_binding_error_count"] = 1
    elif mutation == "pixel_binding":
        result["metrics"]["pixel_binding_error_count"] = 1
    elif mutation == "probe_binding":
        result["metrics"]["probe_binding_error_count"] = 1
    elif mutation == "joint_direction":
        result["metrics"]["joint_direction_error_count"] = 1
    elif mutation == "probe_command_shape":
        result["metrics"]["probe_command_shape_error_count"] = 1
    elif mutation == "realized_direction":
        result["metrics"]["realized_direction_error_count"] = 1
    elif mutation == "gripper_close_binding":
        result["metrics"]["gripper_close_binding_error_count"] = 1
    elif mutation == "gripper_reopen_binding":
        result["metrics"]["gripper_reopen_binding_error_count"] = 1
    elif mutation == "unrealistic_action_count":
        result["metrics"]["action_count"] = 47
        result["metrics"]["terminal_action_count"] = 47
        result["metrics"]["telemetry_sample_count"] = 83
    elif mutation == "audited_action_count":
        result["metrics"]["audited_action_count"] = 95
    elif mutation == "command_cap_status":
        result["metrics"]["command_cap_error_count"] = 1
    elif mutation == "telemetry_relation":
        result["metrics"]["telemetry_sample_count"] = 99
    elif mutation == "failed_metric_check_consistency":
        result["passed"] = False
        result["checks"]["network_disabled"] = False
        result["metrics"]["static_position_error_m"] = 99.0
    elif mutation == "failed_count_check_consistency":
        result["passed"] = False
        result["checks"]["network_disabled"] = False
        result["metrics"]["terminal_action_count"] = 87
    else:
        raise AssertionError(mutation)

    with pytest.raises((TypeError, ValueError)):
        grounding.validate_public_grounding_result(
            result,
            expected_task="OpenToasterOvenDoor",
            expected_seed=7,
            expected_release_digest="abcd1234abcd1234",
            expected_artifact_root=(
                "/home/jli/state/qwen-workflow-recovery/"
                "run-100-123/panda-grounding"
            ),
        )


def test_public_grounding_result_accepts_and_binds_expected_identity() -> None:
    grounding = importlib.import_module("adaptive.panda_grounding_smoke")
    result = _grounding_public_result()

    validated = grounding.validate_public_grounding_result(
        result,
        expected_task="OpenToasterOvenDoor",
        expected_seed=7,
        expected_release_digest="abcd1234abcd1234",
        expected_artifact_root=(
            "/home/jli/state/qwen-workflow-recovery/"
            "run-100-123/panda-grounding"
        ),
    )

    assert validated == result


def test_public_grounding_result_rejects_negative_physical_separation_even_when_failed() -> None:
    grounding = importlib.import_module("adaptive.panda_grounding_smoke")
    result = _grounding_public_result()
    result["metrics"]["gripper_reset_separation_m"] = -0.04
    result["metrics"]["gripper_closed_separation_m"] = -0.078
    result["metrics"]["gripper_reopened_separation_m"] = -0.004
    result["checks"]["gripper_close_direction"] = False
    result["checks"]["gripper_reopen_direction"] = False
    result["passed"] = False

    with pytest.raises(ValueError, match="physical separation"):
        grounding.validate_public_grounding_result(
            result,
            expected_task="OpenToasterOvenDoor",
            expected_seed=7,
            expected_release_digest="abcd1234abcd1234",
            expected_artifact_root=(
                "/home/jli/state/qwen-workflow-recovery/"
                "run-100-123/panda-grounding"
            ),
        )


def test_public_grounding_result_round_trips_raw_evaluation(tmp_path: Path) -> None:
    grounding = importlib.import_module("adaptive.panda_grounding_smoke")
    evidence = _grounding_evidence()
    evaluation = grounding._evaluate_grounding(evidence)

    result = grounding._public_result(
        task="OpenToasterOvenDoor",
        seed=7,
        evaluation=evaluation,
        artifact_root=tmp_path,
        wall_s=1.0,
        journal_terminal_sha256=evidence["journal_terminal_sha256"],
        evidence_sha256="a" * 64,
        error_sha256=None,
    )

    assert result["passed"] is True
    assert result["checks"] == grounding._derive_public_grounding_checks(
        result["metrics"], result["provenance"]
    )
    serialized = json.dumps(result, sort_keys=True)
    assert "snapshot_json" not in serialized
    assert "journal_range" not in serialized
    assert "journal_events" not in serialized
    assert "hmac_key" not in serialized


@pytest.mark.parametrize(
    ("mutation", "failed_check", "preserved_check"),
    (
        ("receipt_step_count", "complete_telemetry_sampling", None),
        ("receipt_maximum_step", "telemetry_peaks_reconcile", None),
        (
            "external_action_totals",
            "action_accounting_exact",
            "complete_telemetry_sampling",
        ),
        (
            "missing_probe_audit",
            "all_seven_joints_probed",
            "episode_action_budget",
        ),
    ),
)
def test_public_grounding_result_round_trips_raw_receipt_accounting_failures(
    mutation: str,
    failed_check: str,
    preserved_check: str | None,
    tmp_path: Path,
) -> None:
    grounding = importlib.import_module("adaptive.panda_grounding_smoke")
    evidence = _mutated_grounding_evidence()
    outbound = evidence["probes"][0]["outbound"]
    if mutation == "receipt_step_count":
        outbound["receipt"]["step_count"] += 1
    elif mutation == "receipt_maximum_step":
        outbound["receipt"]["maximum_commanded_step"] += 0.001
    elif mutation == "external_action_totals":
        evidence.raw["action_count"] += 1
        evidence.raw["terminal_action_count"] += 1
    elif mutation == "missing_probe_audit":
        evidence["probes"].pop()
    else:
        raise AssertionError(mutation)
    evaluation = grounding._evaluate_grounding(evidence)
    assert evaluation.checks[failed_check] is False
    if preserved_check is not None:
        assert evaluation.checks[preserved_check] is True

    result = grounding._public_result(
        task="OpenToasterOvenDoor",
        seed=7,
        evaluation=evaluation,
        artifact_root=tmp_path,
        wall_s=1.0,
        journal_terminal_sha256=evidence["journal_terminal_sha256"],
        evidence_sha256="a" * 64,
        error_sha256=None,
    )

    assert result["checks"] == evaluation.checks
    assert result["checks"][failed_check] is False
    if preserved_check is not None:
        assert result["checks"][preserved_check] is True


@pytest.mark.parametrize(
    ("motion", "failed_check"),
    (
        ("asymmetric_close", "gripper_close_direction"),
        ("reversed_reopen", "gripper_reopen_direction"),
    ),
)
def test_public_grounding_result_round_trips_source_consistent_gripper_direction_failure(
    motion: str,
    failed_check: str,
    tmp_path: Path,
) -> None:
    grounding = importlib.import_module("adaptive.panda_grounding_smoke")
    raw = json.loads(json.dumps(dict(_grounding_evidence())))
    if motion == "asymmetric_close":
        raw["gripper"]["closed_qpos"] = [0.03, 0.02]
    elif motion == "reversed_reopen":
        raw["gripper"]["reopened_qpos"] = [-0.03, 0.03]
    else:
        raise AssertionError(motion)
    evidence = _bind_test_grounding_journal(raw)
    evaluation = grounding._evaluate_grounding(evidence)
    assert evaluation.checks[failed_check] is False
    assert evaluation.metrics[
        "gripper_close_binding_error_count"
        if motion == "asymmetric_close"
        else "gripper_reopen_binding_error_count"
    ] == 0

    result = grounding._public_result(
        task="OpenToasterOvenDoor",
        seed=7,
        evaluation=evaluation,
        artifact_root=tmp_path,
        wall_s=1.0,
        journal_terminal_sha256=evidence["journal_terminal_sha256"],
        evidence_sha256="a" * 64,
        error_sha256=None,
    )

    assert result["checks"] == evaluation.checks
    assert result["checks"][failed_check] is False


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    (
        ("authority_label", "qwen_only_target_authority"),
        ("receipt_command_cap", "command_action_cap"),
    ),
)
def test_public_grounding_result_round_trips_adjacent_status_failures(
    mutation: str,
    failed_check: str,
    tmp_path: Path,
) -> None:
    grounding = importlib.import_module("adaptive.panda_grounding_smoke")
    evidence = _mutated_grounding_evidence()
    if mutation == "authority_label":
        evidence["authority"]["normal_episode_target_authority"] = "harness"
    elif mutation == "receipt_command_cap":
        evidence["probes"][0]["outbound"]["receipt"]["step_count"] = 33
    else:
        raise AssertionError(mutation)
    evaluation = grounding._evaluate_grounding(evidence)
    assert evaluation.checks[failed_check] is False

    result = grounding._public_result(
        task="OpenToasterOvenDoor",
        seed=7,
        evaluation=evaluation,
        artifact_root=tmp_path,
        wall_s=1.0,
        journal_terminal_sha256=evidence["journal_terminal_sha256"],
        evidence_sha256="a" * 64,
        error_sha256=None,
    )

    assert result["checks"] == evaluation.checks
    assert result["checks"][failed_check] is False


def test_grounding_security_command_matches_joint_safety_child(
    tmp_path: Path,
) -> None:
    safety = importlib.import_module("adaptive.joint_safety_smoke")
    run = tmp_path / "panda-grounding"
    run.mkdir()
    command = safety._isolated_child_command(
        module="adaptive.panda_grounding_smoke",
        task="OpenToasterOvenDoor",
        seed=7,
        run=run,
        child_run=run / "gate",
        extra_args=("--isolated-child",),
    )

    assert command[:5] == ["sudo", "-n", "/usr/bin/unshare", "-n", "--"]
    assert "--reuid=1001" in command
    assert "--regid=1001" in command
    assert "--groups=1001,44,992" in command
    assert "--inh-caps=-all" in command
    assert "--ambient-caps=-all" in command
    assert "--bounding-set=-all" in command
    assert "/usr/bin/env" in command
    assert "-i" in command
    assert "-m" in command
    assert command[command.index("-m") + 1] == "adaptive.panda_grounding_smoke"
    assert command[-1] == "--isolated-child"
    assert not any(item.startswith("QWEN_") for item in command)


def test_candidate_panda_grounding_cli_uses_digest_release_and_one_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    candidate = importlib.import_module("candidate")
    result = _grounding_public_result()
    calls: list[list[str]] = []

    monkeypatch.setattr(
        candidate,
        "deploy_remote",
        lambda: "/release/abcd1234abcd1234",
    )
    monkeypatch.setattr(candidate.time, "time", lambda: 100)
    monkeypatch.setattr(candidate.os, "getpid", lambda: 123)

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps(result), stderr="")

    monkeypatch.setattr(candidate, "_run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["candidate.py", "--panda-grounding-smoke", "OpenToasterOvenDoor"],
    )

    assert candidate.main() == 0
    assert len(calls) == 1
    assert (
        "PYTHONPATH=/release/abcd1234abcd1234:"
        "/home/jli/work/robocasa-inspect-official"
    ) in calls[0]
    assert calls[0][calls[0].index("-m") + 1] == "adaptive.panda_grounding_smoke"
    assert calls[0][calls[0].index("--task") + 1] == "OpenToasterOvenDoor"
    assert not any(item.startswith("QWEN_") for item in calls[0])
    assert capsys.readouterr().out == json.dumps(
        result, sort_keys=True, separators=(",", ":")
    ) + "\n"


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (
        ("task", "CloseDoor"),
        ("seed", 8),
        ("release_digest", "bbbb1234bbbb1234"),
        (
            "artifact_root",
            "/home/jli/state/qwen-workflow-recovery/"
            "run-100-124/panda-grounding",
        ),
    ),
)
def test_candidate_panda_grounding_cli_rejects_other_valid_identity(
    field: str,
    wrong_value: object,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate = importlib.import_module("candidate")
    result = _grounding_public_result()
    result[field] = wrong_value
    monkeypatch.setattr(
        candidate,
        "deploy_remote",
        lambda: "/release/abcd1234abcd1234",
    )
    monkeypatch.setattr(candidate.time, "time", lambda: 100)
    monkeypatch.setattr(candidate.os, "getpid", lambda: 123)
    monkeypatch.setattr(
        candidate,
        "_run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=json.dumps(result), stderr=""
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["candidate.py", "--panda-grounding-smoke", "OpenToasterOvenDoor"],
    )

    with pytest.raises((TypeError, ValueError)):
        candidate.main()
    assert capsys.readouterr().out == ""


def test_candidate_panda_grounding_cli_rejects_wrong_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = importlib.import_module("candidate")
    monkeypatch.setattr(candidate, "deploy_remote", lambda: "/release/digest")
    monkeypatch.setattr(
        candidate,
        "_run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout='{"schema":"wrong"}', stderr=""
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["candidate.py", "--panda-grounding-smoke", "OpenToasterOvenDoor"],
    )

    with pytest.raises(RuntimeError, match="grounding smoke returned the wrong schema"):
        candidate.main()


def test_candidate_proposal_smoke_selects_protocol_in_remote_invocation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate = importlib.import_module("candidate")
    calls: list[list[str]] = []
    result = {
        "schema": "robocasa-qwen10-run/v1",
        "release_digest": "digest",
    }

    monkeypatch.setattr(candidate, "deploy_remote", lambda: "/release/digest")

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps(result), stderr="")

    monkeypatch.setattr(candidate, "_run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "candidate.py",
            "--smoke-task",
            "OpenToasterOvenDoor",
            "--smoke-protocol",
            "proposal",
            "--smoke-action-budget",
            "900",
        ],
    )

    assert candidate.main() == 0
    assert len(calls) == 1
    assert calls[0][calls[0].index("--smoke-protocol") + 1] == "proposal"
    assert calls[0][calls[0].index("--smoke-action-budget") + 1] == "900"
    assert calls[0][calls[0].index("--release-digest") + 1] == "digest"
    assert calls[0][calls[0].index("--critic-prompt") + 1].endswith(
        "/prompts/proposal_audit_critic.txt"
    )
    assert capsys.readouterr().out == json.dumps(
        result, sort_keys=True, separators=(",", ":")
    ) + "\n"


def test_remote_driver_smoke_forwards_development_action_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from adaptive import remote_driver

    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "seed": 7,
                "tasks": [
                    {
                        "task": "OpenToasterOvenDoor",
                        "split": "pretrain",
                        "family": "articulated",
                    }
                ],
            }
        )
    )
    critic = tmp_path / "critic.txt"
    critic.write_text("critic")
    recorded: dict[str, object] = {}

    monkeypatch.setattr(remote_driver, "_validate_baseline_path", lambda _path: baseline)
    monkeypatch.setattr(remote_driver, "_validate_release_digest", lambda _value: "digest")
    monkeypatch.setattr(remote_driver, "_validate_critic_prompt", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(remote_driver, "load_joint_system_prompt", lambda *_args, **_kwargs: "prompt")
    monkeypatch.setattr(remote_driver, "joint_system_prompt_parts", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(remote_driver, "_smoke_pass", lambda *_args, **_kwargs: (False, {}))

    def fake_run_one(**kwargs: object) -> tuple[dict[str, object], CriticContext]:
        recorded.update(kwargs)
        return (
            {
                "status": "policy_failed_proposal_audit",
                "success": False,
                "receipts": [],
                "video": None,
                "video_sha256": None,
                "wall_s": 1.0,
                "qwen_direct_command_count": 0,
            },
            _context(max_decisions=40),
        )

    monkeypatch.setattr(remote_driver, "run_one", fake_run_one)
    result = remote_driver.execute(
        SimpleNamespace(
            baseline=baseline,
            critic_prompt=critic,
            output_root=tmp_path / "out",
            release_digest="digest",
            smoke_task="OpenToasterOvenDoor",
            smoke_protocol="proposal",
            smoke_decisions=40,
            smoke_action_budget=900,
            prompt_variant="rig",
        )
    )

    assert recorded["action_budget"] == 900
    assert result["smoke_action_budget"] == 900


def test_candidate_legacy_smoke_selects_legacy_critic_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = importlib.import_module("candidate")
    calls: list[list[str]] = []
    monkeypatch.setattr(candidate, "deploy_remote", lambda: "/release/digest")

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(
            stdout=(
                '{"schema":"robocasa-qwen10-run/v1",'
                '"release_digest":"digest"}'
            ),
            stderr="",
        )

    monkeypatch.setattr(candidate, "_run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "candidate.py",
            "--smoke-task",
            "OpenToasterOvenDoor",
            "--smoke-protocol",
            "legacy",
        ],
    )

    assert candidate.main() == 0
    assert calls[0][calls[0].index("--smoke-protocol") + 1] == "legacy"
    assert calls[0][calls[0].index("--critic-prompt") + 1].endswith(
        "/prompts/advisory_critic.txt"
    )


def test_controller_protocol_wrapper_selection_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive import remote_driver

    legacy = type("LegacyWrapper", (), {})
    proposal = type("ProposalWrapper", (), {})
    original = type("OriginalClient", (), {})
    context = _context()
    monkeypatch.setattr(
        remote_driver,
        "_make_client_class",
        lambda _original, _context: legacy,
    )
    monkeypatch.setattr(
        remote_driver,
        "_make_proposal_audit_client_class",
        lambda _original, _context: proposal,
    )

    assert remote_driver._client_class_for_protocol(
        original, context, protocol="legacy"
    ) is legacy
    assert remote_driver._client_class_for_protocol(
        original, context, protocol="proposal"
    ) is proposal
    with pytest.raises(ValueError, match="protocol"):
        remote_driver._client_class_for_protocol(
            original, context, protocol="episode-owned"
        )


def test_legacy_wrapper_strips_proposal_only_execution_context() -> None:
    context = _context()
    wrapped = _make_client_class(_FakeClient, context)()
    command = _controller_command("obs-0")
    _FakeClient.controller_outputs = [command]

    response = wrapped.complete(
        observation_id="obs-0",
        system_prompt="legacy",
        instruction=_instruction(),
        public_state=_state(),
        images=_images(),
        response_schema=CONTROLLER_RESPONSE_SCHEMA,
        max_tokens=CONTROLLER_MAX_TOKENS,
        proposal_audit_context={"must_not": "reach legacy controller"},
    )

    assert response.command is command
    assert "proposal_audit_context" not in _FakeClient.calls[-1][1]


def test_critic_prompt_must_match_the_selected_protocol() -> None:
    from adaptive import remote_driver

    root = Path(__file__).resolve().parents[1]
    advisory = (root / "prompts" / "advisory_critic.txt").read_text()
    proposal = (root / "prompts" / "proposal_audit_critic.txt").read_text()

    remote_driver._validate_critic_prompt(advisory, protocol="legacy")
    remote_driver._validate_critic_prompt(proposal, protocol="proposal")
    with pytest.raises(ValueError, match="critic prompt"):
        remote_driver._validate_critic_prompt(advisory, protocol="proposal")


def test_cached_smoke_budget_requires_exact_release_and_prompt_hashes(
    tmp_path: Path,
) -> None:
    from adaptive import critic_protocol as protocol
    from adaptive import remote_driver

    path = tmp_path / "latest-smoke.json"
    release_digest = "abcd1234abcd1234"
    system_prompt_sha256 = hashlib.sha256(b"controller prompt").hexdigest()
    critic_prompt_sha256 = hashlib.sha256(b"critic prompt").hexdigest()
    record = {
        "schema": "robocasa-qwen-smoke-timing/v1",
        "critic_config_sha256": CRITIC_CONFIG_SHA256,
        "proposal_audit_config_sha256": protocol.PROPOSAL_AUDIT_CONFIG_SHA256,
        "release_digest": release_digest,
        "system_prompt_sha256": system_prompt_sha256,
        "critic_prompt_sha256": critic_prompt_sha256,
        "smoke_protocol": "proposal",
        "wall_s": 12.5,
        "direct_command_count": 3,
        "smoke_pass": True,
    }
    path.write_text(json.dumps(record))

    kwargs = {
        "release_digest": release_digest,
        "system_prompt_sha256": system_prompt_sha256,
        "critic_prompt_sha256": critic_prompt_sha256,
    }
    assert remote_driver._proposal_smoke_budget_inputs(path, **kwargs) == (12.5, 3)
    record["proposal_audit_config_sha256"] = "f" * 64
    path.write_text(json.dumps(record))
    assert remote_driver._proposal_smoke_budget_inputs(path, **kwargs) == (0.0, 0)
    record["proposal_audit_config_sha256"] = protocol.PROPOSAL_AUDIT_CONFIG_SHA256
    for field in ("release_digest", "system_prompt_sha256", "critic_prompt_sha256"):
        original = record[field]
        record[field] = "0" * len(str(original))
        path.write_text(json.dumps(record))
        assert remote_driver._proposal_smoke_budget_inputs(path, **kwargs) == (0.0, 0)
        record[field] = original


def test_remote_driver_measures_the_installed_release_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive import remote_driver

    candidate = importlib.import_module("candidate")
    assert remote_driver._installed_release_digest() == candidate._digest()
    measured = "abcd1234abcd1234"
    monkeypatch.setattr(remote_driver, "_installed_release_digest", lambda: measured)

    assert remote_driver._validate_release_digest(measured) == measured
    with pytest.raises(ValueError, match="installed release digest"):
        remote_driver._validate_release_digest("ffff1234ffff1234")


def test_remote_driver_requires_the_measured_release_baseline(tmp_path: Path) -> None:
    from adaptive import remote_driver

    installed = Path(remote_driver.__file__).resolve().parents[1]
    installed_baseline = installed / "benchmark" / "baseline.json"
    assert remote_driver._validate_baseline_path(installed_baseline) == installed_baseline

    copy = tmp_path / "baseline.json"
    copy.write_bytes(installed_baseline.read_bytes())
    with pytest.raises(ValueError, match="installed release baseline"):
        remote_driver._validate_baseline_path(copy)

    link = tmp_path / "baseline-link.json"
    link.symlink_to(installed_baseline)
    with pytest.raises(ValueError, match="installed release baseline"):
        remote_driver._validate_baseline_path(link)


def test_remote_driver_checks_baseline_path_before_measuring_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from adaptive import remote_driver

    def reject_baseline(_path: Path) -> Path:
        raise ValueError("baseline guard ran first")

    monkeypatch.setattr(remote_driver, "_validate_baseline_path", reject_baseline)
    monkeypatch.setattr(
        remote_driver,
        "_validate_release_digest",
        lambda _claimed: pytest.fail("release bytes read before baseline guard"),
    )
    args = SimpleNamespace(
        baseline=tmp_path / "baseline.json",
        release_digest="abcd1234abcd1234",
    )

    with pytest.raises(ValueError, match="baseline guard ran first"):
        remote_driver.execute(args)


@pytest.mark.parametrize(
    "argv",
    (
        ["candidate.py", "--smoke-task", "OpenToasterOvenDoor"],
        ["candidate.py", "--smoke-protocol", "proposal"],
    ),
)
def test_candidate_smoke_requires_task_and_protocol_together_before_deploy(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = importlib.import_module("candidate")
    monkeypatch.setattr(
        candidate,
        "deploy_remote",
        lambda: pytest.fail("invalid smoke invocation reached deployment"),
    )
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit):
        candidate.main()


def test_remote_driver_smoke_cli_requires_explicit_protocol_before_execute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from adaptive import remote_driver

    monkeypatch.setattr(
        remote_driver,
        "execute",
        lambda _args: pytest.fail("invalid smoke invocation reached execution"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remote_driver.py",
            "--baseline",
            str(tmp_path / "baseline.json"),
            "--critic-prompt",
            str(tmp_path / "critic.txt"),
            "--output-root",
            str(tmp_path / "out"),
            "--smoke-task",
            "OpenToasterOvenDoor",
        ],
    )

    with pytest.raises(SystemExit):
        remote_driver.main()


def test_failed_grounding_public_result_sanitizes_nonfinite_aggregate(
    tmp_path: Path,
) -> None:
    grounding = importlib.import_module("adaptive.panda_grounding_smoke")
    evaluation = grounding._Evaluation(
        False,
        {name: False for name in grounding.GROUNDING_CHECKS},
        {
            name: float("inf")
            if name == "max_dynamic_translation_relative_error"
            else None
            for name in grounding.GROUNDING_METRIC_FIELDS
        },
    )

    result = grounding._public_result(
        task="OpenToasterOvenDoor",
        seed=7,
        evaluation=evaluation,
        artifact_root=tmp_path,
        wall_s=1.0,
        journal_terminal_sha256=None,
        evidence_sha256=None,
        error_sha256="0" * 64,
    )

    assert result["metrics"]["max_dynamic_translation_relative_error"] is None
    encoded = json.dumps(result, allow_nan=False)
    assert "target_qpos" not in encoded
    assert "oracle_start_pose" not in encoded


def test_grounding_failure_diagnostic_is_stage_and_class_only() -> None:
    grounding = importlib.import_module("adaptive.panda_grounding_smoke")
    error = RuntimeError("private target [1.2, 3.4, 5.6]")

    label = grounding._safe_failure_label("baseline", error)

    assert label == "grounding-stage=baseline error=RuntimeError"
    assert "private" not in label
    assert "1.2" not in label


def test_grounding_failure_trace_contains_only_function_names() -> None:
    grounding = importlib.import_module("adaptive.panda_grounding_smoke")

    def raise_private() -> None:
        raise RuntimeError("private target [1.2, 3.4, 5.6]")

    try:
        raise_private()
    except RuntimeError as error:
        label = grounding._safe_failure_trace(error)

    assert "raise_private" in label
    assert "private target" not in label
    assert "1.2" not in label
    assert not any(character.isdigit() for character in label)


def test_compose_world_pose_accepts_official_numpy_shaped_iterables() -> None:
    class ArrayLike:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        def __iter__(self):
            return iter(self._values)

    pose = compose_world_pose(
        ArrayLike([0.0, 0.0, 0.0]),
        ArrayLike([0.0, 0.0, 0.0, 1.0]),
        ArrayLike([0.2, 0.1, -2.0]),
        ArrayLike([0.0, 0.0, 0.0, 1.0]),
    )

    assert pose.position_m == pytest.approx((0.2, 0.1, -2.0))
    assert pose.rotation_matrix == (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


def test_grounding_evaluator_accepts_official_video_camera_identity() -> None:
    grounding = importlib.import_module("adaptive.panda_grounding_smoke")
    evidence = _grounding_evidence()
    for camera in ("left", "right"):
        evidence["camera_calibration"][camera]["camera_name"] = (
            f"video.robot0_agentview_{camera}"
        )

    passed, checks = grounding.evaluate_grounding_smoke(evidence)

    assert passed is True
    assert checks["official_camera_calibration"] is True


def test_grounding_gate_roots_pure_fk_at_namespaced_panda_link0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grounding = importlib.import_module("adaptive.panda_grounding_smoke")
    requested_bodies: list[str] = []

    class Array:
        def __init__(self, values: object) -> None:
            self.values = values
            self.shape = (3, 3) if isinstance(values[0], list) else (3,)

        def reshape(self, *_shape: int) -> Array:
            return self

        def tolist(self) -> object:
            return self.values

    monkeypatch.setitem(
        sys.modules,
        "numpy",
        SimpleNamespace(
            float64=float,
            asarray=lambda values, dtype: Array(values),
            isfinite=lambda _values: SimpleNamespace(all=lambda: True),
        ),
    )

    class Data:
        def get_body_xpos(self, name: str) -> list[float]:
            requested_bodies.append(name)
            return [0.0, 0.0, 0.0]

        def get_body_xmat(self, name: str) -> list[float]:
            requested_bodies.append(name)
            return [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]

    robot_model = SimpleNamespace(
        root_body="robot0_base",
        correct_naming=lambda name: f"robot0_{name}",
    )
    environment = SimpleNamespace(
        unwrapped=SimpleNamespace(
            robots=[SimpleNamespace(robot_model=robot_model)],
            sim=SimpleNamespace(data=Data()),
        )
    )

    grounding._raw_pose(environment, site=False)

    assert requested_bodies == ["robot0_link0", "robot0_link0"]
def _cartesian_identity_public_state():
    return {
        "state.arm_joint_position": [
            -0.01,
            -0.16,
            -0.02,
            -0.406,
            -0.014,
            0.323,
            0.218,
        ],
        "state.arm_translation_jacobian": [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        "state.arm_rotation_jacobian": [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ],
    }


def _image_servo_public_state(*, base_yaw_quarter_turn: bool = False):
    state = _cartesian_identity_public_state()
    state.update({
        "state.base_position": [0.0, 0.0, 0.0],
        "state.base_rotation": (
            [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]
            if base_yaw_quarter_turn
            else [0.0, 0.0, 0.0, 1.0]
        ),
        "state.end_effector_position_relative": [0.0, 0.0, -1.0],
        "state.end_effector_external_pixels": {
            "left": {
                "u_px": 50.0,
                "v_px": 50.0,
                "visible": True,
                "depth_valid": True,
            },
            "right": {
                "u_px": 50.0,
                "v_px": 50.0,
                "visible": True,
                "depth_valid": True,
            },
        },
    })
    return state


def _image_servo_calibration():
    camera = {
        "image_width_px": 100,
        "image_height_px": 100,
        "fx_px": 100.0,
        "fy_px": 100.0,
        "cx_px": 50.0,
        "cy_px": 50.0,
        "camera_position_world_m": [0.0, 0.0, 0.0],
        "camera_xmat_world": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "projection": "mujoco-camera-x-right-y-up-minus-z-forward",
    }
    return {"left": camera, "right": dict(camera)}


def _image_servo_payload():
    return {
        "kind": "image_servo",
        "observation_id": "obs",
        "camera": "left",
        "target_pixel": [60.0, 50.0],
        "target_role": "source_object",
        "depth_delta_m": 0.0,
        "step_m": 0.02,
        "gripper": "open",
        "note": "milestone=approach; source center in left RGB",
    }


def test_image_servo_maps_target_pixel_to_bounded_base_translation() -> None:
    from adaptive.image_servo import decode_image_servo, resolve_image_servo

    command = decode_image_servo(_image_servo_payload(), observation_id="obs")
    resolution = resolve_image_servo(
        command,
        _image_servo_public_state(),
        _image_servo_calibration(),
        current_gripper=1.0,
    )

    assert resolution.current_pixel == pytest.approx((50.0, 50.0))
    assert resolution.current_depth_m == pytest.approx(1.0)
    assert resolution.target_depth_m == pytest.approx(1.0)
    assert resolution.translation_m == pytest.approx((0.08, 0.0, 0.0))
    assert resolution.joint_endpoint[0] > _cartesian_identity_public_state()[
        "state.arm_joint_position"
    ][0]


def test_articulation_image_servo_is_one_qwen_authored_increment() -> None:
    from adaptive.image_servo import decode_image_servo, resolve_image_servo

    payload = {
        **_image_servo_payload(),
        "target_role": "articulation_motion",
        "step_m": 0.005,
        "gripper": "hold",
        "note": "milestone=actuate; preserve sealed contact while moving handle",
    }
    command = decode_image_servo(payload, observation_id="obs")
    state = _image_servo_public_state()
    state["state.arm_joint_position"][0] = -1.0
    state["state.arm_translation_jacobian"][0][0] = 0.1

    resolution = resolve_image_servo(
        command,
        state,
        _image_servo_calibration(),
        current_gripper=0.0,
    )

    assert math.dist(resolution.translation_m, (0.0, 0.0, 0.0)) == pytest.approx(
        payload["step_m"]
    )
    assert math.dist(resolution.predicted_delta[:3], (0.0, 0.0, 0.0)) <= (
        payload["step_m"] + 1e-12
    )


def test_articulation_image_servo_receipt_reuses_bounded_resolution() -> None:
    from adaptive.joint_runner import (
        finalize_image_servo_receipt,
        prepare_joint_mailbox,
    )

    payload = {
        **_image_servo_payload(),
        "target_role": "articulation_motion",
        "step_m": 0.005,
        "gripper": "hold",
        "note": "milestone=actuate; preserve sealed contact while moving handle",
    }
    state, _unused_evidence, _unused_calibration = (
        _stationary_receipt_boundary_inputs()
    )
    state.update(_image_servo_public_state())
    state["state.arm_joint_position"][0] = -1.0
    state["state.arm_translation_jacobian"][0][0] = 0.1
    calibration = _image_servo_calibration()
    mailbox, command = prepare_joint_mailbox(
        payload,
        source="controller",
        observation_id="obs",
        current_qpos=state["state.arm_joint_position"],
        current_gripper=0.0,
        public_state=state,
        camera_calibration=calibration,
        remaining_actions=160,
        sequence=0,
    )
    telemetry = _telemetry_sample(
        qpos=state["state.arm_joint_position"],
        qvel=[0.0] * 7,
        torque=[0.0] * 7,
        force=[0.0] * 3,
        wrench_torque=[0.0] * 3,
    )
    execution = {
        "kind": "move_joints",
        "accepted": True,
        "bounded_endpoint": mailbox["endpoint"],
        "gripper_intent": 0.0,
        "step_count": 4,
        "maximum_commanded_step": 0.01,
        "realized_arm_qpos": state["state.arm_joint_position"],
        "endpoint_error": max(
            abs(target - actual)
            for target, actual in zip(
                mailbox["endpoint"],
                state["state.arm_joint_position"],
                strict=True,
            )
        ),
        "minimum_hard_limit_margin": 0.5,
        "tracking_pause_count": 0,
        "telemetry_summary": summarize_telemetry_samples(
            telemetry, [telemetry], telemetry
        ),
        "end_effector_external_pixels_before": state[
            "state.end_effector_external_pixels"
        ],
        "end_effector_external_pixels_after": state[
            "state.end_effector_external_pixels"
        ],
    }

    receipt = finalize_image_servo_receipt(
        command,
        mailbox,
        execution,
        before_state=state,
        after_state=state,
        before_calibration=calibration,
        after_calibration=calibration,
    )

    assert receipt["derived_joint_endpoint"] == mailbox["endpoint"]
    assert math.dist(receipt["predicted_cartesian_delta"][:3], [0.0] * 3) <= (
        payload["step_m"] + 1e-12
    )


def test_image_servo_accepts_public_float32_projection_roundoff() -> None:
    from adaptive.image_servo import decode_image_servo, resolve_image_servo

    state = _image_servo_public_state()
    state["state.end_effector_external_pixels"]["left"].update(
        {"u_px": 50.00004, "v_px": 49.99996}
    )
    command = decode_image_servo(_image_servo_payload(), observation_id="obs")

    resolution = resolve_image_servo(
        command,
        state,
        _image_servo_calibration(),
        current_gripper=1.0,
    )

    assert resolution.current_pixel == pytest.approx((50.0, 50.0))


def test_image_servo_converts_world_direction_into_rotated_base_frame() -> None:
    from adaptive.image_servo import decode_image_servo, resolve_image_servo

    command = decode_image_servo(_image_servo_payload(), observation_id="obs")
    resolution = resolve_image_servo(
        command,
        _image_servo_public_state(base_yaw_quarter_turn=True),
        _image_servo_calibration(),
        current_gripper=1.0,
    )

    assert resolution.translation_m == pytest.approx((0.0, -0.08, 0.0), abs=1e-12)


def test_image_servo_mailbox_preserves_qwen_request_and_uses_joint_child() -> None:
    from adaptive.image_servo import ImageServoCommand
    from adaptive.joint_runner import (
        CONTROLLER_RESPONSE_SCHEMA,
        IMAGE_SERVO_RESPONSE_SCHEMA,
        _command_dict,
        prepare_joint_mailbox,
    )

    state = _image_servo_public_state()
    mailbox, command = prepare_joint_mailbox(
        _image_servo_payload(),
        source="controller",
        observation_id="obs",
        current_qpos=state["state.arm_joint_position"],
        current_gripper=1.0,
        public_state=state,
        camera_calibration=_image_servo_calibration(),
        remaining_actions=160,
        sequence=0,
    )

    assert isinstance(command, ImageServoCommand)
    assert IMAGE_SERVO_RESPONSE_SCHEMA in CONTROLLER_RESPONSE_SCHEMA["oneOf"]
    assert IMAGE_SERVO_RESPONSE_SCHEMA["additionalProperties"] is False
    assert _command_dict(command) == _image_servo_payload()
    assert mailbox["kind"] == "move_joints"
    assert mailbox["explicit_mask"] == [True] * 7
    assert mailbox["endpoint"] != state["state.arm_joint_position"]


def test_image_servo_receipt_binds_pixel_request_to_recomputed_resolution() -> None:
    from adaptive.joint_runner import (
        PUBLIC_IMAGE_SERVO_RECEIPT_FIELDS,
        finalize_image_servo_receipt,
        prepare_joint_mailbox,
        validate_public_receipt,
    )

    state, evidence, _unused_calibration = _stationary_receipt_boundary_inputs()
    state.update(_image_servo_public_state())
    state["state.arm_joint_position"] = _reset_qpos()
    evidence["end_effector_external_pixels_before"] = state[
        "state.end_effector_external_pixels"
    ]
    evidence["end_effector_external_pixels_after"] = state[
        "state.end_effector_external_pixels"
    ]
    calibration = _image_servo_calibration()
    mailbox, command = prepare_joint_mailbox(
        _image_servo_payload(),
        source="controller",
        observation_id="obs",
        current_qpos=_reset_qpos(),
        current_gripper=1.0,
        public_state=state,
        camera_calibration=calibration,
        remaining_actions=160,
        sequence=0,
    )
    execution = {
        "kind": "move_joints",
        "accepted": True,
        "bounded_endpoint": mailbox["endpoint"],
        "gripper_intent": 1.0,
        "step_count": 4,
        "maximum_commanded_step": 0.01,
        "endpoint_error": max(
            abs(target - actual)
            for target, actual in zip(
                mailbox["endpoint"], _reset_qpos(), strict=True
            )
        ),
        "minimum_hard_limit_margin": 0.5,
        "tracking_pause_count": 0,
        **evidence,
    }

    receipt = finalize_image_servo_receipt(
        command,
        mailbox,
        execution,
        before_state=state,
        after_state=state,
        before_calibration=calibration,
        after_calibration=calibration,
    )

    assert set(receipt) == PUBLIC_IMAGE_SERVO_RECEIPT_FIELDS
    assert receipt["kind"] == "image_servo"
    assert receipt["requested_camera"] == "left"
    assert receipt["requested_target_pixel"] == [60.0, 50.0]
    assert receipt["requested_target_role"] == "source_object"
    assert receipt["resolved_translation_m"] == pytest.approx([0.08, 0.0, 0.0])
    assert receipt["derived_joint_endpoint"] == mailbox["endpoint"]
    assert validate_public_receipt(receipt)["kind"] == "image_servo"
    overlong_actuation = _json_copy(receipt)
    assert isinstance(overlong_actuation, dict)
    overlong_actuation["requested_target_role"] = "articulation_motion"
    with pytest.raises(ValueError, match="image-servo geometry"):
        validate_public_receipt(overlong_actuation)

    from adaptive import critic_protocol
    from adaptive.joint_runner import _seal_receipt_rgb_change

    rgb_change = {"left": 0.0, "right": 0.0, "wrist": 0.0}
    sealed = _seal_receipt_rgb_change(
        receipt,
        rgb_change,
        source_sequence=0,
        fresh_sequence=1,
    )
    fresh_state = _state()
    fresh_state.update(_image_servo_public_state())
    fresh_state["state.arm_joint_position"] = _reset_qpos()
    fresh_state["state.arm_joint_velocity"] = [0.0] * 7
    fresh_state["state.arm_applied_torque"] = {
        "available": True,
        "values_nm": [0.0] * 7,
    }
    fresh_state["state.end_effector_wrench"] = {
        "force_n": [0.0] * 3,
        "torque_nm": [0.0] * 3,
    }
    fresh_state["state.gripper_qpos"] = [0.0, 0.0]
    reason, selected = critic_protocol.select_sealed_public_receipt(
        _image_servo_payload(),
        [sealed],
        fresh_state,
        rgb_change=rgb_change,
    )
    assert reason == "eligible"
    assert selected == sealed


def test_cartesian_linear_skill_resolves_identity_jacobian() -> None:
    from adaptive.cartesian_skill import (
        CartesianDeltaCommand,
    )
    from adaptive.cartesian_skill import (
        _resolve_linear_cartesian_delta as resolve_cartesian_delta,
    )

    command = CartesianDeltaCommand(
        observation_id="obs",
        translation_m=(0.01, -0.01, 0.005),
        rotation_axis_angle_rad=(0.02, 0.0, -0.01),
        gripper="hold",
        note="milestone=approach; evidence=RGB; expect=EEF moves",
    )
    result = resolve_cartesian_delta(
        command,
        _cartesian_identity_public_state(),
        current_gripper=1.0,
    )
    factor = 1.0 / 1.0025
    start = _cartesian_identity_public_state()["state.arm_joint_position"]
    assert result.joint_endpoint[:6] == pytest.approx(
        [
            start[0] + 0.01 * factor,
            start[1] - 0.01 * factor,
            start[2] + 0.005 * factor,
            start[3] + 0.02 * factor,
            start[4],
            start[5] - 0.01 * factor,
        ],
        abs=2e-3,
    )
    assert result.joint_endpoint[6] == pytest.approx(start[6] - 0.10)
    assert result.predicted_delta == pytest.approx(
        [0.01 * factor, -0.01 * factor, 0.005 * factor, 0.02 * factor, 0.0, -0.01 * factor],
        abs=2e-3,
    )
    assert result.residual_norm < 0.0025
    assert result.gripper_open == 1.0


def test_cartesian_linear_skill_uses_redundancy_to_center_nullspace_joint() -> None:
    from adaptive.cartesian_skill import (
        CartesianDeltaCommand,
    )
    from adaptive.cartesian_skill import (
        _resolve_linear_cartesian_delta as resolve_cartesian_delta,
    )

    state = _cartesian_identity_public_state()
    start_joint7 = state["state.arm_joint_position"][6]
    command = CartesianDeltaCommand(
        observation_id="obs",
        translation_m=(0.01, 0.0, 0.0),
        rotation_axis_angle_rad=(0.0, 0.0, 0.0),
        gripper="hold",
        note="milestone=approach; evidence=RGB; expect=EEF moves",
    )

    result = resolve_cartesian_delta(command, state, current_gripper=1.0)

    assert abs(result.joint_endpoint[6]) < abs(start_joint7)
    assert result.predicted_delta[0] == pytest.approx(0.01 / 1.0025, abs=1e-4)


def test_cartesian_linear_skill_allows_interpolated_quarter_radian_endpoint() -> None:
    from adaptive.cartesian_skill import (
        CartesianDeltaCommand,
    )
    from adaptive.cartesian_skill import (
        _resolve_linear_cartesian_delta as resolve_cartesian_delta,
    )

    state = _cartesian_identity_public_state()
    state["state.arm_translation_jacobian"][0][0] = 0.2
    command = CartesianDeltaCommand(
        observation_id="obs",
        translation_m=(0.04, 0.0, 0.0),
        rotation_axis_angle_rad=(0.0, 0.0, 0.0),
        gripper="hold",
        note="milestone=approach; evidence=RGB; expect=EEF moves",
    )

    result = resolve_cartesian_delta(command, state, current_gripper=1.0)

    joint1_delta = result.joint_endpoint[0] - state["state.arm_joint_position"][0]
    assert 0.12 < joint1_delta < 0.25


def test_cartesian_linear_skill_soft_orientation_recovers_coupled_translation() -> None:
    from adaptive.cartesian_skill import (
        CartesianDeltaCommand,
    )
    from adaptive.cartesian_skill import (
        _resolve_linear_cartesian_delta as resolve_cartesian_delta,
    )

    state = _cartesian_identity_public_state()
    state["state.arm_rotation_jacobian"][0] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    command = CartesianDeltaCommand(
        observation_id="obs",
        translation_m=(0.0, 0.04, 0.0),
        rotation_axis_angle_rad=(0.0, 0.0, 0.0),
        gripper="hold",
        note="milestone=approach; evidence=RGB; expect=EEF moves",
    )

    strict = resolve_cartesian_delta(command, state, current_gripper=1.0)
    softened = resolve_cartesian_delta(
        command,
        state,
        current_gripper=1.0,
        orientation_weight=0.25,
    )

    assert strict.predicted_delta[1] < 0.025
    assert softened.predicted_delta[1] > 0.035
    assert abs(softened.predicted_delta[3]) < 0.05


def test_cartesian_skill_rejects_zero_hold_and_oversized_motion() -> None:
    from adaptive.cartesian_skill import decode_cartesian_delta

    base = {
        "kind": "cartesian_delta",
        "observation_id": "obs",
        "translation_m": [0.0, 0.0, 0.0],
        "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        "gripper": "hold",
        "note": "milestone=approach; evidence=RGB; expect=EEF moves",
    }
    with pytest.raises(ValueError, match="zero Cartesian delta"):
        decode_cartesian_delta(base, observation_id="obs")
    with pytest.raises(ValueError, match="translation norm"):
        decode_cartesian_delta(
            {**base, "translation_m": [0.03, 0.03, 0.0]},
            observation_id="obs",
        )


def test_cartesian_skill_allows_gripper_only_transition() -> None:
    from adaptive.cartesian_skill import decode_cartesian_delta, resolve_cartesian_delta

    value = {
        "kind": "cartesian_delta",
        "observation_id": "obs",
        "translation_m": [0.0, 0.0, 0.0],
        "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        "gripper": "close",
        "note": "milestone=grasp; evidence=aligned RGB; expect=fingers close",
    }
    command = decode_cartesian_delta(value, observation_id="obs")
    result = resolve_cartesian_delta(
        command,
        _cartesian_identity_public_state(),
        current_gripper=1.0,
    )
    assert result.joint_endpoint == pytest.approx(
        _cartesian_identity_public_state()["state.arm_joint_position"]
    )
    assert result.gripper_open == 0.0


def test_cartesian_skill_rejects_invalid_public_jacobian() -> None:
    from adaptive.cartesian_skill import CartesianDeltaCommand, resolve_cartesian_delta

    state = _cartesian_identity_public_state()
    state["state.arm_rotation_jacobian"] = [[0.0] * 7] * 2
    command = CartesianDeltaCommand(
        observation_id="obs",
        translation_m=(0.01, 0.0, 0.0),
        rotation_axis_angle_rad=(0.0, 0.0, 0.0),
        gripper="hold",
        note="milestone=approach; evidence=RGB; expect=EEF moves",
    )
    with pytest.raises(ValueError, match="rotation Jacobian"):
        resolve_cartesian_delta(command, state, current_gripper=1.0)


def _cartesian_payload() -> dict[str, object]:
    return {
        "kind": "cartesian_delta",
        "observation_id": "obs",
        "translation_m": [0.01, 0.0, 0.0],
        "rotation_axis_angle_rad": [0.0, 0.0, 0.01],
        "gripper": "hold",
        "note": "milestone=approach; evidence=RGB; expect=EEF moves",
    }


def test_cartesian_mailbox_uses_derived_endpoint_without_changing_child_schema() -> None:
    from adaptive.cartesian_skill import CartesianDeltaCommand
    from adaptive.joint_runner import prepare_joint_mailbox

    state = _cartesian_identity_public_state()
    mailbox, command = prepare_joint_mailbox(
        _cartesian_payload(),
        source="controller",
        observation_id="obs",
        current_qpos=state["state.arm_joint_position"],
        current_gripper=1.0,
        public_state=state,
        remaining_actions=160,
        sequence=0,
    )
    assert isinstance(command, CartesianDeltaCommand)
    assert command.kind == "cartesian_delta"
    assert set(mailbox) == {
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
    assert mailbox["kind"] == "move_joints"
    assert mailbox["explicit_mask"] == [True] * 7
    assert mailbox["endpoint"] != state["state.arm_joint_position"]


def test_derived_skill_mailboxes_leave_room_for_release_and_verification() -> None:
    from adaptive.joint_protocol import MAX_COMMAND_ACTIONS
    from adaptive.joint_runner import (
        COFFEE_CONTROL_CONTACT_PATH_NOTE,
        COFFEE_CONTROL_CONTACT_WAYPOINTS,
        DERIVED_SKILL_MAX_ACTIONS,
        prepare_joint_mailbox,
    )
    from adaptive.joint_sim_child import EPISODE_ACTION_BUDGET

    cartesian_state = _cartesian_identity_public_state()
    cartesian_mailbox, _ = prepare_joint_mailbox(
        _cartesian_payload(),
        source="controller",
        observation_id="obs",
        current_qpos=cartesian_state["state.arm_joint_position"],
        current_gripper=1.0,
        public_state=cartesian_state,
        remaining_actions=EPISODE_ACTION_BUDGET,
        sequence=0,
    )
    image_state = _image_servo_public_state()
    image_mailbox, _ = prepare_joint_mailbox(
        _image_servo_payload(),
        source="controller",
        observation_id="obs",
        current_qpos=image_state["state.arm_joint_position"],
        current_gripper=1.0,
        public_state=image_state,
        camera_calibration=_image_servo_calibration(),
        remaining_actions=EPISODE_ACTION_BUDGET,
        sequence=1,
    )
    joint_mailbox, _ = prepare_joint_mailbox(
        {
            "kind": "move_joints",
            "observation_id": "obs",
            "targets": {"gripper": 1.0},
            "note": "milestone=observe; establish a known open gripper",
        },
        source="controller",
        observation_id="obs",
        current_qpos=_reset_qpos(),
        current_gripper=0.5,
        remaining_actions=EPISODE_ACTION_BUDGET,
        sequence=2,
    )
    contact_path_mailbox, _ = prepare_joint_mailbox(
        {
            "kind": "move_joints",
            "observation_id": "obs",
            "targets": {
                **{
                    f"joint{index + 1}": value
                    for index, value in enumerate(
                        COFFEE_CONTROL_CONTACT_WAYPOINTS[0]
                    )
                },
                "gripper": 1.0,
            },
            "tracking_mode": "lag_pause",
            "note": COFFEE_CONTROL_CONTACT_PATH_NOTE,
        },
        source="controller",
        observation_id="obs",
        current_qpos=_reset_qpos(),
        current_gripper=1.0,
        remaining_actions=EPISODE_ACTION_BUDGET,
        sequence=3,
    )

    assert DERIVED_SKILL_MAX_ACTIONS == 18
    assert cartesian_mailbox["max_actions"] == DERIVED_SKILL_MAX_ACTIONS
    assert image_mailbox["max_actions"] == DERIVED_SKILL_MAX_ACTIONS
    assert joint_mailbox["max_actions"] == MAX_COMMAND_ACTIONS
    assert contact_path_mailbox["max_actions"] == DERIVED_SKILL_MAX_ACTIONS
    assert "tracking_mode" not in contact_path_mailbox
    assert MAX_COMMAND_ACTIONS + 23 * DERIVED_SKILL_MAX_ACTIONS <= EPISODE_ACTION_BUDGET


def test_cartesian_response_schema_is_closed_and_command_dict_is_original() -> None:
    from adaptive.cartesian_skill import decode_cartesian_delta
    from adaptive.joint_runner import (
        CARTESIAN_RESPONSE_SCHEMA,
        CONTROLLER_RESPONSE_SCHEMA,
        _command_dict,
    )

    assert CARTESIAN_RESPONSE_SCHEMA in CONTROLLER_RESPONSE_SCHEMA["oneOf"]
    assert CARTESIAN_RESPONSE_SCHEMA["additionalProperties"] is False
    command = decode_cartesian_delta(_cartesian_payload(), observation_id="obs")
    assert _command_dict(command) == _cartesian_payload()


def test_cartesian_receipt_preserves_request_and_recomputable_resolution() -> None:
    from adaptive import critic_protocol
    from adaptive.joint_runner import (
        PUBLIC_CARTESIAN_RECEIPT_FIELDS,
        _seal_receipt_rgb_change,
        finalize_cartesian_receipt,
        prepare_joint_mailbox,
        validate_public_receipt,
    )

    payload = {
        **_cartesian_payload(),
        "translation_m": [0.0, 0.0, 0.0],
        "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        "gripper": "close",
    }
    state, evidence, calibration = _stationary_receipt_boundary_inputs()
    state.update(_cartesian_identity_public_state())
    state["state.arm_joint_position"] = _reset_qpos()
    mailbox, command = prepare_joint_mailbox(
        payload,
        source="controller",
        observation_id="obs",
        current_qpos=_reset_qpos(),
        current_gripper=1.0,
        public_state=state,
        remaining_actions=MIN_GRIPPER_ACTIONS,
        sequence=0,
    )
    execution = {
        "kind": "move_joints",
        "accepted": True,
        "bounded_endpoint": mailbox["endpoint"],
        "gripper_intent": 0.0,
        "step_count": MIN_GRIPPER_ACTIONS,
        "maximum_commanded_step": 0.0,
        "endpoint_error": 0.0,
        "minimum_hard_limit_margin": 0.5,
        "tracking_pause_count": 0,
        **evidence,
    }
    receipt = finalize_cartesian_receipt(
        command,
        mailbox,
        execution,
        before_state=state,
        after_state=state,
        before_calibration=calibration,
        after_calibration=calibration,
    )
    assert set(receipt) == PUBLIC_CARTESIAN_RECEIPT_FIELDS
    assert receipt["kind"] == "cartesian_delta"
    assert receipt["requested_translation_m"] == [0.0, 0.0, 0.0]
    assert receipt["requested_rotation_axis_angle_rad"] == [0.0, 0.0, 0.0]
    assert receipt["requested_gripper"] == "close"
    assert receipt["derived_joint_endpoint"] == mailbox["endpoint"]
    assert validate_public_receipt(receipt)["kind"] == "cartesian_delta"

    rgb_change = {"left": 0.0, "right": 0.0, "wrist": 0.0}
    sealed = _seal_receipt_rgb_change(
        receipt,
        rgb_change,
        source_sequence=0,
        fresh_sequence=1,
    )
    fresh_state = _state()
    fresh_state.update(state)
    fresh_state.update(_cartesian_identity_public_state())
    fresh_state["state.arm_joint_position"] = _reset_qpos()
    fresh_state["state.arm_joint_velocity"] = [0.0] * 7
    fresh_state["state.arm_applied_torque"] = {
        "available": True,
        "values_nm": [0.0] * 7,
    }
    fresh_state["state.end_effector_wrench"] = {
        "force_n": [0.0] * 3,
        "torque_nm": [0.0] * 3,
    }
    reason, selected = critic_protocol.select_sealed_public_receipt(
        payload,
        [sealed],
        fresh_state,
        rgb_change=rgb_change,
    )
    assert reason == "eligible"
    assert selected == sealed


def _cartesian_smoke_receipt(
    requested_x: float,
    actual_x: float,
    *,
    gripper: str = "hold",
    start_separation: float = 0.08,
    end_separation: float = 0.08,
) -> dict[str, object]:
    return {
        "kind": "cartesian_delta",
        "accepted": True,
        "requested_translation_m": [requested_x, 0.0, 0.0],
        "requested_rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        "requested_gripper": gripper,
        "end_effector_pose_delta": {
            "translation_m": [actual_x, 0.0, 0.0],
            "rotation_axis_angle_rad": [0.0, 0.001, 0.0],
        },
        "gripper_residual": {
            "start_qpos": [start_separation / 2.0, -start_separation / 2.0],
            "end_qpos": [end_separation / 2.0, -end_separation / 2.0],
            "qpos_delta": [
                (end_separation - start_separation) / 2.0,
                -(end_separation - start_separation) / 2.0,
            ],
            "measured_end_finger_separation": end_separation,
        },
    }


def test_cartesian_skill_smoke_requires_motion_return_orientation_and_gripper() -> None:
    from adaptive.cartesian_skill_smoke import evaluate_cartesian_skill_episode

    episode = {
        "status": "policy_finished_false",
        "simulator_steps": 64,
        "terminal_outcome": {"status": "finished_false"},
        "receipts": [
            _cartesian_smoke_receipt(0.01, 0.009),
            _cartesian_smoke_receipt(-0.01, -0.0085),
            _cartesian_smoke_receipt(
                0.0,
                0.0,
                gripper="close",
                start_separation=0.08,
                end_separation=0.01,
            ),
            _cartesian_smoke_receipt(
                0.0,
                0.0,
                gripper="open",
                start_separation=0.01,
                end_separation=0.075,
            ),
        ],
    }

    passed, checks, metrics = evaluate_cartesian_skill_episode(episode)

    assert passed is True
    assert all(checks.values())
    assert metrics["outbound_translation_m"] == pytest.approx(0.009)
    assert metrics["return_translation_error_m"] == pytest.approx(0.0005)


def test_candidate_cartesian_skill_smoke_dispatches_one_remote_module(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate = importlib.import_module("candidate")
    calls: list[list[str]] = []
    result = {
        "schema": "robocasa-inspect-cartesian-skill-smoke/v1",
        "task": "OpenToasterOvenDoor",
        "seed": 7,
        "release_digest": "digest",
        "artifact_root": (
            "/home/jli/state/qwen-workflow-recovery/"
            "run-100-123/cartesian-skill"
        ),
        "passed": True,
        "checks": {"closed_cartesian_receipts": True},
        "metrics": {},
        "episode_result_sha256": "a" * 64,
        "video": "/state/run/cartesian-skill/video.mp4",
        "video_sha256": "b" * 64,
        "wall_s": 1.0,
    }
    monkeypatch.setattr(candidate, "deploy_remote", lambda: "/release/digest")
    monkeypatch.setattr(candidate.time, "time", lambda: 100)
    monkeypatch.setattr(candidate.os, "getpid", lambda: 123)

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps(result), stderr="")

    monkeypatch.setattr(candidate, "_run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["candidate.py", "--cartesian-skill-smoke", "OpenToasterOvenDoor"],
    )

    assert candidate.main() == 0
    assert len(calls) == 1
    assert calls[0][calls[0].index("-m") + 1] == "adaptive.cartesian_skill_smoke"
    assert calls[0][calls[0].index("--task") + 1] == "OpenToasterOvenDoor"
    assert capsys.readouterr().out == json.dumps(
        result, sort_keys=True, separators=(",", ":")
    ) + "\n"


def test_candidate_image_servo_smoke_dispatches_one_remote_module(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate = importlib.import_module("candidate")
    calls: list[list[str]] = []
    result = {
        "schema": "robocasa-inspect-image-servo-smoke/v1",
        "task": "OpenToasterOvenDoor",
        "seed": 7,
        "release_digest": "digest",
        "artifact_root": (
            "/home/jli/state/qwen-workflow-recovery/"
            "run-100-123/image-servo"
        ),
        "passed": True,
        "checks": {"closed_image_servo_receipts": True},
        "metrics": {},
        "episode_result_sha256": "a" * 64,
        "video": "/state/run/image-servo/video.mp4",
        "video_sha256": "b" * 64,
        "wall_s": 1.0,
    }
    monkeypatch.setattr(candidate, "deploy_remote", lambda: "/release/digest")
    monkeypatch.setattr(candidate.time, "time", lambda: 100)
    monkeypatch.setattr(candidate.os, "getpid", lambda: 123)

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps(result), stderr="")

    monkeypatch.setattr(candidate, "_run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["candidate.py", "--image-servo-smoke", "OpenToasterOvenDoor"],
    )

    assert candidate.main() == 0
    assert len(calls) == 1
    assert calls[0][calls[0].index("-m") + 1] == "adaptive.image_servo_smoke"
    assert calls[0][calls[0].index("--task") + 1] == "OpenToasterOvenDoor"
    assert capsys.readouterr().out == json.dumps(
        result, sort_keys=True, separators=(",", ":")
    ) + "\n"


def test_grounded_diagnostic_gate_requires_three_effects_two_advances_close_success() -> None:
    from adaptive.grounded_diagnostic import evaluate_grounded_diagnostic

    episodes = []
    for index, (task, family, history) in enumerate(
        (
            ("AdjustWaterTemperature", "control", ["observe", "approach", "engage"]),
            ("OpenToasterOvenDoor", "articulated", ["observe", "approach", "engage"]),
            ("PickPlaceToasterToCounter", "grasp_place", ["observe", "approach"]),
        )
    ):
        receipt = _cartesian_smoke_receipt(0.01, 0.006)
        if index == 0:
            receipt["kind"] = "image_servo"
        if index == 1:
            receipt = _cartesian_smoke_receipt(
                0.0,
                0.0,
                gripper="close",
                start_separation=0.08,
                end_separation=0.02,
            )
        episodes.append({
            "task": task,
            "family": family,
            "status": "success" if index == 0 else "policy_failed_decision_budget",
            "success": index == 0,
            "simulator_steps": 100,
            "receipts": [receipt],
            "milestone_history": history,
            "proposal_audit_closure": {"valid": True},
        })

    passed, checks, metrics = evaluate_grounded_diagnostic(episodes)

    assert passed is True
    assert all(checks.values())
    assert metrics["tasks_with_cartesian_effect"] == 3
    assert metrics["tasks_beyond_approach"] == 2
    assert metrics["tasks_with_close_effect"] == 1
    assert metrics["official_material_successes"] == 1


def test_candidate_grounded_diagnostic_dispatches_controlled_seed_1711_replay(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate = importlib.import_module("candidate")
    calls: list[list[str]] = []
    result = {
        "schema": "robocasa-grounded-panda-diagnostic/v6",
        "release_digest": "digest",
        "seed": 1711,
        "tasks": [
            "AdjustWaterTemperature",
            "OpenToasterOvenDoor",
            "PickPlaceToasterToCounter",
        ],
        "passed": False,
        "checks": {},
        "metrics": {},
        "episodes": [],
        "model_calls": 0,
        "controller_calls": 0,
        "critic_attempts": 0,
        "artifact_root": (
            "/home/jli/state/qwen-workflow-recovery/"
            "run-100-123/grounded-diagnostic"
        ),
        "wall_s": 1.0,
    }
    monkeypatch.setattr(candidate, "deploy_remote", lambda: "/release/digest")
    monkeypatch.setattr(candidate.time, "time", lambda: 100)
    monkeypatch.setattr(candidate.os, "getpid", lambda: 123)

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps(result), stderr="")

    monkeypatch.setattr(candidate, "_run", fake_run)
    monkeypatch.setattr(sys, "argv", ["candidate.py", "--grounded-diagnostic"])

    assert candidate.main() == 0
    assert len(calls) == 1
    assert calls[0][calls[0].index("-m") + 1] == "adaptive.grounded_diagnostic"
    assert calls[0][calls[0].index("--seed") + 1] == "1711"
    assert capsys.readouterr().out == json.dumps(
        result, sort_keys=True, separators=(",", ":")
    ) + "\n"


def test_image_servo_smoke_requires_pixel_motion_and_return() -> None:
    from adaptive.image_servo_smoke import evaluate_image_servo_episode

    common = {
        "kind": "image_servo",
        "accepted": True,
        "requested_camera": "left",
        "end_effector_pose_delta": {
            "translation_m": [0.009, 0.0, 0.0],
            "rotation_axis_angle_rad": [0.0, 0.001, 0.0],
        },
    }
    episode = {
        "status": "policy_finished_false",
        "simulator_steps": 24,
        "terminal_outcome": {"status": "finished_false"},
        "receipts": [
            {
                **common,
                "requested_target_pixel": [106.0, 80.0],
                "end_effector_external_pixel_displacement": {
                    "left": {
                        "start_px": [100.0, 80.0],
                        "end_px": [104.0, 80.2],
                    }
                },
            },
            {
                **common,
                "requested_target_pixel": [100.0, 80.0],
                "end_effector_pose_delta": {
                    "translation_m": [-0.008, 0.0, 0.0],
                    "rotation_axis_angle_rad": [0.0, 0.001, 0.0],
                },
                "end_effector_external_pixel_displacement": {
                    "left": {
                        "start_px": [104.0, 80.2],
                        "end_px": [100.8, 80.1],
                    }
                },
            },
        ],
    }

    passed, checks, metrics = evaluate_image_servo_episode(episode)

    assert passed is True
    assert all(checks.values())
    assert metrics["outbound_pixel_motion_px"] == pytest.approx(4.0)
    assert metrics["return_pixel_error_px"] == pytest.approx(
        math.hypot(0.8, 0.1)
    )


def test_single_toaster_empty_target_requires_open_joint7_reorientation() -> None:
    from adaptive import remote_driver
    from adaptive.critic_protocol import validate_milestone_action_semantics

    failure = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [110.0, 120.0],
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "close",
        "gripper_residual": {"measured_end_finger_separation": 0.00237},
    }
    retreat = {
        "kind": "cartesian_delta",
        "requested_gripper": "open",
        "gripper_residual": {"measured_end_finger_separation": 0.078},
    }
    forbidden = [{"camera": "right", "target_pixel": [110.0, 120.0]}]
    another_pixel = {
        "kind": "image_servo",
        "observation_id": "obs-single-toaster",
        "camera": "right",
        "target_pixel": [150.0, 120.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.01,
        "step_m": 0.01,
        "gripper": "open",
        "note": "milestone=engage; try a second handle point",
    }
    wrist_roll = {
        "kind": "move_joints",
        "observation_id": "obs-single-toaster",
        "targets": {"joint7": -0.86, "gripper": 1.0},
        "note": "milestone=engage; roll the open finger gap across the pull",
    }

    with pytest.raises(ValueError, match="after 1 empty"):
        validate_milestone_action_semantics(
            "articulated",
            "engage",
            another_pixel,
            task="OpenToasterOvenDoor",
            milestone_history=["observe", "approach", "engage"],
            immediate_prior_receipt=retreat,
            recent_receipts=[failure, retreat],
            articulated_forbidden_targets=forbidden,
        )
    validate_milestone_action_semantics(
        "articulated",
        "engage",
        wrist_roll,
        task="OpenToasterOvenDoor",
        milestone_history=["observe", "approach", "engage"],
        immediate_prior_receipt=retreat,
        recent_receipts=[failure, retreat],
        articulated_forbidden_targets=forbidden,
    )
    for other_task in (None, "PickPlaceCounterToDrawer"):
        validate_milestone_action_semantics(
            "articulated",
            "engage",
            another_pixel,
            task=other_task,
            milestone_history=["observe", "approach", "engage"],
            immediate_prior_receipt=retreat,
            recent_receipts=[failure, retreat],
            articulated_forbidden_targets=forbidden,
        )

    generic_instruction = remote_driver._controller_articulated_recovery_instruction(
        "base", forbidden, retreat
    ).casefold()
    assert "orientation_recovery_required" not in generic_instruction
    toaster_instruction = remote_driver._controller_articulated_recovery_instruction(
        "base", forbidden, retreat, task="OpenToasterOvenDoor"
    ).casefold()
    assert "visually screened toaster handle target produced an empty" in (
        toaster_instruction
    )
    assert '"distinct_empty_target_count":1' in toaster_instruction

    context = _context()
    context.milestone_history = ["observe", "approach", "engage"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    _FakeClient.controller_outputs = [another_pixel, wrist_roll]
    _FakeClient.critic_outputs = [_proposal_audit()]
    state = _state()
    state.update(_image_servo_public_state())

    response = _proposal_complete(
        wrapped,
        "obs-single-toaster",
        state,
        _images(),
        receipts=[failure, retreat],
        camera_calibration=_image_servo_calibration(),
    )

    assert response.command is wrist_roll
    assert [role for role, _ in _FakeClient.calls] == [
        "controller",
        "controller",
        "critic",
    ]
    assert context.proposal_records[0]["status"] == "rejected_by_protocol"
    first_instruction = str(_FakeClient.calls[0][1]["instruction"]).casefold()
    assert "visually screened toaster handle target produced an empty" in (
        first_instruction
    )
    revision = str(_FakeClient.calls[1][1]["instruction"]).casefold()
    assert "visually screened toaster handle target has already" in revision
    assert "stop selecting another handle pixel" in revision
    assert "only `joint7` and `gripper`" in revision
    critic_payload = json.loads(str(_FakeClient.calls[2][1]["instruction"]))
    assert critic_payload["repeated_empty_handle_orientation_rule"] == {
        "default_verdict": "approve",
        "effect_is_future_evidence": True,
        "gripper_must_remain_open": True,
        "joint7_roll_only": True,
        "single_visually_screened_toaster_target": True,
    }

    other_context = dataclasses.replace(_context(), task="PickPlaceCounterToDrawer")
    other_payload = json.loads(
        remote_driver._proposal_critic_instruction(
            other_context,
            task_instruction="pick the bowl from the counter into the drawer",
            observation_id="obs-single-toaster",
            draft=wrist_roll,
            claimed_milestone="engage",
            public_state=state,
            images=_images(),
            immediate_prior_receipt=retreat,
            recent_receipts=[failure, retreat],
        )
    )
    assert other_payload["repeated_empty_handle_orientation_rule"] is None


def test_prompt_variant_selects_rig_files_and_binds_hashes() -> None:
    from adaptive.joint_runner import (
        PROMPT_VARIANTS,
        joint_system_prompt_parts,
        load_joint_system_prompt,
    )

    root = Path(__file__).resolve().parents[1]
    assert set(PROMPT_VARIANTS) == {"baseline", "rig"}
    assert load_joint_system_prompt(root, variant="baseline") == (
        load_joint_system_prompt(root)
    )
    rig = load_joint_system_prompt(root, variant="rig")
    expected = (
        (root / "prompts" / "joint_system_rig.txt").read_bytes()
        + b"\n"
        + (root / "prompts" / "panda_rig_facts.txt").read_bytes()
        + b"\n"
        + (root / "prompts" / "proposal_audit_compatibility.txt").read_bytes()
    )
    assert rig.encode() == expected
    assert 12_000 <= len(expected) <= 36_000
    parts = joint_system_prompt_parts(root, protocol="proposal", variant="rig")
    assert list(parts) == [
        "joint_system_rig.txt",
        "panda_rig_facts.txt",
        "proposal_audit_compatibility.txt",
    ]
    assert parts["panda_rig_facts.txt"] == hashlib.sha256(
        (root / "prompts" / "panda_rig_facts.txt").read_bytes()
    ).hexdigest()
    assert load_joint_system_prompt(root, protocol="legacy", variant="rig") == (
        load_joint_system_prompt(root, protocol="legacy")
    )
    with pytest.raises(ValueError, match="variant"):
        load_joint_system_prompt(root, variant="nope")


def test_rig_prompt_planar_ik_matches_installed_fk() -> None:
    import random

    from adaptive.panda_embodiment import panda_fk

    root = Path(__file__).resolve().parents[1]
    facts = (root / "prompts" / "panda_rig_facts.txt").read_text()
    for token in (
        "L1 = 0.3266",
        "a1 = 0.2554",
        "L2 = 0.3928",
        "a2 = 0.2116",
        "L3 = 0.2217",
        "a3 = 0.4081",
        "L0 = 0.333",
        "0.7194",
    ):
        assert token in facts, token
    L0 = 0.333
    L1, a1 = math.hypot(0.316, 0.0825), math.atan2(0.0825, 0.316)
    L2, a2 = math.hypot(0.384, 0.0825), math.atan2(0.0825, 0.384)
    L3, a3 = math.hypot(0.088, 0.2035), math.atan2(0.088, 0.2035)
    assert (round(L1, 4), round(a1, 4), round(L2, 4), round(a2, 4)) == (
        0.3266,
        0.2554,
        0.3928,
        0.2116,
    )
    assert (round(L3, 4), round(a3, 4), round(L1 + L2, 4)) == (0.2217, 0.4081, 0.7194)

    def wrap(value: float) -> float:
        return (value + math.pi) % (2 * math.pi) - math.pi

    def inverse(x: float, z: float, phi: float, branch: int) -> tuple[float, ...]:
        c_angle = phi + math.pi - a3
        wx, wz = x - L3 * math.sin(c_angle), z - L3 * math.cos(c_angle)
        px, pz = wx, wz - L0
        d = math.hypot(px, pz)
        cos_a = max(-1.0, min(1.0, (d * d + L1 * L1 - L2 * L2) / (2 * L1 * d)))
        a_angle = math.atan2(px, pz) + branch * math.acos(cos_a)
        joint2 = wrap(a_angle - a1)
        b_angle = math.atan2(
            wx - L1 * math.sin(a_angle), wz - L0 - L1 * math.cos(a_angle)
        )
        joint4 = wrap(joint2 - a2 - b_angle)
        return joint2, joint4, wrap(joint2 - joint4 - phi)

    rng = random.Random(7)
    minus_branch_hits = 0
    for _ in range(500):
        q = [
            0.0,
            rng.uniform(-1.7, 1.7),
            0.0,
            rng.uniform(-3.0, -0.1),
            0.0,
            rng.uniform(0.0, 3.7),
            0.0,
        ]
        p = panda_fk(q).position_m
        phi = q[1] - q[3] - q[5]
        a_angle, b_angle = q[1] + a1, q[1] - q[3] - a2
        c_angle = phi + math.pi - a3
        assert math.isclose(
            p[0],
            L1 * math.sin(a_angle) + L2 * math.sin(b_angle) + L3 * math.sin(c_angle),
            abs_tol=1e-9,
        )
        assert math.isclose(
            p[2],
            L0 + L1 * math.cos(a_angle) + L2 * math.cos(b_angle) + L3 * math.cos(c_angle),
            abs_tol=1e-9,
        )
        assert abs(p[1]) < 1e-9
        truth = (q[1], q[3], q[5])
        matches = [
            all(
                abs(wrap(a - b)) < 1e-6
                for a, b in zip(truth, inverse(p[0], p[2], phi, branch), strict=True)
            )
            for branch in (-1, 1)
        ]
        assert any(matches)
        minus_branch_hits += int(matches[0])
    assert minus_branch_hits > 400


def test_rig_prompt_keeps_the_command_and_milestone_contracts() -> None:
    root = Path(__file__).resolve().parents[1]
    baseline = (root / "prompts" / "joint_system.txt").read_text()
    rig = (root / "prompts" / "joint_system_rig.txt").read_text()
    marker = "cells to locate targets; the grid is not a task object.\n"
    assert rig.endswith(baseline[baseline.index(marker) + len(marker) :])
    for heading in (
        "Command contract",
        "Authority contract",
        "Allowed JSON forms",
        "Proposal-audit and milestone contract",
    ):
        assert heading in rig, heading
    facts = (root / "prompts" / "panda_rig_facts.txt").read_text()
    for heading in ("# Embodiment notes", "# Rig facts", "# Rig formulas", "# Rig advice"):
        assert heading in facts, heading
    for forbidden in ("toaster_oven_main_group", "OpenToasterOvenDoor", "92.80", "153.43"):
        assert forbidden not in facts, forbidden
    assert "0.0032" in facts and "0.0024" in facts and "joint7" in facts


def _ab_episode(*, close_separation: float, status: str, steps: int) -> dict[str, object]:
    servo = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [110.0, 120.0],
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "open",
        "gripper_residual": {"measured_end_finger_separation": 0.0798},
        "step_count": 18,
    }
    close = {
        "kind": "move_joints",
        "requested_targets": {"gripper": 0.0},
        "gripper_intent": 0.0,
        "gripper_residual": {"measured_end_finger_separation": close_separation},
        "step_count": 16,
    }
    retreat = {
        "kind": "cartesian_delta",
        "requested_gripper": "open",
        "gripper_residual": {"measured_end_finger_separation": 0.0779},
        "step_count": 18,
    }
    roll = {
        "kind": "move_joints",
        "requested_targets": {"gripper": 1.0, "joint7": -0.86},
        "gripper_intent": 1.0,
        "gripper_residual": {"measured_end_finger_separation": 0.0791},
        "step_count": 32,
    }
    contact = close_separation > 0.0032
    receipts = [servo] * 3 + [close] + ([] if contact else [retreat, roll])
    return {
        "task": "OpenToasterOvenDoor",
        "seed": 7,
        "family": "articulated",
        "status": status,
        "success": status == "success",
        "served_model_id": "qwen3.8-27b-bf16-x",
        "snapshot_digest": "snap",
        "system_prompt_sha256": "abc123",
        "system_prompt_parts_sha256": {"joint_system.txt": "s1"},
        "critic_prompt_sha256": "crit",
        "decisions": len(receipts),
        "qwen_controller_calls": 9,
        "qwen_critic_attempts": 6,
        "simulator_steps": steps,
        "wall_s": 180.0,
        "milestone_history": ["observe", "approach", "engage", "engage"]
        + (["actuate"] if contact else []),
        "receipts": receipts,
        "proposal_audit_closure": {"rejected_count": 3},
        "proposal_audit_records": [
            {"status": "approved_for_execution", "contradiction": "none"},
            {
                "status": "rejected_by_critic",
                "contradiction": "visual_alignment_unverified",
            },
        ],
        "video": "/remote/run.mp4",
    }


def test_ab_harness_forwards_development_action_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from adaptive import prompt_ab_harness

    recorded: dict[str, object] = {}

    def fake_run_variant(**kwargs: object) -> dict[str, object]:
        recorded.update(kwargs)
        return {"variant": "rig"}

    monkeypatch.setattr(prompt_ab_harness, "_run_variant", fake_run_variant)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prompt_ab_harness.py",
            "run",
            "--task",
            "OpenToasterOvenDoor",
            "--decisions",
            "40",
            "--variants",
            "rig",
            "--smoke-action-budget",
            "900",
            "--out",
            str(tmp_path / "ab"),
        ],
    )

    assert prompt_ab_harness.main() == 0
    assert recorded["action_budget"] == 900


def test_ab_harness_scores_contact_and_failure_class() -> None:
    from adaptive.prompt_ab_harness import (
        ROW_FIELDS,
        build_ab_report,
        classify_failure,
        render_markdown,
        summarize_episode,
    )

    empty = _ab_episode(
        close_separation=0.00237, status="policy_failed_proposal_audit", steps=228
    )
    contact = _ab_episode(close_separation=0.012, status="success", steps=300)
    budget = _ab_episode(
        close_separation=0.00237, status="policy_failed_proposal_audit", steps=450
    )
    extended_mid_failure = _ab_episode(
        close_separation=0.00237, status="policy_failed_proposal_audit", steps=600
    )
    extended_budget = _ab_episode(
        close_separation=0.00237, status="policy_failed_proposal_audit", steps=900
    )
    empty_row = summarize_episode(empty, variant="baseline", run_dir="baseline-1")
    contact_row = summarize_episode(
        contact, variant="rig", run_dir="rig-1", release_digest="deadbeef"
    )
    assert tuple(empty_row) == ROW_FIELDS
    assert empty_row["first_handle_contact_decision"] is None
    assert empty_row["empty_close_count"] == 1
    assert empty_row["wrist_roll_count"] == 1
    assert empty_row["deepest_milestone"] == "engage"
    assert empty_row["max_close_separation_m"] == pytest.approx(0.00237)
    assert empty_row["failure_class"] == (
        "revisions_exhausted:visual_alignment_unverified"
    )
    assert empty_row["rejected_proposals"] == 3
    assert contact_row["first_handle_contact_decision"] == 3
    assert contact_row["empty_close_count"] == 0
    assert contact_row["deepest_milestone"] == "actuate"
    assert contact_row["failure_class"] == "success"
    assert contact_row["release_digest"] == "deadbeef"
    assert classify_failure(budget) == "action_budget_exhausted"
    assert classify_failure(extended_mid_failure, action_budget=900) == (
        "revisions_exhausted:visual_alignment_unverified"
    )
    assert classify_failure(extended_budget, action_budget=900) == (
        "action_budget_exhausted"
    )
    assert classify_failure({**empty, "status": "policy_failed_wall_budget"}) == (
        "wall_budget"
    )
    assert classify_failure({**empty, "status": "policy_gave_up"}) == "gave_up"

    report = build_ab_report([empty_row, contact_row])
    assert report["schema"] == "robocasa-qwen-prompt-ab-report/v1"
    assert report["summary"]["baseline"]["runs_with_handle_contact"] == 0
    assert report["summary"]["rig"]["earliest_handle_contact_decision"] == 3
    assert report["summary"]["rig"]["successes"] == 1
    markdown = render_markdown(report)
    assert markdown.count("\n| ") == 3
    assert "- **baseline**: 1 run(s), 0 success(es), 0 with handle contact" in markdown
    assert "- **rig**: 1 run(s), 1 success(es), 1 with handle contact" in markdown


def test_orientation_roll_revision_targets_the_note_when_the_form_was_right() -> None:
    from adaptive import remote_driver

    failure = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [110.0, 120.0],
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "close",
        "gripper_residual": {"measured_end_finger_separation": 0.00237},
    }
    retreat = {
        "kind": "cartesian_delta",
        "requested_gripper": "open",
        "gripper_residual": {"measured_end_finger_separation": 0.078},
    }
    wrist_roll = {
        "kind": "move_joints",
        "observation_id": "obs-roll-note",
        "targets": {"joint7": -0.8586, "gripper": 1.0},
        "note": "milestone=engage; I am retreating with a cartesian_delta",
    }
    another_pixel = {
        "kind": "image_servo",
        "observation_id": "obs-roll-note",
        "camera": "right",
        "target_pixel": [150.0, 120.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.01,
        "step_m": 0.01,
        "gripper": "open",
        "note": "milestone=engage; try a second handle point",
    }

    def revision(rejected: dict[str, object]) -> str:
        return remote_driver._proposal_revision_instruction(
            "base",
            contradiction="motion_effect_mismatch",
            evidence=["external_rgb", "wrist_rgb", "jacobian_projection"],
            suggested_correction="revise_alignment",
            confidence="high",
            rejected_draft=rejected,
            required_observation_id="obs-roll-note",
            immediate_prior_receipt=retreat,
            recent_receipts=[failure, retreat],
            task="OpenToasterOvenDoor",
        ).casefold()

    roll_revision = revision(wrist_roll)
    assert "already had the required form" in roll_revision
    assert "rewrite the note for this roll" in roll_revision
    assert "do not describe or repeat it" in roll_revision
    assert "stop selecting another handle pixel" not in roll_revision

    pixel_revision = revision(another_pixel)
    assert "stop selecting another handle pixel" in pixel_revision
    assert "do not describe or repeat it" in pixel_revision
    assert "already had the required form" not in pixel_revision

    generic_revision = remote_driver._proposal_revision_instruction(
        "base",
        contradiction="motion_effect_mismatch",
        evidence=["external_rgb"],
        suggested_correction="revise_alignment",
        confidence="high",
        rejected_draft=wrist_roll,
        required_observation_id="obs-roll-note",
        immediate_prior_receipt=retreat,
        recent_receipts=[failure, retreat],
        task="PickPlaceCounterToDrawer",
    ).casefold()
    assert "already had the required form" not in generic_revision

    controller = remote_driver._controller_articulated_recovery_instruction(
        "base",
        [{"camera": "right", "target_pixel": [110.0, 120.0]}],
        retreat,
        task="OpenToasterOvenDoor",
    ).casefold()
    assert "do not describe or repeat it" in controller
    assert "finger gap rotates about the tool axis" in controller


class _TwoViewImage:
    """Synthetic 256x256 view with the grid and one dark horizontal pull."""

    size = (256, 256)

    def __init__(self, *, pull_u: tuple[int, int], pull_v: tuple[int, int]) -> None:
        self.pull_u = pull_u
        self.pull_v = pull_v

    def getpixel(self, point: tuple[int, int]) -> tuple[int, int, int]:
        x, y = point
        if x % 32 == 0:
            return (255, 255, 0)
        if y % 32 == 0:
            return (0, 255, 255)
        if self.pull_u[0] <= x <= self.pull_u[1] and self.pull_v[0] <= y <= self.pull_v[1]:
            return (20, 20, 20)
        return (110, 110, 110)


def test_cross_view_status_flags_grip_site_off_the_pull_in_the_other_view() -> None:
    from adaptive import remote_driver

    left = _TwoViewImage(pull_u=(131, 175), pull_v=(121, 129))
    off = remote_driver._cross_view_status_from_image(
        left,
        other_camera="left",
        pixel_record={"u_px": 263.3, "v_px": 106.3, "visible": False, "depth_valid": True},
    )
    assert off == {
        "other_camera": "left",
        "grip_site_pixel": [263.3, 106.3],
        "grip_site_visible": False,
        "horizontal_pull_detected": True,
        "pull_box": {"u_min": 131, "u_max": 175, "v_min": 121, "v_max": 129},
        "grip_site_offset_px": [88.3, 14.7],
        "tolerance_px": 12,
        "grip_site_on_horizontal_pull": False,
    }
    assert remote_driver._cross_view_close_blocked(off) is True
    on = remote_driver._cross_view_status_from_image(
        left,
        other_camera="left",
        pixel_record={"u_px": 150.0, "v_px": 118.0, "visible": True, "depth_valid": True},
    )
    assert on is not None
    assert on["grip_site_offset_px"] == [0.0, 3.0]
    assert on["grip_site_on_horizontal_pull"] is True
    assert remote_driver._cross_view_close_blocked(on) is False
    blank = remote_driver._cross_view_status_from_image(
        _TwoViewImage(pull_u=(0, 0), pull_v=(0, 0)),
        other_camera="left",
        pixel_record={"u_px": 150.0, "v_px": 118.0, "visible": True, "depth_valid": True},
    )
    assert blank is not None
    assert blank["horizontal_pull_detected"] is False
    assert remote_driver._cross_view_close_blocked(blank) is False
    assert remote_driver._toaster_cross_view_status(
        "PickPlaceCounterToDrawer", "right", {}, {"left": b"", "right": b""}
    ) is None


def test_cross_view_depth_mismatch_blocks_the_close_and_routes_a_depth_servo() -> None:
    from adaptive import remote_driver
    from adaptive.critic_protocol import validate_milestone_action_semantics

    ready = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [110.0, 120.0],
        "requested_target_role": "fixture_handle",
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "open",
        "gripper_residual": {"measured_end_finger_separation": 0.0798},
        "end_effector_external_pixel_displacement": {
            "right": {"end_px": [110.2, 117.2]},
            "left": {"end_px": [263.3, 106.3]},
        },
    }
    close = {
        "kind": "move_joints",
        "observation_id": "obs-cross-view",
        "targets": {"gripper": 0.0},
        "note": "milestone=engage; stationary close on the aligned pull",
    }
    depth_servo = {
        "kind": "image_servo",
        "observation_id": "obs-cross-view",
        "camera": "right",
        "target_pixel": [110.0, 120.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.03,
        "step_m": 0.03,
        "gripper": "open",
        "note": "milestone=engage; push depth along the right ray",
    }
    zero_depth_same_camera = {**depth_servo, "depth_delta_m": 0.0}
    other_camera_servo = {**zero_depth_same_camera, "camera": "left", "target_pixel": [153.0, 125.0]}
    history = ["observe", "approach", "engage"]

    validate_milestone_action_semantics(
        "articulated", "engage", close, task="OpenToasterOvenDoor",
        milestone_history=history, immediate_prior_receipt=ready, recent_receipts=[ready],
    )
    with pytest.raises(ValueError, match="other external view"):
        validate_milestone_action_semantics(
            "articulated", "engage", close, task="OpenToasterOvenDoor",
            milestone_history=history, immediate_prior_receipt=ready,
            recent_receipts=[ready], cross_view_depth_mismatch=True,
        )
    with pytest.raises(ValueError, match="other external view"):
        validate_milestone_action_semantics(
            "articulated", "engage", zero_depth_same_camera, task="OpenToasterOvenDoor",
            milestone_history=history, immediate_prior_receipt=ready,
            recent_receipts=[ready], cross_view_depth_mismatch=True,
        )
    for allowed in (depth_servo, other_camera_servo):
        validate_milestone_action_semantics(
            "articulated", "engage", allowed, task="OpenToasterOvenDoor",
            milestone_history=history, immediate_prior_receipt=ready,
            recent_receipts=[ready], cross_view_depth_mismatch=True,
        )

    record: dict[str, object] = {}
    advice = remote_driver._protocol_rejection(
        record,
        error=ValueError(
            "articulated insertion is off the handle in the other external view; "
            "author an open image_servo that corrects depth before any stationary close"
        ),
    )
    assert record["contradiction"] == "visual_alignment_unverified"
    assert record["suggested_correction"] == "revise_alignment"
    assert advice["evidence"] == ["external_rgb", "jacobian_projection"]
    protocol_message = (
        "articulated insertion is off the handle in the other external view; "
        "author an open image_servo that corrects depth before any stationary close"
    )

    revision = remote_driver._proposal_revision_instruction(
        "base",
        contradiction="visual_alignment_unverified",
        evidence=["external_rgb", "jacobian_projection"],
        suggested_correction="revise_alignment",
        confidence="high",
        rejected_draft=close,
        required_observation_id="obs-cross-view",
        immediate_prior_receipt=ready,
        recent_receipts=[ready],
        task="OpenToasterOvenDoor",
        protocol_message=protocol_message,
    ).casefold()
    assert "wrong depth along the selected camera ray" in revision
    assert "nonzero `depth_delta_m`" in revision
    assert '"protocol_rejection"' in revision

    cross_view = {
        "other_camera": "left",
        "grip_site_pixel": [263.3, 106.3],
        "grip_site_visible": False,
        "horizontal_pull_detected": True,
        "pull_box": {"u_min": 131, "u_max": 175, "v_min": 121, "v_max": 129},
        "grip_site_offset_px": [88.3, 14.7],
        "tolerance_px": 12,
        "grip_site_on_horizontal_pull": False,
    }
    controller_raw = remote_driver._controller_cross_view_instruction(
        "base", cross_view, selected_camera="right", close_blocked=True
    )
    controller = controller_raw.casefold()
    assert "toaster_cross_view_context" in controller
    assert "wrong depth along the right ray" in controller
    assert '"close_blocked":true' in controller
    with pytest.raises(remote_driver.ProposalAuditFailedClosed):
        remote_driver._controller_cross_view_instruction(
            controller_raw, cross_view, selected_camera="right", close_blocked=True
        )

    state = _state()
    state.update(_image_servo_public_state())
    payload = json.loads(
        remote_driver._proposal_critic_instruction(
            _context(),
            task_instruction="open the toaster oven door",
            observation_id="obs-cross-view",
            draft=depth_servo,
            claimed_milestone="engage",
            public_state=state,
            images=_images(),
            immediate_prior_receipt=ready,
            recent_receipts=[ready],
            cross_view=cross_view,
        )
    )
    assert payload["toaster_cross_view"] == cross_view
    assert payload["cross_view_depth_servo_rule"] == {
        "default_verdict": "approve",
        "effect_is_future_evidence": True,
        "gripper_must_remain_open": True,
        "grip_site_off_pull_in_other_view": True,
        "stationary_close_prohibited": True,
    }
    assert payload["stationary_handle_close_rule"] is None
    agreed = {**cross_view, "grip_site_on_horizontal_pull": True}
    payload = json.loads(
        remote_driver._proposal_critic_instruction(
            _context(),
            task_instruction="open the toaster oven door",
            observation_id="obs-cross-view",
            draft=close,
            claimed_milestone="engage",
            public_state=state,
            images=_images(),
            immediate_prior_receipt=ready,
            recent_receipts=[ready],
            cross_view=agreed,
        )
    )
    assert payload["cross_view_depth_servo_rule"] is None
    assert payload["stationary_handle_close_rule"] is not None


def test_toaster_gate_rejection_names_the_detected_pull_span() -> None:
    from adaptive import remote_driver

    image = _TwoViewImage(pull_u=(72, 115), pull_v=(121, 128))
    box = remote_driver._pull_box(remote_driver._horizontal_dark_pull_runs(image, [100, 100]))
    # Row 128 is a cyan grid line, so the bar's last dark row is 127; the
    # yellow line at u=96 no longer splits the run.
    assert box == {"u_min": 72, "u_max": 115, "v_min": 121, "v_max": 127}
    revision = remote_driver._proposal_revision_instruction(
        "base",
        contradiction="visual_alignment_unverified",
        evidence=["external_rgb"],
        suggested_correction="revise_alignment",
        confidence="high",
        rejected_draft={
            "kind": "image_servo",
            "camera": "right",
            "target_pixel": [128, 128],
            "target_role": "fixture_handle",
            "depth_delta_m": 0.0,
            "step_m": 0.03,
            "gripper": "open",
            "note": "milestone=approach; centre",
        },
        required_observation_id="obs-gate",
        task="OpenToasterOvenDoor",
        protocol_message=(
            "toaster oven target center is outside the visible narrow horizontal "
            "handle; the dark pull spans u=72..115, v=121..128 in the right view"
        ),
    ).casefold()
    assert "the dark pull spans u=72..115, v=121..128 in the right view" in revision
    assert "interior midline of that span" in revision


def test_engage_ready_schema_limits_the_decision_to_close_or_servo() -> None:
    from adaptive.joint_runner import controller_engage_ready_schema_for_observation

    with_close = controller_engage_ready_schema_for_observation("obs-x", allow_close=True)
    kinds = [branch["properties"]["kind"] for branch in with_close["oneOf"]]
    assert kinds == [{"const": "image_servo"}, {"const": "move_joints"}]
    close_targets = with_close["oneOf"][1]["properties"]["targets"]
    assert close_targets["properties"] == {"gripper": {"const": 0.0}}
    assert close_targets["required"] == ["gripper"]
    assert close_targets["maxProperties"] == 1
    assert with_close["oneOf"][0]["properties"]["observation_id"] == {"const": "obs-x"}
    assert with_close["oneOf"][1]["properties"]["observation_id"] == {"const": "obs-x"}
    servo_only = controller_engage_ready_schema_for_observation("obs-x", allow_close=False)
    assert [b["properties"]["kind"] for b in servo_only["oneOf"]] == [
        {"const": "image_servo"}
    ]


def test_orientation_recovery_uses_the_strict_wrist_roll_schema_on_first_proposal() -> None:
    from adaptive import remote_driver

    failure = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [110.0, 120.0],
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "close",
        "gripper_residual": {"measured_end_finger_separation": 0.00237},
    }
    retreat = {
        "kind": "cartesian_delta",
        "requested_gripper": "open",
        "gripper_residual": {"measured_end_finger_separation": 0.078},
    }
    wrist_roll = {
        "kind": "move_joints",
        "observation_id": "obs-schema",
        "targets": {"joint7": -0.86, "gripper": 1.0},
        "note": "milestone=engage; roll the open finger gap across the pull",
    }
    context = _context()
    context.milestone_history = ["observe", "approach", "engage"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    _FakeClient.controller_outputs = [wrist_roll]
    _FakeClient.critic_outputs = [_proposal_audit()]
    state = _state()
    state.update(_image_servo_public_state())
    response = _proposal_complete(
        wrapped,
        "obs-schema",
        state,
        _images(),
        receipts=[failure, retreat],
        camera_calibration=_image_servo_calibration(),
    )
    assert response.command is wrist_roll
    schema = _FakeClient.calls[0][1]["response_schema"]
    assert schema["properties"]["targets"]["required"] == ["gripper", "joint7"]


def test_thin_bar_detector_ignores_the_dark_appliance_body() -> None:
    from adaptive import remote_driver

    class BodyAndPull:
        size = (256, 256)

        @staticmethod
        def getpixel(point: tuple[int, int]) -> tuple[int, int, int]:
            x, y = point
            if x % 32 == 0:
                return (255, 255, 0)
            if y % 32 == 0:
                return (0, 255, 255)
            if 61 <= x <= 175 and 67 <= y <= 112:  # dark toaster body (tall)
                return (30, 30, 30)
            if 72 <= x <= 115 and 121 <= y <= 128:  # the pull (thin)
                return (20, 20, 20)
            return (120, 120, 120)

    image = BodyAndPull()
    runs = remote_driver._horizontal_dark_pull_runs(image, [128, 128], half_width=256, half_height=256)
    bars = remote_driver._thin_dark_bars(runs)
    assert bars == [{"u_min": 72, "u_max": 115, "v_min": 121, "v_max": 127}]
    assert remote_driver._pull_box(runs, near_pixel=[100, 100]) == bars[0]
    # A pixel inside the dark body is NOT on the pull.
    body = remote_driver._horizontal_dark_pull_target_status(image, [118.0, 108.0])
    assert body == {"horizontal_pull_detected": True, "target_on_horizontal_pull": False}
    pull = remote_driver._horizontal_dark_pull_target_status(image, [95.0, 124.0])
    assert pull == {"horizontal_pull_detected": True, "target_on_horizontal_pull": True}
    status = remote_driver._cross_view_status_from_image(
        image,
        other_camera="left",
        pixel_record={"u_px": 118.0, "v_px": 100.0, "visible": True, "depth_valid": True},
    )
    assert status is not None
    assert status["pull_box"] == bars[0]
    assert status["grip_site_on_horizontal_pull"] is False


def _stereo_calibration():
    left = {
        "image_width_px": 100,
        "image_height_px": 100,
        "fx_px": 100.0,
        "fy_px": 100.0,
        "cx_px": 50.0,
        "cy_px": 50.0,
        "camera_position_world_m": [0.0, 0.0, 0.0],
        "camera_xmat_world": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "projection": "mujoco-camera-x-right-y-up-minus-z-forward",
    }
    right = {**left, "camera_position_world_m": [0.5, 0.0, 0.0]}
    return {"left": left, "right": right}


def test_stereo_image_servo_triangulates_two_public_rays() -> None:
    from adaptive.image_servo import decode_image_servo, resolve_image_servo
    from adaptive.panda_embodiment import project_world_point

    calibration = _stereo_calibration()
    point = (0.1, 0.05, -1.0)
    left_px = project_world_point(point, calibration["left"])
    right_px = project_world_point(point, calibration["right"])
    assert (round(left_px["u_px"]), round(left_px["v_px"])) == (60, 45)
    assert (round(right_px["u_px"]), round(right_px["v_px"])) == (10, 45)
    payload = {
        **_image_servo_payload(),
        "target_pixel": [left_px["u_px"], left_px["v_px"]],
        "other_view_pixel": [right_px["u_px"], right_px["v_px"]],
        "target_role": "fixture_handle",
        "step_m": 0.03,
    }
    command = decode_image_servo(payload, observation_id="obs")
    assert command.other_view_pixel == (
        pytest.approx(left_px["u_px"] + 0.0),
        pytest.approx(45.0),
    ) or command.other_view_pixel[0] == pytest.approx(10.0)
    resolution = resolve_image_servo(
        command, _image_servo_public_state(), calibration, current_gripper=1.0
    )
    assert resolution.stereo_ray_gap_m == pytest.approx(0.0, abs=1e-9)
    assert resolution.target_depth_m == pytest.approx(1.0, abs=1e-9)
    direction = resolution.translation_m
    norm = math.sqrt(sum(v * v for v in direction))
    # 0.112 m to the point is below the 0.15 m stereo cap, so no scaling.
    assert norm == pytest.approx(math.hypot(0.1, 0.05), abs=1e-9)
    assert direction[0] / norm == pytest.approx(0.1 / math.hypot(0.1, 0.05), abs=1e-6)
    assert direction[1] / norm == pytest.approx(0.05 / math.hypot(0.1, 0.05), abs=1e-6)
    assert abs(direction[2]) < 1e-9

    skew = decode_image_servo(
        {**payload, "other_view_pixel": [right_px["u_px"], 80.0]},
        observation_id="obs",
    )
    with pytest.raises(ValueError, match="ray gap"):
        resolve_image_servo(
            skew, _image_servo_public_state(), calibration, current_gripper=1.0
        )
    single = decode_image_servo({**payload, "other_view_pixel": None}, observation_id="obs")
    assert single.other_view_pixel is None
    single_resolution = resolve_image_servo(
        single, _image_servo_public_state(), calibration, current_gripper=1.0
    )
    assert single_resolution.stereo_ray_gap_m is None
    assert decode_image_servo(_image_servo_payload(), observation_id="obs").other_view_pixel is None
    with pytest.raises(ValueError, match="fields drifted"):
        decode_image_servo({**payload, "bogus": 1}, observation_id="obs")


def test_stereo_pixel_is_optional_in_the_response_schema_and_gated_in_both_views() -> None:
    from adaptive import remote_driver
    from adaptive.joint_runner import IMAGE_SERVO_RESPONSE_SCHEMA

    assert "other_view_pixel" in IMAGE_SERVO_RESPONSE_SCHEMA["properties"]
    assert "other_view_pixel" not in IMAGE_SERVO_RESPONSE_SCHEMA["required"]

    right = _TwoViewImage(pull_u=(72, 115), pull_v=(121, 128))
    left = _TwoViewImage(pull_u=(131, 175), pull_v=(121, 129))
    draft = {
        "kind": "image_servo",
        "camera": "right",
        "target_pixel": [95.0, 124.0],
        "other_view_pixel": [153.0, 100.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.0,
        "step_m": 0.03,
        "gripper": "open",
        "note": "milestone=approach; stereo",
    }
    status = remote_driver._toaster_gate_from_images(draft, {"right": right, "left": left})
    assert status is not None
    assert status["target_on_horizontal_pull"] is True
    assert status["other_view"]["camera"] == "left"
    assert status["other_view"]["target_on_horizontal_pull"] is False
    assert status["other_view"]["pull_box"] == {"u_min": 131, "u_max": 175, "v_min": 121, "v_max": 129}
    good = remote_driver._toaster_gate_from_images(
        {**draft, "other_view_pixel": [153.0, 125.0]}, {"right": right, "left": left}
    )
    assert good is not None and good["other_view"]["target_on_horizontal_pull"] is True
    none = remote_driver._toaster_gate_from_images(
        {**draft, "other_view_pixel": None}, {"right": right, "left": left}
    )
    assert none is not None and none["other_view"] is None


def test_pull_candidates_are_public_facts_and_stereo_gap_routes_to_alignment() -> None:
    from adaptive import remote_driver

    image = _TwoViewImage(pull_u=(72, 115), pull_v=(121, 128))
    bars = remote_driver._thin_bars_in_image(image)
    assert bars == [{"u_min": 72, "u_max": 115, "v_min": 121, "v_max": 127}]
    text = remote_driver._controller_pull_candidates_instruction(
        "base", {"left": [], "right": bars}
    )
    assert "TOASTER_PULL_CANDIDATES" in text
    assert '"right":[{"u_max":115,"u_min":72,"v_max":127,"v_min":121}]' in text
    assert "not a chosen target" in text
    with pytest.raises(remote_driver.ProposalAuditFailedClosed):
        remote_driver._controller_pull_candidates_instruction(text, {"left": []})
    assert remote_driver._toaster_pull_candidates("RestockPantry", _images()) is None

    record: dict[str, object] = {}
    advice = remote_driver._protocol_rejection(
        record,
        error=ValueError(
            "image-servo stereo pixels do not name one point: ray gap 0.173 m "
            "exceeds 0.040 m"
        ),
    )
    assert record["contradiction"] == "visual_alignment_unverified"
    assert advice["suggested_correction"] == "revise_alignment"
    revision = remote_driver._proposal_revision_instruction(
        "base",
        contradiction="visual_alignment_unverified",
        evidence=["external_rgb", "jacobian_projection"],
        suggested_correction="revise_alignment",
        confidence="high",
        rejected_draft={
            "kind": "image_servo",
            "camera": "right",
            "target_pixel": [94.0, 118.0],
            "other_view_pixel": [217.0, 66.0],
            "target_role": "fixture_handle",
            "depth_delta_m": 0.01,
            "step_m": 0.03,
            "gripper": "open",
            "note": "milestone=approach; stereo",
        },
        required_observation_id="obs-gap",
        task="OpenToasterOvenDoor",
        protocol_message=(
            "image-servo stereo pixels do not name one point: ray gap 0.173 m "
            "exceeds 0.040 m"
        ),
    ).casefold()
    assert "do not name one physical point" in revision
    assert "set `other_view_pixel` to null" in revision
    assert "never use the published grip-site pixel" in revision


def test_stereo_image_servo_receipt_survives_every_public_receipt_validator() -> None:
    from adaptive.joint_runner import (
        PUBLIC_IMAGE_SERVO_RECEIPT_FIELDS,
        STEREO_RECEIPT_FIELDS,
        finalize_image_servo_receipt,
        prepare_joint_mailbox,
        validate_public_receipt,
    )
    from adaptive.panda_embodiment import project_world_point

    state, evidence, _unused_calibration = _stationary_receipt_boundary_inputs()
    state.update(_image_servo_public_state())
    state["state.arm_joint_position"] = _reset_qpos()
    evidence["end_effector_external_pixels_before"] = state[
        "state.end_effector_external_pixels"
    ]
    evidence["end_effector_external_pixels_after"] = state[
        "state.end_effector_external_pixels"
    ]
    calibration = _stereo_calibration()
    point = (0.1, 0.05, -1.0)
    left_px = project_world_point(point, calibration["left"])
    right_px = project_world_point(point, calibration["right"])
    payload = {
        **_image_servo_payload(),
        "target_pixel": [left_px["u_px"], left_px["v_px"]],
        "other_view_pixel": [right_px["u_px"], right_px["v_px"]],
        "target_role": "fixture_handle",
        "step_m": 0.03,
    }
    mailbox, command = prepare_joint_mailbox(
        payload,
        source="controller",
        observation_id="obs",
        current_qpos=_reset_qpos(),
        current_gripper=1.0,
        public_state=state,
        camera_calibration=calibration,
        remaining_actions=160,
        sequence=0,
    )
    assert command.other_view_pixel is not None
    execution = {
        "kind": "move_joints",
        "accepted": True,
        "bounded_endpoint": mailbox["endpoint"],
        "gripper_intent": 1.0,
        "step_count": 4,
        "maximum_commanded_step": 0.01,
        "endpoint_error": max(
            abs(target - actual)
            for target, actual in zip(mailbox["endpoint"], _reset_qpos(), strict=True)
        ),
        "minimum_hard_limit_margin": 0.5,
        "tracking_pause_count": 0,
        **evidence,
    }
    receipt = finalize_image_servo_receipt(
        command,
        mailbox,
        execution,
        before_state=state,
        after_state=state,
        before_calibration=calibration,
        after_calibration=calibration,
    )
    assert set(receipt) - STEREO_RECEIPT_FIELDS == PUBLIC_IMAGE_SERVO_RECEIPT_FIELDS
    assert receipt["requested_other_view_pixel"] == [
        pytest.approx(right_px["u_px"]),
        pytest.approx(right_px["v_px"]),
    ]
    assert receipt["stereo_ray_gap_m"] == pytest.approx(0.0, abs=1e-9)
    validated = validate_public_receipt(receipt)
    assert validated["requested_other_view_pixel"] == receipt["requested_other_view_pixel"]
    assert validated["stereo_ray_gap_m"] == pytest.approx(0.0, abs=1e-9)
    # A single-view receipt keeps the exact historical field set.
    plain_mailbox, plain_command = prepare_joint_mailbox(
        {**payload, "other_view_pixel": None},
        source="controller",
        observation_id="obs",
        current_qpos=_reset_qpos(),
        current_gripper=1.0,
        public_state=state,
        camera_calibration=calibration,
        remaining_actions=160,
        sequence=0,
    )
    plain = finalize_image_servo_receipt(
        plain_command,
        plain_mailbox,
        {**execution, "bounded_endpoint": plain_mailbox["endpoint"], "endpoint_error": max(abs(target-actual) for target,actual in zip(plain_mailbox["endpoint"], _reset_qpos(), strict=True))},
        before_state=state,
        after_state=state,
        before_calibration=calibration,
        after_calibration=calibration,
    )
    assert set(plain) == PUBLIC_IMAGE_SERVO_RECEIPT_FIELDS
    validate_public_receipt(plain)


def test_open_finger_obstruction_makes_an_insertion_ready() -> None:
    from adaptive.critic_protocol import (
        articulated_handle_insertion_is_ready,
        articulated_open_finger_obstruction,
    )

    receipt = {
        "kind": "image_servo",
        "requested_camera": "left",
        "requested_target_pixel": [148.0, 153.0],
        "requested_target_role": "fixture_handle",
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "open",
        "gripper_residual": {"measured_end_finger_separation": 0.0678},
        "end_effector_external_pixel_displacement": {
            "left": {"end_px": [156.0, 124.0]},
            "right": {"end_px": [93.0, 123.0]},
        },
    }
    assert articulated_open_finger_obstruction(receipt) is True
    assert articulated_handle_insertion_is_ready(receipt) is True
    fully_open = {
        **receipt,
        "gripper_residual": {"measured_end_finger_separation": 0.0799},
    }
    assert articulated_open_finger_obstruction(fully_open) is False
    assert articulated_handle_insertion_is_ready(fully_open) is False


def test_stalled_approach_routes_to_reorientation_instead_of_repeated_servo() -> None:
    from adaptive import remote_driver
    from adaptive.critic_protocol import (
        articulated_approach_stall_status,
        validate_milestone_action_semantics,
    )

    def servo(dx: float) -> dict[str, object]:
        return {
            "kind": "image_servo",
            "requested_camera": "right",
            "requested_target_pixel": [94.0, 118.0],
            "requested_target_role": "fixture_handle",
            "requested_depth_delta_m": 0.03,
            "requested_gripper": "open",
            "gripper_residual": {"measured_end_finger_separation": 0.0797},
            "end_effector_pose_delta": {"translation_m": [dx, 0.002, 0.0]},
            "end_effector_external_pixel_displacement": {
                "right": {"end_px": [90.0, 100.0]},
                "left": {"end_px": [151.0, 101.0]},
            },
        }

    moving = [servo(0.025), servo(0.015)]
    stalled = [servo(0.025), servo(0.004), servo(0.003)]
    assert articulated_approach_stall_status(moving) is None
    stall = articulated_approach_stall_status(stalled)
    assert stall == {
        "stalled": True,
        "camera": "right",
        "target_pixel": [94.0, 118.0],
        "realized_translation_m": [pytest.approx(0.0045, abs=1e-3), pytest.approx(0.0036, abs=1e-3)],
        "pixel_error_px": pytest.approx(18.4, abs=0.1),
        "min_progress_m": 0.006,
    }
    history = ["observe", "approach", "engage"]
    repeat = {
        "kind": "image_servo",
        "observation_id": "obs-stall",
        "camera": "right",
        "target_pixel": [94.0, 118.0],
        "other_view_pixel": [147.0, 119.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.03,
        "step_m": 0.01,
        "gripper": "open",
        "note": "milestone=engage; try the same servo again",
    }
    reorient = {
        "kind": "move_joints",
        "observation_id": "obs-stall",
        "targets": {"joint2": -0.6, "joint4": -1.9, "joint6": 2.9, "joint7": 1.3, "gripper": 1.0},
        "note": "milestone=engage; point the tool forward with a vertical gap",
    }
    retreat = {
        "kind": "cartesian_delta",
        "observation_id": "obs-stall",
        "translation_m": [-0.03, 0.0, 0.02],
        "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        "gripper": "open",
        "note": "milestone=engage; back and up before a front approach",
    }
    stationary_close = {
        "kind": "move_joints",
        "observation_id": "obs-stall",
        "targets": {"gripper": 0.0},
        "note": "milestone=engage; probe the door top edge already between the fingers",
    }
    with pytest.raises(ValueError, match="approach stalled"):
        validate_milestone_action_semantics(
            "articulated", "engage", repeat, task="OpenToasterOvenDoor",
            milestone_history=history, immediate_prior_receipt=stalled[-1],
            recent_receipts=stalled, approach_stall=stall,
        )
    for allowed in (reorient, retreat, stationary_close):
        validate_milestone_action_semantics(
            "articulated", "engage", allowed, task="OpenToasterOvenDoor",
            milestone_history=history, immediate_prior_receipt=stalled[-1],
            recent_receipts=stalled, approach_stall=stall,
        )
    with pytest.raises(ValueError, match="open insertion"):
        validate_milestone_action_semantics(
            "articulated", "engage", stationary_close, task="OpenCabinetDoor",
            milestone_history=history, immediate_prior_receipt=stalled[-1],
            recent_receipts=stalled, approach_stall=stall,
        )
    with pytest.raises(ValueError, match="open insertion"):
        validate_milestone_action_semantics(
            "articulated", "engage", reorient, task="OpenToasterOvenDoor",
            milestone_history=history, immediate_prior_receipt=moving[-1],
            recent_receipts=moving,
        )
    record: dict[str, object] = {}
    advice = remote_driver._protocol_rejection(
        record, error=ValueError("approach stalled: the grip site no longer moves")
    )
    assert record["contradiction"] == "stagnation"
    assert advice["suggested_correction"] == "revise_approach"
    text = remote_driver._controller_approach_stall_instruction(
        "base", stall, task="OpenToasterOvenDoor",
        immediate_prior_receipt=stalled[-1],
    ).casefold()
    assert "approach_stall_context" in text
    assert "physically blocked" in text
    assert "door top edge" in text
    assert "stationary close now" in text
    assert "do not re-orient" in text
    assert "downward image waypoint" in text
    state = _state()
    state.update(_image_servo_public_state())
    payload = json.loads(
        remote_driver._proposal_critic_instruction(
            _context(),
            task_instruction="open the toaster oven door",
            observation_id="obs-stall",
            draft=stationary_close,
            claimed_milestone="engage",
            public_state=state,
            images=_images(),
            immediate_prior_receipt=stalled[-1],
            recent_receipts=stalled,
            approach_stall=stall,
        )
    )
    assert payload["approach_stall"] == stall
    assert payload["approach_stall_rule"]["blocked_approach_needs_new_geometry"] is False
    assert payload["approach_stall_rule"]["toaster_stall_close_probe_allowed"] is True
    assert payload["approach_stall_rule"]["stationary_close_required"] is True
    assert payload["approach_stall_rule"]["contact_effect_is_future_evidence"] is True


def test_toaster_stall_close_probe_is_explicitly_bounded_in_both_prompts() -> None:
    root = Path(__file__).resolve().parents[1]
    controller = " ".join(
        load_joint_system_prompt(root, variant="rig").casefold().split()
    )
    critic = " ".join(
        (root / "prompts" / "proposal_audit_critic.txt")
        .read_text(encoding="utf-8")
        .casefold()
        .split()
    )

    assert "door top edge" in controller
    assert "only after" in controller and "front-pull approach" in controller
    assert "stationary close now" in controller
    assert "do not re-orient" in controller
    assert "downward image waypoint" in controller
    assert "toaster_stall_close_probe_allowed=true" in critic
    assert "approve the stationary close probe" in critic
    assert "future sealed receipt" in critic


def test_toaster_stall_offers_and_approves_the_cheap_stationary_close() -> None:
    from adaptive import remote_driver
    from adaptive.joint_runner import (
        controller_engage_ready_schema_for_observation,
    )

    def stalled_servo(dx: float) -> dict[str, object]:
        return {
            "kind": "image_servo",
            "requested_camera": "right",
            "requested_target_pixel": [94.0, 118.0],
            "requested_target_role": "fixture_handle",
            "requested_depth_delta_m": 0.01,
            "requested_gripper": "open",
            "gripper_residual": {"measured_end_finger_separation": 0.0797},
            "end_effector_pose_delta": {"translation_m": [dx, 0.0, 0.0]},
            "end_effector_external_pixel_displacement": {
                "right": {"end_px": [94.0, 107.0]},
                "left": {"end_px": [149.0, 108.0]},
            },
        }

    observation_id = "obs-toaster-stall-close"
    close = {
        "kind": "move_joints",
        "observation_id": observation_id,
        "targets": {"gripper": 0.0},
        "note": "milestone=engage; test the door top edge with no arm motion",
    }
    context = _context()
    context.milestone_history = ["observe", "approach", "engage"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    _FakeClient.controller_outputs = [close]
    _FakeClient.critic_outputs = [_proposal_audit()]
    state = _state()
    state.update(_image_servo_public_state())

    response = _proposal_complete(
        wrapped,
        observation_id,
        state,
        _images(),
        receipts=[stalled_servo(0.004), stalled_servo(0.003)],
    )

    assert response.command is close
    controller_call = _FakeClient.calls[0][1]
    assert controller_call["response_schema"] == (
        controller_engage_ready_schema_for_observation(
            observation_id, allow_close=True
        )
    )
    assert "stationary close now" in str(controller_call["instruction"]).casefold()
    critic_payload = json.loads(str(_FakeClient.calls[1][1]["instruction"]))
    assert critic_payload["approach_stall_rule"][
        "toaster_stall_close_probe_allowed"
    ] is True
    assert context.proposal_records[-1]["status"] == "approved_for_execution"


def test_stereo_mismatch_names_the_bar_under_each_pixel_and_forbids_repeats() -> None:
    from adaptive import remote_driver

    candidates = {
        "right": [
            {"u_min": 60, "u_max": 128, "v_min": 114, "v_max": 123},
            {"u_min": 70, "u_max": 129, "v_min": 150, "v_max": 154},
        ],
        "left": [
            {"u_min": 109, "u_max": 186, "v_min": 115, "v_max": 123},
            {"u_min": 117, "u_max": 189, "v_min": 150, "v_max": 155},
        ],
    }
    draft = {
        "kind": "image_servo",
        "camera": "right",
        "target_pixel": [106.0, 153.0],
        "other_view_pixel": [130.0, 121.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.0,
        "step_m": 0.03,
        "gripper": "open",
        "note": "milestone=approach; stereo",
    }
    diagnosis = remote_driver._stereo_bar_diagnosis(draft, candidates)
    assert "the right pixel [106.0, 153.0] is on bar u=70..129, v=150..154" in diagnosis
    assert "the left pixel [130.0, 121.0] is on bar u=109..186, v=115..123" in diagnosis
    assert "DIFFERENT bars" in diagnosis
    same = remote_driver._stereo_bar_diagnosis(
        {**draft, "target_pixel": [94.0, 118.0]}, candidates
    )
    assert "same height" in same
    assert remote_driver._stereo_bar_diagnosis(draft, None) == ""

    revision = remote_driver._proposal_revision_instruction(
        "base",
        contradiction="visual_alignment_unverified",
        evidence=["external_rgb", "jacobian_projection"],
        suggested_correction="revise_alignment",
        confidence="high",
        rejected_draft=draft,
        required_observation_id="obs-hist",
        task="OpenToasterOvenDoor",
        protocol_message="image-servo stereo pixels do not name one point: ray gap 0.17 m exceeds 0.040 m" + diagnosis,
        rejected_history=[
            {"kind": "image_servo", "camera": "right", "target_pixel": [106.0, 153.0], "other_view_pixel": [130.0, 121.0], "targets": None, "contradiction": "visual_alignment_unverified"},
        ],
    )
    assert '"rejected_this_observation"' in revision
    assert '"revision_number":1' in revision
    assert "unchanged complete draft with the same critic input" in revision
    assert "DIFFERENT bars" in revision


def test_toaster_gate_rejects_end_cap_pixels_with_the_interior_range() -> None:
    from adaptive import remote_driver

    right = _TwoViewImage(pull_u=(60, 128), pull_v=(114, 123))
    base = {
        "kind": "image_servo",
        "camera": "right",
        "target_role": "fixture_handle",
        "depth_delta_m": 0.01,
        "step_m": 0.03,
        "gripper": "open",
        "note": "milestone=approach; stereo",
    }
    end = {**base, "target_pixel": [128.0, 118.0]}
    status = remote_driver._toaster_gate_from_images(end, {"right": right})
    assert status is not None and status["target_on_horizontal_pull"] is True
    message = remote_driver._toaster_end_cap_violation(status, end)
    assert message is not None
    assert "end cap of the pull" in message
    assert "u between 68 and 120 in the right view" in message
    mid = {**base, "target_pixel": [94.0, 118.0]}
    assert remote_driver._toaster_end_cap_violation(
        remote_driver._toaster_gate_from_images(mid, {"right": right}), mid
    ) is None
    closing = {**end, "gripper": "close"}
    assert remote_driver._toaster_end_cap_violation(status, closing) is None
    record: dict[str, object] = {}
    remote_driver._protocol_rejection(record, error=ValueError(message))
    assert record["contradiction"] == "visual_alignment_unverified"
    revision = remote_driver._proposal_revision_instruction(
        "base",
        contradiction="visual_alignment_unverified",
        evidence=["external_rgb"],
        suggested_correction="revise_alignment",
        confidence="high",
        rejected_draft=end,
        required_observation_id="obs-end",
        task="OpenToasterOvenDoor",
        protocol_message=message,
    ).casefold()
    assert "near the bar's midpoint" in revision


def test_stereo_required_schema_forbids_single_view_drift_near_the_handle() -> None:
    from adaptive import remote_driver
    from adaptive.joint_runner import controller_engage_ready_schema_for_observation

    schema = controller_engage_ready_schema_for_observation(
        "obs-s", allow_close=False, require_stereo=True
    )
    servo = schema["oneOf"][0]
    assert "other_view_pixel" in servo["required"]
    assert servo["properties"]["other_view_pixel"]["type"] == "array"
    plain = controller_engage_ready_schema_for_observation("obs-s", allow_close=False)
    assert "other_view_pixel" not in plain["oneOf"][0]["required"]
    near = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [98.0, 152.0],
        "requested_target_role": "fixture_handle",
        "requested_gripper": "open",
        "end_effector_external_pixel_displacement": {"right": {"end_px": [100.0, 140.0]}},
    }
    assert remote_driver._handle_servo_near_but_unaligned(near) is True
    far = {**near, "end_effector_external_pixel_displacement": {"right": {"end_px": [30.0, 40.0]}}}
    assert remote_driver._handle_servo_near_but_unaligned(far) is False
    aligned = {**near, "end_effector_external_pixel_displacement": {"right": {"end_px": [99.0, 150.0]}}}
    assert remote_driver._handle_servo_near_but_unaligned(aligned) is False


def test_standing_stereo_target_is_projected_and_reissues_bypass_the_gate() -> None:
    from adaptive import remote_driver

    calibration = _stereo_calibration()
    state = _image_servo_public_state()
    receipts = [
        {"kind": "cartesian_delta", "requested_gripper": "open"},
        {
            "kind": "image_servo",
            "requested_camera": "left",
            "requested_target_pixel": [60.0, 45.0],
            "requested_other_view_pixel": [10.0, 45.0],
            "requested_target_role": "fixture_handle",
            "requested_gripper": "open",
            "stereo_target_base_m": [0.1, 0.05, -1.0],
        },
        {"kind": "move_joints", "requested_targets": {"gripper": 1.0}},
    ]
    standing = remote_driver._standing_stereo_target(receipts, state, calibration)
    assert standing == {
        "base_m": [0.1, 0.05, -1.0],
        "left_px": [60.0, 45.0],
        "right_px": [10.0, 45.0],
        "distance_to_grip_site_m": pytest.approx(math.hypot(0.1, 0.05), abs=1e-3),
        "source_receipt_index": 1,
        "match_tolerance_px": 12.0,
    }
    assert remote_driver._standing_stereo_target(receipts[:1], state, calibration) is None
    reissue = {
        "kind": "image_servo",
        "camera": "right",
        "target_pixel": [14.0, 48.0],
        "other_view_pixel": [58.0, 44.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.02,
        "step_m": 0.03,
        "gripper": "open",
        "note": "milestone=engage; re-issue the standing target",
    }
    assert remote_driver._draft_matches_standing_target(reissue, standing) is True
    assert remote_driver._draft_matches_standing_target(
        {**reissue, "other_view_pixel": [90.0, 44.0]}, standing
    ) is False
    assert remote_driver._draft_matches_standing_target(
        {**reissue, "other_view_pixel": None}, standing
    ) is False
    text = remote_driver._controller_standing_target_instruction("base", standing)
    assert "STANDING_STEREO_TARGET" in text
    assert "exempt from the pull gate" in text
    payload = json.loads(
        remote_driver._proposal_critic_instruction(
            _context(),
            task_instruction="open the toaster oven door",
            observation_id="obs-standing",
            draft=reissue,
            claimed_milestone="approach",
            public_state=_state(),
            images=_images(),
            immediate_prior_receipt=receipts[-1],
            recent_receipts=receipts,
            standing_target=standing,
        )
    )
    assert payload["standing_stereo_target"] == standing
    assert payload["standing_target_reissue"] is True


def test_coffee_standing_target_survives_occluded_candidate_detection() -> None:
    from adaptive import remote_driver

    calibration = _stereo_calibration()
    state = _image_servo_public_state()
    receipts = [
        {
            "kind": "image_servo",
            "requested_camera": "left",
            "requested_target_pixel": [60.0, 45.0],
            "requested_other_view_pixel": [10.0, 45.0],
            "requested_target_role": "control_target",
            "requested_gripper": "open",
            "stereo_target_base_m": [0.1, 0.05, -1.0],
        }
    ]
    standing = remote_driver._task_standing_stereo_target(
        "StartCoffeeMachine", "control", receipts, state, calibration
    )
    assert standing is not None
    assert standing["match_tolerance_px"] == 4.0
    assert (
        remote_driver._task_standing_stereo_target(
            "AdjustWaterTemperature", "control", receipts, state, calibration
        )
        is None
    )
    assert remote_driver._task_standing_stereo_target(
        "OpenToasterOvenDoor", "articulated", receipts, state, calibration
    )["match_tolerance_px"] == 12.0

    occluded_candidates = {
        "left": [
            {"u_min": 143, "u_max": 148, "v_min": 112, "v_max": 112},
            {"u_min": 192, "u_max": 198, "v_min": 119, "v_max": 120},
            {"u_min": 143, "u_max": 147, "v_min": 121, "v_max": 122},
        ],
        "right": [
            {"u_min": 108, "u_max": 113, "v_min": 112, "v_max": 112},
            {"u_min": 109, "u_max": 113, "v_min": 122, "v_max": 122},
        ],
    }
    reissue = {
        **_image_servo_payload(),
        "camera": "left",
        "target_pixel": [172.0, 109.0],
        "other_view_pixel": [110.0, 111.0],
        "target_role": "control_target",
    }
    coffee_standing = {
        "left_px": [172.0, 109.0],
        "right_px": [110.0, 111.0],
        "match_tolerance_px": 4.0,
    }
    assert remote_driver._coffee_button_target_violation(
        "StartCoffeeMachine", reissue, occluded_candidates
    ) is not None
    assert (
        remote_driver._coffee_button_target_violation(
            "StartCoffeeMachine",
            reissue,
            occluded_candidates,
            standing_target=coffee_standing,
        )
        is None
    )
    assert remote_driver._draft_matches_standing_target(
        {**reissue, "target_pixel": [175.0, 109.0]}, coffee_standing
    ) is True
    assert remote_driver._draft_matches_standing_target(
        {**reissue, "target_pixel": [177.0, 109.0]}, coffee_standing
    ) is False
    assert (
        remote_driver._coffee_button_target_violation(
            "StartCoffeeMachine",
            {**reissue, "target_pixel": [177.0, 109.0]},
            occluded_candidates,
            standing_target=coffee_standing,
        )
        is not None
    )
    instruction = remote_driver._controller_standing_target_instruction(
        "base", coffee_standing, task="StartCoffeeMachine"
    )
    assert "coffee-button candidate gate" in instruction
    critic = (
        Path(__file__).resolve().parents[1] / "prompts" / "proposal_audit_critic.txt"
    ).read_text()
    assert "standing_target_reissue=true" in critic
    assert "selected surface is temporarily occluded" in " ".join(critic.split())


def test_image_servo_halves_the_step_when_the_joint_delta_bound_would_reject() -> None:
    from adaptive import image_servo
    from adaptive.image_servo import decode_image_servo, resolve_image_servo

    calibration = _stereo_calibration()
    payload = {
        **_image_servo_payload(),
        "target_pixel": [60.0, 45.0],
        "other_view_pixel": [10.0, 45.0],
        "target_role": "fixture_handle",
        "step_m": 0.03,
    }
    command = decode_image_servo(payload, observation_id="obs")
    calls: list[float] = []
    original = image_servo.resolve_cartesian_delta

    def strict(cartesian, *args, **kwargs):  # type: ignore[no-untyped-def]
        norm = math.sqrt(sum(v * v for v in cartesian.translation_m))
        calls.append(norm)
        if norm > 0.05:
            raise ValueError("derived Cartesian joint delta exceeds its bound")
        return original(cartesian, *args, **kwargs)

    image_servo.resolve_cartesian_delta = strict  # type: ignore[assignment]
    try:
        resolution = resolve_image_servo(
            command, _image_servo_public_state(), calibration, current_gripper=1.0
        )
    finally:
        image_servo.resolve_cartesian_delta = original  # type: ignore[assignment]
    assert len(calls) == 3
    assert calls[0] == pytest.approx(math.hypot(0.1, 0.05), abs=1e-9)
    assert calls[-1] == pytest.approx(math.hypot(0.1, 0.05) / 4, abs=1e-9)
    norm = math.sqrt(sum(v * v for v in resolution.translation_m))
    assert norm == pytest.approx(math.hypot(0.1, 0.05) / 4, abs=1e-9)


def test_reach_limit_rejection_explains_the_joint_and_offers_the_base() -> None:
    from adaptive import remote_driver
    from adaptive.joint_runner import controller_engage_ready_schema_for_observation

    state = _state()
    state["state.arm_joint_position"] = [0.026, 0.001, -0.157, -0.252, -0.012, 0.279, 0.177]
    text = remote_driver._reach_limit_diagnosis(
        "derived Cartesian joint4 endpoint is unsafe",
        state,
        {"distance_to_grip_site_m": 0.62},
    )
    assert "joint4 is at -0.252 rad" in text
    assert "beyond arm reach" in text
    assert "0.62 m from the grip site" in text
    assert remote_driver._reach_limit_diagnosis("something else", state, None) == ""
    revision = remote_driver._proposal_revision_instruction(
        "base",
        contradiction="safety_bound_risk",
        evidence=["joint_tracking"],
        suggested_correction="revise_within_bounds",
        confidence="high",
        rejected_draft={"kind": "image_servo", "camera": "right", "target_pixel": [117.0, 120.0], "other_view_pixel": [147.0, 119.0], "target_role": "fixture_handle", "depth_delta_m": 0.0, "step_m": 0.03, "gripper": "open", "note": "milestone=approach; x"},
        required_observation_id="obs-reach",
        task="OpenToasterOvenDoor",
        protocol_message="derived Cartesian joint4 endpoint is unsafe" + text,
    ).casefold()
    assert "author a bounded `base_action`" in revision
    schema = controller_engage_ready_schema_for_observation(
        "obs-reach", allow_close=False, require_stereo=True, allow_base=True
    )
    kinds = [b["properties"]["kind"] for b in schema["oneOf"]]
    assert kinds == [{"const": "image_servo"}, {"const": "base_action"}]


def test_base_pulse_switches_the_composite_controller_into_base_mode() -> None:
    from adaptive.joint_sim_child import joint_unmap_action

    arm = joint_unmap_action({
        "joint_position": _reset_qpos(),
        "gripper_open": 1.0,
        "base_motion": [0.0, 0.0, 0.0],
        "torso": 0.0,
    })
    assert arm["robot0_base_mode"] == pytest.approx(-1.0)
    base = joint_unmap_action({
        "joint_position": _reset_qpos(),
        "gripper_open": 1.0,
        "base_motion": [0.25, 0.0, 0.0],
        "torso": 0.0,
    })
    assert base["robot0_base_mode"] == pytest.approx(1.0)
    assert base["robot0_base"] == [0.25, 0.0, 0.0]


def test_planar_ik_table_matches_installed_fk_and_flags_limits() -> None:
    from adaptive import remote_driver
    from adaptive.panda_embodiment import panda_fk

    qpos = [-0.048, -0.643, 0.115, -2.202, 0.011, 2.053, 0.387]
    position = panda_fk(qpos).position_m
    state = _state()
    state["state.arm_joint_position"] = qpos
    state["state.end_effector_position_relative"] = list(position)
    table = remote_driver._planar_ik_table(state)
    assert table is not None
    assert table["radius_m"] == pytest.approx(math.hypot(position[0], position[1]), abs=1e-3)
    assert table["current_tool_pitch_rad"] == pytest.approx(-0.643 + 2.202 - 2.053, abs=1e-3)
    rows = table["candidates"]
    assert rows and all(set(r) >= {"tool_pitch_rad", "joint2", "joint4", "joint6", "within_limits"} for r in rows)
    legal = [r for r in rows if r["within_limits"]]
    assert legal
    for row in legal:
        q = [qpos[0], row["joint2"], 0.0, row["joint4"], 0.0, row["joint6"], qpos[6]]
        p = panda_fk(q).position_m
        assert math.hypot(p[0], p[1]) == pytest.approx(table["radius_m"], abs=0.01)
        assert p[2] == pytest.approx(table["height_m"], abs=0.01)
        assert row["joint2"] - row["joint4"] - row["joint6"] == pytest.approx(row["tool_pitch_rad"], abs=2e-3)
    text = remote_driver._controller_approach_stall_instruction(
        "base", {"stalled": True, "camera": "right", "target_pixel": [94.0, 118.0]}, ik_table=table
    )
    assert "planar_ik_table" in text and "within_limits" in text


def test_sweep_fixes_family_override_ready_pose_and_control_press() -> None:
    from adaptive import remote_driver

    assert remote_driver.effective_family("PickPlaceCounterToDrawer", "articulated") == "grasp_place"
    assert remote_driver.effective_family("ToastOnCorrectRack", "articulated") == "grasp_place"
    assert remote_driver.effective_family("OpenToasterOvenDoor", "articulated") == "articulated"
    assert remote_driver.effective_family("RestockPantry", "grasp_place") == "grasp_place"

    state = _state()
    state["state.arm_joint_position"] = [0.0, -0.04, 0.0, -0.15, 0.0, 0.25, 0.22]
    ready = {
        "kind": "move_joints",
        "observation_id": "obs-ready",
        "targets": {"joint2": -0.5, "joint4": -2.0, "joint6": 2.0, "gripper": 1.0},
        "note": "milestone=observe; bend the straight arm into a ready pose",
    }
    payload = json.loads(
        remote_driver._proposal_critic_instruction(
            _context(), task_instruction="open the toaster oven door", observation_id="obs-ready",
            draft=ready, claimed_milestone="observe", public_state=state, images=_images(),
            immediate_prior_receipt=None, recent_receipts=[],
        )
    )
    assert payload["ready_pose_rule"]["arm_nearly_straight"] is True
    bent = dict(state)
    bent["state.arm_joint_position"] = [0.0, -0.5, 0.0, -2.0, 0.0, 2.0, 0.22]
    payload = json.loads(
        remote_driver._proposal_critic_instruction(
            _context(), task_instruction="x", observation_id="obs-ready", draft=ready,
            claimed_milestone="observe", public_state=bent, images=_images(),
            immediate_prior_receipt=None, recent_receipts=[],
        )
    )
    assert payload["ready_pose_rule"] is None

    control = dataclasses.replace(_context(), task="StartCoffeeMachine", family="control")
    close_receipt = {
        "kind": "image_servo",
        "requested_gripper": "close",
        "requested_target_role": "control_target",
        "gripper_residual": {"measured_end_finger_separation": 0.0024},
    }
    press = {
        "kind": "cartesian_delta",
        "observation_id": "obs-press",
        "translation_m": [0.02, 0.0, 0.0],
        "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        "gripper": "close",
        "note": "milestone=actuate; push the start button",
    }
    payload = json.loads(
        remote_driver._proposal_critic_instruction(
            control, task_instruction="start the coffee machine", observation_id="obs-press",
            draft=press, claimed_milestone="actuate", public_state=state, images=_images(),
            immediate_prior_receipt=close_receipt, recent_receipts=[close_receipt],
        )
    )
    assert payload["control_press_rule"]["button_is_pressed_not_grasped"] is True
    ray_press = {
        **_image_servo_payload(),
        "observation_id": "obs-ray-press",
        "target_role": "control_target",
        "depth_delta_m": 0.02,
        "gripper": "close",
        "note": "milestone=actuate; press the start button along its camera ray",
    }
    payload = json.loads(
        remote_driver._proposal_critic_instruction(
            control,
            task_instruction="start the coffee machine",
            observation_id="obs-ray-press",
            draft=ray_press,
            claimed_milestone="actuate",
            public_state=state,
            images=_images(),
            immediate_prior_receipt=close_receipt,
            recent_receipts=[close_receipt],
        )
    )
    assert payload["control_press_rule"]["camera_ray_press"] is True
    payload = json.loads(
        remote_driver._proposal_critic_instruction(
            _context(), task_instruction="open the toaster oven door", observation_id="obs-press",
            draft=press, claimed_milestone="actuate", public_state=state, images=_images(),
            immediate_prior_receipt=close_receipt, recent_receipts=[close_receipt],
        )
    )
    assert payload["control_press_rule"] is None
    source_servo = {
        "kind": "image_servo",
        "requested_camera": "left",
        "requested_target_pixel": [100.0, 130.0],
        "requested_target_role": "source_object",
        "requested_gripper": "open",
        "end_effector_external_pixel_displacement": {"left": {"end_px": [102.0, 112.0]}},
    }
    assert remote_driver._handle_servo_near_but_unaligned(source_servo) is True


def test_coffee_controller_grounding_compensates_finger_side_contact() -> None:
    from adaptive import remote_driver

    grounded = remote_driver._controller_task_grounding_instruction(
        "BASE", task="StartCoffeeMachine"
    )
    assert grounded.startswith("BASE\n\nCONTROL_TARGET_GROUNDING_CONTEXT:\n")
    assert "small 2x2 array" in grounded
    assert "calibrated upper-right contact point" in grounded
    assert "u exactly u_max" in grounded
    assert "left v one pixel above v_min" in grounded
    assert "right v two pixels above v_min" in grounded
    assert "finger-side contact" in grounded
    assert "midpoint between buttons" in grounded
    assert "same row-and-column button" in grounded
    assert "Qwen chooses the camera and exact pixels" in grounded
    assert (
        remote_driver._controller_task_grounding_instruction(
            "BASE", task="AdjustWaterTemperature"
        )
        == "BASE"
    )


def test_coffee_button_candidates_gate_cluster_midpoint_from_public_rgb() -> None:
    from adaptive import remote_driver

    class CoffeeButtonImage:
        size = (256, 256)
        buttons = (
            (82, 87, 108, 110),
            (108, 113, 110, 112),
            (83, 88, 119, 121),
            (109, 114, 121, 123),
        )

        @classmethod
        def getpixel(cls, point: tuple[int, int]) -> tuple[int, int, int]:
            x, y = point
            if x % 32 == 0:
                return (255, 255, 0)
            if y % 32 == 0:
                return (0, 255, 255)
            if any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in cls.buttons):
                return (45, 45, 45)
            return (155, 155, 155)

    boxes = remote_driver._coffee_button_boxes(CoffeeButtonImage())
    assert boxes == [
        {"u_min": 82, "u_max": 87, "v_min": 108, "v_max": 110},
        {"u_min": 108, "u_max": 113, "v_min": 110, "v_max": 112},
        {"u_min": 83, "u_max": 88, "v_min": 119, "v_max": 121},
        {"u_min": 109, "u_max": 114, "v_min": 121, "v_max": 123},
    ]
    candidates = {"left": [], "right": boxes}
    bad = {
        **_image_servo_payload(),
        "camera": "right",
        "target_pixel": [100.0, 100.0],
        "target_role": "control_target",
    }
    violation = remote_driver._coffee_button_target_violation(
        "StartCoffeeMachine", bad, candidates
    )
    assert violation is not None
    assert "not on one distinct visible button" in violation
    assert "u=82..87, v=108..110" in violation
    center_violation = remote_driver._coffee_button_target_violation(
        "StartCoffeeMachine",
        {**bad, "target_pixel": [85.0, 109.0]},
        candidates,
    )
    assert center_violation is not None
    assert "calibrated contact point" in center_violation
    assert "u = 87.0" in center_violation
    assert "v = 106.0" in center_violation
    edge_drift = remote_driver._coffee_button_target_violation(
        "StartCoffeeMachine",
        {**bad, "target_pixel": [88.0, 106.0]},
        candidates,
    )
    assert edge_drift is not None
    assert "calibrated contact point" in edge_drift
    assert remote_driver._coffee_button_target_violation(
        "StartCoffeeMachine",
        {**bad, "target_pixel": [87.0, 106.0]},
        candidates,
    ) is None
    assert remote_driver._coffee_button_target_violation("RestockPantry", bad, candidates) is None
    assert (
        remote_driver._coffee_button_target_violation(
            "StartCoffeeMachine",
            {**bad, "depth_delta_m": -0.03},
            candidates,
        )
        is None
    )

    instruction = remote_driver._controller_coffee_button_candidates_instruction(
        "BASE", candidates
    )
    assert "COFFEE_BUTTON_CANDIDATES" in instruction
    assert "fresh public" in instruction
    assert "Qwen chooses one box" in instruction
    assert "u = u_max" in instruction
    assert "left v = v_min - 1" in instruction
    assert "right v = v_min - 2" in instruction

    record: dict[str, object] = {}
    advice = remote_driver._protocol_rejection(
        record, error=ValueError(violation)
    )
    assert record["contradiction"] == "visual_alignment_unverified"
    revision = remote_driver._proposal_revision_instruction(
        "BASE",
        contradiction=str(advice["contradiction"]),
        evidence=list(advice["evidence"]),
        suggested_correction=str(advice["suggested_correction"]),
        confidence=str(advice["confidence"]),
        rejected_draft=bad,
        required_observation_id="obs-coffee-button",
        task="StartCoffeeMachine",
        protocol_message=violation,
    )
    assert "COFFEE_BUTTON_CANDIDATES" in revision
    assert "do not repeat the cluster midpoint" in revision
    assert "calibrated contact point" in revision


def test_coffee_button_gate_rejects_crossed_stereo_grid_cells() -> None:
    from adaptive import remote_driver

    candidates = {
        "left": [
            {"u_min": 170, "u_max": 174, "v_min": 109, "v_max": 110},
            {"u_min": 143, "u_max": 148, "v_min": 112, "v_max": 112},
            {"u_min": 143, "u_max": 147, "v_min": 121, "v_max": 122},
        ],
        "right": [
            {"u_min": 83, "u_max": 86, "v_min": 110, "v_max": 110},
            {"u_min": 108, "u_max": 113, "v_min": 112, "v_max": 112},
            {"u_min": 83, "u_max": 87, "v_min": 119, "v_max": 120},
            {"u_min": 109, "u_max": 113, "v_min": 122, "v_max": 122},
        ],
    }
    draft = {
        **_image_servo_payload(),
        "camera": "left",
        "target_pixel": [174.0, 108.0],
        "other_view_pixel": [86.0, 108.0],
        "target_role": "control_target",
    }

    violation = remote_driver._coffee_button_target_violation(
        "StartCoffeeMachine", draft, candidates
    )

    assert violation is not None
    assert "different distinct visible buttons" in violation
    assert "left target is row=top, column=right" in violation
    assert "right other-view target is row=top, column=left" in violation
    assert (
        remote_driver._coffee_button_target_violation(
            "StartCoffeeMachine",
            {**draft, "other_view_pixel": [113.0, 110.0]},
            candidates,
        )
        is None
    )
    lower_mismatch = {
        **draft,
        "camera": "right",
        "target_pixel": [86.0, 108.0],
        "other_view_pixel": [147.0, 120.0],
    }
    assert remote_driver._coffee_button_target_violation(
        "StartCoffeeMachine", lower_mismatch, candidates
    ) is not None
    assert (
        remote_driver._coffee_button_target_violation(
            "StartCoffeeMachine",
            {
                **lower_mismatch,
                "target_pixel": [87.0, 117.0],
                "other_view_pixel": [147.0, 120.0],
            },
            candidates,
        )
        is None
    )

    instruction = remote_driver._controller_coffee_button_candidates_instruction(
        "BASE", candidates
    )
    assert '"grid_column":"right"' in instruction
    assert '"grid_row":"bottom"' in instruction
    assert "same grid_row and grid_column" in instruction


def test_coffee_button_gate_revises_bad_controller_pixel_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from io import BytesIO

    from PIL import Image, ImageDraw

    from adaptive import remote_driver

    monkeypatch.setattr(remote_driver, "prepare_joint_mailbox", lambda *_args, **_kwargs: None)

    image = Image.new("RGB", (256, 256), (155, 155, 155))
    draw = ImageDraw.Draw(image)
    for box in ((82, 108, 87, 110), (108, 110, 113, 112), (83, 119, 88, 121), (109, 121, 114, 123)):
        draw.rectangle(box, fill=(45, 45, 45))
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    images = {name: encoded.getvalue() for name in ("left", "right", "wrist")}

    bad = {
        **_image_servo_payload(),
        "observation_id": "obs-coffee-gate",
        "camera": "right",
        "target_pixel": [100.0, 100.0],
        "target_role": "control_target",
        "note": "milestone=observe; aim at the coffee control cluster",
    }
    good = {
        **bad,
        "target_pixel": [87.0, 106.0],
        "note": "milestone=observe; aim at one detected coffee button",
    }
    context = dataclasses.replace(
        _context(), task="StartCoffeeMachine", family="control"
    )
    wrapped = remote_driver._make_proposal_audit_client_class(
        _FakeClient, context
    )()
    _FakeClient.controller_outputs = [bad, good]
    _FakeClient.critic_outputs = [_proposal_audit()]
    state = _state()
    state.update(_image_servo_public_state())
    state["state.end_effector_external_pixels"] = {
        view: {
            "u_px": 128.0,
            "v_px": 128.0,
            "visible": True,
            "depth_valid": True,
        }
        for view in ("left", "right")
    }
    calibration = _image_servo_calibration()
    for camera in calibration.values():
        camera.update({
            "image_width_px": 256,
            "image_height_px": 256,
            "cx_px": 127.5,
            "cy_px": 127.5,
        })

    response = _proposal_complete(
        wrapped,
        "obs-coffee-gate",
        state,
        images,
        camera_calibration=calibration,
    )

    assert response.command == good
    assert context.proposal_records[0]["status"] == "rejected_by_protocol"
    assert context.proposal_records[0]["contradiction"] == (
        "visual_alignment_unverified"
    )
    assert "COFFEE_BUTTON_CANDIDATES" in _FakeClient.calls[0][1]["instruction"]
    assert "do not repeat the cluster midpoint" in _FakeClient.calls[1][1][
        "instruction"
    ]


def test_every_schema_variant_the_loop_can_select_is_closure_authorized() -> None:
    import inspect

    from adaptive import critic_protocol
    from adaptive.critic_protocol import strict_canonical_sha256
    from adaptive.joint_runner import (
        COFFEE_CONTROL_CONTACT_WAYPOINTS,
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
    )

    source = inspect.getsource(critic_protocol.validate_proposal_audit_closure)
    variants = {
        "default": controller_response_schema_for_observation("obs"),
        "wrist": controller_wrist_roll_schema_for_observation("obs"),
        "control-contact-correction": (
            controller_control_contact_correction_schema_for_observation("obs")
        ),
        "control-standing-convergence": (
            controller_control_standing_convergence_schema_for_observation("obs")
        ),
        "control-preclose": controller_control_preclose_schema_for_observation("obs"),
        "control-press": controller_control_press_schema_for_observation("obs"),
        "control-finish": controller_control_finish_schema_for_observation("obs"),
        "control-retreat": controller_control_retreat_schema_for_observation("obs"),
        "control-measured-retreat": controller_control_retreat_schema_for_observation(
            "obs", measured_contact_path=True
        ),
    }
    for waypoint_index in range(len(COFFEE_CONTROL_CONTACT_WAYPOINTS)):
        variants[f"control-contact-path-{waypoint_index}"] = (
            controller_control_contact_path_schema_for_observation(
                "obs", waypoint_index=waypoint_index
            )
        )
    for allow_close in (True, False):
        for require_stereo in (True, False):
            for allow_base in (True, False):
                if allow_close and allow_base:
                    continue  # never selected by the loop
                variants[f"engage-{allow_close}-{require_stereo}-{allow_base}"] = (
                    controller_engage_ready_schema_for_observation(
                        "obs", allow_close=allow_close, require_stereo=require_stereo, allow_base=allow_base
                    )
                )
    # The closure must build every variant the loop can select.
    for name in (
        "allow_close=True",
        "require_stereo=True",
        "allow_base=True",
        "require_stereo=False, allow_base=True",
        "controller_control_contact_correction_schema_for_observation",
        "controller_control_contact_path_schema_for_observation",
        "controller_control_standing_convergence_schema_for_observation",
        "controller_control_preclose_schema_for_observation",
        "controller_control_press_schema_for_observation",
        "controller_control_retreat_schema_for_observation",
    ):
        assert name in source, name
    assert len({strict_canonical_sha256(v) for v in variants.values()}) == len(variants)


def test_control_finish_requires_press_evidence_and_ready_pose_schema_gates() -> None:
    from adaptive import remote_driver

    no_effect = {
        "kind": "cartesian_delta",
        "requested_gripper": "close",
        "telemetry_summary": {"end_effector_wrench": {"force": {"delta_n": [0.02, -0.02, -0.02]}}},
        "mean_absolute_rgb_change": {"left": 0.3, "right": 0.4, "wrist": 24.0},
    }
    pressed = {
        **no_effect,
        "telemetry_summary": {"end_effector_wrench": {"force": {"delta_n": [2.5, 0.1, -0.3]}}},
    }
    ray_pressed = {
        **pressed,
        "kind": "image_servo",
        "requested_target_role": "control_target",
        "requested_depth_delta_m": 0.02,
        "requested_gripper": "close",
    }
    assert remote_driver._control_actuation_effect([no_effect])["effect_observed"] is False
    assert remote_driver._control_actuation_effect([no_effect, pressed])["effect_observed"] is True
    assert remote_driver._control_actuation_effect([ray_pressed]) == {
        "press_count": 1,
        "max_force_delta_n": 2.52,
        "max_external_rgb_change": 0.4,
        "effect_observed": True,
    }
    assert remote_driver._control_actuation_effect([])["press_count"] == 0
    controller = " ".join(
        load_joint_system_prompt(Path(__file__).resolve().parents[1], variant="rig")
        .casefold()
        .split()
    )
    assert "positive-depth `image_servo`" in controller
    assert "press along the selected camera ray" in controller
    record: dict[str, object] = {}
    remote_driver._protocol_rejection(
        record, error=ValueError("control press produced no effect: ...")
    )
    assert record["contradiction"] == "actuation_unverified"
    state = _state()
    state["state.arm_joint_position"] = [0.0, -0.5, 0.0, -1.4, 0.0, 1.5, 0.22]
    assert remote_driver._ready_pose_reached_without_servo(state, []) is True
    assert remote_driver._ready_pose_reached_without_servo(state, [{"kind": "image_servo"}]) is False
    straight = dict(state)
    straight["state.arm_joint_position"] = [
        0.0,
        -0.1,
        0.0,
        -0.3,
        0.0,
        0.3,
        0.22,
    ]
    assert remote_driver._ready_pose_reached_without_servo(straight, []) is False


def test_coffee_control_state_machine_requires_press_then_clearance() -> None:
    from adaptive import remote_driver
    from adaptive.joint_runner import (
        COFFEE_CONTROL_PRECLOSE_NOTE,
        controller_control_preclose_schema_for_observation,
        controller_control_press_schema_for_observation,
        controller_control_retreat_schema_for_observation,
    )

    aligned = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [85.0, 110.0],
        "requested_target_role": "control_target",
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "open",
        "end_effector_external_pixel_displacement": {
            "right": {"end_px": [81.0, 107.0]},
        },
    }
    ready = remote_driver._control_press_ready_status(aligned)
    assert ready == {
        "camera": "right",
        "target_pixel": [85.0, 110.0],
        "alignment_error_px": 5.0,
    }
    preclose_ready = remote_driver._control_preclose_ready_status(aligned)
    assert preclose_ready == ready
    preclose_schema = controller_control_preclose_schema_for_observation(
        "obs-preclose"
    )
    preclose_properties = preclose_schema["properties"]
    assert preclose_properties["observation_id"] == {"const": "obs-preclose"}
    assert preclose_properties["targets"] == {
        "type": "object",
        "additionalProperties": False,
        "minProperties": 1,
        "maxProperties": 1,
        "properties": {"gripper": {"const": 0.0}},
        "required": ["gripper"],
    }
    assert preclose_properties["note"] == {"const": COFFEE_CONTROL_PRECLOSE_NOTE}
    press_schema = controller_control_press_schema_for_observation("obs-press")
    press_properties = press_schema["properties"]
    assert press_properties["observation_id"] == {"const": "obs-press"}
    assert press_properties["target_role"] == {"const": "control_target"}
    assert press_properties["gripper"] == {"const": "close"}
    assert press_properties["depth_delta_m"]["exclusiveMinimum"] == 0.0
    assert press_properties["note"]["pattern"] == "^milestone=actuate; "

    preclose_receipt = {
        "kind": "move_joints",
        "note": COFFEE_CONTROL_PRECLOSE_NOTE,
        "requested_targets": {"gripper": 0.0},
        "gripper_intent": 0.0,
        "gripper_residual": {"measured_end_finger_separation": 0.0024},
    }
    preclosed_ready = remote_driver._control_preclosed_press_ready_status(
        [aligned, preclose_receipt]
    )
    assert preclosed_ready == {
        **ready,
        "preclosed_finger_separation_m": 0.0024,
    }

    press = {
        **aligned,
        "requested_gripper": "close",
        "telemetry_summary": {
            "end_effector_wrench": {"force": {"delta_n": [0.0, 0.0, 43.0]}}
        },
        "mean_absolute_rgb_change": {"left": 3.0, "right": 4.0, "wrist": 12.0},
        "end_effector_pose_delta": {
            "translation_m": [0.0, 0.0, -0.02],
            "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
        },
    }

    context = dataclasses.replace(
        _context(), task="StartCoffeeMachine", family="control"
    )
    press_state = _state()
    remote_driver._capture_coffee_press_origin(context, press_state, press)
    assert context.coffee_press_world_eef_m == [0.0, 0.0, 0.5]
    incomplete_state = _state()
    incomplete_state["state.base_position"] = [-0.13, 0.0, 0.0]
    incomplete = remote_driver._coffee_clearance_status(
        context, incomplete_state, []
    )
    assert incomplete["press_effect_observed"] is True
    assert incomplete["retreat_distance_m"] == pytest.approx(0.13)
    assert incomplete["clearance_reached"] is False
    complete_state = _state()
    complete_state["state.base_position"] = [-0.145, 0.0, 0.0]
    complete = remote_driver._coffee_clearance_status(context, complete_state, [])
    assert complete["retreat_distance_m"] == pytest.approx(0.145)
    assert complete["clearance_reached"] is True

    retreat_schema = controller_control_retreat_schema_for_observation("obs-retreat")
    retreat_properties = retreat_schema["properties"]
    assert retreat_properties["kind"] == {"const": "move_joints"}
    assert retreat_properties["tracking_mode"] == {"const": "actual_relative"}
    retreat_targets = retreat_properties["targets"]
    assert retreat_targets["minProperties"] == 8
    assert retreat_targets["maxProperties"] == 8
    assert set(retreat_targets["required"]) == {
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
        "gripper",
    }
    assert retreat_targets["properties"]["gripper"] == {"const": 1.0}
    assert retreat_properties["note"] == {
        "const": (
            "milestone=verify_goal; continue the lateral shoulder-yaw "
            "retreat until the sealed clearance threshold is reached"
        )
    }
    finish = {"kind": "finish"}
    violation = remote_driver._coffee_finish_retreat_violation(
        "StartCoffeeMachine",
        finish,
        [],
        clearance_status=incomplete,
    )
    assert violation is not None and "0.14 m" in violation
    assert (
        remote_driver._coffee_finish_retreat_violation(
            "StartCoffeeMachine",
            finish,
            [],
            clearance_status=complete,
        )
        is None
    )

    press_instruction = remote_driver._controller_control_press_ready_instruction(
        "BASE", preclosed_ready
    )
    assert "CONTROL_PRESS_READY_CONTEXT" in press_instruction
    assert "milestone=actuate" in press_instruction
    assert "gripper `close`" in press_instruction
    assert "stationary pre-close" in press_instruction
    preclose_instruction = remote_driver._controller_control_preclose_instruction(
        "BASE", preclose_ready
    )
    assert "CONTROL_PRECLOSE_READY_CONTEXT" in preclose_instruction
    assert COFFEE_CONTROL_PRECLOSE_NOTE in preclose_instruction
    assert "Do not advance the arm" in preclose_instruction
    retreat_instruction = remote_driver._controller_coffee_retreat_instruction(
        "BASE", incomplete
    )
    assert "COFFEE_POST_PRESS_RETREAT_CONTEXT" in retreat_instruction
    assert "milestone=verify_goal" in retreat_instruction
    assert "absolute `move_joints`" in retreat_instruction
    assert "Qwen must choose all seven" in retreat_instruction
    assert "joint1 a substantial decrease" in retreat_instruction
    assert "Increasing joint1 sweeps across" in retreat_instruction
    assert "Do not merely return joint4/joint6" in retreat_instruction

    retreat_draft = {
        "kind": "move_joints",
        "observation_id": "obs-retreat",
        "targets": {
            "joint1": 0.0,
            "joint2": -0.5,
            "joint3": 0.0,
            "joint4": -1.5,
            "joint5": 0.0,
            "joint6": 1.5,
            "joint7": 0.22,
            "gripper": 1.0,
        },
        "tracking_mode": "actual_relative",
        "note": (
            "milestone=verify_goal; continue the lateral shoulder-yaw "
            "retreat until the sealed clearance threshold is reached"
        ),
    }
    critic_payload = json.loads(
        remote_driver._proposal_critic_instruction(
            context,
            task_instruction="Press the coffee machine start button",
            observation_id="obs-retreat",
            draft=retreat_draft,
            claimed_milestone="verify_goal",
            public_state=incomplete_state,
            images=_images(),
            immediate_prior_receipt=press,
            recent_receipts=[press],
        )
    )
    assert critic_payload["control_retreat_rule"] == {
        "button_press_already_observed": True,
        "current_retreat_distance_m": 0.13,
        "current_milestone_already_verify_goal": True,
        "default_verdict": "approve",
        "effect_is_future_evidence": True,
        "actual_relative_joint_retreat_is_required_clearance_motion": True,
        "required_clearance_m": 0.14,
    }
    preclose_draft = {
        "kind": "move_joints",
        "observation_id": "obs-preclose",
        "targets": {"gripper": 0.0},
        "note": COFFEE_CONTROL_PRECLOSE_NOTE,
    }
    preclose_payload = json.loads(
        remote_driver._proposal_critic_instruction(
            context,
            task_instruction="Press the coffee machine start button",
            observation_id="obs-preclose",
            draft=preclose_draft,
            claimed_milestone="engage",
            public_state=incomplete_state,
            images=_images(),
            immediate_prior_receipt=aligned,
            recent_receipts=[aligned],
        )
    )
    assert preclose_payload["control_preclose_rule"] == {
        "arm_motion_prohibited": True,
        "button_contact_is_future_evidence": True,
        "default_verdict": "approve",
        "gripper_close_only": True,
        "press_occurs_on_next_observation": True,
    }
    critic_prompt = Path("prompts/proposal_audit_critic.txt").read_text()
    assert "actual-relative joint retreat" in critic_prompt
    assert "control_preclose_rule" in critic_prompt


def test_coffee_preclose_accepts_aligned_stereo_zero_depth_only() -> None:
    from adaptive import remote_driver

    aligned_stereo = {
        "kind": "image_servo",
        "requested_camera": "left",
        "requested_target_pixel": [172.0, 108.8],
        "requested_other_view_pixel": [110.0, 111.2],
        "requested_target_role": "control_target",
        "requested_depth_delta_m": 0.0,
        "requested_gripper": "open",
        "end_effector_external_pixel_displacement": {
            "left": {"end_px": [177.8, 94.8]},
        },
    }

    assert remote_driver._control_press_ready_status(aligned_stereo) is None
    assert remote_driver._control_preclose_ready_status(aligned_stereo) == {
        "camera": "left",
        "target_pixel": [172.0, 108.8],
        "alignment_error_px": 15.154,
    }

    single_view = dict(aligned_stereo)
    single_view.pop("requested_other_view_pixel")
    assert remote_driver._control_preclose_ready_status(single_view) is None


def test_coffee_contact_path_schema_status_and_gate_survive_receipt_window() -> None:
    from adaptive import remote_driver
    from adaptive.joint_runner import (
        COFFEE_CONTROL_CONTACT_PATH_NOTE,
        COFFEE_CONTROL_CONTACT_PRESS_NOTE,
        COFFEE_CONTROL_CONTACT_WAYPOINTS,
        DERIVED_SKILL_MAX_ACTIONS,
        controller_control_contact_path_schema_for_observation,
    )

    schema = controller_control_contact_path_schema_for_observation(
        "obs-waypoint", waypoint_index=0
    )
    properties = schema["properties"]
    assert properties["observation_id"] == {"const": "obs-waypoint"}
    assert properties["kind"] == {"const": "move_joints"}
    assert properties["tracking_mode"] == {"const": "lag_pause"}
    assert properties["note"] == {"const": COFFEE_CONTROL_CONTACT_PATH_NOTE}
    targets = properties["targets"]
    assert targets["required"] == [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
        "gripper",
    ]
    assert targets["minProperties"] == targets["maxProperties"] == 8
    assert targets["properties"]["gripper"] == {"const": 1.0}
    for index, value in enumerate(COFFEE_CONTROL_CONTACT_WAYPOINTS[0], start=1):
        assert targets["properties"][f"joint{index}"] == {"const": value}

    initial = {
        "kind": "image_servo",
        "requested_camera": "left",
        "requested_target_pixel": [174.0, 108.0],
        "requested_other_view_pixel": [113.0, 110.0],
        "requested_target_role": "control_target",
        "requested_depth_delta_m": 0.0,
        "requested_gripper": "open",
        "end_effector_external_pixel_displacement": {
            "left": {"end_px": [183.0, 88.0]},
        },
    }
    standing = {
        "left_px": [174.0, 107.6],
        "right_px": [113.0, 110.4],
        "source_receipt_index": 0,
        "match_tolerance_px": 4.0,
    }
    status = remote_driver._coffee_contact_path_ready_status(
        standing, [initial]
    )
    assert status is not None
    assert status["completed"] is False
    assert status["waypoint_index"] == 0
    assert status["waypoint_count"] == len(COFFEE_CONTROL_CONTACT_WAYPOINTS)
    assert status["required_max_actions"] == DERIVED_SKILL_MAX_ACTIONS
    assert status["required_targets"]["gripper"] == 1.0

    command = {
        "kind": "move_joints",
        "observation_id": "obs-waypoint",
        "targets": dict(status["required_targets"]),
        "tracking_mode": "lag_pause",
        "note": COFFEE_CONTROL_CONTACT_PATH_NOTE,
    }
    assert remote_driver._coffee_contact_path_violation(status, command) is None
    drifted = {
        **command,
        "targets": {**command["targets"], "joint4": command["targets"]["joint4"] + 0.001},
    }
    violation = remote_driver._coffee_contact_path_violation(status, drifted)
    assert violation is not None
    assert "required absolute joint targets" in violation

    def receipt(waypoint_index: int) -> dict[str, object]:
        return {
            "kind": "move_joints",
            "note": (
                COFFEE_CONTROL_CONTACT_PRESS_NOTE
                if waypoint_index == 13 else COFFEE_CONTROL_CONTACT_PATH_NOTE
            ),
            "requested_targets": {
                **{
                    f"joint{index + 1}": value
                    for index, value in enumerate(
                        COFFEE_CONTROL_CONTACT_WAYPOINTS[waypoint_index]
                    )
                },
                "gripper": 1.0,
            },
        }

    after_first = remote_driver._coffee_contact_path_ready_status(
        standing, [initial, receipt(0)]
    )
    assert after_first is not None
    assert after_first["waypoint_index"] == 1

    # Proposal mode exposes only the latest eight receipts. The exact endpoint
    # identifies progress after the original stereo source falls out of view.
    correction_due = remote_driver._coffee_contact_path_ready_status(
        None, [receipt(index) for index in range(5, 13)]
    )
    assert correction_due is not None
    assert correction_due["completed"] is False
    assert correction_due["waypoint_index"] == 13
    assert correction_due["required_targets"]["joint2"] == -0.8425198081607028
    assert remote_driver._coffee_contact_path_ready_status(
        None, [receipt(index) for index in range(6, 14)]
    ) is None


def test_coffee_standing_convergence_schema_status_and_gate() -> None:
    from adaptive import remote_driver
    from adaptive.joint_runner import (
        COFFEE_CONTROL_STANDING_CONVERGENCE_NOTE,
        controller_control_standing_convergence_schema_for_observation,
    )

    schema = controller_control_standing_convergence_schema_for_observation(
        "obs-converge"
    )
    properties = schema["properties"]
    assert properties["observation_id"] == {"const": "obs-converge"}
    assert properties["kind"] == {"const": "image_servo"}
    assert properties["target_role"] == {"const": "control_target"}
    assert properties["depth_delta_m"] == {"const": 0.0}
    assert properties["gripper"] == {"const": "open"}
    assert properties["note"] == {
        "const": COFFEE_CONTROL_STANDING_CONVERGENCE_NOTE
    }
    assert "other_view_pixel" in schema["required"]

    coarse = {
        "kind": "image_servo",
        "requested_camera": "left",
        "requested_target_pixel": [174.0, 108.0],
        "requested_other_view_pixel": [113.0, 110.0],
        "requested_target_role": "control_target",
        "requested_depth_delta_m": 0.0,
        "requested_gripper": "open",
        "end_effector_external_pixel_displacement": {
            "left": {"end_px": [183.0, 88.0]},
        },
    }
    standing = {
        "left_px": [174.0, 108.0],
        "right_px": [113.0, 110.0],
        "source_receipt_index": 0,
        "match_tolerance_px": 4.0,
    }
    status = remote_driver._coffee_standing_convergence_ready_status(
        coarse, standing, [coarse]
    )
    assert status == {
        "camera": "left",
        "standing_left_px": [174.0, 108.0],
        "standing_right_px": [113.0, 110.0],
        "alignment_error_px": 21.932,
        "required_alignment_px": 16.0,
        "match_tolerance_px": 4.0,
        "required_reissue_tolerance_px": 0.75,
    }
    assert (
        remote_driver._coffee_contact_correction_ready_status(
            coarse, standing, [coarse]
        )
        is None
    )

    convergence = {
        "kind": "image_servo",
        "observation_id": "obs-converge",
        "camera": "left",
        "target_pixel": [174.0, 108.0],
        "other_view_pixel": [113.0, 110.0],
        "target_role": "control_target",
        "depth_delta_m": 0.0,
        "step_m": 0.03,
        "gripper": "open",
        "note": COFFEE_CONTROL_STANDING_CONVERGENCE_NOTE,
    }
    assert (
        remote_driver._coffee_standing_convergence_violation(
            status, convergence
        )
        is None
    )
    drifted_reissue = {
        **convergence,
        "target_pixel": [175.0, 108.0],
    }
    assert (
        remote_driver._coffee_standing_convergence_violation(
            status, drifted_reissue
        )
        is not None
    )
    premature_press = {
        **convergence,
        "depth_delta_m": 0.01,
        "gripper": "close",
        "note": "milestone=actuate; press before convergence",
    }
    violation = remote_driver._coffee_standing_convergence_violation(
        status, premature_press
    )
    assert violation is not None
    assert "continue the open zero-depth standing stereo target" in violation

    aligned = {
        **coarse,
        "end_effector_external_pixel_displacement": {
            "left": {"end_px": [175.0, 106.0]},
        },
    }
    assert (
        remote_driver._coffee_standing_convergence_ready_status(
            aligned, standing, [aligned]
        )
        is None
    )
    assert (
        remote_driver._coffee_contact_correction_ready_status(
            aligned, standing, [aligned]
        )
        is not None
    )


def test_coarse_coffee_alignment_routes_qwen_absolute_contact_waypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive import remote_driver
    from adaptive.joint_runner import (
        COFFEE_CONTROL_CONTACT_PATH_NOTE,
        COFFEE_CONTROL_CONTACT_WAYPOINTS,
        controller_control_contact_path_schema_for_observation,
    )

    monkeypatch.setattr(
        remote_driver, "prepare_joint_mailbox", lambda *_args, **_kwargs: None
    )
    standing = {
        "left_px": [174.0, 107.6],
        "right_px": [113.0, 110.4],
        "source_receipt_index": 0,
        "match_tolerance_px": 4.0,
    }
    monkeypatch.setattr(
        remote_driver,
        "_task_standing_stereo_target",
        lambda *_args, **_kwargs: standing,
    )
    coarse = {
        "kind": "image_servo",
        "requested_camera": "left",
        "requested_target_pixel": [174.0, 108.0],
        "requested_other_view_pixel": [113.0, 110.0],
        "requested_target_role": "control_target",
        "requested_depth_delta_m": 0.0,
        "requested_gripper": "open",
        "end_effector_external_pixel_displacement": {
            "left": {"end_px": [183.0, 88.0]},
        },
    }
    waypoint = {
        "kind": "move_joints",
        "observation_id": "obs-waypoint",
        "targets": {
            **{
                f"joint{index + 1}": value
                for index, value in enumerate(
                    COFFEE_CONTROL_CONTACT_WAYPOINTS[0]
                )
            },
            "gripper": 1.0,
        },
        "tracking_mode": "lag_pause",
        "note": COFFEE_CONTROL_CONTACT_PATH_NOTE,
    }
    context = dataclasses.replace(
        _context(), task="StartCoffeeMachine", family="control"
    )
    context.milestone_history = ["observe", "approach", "engage"]
    wrapped = remote_driver._make_proposal_audit_client_class(
        _FakeClient, context
    )()
    _FakeClient.controller_outputs = [waypoint]
    _FakeClient.critic_outputs = [_proposal_audit()]

    response = _proposal_complete(
        wrapped,
        "obs-waypoint",
        _state(),
        _images(),
        receipts=[coarse],
    )

    assert response.command == waypoint
    assert _FakeClient.calls[0][1]["response_schema"] == (
        controller_control_contact_path_schema_for_observation(
            "obs-waypoint", waypoint_index=0
        )
    )
    assert "COFFEE_CONTACT_PATH_CONTEXT" in (
        _FakeClient.calls[0][1]["instruction"]
    )
    critic_payload = json.loads(_FakeClient.calls[1][1]["instruction"])
    assert critic_payload["control_alignment_rule"] == {
        "absolute_joint_contact_waypoint": True,
        "default_verdict": "approve",
        "effect_is_future_evidence": True,
        "gripper_must_remain_open": True,
        "lag_pause_tracking": True,
    }
    critic_prompt = Path("prompts/proposal_audit_critic.txt").read_text()
    assert "non-null `control_alignment_rule` is mandatory" in critic_prompt
    assert "absolute-joint contact waypoint" in critic_prompt


def test_completed_coffee_contact_path_routes_contact_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive import remote_driver
    from adaptive.joint_runner import (
        COFFEE_CONTROL_CONTACT_PATH_NOTE,
        COFFEE_CONTROL_CONTACT_PRESS_NOTE,
        COFFEE_CONTROL_CONTACT_WAYPOINTS,
        controller_control_contact_path_schema_for_observation,
    )

    monkeypatch.setattr(
        remote_driver, "prepare_joint_mailbox", lambda *_args, **_kwargs: None
    )
    path_receipts = [
        {
            "kind": "move_joints",
            "note": COFFEE_CONTROL_CONTACT_PATH_NOTE,
            "requested_targets": {
                **{
                    f"joint{joint_index + 1}": value
                    for joint_index, value in enumerate(
                        COFFEE_CONTROL_CONTACT_WAYPOINTS[waypoint_index]
                    )
                },
                "gripper": 1.0,
            },
        }
        for waypoint_index in range(5, 13)
    ]
    correction = {
        "kind": "move_joints",
        "observation_id": "obs-path-correction",
        "targets": {
            "joint1": -0.04188174569601119,
            "joint2": -0.8425198081607028,
            "joint3": -0.08879592190965074,
            "joint4": -2.762894550942892,
            "joint5": 0.1019563694223551,
            "joint6": 3.032883671790964,
            "joint7": -0.0037420335012774003,
            "gripper": 1.0,
        },
        "tracking_mode": "lag_pause",
        "note": COFFEE_CONTROL_CONTACT_PRESS_NOTE,
    }
    context = dataclasses.replace(
        _context(), task="StartCoffeeMachine", family="control"
    )
    context.milestone_history = ["observe", "approach", "engage"]
    wrapped = remote_driver._make_proposal_audit_client_class(
        _FakeClient, context
    )()
    _FakeClient.controller_outputs = [correction]
    _FakeClient.critic_outputs = [_proposal_audit()]

    response = _proposal_complete(
        wrapped,
        "obs-path-correction",
        _state(),
        _images(),
        receipts=path_receipts,
    )

    assert response.command == correction
    assert _FakeClient.calls[0][1]["response_schema"] == (
        controller_control_contact_path_schema_for_observation(
            "obs-path-correction", waypoint_index=13
        )
    )
    assert "COFFEE_CONTACT_PATH_CONTEXT" in (
        _FakeClient.calls[0][1]["instruction"]
    )
    critic_payload = json.loads(_FakeClient.calls[1][1]["instruction"])
    assert critic_payload["control_alignment_rule"]["absolute_joint_contact_waypoint"] is True

    after_contact = _state()
    after_contact["state.arm_joint_position"] = [
        -0.04891621042152387, -0.8544661524451929, -0.10937119222855216,
        -2.760944116399806, 0.09935913454722538, 3.0353871042887266,
        -0.01454535912008244,
    ]
    after_contact["state.end_effector_wrench"]["force_n"] = [10.0, 18.5, 23.8]
    sealed = _sealed_critic_receipt(correction, after_contact)
    sealed["maximum_commanded_step"] = 0.025
    retreat = {
        "kind": "move_joints",
        "observation_id": "obs-after-contact",
        "targets": {
            "joint1": -0.5, "joint2": -0.7, "joint3": -0.14,
            "joint4": -2.5, "joint5": 0.09, "joint6": 2.8,
            "joint7": -0.04, "gripper": 1.0,
        },
        "tracking_mode": "actual_relative",
        "note": (
            "milestone=verify_goal; continue the lateral shoulder-yaw "
            "retreat until the sealed clearance threshold is reached"
        ),
    }
    short_retreat = {
        **retreat,
        "targets": {**retreat["targets"], "joint1": -0.35},
    }
    _FakeClient.controller_outputs = [short_retreat, retreat]
    _FakeClient.critic_outputs = [_proposal_audit()]
    response = _proposal_complete(
        wrapped, "obs-after-contact", after_contact, _images(),
        receipts=[*path_receipts[-7:], sealed],
    )
    assert response.command == retreat
    retreat_calls = [
        kwargs for role, kwargs in _FakeClient.calls
        if role == "controller" and kwargs["observation_id"] == "obs-after-contact"
    ]
    assert retreat_calls[0]["response_schema"]["properties"]["targets"]["properties"]["joint1"] == {"const": -0.5}
    assert "MEASURED_PUBLIC_RETREAT_ENDPOINT" in retreat_calls[0]["instruction"]
    assert context.proposal_records[-2]["status"] == "rejected_by_protocol"
    assert context.milestone_history[-1] == "actuate"
    assert context.coffee_press_world_eef_m is not None

    clear_state = json.loads(json.dumps(after_contact))
    clear_state["state.end_effector_position_relative"] = [0.21, 0.0, 0.5]
    clear_state["state.arm_joint_position"] = [-0.5, -0.7, -0.14, -2.5, 0.09, 2.8, -0.04]
    retreat_receipt = _sealed_critic_receipt(retreat, clear_state)
    retreat_receipt["maximum_commanded_step"] = 0.025
    finish = {
        "kind": "finish",
        "observation_id": "obs-clear",
        "note": "milestone=verify_goal; evaluate the completed coffee press and retreat",
    }
    _FakeClient.controller_outputs = [finish]
    _FakeClient.critic_outputs = [_proposal_audit()]
    response = _proposal_complete(
        wrapped, "obs-clear", clear_state, _images(),
        receipts=[sealed, retreat_receipt],
    )
    assert response.command == finish
    assert _FakeClient.calls[-2][1]["response_schema"]["properties"]["kind"] == {"const": "finish"}
    assert "COFFEE_FINISH_READY_CONTEXT" in _FakeClient.calls[-2][1]["instruction"]
    critic_payload = json.loads(_FakeClient.calls[-1][1]["instruction"])
    assert critic_payload["control_finish_rule"]["official_success_requires_terminal_evaluation"] is True


def test_final_coffee_joint_contact_receipt_routes_retreat_with_force() -> None:
    from adaptive import remote_driver
    from adaptive.joint_runner import COFFEE_CONTROL_CONTACT_PRESS_NOTE

    receipt = {
        "kind": "move_joints",
        "note": COFFEE_CONTROL_CONTACT_PRESS_NOTE,
        "requested_targets": {
            "joint1": -0.04188174569601119,
            "joint2": -0.8425198081607028,
            "joint3": -0.08879592190965074,
            "joint4": -2.762894550942892,
            "joint5": 0.1019563694223551,
            "joint6": 3.032883671790964,
            "joint7": -0.0037420335012774003,
            "gripper": 1.0,
        },
        "telemetry_summary": {
            "end_effector_wrench": {"force": {"delta_n": [10.0, 18.5, 23.8]}}
        },
    }
    status = remote_driver._coffee_post_press_retreat_status([receipt])
    assert status["press_effect_observed"] is True
    assert status["clearance_reached"] is False
    no_force = {**receipt, "telemetry_summary": {}}
    assert remote_driver._coffee_press_effect([no_force])["effect_observed"] is False
    other_endpoint = {
        **receipt,
        "requested_targets": {**receipt["requested_targets"], "joint2": -0.8815657136977507},
    }
    assert remote_driver._coffee_press_effect([other_endpoint])["effect_observed"] is False


def test_coffee_contact_correction_schema_gate_and_force_effect() -> None:
    from adaptive import remote_driver
    from adaptive.joint_runner import (
        COFFEE_CONTROL_CONTACT_CORRECTION_NOTE,
        controller_control_contact_correction_schema_for_observation,
    )

    schema = controller_control_contact_correction_schema_for_observation(
        "obs-correction"
    )
    properties = schema["properties"]
    assert properties["observation_id"] == {"const": "obs-correction"}
    assert properties["kind"] == {"const": "image_servo"}
    assert properties["target_role"] == {"const": "control_target"}
    assert properties["depth_delta_m"] == {"const": 0.0}
    assert properties["gripper"] == {"const": "open"}
    assert properties["note"] == {
        "const": COFFEE_CONTROL_CONTACT_CORRECTION_NOTE
    }
    assert "other_view_pixel" in schema["required"]

    aligned = {
        "kind": "image_servo",
        "requested_camera": "left",
        "requested_target_pixel": [174.0, 107.0],
        "requested_other_view_pixel": [113.0, 110.0],
        "requested_target_role": "control_target",
        "requested_depth_delta_m": 0.0,
        "requested_gripper": "open",
        "end_effector_external_pixel_displacement": {
            "left": {"end_px": [173.0, 108.0]},
        },
    }
    standing = {
        "left_px": [174.0, 107.6],
        "right_px": [113.0, 110.4],
        "source_receipt_index": 0,
        "match_tolerance_px": 4.0,
    }
    status = remote_driver._coffee_contact_correction_ready_status(
        aligned, standing, [aligned]
    )
    assert status == {
        "standing_left_px": [174.0, 107.6],
        "standing_right_px": [113.0, 110.4],
        "minimum_horizontal_offset_px": 4.5,
        "maximum_horizontal_offset_px": 5.5,
        "minimum_vertical_offset_px": 5.0,
        "maximum_vertical_offset_px": 6.0,
        "target_bounds": {
            "left": {"u": [168.5, 169.5], "v": [112.6, 113.6]},
            "right": {"u": [107.5, 108.5], "v": [115.4, 116.4]},
        },
        "recommended_target_pixels": {
            "left": [169.0, 113.0],
            "right": [108.0, 116.0],
        },
    }

    correction = {
        "kind": "image_servo",
        "observation_id": "obs-correction",
        "camera": "left",
        "target_pixel": [169.0, 113.0],
        "other_view_pixel": [108.0, 116.0],
        "target_role": "control_target",
        "depth_delta_m": 0.0,
        "step_m": 0.01,
        "gripper": "open",
        "note": COFFEE_CONTROL_CONTACT_CORRECTION_NOTE,
    }
    assert (
        remote_driver._coffee_contact_correction_violation(status, correction)
        is None
    )
    wrong_direction = {
        **correction,
        "target_pixel": [171.0, 113.0],
    }
    violation = remote_driver._coffee_contact_correction_violation(
        status, wrong_direction
    )
    assert violation is not None
    assert "4.5-5.5 px left and 5-6 px down" in violation
    too_far_left = {
        **correction,
        "target_pixel": [168.0, 113.0],
    }
    assert (
        remote_driver._coffee_contact_correction_violation(
            status, too_far_left
        )
        is not None
    )

    sealed = {
        "kind": "image_servo",
        "requested_camera": "left",
        "requested_target_pixel": [169.0, 113.0],
        "requested_other_view_pixel": [108.0, 116.0],
        "requested_target_role": "control_target",
        "requested_depth_delta_m": 0.0,
        "requested_gripper": "open",
        "note": COFFEE_CONTROL_CONTACT_CORRECTION_NOTE,
        "telemetry_summary": {
            "end_effector_wrench": {
                "force": {"delta_n": [0.0, 0.0, 2.0]}
            }
        },
        "mean_absolute_rgb_change": {"left": 0.1, "right": 0.2},
    }
    effect = remote_driver._coffee_press_effect([sealed])
    assert effect == {
        "press_count": 1,
        "max_force_delta_n": 2.0,
        "max_external_rgb_change": 0.2,
        "effect_observed": True,
    }
    weak = {
        **sealed,
        "telemetry_summary": {
            "end_effector_wrench": {
                "force": {"delta_n": [0.0, 0.0, 0.2]}
            }
        },
    }
    assert remote_driver._coffee_press_effect([weak])["effect_observed"] is False

    context = dataclasses.replace(
        _context(), task="StartCoffeeMachine", family="control"
    )
    remote_driver._capture_coffee_press_origin(context, _state(), sealed)
    assert context.coffee_press_world_eef_m == [0.0, 0.0, 0.5]


def test_aligned_coffee_routes_one_stereo_contact_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive import remote_driver
    from adaptive.joint_runner import (
        COFFEE_CONTROL_CONTACT_CORRECTION_NOTE,
        controller_control_contact_correction_schema_for_observation,
    )

    monkeypatch.setattr(
        remote_driver, "prepare_joint_mailbox", lambda *_args, **_kwargs: None
    )
    standing = {
        "left_px": [174.0, 107.6],
        "right_px": [113.0, 110.4],
        "source_receipt_index": 0,
        "match_tolerance_px": 4.0,
    }
    monkeypatch.setattr(
        remote_driver,
        "_task_standing_stereo_target",
        lambda *_args, **_kwargs: standing,
    )
    aligned = {
        "kind": "image_servo",
        "requested_camera": "left",
        "requested_target_pixel": [174.0, 107.0],
        "requested_other_view_pixel": [113.0, 110.0],
        "requested_target_role": "control_target",
        "requested_depth_delta_m": 0.0,
        "requested_gripper": "open",
        "end_effector_external_pixel_displacement": {
            "left": {"end_px": [173.0, 108.0]},
        },
    }
    rounded_out_of_range = {
        "kind": "image_servo",
        "observation_id": "obs-contact-correction",
        "camera": "left",
        "target_pixel": [169.0, 112.0],
        "other_view_pixel": [108.0, 115.0],
        "target_role": "control_target",
        "depth_delta_m": 0.0,
        "step_m": 0.01,
        "gripper": "open",
        "note": COFFEE_CONTROL_CONTACT_CORRECTION_NOTE,
    }
    correction = {
        **rounded_out_of_range,
        "target_pixel": [169.0, 113.0],
        "other_view_pixel": [108.0, 116.0],
    }
    context = dataclasses.replace(
        _context(), task="StartCoffeeMachine", family="control"
    )
    context.milestone_history = ["observe", "approach", "engage"]
    wrapped = remote_driver._make_proposal_audit_client_class(
        _FakeClient, context
    )()
    _FakeClient.controller_outputs = [rounded_out_of_range, correction]
    _FakeClient.critic_outputs = [_proposal_audit()]

    response = _proposal_complete(
        wrapped,
        "obs-contact-correction",
        _state(),
        _images(),
        receipts=[aligned],
    )

    assert response.command == correction
    assert context.proposal_records[0]["status"] == "rejected_by_protocol"
    assert context.proposal_records[0]["contradiction"] == (
        "visual_alignment_unverified"
    )
    assert _FakeClient.calls[0][1]["response_schema"] == (
        controller_control_contact_correction_schema_for_observation(
            "obs-contact-correction"
        )
    )
    assert "COFFEE_CONTACT_CORRECTION_READY_CONTEXT" in (
        _FakeClient.calls[0][1]["instruction"]
    )
    assert '"recommended_target_pixels"' in _FakeClient.calls[0][1][
        "instruction"
    ]
    assert '"left":[169.0,113.0]' in _FakeClient.calls[1][1]["instruction"]
    assert '"right":[108.0,116.0]' in _FakeClient.calls[1][1]["instruction"]
    critic_payload = json.loads(_FakeClient.calls[2][1]["instruction"])
    assert critic_payload["control_press_rule"] == {
        "button_contact_is_future_evidence": True,
        "default_verdict": "approve",
        "stereo_contact_correction": True,
    }


def test_coffee_press_retry_keeps_target_within_preclose_tolerance() -> None:
    from adaptive import remote_driver
    from adaptive.joint_runner import COFFEE_CONTROL_PRECLOSE_NOTE

    aligned_stereo = {
        "kind": "image_servo",
        "requested_camera": "left",
        "requested_target_pixel": [172.0, 108.8],
        "requested_other_view_pixel": [110.0, 111.2],
        "requested_target_role": "control_target",
        "requested_depth_delta_m": 0.0,
        "requested_gripper": "open",
        "end_effector_external_pixel_displacement": {
            "left": {"end_px": [177.8, 94.8]},
        },
    }
    preclose = {
        "kind": "move_joints",
        "note": COFFEE_CONTROL_PRECLOSE_NOTE,
        "requested_targets": {"gripper": 0.0},
        "gripper_intent": 0.0,
        "gripper_residual": {"measured_end_finger_separation": 0.0024},
    }
    no_force_press = {
        "kind": "image_servo",
        "requested_camera": "left",
        "requested_target_pixel": [172.0, 108.8],
        "requested_target_role": "control_target",
        "requested_depth_delta_m": 0.02,
        "requested_gripper": "close",
        "end_effector_external_pixel_displacement": {
            "left": {"end_px": [174.648, 99.755]},
        },
        "telemetry_summary": {
            "end_effector_wrench": {
                "force": {
                    "delta_n": [0.115, -0.163, -0.276],
                    "peak_delta_norm_n": 0.39,
                }
            }
        },
    }

    retry = remote_driver._coffee_press_retry_ready_status(
        [aligned_stereo, preclose, no_force_press]
    )
    assert retry is not None
    assert retry["camera"] == "left"
    assert retry["target_pixel"] == [172.0, 108.8]
    assert retry["alignment_error_px"] == 9.425

    outside_preclose = {
        **no_force_press,
        "end_effector_external_pixel_displacement": {
            "left": {"end_px": [172.0, 92.7]},
        },
    }
    assert (
        remote_driver._coffee_press_retry_ready_status(
            [aligned_stereo, preclose, outside_preclose]
        )
        is None
    )


def test_coffee_press_contact_requires_force_instead_of_rgb_motion() -> None:
    from adaptive import remote_driver

    no_contact = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [85.0, 110.0],
        "requested_target_role": "control_target",
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "close",
        "telemetry_summary": {
            "end_effector_wrench": {
                "force": {"delta_n": [0.123, -0.197, -0.218]}
            }
        },
        "mean_absolute_rgb_change": {
            "left": 10.03,
            "right": 8.85,
            "wrist": 27.30,
        },
    }
    contact = {
        **no_contact,
        "telemetry_summary": {
            "end_effector_wrench": {"force": {"delta_n": [2.0, 0.0, 0.0]}}
        },
    }

    assert remote_driver._coffee_press_effect([no_contact]) == {
        "press_count": 1,
        "max_force_delta_n": 0.319,
        "max_external_rgb_change": 10.03,
        "effect_observed": False,
    }
    assert remote_driver._coffee_press_effect([contact])["effect_observed"] is True

    context = dataclasses.replace(
        _context(), task="StartCoffeeMachine", family="control"
    )
    remote_driver._capture_coffee_press_origin(context, _state(), no_contact)
    assert context.coffee_press_world_eef_m is None
    assert remote_driver._coffee_post_press_retreat_status(
        [no_contact]
    )["press_effect_observed"] is False
    remote_driver._capture_coffee_press_origin(context, _state(), contact)
    assert context.coffee_press_world_eef_m == [0.0, 0.0, 0.5]


def test_coffee_retreat_persists_and_reuses_first_qwen_endpoint() -> None:
    from adaptive import remote_driver

    context = dataclasses.replace(
        _context(), task="StartCoffeeMachine", family="control"
    )
    context.coffee_press_world_eef_m = [0.0, 0.0, 0.5]
    first_targets = {
        "joint1": -0.5,
        "joint2": -0.8661980667309608,
        "joint3": 0.14862015852551763,
        "joint4": -2.482998517302737,
        "joint5": 0.03236437557719854,
        "joint6": 2.317830651640853,
        "joint7": 0.2815887269615587,
        "gripper": 1.0,
    }
    first_command = {
        "kind": "move_joints",
        "observation_id": "obs-first-retreat",
        "targets": first_targets,
        "tracking_mode": "actual_relative",
        "note": (
            "milestone=verify_goal; continue the lateral shoulder-yaw "
            "retreat until the sealed clearance threshold is reached"
        ),
    }
    press_state = _state()
    press_state["state.arm_joint_position"] = [
        -0.08,
        -0.94,
        -0.06,
        -2.5,
        0.06,
        2.28,
        0.05,
    ]
    assert (
        remote_driver._coffee_retreat_direction_violation(
            context, first_command, press_state
        )
        is None
    )
    across_array = {
        **first_command,
        "targets": {**first_targets, "joint1": 0.35},
    }
    direction_violation = remote_driver._coffee_retreat_direction_violation(
        context, across_array, press_state
    )
    assert direction_violation is not None
    assert "decrease joint1" in direction_violation

    remote_driver._capture_coffee_retreat_endpoint(context, first_command)

    assert context.coffee_retreat_targets == first_targets
    assert context.coffee_retreat_targets is not first_targets
    assert (
        remote_driver._coffee_retreat_endpoint_violation(context, first_command)
        is None
    )

    drifted_command = {
        **first_command,
        "observation_id": "obs-later-retreat",
        "targets": {**first_targets, "joint3": 0.225, "joint5": 0.081},
    }
    violation = remote_driver._coffee_retreat_endpoint_violation(
        context, drifted_command
    )
    assert violation is not None
    assert "first Qwen-authored retreat endpoint" in violation
    assert json.dumps(
        first_targets, sort_keys=True, separators=(",", ":")
    ) in violation

    remote_driver._capture_coffee_retreat_endpoint(context, drifted_command)
    assert context.coffee_retreat_targets == first_targets
    instruction = remote_driver._controller_coffee_retreat_instruction(
        "BASE",
        {
            "press_effect_observed": True,
            "retreat_distance_m": 0.08,
            "required_clearance_m": 0.14,
            "clearance_reached": False,
        },
        first_endpoint=context.coffee_retreat_targets,
    )
    assert "FIRST_QWEN_AUTHORED_RETREAT_ENDPOINT" in instruction
    assert json.dumps(
        first_targets, sort_keys=True, separators=(",", ":")
    ) in instruction
    assert "Reissue this exact endpoint" in instruction


def test_coffee_retreat_closure_captures_then_enforces_first_endpoint() -> None:
    from adaptive import remote_driver

    context = dataclasses.replace(
        _context(), task="StartCoffeeMachine", family="control"
    )
    context.coffee_press_world_eef_m = [0.0, 0.0, 0.5]
    context.milestone_history = ["observe", "approach", "engage", "actuate"]
    first_targets = {
        "joint1": -0.05,
        "joint2": -1.0,
        "joint3": 0.0,
        "joint4": -2.2,
        "joint5": 0.0,
        "joint6": 1.5,
        "joint7": 0.7,
        "gripper": 1.0,
    }
    note = (
        "milestone=verify_goal; continue the lateral shoulder-yaw "
        "retreat until the sealed clearance threshold is reached"
    )
    first = {
        "kind": "move_joints",
        "observation_id": "obs-retreat-1",
        "targets": first_targets,
        "tracking_mode": "actual_relative",
        "note": note,
    }
    wrong_direction = {
        **first,
        "targets": {**first_targets, "joint1": 0.35},
    }
    changed = {
        **first,
        "observation_id": "obs-retreat-2",
        "targets": {**first_targets, "joint3": 0.05},
    }
    repeated = {
        **first,
        "observation_id": "obs-retreat-2",
    }
    wrapped = remote_driver._make_proposal_audit_client_class(
        _FakeClient, context
    )()
    _FakeClient.controller_outputs = [wrong_direction, first, changed, repeated]
    _FakeClient.critic_outputs = [_proposal_audit(), _proposal_audit()]

    first_response = _proposal_complete(
        wrapped, "obs-retreat-1", _state(), _images()
    )
    after_first = _state(-0.02)
    receipt = _sealed_critic_receipt(first, after_first)
    second_response = _proposal_complete(
        wrapped,
        "obs-retreat-2",
        after_first,
        _images(),
        receipts=[receipt],
    )

    assert first_response.command == first
    assert second_response.command == repeated
    assert context.coffee_retreat_targets == first_targets
    first_observation_records = [
        record
        for record in context.proposal_records
        if record["observation_id"] == "obs-retreat-1"
    ]
    assert [record["status"] for record in first_observation_records] == [
        "rejected_by_protocol",
        "approved_for_execution",
    ]
    second_observation_records = [
        record
        for record in context.proposal_records
        if record["observation_id"] == "obs-retreat-2"
    ]
    assert [record["status"] for record in second_observation_records] == [
        "rejected_by_protocol",
        "approved_for_execution",
    ]
    assert second_observation_records[0]["contradiction"] == (
        "tracking_not_settled"
    )
    second_instructions = [
        call[1]["instruction"]
        for call in _FakeClient.calls
        if call[0] == "controller"
        and call[1]["observation_id"] == "obs-retreat-2"
    ]
    assert len(second_instructions) == 2
    assert all(
        "FIRST_QWEN_AUTHORED_RETREAT_ENDPOINT" in instruction
        for instruction in second_instructions
    )


def test_aligned_coffee_receipt_selects_strict_preclose_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive import remote_driver
    from adaptive.joint_runner import (
        COFFEE_CONTROL_PRECLOSE_NOTE,
        controller_control_preclose_schema_for_observation,
    )

    monkeypatch.setattr(
        remote_driver, "prepare_joint_mailbox", lambda *_args, **_kwargs: None
    )
    aligned = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [85.0, 110.0],
        "requested_target_role": "control_target",
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "open",
        "end_effector_external_pixel_displacement": {
            "right": {"end_px": [81.0, 107.0]},
        },
    }
    preclose = {
        "kind": "move_joints",
        "observation_id": "obs-control-press",
        "targets": {"gripper": 0.0},
        "note": COFFEE_CONTROL_PRECLOSE_NOTE,
    }
    context = dataclasses.replace(
        _context(), task="StartCoffeeMachine", family="control"
    )
    context.milestone_history = ["observe", "approach", "engage"]
    wrapped = remote_driver._make_proposal_audit_client_class(
        _FakeClient, context
    )()
    _FakeClient.controller_outputs = [preclose]
    _FakeClient.critic_outputs = [_proposal_audit()]

    response = _proposal_complete(
        wrapped,
        "obs-control-press",
        _state(),
        _images(),
        receipts=[aligned],
    )

    assert response.command == preclose
    assert _FakeClient.calls[0][1]["response_schema"] == (
        controller_control_preclose_schema_for_observation("obs-control-press")
    )
    assert "CONTROL_PRECLOSE_READY_CONTEXT" in _FakeClient.calls[0][1]["instruction"]
    critic_payload = json.loads(_FakeClient.calls[1][1]["instruction"])
    assert critic_payload["control_preclose_rule"]["gripper_close_only"] is True


def test_preclosed_coffee_receipt_selects_strict_press_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive import remote_driver
    from adaptive.joint_runner import (
        COFFEE_CONTROL_PRECLOSE_NOTE,
        controller_control_press_schema_for_observation,
    )

    monkeypatch.setattr(
        remote_driver, "prepare_joint_mailbox", lambda *_args, **_kwargs: None
    )
    aligned = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [85.0, 110.0],
        "requested_target_role": "control_target",
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "open",
        "end_effector_external_pixel_displacement": {
            "right": {"end_px": [75.0, 101.0]},
        },
    }
    preclose = {
        "kind": "move_joints",
        "note": COFFEE_CONTROL_PRECLOSE_NOTE,
        "requested_targets": {"gripper": 0.0},
        "gripper_intent": 0.0,
        "gripper_residual": {"measured_end_finger_separation": 0.0024},
    }
    press = {
        "kind": "image_servo",
        "observation_id": "obs-control-press",
        "camera": "right",
        "target_pixel": [85.0, 110.0],
        "target_role": "control_target",
        "depth_delta_m": 0.02,
        "step_m": 0.02,
        "gripper": "close",
        "note": "milestone=actuate; press one aligned coffee button",
    }
    context = dataclasses.replace(
        _context(), task="StartCoffeeMachine", family="control"
    )
    context.milestone_history = ["observe", "approach", "engage"]
    wrapped = remote_driver._make_proposal_audit_client_class(
        _FakeClient, context
    )()
    _FakeClient.controller_outputs = [press]
    _FakeClient.critic_outputs = [_proposal_audit()]

    response = _proposal_complete(
        wrapped,
        "obs-control-press",
        _state(),
        _images(),
        receipts=[aligned, preclose],
    )

    assert response.command == press
    assert _FakeClient.calls[0][1]["response_schema"] == (
        controller_control_press_schema_for_observation("obs-control-press")
    )
    assert "CONTROL_PRESS_READY_CONTEXT" in _FakeClient.calls[0][1]["instruction"]
    critic_payload = json.loads(_FakeClient.calls[1][1]["instruction"])
    assert critic_payload["control_press_rule"]["camera_ray_press"] is True


def test_no_force_coffee_press_retries_with_strict_same_target_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive import remote_driver
    from adaptive.joint_runner import (
        COFFEE_CONTROL_PRECLOSE_NOTE,
        controller_control_press_schema_for_observation,
    )

    monkeypatch.setattr(
        remote_driver, "prepare_joint_mailbox", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        remote_driver,
        "_coffee_button_candidates",
        lambda *_args, **_kwargs: {
            "left": [],
            "right": [
                {"u_min": 83, "u_max": 88, "v_min": 117, "v_max": 121}
            ],
        },
    )
    aligned = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [85.0, 110.0],
        "requested_target_role": "control_target",
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "open",
        "end_effector_external_pixel_displacement": {
            "right": {"end_px": [72.64, 104.44]},
        },
    }
    preclose = {
        "kind": "move_joints",
        "note": COFFEE_CONTROL_PRECLOSE_NOTE,
        "requested_targets": {"gripper": 0.0},
        "gripper_intent": 0.0,
        "gripper_residual": {"measured_end_finger_separation": 0.0024},
    }
    no_force_press = {
        "kind": "image_servo",
        "requested_camera": "right",
        "requested_target_pixel": [85.0, 110.0],
        "requested_target_role": "control_target",
        "requested_depth_delta_m": 0.01,
        "requested_gripper": "close",
        "end_effector_external_pixel_displacement": {
            "right": {"end_px": [81.47, 105.35]},
        },
        "telemetry_summary": {
            "end_effector_wrench": {
                "force": {"delta_n": [0.123, -0.197, -0.218]}
            }
        },
        "mean_absolute_rgb_change": {"left": 10.03, "right": 8.85},
    }
    retry_status = remote_driver._coffee_press_retry_ready_status(
        [aligned, preclose, no_force_press]
    )
    assert retry_status == {
        "camera": "right",
        "target_pixel": [85.0, 110.0],
        "alignment_error_px": 5.838,
        "preclosed_finger_separation_m": 0.0024,
        "prior_press_force_confirmed": False,
        "prior_press_force_delta_n": 0.319,
    }

    retry = {
        "kind": "image_servo",
        "observation_id": "obs-control-retry",
        "camera": "right",
        "target_pixel": [85.0, 110.0],
        "target_role": "control_target",
        "depth_delta_m": 0.01,
        "step_m": 0.01,
        "gripper": "close",
        "note": "milestone=actuate; continue the aligned coffee-button press",
    }
    button_hop = {
        **retry,
        "target_pixel": [85.0, 119.0],
        "note": "milestone=actuate; try the lower-left coffee button instead",
    }
    context = dataclasses.replace(
        _context(), task="StartCoffeeMachine", family="control"
    )
    context.milestone_history = ["observe", "approach", "engage", "actuate"]
    wrapped = remote_driver._make_proposal_audit_client_class(
        _FakeClient, context
    )()
    _FakeClient.controller_outputs = [button_hop, retry]
    _FakeClient.critic_outputs = [_proposal_audit()]

    response = _proposal_complete(
        wrapped,
        "obs-control-retry",
        _state(),
        _images(),
        receipts=[aligned, preclose, no_force_press],
    )

    assert response.command == retry
    assert [record["status"] for record in context.proposal_records] == [
        "rejected_by_protocol",
        "approved_for_execution",
    ]
    assert context.proposal_records[0]["contradiction"] == (
        "visual_alignment_unverified"
    )
    assert _FakeClient.calls[0][1]["response_schema"] == (
        controller_control_press_schema_for_observation("obs-control-retry")
    )
    controller_instructions = [
        call[1]["instruction"]
        for call in _FakeClient.calls
        if call[0] == "controller"
    ]
    assert all(
        "CONTROL_PRESS_READY_CONTEXT" in instruction
        for instruction in controller_instructions
    )
    assert all(
        "COFFEE_BUTTON_CANDIDATES" not in instruction
        for instruction in controller_instructions
    )
    assert "must reuse the sealed Qwen-authored coffee button" in (
        controller_instructions[1]
    )
    critic_payload = json.loads(_FakeClient.calls[-1][1]["instruction"])
    assert (
        critic_payload["control_press_rule"][
            "prior_no_force_press_requires_continuation"
        ]
        is True
    )


def test_contact_confirmation_rule_covers_a_stationary_close_receipt() -> None:
    from adaptive import remote_driver

    stationary_close = {
        "kind": "move_joints",
        "requested_targets": {"gripper": 0.0},
        "gripper_intent": 0.0,
        "gripper_residual": {"measured_end_finger_separation": 0.0611, "qpos_delta": [-0.0186, 0.0007]},
    }
    confirm = {
        "kind": "move_joints",
        "observation_id": "obs-confirm",
        "targets": {"gripper": 0.0},
        "note": "milestone=actuate; confirm the obstruction with a stationary close",
    }
    payload = json.loads(
        remote_driver._proposal_critic_instruction(
            _context(), task_instruction="open the toaster oven door", observation_id="obs-confirm",
            draft=confirm, claimed_milestone="actuate", public_state=_state(), images=_images(),
            immediate_prior_receipt=stationary_close, recent_receipts=[stationary_close],
        )
    )
    assert payload["contact_confirmation_rule"] == {
        "arm_motion_prohibited": True,
        "contact_persistence_is_future_evidence": True,
        "default_verdict": "approve",
        "gripper_close_only": True,
    }


def test_approach_servo_rule_defaults_to_approve_before_alignment_exists() -> None:
    from adaptive import remote_driver

    servo = {
        "kind": "image_servo",
        "observation_id": "obs-first",
        "camera": "right",
        "target_pixel": [100.0, 112.0],
        "target_role": "fixture_handle",
        "depth_delta_m": 0.0,
        "step_m": 0.03,
        "gripper": "open",
        "note": "milestone=approach; first servo toward the rack handle",
    }
    payload = json.loads(
        remote_driver._proposal_critic_instruction(
            _context(), task_instruction="toast", observation_id="obs-first", draft=servo,
            claimed_milestone="approach", public_state=_state(), images=_images(),
            immediate_prior_receipt=None, recent_receipts=[],
        )
    )
    assert payload["approach_servo_rule"]["default_verdict"] == "approve"
    assert payload["approach_servo_rule"]["zero_depth_is_still_lateral_motion"] is True
    payload = json.loads(
        remote_driver._proposal_critic_instruction(
            _context(), task_instruction="toast", observation_id="obs-first", draft={**servo, "gripper": "close"},
            claimed_milestone="approach", public_state=_state(), images=_images(),
            immediate_prior_receipt=None, recent_receipts=[],
        )
    )
    assert payload["approach_servo_rule"] is None


def test_final_coffee_contact_waypoint_closes_actuate_before_retreat() -> None:
    from adaptive.critic_protocol import (
        milestone_from_note,
        validate_milestone_transition,
    )
    from adaptive.joint_runner import (
        controller_control_contact_path_schema_for_observation,
        prepare_joint_mailbox,
    )

    schema = controller_control_contact_path_schema_for_observation(
        "obs-final-contact", waypoint_index=13
    )
    milestone = milestone_from_note(schema["properties"]["note"]["const"])
    assert milestone == "actuate"
    history = ["observe", "approach", "engage"]
    validate_milestone_transition("control", milestone, history)
    history.append(milestone)
    validate_milestone_transition("control", "verify_goal", history)

    properties = schema["properties"]
    command = {
        key: value["const"] for key, value in properties.items() if key != "targets"
    }
    command["targets"] = {
        key: value["const"] for key, value in properties["targets"]["properties"].items()
    }
    mailbox, _ = prepare_joint_mailbox(
        command, source="controller", observation_id="obs-final-contact",
        current_qpos=[-0.034, -0.911, -0.145, -2.769, 0.103, 3.035, -0.057],
        current_gripper=1.0, remaining_actions=102, sequence=17,
    )
    assert mailbox["max_actions"] == 18
    assert mailbox.get("tracking_mode", "lag_pause") == "lag_pause"


def _source_atlas_rgb_fixture() -> dict[str, bytes]:
    from io import BytesIO

    from PIL import Image

    images = {}
    for name, channel in (("left", 11), ("right", 29), ("wrist", 47)):
        view = Image.new("RGB", (256, 256))
        view.putdata([(u, v, channel) for v in range(256) for u in range(256)])
        output = BytesIO()
        view.save(output, format="PNG")
        images[name] = output.getvalue()
    return images


def test_source_atlas_covers_every_original_pixel_continuously() -> None:
    from io import BytesIO

    from PIL import Image

    from adaptive import remote_driver

    raw = _source_atlas_rgb_fixture()
    transformed, layout = remote_driver._source_observation_atlas(raw)
    assert set(transformed) == {"left", "right", "wrist"}
    assert transformed["wrist"] == raw["wrist"]
    expected_panels = [
        ([0, 0, 256, 256], [0, 0, 256, 256], 1),
        ([0, 0, 256, 256], [288, 32, 800, 544], 2),
    ]
    assert [
        (p["original_box_xyxy"], p["atlas_box_xyxy"], p["scale"])
        for p in layout["panels"]
    ] == expected_panels
    for camera, channel in (("left", 11), ("right", 29)):
        atlas = Image.open(BytesIO(transformed[camera]))
        assert atlas.size == (816, 544)
        assert atlas.crop((0, 0, 256, 256)).tobytes() == Image.open(BytesIO(raw[camera])).tobytes()
        for original, displayed, scale in expected_panels[1:]:
            # Every original pixel, including quadrant seams and frame edges,
            # appears at its independently derived 2x atlas position.
            for v in range(original[1], original[3]):
                for u in range(original[0], original[2]):
                    x = displayed[0] + (u - original[0]) * scale
                    y = displayed[1] + (v - original[1]) * scale
                    assert atlas.getpixel((x, y)) == (u, v, channel)
                    assert atlas.getpixel((x + 1, y + 1)) == (u, v, channel)


def test_source_atlas_critic_marks_original_target_without_losing_full_view() -> None:
    from io import BytesIO

    from PIL import Image

    from adaptive import remote_driver

    raw = _source_atlas_rgb_fixture()
    draft = {"kind": "image_servo", "camera": "right", "target_pixel": [150, 170]}
    transformed, layout = remote_driver._source_observation_atlas(raw, draft=draft)
    right = Image.open(BytesIO(transformed["right"]))
    # Original [150,170] maps to [636,404] in the continuous full-view panel.
    assert right.getpixel((588, 372)) == (150, 170, 29)
    assert right.getpixel((608, 372)) == (255, 0, 255)
    assert right.crop((0, 0, 256, 256)).tobytes() == Image.open(BytesIO(raw["right"])).tobytes()
    assert layout["marked_target"] == {
        "camera": "right", "original_pixel": [150, 170], "atlas_pixel": [588, 372],
    }
    assert transformed["wrist"] == raw["wrist"]


@pytest.mark.parametrize("family,contact,milestone,active", [
    ("grasp_place", "empty", "observe", True),
    ("grasp_place", "none", "observe", True),
    ("grasp_place", "nonempty", "observe", True),
    ("grasp_place", "none", "transport", False),
    ("grasp_place", "empty", "transport", False),
    ("control", "empty", "observe", False),
])
def test_source_atlas_runtime_routes_both_models_and_preserves_raw_closure(
    family: str, contact: str, milestone: str, active: bool,
) -> None:
    from io import BytesIO

    from PIL import Image

    from adaptive import critic_protocol, remote_driver

    context = _context()
    context.family = family
    context.task = "PickPlaceCounterToStandMixer" if family == "grasp_place" else "StartCoffeeMachine"
    if milestone == "transport":
        context.milestone_history = ["observe", "approach", "pregrasp", "grasp", "transport"]
    receipts = _source_stationary_empty_recovery_receipts() if contact != "none" else []
    if contact == "nonempty":
        receipts[1]["gripper_residual"]["measured_end_finger_separation"] = 0.0066
    raw = _source_atlas_rgb_fixture()
    command = {
        "kind": "move_joints", "observation_id": "obs-atlas",
        "targets": {"gripper": 1.0}, "note": f"milestone={milestone}; inspect fresh RGB with open gripper",
    }
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    _FakeClient.controller_outputs = [command]
    _FakeClient.critic_outputs = [_proposal_audit()]
    response = _proposal_complete(wrapped, "obs-atlas", _state(), raw, receipts=receipts)
    controller_call, critic_call = [kwargs for _, kwargs in _FakeClient.calls]
    for model_call in (controller_call, critic_call):
        assert set(model_call["images"]) == {"left", "right", "wrist"}
        assert Image.open(BytesIO(model_call["images"]["wrist"])).crop((0,0,256,256)).tobytes() == Image.open(BytesIO(raw["wrist"])).tobytes()
        for camera in ("left", "right"):
            assert Image.open(BytesIO(model_call["images"][camera])).size == (
                (816, 544) if active else (256, 256)
            )
            if not active:
                assert model_call["images"][camera] == raw[camera]
    critic_payload = json.loads(critic_call["instruction"])
    assert ("source_observation_atlas" in critic_payload) is active
    assert ("SOURCE_OBSERVATION_ATLAS" in controller_call["instruction"]) is active
    raw_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()}
    model_hashes = {
        name: hashlib.sha256(data).hexdigest()
        for name, data in controller_call["images"].items()
    }
    assert context.proposal_records[0]["fresh_public_rgb_sha256"] == raw_hashes
    assert critic_payload["fresh_public_rgb_sha256"] == raw_hashes
    assert response.evidence["controller_input_image_sha256"] == model_hashes
    assert critic_payload["critic_input_rgb_sha256"] == {
        name: hashlib.sha256(data).hexdigest() for name, data in critic_call["images"].items()
    }
    assert context.controller_records[0]["controller_call_manifest_sha256"] == remote_driver._model_call_manifest_sha256(controller_call)
    assert context.critic_records[0]["critic_call_manifest_sha256"] == remote_driver._model_call_manifest_sha256(critic_call)
    assert response.command is command
    assert critic_protocol.validate_proposal_audit_closure(context)["approved_count"] == 1


def test_source_atlas_image_retry_keeps_original_coordinates_and_audit_binding() -> None:
    from io import BytesIO

    from PIL import Image

    from adaptive import critic_protocol, remote_driver

    context = _context()
    context.family = "grasp_place"
    context.task = "PickPlaceCounterToStandMixer"
    raw = _source_atlas_rgb_fixture()
    first = {
        **_image_servo_payload(), "observation_id": "obs-atlas-revision",
        "depth_delta_m": 0.02, "note": "milestone=observe; inspect a source point",
    }
    revised = {**first, "target_pixel": [64.0, 52.0]}
    _FakeClient.controller_outputs = [first, revised]
    _FakeClient.critic_outputs = [_proposal_audit("revise"), _proposal_audit()]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    state = _state()
    state.update(_image_servo_public_state())
    calibration = _image_servo_calibration()
    for camera in calibration.values():
        camera.update(image_width_px=256, image_height_px=256)
    response = _proposal_complete(
        wrapped, "obs-atlas-revision", state, raw,
        receipts=_source_stationary_empty_recovery_receipts(),
        camera_calibration=calibration,
    )
    assert response.command is revised
    controllers = [kw for role, kw in _FakeClient.calls if role == "controller"]
    critics = [kw for role, kw in _FakeClient.calls if role == "critic"]
    assert len(controllers) == len(critics) == 2
    for camera in ('left', 'right'):
        assert controllers[0]['images'][camera] == controllers[1]['images'][camera]
    for call in controllers:
        assert Image.open(BytesIO(call['images']['wrist'])).crop((0,0,256,256)).tobytes() == Image.open(BytesIO(raw['wrist'])).tobytes()
    assert 'ACTION_PREVIEW' in controllers[1]['instruction']
    assert critics[0]["images"]["left"] != critics[1]["images"]["left"]
    for call, target, atlas_pixel in zip(critics, [[60.0, 50.0], [64.0, 52.0]], [[408, 132], [416, 136]], strict=True):
        payload = json.loads(call["instruction"])
        assert payload["draft"]["target_pixel"] == target
        assert payload["source_observation_atlas"]["marked_target"]["atlas_pixel"] == atlas_pixel
        image = Image.open(BytesIO(call["images"]["left"]))
        assert image.getpixel((atlas_pixel[0] + 20, atlas_pixel[1])) == (255, 0, 255)
        assert image.crop((0, 0, 256, 256)).tobytes() == Image.open(BytesIO(raw["left"])).tobytes()
    assert [p["status"] for p in context.proposal_records] == ["rejected_by_critic", "approved_for_execution"]
    assert context.proposal_records[0]["fresh_public_rgb_sha256"] == context.proposal_records[1]["fresh_public_rgb_sha256"]
    closure = critic_protocol.validate_proposal_audit_closure(context)
    assert closure["rejected_count"] == 1
    assert closure["approved_count"] == 1


@pytest.mark.parametrize("family,recovery,active", [
    ("grasp_place", False, True),
    ("grasp_place", True, True),
    ("control", False, False),
])
def test_atlas_critic_effective_system_matches_image_layout_and_manifest(
    family: str, recovery: bool, active: bool,
) -> None:
    from io import BytesIO

    from PIL import Image

    from adaptive import critic_protocol, remote_driver

    context = _context()
    context.family = family
    context.task = "PickPlaceCounterToStandMixer" if family == "grasp_place" else "StartCoffeeMachine"
    static_prompt = (Path(remote_driver.__file__).parents[1] / "prompts/proposal_audit_critic.txt").read_text()
    context.critic_prompt = static_prompt
    receipts = _source_stationary_empty_recovery_receipts() if recovery else []
    if recovery:
        context.milestone_history = ["observe", "approach", "pregrasp", "grasp"]
    phase = "grasp" if recovery else "observe"
    command = {
        "kind": "move_joints", "observation_id": "obs-effective-layout",
        "targets": {"gripper": 1.0}, "note": f"milestone={phase}; inspect the fresh source view",
    }
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    _FakeClient.controller_outputs = [command]
    _FakeClient.critic_outputs = [_proposal_audit()]
    response = _proposal_complete(
        wrapped, "obs-effective-layout", _state(), _source_atlas_rgb_fixture(), receipts=receipts,
    )
    critic_call = next(kw for role, kw in _FakeClient.calls if role == "critic")
    effective = critic_call["system_prompt"]
    assert Image.open(BytesIO(critic_call["images"]["right"])).size == ((816, 544) if active else (256, 256))
    if active:
        assert "the left half is the full view" not in effective
        assert "target crop" not in effective
        assert "upper-left" in effective
        assert "816x544" in effective
        assert "rulers" in effective and "gutters" in effective
        assert "ORIGINAL u/v" in effective
        assert "continuous" in effective and "quadrants" not in effective
        assert "ORIGINAL" in effective and "256" in effective
        assert effective != static_prompt
        # Layout repair must not weaken the existing source/contact audit policy.
        assert effective.split("An `image_servo` draft is", 1)[1] == static_prompt.split("An `image_servo` draft is", 1)[1]
        assert "Revise a\npixel on background, robot body, empty destination before\ngrasp, or the wrong task entity." in effective
    else:
        assert effective == static_prompt
    assert context.critic_prompt == static_prompt
    assert context.critic_records[0]["critic_call_manifest_sha256"] == remote_driver._model_call_manifest_sha256(critic_call)
    assert response.command is command
    assert critic_protocol.validate_proposal_audit_closure(context)["approved_count"] == 1



def test_source_atlas_original_numeric_rulers_live_outside_full_image(monkeypatch) -> None:
    from io import BytesIO

    from PIL import Image, ImageDraw

    from adaptive import remote_driver

    labels = []
    horizontal_bounds = []
    original_text = ImageDraw.ImageDraw.text

    def record_text(draw, xy, text, *args, **kwargs):
        if str(text).isdigit():
            labels.append((text, tuple(xy), kwargs.get("anchor")))
            if kwargs.get("anchor") == "mt":
                horizontal_bounds.append(draw.textbbox(xy, text, font=kwargs["font"], anchor="mt"))
        return original_text(draw, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record_text)
    transformed, layout = remote_driver._source_observation_atlas(_source_atlas_rgb_fixture())
    expected = []
    for u0, v0, x0, y0 in [(0, 0, 288, 32)]:
        for offset in [*range(0, 256, 16), 255]:
            expected.append((str(u0 + offset), (x0 + 2 * offset, y0 - 26), "mt"))
            expected.append((str(v0 + offset), (x0 - 8, max(y0 + 8, min(y0 + 2 * offset, y0 + 504))), "rm"))
        image = Image.open(BytesIO(transformed["right"]))
        # Exact ticks are in the gutters, adjacent to the unchanged image pixel.
        for offset in [*range(0, 256, 16), 255]:
            assert image.getpixel((x0 + 2 * offset, y0 - 1)) == (240, 240, 240)
            assert image.getpixel((x0 - 1, y0 + 2 * offset)) == (240, 240, 240)
            assert image.getpixel((x0 + 2 * offset, y0)) == (u0 + offset, v0, 29)
            assert image.getpixel((x0, y0 + 2 * offset)) == (u0, v0 + offset, 29)
    assert labels == expected * 2
    for start in range(0, len(horizontal_bounds), 17):
        group = horizontal_bounds[start:start + 17]
        assert all(previous[2] + 2 <= following[0] for previous, following in zip(group, group[1:]))
        assert all(0 <= box[0] < box[2] <= layout["atlas_size_px"][0] for box in group)
    assert "ORIGINAL u/v" in layout["instruction"]
    assert "16" in layout["instruction"] and "gutters" in layout["instruction"]


@pytest.mark.parametrize("target,expected", [
    ([127, 127], [542, 286]),
    ([128, 128], [544, 288]),
    ([132, 189], [552, 410]),
    ([170, 195], [628, 422]),
])
def test_source_atlas_rulers_preserve_boundary_and_near_boundary_target_mapping(target, expected) -> None:
    from io import BytesIO

    from PIL import Image

    from adaptive import remote_driver

    raw = _source_atlas_rgb_fixture()
    unmarked, _ = remote_driver._source_observation_atlas(raw)
    marked, layout = remote_driver._source_observation_atlas(
        raw, draft={"kind": "image_servo", "camera": "right", "target_pixel": target},
    )
    assert layout["marked_target"]["atlas_pixel"] == expected
    image = Image.open(BytesIO(marked["right"]))
    assert image.getpixel(tuple(expected)) == (*target, 29)
    # Marker arms may clip at an image edge, but must never alter rulers or full RGB.
    before = Image.open(BytesIO(unmarked["right"]))
    assert image.crop((0, 0, 256, 256)).tobytes() == before.crop((0, 0, 256, 256)).tobytes()
    for x0, y0 in [(288, 32)]:
        for gutter in [(x0 - 32, y0 - 32, x0 + 512, y0), (x0 - 32, y0, x0, y0 + 512)]:
            assert image.crop(gutter).tobytes() == before.crop(gutter).tobytes()


def test_exact_rejected_draft_skips_duplicate_critic_without_losing_ten_controller_attempts() -> None:
    from adaptive import critic_protocol, remote_driver

    context = _context()
    context.family = "grasp_place"
    context.task = "PickPlaceCounterToStandMixer"
    command = {**_image_servo_payload(), "observation_id": "obs-repeat",
               "note": "milestone=observe; source point in fresh RGB"}
    _FakeClient.controller_outputs = [dict(command) for _ in range(10)]
    _FakeClient.critic_outputs = [_proposal_audit("revise") for _ in range(10)]
    state = _state()
    state.update(_image_servo_public_state())
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    with pytest.raises(critic_protocol.ProposalAuditExhausted):
        _proposal_complete(wrapped, "obs-repeat", state, _images(), camera_calibration=_image_servo_calibration())
    assert [role for role, _ in _FakeClient.calls].count("controller") == 10
    assert [role for role, _ in _FakeClient.calls].count("critic") == 1
    assert context.critic_attempts == context.critic_successes == 1
    assert context.model_calls == 11
    assert context.proposal_revisions_used == 9
    assert [p["status"] for p in context.proposal_records] == ["rejected_by_critic"] + ["rejected_by_protocol"] * 9
    for record in context.proposal_records[1:]:
        assert record["contradiction"] == "stagnation"
        assert record["suggested_correction"] == "revise_approach"
        for field in ["critic_call_index", "critic_request_sha256", "critic_call_manifest_sha256",
                      "critic_attempt_record_sha256", "audit", "audit_sha256"]:
            assert record[field] is None
        assert record["controller_attempt_record_sha256"] in {
            attempt["record_sha256"] for attempt in context.attempt_records if attempt["role"] == "controller"
        }
    assert len(context.critic_records) == 1
    assert len(context.attempt_records) == 11
    closure = critic_protocol.validate_proposal_audit_closure(context)
    assert closure["rejected_count"] == 10
    assert closure["approved_count"] == 0
    assert all(p["executed"] is False and p["mailbox_count"] == 0 for p in context.proposal_records)


@pytest.mark.parametrize("change", [
    {"note": "milestone=observe; corrected description of the same source point"},
    {"depth_delta_m": 0.01},
    {"step_m": 0.01},
    {"other_view_pixel": [40.0, 50.0]},
    {"target_pixel": [61.0, 50.0]},
])
def test_rejected_source_draft_changes_remain_eligible_for_real_critic(change) -> None:
    from adaptive import critic_protocol, remote_driver

    context = _context()
    context.family = "grasp_place"
    context.task = "PickPlaceCounterToStandMixer"
    first = {**_image_servo_payload(), "observation_id": "obs-change",
             "note": "milestone=observe; source point in fresh RGB"}
    second = {**first, **change}
    _FakeClient.controller_outputs = [first, second]
    _FakeClient.critic_outputs = [_proposal_audit("revise"), _proposal_audit()]
    state = _state()
    state.update(_image_servo_public_state())
    calibration = _image_servo_calibration()
    calibration["right"]["camera_position_world_m"] = [0.2, 0.0, 0.0]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    response = _proposal_complete(wrapped, "obs-change", state, _images(), camera_calibration=calibration)
    assert response.command is second
    assert [role for role, _ in _FakeClient.calls] == ["controller", "critic", "controller", "critic"]
    assert context.critic_records[0]["critic_call_manifest_sha256"] != context.critic_records[1]["critic_call_manifest_sha256"]
    assert len(context.attempt_records) == 4
    closure = critic_protocol.validate_proposal_audit_closure(context)
    assert closure["rejected_count"] == closure["approved_count"] == 1


def test_new_observation_audits_previously_rejected_motor_payload() -> None:
    from adaptive import remote_driver

    context = _context()
    first = _milestone_command("obs-old", "observe", 0.01)
    approved = _milestone_command("obs-old", "observe", 0.02)
    next_command = {**first, "observation_id": "obs-new"}
    _FakeClient.controller_outputs = [first, approved, next_command]
    _FakeClient.critic_outputs = [_proposal_audit("revise"), _proposal_audit(), _proposal_audit()]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    _proposal_complete(wrapped, "obs-old", _state(), _images())
    receipt = _sealed_critic_receipt(approved, _state(0.02))
    response = _proposal_complete(wrapped, "obs-new", _state(0.02), _images(), receipts=[receipt])
    assert response.command is next_command
    assert context.critic_attempts == 3
    assert context.proposal_records[-1]["revision_index"] == 0
    assert context.proposal_records[-1]["status"] == "approved_for_execution"
    assert context.proposal_records[-1]["critic_attempt_record_sha256"] == context.attempt_records[-1]["record_sha256"]


def test_repeat_revision_guidance_preserves_supported_same_pixel_and_note_corrections() -> None:
    from adaptive import remote_driver

    draft = _image_servo_payload()
    text = remote_driver._proposal_revision_instruction(
        "base", contradiction="motion_effect_mismatch", evidence=["external_rgb"],
        suggested_correction="revise_alignment", confidence="high", rejected_draft=draft,
        rejected_history=[draft], required_observation_id="obs",
    )
    assert "camera, pixels, or targets equal" not in text
    assert "unchanged complete draft" in text
    assert "same critic input" in text
    assert "note, point, depth, step, stereo" in text
    assert "preserve the command kind and every motor-bearing field exactly" in text
    record = {}
    advice = remote_driver._protocol_rejection(
        record, error=ValueError("unchanged rejected draft has identical critic input on this observation"),
    )
    assert advice["contradiction"] == "stagnation"
    assert advice["suggested_correction"] == "revise_approach"
    assert advice["evidence"] == []



def _wrist_contact_calibration():
    cameras = _image_servo_calibration()
    cameras["wrist"] = {**cameras["left"], "image_width_px": 256,
                        "image_height_px": 256, "cx_px": 127.5, "cy_px": 127.5}
    return cameras


def test_wrist_tool_view_uses_public_projection_and_preserves_raw_panel():
    from io import BytesIO

    from PIL import Image

    from adaptive import remote_driver
    raw = _source_atlas_rgb_fixture()
    state = _image_servo_public_state()
    transformed, layout = remote_driver._source_contact_wrist_view(
        raw, state, _wrist_contact_calibration())
    assert layout["current_tool_pixel"] == [127.5, 127.5]
    assert layout["marker_meaning"] == "current measured grip-site; not an object target"
    assert transformed["left"] == raw["left"] and transformed["right"] == raw["right"]
    view = Image.open(BytesIO(transformed["wrist"]))
    original = Image.open(BytesIO(raw["wrist"]))
    assert view.size == (768, 512)
    assert view.crop((0, 0, 256, 256)).tobytes() == original.tobytes()
    # The marker's center stays open; cyan arms identify the robot reference.
    assert view.getpixel((511, 255)) == original.getpixel((127, 127))
    assert view.getpixel((521, 255)) == (0, 220, 255)
    moved = {**state, "state.base_position": [0.1, 0.0, 0.0]}
    _, moved_layout = remote_driver._source_contact_wrist_view(raw, moved, _wrist_contact_calibration())
    assert moved_layout["current_tool_pixel"] == [137.5, 127.5]


@pytest.mark.parametrize("family,milestone,active", [
    ("grasp_place", "pregrasp", True), ("grasp_place", "approach", False),
    ("control", "observe", False), ("grasp_place", "transport", False),
])
def test_wrist_tool_runtime_routes_same_public_reference_without_motor_changes(family, milestone, active):
    from adaptive import critic_protocol, remote_driver
    context = _context()
    context.family = family
    context.task = "StartCoffeeMachine" if family == "control" else "PickPlaceCounterToStandMixer"
    phases = ["observe", "approach", "pregrasp", "grasp", "transport"]
    context.milestone_history = phases[:phases.index(milestone) + 1]
    raw = _source_atlas_rgb_fixture()
    state = _state(); state.update(_image_servo_public_state())
    command = {"kind": "move_joints", "observation_id": "obs-wrist-tool",
               "targets": {"gripper": 1.0}, "note": f"milestone={milestone}; inspect current public view"}
    _FakeClient.controller_outputs = [command]
    _FakeClient.critic_outputs = [_proposal_audit()]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    response = _proposal_complete(wrapped, "obs-wrist-tool", state, raw,
                                  camera_calibration=_wrist_contact_calibration())
    controller, critic = [kwargs for _, kwargs in _FakeClient.calls]
    assert response.command is command
    assert ("SOURCE_CONTACT_WRIST_VIEW" in controller["instruction"]) is active
    packet = json.loads(critic["instruction"])
    assert ("source_contact_wrist_view" in packet) is active
    from io import BytesIO
    from PIL import Image
    for call in (controller, critic):
        assert Image.open(BytesIO(call["images"]["wrist"])).crop((0,0,256,256)).tobytes() == Image.open(BytesIO(raw["wrist"])).tobytes()
    assert "ACTION_PREVIEW" in controller["instruction"]
    assert "action_preview" in packet
    assert context.proposal_records[0]["fresh_public_rgb_sha256"] == critic_protocol.image_hashes(raw)
    assert response.evidence["controller_input_image_sha256"] == critic_protocol.image_hashes(controller["images"])
    assert packet["critic_input_rgb_sha256"] == critic_protocol.image_hashes(critic["images"])
    assert critic_protocol.validate_proposal_audit_closure(context)["approved_count"] == 1
    if active:
        assert "Wrist is unchanged." not in controller["instruction"]
        assert "Wrist is unchanged." not in critic["instruction"]
        assert "wrist RGB is unchanged" not in critic["system_prompt"]
        assert "current measured grip-site" in critic["system_prompt"]



@pytest.mark.parametrize("milestone,active", [("pregrasp", True), ("approach", True), ("observe", False)])
def test_measured_receipt_context_omits_only_prior_controller_narration(milestone, active):
    from adaptive import critic_protocol, remote_driver

    context = _context(); context.family = "grasp_place"; context.task = "PickPlaceCounterToStandMixer"
    context.milestone_history = ["observe"] + ([] if milestone=="observe" else ["approach"]) + (["pregrasp"] if milestone=="pregrasp" else [])
    state = _state(); state.update(_image_servo_public_state())
    previous = {"kind": "move_joints", "observation_id": "obs-before",
                "targets": {"joint1": 0.01}, "note": f"milestone={milestone}; old unverified alignment claim"}
    receipt = _sealed_critic_receipt(previous, state)
    before = json.loads(json.dumps(receipt))
    command = {"kind": "move_joints", "observation_id": "obs-measured-context",
               "targets": {"gripper": 1.0}, "note": f"milestone={milestone}; inspect current view"}
    _FakeClient.controller_outputs = [command]; _FakeClient.critic_outputs = [_proposal_audit()]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    response = _proposal_complete(wrapped, "obs-measured-context", state, _source_atlas_rgb_fixture(),
                                  receipts=[receipt], camera_calibration=_wrist_contact_calibration())
    controller, critic = [kwargs for _, kwargs in _FakeClient.calls]
    packet, _ = json.JSONDecoder().raw_decode(controller["instruction"])
    if active:
        latest = packet["recent_receipts"][0]
        assert latest["end_effector_pose_delta"] == before["end_effector_pose_delta"]
        assert latest["gripper_residual"] == before["gripper_residual"]
        assert "note" not in latest and "telemetry_summary" not in latest
    else: assert packet["recent_receipts"] == [before]
    assert receipt == before
    assert json.loads(critic["instruction"])["immediate_prior_receipt"] == before
    assert response.command is command
    assert critic_protocol.validate_proposal_audit_closure(context)["approved_count"] == 1


# Recorded public arm pose before the failed Mixer source close.
_NONMODEL_CONTACT_QPOS = [-0.609147961110709, -1.3759474752954173, -0.7255610945162245, -2.503964246346215, -0.1849259842129682, 1.4651741633262962, -0.7068809581550901]


@pytest.mark.parametrize("orientation_weight", [1.0, 0.25])
def test_posture_centering_does_not_change_requested_task_twist(orientation_weight):
    from adaptive.cartesian_skill import _nullspace_centering_delta
    from adaptive.panda_embodiment import panda_local_jacobian

    jac = panda_local_jacobian(_NONMODEL_CONTACT_QPOS)
    rows = jac["translation"] + [[orientation_weight * x for x in row] for row in jac["rotation"]]
    correction = _nullspace_centering_delta(rows, _NONMODEL_CONTACT_QPOS)
    assert max(abs(sum(a*b for a,b in zip(row,correction))) for row in rows) < 1e-10
    assert max(abs(x) for x in correction) <= 0.1000000001


@pytest.mark.parametrize("translation", [(0.01,0,0), (-0.01,0,0), (0,0.01,0), (0,-0.01,0), (0,0,0.01), (0,0,-0.02)])
def test_contact_cartesian_resolution_tracks_small_requested_translation(translation):
    from adaptive.cartesian_skill import CartesianDeltaCommand, resolve_cartesian_delta
    from adaptive.panda_embodiment import panda_fk, panda_local_jacobian

    jac = panda_local_jacobian(_NONMODEL_CONTACT_QPOS)
    state = {"state.arm_joint_position": _NONMODEL_CONTACT_QPOS,
             "state.arm_translation_jacobian": jac["translation"], "state.arm_rotation_jacobian": jac["rotation"]}
    command = CartesianDeltaCommand("obs-fixed-diagnostic",translation,(0,0,0),"open","fixed public pose diagnostic")
    result = resolve_cartesian_delta(command,state,current_gripper=1)
    before = panda_fk(_NONMODEL_CONTACT_QPOS).position_m
    after = panda_fk(result.joint_endpoint).position_m
    error = sum((b-a-t)**2 for a,b,t in zip(before,after,translation))**0.5
    assert error < 0.003


@pytest.mark.parametrize("translation,rotation", [
    ((0.01,0,0),(0,0,0)), ((0,0.01,0),(0,0,0)), ((0,0,-0.02),(0,0,0)),
    ((0.01,0,0),(0,0,0.03)), ((0,0,0),(0.02,0,0)),
])
def test_cartesian_endpoint_refines_the_requested_physical_pose(translation, rotation):
    from adaptive.cartesian_skill import CartesianDeltaCommand, resolve_cartesian_delta
    from adaptive.panda_embodiment import (
        _rotation_log_vector,
        _rotation_times_transpose,
        panda_fk,
        panda_local_jacobian,
    )

    q = _NONMODEL_CONTACT_QPOS
    j = panda_local_jacobian(q)
    state = {"state.arm_joint_position":q,"state.arm_translation_jacobian":j["translation"],"state.arm_rotation_jacobian":j["rotation"]}
    result = resolve_cartesian_delta(CartesianDeltaCommand("pose-probe",translation,rotation,"open","public pose test"),state,current_gripper=1)
    before = panda_fk(q); after = panda_fk(result.joint_endpoint)
    actual_translation = [b-a for a,b in zip(before.position_m,after.position_m)]
    actual_rotation = _rotation_log_vector(_rotation_times_transpose(after.rotation_matrix,before.rotation_matrix))
    assert sum((a-b)**2 for a,b in zip(actual_translation,translation))**.5 < .0003
    assert sum((a-b)**2 for a,b in zip(actual_rotation,rotation))**.5 < .001
    assert result.predicted_delta == pytest.approx([*actual_translation,*actual_rotation],abs=1e-9)


def test_long_malformed_retry_feedback_is_bounded_without_mutating_evidence():
    from adaptive import remote_driver
    raw = '{"kind":"image_servo"}' + r"\n" * 16000
    context = _context()
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    valid = _milestone_command("obs-0", "observe", 0.02)
    _FakeClient.controller_errors_after_log = [_FakeMalformedResponse("malformed json"), None]
    _FakeClient.controller_attempt_records = [[
        _attempt_record("a" * 64, observation_id="obs-0",
                        response_schema_sha256=_json_sha256(CONTROLLER_RESPONSE_SCHEMA))
        | {"sanitized_raw_command": raw}
    ]]
    _FakeClient.controller_outputs = [valid]
    _FakeClient.critic_outputs = [_proposal_audit()]
    response = _proposal_complete(wrapped, "obs-0", _state(), _images())
    controller = [kw for role, kw in _FakeClient.calls if role == "controller"]
    assert response.command is valid
    assert context.proposal_records[0]["draft"] == raw
    assert context.attempt_records[0]["record"]["sanitized_raw_command"] == raw
    assert len(controller[1]["instruction"]) - len(controller[0]["instruction"]) < 12000
    assert context.proposal_revisions_used == 1
    assert sum(role == "critic" for role, _ in _FakeClient.calls) == 1


@pytest.mark.parametrize("draft", ["short invalid JSON", {"kind": "image_servo", "target_pixel": [128, 192]}])
def test_bounded_malformed_retry_preserves_normal_drafts(draft):
    from adaptive import remote_driver
    value = remote_driver._proposal_revision_instruction(
        "base", contradiction="invalid_proposal", evidence=[],
        suggested_correction="revise_proposal", confidence="high",
        rejected_draft=draft, required_observation_id="obs-0")
    assert "rejected_draft_omitted_chars" not in value
    assert _json_sha256(draft) in value


def test_failure_skills_learn_empty_close_once_and_retrieve_relevant_cards():
    from adaptive import remote_driver
    context = _context()
    context.family = "grasp_place"
    context.milestone_history = ["observe", "approach", "pregrasp", "grasp"]
    context.proposal_observation_id = "current"
    receipts = _source_stationary_empty_recovery_receipts()
    receipts[1]["observation_id"] = "failed-close"
    first = remote_driver._failure_skill_packet(context, receipts)
    second = remote_driver._failure_skill_packet(context, receipts)
    assert first == second
    assert [s["id"] for s in first["skills"]] == ["source-contact-verification", "empty-close-information-gain"]
    assert len(context.failure_skill_memory) == 1
    assert first["recent_cases"][0]["observation_id"] == "failed-close"
    assert first["recent_cases"][0]["outcome"] == "empty_source_close"
    assert len(json.dumps(first)) < 4500
    assert first["recent_cases"][0]["failed_approach"]["target_pixel"] == [100.0, 128.0]
    held = {**receipts[1], "observation_id": "held", "gripper_residual": {"measured_end_finger_separation": 0.035}}
    assert remote_driver._failure_skill_packet(context, [*receipts, held]) is None
    assert len(context.failure_skill_memory) == 1


def test_failure_skills_do_not_change_unrelated_control_input():
    from adaptive import remote_driver
    context = _context()
    context.family = "control"
    context.task = "StartCoffeeMachine"
    assert remote_driver._failure_skill_packet(context, []) is None


def test_failure_skills_record_malformed_as_unexecuted_without_raw_text():
    from adaptive import remote_driver
    context = _context()
    context.proposal_observation_id = "obs-format"
    context.proposal_records = [{"observation_id": "obs-format", "proposal_index": 1, "status": "rejected_by_protocol", "failure_class": "MalformedResponse", "draft": "x" * 30000, "contradiction": "invalid_proposal"}]
    packet = remote_driver._failure_skill_packet(context, [])
    assert [s["id"] for s in packet["skills"]] == ["malformed-proposal-recovery"]
    assert packet["recent_cases"][0]["outcome"] == "proposal_not_executed"
    assert "x" * 100 not in json.dumps(packet)


def test_failure_skills_reach_controller_while_command_and_critic_authority_stay_intact():
    from adaptive import remote_driver
    context = _context()
    context.task = "PickPlaceCounterToStandMixer"
    context.family = "grasp_place"
    context.milestone_history = ["observe", "approach", "pregrasp", "grasp"]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    receipts = _source_stationary_empty_recovery_receipts()
    for index in (0, -1):receipts[index]["requested_target_pixel"] = [60.0, 50.0]
    revised = {**_image_servo_payload(), "observation_id": "obs-memory", "camera": "right", "target_pixel": [60, 50], "depth_delta_m": 0.02, "gripper": "open", "note": "milestone=grasp; inspect changed depth"}
    _FakeClient.controller_outputs = [revised]
    _FakeClient.critic_outputs = [_proposal_audit()]
    state = _state();state.update(_image_servo_public_state())
    response = _proposal_complete(wrapped, "obs-memory", state, _images(), receipts=receipts, camera_calibration=_image_servo_calibration())
    controller = next(kw for role, kw in _FakeClient.calls if role == "controller")
    critic = next(kw for role, kw in _FakeClient.calls if role == "critic")
    assert "QWEN_FAILURE_SKILLS" in controller["instruction"]
    assert "QWEN_FAILURE_SKILLS" not in critic["instruction"]
    assert response.command is revised
    assert context.failure_skill_uses[-1]["controller_call_index"] == 1
    assert context.proposal_records[-1]["critic_origin_execution"] is False


def test_failure_skills_retrieve_prior_episode_memory():
    from adaptive import remote_driver
    context = _context()
    context.family = "grasp_place"
    context.milestone_history = ["observe", "approach", "pregrasp", "grasp"]
    packet = remote_driver._failure_skill_packet(context, _source_stationary_empty_recovery_receipts())
    assert packet["prior_cases"]
    assert all(case["kind"] == "empty_source_close" for case in packet["prior_cases"])
    assert "target_pixel" not in json.dumps(packet["prior_cases"])


def test_failure_skills_remember_unresolved_empty_close_beyond_receipt_window():
    from adaptive import remote_driver
    context = _context()
    context.family = "grasp_place"
    context.milestone_history = ["observe", "approach", "pregrasp", "grasp"]
    receipts = _source_stationary_empty_recovery_receipts()
    receipts[1]["observation_id"] = "empty-close"
    packet = remote_driver._failure_skill_packet(context, receipts)
    aged = remote_driver._failure_skill_packet(context, receipts[-1:])
    assert aged is not None
    assert aged["recent_cases"] == packet["recent_cases"]
    assert "empty-close-information-gain" in {s["id"] for s in aged["skills"]}
    held = {**receipts[1], "observation_id": "later-nonempty-close", "gripper_residual": {"measured_end_finger_separation": 0.035}}
    assert remote_driver._failure_skill_packet(context, [held]) is None
    assert remote_driver._failure_skill_packet(context, receipts[-1:]) is None
    assert len(context.failure_skill_memory) == 1


@pytest.mark.parametrize("approved_attempt", [6, 10])
def test_controller_can_recover_after_previous_five_attempt_limit(approved_attempt) -> None:
    from adaptive import critic_protocol, remote_driver

    context = _context()
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    commands = [_milestone_command("obs-0", "observe", (i + 1) * 0.001)
                for i in range(approved_attempt)]
    _FakeClient.controller_outputs = list(commands)
    _FakeClient.critic_outputs = ([_proposal_audit("revise") for _ in range(approved_attempt - 1)]
                                 + [_proposal_audit()])
    response = _proposal_complete(wrapped, "obs-0", _state(), _images())
    assert response.command is commands[-1]
    assert context.proposal_revisions_used == approved_attempt - 1
    assert len(context.proposal_records) == approved_attempt
    assert all(record["executed"] is False and record["mailbox_count"] == 0
               for record in context.proposal_records[:-1])
    closure = critic_protocol.validate_proposal_audit_closure(context)
    assert closure["rejected_count"] == approved_attempt - 1
    assert closure["approved_count"] == 1


def test_critic_revision_drops_resolved_previous_protocol_error() -> None:
    from adaptive import remote_driver

    context = _context()
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    invalid = _milestone_command("obs-0", "observe", 0.001)
    invalid["note"] = "missing milestone claim"
    _FakeClient.controller_outputs = [invalid,
                                      _milestone_command("obs-0", "observe", 0.002),
                                      _milestone_command("obs-0", "observe", 0.003)]
    _FakeClient.critic_outputs = [_proposal_audit("revise"), _proposal_audit()]
    response = _proposal_complete(wrapped, "obs-0", _state(), _images())
    inputs = [kw["instruction"] for role, kw in _FakeClient.calls if role == "controller"]
    assert '"protocol_rejection":' in inputs[1]
    assert '"protocol_rejection":' not in inputs[2]
    assert [r["status"] for r in context.proposal_records] == [
        "rejected_by_protocol", "rejected_by_critic", "approved_for_execution"]
    assert response.command["targets"] == {"joint1": 0.003}


def test_simulator_selects_renderer_that_frees_old_context_before_reset(monkeypatch, tmp_path):
    from adaptive import joint_sim_child

    class CapturedFactory(BaseException):
        pass

    calls = []
    gym = ModuleType("gymnasium")
    def make(*args, **kwargs):
        calls.append((args, kwargs))
        raise CapturedFactory()
    gym.make = make
    active_skills = ModuleType("robocasa_inspect.active_skills")
    active_skills.TerminalOutcomePersister = object
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    monkeypatch.setitem(sys.modules, "robocasa", ModuleType("robocasa"))
    monkeypatch.setitem(sys.modules, "robocasa_inspect.active_skills", active_skills)
    monkeypatch.setattr(joint_sim_child, "_network_denied", lambda: "OSError")
    monkeypatch.setattr(joint_sim_child, "_install_joint_controller", lambda: None)
    with pytest.raises(CapturedFactory):
        joint_sim_child.run("StoreLeftoversByType", 7, tmp_path / "run")
    assert calls == [(("robocasa/StoreLeftoversByType",),
                      {"split": "pretrain", "seed": 7, "renderer": "mujoco"})]



@pytest.mark.parametrize("milestone,active", [("pregrasp", True), ("approach", True), ("observe", False)])
def test_source_controller_alignment_is_measured_pixel_not_contact(monkeypatch, milestone, active):
    from adaptive import remote_driver

    context = _context(); context.family = "grasp_place"
    context.milestone_history = ["observe"] + ([] if milestone=="observe" else ["approach"]) + (["pregrasp"] if milestone=="pregrasp" else [])
    state = _state()
    previous = {"kind": "move_joints", "observation_id": "before-alignment",
        "targets": {"joint1": 0.01}, "note": f"milestone={milestone}; old request"}
    receipt = _sealed_critic_receipt(previous, state)
    alignment = {"camera": "right", "target_pixel": [128,192], "error_px": 0.04,
        "within_one_grid_cell": True, "controller_rule": "alignment gate is satisfied: advance now"}
    old_instruction = _instruction
    def instruction(receipts=None):
        value = json.loads(old_instruction(receipts)); value["image_servo_alignment"] = alignment
        return json.dumps(value)
    monkeypatch.setitem(globals(), "_instruction", instruction)
    command = {"kind": "move_joints", "observation_id": "after-alignment",
        "targets": {"gripper": 1.0}, "note": f"milestone={milestone}; inspect fresh geometry"}
    _FakeClient.controller_outputs = [command]; _FakeClient.critic_outputs = [_proposal_audit()]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    response = _proposal_complete(wrapped, "after-alignment", state, _images(), receipts=[receipt])
    packet = json.JSONDecoder().raw_decode(_FakeClient.calls[0][1]["instruction"])[0]
    if active:
        assert "image_servo_alignment" not in packet
    else: assert packet["image_servo_alignment"] == alignment
    assert response.command is command


@pytest.mark.parametrize("family,active", [("grasp_place", True), ("articulated", False)])
def test_source_critic_alignment_keeps_measurement_scope(family, active):
    from adaptive import remote_driver

    context = _context(); context.family = family
    context.milestone_history = ["observe", "approach", "pregrasp"] if active else ["observe", "approach", "engage"]
    receipt = {"kind": "image_servo", "requested_camera": "right", "requested_target_pixel": [128.0,192.0],
        "end_effector_external_pixel_displacement": {"right": {"end_px": [127.99,192.04]}}}
    packet = json.loads(remote_driver._proposal_critic_instruction(context,
        task_instruction="move the visible source", observation_id="current",
        draft=_cartesian_milestone_command("current", "pregrasp" if active else "engage", 0.01),
        claimed_milestone="pregrasp" if active else "engage", public_state=_state(), images=_images(),
        immediate_prior_receipt=receipt))["image_servo_alignment"]
    assert packet["target_pixel"] == [128.0,192.0]
    assert packet["error_px"] == pytest.approx(math.hypot(0.01,0.04))
    assert packet["within_one_grid_cell"] is True
    assert ("previously requested pixel" in packet["controller_rule"]) is active



@pytest.mark.parametrize("milestone,active", [("pregrasp", True), ("approach", True), ("observe", False)])
def test_source_controller_uses_latest_physical_outcome_without_old_target(milestone, active):
    from adaptive import remote_driver

    context = _context(); context.family = "grasp_place"; context.task = "PickPlaceCounterToStandMixer"
    context.milestone_history = ["observe"] + ([] if milestone=="observe" else ["approach"]) + (["pregrasp"] if milestone=="pregrasp" else [])
    state = _state()
    previous = {"kind": "move_joints", "observation_id": "prior",
        "targets": {"joint1": 0.01}, "note": f"milestone={milestone}; old target"}
    receipt = _sealed_critic_receipt(previous, state)
    receipt.update(requested_target_pixel=[37,53], requested_camera="left", requested_depth_delta_m=0.02,
        end_effector_pose_delta={"translation_m": [0,0,0.01]})
    older = {**receipt, "observation_id": "older"}
    before = json.loads(json.dumps([older,receipt]))
    command = {"kind": "move_joints", "observation_id": "fresh",
        "targets": {"gripper": 1.0}, "note": f"milestone={milestone}; inspect"}
    _FakeClient.controller_outputs = [command]; _FakeClient.critic_outputs = [_proposal_audit()]
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    response = _proposal_complete(wrapped, "fresh", state, _images(), receipts=[older,receipt])
    controller,critic = [kwargs for _,kwargs in _FakeClient.calls]
    value = json.JSONDecoder().raw_decode(controller["instruction"])[0]
    if active:
        assert len(value["recent_receipts"]) == 1
        latest=value["recent_receipts"][0]
        assert latest["end_effector_pose_delta"] == receipt["end_effector_pose_delta"]
        assert latest["gripper_residual"] == receipt["gripper_residual"]
        assert "requested_target_pixel" not in latest and "requested_camera" not in latest
        assert "telemetry_summary" not in latest
    else: assert value["recent_receipts"] == before
    assert [older,receipt] == before
    assert json.loads(critic["instruction"])["immediate_prior_receipt"] == before[-1]
    assert response.command is command


def test_controller_failure_view_keeps_measurement_without_mutating_memory():
    from adaptive import remote_driver

    failure = {"camera":"left", "target_pixel":[37,53], "depth_delta_m":0.02,
        "other_view_pixel":None, "measured_end_finger_separation":0.0024}
    before=json.loads(json.dumps(failure))
    assert remote_driver._source_close_measurement(failure) == {"measured_end_finger_separation":0.0024}
    assert failure == before


@pytest.fixture(autouse=True)
def source_perception_boundary(monkeypatch):
    """Existing proposal-unit scenarios isolate the separately tested perception calls."""
    from adaptive import remote_driver as driver
    original = getattr(driver, '_maybe_source_perception', None)
    monkeypatch.setattr(driver, '_maybe_source_perception', lambda *a, **k: None, raising=False)
    return original


def test_live_perception_only_source_stage_and_never_reuses_stale_geometry(monkeypatch, source_perception_boundary):
    from adaptive import remote_driver as driver
    context = CriticContext(task='PickPlaceCounterToDrawer', family='grasp_place', run=Path('/tmp/unused'), critic_prompt='', max_decisions=70)
    calls = []
    def fake(context, client, **kwargs):
        calls.append(kwargs['observation_id'])
        return {'observation_id': kwargs['observation_id']}
    monkeypatch.setattr(driver, '_source_perception_packet', fake)
    kwargs = dict(observation_id='one', task_instruction='pick the ladle', images={}, public_state={}, camera_calibration={})
    context.milestone_history = ['observe']
    assert source_perception_boundary(context, None, **kwargs) == {'observation_id': 'one'}
    context.milestone_history = ['approach']
    assert source_perception_boundary(context, None, **kwargs) == {'observation_id': 'one'}
    context.milestone_history = ['pregrasp']
    assert source_perception_boundary(context, None, **kwargs) == {'observation_id': 'one'}
    assert source_perception_boundary(context, None, **kwargs) == {'observation_id': 'one'}
    assert calls == ['one']
    kwargs['observation_id'] = 'two'
    assert source_perception_boundary(context, None, **kwargs) == {'observation_id': 'two'}
    context.family = 'control'
    kwargs['observation_id'] = 'three'
    assert source_perception_boundary(context, None, **kwargs) is None
    assert context.source_perception_packet is None
    assert calls == ['one', 'two']


def test_live_perception_geometry_uses_only_qwen_selected_points(monkeypatch):
    from adaptive import remote_driver as driver
    calibration = {'left': {'fx_px': 200, 'fy_px': 200, 'cx_px': 128, 'cy_px': 128,
        'camera_position_world_m': [0, 0, 1], 'camera_xmat_world': [[1,0,0],[0,1,0],[0,0,1]]},
        'wrist': {'fx_px': 200, 'fy_px': 200, 'cx_px': 128, 'cy_px': 128,
        'camera_position_world_m': [.2, 0, 1], 'camera_xmat_world': [[1,0,0],[0,1,0],[0,0,1]]}}
    selected = [{'camera':'left','selected_pixel':[128,128],'qwen_observation':{'visible':True}},
                {'camera':'wrist','selected_pixel':[88,128],'qwen_observation':{'visible':True}}]
    state = {'state.base_rotation':[0,0,0,1], 'state.base_position':[0,0,0], 'state.end_effector_position_relative':[0,0,.2]}
    packet = driver._source_landmark_geometry(selected, state, calibration)
    assert packet['estimated_landmark_position_robot_base_m'] == pytest.approx([0,0,0], abs=1e-9)
    assert packet['current_end_effector_position_robot_base_m'] == [0,0,.2]
    selected[1]['selected_pixel'] = [88,200]
    assert driver._source_landmark_geometry(selected, state, calibration) is None
    assert driver._source_landmark_geometry(selected[:1], state, calibration) is None
    selected[1]['selected_pixel'] = [128,128]
    assert driver._source_landmark_geometry(selected, state, calibration) is None


@pytest.mark.parametrize('missing_log', [False, True])
def test_live_perception_records_every_call_and_requires_linked_output(tmp_path, missing_log):
    from adaptive import remote_driver as driver
    context = _context()
    context.run = tmp_path
    context.proposal_image_sha256 = driver.image_hashes(_images())
    result = {'visible': True, 'keypoint_id': 3, 'description': 'object edge'}
    class Peer:
        def complete(self, **kwargs):
            if not missing_log:
                kwargs['attempt_log'].append(_default_attempt_record('perception', kwargs, sanitized_raw_command=json.dumps(result)))
            return SimpleNamespace(command=result, evidence={'served_model_id':'qwen-fake-served-model', 'image_sha256':driver.image_hashes(kwargs['images'])})
    kwargs = {'observation_id':'one', 'system_prompt':'locate', 'instruction':'locate the object',
              'images':_images(), 'public_state':{}, 'response_schema':{'type':'object'}, 'max_tokens':512}
    actual = driver._source_perception_call(context, Peer(), stage='coarse', kwargs=kwargs)
    assert context.model_calls == 1
    assert context.controller_calls == context.critic_attempts == 0
    assert context.attempt_records == context.proposal_records == []
    assert len(context.source_perception_records) == 1
    assert actual == (None if missing_log else result)
    closure = driver._validate_source_perception_evidence(context, 'qwen-fake-served-model')
    assert closure == {'valid':True,'calls':1,'completed':0 if missing_log else 1}
    assert json.loads((tmp_path/'source-perception/call-0001-coarse/result.json').read_text())['status'] == ('unavailable' if missing_log else 'complete')
    if not missing_log:
        context.model_calls = 0
        with pytest.raises(ValueError, match='omit source perception'):
            driver._validate_source_perception_evidence(context, 'qwen-fake-served-model')
        context.model_calls = 1
        with pytest.raises(ValueError, match='evidence drifted'):
            driver._validate_source_perception_evidence(context, 'another-model')


def test_live_perception_leaves_final_wall_budget_for_control(tmp_path):
    from adaptive import remote_driver as driver
    context = _context(); context.run = tmp_path; context.source_perception_deadline = 0
    class Peer:
        def complete(self, **kwargs):
            raise AssertionError('no perception call after deadline')
    assert driver._source_perception_call(context, Peer(), stage='coarse', kwargs={}) is None
    assert context.model_calls == 0 and context.source_perception_records == []


def test_live_perception_checks_alternative_camera_after_coarse_false_positive(monkeypatch):
    from adaptive import remote_driver as driver
    context = _context()
    stages = []
    visible = {'visible':True,'description':'ladle','point_2d':[500,500]}
    responses = {'coarse':{'target':'ladle','left':visible,'right':visible,'wrist':visible},
                 'left':{'visible':False,'description':'occluded','keypoint_id':None},
                 'right':{'visible':True,'description':'ladle bowl','keypoint_id':1},
                 'wrist':{'visible':True,'description':'ladle bowl','keypoint_id':1}}
    def call(context, client, *, stage, **kwargs):
        stages.append(stage)
        return responses[stage]
    monkeypatch.setattr(driver, '_source_perception_call', call)
    monkeypatch.setattr(driver, '_source_landmark_images', lambda image, center: ({}, [{'id':1,'pixel':[128,128] if image==b'right' else [88,128]}], [0,0,128,128]))
    calibration = {'right':{'fx_px':200,'fy_px':200,'cx_px':128,'cy_px':128,'camera_position_world_m':[0,0,1],'camera_xmat_world':[[1,0,0],[0,1,0],[0,0,1]]},
                   'wrist':{'fx_px':200,'fy_px':200,'cx_px':128,'cy_px':128,'camera_position_world_m':[.2,0,1],'camera_xmat_world':[[1,0,0],[0,1,0],[0,0,1]]}}
    state = {'state.base_rotation':[0,0,0,1], 'state.base_position':[0,0,0], 'state.end_effector_position_relative':[0,0,.2]}
    result = driver._source_perception_packet(context, None, observation_id='one', task_instruction='pick the ladle',
                                            images={'left':b'left','right':b'right','wrist':b'wrist'}, public_state=state, camera_calibration=calibration)
    assert result is not None
    assert stages == ['coarse','wrist','left','right']
    assert result['public_geometry_from_qwen_landmarks']['camera_pair'] == ['wrist','right']
    assert [selection['camera'] for selection in result['observation']] == ['wrist','right']
    assert result['public_geometry_from_qwen_landmarks']['estimated_landmark_position_robot_base_m'] == pytest.approx([0,0,0],abs=1e-9)


def test_live_perception_rejects_pixel_unstable_intersection():
    from adaptive import remote_driver as driver
    calibration = {'left':{'fx_px':200,'fy_px':200,'cx_px':128,'cy_px':128,'camera_position_world_m':[0,0,1], 'camera_xmat_world':[[1,0,0],[0,1,0],[0,0,1]]},
                   'wrist':{'fx_px':200,'fy_px':200,'cx_px':128,'cy_px':128,'camera_position_world_m':[.2,0,1], 'camera_xmat_world':[[1,0,0],[0,1,0],[0,0,1]]}}
    state = {'state.base_rotation':[0,0,0,1],'state.base_position':[0,0,0],'state.end_effector_position_relative':[0,0,.2]}
    selected = [{'camera':'left','selected_pixel':[128,128],'qwen_observation':{'visible':True}},
                {'camera':'wrist','selected_pixel':[118,128],'qwen_observation':{'visible':True}}]
    # The rays meet exactly at depth4m, but one pixel changes depth by >.3m.
    assert driver._source_landmark_geometry(selected,state,calibration) is None
    selected[1]['selected_pixel'] = [88,128]
    assert driver._source_landmark_geometry(selected,state,calibration)['estimated_landmark_position_robot_base_m'] == pytest.approx([0,0,0],abs=1e-9)

def test_source_stereo_explicit_wrist_camera_is_resolved_and_preserved():
    from adaptive import remote_driver
    from adaptive.image_servo import decode_image_servo, resolve_image_servo
    from adaptive.joint_runner import _command_dict
    from adaptive.panda_embodiment import project_world_point
    calibration = _stereo_calibration()
    calibration['wrist'] = {**calibration['left'], 'camera_position_world_m':[.25,0,0]}
    point = [.1,.05,-1.]
    left = project_world_point(point, calibration['left'])
    wrist = project_world_point(point, calibration['wrist'])
    payload = {**_image_servo_payload(), 'target_role':'source_object',
               'target_pixel':[left['u_px'],left['v_px']],
               'other_view_pixel':[wrist['u_px'],wrist['v_px']], 'other_view_camera':'wrist'}
    command = decode_image_servo(payload, observation_id='obs')
    assert command.other_view_camera == 'wrist'
    resolution = resolve_image_servo(command, _image_servo_public_state(), calibration, current_gripper=1.)
    assert resolution.stereo_target_base_m == pytest.approx(point)
    assert resolution.stereo_ray_gap_m == pytest.approx(0.,abs=1e-9)
    assert _command_dict(command)['other_view_camera'] == 'wrist'
    assert remote_driver._normalized_controller_command(payload, observation_id='obs')['other_view_camera'] == 'wrist'


@pytest.mark.parametrize('change', [{'other_view_pixel':None}, {'target_role':'fixture_handle'}, {'other_view_camera':'left'}])
def test_wrist_secondary_requires_a_source_stereo_request(change):
    from adaptive.image_servo import decode_image_servo
    payload = {**_image_servo_payload(), 'target_role':'source_object',
               'other_view_pixel':[35,45], 'other_view_camera':'wrist', **change}
    with pytest.raises(ValueError, match='secondary'):
        decode_image_servo(payload, observation_id='obs')


def test_explicit_wrist_stereo_receipt_binds_camera_through_audit() -> None:
    from adaptive.joint_runner import (
        PUBLIC_IMAGE_SERVO_RECEIPT_FIELDS,
        STEREO_RECEIPT_FIELDS,
        finalize_image_servo_receipt,
        prepare_joint_mailbox,
        validate_public_receipt,
    )
    from adaptive.panda_embodiment import project_world_point

    state, evidence, _unused_calibration = _stationary_receipt_boundary_inputs()
    state.update(_image_servo_public_state())
    state["state.arm_joint_position"] = _reset_qpos()
    evidence["end_effector_external_pixels_before"] = state[
        "state.end_effector_external_pixels"
    ]
    evidence["end_effector_external_pixels_after"] = state[
        "state.end_effector_external_pixels"
    ]
    calibration = _stereo_calibration()
    calibration["wrist"] = {**calibration["left"], "camera_position_world_m":[.25,0,0]}
    point = (0.1, 0.05, -1.0)
    left_px = project_world_point(point, calibration["left"])
    right_px = project_world_point(point, calibration["wrist"])
    payload = {
        **_image_servo_payload(),
        "target_pixel": [left_px["u_px"], left_px["v_px"]],
        "other_view_pixel": [right_px["u_px"], right_px["v_px"]],
        "target_role": "source_object",
        "other_view_camera": "wrist",
        "step_m": 0.03,
    }
    mailbox, command = prepare_joint_mailbox(
        payload,
        source="controller",
        observation_id="obs",
        current_qpos=_reset_qpos(),
        current_gripper=1.0,
        public_state=state,
        camera_calibration=calibration,
        remaining_actions=160,
        sequence=0,
    )
    assert command.other_view_pixel is not None
    execution = {
        "kind": "move_joints",
        "accepted": True,
        "bounded_endpoint": mailbox["endpoint"],
        "gripper_intent": 1.0,
        "step_count": 4,
        "maximum_commanded_step": 0.01,
        "endpoint_error": max(
            abs(target - actual)
            for target, actual in zip(mailbox["endpoint"], _reset_qpos(), strict=True)
        ),
        "minimum_hard_limit_margin": 0.5,
        "tracking_pause_count": 0,
        **evidence,
    }
    receipt = finalize_image_servo_receipt(
        command,
        mailbox,
        execution,
        before_state=state,
        after_state=state,
        before_calibration=calibration,
        after_calibration=calibration,
    )
    assert set(receipt) - STEREO_RECEIPT_FIELDS - {"requested_other_view_camera"} == PUBLIC_IMAGE_SERVO_RECEIPT_FIELDS
    assert receipt["requested_other_view_pixel"] == [
        pytest.approx(right_px["u_px"]),
        pytest.approx(right_px["v_px"]),
    ]
    assert receipt["stereo_ray_gap_m"] == pytest.approx(0.0, abs=1e-9)
    validated = validate_public_receipt(receipt)
    assert validated["requested_other_view_pixel"] == receipt["requested_other_view_pixel"]
    assert validated["stereo_ray_gap_m"] == pytest.approx(0.0, abs=1e-9)

    from adaptive import critic_protocol, remote_driver
    assert receipt['requested_other_view_camera'] == 'wrist'
    assert validated['requested_other_view_camera'] == 'wrist'
    assert remote_driver._receipt_matches_command(receipt, payload)
    assert critic_protocol._receipt_command_mismatch(receipt, payload) is None
    tampered = {**receipt, 'requested_other_view_camera':None}
    assert not remote_driver._receipt_matches_command(tampered, payload)
    assert critic_protocol._receipt_command_mismatch(tampered, payload) == 'command_mismatch'
    assert critic_protocol.source_grasp_approach_target([receipt])['other_view_camera'] == 'wrist'
    from adaptive.joint_runner import _seal_receipt_rgb_change
    sealed = _seal_receipt_rgb_change(receipt, {'left':1.,'right':1.,'wrist':1.}, source_sequence=0, fresh_sequence=1)
    assert sealed['requested_other_view_camera'] == 'wrist'

def test_qwen_wrist_stereo_draft_survives_full_proposal_review():
    from adaptive import remote_driver
    context = _context(); context.task = 'PickPlaceCounterToStandMixer'; context.family = 'grasp_place'
    context.milestone_history = ['observe','approach','pregrasp']
    calibration = _stereo_calibration()
    calibration['wrist'] = {**calibration['left'], 'camera_position_world_m':[.25,0,0]}
    command = {**_image_servo_payload(), 'observation_id':'current', 'target_role':'source_object',
               'target_pixel':[60,45], 'other_view_pixel':[35,45], 'other_view_camera':'wrist',
               'note':'milestone=pregrasp; approach the source in left and wrist views'}
    _FakeClient.controller_outputs = [command]
    _FakeClient.critic_outputs = [_proposal_audit()]
    state = _state(); state.update(_image_servo_public_state())
    wrapped = remote_driver._make_proposal_audit_client_class(_FakeClient, context)()
    response = _proposal_complete(wrapped, 'current', state, _source_atlas_rgb_fixture(), camera_calibration=calibration)
    assert response.command == command
    assert context.proposal_records[-1]['draft']['other_view_camera'] == 'wrist'
    assert context.proposal_records[-1]['status'] == 'approved_for_execution'
    critic_input = json.loads(_FakeClient.calls[-1][1]['instruction'])
    assert critic_input['draft']['other_view_camera'] == 'wrist'


@pytest.mark.parametrize("actually_visible", [True, False])
def test_focused_source_verifies_views_after_coarse_false_negative(monkeypatch, actually_visible):
    from adaptive import remote_driver as driver
    context = _context(); stages = []; centers = []
    missing = {'visible':False, 'description':'not visible', 'point_2d':None}
    responses = {'coarse':{'target':'ladle', 'left':missing, 'right':missing, 'wrist':missing},
                 **{camera:{'visible':actually_visible and camera != 'left', 'description':'verified image',
                            'keypoint_id':1 if actually_visible and camera != 'left' else None}
                    for camera in ['wrist','left','right']}}
    def call(context, client, *, stage, **kwargs):
        stages.append(stage)
        if stage != 'coarse':
            assert kwargs['details']['crop_center_source'] == 'image_center_fallback'
            assert kwargs['details']['crop_center_from_qwen'] is None
        return responses[stage]
    def marked(image, center):
        centers.append(center)
        return {}, [{'id':1, 'pixel':[128,128] if image==b'right' else [88,128]}], [64,64,192,192]
    monkeypatch.setattr(driver, '_source_perception_call', call)
    monkeypatch.setattr(driver, '_source_landmark_images', marked)
    calibration = {'right':{'fx_px':200,'fy_px':200,'cx_px':128,'cy_px':128,'camera_position_world_m':[0,0,1],'camera_xmat_world':[[1,0,0],[0,1,0],[0,0,1]]},
                   'wrist':{'fx_px':200,'fy_px':200,'cx_px':128,'cy_px':128,'camera_position_world_m':[.2,0,1],'camera_xmat_world':[[1,0,0],[0,1,0],[0,0,1]]}}
    state = {'state.base_rotation':[0,0,0,1], 'state.base_position':[0,0,0], 'state.end_effector_position_relative':[0,0,.2]}
    result = driver._source_perception_packet(context, None, observation_id='one', task_instruction='pick the ladle',
        images={'left':b'left','right':b'right','wrist':b'wrist'}, public_state=state, camera_calibration=calibration)
    assert stages == ['coarse','wrist','left','right']
    assert centers == [[128.,128.]] * 3
    if actually_visible:
        assert result['public_geometry_from_qwen_landmarks']['camera_pair'] == ['wrist','right']
        assert result['public_geometry_from_qwen_landmarks']['estimated_landmark_position_robot_base_m'] == pytest.approx([0,0,0],abs=1e-9)
    else:
        assert result is None


def test_source_geometry_supplies_fresh_relative_displacement_without_action():
    from adaptive import remote_driver as driver
    calibration = {'left':{'fx_px':200,'fy_px':200,'cx_px':128,'cy_px':128,'camera_position_world_m':[0,0,1], 'camera_xmat_world':[[1,0,0],[0,1,0],[0,0,1]]},
                   'wrist':{'fx_px':200,'fy_px':200,'cx_px':128,'cy_px':128,'camera_position_world_m':[.2,0,1],'camera_xmat_world':[[1,0,0],[0,1,0],[0,0,1]]}}
    state = {'state.base_rotation':[0,0,0,1], 'state.base_position':[0,0,0], 'state.end_effector_position_relative':[-.2,.1,.3]}
    selected = [{'camera':'left','selected_pixel':[128,128],'qwen_observation':{'visible':True}},
                {'camera':'wrist','selected_pixel':[88,128],'qwen_observation':{'visible':True}}]
    result = driver._source_landmark_geometry(selected,state,calibration)
    assert result['landmark_minus_end_effector_robot_base_m'] == pytest.approx([.2,-.1,-.3],abs=1e-9)
    assert result['end_effector_to_landmark_distance_m'] == pytest.approx(math.sqrt(.14))
    assert 'translation_m' not in result and 'step_m' not in result
    state['state.end_effector_position_relative'] = [.1,.2,.3]
    fresh = driver._source_landmark_geometry(selected,state,calibration)
    assert fresh['landmark_minus_end_effector_robot_base_m'] == pytest.approx([-.1,-.2,-.3],abs=1e-9)
    assert result['landmark_minus_end_effector_robot_base_m'] == pytest.approx([.2,-.1,-.3],abs=1e-9)
