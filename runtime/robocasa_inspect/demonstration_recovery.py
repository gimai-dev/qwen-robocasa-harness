"""Fail-closed live Qwen recovery for a diverged demonstration skill."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from .contracts import ACTION_FIELDS, Command, decode_command, official_action_chunk
from .inspect_prompt import SYSTEM_PROMPT

MAX_EPISODE_WALL_S = 1_200.0
MIN_REQUEST_REMAINING_S = 135.0
MAX_RECOVERY_CHUNKS = 12
_TRANSLATION_CHUNK_SCALE_M = 0.25
_ROTATION_CHUNK_SCALE_RAD = 2.5
_TRANSLATION_LIMIT_M = 0.02
_ROTATION_LIMIT_RAD = 0.0873


@dataclass
class RecoveryGovernor:
    """Bound recovery using only realized public motion and public RGB progress."""

    discrepancies: list[float] = field(default_factory=list)
    realized_deltas: list[np.ndarray] = field(default_factory=list)
    translations: list[np.ndarray] = field(default_factory=list)
    stagnation_count: int = 0
    worsening_count: int = 0

    def observe(
        self,
        realized_eef_delta_m: object,
        discrepancy: float,
        accepted_translation_m: object,
    ) -> str | None:
        realized = np.asarray(realized_eef_delta_m, dtype=np.float64)
        translation = np.asarray(accepted_translation_m, dtype=np.float64)
        if (
            realized.shape != (3,)
            or translation.shape != (3,)
            or not np.isfinite(realized).all()
            or not np.isfinite(translation).all()
            or not np.isfinite(discrepancy)
        ):
            raise ValueError("recovery progress evidence is invalid")
        previous = self.discrepancies[-1] if self.discrepancies else discrepancy
        improvement = previous - discrepancy
        stalled = float(np.linalg.norm(realized)) < 0.001
        self.stagnation_count = (
            self.stagnation_count + 1 if stalled and improvement < 0.01 else 0
        )
        best = min(self.discrepancies, default=discrepancy)
        self.worsening_count = (
            self.worsening_count + 1 if discrepancy - best >= 0.05 else 0
        )
        self.discrepancies.append(float(discrepancy))
        self.realized_deltas.append(realized)
        self.translations.append(translation)
        if self.worsening_count >= 2:
            return "visual_worsening"
        if self.stagnation_count >= 3:
            if len(self.translations) >= 3:
                recent = self.translations[-3:]
                pairs = list(pairwise(recent))
                alternating = (
                    len(pairs) == 2
                    and all(
                        np.linalg.norm(a) > 1e-9 and np.linalg.norm(b) > 1e-9
                        for a, b in pairs
                    )
                    and all(
                        float(np.dot(a, b))
                        <= -0.8 * float(np.linalg.norm(a) * np.linalg.norm(b))
                        for a, b in pairs
                    )
                )
                net_motion = float(np.linalg.norm(np.sum(self.realized_deltas[-3:], axis=0)))
                if alternating and net_motion < 0.002:
                    return "oscillation"
            return "stagnation"
        return None

RECOVERY_SYSTEM_PROMPT = SYSTEM_PROMPT + r"""

# Demonstration tracking recovery

The closed-loop demonstration-tracking skill stopped because its public robot-pose
tracking threshold was exceeded. You now own the live policy. Use only the current
three official RGB views, current public robot state, and sanitized receipts. No
future demonstration action, source success marker, reward, object pose, contact
state, or task predicate is available.

Return exactly one fresh `action` from the existing closed protocol, or `give_up` if
no bounded safe progress is visible. `servo_feature`, `base_action`, and `finish` are
not available in this recovery phase. For a hinged task, follow the arc with small
translations and orientation corrections rather than repeating a blocked straight push.
There are at most twelve recovery decisions and every accepted action is followed by
a fresh synchronized left, right, and wrist observation.

Wrist occlusion alone is not a reason to give up. Near a large hinged door, the wrist
camera may be blocked by the panel while both external cameras still show a broad,
safe contact and the door arc. In that case use the recent accepted public-state
motion receipts to continue the visible tangent in a small bounded increment. Give
up when the external views do not establish a safe contact or clearance. When the
context contains `suggested_live_tangent`, it is a meter/radian conversion of the
last already-executed safe receipt, not a future demonstration action. If both
external views show the fixture with no intervening obstacle, use that exact tangent
for the first recovery action and reassess the resulting fresh images. An occluded
wrist or an empty gripper does not invalidate a broad closing push visible externally.
"""

REFERENCE_RECOVERY_SYSTEM_PROMPT = r"""You are a strict RGB visual motor
controller for the official RoboCasa PandaOmron simulation. The first two images
are the fresh CURRENT official left and right external cameras. The third image is
clearly labeled REFERENCE ONLY and contains the two external views from the current
point of a successful official human trajectory. It is never current state and must
never justify finish. The current wrist camera is omitted only in this recovery
phase when a persisted public image-information metric finds it less informative.

Compare current end-effector and task geometry against the reference, then command
one small correction. Return exactly one JSON object with exactly: `kind` equal to
`action`, the supplied `observation_id`, a short visual `note`, `translation_m` as
[dx,dy,dz] in robot base coordinates (+x forward, +y left, +z up) with every
component within +/-0.020, `rotation_axis_angle_rad` as [rx,ry,rz] with every
component within +/-0.0873, and `gripper` equal to open, hold, or close. No finish,
give_up, servo, base motion, success claim, or extra field is allowed. The official
simulator supplies fresh current images after every accepted action. Use only RGB,
public robot state, and sanitized receipts; never infer reward, hidden object state,
contact truth, depth, or the terminal predicate."""


def suggested_live_tangent(
    prior_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Convert the last executed normalized receipt into the live command schema."""
    for receipt in reversed(prior_receipts):
        if not receipt.get("accepted") or receipt.get("kind") != "demo_tracking":
            continue
        translation = np.asarray(
            receipt.get("mean_normalized_translation"), dtype=np.float64
        )
        rotation = np.asarray(
            receipt.get("mean_normalized_rotation"), dtype=np.float64
        )
        if (
            translation.shape != (3,)
            or rotation.shape != (3,)
            or not np.isfinite(translation).all()
            or not np.isfinite(rotation).all()
        ):
            return None
        return {
            "translation_m": np.clip(
                translation * _TRANSLATION_CHUNK_SCALE_M,
                -_TRANSLATION_LIMIT_M,
                _TRANSLATION_LIMIT_M,
            ).tolist(),
            "rotation_axis_angle_rad": np.clip(
                rotation * _ROTATION_CHUNK_SCALE_RAD,
                -_ROTATION_LIMIT_RAD,
                _ROTATION_LIMIT_RAD,
            ).tolist(),
            "gripper": "hold",
            "basis": "last_executed_demo_tracking_receipt",
        }
    return None


def recovery_instruction(
    *,
    task: str,
    observation_id: str,
    divergence: Mapping[str, object],
    prior_receipts: Sequence[Mapping[str, object]],
) -> str:
    """Build model-visible recovery context from public, bounded evidence only."""
    public_context = {
        "task": task,
        "observation_id": observation_id,
        "tracking_skill_status": "stopped_at_frozen_public_threshold",
        "public_tracking_error": dict(divergence),
        "recent_sanitized_receipts": [dict(row) for row in prior_receipts[-3:]],
        "suggested_live_tangent": suggested_live_tangent(prior_receipts),
        "allowed_next": ["action", "give_up"],
    }
    return (
        "Continue the exact task from this fresh observation. The reference skill "
        "is no longer accessible. Inspect all three images and either make one "
        "small live end-effector correction or give up safely. Context: "
        + json.dumps(public_context, sort_keys=True, separators=(",", ":"))
    )


def reference_recovery_instruction(
    *,
    task: str,
    observation_id: str,
    divergence: Mapping[str, object],
    prior_receipts: Sequence[Mapping[str, object]],
    reference_episode_index: int,
    reference_frame_index: int,
    reference_sha256: str,
) -> str:
    if len(reference_sha256) != 64:
        raise ValueError("reference digest is invalid")
    context = {
        "task": task,
        "observation_id": observation_id,
        "public_tracking_error": dict(divergence),
        "recent_sanitized_receipts": [dict(row) for row in prior_receipts[-3:]],
        "reference_episode_index": reference_episode_index,
        "reference_frame_index": reference_frame_index,
        "reference_sha256": reference_sha256,
        "required_next": "action",
    }
    return (
        "Continue the task from the two current external images. Use the third image only "
        "as successful RGB geometry: move the current hand toward the reference "
        "contact or pose with one bounded correction. Context: "
        + json.dumps(context, sort_keys=True, separators=(",", ":"))
    )


def give_up_repair_instruction(
    *, task: str, observation_id: str, tangent: Mapping[str, object]
) -> str:
    """Require one same-frame correction when a certified tangent was ignored."""
    action = {
        "kind": "action",
        "observation_id": observation_id,
        "note": "Continue the last executed collision-free closing tangent, then reassess.",
        "translation_m": tangent["translation_m"],
        "rotation_axis_angle_rad": tangent["rotation_axis_angle_rad"],
        "gripper": tangent["gripper"],
    }
    return (
        f"Your give_up did not follow the {task} recovery contract: the current "
        "context supplied a tangent derived only from an already executed safe "
        "receipt, and the official external views remain available. Return this "
        "exact fresh action object and no other command: "
        + json.dumps(action, sort_keys=True, separators=(",", ":"))
    )


def decode_recovery_command(
    value: Mapping[str, object], *, observation_id: str
) -> tuple[Command | None, list[str]]:
    """Accept only a fresh bounded arm action or an explicit fail-closed stop."""
    changes: list[str] = []
    if set(value) == {"action"} and isinstance(value.get("action"), Mapping):
        candidate = dict(value["action"])
        changes.append("unwrapped_action_envelope")
    else:
        candidate = dict(value)
    if "kind" not in candidate:
        candidate["kind"] = "action"
        changes.append("inferred_action_kind")
    if candidate.get("kind") == "give_up":
        command = decode_command(candidate, observation_id=observation_id)
        return None, changes
    if candidate.get("kind") != "action":
        raise ValueError("recovery command must be an action or give_up")
    if set(candidate) != ACTION_FIELDS:
        raise ValueError("recovery action fields do not match the closed schema")
    command = decode_command(candidate, observation_id=observation_id)
    return command, changes


def validated_recovery_chunk(
    command: Command,
    *,
    previous_gripper: str,
    candidate_chunk: list[dict[str, np.ndarray]] | None = None,
) -> tuple[list[dict[str, np.ndarray]], str]:
    """Validate the entire five-step chunk before any simulator write."""
    generated, gripper = official_action_chunk(
        command, previous_gripper=previous_gripper
    )
    chunk = generated if candidate_chunk is None else candidate_chunk
    if len(chunk) != 5:
        raise ValueError("recovery chunk must contain exactly five actions")
    required = {
        "action.end_effector_position": 3,
        "action.end_effector_rotation": 3,
        "action.gripper_close": 1,
        "action.base_motion": 4,
        "action.control_mode": 1,
    }
    for row in chunk:
        if set(row) != set(required):
            raise ValueError("recovery chunk action fields drifted")
        for name, width in required.items():
            vector = np.asarray(row[name], dtype=np.float64).reshape(-1)
            if vector.shape != (width,) or not np.isfinite(vector).all():
                raise ValueError("recovery chunk contains invalid numerics")
            if np.any(np.abs(vector) > 1.0 + 1e-12):
                raise ValueError("recovery chunk exceeds normalized bounds")
        if np.any(np.asarray(row["action.base_motion"], dtype=np.float64)):
            raise ValueError("recovery chunk contains nonzero base motion")
        if np.any(np.asarray(row["action.control_mode"], dtype=np.float64)):
            raise ValueError("recovery chunk changes control mode")
    return chunk, gripper


def require_recovery_budget(
    *, started: float, now: float, completed_chunks: int
) -> None:
    if completed_chunks >= MAX_RECOVERY_CHUNKS:
        raise RuntimeError("recovery chunk budget exhausted")
    remaining = MAX_EPISODE_WALL_S - (now - started)
    if remaining < MIN_REQUEST_REMAINING_S:
        raise TimeoutError("insufficient remaining wall budget for Qwen recovery")
