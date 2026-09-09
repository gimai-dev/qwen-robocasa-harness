"""Fable-accepted Cycle 11 source-skill authorization protocol."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from .cycle10_policy import GRIPPERS, NOTE_PATTERN, PHASES
from .goal30_cycle10 import RESIDUAL_IDS

SKILLS = ("execute_source", "reobserve", "give_up")
OBSERVATION_TOKEN = re.compile(r"[0-9a-f]{12}")

CYCLE11_AGENCY_WORDING = (
    "Qwen's contribution is authorization and veto; its discrimination competence "
    "is evidenced by masked semantic preflight, not per-episode counterfactuals. "
    "Counted successes are source-skill successes authorized by Qwen."
)

CYCLE11_SYSTEM_PROMPT = r"""You are an Inspect-style visual source-skill policy
for the official RoboCasa PandaOmron simulation. You receive exactly three public
images. Image 1 is official LEFT with CURRENT on the left half and SOURCE
REFERENCE on the right. Image 2 is the same layout for official RIGHT. Image 3
is unmodified CURRENT official wrist RGB. The source is a proposed motion prior,
not current state, reward, success, depth, object pose, contact, or an action.

First decide whether the proposed source belongs to the same task and layout and
is the needed current or next step. execute_source authorizes the harness to
servo to that source keyframe plus your closed residual and gripper choice.
reobserve performs no robot motion and requests fresh cameras when current is
already aligned or visual evidence is temporarily uncertain. give_up performs no
motion and vetoes a mismatched task, layout, appliance, object arrangement, or
unsafe source. Never authorize a visibly cross-task or different-layout source.

residual_id selects one closed persistent base-frame correction. Prefer zero
unless CURRENT and a valid SOURCE are visibly offset. gripper source follows the
source keyframe. phase describes visible intent. Never output coordinates,
pixels, joints, done, verify, reward, predicates, hidden state, depth, or tools.
Return exactly one schema object."""


@dataclass(frozen=True)
class Cycle11Decision:
    observation_token: str
    skill: str
    residual_id: str
    gripper: str
    phase: str
    note: str


def cycle11_response_schema(observation_token: str) -> dict[str, object]:
    if OBSERVATION_TOKEN.fullmatch(observation_token) is None:
        raise ValueError("Cycle 11 observation token must be twelve lowercase hex")
    fields = {
        "observation_token": {"const": observation_token},
        "skill": {"enum": list(SKILLS)},
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


def decode_cycle11_decision(
    value: Mapping[str, object], *, observation_token: str
) -> Cycle11Decision:
    required = {
        "observation_token",
        "skill",
        "residual_id",
        "gripper",
        "phase",
        "note",
    }
    if set(value) != required:
        raise ValueError("Cycle 11 decision fields drifted")
    if value.get("observation_token") != observation_token:
        raise ValueError("Cycle 11 observation token is stale")
    skill = value.get("skill")
    residual_id = value.get("residual_id")
    gripper = value.get("gripper")
    phase = value.get("phase")
    note = value.get("note")
    if skill not in SKILLS:
        raise ValueError("Cycle 11 skill is outside the closed vocabulary")
    if residual_id not in RESIDUAL_IDS:
        raise ValueError("Cycle 11 residual is outside the closed vocabulary")
    if gripper not in GRIPPERS:
        raise ValueError("Cycle 11 gripper is outside the closed vocabulary")
    if phase not in PHASES:
        raise ValueError("Cycle 11 phase is outside the closed vocabulary")
    if not isinstance(note, str) or NOTE_PATTERN.fullmatch(note) is None:
        raise ValueError("Cycle 11 note is invalid")
    if skill != "execute_source" and residual_id != "zero":
        raise ValueError("Cycle 11 no-motion skills require zero residual")
    return Cycle11Decision(
        observation_token=observation_token,
        skill=str(skill),
        residual_id=str(residual_id),
        gripper=str(gripper),
        phase=str(phase),
        note=note,
    )
