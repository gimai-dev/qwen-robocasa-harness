"""Minimal Cycle 13 source-authorization protocol."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

SKILLS = ("authorize_source", "reobserve", "give_up")
TOKEN = re.compile(r"[0-9a-f]{12}")

CYCLE13_SYSTEM_PROMPT = r"""You are an Inspect-style source-authorizing policy
for official RoboCasa PandaOmron simulation. LEFT and RIGHT images show CURRENT
on the left and a proposed SOURCE REFERENCE on the right. The third image is
unmodified CURRENT official wrist RGB.

authorize_source when the proposal is from the same task, compatible layout, and
a legitimate current or future route state. Authorize it even if CURRENT is
already aligned; the harness separately decides whether motion is needed.
reobserve only when visibility is genuinely insufficient. give_up vetoes a
cross-task, different-appliance, different-object, or incompatible-layout source.
reobserve and give_up cause zero motion. Return exactly the required two fields.
Never output coordinates, joints, residuals, gripper, notes, tools, done, verify,
reward, predicates, depth, object pose, contact, or hidden state."""


@dataclass(frozen=True)
class Cycle13Decision:
    observation_token: str
    skill: str


def cycle13_response_schema(observation_token: str) -> dict[str, object]:
    if TOKEN.fullmatch(observation_token) is None:
        raise ValueError("Cycle 13 token must be twelve lowercase hex")
    fields = {
        "observation_token": {"const": observation_token},
        "skill": {"enum": list(SKILLS)},
    }
    return {
        "type": "object",
        "properties": fields,
        "required": list(fields),
        "additionalProperties": False,
    }


def decode_cycle13_decision(
    value: Mapping[str, object], *, observation_token: str
) -> Cycle13Decision:
    if set(value) != {"observation_token", "skill"}:
        raise ValueError("Cycle 13 decision fields drifted")
    if value.get("observation_token") != observation_token:
        raise ValueError("Cycle 13 token is stale")
    skill = value.get("skill")
    if skill not in SKILLS:
        raise ValueError("Cycle 13 skill is outside the closed vocabulary")
    return Cycle13Decision(observation_token=observation_token, skill=str(skill))


def cycle13_guided_choices(observation_token: str) -> tuple[str, ...]:
    if TOKEN.fullmatch(observation_token) is None:
        raise ValueError("Cycle 13 token must be twelve lowercase hex")
    return tuple(f"{observation_token}:{skill}" for skill in SKILLS)


def decode_cycle13_guided_choice(value: str, *, observation_token: str) -> Cycle13Decision:
    choices = cycle13_guided_choices(observation_token)
    if value not in choices:
        raise ValueError("Cycle 13 guided choice is stale or invalid")
    return Cycle13Decision(
        observation_token=observation_token,
        skill=value.split(":", 1)[1],
    )
