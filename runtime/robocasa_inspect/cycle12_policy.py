"""Fable-accepted Cycle 12 same-task source authorization protocol."""

from __future__ import annotations

from .cycle11_policy import (
    Cycle11Decision,
    cycle11_response_schema,
    decode_cycle11_decision,
)

SKILLS = ("authorize_source", "reobserve", "give_up")

CYCLE12_SYSTEM_PROMPT = r"""You are an Inspect-style visual source-authorizing
policy for official RoboCasa PandaOmron simulation. You receive exactly three
public images. LEFT and RIGHT each show CURRENT on the left half and a proposed
SOURCE REFERENCE on the right. The third image is unmodified CURRENT official
wrist RGB. Source pixels are a proposed motion prior, not current state, reward,
success, depth, object pose, contact, or an action.

authorize_source means the proposed source is from the same task and compatible
layout and is a legitimate current or future route state. Authorization permits
the harness to use it; the harness separately determines from public EEF error
whether servo motion is needed or the authorized source is already aligned.
Therefore authorize a valid same-task/layout source even when CURRENT is already
aligned with it. reobserve means visibility is genuinely insufficient and causes
zero motion. give_up vetoes a cross-task, different-appliance, different-object,
or incompatible-layout source and causes zero motion.

An always-authorize policy is unsafe: inspect both external panels and wrist RGB
for semantic and layout consistency. residual_id is an optional closed persistent
base-frame correction and must be zero for reobserve or give_up. gripper source
follows the authorized source. Never output coordinates, pixels, joints, done,
verify, reward, predicates, hidden state, depth, or tools. Return exactly one
schema object."""


def cycle12_response_schema(observation_token: str) -> dict[str, object]:
    schema = cycle11_response_schema(observation_token)
    schema["properties"]["skill"]["enum"] = list(SKILLS)
    return schema


def decode_cycle12_decision(
    value: dict[str, object], *, observation_token: str
) -> Cycle11Decision:
    translated = dict(value)
    skill = translated.get("skill")
    if skill not in SKILLS:
        raise ValueError("Cycle 12 skill is outside the closed vocabulary")
    translated["skill"] = (
        "execute_source" if skill == "authorize_source" else skill
    )
    decision = decode_cycle11_decision(
        translated, observation_token=observation_token
    )
    return Cycle11Decision(
        observation_token=decision.observation_token,
        skill=str(skill),
        residual_id=decision.residual_id,
        gripper=decision.gripper,
        phase=decision.phase,
        note=decision.note,
    )
