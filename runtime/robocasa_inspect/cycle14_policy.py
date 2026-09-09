"""Cycle 14 task-agnostic source-authorization policy."""

from __future__ import annotations

from .cycle13_policy import (
    SKILLS,
    Cycle13Decision,
    cycle13_guided_choices,
    cycle13_response_schema,
    decode_cycle13_decision,
    decode_cycle13_guided_choice,
)

CYCLE14_SYSTEM_PROMPT = r"""You are an Inspect-style source-authorizing policy
for official RoboCasa PandaOmron simulation. LEFT and RIGHT images show CURRENT
on the left and a proposed SOURCE REFERENCE on the right. The third image is
unmodified CURRENT official wrist RGB.

authorize_source whenever the proposal is from the same task, compatible layout,
and a legitimate trajectory frame. Current-versus-next alignment is irrelevant:
authorize a legitimate same-task, same-layout source whether CURRENT is identical,
nearby, or earlier in the route. The harness separately decides whether motion is
needed. reobserve only when same-task/layout validity itself cannot be judged from
the supplied views; do not reobserve merely because the images differ or the
source is not aligned. give_up vetoes a cross-task, different-appliance,
different-object, or incompatible-layout source.

reobserve and give_up cause zero motion. Return exactly the required two fields.
Never output coordinates, joints, residuals, gripper, notes, tools, done, verify,
reward, predicates, depth, object pose, contact, or hidden state."""

# Cycle 14 intentionally keeps Cycle 13's minimal closed wire protocol.
cycle14_response_schema = cycle13_response_schema
decode_cycle14_decision = decode_cycle13_decision
cycle14_guided_choices = cycle13_guided_choices
decode_cycle14_guided_choice = decode_cycle13_guided_choice

__all__ = [
    "CYCLE14_SYSTEM_PROMPT",
    "SKILLS",
    "Cycle13Decision",
    "cycle14_guided_choices",
    "cycle14_response_schema",
    "decode_cycle14_decision",
    "decode_cycle14_guided_choice",
]
