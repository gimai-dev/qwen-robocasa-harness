"""Closed RGB-only completion verifier for bounded manipulation trials."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

_FIELDS = {
    "kind",
    "observation_id",
    "left_open_gap_visible",
    "right_open_gap_visible",
    "evidence",
}
_KINDS = {"continue_push", "finish", "give_up"}


@dataclass(frozen=True)
class VisualCompletion:
    kind: str
    observation_id: str
    left_open_gap_visible: bool
    right_open_gap_visible: bool
    evidence: str


@dataclass(frozen=True)
class VisualOpenCompletion:
    kind: str
    observation_id: str
    left_extension_visible: bool
    right_extension_visible: bool
    evidence: str


def decode_visual_completion(
    value: Mapping[str, object], *, observation_id: str
) -> VisualCompletion:
    if set(value) != _FIELDS or value.get("kind") not in _KINDS:
        raise ValueError("visual completion response violates the closed schema")
    if value.get("observation_id") != observation_id:
        raise ValueError("visual completion cited a stale observation")
    left_open = value.get("left_open_gap_visible")
    right_open = value.get("right_open_gap_visible")
    if type(left_open) is not bool or type(right_open) is not bool:
        raise ValueError("visual completion gap fields must be booleans")
    if value["kind"] == "finish" and (left_open or right_open):
        raise ValueError("finish contradicts a visible external-view open gap")
    evidence = value.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 320:
        raise ValueError("visual completion evidence is invalid")
    if value["kind"] == "finish" and not all(
        name in evidence.lower() for name in ("left", "right")
    ):
        raise ValueError("finish evidence must cite both external views")
    return VisualCompletion(
        str(value["kind"]), observation_id, left_open, right_open, evidence
    )


def decode_visual_open_completion(
    value: Mapping[str, object], *, observation_id: str
) -> VisualOpenCompletion:
    fields = {
        "kind",
        "observation_id",
        "left_extension_visible",
        "right_extension_visible",
        "evidence",
    }
    if set(value) != fields or value.get("kind") not in {
        "continue_pull",
        "finish",
        "give_up",
    }:
        raise ValueError("visual open completion response violates the closed schema")
    if value.get("observation_id") != observation_id:
        raise ValueError("visual open completion cited a stale observation")
    left_visible = value.get("left_extension_visible")
    right_visible = value.get("right_extension_visible")
    if type(left_visible) is not bool or type(right_visible) is not bool:
        raise ValueError("visual open completion fields must be booleans")
    if value["kind"] == "finish" and not (left_visible and right_visible):
        raise ValueError("finish requires both external views")
    evidence = value.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 320:
        raise ValueError("visual open completion evidence is invalid")
    if value["kind"] == "finish" and not all(
        name in evidence.lower() for name in ("left", "right")
    ):
        raise ValueError("finish evidence must cite both external views")
    return VisualOpenCompletion(
        str(value["kind"]),
        observation_id,
        left_visible,
        right_visible,
        evidence,
    )


def close_drawer_verifier_prompt(task: str = "CloseDrawer") -> str:
    return f"""You are the terminal visual verifier in an Inspect-style robot harness.
You receive exactly the three synchronized official RoboCasa RGB views (left external,
right external, wrist) and the official public robot state. Use images only to judge
the drawer; do not infer depth, joints, contacts, reward, or hidden task state.

The task is {task}. Return exactly one JSON object and no prose:
{{"kind":"continue_push|finish|give_up","observation_id":"...","left_open_gap_visible":true|false,"right_open_gap_visible":true|false,"evidence":"..."}}

Return finish only when both the left and right external views visibly agree that the
manipulated moving front is flush with or fully seated against its surrounding fixed
frame (for a drawer, flush with its surrounding cabinet frame), with no protruding
moving edge or open gap in either view. If any external view
still shows the task fixture open, return continue_push. One view alone is never
sufficient because the robot can occlude the open side. Do not use arm pose, effort, elapsed actions, or a
partially occluded wrist view as proof. Return continue_push when the same moving front
remains visibly protruding and a straight bounded push remains visually appropriate.
Return give_up if the drawer is not identifiable, the robot is misaligned, or safe
progress cannot be verified. Finish evidence must explicitly cite both the left and
right external views and their visible drawer/frame relationship in one short
sentence. Never output motion coordinates or any command outside this vocabulary."""


def open_drawer_verifier_prompt(task: str = "OpenDrawer") -> str:
    return f"""You are the terminal visual verifier in an Inspect-style robot harness.
You receive exactly the three synchronized official RoboCasa RGB views (left external,
right external, wrist) and the official public robot state. Use images only to judge
the drawer; do not infer depth, joints, contacts, reward, or hidden task state.

The task is {task}. Return exactly one JSON object and no prose:
{{"kind":"continue_pull|finish|give_up","observation_id":"...","left_extension_visible":true|false,"right_extension_visible":true|false,"evidence":"..."}}

Return finish only when both the left and right external views agree that the
manipulated drawer or moving front is visibly extended or swung away from its fixed
frame and an open gap, side wall, or interior is visible. If either external view still
shows it closed or only slightly ajar, return continue_pull. Do not use arm pose, effort, elapsed actions, or a wrist
occlusion as proof. Return give_up if the drawer is not identifiable, the grasp is
visibly lost, or safe progress cannot be verified. Finish evidence must explicitly
cite both the left and right external views. Never output motion coordinates or any
command outside this vocabulary."""
