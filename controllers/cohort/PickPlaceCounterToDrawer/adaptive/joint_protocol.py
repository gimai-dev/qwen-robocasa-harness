"""Closed protocol for bounded absolute Panda joint commands."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence

JOINT_LIMITS: tuple[tuple[float, float], ...] = (
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
)
JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
JOINT_LIMIT_MARGIN = 0.02
MAX_JOINT_STEP = 0.025
JOINT_STEP_TOLERANCE = 0.002
TRACKING_LAG_PAUSE = 0.05
LAG_PAUSE_TRACKING = "lag_pause"
ACTUAL_RELATIVE_TRACKING = "actual_relative"
JOINT_TRACKING_MODES = frozenset({LAG_PAUSE_TRACKING, ACTUAL_RELATIVE_TRACKING})
MAX_COMMAND_ACTIONS = 32
SETTLE_ACTIONS = 2
MIN_GRIPPER_ACTIONS = 16


def _strict_object(properties: dict[str, object]) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": sorted(properties),
    }


JOINT_RESPONSE_SCHEMA: dict[str, object] = _strict_object({
    "kind": {"const": "move_joints"},
    "observation_id": {"type": "string", "minLength": 1},
    "targets": {
        "type": "object",
        "additionalProperties": False,
        "minProperties": 1,
        "maxProperties": 8,
        "properties": {
            **{name: {"type": "number"} for name in JOINT_NAMES},
            "gripper": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    },
    "note": {"type": "string", "minLength": 1, "maxLength": 320},
})


@dataclasses.dataclass(frozen=True)
class JointCommand:
    observation_id: str
    targets: dict[str, float]
    note: str
    tracking_mode: str = LAG_PAUSE_TRACKING
    kind: str = "move_joints"


@dataclasses.dataclass(frozen=True)
class JointTrajectory:
    bounded_target: tuple[float, ...]
    explicit_mask: tuple[bool, ...]
    gripper_open: float
    waypoints: tuple[tuple[float, ...], ...] = ()
    remaining_error: float = 0.0


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{label} must be a finite number")
    return output


def decode_joint_command(
    value: Mapping[str, object], *, observation_id: str
) -> JointCommand:
    required_fields = {"kind", "observation_id", "targets", "note"}
    if set(value) not in {frozenset(required_fields), frozenset({
        *required_fields, "tracking_mode",
    })}:
        raise ValueError("joint command fields drifted")
    if value.get("kind") != "move_joints":
        raise ValueError("joint command kind is invalid")
    if value.get("observation_id") != observation_id:
        raise ValueError("joint command has a stale observation_id")
    note = value.get("note")
    if not isinstance(note, str) or not 1 <= len(note) <= 320:
        raise ValueError("joint command note is outside its bound")
    tracking_mode = value.get("tracking_mode", LAG_PAUSE_TRACKING)
    if tracking_mode not in JOINT_TRACKING_MODES:
        raise ValueError("joint tracking mode is invalid")
    raw_targets = value.get("targets")
    if not isinstance(raw_targets, Mapping) or not 1 <= len(raw_targets) <= 8:
        raise ValueError("joint targets must be a non-empty mapping")
    unknown = set(raw_targets) - {*JOINT_NAMES, "gripper"}
    if unknown:
        raise ValueError(f"joint targets contain unknown dimensions: {sorted(unknown)}")

    targets: dict[str, float] = {}
    for name, raw in raw_targets.items():
        target = _finite_number(raw, label=name)
        if name == "gripper":
            if not 0.0 <= target <= 1.0:
                raise ValueError("gripper target must be between 0 and 1")
        else:
            index = JOINT_NAMES.index(name)
            lower, upper = JOINT_LIMITS[index]
            if not lower + JOINT_LIMIT_MARGIN <= target <= upper - JOINT_LIMIT_MARGIN:
                raise ValueError(f"{name} target is outside the safe joint-limit inset")
        targets[name] = target
    return JointCommand(
        observation_id=observation_id,
        targets=targets,
        note=note,
        tracking_mode=str(tracking_mode),
    )


def _current_qpos(value: Sequence[object]) -> tuple[float, ...]:
    if len(value) != 7:
        raise ValueError("current arm qpos must contain seven values")
    output = tuple(
        _finite_number(item, label=f"current {JOINT_NAMES[index]}")
        for index, item in enumerate(value)
    )
    for index, (item, (lower, upper)) in enumerate(zip(output, JOINT_LIMITS, strict=True)):
        if not lower <= item <= upper:
            raise ValueError(f"held {JOINT_NAMES[index]} is outside its hard limit")
    return output


def interpolate_joint_command(
    command: JointCommand,
    *,
    current_qpos: Sequence[object],
    current_gripper: object,
) -> JointTrajectory:
    current = _current_qpos(current_qpos)
    gripper = _finite_number(current_gripper, label="current gripper")
    if not 0.0 <= gripper <= 1.0:
        raise ValueError("current gripper must be between 0 and 1")
    target = tuple(
        command.targets.get(name, current[index])
        for index, name in enumerate(JOINT_NAMES)
    )
    explicit = tuple(name in command.targets for name in JOINT_NAMES)
    waypoints: list[tuple[float, ...]] = []
    prior: tuple[float, ...] | None = None
    actual = current
    while len(waypoints) < MAX_COMMAND_ACTIONS:
        waypoint = next_joint_waypoint(actual, prior, target)
        waypoints.append(waypoint)
        prior = waypoint
        actual = waypoint
        if waypoint == target:
            break

    reached = bool(waypoints) and waypoints[-1] == target
    if reached:
        required_actions = len(waypoints) + SETTLE_ACTIONS
        if command.targets.get("gripper", gripper) != gripper:
            required_actions = max(required_actions, MIN_GRIPPER_ACTIONS)
        while len(waypoints) < min(required_actions, MAX_COMMAND_ACTIONS):
            waypoints.append(target)

    remaining_error = max(
        abs(expected - realized)
        for expected, realized in zip(target, waypoints[-1], strict=True)
    )
    return JointTrajectory(
        bounded_target=target,
        explicit_mask=explicit,
        gripper_open=command.targets.get("gripper", gripper),
        waypoints=tuple(waypoints),
        remaining_error=remaining_error,
    )


def next_joint_waypoint(
    actual_qpos: Sequence[object],
    prior_commanded_qpos: Sequence[object] | None,
    endpoint: Sequence[object],
    *,
    tracking_mode: str = LAG_PAUSE_TRACKING,
) -> tuple[float, ...]:
    actual = _current_qpos(actual_qpos)
    target = _current_qpos(endpoint)
    if tracking_mode not in JOINT_TRACKING_MODES:
        raise ValueError("joint tracking mode is invalid")
    if tracking_mode == ACTUAL_RELATIVE_TRACKING:
        reference = actual
    elif prior_commanded_qpos is None:
        reference = actual
    else:
        prior = _current_qpos(prior_commanded_qpos)
        if max(
            abs(commanded - realized)
            for commanded, realized in zip(prior, actual, strict=True)
        ) >= TRACKING_LAG_PAUSE:
            return prior
        reference = prior
    return tuple(
        current + max(-MAX_JOINT_STEP, min(MAX_JOINT_STEP, desired - current))
        for current, desired in zip(reference, target, strict=True)
    )
