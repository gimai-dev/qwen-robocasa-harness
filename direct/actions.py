"""Numerical action protocol shared by every condition.

Compact keys keep a 45-slot full trajectory inside the client's output bound:
  k  kind: "ee" | "joint" | "base" | "hold" | "stop"
  p  ee position [x, y, z] metres, WORLD frame (absolute; relative in H2)
  o  ee orientation quaternion [x, y, z, w], WORLD frame, of grip_site
  q  seven arm joint targets in radians, joint1..joint7 (absolute; delta in H2)
  g  gripper command: 0 closed, 1 open (public convention)
  a  base axis: "x" | "y" | "yaw"      v  normalized base velocity in [-0.5, 0.5]
  n  short note
Every accepted action occupies one control slot of SLOT_STEPS simulator steps
unless a condition shortens the slot (H4). A rejected action consumes no steps.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .kinematics import JOINT_LIMITS, JOINT_LIMIT_MARGIN, quat_xyzw_to_matrix

SLOT_STEPS = 20
MAX_FULL_SLOTS = 45
BASE_VELOCITY_LIMIT = 0.5
BASE_MOTION_STEPS = 16          # base velocity is applied for these steps, then braked for the rest of the slot
KINDS = ("ee", "joint", "base", "hold", "stop")
AXES = ("x", "y", "yaw")


@dataclass(frozen=True)
class Action:
    kind: str
    position_m: tuple[float, float, float] | None = None
    quat_xyzw: tuple[float, float, float, float] | None = None
    q_rad: tuple[float, ...] | None = None
    gripper: int | None = None
    axis: str | None = None
    velocity: float | None = None
    note: str = ""
    raw: dict = field(default_factory=dict)

    def summary(self) -> dict:
        out: dict = {"k": self.kind}
        if self.position_m is not None:
            out["p"] = [round(v, 4) for v in self.position_m]
        if self.quat_xyzw is not None:
            out["o"] = [round(v, 4) for v in self.quat_xyzw]
        if self.q_rad is not None:
            out["q"] = [round(v, 4) for v in self.q_rad]
        if self.gripper is not None:
            out["g"] = self.gripper
        if self.axis is not None:
            out["a"] = self.axis
            out["v"] = self.velocity
        if self.note:
            out["n"] = self.note
        return out


def _finite_list(value: object, width: int, label: str) -> list[float]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence) or len(value) != width:
        raise ValueError(f"{label} must contain {width} numbers")
    out = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ValueError(f"{label} must contain finite numbers")
        out.append(float(item))
    return out


def action_schema(*, interface: str, allow_stop: bool = True, extra: Mapping[str, object] | None = None) -> dict:
    """Strict JSON schema for one action; ``interface`` is "ee" or "joint"."""
    if interface not in ("ee", "joint"):
        raise ValueError("interface must be ee or joint")
    kinds = [interface, "base", "hold"] + (["stop"] if allow_stop else [])
    number = {"type": "number"}
    nullable_vec = lambda n: {"anyOf": [{"type": "array", "items": number, "minItems": n, "maxItems": n}, {"type": "null"}]}
    properties: dict = {"k": {"type": "string", "enum": kinds}}
    if interface == "ee":
        properties["p"] = nullable_vec(3)
        properties["o"] = nullable_vec(4)
    else:
        properties["q"] = nullable_vec(7)
    properties.update({
        "g": {"anyOf": [{"type": "integer", "enum": [0, 1]}, {"type": "null"}]},
        "a": {"anyOf": [{"type": "string", "enum": list(AXES)}, {"type": "null"}]},
        "v": {"anyOf": [number, {"type": "null"}]},
        "n": {"type": "string"},
    })
    if extra:
        properties.update(extra)
    return {"type": "object", "additionalProperties": False, "properties": properties,
            "required": list(properties)}


def short_response_schema(*, interface: str, action_extra: Mapping[str, object] | None = None,
                          top_extra: Mapping[str, object] | None = None) -> dict:
    properties = {"reasoning": {"type": "string"}, "action": action_schema(interface=interface, extra=action_extra)}
    if top_extra:
        properties.update(top_extra)
    return {"type": "object", "additionalProperties": False, "properties": properties, "required": list(properties)}


def full_response_schema(*, interface: str, max_slots: int = MAX_FULL_SLOTS) -> dict:
    return {"type": "object", "additionalProperties": False,
            "properties": {"reasoning": {"type": "string"},
                           "sequence": {"type": "array", "minItems": 1, "maxItems": max_slots,
                                        "items": action_schema(interface=interface, allow_stop=False)}},
            "required": ["reasoning", "sequence"]}


def decode_action(value: Mapping[str, object], *, interface: str, representation: str = "absolute",
                  current_tcp_world: Sequence[float] | None = None,
                  current_q: Sequence[float] | None = None) -> Action:
    """Validate one model action. ``representation`` "relative" (H2) converts
    displacements / delta-q into absolute targets deterministically."""
    if not isinstance(value, Mapping):
        raise ValueError("action must be an object")
    kind = value.get("k")
    if kind not in KINDS:
        raise ValueError(f"unknown action kind {kind!r}")
    note = value.get("n", "")
    note = note if isinstance(note, str) else ""
    gripper = value.get("g")
    if gripper is not None:
        if isinstance(gripper, bool) or gripper not in (0, 1):
            raise ValueError("gripper must be 0 (closed) or 1 (open)")
        gripper = int(gripper)
    raw = dict(value)
    if kind == "ee":
        if interface != "ee":
            raise ValueError("ee action is not available in the joint interface")
        p = _finite_list(value.get("p"), 3, "position p")
        o = _finite_list(value.get("o"), 4, "orientation o")
        norm = math.sqrt(sum(v * v for v in o))
        if norm < 1e-6:
            raise ValueError("orientation quaternion has zero norm")
        o = [v / norm for v in o]
        if representation == "relative":
            if current_tcp_world is None:
                raise ValueError("relative ee action needs the current TCP position")
            p = [float(c) + d for c, d in zip(current_tcp_world, p, strict=True)]
        return Action("ee", position_m=tuple(p), quat_xyzw=tuple(o), gripper=gripper, note=note, raw=raw)
    if kind == "joint":
        if interface != "joint":
            raise ValueError("joint action is not available in the ee interface")
        q = _finite_list(value.get("q"), 7, "joint target q")
        if representation == "relative":
            if current_q is None:
                raise ValueError("relative joint action needs the current q")
            q = [float(c) + d for c, d in zip(current_q, q, strict=True)]
        for index, (v, (lower, upper)) in enumerate(zip(q, JOINT_LIMITS, strict=True)):
            if not lower + JOINT_LIMIT_MARGIN <= v <= upper - JOINT_LIMIT_MARGIN:
                raise ValueError(f"joint{index + 1} target {v:.3f} is outside the safe range "
                                 f"[{lower + JOINT_LIMIT_MARGIN:.3f}, {upper - JOINT_LIMIT_MARGIN:.3f}]")
        return Action("joint", q_rad=tuple(q), gripper=gripper, note=note, raw=raw)
    if kind == "base":
        axis = value.get("a")
        if axis not in AXES:
            raise ValueError("base action needs axis a in x|y|yaw")
        velocity = value.get("v")
        if isinstance(velocity, bool) or not isinstance(velocity, (int, float)) or not math.isfinite(float(velocity)):
            raise ValueError("base action needs a finite velocity v")
        velocity = float(velocity)
        if not 0.0 < abs(velocity) <= BASE_VELOCITY_LIMIT:
            raise ValueError(f"base velocity must be non-zero with |v| <= {BASE_VELOCITY_LIMIT}")
        return Action("base", axis=str(axis), velocity=velocity, gripper=gripper, note=note, raw=raw)
    if kind == "hold":
        return Action("hold", gripper=gripper, note=note, raw=raw)
    return Action("stop", note=note, raw=raw)


def ee_rotation_matrix(action: Action) -> np.ndarray:
    if action.quat_xyzw is None:
        raise ValueError("action has no orientation")
    return quat_xyzw_to_matrix(action.quat_xyzw)
