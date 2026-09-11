"""Policy-visible observation window (clean) and its JSON rendering."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path


def _r(value: object, digits: int = 3) -> object:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [_r(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: _r(v, digits) for k, v in value.items()}
    return value


def state_summary(observation: Mapping[str, object]) -> dict:
    s = observation["public_state"]
    force = s.get("contact_force_delta_n", s["wrench"]["force_n"])
    return _r({
        "tcp_world_m": s["tcp_world_position_m"],
        "tcp_quat_xyzw": s["tcp_world_quat_xyzw"],
        "arm_q_rad": s["arm_q_rad"],
        "gripper_command": int(round(float(observation["gripper_command"]))),
        "gripper_width_m": s["gripper_width_m"],
        "contact_force_n": [force[0], force[1], force[2]],
        "base_world_m": s["base_world_position_m"],
        "base_yaw_rad": s["base_world_yaw_rad"],
        "tcp_pixels": {k: [v["u"], v["v"]] if v["visible"] else None for k, v in s["tcp_pixels"].items()},
    })


def regions_summary(regions: Mapping[str, object]) -> dict:
    out = []
    for region in regions["regions"]:
        out.append({"id": region["id"], "world_m": region["world_m"], "size_m": region["approx_size_m"],
                    "rgb": region["mean_rgb"],
                    "pixels": {view: v["uv"] for view, v in region["views"].items()}})
    unpaired = {view: [{"uv": m["uv"], "bbox": m["bbox_xywh"], "rgb": m["mean_rgb"]} for m in masks]
                for view, masks in regions["unpaired"].items()}
    return {"regions": out, "unpaired": unpaired}


def read_images(observation: Mapping[str, object]) -> dict[str, bytes]:
    return {label: Path(record["path"]).read_bytes() for label, record in observation["images"].items()}


def build_user_message(*, observation: Mapping[str, object], regions: Mapping[str, object] | None,
                       decision: int, max_decisions: int, previous: Mapping[str, object] | None,
                       extra: Mapping[str, object] | None = None) -> str:
    payload: dict = {
        "task": observation["instruction"],
        "budget": {"decision": decision, "decisions_left": max_decisions - decision,
                   "steps_used": observation["steps_used"], "steps_left": observation["steps_budget"] - observation["steps_used"],
                   "wall_left_s": int(observation["wall_budget_s"] - observation["wall_used_s"])},
        "state": state_summary(observation),
    }
    if regions is not None:
        payload.update(regions_summary(regions))
    payload["previous"] = previous
    if extra:
        payload.update(extra)
    return json.dumps(payload, separators=(",", ":"))


def image_list(current: Mapping[str, bytes], previous: Mapping[str, bytes] | None) -> list[tuple[str, bytes]]:
    images = [(f"current_{label}", current[label]) for label in ("left", "right", "wrist")]
    if previous is not None:
        images += [(f"before_previous_action_{label}", previous[label]) for label in ("left", "right", "wrist")]
    return images
