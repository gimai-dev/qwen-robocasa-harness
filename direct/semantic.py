"""Show-Harness-style semantic action units for the RoboCasa PandaOmron.

The VLM emits ONE token per decision; this interpreter supplies the metric
magnitude and turns it into an Action for direct.executor.execute. Moves are
fixed steps in the robot BASE frame (+x forward, +y left, +z up), rotated
into the world frame with the measured base yaw. Base tokens drive the
mobile base one slot (our embodiment extension; Show-Harness arms are fixed).
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from scipy.spatial.transform import Rotation

from .actions import Action
from .kinematics import matrix_to_quat_xyzw, quat_xyzw_to_matrix

TOKENS = ("MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN",
          "ROTATE_CW", "ROTATE_CCW", "GRASP", "RELEASE", "DONE",
          "BASE_FWD", "BASE_BACK", "BASE_LEFT", "BASE_RIGHT")
_MOVE = {"MV_FWD": (1, 0, 0), "MV_BACK": (-1, 0, 0), "MV_LEFT": (0, 1, 0),
         "MV_RIGHT": (0, -1, 0), "MV_UP": (0, 0, 1), "MV_DOWN": (0, 0, -1)}
_BASE = {"BASE_FWD": ("x", 1), "BASE_BACK": ("x", -1), "BASE_LEFT": ("y", 1), "BASE_RIGHT": ("y", -1)}


def parse_token(text: str) -> str:
    stripped = (text or "").strip()
    token = stripped.split()[0].strip(".,;:\"'`") if stripped else ""
    if token not in TOKENS:
        raise ValueError(f"not an action token: {text!r}")
    return token


class Interpreter:
    def __init__(self, step_fine_m: float = 0.02, step_coarse_m: float = 0.04,
                 yaw_step_rad: float = math.radians(15), base_step_v: float = 0.5) -> None:
        self.step_fine_m, self.step_coarse_m = step_fine_m, step_coarse_m
        self.yaw_step_rad, self.base_step_v = yaw_step_rad, base_step_v

    def to_action(self, token: str, state: Mapping[str, object], *, fine: bool) -> Action | None:
        tcp = tuple(float(v) for v in state["tcp_world_position_m"])
        quat = tuple(float(v) for v in state["tcp_world_quat_xyzw"])
        if token == "DONE":
            return None
        if token in _MOVE:
            step = self.step_fine_m if fine else self.step_coarse_m
            yaw = float(state["base_world_yaw_rad"])
            dx, dy, dz = _MOVE[token]
            world = (tcp[0] + step * (dx * math.cos(yaw) - dy * math.sin(yaw)),
                     tcp[1] + step * (dx * math.sin(yaw) + dy * math.cos(yaw)),
                     tcp[2] + step * dz)
            return Action("ee", position_m=world, quat_xyzw=quat, note=token)
        if token in ("ROTATE_CW", "ROTATE_CCW"):
            sign = -1.0 if token == "ROTATE_CW" else 1.0
            rot = quat_xyzw_to_matrix(quat) @ Rotation.from_euler("z", sign * self.yaw_step_rad).as_matrix()
            return Action("ee", position_m=tcp, quat_xyzw=tuple(matrix_to_quat_xyzw(rot)), note=token)
        if token == "GRASP":
            return Action("hold", gripper=0, note=token)
        if token == "RELEASE":
            return Action("hold", gripper=1, note=token)
        axis, sign = _BASE[token]
        return Action("base", axis=axis, velocity=sign * self.base_step_v, note=token)


def proprio_text(observation: Mapping[str, object], receipt: Mapping[str, object] | None) -> dict:
    """The proprioception plugin's text: gripper height, gripper state, contact, last step effect."""
    s = observation["public_state"]
    width = float(s["gripper_width_m"])
    command = int(round(float(observation["gripper_command"])))
    gripper = ("closed on nothing" if command == 0 and width < 0.005 else
               f"closed, holding something (gap {width:.3f} m)" if command == 0 else "open")
    out = {"gripper_height_m": round(float(s["tcp_world_position_m"][2]), 3), "gripper": gripper,
           "contact_force_n": s.get("contact_force_delta_n"),
           "base_heading_rad": round(float(s["base_world_yaw_rad"]), 2)}
    if receipt:
        out["last_action_effect"] = {"status": receipt.get("status"), "tcp_moved_m": receipt.get("tcp_moved_m"),
                                     "base_moved_m": receipt.get("base_moved_m"), "reason": receipt.get("reason")}
    return out
