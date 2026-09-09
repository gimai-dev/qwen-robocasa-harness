"""Public-state geometry and completion gates for reviewed pull skills."""

from __future__ import annotations

import numpy as np

GRASP_ENTRY_DEPTHS_M = (0.07, 0.08, 0.09, 0.10, 0.11)
MIN_OCCUPIED_GRIPPER_WIDTH_M = 0.004
MIN_PUBLIC_PULL_M = 0.20


def _unit_axis(value: object) -> np.ndarray:
    axis = np.asarray(value, dtype=np.float64)
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise ValueError("pull axis is invalid")
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("pull axis is degenerate")
    return axis / norm


def grasp_search_entry_depths() -> tuple[float, ...]:
    return GRASP_ENTRY_DEPTHS_M


def public_grasp_aperture_m(finger_qpos: object) -> float:
    qpos = np.asarray(finger_qpos, dtype=np.float64)
    if qpos.shape != (2,) or not np.isfinite(qpos).all():
        raise ValueError("public gripper state is invalid")
    return float(abs(qpos[0] - qpos[1]))


def public_grasp_occupied(finger_qpos: object) -> bool:
    return public_grasp_aperture_m(finger_qpos) >= MIN_OCCUPIED_GRIPPER_WIDTH_M


def pull_grasp_entry_goal(
    standoff_world_m: object,
    push_axis_world: object,
    *,
    entry_depth_m: float = GRASP_ENTRY_DEPTHS_M[0],
) -> np.ndarray:
    standoff = np.asarray(standoff_world_m, dtype=np.float64)
    if standoff.shape != (3,) or not np.isfinite(standoff).all():
        raise ValueError("pull standoff is invalid")
    if (
        not np.isfinite(entry_depth_m)
        or entry_depth_m < GRASP_ENTRY_DEPTHS_M[0]
        or entry_depth_m > GRASP_ENTRY_DEPTHS_M[-1]
    ):
        raise ValueError("grasp entry depth is outside the reviewed range")
    return standoff + entry_depth_m * _unit_axis(push_axis_world)


def pull_target(
    grasp_world_m: object,
    push_axis_world: object,
    *,
    distance_m: float,
) -> np.ndarray:
    grasp = np.asarray(grasp_world_m, dtype=np.float64)
    if grasp.shape != (3,) or not np.isfinite(grasp).all():
        raise ValueError("pull grasp position is invalid")
    if not np.isfinite(distance_m) or not 0.0 <= distance_m <= 0.40:
        raise ValueError("pull distance is outside the reviewed range")
    return grasp - distance_m * _unit_axis(push_axis_world)


def pull_finish_gate(
    *,
    realized_pull_m: float,
    visual_kind: str,
    previous_streak: int,
) -> tuple[int, bool]:
    if not np.isfinite(realized_pull_m) or realized_pull_m < 0.0:
        raise ValueError("realized pull travel is invalid")
    if previous_streak < 0:
        raise ValueError("finish streak is invalid")
    if realized_pull_m < MIN_PUBLIC_PULL_M or visual_kind != "finish":
        return 0, False
    streak = previous_streak + 1
    return streak, streak >= 3
