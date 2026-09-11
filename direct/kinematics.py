"""Pure Panda kinematics: FK, bounded IK, Cartesian segments, frame transforms.

Positions are metres, angles radians. ``panda_fk`` maps the seven arm joints to
RoboSuite's right-gripper ``grip_site`` in the Panda ``link0`` base frame, the
same frame as the public ``state.end_effector_position_relative`` observable.
Public quaternions are vector-first ``(x, y, z, w)``. Nothing here reads the
simulator or chooses targets; callers decide where the gripper should go.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

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
MAX_JOINT_STEP = 0.08           # rad per simulator step commanded joint rate cap (1.6 rad/s at 20 Hz)
MAX_WAYPOINT_JUMP = 0.6         # rad; larger consecutive IK jumps mean a branch change, not a path
POSITION_TOLERANCE_M = 0.003
REGULARIZATION = 0.01           # IK pull toward the start configuration (m per rad)
ORIENTATION_TOLERANCE_RAD = math.radians(3)
_BOUNDS = np.asarray(JOINT_LIMITS).T + np.array([[JOINT_LIMIT_MARGIN], [-JOINT_LIMIT_MARGIN]])


def _quat_wxyz_matrix(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _transform(position: Sequence[float], quat_wxyz: Sequence[float]) -> np.ndarray:
    output = np.eye(4)
    output[:3, :3] = _quat_wxyz_matrix(quat_wxyz)
    output[:3, 3] = np.asarray(position, dtype=float)
    return output


# Parent-body transforms transcribed from the Panda MJCF (link1..link7); every
# joint rotates about its child body's local +z axis.
_PANDA_JOINT_TRANSFORMS = tuple(_transform(p, q) for p, q in (
    ((0.0, 0.0, 0.333), (1.0, 0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.707107, -0.707107, 0.0, 0.0)),
    ((0.0, -0.316, 0.0), (0.707107, 0.707107, 0.0, 0.0)),
    ((0.0825, 0.0, 0.0), (0.707107, 0.707107, 0.0, 0.0)),
    ((-0.0825, 0.384, 0.0), (0.707107, -0.707107, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.707107, 0.707107, 0.0, 0.0)),
    ((0.088, 0.0, 0.0), (0.707107, 0.707107, 0.0, 0.0)),
))
# Fixed chain after joint7: right_hand, PandaGripper root, eef body (grip_site).
_FLANGE_TO_GRIP_SITE = np.eye(4)
for _p, _q in (
    ((0.0, 0.0, 0.1065), (0.924, 0.0, 0.0, -0.383)),
    ((0.0, 0.0, 0.0), (0.707107, 0.0, 0.0, -0.707107)),
    ((0.0, 0.0, 0.097), (1.0, 0.0, 0.0, 0.0)),
):
    _FLANGE_TO_GRIP_SITE = _FLANGE_TO_GRIP_SITE @ _transform(_p, _q)


def panda_fk(q: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return (position, rotation matrix) of grip_site in the Panda base frame."""
    values = np.asarray(q, dtype=float).reshape(-1)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("Panda qpos must contain seven finite values")
    transform = np.eye(4)
    for fixed, angle in zip(_PANDA_JOINT_TRANSFORMS, values, strict=True):
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        transform = transform @ fixed @ rotation
    transform = transform @ _FLANGE_TO_GRIP_SITE
    return transform[:3, 3].copy(), transform[:3, :3].copy()


def quat_xyzw_to_matrix(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return _quat_wxyz_matrix((w, x, y, z))


def matrix_to_quat_xyzw(rotation: np.ndarray) -> list[float]:
    quat = Rotation.from_matrix(np.asarray(rotation, dtype=float)).as_quat()  # xyzw
    if quat[3] < 0:
        quat = -quat
    return [float(v) for v in quat]


def base_to_world(position_base: Sequence[float], rotation_base: np.ndarray | None,
                  base_position: Sequence[float], base_quat_xyzw: Sequence[float]) -> tuple[np.ndarray, np.ndarray | None]:
    """Compose a base-frame pose with the measured public base pose."""
    rotation = quat_xyzw_to_matrix(base_quat_xyzw)
    translation = np.asarray(base_position, dtype=float)
    position = translation + rotation @ np.asarray(position_base, dtype=float)
    return position, (None if rotation_base is None else rotation @ np.asarray(rotation_base))


def world_to_base(position_world: Sequence[float], rotation_world: np.ndarray | None,
                  base_position: Sequence[float], base_quat_xyzw: Sequence[float]) -> tuple[np.ndarray, np.ndarray | None]:
    rotation = quat_xyzw_to_matrix(base_quat_xyzw)
    translation = np.asarray(base_position, dtype=float)
    position = rotation.T @ (np.asarray(position_world, dtype=float) - translation)
    return position, (None if rotation_world is None else rotation.T @ np.asarray(rotation_world))


def pose_errors(q: Sequence[float], position: np.ndarray, rotation: np.ndarray) -> tuple[float, float]:
    actual_position, actual_rotation = panda_fk(q)
    return (float(np.linalg.norm(position - actual_position)),
            float(np.linalg.norm(Rotation.from_matrix(rotation @ actual_rotation.T).as_rotvec())))


def solve_pose(q_start: Sequence[float], position: Sequence[float],
               rotation: Sequence[Sequence[float]], *, max_nfev: int = 180) -> dict:
    """Bounded full-pose IK from ``q_start``; inspect ``status`` before use.

    An unresolved local solve does not prove global impossibility; the residuals
    are returned so the policy can move the target.
    """
    start = np.asarray(q_start, dtype=float)
    target = np.asarray(position, dtype=float)
    target_rotation = np.asarray(rotation, dtype=float)

    def residual(q: np.ndarray) -> np.ndarray:
        actual_position, actual_rotation = panda_fk(q)
        # The weak pull toward q_start selects the nearest redundant solution
        # (joint3/joint5 are interchangeable when the arm is straight).
        return np.r_[actual_position - target,
                     0.25 * Rotation.from_matrix(target_rotation @ actual_rotation.T).as_rotvec(),
                     REGULARIZATION * (q - start)]

    result = least_squares(residual, np.clip(start, *_BOUNDS), bounds=_BOUNDS,
                           max_nfev=max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    position_error, orientation_error = pose_errors(result.x, target, target_rotation)
    reachable = position_error <= POSITION_TOLERANCE_M and orientation_error <= ORIENTATION_TOLERANCE_RAD
    return {
        "status": "kinematically_reachable" if reachable else "kinematically_unresolved",
        "q": result.x.tolist(),
        "position_error_m": position_error,
        "orientation_error_rad": orientation_error,
        "min_joint_margin_rad": float(np.min(np.r_[result.x - _BOUNDS[0], _BOUNDS[1] - result.x])),
        "nfev": int(result.nfev),
    }


def plan_pose_segment(q_start: Sequence[float], target_position: Sequence[float],
                      target_rotation: Sequence[Sequence[float]], *,
                      translation_step_m: float = 0.01,
                      rotation_step_rad: float = math.radians(2.5)) -> dict:
    """Straight-line position / shortest-rotation segment as joint waypoints.

    Returns ``{"status", "waypoints": [q, ...]}``; a failed segment returns no
    executable prefix so the caller reports the target as unreachable.
    """
    p0, r0 = panda_fk(q_start)
    p1, r1 = np.asarray(target_position, dtype=float), np.asarray(target_rotation, dtype=float)
    angle = float(np.linalg.norm(Rotation.from_matrix(r1 @ r0.T).as_rotvec()))
    count = max(1, math.ceil(np.linalg.norm(p1 - p0) / translation_step_m),
                math.ceil(angle / rotation_step_rad))
    interpolate = Slerp([0, 1], Rotation.from_matrix([r0, r1]))
    q = np.asarray(q_start, dtype=float)
    waypoints: list[list[float]] = []
    for fraction in np.linspace(0, 1, count + 1)[1:]:
        position = p0 + fraction * (p1 - p0)
        rotation = interpolate(fraction).as_matrix()
        result = solve_pose(q, position, rotation)
        if result["status"] != "kinematically_reachable":
            return {"status": "kinematically_unresolved", "waypoints": [],
                    "failed_fraction": float(fraction), "failed_result": result,
                    "segment_length_m": float(np.linalg.norm(p1 - p0)), "segment_angle_rad": angle}
        jump = float(np.max(np.abs(np.asarray(result["q"]) - q)))
        if jump > MAX_WAYPOINT_JUMP:
            result["max_joint_jump_rad"] = jump
            return {"status": "kinematically_unresolved", "waypoints": [],
                    "failed_fraction": float(fraction), "failed_result": result, "reason": "ik_discontinuity",
                    "segment_length_m": float(np.linalg.norm(p1 - p0)), "segment_angle_rad": angle}
        waypoints.append(result["q"])
        q = np.asarray(result["q"])
    return {"status": "kinematically_reachable", "waypoints": waypoints,
            "segment_length_m": float(np.linalg.norm(p1 - p0)), "segment_angle_rad": angle}


def inside_safe_limits(q: Sequence[float]) -> bool:
    return all(lower + JOINT_LIMIT_MARGIN <= float(v) <= upper - JOINT_LIMIT_MARGIN
               for v, (lower, upper) in zip(q, JOINT_LIMITS, strict=True))


def bounded_step(actual: Sequence[float], target: Sequence[float], maximum: float = MAX_JOINT_STEP) -> list[float]:
    """One tracking step toward ``target`` with every joint bounded by ``maximum``."""
    a, t = np.asarray(actual, dtype=float), np.asarray(target, dtype=float)
    delta = t - a
    largest = float(np.max(np.abs(delta)))
    if largest <= maximum:
        return t.tolist()
    return (a + delta * (maximum / largest)).tolist()


ELBOW_SEEDS = (
    (0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.785),
    (0.0, 0.2, 0.0, -1.6, 0.0, 1.8, 0.785),
    (0.0, 0.6, 0.0, -1.2, 0.0, 1.8, 0.785),
    (0.0, -0.8, 0.0, -2.4, 0.0, 1.6, 0.785),
)
MAX_FALLBACK_JOINT_DELTA = 1.6   # rad; a joint-space fallback must complete within one 20-step slot


def solve_pose_multistart(q_start: Sequence[float], position: Sequence[float],
                          rotation: Sequence[Sequence[float]]) -> dict:
    """IK for the final target from several seeds; returns the reachable solution
    closest (max joint delta) to ``q_start`` or the best unresolved attempt."""
    start = np.asarray(q_start, dtype=float)
    best_reachable = None
    best_unresolved = None
    for seed in (tuple(start), *ELBOW_SEEDS):
        result = solve_pose(seed, position, rotation, max_nfev=300)
        delta = float(np.max(np.abs(np.asarray(result["q"]) - start)))
        result["max_joint_delta_rad"] = delta
        if result["status"] == "kinematically_reachable":
            if best_reachable is None or delta < best_reachable["max_joint_delta_rad"]:
                best_reachable = result
        elif best_unresolved is None or result["position_error_m"] < best_unresolved["position_error_m"]:
            best_unresolved = result
    return best_reachable if best_reachable is not None else best_unresolved
