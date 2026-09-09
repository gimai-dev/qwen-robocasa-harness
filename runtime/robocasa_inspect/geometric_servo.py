"""Public-state waypoint construction for RGB-parallax task features."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

PANDA_GRIPPER_PLUS_Z_SUPPORT_M = 0.04598793422755339
PUBLIC_RGB_STANDOFF_MARGIN_M = 0.05
MIN_WAYPOINT_INCREMENT_BUDGET = 16
MAX_WAYPOINT_INCREMENT_BUDGET = 64
CERTIFIED_WAYPOINT_PROGRESS_M = 0.004


def quaternion_xyzw_matrix(value: object) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("base quaternion is invalid")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("base quaternion is degenerate")
    x, y, z, w = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def public_eef_world(state: Mapping[str, object]) -> np.ndarray:
    base = np.asarray(state["state.base_position"], dtype=np.float64)
    relative = np.asarray(
        state["state.end_effector_position_relative"], dtype=np.float64
    )
    rotation = quaternion_xyzw_matrix(state["state.base_rotation"])
    if base.shape != (3,) or relative.shape != (3,):
        raise ValueError("public position state is invalid")
    return base + rotation @ relative


def world_quaternion_to_base_xyzw(
    state: Mapping[str, object], world_xyzw: object
) -> np.ndarray:
    base = np.asarray(state["state.base_rotation"], dtype=np.float64)
    world = np.asarray(world_xyzw, dtype=np.float64)
    if base.shape != (4,) or world.shape != (4,):
        raise ValueError("public/world quaternion is invalid")
    base /= np.linalg.norm(base)
    world /= np.linalg.norm(world)
    x1, y1, z1, w1 = (-base[0], -base[1], -base[2], base[3])
    x2, y2, z2, w2 = world
    relative = np.asarray(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )
    return relative / np.linalg.norm(relative)


def push_site_orientation_world_xyzw(
    state: Mapping[str, object], target_world_m: object
) -> np.ndarray:
    """Keep the tool axis surface-normal and its orthogonal axis vertically down."""
    base = np.asarray(state["state.base_position"], dtype=np.float64)
    target = np.asarray(target_world_m, dtype=np.float64)
    return push_site_orientation_for_direction_world_xyzw(target - base)


def push_site_orientation_for_direction_world_xyzw(
    direction_world: object,
    *,
    vertical_sign: int = -1,
) -> np.ndarray:
    """Point site +Z along a public-RGB-derived horizontal push direction."""
    forward = np.asarray(direction_world, dtype=np.float64).copy()
    if forward.shape != (3,) or not np.isfinite(forward).all():
        raise ValueError("push direction is invalid")
    forward[2] = 0.0
    if np.linalg.norm(forward) <= 1e-12:
        raise ValueError("push direction is degenerate")
    forward /= np.linalg.norm(forward)
    if isinstance(vertical_sign, bool) or vertical_sign not in {-1, 1}:
        raise ValueError("push vertical sign is invalid")
    local_x = np.asarray([0.0, 0.0, float(vertical_sign)])
    local_y = np.cross(forward, local_x)
    local_y /= np.linalg.norm(local_y)
    matrix = np.column_stack((local_x, local_y, forward))
    return _orientation_matrix_to_xyzw(matrix)


def downward_push_site_orientation_world_xyzw(*, roll_sign: int = 1) -> np.ndarray:
    """Point site +Z down with a deterministic world-aligned wrist roll."""

    if isinstance(roll_sign, bool) or roll_sign not in {-1, 1}:
        raise ValueError("downward push roll sign is invalid")
    local_x = np.asarray([float(roll_sign), 0.0, 0.0])
    local_y = np.asarray([0.0, -float(roll_sign), 0.0])
    local_z = np.asarray([0.0, 0.0, -1.0])
    return _orientation_matrix_to_xyzw(np.column_stack((local_x, local_y, local_z)))


def grasp_site_orientation_for_direction_world_xyzw(
    direction_world: object,
    *,
    vertical_sign: int = 1,
) -> np.ndarray:
    """Point +Z at a handle while the Panda finger-separation X stays vertical."""
    forward = np.asarray(direction_world, dtype=np.float64).copy()
    if forward.shape != (3,) or not np.isfinite(forward).all():
        raise ValueError("grasp direction is invalid")
    forward[2] = 0.0
    if np.linalg.norm(forward) <= 1e-12:
        raise ValueError("grasp direction is degenerate")
    forward /= np.linalg.norm(forward)
    if isinstance(vertical_sign, bool) or vertical_sign not in {-1, 1}:
        raise ValueError("grasp vertical sign is invalid")
    local_x = np.asarray([0.0, 0.0, float(vertical_sign)])
    local_y = np.cross(forward, local_x)
    local_y /= np.linalg.norm(local_y)
    return _orientation_matrix_to_xyzw(
        np.column_stack((local_x, local_y, forward))
    )


def _orientation_matrix_to_xyzw(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.asarray(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        diagonal = int(np.argmax(np.diag(matrix)))
        indices = ((1, 2), (2, 0), (0, 1))
        first, second = indices[diagonal]
        scale = 2.0 * np.sqrt(
            1.0 + matrix[diagonal, diagonal]
            - matrix[first, first]
            - matrix[second, second]
        )
        quaternion = np.empty(4, dtype=np.float64)
        quaternion[diagonal] = 0.25 * scale
        quaternion[first] = (matrix[first, diagonal] + matrix[diagonal, first]) / scale
        quaternion[second] = (
            matrix[second, diagonal] + matrix[diagonal, second]
        ) / scale
        quaternion[3] = (matrix[second, first] - matrix[first, second]) / scale
    if quaternion[3] < 0:
        quaternion *= -1
    return quaternion / np.linalg.norm(quaternion)


def orient_surface_normal_from_base(
    normal_world: object, target_world_m: object, base_world_m: object
) -> np.ndarray:
    """Choose the RGB plane-normal sign that points from the robot into the target."""
    normal = np.asarray(normal_world, dtype=np.float64).copy()
    target = np.asarray(target_world_m, dtype=np.float64)
    base = np.asarray(base_world_m, dtype=np.float64)
    if normal.shape != (3,) or target.shape != (3,) or base.shape != (3,):
        raise ValueError("surface-normal geometry is invalid")
    normal[2] = 0.0
    normal /= np.linalg.norm(normal)
    if np.dot(normal, target - base) < 0:
        normal *= -1
    return normal


def final_push_direction_after_base_reposition(
    *,
    initial_direction: object,
    target_world_m: object,
    final_base_world_m: object,
    preserve_rgb_surface_normal: bool,
) -> np.ndarray:
    """Refresh point-target geometry after mobile-base motion.

    A handle supplies a public-RGB plane normal whose axis remains authoritative; only
    its sign depends on the final base pose. A point-like button has no observed plane
    normal, so its approach ray must be recomputed from the final base rather than
    retaining a stale pre-reposition ray.
    """

    initial = np.asarray(initial_direction, dtype=np.float64)
    target = np.asarray(target_world_m, dtype=np.float64)
    final_base = np.asarray(final_base_world_m, dtype=np.float64)
    if (
        initial.shape != (3,)
        or target.shape != (3,)
        or final_base.shape != (3,)
        or not np.isfinite(initial).all()
        or not np.isfinite(target).all()
        or not np.isfinite(final_base).all()
    ):
        raise ValueError("final push-direction geometry is invalid")
    if preserve_rgb_surface_normal:
        return orient_surface_normal_from_base(initial, target, final_base)
    refreshed = target - final_base
    refreshed[2] = 0.0
    norm = float(np.linalg.norm(refreshed))
    if norm <= 1e-12:
        raise ValueError("final point-target push direction is degenerate")
    return refreshed / norm


def bounded_base_frame_delta(
    state: Mapping[str, object], target_world_m: object, *, max_step_m: float = 0.02
) -> tuple[np.ndarray, float]:
    target = np.asarray(target_world_m, dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all() or max_step_m <= 0:
        raise ValueError("geometric servo target is invalid")
    base = np.asarray(state["state.base_position"], dtype=np.float64)
    rotation = quaternion_xyzw_matrix(state["state.base_rotation"])
    target_relative = rotation.T @ (target - base)
    current_relative = np.asarray(
        state["state.end_effector_position_relative"], dtype=np.float64
    )
    error = target_relative - current_relative
    distance = float(np.linalg.norm(error))
    if distance > max_step_m:
        error *= max_step_m / distance
    return error, distance


def waypoint_increment_budget(distance_m: float) -> int:
    """Bound servo effort from a certified minimum per-chunk progress rate."""

    if not np.isfinite(distance_m) or distance_m < 0.0:
        raise ValueError("waypoint distance is invalid")
    required = int(np.ceil(distance_m / CERTIFIED_WAYPOINT_PROGRESS_M)) + 4
    return min(
        MAX_WAYPOINT_INCREMENT_BUDGET,
        max(MIN_WAYPOINT_INCREMENT_BUDGET, required),
    )


def planned_waypoint_requires_abort(
    *,
    is_final: bool,
    position_error_m: float,
    orientation_error_rad: float,
) -> bool:
    """Let only the final waypoint proceed to the stricter endpoint settle loop."""
    if (
        not np.isfinite(position_error_m)
        or position_error_m < 0.0
        or not np.isfinite(orientation_error_rad)
        or orientation_error_rad < 0.0
    ):
        raise ValueError("planned waypoint error is invalid")
    if is_final:
        return False
    return position_error_m > 0.035 or orientation_error_rad > 0.06


def contact_free_waypoints(target_world_m: object) -> tuple[np.ndarray, np.ndarray]:
    """Approach a front-facing kitchen feature from above, then from robot-side x."""
    target = np.asarray(target_world_m, dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("waypoint target is invalid")
    high = target + np.asarray([-0.10, 0.0, 0.30])
    standoff = target + np.asarray(
        [-(PANDA_GRIPPER_PLUS_Z_SUPPORT_M + PUBLIC_RGB_STANDOFF_MARGIN_M), 0.0, 0.0]
    )
    return high, standoff


def base_reposition_request(
    state: Mapping[str, object],
    target_world_m: object,
    *,
    desired_forward_m: float = 0.55,
) -> tuple[str, int] | None:
    """Place a public target in the arm's reviewed mobile-base workspace."""
    if not np.isfinite(desired_forward_m) or not 0.35 <= desired_forward_m <= 0.60:
        raise ValueError("workspace distance is outside the reviewed range")
    target = np.asarray(target_world_m, dtype=np.float64)
    base = np.asarray(state["state.base_position"], dtype=np.float64)
    rotation = quaternion_xyzw_matrix(state["state.base_rotation"])
    relative = rotation.T @ (target - base)
    if abs(relative[1]) > 0.06:
        return "y", 1 if relative[1] > 0 else -1
    if relative[0] > desired_forward_m + 0.01:
        return "x", 1
    if relative[0] < desired_forward_m - 0.01:
        return "x", -1
    return None


def torso_reposition_direction(
    state: Mapping[str, object], target_world_m: object
) -> int | None:
    """Put a public feature in the reviewed vertical arm workspace."""
    target = np.asarray(target_world_m, dtype=np.float64)
    base = np.asarray(state["state.base_position"], dtype=np.float64)
    relative_height = float(target[2] - base[2])
    if relative_height < 0.20:
        return -1
    if relative_height > 0.30:
        return 1
    return None


def bounded_orientation_delta(
    state: Mapping[str, object],
    desired_xyzw: object,
    *,
    max_step_rad: float = 0.0873,
) -> tuple[np.ndarray, float]:
    """Return a bounded local orientation correction to a reviewed target."""
    current = np.asarray(
        state["state.end_effector_rotation_relative"], dtype=np.float64
    )
    current /= np.linalg.norm(current)
    desired = np.asarray(desired_xyzw, dtype=np.float64)
    if desired.shape != (4,) or not np.isfinite(desired).all():
        raise ValueError("desired orientation is invalid")
    desired /= np.linalg.norm(desired)
    inverse = np.asarray([-current[0], -current[1], -current[2], current[3]])
    x1, y1, z1, w1 = desired
    x2, y2, z2, w2 = inverse
    difference = np.asarray(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )
    if difference[3] < 0:
        difference *= -1
    vector_norm = float(np.linalg.norm(difference[:3]))
    if vector_norm <= 1e-10:
        error = np.zeros(3)
    else:
        angle = 2.0 * np.arctan2(vector_norm, float(difference[3]))
        error = difference[:3] * (angle / vector_norm)
    distance = float(np.linalg.norm(error))
    if distance > max_step_rad:
        error *= max_step_rad / distance
    return error, distance
