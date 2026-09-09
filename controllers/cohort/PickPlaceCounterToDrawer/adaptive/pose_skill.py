"""Robot-only pose IK and measured waypoint tracking for Qwen-authored skills.

Positions and rotations are expressed in the Panda base frame. Gripper +z is
approach, and +x is jaw separation. Reachability here is kinematic; callers own
RGB target selection, environmental clearance, action budgets, and grasp proof.
Requires NumPy and SciPy; imports no simulator and reads no environment state.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from .joint_protocol import JOINT_LIMIT_MARGIN, JOINT_LIMITS, MAX_JOINT_STEP
from .panda_embodiment import panda_fk

POSITION_TOLERANCE_M = .003
ORIENTATION_TOLERANCE_RAD = math.radians(3)
_BOUNDS = np.asarray(JOINT_LIMITS).T + np.array([[JOINT_LIMIT_MARGIN], [-JOINT_LIMIT_MARGIN]])


def _pose_arrays(q: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    pose = panda_fk(np.asarray(q, dtype=float).tolist())
    return np.asarray(pose.position_m), np.asarray(pose.rotation_matrix)


def _errors(q: Sequence[float], position: np.ndarray, rotation: np.ndarray) -> tuple[float, float]:
    actual_position, actual_rotation = _pose_arrays(q)
    return (float(np.linalg.norm(position-actual_position)),
            float(np.linalg.norm(Rotation.from_matrix(rotation @ actual_rotation.T).as_rotvec())))


def solve_pose(q_start: Sequence[float], position: Sequence[float],
               rotation: Sequence[Sequence[float]], *, max_nfev: int = 180) -> dict:
    """Bounded full-pose IK; inspect status before executing diagnostic ``q``.

    An unresolved local solve does not prove global impossibility. It returns
    the actual position/orientation residual so Qwen can alter the approach.
    """
    start = np.asarray(q_start, dtype=float)
    target = np.asarray(position, dtype=float)
    target_rotation = np.asarray(rotation, dtype=float)

    def residual(q):
        actual_position, actual_rotation = _pose_arrays(q)
        return np.r_[actual_position-target, .25*Rotation.from_matrix(
            target_rotation @ actual_rotation.T).as_rotvec()]

    result = least_squares(residual, np.clip(start, *_BOUNDS), bounds=_BOUNDS,
                           max_nfev=max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    position_error, orientation_error = _errors(result.x, target, target_rotation)
    reachable = position_error <= POSITION_TOLERANCE_M and orientation_error <= ORIENTATION_TOLERANCE_RAD
    return {
        'status': 'kinematically_reachable' if reachable else 'kinematically_unresolved',
        'q': result.x.tolist(), 'position_error_m': position_error,
        'orientation_error_rad': orientation_error,
        'min_joint_margin_rad': float(np.min(np.r_[result.x-_BOUNDS[0], _BOUNDS[1]-result.x])),
        'nfev': result.nfev,
    }


def pregrasp_targets(point: Sequence[float], yaw: float = 0.,
                     clearance: float = .15) -> tuple[np.ndarray, np.ndarray]:
    """Return overhead position and downward rotation; yaw specifies jaw +x.

    For an elongated horizontal object, the jaw x axis should normally cross
    its long axis. Selecting that axis remains a visual/planning decision.
    """
    rotation = Rotation.from_euler('z', yaw).as_matrix() @ np.diag([1., -1., -1.])
    return np.asarray(point, dtype=float)+[0., 0., clearance], rotation


def retreat_target(q: Sequence[float], distance: float = .1) -> tuple[np.ndarray, np.ndarray]:
    """Return a pose behind the current grip site along tool -z."""
    position, rotation = _pose_arrays(q)
    return position-distance*rotation[:, 2], rotation


def plan_pose_segment(q_start: Sequence[float], target_position: Sequence[float],
                      target_rotation: Sequence[Sequence[float]], *,
                      translation_step_m: float = .02,
                      rotation_step_rad: float = math.radians(5)) -> dict:
    """Solve a straight position / shortest-rotation segment from current FK.

    Each waypoint includes its intended pose, for measured tracking. Failed
    segments return no executable prefix: callers must change/replan the route.
    Joint interpolation and contact monitoring remain executor responsibilities.
    """
    p0, r0 = _pose_arrays(q_start)
    p1, r1 = np.asarray(target_position), np.asarray(target_rotation)
    if translation_step_m <= 0 or rotation_step_rad <= 0:
        raise ValueError('pose sampling steps must be positive')
    angle = np.linalg.norm(Rotation.from_matrix(r1 @ r0.T).as_rotvec())
    count = max(1, math.ceil(np.linalg.norm(p1-p0)/translation_step_m),
                math.ceil(angle/rotation_step_rad))
    interpolate = Slerp([0, 1], Rotation.from_matrix([r0, r1]))
    q = np.asarray(q_start, dtype=float)
    waypoints = []
    for fraction in np.linspace(0, 1, count+1)[1:]:
        position = p0+fraction*(p1-p0)
        rotation = interpolate(fraction).as_matrix()
        result = solve_pose(q, position, rotation)
        if result['status'] != 'kinematically_reachable':
            return {'status': 'kinematically_unresolved', 'waypoints': [],
                    'failed_fraction': float(fraction), 'failed_result': result}
        result.update(target_position_m=position.tolist(), target_rotation=rotation.tolist(),
                      max_joint_jump_rad=float(np.max(np.abs(np.asarray(result['q'])-q))))
        waypoints.append(result)
        q = np.asarray(result['q'])
    return {'status': 'kinematically_reachable', 'waypoints': waypoints}


def _bounded_step(actual: np.ndarray, target: np.ndarray, maximum: float) -> list[float]:
    delta = target-actual
    largest = float(np.max(np.abs(delta)))
    # One common scale preserves the joint direction toward this nearby pose.
    return (actual+delta*min(1., maximum/max(largest, 1e-12))).tolist()


def tracking_step(*, actual_q: Sequence[float], target_q: Sequence[float],
                  previous_q: Sequence[float], force: Sequence[float],
                  unloaded_force: Sequence[float], executed_q: Sequence[Sequence[float]],
                  max_joint_step: float = MAX_JOINT_STEP,
                  force_increase_n: float = 20., minimum_progress_fraction: float = .05) -> dict:
    """Advance, finish, or reverse a loaded/stalled measured waypoint path.

    ``previous_q`` is an earlier measured sample in the same tracking window;
    use a window long enough for the position controller to respond.
    ``executed_q`` is chronological *measured* motion starting at an unloaded
    pose, not merely commanded targets. On blockage, follow returned reversed
    waypoints without advancing the grasp phase. Keep the unloaded force
    baseline for the whole approach; do not reset it during loaded contact.
    """
    actual, target = np.asarray(actual_q), np.asarray(target_q)
    p_target, r_target = _pose_arrays(target)
    pos_error, rot_error = _errors(actual, p_target, r_target)
    prev_pos_error, prev_rot_error = _errors(previous_q, p_target, r_target)
    error = pos_error+.1*rot_error
    previous_error = prev_pos_error+.1*prev_rot_error
    progress = (previous_error-error)/max(previous_error, 1e-12)
    # Norm growth is insensitive to rotating the wrist sensor frame.
    load = float(np.linalg.norm(force)-np.linalg.norm(unloaded_force))
    reached = pos_error <= POSITION_TOLERANCE_M and rot_error <= ORIENTATION_TOLERANCE_RAD
    blocked = not reached and load >= force_increase_n and progress < minimum_progress_fraction
    retreat = []
    if blocked:
        # Skip the present sample, then replay the actual approach in reverse.
        retreat = [np.asarray(q).tolist() for q in reversed(executed_q)
                   if np.max(np.abs(np.asarray(q)-actual)) > 1e-6]
        next_q = _bounded_step(actual, np.asarray(retreat[0]), max_joint_step) if retreat else actual.tolist()
    elif reached:
        next_q = actual.tolist()
    else:
        next_q = _bounded_step(actual, target, max_joint_step)
    return {'status': 'blocked' if blocked else 'reached' if reached else 'advance',
            'next_q': next_q, 'retreat_waypoints': retreat,
            'position_error_m': pos_error, 'orientation_error_rad': rot_error,
            'progress_fraction': float(progress), 'force_increase_n': load}


def choose_parallel_jaw_path(q: Sequence[float], position: Sequence[float],
                             rotation: Sequence[Sequence[float]]) -> dict:
    """Choose the cheaper reachable path for two equivalent jaw directions.

    Flipping tool x/y preserves approach +z and the opposing jaw line. For a
    downward grasp this is the requested yaw plus pi; multiplying on the right
    also avoids applying a world-z rotation to a nonvertical approach. Cost is
    the ideal number of existing MAX_JOINT_STEP-bounded actions, not sim time.
    """
    original = np.asarray(rotation, dtype=float)
    alternatives = (original, original @ np.diag([-1., -1., 1.]))
    candidates = []
    reachable = []
    for offset, candidate_rotation in zip((0., math.pi), alternatives):
        plan = plan_pose_segment(q, position, candidate_rotation)
        summary = {'yaw_offset_rad': offset, 'status': plan['status'], 'ideal_actions': None}
        if plan['status'] == 'kinematically_reachable':
            configurations = np.asarray([list(q)]+[w['q'] for w in plan['waypoints']])
            largest_deltas = np.abs(np.diff(configurations, axis=0)).max(axis=1)
            actions = int(np.ceil(largest_deltas/MAX_JOINT_STEP).sum())
            summary['ideal_actions'] = actions
            reachable.append((actions, offset, candidate_rotation, plan))
        else:
            summary['failed_fraction'] = plan['failed_fraction']
            summary['failed_result'] = plan['failed_result']
        candidates.append(summary)
    if not reachable:
        return {'status': 'kinematically_unresolved', 'waypoints': [],
                'selected_yaw_offset_rad': None, 'target_rotation': None,
                'ideal_actions': None, 'candidate_summaries': candidates}
    actions, offset, selected_rotation, plan = min(reachable, key=lambda item: item[0])
    return {**plan, 'selected_yaw_offset_rad': offset,
            'target_rotation': selected_rotation.tolist(), 'ideal_actions': actions,
            'candidate_summaries': candidates}


def detect_stalled_window(samples: Sequence[dict]) -> bool:
    """Detect three-observation pose stagnation even without wrist force.

    Samples contain ``position_error_m`` and ``orientation_error_rad`` measured
    against the SAME goal. Reset the window when that goal changes, or compute
    errors against the fixed final pose of a stage. This catches the observed
    high-wrist tracking plateau while permitting unloaded progress.
    """
    if len(samples) < 3:
        return False
    first, _, last = samples[-3:]
    position = float(last['position_error_m'])
    orientation = float(last['orientation_error_rad'])
    if position <= .005 and orientation <= math.radians(4):
        return False
    initial_error = float(first['position_error_m'])+.1*float(first['orientation_error_rad'])
    final_error = position+.1*orientation
    progress = (initial_error-final_error)/max(initial_error, 1e-12)
    return progress < .03
