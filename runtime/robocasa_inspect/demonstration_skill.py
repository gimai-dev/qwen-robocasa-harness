"""Bounded adapters for official RoboCasa human-demonstration skills.

The model may select a demonstration as a high-level motion prior, but the
simulator receives only the official RoboCasa action dictionary.  The helpers in
this module deliberately use only the public 16-value robot state for live
trajectory checks; task state, reward, and the success predicate remain hidden.
"""

from __future__ import annotations

import hashlib

import numpy as np

GROOT_ACTION_WIDTH = 12
GROOT_STATE_WIDTH = 16
MAX_INITIAL_EEF_POSITION_ERROR_M = 0.03
MAX_INITIAL_EEF_ORIENTATION_ERROR_RAD = 0.35
MAX_TRACKING_EEF_POSITION_ERROR_M = 0.08
MAX_TRACKING_EEF_ORIENTATION_ERROR_RAD = 0.70
MAX_POSITION_CORRECTION_NORMALIZED = 0.25
MAX_ROTATION_CORRECTION_NORMALIZED = 0.15
MAX_REFERENCE_ADVANCE_STEPS = 10


def acknowledged_source_cursor(
    *, source_start: int, emitted_steps: int, selected_reference: int, length: int
) -> int:
    """Advance past acknowledged source actions while retaining monotone pose alignment."""
    if (
        source_start < 0
        or emitted_steps <= 0
        or selected_reference < source_start
        or length <= 0
    ):
        raise ValueError("source cursor evidence is invalid")
    consumed = min(source_start + emitted_steps, length - 1)
    return min(max(consumed, selected_reference), length - 1)


def _finite_vector(value: object, *, width: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (width,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain exactly {width} finite values")
    return vector


def groot_action_to_official(value: object) -> dict[str, np.ndarray]:
    """Map the published GR00T/LeRobot ordering into RoboCasa's Gym action."""
    action = _finite_vector(value, width=GROOT_ACTION_WIDTH, name="demo action")
    if np.any(np.abs(action) > 1.0 + 1e-12):
        raise ValueError("demo action exceeds the official normalized action bound")
    return {
        "action.end_effector_position": action[5:8].copy(),
        "action.end_effector_rotation": action[8:11].copy(),
        "action.gripper_close": action[11:12].copy(),
        "action.base_motion": action[0:4].copy(),
        "action.control_mode": action[4:5].copy(),
    }


def stationary_base_action_chunks(
    actions: object, *, chunk_steps: int = 5, max_steps: int = 450
) -> list[list[dict[str, np.ndarray]]]:
    """Validate and chunk a stationary-base demonstration for the live mailbox."""
    source = np.asarray(actions, dtype=np.float64)
    if (
        source.ndim != 2
        or source.shape[1] != GROOT_ACTION_WIDTH
        or not np.isfinite(source).all()
    ):
        raise ValueError("demonstration actions have an invalid shape or value")
    if chunk_steps <= 0 or len(source) == 0:
        raise ValueError("demonstration must contain nonempty action chunks")
    padded_steps = ((len(source) + chunk_steps - 1) // chunk_steps) * chunk_steps
    if padded_steps > max_steps:
        raise ValueError("demonstration exceeds the official simulator-step budget")
    if np.any(np.abs(source[:, :4]) > 1e-12):
        raise ValueError("stationary-base skill refuses mobile-base demonstration motion")
    converted = [groot_action_to_official(row) for row in source]
    if remainder := padded_steps - len(source):
        final = converted[-1]
        padding = {
            "action.end_effector_position": np.zeros(3),
            "action.end_effector_rotation": np.zeros(3),
            "action.gripper_close": final["action.gripper_close"].copy(),
            "action.base_motion": np.zeros(4),
            "action.control_mode": final["action.control_mode"].copy(),
        }
        converted.extend(
            {key: value.copy() for key, value in padding.items()}
            for _ in range(remainder)
        )
    return [
        converted[index : index + chunk_steps]
        for index in range(0, len(converted), chunk_steps)
    ]


def quaternion_delta_rotvec(current: object, target: object) -> np.ndarray:
    """Return the shortest current-to-target rotation vector for xyzw quaternions."""
    current = _finite_vector(current, width=4, name="current quaternion")
    target = _finite_vector(target, width=4, name="target quaternion")
    if np.linalg.norm(current) <= 1e-12 or np.linalg.norm(target) <= 1e-12:
        raise ValueError("quaternion is degenerate")
    current = current / np.linalg.norm(current)
    target = target / np.linalg.norm(target)
    current_conjugate = np.asarray([-current[0], -current[1], -current[2], current[3]])
    x1, y1, z1, w1 = target
    x2, y2, z2, w2 = current_conjugate
    delta = np.asarray(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )
    if delta[3] < 0:
        delta *= -1
    sine = float(np.linalg.norm(delta[:3]))
    if sine <= 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(sine, float(delta[3]))
    return delta[:3] * (angle / sine)


def tracking_corrected_chunk(
    chunk: list[dict[str, np.ndarray]],
    public_state: object,
    demo_state: object,
) -> tuple[list[dict[str, np.ndarray]], dict[str, list[float]]]:
    """Track a human path using only bounded public EEF pose feedback."""
    if len(chunk) != 5:
        raise ValueError("tracking correction requires one five-step chunk")
    live = _finite_vector(public_state, width=GROOT_STATE_WIDTH, name="public state")
    demo = _finite_vector(demo_state, width=GROOT_STATE_WIDTH, name="demo state")
    position = np.clip(
        (demo[7:10] - live[7:10]) * 15.0,
        -MAX_POSITION_CORRECTION_NORMALIZED,
        MAX_POSITION_CORRECTION_NORMALIZED,
    )
    rotation = np.clip(
        quaternion_delta_rotvec(live[10:14], demo[10:14]) * 2.0,
        -MAX_ROTATION_CORRECTION_NORMALIZED,
        MAX_ROTATION_CORRECTION_NORMALIZED,
    )
    corrected: list[dict[str, np.ndarray]] = []
    for original in chunk:
        action = {key: np.asarray(value, dtype=np.float64).copy() for key, value in original.items()}
        action["action.end_effector_position"] = np.clip(
            action["action.end_effector_position"] + position, -1.0, 1.0
        )
        action["action.end_effector_rotation"] = np.clip(
            action["action.end_effector_rotation"] + rotation, -1.0, 1.0
        )
        corrected.append(action)
    return corrected, {
        "position_normalized": position.tolist(),
        "rotation_normalized": rotation.tolist(),
    }


def public_state_to_groot(value: dict[str, object]) -> np.ndarray:
    """Reconstruct the documented public GR00T state ordering."""
    groups = (
        ("state.base_position", 3),
        ("state.base_rotation", 4),
        ("state.end_effector_position_relative", 3),
        ("state.end_effector_rotation_relative", 4),
        ("state.gripper_qpos", 2),
    )
    parts = [
        _finite_vector(value.get(key), width=width, name=key)
        for key, width in groups
    ]
    result = np.concatenate(parts)
    if result.shape != (GROOT_STATE_WIDTH,):
        raise AssertionError("public state width drift")
    return result


def quaternion_distance_rad(first: object, second: object) -> float:
    left = _finite_vector(first, width=4, name="first quaternion")
    right = _finite_vector(second, width=4, name="second quaternion")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        raise ValueError("quaternion is degenerate")
    similarity = float(abs(np.dot(left / left_norm, right / right_norm)))
    return float(2.0 * np.arccos(np.clip(similarity, -1.0, 1.0)))


def trajectory_error(public_state: object, demo_state: object) -> dict[str, float]:
    live = _finite_vector(public_state, width=GROOT_STATE_WIDTH, name="public state")
    demo = _finite_vector(demo_state, width=GROOT_STATE_WIDTH, name="demo state")
    return {
        "eef_position_error_m": float(np.linalg.norm(live[7:10] - demo[7:10])),
        "eef_orientation_error_rad": quaternion_distance_rad(
            live[10:14], demo[10:14]
        ),
        "gripper_error_m": float(np.max(np.abs(live[14:16] - demo[14:16]))),
    }


def select_tracking_reference(
    public_state: object,
    demo_states: object,
    *,
    previous_reference_index: int,
) -> tuple[int, dict[str, float]]:
    """Monotonically align live public pose to a bounded demo path window."""
    live = _finite_vector(public_state, width=GROOT_STATE_WIDTH, name="public state")
    states = np.asarray(demo_states, dtype=np.float64)
    if (
        states.ndim != 2
        or states.shape[1] != GROOT_STATE_WIDTH
        or not np.isfinite(states).all()
    ):
        raise ValueError("demonstration states have an invalid shape or value")
    if not 0 <= previous_reference_index < len(states):
        raise ValueError("previous demonstration reference is out of range")
    stop = min(
        len(states) - 1,
        previous_reference_index + MAX_REFERENCE_ADVANCE_STEPS,
    )
    candidates: list[tuple[float, int, dict[str, float]]] = []
    for index in range(previous_reference_index, stop + 1):
        error = trajectory_error(live, states[index])
        score = (
            error["eef_position_error_m"]
            + 0.02 * error["eef_orientation_error_rad"]
            + error["gripper_error_m"]
        )
        candidates.append((score, index, error))
    _, selected, _ = min(candidates, key=lambda row: (row[0], row[1]))
    error = trajectory_error(live, states[selected])
    return selected, error


def validate_initial_match(public_state: object, demo_state: object) -> dict[str, float]:
    error = trajectory_error(public_state, demo_state)
    if error["eef_position_error_m"] > MAX_INITIAL_EEF_POSITION_ERROR_M:
        raise ValueError("live EEF position is too far from the retrieved demonstration")
    if (
        error["eef_orientation_error_rad"]
        > MAX_INITIAL_EEF_ORIENTATION_ERROR_RAD
    ):
        raise ValueError(
            "live EEF orientation is too far from the retrieved demonstration"
        )
    return error


def validate_tracking_match(
    public_state: object, demo_state: object
) -> dict[str, float]:
    error = trajectory_error(public_state, demo_state)
    if error["eef_position_error_m"] > MAX_TRACKING_EEF_POSITION_ERROR_M:
        raise ValueError("live EEF position diverged from the demonstration")
    if (
        error["eef_orientation_error_rad"]
        > MAX_TRACKING_EEF_ORIENTATION_ERROR_RAD
    ):
        raise ValueError("live EEF orientation diverged from the demonstration")
    return error


def trajectory_sha256(actions: object, states: object) -> str:
    action_array = np.asarray(actions, dtype="<f8")
    state_array = np.asarray(states, dtype="<f8")
    if (
        action_array.ndim != 2
        or action_array.shape[1] != GROOT_ACTION_WIDTH
        or state_array.shape != (action_array.shape[0], GROOT_STATE_WIDTH)
        or not np.isfinite(action_array).all()
        or not np.isfinite(state_array).all()
    ):
        raise ValueError("demonstration trajectory shape or values are invalid")
    digest = hashlib.sha256()
    digest.update(action_array.tobytes(order="C"))
    digest.update(state_array.tobytes(order="C"))
    return digest.hexdigest()
