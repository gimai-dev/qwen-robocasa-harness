"""Public-Jacobian resolution for Qwen-authored Panda Cartesian deltas."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence

from .joint_protocol import JOINT_LIMIT_MARGIN, JOINT_LIMITS
from .panda_embodiment import (
    _rotation_log_vector,
    _rotation_times_transpose,
    panda_fk,
    panda_local_jacobian,
)

DLS_DAMPING_SQUARED = 0.0025
MAX_TRANSLATION_COMPONENT_M = 0.03
MAX_TRANSLATION_NORM_M = 0.04
MAX_ROTATION_COMPONENT_RAD = 0.08
MAX_ROTATION_NORM_RAD = 0.10
MAX_DERIVED_JOINT_DELTA_RAD = 0.8
MAX_NULLSPACE_CENTERING_DELTA_RAD = 0.10


@dataclasses.dataclass(frozen=True)
class CartesianDeltaCommand:
    observation_id: str
    translation_m: tuple[float, float, float]
    rotation_axis_angle_rad: tuple[float, float, float]
    gripper: str
    note: str
    kind: str = "cartesian_delta"


@dataclasses.dataclass(frozen=True)
class CartesianResolution:
    joint_endpoint: tuple[float, ...]
    gripper_open: float
    predicted_delta: tuple[float, ...]
    residual_norm: float


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{label} must be a finite number")
    return output


def _vector(value: object, *, width: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must contain exactly {width} values")
    if len(value) != width:
        raise ValueError(f"{label} must contain exactly {width} values")
    return tuple(
        _finite(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(item * item for item in value))


def decode_cartesian_delta(
    value: Mapping[str, object], *, observation_id: str
) -> CartesianDeltaCommand:
    required = {
        "kind",
        "observation_id",
        "translation_m",
        "rotation_axis_angle_rad",
        "gripper",
        "note",
    }
    if set(value) != required or value.get("kind") != "cartesian_delta":
        raise ValueError("Cartesian command fields drifted")
    if value.get("observation_id") != observation_id:
        raise ValueError("Cartesian command has a stale observation_id")
    note = value.get("note")
    if not isinstance(note, str) or not 1 <= len(note) <= 320:
        raise ValueError("Cartesian command note is outside its bound")
    gripper = value.get("gripper")
    if gripper not in {"open", "hold", "close"}:
        raise ValueError("Cartesian gripper intent is invalid")
    translation = _vector(value.get("translation_m"), width=3, label="translation")
    rotation = _vector(
        value.get("rotation_axis_angle_rad"), width=3, label="rotation"
    )
    if any(abs(item) > MAX_TRANSLATION_COMPONENT_M for item in translation):
        raise ValueError("Cartesian translation component exceeds its bound")
    if _norm(translation) > MAX_TRANSLATION_NORM_M:
        raise ValueError("Cartesian translation norm exceeds its bound")
    if any(abs(item) > MAX_ROTATION_COMPONENT_RAD for item in rotation):
        raise ValueError("Cartesian rotation component exceeds its bound")
    if _norm(rotation) > MAX_ROTATION_NORM_RAD:
        raise ValueError("Cartesian rotation norm exceeds its bound")
    if not any(translation) and not any(rotation) and gripper == "hold":
        raise ValueError("zero Cartesian delta requires a gripper transition")
    return CartesianDeltaCommand(
        observation_id=observation_id,
        translation_m=(translation[0], translation[1], translation[2]),
        rotation_axis_angle_rad=(rotation[0], rotation[1], rotation[2]),
        gripper=str(gripper),
        note=note,
    )


def _matrix(value: object, *, label: str) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a 3x7 matrix")
    if len(value) != 3:
        raise ValueError(f"{label} must be a 3x7 matrix")
    rows = tuple(_vector(row, width=7, label=f"{label} row") for row in value)
    return rows


def _solve(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, ...]:
    width = len(vector)
    augmented = [list(matrix[row]) + [float(vector[row])] for row in range(width)]
    for column in range(width):
        pivot = max(range(column, width), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("Cartesian DLS solve is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [item / scale for item in augmented[column]]
        for row in range(width):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                item - factor * pivot_item
                for item, pivot_item in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return tuple(augmented[row][-1] for row in range(width))


def _nullspace_centering_delta(
    jacobian: Sequence[Sequence[float]],
    current: Sequence[float],
) -> tuple[float, ...]:
    # Damping belongs to the primary task solve. A damped inverse here leaks
    # posture correction into the authored end-effector motion.
    row_basis: list[tuple[float, ...]] = []
    for row in jacobian:
        residual = tuple(row)
        for _ in range(2):
            for basis in row_basis:
                coefficient = sum(a * b for a, b in zip(residual, basis, strict=True))
                residual = tuple(a - coefficient * b for a, b in zip(residual, basis, strict=True))
        length = _norm(residual)
        if length > 1e-10:
            row_basis.append(tuple(value / length for value in residual))
    toward_center = tuple(
        (lower + upper) / 2.0 - current[index]
        for index, (lower, upper) in enumerate(JOINT_LIMITS)
    )
    nullspace = toward_center
    for basis in row_basis:
        coefficient = sum(a * b for a, b in zip(nullspace, basis, strict=True))
        nullspace = tuple(a - coefficient * b for a, b in zip(nullspace, basis, strict=True))
    largest = max(abs(item) for item in nullspace)
    if largest <= 1e-12:
        return (0.0,) * 7
    scale = min(1.0, MAX_NULLSPACE_CENTERING_DELTA_RAD / largest)
    return tuple(item * scale for item in nullspace)


def _resolve_linear_cartesian_delta(
    command: CartesianDeltaCommand,
    public_state: Mapping[str, object],
    *,
    current_gripper: object,
    orientation_weight: float = 1.0,
    max_predicted_translation_m: float | None = None,
) -> CartesianResolution:
    current = _vector(
        public_state.get("state.arm_joint_position"),
        width=7,
        label="arm joint position",
    )
    translation_jacobian = _matrix(
        public_state.get("state.arm_translation_jacobian"),
        label="translation Jacobian",
    )
    rotation_jacobian = _matrix(
        public_state.get("state.arm_rotation_jacobian"),
        label="rotation Jacobian",
    )
    weight = _finite(orientation_weight, label="orientation weight")
    if not 0.0 < weight <= 1.0:
        raise ValueError("orientation weight must be in (0, 1]")
    unweighted_jacobian = translation_jacobian + rotation_jacobian
    jacobian = translation_jacobian + tuple(
        tuple(weight * item for item in row) for row in rotation_jacobian
    )
    unweighted_requested = command.translation_m + command.rotation_axis_angle_rad
    requested = command.translation_m + tuple(
        weight * item for item in command.rotation_axis_angle_rad
    )
    normal = [
        [
            sum(jacobian[row][joint] * jacobian[column][joint] for joint in range(7))
            + (DLS_DAMPING_SQUARED if row == column else 0.0)
            for column in range(6)
        ]
        for row in range(6)
    ]
    dual = _solve(normal, requested)
    primary_delta = tuple(
        sum(jacobian[row][joint] * dual[row] for row in range(6))
        for joint in range(7)
    )
    if any(abs(item) > MAX_DERIVED_JOINT_DELTA_RAD for item in primary_delta):
        raise ValueError("derived Cartesian joint delta exceeds its bound")
    posture_delta = (
        _nullspace_centering_delta(jacobian, current)
        if any(requested)
        else (0.0,) * 7
    )
    alpha_low = 0.0
    alpha_high = 1.0
    for index, correction in enumerate(posture_delta):
        if correction == 0.0:
            continue
        base_endpoint = current[index] + primary_delta[index]
        lower, upper = JOINT_LIMITS[index]
        endpoint_bounds = sorted((
            (lower + JOINT_LIMIT_MARGIN - base_endpoint) / correction,
            (upper - JOINT_LIMIT_MARGIN - base_endpoint) / correction,
        ))
        delta_bounds = sorted((
            (-MAX_DERIVED_JOINT_DELTA_RAD - primary_delta[index]) / correction,
            (MAX_DERIVED_JOINT_DELTA_RAD - primary_delta[index]) / correction,
        ))
        alpha_low = max(alpha_low, endpoint_bounds[0], delta_bounds[0])
        alpha_high = min(alpha_high, endpoint_bounds[1], delta_bounds[1])
    if alpha_low <= alpha_high and alpha_high >= 0.0 and alpha_low <= 1.0:
        alpha = min(1.0, alpha_high)
    else:
        alpha = 0.0
    joint_delta = tuple(
        primary_delta[index] + alpha * posture_delta[index] for index in range(7)
    )
    if max_predicted_translation_m is not None:
        translation_limit = _finite(
            max_predicted_translation_m,
            label="predicted Cartesian translation limit",
        )
        if not 0.0 < translation_limit <= MAX_TRANSLATION_NORM_M:
            raise ValueError("predicted Cartesian translation limit is outside its bound")
        predicted_translation = tuple(
            sum(
                translation_jacobian[row][joint] * joint_delta[joint]
                for joint in range(7)
            )
            for row in range(3)
        )
        predicted_translation_norm = _norm(predicted_translation)
        if predicted_translation_norm > translation_limit:
            scale = translation_limit / predicted_translation_norm
            joint_delta = tuple(item * scale for item in joint_delta)
    endpoint = tuple(
        current[index] + joint_delta[index] for index in range(7)
    )
    for index, (value, (lower, upper)) in enumerate(
        zip(endpoint, JOINT_LIMITS, strict=True)
    ):
        if not lower + JOINT_LIMIT_MARGIN <= value <= upper - JOINT_LIMIT_MARGIN:
            raise ValueError(f"derived Cartesian joint{index + 1} endpoint is unsafe")
    predicted = tuple(
        sum(
            unweighted_jacobian[row][joint] * joint_delta[joint]
            for joint in range(7)
        )
        for row in range(6)
    )
    residual = _norm(
        tuple(
            expected - actual
            for expected, actual in zip(
                unweighted_requested,
                predicted,
                strict=True,
            )
        )
    )
    gripper = _finite(current_gripper, label="current gripper")
    if not 0.0 <= gripper <= 1.0:
        raise ValueError("current gripper must be between 0 and 1")
    gripper_open = {"open": 1.0, "hold": gripper, "close": 0.0}[command.gripper]
    if not any(requested) and gripper_open == gripper:
        raise ValueError("zero Cartesian delta has no gripper transition")
    return CartesianResolution(
        joint_endpoint=endpoint,
        gripper_open=gripper_open,
        predicted_delta=predicted,
        residual_norm=residual,
    )


def resolve_cartesian_delta(
    command: CartesianDeltaCommand,
    public_state: Mapping[str, object],
    *,
    current_gripper: object,
    orientation_weight: float = 1.0,
    max_predicted_translation_m: float | None = None,
) -> CartesianResolution:
    initial = _resolve_linear_cartesian_delta(
        command, public_state, current_gripper=current_gripper,
        orientation_weight=orientation_weight,
        max_predicted_translation_m=max_predicted_translation_m,
    )
    if not any(command.translation_m + command.rotation_axis_angle_rad):
        return initial
    current = _vector(public_state["state.arm_joint_position"], width=7, label="arm joint position")
    origin = panda_fk(current)
    translation = command.translation_m
    if max_predicted_translation_m is not None and _norm(translation) > max_predicted_translation_m:
        scale = max_predicted_translation_m / _norm(translation)
        translation = tuple(value * scale for value in translation)
    target_position = tuple(a + b for a, b in zip(origin.position_m, translation, strict=True))
    angle = _norm(command.rotation_axis_angle_rad)
    if angle == 0.0:
        target_rotation = origin.rotation_matrix
    else:
        x, y, z = (value / angle for value in command.rotation_axis_angle_rad)
        skew = ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0))
        increment = tuple(tuple(
            (1.0 if i == j else 0.0) + math.sin(angle) * skew[i][j]
            + (1.0 - math.cos(angle)) * sum(skew[i][k] * skew[k][j] for k in range(3))
            for j in range(3)) for i in range(3))
        target_rotation = tuple(tuple(
            sum(increment[i][k] * origin.rotation_matrix[k][j] for k in range(3))
            for j in range(3)) for i in range(3))

    def error_at(endpoint: Sequence[float]) -> tuple[float, ...]:
        pose = panda_fk(endpoint)
        position_error = tuple(a - b for a, b in zip(target_position, pose.position_m, strict=True))
        rotation_error = _rotation_log_vector(_rotation_times_transpose(target_rotation, pose.rotation_matrix))
        return position_error + tuple(orientation_weight * value for value in rotation_error)

    endpoint = initial.joint_endpoint
    # Correct the finite-step linearization error against the same authored
    # pose, using only the public robot kinematics. Send one final endpoint.
    for _ in range(2):
        error = error_at(endpoint)
        local = panda_local_jacobian(endpoint)
        jacobian = local["translation"] + [
            [orientation_weight * value for value in row] for row in local["rotation"]
        ]
        normal = [[
            sum(jacobian[i][k] * jacobian[j][k] for k in range(7))
            + (DLS_DAMPING_SQUARED if i == j else 0.0)
            for j in range(6)] for i in range(6)]
        dual = _solve(normal, error)
        trial = tuple(endpoint[k] + sum(jacobian[i][k] * dual[i] for i in range(6)) for k in range(7))
        if any(
            abs(value - current[i]) > MAX_DERIVED_JOINT_DELTA_RAD
            or not lower + JOINT_LIMIT_MARGIN <= value <= upper - JOINT_LIMIT_MARGIN
            for i, (value, (lower, upper)) in enumerate(zip(trial, JOINT_LIMITS, strict=True))
        ) or _norm(error_at(trial)) >= _norm(error):
            break
        endpoint = trial
    final_pose = panda_fk(endpoint)
    predicted_translation = tuple(b - a for a, b in zip(origin.position_m, final_pose.position_m, strict=True))
    if max_predicted_translation_m is not None and _norm(predicted_translation) > max_predicted_translation_m:
        return initial
    predicted_rotation = _rotation_log_vector(_rotation_times_transpose(final_pose.rotation_matrix, origin.rotation_matrix))
    predicted = predicted_translation + predicted_rotation
    residual = _norm(tuple(a - b for a, b in zip(command.translation_m + command.rotation_axis_angle_rad, predicted, strict=True)))
    return CartesianResolution(endpoint, initial.gripper_open, predicted, residual)
