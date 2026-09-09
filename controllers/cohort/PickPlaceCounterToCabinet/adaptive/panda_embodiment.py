"""Pure RoboSuite Panda grip-site geometry.

All positions are metres and all joint angles are radians. ``panda_fk`` maps
the Panda ``link0`` robot-base frame to the exact right-gripper ``grip_site``
used by RoboSuite's public base-relative EEF position observable. Rotation
matrices map coordinates from the described local frame into its parent frame.

MJCF quaternions are scalar-first ``(w, x, y, z)``. Public observation
quaternions handled by this module are documented separately at their boundary.
The implementation intentionally imports neither RoboSuite nor MuJoCo.

Installed-model provenance (read from the H200 validation installation):

* ``/home/jli/work/robocasa-inspect-official/robosuite/robosuite/models/``
  ``assets/robots/panda/robot.xml``
  SHA-256 ``67f62e98681d2ae0d5c50352105492b0041bcc9b9502e83fed02823195338c90``
* ``/home/jli/work/robocasa-inspect-official/robosuite/robosuite/models/``
  ``assets/grippers/panda_gripper.xml``
  SHA-256 ``066fd7ae45afb11e1844869360e587646e9768210ba5389aa3cd35292cde29c1``
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

ROBOT_MODEL_SOURCE_PATH = (
    "/home/jli/work/robocasa-inspect-official/robosuite/robosuite/models/"
    "assets/robots/panda/robot.xml"
)
ROBOT_MODEL_SHA256 = "67f62e98681d2ae0d5c50352105492b0041bcc9b9502e83fed02823195338c90"
GRIPPER_MODEL_SOURCE_PATH = (
    "/home/jli/work/robocasa-inspect-official/robosuite/robosuite/models/"
    "assets/grippers/panda_gripper.xml"
)
GRIPPER_MODEL_SHA256 = (
    "066fd7ae45afb11e1844869360e587646e9768210ba5389aa3cd35292cde29c1"
)

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]
Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]

OFFICIAL_PUBLIC_STATE_KEYS = frozenset(
    {
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.base_position",
        "state.base_rotation",
        "state.gripper_qpos",
    }
)
PUBLIC_TELEMETRY_KEYS = frozenset(
    {
        "state.arm_joint_position",
        "state.arm_joint_velocity",
        "state.arm_applied_torque",
        "state.end_effector_wrench",
        "state.arm_translation_jacobian",
        "state.arm_rotation_jacobian",
        "state.end_effector_external_pixels",
    }
)
PUBLIC_STATE_FIELDS = OFFICIAL_PUBLIC_STATE_KEYS | PUBLIC_TELEMETRY_KEYS


@dataclass(frozen=True)
class Pose:
    """Immutable position and orientation of one frame in another frame."""

    position_m: Vector3
    rotation_matrix: Matrix3


_IDENTITY_4: Matrix4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _matmul4(left: Matrix4, right: Matrix4) -> Matrix4:
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][column] for inner in range(4))
            for column in range(4)
        )
        for row in range(4)
    )  # type: ignore[return-value]


def _quaternion_wxyz_rotation(quaternion: Sequence[float]) -> Matrix3:
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("quaternion must have finite nonzero norm")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def _transform(
    position_m: Sequence[float] = (0.0, 0.0, 0.0),
    quaternion_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
) -> Matrix4:
    rotation = _quaternion_wxyz_rotation(quaternion_wxyz)
    position = tuple(float(value) for value in position_m)
    return (
        (*rotation[0], position[0]),
        (*rotation[1], position[1]),
        (*rotation[2], position[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def _rotation_z(angle_rad: float) -> Matrix4:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return (
        (cosine, -sine, 0.0, 0.0),
        (sine, cosine, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _pose(transform: Matrix4) -> Pose:
    return Pose(
        position_m=(transform[0][3], transform[1][3], transform[2][3]),
        rotation_matrix=(transform[0][:3], transform[1][:3], transform[2][:3]),
    )


# Parent-body transforms transcribed verbatim from the nested Panda MJCF bodies
# link1 through link7. Every joint rotates about its child body's local +z axis.
_PANDA_JOINT_FRAMES = (
    ((0.0, 0.0, 0.333), (1.0, 0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.707107, -0.707107, 0.0, 0.0)),
    ((0.0, -0.316, 0.0), (0.707107, 0.707107, 0.0, 0.0)),
    ((0.0825, 0.0, 0.0), (0.707107, 0.707107, 0.0, 0.0)),
    ((-0.0825, 0.384, 0.0), (0.707107, -0.707107, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.707107, 0.707107, 0.0, 0.0)),
    ((0.088, 0.0, 0.0), (0.707107, 0.707107, 0.0, 0.0)),
)
_PANDA_JOINT_TRANSFORMS = tuple(
    _transform(position, quaternion) for position, quaternion in _PANDA_JOINT_FRAMES
)

# Exact fixed chain after joint7: Panda right_hand, mounted PandaGripper root,
# then its eef body. grip_site is at the eef-body origin.
_FLANGE_TO_GRIP_SITE_FRAMES = (
    ((0.0, 0.0, 0.1065), (0.924, 0.0, 0.0, -0.383)),
    ((0.0, 0.0, 0.0), (0.707107, 0.0, 0.0, -0.707107)),
    ((0.0, 0.0, 0.097), (1.0, 0.0, 0.0, 0.0)),
)


def _compose_fixed_frames(
    frames: Sequence[tuple[Sequence[float], Sequence[float]]],
) -> Matrix4:
    transform = _IDENTITY_4
    for position, quaternion in frames:
        transform = _matmul4(transform, _transform(position, quaternion))
    return transform


_FLANGE_TO_GRIP_SITE_MATRIX = _compose_fixed_frames(_FLANGE_TO_GRIP_SITE_FRAMES)
FLANGE_TO_GRIP_SITE = _pose(_FLANGE_TO_GRIP_SITE_MATRIX)


def _joint_values(qpos: object) -> tuple[float, ...]:
    if isinstance(qpos, (str, bytes)) or not isinstance(qpos, Sequence):
        raise TypeError("Panda qpos must be a seven-value sequence")
    if len(qpos) != 7:
        raise ValueError("Panda qpos must contain exactly seven values")
    values: list[float] = []
    for value in qpos:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("Panda qpos values must be real numbers")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Panda qpos values must be finite")
        values.append(numeric)
    return tuple(values)


def panda_fk(qpos: object) -> Pose:
    """Return the base-frame pose of RoboSuite's Panda grip site."""

    joint_values = _joint_values(qpos)
    transform = _IDENTITY_4
    for fixed_transform, joint_value in zip(
        _PANDA_JOINT_TRANSFORMS, joint_values, strict=True
    ):
        transform = _matmul4(transform, fixed_transform)
        transform = _matmul4(transform, _rotation_z(joint_value))
    return _pose(_matmul4(transform, _FLANGE_TO_GRIP_SITE_MATRIX))


def _rotation_times_transpose(left: Matrix3, right: Matrix3) -> Matrix3:
    return tuple(
        tuple(
            sum(left[row][inner] * right[column][inner] for inner in range(3))
            for column in range(3)
        )
        for row in range(3)
    )  # type: ignore[return-value]


def _rotation_log_vector(rotation: Matrix3) -> Vector3:
    """Return the axis-angle log vector of a proper near-identity rotation."""

    cosine = max(
        -1.0,
        min(
            1.0,
            (rotation[0][0] + rotation[1][1] + rotation[2][2] - 1.0) / 2.0,
        ),
    )
    angle = math.acos(cosine)
    skew = (
        rotation[2][1] - rotation[1][2],
        rotation[0][2] - rotation[2][0],
        rotation[1][0] - rotation[0][1],
    )
    if angle < 1e-7:
        return tuple(0.5 * value for value in skew)  # type: ignore[return-value]
    scale = angle / (2.0 * math.sin(angle))
    return tuple(scale * value for value in skew)  # type: ignore[return-value]


def panda_local_jacobian(
    qpos: object, epsilon: float = 1e-5
) -> dict[str, list[list[float]]]:
    """Return a base-frame finite-difference 6x7 geometric Jacobian.

    Translation rows are metres per radian. Rotation rows are spatial
    axis-angle radians per radian, computed from ``R_plus @ R_minus.T``.
    """

    joint_values = _joint_values(qpos)
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
        raise TypeError("Jacobian epsilon must be a real number")
    step = float(epsilon)
    if not math.isfinite(step) or not 1e-7 <= step <= 1e-2:
        raise ValueError("Jacobian epsilon must be in [1e-7, 1e-2]")
    translation = [[0.0 for _ in range(7)] for _ in range(3)]
    rotation = [[0.0 for _ in range(7)] for _ in range(3)]
    denominator = 2.0 * step
    for joint_index in range(7):
        plus = list(joint_values)
        minus = list(joint_values)
        plus[joint_index] += step
        minus[joint_index] -= step
        plus_pose = panda_fk(plus)
        minus_pose = panda_fk(minus)
        for axis in range(3):
            translation[axis][joint_index] = (
                plus_pose.position_m[axis] - minus_pose.position_m[axis]
            ) / denominator
        relative_rotation = _rotation_times_transpose(
            plus_pose.rotation_matrix, minus_pose.rotation_matrix
        )
        rotation_delta = _rotation_log_vector(relative_rotation)
        for axis in range(3):
            rotation[axis][joint_index] = rotation_delta[axis] / denominator
    return {"translation": translation, "rotation": rotation}


def _finite_vector(value: object, *, width: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError(f"{label} must be a {width}-value sequence")
    try:
        items = list(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(
            f"{label} must be a {width}-value sequence"
        ) from error
    if len(items) != width:
        raise ValueError(f"{label} must contain exactly {width} values")
    result: list[float] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(f"{label} values must be real numbers")
        numeric = float(item)
        if not math.isfinite(numeric):
            raise ValueError(f"{label} values must be finite")
        result.append(numeric)
    return tuple(result)


def compose_world_pose(
    base_position_world_m: object,
    base_rotation_xyzw: object,
    relative_position_m: object,
    relative_rotation_xyzw: object,
) -> Pose:
    """Compose public base and base-relative EEF poses into a world pose.

    Public RoboSuite observation quaternions use vector-first ``(x, y, z, w)``.
    The relative translation is expressed in the robot-base frame.
    """

    base_position = _finite_vector(
        base_position_world_m, width=3, label="base position"
    )
    relative_position = _finite_vector(
        relative_position_m, width=3, label="relative EEF position"
    )
    base_xyzw = _finite_vector(base_rotation_xyzw, width=4, label="base rotation")
    relative_xyzw = _finite_vector(
        relative_rotation_xyzw, width=4, label="relative EEF rotation"
    )
    base_transform = _transform(base_position, (base_xyzw[3], *base_xyzw[:3]))
    relative_transform = _transform(
        relative_position, (relative_xyzw[3], *relative_xyzw[:3])
    )
    return _pose(_matmul4(base_transform, relative_transform))


def _finite_scalar(value: object, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def _camera_rotation(value: object) -> Matrix3:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("camera rotation must be a 3x3 sequence")
    if len(value) != 3:
        raise ValueError("camera rotation must be 3x3")
    rows = tuple(
        _finite_vector(row, width=3, label="camera rotation row") for row in value
    )
    for first in range(3):
        for second in range(3):
            dot = sum(rows[row][first] * rows[row][second] for row in range(3))
            expected = 1.0 if first == second else 0.0
            if not math.isclose(dot, expected, abs_tol=1e-6, rel_tol=0.0):
                raise ValueError("camera rotation must be orthonormal")
    determinant = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    if not math.isclose(determinant, 1.0, abs_tol=1e-6, rel_tol=0.0):
        raise ValueError("camera rotation must be proper")
    return rows  # type: ignore[return-value]


def _image_dimension(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def project_world_point(
    point_world: object, calibration: Mapping[str, object]
) -> dict[str, object]:
    """Project one world point with official fixed-camera calibration.

    MuJoCo camera axes are +x right, +y up, and -z forward. Pixels use +u
    right and +v down. Off-image points retain their unclamped finite pixels and
    return ``visible=False``; points at or behind the camera are rejected.
    """

    if not isinstance(calibration, Mapping):
        raise TypeError("camera calibration must be a mapping")
    required = {
        "image_width_px",
        "image_height_px",
        "fx_px",
        "fy_px",
        "cx_px",
        "cy_px",
        "camera_position_world_m",
        "camera_xmat_world",
        "projection",
    }
    if not required.issubset(calibration):
        raise ValueError("camera calibration is missing required fields")
    if calibration["projection"] != ("mujoco-camera-x-right-y-up-minus-z-forward"):
        raise ValueError("camera projection convention is unsupported")
    width = _image_dimension(calibration["image_width_px"], label="image width")
    height = _image_dimension(calibration["image_height_px"], label="image height")
    fx = _finite_scalar(calibration["fx_px"], label="fx", positive=True)
    fy = _finite_scalar(calibration["fy_px"], label="fy", positive=True)
    cx = _finite_scalar(calibration["cx_px"], label="cx")
    cy = _finite_scalar(calibration["cy_px"], label="cy")
    point = _finite_vector(point_world, width=3, label="world point")
    camera_position = _finite_vector(
        calibration["camera_position_world_m"],
        width=3,
        label="camera position",
    )
    camera_rotation = _camera_rotation(calibration["camera_xmat_world"])
    offset = tuple(point[axis] - camera_position[axis] for axis in range(3))
    camera_point = tuple(
        sum(camera_rotation[row][axis] * offset[row] for row in range(3))
        for axis in range(3)
    )
    depth = -camera_point[2]
    if depth <= 0.0:
        raise ValueError("world point is not in front of the camera")
    u = cx + fx * camera_point[0] / depth
    v = cy - fy * camera_point[1] / depth
    if not all(math.isfinite(value) for value in (u, v, depth)):
        raise ValueError("camera projection is non-finite")
    return {
        "u_px": u,
        "v_px": v,
        "depth_m": depth,
        "visible": 0.0 <= u < width and 0.0 <= v < height,
    }


def _iterable_items(value: object, *, label: str) -> list[object]:
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError(f"{label} must be a sequence")
    try:
        return list(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{label} must be a sequence") from error


def _joint_names(value: object, *, label: str) -> list[str]:
    items = _iterable_items(value, label=label)
    if any(not isinstance(item, str) or not item for item in items):
        raise TypeError(f"{label} must contain non-empty strings")
    names = list(items)
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must be unique")
    return names  # type: ignore[return-value]


def select_named_arm_vector(
    values: object,
    robot_names: object,
    arm_names: object,
    label: str,
) -> list[float]:
    """Select exactly seven finite arm values by joint name."""

    if not isinstance(label, str) or not label:
        raise TypeError("arm vector label must be a non-empty string")
    items = _iterable_items(values, label=label)
    names = _joint_names(robot_names, label=f"{label} robot names")
    selected_names = _joint_names(arm_names, label=f"{label} arm names")
    if len(items) != len(names):
        raise ValueError(f"{label} values and robot names disagree")
    if len(selected_names) != 7:
        raise ValueError(f"{label} must select exactly seven arm joints")
    by_name = {
        name: _finite_scalar(value, label=f"{label} {name}")
        for name, value in zip(names, items, strict=True)
    }
    missing = [name for name in selected_names if name not in by_name]
    if missing:
        raise ValueError(f"{label} is missing named arm joints")
    return [by_name[name] for name in selected_names]


def _measured_vector(value: object, *, width: int, label: str) -> list[float]:
    items = _iterable_items(value, label=label)
    if len(items) != width:
        raise ValueError(f"{label} must contain exactly {width} values")
    return [_finite_scalar(item, label=label) for item in items]


def _official_proprioception(environment: object) -> Mapping[str, object]:
    source = environment.unwrapped
    if getattr(source, "viewer_get_obs", False):
        observations = source.viewer._get_observations(force_update=False)
    else:
        observations = source._get_observations(force_update=False)
    if not isinstance(observations, Mapping):
        raise TypeError("official proprioceptive observation must be a mapping")
    return observations


def _external_pixel_record(
    point_world: object, calibration: Mapping[str, object]
) -> dict[str, object]:
    try:
        projected = project_world_point(point_world, calibration)
    except ValueError as error:
        if str(error) != "world point is not in front of the camera":
            raise
        return {
            "u_px": None,
            "v_px": None,
            "visible": False,
            "depth_valid": False,
        }
    return {
        "u_px": projected["u_px"],
        "v_px": projected["v_px"],
        "visible": projected["visible"],
        "depth_valid": True,
    }


def read_public_telemetry(
    environment: object, camera_calibration: Mapping[str, object]
) -> dict[str, object]:
    """Read strictly validated Panda proprioception and public geometry."""

    source = environment.unwrapped
    robots = source.robots
    if isinstance(robots, (str, bytes, Mapping)) or len(robots) != 1:
        raise ValueError("telemetry requires exactly one robot")
    robot = robots[0]
    arm_names = _joint_names(robot.robot_arm_joints, label="arm joint names")
    if len(arm_names) != 7:
        raise ValueError("Panda arm joint names must contain exactly seven names")
    qpos = select_named_arm_vector(
        robot.get_robot_joint_positions(),
        robot.robot_joints,
        arm_names,
        "arm joint position",
    )
    raw = _official_proprioception(environment)
    if "robot0_joint_vel" not in raw:
        raise ValueError("official robot0_joint_vel observation is missing")
    qvel = select_named_arm_vector(
        raw["robot0_joint_vel"],
        robot.robot_joints,
        arm_names,
        "arm joint velocity",
    )

    controller = robot.part_controllers["right"]
    raw_torques = controller.torques
    if raw_torques is None:
        if source.timestep != 0:
            raise ValueError("post-action applied torque is unavailable")
        applied_torque: dict[str, object] = {
            "available": False,
            "values_nm": None,
        }
    else:
        applied_torque = {
            "available": True,
            "values_nm": select_named_arm_vector(
                raw_torques,
                arm_names,
                arm_names,
                "applied torque",
            ),
        }
    force = _measured_vector(
        robot.ee_force["right"], width=3, label="end-effector force"
    )
    torque = _measured_vector(
        robot.ee_torque["right"], width=3, label="end-effector torque"
    )

    public_pose_keys = {
        "robot0_base_pos",
        "robot0_base_quat",
        "robot0_base_to_eef_pos",
        "robot0_base_to_eef_quat",
    }
    if not public_pose_keys.issubset(raw):
        raise ValueError("public base-relative end-effector pose is missing")
    world_eef = compose_world_pose(
        raw["robot0_base_pos"],
        raw["robot0_base_quat"],
        raw["robot0_base_to_eef_pos"],
        raw["robot0_base_to_eef_quat"],
    )
    if not isinstance(camera_calibration, Mapping):
        raise TypeError("camera calibration must be a mapping")
    if not {"left", "right"}.issubset(camera_calibration):
        raise ValueError("fixed external camera calibration is missing")
    external_pixels: dict[str, object] = {}
    for label in ("left", "right"):
        calibration = camera_calibration[label]
        if not isinstance(calibration, Mapping):
            raise TypeError(f"{label} camera calibration must be a mapping")
        external_pixels[label] = _external_pixel_record(
            world_eef.position_m, calibration
        )

    jacobian = panda_local_jacobian(qpos)
    return {
        "state.arm_joint_position": qpos,
        "state.arm_joint_velocity": qvel,
        "state.arm_applied_torque": applied_torque,
        "state.end_effector_wrench": {
            "force_n": force,
            "torque_nm": torque,
        },
        "state.arm_translation_jacobian": jacobian["translation"],
        "state.arm_rotation_jacobian": jacobian["rotation"],
        "state.end_effector_external_pixels": external_pixels,
    }


def _canonical_json_snapshot(value: object) -> object:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _closed_public_mapping(
    value: object, *, fields: frozenset[str] | set[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"public state {label} schema drifted")
    return value


def _public_matrix(
    value: object, *, rows: int, columns: int, label: str
) -> list[list[float]]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise TypeError(f"public state {label} must be a {rows}x{columns} matrix")
    if len(value) != rows:
        raise ValueError(f"public state {label} must contain exactly {rows} rows")
    return [
        _measured_vector(row, width=columns, label=f"public state {label} row")
        for row in value
    ]


def _public_external_pixels(value: object) -> dict[str, object]:
    pixels = _closed_public_mapping(
        value,
        fields=frozenset({"left", "right"}),
        label="external pixel root",
    )
    output: dict[str, object] = {}
    for camera in ("left", "right"):
        record = _closed_public_mapping(
            pixels[camera],
            fields=frozenset({"u_px", "v_px", "visible", "depth_valid"}),
            label=f"{camera} external pixel",
        )
        visible = record["visible"]
        depth_valid = record["depth_valid"]
        if type(visible) is not bool or type(depth_valid) is not bool:
            raise TypeError(
                f"public state {camera} external pixel flags must be boolean"
            )
        if depth_valid:
            u_px: float | None = _finite_scalar(
                record["u_px"], label=f"public state {camera} u pixel"
            )
            v_px: float | None = _finite_scalar(
                record["v_px"], label=f"public state {camera} v pixel"
            )
        else:
            if record["u_px"] is not None or record["v_px"] is not None or visible:
                raise ValueError(
                    f"public state {camera} invalid-depth pixels "
                    "must be null and hidden"
                )
            u_px = None
            v_px = None
        output[camera] = {
            "u_px": u_px,
            "v_px": v_px,
            "visible": visible,
            "depth_valid": depth_valid,
        }
    return output


def validate_public_state(
    value: object, *, require_torque_available: bool = False
) -> dict[str, object]:
    """Validate and copy the exact Task-3 model-visible public-state schema.

    The producer boundary may publish the documented initial
    ``available=false, values_nm=null`` torque marker. Consumers that require a
    complete post-action observation, including the critic, set
    ``require_torque_available=True``.
    """

    state = _closed_public_mapping(
        value,
        fields=PUBLIC_STATE_FIELDS,
        label="root",
    )
    torque = _closed_public_mapping(
        state["state.arm_applied_torque"],
        fields=frozenset({"available", "values_nm"}),
        label="applied torque",
    )
    torque_available = torque["available"]
    if type(torque_available) is not bool:
        raise TypeError("public state applied torque availability must be boolean")
    if torque_available:
        torque_values: list[float] | None = _measured_vector(
            torque["values_nm"], width=7, label="public state applied torque"
        )
    else:
        if torque["values_nm"] is not None:
            raise ValueError("public state unavailable torque values must be null")
        if require_torque_available:
            raise ValueError("public state applied torque must be available")
        torque_values = None
    wrench = _closed_public_mapping(
        state["state.end_effector_wrench"],
        fields=frozenset({"force_n", "torque_nm"}),
        label="end-effector wrench",
    )
    output = {
        "state.base_position": _measured_vector(
            state["state.base_position"], width=3, label="public state base position"
        ),
        "state.base_rotation": _measured_vector(
            state["state.base_rotation"], width=4, label="public state base rotation"
        ),
        "state.end_effector_position_relative": _measured_vector(
            state["state.end_effector_position_relative"],
            width=3,
            label="public state relative EEF position",
        ),
        "state.end_effector_rotation_relative": _measured_vector(
            state["state.end_effector_rotation_relative"],
            width=4,
            label="public state relative EEF rotation",
        ),
        "state.gripper_qpos": _measured_vector(
            state["state.gripper_qpos"],
            width=2,
            label="public state gripper qpos",
        ),
        "state.arm_joint_position": _measured_vector(
            state["state.arm_joint_position"],
            width=7,
            label="public state arm joint position",
        ),
        "state.arm_joint_velocity": _measured_vector(
            state["state.arm_joint_velocity"],
            width=7,
            label="public state arm joint velocity",
        ),
        "state.arm_applied_torque": {
            "available": torque_available,
            "values_nm": torque_values,
        },
        "state.end_effector_wrench": {
            "force_n": _measured_vector(
                wrench["force_n"], width=3, label="public state EEF force"
            ),
            "torque_nm": _measured_vector(
                wrench["torque_nm"], width=3, label="public state EEF torque"
            ),
        },
        "state.arm_translation_jacobian": _public_matrix(
            state["state.arm_translation_jacobian"],
            rows=3,
            columns=7,
            label="translation jacobian",
        ),
        "state.arm_rotation_jacobian": _public_matrix(
            state["state.arm_rotation_jacobian"],
            rows=3,
            columns=7,
            label="rotation jacobian",
        ),
        "state.end_effector_external_pixels": _public_external_pixels(
            state["state.end_effector_external_pixels"]
        ),
    }
    return _canonical_json_snapshot(output)  # type: ignore[return-value]


@dataclass(frozen=True)
class _TelemetryVectors:
    qpos: list[float]
    qvel: list[float]
    torque: list[float] | None
    force: list[float]
    wrench_torque: list[float]


def _summary_telemetry_vectors(
    telemetry: object, *, label: str, require_torque: bool
) -> _TelemetryVectors:
    if not isinstance(telemetry, Mapping):
        raise TypeError(f"{label} telemetry must be a mapping")
    required = {
        "state.arm_joint_position",
        "state.arm_joint_velocity",
        "state.arm_applied_torque",
        "state.end_effector_wrench",
    }
    if not required.issubset(telemetry):
        raise ValueError(f"{label} telemetry is missing required fields")
    torque_record = telemetry["state.arm_applied_torque"]
    if not isinstance(torque_record, Mapping) or set(torque_record) != {
        "available",
        "values_nm",
    }:
        raise ValueError(f"{label} applied torque record drifted")
    available = torque_record["available"]
    if type(available) is not bool:
        raise TypeError(f"{label} applied torque availability must be boolean")
    if available:
        applied_torque = _measured_vector(
            torque_record["values_nm"], width=7, label=f"{label} applied torque"
        )
    else:
        if torque_record["values_nm"] is not None:
            raise ValueError(f"{label} unavailable applied torque must be null")
        if require_torque:
            raise ValueError(f"{label} applied torque must be available")
        applied_torque = None
    wrench = telemetry["state.end_effector_wrench"]
    if not isinstance(wrench, Mapping) or set(wrench) != {"force_n", "torque_nm"}:
        raise ValueError(f"{label} end-effector wrench record drifted")
    return _TelemetryVectors(
        qpos=_measured_vector(
            telemetry["state.arm_joint_position"],
            width=7,
            label=f"{label} arm joint position",
        ),
        qvel=_measured_vector(
            telemetry["state.arm_joint_velocity"],
            width=7,
            label=f"{label} arm joint velocity",
        ),
        torque=applied_torque,
        force=_measured_vector(
            wrench["force_n"], width=3, label=f"{label} end-effector force"
        ),
        wrench_torque=_measured_vector(
            wrench["torque_nm"], width=3, label=f"{label} end-effector torque"
        ),
    )


def _vector_delta(end: Sequence[float], start: Sequence[float]) -> list[float]:
    return [
        end_value - start_value
        for start_value, end_value in zip(start, end, strict=True)
    ]


def _peak_abs_delta(
    values: Sequence[Sequence[float]], baseline: Sequence[float]
) -> list[float]:
    return [
        max(abs(value[index] - baseline[index]) for value in values)
        for index in range(len(baseline))
    ]


def _peak_delta_norm(
    values: Sequence[Sequence[float]], baseline: Sequence[float]
) -> float:
    return max(
        math.sqrt(
            sum((value[index] - baseline[index]) ** 2 for index in range(len(baseline)))
        )
        for value in values
    )


def _closed_summary_mapping(
    value: object, *, fields: set[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"telemetry summary {label} schema drifted")
    return value


def _nonnegative_summary_scalar(value: object, *, label: str) -> float:
    output = _finite_scalar(value, label=f"telemetry summary {label}")
    if output < 0.0:
        raise ValueError(f"telemetry summary {label} must be nonnegative")
    return output


def _nonnegative_summary_vector(
    value: object, *, width: int, label: str
) -> list[float]:
    output = _measured_vector(value, width=width, label=f"telemetry summary {label}")
    if any(item < 0.0 for item in output):
        raise ValueError(f"telemetry summary {label} must be nonnegative")
    return output


def validate_telemetry_summary(value: object) -> dict[str, object]:
    """Validate and copy the complete closed raw action-summary schema."""

    summary = _closed_summary_mapping(
        value,
        fields={
            "arm_joint_position",
            "arm_joint_velocity",
            "arm_applied_torque",
            "end_effector_wrench",
        },
        label="root",
    )
    qpos = _closed_summary_mapping(
        summary["arm_joint_position"],
        fields={
            "start_rad",
            "end_rad",
            "delta_rad",
            "peak_abs_delta_from_start_rad",
        },
        label="arm joint position",
    )
    qvel = _closed_summary_mapping(
        summary["arm_joint_velocity"],
        fields={"start_rad_s", "end_rad_s", "maximum_abs_rad_s"},
        label="arm joint velocity",
    )
    applied_torque = _closed_summary_mapping(
        summary["arm_applied_torque"],
        fields={
            "start_nm",
            "end_nm",
            "delta_nm",
            "peak_abs_delta_from_start_nm",
        },
        label="arm applied torque",
    )
    wrench = _closed_summary_mapping(
        summary["end_effector_wrench"],
        fields={"force", "torque"},
        label="end-effector wrench",
    )
    force = _closed_summary_mapping(
        wrench["force"],
        fields={"start_n", "end_n", "delta_n", "peak_delta_norm_n"},
        label="end-effector force",
    )
    wrench_torque = _closed_summary_mapping(
        wrench["torque"],
        fields={"start_nm", "end_nm", "delta_nm", "peak_delta_norm_nm"},
        label="end-effector torque",
    )

    start_torque = applied_torque["start_nm"]
    if start_torque is None:
        if (
            applied_torque["delta_nm"] is not None
            or applied_torque["peak_abs_delta_from_start_nm"] is not None
        ):
            raise ValueError("telemetry summary unavailable start torque drifted")
        normalized_start_torque = None
        normalized_delta_torque = None
        normalized_peak_torque = None
    else:
        normalized_start_torque = _measured_vector(
            start_torque, width=7, label="telemetry summary start applied torque"
        )
        normalized_delta_torque = _measured_vector(
            applied_torque["delta_nm"],
            width=7,
            label="telemetry summary applied torque delta",
        )
        normalized_peak_torque = _nonnegative_summary_vector(
            applied_torque["peak_abs_delta_from_start_nm"],
            width=7,
            label="peak applied torque delta",
        )

    return {
        "arm_joint_position": {
            "start_rad": _measured_vector(
                qpos["start_rad"], width=7, label="telemetry summary start arm qpos"
            ),
            "end_rad": _measured_vector(
                qpos["end_rad"], width=7, label="telemetry summary end arm qpos"
            ),
            "delta_rad": _measured_vector(
                qpos["delta_rad"], width=7, label="telemetry summary arm qpos delta"
            ),
            "peak_abs_delta_from_start_rad": _nonnegative_summary_vector(
                qpos["peak_abs_delta_from_start_rad"],
                width=7,
                label="peak arm qpos delta",
            ),
        },
        "arm_joint_velocity": {
            "start_rad_s": _measured_vector(
                qvel["start_rad_s"],
                width=7,
                label="telemetry summary start arm qvel",
            ),
            "end_rad_s": _measured_vector(
                qvel["end_rad_s"], width=7, label="telemetry summary end arm qvel"
            ),
            "maximum_abs_rad_s": _nonnegative_summary_vector(
                qvel["maximum_abs_rad_s"],
                width=7,
                label="maximum absolute arm qvel",
            ),
        },
        "arm_applied_torque": {
            "start_nm": normalized_start_torque,
            "end_nm": _measured_vector(
                applied_torque["end_nm"],
                width=7,
                label="telemetry summary end applied torque",
            ),
            "delta_nm": normalized_delta_torque,
            "peak_abs_delta_from_start_nm": normalized_peak_torque,
        },
        "end_effector_wrench": {
            "force": {
                "start_n": _measured_vector(
                    force["start_n"], width=3, label="telemetry summary start force"
                ),
                "end_n": _measured_vector(
                    force["end_n"], width=3, label="telemetry summary end force"
                ),
                "delta_n": _measured_vector(
                    force["delta_n"], width=3, label="telemetry summary force delta"
                ),
                "peak_delta_norm_n": _nonnegative_summary_scalar(
                    force["peak_delta_norm_n"], label="peak force delta norm"
                ),
            },
            "torque": {
                "start_nm": _measured_vector(
                    wrench_torque["start_nm"],
                    width=3,
                    label="telemetry summary start wrench torque",
                ),
                "end_nm": _measured_vector(
                    wrench_torque["end_nm"],
                    width=3,
                    label="telemetry summary end wrench torque",
                ),
                "delta_nm": _measured_vector(
                    wrench_torque["delta_nm"],
                    width=3,
                    label="telemetry summary wrench torque delta",
                ),
                "peak_delta_norm_nm": _nonnegative_summary_scalar(
                    wrench_torque["peak_delta_norm_nm"],
                    label="peak wrench torque delta norm",
                ),
            },
        },
    }


def summarize_telemetry_samples(
    before: object, samples: object, after: object
) -> dict[str, object]:
    """Summarize raw action telemetry without inferring physical verdicts."""

    sample_items = _iterable_items(samples, label="post-action telemetry samples")
    if not sample_items:
        raise ValueError("post-action telemetry samples must not be empty")
    start = _summary_telemetry_vectors(before, label="pre-action", require_torque=False)
    measured = [
        _summary_telemetry_vectors(
            sample, label=f"post-step sample {index}", require_torque=True
        )
        for index, sample in enumerate(sample_items)
    ]
    end = _summary_telemetry_vectors(
        after, label="post-action completion", require_torque=True
    )
    post = [*measured, end]
    post_qpos = [value.qpos for value in post]
    all_qvel = [start.qvel, *(value.qvel for value in post)]
    post_force = [value.force for value in post]
    post_wrench_torque = [value.wrench_torque for value in post]

    start_torque = start.torque
    end_torque = end.torque
    assert end_torque is not None
    post_torque = [value.torque for value in post]
    assert all(value is not None for value in post_torque)
    if start_torque is None:
        torque_summary: dict[str, object] = {
            "start_nm": None,
            "end_nm": end_torque,
            "delta_nm": None,
            "peak_abs_delta_from_start_nm": None,
        }
    else:
        torque_summary = {
            "start_nm": start_torque,
            "end_nm": end_torque,
            "delta_nm": _vector_delta(end_torque, start_torque),
            "peak_abs_delta_from_start_nm": _peak_abs_delta(
                post_torque,
                start_torque,  # type: ignore[arg-type]
            ),
        }
    return {
        "arm_joint_position": {
            "start_rad": start.qpos,
            "end_rad": end.qpos,
            "delta_rad": _vector_delta(end.qpos, start.qpos),
            "peak_abs_delta_from_start_rad": _peak_abs_delta(post_qpos, start.qpos),
        },
        "arm_joint_velocity": {
            "start_rad_s": start.qvel,
            "end_rad_s": end.qvel,
            "maximum_abs_rad_s": [
                max(abs(value[index]) for value in all_qvel) for index in range(7)
            ],
        },
        "arm_applied_torque": torque_summary,
        "end_effector_wrench": {
            "force": {
                "start_n": start.force,
                "end_n": end.force,
                "delta_n": _vector_delta(end.force, start.force),
                "peak_delta_norm_n": _peak_delta_norm(post_force, start.force),
            },
            "torque": {
                "start_nm": start.wrench_torque,
                "end_nm": end.wrench_torque,
                "delta_nm": _vector_delta(end.wrench_torque, start.wrench_torque),
                "peak_delta_norm_nm": _peak_delta_norm(
                    post_wrench_torque, start.wrench_torque
                ),
            },
        },
    }
