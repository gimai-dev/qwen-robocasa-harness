"""Public-camera image servo resolved through the Panda Cartesian skill."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence

from .cartesian_skill import CartesianDeltaCommand, resolve_cartesian_delta
from .panda_embodiment import project_world_point

MAX_DEPTH_DELTA_M = 0.03
MIN_STEP_M = 0.005
MAX_STEP_M = 0.03
MAX_INTERNAL_SERVO_STEPS = 4
MAX_SERVO_TRAJECTORY_M = 0.08
MAX_STEREO_SERVO_TRAJECTORY_M = 0.15
MAX_STEREO_RAY_GAP_M = 0.06
IMAGE_SERVO_ORIENTATION_WEIGHT = 0.25
PUBLIC_PIXEL_ROUNDOFF_TOLERANCE_PX = 0.001


@dataclasses.dataclass(frozen=True)
class ImageServoCommand:
    observation_id: str
    camera: str
    target_pixel: tuple[float, float]
    target_role: str
    depth_delta_m: float
    step_m: float
    gripper: str
    note: str
    other_view_pixel: tuple[float, float] | None = None
    other_view_camera: str | None = None
    kind: str = "image_servo"


@dataclasses.dataclass(frozen=True)
class ImageServoResolution:
    current_pixel: tuple[float, float]
    current_depth_m: float
    target_depth_m: float
    translation_m: tuple[float, float, float]
    joint_endpoint: tuple[float, ...]
    gripper_open: float
    predicted_delta: tuple[float, ...]
    residual_norm: float
    stereo_ray_gap_m: float | None = None
    stereo_target_base_m: tuple[float, float, float] | None = None


def image_servo_trajectory_limit_m(command: ImageServoCommand) -> float:
    """Return the Qwen-authored trajectory cap for one accepted servo action."""

    increments = (
        1 if command.target_role == "articulation_motion" else MAX_INTERNAL_SERVO_STEPS
    )
    cap = (
        MAX_STEREO_SERVO_TRAJECTORY_M
        if command.other_view_pixel is not None
        and command.target_role != "articulation_motion"
        else MAX_SERVO_TRAJECTORY_M
    )
    return min(command.step_m * increments, cap)


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _vector(value: object, *, width: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must contain exactly {width} values")
    if len(value) != width:
        raise ValueError(f"{label} must contain exactly {width} values")
    return tuple(
        _finite(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _rotation_xyzw(value: object) -> tuple[tuple[float, float, float], ...]:
    x, y, z, w = _vector(value, width=4, label="base rotation")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isclose(norm, 1.0, abs_tol=1e-5, rel_tol=0.0):
        raise ValueError("base rotation must be a unit quaternion")
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def _matvec(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(
        sum(matrix[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _transpose_matvec(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(
        sum(matrix[row][column] * vector[row] for row in range(3))
        for column in range(3)
    )  # type: ignore[return-value]


def decode_image_servo(
    value: Mapping[str, object], *, observation_id: str
) -> ImageServoCommand:
    required = {
        "kind",
        "observation_id",
        "camera",
        "target_pixel",
        "target_role",
        "depth_delta_m",
        "step_m",
        "gripper",
        "note",
    }
    if (
        not required <= set(value) <= required | {"other_view_pixel", "other_view_camera"}
        or value.get("kind") != "image_servo"
    ):
        raise ValueError("image-servo command fields drifted")
    if value.get("observation_id") != observation_id:
        raise ValueError("image-servo command has a stale observation_id")
    camera = value.get("camera")
    if camera not in {"left", "right"}:
        raise ValueError("image-servo camera must be left or right")
    target_role = value.get("target_role")
    if target_role not in {
        "source_object",
        "destination_receptacle",
        "fixture_handle",
        "control_target",
        "articulation_motion",
    }:
        raise ValueError("image-servo target role is invalid")
    target_pixel = _vector(value.get("target_pixel"), width=2, label="target pixel")
    depth_delta = _finite(value.get("depth_delta_m"), label="depth delta")
    if abs(depth_delta) > MAX_DEPTH_DELTA_M:
        raise ValueError("image-servo depth delta exceeds its bound")
    step = _finite(value.get("step_m"), label="image-servo step")
    if not MIN_STEP_M <= step <= MAX_STEP_M:
        raise ValueError("image-servo step is outside its bound")
    gripper = value.get("gripper")
    if gripper not in {"open", "hold", "close"}:
        raise ValueError("image-servo gripper intent is invalid")
    note = value.get("note")
    if not isinstance(note, str) or not 1 <= len(note) <= 320:
        raise ValueError("image-servo note is outside its bound")
    raw_other = value.get("other_view_pixel")
    other_view_pixel = (
        None
        if raw_other is None
        else _vector(raw_other, width=2, label="other view pixel")
    )
    other_view_camera = value.get("other_view_camera")
    if other_view_camera is not None and (
        other_view_camera != "wrist" or other_view_pixel is None
        or target_role != "source_object"
    ):
        raise ValueError("wrist secondary requires a source-object stereo request")
    return ImageServoCommand(
        observation_id=observation_id,
        camera=str(camera),
        other_view_camera=other_view_camera,
        target_pixel=(target_pixel[0], target_pixel[1]),
        target_role=str(target_role),
        depth_delta_m=depth_delta,
        step_m=step,
        gripper=str(gripper),
        note=note,
        other_view_pixel=(
            None
            if other_view_pixel is None
            else (other_view_pixel[0], other_view_pixel[1])
        ),
    )


def _camera_geometry(
    calibration: Mapping[str, object],
) -> tuple[float, float, float, float, tuple[float, ...], tuple[tuple[float, ...], ...]]:
    fx = _finite(calibration.get("fx_px"), label="camera fx")
    fy = _finite(calibration.get("fy_px"), label="camera fy")
    cx = _finite(calibration.get("cx_px"), label="camera cx")
    cy = _finite(calibration.get("cy_px"), label="camera cy")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("image-servo focal length must be positive")
    camera_position = _vector(
        calibration.get("camera_position_world_m"),
        width=3,
        label="camera position",
    )
    camera_rotation_raw = calibration.get("camera_xmat_world")
    if (
        isinstance(camera_rotation_raw, (str, bytes, Mapping))
        or not isinstance(camera_rotation_raw, Sequence)
        or len(camera_rotation_raw) != 3
    ):
        raise ValueError("image-servo camera rotation is invalid")
    camera_rotation = tuple(
        _vector(row, width=3, label="camera rotation row")
        for row in camera_rotation_raw
    )
    return fx, fy, cx, cy, camera_position, camera_rotation


def _pixel_ray_world(
    pixel: tuple[float, float],
    geometry: tuple[
        float, float, float, float, tuple[float, ...], tuple[tuple[float, ...], ...]
    ],
) -> tuple[tuple[float, ...], tuple[float, float, float]]:
    fx, fy, cx, cy, position, rotation = geometry
    direction = _matvec(rotation, ((pixel[0] - cx) / fx, -(pixel[1] - cy) / fy, -1.0))
    norm = math.sqrt(sum(item * item for item in direction))
    unit = (direction[0] / norm, direction[1] / norm, direction[2] / norm)
    return position, unit


def _closest_ray_points(
    origin_a: Sequence[float],
    direction_a: Sequence[float],
    origin_b: Sequence[float],
    direction_b: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...], float, float, float]:
    w0 = tuple(origin_a[i] - origin_b[i] for i in range(3))
    a = sum(direction_a[i] * direction_a[i] for i in range(3))
    b = sum(direction_a[i] * direction_b[i] for i in range(3))
    c = sum(direction_b[i] * direction_b[i] for i in range(3))
    d = sum(direction_a[i] * w0[i] for i in range(3))
    e = sum(direction_b[i] * w0[i] for i in range(3))
    denominator = a * c - b * b
    if denominator <= 1e-12:
        raise ValueError("image-servo stereo rays are parallel")
    s = (b * e - c * d) / denominator
    t = (a * e - b * d) / denominator
    point_a = tuple(origin_a[i] + s * direction_a[i] for i in range(3))
    point_b = tuple(origin_b[i] + t * direction_b[i] for i in range(3))
    gap = math.sqrt(sum((point_a[i] - point_b[i]) ** 2 for i in range(3)))
    return point_a, point_b, gap, s, t


def resolve_image_servo(
    command: ImageServoCommand,
    public_state: Mapping[str, object],
    camera_calibration: Mapping[str, object],
    *,
    current_gripper: object,
) -> ImageServoResolution:
    calibration = camera_calibration.get(command.camera)
    if not isinstance(calibration, Mapping):
        raise ValueError("image-servo external camera calibration is missing")
    width = calibration.get("image_width_px")
    height = calibration.get("image_height_px")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("image-servo image width is invalid")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("image-servo image height is invalid")
    target_u, target_v = command.target_pixel
    if not 0.0 <= target_u < width or not 0.0 <= target_v < height:
        raise ValueError("image-servo target pixel is outside the image")

    base_position = _vector(
        public_state.get("state.base_position"), width=3, label="base position"
    )
    base_rotation = _rotation_xyzw(public_state.get("state.base_rotation"))
    relative_eef = _vector(
        public_state.get("state.end_effector_position_relative"),
        width=3,
        label="relative end-effector position",
    )
    relative_world = _matvec(base_rotation, relative_eef)
    current_world = tuple(
        base_position[index] + relative_world[index] for index in range(3)
    )
    projected = project_world_point(current_world, calibration)
    if projected["visible"] is not True:
        raise ValueError("image-servo end effector is not visible in the selected camera")
    current_u = _finite(projected["u_px"], label="current end-effector u pixel")
    current_v = _finite(projected["v_px"], label="current end-effector v pixel")
    current_depth = _finite(projected["depth_m"], label="current end-effector depth")
    public_pixels = public_state.get("state.end_effector_external_pixels")
    if not isinstance(public_pixels, Mapping):
        raise ValueError("image-servo public external pixels are missing")
    public_pixel = public_pixels.get(command.camera)
    if (
        not isinstance(public_pixel, Mapping)
        or public_pixel.get("visible") is not True
        or public_pixel.get("depth_valid") is not True
    ):
        raise ValueError("image-servo public end-effector pixel is unavailable")
    measured_u = _finite(public_pixel.get("u_px"), label="public end-effector u pixel")
    measured_v = _finite(public_pixel.get("v_px"), label="public end-effector v pixel")
    if not math.isclose(
        current_u,
        measured_u,
        abs_tol=PUBLIC_PIXEL_ROUNDOFF_TOLERANCE_PX,
        rel_tol=0.0,
    ) or not math.isclose(
        current_v,
        measured_v,
        abs_tol=PUBLIC_PIXEL_ROUNDOFF_TOLERANCE_PX,
        rel_tol=0.0,
    ):
        raise ValueError("image-servo calibration disagrees with public end-effector pixel")

    geometry = _camera_geometry(calibration)
    fx, fy, cx, cy, camera_position, camera_rotation = geometry
    stereo_ray_gap: float | None = None
    stereo_target_base: tuple[float, ...] | None = None
    if command.other_view_pixel is None:
        target_depth = current_depth + command.depth_delta_m
        if target_depth <= 0.0:
            raise ValueError(
                "image-servo target depth must remain in front of the camera"
            )
        camera_target = (
            (target_u - cx) * target_depth / fx,
            -(target_v - cy) * target_depth / fy,
            -target_depth,
        )
        target_offset_world = _matvec(camera_rotation, camera_target)
        target_world = tuple(
            camera_position[index] + target_offset_world[index] for index in range(3)
        )
    else:
        other_name = command.other_view_camera or ("left" if command.camera == "right" else "right")
        other = camera_calibration.get(other_name)
        if not isinstance(other, Mapping):
            raise ValueError("image-servo other-view camera calibration is missing")
        other_width = other.get("image_width_px")
        other_height = other.get("image_height_px")
        if (
            isinstance(other_width, bool)
            or not isinstance(other_width, int)
            or other_width <= 0
            or isinstance(other_height, bool)
            or not isinstance(other_height, int)
            or other_height <= 0
        ):
            raise ValueError("image-servo other-view image size is invalid")
        other_u, other_v = command.other_view_pixel
        if not 0.0 <= other_u < other_width or not 0.0 <= other_v < other_height:
            raise ValueError("image-servo other-view pixel is outside the image")
        origin_a, direction_a = _pixel_ray_world(command.target_pixel, geometry)
        origin_b, direction_b = _pixel_ray_world(
            command.other_view_pixel, _camera_geometry(other)
        )
        point_a, point_b, gap, along_a, along_b = _closest_ray_points(
            origin_a, direction_a, origin_b, direction_b
        )
        if along_a <= 0.0 or along_b <= 0.0:
            raise ValueError("image-servo stereo pixels meet behind a camera")
        if gap > MAX_STEREO_RAY_GAP_M:
            raise ValueError(
                "image-servo stereo pixels do not name one point: ray gap "
                f"{gap:.3f} m exceeds {MAX_STEREO_RAY_GAP_M:.3f} m"
            )
        stereo_ray_gap = gap
        stereo_target_base = _transpose_matvec(
            base_rotation,
            tuple(midpoint_component - base_position[i] for i, midpoint_component in enumerate(
                tuple((point_a[k] + point_b[k]) / 2.0 for k in range(3))
            )),
        )
        forward = _matvec(camera_rotation, (0.0, 0.0, -1.0))
        midpoint = tuple((point_a[i] + point_b[i]) / 2.0 for i in range(3))
        midpoint_depth = sum(
            (midpoint[i] - camera_position[i]) * forward[i] for i in range(3)
        )
        target_depth = midpoint_depth + command.depth_delta_m
        cosine = sum(direction_a[i] * forward[i] for i in range(3))
        if midpoint_depth <= 0.0 or target_depth <= 0.0 or cosine <= 0.0:
            raise ValueError(
                "image-servo stereo target must remain in front of the camera"
            )
        target_world = tuple(
            camera_position[i] + direction_a[i] * (target_depth / cosine)
            for i in range(3)
        )
    desired_world = tuple(
        target_world[index] - current_world[index] for index in range(3)
    )
    desired_base = _transpose_matvec(base_rotation, desired_world)
    desired_norm = math.sqrt(sum(item * item for item in desired_base))
    if desired_norm <= 1e-9:
        raise ValueError("image-servo target does not define a motion direction")
    scale = min(
        1.0,
        image_servo_trajectory_limit_m(command) / desired_norm,
    )
    # The direction is Qwen's; the harness only bounds the magnitude. When the
    # fixed damped least-squares map needs more joint motion than one command
    # may carry, halve the same-direction step instead of rejecting the ray.
    last_error: ValueError | None = None
    for _attempt in range(5):
        translation = tuple(item * scale for item in desired_base)
        cartesian = CartesianDeltaCommand(
            observation_id=command.observation_id,
            translation_m=translation,
            rotation_axis_angle_rad=(0.0, 0.0, 0.0),
            gripper=command.gripper,
            note=command.note,
        )
        try:
            resolved = resolve_cartesian_delta(
                cartesian,
                public_state,
                current_gripper=current_gripper,
                orientation_weight=IMAGE_SERVO_ORIENTATION_WEIGHT,
                max_predicted_translation_m=(
                    image_servo_trajectory_limit_m(command)
                    if command.target_role == "articulation_motion"
                    else None
                ),
            )
            break
        except ValueError as error:
            text = str(error)
            if (
                "derived Cartesian joint delta exceeds its bound" not in text
                and "endpoint is unsafe" not in text
            ):
                raise
            last_error = error
            scale /= 2.0
    else:
        assert last_error is not None
        raise last_error
    return ImageServoResolution(
        current_pixel=(current_u, current_v),
        current_depth_m=current_depth,
        target_depth_m=target_depth,
        translation_m=translation,
        joint_endpoint=resolved.joint_endpoint,
        gripper_open=resolved.gripper_open,
        predicted_delta=resolved.predicted_delta,
        residual_norm=resolved.residual_norm,
        stereo_ray_gap_m=stereo_ray_gap,
        stereo_target_base_m=(
            (stereo_target_base[0], stereo_target_base[1], stereo_target_base[2])
            if stereo_target_base is not None
            else None
        ),
    )
