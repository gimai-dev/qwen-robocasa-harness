"""Public-RGB capacity evidence and sparse-keyframe geometry for Cycle 9."""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .demonstration_skill import quaternion_distance_rad


@dataclass(frozen=True)
class SparseKeyframe:
    start: int
    stop: int
    translation_travel_m: float
    rotation_travel_rad: float
    gripper_transition: bool
    residual_cap_m: float


@dataclass(frozen=True)
class PublicPatchMatch:
    source_pixel: tuple[float, float]
    current_pixel: tuple[float, float]
    cosine_similarity: float
    cosine_margin: float
    ncc: float
    ncc_margin: float


def sparse_keyframes(states: object, actions: object) -> tuple[SparseKeyframe, ...]:
    """Apply the frozen 40 mm / 0.20 rad / gripper-transition rule."""
    state = np.asarray(states, dtype=np.float64)
    action = np.asarray(actions, dtype=np.float64)
    if (
        state.ndim != 2
        or state.shape[1] != 16
        or action.ndim != 2
        or action.shape[1] < 12
        or len(state) != len(action)
        or len(state) < 2
        or not np.isfinite(state).all()
        or not np.isfinite(action).all()
    ):
        raise ValueError("Cycle 9 source trajectory is invalid")
    result: list[SparseKeyframe] = []
    start = 0
    translation_travel = 0.0
    rotation_travel = 0.0
    for index in range(1, len(state)):
        translation_travel += float(
            np.linalg.norm(state[index, 7:10] - state[index - 1, 7:10])
        )
        rotation_travel += quaternion_distance_rad(
            state[index - 1, 10:14], state[index, 10:14]
        )
        transitioned = bool((action[index, 11] >= 0.5) != (action[index - 1, 11] >= 0.5))
        terminal = index == len(state) - 1
        if (
            translation_travel >= 0.040 - 1e-12
            or rotation_travel >= 0.20 - 1e-12
            or transitioned
            or terminal
        ):
            result.append(
                SparseKeyframe(
                    start=start,
                    stop=index,
                    translation_travel_m=translation_travel,
                    rotation_travel_rad=rotation_travel,
                    gripper_transition=transitioned,
                    residual_cap_m=min(0.025, 0.25 * translation_travel),
                )
            )
            start = index
            translation_travel = 0.0
            rotation_travel = 0.0
    return tuple(result)


def approach_residual_capacity_m(keyframes: Sequence[SparseKeyframe]) -> float:
    """Sum capacity through the first source gripper transition, or the route."""
    total = 0.0
    for row in keyframes:
        total += row.residual_cap_m
        if row.gripper_transition:
            break
    return total


def point_annotation_schema() -> dict[str, object]:
    point = {
        "type": "array",
        "prefixItems": [
            {"type": "number", "minimum": 0.0, "maximum": 255.0},
            {"type": "number", "minimum": 0.0, "maximum": 255.0},
        ],
        "minItems": 2,
        "maxItems": 2,
    }
    fields = {
        "landmark": {
            "type": "string",
            "minLength": 1,
            "maxLength": 80,
            "pattern": "^[A-Za-z0-9 .,;:!?()'/_+\\-]{1,80}$",
        },
        "left_px": point,
        "right_px": point,
    }
    return {
        "type": "object",
        "properties": fields,
        "required": sorted(fields),
        "additionalProperties": False,
    }


def landmark_box_schema() -> dict[str, object]:
    """Closed single-view semantic box schema for evidence-only Qwen calls."""
    coordinate = {"type": "number", "minimum": 0.0, "maximum": 255.0}
    fields = {
        "landmark": {
            "type": "string",
            "minLength": 1,
            "maxLength": 80,
            "pattern": "^[A-Za-z0-9 .,;:!?()'/_+\\-]{1,80}$",
        },
        "box_xyxy": {
            "type": "array",
            "prefixItems": [coordinate, coordinate, coordinate, coordinate],
            "minItems": 4,
            "maxItems": 4,
        },
    }
    return {
        "type": "object",
        "properties": fields,
        "required": sorted(fields),
        "additionalProperties": False,
    }


def consensus_box(first: Sequence[float], second: Sequence[float]) -> np.ndarray:
    """Validate two boxes, require IoU >=0.5, and return their intersection."""
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != (4,) or right.shape != (4,):
        raise ValueError("landmark box must contain xyxy coordinates")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("landmark box is nonfinite")
    if (
        np.any(left < 0.0)
        or np.any(right < 0.0)
        or np.any(left > 255.0)
        or np.any(right > 255.0)
    ):
        raise ValueError("landmark box is outside the official image")
    if (
        left[2] <= left[0]
        or left[3] <= left[1]
        or right[2] <= right[0]
        or right[3] <= right[1]
    ):
        raise ValueError("landmark box has nonpositive area")
    low = np.maximum(left[:2], right[:2])
    high = np.minimum(left[2:], right[2:])
    intersection = float(np.prod(np.maximum(high - low, 0.0)))
    union = float(
        np.prod(left[2:] - left[:2])
        + np.prod(right[2:] - right[:2])
        - intersection
    )
    if union <= 0.0 or intersection / union < 0.5 - 1e-12:
        raise ValueError("independent landmark boxes have IoU below 0.5")
    if np.any(high - low < 8.0):
        raise ValueError("landmark consensus box is too small")
    return np.concatenate((low, high))


def _patch_candidates(box: Sequence[float], grid: int) -> np.ndarray:
    value = np.asarray(box, dtype=np.float64)
    centers = (np.arange(grid, dtype=np.float64) + 0.5) * (256.0 / grid)
    yy, xx = np.meshgrid(centers, centers, indexing="ij")
    selected = (
        (xx >= value[0])
        & (yy >= value[1])
        & (xx <= value[2])
        & (yy <= value[3])
    )
    result = np.column_stack(np.nonzero(selected))
    if len(result) < 2:
        raise ValueError("landmark box contains fewer than two DINO patches")
    return result


def mutual_patch_match(
    source_features: object,
    current_features: object,
    *,
    source_box: Sequence[float],
    current_box: Sequence[float],
    minimum_similarity: float,
    minimum_margin: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Select one unique mutual-nearest DINO patch pair inside public boxes."""
    source = np.asarray(source_features, dtype=np.float64)
    current = np.asarray(current_features, dtype=np.float64)
    if (
        source.ndim != 3
        or current.shape != source.shape
        or source.shape[0] != source.shape[1]
    ):
        raise ValueError("DINO patch features are invalid")
    if not np.isfinite(source).all() or not np.isfinite(current).all():
        raise ValueError("DINO patch features are nonfinite")
    grid = source.shape[0]
    source_indices = _patch_candidates(source_box, grid)
    current_indices = _patch_candidates(current_box, grid)
    source_vectors = source[source_indices[:, 0], source_indices[:, 1]]
    current_vectors = current[current_indices[:, 0], current_indices[:, 1]]
    source_vectors /= np.linalg.norm(source_vectors, axis=1, keepdims=True)
    current_vectors /= np.linalg.norm(current_vectors, axis=1, keepdims=True)
    similarity = source_vectors @ current_vectors.T
    source_best = np.argmax(similarity, axis=1)
    current_best = np.argmax(similarity, axis=0)
    pairs: list[tuple[float, int, int]] = []
    for source_row, current_row in enumerate(source_best):
        if current_best[current_row] == source_row:
            pairs.append(
                (
                    float(similarity[source_row, current_row]),
                    source_row,
                    int(current_row),
                )
            )
    pairs.sort(reverse=True)
    if not pairs or pairs[0][0] < minimum_similarity - 1e-12:
        raise ValueError("no sufficiently similar mutual DINO match")
    next_similarity = pairs[1][0] if len(pairs) > 1 else -1.0
    margin = pairs[0][0] - next_similarity
    if margin < minimum_margin - 1e-12:
        raise ValueError("DINO match is not unique")
    _, source_row, current_row = pairs[0]
    scale = 256.0 / grid
    source_pixel = (source_indices[source_row][::-1].astype(np.float64) + 0.5) * scale
    current_pixel = (current_indices[current_row][::-1].astype(np.float64) + 0.5) * scale
    return source_pixel, current_pixel, pairs[0][0], margin


def _ncc(first: np.ndarray, second: np.ndarray) -> float:
    left = first.astype(np.float64).reshape(-1)
    right = second.astype(np.float64).reshape(-1)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return -1.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)


def refine_match_ncc(
    source_image: object,
    current_image: object,
    *,
    source_pixel: Sequence[float],
    current_pixel: Sequence[float],
    patch_radius_px: int,
    search_radius_px: int,
    minimum_ncc: float,
    minimum_margin: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Refine the current point at integer-pixel resolution with local RGB NCC."""
    source = np.asarray(source_image, dtype=np.uint8)
    current = np.asarray(current_image, dtype=np.uint8)
    if source.shape != (256, 256, 3) or current.shape != source.shape:
        raise ValueError("NCC requires official RGB")
    sx, sy = np.rint(source_pixel).astype(int)
    cx, cy = np.rint(current_pixel).astype(int)
    radius = int(patch_radius_px)
    search = int(search_radius_px)
    if radius < 2 or search < 0:
        raise ValueError("NCC geometry is invalid")
    if (
        sx - radius < 0
        or sy - radius < 0
        or sx + radius >= 256
        or sy + radius >= 256
    ):
        raise ValueError("source NCC patch crosses image boundary")
    template = source[sy - radius : sy + radius + 1, sx - radius : sx + radius + 1]
    candidates: list[tuple[float, int, int]] = []
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            x, y = cx + dx, cy + dy
            if (
                x - radius < 0
                or y - radius < 0
                or x + radius >= 256
                or y + radius >= 256
            ):
                continue
            patch = current[y - radius : y + radius + 1, x - radius : x + radius + 1]
            candidates.append((_ncc(template, patch), x, y))
    candidates.sort(reverse=True)
    if not candidates or candidates[0][0] < minimum_ncc - 1e-12:
        raise ValueError("NCC match is below the frozen threshold")
    margin = candidates[0][0] - (candidates[1][0] if len(candidates) > 1 else -1.0)
    if margin < minimum_margin - 1e-12:
        raise ValueError("NCC match is not unique")
    score, x, y = candidates[0]
    return np.asarray([sx, sy], dtype=np.float64), np.asarray([x, y], dtype=np.float64), score, margin


def select_calibrated_safety_factor(
    raw_estimates_m: Sequence[float],
    true_displacements_m: Sequence[float],
    factor_grid: Sequence[float],
) -> float:
    """Choose the smallest frozen-grid factor that bounds every calibration pair."""
    raw = np.asarray(raw_estimates_m, dtype=np.float64)
    truth = np.asarray(true_displacements_m, dtype=np.float64)
    factors = np.asarray(factor_grid, dtype=np.float64)
    if (
        raw.ndim != 1
        or truth.shape != raw.shape
        or len(raw) < 1
        or factors.ndim != 1
    ):
        raise ValueError("calibration arrays are invalid")
    if np.any(raw <= 0.0) or np.any(truth <= 0.0) or np.any(factors <= 0.0):
        raise ValueError("calibration values must be positive")
    for factor in sorted({float(value) for value in factors}):
        estimates = raw * factor
        if np.all(estimates + 1e-12 >= truth) and np.all(estimates <= 2.0 * truth + 1e-12):
            return factor
    raise ValueError("no frozen safety factor passes the public calibration cohort")


def _xyzw_rotation(quaternion: Sequence[float]) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("public base quaternion is invalid")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("public base quaternion is degenerate")
    x, y, z, w = value / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def camera_extrinsics_in_base(
    calibration: Mapping[str, object], public_state: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Express one official camera pose in the public robot-base frame."""
    state = np.asarray(public_state, dtype=np.float64)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError("public robot state is invalid")
    base_rotation = _xyzw_rotation(state[3:7])
    base_position = state[:3]
    camera_position = np.asarray(
        calibration["camera_position_world_m"], dtype=np.float64
    )
    camera_rotation = np.asarray(calibration["camera_xmat_world"], dtype=np.float64)
    if camera_position.shape != (3,) or camera_rotation.shape != (3, 3):
        raise ValueError("public camera calibration is invalid")
    return (
        base_rotation.T @ (camera_position - base_position),
        base_rotation.T @ camera_rotation,
    )


def public_eef_camera_depth_m(
    public_state: Sequence[float],
    *,
    camera_position_in_base_m: Sequence[float],
    camera_rotation_in_base: object,
) -> float:
    """Compute official-camera forward depth from public base-relative EEF state."""
    state = np.asarray(public_state, dtype=np.float64)
    camera_position = np.asarray(camera_position_in_base_m, dtype=np.float64)
    camera_rotation = np.asarray(camera_rotation_in_base, dtype=np.float64)
    if state.shape != (16,) or camera_position.shape != (3,) or camera_rotation.shape != (3, 3):
        raise ValueError("public depth inputs are invalid")
    camera_point = camera_rotation.T @ (state[7:10] - camera_position)
    depth = -float(camera_point[2])
    if not np.isfinite(depth) or depth <= 0.0:
        raise ValueError("public EEF has nonpositive official-camera depth")
    return depth


def raw_pixel_metric_estimate_m(
    *,
    pixel_displacement: float,
    endpoint_depths_m: Sequence[float],
    focal_length_px: float,
    depth_margin_m: float,
) -> float:
    """Return the frozen empirically calibrated per-view displacement estimate."""
    depths = np.asarray(endpoint_depths_m, dtype=np.float64)
    if (
        pixel_displacement <= 0.0
        or depths.shape != (2,)
        or np.any(depths <= 0.0)
        or focal_length_px <= 0.0
        or depth_margin_m < 0.0
    ):
        raise ValueError("pixel metric estimate inputs are invalid")
    return float(pixel_displacement * (float(depths.max()) + depth_margin_m) / focal_length_px)


def comparison_panel_png(
    current: object, reference: object, *, current_first: bool
) -> bytes:
    """Build a labeled public current/reference panel for point annotation."""
    current_image = Image.fromarray(np.asarray(current, dtype=np.uint8), mode="RGB")
    reference_image = Image.fromarray(np.asarray(reference, dtype=np.uint8), mode="RGB")
    if current_image.size != (256, 256) or reference_image.size != (256, 256):
        raise ValueError("Cycle 9 point evidence requires 256x256 official RGB")
    order = (
        (("CURRENT", current_image), ("REFERENCE", reference_image))
        if current_first
        else (("REFERENCE", reference_image), ("CURRENT", current_image))
    )
    canvas = Image.new("RGB", (512, 286), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(order):
        canvas.paste(image, (index * 256, 30))
        draw.text((index * 256 + 4, 7), label, fill="red")
    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _camera_ray(
    calibration: Mapping[str, object], pixel: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    if len(pixel) != 2:
        raise ValueError("pixel must contain x and y")
    u, v = (float(pixel[0]), float(pixel[1]))
    rotation = np.asarray(calibration["camera_xmat_world"], dtype=np.float64)
    origin = np.asarray(calibration["camera_position_world_m"], dtype=np.float64)
    if rotation.shape != (3, 3) or origin.shape != (3,):
        raise ValueError("public camera calibration is invalid")
    camera = np.asarray(
        [
            (u - float(calibration["cx_px"])) / float(calibration["fx_px"]),
            -(v - float(calibration["cy_px"])) / float(calibration["fy_px"]),
            -1.0,
        ],
        dtype=np.float64,
    )
    direction = rotation @ camera
    direction /= np.linalg.norm(direction)
    return origin, direction


def triangulate_public_point(
    left_calibration: Mapping[str, object],
    right_calibration: Mapping[str, object],
    left_pixel: Sequence[float],
    right_pixel: Sequence[float],
) -> tuple[np.ndarray, float]:
    """Return the midpoint of two public camera rays and their separation."""
    left_origin, left_ray = _camera_ray(left_calibration, left_pixel)
    right_origin, right_ray = _camera_ray(right_calibration, right_pixel)
    matrix = np.column_stack((left_ray, -right_ray))
    if np.linalg.cond(matrix) > 100.0:
        raise ValueError("public camera rays are ill-conditioned")
    parameters, *_ = np.linalg.lstsq(matrix, right_origin - left_origin, rcond=None)
    if np.min(parameters) <= 0.0:
        raise ValueError("public point has nonpositive camera depth")
    left_point = left_origin + parameters[0] * left_ray
    right_point = right_origin + parameters[1] * right_ray
    return (left_point + right_point) / 2.0, float(
        np.linalg.norm(left_point - right_point)
    )


def reproject_public_point(
    calibration: Mapping[str, object], point_world_m: Sequence[float]
) -> np.ndarray:
    rotation = np.asarray(calibration["camera_xmat_world"], dtype=np.float64)
    origin = np.asarray(calibration["camera_position_world_m"], dtype=np.float64)
    camera = rotation.T @ (np.asarray(point_world_m, dtype=np.float64) - origin)
    if camera[2] >= 0.0:
        raise ValueError("public point is behind the camera")
    return np.asarray(
        [
            float(calibration["cx_px"])
            + float(calibration["fx_px"]) * camera[0] / -camera[2],
            float(calibration["cy_px"])
            - float(calibration["fy_px"]) * camera[1] / -camera[2],
        ]
    )


def validate_public_landmark(
    *,
    calibration: Mapping[str, Mapping[str, object]],
    left_pixel: Sequence[float],
    right_pixel: Sequence[float],
) -> tuple[np.ndarray, float, float]:
    point, ray_residual = triangulate_public_point(
        calibration["left"], calibration["right"], left_pixel, right_pixel
    )
    reprojection = max(
        float(
            np.linalg.norm(
                reproject_public_point(calibration["left"], point) - left_pixel
            )
        ),
        float(
            np.linalg.norm(
                reproject_public_point(calibration["right"], point) - right_pixel
            )
        ),
    )
    if ray_residual > 0.005 + 1e-12:
        raise ValueError("public landmark cross-ray residual exceeds 5 mm")
    if reprojection > 4.0 + 1e-12:
        raise ValueError("public landmark reprojection residual exceeds 4 px")
    lower = np.asarray([0.0, -2.0, 0.0])
    upper = np.asarray([2.5, 1.0, 2.5])
    if np.any(point < lower) or np.any(point > upper):
        raise ValueError("public landmark is outside the certified workspace")
    return point, ray_residual, reprojection
