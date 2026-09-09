"""Public-RGB-only feature tracking and local image-space control."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import cv2
import numpy as np

IMAGE_SIZE = 256
ALIGNMENT_TOLERANCE_PX = 8.0
ALIGNMENT_TOLERANCE_NORMALIZED = ALIGNMENT_TOLERANCE_PX / IMAGE_SIZE
MIN_FLOW_NOISE_PX = 0.5
MIN_EEF_DISPLACEMENT_M = 0.003
PROBE_DISPLACEMENT_MM = 20.0
IMAGE_BORDER_PX = 12.0
MIN_FEATURE_SEPARATION_PX = 24.0
ANCHOR_RADIUS_PX = 32
ANCHOR_MAX_CORNERS = 64
ANCHOR_QUALITY_LEVEL = 0.01
ANCHOR_MIN_DISTANCE_PX = 5
ANCHOR_TEXTURE_FLOOR = 1e-5
ANCHOR_MIN_CLUSTER = 3
ANCHOR_MAGNITUDE_FRACTION = 0.25
ANCHOR_DIRECTION_DEG = 15.0


@dataclass(frozen=True)
class TrackResult:
    points: np.ndarray
    max_forward_backward_error_px: float
    min_texture_eigenvalue: float


@dataclass(frozen=True)
class JacobianFit:
    jacobian: np.ndarray
    rank: int
    condition_number: float
    residual_px: float


@dataclass(frozen=True)
class _LegacyViewAnchors:
    target_initial: np.ndarray
    gripper_initial: np.ndarray
    target_returned: np.ndarray
    gripper_returned: np.ndarray
    gripper_plus: np.ndarray
    report: dict[str, object]


def normalized_to_pixel(uv: tuple[float, float]) -> np.ndarray:
    value = np.asarray(uv, dtype=np.float64)
    if (
        value.shape != (2,)
        or not np.isfinite(value).all()
        or np.any(value < 0)
        or np.any(value > 1)
    ):
        raise ValueError("normalized coordinate is invalid")
    return value * (IMAGE_SIZE - 1)


def parallax_direction_from_probe(
    before_uv_px: object, after_uv_px: object, *, probe_direction: int
) -> int:
    """Keep a lateral probe only when it improves horizontal feature visibility."""

    before = np.asarray(before_uv_px, dtype=np.float64)
    after = np.asarray(after_uv_px, dtype=np.float64)
    if (
        before.shape != (2,)
        or after.shape != (2,)
        or not np.isfinite(before).all()
        or not np.isfinite(after).all()
        or np.any(before < 0)
        or np.any(after < 0)
        or np.any(before > IMAGE_SIZE - 1)
        or np.any(after > IMAGE_SIZE - 1)
    ):
        raise ValueError("parallax probe pixels are invalid")
    if probe_direction not in {-1, 1}:
        raise ValueError("parallax probe direction must be -1 or 1")
    before_margin = min(float(before[0]), IMAGE_SIZE - 1 - float(before[0]))
    after_margin = min(float(after[0]), IMAGE_SIZE - 1 - float(after[0]))
    return -probe_direction if after_margin < before_margin - 0.05 else probe_direction


def _gray(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or value.dtype != np.uint8:
        raise ValueError("RGB frame violates the official 256x256 contract")
    return cv2.cvtColor(value, cv2.COLOR_RGB2GRAY)


def _gray_float(image: np.ndarray) -> np.ndarray:
    return _gray(image).astype(np.float32) / 255.0


def _region_corners(image: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, bool]:
    gray = _gray_float(image)
    x, y = np.asarray(center, dtype=np.float64)
    clipped = bool(
        x - ANCHOR_RADIUS_PX < IMAGE_BORDER_PX
        or y - ANCHOR_RADIUS_PX < IMAGE_BORDER_PX
        or x + ANCHOR_RADIUS_PX > IMAGE_SIZE - 1 - IMAGE_BORDER_PX
        or y + ANCHOR_RADIUS_PX > IMAGE_SIZE - 1 - IMAGE_BORDER_PX
    )
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    cv2.circle(mask, (round(x), round(y)), ANCHOR_RADIUS_PX, 255, -1)
    mask[: int(IMAGE_BORDER_PX), :] = 0
    mask[-int(IMAGE_BORDER_PX) :, :] = 0
    mask[:, : int(IMAGE_BORDER_PX)] = 0
    mask[:, -int(IMAGE_BORDER_PX) :] = 0
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=ANCHOR_MAX_CORNERS,
        qualityLevel=ANCHOR_QUALITY_LEVEL,
        minDistance=ANCHOR_MIN_DISTANCE_PX,
        mask=mask,
        blockSize=3,
    )
    if corners is None:
        return np.empty((0, 2), np.float32), clipped
    return corners.reshape(-1, 2).astype(np.float32), clipped


def _lk_leg(
    previous: np.ndarray, current: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    options = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    before = _gray(previous)
    after = _gray(current)
    forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        before, after, points.astype(np.float32), None, **options
    )
    if forward is None:
        return (
            points.astype(np.float64),
            np.zeros(len(points), bool),
            np.full(len(points), np.inf),
        )
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        after, before, forward, None, **options
    )
    if backward is None:
        return (
            forward.astype(np.float64),
            np.zeros(len(points), bool),
            np.full(len(points), np.inf),
        )
    valid = np.asarray(forward_status).reshape(-1).astype(bool)
    valid &= np.asarray(backward_status).reshape(-1).astype(bool)
    residual = np.linalg.norm(backward - points, axis=1)
    valid &= residual <= 1.0
    return forward.astype(np.float64), valid, residual.astype(np.float64)


def _continuous_candidates(
    frames: list[np.ndarray], candidates: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    positions = [candidates.astype(np.float64)]
    valid = np.ones(len(candidates), bool)
    max_fb = np.zeros(len(candidates), np.float64)
    noise_legs: list[np.ndarray] = []
    for index, (before, after) in enumerate(pairwise(frames)):
        tracked, leg_valid, fb = _lk_leg(before, after, positions[-1])
        valid &= leg_valid
        max_fb = np.maximum(max_fb, fb)
        if index < len(frames) - 3:
            noise_legs.append(np.linalg.norm(tracked - positions[-1], axis=1))
        positions.append(tracked)
    if noise_legs:
        noise_values = np.stack(noise_legs, axis=1)
        median = np.median(noise_values, axis=1)
        mad = np.median(np.abs(noise_values - median[:, None]), axis=1)
        noise = np.maximum(MIN_FLOW_NOISE_PX, median + 5.0 * mad)
    else:
        noise = np.full(len(candidates), MIN_FLOW_NOISE_PX)
    return positions, valid, max_fb, noise


def _legacy_select_view_anchors(
    same_pose_frames: list[np.ndarray],
    plus_frame: np.ndarray,
    returned_frame: np.ndarray,
    *,
    target_center: np.ndarray,
    gripper_center: np.ndarray,
) -> _LegacyViewAnchors:
    """Select motion-grounded anchors from public images without geometry."""
    if len(same_pose_frames) != 5:
        raise ValueError(
            "anchor noise authority requires exactly five same-pose frames"
        )
    frames = [*same_pose_frames, plus_frame, returned_frame]
    target_candidates, target_clipped = _region_corners(
        same_pose_frames[0], target_center
    )
    gripper_candidates, gripper_clipped = _region_corners(
        same_pose_frames[0], gripper_center
    )
    if len(target_candidates) == 0 or len(gripper_candidates) == 0:
        raise ValueError("anchor region contains no textured corners")
    texture = cv2.cornerMinEigenVal(
        _gray_float(same_pose_frames[0]), blockSize=3, ksize=3
    )

    def texture_values(candidates: np.ndarray) -> np.ndarray:
        xy = np.rint(candidates).astype(int)
        return texture[xy[:, 1], xy[:, 0]].astype(np.float64)

    target_positions, target_valid, target_fb, target_noise = _continuous_candidates(
        frames, target_candidates
    )
    gripper_positions, gripper_valid, gripper_fb, gripper_noise = (
        _continuous_candidates(frames, gripper_candidates)
    )
    target_plus = np.linalg.norm(target_positions[-2] - target_positions[0], axis=1)
    target_return = np.linalg.norm(target_positions[-1] - target_positions[0], axis=1)
    target_pass = target_valid & (
        texture_values(target_candidates) >= ANCHOR_TEXTURE_FLOOR
    )
    target_pass &= target_plus <= np.maximum(1.5, 3.0 * target_noise)
    target_pass &= target_return <= np.maximum(1.5, 3.0 * target_noise)
    target_indices = np.flatnonzero(target_pass)
    if len(target_indices) == 0:
        raise ValueError("target region has no continuous static textured anchor")
    target_index = int(
        target_indices[
            np.argmin(
                np.linalg.norm(
                    target_candidates[target_indices] - target_center, axis=1
                )
            )
        ]
    )

    plus_vectors = gripper_positions[-2] - gripper_positions[0]
    return_vectors = gripper_positions[-1] - gripper_positions[-2]
    plus_magnitude = np.linalg.norm(plus_vectors, axis=1)
    return_magnitude = np.linalg.norm(return_vectors, axis=1)
    net_return = np.linalg.norm(gripper_positions[-1] - gripper_positions[0], axis=1)
    denominator = np.maximum(plus_magnitude * return_magnitude, 1e-12)
    cosine = np.sum(plus_vectors * return_vectors, axis=1) / denominator
    ratio = return_magnitude / np.maximum(plus_magnitude, 1e-12)
    gripper_pass = gripper_valid & (
        texture_values(gripper_candidates) >= ANCHOR_TEXTURE_FLOOR
    )
    gripper_pass &= plus_magnitude >= np.maximum(1.5, 3.0 * gripper_noise)
    gripper_pass &= net_return <= np.maximum(1.5, 0.20 * plus_magnitude)
    gripper_pass &= cosine <= -0.9
    gripper_pass &= (ratio >= 0.5) & (ratio <= 2.0)
    gripper_indices = np.flatnonzero(gripper_pass)
    if len(gripper_indices) < ANCHOR_MIN_CLUSTER:
        raise ValueError("gripper region lacks three continuous reversible tracks")
    cluster_vectors = plus_vectors[gripper_indices]
    cluster_magnitude = plus_magnitude[gripper_indices]
    median_magnitude = float(np.median(cluster_magnitude))
    median_vector = np.median(cluster_vectors, axis=0)
    median_direction = median_vector / max(float(np.linalg.norm(median_vector)), 1e-12)
    directions = cluster_vectors / np.maximum(cluster_magnitude[:, None], 1e-12)
    angles = np.degrees(np.arccos(np.clip(directions @ median_direction, -1.0, 1.0)))
    if np.any(
        np.abs(cluster_magnitude - median_magnitude)
        > ANCHOR_MAGNITUDE_FRACTION * median_magnitude
    ):
        raise ValueError("gripper moving tracks are not magnitude-unimodal")
    if np.any(angles > ANCHOR_DIRECTION_DEG):
        raise ValueError("gripper moving tracks are not direction-unimodal")
    gripper_index = int(
        gripper_indices[
            np.argmin(
                np.linalg.norm(
                    gripper_candidates[gripper_indices] - gripper_center, axis=1
                )
            )
        ]
    )
    report = {
        "target_region_clipped": target_clipped,
        "gripper_region_clipped": gripper_clipped,
        "target_candidate_count": len(target_candidates),
        "gripper_candidate_count": len(gripper_candidates),
        "gripper_passing_count": len(gripper_indices),
        "target_citation_px": np.asarray(target_center).tolist(),
        "gripper_citation_px": np.asarray(gripper_center).tolist(),
        "target_anchor_px": target_positions[0][target_index].tolist(),
        "gripper_anchor_px": gripper_positions[0][gripper_index].tolist(),
        "target_anchor_offset_px": (
            target_positions[0][target_index] - np.asarray(target_center)
        ).tolist(),
        "gripper_anchor_offset_px": (
            gripper_positions[0][gripper_index] - np.asarray(gripper_center)
        ).tolist(),
        "target_texture_eigenvalue": float(
            texture_values(target_candidates)[target_index]
        ),
        "gripper_texture_eigenvalue": float(
            texture_values(gripper_candidates)[gripper_index]
        ),
        "target_max_forward_backward_px": float(target_fb[target_index]),
        "gripper_max_forward_backward_px": float(gripper_fb[gripper_index]),
        "gripper_plus_displacement_px": plus_vectors[gripper_index].tolist(),
        "gripper_return_displacement_px": return_vectors[gripper_index].tolist(),
        "gripper_return_error_px": float(net_return[gripper_index]),
        "gripper_reversal_cosine": float(cosine[gripper_index]),
        "gripper_return_ratio": float(ratio[gripper_index]),
        "gripper_noise_px": float(gripper_noise[gripper_index]),
        "cluster_median_magnitude_px": median_magnitude,
        "cluster_max_angle_deg": float(np.max(angles)),
    }
    return _LegacyViewAnchors(
        target_initial=target_positions[0][target_index],
        gripper_initial=gripper_positions[0][gripper_index],
        target_returned=target_positions[-1][target_index],
        gripper_returned=gripper_positions[-1][gripper_index],
        gripper_plus=gripper_positions[-2][gripper_index],
        report=report,
    )


def track_points(
    previous: np.ndarray, current: np.ndarray, points: np.ndarray
) -> TrackResult:
    """Track cited public pixels and fail closed on LK or texture ambiguity."""
    before = _gray(previous)
    after = _gray(current)
    initial = np.asarray(points, dtype=np.float32)
    if initial.ndim != 2 or initial.shape[1] != 2 or not np.isfinite(initial).all():
        raise ValueError("tracking points must be finite Nx2 pixels")
    options = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        before, after, initial, None, **options
    )
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        after, before, forward, None, **options
    )
    if forward is None or backward is None:
        raise ValueError("optical flow unavailable")
    valid = np.asarray(forward_status).reshape(-1) & np.asarray(
        backward_status
    ).reshape(-1)
    if not bool(np.all(valid)):
        raise ValueError("optical flow lost a cited feature")
    fb = np.linalg.norm(backward - initial, axis=1)
    texture = cv2.cornerMinEigenVal(before, blockSize=7, ksize=3)
    eigenvalues: list[float] = []
    for u, v in initial:
        x = int(np.clip(round(float(u)), 0, IMAGE_SIZE - 1))
        y = int(np.clip(round(float(v)), 0, IMAGE_SIZE - 1))
        window = texture[
            max(0, y - 5) : min(IMAGE_SIZE, y + 6),
            max(0, x - 5) : min(IMAGE_SIZE, x + 6),
        ]
        eigenvalues.append(float(np.max(window, initial=0.0)))
    result = TrackResult(
        points=np.asarray(forward, dtype=np.float64),
        max_forward_backward_error_px=float(np.max(fb, initial=0.0)),
        min_texture_eigenvalue=float(min(eigenvalues, default=0.0)),
    )
    if result.max_forward_backward_error_px > 1.0:
        raise ValueError("forward/backward optical-flow residual is too large")
    if result.min_texture_eigenvalue <= 1e-5:
        raise ValueError("cited feature patch lacks trackable texture")
    return result


def fit_image_jacobian(
    eef_deltas_m: np.ndarray,
    pixel_deltas: np.ndarray,
    *,
    noise_floor_px: float,
) -> JacobianFit:
    """Fit d(pixel error)/d(public EEF translation) from public probes."""
    eef = np.asarray(eef_deltas_m, dtype=np.float64)
    pixels = np.asarray(pixel_deltas, dtype=np.float64)
    if eef.ndim != 2 or eef.shape[1] != 3 or pixels.shape != (eef.shape[0], 4):
        raise ValueError("Jacobian samples have the wrong shape")
    if not np.isfinite(eef).all() or not np.isfinite(pixels).all():
        raise ValueError("Jacobian samples must be finite")
    if np.linalg.matrix_rank(eef) < 3:
        raise ValueError("EEF probes do not span rank three")
    jacobian = (np.linalg.pinv(eef) @ pixels).T
    singular = np.linalg.svd(jacobian, compute_uv=False)
    rank = int(np.sum(singular > max(float(noise_floor_px), MIN_FLOW_NOISE_PX)))
    if rank < 3 or singular[-1] <= 0:
        raise ValueError("image Jacobian is rank deficient")
    condition = float(singular[0] / singular[-1])
    if condition > 50.0:
        raise ValueError("image Jacobian is ill conditioned")
    residual = float(np.sqrt(np.mean((pixels - eef @ jacobian.T) ** 2)))
    if residual > max(1.0, 3.0 * max(float(noise_floor_px), MIN_FLOW_NOISE_PX)):
        raise ValueError("image Jacobian residual is too large")
    return JacobianFit(jacobian, rank, condition, residual)


def servo_delta(
    jacobian: np.ndarray,
    pixel_error: np.ndarray,
    *,
    max_step_m: float = 0.01,
    damping: float = 1.0,
) -> np.ndarray:
    matrix = np.asarray(jacobian, dtype=np.float64)
    error = np.asarray(pixel_error, dtype=np.float64)
    if (
        matrix.shape != (4, 3)
        or error.shape != (4,)
        or not np.isfinite(matrix).all()
        or not np.isfinite(error).all()
    ):
        raise ValueError("servo inputs violate the fixed dimensions")
    gram = matrix.T @ matrix + float(damping) ** 2 * np.eye(3)
    delta = np.linalg.solve(gram, matrix.T @ error)
    norm = float(np.linalg.norm(delta))
    if norm > max_step_m:
        delta *= max_step_m / norm
    predicted = float(np.linalg.norm(error - matrix @ delta))
    if predicted >= float(np.linalg.norm(error)) - 1e-9:
        raise ValueError("servo step has no predicted improvement")
    return delta


def probe_is_safe(points: np.ndarray, *, worst_pixels_per_mm: float) -> bool:
    """Preflight the next 20 mm probe using only tracked external-view pixels."""
    value = np.asarray(points, dtype=np.float64)
    if value.shape != (4, 2) or not np.isfinite(value).all() or worst_pixels_per_mm < 0:
        return False
    predicted = worst_pixels_per_mm * PROBE_DISPLACEMENT_MM
    border_margin = float(np.min(np.minimum(value, (IMAGE_SIZE - 1) - value)))
    separations = (
        np.linalg.norm(value[0] - value[1]),
        np.linalg.norm(value[2] - value[3]),
    )
    return (
        border_margin - predicted >= IMAGE_BORDER_PX
        and min(separations) - predicted >= MIN_FEATURE_SEPARATION_PX
    )


# Anchor discovery is isolated so its hash can remain frozen through revision 5.
from .anchor_selection import (  # noqa: F401
    AnchorSelectionError,
    MotionClusterError,
    TwoViewAnchors,
    ViewAnchors,
    select_motion_cluster,
    select_two_view_anchors,
    select_view_anchors,
)
