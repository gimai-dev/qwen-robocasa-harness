"""Target-only visual control from the three official RoboCasa RGB views.

External views ground a feature across viewpoints.  Only the moving wrist camera can
produce a target-pixel Jacobian: a stationary target in a fixed external camera must
fail the rank gate rather than manufacture depth or an end-effector anchor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IMAGE_SIZE = 256
IMAGE_BORDER_PX = 12.0
MAX_CONDITION = 50.0
MAX_ROTATION_DEG = 10.0
MIN_SCALE_RATIO = 0.8
MAX_SCALE_RATIO = 1.25
MIN_NOISE_PX = 0.25


@dataclass(frozen=True)
class TargetJacobianFit:
    jacobian: np.ndarray
    rank: int
    condition_number: float
    residual_px: float


@dataclass(frozen=True)
class ApproachProbeEvidence:
    approach_sign: int
    scale_signal: float
    scale_noise: float
    pixel_return_error: float


@dataclass(frozen=True)
class TrackContinuityEvidence:
    scale_ratio: float
    rotation_deg: float
    residual_px: float


def measure_point_noise(points_px: np.ndarray) -> float:
    """Measure 2-D static point noise from exactly five unchanged observations."""
    points = np.asarray(points_px, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("noise authority requires exactly five finite 2-D points")
    variance = np.var(points, axis=0, ddof=1)
    return max(MIN_NOISE_PX, float(np.sqrt(np.sum(variance))))


def fit_target_jacobian(
    eef_deltas_m: np.ndarray,
    pixel_deltas: np.ndarray,
    *,
    noise_floor_px: float,
) -> TargetJacobianFit:
    """Fit a 2x3 target-pixel Jacobian using only public EEF deltas and RGB flow."""
    eef = np.asarray(eef_deltas_m, dtype=np.float64)
    pixels = np.asarray(pixel_deltas, dtype=np.float64)
    noise = float(noise_floor_px)
    if eef.ndim != 2 or eef.shape[1] != 3 or pixels.shape != (len(eef), 2):
        raise ValueError("target Jacobian samples have the wrong shape")
    if not np.isfinite(eef).all() or not np.isfinite(pixels).all() or noise < 0:
        raise ValueError("target Jacobian evidence must be finite")
    if np.linalg.matrix_rank(eef) < 3:
        raise ValueError("EEF probes do not span rank three")
    jacobian = (np.linalg.pinv(eef) @ pixels).T
    singular = np.linalg.svd(jacobian, compute_uv=False)
    tolerance = max(float(singular[0]) * 1e-8, 1e-9)
    rank = int(np.sum(singular > tolerance))
    if rank != 2 or singular[-1] <= 0:
        raise ValueError("target image Jacobian must have rank two")
    condition = float(singular[0] / singular[-1])
    if condition > MAX_CONDITION:
        raise ValueError("target image Jacobian is ill conditioned")
    residual = float(np.sqrt(np.mean((pixels - eef @ jacobian.T) ** 2)))
    if residual > max(1.0, 3.0 * max(noise, MIN_NOISE_PX)):
        raise ValueError("target image Jacobian residual is too large")
    probe_scale = float(np.max(np.linalg.norm(eef, axis=1), initial=0.0))
    if singular[-1] * probe_scale < 3.0 * max(noise, MIN_NOISE_PX):
        raise ValueError("target image Jacobian is unobservable above pixel noise")
    return TargetJacobianFit(jacobian, rank, condition, residual)


def target_servo_delta(
    jacobian: np.ndarray,
    *,
    current_uv_px: np.ndarray,
    desired_uv_px: np.ndarray,
    max_step_m: float = 0.005,
    damping: float = 1.0,
) -> np.ndarray:
    """Return one bounded damped-least-squares motion with predicted improvement."""
    matrix = np.asarray(jacobian, dtype=np.float64)
    current = np.asarray(current_uv_px, dtype=np.float64)
    desired = np.asarray(desired_uv_px, dtype=np.float64)
    if (
        matrix.shape != (2, 3)
        or current.shape != (2,)
        or desired.shape != (2,)
        or not np.isfinite(matrix).all()
        or not np.isfinite(current).all()
        or not np.isfinite(desired).all()
        or max_step_m <= 0
        or damping <= 0
    ):
        raise ValueError("target servo inputs violate the closed dimensions")
    error = desired - current
    gram = matrix.T @ matrix + float(damping) ** 2 * np.eye(3)
    delta = np.linalg.solve(gram, matrix.T @ error)
    norm = float(np.linalg.norm(delta))
    if norm > max_step_m:
        delta *= max_step_m / norm
    if np.linalg.norm(error - matrix @ delta) >= np.linalg.norm(error) - 1e-9:
        raise ValueError("target servo step has no predicted improvement")
    return delta


def candidate_null_direction(jacobian: np.ndarray) -> np.ndarray:
    """Return the only image-Jacobian null direction, without assigning depth sign."""
    matrix = np.asarray(jacobian, dtype=np.float64)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("null direction requires a finite 2x3 Jacobian")
    _, singular, right = np.linalg.svd(matrix, full_matrices=True)
    if len(singular) != 2 or singular[-1] <= singular[0] * 1e-8:
        raise ValueError("null direction requires a rank-two Jacobian")
    direction = right[-1]
    return direction / np.linalg.norm(direction)


def validate_approach_probe(
    *,
    scale_before: float,
    scale_plus: float,
    scale_returned: float,
    scale_noise: float,
    pixel_return_error: float,
) -> ApproachProbeEvidence:
    """Assign approach sign only from a visible, reversible wrist scale response."""
    values = np.asarray(
        [scale_before, scale_plus, scale_returned, scale_noise, pixel_return_error],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or scale_before <= 0 or scale_noise < 0:
        raise ValueError("approach probe evidence is invalid")
    signal = float(scale_plus - scale_before)
    threshold = max(0.01, 3.0 * scale_noise)
    if abs(signal) <= threshold:
        raise ValueError("approach direction is unobservable")
    if (
        abs(scale_returned - scale_before) > threshold
        or pixel_return_error > 1.0
    ):
        raise ValueError("approach probe failed its return gate")
    return ApproachProbeEvidence(
        approach_sign=1 if signal > 0 else -1,
        scale_signal=signal,
        scale_noise=scale_noise,
        pixel_return_error=pixel_return_error,
    )


def validate_track_continuity(
    previous_points_px: np.ndarray, current_points_px: np.ndarray
) -> TrackContinuityEvidence:
    """Validate a textured feature patch by border and similarity-transform gates."""
    previous = np.asarray(previous_points_px, dtype=np.float64)
    current = np.asarray(current_points_px, dtype=np.float64)
    if (
        previous.ndim != 2
        or previous.shape != current.shape
        or previous.shape[0] < 3
        or previous.shape[1] != 2
        or not np.isfinite(previous).all()
        or not np.isfinite(current).all()
    ):
        raise ValueError("track continuity requires matching finite Nx2 patches")
    margin = np.minimum(current, (IMAGE_SIZE - 1) - current)
    if float(np.min(margin)) < IMAGE_BORDER_PX:
        raise ValueError("tracked feature violates the image border")
    before = previous - previous.mean(axis=0)
    after = current - current.mean(axis=0)
    denominator = float(np.sum(before**2))
    if denominator <= 1e-12:
        raise ValueError("tracked feature patch is degenerate")
    left, singular, right = np.linalg.svd(before.T @ after)
    rotation = left @ right
    if np.linalg.det(rotation) <= 0:
        raise ValueError("tracked feature reflected")
    scale = float(np.sum(singular) / denominator)
    predicted = current.mean(axis=0) + scale * before @ rotation
    residual = float(np.sqrt(np.mean(np.sum((current - predicted) ** 2, axis=1))))
    angle = abs(float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))))
    if angle > MAX_ROTATION_DEG:
        raise ValueError("tracked feature rotation exceeds ten degrees")
    if scale < MIN_SCALE_RATIO or scale > MAX_SCALE_RATIO:
        raise ValueError("tracked feature scale ratio is invalid")
    if residual > 1.0:
        raise ValueError("tracked feature affine residual is too large")
    return TrackContinuityEvidence(scale, angle, residual)
