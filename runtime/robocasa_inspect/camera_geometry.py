"""Public pinhole calibration for the three official RoboCasa RGB cameras."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .contracts import CAMERAS

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
MIN_PARALLAX_BASELINE_M = 0.02
MIN_RAY_ANGLE_DEG = 3.0
MAX_REPROJECTION_ERROR_PX = 2.0


@dataclass(frozen=True)
class TriangulationEvidence:
    point_world_m: tuple[float, float, float]
    baseline_m: float
    ray_angle_deg: float
    reprojection_error_px: tuple[float, float]


def official_camera_calibration(environment: object) -> dict[str, dict[str, object]]:
    """Project MuJoCo's official fixed camera state into a closed public record."""
    simulation = environment.unwrapped.sim
    output: dict[str, dict[str, object]] = {}
    for label, name in zip(("left", "right", "wrist"), CAMERAS, strict=True):
        model_name = name.removeprefix("video.")
        camera_id = simulation.model.camera_name2id(model_name)
        fovy_deg = float(simulation.model.cam_fovy[camera_id])
        position = np.asarray(simulation.data.cam_xpos[camera_id], dtype=np.float64)
        xmat = np.asarray(simulation.data.cam_xmat[camera_id], dtype=np.float64).reshape(
            3, 3
        )
        if (
            not 1.0 <= fovy_deg <= 179.0
            or position.shape != (3,)
            or not np.isfinite(position).all()
            or not np.isfinite(xmat).all()
            or not np.allclose(xmat.T @ xmat, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(xmat), 1.0, atol=1e-6, rtol=0)
        ):
            raise RuntimeError(f"official camera calibration drift: {name}")
        focal = 0.5 * IMAGE_HEIGHT / np.tan(np.deg2rad(fovy_deg) / 2.0)
        output[label] = {
            "camera_name": name,
            "mujoco_camera_name": model_name,
            "image_width_px": IMAGE_WIDTH,
            "image_height_px": IMAGE_HEIGHT,
            "fx_px": float(focal),
            "fy_px": float(focal),
            "cx_px": (IMAGE_WIDTH - 1.0) / 2.0,
            "cy_px": (IMAGE_HEIGHT - 1.0) / 2.0,
            "camera_position_world_m": position.tolist(),
            "camera_xmat_world": xmat.tolist(),
            "projection": "mujoco-camera-x-right-y-up-minus-z-forward",
        }
    return output


def project_world_point(
    calibration: Mapping[str, object], point_world_m: object
) -> tuple[float, float, float]:
    """Return pixel u/v and positive optical depth for one public world point."""
    point = np.asarray(point_world_m, dtype=np.float64)
    position = np.asarray(calibration["camera_position_world_m"], dtype=np.float64)
    xmat = np.asarray(calibration["camera_xmat_world"], dtype=np.float64)
    if point.shape != (3,) or position.shape != (3,) or xmat.shape != (3, 3):
        raise ValueError("invalid camera projection shapes")
    local = xmat.T @ (point - position)
    depth = -float(local[2])
    if not np.isfinite(local).all() or depth <= 0.0:
        raise ValueError("point is not in front of the camera")
    u = float(calibration["cx_px"]) + float(calibration["fx_px"]) * float(
        local[0]
    ) / depth
    v = float(calibration["cy_px"]) - float(calibration["fy_px"]) * float(
        local[1]
    ) / depth
    return u, v, depth


def pixel_ray_world(
    calibration: Mapping[str, object], uv_px: object
) -> tuple[np.ndarray, np.ndarray]:
    """Return the public camera origin and unit world ray for one RGB pixel."""
    uv = np.asarray(uv_px, dtype=np.float64)
    position = np.asarray(calibration["camera_position_world_m"], dtype=np.float64)
    xmat = np.asarray(calibration["camera_xmat_world"], dtype=np.float64)
    if uv.shape != (2,) or not np.isfinite(uv).all():
        raise ValueError("pixel ray requires two finite coordinates")
    local = np.asarray(
        [
            (uv[0] - float(calibration["cx_px"])) / float(calibration["fx_px"]),
            -(uv[1] - float(calibration["cy_px"])) / float(calibration["fy_px"]),
            -1.0,
        ]
    )
    direction = xmat @ local
    direction /= np.linalg.norm(direction)
    return position, direction


def triangulate_two_view_point(
    first_calibration: Mapping[str, object],
    first_uv_px: object,
    second_calibration: Mapping[str, object],
    second_uv_px: object,
    *,
    base_position_world_m: object,
) -> TriangulationEvidence:
    """Triangulate one tracked RGB feature with strict public geometry gates."""
    first_origin, first_ray = pixel_ray_world(first_calibration, first_uv_px)
    second_origin, second_ray = pixel_ray_world(second_calibration, second_uv_px)
    baseline = float(np.linalg.norm(second_origin - first_origin))
    cosine = float(np.clip(np.dot(first_ray, second_ray), -1.0, 1.0))
    ray_angle = float(np.degrees(np.arccos(cosine)))
    if baseline < MIN_PARALLAX_BASELINE_M:
        raise ValueError("parallax baseline is below 2 cm")
    if ray_angle < MIN_RAY_ANGLE_DEG:
        raise ValueError("parallax ray angle is below three degrees")
    projectors = [
        np.eye(3) - np.outer(first_ray, first_ray),
        np.eye(3) - np.outer(second_ray, second_ray),
    ]
    system = projectors[0] + projectors[1]
    if np.linalg.cond(system) > 1_000.0:
        raise ValueError("parallax system is ill conditioned")
    point = np.linalg.solve(
        system, projectors[0] @ first_origin + projectors[1] @ second_origin
    )
    first_projection = project_world_point(first_calibration, point)
    second_projection = project_world_point(second_calibration, point)
    errors = (
        float(
            np.linalg.norm(
                np.asarray(first_projection[:2]) - np.asarray(first_uv_px)
            )
        ),
        float(
            np.linalg.norm(
                np.asarray(second_projection[:2]) - np.asarray(second_uv_px)
            )
        ),
    )
    if max(errors) > MAX_REPROJECTION_ERROR_PX:
        raise ValueError("parallax reprojection exceeds two pixels")
    base = np.asarray(base_position_world_m, dtype=np.float64)
    if base.shape != (3,) or not np.isfinite(base).all():
        raise ValueError("base position is invalid")
    relative = point - base
    if np.linalg.norm(relative[:2]) > 1.5 or not 0.0 <= point[2] <= 2.5:
        raise ValueError("triangulated point violates the certified workspace")
    return TriangulationEvidence(
        tuple(float(value) for value in point), baseline, ray_angle, errors
    )


def best_strict_triangulation(
    samples: list[Mapping[str, object]], *, base_position_world_m: object
) -> tuple[TriangulationEvidence, tuple[int, int]]:
    """Choose the best strictly valid pair from a public RGB parallax track."""
    candidates: list[tuple[float, float, int, int, TriangulationEvidence]] = []
    for first_index, first in enumerate(samples):
        if not isinstance(first, Mapping):
            raise ValueError("parallax sample is invalid")
        for second_index in range(first_index + 1, len(samples)):
            second = samples[second_index]
            if not isinstance(second, Mapping):
                raise ValueError("parallax sample is invalid")
            try:
                evidence = triangulate_two_view_point(
                    first["calibration"],
                    first["uv_px"],
                    second["calibration"],
                    second["uv_px"],
                    base_position_world_m=base_position_world_m,
                )
            except (KeyError, TypeError, ValueError):
                continue
            candidates.append(
                (
                    max(evidence.reprojection_error_px),
                    -evidence.ray_angle_deg,
                    first_index,
                    second_index,
                    evidence,
                )
            )
    if not candidates:
        raise ValueError("no strictly valid parallax pair")
    _, _, first_index, second_index, evidence = min(candidates)
    return evidence, (first_index, second_index)
