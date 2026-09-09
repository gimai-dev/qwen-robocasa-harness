"""Single-official-view target/gripper servo with no camera geometry or depth."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .active_vision import fit_target_jacobian, target_servo_delta
from .harness import ServoObservation
from .visual_servo import MIN_EEF_DISPLACEMENT_M, track_points

ALIGNMENT_PX = 8.0


@dataclass(frozen=True)
class SingleViewExecution:
    status: str
    final_observation: ServoObservation
    report: dict[str, object]


Execute = Callable[[tuple[float, float, float] | None, str], ServoObservation]


def _frame(observation: ServoObservation, view: str) -> np.ndarray:
    if view == "left":
        return observation.left
    if view == "right":
        return observation.right
    raise ValueError("single-view servo requires one official external view")


def _track(
    previous: ServoObservation,
    current: ServoObservation,
    points: np.ndarray,
    *,
    view: str,
) -> tuple[np.ndarray, float]:
    result = track_points(_frame(previous, view), _frame(current, view), points)
    return result.points, result.max_forward_backward_error_px


def run_single_view_center(
    *,
    initial: ServoObservation,
    view: str,
    target_px: np.ndarray,
    gripper_px: np.ndarray,
    reset_eef_position_m: np.ndarray,
    execute: Execute,
    audit_only: bool = False,
    max_servo_steps: int = 16,
) -> SingleViewExecution:
    """Calibrate reversible 3-D EEF probes, then center in one external RGB view."""
    points = np.asarray([target_px, gripper_px], dtype=np.float64)
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError("target and gripper citations must be finite 2-D pixels")
    report: dict[str, object] = {
        "schema": "robocasa-inspect-single-view-center/v1",
        "view": view,
        "target_source": "qwen_target_only",
        "gripper_source": "reversible_rgb_articulation",
        "camera_geometry_used": False,
        "depth_used": False,
        "probes": [],
        "servo_steps": [],
        "audit_only": audit_only,
    }
    current = initial
    eef_samples: list[np.ndarray] = []
    pixel_samples: list[np.ndarray] = []
    try:
        for axis in range(3):
            start = current
            start_points = points.copy()
            delta = np.zeros(3)
            delta[axis] = 0.02
            plus = execute(tuple(delta), f"probe_{'xyz'[axis]}_plus_1")
            plus = execute(tuple(delta), f"probe_{'xyz'[axis]}_plus_2")
            plus_points, plus_fb = _track(
                start, plus, start_points, view=view
            )
            eef_delta = plus.eef_position_m - start.eef_position_m
            if np.linalg.norm(eef_delta) < MIN_EEF_DISPLACEMENT_M:
                raise ValueError("public EEF probe displacement is too small")
            target_motion = float(np.linalg.norm(plus_points[0] - start_points[0]))
            gripper_motion = plus_points[1] - start_points[1]
            if target_motion > 1.5:
                raise ValueError("target patch moved during fixed-view calibration")
            if np.linalg.norm(gripper_motion) < 0.75:
                raise ValueError("gripper response is unobservable")
            returned = execute(
                tuple(-delta), f"probe_{'xyz'[axis]}_return_1"
            )
            for correction_index in range(5):
                return_error = start.eef_position_m - returned.eef_position_m
                if np.linalg.norm(return_error) <= 0.002:
                    break
                correction = np.clip(return_error, -0.01, 0.01)
                returned = execute(
                    tuple(float(value) for value in correction),
                    f"probe_{'xyz'[axis]}_return_correction_{correction_index + 1}",
                )
            returned_points, return_fb = _track(
                plus, returned, plus_points, view=view
            )
            eef_return = float(
                np.linalg.norm(returned.eef_position_m - start.eef_position_m)
            )
            pixel_return = float(
                np.max(np.linalg.norm(returned_points - start_points, axis=1))
            )
            eef_limit = max(0.002, 0.20 * float(np.linalg.norm(eef_delta)))
            pixel_limit = max(1.0, 0.20 * float(np.linalg.norm(gripper_motion)))
            if eef_return > eef_limit or pixel_return > pixel_limit:
                raise ValueError("calibration return gate failed")
            eef_samples.append(eef_delta)
            pixel_samples.append(gripper_motion)
            report["probes"].append(
                {
                    "axis": "xyz"[axis],
                    "requested_delta_m": delta.tolist(),
                    "requested_plus_chunks": 2,
                    "bounded_public_state_return_corrections": 5,
                    "measured_eef_delta_m": eef_delta.tolist(),
                    "target_motion_px": target_motion,
                    "gripper_motion_px": gripper_motion.tolist(),
                    "eef_return_error_m": eef_return,
                    "pixel_return_error_px": pixel_return,
                    "forward_backward_error_px": max(plus_fb, return_fb),
                }
            )
            current, points = returned, returned_points
        fit = fit_target_jacobian(
            np.asarray(eef_samples),
            np.asarray(pixel_samples),
            noise_floor_px=0.25,
        )
        report.update(
            {
                "jacobian": fit.jacobian.tolist(),
                "jacobian_rank": fit.rank,
                "jacobian_condition_number": fit.condition_number,
                "jacobian_residual_px": fit.residual_px,
            }
        )
    except ValueError as error:
        report["failure"] = str(error)
        return SingleViewExecution("calibration_failed", current, report)
    if audit_only:
        return SingleViewExecution("calibration_audit_passed", current, report)

    aligned = 0
    for index in range(max_servo_steps):
        error = points[0] - points[1]
        if np.linalg.norm(error) <= ALIGNMENT_PX:
            aligned += 1
            if aligned == 2:
                report["terminal_error_px"] = error.tolist()
                return SingleViewExecution("centered", current, report)
            following = execute(None, "alignment_confirmation")
        else:
            aligned = 0
            try:
                delta = target_servo_delta(
                    fit.jacobian,
                    current_uv_px=points[1],
                    desired_uv_px=points[0],
                )
            except ValueError as error:
                report["failure"] = str(error)
                return SingleViewExecution("centering_failed", current, report)
            if (
                np.linalg.norm(
                    current.eef_position_m
                    + delta
                    - np.asarray(reset_eef_position_m, dtype=np.float64)
                )
                > 0.20
            ):
                report["failure"] = "reset-relative public EEF workspace bound"
                return SingleViewExecution("centering_failed", current, report)
            following = execute(tuple(delta), f"center_{index + 1}")
            report["servo_steps"].append(
                {
                    "index": index + 1,
                    "requested_delta_m": delta.tolist(),
                    "error_before_px": error.tolist(),
                }
            )
        try:
            following_points, fb = _track(
                current, following, points, view=view
            )
        except ValueError as error:
            report["failure"] = str(error)
            return SingleViewExecution("centering_failed", current, report)
        before = float(np.linalg.norm(error))
        measured = float(np.linalg.norm(following_points[0] - following_points[1]))
        if aligned == 0 and measured >= before - 0.25:
            report["failure"] = "measured centering did not improve"
            return SingleViewExecution("centering_failed", following, report)
        if report["servo_steps"]:
            report["servo_steps"][-1].update(
                {
                    "measured_error_px": measured,
                    "forward_backward_error_px": fb,
                }
            )
        current, points = following, following_points
    report["failure"] = "centering step budget exhausted"
    return SingleViewExecution("centering_failed", current, report)
