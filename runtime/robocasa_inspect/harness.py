"""Task-independent public-RGB visual-servo harness for Qwen citations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .contracts import Command
from .visual_servo import (
    ALIGNMENT_TOLERANCE_NORMALIZED,
    ALIGNMENT_TOLERANCE_PX,
    MIN_EEF_DISPLACEMENT_M,
    MIN_FLOW_NOISE_PX,
    AnchorSelectionError,
    fit_image_jacobian,
    normalized_to_pixel,
    probe_is_safe,
    select_two_view_anchors,
    servo_delta,
    track_points,
)


@dataclass(frozen=True)
class ServoObservation:
    observation_id: str
    left: np.ndarray
    right: np.ndarray
    eef_position_m: np.ndarray
    finger_qpos: np.ndarray


@dataclass(frozen=True)
class ServoExecution:
    status: str
    final_observation: ServoObservation
    report: dict[str, object]
    audit_images: dict[str, np.ndarray] | None = None


Execute = Callable[[tuple[float, float, float] | None, str, str], ServoObservation]


def _cited_points(command: Command) -> np.ndarray:
    if command.kind != "servo_feature" or command.left is None or command.right is None:
        raise ValueError("servo session requires a complete servo_feature command")
    return np.asarray(
        [
            normalized_to_pixel(command.left.target_uv),
            normalized_to_pixel(command.left.gripper_uv),
            normalized_to_pixel(command.right.target_uv),
            normalized_to_pixel(command.right.gripper_uv),
        ],
        dtype=np.float64,
    )


def _track(previous: ServoObservation, current: ServoObservation, points: np.ndarray):
    left = track_points(previous.left, current.left, points[:2])
    right = track_points(previous.right, current.right, points[2:])
    return (
        np.vstack((left.points, right.points)),
        max(left.max_forward_backward_error_px, right.max_forward_backward_error_px),
        min(left.min_texture_eigenvalue, right.min_texture_eigenvalue),
    )


def _pixel_error(points: np.ndarray) -> np.ndarray:
    return np.concatenate((points[0] - points[1], points[2] - points[3]))


def _aligned(points: np.ndarray) -> bool:
    return bool(
        np.linalg.norm(points[0] - points[1]) <= ALIGNMENT_TOLERANCE_PX
        and np.linalg.norm(points[2] - points[3]) <= ALIGNMENT_TOLERANCE_PX
    )


def run_servo_session(
    command: Command,
    initial: ServoObservation,
    *,
    reset_eef_position_m: np.ndarray,
    execute: Execute,
    max_servo_steps: int = 12,
    audit_only: bool = False,
) -> ServoExecution:
    """Calibrate once and servo only in public image space.

    ``execute(None, ...)`` requests a fresh same-pose RGB observation. A numeric delta
    requests one official five-step action chunk. The callback owns all provenance and
    simulator mailbox checks.
    """
    points = _cited_points(command)
    report: dict[str, object] = {
        "schema": "robocasa-public-servo/v1",
        "camera_resolution": [256, 256],
        "alignment_tolerance_px": ALIGNMENT_TOLERANCE_PX,
        "alignment_tolerance_normalized": ALIGNMENT_TOLERANCE_NORMALIZED,
        "wrist_used_by_servo": False,
        "probes": [],
        "servo_steps": [],
        "anchor_audit_only": audit_only,
    }
    current = initial
    try:
        settled = execute((0.0, 0.0, 0.0), "open", "calibration_settle")
        same_pose = [settled]
        for index in range(4):
            same_pose.append(execute(None, "hold", f"same_pose_noise_{index + 2}"))
        current = same_pose[-1]
        if not probe_is_safe(points, worst_pixels_per_mm=0.0):
            raise ValueError("initial citation regions violate probe safety")
        delta = np.asarray([0.02, 0.0, 0.0])
        plus = execute(tuple(delta), "open", "probe_x_plus")
        eef_delta = plus.eef_position_m - current.eef_position_m
        if np.linalg.norm(eef_delta) < MIN_EEF_DISPLACEMENT_M:
            raise ValueError("public EEF probe displacement is too small")
        returned = execute(tuple(-delta), "open", "probe_x_return")
        eef_return = float(
            np.linalg.norm(returned.eef_position_m - current.eef_position_m)
        )
        eef_return_limit = max(0.002, 0.20 * float(np.linalg.norm(eef_delta)))
        if eef_return > eef_return_limit:
            raise ValueError("calibration_return_failed")
        selection = select_two_view_anchors(
            left_same_pose=[item.left for item in same_pose],
            right_same_pose=[item.right for item in same_pose],
            left_plus=plus.left,
            right_plus=plus.right,
            left_returned=returned.left,
            right_returned=returned.right,
            left_target_center=points[0],
            right_target_center=points[2],
            left_gripper_center=points[1],
            right_gripper_center=points[3],
        )
        left, right = selection.left, selection.right
        report["anchor_selection"] = {
            "mode": selection.mode,
            "left": left.report,
            "right": right.report,
            **selection.report,
        }
        report["audit_source_observation_id"] = same_pose[0].observation_id
        report["frozen_probe_contract"] = {
            "same_pose_frame_count": 5,
            "probe_translation_m": delta.tolist(),
            "return_translation_m": (-delta).tolist(),
            "action_chunk_steps": 5,
        }
        if audit_only:
            return ServoExecution(
                "anchor_audit_required",
                returned,
                report,
                {"left": same_pose[0].left, "right": same_pose[0].right},
            )
        points = np.vstack(
            (
                left.target_returned,
                left.gripper_returned,
                right.target_returned,
                right.gripper_returned,
            )
        )
        gripper_delta = np.concatenate(
            (
                left.gripper_plus - left.gripper_initial,
                right.gripper_plus - right.gripper_initial,
            )
        )
        noise = max(
            MIN_FLOW_NOISE_PX,
            float(left.report["gripper_noise_px"]),
            float(right.report["gripper_noise_px"]),
        )
        report["same_pose_noise_px"] = noise
        eef_samples: list[np.ndarray] = [eef_delta]
        pixel_samples: list[np.ndarray] = [gripper_delta]
        worst_pixels_per_mm = float(
            np.linalg.norm(gripper_delta) / (1_000.0 * np.linalg.norm(eef_delta))
        )
        report["probes"].append(
            {
                "axis": "x",
                "requested_m": delta.tolist(),
                "measured_eef_delta_m": eef_delta.tolist(),
                "pixel_delta": gripper_delta.tolist(),
                "return_eef_error_m": eef_return,
                "return_eef_limit_m": eef_return_limit,
                "worst_pixels_per_mm": worst_pixels_per_mm,
                "left_pixels_per_requested_mm": float(
                    np.linalg.norm(gripper_delta[:2]) / 20.0
                ),
                "right_pixels_per_requested_mm": float(
                    np.linalg.norm(gripper_delta[2:]) / 20.0
                ),
            }
        )
        current = returned
        finger_origin = settled.finger_qpos.copy()
        for axis in range(1, 3):
            if not probe_is_safe(points, worst_pixels_per_mm=worst_pixels_per_mm):
                raise ValueError(
                    "remaining probe would leave the registered pixel neighborhood"
                )
            start, start_points = current, points.copy()
            separation = [
                float(np.linalg.norm(points[0] - points[1])),
                float(np.linalg.norm(points[2] - points[3])),
            ]
            delta = np.zeros(3)
            delta[axis] = 0.02
            plus = execute(tuple(delta), "open", f"probe_{'xyz'[axis]}_plus")
            plus_points, plus_fb, _ = _track(start, plus, start_points)
            eef_delta = plus.eef_position_m - start.eef_position_m
            target_motion = max(
                float(np.linalg.norm(plus_points[0] - start_points[0])),
                float(np.linalg.norm(plus_points[2] - start_points[2])),
            )
            gripper_delta = np.concatenate(
                (plus_points[1] - start_points[1], plus_points[3] - start_points[3])
            )
            if np.linalg.norm(eef_delta) < MIN_EEF_DISPLACEMENT_M:
                raise ValueError("public EEF probe displacement is too small")
            if np.linalg.norm(gripper_delta) < max(1.5, 3.0 * noise):
                raise ValueError("gripper pixel response is below the noise gate")
            if target_motion > max(1.5, 3.0 * noise):
                raise ValueError("target patch moved during free-space calibration")
            returned = execute(tuple(-delta), "open", f"probe_{'xyz'[axis]}_return")
            return_points, return_fb, _ = _track(plus, returned, plus_points)
            eef_return = float(
                np.linalg.norm(returned.eef_position_m - start.eef_position_m)
            )
            pixel_return = float(
                np.max(np.linalg.norm(return_points - start_points, axis=1))
            )
            eef_return_limit = max(0.002, 0.20 * float(np.linalg.norm(eef_delta)))
            per_view_plus = max(
                float(np.linalg.norm(gripper_delta[:2])),
                float(np.linalg.norm(gripper_delta[2:])),
            )
            pixel_return_limit = max(1.5, 0.20 * per_view_plus)
            if eef_return > eef_return_limit:
                raise ValueError("calibration_return_failed")
            if pixel_return > pixel_return_limit:
                raise ValueError("probe pixel return gate failed")
            if np.max(np.abs(returned.finger_qpos - finger_origin)) > 0.002:
                raise ValueError("finger qpos changed during open-gripper calibration")
            ppm = float(
                np.linalg.norm(gripper_delta) / (1_000.0 * np.linalg.norm(eef_delta))
            )
            worst_pixels_per_mm = max(worst_pixels_per_mm, ppm)
            eef_samples.append(eef_delta)
            pixel_samples.append(gripper_delta)
            report["probes"].append(
                {
                    "axis": "xyz"[axis],
                    "requested_m": delta.tolist(),
                    "measured_eef_delta_m": eef_delta.tolist(),
                    "pixel_delta": gripper_delta.tolist(),
                    "two_view_separation_px": separation,
                    "forward_backward_px": max(plus_fb, return_fb),
                    "return_eef_error_m": eef_return,
                    "return_eef_limit_m": eef_return_limit,
                    "return_pixel_error_px": pixel_return,
                    "return_pixel_limit_px": pixel_return_limit,
                    "worst_pixels_per_mm": worst_pixels_per_mm,
                }
            )
            current, points = returned, return_points

        fit = fit_image_jacobian(
            np.asarray(eef_samples), np.asarray(pixel_samples), noise_floor_px=noise
        )
        report["jacobian"] = fit.jacobian.tolist()
        report["jacobian_rank"] = fit.rank
        report["jacobian_condition_number"] = fit.condition_number
        report["jacobian_residual_px"] = fit.residual_px
    except AnchorSelectionError as error:
        report["anchor_selection_failure"] = error.report
        report["failure"] = str(error)
        return ServoExecution("servo_never_engaged", current, report)
    except ValueError as error:
        report["failure"] = str(error)
        return ServoExecution("servo_never_engaged", current, report)

    aligned_frames = 0
    no_improvement = 0
    for index in range(max_servo_steps):
        error = _pixel_error(points)
        if _aligned(points):
            aligned_frames += 1
            if aligned_frames == 2:
                report["terminal_pixel_error"] = error.tolist()
                return ServoExecution("aligned", current, report)
            next_observation = execute(None, "hold", "alignment_confirmation")
        else:
            aligned_frames = 0
            try:
                delta = servo_delta(
                    fit.jacobian, error, max_step_m=0.01 / (2**no_improvement)
                )
            except ValueError as control_error:
                report["failure"] = str(control_error)
                return ServoExecution("servo_engaged_failed", current, report)
            predicted = float(np.linalg.norm(error - fit.jacobian @ delta))
            candidate = current.eef_position_m + delta
            if np.linalg.norm(candidate - np.asarray(reset_eef_position_m)) > 0.20:
                report["failure"] = "reset-relative public EEF workspace bound"
                return ServoExecution("servo_engaged_failed", current, report)
            next_observation = execute(
                tuple(delta), command.gripper, f"servo_{index + 1}"
            )
            report["servo_steps"].append(
                {
                    "index": index + 1,
                    "requested_delta_m": delta.tolist(),
                    "error_before_px": error.tolist(),
                    "predicted_error_norm_px": predicted,
                }
            )
        try:
            next_points, fb, _ = _track(current, next_observation, points)
        except ValueError as error:
            report["failure"] = str(error)
            return ServoExecution("servo_engaged_failed", current, report)
        measured = float(np.linalg.norm(_pixel_error(next_points)))
        before = float(np.linalg.norm(error))
        if not _aligned(points) and measured >= before - 0.5:
            no_improvement += 1
            if no_improvement > 1:
                report["failure"] = "trust-region measured improvement gate"
                return ServoExecution("servo_engaged_failed", next_observation, report)
        else:
            no_improvement = 0
        if report["servo_steps"]:
            report["servo_steps"][-1]["measured_error_norm_px"] = measured
            report["servo_steps"][-1]["forward_backward_px"] = fb
        current, points = next_observation, next_points
    report["failure"] = "servo step budget"
    return ServoExecution("servo_engaged_failed", current, report)
