"""Fable-accepted public-RGB calibration amendment for Cycle 9."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

FABLE_VERDICT = "ACCEPT"
FABLE_CLARIFICATION = (
    "Capacity is a pre-implementation evidence gate; dev20 is a separate "
    "post-implementation official-success gate."
)

MATCHER = {
    "qwen_box_calls_per_image": 2,
    "minimum_box_iou": 0.5,
    "dino_feature": "x_norm_patchtokens",
    "dino_grid": [16, 16],
    "dino_minimum_cosine_similarity": 0.55,
    "dino_minimum_best_second_margin": 0.002,
    "ncc_patch_radius_px": 5,
    "ncc_search_radius_px": 8,
    "ncc_minimum_score": 0.25,
    "ncc_minimum_best_second_margin": 0.001,
    "matching": "per-view-mutual-nearest-current-source-only",
}

CALIBRATION = {
    "candidate_pairs": 30,
    "minimum_usable_pairs": 10,
    "minimum_usable_tasks": 3,
    "minimum_public_eef_displacement_m": 0.020,
    "maximum_public_eef_displacement_m": 0.100,
    "same_closed_gripper_interval": True,
    "camera_depth_margin_m": 0.100,
    "safety_factor_grid": [round(0.25 + 0.05 * index, 2) for index in range(76)],
    "every_estimate_at_least_true_displacement": True,
    "every_estimate_at_most_twice_true_displacement": True,
    "disjoint_from": ["dev20", "family10_slices", "held_out_matrices"],
}


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_capacity_amendment(
    *, base_cycle9_sha256: str, fable_review_sha256: str
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "robocasa-inspect-goal30-cycle9-capacity-amendment/v1",
        "fable_verdict": FABLE_VERDICT,
        "fable_clarification": FABLE_CLARIFICATION,
        "parents": {
            "base_cycle9_sha256": base_cycle9_sha256,
            "fable_review_sha256": fable_review_sha256,
        },
        "inputs": [
            "official_current_reset_rgb",
            "official_source_demonstration_rgb",
            "official_public_camera_calibration",
            "public_eef_state",
            "source_binary_gripper_metadata",
            "frozen_dinov2_vits14_weights",
            "qwen3.8-27b_box_annotations",
        ],
        "forbidden_inputs": [
            "simulator_object_pose",
            "depth_image",
            "private_simulator_state",
            "reward",
            "success_oracle",
            "manual_annotation",
        ],
        "matcher": dict(MATCHER),
        "calibration": dict(CALIBRATION),
        "capacity_gate": {
            "runtime_use": False,
            "unmeasurable_cannot_count": True,
            "required_capacity": (
                "max(reset_eef_displacement_m,calibrated_landmark_bound_m)+0.010m"
            ),
            "minimum_measured_and_passing_tasks": 6,
            "minimum_passing_families": 3,
            "failure_action": "kill_cycle9_without_live_scoring",
        },
        "postimplementation_dev20_gate": {
            "minimum_official_successes": 6,
            "minimum_successful_families": 3,
            "separate_from_capacity_gate": True,
        },
        "execution": {
            "one_measurement_probe_then_one_frozen_dev20_capacity_run": True,
            "zero_action_harness_invalid_restart_each": 1,
            "restart_requires_no_simulator_action": True,
            "no_threshold_relaxation_or_outcome_tuning": True,
        },
        "unchanged": {
            "runtime_agency_and_sparse_keyframes": True,
            "servo_and_safety_limits": True,
            "official_cameras": True,
            "no_fine_tuning": True,
        },
    }
    value["sha256"] = _digest(value)
    return value


def validate_capacity_amendment(value: Mapping[str, object]) -> None:
    amendment = dict(value)
    digest = amendment.pop("sha256", None)
    if not isinstance(digest, str) or digest != _digest(amendment):
        raise RuntimeError("Cycle 9 capacity amendment digest mismatch")
    if amendment.get("schema") != "robocasa-inspect-goal30-cycle9-capacity-amendment/v1":
        raise RuntimeError("Cycle 9 capacity amendment schema mismatch")
    if amendment.get("fable_verdict") != "ACCEPT":
        raise RuntimeError("Cycle 9 capacity amendment lacks Fable acceptance")
    matcher = amendment.get("matcher")
    calibration = amendment.get("calibration")
    capacity = amendment.get("capacity_gate")
    dev20 = amendment.get("postimplementation_dev20_gate")
    if not all(isinstance(row, Mapping) for row in (matcher, calibration, capacity, dev20)):
        raise TypeError("Cycle 9 capacity amendment is incomplete")
    if dict(matcher) != MATCHER or dict(calibration) != CALIBRATION:
        raise RuntimeError("Cycle 9 matcher or calibration thresholds drifted")
    if capacity.get("runtime_use") is not False:
        raise RuntimeError("Cycle 9 capacity evidence crossed into runtime")
    if capacity.get("minimum_measured_and_passing_tasks") != 6:
        raise RuntimeError("Cycle 9 capacity cardinality drifted")
    if dev20.get("minimum_official_successes") != 6:
        raise RuntimeError("Cycle 9 dev20 cardinality drifted")
