"""Fable-accepted Cycle 10 persistent-residual Inspect policy contract."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

AGENCY_WORDING = (
    "Sparse-keyframe Qwen decisions choose a closed residual skill, gripper state, "
    "and phase from fresh official RGB; the harness only servos to the exact source "
    "keyframe plus the Qwen-authorized accumulated residual. A counted success "
    "requires at least one Qwen residual of at least 3 mm or at least 25 mrad, or a "
    "gripper decision differing from source. Qwen does not author every low-level "
    "EEF tick."
)

TRANSLATION_RESIDUALS = tuple(
    f"t{axis}{sign}{millimeters:02d}"
    for axis in "xyz"
    for sign in ("+", "-")
    for millimeters in (5, 10, 20)
)
ROTATION_RESIDUALS = tuple(
    f"r{axis}{sign}{milliradians:03d}"
    for axis in "xyz"
    for sign in ("+", "-")
    for milliradians in (25, 50)
)
RESIDUAL_IDS = ("zero", *TRANSLATION_RESIDUALS, *ROTATION_RESIDUALS)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_cycle10_contract(
    *,
    cycle9_sha256: str,
    cycle9_calibration_result_sha256: str,
    fable_review_sha256: str,
    goal_authority_sha256: str,
    dev20: list[str],
    family10_slices: Mapping[str, list[str]],
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "robocasa-inspect-goal30-cycle10/v1",
        "fable_verdict": "ACCEPT",
        "parents": {
            "cycle9_sha256": cycle9_sha256,
            "cycle9_calibration_result_sha256": cycle9_calibration_result_sha256,
            "fable_review_sha256": fable_review_sha256,
            "goal_authority_sha256": goal_authority_sha256,
        },
        "cycle9_disposition": {
            "killed_before_live_scoring": True,
            "calibration_candidate_pairs": 30,
            "calibration_usable_pairs": 0,
            "thresholds_changed": False,
        },
        "agency": {
            "wording": AGENCY_WORDING,
            "qwen_authors_every_eef_tick": False,
            "qwen_decides": ["residual_id", "gripper", "phase", "note"],
            "residual_ids": list(RESIDUAL_IDS),
            "gripper_values": ["source", "open", "close"],
            "phase_values": [
                "align",
                "grasp",
                "manipulate",
                "release",
                "advance",
                "give_up",
            ],
        },
        "observation": {
            "images": [
                "official_left_current_source_reference_panel",
                "official_right_current_source_reference_panel",
                "unmodified_current_official_wrist_rgb",
            ],
            "panel_order": "CURRENT_LEFT_HALF|SOURCE_REFERENCE_RIGHT_HALF",
            "public_state": [
                "base_frame_axis_definition",
                "nominal_public_eef_error",
                "persistent_pre_and_proposed_post_residual",
                "source_keyframe_index",
                "remaining_action_model_wall_budgets",
                "compact_receipts",
            ],
            "forbidden": [
                "reward",
                "success",
                "private_simulator_state",
                "depth",
                "object_pose",
                "future_source_frame",
            ],
        },
        "source_route": {
            "translation_keyframe_spacing_m": 0.040,
            "rotation_keyframe_spacing_rad": 0.20,
            "every_gripper_transition": True,
            "terminal_keyframe": True,
            "monotone_same_task_source": True,
            "maximum_keyframes": 60,
        },
        "persistent_residual": {
            "applied_to_every_subsequent_source_keyframe": True,
            "cleared_only_on_task_reset": True,
            "maximum_translation_norm_m": 0.050,
            "maximum_rotation_norm_rad": 0.20,
            "maximum_decisions_per_keyframe": 4,
            "record_pre_post_offset_per_observation": True,
            "harness_may_not_invent_or_modify_increment": True,
        },
        "retrieval": {
            "same_task_public_only": True,
            "stationary_base": True,
            "maximum_initial_public_eef_translation_error_m": 0.030,
            "maximum_initial_public_eef_rotation_error_rad": 0.15,
            "existing_external_rgb_similarity_gate": True,
        },
        "servo": {
            "maximum_authored_translation_per_tick_m": 0.004,
            "maximum_authored_rotation_per_tick_rad": 0.020,
            "translation_controller_gain": 3.5,
            "rotation_controller_gain": 3.5,
            "maximum_realized_translation_per_tick_m": 0.005,
            "maximum_realized_rotation_per_tick_rad": 0.025,
            "maximum_consecutive_public_stalls": 3,
            "advance_translation_error_m": 0.005,
            "advance_rotation_error_rad": 0.025,
            "fresh_three_camera_observation_after_every_decision": True,
        },
        "budgets": {
            "environment_steps": "pinned_official_task_horizon",
            "nominal_servo_tick_reserve_multiplier": 1.30,
            "residual_rejected_if_predicted_ticks_exceed_remaining_horizon": True,
            "fresh_schema_preflight_calls": 400,
            "max_model_calls_per_task": 400,
            "repair_attempt_allowance_per_task": 32,
            "call_eligibility": "keyframes*4+32<=400",
            "wall_multiplier_from_preflight_p99": 4.0,
            "wall_minimum_s": 300.0,
            "wall_hard_cap_s": 3600.0,
            "numeric_task_wall_budget_frozen_after_preflight": True,
            "ineligible_action": "give_up_before_qwen_or_simulator_action",
        },
        "source_cohort_gate": {
            "minimum_eligible_tasks": 20,
            "minimum_families": 4,
            "public_inputs_only": True,
            "no_simulator_actions_or_task_outcomes": True,
            "failure_action": "return_to_fable_before_runtime_implementation",
        },
        "capacity_honesty": {
            "eef_residual_range_covers_retrieval_eef_mismatch": True,
            "object_displacement_capacity": "untested",
            "sole_empirical_object_capability_evidence": (
                "one_frozen_dev20_official_run_at_6_of_20_across_3_families"
            ),
        },
        "evaluation": {
            "smoke": ["OpenFridgeDrawer", "first_frozen_non_fridge_dev_task"],
            "smoke_is_non_scoring": True,
            "zero_action_harness_invalid_restart_smoke": 1,
            "any_action_consumes_run": True,
            "dev20": list(dev20),
            "dev20_minimum_successes": 6,
            "dev20_minimum_successful_families": 3,
            "family10_slices": {
                key: list(tasks) for key, tasks in sorted(family10_slices.items())
            },
            "family10_minimum_successes_each": 3,
            "no_tuning_between_family_slices": True,
            "matrix_seeds": [7, 11],
            "matrix_tasks_each": 100,
            "matrix_minimum_successes_each": 30,
            "no_matrix_tuning": True,
            "red_smoke_dev_or_family_gate": "return_to_fable",
        },
        "evidence": {
            "owner_only": True,
            "full_qwen_decisions_and_receipts": True,
            "per_servo_trace": True,
            "current_and_reference_image_hashes": True,
            "official_predicate_outcome": True,
            "real_three_camera_mp4_per_task": True,
            "exact_source_model_runtime_authorities": True,
            "final_fable_audit": True,
        },
        "unchanged": {
            "model": "Qwen3.8-27B",
            "fine_tuning": False,
            "official_cameras": True,
            "official_predicates": True,
            "model_identity_and_safety": True,
        },
    }
    value["sha256"] = _digest(value)
    return value


def validate_cycle10_contract(value: Mapping[str, object]) -> None:
    contract = dict(value)
    digest = contract.pop("sha256", None)
    if not isinstance(digest, str) or digest != _digest(contract):
        raise RuntimeError("Cycle 10 contract digest mismatch")
    if contract.get("schema") != "robocasa-inspect-goal30-cycle10/v1":
        raise RuntimeError("Cycle 10 contract schema mismatch")
    if contract.get("fable_verdict") != "ACCEPT":
        raise RuntimeError("Cycle 10 lacks Fable acceptance")
    agency = contract.get("agency")
    residual = contract.get("persistent_residual")
    budgets = contract.get("budgets")
    honesty = contract.get("capacity_honesty")
    evaluation = contract.get("evaluation")
    if not all(
        isinstance(row, Mapping)
        for row in (agency, residual, budgets, honesty, evaluation)
    ):
        raise TypeError("Cycle 10 contract is incomplete")
    if agency.get("wording") != AGENCY_WORDING:
        raise RuntimeError("Cycle 10 agency wording drifted")
    if tuple(agency.get("residual_ids", ())) != RESIDUAL_IDS:
        raise RuntimeError("Cycle 10 residual vocabulary drifted")
    if residual.get("applied_to_every_subsequent_source_keyframe") is not True:
        raise RuntimeError("Cycle 10 residual carry-over drifted")
    if budgets.get("environment_steps") != "pinned_official_task_horizon":
        raise RuntimeError("Cycle 10 official horizon authority drifted")
    if honesty.get("object_displacement_capacity") != "untested":
        raise RuntimeError("Cycle 10 object-capacity honesty drifted")
    if evaluation.get("matrix_minimum_successes_each") != 30:
        raise RuntimeError("Cycle 10 goal drifted")
