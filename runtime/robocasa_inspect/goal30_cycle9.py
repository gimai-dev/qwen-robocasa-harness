"""Fable-accepted Cycle 9 sparse-keyframe public-servo contract."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

AGENCY_WORDING = (
    "Sparse-keyframe Qwen decisions (residual/gripper/phase) with public-error "
    "harness servo between keyframes; a counted success requires at least one "
    "residual of at least 3 mm or a gripper decision differing from source."
)


def derive_family10_slices(
    *,
    development_tasks: Sequence[str],
    dev20: Sequence[str],
    task_modules: Mapping[str, str],
) -> dict[str, list[str]]:
    """Derive three disjoint public-development slices without reserve tasks."""
    excluded = set(dev20)

    def family(task: str) -> str | None:
        lowered = task.lower()
        module = task_modules.get(task, "").lower()
        if any(
            token in lowered
            for token in (
                "turnon",
                "turnoff",
                "adjust",
                "preheat",
                "start",
                "boil",
                "defrost",
                "lowerheat",
                "turnsinkspout",
            )
        ):
            return "controls"
        if any(token in lowered for token in ("open", "close", "slide")) and (
            "pickplace" not in lowered
        ):
            return "articulated"
        if "pick_place" in module or any(
            token in lowered
            for token in ("pickplace", "arrange", "gather", "store", "transfer")
        ):
            return "grasp_place"
        return None

    result: dict[str, list[str]] = {}
    used = set(excluded)
    for name in ("articulated", "controls", "grasp_place"):
        candidates = [
            task
            for task in development_tasks
            if task not in used and family(task) == name
        ]
        ranked = sorted(
            candidates,
            key=lambda task: hashlib.sha256(
                f"robocasa-inspect-cycle9-family10/{name}/{task}".encode()
            ).hexdigest(),
        )
        if len(ranked) < 10:
            raise ValueError(f"Cycle 9 lacks ten public {name} tasks")
        result[name] = ranked[:10]
        used.update(result[name])
    return result


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_cycle9_contract(
    *,
    controller_amendment_sha256: str,
    diagnostic_r2_sha256: str,
    fable_review_sha256: str,
    cycle1_contract_sha256: str,
    goal_authority_sha256: str,
    preedit_evidence_sha256: str,
    dev20: Sequence[str],
    family10_slices: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "robocasa-inspect-goal30-cycle9/v1",
        "fable_verdict": "ACCEPT",
        "parents": {
            "controller_amendment_sha256": controller_amendment_sha256,
            "diagnostic_r2_sha256": diagnostic_r2_sha256,
            "fable_review_sha256": fable_review_sha256,
            "cycle1_contract_sha256": cycle1_contract_sha256,
            "goal_authority_sha256": goal_authority_sha256,
            "preedit_evidence_sha256": preedit_evidence_sha256,
        },
        "agency": {
            "wording": AGENCY_WORDING,
            "qwen_authors_every_eef_macro": False,
            "qwen_decides": ["residual_id", "gripper", "phase", "note"],
            "servo_may_only_reduce_error_to_authorized_endpoint": True,
        },
        "keyframes": {
            "monotone_sealed_same_task_public_eef_trajectory": True,
            "translation_spacing_m": 0.040,
            "rotation_spacing_rad": 0.20,
            "emit_on_any_gripper_transition": True,
            "emit_terminal_source_state": True,
            "never_skip_gripper_transition": True,
        },
        "residual_bank": {
            "basis": "deterministic_two_axis_lateral_to_source_segment_tangent",
            "cap": "min(0.025m,0.25*segment_translation_length_m)",
            "fractions": [0.0, 1 / 3, 2 / 3, 1.0],
            "both_signs_on_both_lateral_axes": True,
            "closed_const_bank": True,
            "near_zero_segment_cap_m": 0.0,
        },
        "capacity_preimplementation_gate": {
            "inputs": [
                "sealed_dev20_reset_official_three_camera_rgb",
                "sealed_public_reference_rgb",
                "public_camera_calibration",
                "public_eef_state",
            ],
            "runtime_use": False,
            "qwen_point_annotation_calls_per_state": 2,
            "qwen_point_annotation_calls_total": 4,
            "minimum_view_support": 2,
            "maximum_cross_ray_residual_m": 0.005,
            "maximum_reprojection_residual_px": 4.0,
            "maximum_independent_call_disagreement_m": 0.005,
            "positive_depth_required": True,
            "certified_workspace_required": True,
            "capacity_margin_m": 0.010,
            "minimum_measurable_passing_tasks": 6,
            "minimum_passing_families": 3,
            "failure_action": "kill_cycle9_without_live_scoring",
        },
        "servo": {
            "input": "current_public_eef_error_to_exact_qwen_authorized_endpoint",
            "maximum_authored_translation_per_tick_m": 0.004,
            "maximum_authored_rotation_per_tick_rad": 0.020,
            "translation_controller_gain": 3.5,
            "rotation_controller_gain": 3.5,
            "maximum_realized_translation_per_tick_m": 0.005,
            "maximum_realized_rotation_per_tick_rad": 0.025,
            "maximum_consecutive_public_stalls": 3,
            "fresh_official_three_camera_observation_after_servo": True,
            "no_path_planning_or_target_inference": True,
        },
        "evaluation": {
            "dev20": list(dev20),
            "family10_slices": {
                key: list(tasks) for key, tasks in sorted(family10_slices.items())
            },
            "transport_preflight_calls": 400,
            "smoke": ["OpenFridgeDrawer", "first_frozen_non_fridge_dev_task"],
            "smoke_is_non_scoring": True,
            "zero_action_harness_invalid_restarts_smoke": 1,
            "zero_action_harness_invalid_restarts_dev20": 1,
            "any_executed_action_consumes_run": True,
            "dev20_minimum_successes": 6,
            "dev20_minimum_successful_families": 3,
            "family10_slices_required": 3,
            "family10_tasks_each": 10,
            "family10_minimum_successes_each": 3,
            "no_tuning_between_slices": True,
            "matrix_seeds": [7, 11],
            "matrix_tasks_each": 100,
            "matrix_minimum_successes_each": 30,
            "red_dev_or_family_gate": "return_to_fable",
        },
        "evidence": {
            "owner_only": True,
            "complete_policy_and_servo_trace": True,
            "official_three_camera_hashes": True,
            "real_mp4_per_counted_task": True,
            "final_fable_audit": True,
        },
        "unchanged": {
            "model": "Qwen3.8-27B",
            "fine_tuning": False,
            "official_predicates": True,
            "official_camera_configuration": True,
            "task_split": True,
            "model_identity_and_safety": True,
        },
    }
    value["sha256"] = _digest(value)
    return value


def validate_cycle9_contract(value: Mapping[str, object]) -> None:
    contract = dict(value)
    digest = contract.pop("sha256", None)
    if not isinstance(digest, str) or digest != _digest(contract):
        raise RuntimeError("Cycle 9 contract digest mismatch")
    if contract.get("schema") != "robocasa-inspect-goal30-cycle9/v1":
        raise RuntimeError("Cycle 9 contract schema mismatch")
    if contract.get("fable_verdict") != "ACCEPT":
        raise RuntimeError("Cycle 9 was not accepted by Fable")
    agency = contract.get("agency")
    evaluation = contract.get("evaluation")
    capacity = contract.get("capacity_preimplementation_gate")
    if not all(isinstance(item, Mapping) for item in (agency, evaluation, capacity)):
        raise TypeError("Cycle 9 contract is incomplete")
    if agency.get("wording") != AGENCY_WORDING:
        raise RuntimeError("Cycle 9 agency wording drifted")
    if agency.get("qwen_authors_every_eef_macro") is not False:
        raise RuntimeError("Cycle 9 agency boundary drifted")
    if len(evaluation.get("dev20", [])) != 20:
        raise RuntimeError("Cycle 9 dev20 drifted")
    slices = evaluation.get("family10_slices")
    if not isinstance(slices, Mapping) or len(slices) != 3:
        raise RuntimeError("Cycle 9 family10 slices drifted")
    if any(len(tasks) != 10 for tasks in slices.values()):
        raise RuntimeError("Cycle 9 family10 cardinality drifted")
    flattened = [task for tasks in slices.values() for task in tasks]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("Cycle 9 family10 slices overlap")
    if set(flattened) & set(evaluation.get("dev20", [])):
        raise RuntimeError("Cycle 9 family10 slices overlap dev20")
    if evaluation.get("matrix_minimum_successes_each") != 30:
        raise RuntimeError("Cycle 9 matrix goal drifted")
    if capacity.get("runtime_use") is not False:
        raise RuntimeError("Cycle 9 evidence-only capacity boundary drifted")
