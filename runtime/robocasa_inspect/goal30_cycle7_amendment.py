"""Immutable Fable-reviewed amendment for the final Cycle 7 iteration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_cycle7_amendment(
    *,
    cycle7_contract_sha256: str,
    cycle7_iteration1_early_stop_sha256: str,
    implementation_parent: Mapping[str, str],
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "robocasa-inspect-goal30-cycle7-amendment-c/v1",
        "fable_verdict": "ACCEPT",
        "cycle7_contract_sha256": cycle7_contract_sha256,
        "cycle7_iteration1_early_stop_sha256": cycle7_iteration1_early_stop_sha256,
        "implementation_parent": dict(sorted(implementation_parent.items())),
        "pose_action": {
            "candidate_id": "source_plus_residual",
            "final_equals_source_plus_residual_tolerance": 1e-12,
            "residual_norm_min_m": 0.0,
            "residual_norm_max_m": 0.010,
            "residual_lateral_abs_cosine_max": 0.25,
            "final_translation_component_max_m": 0.020,
            "rotation_is_exact_source": True,
            "qwen_authors_residual_and_gripper_every_macro": True,
            "source_half_and_custom_pose_candidates": False,
        },
        "semantic_action": {
            "candidate_id": "custom",
            "existing_component_caps": True,
        },
        "counted_success": {
            "requires_material_qwen_macro": True,
            "material_residual_norm_min_m": 0.003,
            "different_gripper_counts": True,
            "otherwise_taxonomy": "harness_only_success",
        },
        "stall_governor": {
            "consecutive_stalled_macros_abort": 3,
            "stalled_translation_norm_lt_m": 0.00025,
            "stalled_rotation_lt_rad": 0.002,
            "nonstalled_macro_resets_counter": True,
            "zero_additional_macros_after_activation": True,
            "taxonomy": "contact_stall",
            "activation_independent_of_substep_list_length": True,
            "required_r1_trace_regression": True,
        },
        "execution": {
            "fresh_live_schema_capacity_preflight": True,
            "only_remaining_cycle7_diagnostic_iteration": 2,
            "all_other_frozen_gates_unchanged": True,
        },
    }
    value["sha256"] = _digest(value)
    return value


def validate_cycle7_amendment(value: Mapping[str, object]) -> None:
    contract = dict(value)
    digest = contract.pop("sha256", None)
    if not isinstance(digest, str) or digest != _digest(contract):
        raise RuntimeError("Cycle 7 amendment digest mismatch")
    if contract.get("schema") != "robocasa-inspect-goal30-cycle7-amendment-c/v1":
        raise RuntimeError("Cycle 7 amendment schema mismatch")
    if contract.get("fable_verdict") != "ACCEPT":
        raise RuntimeError("Cycle 7 amendment Fable verdict drifted")
    pose = contract.get("pose_action")
    success = contract.get("counted_success")
    stall = contract.get("stall_governor")
    execution = contract.get("execution")
    if not all(isinstance(item, Mapping) for item in (pose, success, stall, execution)):
        raise RuntimeError("Cycle 7 amendment is incomplete")
    if pose.get("candidate_id") != "source_plus_residual":
        raise RuntimeError("Cycle 7 pose authorship drifted")
    if pose.get("residual_norm_min_m") != 0.0:
        raise RuntimeError("Cycle 7 residual floor drifted")
    if success.get("material_residual_norm_min_m") != 0.003:
        raise RuntimeError("Cycle 7 contribution gate drifted")
    if stall.get("consecutive_stalled_macros_abort") != 3:
        raise RuntimeError("Cycle 7 stall abort drifted")
    if stall.get("zero_additional_macros_after_activation") is not True:
        raise RuntimeError("Cycle 7 stall response drifted")
    if execution.get("only_remaining_cycle7_diagnostic_iteration") != 2:
        raise RuntimeError("Cycle 7 iteration budget drifted")
