"""Immutable Fable-reviewed authority for the seventh 30%-goal cycle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

TRANSLATION_COMPONENT_MAX_M = 0.020
ROTATION_COMPONENT_MAX_RAD = 0.100
CONTROLLER_TRANSLATION_NORM_MAX_M = 0.005
CONTROLLER_ROTATION_NORM_MAX_RAD = 0.025
CANDIDATE_IDS = ("source_full", "source_half", "custom")


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_cycle7_contract(
    *,
    cycle6_contract_sha256: str,
    cycle6_early_stop_sha256: str,
    implementation_parent: Mapping[str, str],
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "robocasa-inspect-goal30-cycle7/v1",
        "fable_verdict": "ACCEPT",
        "cycle6_contract_sha256": cycle6_contract_sha256,
        "cycle6_early_stop_sha256": cycle6_early_stop_sha256,
        "implementation_parent": dict(sorted(implementation_parent.items())),
        "direct_eef": {
            "macro_semantics": "linfinity_components",
            "translation_component_max_m": TRANSLATION_COMPONENT_MAX_M,
            "rotation_component_max_rad": ROTATION_COMPONENT_MAX_RAD,
            "controller_translation_norm_max_m": CONTROLLER_TRANSLATION_NORM_MAX_M,
            "controller_rotation_norm_max_rad": CONTROLLER_ROTATION_NORM_MAX_RAD,
            "subdivision_uses_authored_l2_norm": True,
            "clip_normalize_substitute_or_repair": False,
            "derived_sum_tolerance": 1e-12,
        },
        "candidates": {
            "ids": list(CANDIDATE_IDS),
            "source_translation_norm_max_m": 0.020,
            "source_rotation_norm_max_rad": 0.100,
            "source_id_numeric_tolerance": 1e-12,
            "public_image_correction_candidate": False,
            "analytic_feasibility_only": True,
            "analytic_formula": (
                "sum(max(ceil(segment_translation_l2/0.005),"
                "ceil(segment_rotation_l2/0.025),1))<=reserved_controller_steps"
            ),
            "simulated_source_replay": False,
        },
        "prompt_and_evidence": {
            "required_mean_progress_m_per_remaining_step": True,
            "candidate_norms_and_substeps": True,
            "candidate_id_and_value_hash": True,
            "authored_command_hash": True,
            "derived_substeps_hash": True,
            "realized_progress_per_step": True,
            "failure_taxonomy": [
                "cap_exhaustion",
                "source_half_selection",
                "custom_divergence",
                "contact_stall",
                "visual_terminal_error",
            ],
        },
        "qwen_contribution": {
            "required_for_counted_success": True,
            "translation_difference_min_m": 0.003,
            "rotation_difference_min_rad": 0.020,
            "different_gripper_counts": True,
            "plain_source_selection_counts": False,
        },
        "preserved_gates": {
            "diagnostic_regression_successes": 2,
            "diagnostic_conversion_successes": 2,
            "diagnostic_requires_non_fridge_conversion": True,
            "diagnostic_requires_semantic_composite_terminal": True,
            "diagnostic_iterations": 2,
            "dev20_successes": 6,
            "dev20_successful_families": 3,
            "family10_slices": 3,
            "family10_tasks_each": 10,
            "family10_successes_each": 3,
            "matrix_successes_each_seed": 30,
            "matrix_seeds": [7, 11],
            "matrix_identity_access_before_gates": False,
        },
        "preserved_boundaries": {
            "official_three_cameras": True,
            "image_only_verification": True,
            "task_predicate_during_control": False,
            "hidden_depth": False,
            "same_seed_source_replay": False,
        },
    }
    value["sha256"] = _digest(value)
    return value


def validate_cycle7_contract(contract: Mapping[str, object]) -> None:
    value = dict(contract)
    digest = value.pop("sha256", None)
    if not isinstance(digest, str) or digest != _digest(value):
        raise RuntimeError("Cycle 7 contract digest mismatch")
    if value.get("schema") != "robocasa-inspect-goal30-cycle7/v1":
        raise RuntimeError("Cycle 7 contract schema mismatch")
    if value.get("fable_verdict") != "ACCEPT":
        raise RuntimeError("Cycle 7 Fable verdict drifted")
    direct = value.get("direct_eef")
    candidates = value.get("candidates")
    gates = value.get("preserved_gates")
    boundaries = value.get("preserved_boundaries")
    if not all(isinstance(item, Mapping) for item in (direct, candidates, gates, boundaries)):
        raise RuntimeError("Cycle 7 contract is incomplete")
    if direct.get("macro_semantics") != "linfinity_components":
        raise RuntimeError("Cycle 7 macro semantics drifted")
    if direct.get("clip_normalize_substitute_or_repair") is not False:
        raise RuntimeError("Cycle 7 mutation boundary drifted")
    if candidates.get("ids") != list(CANDIDATE_IDS):
        raise RuntimeError("Cycle 7 candidate vocabulary drifted")
    if candidates.get("analytic_feasibility_only") is not True:
        raise RuntimeError("Cycle 7 source feasibility boundary drifted")
    if gates.get("family10_slices") != 3 or gates.get("matrix_successes_each_seed") != 30:
        raise RuntimeError("Cycle 7 success gates drifted")
    if boundaries.get("hidden_depth") is not False:
        raise RuntimeError("Cycle 7 camera boundary drifted")
