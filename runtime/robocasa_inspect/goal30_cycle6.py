"""Immutable Fable-reviewed authority for the sixth 30%-goal cycle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

TRANSLATION_COMPONENT_MAX_M = 0.0115
ROTATION_COMPONENT_MAX_RAD = 0.0577
TRANSLATION_NORM_MAX_M = 0.020
ROTATION_NORM_MAX_RAD = 0.100


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_cycle6_contract(
    *,
    cycle5_contract_sha256: str,
    cycle5_diagnostic_sha256: str,
    cycle5_schema_preflight_sha256: str,
    implementation_parent: Mapping[str, str],
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "robocasa-inspect-goal30-cycle6/v1",
        "fable_verdict": "ACCEPT",
        "cycle5_contract_sha256": cycle5_contract_sha256,
        "cycle5_diagnostic_sha256": cycle5_diagnostic_sha256,
        "cycle5_schema_preflight_sha256": cycle5_schema_preflight_sha256,
        "implementation_parent": dict(sorted(implementation_parent.items())),
        "direct_eef": {
            "encoding": "cartesian_translation_and_axis_angle",
            "translation_component_max_m": TRANSLATION_COMPONENT_MAX_M,
            "rotation_component_max_rad": ROTATION_COMPONENT_MAX_RAD,
            "translation_norm_max_m": TRANSLATION_NORM_MAX_M,
            "rotation_norm_max_rad": ROTATION_NORM_MAX_RAD,
            "controller_translation_norm_max_m": 0.005,
            "controller_rotation_norm_max_rad": 0.025,
            "clip_normalize_substitute_or_repair": False,
            "candidate_and_model_bounds_identical": True,
            "hash_authored_command": True,
            "hash_derived_substeps": True,
            "derived_sum_tolerance": 1e-12,
        },
        "live_preflight": {
            "calls": 10,
            "branches": ["move", "give_up", "done"],
            "exact_observation_token": True,
            "boundary_values": True,
            "schema_valid_required": 10,
            "max_calls": 439,
            "latency_p95_max_s": 2.046,
        },
        "preserved_cycle5": {
            "official_three_cameras": True,
            "image_only_verification": True,
            "source_path_reserve_multiplier": 1.30,
            "source_pose_automatic_execution": False,
            "qwen_material_contribution_required": True,
            "translation_difference_min_m": 0.003,
            "rotation_difference_min_rad": 0.020,
            "different_gripper_counts": True,
            "public_eef_consecutive_stall_abort": 2,
            "implementation_iterations": 2,
            "matrix_identity_access_before_gates": False,
        },
        "gates": {
            "diagnostic_regression_successes": 2,
            "diagnostic_conversion_successes": 2,
            "diagnostic_requires_non_fridge_conversion": True,
            "diagnostic_requires_semantic_composite_terminal": True,
            "dev20_successes": 6,
            "dev20_successful_families": 3,
            "family10_slices": 3,
            "family10_tasks_each": 10,
            "family10_successes_each": 3,
            "family10_disjoint_and_prefrozen": True,
            "matrix_successes_each_seed": 30,
            "matrix_seeds": [7, 11],
        },
        "sequence": [
            "freeze_contract",
            "red_tests_and_implementation",
            "ten_call_live_preflight",
            "six_case_diagnostic",
            "dev20",
            "three_family10_slices",
            "seed7_matrix",
            "seed11_matrix",
            "audit_and_videos",
            "final_fable_review",
        ],
    }
    value["sha256"] = _digest(value)
    return value


def validate_cycle6_contract(contract: Mapping[str, object]) -> None:
    value = dict(contract)
    digest = value.pop("sha256", None)
    if not isinstance(digest, str) or digest != _digest(value):
        raise RuntimeError("Cycle 6 contract digest mismatch")
    if value.get("schema") != "robocasa-inspect-goal30-cycle6/v1":
        raise RuntimeError("Cycle 6 contract schema mismatch")
    if value.get("fable_verdict") != "ACCEPT":
        raise RuntimeError("Cycle 6 Fable verdict drifted")
    direct = value.get("direct_eef")
    preserved = value.get("preserved_cycle5")
    gates = value.get("gates")
    if not all(isinstance(item, Mapping) for item in (direct, preserved, gates)):
        raise RuntimeError("Cycle 6 contract is incomplete")
    if direct.get("translation_component_max_m") != TRANSLATION_COMPONENT_MAX_M:
        raise RuntimeError("Cycle 6 translation component bound drifted")
    if direct.get("rotation_component_max_rad") != ROTATION_COMPONENT_MAX_RAD:
        raise RuntimeError("Cycle 6 rotation component bound drifted")
    if direct.get("clip_normalize_substitute_or_repair") is not False:
        raise RuntimeError("Cycle 6 command mutation boundary drifted")
    if preserved.get("implementation_iterations") != 2:
        raise RuntimeError("Cycle 6 iteration bound drifted")
    if gates.get("family10_slices") != 3 or gates.get("family10_successes_each") != 3:
        raise RuntimeError("Cycle 6 family10 gate drifted")
    if gates.get("matrix_successes_each_seed") != 30:
        raise RuntimeError("Cycle 6 final goal drifted")
