"""Immutable Fable-reviewed authority for the fifth 30%-goal cycle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_cycle5_contract(
    *,
    cycle4_contract_sha256: str,
    cycle4_diagnostic_sha256: str,
    preedit_evidence_sha256: str,
    implementation_parent: Mapping[str, str],
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "robocasa-inspect-goal30-cycle5/v1",
        "cycle4_contract_sha256": cycle4_contract_sha256,
        "cycle4_diagnostic_sha256": cycle4_diagnostic_sha256,
        "preedit_evidence_sha256": preedit_evidence_sha256,
        "implementation_parent": dict(sorted(implementation_parent.items())),
        "model_protocol": {
            "server_enforced_json_schema": True,
            "branches": ["move", "give_up", "done"],
            "syntax_repair": False,
            "schema_failure_is_harness_bug": True,
            "one_inflight_synchronous_request": True,
            "fresh_10_call_latency_preflight": True,
            "model_call_wall_budget_s": 900,
            "minimum_model_calls": 203,
            "max_calls_formula": "floor(900/fresh_schema_latency_p95_s)",
        },
        "macro": {
            "qwen_authored_numeric_eef": True,
            "translation_norm_max_m": 0.020,
            "rotation_norm_max_rad": 0.100,
            "controller_translation_norm_max_m": 0.005,
            "controller_rotation_norm_max_rad": 0.025,
            "public_eef_consecutive_stall_abort": 2,
            "reobserve_all_official_cameras_after_macro": True,
            "hash_raw_macro_substeps_and_realized_delta": True,
            "automatic_source_execution": False,
        },
        "source_path": {
            "reserve_multiplier": 1.30,
            "substep_translation_m": 0.005,
            "eligibility_formula": (
                "ceil(1.30*ceil(public_eef_path_length_m/0.005))<=official_horizon"
            ),
            "deterministic_next_rank_fallback": True,
            "ineligible_is_semantic_only": True,
            "frozen_macro_call_p95": 156,
        },
        "retrieval": {
            "catalog_count": 373,
            "catalog_metadata_only": True,
            "uses_frozen_matrix_identities": False,
            "manual_per_task_edits": False,
            "cross_task_pose_requires_reset_ssim_eef": True,
            "otherwise_semantic_only": True,
        },
        "skills": {
            "phases": [
                "approach",
                "contact",
                "actuate",
                "transport",
                "release",
                "verify",
            ],
            "family_level_only": True,
            "per_task_numeric_parameters": False,
            "fresh_qwen_macro_per_motion": True,
            "verify_image_only": True,
            "control_reads_task_predicate": False,
            "control_reads_reward_object_or_contact_state": False,
        },
        "qwen_contribution": {
            "required_for_counted_success": True,
            "translation_difference_min_m": 0.003,
            "rotation_difference_min_rad": 0.020,
            "different_gripper_counts": True,
        },
        "budgets": {
            "per_task_environment_steps": "pinned_official_horizon",
            "unregistered_semantic_task_steps": 450,
            "wall_s": 1200,
            "implementation_iterations": 2,
        },
        "gates": {
            "diagnostic_regression_successes": 2,
            "diagnostic_conversion_successes": 2,
            "diagnostic_requires_non_fridge_conversion": True,
            "diagnostic_requires_semantic_composite_terminal": True,
            "dev20_successes": 6,
            "dev20_successful_families": 3,
            "family10_successes_each": 3,
            "matrix_successes_each_seed": 30,
            "matrix_seeds": [7, 11],
        },
    }
    value["sha256"] = _digest(value)
    return value


def validate_cycle5_contract(contract: Mapping[str, object]) -> None:
    value = dict(contract)
    digest = value.pop("sha256", None)
    if not isinstance(digest, str) or digest != _digest(value):
        raise RuntimeError("Cycle 5 contract digest mismatch")
    if value.get("schema") != "robocasa-inspect-goal30-cycle5/v1":
        raise RuntimeError("Cycle 5 contract schema mismatch")
    protocol = value.get("model_protocol")
    macro = value.get("macro")
    source = value.get("source_path")
    retrieval = value.get("retrieval")
    skills = value.get("skills")
    contribution = value.get("qwen_contribution")
    budgets = value.get("budgets")
    gates = value.get("gates")
    if not all(
        isinstance(item, Mapping)
        for item in (
            protocol,
            macro,
            source,
            retrieval,
            skills,
            contribution,
            budgets,
            gates,
        )
    ):
        raise RuntimeError("Cycle 5 contract is incomplete")
    if protocol.get("branches") != ["move", "give_up", "done"]:
        raise RuntimeError("Cycle 5 response branches drifted")
    if protocol.get("syntax_repair") is not False:
        raise RuntimeError("Cycle 5 syntax repair drifted")
    if macro.get("translation_norm_max_m") != 0.020:
        raise RuntimeError("Cycle 5 macro translation bound drifted")
    if macro.get("controller_translation_norm_max_m") != 0.005:
        raise RuntimeError("Cycle 5 controller subdivision drifted")
    if source.get("reserve_multiplier") != 1.30:
        raise RuntimeError("Cycle 5 source reserve drifted")
    if retrieval.get("uses_frozen_matrix_identities") is not False:
        raise RuntimeError("Cycle 5 retrieval leaked matrix identities")
    if skills.get("control_reads_task_predicate") is not False:
        raise RuntimeError("Cycle 5 control predicate boundary drifted")
    if contribution.get("required_for_counted_success") is not True:
        raise RuntimeError("Cycle 5 Qwen contribution gate drifted")
    if budgets.get("implementation_iterations") != 2:
        raise RuntimeError("Cycle 5 implementation iteration drifted")
    if gates.get("diagnostic_regression_successes") != 2:
        raise RuntimeError("Cycle 5 regression gate was loosened")
    if gates.get("diagnostic_conversion_successes") != 2:
        raise RuntimeError("Cycle 5 conversion gate was loosened")
    if gates.get("matrix_successes_each_seed") != 30:
        raise RuntimeError("Cycle 5 final goal drifted")


def cycle5_diagnostic_passed(records: list[Mapping[str, object]]) -> bool:
    regressions = [row for row in records if row.get("family") == "regression"]
    conversions = [row for row in records if row.get("family") != "regression"]
    successful_conversions = [row for row in conversions if row.get("success") is True]
    semantic = next(
        (row for row in records if row.get("task") == "UtensilShuffle"), None
    )
    return bool(
        len(regressions) == 2
        and all(row.get("success") is True for row in regressions)
        and len(successful_conversions) >= 2
        and any("fridge" not in str(row.get("task", "")).lower() for row in successful_conversions)
        and semantic is not None
        and semantic.get("semantic_composite_terminal") is True
    )
