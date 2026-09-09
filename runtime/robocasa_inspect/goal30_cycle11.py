"""Fable-accepted Cycle 11 Qwen-authorized source-skill contract."""

from __future__ import annotations

from collections.abc import Mapping

from .cycle11_policy import CYCLE11_AGENCY_WORDING
from .cycle11_preflight import canonical_sha256


def build_cycle11_contract(
    *,
    cycle10_contract_sha256: str,
    controller_amendment_sha256: str,
    source_cohort_sha256: str,
    physical_smoke_sha256: str,
    goal_authority_sha256: str,
    fable_review_sha256: str,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "robocasa-inspect-goal30-cycle11/v1",
        "fable_verdict": "ACCEPT",
        "parents": {
            "cycle10_contract_sha256": cycle10_contract_sha256,
            "controller_amendment_sha256": controller_amendment_sha256,
            "source_cohort_sha256": source_cohort_sha256,
            "physical_smoke_sha256": physical_smoke_sha256,
            "goal_authority_sha256": goal_authority_sha256,
            "fable_review_sha256": fable_review_sha256,
        },
        "agency": {
            "wording": CYCLE11_AGENCY_WORDING,
            "qwen_role": "source_skill_authorization_and_veto",
            "skills": ["execute_source", "reobserve", "give_up"],
            "per_episode_liveness_not_competence": True,
            "liveness": {
                "minimum_execute_source": 3,
                "requires_before_source_boundary": True,
                "requires_at_or_after_source_boundary": True,
            },
            "report_nonzero_residual_fraction": True,
            "report_zero_nonzero_residual_success_fraction": True,
        },
        "semantic_preflight": {
            "frozen_cases": 500,
            "aligned": 200,
            "aligned_distinct_same_task_minimum": 100,
            "valid_next": 200,
            "invalid_cross_task_or_layout": 100,
            "calls": 900,
            "unmasked_aligned_reobserve_minimum": 0.95,
            "unmasked_valid_execute_minimum": 0.95,
            "masked_aligned_reobserve_minimum": 0.80,
            "masked_distinct_aligned_reobserve_minimum": 0.80,
            "masked_valid_execute_minimum": 0.80,
            "invalid_veto_minimum": 0.90,
            "invalid_execute_maximum": 0.10,
            "masked_inputs": "exact_three_images_task_wording_receipts_only",
            "forbidden_masked_fields": ["eef_error", "source_keyframe_index"],
            "invalid_public_eef_deliberately_execute_range": True,
            "zero_simulator_actions": True,
        },
        "runtime": {
            "execute_source_is_only_motion_authority": True,
            "reobserve_is_zero_motion": True,
            "maximum_reobserves_per_keyframe": 2,
            "same_frame_semantic_repair_after_limit": True,
            "global_repair_limit": 32,
            "cycle10_controller_and_horizon_unchanged": True,
        },
        "gates": {
            "smokes": ["OpenFridgeDrawer", "TurnOnToasterOven"],
            "dev20_minimum_successes": 6,
            "dev20_minimum_families": 3,
            "family10_minimum_each": 3,
            "matrix_seeds": [7, 11],
            "matrix_tasks_each": 100,
            "matrix_minimum_successes_each": 30,
            "red_action": "return_to_fable_before_plan_change",
        },
        "unchanged": {
            "model": "Qwen3.8-27B",
            "fine_tuning": False,
            "official_three_cameras": True,
            "official_predicates": True,
            "source_controller": True,
            "safety_thresholds": True,
            "held_out_identity_hidden": True,
        },
    }
    value["sha256"] = canonical_sha256(value)
    return value


def validate_cycle11_contract(value: Mapping[str, object]) -> None:
    body = dict(value)
    digest = body.pop("sha256", None)
    if digest != canonical_sha256(body):
        raise RuntimeError("Cycle 11 contract digest mismatch")
    if body.get("schema") != "robocasa-inspect-goal30-cycle11/v1":
        raise RuntimeError("Cycle 11 contract schema mismatch")
    if body.get("fable_verdict") != "ACCEPT":
        raise RuntimeError("Cycle 11 lacks Fable acceptance")
    agency = body.get("agency")
    preflight = body.get("semantic_preflight")
    gates = body.get("gates")
    if not all(isinstance(row, Mapping) for row in (agency, preflight, gates)):
        raise RuntimeError("Cycle 11 contract is incomplete")
    if agency.get("wording") != CYCLE11_AGENCY_WORDING:
        raise RuntimeError("Cycle 11 agency wording drifted")
    if preflight.get("calls") != 900:
        raise RuntimeError("Cycle 11 semantic call count drifted")
    if preflight.get("aligned_distinct_same_task_minimum") != 100:
        raise RuntimeError("Cycle 11 Fable distinct-frame condition drifted")
    if gates.get("matrix_minimum_successes_each") != 30:
        raise RuntimeError("Cycle 11 goal drifted")
