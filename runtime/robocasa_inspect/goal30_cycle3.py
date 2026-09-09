"""Immutable Fable-reviewed authority for the third 30%-goal cycle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_cycle3_contract(
    *,
    cycle2_contract_sha256: str,
    cohort: Mapping[str, object],
    implementation_parent: Mapping[str, str],
) -> dict[str, object]:
    """Freeze the accepted source-pose soft-prior experiment before runtime edits."""
    value: dict[str, object] = {
        "schema": "robocasa-inspect-goal30-cycle3/v1",
        "cycle2_contract_sha256": cycle2_contract_sha256,
        "cohort": dict(cohort),
        "implementation_parent": dict(sorted(implementation_parent.items())),
        "source_pose_soft_prior": {
            "translation_gain": 1.0,
            "rotation_gain": 1.0,
            "translation_cap_m": 0.010,
            "rotation_cap_rad": 0.050,
            "translation_error_max_m": 0.120,
            "rotation_error_max_rad": 0.800,
            "phase_authority": "two_fresh_both_view_visual_plus_gripper_only",
            "candidate_may_not_advance_phase": True,
            "automatic_candidate_execution": False,
        },
        "repair": {
            "repairs_per_observation": 1,
            "translation_candidate_floor_m": 0.002,
            "translation_opposed_dot_m2": -1e-6,
            "rotation_candidate_floor_rad": 0.010,
            "rotation_opposed_dot_rad2": -1e-5,
            "orthogonal_avoidance_allowed": True,
            "repair_counts_against_model_budget": True,
            "second_give_up_or_opposed_action": "stop",
        },
        "counting": {
            "requires_qwen_authorized_nonzero_action": True,
            "fresh_observation_required": True,
            "pure_demo_tracking_counts": False,
        },
        "budgets": {
            "model_calls": 64,
            "environment_steps": 450,
            "wall_s": 1200,
            "terminal_hold_steps": 10,
            "implementation_iterations": 2,
        },
        "gates": {
            "diagnostic_regressions": 2,
            "diagnostic_conversions": 2,
            "diagnostic_requires_non_fridge": True,
            "dev20_successes": 6,
            "dev20_successful_families": 3,
            "family10_successes_each": 3,
            "matrix_successes_each_seed": 30,
        },
    }
    value["sha256"] = _digest(value)
    return value


def validate_cycle3_contract(contract: Mapping[str, object]) -> None:
    value = dict(contract)
    digest = value.pop("sha256", None)
    if not isinstance(digest, str) or digest != _digest(value):
        raise RuntimeError("cycle3 contract digest mismatch")
    if value.get("schema") != "robocasa-inspect-goal30-cycle3/v1":
        raise RuntimeError("cycle3 contract schema mismatch")
    prior = value.get("source_pose_soft_prior")
    repair = value.get("repair")
    counting = value.get("counting")
    budgets = value.get("budgets")
    gates = value.get("gates")
    if not all(
        isinstance(item, Mapping)
        for item in (prior, repair, counting, budgets, gates)
    ):
        raise RuntimeError("cycle3 contract is incomplete")
    if prior.get("phase_authority") != "two_fresh_both_view_visual_plus_gripper_only":
        raise RuntimeError("cycle3 visual phase authority drift")
    if prior.get("automatic_candidate_execution") is not False:
        raise RuntimeError("cycle3 candidate execution drift")
    if repair.get("repairs_per_observation") != 1:
        raise RuntimeError("cycle3 repair budget drift")
    if counting.get("requires_qwen_authorized_nonzero_action") is not True:
        raise RuntimeError("cycle3 Qwen authorship gate drift")
    if budgets.get("model_calls") != 64:
        raise RuntimeError("cycle3 model budget drift")
    if gates.get("matrix_successes_each_seed") != 30:
        raise RuntimeError("cycle3 goal gate drift")
