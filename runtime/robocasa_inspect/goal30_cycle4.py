"""Immutable Fable-reviewed authority for the fourth 30%-goal cycle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_cycle4_contract(
    *,
    cycle3_contract_sha256: str,
    preedit_evidence_sha256: str,
    full_catalog_sha256: str,
    cycle3_results: Mapping[str, str],
    implementation_parent: Mapping[str, str],
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "robocasa-inspect-goal30-cycle4/v1",
        "cycle3_contract_sha256": cycle3_contract_sha256,
        "preedit_evidence_sha256": preedit_evidence_sha256,
        "full_catalog_sha256": full_catalog_sha256,
        "cycle3_results": dict(sorted(cycle3_results.items())),
        "implementation_parent": dict(sorted(implementation_parent.items())),
        "candidate_policy": {
            "exact_candidates": [
                "source_full",
                "source_half",
                "translation_only",
                "rotation_only",
            ],
            "visual_lateral_candidate": "visual_lateral",
            "translation_cap_m": 0.010,
            "rotation_cap_rad": 0.050,
            "lateral_residual_cap_m": 0.010,
            "lateral_cosine_abs_max": 0.25,
            "source_translation_floor_m": 0.002,
            "final_component_cap_m": 0.020,
            "automatic_execution": False,
            "source_actions_exposed_or_used": False,
        },
        "freshness": {
            "token_hex_chars": 12,
            "derivation": "sha256(full_observation_id+cycle4_contract_sha256)[:12]",
            "unique_per_episode": True,
            "one_inflight_synchronous_request": True,
            "syntax_retries_per_observation": 1,
            "harness_substitution": False,
        },
        "cursor": {
            "monotone": True,
            "advance_authority": "fresh_public_eef_pose_progress",
            "visual_metrics": "audit_and_stall_veto_only",
            "success_authority": "official_simulator_predicate_only",
            "translation_error_max_m": 0.120,
            "rotation_error_max_rad": 0.800,
            "minimum_progress_m": 0.0005,
        },
        "contact": {
            "registry_authority": "full_official_catalog",
            "runtime_contact_state_exposed": False,
            "phase_sources": ["published_task_affordance", "source_cursor", "binary_gripper"],
            "expected_contact_is_not_success": True,
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


def validate_cycle4_contract(contract: Mapping[str, object]) -> None:
    value = dict(contract)
    digest = value.pop("sha256", None)
    if not isinstance(digest, str) or digest != _digest(value):
        raise RuntimeError("cycle4 contract digest mismatch")
    if value.get("schema") != "robocasa-inspect-goal30-cycle4/v1":
        raise RuntimeError("cycle4 contract schema mismatch")
    candidates = value.get("candidate_policy")
    freshness = value.get("freshness")
    cursor = value.get("cursor")
    contact = value.get("contact")
    budgets = value.get("budgets")
    gates = value.get("gates")
    if not all(
        isinstance(item, Mapping)
        for item in (candidates, freshness, cursor, contact, budgets, gates)
    ):
        raise RuntimeError("cycle4 contract is incomplete")
    if candidates.get("automatic_execution") is not False:
        raise RuntimeError("cycle4 automatic execution drift")
    if candidates.get("source_actions_exposed_or_used") is not False:
        raise RuntimeError("cycle4 source action boundary drift")
    if candidates.get("lateral_residual_cap_m") != 0.010:
        raise RuntimeError("cycle4 lateral bound drift")
    if freshness.get("harness_substitution") is not False:
        raise RuntimeError("cycle4 freshness substitution drift")
    if cursor.get("success_authority") != "official_simulator_predicate_only":
        raise RuntimeError("cycle4 success authority drift")
    if contact.get("runtime_contact_state_exposed") is not False:
        raise RuntimeError("cycle4 contact-state boundary drift")
    if budgets.get("model_calls") != 64 or gates.get("matrix_successes_each_seed") != 30:
        raise RuntimeError("cycle4 budget or goal drift")
