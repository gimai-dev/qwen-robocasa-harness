"""Fable-reviewed const-bank repair for Cycle 7's final diagnostic."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_cycle7_amendment_d(
    *,
    amendment_c_sha256: str,
    original_capacity_preflight_sha256: str,
    harness_invalid_start_sha256: str,
    implementation_parent: Mapping[str, str],
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "robocasa-inspect-goal30-cycle7-amendment-d/v1",
        "fable_verdict": "ACCEPT",
        "amendment_c_sha256": amendment_c_sha256,
        "implementation_parent": dict(sorted(implementation_parent.items())),
        "harness_invalid_starts": [
            {
                "kind": "capacity_preflight",
                "sha256": original_capacity_preflight_sha256,
                "executed_actions": 0,
            },
            {
                "kind": "source_plus_residual_linear_binding",
                "sha256": harness_invalid_start_sha256,
                "executed_actions": 0,
            },
        ],
        "residual_bank": {
            "candidate_id": "source_plus_residual",
            "magnitudes_m": [0.003, 0.006, 0.009],
            "signs": [-1, 1],
            "directions": 2,
            "zero_option": True,
            "maximum_move_branches": 13,
            "deterministic_basis": (
                "normalize(tangent); choose least-aligned +X,+Y,+Z world axis; "
                "Gram-Schmidt first; cross(tangent,first) second; fixed +X,+Y "
                "when tangent norm is zero"
            ),
            "same_basis_for_same_source_tangent": True,
            "filter_final_translation_component_abs_gt_m": 0.020,
            "residual_and_final_are_schema_const": True,
            "source_rotation_is_schema_const": True,
            "numeric_binding_tolerance": 1e-12,
            "record_bank_hash_and_selected_values": True,
        },
        "preflight": {
            "fresh_calls": 10,
            "minimum_model_calls": 203,
            "derive_capacity_from_fresh_p95": True,
            "shrink_order_if_red": ["drop_0.009m_ring"],
        },
        "execution": {
            "harness_invalid_start_does_not_consume_iteration": True,
            "one_remaining_diagnostic_iteration": True,
            "all_other_amendment_c_and_frozen_gates_unchanged": True,
        },
    }
    value["sha256"] = _digest(value)
    return value


def validate_cycle7_amendment_d(value: Mapping[str, object]) -> None:
    contract = dict(value)
    digest = contract.pop("sha256", None)
    if not isinstance(digest, str) or digest != _digest(contract):
        raise RuntimeError("Cycle 7 amendment D digest mismatch")
    if contract.get("schema") != "robocasa-inspect-goal30-cycle7-amendment-d/v1":
        raise RuntimeError("Cycle 7 amendment D schema mismatch")
    if contract.get("fable_verdict") != "ACCEPT":
        raise RuntimeError("Cycle 7 amendment D Fable verdict drifted")
    bank = contract.get("residual_bank")
    preflight = contract.get("preflight")
    execution = contract.get("execution")
    invalid = contract.get("harness_invalid_starts")
    if not all(isinstance(item, Mapping) for item in (bank, preflight, execution)):
        raise RuntimeError("Cycle 7 amendment D is incomplete")
    if not isinstance(invalid, list) or len(invalid) != 2:
        raise RuntimeError("Cycle 7 harness-invalid audit drifted")
    if bank.get("magnitudes_m") != [0.003, 0.006, 0.009]:
        raise RuntimeError("Cycle 7 residual bank drifted")
    if bank.get("maximum_move_branches") != 13:
        raise RuntimeError("Cycle 7 schema bank size drifted")
    if preflight.get("minimum_model_calls") != 203:
        raise RuntimeError("Cycle 7 capacity gate drifted")
    if execution.get("one_remaining_diagnostic_iteration") is not True:
        raise RuntimeError("Cycle 7 diagnostic iteration drifted")
