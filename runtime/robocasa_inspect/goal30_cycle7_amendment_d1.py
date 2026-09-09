"""Fable-reviewed string-const representation for the Cycle 7 bank."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_cycle7_amendment_d1(
    *,
    amendment_d_sha256: str,
    numeric_const_failure_sha256: str,
    implementation_parent: Mapping[str, str],
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "robocasa-inspect-goal30-cycle7-amendment-d1/v1",
        "fable_verdict": "ACCEPT",
        "amendment_d_sha256": amendment_d_sha256,
        "numeric_const_failure_sha256": numeric_const_failure_sha256,
        "implementation_parent": dict(sorted(implementation_parent.items())),
        "pose_representation": {
            "fields": [
                "translation_m_f64_repr",
                "lateral_residual_m_f64_repr",
                "rotation_axis_angle_rad_f64_repr",
            ],
            "array_items": "repr(float)_string_const",
            "decode": "float(decimal_string)",
            "exact_float64_round_trip": True,
            "raw_and_decoded_values_hashed": True,
            "numeric_const_fields_removed_for_pose": True,
            "semantic_custom_remains_numeric": True,
            "fallback_if_repr_preflight_red": "float.hex_string_const",
        },
        "model_protocol": {
            "enable_thinking": False,
            "max_tokens": 1024,
            "server_enforced_closed_schema": True,
        },
        "preflight": {
            "pose_branches_each_exactly_once": 13,
            "semantic_custom_calls": 1,
            "give_up_calls": 1,
            "done_calls": 1,
            "total_calls": 16,
            "capacity_from_p95": "floor(900/p95)",
            "minimum_model_calls": 203,
        },
        "unchanged": {
            "physical_bank": True,
            "branch_selection_is_qwen_authorship": True,
            "binding_lateral_cap_and_contribution_checks": True,
            "remaining_diagnostic_iteration": 1,
        },
    }
    value["sha256"] = _digest(value)
    return value


def validate_cycle7_amendment_d1(value: Mapping[str, object]) -> None:
    contract = dict(value)
    digest = contract.pop("sha256", None)
    if not isinstance(digest, str) or digest != _digest(contract):
        raise RuntimeError("Cycle 7 amendment D1 digest mismatch")
    if contract.get("schema") != "robocasa-inspect-goal30-cycle7-amendment-d1/v1":
        raise RuntimeError("Cycle 7 amendment D1 schema mismatch")
    representation = contract.get("pose_representation")
    protocol = contract.get("model_protocol")
    preflight = contract.get("preflight")
    unchanged = contract.get("unchanged")
    if not all(
        isinstance(item, Mapping)
        for item in (representation, protocol, preflight, unchanged)
    ):
        raise RuntimeError("Cycle 7 amendment D1 is incomplete")
    if representation.get("array_items") != "repr(float)_string_const":
        raise RuntimeError("Cycle 7 D1 representation drifted")
    if representation.get("exact_float64_round_trip") is not True:
        raise RuntimeError("Cycle 7 D1 numeric authority drifted")
    if preflight.get("total_calls") != 16:
        raise RuntimeError("Cycle 7 D1 branch coverage drifted")
    if preflight.get("minimum_model_calls") != 203:
        raise RuntimeError("Cycle 7 D1 capacity drifted")
    if unchanged.get("remaining_diagnostic_iteration") != 1:
        raise RuntimeError("Cycle 7 D1 iteration budget drifted")
