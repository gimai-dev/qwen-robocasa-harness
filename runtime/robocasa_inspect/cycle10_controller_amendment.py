"""Fable-accepted Cycle 10 servo-regime controller supersession."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

FABLE_CONDITIONS = (
    "carry_worst_live_realization_ratio_not_probe_maximum_alone",
    "skip_transport_repeat_only_because_schema_model_server_are_unchanged",
    "second_live_overshoot_reopens_controller_diagnosis_without_another_cap_cut",
    "source_only_success_is_harness_only_and_returns_to_fable",
    "second_smoke_is_first_frozen_nonfridge_dev20_task_passing_amended_source_gate",
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_controller_amendment(
    *,
    contract_sha256: str,
    preflight_authority_sha256: str,
    preflight_result_sha256: str,
    source_cohort_sha256: str,
    smoke_result_sha256: str,
    calibration_result_sha256: str,
    response_schema_sha256: str,
    model_identity_sha256: str,
    server_attestation_sha256: str,
    live_realization_ratio: float,
    selected_translation_cap_m: float,
    selected_rotation_cap_rad: float,
    current_implementation_sha256: Mapping[str, str],
) -> dict[str, object]:
    candidate_bound = selected_translation_cap_m * live_realization_ratio
    if (
        not 1.0 < live_realization_ratio < 2.0
        or selected_translation_cap_m != 0.003
        or selected_rotation_cap_rad != 0.020
        or candidate_bound > 0.0045 + 1e-12
    ):
        raise ValueError("Cycle 10 controller amendment calibration is invalid")
    value: dict[str, object] = {
        "schema": "robocasa-inspect-cycle10-controller-amendment/v1",
        "fable_model": "Fable 5",
        "fable_channel": "signed-in-claude-code-subscription",
        "fable_verdict": "ACCEPT",
        "fable_conditions": list(FABLE_CONDITIONS),
        "parents": {
            "cycle10_contract_sha256": contract_sha256,
            "preflight_authority_sha256": preflight_authority_sha256,
            "preflight_result_sha256": preflight_result_sha256,
            "source_cohort_sha256": source_cohort_sha256,
            "smoke_result_sha256": smoke_result_sha256,
            "calibration_result_sha256": calibration_result_sha256,
        },
        "transport_supersession": {
            "repeat_400_calls": False,
            "reason": "schema_model_and_server_unchanged",
            "response_schema_sha256": response_schema_sha256,
            "model_identity_sha256": model_identity_sha256,
            "server_attestation_sha256": server_attestation_sha256,
        },
        "calibration": {
            "cycle8_macro_regime_superseded_for_cycle10_servo_sizing": True,
            "unexplained_probe_live_regime_difference": True,
            "live_realization_ratio": live_realization_ratio,
            "selected_translation_cap_m": selected_translation_cap_m,
            "selected_rotation_cap_rad": selected_rotation_cap_rad,
            "realized_translation_guard_m": 0.005,
            "realized_rotation_guard_rad": 0.025,
            "margin_translation_m": 0.0045,
            "margin_rotation_rad": 0.0225,
            "controller_gains_unchanged": True,
            "stall_gate_unchanged": True,
        },
        "worst_ratio_candidate_realized_bound_m": candidate_bound,
        "source_cohort_required_before_runtime": True,
        "source_route_reserve_multiplier": 1.30,
        "smoke_selection_rule": (
            "first frozen dev20 non-fridge task in order passing amended public source gate"
        ),
        "harness_invalid_restart_consumed": False,
        "material_agency_rule_unchanged": True,
        "current_implementation_sha256": dict(
            sorted(current_implementation_sha256.items())
        ),
    }
    value["sha256"] = _digest(value)
    return value


def validate_controller_amendment(
    value: Mapping[str, object],
    *,
    contract_sha256: str,
    preflight_authority_sha256: str,
    preflight_result_sha256: str,
    current_implementation_sha256: Mapping[str, str],
) -> None:
    amendment = dict(value)
    digest = amendment.pop("sha256", None)
    if digest != _digest(amendment):
        raise RuntimeError("Cycle 10 controller amendment digest drifted")
    if amendment.get("schema") != "robocasa-inspect-cycle10-controller-amendment/v1":
        raise RuntimeError("Cycle 10 controller amendment schema drifted")
    if amendment.get("fable_verdict") != "ACCEPT":
        raise RuntimeError("Cycle 10 controller amendment lacks Fable acceptance")
    if tuple(amendment.get("fable_conditions", ())) != FABLE_CONDITIONS:
        raise RuntimeError("Cycle 10 controller amendment Fable conditions drifted")
    parents = amendment.get("parents")
    if not isinstance(parents, Mapping) or (
        parents.get("cycle10_contract_sha256") != contract_sha256
        or parents.get("preflight_authority_sha256") != preflight_authority_sha256
        or parents.get("preflight_result_sha256") != preflight_result_sha256
    ):
        raise RuntimeError("Cycle 10 controller amendment parent drifted")
    if amendment.get("current_implementation_sha256") != dict(
        sorted(current_implementation_sha256.items())
    ):
        raise RuntimeError("Cycle 10 controller amendment implementation drifted")
    calibration = amendment.get("calibration")
    if not isinstance(calibration, Mapping) or (
        calibration.get("selected_translation_cap_m") != 0.003
        or calibration.get("selected_rotation_cap_rad") != 0.020
        or calibration.get("realized_translation_guard_m") != 0.005
        or calibration.get("realized_rotation_guard_rad") != 0.025
    ):
        raise RuntimeError("Cycle 10 controller amendment cap drifted")
    if float(amendment.get("worst_ratio_candidate_realized_bound_m", 1.0)) > 0.0045:
        raise RuntimeError("Cycle 10 controller amendment lacks safety margin")
