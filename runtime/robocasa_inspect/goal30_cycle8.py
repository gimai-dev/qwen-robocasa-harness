"""Fable-accepted final transport-reliability cycle for the 30% goal."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def wilson_upper_bound(failures: int, total: int, *, z: float = 1.96) -> float:
    """Return the two-sided Wilson interval's upper endpoint."""
    if total <= 0 or failures < 0 or failures > total:
        raise ValueError("Wilson counts are invalid")
    proportion = failures / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = proportion + z_squared / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z_squared / (4.0 * total * total)
    )
    return (center + radius) / denominator


def build_cycle8_contract(
    *,
    cycle7_d1_sha256: str,
    cycle7_failure_sha256: str,
    fable_review_sha256: str,
    implementation_parent: Mapping[str, str],
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "robocasa-inspect-goal30-cycle8/v1",
        "fable_verdict": "ACCEPT",
        "parents": {
            "cycle7_d1_sha256": cycle7_d1_sha256,
            "cycle7_failure_sha256": cycle7_failure_sha256,
            "fable_review_sha256": fable_review_sha256,
            "implementation_parent": dict(sorted(implementation_parent.items())),
        },
        "purpose": {
            "establishes_30_percent": False,
            "establishes_transport_reliability": True,
            "then_produces_first_clean_behavioral_verdict": True,
            "last_infrastructure_only_cycle": True,
        },
        "attempt_evidence": {
            "owner_only_append_only_jsonl": True,
            "fsync_each_http_response_before_command_parse": True,
            "fields": [
                "request_sha256",
                "observation_id",
                "attempt_index",
                "raw_body_sha256",
                "sanitized_raw_command",
                "http_status",
                "finish_reason",
                "completion_tokens",
                "latency_s",
                "response_schema_sha256",
                "served_model_id",
            ],
            "credentials_forbidden": True,
            "sigkill_survival_required": True,
        },
        "transport_repair": {
            "maximum_total_attempts": 3,
            "maximum_retries": 2,
            "same_observation_schema_images_and_public_state": True,
            "simulator_actions_between_attempts": 0,
            "retry_suffix": "previous response was not schema-valid; re-emit",
            "content_hint_forbidden": True,
            "attempts_count_model_and_wall_budgets": True,
            "exhaustion_taxonomy": "transport_failure",
            "finish_reason_length_is_separate_class": True,
        },
        "output_bound": {
            "note_max_length": 120,
            "max_tokens_formula": "ceil(1.5 * maximum_legal_branch_token_count)",
            "served_tokenizer_snapshot_frozen": True,
            "maximum_legal_branch_census_required": True,
        },
        "reliability_preflight": {
            "fresh_attested_server": True,
            "calls": 400,
            "allowed_decision_losses": 0,
            "allowed_finish_reason_length": 0,
            "wilson_z": 1.96,
            "wilson_upper_max": 0.01,
            "zero_of_400_upper": wilson_upper_bound(0, 400),
            "one_of_400_upper": wilson_upper_bound(1, 400),
            "minimum_model_calls": 203,
            "capacity_formula": "floor(900 / p95_per_attempt_latency_s)",
            "corpus": {
                "sealed_public_observations_only": True,
                "sources": ["cycle5", "cycle6", "cycle7"],
                "official_three_camera_bytes": True,
                "public_state_only": True,
                "hidden_state_forbidden": True,
                "reserve_tasks_forbidden": True,
                "minimum_pose_entries": 200,
                "minimum_semantic_entries": 100,
                "minimum_each_residual_branch": 10,
                "minimum_each_custom_terminal_mode": 25,
            },
        },
        "diagnostic": {
            "maximum_iterations": 2,
            "maximum_free_zero_action_harness_invalid_starts": 1,
            "second_harness_invalid_start_ends_cycle": True,
            "one_or_more_actions_consumes_iteration": True,
            "transport_failure_after_action_consumes_iteration": True,
            "six_case_gate": {
                "both_regressions_required": True,
                "minimum_conversions": 2,
                "non_fridge_conversion_required": True,
                "semantic_composite_terminal_required": True,
            },
        },
        "early_stop": {
            "six_case": "stop_on_any_regression_failure_or_third_conversion_failure",
            "dev20": "stop_on_15th_failure_for_minimum_6_of_20",
            "family10": "stop_on_8th_failure_for_minimum_3_of_10",
            "matrix100": "stop_on_71st_failure_for_minimum_30_of_100",
            "skip_seed11_if_seed7_gate_unreachable": True,
        },
        "behavioral_kill": {
            "trigger": "two_transport_clean_six_case_iterations_fail_gate",
            "per_macro_qwen_servo_then_not_viable": True,
            "dev20_family10_and_matrices_forbidden_after_trigger": True,
            "next_step": "new_fable_reviewed_sparse_keyframe_cycle",
            "silent_role_drift_forbidden": True,
        },
        "unchanged": {
            "qwen_model": "Qwen3.8-27B",
            "fine_tuning": False,
            "official_cameras": ["left", "right", "wrist"],
            "direct_eef_via_reviewed_bank": True,
            "official_predicate_terminal_only": True,
            "horizons_and_feasibility_rules": True,
            "stall_governor": True,
            "reserve_access": False,
            "final_seed7_minimum_successes": 30,
            "final_seed11_minimum_successes": 30,
        },
    }
    value["sha256"] = _digest(value)
    return value


def validate_cycle8_contract(value: Mapping[str, object]) -> None:
    contract = dict(value)
    digest = contract.pop("sha256", None)
    if not isinstance(digest, str) or digest != _digest(contract):
        raise RuntimeError("Cycle 8 digest mismatch")
    if contract.get("schema") != "robocasa-inspect-goal30-cycle8/v1":
        raise RuntimeError("Cycle 8 schema mismatch")
    if contract.get("fable_verdict") != "ACCEPT":
        raise RuntimeError("Cycle 8 Fable verdict drifted")
    purpose = contract.get("purpose")
    repair = contract.get("transport_repair")
    preflight = contract.get("reliability_preflight")
    diagnostic = contract.get("diagnostic")
    kill = contract.get("behavioral_kill")
    if not all(
        isinstance(item, Mapping)
        for item in (purpose, repair, preflight, diagnostic, kill)
    ):
        raise RuntimeError("Cycle 8 contract is incomplete")
    if purpose.get("last_infrastructure_only_cycle") is not True:
        raise RuntimeError("Cycle 8 purpose drifted")
    if repair.get("maximum_total_attempts") != 3:
        raise RuntimeError("Cycle 8 retry budget drifted")
    if preflight.get("calls") != 400 or preflight.get("allowed_decision_losses") != 0:
        raise RuntimeError("Cycle 8 reliability census drifted")
    if not float(preflight.get("zero_of_400_upper", 1.0)) < 0.01:
        raise RuntimeError("Cycle 8 Wilson zero-failure gate drifted")
    if not float(preflight.get("one_of_400_upper", 0.0)) > 0.01:
        raise RuntimeError("Cycle 8 Wilson one-failure rejection drifted")
    if diagnostic.get("maximum_iterations") != 2:
        raise RuntimeError("Cycle 8 diagnostic budget drifted")
    if kill.get("silent_role_drift_forbidden") is not True:
        raise RuntimeError("Cycle 8 behavioral kill drifted")


def cycle8_diagnostic_gate_unreachable(
    records: Sequence[Mapping[str, object]],
) -> bool:
    """Stop before spending calls once the immutable six-case gate cannot pass."""
    if any(
        row.get("family") == "regression" and row.get("success") is not True
        for row in records
    ):
        return True
    conversion_failures = sum(
        row.get("family") != "regression" and row.get("success") is not True
        for row in records
    )
    return conversion_failures >= 3
