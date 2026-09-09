"""Fable-accepted compact pose protocol after Cycle 8 latency evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_amendment_e(
    *,
    cycle8_contract_sha256: str,
    reliability_r1_sha256: str,
    attempt_log_r1_sha256: str,
    fable_review_sha256: str,
    implementation_parent: Mapping[str, str],
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "robocasa-inspect-goal30-cycle8-amendment-e/v1",
        "fable_verdict": "ACCEPT",
        "parents": {
            "cycle8_contract_sha256": cycle8_contract_sha256,
            "reliability_r1_sha256": reliability_r1_sha256,
            "attempt_log_r1_sha256": attempt_log_r1_sha256,
            "fable_review_sha256": fable_review_sha256,
            "implementation_parent": dict(sorted(implementation_parent.items())),
        },
        "r1_evidence": {
            "calls": 400,
            "raw_malformed_attempts": 0,
            "finish_reason_length": 0,
            "decision_losses": 0,
            "wilson_upper": 0.009512640599680667,
            "latency_p95_s": 4.7353234063979475,
            "max_model_calls": 190,
            "failure_gate": "capacity_below_203_only",
        },
        "agency": {
            "qwen_authored": ["residual_id", "phase", "gripper", "note"],
            "harness_operation": "deterministic_public_hashed_bank_lookup",
            "harness_may_infer_optimize_reorder_or_modify": False,
            "arrays_removed_were_schema_const_echoes": True,
            "inspect_style_direct_eef_preserved": True,
        },
        "protocol": {
            "compact_pose_fields": [
                "kind",
                "observation_token",
                "phase",
                "candidate_id",
                "residual_id",
                "gripper",
                "note",
            ],
            "full_human_readable_bank_in_prompt": True,
            "bank_contents": [
                "residual_id",
                "lateral_offset_mm",
                "resulting_translation_m_f64_repr",
            ],
            "presented_bank_hash_per_observation": True,
            "constructed_command_hash_per_decision": True,
            "bit_identical_to_d1_all_13_branches_across_corpus": True,
        },
        "preflight": {
            "new_served_tokenizer_cap": True,
            "one_wholly_fresh_attested_server": True,
            "same_sealed_400_entry_corpus": True,
            "allowed_raw_malformed_attempts": 0,
            "allowed_finish_reason_length": 0,
            "allowed_decision_losses": 0,
            "wilson_upper_max": 0.01,
            "minimum_model_calls": 203,
            "capacity_formula": "floor(900 / p95_per_attempt_latency_s)",
        },
        "terminal": {
            "green": "proceed_to_maximum_two_six_case_diagnostics",
            "red": "return_to_fable",
            "further_transport_amendments_after_red": 0,
        },
        "unchanged": {
            "qwen_model": "Qwen3.8-27B",
            "fine_tuning": False,
            "official_three_cameras": True,
            "task_split_and_reserve": True,
            "safety_and_official_predicates": True,
            "simulator_horizons": True,
            "capacity_gate": True,
        },
    }
    value["sha256"] = _digest(value)
    return value


def validate_amendment_e(value: Mapping[str, object]) -> None:
    amendment = dict(value)
    digest = amendment.pop("sha256", None)
    if not isinstance(digest, str) or digest != _digest(amendment):
        raise RuntimeError("Cycle 8 Amendment E digest mismatch")
    if amendment.get("schema") != "robocasa-inspect-goal30-cycle8-amendment-e/v1":
        raise RuntimeError("Cycle 8 Amendment E schema mismatch")
    if amendment.get("fable_verdict") != "ACCEPT":
        raise RuntimeError("Cycle 8 Amendment E was not accepted")
    agency = amendment.get("agency")
    preflight = amendment.get("preflight")
    terminal = amendment.get("terminal")
    if not all(isinstance(item, Mapping) for item in (agency, preflight, terminal)):
        raise RuntimeError("Cycle 8 Amendment E is incomplete")
    if agency.get("harness_may_infer_optimize_reorder_or_modify") is not False:
        raise RuntimeError("Cycle 8 Amendment E agency boundary drifted")
    if preflight.get("minimum_model_calls") != 203:
        raise RuntimeError("Cycle 8 Amendment E capacity gate drifted")
    if terminal.get("further_transport_amendments_after_red") != 0:
        raise RuntimeError("Cycle 8 Amendment E terminal condition drifted")
