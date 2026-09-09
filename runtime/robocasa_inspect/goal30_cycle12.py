"""Fable-accepted Cycle 12 honest source-authorization contract."""

from __future__ import annotations

from collections.abc import Mapping

from .cycle11_preflight import canonical_sha256

COUNTED_WORDING = (
    "Qwen's contribution is same-task source authorization and mismatch veto; "
    "counted successes are Qwen-authorized source-skill system successes. "
    "Low-level servo and alignment decisions are harness-owned. Runtime retrieval "
    "independently enforces same-task/layout; Qwen's veto is a redundant safety "
    "check. Absent a live veto, the trajectory is identical to a harness-only run."
)


def build_cycle12_contract(
    *,
    cycle11_contract_sha256: str,
    cycle11_corpus_sha256: str,
    cycle11_red_evidence: Mapping[str, object],
    fable_review_sha256: str,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "robocasa-inspect-goal30-cycle12/v1",
        "fable_verdict": "ACCEPT_WITH_MANDATORY_CONDITIONS",
        "parents": {
            "cycle11_contract_sha256": cycle11_contract_sha256,
            "cycle11_corpus_sha256": cycle11_corpus_sha256,
            "cycle11_red_evidence": dict(cycle11_red_evidence),
            "fable_review_sha256": fable_review_sha256,
        },
        "agency": {
            "wording": COUNTED_WORDING,
            "skills": ["authorize_source", "reobserve", "give_up"],
            "report_live_veto_count": True,
            "report_successes_with_qwen_trajectory_change": True,
            "trajectory_change_kinds": [
                "veto",
                "reobserve_delayed_motion",
                "nonzero_residual",
            ],
            "final_claim": "Qwen-authorized source-skill successes, not Qwen task control",
            "user_decides_if_original_objective_is_satisfied": True,
        },
        "preflight": {
            "same_frozen_cycle11_corpus": True,
            "calls": 900,
            "unmasked_aligned_authorize_minimum": 0.90,
            "unmasked_valid_authorize_minimum": 0.90,
            "masked_aligned_authorize_minimum": 0.90,
            "masked_valid_authorize_minimum": 0.90,
            "masked_distinct_aligned_authorize_minimum": 0.85,
            "invalid_veto_minimum": 0.90,
            "invalid_authorize_maximum": 0.10,
            "full_confusion_matrices": True,
            "schema_valid_required": 900,
            "transport_failures_required": 0,
            "zero_simulator_actions": True,
        },
        "runtime": {
            "only_authorize_source_may_progress_or_servo": True,
            "maximum_reobserves_per_keyframe": 2,
            "reobserve_limit_action": "give_up",
            "reobserve_limit_taxonomy": "source_visibility_unresolved",
            "minimum_authorizations": 3,
            "requires_authorization_before_source_boundary": True,
            "requires_authorization_at_or_after_source_boundary": True,
        },
        "unchanged": {
            "controller": True,
            "guards": True,
            "horizons": True,
            "dev20_family_matrix_gates": True,
            "no_tuning": True,
            "split_secrecy": True,
            "model": "Qwen3.8-27B",
            "fine_tuning": False,
            "official_three_cameras": True,
            "official_predicates": True,
        },
    }
    value["sha256"] = canonical_sha256(value)
    return value


def validate_cycle12_contract(value: Mapping[str, object]) -> None:
    body = dict(value)
    digest = body.pop("sha256", None)
    if digest != canonical_sha256(body):
        raise RuntimeError("Cycle 12 contract digest mismatch")
    if body.get("schema") != "robocasa-inspect-goal30-cycle12/v1":
        raise RuntimeError("Cycle 12 contract schema mismatch")
    if body.get("fable_verdict") != "ACCEPT_WITH_MANDATORY_CONDITIONS":
        raise RuntimeError("Cycle 12 lacks Fable acceptance")
    agency = body.get("agency")
    preflight = body.get("preflight")
    runtime = body.get("runtime")
    if not all(isinstance(row, Mapping) for row in (agency, preflight, runtime)):
        raise RuntimeError("Cycle 12 contract is incomplete")
    if agency.get("wording") != COUNTED_WORDING:
        raise RuntimeError("Cycle 12 honesty wording drifted")
    if preflight.get("calls") != 900 or preflight.get("invalid_authorize_maximum") != 0.10:
        raise RuntimeError("Cycle 12 preflight gates drifted")
    if runtime.get("reobserve_limit_action") != "give_up":
        raise RuntimeError("Cycle 12 reobserve fail-closed rule drifted")
