"""Fable-accepted Cycle 13 minimal authorization contract."""

from __future__ import annotations

from collections.abc import Mapping

from .cycle11_preflight import canonical_sha256
from .goal30_cycle12 import COUNTED_WORDING

CYCLE13_WORDING = (
    COUNTED_WORDING
    + " Zero residual and source gripper are used by construction; Qwen performs "
    "no pose or gripper correction."
)


def build_cycle13_contract(
    *,
    cycle12_contract_sha256: str,
    cycle11_corpus_sha256: str,
    cycle12_red_evidence: Mapping[str, object],
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "robocasa-inspect-goal30-cycle13/v1",
        "fable_verdict": "ACCEPT_WITH_CONDITIONS",
        "parents": {
            "cycle12_contract_sha256": cycle12_contract_sha256,
            "cycle11_corpus_sha256": cycle11_corpus_sha256,
            "cycle12_red_evidence": dict(cycle12_red_evidence),
        },
        "agency": {
            "wording": CYCLE13_WORDING,
            "skills": ["authorize_source", "reobserve", "give_up"],
            "zero_residual": True,
            "source_gripper": True,
            "qwen_pose_or_gripper_correction": False,
        },
        "transport": {
            "primary": "two_field_strict_json_schema",
            "case107_probe_before_preflight": True,
            "probe_excluded_from_semantic_tallies": True,
            "fallback_trigger": "any_finish_reason_length",
            "fallback": "token_bound_guided_choice_three_strings",
            "maximum_fallback_switches": 1,
            "fresh_900_after_fallback": True,
            "bind_primary_and_fallback_hashes": True,
        },
        "semantic_gates": {
            "calls": 900,
            "same_cycle11_corpus": True,
            "same_cycle12_class_thresholds": True,
            "schema_valid": 900,
            "transport_failures": 0,
        },
        "unchanged": {
            "honesty_conditions": True,
            "runtime_controller_guards_horizons": True,
            "dev20_family_matrix_gates": True,
            "split_secrecy": True,
            "no_tuning": True,
            "model": "Qwen3.8-27B",
            "fine_tuning": False,
            "official_three_cameras": True,
            "official_predicates": True,
        },
    }
    value["sha256"] = canonical_sha256(value)
    return value


def validate_cycle13_contract(value: Mapping[str, object]) -> None:
    body = dict(value)
    digest = body.pop("sha256", None)
    if digest != canonical_sha256(body):
        raise RuntimeError("Cycle 13 contract digest mismatch")
    if body.get("schema") != "robocasa-inspect-goal30-cycle13/v1":
        raise RuntimeError("Cycle 13 contract schema mismatch")
    if body.get("fable_verdict") != "ACCEPT_WITH_CONDITIONS":
        raise RuntimeError("Cycle 13 lacks Fable acceptance")
    agency = body.get("agency")
    transport = body.get("transport")
    if not isinstance(agency, Mapping) or not isinstance(transport, Mapping):
        raise TypeError("Cycle 13 contract is incomplete")
    if agency.get("wording") != CYCLE13_WORDING:
        raise RuntimeError("Cycle 13 honesty wording drifted")
    if transport.get("maximum_fallback_switches") != 1:
        raise RuntimeError("Cycle 13 fallback bound drifted")
