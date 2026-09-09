"""Fable-accepted Cycle 14 disjoint-corpus semantic authority."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from .cycle11_preflight import canonical_sha256
from .cycle14_policy import CYCLE14_SYSTEM_PROMPT

CORPUS_B_SEED = 14014
CONSTRUCTION_SOURCES = (
    "build_cycle11_semantic_corpus.py",
    "build_cycle14_source_cohort.py",
    "replay_demonstration.py",
    "robocasa_inspect/cycle10_policy.py",
    "robocasa_inspect/cycle10_source.py",
    "robocasa_inspect/cycle9_capacity.py",
    "robocasa_inspect/cycle11_preflight.py",
    "robocasa_inspect/cycle14_corpus.py",
    "robocasa_inspect/demonstration_reference.py",
    "robocasa_inspect/demonstration_skill.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cycle14_construction_authority(root: Path) -> dict[str, str]:
    return {name: _sha256(root / name) for name in CONSTRUCTION_SOURCES}


def cycle14_construction_sha256(root: Path) -> str:
    return canonical_sha256(cycle14_construction_authority(root))


def build_cycle14_contract(
    *,
    cycle13_contract_sha256: str,
    cycle13_transport_sha256: str,
    corpus_a_sha256: str,
    corpus_a_file_sha256: str,
    cycle13_red_evidence: Mapping[str, object],
    prompt_before_sha256: str,
    prompt_diff: str,
    construction_authority: Mapping[str, str],
    fable_review_sha256: str,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "robocasa-inspect-goal30-cycle14/v1",
        "fable_verdict": "ACCEPT",
        "parents": {
            "cycle13_contract_sha256": cycle13_contract_sha256,
            "cycle13_transport_sha256": cycle13_transport_sha256,
            "corpus_a_sha256": corpus_a_sha256,
            "corpus_a_file_sha256": corpus_a_file_sha256,
            "cycle13_red_evidence": dict(cycle13_red_evidence),
            "fable_review_sha256": fable_review_sha256,
        },
        "prompt_amendment": {
            "before_sha256": prompt_before_sha256,
            "after_sha256": hashlib.sha256(
                CYCLE14_SYSTEM_PROMPT.encode()
            ).hexdigest(),
            "unified_diff": prompt_diff,
            "task_agnostic": True,
            "corpus_a_development_only": True,
            "maximum_prompt_revisions": 2,
        },
        "corpus_b": {
            "seed": CORPUS_B_SEED,
            "cases": 500,
            "calls": 900,
            "source_episodes_disjoint_from_a": True,
            "same_development_pool": True,
            "construction_authority": dict(construction_authority),
            "construction_rule_sha256": canonical_sha256(
                dict(construction_authority)
            ),
            "hash_before_any_model_call": True,
            "maximum_scored_runs": 1,
        },
        "semantic_gates": {
            "schema_valid": 900,
            "transport_failures": 0,
            "unmasked_aligned_authorize_min": 0.90,
            "unmasked_valid_authorize_min": 0.90,
            "masked_aligned_authorize_min": 0.90,
            "masked_distinct_aligned_authorize_min": 0.85,
            "masked_valid_authorize_min": 0.90,
            "invalid_veto_min": 0.90,
            "invalid_authorize_max": 0.10,
        },
        "runtime_gate": {
            "simulator_actions_before_corpus_b_green": 0,
            "red_corpus_b_returns_to_fable": True,
        },
        "unchanged": {
            "model": "Qwen3.8-27B",
            "fine_tuning": False,
            "official_three_cameras": True,
            "split_secrecy": True,
            "cycle13_minimal_transport": True,
            "qwen_pose_or_gripper_correction": False,
        },
    }
    value["sha256"] = canonical_sha256(value)
    return value


def validate_cycle14_contract(
    value: Mapping[str, object], *, root: Path | None = None
) -> None:
    body = dict(value)
    digest = body.pop("sha256", None)
    if digest != canonical_sha256(body):
        raise RuntimeError("Cycle 14 contract digest mismatch")
    if body.get("schema") != "robocasa-inspect-goal30-cycle14/v1":
        raise RuntimeError("Cycle 14 contract schema mismatch")
    if body.get("fable_verdict") != "ACCEPT":
        raise RuntimeError("Cycle 14 lacks Fable acceptance")
    prompt = body.get("prompt_amendment")
    corpus = body.get("corpus_b")
    gates = body.get("semantic_gates")
    if not all(isinstance(item, Mapping) for item in (prompt, corpus, gates)):
        raise TypeError("Cycle 14 contract is incomplete")
    assert isinstance(prompt, Mapping)
    assert isinstance(corpus, Mapping)
    assert isinstance(gates, Mapping)
    if prompt.get("after_sha256") != hashlib.sha256(
        CYCLE14_SYSTEM_PROMPT.encode()
    ).hexdigest():
        raise RuntimeError("Cycle 14 prompt authority drifted")
    if corpus.get("seed") != CORPUS_B_SEED:
        raise RuntimeError("Cycle 14 Corpus B seed drifted")
    if corpus.get("maximum_scored_runs") != 1:
        raise RuntimeError("Cycle 14 scored-run bound drifted")
    expected_gates = {
        "schema_valid": 900,
        "transport_failures": 0,
        "unmasked_aligned_authorize_min": 0.90,
        "unmasked_valid_authorize_min": 0.90,
        "masked_aligned_authorize_min": 0.90,
        "masked_distinct_aligned_authorize_min": 0.85,
        "masked_valid_authorize_min": 0.90,
        "invalid_veto_min": 0.90,
        "invalid_authorize_max": 0.10,
    }
    if dict(gates) != expected_gates:
        raise RuntimeError("Cycle 14 semantic gates drifted")
    if root is not None:
        current = cycle14_construction_authority(root)
        if corpus.get("construction_authority") != current:
            raise RuntimeError("Cycle 14 construction authority drifted")
        if corpus.get("construction_rule_sha256") != canonical_sha256(current):
            raise RuntimeError("Cycle 14 construction rule drifted")
