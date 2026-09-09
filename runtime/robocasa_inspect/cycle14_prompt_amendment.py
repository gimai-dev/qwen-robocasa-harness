"""Authority for Cycle 14's second and final development prompt."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from .cycle11_preflight import canonical_sha256
from .cycle14_policy_v2 import CYCLE14_SYSTEM_PROMPT_V2


def build_cycle14_prompt_amendment(
    *,
    cycle14_contract_sha256: str,
    corpus_a_result_file_sha256: str,
    corpus_a_score: Mapping[str, object],
    prompt_before_sha256: str,
    prompt_diff: str,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "robocasa-inspect-cycle14-prompt-amendment/v1",
        "cycle14_contract_sha256": cycle14_contract_sha256,
        "revision_index": 2,
        "maximum_prompt_revisions": 2,
        "development_corpus": "A",
        "corpus_a_result_file_sha256": corpus_a_result_file_sha256,
        "corpus_a_score": dict(corpus_a_score),
        "before_sha256": prompt_before_sha256,
        "after_sha256": hashlib.sha256(
            CYCLE14_SYSTEM_PROMPT_V2.encode()
        ).hexdigest(),
        "unified_diff": prompt_diff,
        "task_agnostic": True,
        "task_object_fixture_names_added": 0,
        "corpus_b_seed_unchanged": True,
        "corpus_b_construction_rule_unchanged": True,
        "corpus_b_scored_runs_remaining": 1,
        "simulator_actions": 0,
    }
    value["sha256"] = canonical_sha256(value)
    return value


def validate_cycle14_prompt_amendment(
    value: Mapping[str, object], *, cycle14_contract_sha256: str
) -> None:
    body = dict(value)
    digest = body.pop("sha256", None)
    if digest != canonical_sha256(body):
        raise RuntimeError("Cycle 14 prompt-amendment digest mismatch")
    if body.get("schema") != "robocasa-inspect-cycle14-prompt-amendment/v1":
        raise RuntimeError("Cycle 14 prompt-amendment schema mismatch")
    if body.get("cycle14_contract_sha256") != cycle14_contract_sha256:
        raise RuntimeError("Cycle 14 prompt-amendment contract drifted")
    if body.get("revision_index") != body.get("maximum_prompt_revisions") or body.get(
        "revision_index"
    ) != 2:
        raise RuntimeError("Cycle 14 prompt-revision bound drifted")
    if body.get("after_sha256") != hashlib.sha256(
        CYCLE14_SYSTEM_PROMPT_V2.encode()
    ).hexdigest():
        raise RuntimeError("Cycle 14 prompt-v2 authority drifted")
    if body.get("task_agnostic") is not True or body.get(
        "task_object_fixture_names_added"
    ) != 0:
        raise RuntimeError("Cycle 14 prompt-v2 is not task agnostic")
    score = body.get("corpus_a_score")
    if not isinstance(score, Mapping):
        raise TypeError("Cycle 14 Corpus A score is missing")
    rates = score.get("rates")
    if not isinstance(rates, Mapping) or rates.get("invalid_authorize") != 0.62:
        raise RuntimeError("Cycle 14 Corpus A development evidence drifted")
