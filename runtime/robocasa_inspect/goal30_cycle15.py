"""Fable-accepted Cycle 15 visual-presentation authority."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from .cycle11_preflight import canonical_sha256
from .cycle13_transport import cycle13_primary_schema_sha256
from .cycle15_policy import CYCLE15_SYSTEM_PROMPT


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_cycle15_contract(
    *,
    cycle14_contract_sha256: str,
    cycle13_contract_sha256: str,
    cycle13_transport_sha256: str,
    corpus_a_sha256: str,
    corpus_a_file_sha256: str,
    cycle14_prompt_revision1_sha256: str,
    cycle14_corpus_a_r1_file_sha256: str,
    cycle14_corpus_a_r2_red: Mapping[str, object],
    invalid_stratification_sha256: str,
    invalid_stratification_file_sha256: str,
    fable_review_sha256: str,
    role_header_diff: str,
    presentation_source_sha256: str,
    corpus_b_seed: int,
    corpus_b_construction_rule_sha256: str,
    corpus_b_construction_authority: Mapping[str, str],
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "robocasa-inspect-goal30-cycle15/v1",
        "fable_verdict": "ACCEPT_WITH_ANTI_OVERFIT_AUTHORITY",
        "parents": {
            "cycle14_contract_sha256": cycle14_contract_sha256,
            "cycle13_contract_sha256": cycle13_contract_sha256,
            "cycle13_transport_sha256": cycle13_transport_sha256,
            "corpus_a_sha256": corpus_a_sha256,
            "corpus_a_file_sha256": corpus_a_file_sha256,
            "cycle14_prompt_revision1_sha256": cycle14_prompt_revision1_sha256,
            "cycle14_corpus_a_r1_file_sha256": cycle14_corpus_a_r1_file_sha256,
            "cycle14_corpus_a_r2_red": dict(cycle14_corpus_a_r2_red),
            "invalid_stratification_sha256": invalid_stratification_sha256,
            "invalid_stratification_file_sha256": (
                invalid_stratification_file_sha256
            ),
            "fable_review_sha256": fable_review_sha256,
        },
        "presentation": {
            "variant": "current_external_mosaic_source_external_mosaic_current_wrist/v1",
            "maximum_variants": 1,
            "source_sha256": presentation_source_sha256,
            "input_panel_dimensions_px": [512, 286],
            "official_external_view_dimensions_px": [256, 256],
            "output_mosaic_dimensions_px": [512, 286],
            "wrist_dimensions_px": [256, 256],
            "wrist_bytes_unmodified": True,
            "all_official_external_pixels_retained_exactly_once": True,
            "additional_downscale": False,
            "learned_processing": False,
            "task_names_in_labels": False,
        },
        "prompt": {
            "role_header_diff": role_header_diff,
            "decision_rules_byte_identical_to_cycle14_revision1": True,
            "final_sha256": hashlib.sha256(
                CYCLE15_SYSTEM_PROMPT.encode()
            ).hexdigest(),
            "response_schema_sha256": cycle13_primary_schema_sha256(),
            "maximum_semantic_rule_edits": 0,
        },
        "development_gate": {
            "corpus": "A",
            "maximum_runs": 1,
            "red_returns_to_fable": True,
        },
        "corpus_b": {
            "seed": corpus_b_seed,
            "construction_rule_sha256": corpus_b_construction_rule_sha256,
            "construction_authority": dict(corpus_b_construction_authority),
            "source_episodes_disjoint_from_a": True,
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
        "unchanged": {
            "model": "Qwen3.8-27B",
            "fine_tuning": False,
            "official_three_cameras": True,
            "cycle13_minimal_transport": True,
            "qwen_pose_or_gripper_correction": False,
            "simulator_actions_before_semantic_green": 0,
        },
    }
    value["sha256"] = canonical_sha256(value)
    return value


def validate_cycle15_contract(value: Mapping[str, object], *, root: Path) -> None:
    body = dict(value)
    digest = body.pop("sha256", None)
    if digest != canonical_sha256(body):
        raise RuntimeError("Cycle 15 contract digest mismatch")
    if body.get("schema") != "robocasa-inspect-goal30-cycle15/v1":
        raise RuntimeError("Cycle 15 contract schema mismatch")
    if body.get("fable_verdict") != "ACCEPT_WITH_ANTI_OVERFIT_AUTHORITY":
        raise RuntimeError("Cycle 15 lacks Fable acceptance")
    presentation = body.get("presentation")
    prompt = body.get("prompt")
    corpus = body.get("corpus_b")
    if not all(isinstance(item, Mapping) for item in (presentation, prompt, corpus)):
        raise TypeError("Cycle 15 contract is incomplete")
    assert isinstance(presentation, Mapping)
    assert isinstance(prompt, Mapping)
    assert isinstance(corpus, Mapping)
    if presentation.get("maximum_variants") != 1:
        raise RuntimeError("Cycle 15 presentation variant bound drifted")
    if presentation.get("source_sha256") != _sha256(
        root / "robocasa_inspect/cycle15_presentation.py"
    ):
        raise RuntimeError("Cycle 15 presentation source drifted")
    if prompt.get("final_sha256") != hashlib.sha256(
        CYCLE15_SYSTEM_PROMPT.encode()
    ).hexdigest():
        raise RuntimeError("Cycle 15 prompt drifted")
    if prompt.get("response_schema_sha256") != cycle13_primary_schema_sha256():
        raise RuntimeError("Cycle 15 response schema drifted")
    if corpus.get("maximum_scored_runs") != 1 or corpus.get("seed") != 14014:
        raise RuntimeError("Cycle 15 Corpus B authority drifted")
