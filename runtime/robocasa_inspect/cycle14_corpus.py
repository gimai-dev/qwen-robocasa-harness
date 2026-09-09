"""Deterministic, disjoint Corpus B source selection and validation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence


def episode_selection_key(*, seed: int, task: str, episode_index: int) -> str:
    return hashlib.sha256(f"{seed}:{task}:{episode_index}".encode()).hexdigest()


def select_disjoint_episode(
    candidates: Sequence[Mapping[str, object]],
    *,
    seed: int,
    task: str,
    excluded_episode_indices: set[int],
) -> Mapping[str, object]:
    eligible = [
        row
        for row in candidates
        if row.get("eligible") is True
        and int(row["episode_index"]) not in excluded_episode_indices
    ]
    if not eligible:
        raise RuntimeError(f"no disjoint eligible source episode for {task}")
    return min(
        eligible,
        key=lambda row: episode_selection_key(
            seed=seed, task=task, episode_index=int(row["episode_index"])
        ),
    )


def validate_cycle14_source_cohort(
    value: Mapping[str, object],
    *,
    seed: int,
    construction_rule_sha256: str,
    corpus_a_sha256: str,
    excluded_episode_indices: set[int],
) -> None:
    if value.get("schema") != "robocasa-inspect-cycle14-source-cohort/v1":
        raise RuntimeError("Cycle 14 source-cohort schema mismatch")
    if value.get("complete") is not True or value.get("passed") is not True:
        raise RuntimeError("Cycle 14 source cohort is not green")
    if value.get("seed") != seed:
        raise RuntimeError("Cycle 14 source-cohort seed drifted")
    if value.get("construction_rule_sha256") != construction_rule_sha256:
        raise RuntimeError("Cycle 14 source-cohort construction drifted")
    if value.get("corpus_a_sha256") != corpus_a_sha256:
        raise RuntimeError("Cycle 14 source-cohort parent drifted")
    records = value.get("records")
    if not isinstance(records, list) or len(records) != 20:
        raise RuntimeError("Cycle 14 source cohort must contain exactly 20 tasks")
    episodes = [int(row["episode_index"]) for row in records]
    if len(set(episodes)) != 20 or set(episodes) & excluded_episode_indices:
        raise RuntimeError("Cycle 14 source episodes are not disjoint and unique")
    if any(row.get("eligible") is not True for row in records):
        raise RuntimeError("Cycle 14 source cohort contains an ineligible route")
    if len({str(row["family"]) for row in records}) < 4:
        raise RuntimeError("Cycle 14 source cohort has fewer than four families")
