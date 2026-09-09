"""Cycle 13 primary and fallback transport authorities."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

from .cycle11_preflight import canonical_sha256
from .cycle13_policy import (
    SKILLS,
    cycle13_guided_choices,
    cycle13_response_schema,
)


def derive_cycle13_token_bound(tokenizer: object) -> dict[str, object]:
    token = "a1b2c3d4e5f6"
    primary_counts = []
    primary_hashes = []
    for skill in SKILLS:
        serialized = json.dumps(
            {"observation_token": token, "skill": skill}, separators=(",", ":")
        )
        encoded = tokenizer.encode(serialized, add_special_tokens=False)  # type: ignore[attr-defined]
        if not isinstance(encoded, Sequence):
            raise TypeError("served tokenizer returned invalid tokens")
        primary_counts.append(len(encoded))
        primary_hashes.append(hashlib.sha256(serialized.encode()).hexdigest())
    choices = cycle13_guided_choices(token)
    choice_counts = [
        len(tokenizer.encode(value, add_special_tokens=False))  # type: ignore[attr-defined]
        for value in choices
    ]
    maximum = max(*primary_counts, *choice_counts)
    return {
        "primary_branch_count": len(primary_counts),
        "primary_branch_sha256": primary_hashes,
        "fallback_choice_template_sha256": canonical_sha256(choices),
        "maximum_legal_branch_token_count": maximum,
        "max_tokens": math.ceil(1.5 * maximum),
        "formula": "ceil(1.5 * maximum_legal_branch_token_count)",
    }


def cycle13_primary_schema_sha256() -> str:
    return canonical_sha256(cycle13_response_schema("a1b2c3d4e5f6"))


def cycle13_fallback_template_sha256() -> str:
    return canonical_sha256(cycle13_guided_choices("a1b2c3d4e5f6"))


def validate_cycle13_transport(
    value: Mapping[str, object], *, contract_sha256: str
) -> None:
    body = dict(value)
    digest = body.pop("sha256", None)
    if digest != canonical_sha256(body):
        raise RuntimeError("Cycle 13 transport digest drifted")
    if body.get("schema") != "robocasa-inspect-cycle13-transport/v1":
        raise RuntimeError("Cycle 13 transport schema drifted")
    if body.get("cycle13_contract_sha256") != contract_sha256:
        raise RuntimeError("Cycle 13 transport contract drifted")
    if body.get("primary_schema_sha256") != cycle13_primary_schema_sha256():
        raise RuntimeError("Cycle 13 primary schema drifted")
    if body.get("fallback_template_sha256") != cycle13_fallback_template_sha256():
        raise RuntimeError("Cycle 13 fallback template drifted")
