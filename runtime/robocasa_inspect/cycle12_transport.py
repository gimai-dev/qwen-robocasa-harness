"""Tokenizer-derived Cycle 12 transport bound and amendment validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

from .cycle10_policy import GRIPPERS, PHASES
from .cycle11_preflight import canonical_sha256
from .cycle12_policy import SKILLS, cycle12_response_schema
from .goal30_cycle10 import RESIDUAL_IDS

NOTE_ASCII_TOKEN_UPPER_BOUND = 120


def derive_cycle12_token_bound(tokenizer: object) -> dict[str, object]:
    """Apply the frozen Cycle 8 1.5x rule to every Cycle 12 branch."""
    token = "a1b2c3d4e5f6"
    counts: list[int] = []
    hashes: list[str] = []
    for skill in SKILLS:
        for residual_id in RESIDUAL_IDS:
            if skill != "authorize_source" and residual_id != "zero":
                continue
            for gripper in GRIPPERS:
                for phase in PHASES:
                    command = {
                        "observation_token": token,
                        "skill": skill,
                        "residual_id": residual_id,
                        "gripper": gripper,
                        "phase": phase,
                        "note": "",
                    }
                    serialized = json.dumps(
                        command, separators=(",", ":"), ensure_ascii=True
                    )
                    encoded = tokenizer.encode(serialized, add_special_tokens=False)  # type: ignore[attr-defined]
                    if not isinstance(encoded, Sequence):
                        raise TypeError("served tokenizer returned invalid tokens")
                    counts.append(len(encoded))
                    hashes.append(hashlib.sha256(serialized.encode()).hexdigest())
    structural = max(counts)
    maximum_legal = structural + NOTE_ASCII_TOKEN_UPPER_BOUND
    return {
        "template_count": len(counts),
        "template_sha256": hashes,
        "structural_max_tokens": structural,
        "note_ascii_token_upper_bound": NOTE_ASCII_TOKEN_UPPER_BOUND,
        "maximum_legal_branch_token_count": maximum_legal,
        "max_tokens": math.ceil(1.5 * maximum_legal),
        "formula": "ceil(1.5 * maximum_legal_branch_token_count)",
        "cycle8_rule_reused": True,
    }


def cycle12_schema_sha256() -> str:
    schema = cycle12_response_schema("a1b2c3d4e5f6")
    if list(schema["properties"])[-1] != "note":
        raise RuntimeError("Cycle 12 note is not the last schema property")
    return canonical_sha256(schema)


def validate_cycle12_transport_amendment(
    value: Mapping[str, object], *, contract_sha256: str
) -> None:
    body = dict(value)
    digest = body.pop("sha256", None)
    if digest != canonical_sha256(body):
        raise RuntimeError("Cycle 12 transport amendment digest drifted")
    if body.get("schema") != "robocasa-inspect-cycle12-transport-amendment/v1":
        raise RuntimeError("Cycle 12 transport amendment schema drifted")
    if body.get("cycle12_contract_sha256") != contract_sha256:
        raise RuntimeError("Cycle 12 transport amendment contract drifted")
    census = body.get("census")
    if not isinstance(census, Mapping):
        raise TypeError("Cycle 12 token census is missing")
    maximum = census.get("maximum_legal_branch_token_count")
    if not isinstance(maximum, int) or census.get("max_tokens") != math.ceil(1.5 * maximum):
        raise RuntimeError("Cycle 12 token formula drifted")
    if body.get("response_schema_sha256") != cycle12_schema_sha256():
        raise RuntimeError("Cycle 12 response schema drifted")
    if body.get("r1_calls_count_toward_r2") is not False:
        raise RuntimeError("Cycle 12 r1 evidence isolation drifted")
