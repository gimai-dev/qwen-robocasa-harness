"""Derive a conservative Qwen output cap from every Cycle 8 branch shape."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

from .cycle5_skill import lateral_residual_bank
from .goal30_cycle7 import ROTATION_COMPONENT_MAX_RAD, TRANSLATION_COMPONENT_MAX_M

NOTE_ASCII_TOKEN_UPPER_BOUND = 120


def _repr_vector(value: object) -> list[str]:
    return [repr(float(item)) for item in value]  # type: ignore[arg-type]


def legal_branch_templates() -> tuple[dict[str, object], ...]:
    """Return all structural command shapes with a conservative numeric census."""
    token = "a1b2c3d4e5f6"
    source_translation = [0.0123456789012345, 0.0, 0.0]
    source_rotation = [0.09999999999999999, -0.09999999999999999, 0.0]
    pose = [
        {
            "kind": "move",
            "observation_token": token,
            "phase": "approach",
            "candidate_id": "source_plus_residual",
            "residual_id": option["residual_id"],
            "translation_m_f64_repr": _repr_vector(option["translation_m"]),
            "lateral_residual_m_f64_repr": _repr_vector(
                option["lateral_residual_m"]
            ),
            "rotation_axis_angle_rad_f64_repr": _repr_vector(source_rotation),
            "gripper": "close",
            "note": "",
        }
        for option in lateral_residual_bank(source_translation)
    ]
    custom = []
    for sign in (-1.0, 1.0):
        custom.append(
            {
                "kind": "move",
                "observation_token": token,
                "phase": "transport",
                "candidate_id": "custom",
                "translation_m": [sign * TRANSLATION_COMPONENT_MAX_M] * 3,
                "rotation_axis_angle_rad": [sign * ROTATION_COMPONENT_MAX_RAD] * 3,
                "gripper": "close",
                "note": "",
            }
        )
    terminal = [
        {"kind": kind, "observation_token": token, "note": ""}
        for kind in ("give_up", "done")
    ]
    return (*pose, *custom, *terminal)


def compact_legal_branch_templates() -> tuple[dict[str, object], ...]:
    token = "a1b2c3d4e5f6"
    pose = [
        {
            "kind": "move",
            "observation_token": token,
            "phase": "transport",
            "candidate_id": "source_plus_residual",
            "residual_id": option["residual_id"],
            "gripper": "close",
            "note": "",
        }
        for option in lateral_residual_bank([0.0123456789012345, 0.0, 0.0])
    ]
    custom_and_terminal = [
        command
        for command in legal_branch_templates()
        if command.get("candidate_id") == "custom"
        or command.get("kind") in {"give_up", "done"}
    ]
    return (*pose, *custom_and_terminal)


def derive_token_bound(tokenizer: object) -> dict[str, object]:
    """Apply Fable's 1.5x formula with a provable ASCII-note upper bound."""
    templates = legal_branch_templates()
    counts = []
    hashes = []
    for command in templates:
        serialized = json.dumps(command, sort_keys=True, separators=(",", ":"))
        encoded = tokenizer.encode(serialized, add_special_tokens=False)  # type: ignore[attr-defined]
        if not isinstance(encoded, Sequence):
            raise TypeError("served tokenizer returned an invalid token sequence")
        counts.append(len(encoded))
        hashes.append(hashlib.sha256(serialized.encode()).hexdigest())
    structural_max = max(counts)
    maximum_legal = structural_max + NOTE_ASCII_TOKEN_UPPER_BOUND
    return {
        "template_count": len(templates),
        "template_sha256": hashes,
        "structural_max_tokens": structural_max,
        "note_ascii_token_upper_bound": NOTE_ASCII_TOKEN_UPPER_BOUND,
        "maximum_legal_branch_token_count": maximum_legal,
        "max_tokens": math.ceil(1.5 * maximum_legal),
    }


def derive_compact_token_bound(tokenizer: object) -> dict[str, object]:
    templates = compact_legal_branch_templates()
    counts = []
    hashes = []
    for command in templates:
        serialized = json.dumps(command, sort_keys=True, separators=(",", ":"))
        encoded = tokenizer.encode(serialized, add_special_tokens=False)  # type: ignore[attr-defined]
        if not isinstance(encoded, Sequence):
            raise TypeError("served tokenizer returned an invalid token sequence")
        counts.append(len(encoded))
        hashes.append(hashlib.sha256(serialized.encode()).hexdigest())
    structural_max = max(counts)
    maximum_legal = structural_max + NOTE_ASCII_TOKEN_UPPER_BOUND
    return {
        "template_count": len(templates),
        "template_sha256": hashes,
        "structural_max_tokens": structural_max,
        "note_ascii_token_upper_bound": NOTE_ASCII_TOKEN_UPPER_BOUND,
        "maximum_legal_branch_token_count": maximum_legal,
        "max_tokens": math.ceil(1.5 * maximum_legal),
    }


def validate_token_bound(value: Mapping[str, object]) -> None:
    if value.get("schema") != "robocasa-inspect-cycle8-token-bound/v1":
        raise RuntimeError("Cycle 8 token-bound schema drifted")
    census = value.get("census")
    if not isinstance(census, Mapping):
        raise TypeError("Cycle 8 token census is missing")
    maximum = census.get("maximum_legal_branch_token_count")
    if not isinstance(maximum, int) or census.get("max_tokens") != math.ceil(
        1.5 * maximum
    ):
        raise RuntimeError("Cycle 8 token formula drifted")
    if not 32 <= int(census["max_tokens"]) <= 1024:
        raise RuntimeError("Cycle 8 token cap is outside the reviewed bound")


def validate_compact_token_bound(value: Mapping[str, object]) -> None:
    if value.get("schema") != "robocasa-inspect-cycle8-compact-token-bound/v1":
        raise RuntimeError("Cycle 8 compact token-bound schema drifted")
    census = value.get("census")
    if not isinstance(census, Mapping):
        raise TypeError("Cycle 8 compact token census is missing")
    maximum = census.get("maximum_legal_branch_token_count")
    if not isinstance(maximum, int) or census.get("max_tokens") != math.ceil(
        1.5 * maximum
    ):
        raise RuntimeError("Cycle 8 compact token formula drifted")
    if not 32 <= int(census["max_tokens"]) <= 1024:
        raise RuntimeError("Cycle 8 compact token cap is outside the reviewed bound")
