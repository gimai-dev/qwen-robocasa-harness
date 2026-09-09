"""Frozen semantic discrimination gates for Cycle 11."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

CASE_KINDS = ("aligned", "valid_next", "invalid_reference")
VARIANTS = ("unmasked", "masked", "invalid")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_semantic_corpus(value: Mapping[str, object]) -> None:
    if value.get("schema") != "robocasa-inspect-cycle11-semantic-corpus/v1":
        raise RuntimeError("Cycle 11 corpus schema mismatch")
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != 500:
        raise RuntimeError("Cycle 11 corpus must contain exactly 500 cases")
    kinds = {kind: 0 for kind in CASE_KINDS}
    distinct = 0
    identifiers: set[str] = set()
    for row in cases:
        if not isinstance(row, Mapping) or row.get("kind") not in kinds:
            raise RuntimeError("Cycle 11 corpus case is invalid")
        kinds[str(row["kind"])] += 1
        identifier = row.get("case_id")
        if not isinstance(identifier, str) or identifier in identifiers:
            raise RuntimeError("Cycle 11 corpus case IDs are invalid")
        identifiers.add(identifier)
        if row.get("kind") == "aligned" and row.get("distinct_frame") is True:
            distinct += 1
            if float(row.get("translation_error_m", 1.0)) > 0.005 + 1e-12:
                raise RuntimeError("distinct aligned translation gate drifted")
            if float(row.get("rotation_error_rad", 1.0)) > 0.025 + 1e-12:
                raise RuntimeError("distinct aligned rotation gate drifted")
    if kinds != {"aligned": 200, "valid_next": 200, "invalid_reference": 100}:
        raise RuntimeError("Cycle 11 corpus class counts drifted")
    if distinct < 100:
        raise RuntimeError("Cycle 11 needs 100 distinct-frame aligned cases")
    digest = value.get("sha256")
    body = dict(value)
    body.pop("sha256", None)
    if digest != canonical_sha256(body):
        raise RuntimeError("Cycle 11 corpus digest mismatch")


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def score_semantic_attempts(attempts: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(attempts) != 900:
        raise ValueError("Cycle 11 preflight requires exactly 900 attempts")
    buckets = {
        "unmasked_aligned": [0, 0],
        "unmasked_valid_next": [0, 0],
        "masked_aligned": [0, 0],
        "masked_aligned_distinct": [0, 0],
        "masked_valid_next": [0, 0],
        "invalid_veto": [0, 0],
        "invalid_execute": [0, 0],
    }
    schema_valid = 0
    transport_failures = 0
    latencies: list[float] = []
    for row in attempts:
        latency = float(row.get("latency_s", 0.0))
        if latency <= 0.0 or not math.isfinite(latency):
            raise ValueError("Cycle 11 attempt latency is invalid")
        latencies.append(latency)
        if row.get("transport_failure") is True:
            transport_failures += 1
            continue
        if row.get("schema_valid") is not True:
            continue
        schema_valid += 1
        variant = str(row.get("variant"))
        kind = str(row.get("kind"))
        skill = str(row.get("skill"))
        if variant == "unmasked" and kind in {"aligned", "valid_next"}:
            key = f"unmasked_{kind}"
            buckets[key][1] += 1
            buckets[key][0] += int(
                skill == ("reobserve" if kind == "aligned" else "execute_source")
            )
        elif variant == "masked" and kind in {"aligned", "valid_next"}:
            key = f"masked_{kind}"
            buckets[key][1] += 1
            buckets[key][0] += int(
                skill == ("reobserve" if kind == "aligned" else "execute_source")
            )
            if kind == "aligned" and row.get("distinct_frame") is True:
                buckets["masked_aligned_distinct"][1] += 1
                buckets["masked_aligned_distinct"][0] += int(skill == "reobserve")
        elif variant == "invalid" and kind == "invalid_reference":
            buckets["invalid_veto"][1] += 1
            buckets["invalid_veto"][0] += int(skill in {"give_up", "reobserve"})
            buckets["invalid_execute"][1] += 1
            buckets["invalid_execute"][0] += int(skill == "execute_source")
        else:
            raise ValueError("Cycle 11 attempt variant/class drifted")
    rates = {key: _rate(*counts) for key, counts in buckets.items()}
    values = sorted(latencies)
    p99 = values[math.ceil(0.99 * len(values)) - 1]
    wall = min(3600.0, max(300.0, p99 * 4.0 * 400.0))
    gates = {
        "schema_transport": schema_valid == 900 and transport_failures == 0,
        "unmasked_aligned": rates["unmasked_aligned"] >= 0.95,
        "unmasked_valid_next": rates["unmasked_valid_next"] >= 0.95,
        "masked_aligned": rates["masked_aligned"] >= 0.80,
        "masked_aligned_distinct": rates["masked_aligned_distinct"] >= 0.80,
        "masked_valid_next": rates["masked_valid_next"] >= 0.80,
        "invalid_veto": rates["invalid_veto"] >= 0.90,
        "invalid_execute": rates["invalid_execute"] <= 0.10,
    }
    return {
        "calls": len(attempts),
        "schema_valid": schema_valid,
        "transport_failures": transport_failures,
        "rates": rates,
        "gates": gates,
        "passed": all(gates.values()),
        "latency_p99_s": p99,
        "numeric_task_wall_budget_s": wall,
    }
