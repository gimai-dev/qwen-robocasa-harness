"""Cycle 12 frozen same-task authorization/veto gates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def score_cycle12_attempts(
    attempts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(attempts) != 900:
        raise ValueError("Cycle 12 requires exactly 900 attempts")
    buckets = {
        "unmasked_aligned_authorize": [0, 0],
        "unmasked_valid_authorize": [0, 0],
        "masked_aligned_authorize": [0, 0],
        "masked_distinct_aligned_authorize": [0, 0],
        "masked_valid_authorize": [0, 0],
        "invalid_veto": [0, 0],
        "invalid_authorize": [0, 0],
    }
    confusion: dict[str, dict[str, int]] = {}
    schema_valid = 0
    transport_failures = 0
    latencies: list[float] = []
    for row in attempts:
        latency = float(row.get("latency_s", 0.0))
        if latency <= 0 or not math.isfinite(latency):
            raise ValueError("Cycle 12 attempt latency is invalid")
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
        label = f"{variant}:{kind}"
        confusion.setdefault(label, {})[skill] = confusion.setdefault(label, {}).get(skill, 0) + 1
        if variant in {"unmasked", "masked"} and kind in {"aligned", "valid_next"}:
            short = "valid" if kind == "valid_next" else "aligned"
            key = f"{variant}_{short}_authorize"
            buckets[key][1] += 1
            buckets[key][0] += int(skill == "authorize_source")
            if variant == "masked" and kind == "aligned" and row.get("distinct_frame") is True:
                key = "masked_distinct_aligned_authorize"
                buckets[key][1] += 1
                buckets[key][0] += int(skill == "authorize_source")
        elif variant == "invalid" and kind == "invalid_reference":
            buckets["invalid_veto"][1] += 1
            buckets["invalid_veto"][0] += int(skill in {"give_up", "reobserve"})
            buckets["invalid_authorize"][1] += 1
            buckets["invalid_authorize"][0] += int(skill == "authorize_source")
        else:
            raise ValueError("Cycle 12 attempt class drifted")
    rates = {key: _rate(*counts) for key, counts in buckets.items()}
    values = sorted(latencies)
    p99 = values[math.ceil(0.99 * len(values)) - 1]
    gates = {
        "schema_transport": schema_valid == 900 and transport_failures == 0,
        "unmasked_aligned": rates["unmasked_aligned_authorize"] >= 0.90,
        "unmasked_valid": rates["unmasked_valid_authorize"] >= 0.90,
        "masked_aligned": rates["masked_aligned_authorize"] >= 0.90,
        "masked_distinct_aligned": rates["masked_distinct_aligned_authorize"] >= 0.85,
        "masked_valid": rates["masked_valid_authorize"] >= 0.90,
        "invalid_veto": rates["invalid_veto"] >= 0.90,
        "invalid_authorize": rates["invalid_authorize"] <= 0.10,
    }
    return {
        "calls": 900,
        "schema_valid": schema_valid,
        "transport_failures": transport_failures,
        "confusion_matrices": confusion,
        "rates": rates,
        "gates": gates,
        "passed": all(gates.values()),
        "latency_p99_s": p99,
        "numeric_task_wall_budget_s": min(3600.0, max(300.0, p99 * 4.0 * 400.0)),
    }
