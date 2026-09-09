"""Cycle 10 exact-schema transport and wall-budget authority."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def numeric_task_wall_budget_s(
    latencies_s: Sequence[float],
    *,
    maximum_model_calls: int = 400,
    multiplier: float = 4.0,
    minimum_s: float = 300.0,
    hard_cap_s: float = 3600.0,
) -> tuple[float, float]:
    values = sorted(float(value) for value in latencies_s)
    if (
        len(values) != 400
        or any(value <= 0.0 or not math.isfinite(value) for value in values)
        or maximum_model_calls != 400
        or multiplier < 1.0
        or minimum_s <= 0.0
        or hard_cap_s < minimum_s
    ):
        raise ValueError("Cycle 10 preflight latency authority is invalid")
    index = math.ceil(0.99 * len(values)) - 1
    p99 = values[index]
    budget = min(hard_cap_s, max(minimum_s, p99 * multiplier * maximum_model_calls))
    return p99, budget


def validate_cycle10_preflight(
    value: Mapping[str, object], *, contract_sha256: str
) -> None:
    if value.get("schema") != "robocasa-inspect-cycle10-preflight/v1":
        raise RuntimeError("Cycle 10 preflight schema mismatch")
    if value.get("passed") is not True or value.get("complete") is not True:
        raise RuntimeError("Cycle 10 preflight is not green")
    if value.get("cycle10_contract_sha256") != contract_sha256:
        raise RuntimeError("Cycle 10 preflight contract drifted")
    if value.get("calls") != 400 or value.get("schema_valid") != 400:
        raise RuntimeError("Cycle 10 preflight call count drifted")
    if value.get("transport_failures") != 0:
        raise RuntimeError("Cycle 10 preflight transport is red")
    wall = float(value.get("numeric_task_wall_budget_s", 0.0))
    if not 300.0 <= wall <= 3600.0:
        raise RuntimeError("Cycle 10 numeric task wall budget drifted")
