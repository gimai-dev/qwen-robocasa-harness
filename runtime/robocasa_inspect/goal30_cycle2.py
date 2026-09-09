"""Immutable reviewed authority for the second 30%-goal cycle."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_cycle2_contract(
    *,
    goal_authority_sha256: str,
    cycle1_contract_sha256: str,
    cohort: Mapping[str, object],
    calibration: Mapping[str, object],
    dino_authority: Mapping[str, object],
    latency: Mapping[str, object],
    implementation_parent: Mapping[str, str],
) -> dict[str, object]:
    if latency.get("calls") != 10:
        raise ValueError("latency preflight must contain exactly ten calls")
    p95 = latency.get("p95_s")
    if not isinstance(p95, (int, float)) or not math.isfinite(p95) or p95 <= 0:
        raise ValueError("latency p95 is invalid")
    model_calls = min(64, math.floor(900.0 / float(p95)))
    if model_calls < 16:
        raise ValueError("latency budget must permit at least sixteen calls")
    value: dict[str, object] = {
        "schema": "robocasa-inspect-goal30-cycle2/v1",
        "goal_authority_sha256": goal_authority_sha256,
        "cycle1_contract_sha256": cycle1_contract_sha256,
        "cohort": dict(cohort),
        "calibration": dict(calibration),
        "dino_authority": dict(dino_authority),
        "latency_preflight": dict(latency),
        "implementation_parent": dict(sorted(implementation_parent.items())),
        "budgets": {
            "decisions_per_keyframe": 8,
            "model_calls": model_calls,
            "environment_steps": 450,
            "wall_s": 1200,
            "terminal_hold_steps": 10,
            "implementation_iterations": 2,
        },
        "gates": {
            "diagnostic_regressions": 2,
            "diagnostic_conversions": 2,
            "diagnostic_requires_non_fridge": True,
            "dev20_successes": 6,
            "dev20_successful_families": 3,
            "family10_successes_each": 3,
            "matrix_successes_each_seed": 30,
        },
    }
    value["sha256"] = _digest(value)
    return value


def validate_cycle2_contract(contract: Mapping[str, object]) -> None:
    value = dict(contract)
    digest = value.pop("sha256", None)
    if not isinstance(digest, str) or digest != _digest(value):
        raise RuntimeError("cycle2 contract digest mismatch")
    if value.get("schema") != "robocasa-inspect-goal30-cycle2/v1":
        raise RuntimeError("cycle2 contract schema mismatch")
    budgets = value.get("budgets")
    gates = value.get("gates")
    if not isinstance(budgets, Mapping) or not isinstance(gates, Mapping):
        raise RuntimeError("cycle2 contract is incomplete")
    if budgets.get("model_calls") not in range(16, 65):
        raise RuntimeError("cycle2 model-call budget drift")
    if gates.get("matrix_successes_each_seed") != 30:
        raise RuntimeError("cycle2 goal gate drift")
