"""Immutable Cycle 10 transport and numeric wall-budget authority."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from .cycle10_preflight import validate_cycle10_preflight

RUNTIME_SOURCE_FILES = (
    "preflight_cycle10_transport.py",
    "run_cycle10_skill.py",
    "replay_demonstration.py",
    "robocasa_inspect/contracts.py",
    "robocasa_inspect/cycle8_controller_calibration.py",
    "robocasa_inspect/cycle9_capacity.py",
    "robocasa_inspect/cycle10_policy.py",
    "robocasa_inspect/cycle10_controller_amendment.py",
    "robocasa_inspect/cycle10_preflight.py",
    "robocasa_inspect/cycle10_preflight_authority.py",
    "robocasa_inspect/cycle10_runtime.py",
    "robocasa_inspect/cycle10_servo_calibration.py",
    "robocasa_inspect/cycle10_source.py",
    "robocasa_inspect/demonstration_reference.py",
    "robocasa_inspect/demonstration_retrieval.py",
    "robocasa_inspect/demonstration_skill.py",
    "robocasa_inspect/goal30_cycle10.py",
    "robocasa_inspect/model_client.py",
    "robocasa_inspect/runner.py",
    "robocasa_inspect/sim_child.py",
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def runtime_implementation_sha256(root: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in RUNTIME_SOURCE_FILES
    }


def build_preflight_authority(
    preflight: Mapping[str, object],
    *,
    preflight_file_sha256: str,
    attempt_log_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> dict[str, object]:
    contract_sha256 = str(preflight.get("cycle10_contract_sha256"))
    validate_cycle10_preflight(preflight, contract_sha256=contract_sha256)
    value: dict[str, object] = {
        "schema": "robocasa-inspect-cycle10-preflight-authority/v1",
        "cycle10_contract_sha256": contract_sha256,
        "preflight_file_sha256": preflight_file_sha256,
        "attempt_log_sha256": attempt_log_sha256,
        "calls": 400,
        "schema_valid": 400,
        "transport_failures": 0,
        "latency_p99_s": float(preflight["latency_p99_s"]),
        "wall_multiplier": 4.0,
        "numeric_task_wall_budget_s": float(preflight["numeric_task_wall_budget_s"]),
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
    }
    value["sha256"] = _digest(value)
    return value


def validate_preflight_authority(
    value: Mapping[str, object],
    *,
    contract_sha256: str,
    preflight_file_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> None:
    authority = dict(value)
    digest = authority.pop("sha256", None)
    if digest != _digest(authority):
        raise RuntimeError("Cycle 10 preflight authority digest drifted")
    if authority.get("schema") != "robocasa-inspect-cycle10-preflight-authority/v1":
        raise RuntimeError("Cycle 10 preflight authority schema drifted")
    if authority.get("cycle10_contract_sha256") != contract_sha256:
        raise RuntimeError("Cycle 10 preflight authority contract drifted")
    if authority.get("preflight_file_sha256") != preflight_file_sha256:
        raise RuntimeError("Cycle 10 preflight authority result drifted")
    if authority.get("implementation_sha256") != dict(sorted(implementation_sha256.items())):
        raise RuntimeError("Cycle 10 preflight authority implementation drifted")
    if authority.get("calls") != 400 or authority.get("schema_valid") != 400:
        raise RuntimeError("Cycle 10 preflight authority call count drifted")
    wall = float(authority.get("numeric_task_wall_budget_s", 0.0))
    if not 300.0 <= wall <= 3600.0:
        raise RuntimeError("Cycle 10 preflight authority wall budget drifted")
