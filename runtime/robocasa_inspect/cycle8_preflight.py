"""Pure command and gate logic for the Cycle 8 400-call census."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .cycle5_skill import (
    expand_compact_pose_command,
    lateral_residual_bank,
    presented_residual_bank,
)
from .goal30_cycle8 import wilson_upper_bound

PREFLIGHT_NOTE = "Transport census only; no simulator action is executed."
FRESH_SERVER_MARKER_SCHEMA = "robocasa-inspect-cycle8-fresh-server/v1"


def _repr_vector(value: object) -> list[str]:
    return [repr(float(item)) for item in value]  # type: ignore[arg-type]


def observation_token(entry_id: str, index: int) -> str:
    return hashlib.sha256(f"cycle8:{index}:{entry_id}".encode()).hexdigest()[:12]


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def validate_fresh_server_marker(
    value: Mapping[str, object],
    *,
    attestation: Mapping[str, object],
    attestation_path: Path,
    corpus_sha256: str,
    token_bound_sha256: str,
    contract_sha256: str,
    amendment_e_sha256: str,
) -> None:
    marker = dict(value)
    digest = marker.pop("sha256", None)
    if not isinstance(digest, str) or digest != canonical_digest(marker):
        raise RuntimeError("Cycle 8 fresh-server marker digest mismatch")
    if marker.get("schema") != FRESH_SERVER_MARKER_SCHEMA:
        raise RuntimeError("Cycle 8 fresh-server marker schema mismatch")
    if marker.get("attestation_sha256") != hashlib.sha256(
        attestation_path.read_bytes()
    ).hexdigest():
        raise RuntimeError("Cycle 8 fresh-server attestation bytes drifted")
    expected = {
        "launch_id": attestation.get("launch_id"),
        "pid": attestation.get("pid"),
        "process_start_ticks": attestation.get("process_start_ticks"),
        "served_model_id": attestation.get("served_model_id"),
        "corpus_sha256": corpus_sha256,
        "token_bound_sha256": token_bound_sha256,
        "cycle8_contract_sha256": contract_sha256,
        "amendment_e_sha256": amendment_e_sha256,
    }
    if any(marker.get(key) != item for key, item in expected.items()):
        raise RuntimeError("Cycle 8 fresh-server marker authority drifted")
    attestation_mtime_ns = marker.get("attestation_mtime_ns")
    created_ns = marker.get("created_ns")
    if (
        not isinstance(attestation_mtime_ns, int)
        or attestation_mtime_ns != attestation_path.stat().st_mtime_ns
        or not isinstance(created_ns, int)
        or created_ns < attestation_mtime_ns
    ):
        raise RuntimeError("Cycle 8 server is not freshly attested")


def required_command(
    entry: Mapping[str, object], *, index: int, compact_pose: bool = False
) -> dict[str, object]:
    token = observation_token(str(entry["entry_id"]), index)
    schema_mode = entry["schema_mode"]
    if schema_mode == "pose":
        source = entry["source_full"]
        if not isinstance(source, Mapping):
            raise TypeError("pose preflight source is missing")
        option = next(
            value
            for value in lateral_residual_bank(source["translation_m"])
            if value["residual_id"] == entry["residual_id"]
        )
        compact = {
            "kind": "move",
            "observation_token": token,
            "phase": "approach",
            "candidate_id": "source_plus_residual",
            "residual_id": option["residual_id"],
            "gripper": "open",
            "note": PREFLIGHT_NOTE,
        }
        if compact_pose:
            return compact
        return {
            **compact,
            "translation_m_f64_repr": _repr_vector(option["translation_m"]),
            "lateral_residual_m_f64_repr": _repr_vector(
                option["lateral_residual_m"]
            ),
            "rotation_axis_angle_rad_f64_repr": _repr_vector(
                source["rotation_axis_angle_rad"]
            ),
        }
    if schema_mode == "custom":
        return {
            "kind": "move",
            "observation_token": token,
            "phase": "approach",
            "candidate_id": "custom",
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
            "gripper": "open",
            "note": PREFLIGHT_NOTE,
        }
    if schema_mode in {"give_up", "done"}:
        return {
            "kind": schema_mode,
            "observation_token": token,
            "note": PREFLIGHT_NOTE,
        }
    raise RuntimeError("Cycle 8 preflight schema mode drifted")


def presented_bank(source: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    return presented_residual_bank(source["translation_m"])


def validate_compact_source_equivalence(
    source: Mapping[str, object], *, observation_token: str = "a1b2c3d4e5f6"
) -> tuple[str, ...]:
    """Prove compact lookup is bit-identical to every legacy D1 const branch."""
    hashes = []
    for option in lateral_residual_bank(source["translation_m"]):
        compact = {
            "kind": "move",
            "observation_token": observation_token,
            "phase": "approach",
            "candidate_id": "source_plus_residual",
            "residual_id": option["residual_id"],
            "gripper": "open",
            "note": PREFLIGHT_NOTE,
        }
        expanded = expand_compact_pose_command(compact, source=source)
        expected = {
            **compact,
            "translation_m_f64_repr": _repr_vector(option["translation_m"]),
            "lateral_residual_m_f64_repr": _repr_vector(
                option["lateral_residual_m"]
            ),
            "rotation_axis_angle_rad_f64_repr": _repr_vector(
                source["rotation_axis_angle_rad"]
            ),
        }
        if expanded != expected:
            raise RuntimeError("compact pose reconstruction drifted from D1")
        hashes.append(canonical_digest(expanded))
    return tuple(hashes)


def transport_gate(
    *,
    completed: int,
    raw_malformed_attempts: int,
    decision_losses: int,
    length_finishes: int,
    attempt_latencies_s: Sequence[float],
) -> dict[str, object]:
    if not attempt_latencies_s:
        raise ValueError("Cycle 8 transport gate has no latency evidence")
    p95 = float(np.percentile(attempt_latencies_s, 95))
    max_model_calls = math.floor(900.0 / p95)
    wilson = wilson_upper_bound(raw_malformed_attempts, completed)
    return {
        "calls": completed,
        "raw_malformed_attempts": raw_malformed_attempts,
        "decision_losses": decision_losses,
        "finish_reason_length": length_finishes,
        "wilson_upper": wilson,
        "latency_p95_s": p95,
        "max_model_calls": max_model_calls,
        "passed": completed == 400
        and raw_malformed_attempts == 0
        and decision_losses == 0
        and length_finishes == 0
        and wilson <= 0.01
        and max_model_calls >= 203,
    }
