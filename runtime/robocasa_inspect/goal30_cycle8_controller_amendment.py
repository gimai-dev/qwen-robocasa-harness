"""Fable-accepted Cycle 8 controller-realization amendment."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_controller_amendment(
    *,
    amendment_e_sha256: str,
    reliability_preflight_sha256: str,
    diagnostic_r1_sha256: str,
    diagnostic_r1_video_sha256: str,
    fable_review_sha256: str,
    trace_evidence: Sequence[Mapping[str, object]],
    feasibility_rows: Sequence[Mapping[str, object]],
    implementation_parent: Mapping[str, str],
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "robocasa-inspect-goal30-cycle8-controller-amendment/v1",
        "fable_verdict": "ACCEPT",
        "parents": {
            "amendment_e_sha256": amendment_e_sha256,
            "reliability_preflight_sha256": reliability_preflight_sha256,
            "diagnostic_r1_sha256": diagnostic_r1_sha256,
            "diagnostic_r1_video_sha256": diagnostic_r1_video_sha256,
            "fable_review_sha256": fable_review_sha256,
            "implementation_parent": dict(sorted(implementation_parent.items())),
        },
        "sealed_trace_evidence": [dict(row) for row in trace_evidence],
        "calibration": {
            "translation_controller_gain": 3.5,
            "rotation_controller_gain": 3.5,
            "worst_trace_translation_median": 0.224170,
            "worst_trace_rotation_median": 0.210763,
            "predicted_translation_realization": 0.784595,
            "predicted_rotation_realization": 0.7376705,
            "normalized_translation_component_cap": 0.35,
            "normalized_rotation_component_cap": 0.175,
            "maximum_realized_translation_per_step_m": 0.005,
            "maximum_realized_rotation_per_step_rad": 0.025,
            "overshoot_taxonomy": "controller_overshoot",
            "adjust_water_rotation_low_confidence": True,
            "hash_progress_stall_and_cursor_coordinates": "authored_or_measured_public",
        },
        "cohort_feasibility": [dict(row) for row in feasibility_rows],
        "terminal": {
            "diagnostic2_is_final_per_macro_run": True,
            "failure_action": "behavioral_kill_and_sparse_keyframe_pivot",
        },
    }
    value["sha256"] = _digest(value)
    return value


def validate_controller_amendment(value: Mapping[str, object]) -> None:
    amendment = dict(value)
    digest = amendment.pop("sha256", None)
    if not isinstance(digest, str) or digest != _digest(amendment):
        raise RuntimeError("Cycle 8 controller amendment digest mismatch")
    if amendment.get("schema") != (
        "robocasa-inspect-goal30-cycle8-controller-amendment/v1"
    ):
        raise RuntimeError("Cycle 8 controller amendment schema mismatch")
    if amendment.get("fable_verdict") != "ACCEPT":
        raise RuntimeError("Cycle 8 controller amendment was not accepted")
    calibration = amendment.get("calibration")
    rows = amendment.get("cohort_feasibility")
    terminal = amendment.get("terminal")
    if not isinstance(calibration, Mapping) or not isinstance(rows, list):
        raise TypeError("Cycle 8 controller amendment is incomplete")
    if calibration.get("translation_controller_gain") != 3.5:
        raise RuntimeError("Cycle 8 translation calibration drifted")
    if calibration.get("rotation_controller_gain") != 3.5:
        raise RuntimeError("Cycle 8 rotation calibration drifted")
    if calibration.get("maximum_realized_translation_per_step_m") != 0.005:
        raise RuntimeError("Cycle 8 translation guard drifted")
    if calibration.get("maximum_realized_rotation_per_step_rad") != 0.025:
        raise RuntimeError("Cycle 8 rotation guard drifted")
    if len(rows) != 6 or {str(row.get("task")) for row in rows} != {
        "OpenFridgeDrawer",
        "CloseFridge",
        "OpenBlenderLid",
        "AdjustWaterTemperature",
        "PickPlaceCounterToDrawer",
        "UtensilShuffle",
    }:
        raise RuntimeError("Cycle 8 cohort feasibility is incomplete")
    for row in rows:
        if row.get("task") == "UtensilShuffle":
            if row.get("feasibility_mode") != "semantic_step_budget":
                raise RuntimeError("Cycle 8 semantic feasibility drifted")
            continue
        if row.get("fits_reserved_controller_steps") is not True:
            raise RuntimeError("Cycle 8 controller calibration exceeds a reserve")
        expected = max(
            math.ceil(float(row["translation_component_steps"]) / 0.784595),
            math.ceil(float(row["rotation_component_steps"]) / 0.7376705),
        )
        if row.get("predicted_controller_steps") != expected:
            raise RuntimeError("Cycle 8 feasibility arithmetic drifted")
    if not isinstance(terminal, Mapping) or terminal.get(
        "diagnostic2_is_final_per_macro_run"
    ) is not True:
        raise RuntimeError("Cycle 8 diagnostic terminal condition drifted")
