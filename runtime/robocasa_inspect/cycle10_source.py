"""Public source-route feasibility proof for Cycle 10."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .cycle9_capacity import sparse_keyframes
from .demonstration_skill import quaternion_distance_rad


def task_family(task: str, module: str) -> str:
    lowered = task.lower()
    if "pick_place" in module.lower() or any(
        token in lowered
        for token in ("pickplace", "arrange", "gather", "store", "transfer")
    ):
        return "pick_place"
    if any(token in lowered for token in ("open", "close", "slide")):
        return "articulated"
    if any(
        token in lowered
        for token in (
            "turnon",
            "turnoff",
            "adjust",
            "preheat",
            "start",
            "lowerheat",
            "boil",
            "defrost",
        )
    ):
        return "control"
    return "composite"


def source_route_feasibility(
    states: object,
    actions: object,
    *,
    official_horizon: int,
    maximum_keyframes: int = 60,
    model_call_limit: int = 400,
    decisions_per_keyframe: int = 4,
    repair_allowance: int = 32,
    reserve_multiplier: float = 1.30,
    translation_tick_cap_m: float = 0.003,
    rotation_tick_cap_rad: float = 0.020,
) -> dict[str, object]:
    """Prove sparse source endpoints fit official action and model-call budgets."""
    state = np.asarray(states, dtype=np.float64)
    action = np.asarray(actions, dtype=np.float64)
    if (
        not 0.0 < translation_tick_cap_m <= 0.003
        or not 0.0 < rotation_tick_cap_rad <= 0.020
    ):
        raise ValueError("Cycle 10 amended servo cap is invalid")
    rows = sparse_keyframes(state, action)
    servo_ticks = 0
    previous = 0
    endpoint_records: list[dict[str, object]] = []
    workspace_ok = True
    for row in rows:
        stop = row.stop
        translation = float(np.linalg.norm(state[stop, 7:10] - state[previous, 7:10]))
        rotation = quaternion_distance_rad(state[previous, 10:14], state[stop, 10:14])
        ticks = max(
            math.ceil(translation / translation_tick_cap_m),
            math.ceil(rotation / rotation_tick_cap_rad),
        )
        servo_ticks += ticks
        endpoint = state[stop, 7:10]
        in_workspace = bool(
            -0.75 <= endpoint[0] <= 0.90
            and -0.90 <= endpoint[1] <= 0.90
            and 0.0 <= endpoint[2] <= 1.60
        )
        workspace_ok &= in_workspace
        endpoint_records.append(
            {
                "source_index": stop,
                "translation_m": translation,
                "rotation_rad": rotation,
                "nominal_servo_ticks": ticks,
                "workspace_ok": in_workspace,
            }
        )
        previous = stop
    reserved_ticks = math.ceil(reserve_multiplier * servo_ticks)
    maximum_calls = len(rows) * decisions_per_keyframe + repair_allowance
    return {
        "keyframes": len(rows),
        "nominal_servo_ticks": servo_ticks,
        "reserved_servo_ticks": reserved_ticks,
        "official_horizon": official_horizon,
        "translation_tick_cap_m": translation_tick_cap_m,
        "rotation_tick_cap_rad": rotation_tick_cap_rad,
        "maximum_model_calls": maximum_calls,
        "workspace_ok": workspace_ok,
        "keyframe_gate": len(rows) <= maximum_keyframes,
        "horizon_gate": reserved_ticks <= official_horizon,
        "model_call_gate": maximum_calls <= model_call_limit,
        "eligible": bool(
            rows
            and len(rows) <= maximum_keyframes
            and reserved_ticks <= official_horizon
            and maximum_calls <= model_call_limit
            and workspace_ok
        ),
        "endpoints": endpoint_records,
    }


def validate_source_cohort_result(
    value: Mapping[str, object],
    *,
    contract_sha256: str,
    controller_amendment_sha256: str | None = None,
) -> None:
    if value.get("schema") != "robocasa-inspect-cycle10-source-cohort/v1":
        raise RuntimeError("Cycle 10 source-cohort schema mismatch")
    if value.get("complete") is not True or value.get("passed") is not True:
        raise RuntimeError("Cycle 10 source-cohort gate is not green")
    if value.get("cycle10_contract_sha256") != contract_sha256:
        raise RuntimeError("Cycle 10 source-cohort contract drifted")
    if (
        controller_amendment_sha256 is not None
        and value.get("controller_amendment_sha256")
        != controller_amendment_sha256
    ):
        raise RuntimeError("Cycle 10 source cohort controller amendment drifted")
    if int(value.get("eligible_tasks", 0)) < 20:
        raise RuntimeError("Cycle 10 source cohort has fewer than 20 tasks")
    if len(value.get("eligible_families", [])) < 4:
        raise RuntimeError("Cycle 10 source cohort has fewer than four families")
