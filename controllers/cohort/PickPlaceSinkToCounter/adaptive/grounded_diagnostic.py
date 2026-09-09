"""Controlled seed-1711 replay with enough low-level budget to execute close."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from .joint_runner import _atomic_json

RESULT_SCHEMA = "robocasa-grounded-panda-diagnostic/v6"
DIAGNOSTIC_TASKS = (
    "AdjustWaterTemperature",
    "OpenToasterOvenDoor",
    "PickPlaceToasterToCounter",
)
DIAGNOSTIC_SEED = 1711
MAX_DECISIONS = 16
MAX_ACTIONS = 320
MAX_WALL_S = 600.0
MIN_CARTESIAN_EFFECT_M = 0.001
MIN_CARTESIAN_ROTATION_RAD = 0.01
MIN_GRIPPER_EFFECT_M = 0.005


def _is_safety_abort(episode: Mapping[str, object]) -> bool:
    status = str(episode.get("status", "")).lower()
    return any(
        label in status
        for label in ("safety_abort", "collision_abort", "simulator_safety")
    )


def _vector(value: object, width: int) -> list[float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != width
    ):
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(item * item for item in value))


def _receipt_effect(receipt: Mapping[str, object]) -> bool:
    if receipt.get("kind") not in {
        "cartesian_delta",
        "image_servo",
    } or receipt.get("accepted") is not True:
        return False
    pose = receipt.get("end_effector_pose_delta")
    if isinstance(pose, Mapping):
        translation = _vector(pose.get("translation_m"), 3)
        rotation = _vector(pose.get("rotation_axis_angle_rad"), 3)
        if (
            translation is not None
            and _norm(translation) >= MIN_CARTESIAN_EFFECT_M
        ) or (
            rotation is not None
            and _norm(rotation) >= MIN_CARTESIAN_ROTATION_RAD
        ):
            return True
    residual = receipt.get("gripper_residual")
    if not isinstance(residual, Mapping):
        return False
    start = _vector(residual.get("start_qpos"), 2)
    end = _vector(residual.get("end_qpos"), 2)
    return (
        start is not None
        and end is not None
        and abs(abs(end[0] - end[1]) - abs(start[0] - start[1]))
        >= MIN_GRIPPER_EFFECT_M
    )


def _close_effect(receipt: Mapping[str, object]) -> bool:
    if receipt.get("requested_gripper") != "close":
        return False
    residual = receipt.get("gripper_residual")
    if not isinstance(residual, Mapping):
        return False
    start = _vector(residual.get("start_qpos"), 2)
    end = _vector(residual.get("end_qpos"), 2)
    return (
        start is not None
        and end is not None
        and abs(end[0] - end[1]) <= abs(start[0] - start[1]) - MIN_GRIPPER_EFFECT_M
    )


def _beyond_approach(episode: Mapping[str, object]) -> bool:
    history = episode.get("milestone_history")
    return isinstance(history, list) and any(
        milestone in {"pregrasp", "grasp", "transport", "release", "engage", "actuate"}
        for milestone in history
    )


def evaluate_grounded_diagnostic(
    episodes: Sequence[Mapping[str, object]],
) -> tuple[bool, dict[str, bool], dict[str, int]]:
    exact_tasks = [episode.get("task") for episode in episodes] == list(DIAGNOSTIC_TASKS)
    safe = len(episodes) == 3 and all(
        not _is_safety_abort(episode)
        and type(episode.get("simulator_steps")) is int
        and 0 <= int(episode["simulator_steps"]) <= MAX_ACTIONS
        for episode in episodes
    )
    closure = len(episodes) == 3 and all(
        isinstance(episode.get("proposal_audit_closure"), Mapping)
        and episode["proposal_audit_closure"].get("valid") is True
        for episode in episodes
    )
    effect_tasks = 0
    close_tasks = 0
    for episode in episodes:
        raw_receipts = episode.get("receipts")
        receipts = (
            [receipt for receipt in raw_receipts if isinstance(receipt, Mapping)]
            if isinstance(raw_receipts, list)
            else []
        )
        effect_tasks += int(any(_receipt_effect(receipt) for receipt in receipts))
        close_tasks += int(any(_close_effect(receipt) for receipt in receipts))
    beyond = sum(_beyond_approach(episode) for episode in episodes)
    successes = sum(episode.get("success") is True for episode in episodes)
    metrics = {
        "tasks_with_cartesian_effect": effect_tasks,
        "tasks_beyond_approach": beyond,
        "tasks_with_close_effect": close_tasks,
        "official_material_successes": successes,
    }
    checks = {
        "exact_seed1711_replay_task_order": exact_tasks,
        "three_tasks_safe_and_action_bounded": safe,
        "three_proposal_closures_valid": closure,
        "three_tasks_have_cartesian_effect": effect_tasks == 3,
        "two_tasks_advance_beyond_approach": beyond >= 2,
        "one_task_has_close_effect": close_tasks >= 1,
        "one_official_material_success": successes >= 1,
    }
    return all(checks.values()), checks, metrics


def validate_public_grounded_diagnostic(
    value: object,
    *,
    expected_release_digest: str,
    expected_artifact_root: str,
) -> dict[str, object]:
    fields = {
        "schema",
        "release_digest",
        "seed",
        "tasks",
        "passed",
        "checks",
        "metrics",
        "episodes",
        "model_calls",
        "controller_calls",
        "critic_attempts",
        "artifact_root",
        "wall_s",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("grounded diagnostic schema drifted")
    result = dict(value)
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("release_digest") != expected_release_digest
        or result.get("seed") != DIAGNOSTIC_SEED
        or result.get("tasks") != list(DIAGNOSTIC_TASKS)
        or result.get("artifact_root") != expected_artifact_root
        or type(result.get("passed")) is not bool
        or not isinstance(result.get("checks"), Mapping)
        or not isinstance(result.get("metrics"), Mapping)
        or not isinstance(result.get("episodes"), list)
    ):
        raise ValueError("grounded diagnostic identity drifted")
    return result


def execute(args: argparse.Namespace) -> dict[str, object]:
    from .remote_driver import (
        _validate_baseline_path,
        _validate_critic_prompt,
        _validate_release_digest,
        run_one,
    )

    started = time.monotonic()
    if args.seed != DIAGNOSTIC_SEED:
        raise ValueError("grounded diagnostic replay seed must remain 1711")
    baseline_path = _validate_baseline_path(args.baseline)
    release_digest = _validate_release_digest(args.release_digest)
    critic_prompt = args.critic_prompt.read_text(encoding="utf-8")
    _validate_critic_prompt(critic_prompt, protocol="proposal")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    rows = {row["task"]: row for row in baseline["tasks"]}
    args.output_root.mkdir(parents=True, mode=0o700)
    episodes: list[dict[str, object]] = []
    contexts = []
    for task in DIAGNOSTIC_TASKS:
        episode, context = run_one(
            item=rows[task],
            seed=DIAGNOSTIC_SEED,
            run=args.output_root / task,
            max_decisions=MAX_DECISIONS,
            critic_prompt=critic_prompt,
            protocol="proposal",
            action_budget=MAX_ACTIONS,
            wall_budget_s=MAX_WALL_S,
        )
        episodes.append(episode)
        contexts.append(context)
    passed, checks, metrics = evaluate_grounded_diagnostic(episodes)
    result = {
        "schema": RESULT_SCHEMA,
        "release_digest": release_digest,
        "seed": DIAGNOSTIC_SEED,
        "tasks": list(DIAGNOSTIC_TASKS),
        "passed": passed,
        "checks": checks,
        "metrics": metrics,
        "episodes": episodes,
        "model_calls": sum(context.model_calls for context in contexts),
        "controller_calls": sum(context.controller_calls for context in contexts),
        "critic_attempts": sum(context.critic_attempts for context in contexts),
        "artifact_root": str(args.output_root),
        "wall_s": time.monotonic() - started,
    }
    _atomic_json(args.output_root / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--critic-prompt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--release-digest", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
