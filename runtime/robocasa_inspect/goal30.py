"""Frozen authority and evidence helpers for the 30%-success program."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

SPLIT_SALT = "robocasa-inspect-goal30/v1/held-out"
REQUIRED_SEEDS = (7, 11)
SLICE_SEEDS = (7, 11, 19)
MINIMUM_SUCCESSES = 30
MAX_MODEL_DECISIONS = 90
MAX_WALL_SECONDS = 1200


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _selection_tasks(selection: object) -> list[dict[str, str]]:
    if not isinstance(selection, Mapping):
        raise ValueError("selection must be an object")
    tasks = selection.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 100:
        raise ValueError("selection must contain exactly 100 tasks")
    normalized: list[dict[str, str]] = []
    for item in tasks:
        if not isinstance(item, Mapping) or set(item) != {"task", "category"}:
            raise ValueError("selection task fields are not closed")
        task = item["task"]
        category = item["category"]
        if not isinstance(task, str) or category not in {"atomic", "composite"}:
            raise ValueError("selection task is invalid")
        normalized.append({"task": task, "category": category})
    if len({item["task"] for item in normalized}) != 100:
        raise ValueError("selection tasks must be unique")
    return normalized


def _split(tasks: Sequence[str]) -> tuple[list[str], list[str]]:
    ranked = sorted(
        tasks,
        key=lambda task: hashlib.sha256(f"{SPLIT_SALT}:{task}".encode()).digest(),
    )
    held_out = sorted(ranked[:20])
    development = sorted(set(tasks) - set(held_out))
    return development, held_out


def _slice_candidates(tasks: Sequence[str], slice_name: str) -> list[str]:
    keywords = {
        "articulated": (
            "drawer",
            "cabinet",
            "fridge",
            "microwave",
            "oven",
            "dishwasher",
            "lid",
            "rack",
            "mixerhead",
        ),
        "controls": (
            "turnon",
            "turnoff",
            "adjust",
            "preheat",
            "lowerheat",
            "startcoffee",
            "sinkspout",
        ),
        "grasp_place": (
            "pickplace",
            "arrange",
            "pack",
            "setup",
            "gather",
            "store",
            "transfer",
            "distribute",
            "restock",
        ),
    }[slice_name]
    matching = [
        task for task in tasks if any(key in task.lower() for key in keywords)
    ]
    return sorted(
        matching,
        key=lambda task: hashlib.sha256(
            f"robocasa-inspect-goal30/v1/gate/{slice_name}:{task}".encode()
        ).digest(),
    )


def _slice_gates(development: Sequence[str], held_out: Sequence[str]) -> dict[str, object]:
    gates: dict[str, object] = {}
    for slice_name in ("articulated", "controls", "grasp_place"):
        dev = _slice_candidates(development, slice_name)[:3]
        reserve = _slice_candidates(held_out, slice_name)[:3]
        if len(dev) != 3 or len(reserve) != 3:
            raise ValueError(f"insufficient {slice_name} tasks for fixed slice gate")
        gates[slice_name] = {
            "development_tasks": dev,
            "held_out_tasks": reserve,
        }
    return gates


def _baseline_facts(result: object, *, raw_sha256: str) -> dict[str, object]:
    if not isinstance(result, Mapping):
        raise ValueError("baseline result must be an object")
    required = {
        "schema": "robocasa-inspect-100-task-eval/v1",
        "seed": 7,
        "selected": 100,
        "completed": 100,
        "successes": 3,
        "complete": True,
    }
    if any(result.get(key) != value for key, value in required.items()):
        raise ValueError("baseline is not the frozen 3/100 seed-7 run")
    if result.get("success_rate") != 0.03:
        raise ValueError("baseline success rate is not 3/100")
    records = result.get("records")
    if not isinstance(records, list) or len(records) != 100:
        raise ValueError("baseline records are incomplete")
    return {
        "seed": 7,
        "selected": 100,
        "successes": 3,
        "success_rate": 0.03,
        "status_counts": dict(
            sorted(Counter(str(item.get("status")) for item in records).items())
        ),
        "sha256": raw_sha256,
    }


def build_goal_authority(
    *,
    selection_path: Path,
    baseline_result_path: Path,
    execution_authority: Mapping[str, object],
) -> dict[str, object]:
    """Build the immutable 30%-goal contract before behavior development."""
    selection_raw = selection_path.read_bytes()
    baseline_raw = baseline_result_path.read_bytes()
    selection = _load_json(selection_path)
    task_items = _selection_tasks(selection)
    task_names = [item["task"] for item in task_items]
    development, held_out = _split(task_names)
    execution = dict(execution_authority)
    if not execution or not isinstance(execution.get("sha256"), str):
        raise ValueError("execution authority is incomplete")
    authority: dict[str, object] = {
        "schema": "robocasa-inspect-goal30-authority/v1",
        "goal": {
            "minimum_successes": MINIMUM_SUCCESSES,
            "denominator": 100,
            "required_seeds": list(REQUIRED_SEEDS),
        },
        "selection": {
            "schema": selection.get("schema"),
            "tasks": task_items,
            "sha256": _sha256_bytes(selection_raw),
        },
        "split": {
            "algorithm": "lowest-20-sha256-ranks-are-held-out",
            "salt": SPLIT_SALT,
            "development_tasks": development,
            "held_out_tasks": held_out,
            "held_out_reporting": "sealed-until-dev80-reaches-30-successes",
        },
        "baseline": _baseline_facts(
            _load_json(baseline_result_path), raw_sha256=_sha256_bytes(baseline_raw)
        ),
        "execution_authority": execution,
        "limits": {
            "max_model_decisions_per_task": MAX_MODEL_DECISIONS,
            "max_wall_seconds_per_task": MAX_WALL_SECONDS,
        },
        "slice_gate": {
            "seeds": list(SLICE_SEEDS),
            "required_successes_per_task": 2,
            "required_tasks_per_split": 2,
            "task_count_per_split": 3,
            "existing_success_sentinels": [
                "CloseDrawer",
                "TurnOffMicrowave",
                "TurnOnMicrowave",
            ],
            "zero_safety_aborts": True,
            "tasks": _slice_gates(development, held_out),
        },
        "revision_policy": {
            "strike_unit": "aggregate-fixed-dev-milestone-or-task-seed-regression",
            "max_failed_revisions_before_fable_rereview": 3,
            "diagnostic_double_decision_budget_is_non_candidate": True,
            "restore_last_pinned_bundle_on_regression": True,
        },
    }
    authority["sha256"] = _sha256_bytes(_canonical_json(authority))
    return authority


def validate_goal_authority(
    expected: Mapping[str, object],
    *,
    selection_path: Path,
    baseline_result_path: Path,
    execution_authority: Mapping[str, object],
) -> None:
    current = build_goal_authority(
        selection_path=selection_path,
        baseline_result_path=baseline_result_path,
        execution_authority=execution_authority,
    )
    if current != dict(expected):
        raise RuntimeError("goal authority drift")


def _milestones(receipts: object, success: bool) -> dict[str, bool]:
    accepted = [
        item
        for item in receipts if isinstance(item, Mapping) and item.get("accepted") is True
    ] if isinstance(receipts, list) else []
    closed_at = next(
        (
            index
            for index, item in enumerate(accepted)
            if item.get("gripper") in {"closed", "close"}
        ),
        None,
    )
    after_close = accepted[closed_at + 1 :] if closed_at is not None else []
    transport = any(
        isinstance(item.get("translation_m"), list)
        and any(abs(float(value)) > 1e-6 for value in item["translation_m"])
        for item in after_close
    )
    release = any(item.get("gripper") in {"open", "release"} for item in after_close)
    return {
        "accepted_action": bool(accepted),
        "gripper_closed": closed_at is not None,
        "transport_after_close": transport,
        "release_after_close": release,
        "official_success": success,
    }


def _earliest_failure(milestones: Mapping[str, bool]) -> str | None:
    names = {
        "accepted_action": "action_execution",
        "gripper_closed": "contact_or_actuation",
        "transport_after_close": "transport",
        "release_after_close": "release",
        "official_success": "official_completion",
    }
    for key, label in names.items():
        if not milestones[key]:
            return label
    return None


def build_milestone_report(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Extract sanitized progress proxies; never preserve requests or model text."""
    tasks: list[dict[str, object]] = []
    for record in records:
        result_path = Path(str(record.get("episode_result", "")))
        episode: Mapping[str, object] = {}
        if result_path.is_file():
            loaded = _load_json(result_path)
            if isinstance(loaded, Mapping):
                episode = loaded
        success = record.get("success") is True
        milestones = _milestones(episode.get("receipts", []), success)
        tasks.append(
            {
                "task": str(record.get("task")),
                "category": str(record.get("category")),
                "status": str(record.get("status")),
                "milestones": milestones,
                "earliest_failure": _earliest_failure(milestones),
                "counts": {
                    "accepted_receipts": len(
                        [
                            item
                            for item in episode.get("receipts", [])
                            if isinstance(item, Mapping)
                            and item.get("accepted") is True
                        ]
                    )
                    if isinstance(episode.get("receipts"), list)
                    else 0,
                    "model_decisions": int(episode.get("model_decisions", 0)),
                    "simulator_steps": int(episode.get("simulator_steps", 0)),
                },
            }
        )
    return {
        "schema": "robocasa-inspect-goal30-milestones/v1",
        "tasks": tasks,
        "earliest_failure_counts": dict(
            sorted(Counter(str(item["earliest_failure"]) for item in tasks).items())
        ),
    }


def validate_candidate_result(
    result: Mapping[str, object],
    *,
    selection: Mapping[str, object],
    required_seed: int,
) -> None:
    expected_tasks = [item["task"] for item in _selection_tasks(selection)]
    records = result.get("records")
    actual_tasks = (
        [item.get("task") for item in records if isinstance(item, Mapping)]
        if isinstance(records, list)
        else []
    )
    if (
        result.get("schema") != "robocasa-inspect-100-task-eval/v1"
        or result.get("seed") != required_seed
        or result.get("selected") != 100
        or result.get("completed") != 100
        or result.get("complete") is not True
        or actual_tasks != expected_tasks
    ):
        raise ValueError("candidate is not a complete exact-cohort result")
    successes = sum(
        item.get("success") is True for item in records if isinstance(item, Mapping)
    )
    if result.get("successes") != successes or successes < MINIMUM_SUCCESSES:
        raise ValueError("candidate must contain at least 30 successes")
    if result.get("success_rate") != successes / 100:
        raise ValueError("candidate success rate is inconsistent")
