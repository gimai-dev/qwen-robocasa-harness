"""Frozen disclosed-development slice contract for the first 30%-goal cycle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

DEV20_SALT = "robocasa-inspect-goal30/cycle1/dev20"
FAMILY_SALT = "robocasa-inspect-goal30/cycle1/family"
FAMILY_KEYWORDS = {
    "articulated": (
        "drawer", "cabinet", "fridge", "microwave", "oven", "dishwasher", "lid", "rack",
    ),
    "controls": (
        "turnon", "turnoff", "adjust", "preheat", "lowerheat", "startcoffee", "sinkspout",
    ),
    "grasp_place": (
        "pickplace", "arrange", "pack", "setup", "gather", "store", "transfer", "restock",
    ),
}


def _digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def _rank(tasks: Sequence[str], salt: str) -> list[str]:
    return sorted(tasks, key=lambda task: hashlib.sha256(f"{salt}:{task}".encode()).digest())


def build_cycle_contract(
    goal_authority: Mapping[str, object],
    implementation_authority: Mapping[str, str],
) -> dict[str, object]:
    split = goal_authority.get("split")
    if not isinstance(split, Mapping):
        raise TypeError("goal authority split is missing")
    development = split.get("development_tasks")
    reserve = split.get("held_out_tasks")
    if not isinstance(development, list) or not isinstance(reserve, list):
        raise TypeError("goal authority task split is invalid")
    if len(development) != 80 or len(reserve) != 20:
        raise ValueError("goal authority split size drift")
    if not all(isinstance(task, str) for task in development + reserve):
        raise ValueError("goal authority task identity is invalid")
    families: dict[str, list[str]] = {}
    for name, keywords in FAMILY_KEYWORDS.items():
        matching = [
            task
            for task in development
            if any(keyword in task.lower() for keyword in keywords)
        ]
        ranked = _rank(matching, f"{FAMILY_SALT}/{name}")[:10]
        if len(ranked) != 10:
            raise ValueError(f"insufficient disclosed tasks for {name}")
        families[name] = ranked
    contract: dict[str, object] = {
        "schema": "robocasa-inspect-goal30-cycle/v1",
        "goal_authority_sha256": goal_authority.get("sha256"),
        "reserve": {"count": 20, "identities_included": False},
        "implementation_authority": dict(sorted(implementation_authority.items())),
        "dev20": _rank(development, DEV20_SALT)[:20],
        "families": families,
        "gates": {
            "dev20_minimum_successes": 6,
            "dev20_each_represented_family_minimum": 1,
            "family10_minimum_successes": 3,
            "family10_required_families": 3,
            "max_dev_iterations": 3,
            "max_full_matrix_rounds": 1,
            "full_matrix_minimum_successes_each_seed": 30,
        },
    }
    contract["sha256"] = _digest(contract)
    return contract


def validate_cycle_contract(
    contract: Mapping[str, object],
    goal_authority: Mapping[str, object],
    implementation_authority: Mapping[str, str],
) -> None:
    if dict(contract) != build_cycle_contract(goal_authority, implementation_authority):
        raise RuntimeError("goal30 cycle contract drift")
