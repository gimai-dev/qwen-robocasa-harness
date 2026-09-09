"""Frozen task assignment and evidence gates for the skill-harness evaluation."""

from __future__ import annotations

from collections.abc import Sequence

from .task_recipes import TASK_RECIPES

HARNESS_TASKS = frozenset(TASK_RECIPES)
BASELINE_ZERO_ACTION_EPISODES = 50


def policy_for_task(task: str) -> str:
    """Route reviewed recipes to the skill harness and all others to Qwen."""

    return "skill_harness" if task in HARNESS_TASKS else "direct_fallback"


def build_assignment_manifest(
    cohort: Sequence[tuple[str, str]],
) -> dict[str, object]:
    """Build the immutable reviewed-harness/direct-fallback assignment contract."""
    if len(cohort) != 100:
        raise ValueError("evaluation cohort must contain exactly 100 tasks")
    names = [task for task, _ in cohort]
    if len(set(names)) != len(names):
        raise ValueError("evaluation cohort tasks must be unique")
    if not HARNESS_TASKS.issubset(names):
        raise ValueError("evaluation cohort is missing reviewed harness tasks")
    assignments = [
        {
            "task": task,
            "category": category,
            "policy": policy_for_task(task),
        }
        for task, category in cohort
    ]
    return {
        "schema": "robocasa-inspect-evaluation-contract/v2",
        "baseline": {
            "successes": 0,
            "selected": 100,
            "zero_action_episodes": BASELINE_ZERO_ACTION_EPISODES,
        },
        "pilot": {
            "required_seed": 7,
            "robustness_seed": 19,
            "optional_seed": 31,
            "replication_count": 2,
            "minimum_replicated_successes": 1,
            "target_seed7_successes": 3,
        },
        "terminal_evaluator": {
            "query": "once_after_finish",
            "intermediate_queries": False,
        },
        "assignments": assignments,
    }
