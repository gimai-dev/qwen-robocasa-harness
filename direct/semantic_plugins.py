"""Show-Harness-style plugins: action history, subtask planning with completion checks, grasp recovery."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .policy import MAX_TOKENS_SHORT, QwenDirectClient
from .semantic import TOKENS

PROMPTS = Path(__file__).resolve().with_name("prompts")
PLAN_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["subtasks"], "properties": {"subtasks": {
    "type": "array", "minItems": 3, "maxItems": 6, "items": {"type": "object", "additionalProperties": False,
    "required": ["name", "done_when", "phase"], "properties": {"name": {"type": "string"}, "done_when": {"type": "string"},
    "phase": {"type": "string", "enum": ["approach", "align"]}}}}}}
CHECK_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["subtask_done", "token"],
                "properties": {"subtask_done": {"type": "boolean"}, "token": {"type": "string", "enum": list(TOKENS)}}}


class RecentMoves:
    def __init__(self, n: int = 8) -> None:
        self.n, self.items = n, []

    def push(self, token: str, status: str) -> None:
        self.items.append(token if status in ("completed", "partial") else f"{token}({status})")
        self.items = self.items[-self.n:]

    def text(self) -> list[str]:
        return list(self.items)


class SubgoalPlanner:
    def __init__(self, client: QwenDirectClient, run: Path) -> None:
        self.client, self.run, self.subtasks, self.index = client, run, [], 0

    def plan(self, task: str, images: Sequence[tuple[str, bytes]], system_prompt: str) -> list[dict]:
        text = (PROMPTS / "semantic_planner.txt").read_text().format(task=task)
        call = self.client.complete(system_prompt=system_prompt, user_text=text, images=list(images), response_schema=PLAN_SCHEMA,
                                    max_tokens=MAX_TOKENS_SHORT, category="plan", decision=0)
        self.subtasks = call["parsed"]["subtasks"]
        self.index = 0
        (self.run / "subtasks.json").write_text(json.dumps(self.subtasks, indent=1))
        return self.subtasks

    def current(self) -> dict | None:
        return self.subtasks[self.index] if self.index < len(self.subtasks) else None

    def advance(self) -> None:
        self.index = min(self.index + 1, len(self.subtasks))

    def rollback_to_grasp(self) -> None:
        for i, s in enumerate(self.subtasks):
            if "grasp" in s["name"].lower() or "align" in s["name"].lower():
                self.index = i
                return
        self.index = 0

    def fine(self) -> bool:
        cur = self.current()
        return bool(cur and cur["phase"] == "align")


class Recovery:
    """Empty close -> force RELEASE and roll the plan back to the grasp subtask (Show-Harness `recovery`)."""
    def __init__(self) -> None:
        self.events = 0

    def check(self, token: str, receipt: Mapping[str, object], proprio: Mapping[str, object]) -> tuple[str | None, bool]:
        if token == "GRASP" and (receipt.get("gripper_width_after_m") or 1.0) < 0.005:
            self.events += 1
            return "RELEASE", True
        return None, False
