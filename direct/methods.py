"""Experimental conditions. ``clean`` is the baseline; H1-H8 override hooks.

Every method keeps Qwen as the sole author of numbers. Hooks:
  prompt_suffix()            extra system-prompt text for the condition
  observe(ctx)               returns (extra_json_fields, extra_images)
  representation             "absolute" | "relative" (H2)
  slot_steps(action, current_gripper=None)  simulator steps for the slot (H4)
  after_receipt(ctx, ...)    memory / bank updates (H5-H8)
  extra_calls(ctx, ...)      additional counted Qwen calls (H3, H8)
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .actions import SLOT_STEPS

METHOD_NAMES = ("clean", "h1", "h2", "h3", "h4", "h4c", "h5", "h6", "h7", "h8")


class Method:
    name = "clean"
    representation = "absolute"

    def __init__(self, *, run: Path, config: Mapping[str, object]) -> None:
        self.run = run
        self.config = dict(config)

    def prompt_suffix(self) -> str:
        return ""

    def observe(self, ctx: Mapping[str, object]) -> tuple[dict, list[tuple[str, bytes]]]:
        return {}, []

    def images(self, ctx: Mapping[str, object], current: Mapping[str, bytes],
               previous: Mapping[str, bytes] | None) -> list[tuple[str, bytes]] | None:
        return None  # None = clean image window

    def slot_steps(self, action, current_gripper=None) -> int:
        return SLOT_STEPS

    def after_receipt(self, ctx: Mapping[str, object], action, receipt) -> None:
        return None

    def wants_revision(self, ctx, decision_record, action) -> bool:
        return False

    def finalize(self) -> dict:
        return {}


def make_method(name: str, *, run: Path, config: Mapping[str, object]) -> Method:
    if name == "clean":
        return Method(run=run, config=config)
    if "+" in name:
        from .combos import make_combo
        return make_combo(name, run=run, config=config)
    from . import harnesses
    return harnesses.make(name, run=run, config=config)
