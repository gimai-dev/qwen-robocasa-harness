"""Combination conditions: compose independent harness methods (e.g. h1+h2, h3+h4).

Method name "h1+h2" builds a Combo of the member conditions. Rules:
  prompt         member suffixes concatenated
  representation "relative" if H2 is a member
  images         H1's annotated window if H1 is a member
  schema         H3's candidate schema if H3 is a member (with H4's `s` field when H4 is present),
                 else H4's, else the default
  slot_steps     H4's rule if H4 is a member
  choose         H3's preview-and-select call if H3 is a member
  observe/after  every member's hooks, merged
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .actions import SLOT_STEPS, action_schema, short_response_schema
from .methods import Method


class Combo(Method):
    def __init__(self, *, run: Path, config: Mapping[str, object], members):
        super().__init__(run=run, config=config)
        self.members = members
        self.name = "+".join(m.name for m in members)
        self.representation = "relative" if any(m.name == "h2" for m in members) else "absolute"
        for m in members:
            m.representation = self.representation
        self.h1 = next((m for m in members if m.name == "h1"), None)
        self.h3 = next((m for m in members if m.name == "h3"), None)
        self.h4 = next((m for m in members if m.name in ("h4", "h4c")), None)

    def prompt_suffix(self) -> str:
        return "\n".join(m.prompt_suffix() for m in self.members if m.prompt_suffix())

    def short_schema(self, interface):
        extra = {"s": {"type": "integer", "enum": [5, 10, 20]}} if self.h4 and self.h4.name == "h4" else None
        if self.h3:
            self.h3.interface = interface
            return {"type": "object", "additionalProperties": False,
                    "properties": {"reasoning": {"type": "string"},
                                   "candidates": {"type": "array", "minItems": 2, "maxItems": 3,
                                                  "items": action_schema(interface=interface, extra=extra)}},
                    "required": ["reasoning", "candidates"]}
        top = {}
        for m in self.members:
            if hasattr(m, "short_schema") and m.name not in ("h3", "h4"):
                schema = m.short_schema(interface)
                for k, v in schema["properties"].items():
                    if k not in ("reasoning", "action"):
                        top[k] = v
        return short_response_schema(interface=interface, action_extra=extra, top_extra=top or None)

    def observe(self, ctx):
        extra, images = {}, []
        for m in self.members:
            e, i = m.observe(ctx)
            extra.update(e); images += i
        return extra, images

    def images(self, ctx, current, previous):
        return self.h1.images(ctx, current, previous) if self.h1 else None

    def slot_steps(self, action, current_gripper=None) -> int:
        return self.h4.slot_steps(action, current_gripper=current_gripper) if self.h4 else SLOT_STEPS

    def after_receipt(self, ctx, action, receipt) -> None:
        for m in self.members:
            m.after_receipt(ctx, action, receipt)

    def finalize(self) -> dict:
        return {m.name: m.finalize() for m in self.members}


def make_combo(name: str, *, run: Path, config: Mapping[str, object]) -> Combo:
    from .methods import make_method
    members = [make_method(part, run=run, config=config) for part in name.split("+")]
    combo = Combo(run=run, config=config, members=members)
    if combo.h3:
        combo.choose = combo.h3.choose  # type: ignore[attr-defined]
    return combo
