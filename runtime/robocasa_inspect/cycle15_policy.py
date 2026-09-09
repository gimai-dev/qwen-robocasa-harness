"""Cycle 15 prompt: truthful role header plus byte-identical Cycle 14 rules."""

from __future__ import annotations

from .cycle14_policy import CYCLE14_SYSTEM_PROMPT

CYCLE15_ROLE_HEADER = r"""You are an Inspect-style source-authorizing policy
for official RoboCasa PandaOmron simulation. The first image is CURRENT official
external LEFT+RIGHT; the second is SOURCE REFERENCE official external LEFT+RIGHT;
the third is unmodified CURRENT official wrist RGB."""

_CYCLE14_HEADER, CYCLE15_DECISION_RULES = CYCLE14_SYSTEM_PROMPT.split("\n\n", 1)
CYCLE15_SYSTEM_PROMPT = CYCLE15_ROLE_HEADER + "\n\n" + CYCLE15_DECISION_RULES

__all__ = [
    "CYCLE15_DECISION_RULES",
    "CYCLE15_ROLE_HEADER",
    "CYCLE15_SYSTEM_PROMPT",
]
