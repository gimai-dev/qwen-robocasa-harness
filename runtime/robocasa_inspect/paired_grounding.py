"""Closed two-view semantic grounding for official RoboCasa RGB cameras."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

_FIELDS = {
    "kind",
    "observation_id",
    "feature_kind",
    "left_uv",
    "right_uv",
}
_BORDER = 12.0 / 255.0


@dataclass(frozen=True)
class PairedGrounding:
    observation_id: str
    feature_kind: str
    left_uv: tuple[float, float]
    right_uv: tuple[float, float]


def _uv(value: object, *, view: str) -> tuple[float, float]:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"{view} citation must contain two finite values")
    if np.any(point < _BORDER) or np.any(point > 1.0 - _BORDER):
        raise ValueError(f"{view} citation violates the 12-pixel border")
    return float(point[0]), float(point[1])


def decode_paired_grounding(
    value: Mapping[str, object],
    *,
    observation_id: str,
    expected_feature_kind: str,
) -> PairedGrounding:
    if set(value) != _FIELDS:
        raise ValueError("paired grounding fields do not match the closed schema")
    if value.get("kind") != "ground_feature_pair":
        raise ValueError("unsupported paired grounding kind")
    if value.get("observation_id") != observation_id:
        raise ValueError("stale observation citation")
    if value.get("feature_kind") != expected_feature_kind:
        raise ValueError("paired grounding feature kind mismatch")
    return PairedGrounding(
        observation_id=observation_id,
        feature_kind=expected_feature_kind,
        left_uv=_uv(value.get("left_uv"), view="left"),
        right_uv=_uv(value.get("right_uv"), view="right"),
    )


def paired_grounding_prompt(*, task: str, feature_kind: str, guidance: str) -> str:
    return f"""You are the visual grounding component of an Inspect-style robot
controller for official RoboCasa365. You receive exactly three synchronized 256x256
RGB images in this order: left external, right external, wrist. Return exactly one
JSON object and no prose:
{{"kind":"ground_feature_pair","observation_id":"...","feature_kind":"{feature_kind}","left_uv":[u,v],"right_uv":[u,v]}}

Copy observation_id exactly. Coordinates are normalized to [0,1] and must be at least
12 pixels from every border. Both citations must mark the same physical feature in
the two external images. Use the wrist image only as semantic context; never cite it.
Do not infer or output depth, camera calibration, world coordinates, object pose,
contacts, reward, success, actions, or any extra field. If the feature is partially
occluded, cite the same stable visible corner or edge in both views rather than two
different semantic regions.

For task {task}, ground {guidance}. The required feature kind is {feature_kind}.
The deterministic harness, not you, will validate geometry and execute motion."""
