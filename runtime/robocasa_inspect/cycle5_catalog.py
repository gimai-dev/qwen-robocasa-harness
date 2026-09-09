"""Deterministic public-metadata retrieval authority for Cycle 5."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping

_CLASS_VOCABULARY = (
    "appliance",
    "blender",
    "cabinet",
    "counter",
    "drawer",
    "faucet",
    "fridge",
    "handle",
    "microwave",
    "object",
    "oven",
    "sink",
    "stove",
    "utensil",
)
_STOP = {
    "a",
    "an",
    "and",
    "class",
    "for",
    "in",
    "into",
    "of",
    "on",
    "only",
    "task",
    "the",
    "to",
}


def metadata_tokens(name: str, instruction_text: str) -> tuple[str, ...]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    tokens = re.findall(r"[a-z0-9]+", f"{expanded} {instruction_text}".lower())
    return tuple(sorted({token for token in tokens if token not in _STOP}))


def fixture_object_classes(name: str, instruction_text: str) -> tuple[str, ...]:
    text = " ".join(metadata_tokens(name, instruction_text))
    return tuple(item for item in _CLASS_VOCABULARY if item in text)


def horizon_path_evidence(
    *, path_length_m: float, official_horizon: int, reserve_multiplier: float = 1.30
) -> dict[str, object]:
    if not math.isfinite(path_length_m) or path_length_m < 0:
        raise ValueError("source path length must be finite and nonnegative")
    if official_horizon <= 0 or reserve_multiplier < 1:
        raise ValueError("source horizon authority is invalid")
    substeps = math.ceil(path_length_m / 0.005)
    reserved = math.ceil(reserve_multiplier * substeps)
    return {
        "path_length_m": path_length_m,
        "controller_substeps": substeps,
        "reserved_controller_steps": reserved,
        "official_horizon": official_horizon,
        "horizon_eligible": reserved <= official_horizon,
    }


def _quaternion_distance_rad(left: object, right: object) -> float:
    import numpy as np

    q0 = np.asarray(left, dtype=np.float64)
    q1 = np.asarray(right, dtype=np.float64)
    if q0.shape != (4,) or q1.shape != (4,):
        raise ValueError("source quaternion is invalid")
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    return 2.0 * math.acos(float(np.clip(abs(np.dot(q0, q1)), -1.0, 1.0)))


def cursor_path_evidence(
    states: object, *, official_horizon: int, reserve_multiplier: float = 1.30
) -> dict[str, object]:
    """Compute source feasibility analytically using the live monotone cursor rule."""
    import numpy as np

    values = np.asarray(states, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 16
        or len(values) < 1
        or not np.isfinite(values).all()
        or official_horizon <= 0
        or reserve_multiplier < 1
    ):
        raise ValueError("Cycle 7 cursor path authority is invalid")
    cursor = 0
    controller_substeps = 0
    segments: list[dict[str, object]] = []
    while cursor < len(values) - 1:
        distance = 0.0
        selected = cursor + 1
        for index in range(cursor + 1, len(values)):
            distance += float(
                np.linalg.norm(values[index, 7:10] - values[index - 1, 7:10])
            )
            selected = index
            if distance >= 0.018 or index - cursor >= 20:
                break
        translation = float(
            np.linalg.norm(values[selected, 7:10] - values[cursor, 7:10])
        )
        rotation = _quaternion_distance_rad(
            values[cursor, 10:14], values[selected, 10:14]
        )
        steps = max(
            1,
            math.ceil(translation / 0.005 - 1e-12),
            math.ceil(rotation / 0.025 - 1e-12),
        )
        controller_substeps += steps
        segments.append(
            {
                "start": cursor,
                "stop": selected,
                "translation_m": translation,
                "rotation_rad": rotation,
                "controller_substeps": steps,
            }
        )
        cursor = selected
    reserved = math.ceil(reserve_multiplier * controller_substeps)
    return {
        "controller_substeps": controller_substeps,
        "reserved_controller_steps": reserved,
        "official_horizon": official_horizon,
        "horizon_eligible": reserved <= official_horizon,
        "segments": segments,
    }


def build_catalog_mapping(
    catalog: Iterable[Mapping[str, object]], source_tasks: Iterable[str]
) -> list[dict[str, object]]:
    rows = [dict(row) for row in catalog]
    by_name = {str(row["task"]): row for row in rows}
    sources = sorted(set(source_tasks) & set(by_name))
    if not sources:
        raise ValueError("catalog mapping requires official source tasks")
    result: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda value: str(value["task"])):
        task = str(row["task"])
        tokens = set(row["tokens"])
        classes = set(row["fixture_object_classes"])
        ranked: list[tuple[tuple[object, ...], dict[str, object]]] = []
        for source in sources:
            candidate = by_name[source]
            source_tokens = set(candidate["tokens"])
            union = tokens | source_tokens
            jaccard = len(tokens & source_tokens) / len(union) if union else 0.0
            class_overlap = len(classes & set(candidate["fixture_object_classes"]))
            destination = source.rsplit("To", 1)[-1].lower() if "To" in source else ""
            instruction = str(row.get("instruction_text", "")).lower()
            # Composite docs put the positive goal before a ``Steps:`` section;
            # later lines often describe objects that must deliberately remain.
            positive_goal = instruction.split("steps:", 1)[0]
            destination_match = bool(
                destination
                and re.search(
                    rf"\b(?:in|into|to)\s+(?:the\s+)?{re.escape(destination)}\b",
                    positive_goal,
                )
            )
            score = {
                "exact_task": task == source,
                "same_module": row["module"] == candidate["module"],
                "same_affordance": row["affordance"] == candidate["affordance"],
                "destination_match": destination_match,
                "class_overlap": class_overlap,
                "token_jaccard": jaccard,
            }
            key = (
                -int(score["exact_task"]),
                -int(score["same_module"]),
                -int(destination_match),
                -int(score["same_affordance"]),
                -class_overlap,
                -jaccard,
                source,
            )
            ranked.append((key, {"source_task": source, "score": score}))
        ranked.sort(key=lambda item: item[0])
        choice = ranked[0][1]
        result.append(
            {
                "task": task,
                "source_task": choice["source_task"],
                "mode": "pose_candidate" if choice["source_task"] == task else "semantic_only",
                "score": choice["score"],
            }
        )
    return result
