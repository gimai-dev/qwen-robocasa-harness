"""Closed manifest validation for the Cycle 8 public-observation census."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence

from .cycle5_skill import lateral_residual_bank

ENTRY_FIELDS = {
    "entry_id",
    "source_cycle",
    "task",
    "mode",
    "schema_mode",
    "residual_id",
    "observation_path",
    "observation_sha256",
    "observation_id",
    "instruction_sha256",
    "public_state_sha256",
    "image_path",
    "image_sha256",
    "source_full",
    "source_result_sha256",
}
PUBLIC_STATE_FIELDS = {
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
}


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_corpus_manifest(
    value: Mapping[str, object], *, reserve_tasks: set[str] | None = None
) -> dict[str, object]:
    manifest = dict(value)
    digest = manifest.pop("sha256", None)
    if not isinstance(digest, str) or digest != _digest(manifest):
        raise RuntimeError("Cycle 8 corpus digest mismatch")
    if manifest.get("schema") != "robocasa-inspect-cycle8-public-corpus/v1":
        raise RuntimeError("Cycle 8 corpus schema mismatch")
    entries = manifest.get("entries")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise TypeError("Cycle 8 corpus entries are missing")
    if len(entries) != 400:
        raise RuntimeError("Cycle 8 corpus must contain exactly 400 entries")
    modes: Counter[str] = Counter()
    residuals: Counter[str] = Counter()
    entry_ids: set[str] = set()
    reserve_tasks = reserve_tasks or set()
    for item in entries:
        if not isinstance(item, Mapping) or set(item) != ENTRY_FIELDS:
            raise RuntimeError("Cycle 8 corpus entry fields drifted")
        entry_id = item.get("entry_id")
        if not isinstance(entry_id, str) or entry_id in entry_ids:
            raise RuntimeError("Cycle 8 corpus entry identity drifted")
        entry_ids.add(entry_id)
        if item.get("source_cycle") not in {"cycle5", "cycle6", "cycle7"}:
            raise RuntimeError("Cycle 8 corpus source drifted")
        task = item.get("task")
        if not isinstance(task, str) or task in reserve_tasks:
            raise RuntimeError("Cycle 8 corpus contains a reserve task")
        mode = item.get("mode")
        schema_mode = item.get("schema_mode")
        if mode == "pose":
            source = item.get("source_full")
            residual_id = item.get("residual_id")
            if not isinstance(source, Mapping) or not isinstance(residual_id, str):
                raise RuntimeError("Cycle 8 pose entry is incomplete")
            bank = lateral_residual_bank(source.get("translation_m"))
            if residual_id not in {option["residual_id"] for option in bank}:
                raise RuntimeError("Cycle 8 pose residual is outside its bank")
            if schema_mode != "pose":
                raise RuntimeError("Cycle 8 pose schema mode drifted")
            residuals[residual_id] += 1
        elif mode == "semantic":
            if item.get("source_full") is not None or item.get("residual_id") is not None:
                raise RuntimeError("Cycle 8 semantic entry leaked a pose source")
            if schema_mode not in {"custom", "give_up", "done"}:
                raise RuntimeError("Cycle 8 semantic schema mode drifted")
            modes[str(schema_mode)] += 1
        else:
            raise RuntimeError("Cycle 8 corpus mode drifted")
        image_paths = item.get("image_path")
        image_hashes = item.get("image_sha256")
        if not isinstance(image_paths, Mapping) or set(image_paths) != {
            "left",
            "right",
            "wrist",
        }:
            raise RuntimeError("Cycle 8 corpus camera inventory drifted")
        if not isinstance(image_hashes, Mapping) or set(image_hashes) != set(
            image_paths
        ):
            raise RuntimeError("Cycle 8 corpus camera hashes drifted")
    pose_count = sum(residuals.values())
    semantic_count = sum(modes.values())
    if pose_count < 200 or semantic_count < 100:
        raise RuntimeError("Cycle 8 corpus mode strata drifted")
    if len(residuals) != 13 or min(residuals.values(), default=0) < 10:
        raise RuntimeError("Cycle 8 corpus residual strata drifted")
    if any(modes[name] < 25 for name in ("custom", "give_up", "done")):
        raise RuntimeError("Cycle 8 corpus semantic strata drifted")
    return {
        "entries": len(entries),
        "pose": pose_count,
        "semantic": semantic_count,
        "residual_counts": dict(sorted(residuals.items())),
        "semantic_mode_counts": dict(sorted(modes.items())),
    }
