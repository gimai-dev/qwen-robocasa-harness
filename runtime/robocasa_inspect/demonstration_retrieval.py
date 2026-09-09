"""Deterministic same-task demonstration retrieval from official public reset data."""

from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .demonstration_reference import (
    RetrievalCandidate,
    load_reference_views,
    masked_ssim_discrepancy,
    rank_retrieval_candidates,
)
from .demonstration_skill import (
    MAX_INITIAL_EEF_ORIENTATION_ERROR_RAD,
    MAX_INITIAL_EEF_POSITION_ERROR_M,
    public_state_to_groot,
    quaternion_distance_rad,
)


def _current_rgb(value: bytes) -> np.ndarray:
    image = np.asarray(Image.open(io.BytesIO(value)).convert("RGB"), dtype=np.uint8)
    if image.shape != (256, 256, 3):
        raise ValueError("current official camera contract drift")
    return image


def _episode_tables(dataset: Path) -> pd.DataFrame:
    paths = sorted((dataset / "meta/episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise ValueError("official episode metadata is unavailable")
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def _trajectory_facts(
    dataset: Path, episodes: pd.DataFrame
) -> dict[int, tuple[np.ndarray, bool]]:
    result: dict[int, tuple[np.ndarray, bool]] = {}
    grouped = episodes.groupby(["data/chunk_index", "data/file_index"], sort=True)
    for (chunk, file_index), group in grouped:
        path = dataset / f"data/chunk-{int(chunk):03d}/file-{int(file_index):03d}.parquet"
        data = pd.read_parquet(
            path,
            columns=["episode_index", "frame_index", "observation.state", "action"],
        )
        first = data[data["frame_index"] == 0].set_index("episode_index")
        for episode_index in group["episode_index"]:
            value = first.loc[int(episode_index), "observation.state"]
            actions = np.stack(
                data[data["episode_index"] == int(episode_index)]["action"].to_numpy()
            ).astype(np.float64)
            stationary = bool(np.all(np.abs(actions[:, :4]) <= 1e-12))
            result[int(episode_index)] = (
                np.asarray(value, dtype=np.float64),
                stationary,
            )
    return result


def retrieve_same_task_episode(
    dataset: Path,
    *,
    task: str,
    current_images: Mapping[str, bytes],
    current_public_state: Mapping[str, object],
) -> tuple[object | None, list[dict[str, object]]]:
    """Rank official same-task episodes and return the first frozen eligible match."""
    if set(current_images) != {"left", "right", "wrist"}:
        raise ValueError("retrieval requires the exact official current camera inventory")
    episodes = _episode_tables(dataset)
    marker = f"/{task}/"
    candidates = episodes[
        episodes["source_prefix"].map(lambda value: marker in f"/{str(value).strip('/')}/")
    ]
    if candidates.empty:
        return None, []
    facts = _trajectory_facts(dataset, candidates)
    live = public_state_to_groot(dict(current_public_state))
    mask = np.zeros((256, 256), dtype=np.bool_)
    mask[16:240, 16:240] = True
    current = {name: _current_rgb(current_images[name]) for name in ("left", "right")}
    measured: list[RetrievalCandidate] = []
    for _, episode in candidates.sort_values("episode_index").iterrows():
        views, _ = load_reference_views(dataset, episode, frame_index=0)
        similarities = [
            1.0 - masked_ssim_discrepancy(current[name], views[name], mask)
            for name in ("left", "right")
        ]
        source, stationary = facts[int(episode["episode_index"])]
        measured.append(
            RetrievalCandidate(
                episode_index=int(episode["episode_index"]),
                mean_external_ssim=float(np.mean(similarities)),
                eef_distance_m=float(np.linalg.norm(live[7:10] - source[7:10])),
                eef_orientation_error_rad=quaternion_distance_rad(
                    live[10:14], source[10:14]
                ),
                stationary_base=stationary,
            )
        )
    ranked = rank_retrieval_candidates(measured)
    tracker_eligible = [
        row
        for row in ranked
        if row.eef_distance_m <= MAX_INITIAL_EEF_POSITION_ERROR_M
        and row.eef_orientation_error_rad <= MAX_INITIAL_EEF_ORIENTATION_ERROR_RAD
    ]
    evidence = [
        {
            **asdict(row),
            "eligible": row in ranked,
            "rank": ranked.index(row) if row in ranked else None,
            "tracker_start_eligible": row in tracker_eligible,
            "selected": bool(tracker_eligible and row == tracker_eligible[0]),
        }
        for row in measured
    ]
    if not tracker_eligible:
        return None, evidence
    chosen = candidates[
        candidates["episode_index"] == tracker_eligible[0].episode_index
    ]
    if len(chosen) != 1:
        raise ValueError("retrieved episode is missing or duplicated")
    return chosen.iloc[0], evidence
