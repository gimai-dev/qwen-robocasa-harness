"""Official RGB reference frames for goal-conditioned recovery."""

from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

REFERENCE_CAMERAS = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
)

REFERENCE_ALIGNMENT_RULE = "last-acknowledged-emitted-source-step/v1"
MIN_RETRIEVAL_SSIM = 0.30
MAX_RETRIEVAL_EEF_DISTANCE_M = 0.10


@dataclass(frozen=True)
class RetrievalCandidate:
    episode_index: int
    mean_external_ssim: float
    eef_distance_m: float
    eef_orientation_error_rad: float = 0.0
    stationary_base: bool = True


def aligned_reference_index(*, source_start: int, emitted_steps: int, length: int) -> int:
    """Freeze the last source step already emitted and acknowledged, never look ahead."""
    if source_start < 0 or emitted_steps <= 0 or length <= 0:
        raise ValueError("reference alignment inputs are invalid")
    return min(source_start + emitted_steps - 1, length - 1)


def rank_retrieval_candidates(
    candidates: list[RetrievalCandidate],
) -> list[RetrievalCandidate]:
    """Apply the frozen same-task eligibility gate and deterministic tie break."""
    eligible = [
        row
        for row in candidates
        if row.mean_external_ssim >= MIN_RETRIEVAL_SSIM
        and row.eef_distance_m <= MAX_RETRIEVAL_EEF_DISTANCE_M
        and row.stationary_base
    ]
    return sorted(
        eligible,
        key=lambda row: (-row.mean_external_ssim, row.eef_distance_m, row.episode_index),
    )


def motion_roi_mask(reset: object, reference: object) -> np.ndarray:
    """Derive a task-region mask solely from motion in official source RGB."""
    first = _rgb(reset).astype(np.int16)
    last = _rgb(reference).astype(np.int16)
    mask = np.mean(np.abs(last - first), axis=2) >= 12.0
    # Deterministic 13x13 binary dilation retains thin handles and door edges.
    padded = np.pad(mask, 6)
    dilated = np.zeros_like(mask)
    for dy in range(13):
        for dx in range(13):
            dilated |= padded[dy : dy + 256, dx : dx + 256]
    if int(dilated.sum()) < 64:
        raise ValueError("source RGB does not establish a legible motion region")
    return dilated


def masked_ssim_discrepancy(current: object, reference: object, mask: object) -> float:
    """Return 1-SSIM over a public RGB task mask (zero is an exact match)."""
    left = _rgb(current).astype(np.float64) / 255.0
    right = _rgb(reference).astype(np.float64) / 255.0
    selected = np.asarray(mask, dtype=np.bool_)
    if selected.shape != (256, 256) or int(selected.sum()) < 64:
        raise ValueError("reference progress mask is invalid")
    x = left[selected].reshape(-1)
    y = right[selected].reshape(-1)
    mean_x = float(x.mean())
    mean_y = float(y.mean())
    var_x = float(x.var())
    var_y = float(y.var())
    covariance = float(np.mean((x - mean_x) * (y - mean_y)))
    c1 = 0.01**2
    c2 = 0.03**2
    similarity = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2)
    )
    return float(np.clip(1.0 - similarity, 0.0, 2.0))


def _rgb(value: object) -> np.ndarray:
    image = np.asarray(value, dtype=np.uint8)
    if image.shape != (256, 256, 3):
        raise ValueError("reference camera must be 256x256 RGB")
    return image


def reference_mosaic_png(
    left: object, right: object, *, episode_index: int, frame_index: int
) -> bytes:
    """Encode two official demo views with an unambiguous non-current label."""
    if episode_index < 0 or frame_index < 0:
        raise ValueError("reference provenance must be nonnegative")
    canvas = Image.new("RGB", (512, 286), "white")
    canvas.paste(Image.fromarray(_rgb(left), mode="RGB"), (0, 30))
    canvas.paste(Image.fromarray(_rgb(right), mode="RGB"), (256, 30))
    ImageDraw.Draw(canvas).text(
        (4, 7),
        f"REFERENCE ONLY - NOT CURRENT | episode {episode_index} frame {frame_index}",
        fill="red",
    )
    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _video_frame(path: Path, timestamp_s: float) -> np.ndarray:
    if not path.is_file() or timestamp_s < 0:
        raise ValueError("official reference video provenance is invalid")
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        container.seek(
            int(timestamp_s / float(stream.time_base)),
            stream=stream,
            any_frame=False,
            backward=True,
        )
        selected: np.ndarray | None = None
        for frame in container.decode(stream):
            frame_time = float(frame.pts * stream.time_base)
            if frame_time > timestamp_s + 1e-6 and selected is not None:
                break
            selected = frame.to_ndarray(format="rgb24")
        if selected is None:
            raise ValueError("official reference frame is unavailable")
        return _rgb(selected)
    finally:
        container.close()


def load_reference_mosaic(
    dataset: Path,
    episode: Mapping[str, object],
    *,
    frame_index: int,
) -> bytes:
    """Read one same-episode public RGB keyframe from the pinned video files."""
    episode_index = int(episode["episode_index"])
    length = int(episode["length"])
    if not 0 <= frame_index < length:
        raise ValueError("reference frame index is outside the episode")
    images: list[np.ndarray] = []
    for camera in REFERENCE_CAMERAS:
        chunk = int(episode[f"videos/{camera}/chunk_index"])
        file_index = int(episode[f"videos/{camera}/file_index"])
        start = float(episode[f"videos/{camera}/from_timestamp"])
        video = dataset / f"videos/{camera}/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
        images.append(_video_frame(video, start + frame_index / 20.0))
    return reference_mosaic_png(
        images[0],
        images[1],
        episode_index=episode_index,
        frame_index=frame_index,
    )


def load_reference_views(
    dataset: Path,
    episode: Mapping[str, object],
    *,
    frame_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, Path]]:
    """Load exact source external views and their immutable file authorities."""
    length = int(episode["length"])
    if not 0 <= frame_index < length:
        raise ValueError("reference frame index is outside the episode")
    views: dict[str, np.ndarray] = {}
    paths: dict[str, Path] = {}
    for short_name, camera in zip(("left", "right"), REFERENCE_CAMERAS, strict=True):
        chunk = int(episode[f"videos/{camera}/chunk_index"])
        file_index = int(episode[f"videos/{camera}/file_index"])
        start = float(episode[f"videos/{camera}/from_timestamp"])
        path = dataset / f"videos/{camera}/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
        views[short_name] = _video_frame(path, start + frame_index / 20.0)
        paths[short_name] = path
    return views, paths
