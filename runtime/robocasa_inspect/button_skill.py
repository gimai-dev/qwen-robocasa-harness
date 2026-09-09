"""Public-RGB identity and retreat contracts for appliance-button skills."""

from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

IMAGE_SIZE_PX = 256
MIN_BORDER_MARGIN_PX = 12.0
MAX_ADVISORY_DISAGREEMENT_PX = 12.0
BUTTON_RETREAT_CLEARANCE_M = 0.18
VERTICAL_PRESS_TASKS = frozenset(
    {
        "TurnOnElectricKettle",
        "TurnOnToaster",
    }
)
PRESS_TASKS = frozenset(
    {
        "TurnOnMicrowave",
        "TurnOffMicrowave",
        "TurnOnBlender",
        "OpenElectricKettleLid",
        *VERTICAL_PRESS_TASKS,
    }
)


def task_press_direction_world(task: str, surface_direction_world: object) -> np.ndarray:
    """Return the reviewed press axis without exposing scene or task state.

    Face controls retain the public-RGB-derived surface normal. Official lever tasks
    explicitly say "press down", so their task-level recipe replaces only the motion
    axis with world -Z after Qwen has grounded the visible lever.
    """

    if task not in PRESS_TASKS:
        raise ValueError("task is not a reviewed press task")
    if task in VERTICAL_PRESS_TASKS:
        return np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    direction = np.asarray(surface_direction_world, dtype=np.float64).copy()
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError("press direction is invalid")
    direction[2] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("press direction is degenerate")
    return direction / norm


@dataclass(frozen=True)
class ButtonIdentityLock:
    view: str
    target_uv_px: tuple[float, float]
    authority: str
    advisory_consistent: bool
    advisory_disagreement_px: float | None


@dataclass(frozen=True)
class ButtonRefinement:
    anchor_uv_px: tuple[float, float]
    area_px: int
    semantic_disagreement_px: float


@dataclass(frozen=True)
class ButtonCandidate:
    label: str
    center_uv_px: tuple[float, float]
    area_px: int


@dataclass(frozen=True)
class MicrowaveButtonPair:
    layout: str
    candidates: tuple[ButtonCandidate, ButtonCandidate]
    crop_xywh: tuple[int, int, int, int]


def button_candidate_system_prompt(task: str) -> str:
    if task == "TurnOnMicrowave":
        requested = "START, POWER, or QUICK-START control that turns the microwave on"
        rejected = "STOP, CANCEL"
    elif task == "TurnOffMicrowave":
        requested = "STOP or CANCEL control that turns the microwave off"
        rejected = "START, POWER, QUICK-START"
    else:
        raise ValueError("task has no reviewed button-candidate prompt")
    return f"""You are the semantic control-panel
disambiguation component of an Inspect-style robot controller. You receive exactly
three public images in this order: the official left external camera, the official
right external camera, and a deterministic official-RGB panel zoom annotated with two
detected physical button candidates: A in cyan and B in magenta. Choose which candidate
is the {requested}. Reject {rejected}, displays, number keys, timers, dials, doors, and
handles. On the detected dense
keypad horizontal-pair layout, the reviewed visible appliance convention is START on
the left and STOP/CANCEL on the right. Return exactly one JSON object:
{{"kind":"select_button_candidate","observation_id":"...","candidate":"A|B","evidence":"short visible-layout reason"}}
Use only the supplied public RGB derivation and reviewed layout rule. Do not return a
pixel coordinate, motion, depth, hidden state, done, or success."""


BUTTON_CANDIDATE_SYSTEM_PROMPT = button_candidate_system_prompt("TurnOnMicrowave")


def _valid_pixel(value: object, *, label: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"{label} must be two finite pixels")
    if np.any(point < MIN_BORDER_MARGIN_PX) or np.any(
        point > IMAGE_SIZE_PX - 1 - MIN_BORDER_MARGIN_PX
    ):
        raise ValueError(f"{label} is outside the certified image interior")
    return point


def preserve_tracked_button_identity(
    *,
    view: str,
    tracked_uv_px: object,
    advisory_view: str,
    advisory_uv_px: object,
    authority: str = "initial_qwen_plus_public_optical_flow",
) -> ButtonIdentityLock:
    """Keep a continuously tracked button; use a later VLM citation as an audit.

    A second semantic call cannot replace an identity that has already been grounded
    by Qwen and propagated through accepted, bounded camera motion with public RGB
    optical flow. This prevents a fresh citation from silently jumping to a nearby
    cabinet or control while still recording whether the advisory citation agrees.
    """

    if view not in {"left", "right"} or advisory_view not in {"left", "right"}:
        raise ValueError("button identity requires an official external view")
    if authority not in {
        "initial_qwen_plus_public_optical_flow",
        "initial_qwen_closed_pair_plus_public_optical_flow",
    }:
        raise ValueError("button identity authority is not reviewed")
    tracked = _valid_pixel(tracked_uv_px, label="tracked button")
    advisory = _valid_pixel(advisory_uv_px, label="advisory button")
    disagreement = (
        float(np.linalg.norm(tracked - advisory)) if view == advisory_view else None
    )
    consistent = (
        disagreement is not None
        and disagreement <= MAX_ADVISORY_DISAGREEMENT_PX
    )
    return ButtonIdentityLock(
        view=view,
        target_uv_px=(float(tracked[0]), float(tracked[1])),
        authority=authority,
        advisory_consistent=consistent,
        advisory_disagreement_px=disagreement,
    )


def _neutral_components(
    image: np.ndarray, semantic: np.ndarray, *, radius: int
) -> list[tuple[float, int, float, float, int, int]]:
    x0 = max(0, int(np.floor(semantic[0])) - radius)
    x1 = min(IMAGE_SIZE_PX, int(np.ceil(semantic[0])) + radius + 1)
    y0 = max(0, int(np.floor(semantic[1])) - radius)
    y1 = min(IMAGE_SIZE_PX, int(np.ceil(semantic[1])) + radius + 1)
    crop = image[y0:y1, x0:x1].astype(np.float64)
    gray = np.mean(crop, axis=2)
    neutral = np.ptp(crop, axis=2) <= 8.0
    threshold = max(60.0, float(np.median(gray)) + 18.0)
    mask = neutral & (gray >= threshold)
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[tuple[float, int, float, float, int, int]] = []
    height, width = mask.shape
    for row in range(height):
        for column in range(width):
            if not mask[row, column] or visited[row, column]:
                continue
            stack = [(row, column)]
            visited[row, column] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                current_row, current_column = stack.pop()
                pixels.append((current_row, current_column))
                for row_delta in (-1, 0, 1):
                    for column_delta in (-1, 0, 1):
                        next_row = current_row + row_delta
                        next_column = current_column + column_delta
                        if (
                            0 <= next_row < height
                            and 0 <= next_column < width
                            and not visited[next_row, next_column]
                            and mask[next_row, next_column]
                        ):
                            visited[next_row, next_column] = True
                            stack.append((next_row, next_column))
            rows = np.asarray([pixel[0] for pixel in pixels], dtype=np.float64)
            columns = np.asarray([pixel[1] for pixel in pixels], dtype=np.float64)
            component_height = int(np.ptp(rows) + 1)
            component_width = int(np.ptp(columns) + 1)
            if not 4 <= len(pixels) <= 100:
                continue
            if component_height > 18 or component_width > 18:
                continue
            u = float(np.mean(columns) + x0)
            v = float(np.mean(rows) + y0)
            distance = float(np.linalg.norm(np.asarray([u, v]) - semantic))
            components.append(
                (
                    distance,
                    len(pixels),
                    u,
                    v,
                    component_width,
                    component_height,
                )
            )
    return components


def detect_microwave_button_pair(
    image: object, semantic_uv_px: object
) -> MicrowaveButtonPair | None:
    """Detect a cited dense-keypad START/STOP pair from official RGB only."""

    rgb = np.asarray(image)
    if rgb.shape != (IMAGE_SIZE_PX, IMAGE_SIZE_PX, 3):
        raise ValueError("button-pair detection requires one 256x256 RGB image")
    semantic = _valid_pixel(semantic_uv_px, label="semantic button")
    candidates = [
        component
        for component in _neutral_components(rgb, semantic, radius=32)
        if 12 <= component[1] <= 80
        and 4 <= component[4] <= 12
        and 3 <= component[5] <= 12
        and component[0] <= 32.0
    ]
    pairs: list[
        tuple[float, tuple[float, int, float, float, int, int], tuple[float, int, float, float, int, int]]
    ] = []
    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            left, right = sorted((first, second), key=lambda item: item[2])
            separation = right[2] - left[2]
            vertical_error = abs(right[3] - left[3])
            area_ratio = min(left[1], right[1]) / max(left[1], right[1])
            if not 12.0 <= separation <= 28.0 or vertical_error > 4.0:
                continue
            if area_ratio < 0.6:
                continue
            if abs(left[4] - right[4]) > 2 or abs(left[5] - right[5]) > 2:
                continue
            if min(left[0], right[0]) > 6.0:
                continue
            score = min(left[0], right[0]) + vertical_error + abs(separation - 18.0)
            pairs.append((score, left, right))
    if not pairs:
        return None
    _, left, right = min(pairs, key=lambda item: item[0])
    midpoint = np.asarray([(left[2] + right[2]) / 2, (left[3] + right[3]) / 2])
    crop_size = 64
    crop_x = int(np.clip(round(midpoint[0]) - crop_size // 2, 0, 256 - crop_size))
    crop_y = int(np.clip(round(midpoint[1]) - crop_size // 2, 0, 256 - crop_size))
    return MicrowaveButtonPair(
        layout="dense_keypad_horizontal_pair",
        candidates=(
            ButtonCandidate("A", (left[2], left[3]), left[1]),
            ButtonCandidate("B", (right[2], right[3]), right[1]),
        ),
        crop_xywh=(crop_x, crop_y, crop_size, crop_size),
    )


def annotate_microwave_button_pair(
    image: object, pair: MicrowaveButtonPair
) -> bytes:
    """Return a deterministic annotated zoom derived only from official RGB."""

    rgb = np.asarray(image)
    if rgb.shape != (IMAGE_SIZE_PX, IMAGE_SIZE_PX, 3):
        raise ValueError("button-pair annotation requires one 256x256 RGB image")
    x, y, width, height = pair.crop_xywh
    crop = Image.fromarray(rgb).crop((x, y, x + width, y + height)).resize(
        (256, 256), resample=Image.Resampling.NEAREST
    )
    draw = ImageDraw.Draw(crop)
    colors = {"A": (0, 255, 255), "B": (255, 0, 255)}
    for candidate in pair.candidates:
        center_x = (candidate.center_uv_px[0] - x) * 4
        center_y = (candidate.center_uv_px[1] - y) * 4
        color = colors[candidate.label]
        draw.ellipse(
            (center_x - 16, center_y - 16, center_x + 16, center_y + 16),
            outline=color,
            width=5,
        )
        draw.text(
            (center_x - 12, center_y - 42),
            candidate.label,
            fill=color,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    encoded = io.BytesIO()
    crop.save(encoded, format="PNG")
    return encoded.getvalue()


def decode_button_candidate(
    command: Mapping[str, object], *, observation_id: str
) -> str:
    """Decode one observation-bound choice from the closed A/B vocabulary."""

    if set(command) != {"kind", "observation_id", "candidate", "evidence"}:
        raise ValueError("button candidate response violates the closed schema")
    if command.get("kind") != "select_button_candidate":
        raise ValueError("button candidate response has the wrong kind")
    if command.get("observation_id") != observation_id:
        raise ValueError("button candidate response cites a stale observation")
    candidate = command.get("candidate")
    evidence = command.get("evidence")
    if candidate not in {"A", "B"}:
        raise ValueError("button candidate must be A or B")
    if not isinstance(evidence, str) or not 1 <= len(evidence) <= 256:
        raise ValueError("button candidate evidence must be concise text")
    return candidate


def refine_neutral_microwave_button(
    image: object, semantic_uv_px: object
) -> ButtonRefinement:
    """Refine a Qwen citation to the nearest small neutral control highlight.

    Official microwave start controls render as a compact achromatic raised patch.
    The neighboring stop indicator can be red, so chromatic pixels are deliberately
    excluded. The semantic citation still supplies the appliance/control identity;
    this routine only performs bounded, local public-RGB subpixel refinement.
    """

    rgb = np.asarray(image)
    if rgb.shape != (IMAGE_SIZE_PX, IMAGE_SIZE_PX, 3):
        raise ValueError("button refinement requires one 256x256 RGB image")
    semantic = _valid_pixel(semantic_uv_px, label="semantic button")
    radius = 18
    x0 = max(0, int(np.floor(semantic[0])) - radius)
    x1 = min(IMAGE_SIZE_PX, int(np.ceil(semantic[0])) + radius + 1)
    y0 = max(0, int(np.floor(semantic[1])) - radius)
    y1 = min(IMAGE_SIZE_PX, int(np.ceil(semantic[1])) + radius + 1)
    crop = rgb[y0:y1, x0:x1].astype(np.float64)
    gray = np.mean(crop, axis=2)
    neutral = np.ptp(crop, axis=2) <= 8.0
    threshold = max(60.0, float(np.median(gray)) + 18.0)
    mask = neutral & (gray >= threshold)
    visited = np.zeros(mask.shape, dtype=bool)
    candidates: list[tuple[float, int, float, float]] = []
    height, width = mask.shape
    for row in range(height):
        for column in range(width):
            if not mask[row, column] or visited[row, column]:
                continue
            stack = [(row, column)]
            visited[row, column] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                current_row, current_column = stack.pop()
                pixels.append((current_row, current_column))
                for row_delta in (-1, 0, 1):
                    for column_delta in (-1, 0, 1):
                        next_row = current_row + row_delta
                        next_column = current_column + column_delta
                        if (
                            0 <= next_row < height
                            and 0 <= next_column < width
                            and not visited[next_row, next_column]
                            and mask[next_row, next_column]
                        ):
                            visited[next_row, next_column] = True
                            stack.append((next_row, next_column))
            if not 4 <= len(pixels) <= 80:
                continue
            rows = np.asarray([pixel[0] for pixel in pixels], dtype=np.float64)
            columns = np.asarray([pixel[1] for pixel in pixels], dtype=np.float64)
            if np.ptp(rows) + 1 > 16 or np.ptp(columns) + 1 > 16:
                continue
            u = float(np.mean(columns) + x0)
            v = float(np.mean(rows) + y0)
            distance = float(np.linalg.norm(np.asarray([u, v]) - semantic))
            if distance <= radius:
                candidates.append((distance, len(pixels), u, v))
    if not candidates:
        raise ValueError("no local neutral button component matches the citation")
    distance, area, u, v = min(candidates, key=lambda item: (item[0], -item[1]))
    return ButtonRefinement((u, v), area, distance)


def _refine_red_microwave_stop(
    image: object, semantic_uv_px: object
) -> ButtonRefinement:
    """Refine a public semantic citation to the nearby red STOP control."""

    rgb = np.asarray(image)
    if rgb.shape != (IMAGE_SIZE_PX, IMAGE_SIZE_PX, 3):
        raise ValueError("button refinement requires one 256x256 RGB image")
    semantic = _valid_pixel(semantic_uv_px, label="semantic button")
    radius = 20
    x0 = max(0, int(np.floor(semantic[0])) - radius)
    x1 = min(IMAGE_SIZE_PX, int(np.ceil(semantic[0])) + radius + 1)
    y0 = max(0, int(np.floor(semantic[1])) - radius)
    y1 = min(IMAGE_SIZE_PX, int(np.ceil(semantic[1])) + radius + 1)
    crop = rgb[y0:y1, x0:x1].astype(np.float64)
    mask = (
        (crop[:, :, 0] >= 70.0)
        & (crop[:, :, 0] >= crop[:, :, 1] + 20.0)
        & (crop[:, :, 0] >= crop[:, :, 2] + 20.0)
    )
    rows, columns = np.nonzero(mask)
    if not 4 <= len(rows) <= 80:
        raise ValueError("no local red stop component matches the citation")
    u = float(np.mean(columns) + x0)
    v = float(np.mean(rows) + y0)
    distance = float(np.linalg.norm(np.asarray([u, v]) - semantic))
    if distance > radius:
        raise ValueError("red stop component is outside the refinement radius")
    return ButtonRefinement((u, v), len(rows), distance)


def refine_task_microwave_button(
    image: object, semantic_uv_px: object, *, task: str
) -> ButtonRefinement:
    if task == "TurnOnMicrowave":
        return refine_neutral_microwave_button(image, semantic_uv_px)
    if task == "TurnOffMicrowave":
        return _refine_red_microwave_stop(image, semantic_uv_px)
    raise ValueError("task has no reviewed button refinement")


def button_retreat_goal(button_world_m: object, push_axis_world: object) -> np.ndarray:
    """Return a straight, robot-side retreat beyond RoboCasa's 15 cm gate."""

    button = np.asarray(button_world_m, dtype=np.float64)
    axis = np.asarray(push_axis_world, dtype=np.float64)
    if (
        button.shape != (3,)
        or axis.shape != (3,)
        or not np.isfinite(button).all()
        or not np.isfinite(axis).all()
    ):
        raise ValueError("button retreat geometry is invalid")
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("button retreat axis is degenerate")
    return button - BUTTON_RETREAT_CLEARANCE_M * (axis / norm)
