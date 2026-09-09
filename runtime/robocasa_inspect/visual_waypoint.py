"""Source-only visual phase contracts for direct EEF policies."""

from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw

from .contracts import Command, adapt_bounded_action, decode_command
from .demonstration_skill import quaternion_delta_rotvec


@dataclass(frozen=True)
class SourceThresholds:
    dino_similarity_min: float
    ssim_discrepancy_max: float
    skip_margin: float
    gripper_width_boundary_m: float


class PhaseEvidence(NamedTuple):
    observation_id: str
    both_views_match: bool
    closer_to_next: bool
    gripper_matches: bool


@dataclass(frozen=True)
class SourcePoseSuggestion:
    translation_m: tuple[float, float, float]
    rotation_axis_angle_rad: tuple[float, float, float]
    position_error_m: float
    rotation_error_rad: float


def source_pose_suggestion(
    live_state: object,
    source_target_state: object,
    *,
    translation_cap_m: float,
    rotation_cap_rad: float,
    translation_error_max_m: float,
    rotation_error_max_rad: float,
) -> SourcePoseSuggestion:
    """Build a bounded soft prior from same-frame public robot pose only."""
    live = np.asarray(live_state, dtype=np.float64)
    target = np.asarray(source_target_state, dtype=np.float64)
    if (
        live.shape != (16,)
        or target.shape != (16,)
        or not np.isfinite(live).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("source pose suggestion requires two finite public states")
    translation = target[7:10] - live[7:10]
    rotation = quaternion_delta_rotvec(live[10:14], target[10:14])
    position_error = float(np.linalg.norm(translation))
    rotation_error = float(np.linalg.norm(rotation))
    if position_error > translation_error_max_m:
        raise ValueError("source pose translation is outside the reviewed soft-prior guard")
    if rotation_error > rotation_error_max_rad:
        raise ValueError("source pose rotation is outside the reviewed soft-prior guard")

    def capped(value: np.ndarray, limit: float) -> np.ndarray:
        norm = float(np.linalg.norm(value))
        return value if norm <= limit else value * (limit / norm)

    translation = capped(translation, translation_cap_m)
    rotation = capped(rotation, rotation_cap_rad)
    return SourcePoseSuggestion(
        translation_m=tuple(float(item) for item in translation),
        rotation_axis_angle_rad=tuple(float(item) for item in rotation),
        position_error_m=position_error,
        rotation_error_rad=rotation_error,
    )


def select_guarded_source_pose(
    live_state: object,
    source_states: object,
    *,
    start_index: int,
    stop_index: int,
    previous_index: int,
    translation_cap_m: float,
    rotation_cap_rad: float,
    translation_error_max_m: float,
    rotation_error_max_rad: float,
) -> tuple[int, SourcePoseSuggestion]:
    """Select the furthest monotone source pose inside the reviewed live guard."""
    states = np.asarray(source_states, dtype=np.float64)
    if (
        states.ndim != 2
        or states.shape[1] != 16
        or not np.isfinite(states).all()
        or not 0 <= start_index <= previous_index <= stop_index < len(states)
    ):
        raise ValueError("guarded source pose range is invalid")
    selected: tuple[int, SourcePoseSuggestion] | None = None
    for index in range(previous_index, stop_index + 1):
        try:
            suggestion = source_pose_suggestion(
                live_state,
                states[index],
                translation_cap_m=translation_cap_m,
                rotation_cap_rad=rotation_cap_rad,
                translation_error_max_m=translation_error_max_m,
                rotation_error_max_rad=rotation_error_max_rad,
            )
        except ValueError:
            continue
        selected = index, suggestion
    if selected is None:
        raise ValueError("no monotone source pose is inside the reviewed live guard")
    return selected


def action_opposes_suggestion(
    command: Command,
    suggestion: SourcePoseSuggestion,
    *,
    translation_candidate_floor_m: float,
    translation_opposed_dot_m2: float,
    rotation_candidate_floor_rad: float,
    rotation_opposed_dot_rad2: float,
) -> bool:
    """Apply the frozen same-frame opposite-direction repair predicate."""
    if command.kind != "action":
        return False
    translation = np.asarray(suggestion.translation_m)
    action_translation = np.asarray(command.translation_m)
    rotation = np.asarray(suggestion.rotation_axis_angle_rad)
    action_rotation = np.asarray(command.rotation_axis_angle_rad)
    translation_opposed = (
        np.linalg.norm(translation) >= translation_candidate_floor_m
        and float(np.dot(action_translation, translation)) < translation_opposed_dot_m2
    )
    rotation_opposed = (
        np.linalg.norm(rotation) >= rotation_candidate_floor_rad
        and float(np.dot(action_rotation, rotation)) < rotation_opposed_dot_rad2
    )
    return bool(translation_opposed or rotation_opposed)


def is_qwen_nonzero_action(command: Command, *, previous_gripper: str) -> bool:
    """Count only actual fresh Qwen-authored motion or gripper transitions."""
    return command.kind == "action" and (
        np.linalg.norm(command.translation_m) > 1e-12
        or np.linalg.norm(command.rotation_axis_angle_rad) > 1e-12
        or command.gripper not in {"hold", previous_gripper}
    )


def reference_montage_png(
    left: object, right: object, *, episode_index: int, frame_index: int
) -> bytes:
    """Fit a labeled two-view source reference into the third 256-square slot."""
    images = [np.asarray(value, dtype=np.uint8) for value in (left, right)]
    if (
        any(value.shape != (256, 256, 3) for value in images)
        or episode_index < 0
        or frame_index < 0
    ):
        raise ValueError("waypoint reference provenance is invalid")
    canvas = Image.new("RGB", (256, 256), "white")
    for index, value in enumerate(images):
        resized = Image.fromarray(value, mode="RGB").resize(
            (256, 112), resample=Image.Resampling.BICUBIC
        )
        canvas.paste(resized, (0, 28 + 112 * index))
    ImageDraw.Draw(canvas).text(
        (4, 4),
        f"REFERENCE ONLY | ep {episode_index} frame {frame_index}",
        fill="red",
    )
    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=False)
    return output.getvalue()


def decode_waypoint_command(
    value: Mapping[str, object], *, observation_id: str, terminal_ready: bool
) -> Command:
    """Keep visual-waypoint control inside direct EEF and explicit stop tools."""
    if set(value) == {"action"} and isinstance(value.get("action"), Mapping):
        candidate = dict(value["action"])
    else:
        candidate = dict(value)
    if "kind" not in candidate:
        candidate["kind"] = "action"
    note = candidate.get("note")
    if isinstance(note, str) and len(note) > 320:
        candidate["note"] = note[:320]
    if candidate.get("kind") == "action":
        command, _ = adapt_bounded_action(candidate, observation_id=observation_id)
    else:
        command = decode_command(candidate, observation_id=observation_id)
    if command.kind == "finish" and not terminal_ready:
        raise ValueError("finish is unavailable before the final reference")
    if command.kind not in {"action", "finish", "give_up"}:
        raise ValueError("waypoint policy permits only direct action or stop")
    return command


def derive_keyframe_indices(
    motion_energy: object, gripper_commands: object, *, maximum: int = 8
) -> tuple[tuple[int, ...], frozenset[int]]:
    """Choose motion quantiles while retaining every binary gripper transition."""
    motion = np.asarray(motion_energy, dtype=np.float64)
    gripper = np.asarray(gripper_commands, dtype=np.float64)
    if (
        motion.ndim != 1
        or gripper.shape != motion.shape
        or len(motion) < 2
        or not np.isfinite(motion).all()
        or not np.isfinite(gripper).all()
        or maximum < 2
    ):
        raise ValueError("source keyframe inputs are invalid")
    transitions = frozenset(
        int(index)
        for index in range(1, len(gripper))
        if bool(gripper[index] >= 0) != bool(gripper[index - 1] >= 0)
    )
    required = {0, len(motion) - 1, *transitions}
    if len(required) > maximum:
        raise ValueError("gripper transitions exceed the keyframe budget")
    selected = set(required)
    cumulative = np.cumsum(np.maximum(motion, 0.0))
    total = float(cumulative[-1])
    remaining = maximum - len(selected)
    if remaining:
        if total <= 1e-12:
            candidates = np.linspace(0, len(motion) - 1, remaining + 2)[1:-1]
            selected.update(round(value) for value in candidates)
        else:
            for fraction in np.linspace(0.0, 1.0, remaining + 2)[1:-1]:
                selected.add(int(np.searchsorted(cumulative, fraction * total)))
    if len(selected) < maximum:
        ranked = sorted(
            range(1, len(motion) - 1),
            key=lambda index: (-motion[index], index),
        )
        for index in ranked:
            selected.add(index)
            if len(selected) == maximum:
                break
    return tuple(sorted(selected)), transitions


def derive_source_thresholds(
    *,
    adjacent_dino_similarity: object,
    adjacent_ssim_discrepancy: object,
    gripper_widths: object,
) -> SourceThresholds:
    """Derive immutable progress thresholds exclusively from source episodes."""
    dino = np.asarray(adjacent_dino_similarity, dtype=np.float64)
    ssim = np.asarray(adjacent_ssim_discrepancy, dtype=np.float64)
    widths = np.sort(np.asarray(gripper_widths, dtype=np.float64))
    if (
        dino.ndim != 1
        or ssim.ndim != 1
        or widths.ndim != 1
        or min(len(dino), len(ssim), len(widths)) < 2
        or not np.isfinite(dino).all()
        or not np.isfinite(ssim).all()
        or not np.isfinite(widths).all()
    ):
        raise ValueError("source threshold samples are invalid")
    gaps = np.diff(widths)
    split = int(np.argmax(gaps))
    if float(gaps[split]) < 0.005:
        raise ValueError("source gripper states do not establish open and closed modes")
    dino_min = float(np.min(dino))
    ssim_max = float(np.max(ssim))
    if not 0.0 <= dino_min <= 1.0 or not 0.0 <= ssim_max <= 2.0:
        raise ValueError("source visual threshold is outside its metric range")
    separation = max(0.01, float(np.median(1.0 - dino)) * 0.10)
    return SourceThresholds(
        dino_similarity_min=dino_min,
        ssim_discrepancy_max=ssim_max,
        skip_margin=separation,
        gripper_width_boundary_m=float((widths[split] + widths[split + 1]) / 2.0),
    )


class VisualPhaseGovernor:
    """Two-fresh-frame phase advancement with non-skippable gripper edges."""

    def __init__(self, *, phase_count: int, gripper_transitions: set[int]) -> None:
        if phase_count < 2 or any(not 0 < item < phase_count for item in gripper_transitions):
            raise ValueError("visual phase contract is invalid")
        self.phase_count = phase_count
        self.gripper_transitions = frozenset(gripper_transitions)
        self.phase = 0
        self._last_observation_id: str | None = None
        self._matches = 0

    def observe(self, evidence: PhaseEvidence) -> int:
        if not evidence.observation_id:
            raise ValueError("phase evidence needs a fresh observation identity")
        if evidence.observation_id == self._last_observation_id:
            return self.phase
        self._last_observation_id = evidence.observation_id
        if (
            evidence.closer_to_next
            and self.phase + 1 < self.phase_count
            and self.phase + 1 not in self.gripper_transitions
        ):
            self.phase += 1
            self._matches = 0
            return self.phase
        if evidence.both_views_match and evidence.gripper_matches:
            self._matches += 1
        else:
            self._matches = 0
        if self._matches >= 2 and self.phase + 1 < self.phase_count:
            self.phase += 1
            self._matches = 0
        return self.phase
