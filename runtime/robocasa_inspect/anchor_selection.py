"""Frozen public-RGB authority for deterministic two-view robot anchors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

IMAGE_SIZE = 256
BORDER = 12
RADIUS = 32
MIN_CLUSTER = 3
MIN_NOISE = 0.5
CONFIG: dict[str, object] = {
    "schema": "robocasa-public-anchor-selection/v1",
    "image_size": IMAGE_SIZE,
    "border_px": BORDER,
    "roi_radius_px": RADIUS,
    "roi_max_corners": 64,
    "full_frame_max_corners": 256,
    "quality_level": 0.01,
    "min_distance_px": 5,
    "block_size": 3,
    "texture_floor": 1e-5,
    "min_cluster": MIN_CLUSTER,
    "magnitude_fraction": 0.25,
    "direction_deg": 15.0,
    "minimum_motion_px": 1.5,
    "noise_multiplier": 3.0,
    "maximum_forward_backward_error_px": 1.0,
    "maximum_return_fraction": 0.20,
    "maximum_reversal_cosine": -0.9,
    "return_ratio": [0.5, 2.0],
    "probe_translation_m": [0.02, 0.0, 0.0],
    "same_pose_frame_count": 5,
    "fallback_scope": "both_views",
    "component_policy": "discard_noncomplete_whole_component",
    "medoid_tie_break": "raw_float64_initial_xy_lexicographic",
}


class MotionClusterError(ValueError):
    def __init__(self, message: str, report: dict[str, object]):
        super().__init__(message)
        self.report = report


class AnchorSelectionError(ValueError):
    def __init__(self, message: str, report: dict[str, object]):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class ViewAnchors:
    target_initial: np.ndarray
    gripper_initial: np.ndarray
    target_returned: np.ndarray
    gripper_returned: np.ndarray
    gripper_plus: np.ndarray
    report: dict[str, object]


@dataclass(frozen=True)
class TwoViewAnchors:
    left: ViewAnchors
    right: ViewAnchors
    mode: str
    report: dict[str, object]


def selection_authority() -> dict[str, object]:
    canonical = json.dumps(CONFIG, sort_keys=True, separators=(",", ":")).encode()
    return {
        "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(canonical).hexdigest(),
        "config": CONFIG,
    }


def _gray(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.shape != (256, 256, 3) or value.dtype != np.uint8:
        raise ValueError("RGB frame violates the official 256x256 contract")
    return cv2.cvtColor(value, cv2.COLOR_RGB2GRAY)


def _corners(image: np.ndarray, center: np.ndarray | None) -> tuple[np.ndarray, bool]:
    mask = np.zeros((256, 256), np.uint8)
    mask[BORDER:-BORDER, BORDER:-BORDER] = 255
    clipped = False
    maximum = 256
    if center is not None:
        x, y = np.asarray(center, dtype=np.float64)
        clipped = bool(
            x - RADIUS < BORDER
            or y - RADIUS < BORDER
            or x + RADIUS > 255 - BORDER
            or y + RADIUS > 255 - BORDER
        )
        region = np.zeros_like(mask)
        cv2.circle(region, (round(float(x)), round(float(y))), RADIUS, 255, -1)
        mask &= region
        maximum = 64
    found = cv2.goodFeaturesToTrack(
        _gray(image).astype(np.float32) / 255,
        maxCorners=maximum,
        qualityLevel=0.01,
        minDistance=5,
        mask=mask,
        blockSize=3,
    )
    if found is None:
        return np.empty((0, 2), np.float32), clipped
    return found.reshape(-1, 2).astype(np.float32), clipped


def _leg(before: np.ndarray, after: np.ndarray, points: np.ndarray):
    options = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    forward, forward_ok, _ = cv2.calcOpticalFlowPyrLK(
        _gray(before), _gray(after), points.astype(np.float32), None, **options
    )
    if forward is None:
        return points.astype(np.float64), np.zeros(len(points), bool), np.full(
            len(points), np.inf
        )
    backward, backward_ok, _ = cv2.calcOpticalFlowPyrLK(
        _gray(after), _gray(before), forward, None, **options
    )
    if backward is None:
        return forward.astype(np.float64), np.zeros(len(points), bool), np.full(
            len(points), np.inf
        )
    valid = np.asarray(forward_ok).reshape(-1).astype(bool)
    valid &= np.asarray(backward_ok).reshape(-1).astype(bool)
    residual = np.linalg.norm(backward - points, axis=1)
    valid &= residual <= 1.0
    return forward.astype(np.float64), valid, residual.astype(np.float64)


def _track_sequence(frames: list[np.ndarray], points: np.ndarray):
    positions = [points.astype(np.float64)]
    valid = np.ones(len(points), bool)
    maximum_fb = np.zeros(len(points), np.float64)
    noise_legs: list[np.ndarray] = []
    for index, (before, after) in enumerate(pairwise(frames)):
        tracked, leg_valid, fb = _leg(before, after, positions[-1])
        valid &= leg_valid
        maximum_fb = np.maximum(maximum_fb, fb)
        if index < len(frames) - 3:
            noise_legs.append(np.linalg.norm(tracked - positions[-1], axis=1))
        positions.append(tracked)
    values = np.stack(noise_legs, axis=1)
    median = np.median(values, axis=1)
    mad = np.median(np.abs(values - median[:, None]), axis=1)
    noise = np.maximum(MIN_NOISE, median + 5 * mad)
    return positions, valid, maximum_fb, noise


def _edge(first: np.ndarray, second: np.ndarray) -> bool:
    first_magnitude = float(np.linalg.norm(first))
    second_magnitude = float(np.linalg.norm(second))
    cosine = float(first @ second) / max(first_magnitude * second_magnitude, 1e-12)
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    magnitude_ok = abs(first_magnitude - second_magnitude) <= 0.25 * max(
        first_magnitude, second_magnitude
    )
    return angle <= 15.0 and magnitude_ok


def select_motion_cluster(
    initial_points: np.ndarray, plus_vectors: np.ndarray, passing: np.ndarray
) -> tuple[int, dict[str, object]]:
    points = np.asarray(initial_points, dtype=np.float64)
    vectors = np.asarray(plus_vectors, dtype=np.float64)
    indices = np.flatnonzero(np.asarray(passing, dtype=bool))
    adjacency = {int(index): {int(index)} for index in indices}
    failed_edges = 0
    for offset, first in enumerate(indices):
        for second in indices[offset + 1 :]:
            if _edge(vectors[first], vectors[second]):
                adjacency[int(first)].add(int(second))
                adjacency[int(second)].add(int(first))
            else:
                failed_edges += 1
    components: list[list[int]] = []
    unseen = set(adjacency)
    while unseen:
        pending = [min(unseen)]
        component: set[int] = set()
        while pending:
            node = pending.pop()
            if node in component:
                continue
            component.add(node)
            pending.extend(adjacency[node] - component)
        unseen -= component
        components.append(sorted(component))
    complete: list[list[int]] = []
    discarded: list[list[int]] = []
    for component in components:
        destination = (
            complete
            if all(other in adjacency[node] for node in component for other in component)
            else discarded
        )
        destination.append(component)
    sizes = sorted((len(component) for component in complete), reverse=True)
    report: dict[str, object] = {
        "passing_count": len(indices),
        "connected_components": components,
        "complete_components": complete,
        "discarded_noncomplete_components": discarded,
        "largest_discarded_noncomplete_component": max(
            (len(component) for component in discarded), default=0
        ),
        "failed_similarity_edges": failed_edges,
        "largest_complete_component": sizes[0] if sizes else 0,
        "runner_up_complete_component": sizes[1] if len(sizes) > 1 else 0,
    }
    if not sizes or sizes[0] < MIN_CLUSTER:
        raise MotionClusterError("no complete motion cluster of at least three", report)
    if len(sizes) > 1 and sizes[0] == sizes[1]:
        raise MotionClusterError("motion clusters have an equal maximum", report)
    chosen = next(component for component in complete if len(component) == sizes[0])
    ranked = []
    for index in chosen:
        cost = float(
            sum(np.linalg.norm(vectors[index] - vectors[other]) for other in chosen)
        )
        ranked.append((cost, float(points[index, 0]), float(points[index, 1]), index))
    ranked.sort()
    selected = int(ranked[0][3])
    report.update({"selected_component": chosen, "selected_index": selected})
    return selected, report


def _texture(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = cv2.cornerMinEigenVal(
        _gray(image).astype(np.float32) / 255, blockSize=3, ksize=3
    )
    xy = np.rint(points).astype(int)
    return values[xy[:, 1], xy[:, 0]].astype(np.float64)


def _attempt_view(
    same: list[np.ndarray],
    plus: np.ndarray,
    returned: np.ndarray,
    *,
    target_center: np.ndarray,
    gripper_center: np.ndarray,
    mode: str,
) -> ViewAnchors:
    frames = [*same, plus, returned]
    target_candidates, target_clipped = _corners(same[0], target_center)
    gripper_candidates, gripper_clipped = _corners(
        same[0], gripper_center if mode == "cited_region" else None
    )
    report: dict[str, object] = {
        "passed": False,
        "selection_mode": mode,
        "target_region_clipped": target_clipped,
        "gripper_region_clipped": gripper_clipped,
        "target_candidate_count": len(target_candidates),
        "gripper_candidate_count": len(gripper_candidates),
        "target_citation_px": np.asarray(target_center).tolist(),
        "gripper_citation_px": np.asarray(gripper_center).tolist(),
    }
    if not len(target_candidates) or not len(gripper_candidates):
        raise AnchorSelectionError("anchor candidate pool is empty", report)
    target_positions, target_valid, target_fb, target_noise = _track_sequence(
        frames, target_candidates
    )
    gripper_positions, gripper_valid, gripper_fb, gripper_noise = _track_sequence(
        frames, gripper_candidates
    )
    target_texture = _texture(same[0], target_candidates)
    gripper_texture = _texture(same[0], gripper_candidates)
    target_plus = np.linalg.norm(target_positions[-2] - target_positions[0], axis=1)
    target_return = np.linalg.norm(target_positions[-1] - target_positions[0], axis=1)
    target_pass = target_valid & (target_texture >= 1e-5)
    target_pass &= target_plus <= np.maximum(1.5, 3 * target_noise)
    target_pass &= target_return <= np.maximum(1.5, 3 * target_noise)
    target_indices = np.flatnonzero(target_pass)
    if not len(target_indices):
        report["failure"] = "target region has no continuous static textured anchor"
        raise AnchorSelectionError(str(report["failure"]), report)
    target_index = int(
        target_indices[
            np.argmin(
                np.linalg.norm(target_candidates[target_indices] - target_center, axis=1)
            )
        ]
    )
    plus_vectors = gripper_positions[-2] - gripper_positions[0]
    return_vectors = gripper_positions[-1] - gripper_positions[-2]
    plus_magnitude = np.linalg.norm(plus_vectors, axis=1)
    return_magnitude = np.linalg.norm(return_vectors, axis=1)
    net_return = np.linalg.norm(gripper_positions[-1] - gripper_positions[0], axis=1)
    cosine = np.sum(plus_vectors * return_vectors, axis=1) / np.maximum(
        plus_magnitude * return_magnitude, 1e-12
    )
    ratio = return_magnitude / np.maximum(plus_magnitude, 1e-12)
    gates = {
        "continuous": gripper_valid,
        "textured": gripper_texture >= 1e-5,
        "moving": plus_magnitude >= np.maximum(1.5, 3 * gripper_noise),
        "returned": net_return <= np.maximum(1.5, 0.20 * plus_magnitude),
        "reversed": cosine <= -0.9,
        "return_ratio": (ratio >= 0.5) & (ratio <= 2.0),
    }
    passing = np.logical_and.reduce(list(gates.values()))
    candidates: list[dict[str, object]] = []
    for index, point in enumerate(gripper_candidates):
        candidates.append(
            {
                "index": index,
                "initial_px": point.astype(np.float64).tolist(),
                "plus_vector_px": plus_vectors[index].tolist(),
                "return_vector_px": return_vectors[index].tolist(),
                "noise_px": float(gripper_noise[index]),
                "texture_eigenvalue": float(gripper_texture[index]),
                "max_forward_backward_px": float(gripper_fb[index]),
                "return_error_px": float(net_return[index]),
                "reversal_cosine": float(cosine[index]),
                "return_ratio": float(ratio[index]),
                "citation_offset_px": float(np.linalg.norm(point - gripper_center)),
                "gates": {key: bool(values[index]) for key, values in gates.items()},
                "passed": bool(passing[index]),
            }
        )
    report["candidates"] = candidates
    report["near_miss_counts"] = {
        key: int(np.count_nonzero(~values)) for key, values in gates.items()
    }
    try:
        gripper_index, cluster = select_motion_cluster(
            gripper_positions[0], plus_vectors, passing
        )
    except MotionClusterError as error:
        report.update({"cluster": error.report, "failure": str(error)})
        raise AnchorSelectionError(str(error), report) from error
    report.update(
        {
            "passed": True,
            "gripper_passing_count": int(np.count_nonzero(passing)),
            "cluster": cluster,
            "target_anchor_px": target_positions[0][target_index].tolist(),
            "gripper_anchor_px": gripper_positions[0][gripper_index].tolist(),
            "target_anchor_offset_px": (
                target_positions[0][target_index] - target_center
            ).tolist(),
            "gripper_anchor_offset_px": (
                gripper_positions[0][gripper_index] - gripper_center
            ).tolist(),
            "target_texture_eigenvalue": float(target_texture[target_index]),
            "gripper_texture_eigenvalue": float(gripper_texture[gripper_index]),
            "target_max_forward_backward_px": float(target_fb[target_index]),
            "gripper_max_forward_backward_px": float(gripper_fb[gripper_index]),
            "gripper_plus_displacement_px": plus_vectors[gripper_index].tolist(),
            "gripper_return_displacement_px": return_vectors[gripper_index].tolist(),
            "gripper_return_error_px": float(net_return[gripper_index]),
            "gripper_reversal_cosine": float(cosine[gripper_index]),
            "gripper_return_ratio": float(ratio[gripper_index]),
            "gripper_noise_px": float(gripper_noise[gripper_index]),
        }
    )
    return ViewAnchors(
        target_positions[0][target_index],
        gripper_positions[0][gripper_index],
        target_positions[-1][target_index],
        gripper_positions[-1][gripper_index],
        gripper_positions[-2][gripper_index],
        report,
    )


def select_view_anchors(
    same_pose_frames: list[np.ndarray],
    plus_frame: np.ndarray,
    returned_frame: np.ndarray,
    *,
    target_center: np.ndarray,
    gripper_center: np.ndarray,
) -> ViewAnchors:
    if len(same_pose_frames) != 5:
        raise ValueError("anchor noise authority requires exactly five same-pose frames")
    return _attempt_view(
        same_pose_frames,
        plus_frame,
        returned_frame,
        target_center=target_center,
        gripper_center=gripper_center,
        mode="cited_region",
    )


def select_two_view_anchors(**arguments: object) -> TwoViewAnchors:
    if len(arguments["left_same_pose"]) != 5 or len(arguments["right_same_pose"]) != 5:
        raise ValueError("anchor noise authority requires exactly five same-pose frames")
    views = {
        "left": (
            arguments["left_same_pose"], arguments["left_plus"],
            arguments["left_returned"], arguments["left_target_center"],
            arguments["left_gripper_center"],
        ),
        "right": (
            arguments["right_same_pose"], arguments["right_plus"],
            arguments["right_returned"], arguments["right_target_center"],
            arguments["right_gripper_center"],
        ),
    }
    cited: dict[str, ViewAnchors] = {}
    cited_reports: dict[str, object] = {}
    for name, values in views.items():
        try:
            cited[name] = _attempt_view(
                values[0], values[1], values[2], target_center=values[3],
                gripper_center=values[4], mode="cited_region",
            )
            cited_reports[name] = cited[name].report
        except AnchorSelectionError as error:
            cited_reports[name] = error.report
    authority = selection_authority()
    if len(cited) == 2:
        return TwoViewAnchors(
            cited["left"], cited["right"], "cited_region",
            {"mode": "cited_region", "cited_region_attempts": cited_reports,
             "selection_authority": authority},
        )
    fallback: dict[str, ViewAnchors] = {}
    fallback_reports: dict[str, object] = {}
    for name, values in views.items():
        try:
            fallback[name] = _attempt_view(
                values[0], values[1], values[2], target_center=values[3],
                gripper_center=values[4], mode="whole_frame",
            )
            fallback_reports[name] = fallback[name].report
        except AnchorSelectionError as error:
            fallback_reports[name] = error.report
    report = {
        "mode": "whole_frame", "cited_region_attempts": cited_reports,
        "whole_frame_attempts": fallback_reports, "selection_authority": authority,
    }
    if len(fallback) != 2:
        raise AnchorSelectionError("two-view whole-frame anchor selection failed", report)
    return TwoViewAnchors(
        fallback["left"], fallback["right"], "whole_frame", report
    )


def render_anchor_overlay(
    images: dict[str, np.ndarray], selection: dict[str, object], target: Path
) -> dict[str, object]:
    canvas = Image.new("RGB", (512, 256))
    inputs: dict[str, object] = {}
    for view_index, view in enumerate(("left", "right")):
        source = np.asarray(images[view], dtype=np.uint8)
        frame = Image.fromarray(source)
        draw = ImageDraw.Draw(frame)
        evidence = selection[view]
        for candidate in evidence.get("candidates", []):
            if candidate["passed"]:
                u, v = candidate["initial_px"]
                draw.ellipse((u - 2, v - 2, u + 2, v + 2), outline="magenta")
        for key, color, radius in (
            ("target_citation_px", "yellow", 6),
            ("gripper_citation_px", "orange", 6),
            ("target_anchor_px", "red", 4),
            ("gripper_anchor_px", "cyan", 4),
        ):
            u, v = evidence[key]
            draw.ellipse((u-radius, v-radius, u+radius, v+radius), outline=color, width=2)
        canvas.paste(frame, (view_index * 256, 0))
        inputs[view] = {
            "raw_rgb_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
            "shape": list(source.shape),
            **{key: evidence[key] for key in (
                "target_citation_px", "gripper_citation_px",
                "target_anchor_px", "gripper_anchor_px")},
        }
    canvas.save(target, format="PNG")
    target.chmod(0o600)
    return {
        "path": str(target),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "inputs": inputs,
        "renderer_module_sha256": selection_authority()["module_sha256"],
    }
