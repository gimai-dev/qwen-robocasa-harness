"""Deterministic public-path smoke for external-camera image servo."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

from .joint_runner import _atomic_json, run_episode

RESULT_SCHEMA = "robocasa-inspect-image-servo-smoke/v1"
PIXEL_OFFSET = 6.0
STEP_M = 0.01
MIN_PIXEL_MOTION = 0.5
MAX_RETURN_ERROR_PX = 2.0
MAX_ORIENTATION_EFFECT_RAD = 0.05
MAX_ACTIONS = 160


def _vector(value: object, width: int) -> list[float] | None:
    if (
        isinstance(value, (str, bytes, Mapping))
        or not isinstance(value, Sequence)
        or len(value) != width
    ):
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(item * item for item in value))


def _selected_pixels(receipt: Mapping[str, object]) -> tuple[list[float], list[float]] | None:
    displacement = receipt.get("end_effector_external_pixel_displacement")
    camera = receipt.get("requested_camera")
    if not isinstance(displacement, Mapping) or camera not in {"left", "right"}:
        return None
    record = displacement.get(camera)
    if not isinstance(record, Mapping):
        return None
    start = _vector(record.get("start_px"), 2)
    end = _vector(record.get("end_px"), 2)
    return (start, end) if start is not None and end is not None else None


def evaluate_image_servo_episode(
    episode: Mapping[str, object],
) -> tuple[bool, dict[str, bool], dict[str, float | int | None]]:
    raw = episode.get("receipts")
    receipts = (
        [dict(item) for item in raw if isinstance(item, Mapping)]
        if isinstance(raw, list)
        else []
    )
    closed = len(receipts) == 2 and all(
        receipt.get("kind") == "image_servo"
        and receipt.get("accepted") is True
        and receipt.get("requested_camera") == "left"
        for receipt in receipts
    )
    pixels = [_selected_pixels(receipt) for receipt in receipts]
    first_start = pixels[0][0] if len(pixels) == 2 and pixels[0] else None
    first_end = pixels[0][1] if len(pixels) == 2 and pixels[0] else None
    second_end = pixels[1][1] if len(pixels) == 2 and pixels[1] else None
    outbound = (
        first_end[0] - first_start[0]
        if first_start is not None and first_end is not None
        else None
    )
    return_error = (
        _norm([second_end[index] - first_start[index] for index in range(2)])
        if first_start is not None and second_end is not None
        else None
    )
    poses = [
        receipt.get("end_effector_pose_delta")
        if isinstance(receipt.get("end_effector_pose_delta"), Mapping)
        else None
        for receipt in receipts
    ]
    translations = [
        _vector(pose.get("translation_m"), 3) if pose is not None else None
        for pose in poses
    ]
    rotations = [
        _vector(pose.get("rotation_axis_angle_rad"), 3)
        if pose is not None
        else None
        for pose in poses
    ]
    targets = [_vector(receipt.get("requested_target_pixel"), 2) for receipt in receipts]
    requested = (
        closed
        and first_start is not None
        and targets
        == [
            [first_start[0] + PIXEL_OFFSET, first_start[1]],
            [first_start[0], first_start[1]],
        ]
    )
    simulator_steps = episode.get("simulator_steps")
    checks = {
        "closed_image_servo_receipts": closed,
        "requested_pixel_sequence_exact": requested,
        "outbound_pixel_motion_observed": (
            outbound is not None and outbound >= MIN_PIXEL_MOTION
        ),
        "return_pixel_error_bounded": (
            return_error is not None and return_error <= MAX_RETURN_ERROR_PX
        ),
        "task_space_motion_observed": (
            len(translations) == 2
            and all(value is not None for value in translations)
            and _norm(translations[0]) >= 0.001
        ),
        "orientation_constrained": (
            len(rotations) == 2
            and all(value is not None for value in rotations)
            and max(_norm(value) for value in rotations if value is not None)
            <= MAX_ORIENTATION_EFFECT_RAD
        ),
        "action_budget_bounded": (
            type(simulator_steps) is int and 0 < simulator_steps <= MAX_ACTIONS
        ),
        "official_terminal_closed": (
            episode.get("status") in {"success", "policy_finished_false"}
            and isinstance(episode.get("terminal_outcome"), Mapping)
        ),
    }
    metrics: dict[str, float | int | None] = {
        "outbound_pixel_motion_px": outbound,
        "return_pixel_error_px": return_error,
        "simulator_steps": simulator_steps if type(simulator_steps) is int else None,
    }
    return all(checks.values()), checks, metrics


def validate_public_image_servo_result(
    value: object,
    *,
    expected_task: str,
    expected_seed: int,
    expected_release_digest: str,
    expected_artifact_root: str,
) -> dict[str, object]:
    fields = {
        "schema",
        "task",
        "seed",
        "release_digest",
        "artifact_root",
        "passed",
        "checks",
        "metrics",
        "episode_result_sha256",
        "video",
        "video_sha256",
        "wall_s",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("image-servo smoke result schema drifted")
    result = dict(value)
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("task") != expected_task
        or result.get("seed") != expected_seed
        or result.get("release_digest") != expected_release_digest
        or result.get("artifact_root") != expected_artifact_root
        or type(result.get("passed")) is not bool
        or not isinstance(result.get("checks"), Mapping)
        or not isinstance(result.get("metrics"), Mapping)
    ):
        raise ValueError("image-servo smoke result identity drifted")
    for field in ("episode_result_sha256", "video_sha256"):
        digest = result.get(field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"image-servo smoke {field} is invalid")
    return result


class _DeterministicImageServoClient:
    def __init__(self, **_kwargs: object) -> None:
        self.call_index = 0
        self.initial_pixel: list[float] | None = None

    def verify(self) -> None:
        return None

    def close(self) -> None:
        return None

    def complete(self, **kwargs: object) -> SimpleNamespace:
        observation_id = str(kwargs["observation_id"])
        public_state = kwargs["public_state"]
        assert isinstance(public_state, Mapping)
        pixels = public_state["state.end_effector_external_pixels"]
        assert isinstance(pixels, Mapping) and isinstance(pixels["left"], Mapping)
        current = [float(pixels["left"]["u_px"]), float(pixels["left"]["v_px"])]
        index = self.call_index
        self.call_index += 1
        if index == 0:
            self.initial_pixel = current
            target = [current[0] + PIXEL_OFFSET, current[1]]
        elif index == 1:
            assert self.initial_pixel is not None
            target = self.initial_pixel
        else:
            return SimpleNamespace(
                command={
                    "kind": "finish",
                    "observation_id": observation_id,
                    "note": "deterministic image-servo smoke complete",
                },
                evidence={"source": "deterministic_image_servo_smoke"},
            )
        return SimpleNamespace(
            command={
                "kind": "image_servo",
                "observation_id": observation_id,
                "camera": "left",
                "target_pixel": target,
                "target_role": "fixture_handle",
                "depth_delta_m": 0.0,
                "step_m": STEP_M,
                "gripper": "open",
                "note": "state.end_effector_external_pixels left offset smoke",
            },
            evidence={"source": "deterministic_image_servo_smoke"},
        )


def run_image_servo_smoke(*, task: str, seed: int, run: Path) -> dict[str, object]:
    started = time.monotonic()
    episode_run = run / "episode"
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    episode = run_episode(
        task=task,
        seed=seed,
        run=episode_run,
        max_decisions=3,
        action_budget=MAX_ACTIONS,
        client_class=_DeterministicImageServoClient,
        protocol="legacy",
    )
    passed, checks, metrics = evaluate_image_servo_episode(episode)
    episode_path = episode_run / "result.json"
    result = {
        "schema": RESULT_SCHEMA,
        "task": task,
        "seed": seed,
        "release_digest": Path(__file__).resolve().parents[1].name,
        "artifact_root": str(run),
        "passed": passed,
        "checks": checks,
        "metrics": metrics,
        "episode_result_sha256": hashlib.sha256(episode_path.read_bytes()).hexdigest(),
        "video": episode["video"],
        "video_sha256": episode["video_sha256"],
        "wall_s": time.monotonic() - started,
    }
    _atomic_json(run / "image-servo-smoke.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="OpenToasterOvenDoor")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_image_servo_smoke(
        task=args.task, seed=args.seed, run=args.run_dir.resolve()
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
