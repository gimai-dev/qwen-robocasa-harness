"""Deterministic public-path smoke for Panda Cartesian delta execution."""

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

RESULT_SCHEMA = "robocasa-inspect-cartesian-skill-smoke/v1"
OUTBOUND_TRANSLATION_M = 0.01
MIN_OBSERVED_TRANSLATION_M = 0.002
MAX_RETURN_TRANSLATION_ERROR_M = 0.01
MAX_ORIENTATION_EFFECT_RAD = 0.05
MAX_ACTIONS = 160


def _vector(value: object, *, width: int) -> list[float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != width
    ):
        return None
    try:
        output = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return output if all(math.isfinite(item) for item in output) else None


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(item * item for item in value))


def _pose_delta(receipt: Mapping[str, object]) -> tuple[list[float], list[float]] | None:
    value = receipt.get("end_effector_pose_delta")
    if not isinstance(value, Mapping):
        return None
    translation = _vector(value.get("translation_m"), width=3)
    rotation = _vector(value.get("rotation_axis_angle_rad"), width=3)
    if translation is None or rotation is None:
        return None
    return translation, rotation


def _separation(receipt: Mapping[str, object], field: str) -> float | None:
    residual = receipt.get("gripper_residual")
    if not isinstance(residual, Mapping):
        return None
    if field == "end":
        value = residual.get("measured_end_finger_separation")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        output = float(value)
        return output if math.isfinite(output) and output >= 0.0 else None
    qpos = _vector(residual.get("start_qpos"), width=2)
    return abs(qpos[0] - qpos[1]) if qpos is not None else None


def evaluate_cartesian_skill_episode(
    episode: Mapping[str, object],
) -> tuple[bool, dict[str, bool], dict[str, float | int | None]]:
    raw_receipts = episode.get("receipts")
    receipts = (
        [dict(item) for item in raw_receipts if isinstance(item, Mapping)]
        if isinstance(raw_receipts, list)
        else []
    )
    closed_receipts = len(receipts) == 4 and all(
        receipt.get("kind") == "cartesian_delta"
        and receipt.get("accepted") is True
        for receipt in receipts
    )
    requested = [
        (
            _vector(receipt.get("requested_translation_m"), width=3),
            _vector(receipt.get("requested_rotation_axis_angle_rad"), width=3),
            receipt.get("requested_gripper"),
        )
        for receipt in receipts
    ]
    requested_sequence = closed_receipts and requested == [
        ([OUTBOUND_TRANSLATION_M, 0.0, 0.0], [0.0, 0.0, 0.0], "hold"),
        ([-OUTBOUND_TRANSLATION_M, 0.0, 0.0], [0.0, 0.0, 0.0], "hold"),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], "close"),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], "open"),
    ]
    poses = [_pose_delta(receipt) for receipt in receipts]
    outbound_translation = (
        _norm(poses[0][0]) if len(poses) == 4 and poses[0] is not None else None
    )
    orientation_effect = (
        max(_norm(pose[1]) for pose in poses[:2] if pose is not None)
        if len(poses) == 4 and all(pose is not None for pose in poses[:2])
        else None
    )
    return_error = (
        _norm([poses[0][0][index] + poses[1][0][index] for index in range(3)])
        if len(poses) == 4 and poses[0] is not None and poses[1] is not None
        else None
    )
    close_start = _separation(receipts[2], "start") if len(receipts) == 4 else None
    close_end = _separation(receipts[2], "end") if len(receipts) == 4 else None
    open_start = _separation(receipts[3], "start") if len(receipts) == 4 else None
    open_end = _separation(receipts[3], "end") if len(receipts) == 4 else None
    steps = episode.get("simulator_steps")
    checks = {
        "closed_cartesian_receipts": closed_receipts,
        "requested_sequence_exact": requested_sequence,
        "outbound_translation_observed": (
            outbound_translation is not None
            and outbound_translation >= MIN_OBSERVED_TRANSLATION_M
        ),
        "six_axis_orientation_constrained": (
            orientation_effect is not None
            and orientation_effect <= MAX_ORIENTATION_EFFECT_RAD
        ),
        "return_translation_bounded": (
            return_error is not None
            and return_error <= MAX_RETURN_TRANSLATION_ERROR_M
        ),
        "gripper_close_observed": (
            close_start is not None
            and close_end is not None
            and close_end <= close_start - 0.005
        ),
        "gripper_reopen_observed": (
            open_start is not None
            and open_end is not None
            and open_end >= open_start + 0.005
        ),
        "action_budget_bounded": (
            type(steps) is int and 0 < steps <= MAX_ACTIONS
        ),
        "official_terminal_closed": (
            episode.get("status") in {"success", "policy_finished_false"}
            and isinstance(episode.get("terminal_outcome"), Mapping)
        ),
    }
    metrics: dict[str, float | int | None] = {
        "outbound_translation_m": outbound_translation,
        "maximum_orientation_effect_rad": orientation_effect,
        "return_translation_error_m": return_error,
        "close_start_separation_m": close_start,
        "close_end_separation_m": close_end,
        "reopen_start_separation_m": open_start,
        "reopen_end_separation_m": open_end,
        "simulator_steps": steps if type(steps) is int else None,
    }
    return all(checks.values()), checks, metrics


class _DeterministicCartesianClient:
    """Exercise the production runner without issuing a model request."""

    def __init__(self, **_kwargs: object) -> None:
        self.call_index = 0

    def verify(self) -> None:
        return None

    def close(self) -> None:
        return None

    def complete(self, **kwargs: object) -> SimpleNamespace:
        observation_id = str(kwargs["observation_id"])
        index = self.call_index
        self.call_index += 1
        if index == 0:
            command = {
                "kind": "cartesian_delta",
                "observation_id": observation_id,
                "translation_m": [OUTBOUND_TRANSLATION_M, 0.0, 0.0],
                "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
                "gripper": "hold",
                "note": "state.arm_translation_jacobian outbound smoke",
            }
        elif index == 1:
            command = {
                "kind": "cartesian_delta",
                "observation_id": observation_id,
                "translation_m": [-OUTBOUND_TRANSLATION_M, 0.0, 0.0],
                "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
                "gripper": "hold",
                "note": "state.arm_translation_jacobian return smoke",
            }
        elif index in {2, 3}:
            command = {
                "kind": "cartesian_delta",
                "observation_id": observation_id,
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
                "gripper": "close" if index == 2 else "open",
                "note": "state.gripper_qpos transition smoke",
            }
        else:
            command = {
                "kind": "finish",
                "observation_id": observation_id,
                "note": "deterministic smoke complete",
            }
        return SimpleNamespace(
            command=command,
            evidence={
                "source": "deterministic_cartesian_skill_smoke",
                "call_index": index + 1,
            },
        )


def validate_public_cartesian_skill_result(
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
        raise ValueError("Cartesian skill smoke result schema drifted")
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
        raise ValueError("Cartesian skill smoke result identity drifted")
    for field in ("episode_result_sha256", "video_sha256"):
        digest = result.get(field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Cartesian skill smoke {field} is invalid")
    return result


def run_cartesian_skill_smoke(*, task: str, seed: int, run: Path) -> dict[str, object]:
    started = time.monotonic()
    episode_run = run / "episode"
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    episode = run_episode(
        task=task,
        seed=seed,
        run=episode_run,
        max_decisions=5,
        client_class=_DeterministicCartesianClient,
        protocol="legacy",
    )
    passed, checks, metrics = evaluate_cartesian_skill_episode(episode)
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
    _atomic_json(run / "cartesian-skill-smoke.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="OpenToasterOvenDoor")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_cartesian_skill_smoke(
        task=args.task,
        seed=args.seed,
        run=args.run_dir.resolve(),
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
