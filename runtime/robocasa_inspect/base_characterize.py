"""One-shot no-task-interaction certification of official base axes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .base_active import BASE_AXES, base_pulse_chunk, characterize_axis
from .runner import _atomic_json, _launch_simulator, _serialize_actions, _wait_json

PULSE = 0.10


def _images(observation: dict[str, object]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name in ("left", "right"):
        path = Path(observation["images"][name]["path"])
        output[name] = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    return output


def _yaw(quat_xyzw: np.ndarray) -> float:
    x, y, z, w = quat_xyzw
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _coordinate(observation: dict[str, object], axis: str) -> float:
    state = observation["public_state"]
    if axis == "yaw":
        return _yaw(np.asarray(state["state.base_rotation"], dtype=np.float64))
    return float(np.asarray(state["state.base_position"], dtype=np.float64)[BASE_AXES[axis]])


def _send_action(run: Path, sequence: int, actions: list[dict[str, np.ndarray]]) -> None:
    _atomic_json(
        run / "sim" / "mailbox" / f"command-{sequence:06d}.json",
        {
            "schema": "robocasa-inspect-command/v1",
            "sequence": sequence,
            "kind": "action",
            "actions": _serialize_actions(actions),
        },
    )


def characterize(*, run: Path, task: str = "CloseDrawer", seed: int = 7) -> dict[str, object]:
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    run.chmod(0o700)
    log_path = run / "simulator.log"
    log = log_path.open("w", encoding="utf-8")
    os.chmod(log_path, 0o600)
    process = _launch_simulator(task=task, seed=seed, run=run, log_handle=log)
    sequence = 0
    reports: dict[str, object] = {}
    try:
        observation = _wait_json(
            run / "sim" / "mailbox" / "observation-000000.json", timeout_s=180
        )
        reset_eef = np.asarray(
            observation["public_state"]["state.end_effector_position_relative"],
            dtype=np.float64,
        )
        for axis in ("x", "y", "yaw"):
            excursions: list[float] = []
            returns: list[float] = []
            changes: list[float] = []
            maes: list[float] = []
            eef_drifts: list[float] = []
            for _ in range(2):
                initial = observation
                initial_coordinate = _coordinate(initial, axis)
                initial_images = _images(initial)
                _send_action(
                    run,
                    sequence,
                    base_pulse_chunk(axis=axis, normalized_velocity=PULSE),
                )
                sequence += 1
                moved = _wait_json(
                    run / "sim" / "mailbox" / f"observation-{sequence:06d}.json",
                    timeout_s=180,
                )
                moved_coordinate = _coordinate(moved, axis)
                moved_images = _images(moved)
                _send_action(
                    run,
                    sequence,
                    base_pulse_chunk(axis=axis, normalized_velocity=-PULSE),
                )
                sequence += 1
                observation = _wait_json(
                    run / "sim" / "mailbox" / f"observation-{sequence:06d}.json",
                    timeout_s=180,
                )
                returned_coordinate = _coordinate(observation, axis)
                returned_images = _images(observation)
                excursions.append(abs(moved_coordinate - initial_coordinate))
                returns.append(abs(returned_coordinate - initial_coordinate))
                changes.append(
                    max(
                        float(
                            np.mean(
                                np.max(np.abs(moved_images[name] - initial_images[name]), axis=2)
                                > 8
                            )
                        )
                        for name in initial_images
                    )
                )
                maes.append(
                    max(
                        float(np.mean(np.abs(returned_images[name] - initial_images[name])))
                        for name in initial_images
                    )
                )
                eef = np.asarray(
                    observation["public_state"]["state.end_effector_position_relative"],
                    dtype=np.float64,
                )
                eef_drifts.append(float(np.linalg.norm(eef - reset_eef)))
            report = characterize_axis(
                axis=axis,
                excursions=excursions,
                return_errors=returns,
                changed_fractions=changes,
                returned_mae=maes,
            )
            report["eef_reset_relative_drift_m"] = eef_drifts
            report["enabled"] = bool(report["enabled"] and max(eef_drifts) <= 0.02)
            reports[axis] = report
        _atomic_json(
            run / "sim" / "mailbox" / f"command-{sequence:06d}.json",
            {"schema": "robocasa-inspect-command/v1", "sequence": sequence, "kind": "close"},
        )
        process.wait(timeout=180)
        result = {
            "schema": "robocasa-inspect-base-characterization/v1",
            "task": task,
            "seed": seed,
            "normalized_velocity": PULSE,
            "repetitions": 2,
            "axes": reports,
            "enabled_axes": [name for name, value in reports.items() if value["enabled"]],
            "success_was_queried": False,
        }
        _atomic_json(run / "base-characterization.json", result)
        return result
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        log.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(characterize(run=args.run_dir.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()

