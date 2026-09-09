"""Reversible official-EEF viewpoint characterization without task evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .contracts import Command, official_action_chunk
from .runner import _atomic_json, _launch_simulator, _serialize_actions, _wait_json


def viewpoint_schedule() -> list[tuple[str, np.ndarray]]:
    result: list[tuple[str, np.ndarray]] = []
    for axis, name in enumerate("xyz"):
        plus = np.zeros(3, dtype=np.float64)
        plus[axis] = 0.02
        result.extend(
            [
                (f"{name}_plus", plus.copy()),
                (f"{name}_return", -plus.copy()),
                (f"{name}_minus", -plus.copy()),
                (f"{name}_return_2", plus.copy()),
            ]
        )
    return result


def _send(run: Path, sequence: int, delta: np.ndarray) -> None:
    command = Command(
        kind="action",
        observation_id="characterization-only",
        translation_m=tuple(float(value) for value in delta),
        gripper="open",
    )
    actions, _ = official_action_chunk(command, previous_gripper="open")
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
    log_path = run / "simulator.log"
    log = log_path.open("w", encoding="utf-8")
    os.chmod(log_path, 0o600)
    process = _launch_simulator(task=task, seed=seed, run=run, log_handle=log)
    records: list[dict[str, object]] = []
    sequence = 0
    try:
        observation = _wait_json(
            run / "sim" / "mailbox" / "observation-000000.json", timeout_s=180
        )
        initial = np.asarray(
            observation["public_state"]["state.end_effector_position_relative"],
            dtype=np.float64,
        )
        for label, delta in viewpoint_schedule():
            _send(run, sequence, delta)
            sequence += 1
            observation = _wait_json(
                run / "sim" / "mailbox" / f"observation-{sequence:06d}.json",
                timeout_s=180,
            )
            wrist_path = Path(observation["images"]["wrist"]["path"])
            gray = np.asarray(Image.open(wrist_path).convert("L"), dtype=np.uint8)
            counts = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
            probability = counts[counts > 0] / counts.sum()
            records.append(
                {
                    "label": label,
                    "requested_delta_m": delta.tolist(),
                    "eef_position_m": observation["public_state"][
                        "state.end_effector_position_relative"
                    ],
                    "wrist_entropy_bits": float(-np.sum(probability * np.log2(probability))),
                    "wrist_sha256": hashlib.sha256(wrist_path.read_bytes()).hexdigest(),
                }
            )
        _atomic_json(
            run / "sim" / "mailbox" / f"command-{sequence:06d}.json",
            {
                "schema": "robocasa-inspect-command/v1",
                "sequence": sequence,
                "kind": "close",
            },
        )
        process.wait(timeout=180)
        final = np.asarray(
            observation["public_state"]["state.end_effector_position_relative"],
            dtype=np.float64,
        )
        result = {
            "schema": "robocasa-inspect-viewpoint-characterization/v1",
            "task": task,
            "seed": seed,
            "records": records,
            "reset_return_error_m": float(np.linalg.norm(final - initial)),
            "success_was_queried": False,
        }
        _atomic_json(run / "viewpoint-characterization.json", result)
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
