"""Certify RGB-only gripper articulation anchors in the official external views."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .contracts import Command, official_action_chunk
from .gripper_articulation import locate_gripper_anchor
from .runner import _atomic_json, _launch_simulator, _serialize_actions, _wait_json


def _images(observation: dict[str, object]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(Image.open(observation["images"][name]["path"]).convert("RGB"))
        for name in ("left", "right")
    }


def _send(run: Path, sequence: int, gripper: str) -> None:
    command = Command(
        kind="action", observation_id="characterization-only", gripper=gripper
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
    sequence = 0
    try:
        initial_observation = _wait_json(
            run / "sim" / "mailbox" / "observation-000000.json", timeout_s=180
        )
        opened = _images(initial_observation)
        observation = initial_observation
        for gripper in ("close",) * 5:
            _send(run, sequence, gripper)
            sequence += 1
            observation = _wait_json(
                run / "sim" / "mailbox" / f"observation-{sequence:06d}.json",
                timeout_s=180,
            )
        closed = _images(observation)
        for gripper in ("open",) * 5:
            _send(run, sequence, gripper)
            sequence += 1
            observation = _wait_json(
                run / "sim" / "mailbox" / f"observation-{sequence:06d}.json",
                timeout_s=180,
            )
        returned = _images(observation)
        reports: dict[str, object] = {}
        overlay = Image.new("RGB", (512, 256))
        for index, name in enumerate(("left", "right")):
            evidence = locate_gripper_anchor(
                opened[name], closed[name], returned[name], static_noise_px=0.5
            )
            reports[name] = {
                "anchor_uv_px": list(evidence.anchor_uv_px),
                "changed_pixels": evidence.changed_pixels,
                "component_pixels": evidence.component_pixels,
                "return_mae": evidence.return_mae,
            }
            panel = Image.fromarray(closed[name])
            draw = ImageDraw.Draw(panel)
            x, y = evidence.anchor_uv_px
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline="cyan", width=3)
            draw.text((4, 4), name, fill="cyan")
            overlay.paste(panel, (index * 256, 0))
        overlay_path = run / "gripper-articulation-overlay.png"
        overlay.save(overlay_path)
        overlay_path.chmod(0o600)
        _atomic_json(
            run / "sim" / "mailbox" / f"command-{sequence:06d}.json",
            {
                "schema": "robocasa-inspect-command/v1",
                "sequence": sequence,
                "kind": "close",
            },
        )
        process.wait(timeout=180)
        result = {
            "schema": "robocasa-inspect-gripper-characterization/v1",
            "task": task,
            "seed": seed,
            "views": reports,
            "overlay": str(overlay_path),
            "success_was_queried": False,
        }
        _atomic_json(run / "gripper-characterization.json", result)
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
