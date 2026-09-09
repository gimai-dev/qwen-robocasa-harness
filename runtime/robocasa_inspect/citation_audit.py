"""No-motion Qwen audit of target-only citations in both official external views."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw

from .active_skills import ActiveSkillSupervisor, decode_skill_command
from .base_active import base_pulse_chunk
from .model_client import QwenClient, load_authority, verify_process
from .runner import (
    _atomic_json,
    _launch_simulator,
    _read_images,
    _serialize_actions,
    _wait_json,
)

SYSTEM_PROMPT = """You are the target-grounding component of an Inspect-style visual
robot controller for official RoboCasa365 PandaOmron. You receive exactly three
synchronized 256x256 RGB images: left external, right external, wrist, plus only the
official public 16D robot state. The wrist view may be occluded. Do not infer depth,
camera geometry, object pose, joints, contacts, rewards, or success. For the requested
external view, cite the same physical task feature using exactly:
{"kind":"center_feature","observation_id":"...","view":"robot0_agentview_left|robot0_agentview_right","feature_uv":[u,v],"feature_kind":"front_surface"}
Return one JSON object and no prose. u is left-to-right and v top-to-bottom in [0,1].
Use precise pixel-derived values at least 12 pixels inside the image. For CloseDrawer,
cite the center of the protruding moving front panel of the already-open drawer. The
action is a push on that front panel; its handle can be outside the image. A closed
drawer handle is not the target. Do not cite the microwave, cabinet knobs, robot,
countertop, refrigerator, or any closed drawer. Never output gripper pixels or a
motion command."""


def _overlay(
    observation: dict[str, object], citations: dict[str, tuple[float, float]], target: Path
) -> None:
    canvas = Image.new("RGB", (512, 256))
    for index, name in enumerate(("left", "right")):
        panel = Image.open(observation["images"][name]["path"]).convert("RGB")
        draw = ImageDraw.Draw(panel)
        x, y = citations[name]
        x *= 255
        y *= 255
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline="lime", width=3)
        draw.line((x - 12, y, x + 12, y), fill="lime", width=2)
        draw.line((x, y - 12, x, y + 12), fill="lime", width=2)
        draw.text((4, 4), name, fill="lime")
        canvas.paste(panel, (index * 256, 0))
    canvas.save(target)
    target.chmod(0o600)


def yaw_scan_schedule() -> tuple[int, ...]:
    """Visit +15 and -15 degrees approximately, then return to reset."""
    return (1,) * 10 + (-1,) * 20 + (1,) * 10


def _send_yaw(run: Path, sequence: int, direction: int) -> None:
    _atomic_json(
        run / "sim" / "mailbox" / f"command-{sequence:06d}.json",
        {
            "schema": "robocasa-inspect-command/v1",
            "sequence": sequence,
            "kind": "action",
            "actions": _serialize_actions(
                base_pulse_chunk(axis="yaw", normalized_velocity=0.15 * direction)
            ),
        },
    )


def run_audit(
    *,
    run: Path,
    task: str = "CloseDrawer",
    seed: int = 7,
    yaw_scan: bool = False,
) -> dict[str, object]:
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    log_path = run / "simulator.log"
    log = log_path.open("w", encoding="utf-8")
    os.chmod(log_path, 0o600)
    identity, attestation = load_authority(
        Path(os.environ["QWEN_IDENTITY_MANIFEST"]),
        Path(os.environ["QWEN_SERVER_ATTESTATION"]),
    )
    client = QwenClient(
        base_url="http://127.0.0.1:8002/v1",
        api_key=Path(os.environ["QWEN_API_TOKEN_FILE"]).read_text().strip(),
        identity=identity,
        attestation=attestation,
    )
    client.verify()
    process = _launch_simulator(task=task, seed=seed, run=run, log_handle=log)
    supervisor = ActiveSkillSupervisor(max_model_calls=8 if yaw_scan else 4)
    records: list[dict[str, object]] = []
    try:
        observation = _wait_json(
            run / "sim" / "mailbox" / "observation-000000.json", timeout_s=180
        )
        waypoints = [("reset", 0)]
        if yaw_scan:
            waypoints.extend((("yaw_plus", 10), ("yaw_minus", 30)))
        sequence = 0
        overlays: list[Path] = []
        schedule = yaw_scan_schedule() if yaw_scan else ()
        for pose, target_sequence in waypoints:
            while sequence < target_sequence:
                _send_yaw(run, sequence, schedule[sequence])
                sequence += 1
                observation = _wait_json(
                    run
                    / "sim"
                    / "mailbox"
                    / f"observation-{sequence:06d}.json",
                    timeout_s=180,
                )
            images = _read_images(observation)
            pose_citations: dict[str, tuple[float, float]] = {}
            for short, full in (
                ("left", "robot0_agentview_left"),
                ("right", "robot0_agentview_right"),
            ):
                supervisor.record_model_call()
                response = client.complete(
                    observation_id=str(observation["observation_id"]),
                    system_prompt=SYSTEM_PROMPT,
                    instruction=json.dumps(
                        {
                            "task": observation["instruction"],
                            "viewpoint": pose,
                            "required_view": full,
                            "prior_target_citations_this_viewpoint": pose_citations,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    public_state=observation["public_state"],
                    images=images,
                )
                command = decode_skill_command(
                    response.command,
                    observation_id=str(observation["observation_id"]),
                )
                if command.kind != "center_feature" or command.view != full:
                    raise RuntimeError("Qwen did not return the required target citation")
                if (
                    command.feature_kind != "front_surface"
                    or command.feature_uv is None
                ):
                    raise RuntimeError("Qwen target kind drifted")
                pose_citations[short] = command.feature_uv
                records.append(
                    {
                        "viewpoint": pose,
                        "sequence": sequence,
                        "view": full,
                        "feature_kind": command.feature_kind,
                        "feature_uv": list(command.feature_uv),
                        "evidence": response.evidence,
                    }
                )
            overlay = run / f"target-citation-overlay-{pose}.png"
            _overlay(observation, pose_citations, overlay)
            overlays.append(overlay)
        while sequence < len(schedule):
            _send_yaw(run, sequence, schedule[sequence])
            sequence += 1
            _wait_json(
                run / "sim" / "mailbox" / f"observation-{sequence:06d}.json",
                timeout_s=180,
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
        verify_process(attestation)
        result = {
            "schema": "robocasa-inspect-target-citation-audit/v1",
            "task": task,
            "seed": seed,
            "model_calls": supervisor.model_calls,
            "records": records,
            "yaw_scan": yaw_scan,
            "overlays": [
                {
                    "path": str(overlay),
                    "sha256": hashlib.sha256(overlay.read_bytes()).hexdigest(),
                }
                for overlay in overlays
            ],
            "success_was_queried": False,
            "snapshot_digest": identity.get("snapshot_digest"),
        }
        _atomic_json(run / "citation-audit.json", result)
        return result
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        log.close()
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--yaw-scan", action="store_true")
    args = parser.parse_args()
    result = run_audit(run=args.run_dir.resolve(), yaw_scan=args.yaw_scan)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
