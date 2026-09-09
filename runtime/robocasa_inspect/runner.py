"""Closed-loop Qwen runner for the network-isolated official RoboCasa child."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .anchor_selection import render_anchor_overlay
from .base_active import base_pulse_chunk
from .contracts import (
    Command,
    adapt_bounded_action,
    decode_command,
    is_zero_motion,
    official_action_chunk,
)
from .harness import ServoObservation, run_servo_session
from .inspect_prompt import SYSTEM_PROMPT
from .model_client import (
    MalformedResponse,
    QwenClient,
    Response,
    load_authority,
    verify_process,
)

REPAIR_TEXT = """The generic harness rejected the previous response as {reason}.
On this same immutable observation, first compare the exact task wording with all
three images. If the current scene already visibly satisfies the exact task, issue
`finish` now instead of observing again. Otherwise return one valid bounded nonzero
action that makes visible progress. Reobserve is unavailable because it would repeat
the same frozen frame. If visibility is poor, use a small safe upward or
away-from-surface motion to improve the wrist viewpoint, then inspect the fresh frame.
Infer geometry only from the three RGB views and public state. Return give_up only if
even a safe viewpoint move is impossible. Do not repeat the rejected command.
For arm motion copy this complete shape:
{{"kind":"action","observation_id":"CURRENT","note":"WHY",
"translation_m":[0,0,0],"rotation_axis_angle_rad":[0,0,0],
"gripper":"open|hold|close"}}.
For mobile navigation copy this complete shape:
{{"kind":"base_action","observation_id":"CURRENT","note":"WHY",
"axis":"x|y|yaw","normalized_velocity":0.25,
"gripper":"open|hold|close"}}.
Do not omit kind, observation_id, note, or gripper."""

OSCILLATION_REPAIR = """A generic oscillation guard rejected repeated reversals on
the {axis} axis. Reobserve is unavailable and the two external RGB views are
sufficient. Return a bounded action with translation {axis}=0 and at least one other
translation axis nonzero. Choose the direction yourself from the images. Do not
repeat any recent command."""

SERVO_EXHAUSTED_REPAIR = """Both permitted public-RGB servo calibration attempts are
exhausted for this episode. Do not output servo_feature again. Continue with one
bounded nonzero `action` selected from the fresh RGB views, or return `give_up` if no
safe direct action exists. The harness will reject another servo_feature."""

FORMAT_REPAIR = """Your immediately previous response was not a complete single JSON
object. This is one bounded format-only retry on the same immutable observation.
Return exactly one complete allowed JSON object with no prose or markdown. Keep the
decision concise. Do not change the requested task or claim success."""


class _OfficialSuccess(Exception):
    """Internal runner-only stop; never serialized into a model-visible receipt."""


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _wait_json(path: Path, *, timeout_s: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            time.sleep(0.02)
    raise TimeoutError(f"timed out waiting for {path.name}")


def _wait_process_json(
    path: Path,
    *,
    process: subprocess.Popen[str],
    log_path: Path,
    timeout_s: float,
) -> dict[str, object]:
    """Wait for mailbox output while failing immediately if its owner exits."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            pass
        return_code = process.poll()
        if return_code is not None:
            try:
                log_tail = log_path.read_text(errors="replace")[-4_096:].strip()
            except FileNotFoundError:
                log_tail = "simulator log is missing"
            raise RuntimeError(
                f"simulator exited with code {return_code} before {path.name}: "
                f"{log_tail}"
            )
        time.sleep(0.02)
    raise TimeoutError(f"timed out waiting for {path.name}")


def _complete_with_format_retry(
    client: QwenClient,
    *,
    observation_id: str,
    system_prompt: str,
    instruction: str,
    public_state: dict[str, object],
    images: dict[str, bytes],
) -> tuple[Response | None, list[dict[str, object]]]:
    errors: list[dict[str, object]] = []
    arguments = {
        "observation_id": observation_id,
        "system_prompt": system_prompt,
        "instruction": instruction,
        "public_state": public_state,
        "images": images,
    }
    try:
        return client.complete(**arguments), errors
    except MalformedResponse as error:
        errors.append(error.evidence)
    arguments["instruction"] = f"{instruction}\n\n{FORMAT_REPAIR}"
    try:
        return client.complete(**arguments), errors
    except MalformedResponse as error:
        errors.append(error.evidence)
        return None, errors


def _serialize_actions(
    actions: list[dict[str, np.ndarray]],
) -> list[dict[str, list[float]]]:
    return [
        {key: np.asarray(value).reshape(-1).tolist() for key, value in action.items()}
        for action in actions
    ]


def _image_change(
    previous: dict[str, bytes] | None, current: dict[str, bytes]
) -> dict[str, float]:
    if previous is None:
        return {name: 0.0 for name in current}
    result: dict[str, float] = {}
    for name, value in current.items():
        old = np.asarray(
            Image.open(__import__("io").BytesIO(previous[name])), dtype=np.float32
        )
        new = np.asarray(
            Image.open(__import__("io").BytesIO(value)), dtype=np.float32
        )
        result[name] = float(np.mean(np.abs(new - old)))
    return result


def _read_images(observation: dict[str, object]) -> dict[str, bytes]:
    output: dict[str, bytes] = {}
    for name in ("left", "right", "wrist"):
        record = observation["images"][name]
        path = Path(record["path"])
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise RuntimeError("simulator image hash mismatch")
        output[name] = data
    return output


def _servo_observation(
    observation: dict[str, object], images: dict[str, bytes]
) -> ServoObservation:
    state = observation["public_state"]
    return ServoObservation(
        observation_id=str(observation["observation_id"]),
        left=np.asarray(
            Image.open(io.BytesIO(images["left"])).convert("RGB"), dtype=np.uint8
        ),
        right=np.asarray(
            Image.open(io.BytesIO(images["right"])).convert("RGB"), dtype=np.uint8
        ),
        eef_position_m=np.asarray(
            state["state.end_effector_position_relative"], dtype=np.float64
        ),
        finger_qpos=np.asarray(state["state.gripper_qpos"], dtype=np.float64),
    )


def _launch_simulator(
    *, task: str, seed: int, run: Path, log_handle: object
) -> subprocess.Popen[str]:
    root = Path("/home/jli/work/robocasa-inspect-official")
    python = root / ".venv/bin/python"
    driver = Path(
        "/home/jli/state/robocasa-inspect-official/nvidia-egl-580.159.04/rootfs"
    )
    state = Path("/home/jli/state/robocasa-inspect-official")
    lib = driver / "usr/lib/x86_64-linux-gnu"
    vendor = driver / "usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    xdg = run / "xdg"
    xdg.mkdir(mode=0o700)
    command = [
        "sudo",
        "-n",
        "/usr/bin/unshare",
        "-n",
        "--",
        "/usr/bin/setpriv",
        "--reuid=1001",
        "--regid=1001",
        "--groups=1001,44,992",
        "--inh-caps=-all",
        "--ambient-caps=-all",
        "--bounding-set=-all",
        "--",
        "/usr/bin/env",
        "-i",
        "HOME=/home/jli",
        "PATH=/usr/bin:/bin",
        f"PYTHONPATH={root}:{root / 'robocasa'}:{root / 'robosuite'}",
        f"ROBOCASA_ASSET_ROOT={root / 'robocasa/robocasa/models/assets'}",
        f"ROBOCASA_ASSET_MANIFEST={state / 'assets/content-manifest.json'}",
        f"ROBOCASA_RUN_DIR={run / 'sim'}",
        f"ROBOCASA_CHECKOUT_ROOT={root / 'robocasa'}",
        f"ROBOCASA_CACHE_ROOT={state}",
        "MUJOCO_GL=egl",
        "PYOPENGL_PLATFORM=egl",
        "EGL_PLATFORM=surfaceless",
        f"XDG_RUNTIME_DIR={xdg}",
        f"LD_LIBRARY_PATH={lib}:/usr/lib/x86_64-linux-gnu",
        f"__EGL_VENDOR_LIBRARY_FILENAMES={vendor}",
        str(python),
        "-m",
        "robocasa_inspect.sim_child",
        "--task",
        task,
        "--seed",
        str(seed),
        "--run-dir",
        str(run / "sim"),
    ]
    return subprocess.Popen(
        command, stdout=log_handle, stderr=subprocess.STDOUT, text=True
    )


def _instruction(
    observation: dict[str, object],
    *,
    receipts: list[dict[str, object]],
    change: dict[str, float],
    repair: str | None,
) -> str:
    context = {
        "task": observation["instruction"],
        "camera_order": ["left external", "right external", "wrist"],
        "public_state_order": list(observation["public_state"]),
        "recent_receipts": receipts[-12:],
        "mean_absolute_rgb_change_since_previous": change,
    }
    text = json.dumps(context, sort_keys=True, separators=(",", ":"))
    if repair:
        if repair.startswith("oscillation_"):
            text += "\n" + OSCILLATION_REPAIR.format(axis=repair.rsplit("_", 1)[-1])
        elif repair == "servo_recalibration_budget_exhausted":
            text += "\n" + SERVO_EXHAUSTED_REPAIR
        else:
            text += "\n" + REPAIR_TEXT.format(reason=repair)
    return text


def _terminal_status(run: Path) -> str:
    terminal = json.loads((run / "sim" / "simulator-terminal.json").read_text())
    if terminal.get("status") == "success":
        return "success"
    if terminal.get("status") == "finished_false":
        return "policy_finished_false"
    raise RuntimeError("finish did not produce a sealed official simulator outcome")


def _oscillation_axis(command: object, receipts: list[dict[str, object]]) -> str | None:
    movements = [receipt for receipt in receipts if "translation_m" in receipt]
    if getattr(command, "kind", None) != "action" or len(movements) < 2:
        return None
    current = np.asarray(command.translation_m, dtype=np.float64)
    older = np.asarray(movements[-2]["translation_m"], dtype=np.float64)
    latest = np.asarray(movements[-1]["translation_m"], dtype=np.float64)
    axis = int(np.argmax(np.abs(current)))
    if abs(current[axis]) < 0.015:
        return None
    if np.allclose(current, older, atol=1e-9, rtol=0) and np.allclose(
        current, -latest, atol=1e-9, rtol=0
    ):
        return "xyz"[axis]
    return None


def _repair_reason(
    command: Command,
    *,
    previous_gripper: str,
    servo_attempts: int,
    receipts: list[dict[str, object]],
) -> str | None:
    if is_zero_motion(command, previous_gripper=previous_gripper):
        return "zero_motion_while_task_active"
    if command.kind == "servo_feature" and servo_attempts >= 2:
        return "servo_recalibration_budget_exhausted"
    if axis := _oscillation_axis(command, receipts):
        return f"oscillation_axis_{axis}"
    return None


def _render_video(sim_run: Path, target: Path) -> int:
    frames = sorted((sim_run / "frames").glob("*"))
    mosaic_dir = sim_run.parent / "mosaics"
    mosaic_dir.mkdir(mode=0o700)
    for index, frame_dir in enumerate(frames):
        images = [
            Image.open(frame_dir / f"{name}.png").convert("RGB")
            for name in ("left", "right", "wrist")
        ]
        mosaic = Image.new("RGB", (768, 256))
        for camera_index, image in enumerate(images):
            mosaic.paste(image, (camera_index * 256, 0))
        path = mosaic_dir / f"{index:06d}.png"
        mosaic.save(path, format="PNG")
        path.chmod(0o600)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "4",
            "-i",
            str(mosaic_dir / "%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(target),
        ],
        check=True,
    )
    target.chmod(0o600)
    return len(frames)


def run_episode(
    *,
    task: str,
    seed: int,
    run: Path,
    max_decisions: int,
    anchor_audit_only: bool = False,
) -> dict[str, object]:
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    log_path = run / "simulator.log"
    log_handle = log_path.open("w")
    os.chmod(log_path, 0o600)
    identity, attestation = load_authority(
        Path(os.environ["QWEN_IDENTITY_MANIFEST"]),
        Path(os.environ["QWEN_SERVER_ATTESTATION"]),
    )
    token = Path(os.environ["QWEN_API_TOKEN_FILE"]).read_text().strip()
    client = QwenClient(
        base_url="http://127.0.0.1:8002/v1",
        api_key=token,
        identity=identity,
        attestation=attestation,
    )
    client.verify()
    process = _launch_simulator(task=task, seed=seed, run=run, log_handle=log_handle)
    started = time.monotonic()
    receipts: list[dict[str, object]] = []
    requests: list[dict[str, object]] = []
    previous_images: dict[str, bytes] | None = None
    gripper = "open"
    status = "incomplete"
    sequence = 0
    action_chunks = 0
    servo_attempts = 0
    reset_eef_position_m: np.ndarray | None = None
    try:
        for decision in range(max_decisions):
            if time.monotonic() - started > 1_200:
                status = "policy_failed_wall_budget"
                break
            observation = _wait_json(
                run / "sim" / "mailbox" / f"observation-{sequence:06d}.json",
                timeout_s=180,
            )
            if observation.get("sequence") != sequence:
                raise RuntimeError("observation sequence mismatch")
            if observation.get("success") is True:
                status = "success"
                break
            images = _read_images(observation)
            if reset_eef_position_m is None:
                reset_eef_position_m = np.asarray(
                    observation["public_state"]["state.end_effector_position_relative"],
                    dtype=np.float64,
                )
            change = _image_change(previous_images, images)
            response, format_errors = _complete_with_format_retry(
                client,
                observation_id=str(observation["observation_id"]),
                system_prompt=SYSTEM_PROMPT,
                instruction=_instruction(
                    observation, receipts=receipts, change=change, repair=None
                ),
                public_state=observation["public_state"],
                images=images,
            )
            if response is None:
                requests.append(
                    {
                        "decision": decision,
                        "observation_id": observation["observation_id"],
                        "format_errors": format_errors,
                        "post_repair_status": "invalid_json",
                    }
                )
                status = "policy_failed_invalid_json"
                break
            repair_reason: str | None = None
            adapter_changes: list[str] = []
            try:
                command = decode_command(
                    response.command, observation_id=str(observation["observation_id"])
                )
                repair_reason = _repair_reason(
                    command,
                    previous_gripper=gripper,
                    servo_attempts=servo_attempts,
                    receipts=receipts,
                )
            except ValueError:
                try:
                    command, adapter_changes = adapt_bounded_action(
                        response.command,
                        observation_id=str(observation["observation_id"]),
                    )
                    repair_reason = _repair_reason(
                        command,
                        previous_gripper=gripper,
                        servo_attempts=servo_attempts,
                        receipts=receipts,
                    )
                except ValueError:
                    repair_reason = "invalid_closed_command"
            repair_evidence = None
            repaired_command = None
            if repair_reason:
                repaired, repair_format_errors = _complete_with_format_retry(
                    client,
                    observation_id=str(observation["observation_id"]),
                    system_prompt=SYSTEM_PROMPT,
                    instruction=_instruction(
                        observation,
                        receipts=receipts,
                        change=change,
                        repair=repair_reason,
                    ),
                    public_state=observation["public_state"],
                    images=images,
                )
                if repaired is None:
                    requests.append(
                        {
                            "decision": decision,
                            "observation_id": observation["observation_id"],
                            "command": response.command,
                            "evidence": response.evidence,
                            "format_errors": format_errors,
                            "repair_reason": repair_reason,
                            "repair_format_errors": repair_format_errors,
                            "post_repair_status": "invalid_json",
                        }
                    )
                    status = "policy_failed_invalid_json"
                    break
                repaired_command = repaired.command
                repair_evidence = repaired.evidence
                try:
                    command = decode_command(
                        repaired.command,
                        observation_id=str(observation["observation_id"]),
                    )
                except ValueError:
                    try:
                        command, repair_adapter_changes = adapt_bounded_action(
                            repaired.command,
                            observation_id=str(observation["observation_id"]),
                        )
                        adapter_changes.extend(
                            f"repair:{change}" for change in repair_adapter_changes
                        )
                    except ValueError:
                        requests.append(
                            {
                                "decision": decision,
                                "observation_id": observation["observation_id"],
                                "command": response.command,
                                "evidence": response.evidence,
                                "repair_reason": repair_reason,
                                "repair_evidence": repair_evidence,
                                "repaired_command": repaired_command,
                                "post_repair_status": "invalid_closed_command",
                            }
                        )
                        status = "policy_failed_invalid_repair"
                        break
                repeated_reason = _repair_reason(
                    command,
                    previous_gripper=gripper,
                    servo_attempts=servo_attempts,
                    receipts=receipts,
                )
                if repeated_reason is not None:
                    status = f"policy_failed_repair_{repeated_reason}"
                    break
            requests.append(
                {
                    "decision": decision,
                    "observation_id": observation["observation_id"],
                    "command": response.command,
                    "evidence": response.evidence,
                    "format_errors": format_errors,
                    "repair_reason": repair_reason,
                    "adapter_changes": adapter_changes,
                    "repair_evidence": repair_evidence,
                    "repaired_command": repaired_command,
                    "post_repair_command": dataclasses.asdict(command),
                }
            )
            if command.kind == "give_up":
                status = "policy_gave_up"
                break
            if command.kind == "finish":
                _atomic_json(
                    run / "sim" / "mailbox" / f"command-{sequence:06d}.json",
                    {
                        "schema": "robocasa-inspect-command/v1",
                        "sequence": sequence,
                        "kind": "finish",
                    },
                )
                status = "finish_requested"
                break
            if command.kind == "base_action":
                if action_chunks >= 90:
                    status = "policy_failed_step_budget"
                    break
                if command.gripper != "hold":
                    gripper = command.gripper
                actions = base_pulse_chunk(
                    axis=command.base_axis,
                    normalized_velocity=command.base_velocity,
                )
                close = 1.0 if gripper == "close" else 0.0
                for action in actions:
                    action["action.gripper_close"] = np.asarray([close])
                _atomic_json(
                    run / "sim" / "mailbox" / f"command-{sequence:06d}.json",
                    {
                        "schema": "robocasa-inspect-command/v1",
                        "sequence": sequence,
                        "kind": "action",
                        "actions": _serialize_actions(actions),
                    },
                )
                receipts.append(
                    {
                        "kind": "base_action",
                        "observation_id": observation["observation_id"],
                        "accepted": True,
                        "note": command.note,
                        "axis": command.base_axis,
                        "normalized_velocity": command.base_velocity,
                        "gripper": gripper,
                    }
                )
                action_chunks += 1
                previous_images = images
                sequence += 1
                continue
            if command.kind == "servo_feature":
                servo_attempts += 1
                current_harness_observation_id = str(observation["observation_id"])

                def execute_servo(
                    translation: tuple[float, float, float] | None,
                    requested_gripper: str,
                    label: str,
                ) -> ServoObservation:
                    nonlocal sequence, gripper, action_chunks, previous_images
                    nonlocal current_harness_observation_id
                    if translation is None:
                        payload: dict[str, object] = {
                            "schema": "robocasa-inspect-command/v1",
                            "sequence": sequence,
                            "kind": "observe",
                        }
                    else:
                        if action_chunks >= 90:
                            raise ValueError("official 450-step budget exhausted")
                        harness_command = Command(
                            kind="action",
                            observation_id=current_harness_observation_id,
                            note=f"public RGB harness: {label}",
                            translation_m=translation,
                            gripper=requested_gripper,
                        )
                        actions, gripper = official_action_chunk(
                            harness_command, previous_gripper=gripper
                        )
                        payload = {
                            "schema": "robocasa-inspect-command/v1",
                            "sequence": sequence,
                            "kind": "action",
                            "actions": _serialize_actions(actions),
                        }
                        action_chunks += 1
                        receipts.append(
                            {
                                "source": "public_rgb_harness",
                                "label": label,
                                "observation_id": current_harness_observation_id,
                                "accepted": True,
                                "translation_m": list(translation),
                                "rotation_axis_angle_rad": [0.0, 0.0, 0.0],
                                "gripper": gripper,
                            }
                        )
                    _atomic_json(
                        run / "sim" / "mailbox" / f"command-{sequence:06d}.json",
                        payload,
                    )
                    sequence += 1
                    next_observation = _wait_json(
                        run / "sim" / "mailbox" / f"observation-{sequence:06d}.json",
                        timeout_s=180,
                    )
                    if next_observation.get("sequence") != sequence:
                        raise RuntimeError("harness observation sequence mismatch")
                    next_images = _read_images(next_observation)
                    previous_images = next_images
                    current_harness_observation_id = str(
                        next_observation["observation_id"]
                    )
                    if next_observation.get("success") is True:
                        raise _OfficialSuccess
                    return _servo_observation(next_observation, next_images)

                try:
                    servo = run_servo_session(
                        command,
                        _servo_observation(observation, images),
                        reset_eef_position_m=reset_eef_position_m,
                        execute=execute_servo,
                        audit_only=anchor_audit_only,
                    )
                except _OfficialSuccess:
                    requests[-1]["harness_status"] = "official_terminated"
                    status = "success"
                    break
                requests[-1]["harness_status"] = servo.status
                requests[-1]["harness_report"] = servo.report
                if selection := servo.report.get("anchor_selection"):
                    overlay = run / f"anchor-overlay-decision-{decision:03d}.png"
                    if servo.audit_images is None:
                        overlay_images = {
                            name: np.asarray(
                                Image.open(io.BytesIO(images[name])).convert("RGB"),
                                dtype=np.uint8,
                            )
                            for name in ("left", "right")
                        }
                    else:
                        overlay_images = servo.audit_images
                    requests[-1]["anchor_overlay"] = render_anchor_overlay(
                        overlay_images, selection, overlay
                    )
                receipts.append(
                    {
                        "source": "public_rgb_harness",
                        "kind": servo.status,
                        "accepted": servo.status == "aligned",
                        "observation_id": servo.final_observation.observation_id,
                        "detail": (
                            "two-view pixel alignment verified"
                            if servo.status == "aligned"
                            else str(servo.report.get("failure", "servo unavailable"))
                        ),
                    }
                )
                if anchor_audit_only:
                    status = (
                        "anchor_audit_required"
                        if servo.status == "anchor_audit_required"
                        else "gripper_anchor_invalid"
                    )
                    break
                continue
            if command.kind != "action":
                status = "policy_failed_non_action"
                break
            if action_chunks >= 90:
                status = "policy_failed_step_budget"
                break
            actions, gripper = official_action_chunk(command, previous_gripper=gripper)
            _atomic_json(
                run / "sim" / "mailbox" / f"command-{sequence:06d}.json",
                {
                    "schema": "robocasa-inspect-command/v1",
                    "sequence": sequence,
                    "kind": "action",
                    "actions": _serialize_actions(actions),
                },
            )
            receipt = {
                "decision": decision,
                "observation_id": observation["observation_id"],
                "accepted": True,
                "note": command.note,
                "translation_m": list(command.translation_m),
                "rotation_axis_angle_rad": list(command.rotation_axis_angle_rad),
                "gripper": gripper,
            }
            receipts.append(receipt)
            action_chunks += 1
            previous_images = images
            sequence += 1
        else:
            status = "policy_failed_decision_budget"

        if status != "finish_requested" and process.poll() is None:
            _atomic_json(
                run / "sim" / "mailbox" / f"command-{sequence:06d}.json",
                {
                    "schema": "robocasa-inspect-command/v1",
                    "sequence": sequence,
                    "kind": "close",
                },
            )
        return_code = process.wait(timeout=180)
        if status == "finish_requested":
            if return_code != 0:
                raise RuntimeError(f"simulator exited {return_code} during finish")
            status = _terminal_status(run)
        if return_code != 0 and status.startswith("policy_"):
            raise RuntimeError(f"simulator exited {return_code}")
        verify_process(attestation)
        video = run / f"{task}-seed{seed}-{status}.mp4"
        frame_count = _render_video(run / "sim", video)
        result = {
            "schema": "robocasa-inspect-episode/v1",
            "task": task,
            "seed": seed,
            "status": status,
            "success": status == "success",
            "decisions": action_chunks,
            "model_decisions": len(requests),
            "action_chunks": action_chunks,
            "simulator_steps": action_chunks * 5,
            "servo_attempts": servo_attempts,
            "anchor_audit_only": anchor_audit_only,
            "requests": requests,
            "receipts": receipts,
            "frame_sets": frame_count,
            "video": str(video),
            "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "wall_s": time.monotonic() - started,
            "snapshot_digest": identity.get("snapshot_digest"),
            "served_model_id": attestation.get("served_model_id"),
            "system_prompt_sha256": hashlib.sha256(
                SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
        }
        _atomic_json(run / "result.json", result)
        return result
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        log_handle.close()
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="CloseDrawer")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-decisions", type=int, default=90)
    parser.add_argument("--anchor-audit-only", action="store_true")
    args = parser.parse_args()
    result = run_episode(
        task=args.task,
        seed=args.seed,
        run=args.run_dir.resolve(),
        max_decisions=args.max_decisions,
        anchor_audit_only=args.anchor_audit_only,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("task", "seed", "status", "decisions", "video")
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
