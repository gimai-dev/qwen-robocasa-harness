"""Network-isolated official RoboCasa simulator process with a file mailbox."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

import robocasa  # noqa: F401

from .active_skills import TerminalOutcomePersister
from .camera_geometry import official_camera_calibration
from .contracts import CAMERAS, project_observation


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _network_denied() -> str:
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=0.2):
            pass
    except OSError as error:
        return type(error).__name__
    raise RuntimeError("simulator network namespace is not isolated")


def _publish_observation(
    raw: dict[str, object],
    *,
    environment: object,
    run: Path,
    episode: str,
    sequence: int,
) -> dict[str, object]:
    public = project_observation(raw, episode=episode, sequence=sequence)
    frame_dir = run / "frames" / f"{sequence:06d}"
    frame_dir.mkdir(parents=True, mode=0o700)
    images: dict[str, dict[str, str]] = {}
    for label, key in zip(("left", "right", "wrist"), CAMERAS, strict=True):
        target = frame_dir / f"{label}.png"
        Image.fromarray(public.images[key]).save(target, format="PNG")
        target.chmod(0o600)
        images[label] = {
            "path": str(target),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    record = {
        "schema": "robocasa-inspect-observation/v1",
        "episode": episode,
        "sequence": sequence,
        "observation_id": public.observation_id,
        "instruction": public.instruction,
        "images": images,
        "public_state": public.state_groups,
        "camera_calibration": official_camera_calibration(environment),
    }
    _atomic_json(run / "mailbox" / f"observation-{sequence:06d}.json", record)
    return record


def _official_terminal_success(environment: object) -> bool:
    """Invoke the official task predicate once, only for an explicit finish."""
    task = environment.unwrapped
    # This private method is RoboCasa's official task predicate and is intentionally
    # isolated here so no model-visible or intermediate path can query it.
    success = task._check_success()
    if not isinstance(success, (bool, np.bool_)):
        raise TypeError("official success predicate drift")
    return bool(success)


def _terminal_snapshot_sha256(environment: object) -> str:
    """Hash frozen simulator state without exposing it to the control process."""
    simulator = environment.unwrapped.sim
    parts = [
        np.asarray(simulator.data.qpos, dtype="<f8").reshape(-1),
        np.asarray(simulator.data.qvel, dtype="<f8").reshape(-1),
        np.asarray(simulator.data.act, dtype="<f8").reshape(-1),
        np.asarray(simulator.data.ctrl, dtype="<f8").reshape(-1),
        np.asarray([simulator.data.time], dtype="<f8"),
    ]
    digest = hashlib.sha256()
    for part in parts:
        digest.update(np.asarray([len(part)], dtype="<u8").tobytes())
        digest.update(part.tobytes(order="C"))
    return digest.hexdigest()


def _read_command(run: Path, sequence: int, timeout_s: float) -> dict[str, object]:
    target = run / "mailbox" / f"command-{sequence:06d}.json"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            value = json.loads(target.read_text())
        except FileNotFoundError:
            time.sleep(0.01)
            continue
        if value.get("sequence") != sequence:
            raise RuntimeError("mailbox command sequence mismatch")
        return value
    raise TimeoutError("simulator command mailbox timed out")


def run(task: str, seed: int, run: Path) -> None:
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    (run / "mailbox").mkdir(mode=0o700)
    (run / "frames").mkdir(mode=0o700)
    episode_tmp = run / "episode-tmp"
    episode_tmp.mkdir(mode=0o700)
    os.environ["ROBOCASA_EPISODE_TMPDIR"] = str(episode_tmp)
    episode = hashlib.sha256(f"{task}:{seed}:{run.name}".encode()).hexdigest()[:32]
    metadata = {
        "schema": "robocasa-inspect-simulator/v1",
        "task": task,
        "seed": seed,
        "episode": episode,
        "network_probe": _network_denied(),
    }
    _atomic_json(run / "simulator.json", metadata)
    environment = gym.make(f"robocasa/{task}", split="pretrain", seed=seed)
    terminal: dict[str, object] = {"status": "incomplete"}
    accepted_commands: list[dict[str, object]] = []
    try:
        raw, _ = environment.reset()
        sequence = 0
        _publish_observation(
            raw,
            environment=environment,
            run=run,
            episode=episode,
            sequence=sequence,
        )
        while True:
            command = _read_command(run, sequence, timeout_s=1_300)
            if command.get("kind") == "close":
                terminal = {"status": "closed", "sequence": sequence}
                break
            if command.get("kind") == "finish":
                if set(command) != {"schema", "sequence", "kind"}:
                    raise RuntimeError("finish command violates the closed schema")
                # The action loop is halted while this command is handled. Capture
                # the terminal state before the one-shot official evaluator runs.
                snapshot_sha256 = _terminal_snapshot_sha256(environment)
                outcome = TerminalOutcomePersister(run).seal(
                    trace=accepted_commands,
                    receipts=[{"sequence": sequence, "accepted": True}],
                    terminal_snapshot_sha256=snapshot_sha256,
                    evaluate=lambda: _official_terminal_success(environment),
                )
                terminal = {
                    "status": "success" if outcome["success"] else "finished_false",
                    "sequence": sequence,
                    "terminal_outcome_sha256": hashlib.sha256(
                        (run / "terminal-outcome.json").read_bytes()
                    ).hexdigest(),
                    "terminal_snapshot_sha256": snapshot_sha256,
                }
                break
            if command.get("kind") == "observe":
                if set(command) != {"schema", "sequence", "kind"}:
                    raise RuntimeError(
                        "observe mailbox command violates the closed schema"
                    )
                sequence += 1
                accepted_commands.append(
                    {"sequence": sequence - 1, "kind": "observe"}
                )
                _publish_observation(
                    raw,
                    environment=environment,
                    run=run,
                    episode=episode,
                    sequence=sequence,
                )
                continue
            if command.get("kind") != "action" or not isinstance(
                command.get("actions"), list
            ):
                raise RuntimeError("invalid simulator mailbox command")
            if not 1 <= len(command["actions"]) <= 5:
                raise RuntimeError(
                    "simulator action chunk must contain one to five steps"
                )
            for serialized in command["actions"]:
                action = {
                    key: np.asarray(value, dtype=np.float64)
                    for key, value in serialized.items()
                }
                raw, _, _, _, _ = environment.step(action)
            accepted_commands.append(
                {
                    "sequence": sequence,
                    "kind": "action",
                    "actions_sha256": hashlib.sha256(
                        json.dumps(
                            command["actions"],
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest(),
                }
            )
            sequence += 1
            _publish_observation(
                raw,
                environment=environment,
                run=run,
                episode=episode,
                sequence=sequence,
            )
    except Exception as error:
        terminal = {"status": "infrastructure_blocked", "error": repr(error)}
        raise
    finally:
        environment.close()
        terminal["episode_tmp_empty"] = not any(episode_tmp.iterdir())
        _atomic_json(run / "simulator-terminal.json", terminal)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.task, args.seed, args.run_dir.resolve())


if __name__ == "__main__":
    main()
