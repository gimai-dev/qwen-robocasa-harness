"""Deterministic production-path safety smoke for native joint execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from .joint_runner import _atomic_json


def _isolated_child_command(
    *,
    module: str,
    task: str,
    seed: int,
    run: Path,
    child_run: Path,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Build the shared reduced-privilege, network-disabled smoke command."""

    if not module.startswith("adaptive.") or not task:
        raise ValueError("isolated smoke module and task must be explicit")
    release = Path(__file__).resolve().parents[1]
    root = Path("/home/jli/work/robocasa-inspect-official")
    python = root / ".venv/bin/python"
    driver = Path(
        "/home/jli/state/robocasa-inspect-official/nvidia-egl-580.173.02/rootfs"
    )
    state = Path("/home/jli/state/robocasa-inspect-official")
    lib = driver / "usr/lib/x86_64-linux-gnu"
    vendor = driver / "usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    xdg = run / "xdg"
    xdg.mkdir(mode=0o700)
    return [
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
        f"PYTHONPATH={release}:{root}:{root / 'robocasa'}:{root / 'robosuite'}",
        f"ROBOCASA_ASSET_ROOT={root / 'robocasa/robocasa/models/assets'}",
        f"ROBOCASA_ASSET_MANIFEST={state / 'assets/content-manifest.json'}",
        f"ROBOCASA_RUN_DIR={child_run}",
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
        module,
        "--task",
        task,
        "--seed",
        str(seed),
        "--run-dir",
        str(child_run),
        *extra_args,
    ]


def _vector(value: object, width: int) -> list[float] | None:
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


def evaluate_safety_smoke(
    evidence: Mapping[str, object],
) -> tuple[bool, dict[str, bool]]:
    reset = _vector(evidence.get("reset_qpos"), 7)
    outbound = _vector(evidence.get("outbound_qpos"), 7)
    returned = _vector(evidence.get("return_qpos"), 7)
    reset_gripper = _vector(evidence.get("reset_gripper_qpos"), 2)
    closed_gripper = _vector(evidence.get("closed_gripper_qpos"), 2)
    reopened_gripper = _vector(evidence.get("reopened_gripper_qpos"), 2)
    raw_steps = evidence.get("maximum_commanded_steps")
    steps = (
        [float(item) for item in raw_steps]
        if isinstance(raw_steps, Sequence)
        and not isinstance(raw_steps, (str, bytes))
        and all(
            not isinstance(item, bool) and isinstance(item, (int, float))
            for item in raw_steps
        )
        else []
    )
    outbound_ok = (
        reset is not None
        and outbound is not None
        and outbound[0] - reset[0] >= 0.02
    )
    return_ok = (
        reset is not None
        and returned is not None
        and max(abs(a - b) for a, b in zip(returned, reset, strict=True)) <= 0.03
    )
    closed_endpoints = (0.0, 0.0)
    open_endpoints = (0.04, -0.04)
    close_direction = (
        reset_gripper is not None
        and closed_gripper is not None
        and all(
            abs(closed - target) < abs(reset_value - target) - 1e-4
            for reset_value, closed, target in zip(
                reset_gripper, closed_gripper, closed_endpoints, strict=True
            )
        )
    )
    reopen_direction = (
        reset_gripper is not None
        and closed_gripper is not None
        and reopened_gripper is not None
        and all(
            abs(reopened - target) < abs(closed - target) - 1e-4
            for closed, reopened, target in zip(
                closed_gripper, reopened_gripper, open_endpoints, strict=True
            )
        )
    )
    minimum_margin = evidence.get("minimum_hard_limit_margin")
    action_count = evidence.get("action_count")
    terminal_action_count = evidence.get("terminal_action_count")
    checks = {
        "outbound_joint1_motion": outbound_ok,
        "return_error_bounded": return_ok,
        "commanded_steps_bounded": bool(steps)
        and all(math.isfinite(item) and item <= 0.027 + 1e-12 for item in steps),
        "hard_limit_margin_nonnegative": (
            not isinstance(minimum_margin, bool)
            and isinstance(minimum_margin, (int, float))
            and math.isfinite(float(minimum_margin))
            and float(minimum_margin) >= 0.0
        ),
        "gripper_close_direction": close_direction,
        "gripper_reopen_direction": reopen_direction,
        "action_accounting_exact": (
            isinstance(action_count, int)
            and not isinstance(action_count, bool)
            and action_count > 0
            and action_count == terminal_action_count
        ),
        "simulator_clean": evidence.get("simulator_error") is None,
    }
    return all(checks.values()), checks


def _gripper_qpos(observation: Mapping[str, object]) -> list[float]:
    state = observation.get("public_state")
    if not isinstance(state, Mapping):
        raise RuntimeError("smoke observation lacks public state")
    value = _vector(state.get("state.gripper_qpos"), 2)
    if value is None:
        raise RuntimeError("smoke observation has invalid gripper qpos")
    return value


def _arm_qpos(observation: Mapping[str, object]) -> list[float]:
    state = observation.get("public_state")
    if not isinstance(state, Mapping):
        raise RuntimeError("smoke observation lacks public state")
    value = _vector(state.get("state.arm_joint_position"), 7)
    if value is None:
        raise RuntimeError("smoke observation has invalid arm qpos")
    return value


def run_safety_smoke(*, task: str, seed: int, run: Path) -> dict[str, object]:
    from robocasa_inspect.runner import _wait_process_json

    started = time.monotonic()
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    log_path = run / "simulator.log"
    log_handle = log_path.open("w")
    os.chmod(log_path, 0o600)
    process = subprocess.Popen(
        _isolated_child_command(
            module="adaptive.joint_sim_child",
            task=task,
            seed=seed,
            run=run,
            child_run=run / "sim",
        ),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    sequence = 0
    evidence: dict[str, object] = {
        "schema": "robocasa-inspect-joint-safety-evidence/v1",
        "task": task,
        "seed": seed,
        "reset_qpos": None,
        "outbound_qpos": None,
        "return_qpos": None,
        "reset_gripper_qpos": None,
        "closed_gripper_qpos": None,
        "reopened_gripper_qpos": None,
        "maximum_commanded_steps": [],
        "minimum_hard_limit_margin": None,
        "action_count": 0,
        "terminal_action_count": None,
        "simulator_error": None,
    }

    def wait_observation() -> dict[str, object]:
        value = _wait_process_json(
            run / "sim" / "mailbox" / f"observation-{sequence:06d}.json",
            process=process,
            log_path=log_path,
            timeout_s=180,
        )
        if value.get("sequence") != sequence:
            raise RuntimeError("safety smoke observation sequence mismatch")
        return value

    def move(
        observation: Mapping[str, object],
        *,
        endpoint: list[float],
        explicit_mask: list[bool],
        gripper_open: float,
    ) -> tuple[dict[str, object], dict[str, object]]:
        nonlocal sequence
        _atomic_json(
            run / "sim" / "mailbox" / f"command-{sequence:06d}.json",
            {
                "schema": "robocasa-inspect-joint-command/v1",
                "sequence": sequence,
                "kind": "move_joints",
                "observation_id": observation["observation_id"],
                "endpoint": endpoint,
                "explicit_mask": explicit_mask,
                "gripper_open": gripper_open,
                "max_actions": 32,
            },
        )
        sequence += 1
        fresh = wait_observation()
        execution = fresh.get("execution")
        if not isinstance(execution, Mapping):
            raise RuntimeError("safety smoke lacks execution receipt")
        if execution.get("accepted") is not True:
            raise RuntimeError("safety smoke command was not accepted")
        return fresh, dict(execution)

    executions: list[dict[str, object]] = []
    try:
        observation = wait_observation()
        reset = _arm_qpos(observation)
        evidence["reset_qpos"] = reset
        evidence["reset_gripper_qpos"] = _gripper_qpos(observation)

        outbound_target = list(reset)
        outbound_target[0] += 0.05
        observation, execution = move(
            observation,
            endpoint=outbound_target,
            explicit_mask=[True, False, False, False, False, False, False],
            gripper_open=1.0,
        )
        executions.append(execution)
        evidence["outbound_qpos"] = _arm_qpos(observation)

        return_target = _arm_qpos(observation)
        return_target[0] = reset[0]
        observation, execution = move(
            observation,
            endpoint=return_target,
            explicit_mask=[True, False, False, False, False, False, False],
            gripper_open=1.0,
        )
        executions.append(execution)
        evidence["return_qpos"] = _arm_qpos(observation)

        hold = _arm_qpos(observation)
        observation, execution = move(
            observation,
            endpoint=hold,
            explicit_mask=[False] * 7,
            gripper_open=0.0,
        )
        executions.append(execution)
        evidence["closed_gripper_qpos"] = _gripper_qpos(observation)

        hold = _arm_qpos(observation)
        observation, execution = move(
            observation,
            endpoint=hold,
            explicit_mask=[False] * 7,
            gripper_open=1.0,
        )
        executions.append(execution)
        evidence["reopened_gripper_qpos"] = _gripper_qpos(observation)

        _atomic_json(
            run / "sim" / "mailbox" / f"command-{sequence:06d}.json",
            {
                "schema": "robocasa-inspect-joint-command/v1",
                "sequence": sequence,
                "kind": "close",
            },
        )
        return_code = process.wait(timeout=180)
        if return_code != 0:
            raise RuntimeError(f"safety smoke simulator exited {return_code}")
        terminal = json.loads(
            (run / "sim" / "simulator-terminal.json").read_text()
        )
        evidence["maximum_commanded_steps"] = [
            execution["maximum_commanded_step"] for execution in executions
        ]
        evidence["minimum_hard_limit_margin"] = min(
            float(execution["minimum_hard_limit_margin"])
            for execution in executions
        )
        evidence["action_count"] = sum(
            int(execution["step_count"]) for execution in executions
        )
        evidence["terminal_action_count"] = terminal.get("simulator_actions")
    except Exception as error:
        evidence["simulator_error"] = f"{type(error).__name__}: {error}"[:500]
        evidence["simulator_error_sha256"] = hashlib.sha256(
            repr(error).encode()
        ).hexdigest()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        log_handle.close()

    passed, checks = evaluate_safety_smoke(evidence)
    result = {
        "schema": "robocasa-inspect-joint-safety-smoke/v1",
        "passed": passed,
        "checks": checks,
        "evidence": evidence,
        "artifact_root": str(run),
        "wall_s": time.monotonic() - started,
    }
    _atomic_json(run / "joint-safety-smoke.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="OpenToasterOvenDoor")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_safety_smoke(
        task=args.task, seed=args.seed, run=args.run_dir.resolve()
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
