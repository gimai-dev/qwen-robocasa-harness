"""Driver-side executor: sandboxed simulator child, mailbox protocol, IK conversion.

The executor turns a validated numerical ``Action`` into one child command. It
performs IK, straight-line interpolation and bound checks only; it never
chooses targets. A rejected action returns a receipt with zero executed steps.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .actions import Action, BASE_MOTION_STEPS, SLOT_STEPS, ee_rotation_matrix
from .kinematics import plan_pose_segment, solve_pose_multistart, world_to_base

SIM_ROOT = Path("/home/jli/work/robocasa-inspect-official")
SIM_STATE = Path("/home/jli/state/robocasa-inspect-official")
EGL_ROOTFS = SIM_STATE / "nvidia-egl-580.173.02/rootfs"
RELEASE_ROOT = Path(__file__).resolve().parents[1]
COMMAND_SCHEMA = "qwen-direct-command/v1"


@dataclass
class Receipt:
    status: str                 # completed | partial | unreachable | invalid | budget_exhausted | stop | infrastructure
    action: Action | None
    steps: int
    detail: dict
    child: dict | None = None   # raw execution record from the child

    def summary(self) -> dict:
        out = {"status": self.status, "steps": self.steps, **self.detail}
        if self.child is not None:
            c = self.child
            keep = ("steps_executed", "waypoints_reached", "waypoints_total", "final_joint_error_rad",
                    "peak_force_n", "tcp_world_after_m", "tcp_world_quat_after_xyzw", "gripper_width_after_m",
                    "base_world_after_m", "base_yaw_after_rad", "budget_truncated", "tracking_pauses")
            out.update({k: _round(c[k]) for k in keep if k in c})
            if "tcp_world_after_m" in c and "tcp_world_before_m" in c:
                out["tcp_moved_m"] = [round(a - b, 4) for a, b in zip(c["tcp_world_after_m"], c["tcp_world_before_m"])]
            if "base_world_after_m" in c and "base_world_before_m" in c:
                out["base_moved_m"] = [round(a - b, 4) for a, b in zip(c["base_world_after_m"], c["base_world_before_m"])]
                if self.action is not None and self.action.kind == "base" and max(abs(v) for v in out["base_moved_m"][:2]) < 0.01 and self.action.axis != "yaw":
                    out["note"] = "the base barely moved: it is probably blocked by furniture in that direction"
        return out


def _round(value: object, digits: int = 4) -> object:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [_round(v, digits) for v in value]
    return value


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


class Simulator:
    """Owns one sandboxed child process and the mailbox conversation."""

    def __init__(self, *, task: str, seed: int, run: Path, scenes: Path, action_budget: int,
                 wall_budget_s: float, restore_from: Path | None = None) -> None:
        self.task, self.seed, self.run, self.scenes = task, seed, run, scenes
        self.action_budget, self.wall_budget_s, self.restore_from = action_budget, wall_budget_s, restore_from
        self.sim = run / "sim"
        self.sequence = 0
        self.process: subprocess.Popen[str] | None = None
        self.log_path = run / "simulator.log"
        self.observation: dict | None = None

    def launch(self) -> dict:
        self.run.mkdir(parents=True, exist_ok=True)
        xdg = self.run / "xdg"
        xdg.mkdir(mode=0o700, exist_ok=True)
        self.scenes.mkdir(parents=True, exist_ok=True)
        lib = EGL_ROOTFS / "usr/lib/x86_64-linux-gnu"
        vendor = EGL_ROOTFS / "usr/share/glvnd/egl_vendor.d/10_nvidia.json"
        command = [
            "sudo", "-n", "/usr/bin/unshare", "-n", "--", "/usr/bin/setpriv",
            "--reuid=1001", "--regid=1001", "--groups=1001,44,992",
            "--inh-caps=-all", "--ambient-caps=-all", "--bounding-set=-all", "--",
            "/usr/bin/env", "-i", "HOME=/home/jli", "PATH=/usr/bin:/bin",
            f"PYTHONPATH={RELEASE_ROOT}:{RELEASE_ROOT / 'runtime'}:{SIM_ROOT}:{SIM_ROOT / 'robocasa'}:{SIM_ROOT / 'robosuite'}",
            f"ROBOCASA_ASSET_ROOT={SIM_ROOT / 'robocasa/robocasa/models/assets'}",
            f"ROBOCASA_ASSET_MANIFEST={SIM_STATE / 'assets/content-manifest.json'}",
            f"ROBOCASA_RUN_DIR={self.sim}", f"ROBOCASA_CHECKOUT_ROOT={SIM_ROOT / 'robocasa'}",
            f"ROBOCASA_CACHE_ROOT={SIM_STATE}",
            "MUJOCO_GL=egl", "PYOPENGL_PLATFORM=egl", "EGL_PLATFORM=surfaceless",
            f"XDG_RUNTIME_DIR={xdg}", f"LD_LIBRARY_PATH={lib}:/usr/lib/x86_64-linux-gnu",
            f"__EGL_VENDOR_LIBRARY_FILENAMES={vendor}",
            str(SIM_ROOT / ".venv/bin/python"), "-m", "direct.sim_child",
            "--task", self.task, "--seed", str(self.seed), "--run-dir", str(self.sim),
            "--scenes-dir", str(self.scenes), "--action-budget", str(self.action_budget),
            "--wall-budget-s", str(self.wall_budget_s),
        ]
        if self.restore_from is not None:
            command += ["--restore-from", str(self.restore_from)]
        (self.run / "launch.json").write_text(json.dumps(command, indent=1))
        self.log = self.log_path.open("w")
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT, text=True)
        self.observation = self._wait_observation(timeout_s=600)
        return self.observation

    def _wait_observation(self, *, timeout_s: float) -> dict:
        path = self.sim / "mailbox" / f"observation-{self.sequence:06d}.json"
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                value = json.loads(path.read_text())
                if value.get("sequence") != self.sequence:
                    raise RuntimeError("observation sequence mismatch")
                return value
            except FileNotFoundError:
                pass
            code = self.process.poll() if self.process else None
            if code is not None:
                tail = self.log_path.read_text(errors="replace")[-3000:]
                raise RuntimeError(f"simulator exited with code {code} before {path.name}:\n{tail}")
            time.sleep(0.02)
        raise TimeoutError(f"timed out waiting for {path.name}")

    def send(self, command: Mapping[str, object], *, timeout_s: float = 300) -> dict:
        payload = {"schema": COMMAND_SCHEMA, "sequence": self.sequence, **command}
        if command["kind"] in ("slot", "base", "snapshot", "inspect"):
            payload["observation_id"] = self.observation["observation_id"]
        _atomic_json(self.sim / "mailbox" / f"command-{self.sequence:06d}.json", payload)
        if command["kind"] in ("finish", "close"):
            return {}
        self.sequence += 1
        self.observation = self._wait_observation(timeout_s=timeout_s)
        return self.observation

    def finish(self) -> dict:
        self.send({"kind": "finish"})
        return self._terminal()

    def close(self) -> dict:
        if self.process is not None and self.process.poll() is None:
            try:
                self.send({"kind": "close"})
            except Exception:
                pass
        return self._terminal()

    def _terminal(self) -> dict:
        if self.process is not None:
            try:
                self.process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.log.close()
        path = self.sim / "simulator-terminal.json"
        outcome = json.loads(path.read_text()) if path.exists() else {"status": "missing_terminal"}
        terminal_outcome = self.sim / "terminal-outcome.json"
        if terminal_outcome.exists():
            outcome["terminal_outcome"] = json.loads(terminal_outcome.read_text())
        return outcome

    def steps_used(self) -> int:
        return int(self.observation["steps_used"])

    def steps_left(self) -> int:
        return self.action_budget - self.steps_used()


def execute(sim: Simulator, action: Action, *, slot_steps: int = SLOT_STEPS,
            base_motion_steps: int = BASE_MOTION_STEPS) -> Receipt:
    """Convert one action into a child command; return the executed receipt."""
    state = sim.observation["public_state"]
    q = [float(v) for v in state["arm_q_rad"]]
    gripper = action.gripper if action.gripper is not None else int(round(float(sim.observation["gripper_command"])))
    if sim.steps_left() <= 0:
        return Receipt("budget_exhausted", action, 0, {"reason": "simulator step budget exhausted"})
    if action.kind == "stop":
        return Receipt("stop", action, 0, {})
    if action.kind == "ee":
        p_base, r_base = world_to_base(action.position_m, ee_rotation_matrix(action),
                                       state["base_world_position_m"], state["base_world_quat_xyzw"])
        plan = plan_pose_segment(q, p_base, r_base)
        path_type = "straight_line"
        if plan["status"] != "kinematically_reachable":
            # Ordinary fallback: direct IK to the final pose with joint-space
            # interpolation when the straight Cartesian line has no local solution
            # (typical when leaving the straight-arm home posture).
            direct = solve_pose_multistart(q, p_base, r_base)
            if direct["status"] == "kinematically_reachable":
                # The child applies the same bounded joint stepping as a joint
                # action and returns partial when this slot cannot finish it.
                plan = {"status": "kinematically_reachable", "waypoints": [direct["q"]],
                        "segment_length_m": plan["segment_length_m"], "segment_angle_rad": plan["segment_angle_rad"]}
                path_type = "joint_space_interpolation"
            else:
                failed = plan["failed_result"]
                return Receipt("unreachable", action, 0, {
                    "reason": ("joint-space discontinuity along the straight path (IK branch change)" if plan.get("reason") == "ik_discontinuity"
                               else "IK did not resolve this pose within the configured joint limits"),
                    "failed_fraction": round(plan["failed_fraction"], 3),
                    "residual_position_m": round(failed["position_error_m"], 4),
                    "residual_orientation_rad": round(failed["orientation_error_rad"], 4),
                    "direct_ik": {"status": direct["status"], "residual_position_m": round(direct["position_error_m"], 4),
                                  "max_joint_delta_rad": round(direct["max_joint_delta_rad"], 3)},
                    "target_base_m": [round(float(v), 4) for v in p_base],
                    "target_distance_from_shoulder_m": round(float(np.linalg.norm(np.asarray(p_base) - np.array([0.0, 0.0, 0.333]))), 3),
                    "target_horizontal_distance_from_base_m": round(float(np.linalg.norm(np.asarray(p_base)[:2])), 3),
                    "segment_length_m": round(plan["segment_length_m"], 4)})
        command = {"kind": "slot", "waypoints": plan["waypoints"], "g": gripper, "steps": slot_steps}
        detail = {"ik_waypoints": len(plan["waypoints"]), "segment_length_m": round(plan["segment_length_m"], 4),
                  "segment_angle_rad": round(plan["segment_angle_rad"], 4), "path": path_type}
    elif action.kind == "joint":
        command = {"kind": "slot", "waypoints": [list(action.q_rad)], "g": gripper, "steps": slot_steps}
        detail = {"max_joint_delta_rad": round(max(abs(a - b) for a, b in zip(action.q_rad, q, strict=True)), 4)}
    elif action.kind == "hold":
        command = {"kind": "slot", "waypoints": [q], "g": gripper, "steps": slot_steps}
        detail = {}
    elif action.kind == "base":
        command = {"kind": "base", "a": action.axis, "v": action.velocity, "g": gripper,
                   "steps": slot_steps, "motion_steps": min(base_motion_steps, slot_steps)}
        detail = {}
    else:
        return Receipt("invalid", action, 0, {"reason": f"unsupported kind {action.kind}"})
    observation = sim.send(command)
    child = observation["execution"]
    steps = int(child["steps_executed"])
    if child.get("budget_truncated"):
        status = "budget_exhausted"
    elif action.kind == "ee" or action.kind == "joint":
        status = "completed" if child["final_joint_error_rad"] < 0.03 else "partial"
        if status == "partial":
            if child.get("waypoints_commanded") == child.get("waypoints_total") and child.get("peak_force_n", 0) > 15:
                detail["reason"] = (f"motion blocked by contact: the arm pressed on something and stalled short of the target "
                                    f"(peak contact force {child['peak_force_n']:.0f} N); the robot is at tcp_world_after_m; move away from the obstacle before retrying")
            else:
                detail["reason"] = "slot ended before the target was reached (the path was longer than one slot); the robot is at tcp_world_after_m"
    else:
        status = "completed"
    return Receipt(status, action, steps, detail, child)


READY_Q = (0.0, -0.35, 0.0, -2.0, 0.0, 1.65, 0.785)


def move_to_ready(sim: Simulator, max_slots: int = 3) -> Receipt:
    """Task-agnostic pre-episode move out of the straight-arm home singularity.
    The elbow travels about 1.9 rad, more than one 20-step slot allows, so up to
    ``max_slots`` slots are used; the returned receipt is the last one."""
    receipt = None
    for _ in range(max_slots):
        receipt = execute(sim, Action("joint", q_rad=READY_Q, gripper=1, note="ready pose"), slot_steps=SLOT_STEPS)
        if receipt.child is None or receipt.child["final_joint_error_rad"] < 0.05:
            break
    return receipt
