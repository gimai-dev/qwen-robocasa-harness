"""Network-isolated collision-planner launcher and closed result validation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


def _validated_plan(
    value: object,
    *,
    task: str,
    seed: int,
    target_world_m: np.ndarray,
    base_world_m: np.ndarray,
    orientation_world_xyzw: np.ndarray,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("collision planner result is not an object")
    if value.get("schema") != "robocasa-inspect-robot-ik-diagnostic/v1":
        raise RuntimeError("collision planner schema mismatch")
    if value.get("task") != task or value.get("seed") != seed:
        raise RuntimeError("collision planner episode mismatch")
    for name, expected in (
        ("target_world_m", target_world_m),
        ("requested_base_world_m", base_world_m),
        ("requested_orientation_world_xyzw", orientation_world_xyzw),
    ):
        actual = np.asarray(value.get(name), dtype=np.float64)
        expected_array = np.asarray(expected, dtype=np.float64)
        if (
            actual.shape != expected_array.shape
            or not np.isfinite(actual).all()
            or not np.allclose(actual, expected_array, atol=1e-9, rtol=0)
        ):
            raise RuntimeError(f"collision planner {name} mismatch")
    if value.get("success_was_queried") is not False:
        raise RuntimeError("collision planner queried task success")
    if value.get("collision_geometry_used_for_safety") is not True:
        raise RuntimeError("collision planner safety authority is missing")
    if value.get("task_state_was_read") is not False:
        raise RuntimeError("collision planner read task state")
    if value.get("collision_free") is not True or value.get(
        "planned_joint_path_clear"
    ) is not True:
        raise RuntimeError("collision planner found no safe path")
    path = value.get("cartesian_path")
    if not isinstance(path, list) or not 2 <= len(path) <= 256:
        raise RuntimeError("collision planner Cartesian path is invalid")
    for index, waypoint in enumerate(path):
        if not isinstance(waypoint, dict) or set(waypoint) != {
            "path_index",
            "eef_position_world_m",
            "eef_orientation_world_xyzw",
        }:
            raise RuntimeError("collision planner waypoint schema mismatch")
        if waypoint["path_index"] != index:
            raise RuntimeError("collision planner waypoint ordering mismatch")
        position = np.asarray(waypoint["eef_position_world_m"], dtype=np.float64)
        quaternion = np.asarray(
            waypoint["eef_orientation_world_xyzw"], dtype=np.float64
        )
        if (
            position.shape != (3,)
            or quaternion.shape != (4,)
            or not np.isfinite(position).all()
            or not np.isfinite(quaternion).all()
            or not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-8, rtol=0)
        ):
            raise RuntimeError("collision planner waypoint numerics are invalid")
    return value


def plan_collision_free_cartesian_path(
    *,
    task: str,
    seed: int,
    target_world_m: object,
    base_world_m: object,
    orientation_world_xyzw: object,
    run: Path,
) -> dict[str, object]:
    target = np.asarray(target_world_m, dtype=np.float64)
    base = np.asarray(base_world_m, dtype=np.float64)
    orientation = np.asarray(orientation_world_xyzw, dtype=np.float64)
    if target.shape != (3,) or base.shape != (3,) or orientation.shape != (4,):
        raise ValueError("collision planner target/base shape is invalid")
    root = Path("/home/jli/work/robocasa-inspect-official")
    state = Path("/home/jli/state/robocasa-inspect-official")
    driver = state / "nvidia-egl-580.159.04/rootfs"
    planner_run = run / "collision-plan"
    xdg = run / "collision-plan-xdg"
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
        f"ROBOCASA_RUN_DIR={planner_run}",
        f"ROBOCASA_CHECKOUT_ROOT={root / 'robocasa'}",
        f"ROBOCASA_CACHE_ROOT={state}",
        "MUJOCO_GL=egl",
        "PYOPENGL_PLATFORM=egl",
        "EGL_PLATFORM=surfaceless",
        f"XDG_RUNTIME_DIR={xdg}",
        f"LD_LIBRARY_PATH={driver / 'usr/lib/x86_64-linux-gnu'}:/usr/lib/x86_64-linux-gnu",
        f"__EGL_VENDOR_LIBRARY_FILENAMES={driver / 'usr/share/glvnd/egl_vendor.d/10_nvidia.json'}",
        str(root / ".venv/bin/python"),
        "-m",
        "robocasa_inspect.ik_diagnostic",
        "--task",
        task,
        "--seed",
        str(seed),
        "--target",
        *(str(float(item)) for item in target),
        "--base",
        *(str(float(item)) for item in base),
        "--orientation",
        *(str(float(item)) for item in orientation),
        "--run-dir",
        str(planner_run),
    ]
    log_path = run / "collision-planner.log"
    with log_path.open("w", encoding="utf-8") as log:
        log_path.chmod(0o600)
        subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=180,
            check=True,
        )
    result_path = planner_run / "result.json"
    return _validated_plan(
        json.loads(result_path.read_text()),
        task=task,
        seed=seed,
        target_world_m=target,
        base_world_m=base,
        orientation_world_xyzw=orientation,
    )
