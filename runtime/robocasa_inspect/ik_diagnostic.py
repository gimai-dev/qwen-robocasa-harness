"""Offline robot-only position feasibility diagnostic for reviewed RGB targets."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import gymnasium as gym
import numpy as np

import robocasa  # noqa: F401


def _validated_world_point(value: object, *, label: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"{label} must be a finite world XYZ point")
    return point


def _matrix_to_quaternion_xyzw(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation matrix is invalid")
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.asarray(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        diagonal = int(np.argmax(np.diag(matrix)))
        if diagonal == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quaternion = np.asarray(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif diagonal == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quaternion = np.asarray(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quaternion = np.asarray(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    if quaternion[3] < 0:
        quaternion *= -1
    return quaternion / np.linalg.norm(quaternion)


def _quaternion_xyzw_to_matrix(value: object) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("orientation quaternion is invalid")
    quaternion /= np.linalg.norm(quaternion)
    x, y, z, w = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_error_world(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    delta = target @ current.T
    cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    vector = np.asarray(
        [
            delta[2, 1] - delta[1, 2],
            delta[0, 2] - delta[2, 0],
            delta[1, 0] - delta[0, 1],
        ]
    )
    sine = float(np.linalg.norm(vector) / 2.0)
    if angle <= 1e-10:
        return 0.5 * vector
    if sine <= 1e-8:
        eigenvalues, eigenvectors = np.linalg.eig(delta)
        axis = np.real(eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))])
        axis /= np.linalg.norm(axis)
        return angle * axis
    return angle * vector / (2.0 * sine)


def _gripper_axis_support_m(simulation: object, site_id: int) -> dict[str, float]:
    """Conservatively bound official gripper collisions along every signed site axis."""
    site_position = np.asarray(simulation.data.site_xpos[site_id], dtype=np.float64)
    site_xmat = np.asarray(
        simulation.data.site_xmat[site_id], dtype=np.float64
    ).reshape(3, 3)
    directions = {
        f"{axis}_{sign_name}": float(sign) * site_xmat[:, axis_index]
        for axis, axis_index in (("x", 0), ("y", 1), ("z", 2))
        for sign_name, sign in (("plus", 1.0), ("minus", -1.0))
    }
    supports: dict[str, list[float]] = {name: [] for name in directions}
    for geom_id in range(simulation.model.ngeom):
        name = simulation.model.geom_id2name(geom_id) or ""
        if "gripper0" not in name or simulation.model.geom_contype[geom_id] == 0:
            continue
        center = np.asarray(simulation.data.geom_xpos[geom_id], dtype=np.float64)
        radius = float(simulation.model.geom_rbound[geom_id])
        for direction_name, direction in directions.items():
            supports[direction_name].append(
                float(np.dot(center - site_position, direction)) + radius
            )
    if not all(supports.values()):
        raise RuntimeError("official gripper collision geometry is missing")
    result = {name: max(values) for name, values in supports.items()}
    if not all(0.01 <= support <= 0.40 for support in result.values()):
        raise RuntimeError(f"official gripper axis support is implausible: {result}")
    return result


def _rrt_joint_path(
    start: np.ndarray,
    goal: np.ndarray,
    ranges: np.ndarray,
    *,
    edge_is_clear: Callable[[np.ndarray, np.ndarray], bool],
    seed: int = 3074294,
    max_nodes: int = 2_000,
) -> list[np.ndarray] | None:
    """Plan one deterministic bounded joint path without task geometry access."""
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    ranges = np.asarray(ranges, dtype=np.float64)
    if start.shape != goal.shape or ranges.shape != (len(start), 2):
        raise ValueError("RRT joint dimensions are invalid")
    if max_nodes <= 0:
        raise ValueError("RRT node budget must be positive")
    if edge_is_clear(start, goal):
        return [start.copy(), goal.copy()]
    generator = np.random.default_rng(seed)
    spans = np.maximum(ranges[:, 1] - ranges[:, 0], 1e-6)
    nodes = [start.copy()]
    parents = [-1]
    for iteration in range(max_nodes):
        sample = (
            goal
            if iteration % 5 == 0
            else generator.uniform(ranges[:, 0], ranges[:, 1])
        )
        distances = [np.linalg.norm((node - sample) / spans) for node in nodes]
        nearest_index = int(np.argmin(distances))
        difference = sample - nodes[nearest_index]
        scale = min(1.0, 0.12 / max(float(np.max(np.abs(difference))), 1e-12))
        candidate = nodes[nearest_index] + scale * difference
        if not edge_is_clear(nodes[nearest_index], candidate):
            continue
        nodes.append(candidate)
        parents.append(nearest_index)
        candidate_index = len(nodes) - 1
        if not edge_is_clear(candidate, goal):
            continue
        path = [goal.copy(), candidate.copy()]
        parent = parents[candidate_index]
        while parent >= 0:
            path.append(nodes[parent].copy())
            parent = parents[parent]
        path.reverse()
        return path
    return None


def solve(
    task: str,
    seed: int,
    target_world_m: np.ndarray,
    *,
    base_world_m: np.ndarray | None = None,
    orientation_world_xyzw: np.ndarray | None = None,
) -> dict[str, object]:
    from robocasa.utils import env_utils

    target_world_m = _validated_world_point(target_world_m, label="target")
    if base_world_m is not None:
        base_world_m = _validated_world_point(base_world_m, label="base")
    target_orientation = (
        None
        if orientation_world_xyzw is None
        else _quaternion_xyzw_to_matrix(orientation_world_xyzw)
    )
    environment = gym.make(f"robocasa/{task}", split="pretrain", seed=seed)
    try:
        environment.reset()
        kitchen = environment.unwrapped
        simulation = kitchen.sim
        robot = kitchen.robots[0]
        base_site_id = simulation.model.site_name2id(
            robot.robot_model.base.correct_naming("center")
        )
        initial_base_world_m = np.asarray(
            simulation.data.site_xpos[base_site_id], dtype=np.float64
        ).copy()
        if base_world_m is not None:
            env_utils.set_robot_to_position(kitchen, base_world_m)
            simulation.forward()
        realized_base_world_m = np.asarray(
            simulation.data.site_xpos[base_site_id], dtype=np.float64
        ).copy()
        expected_base_world_m = (
            initial_base_world_m if base_world_m is None else base_world_m
        )
        if not np.allclose(
            realized_base_world_m[:2], expected_base_world_m[:2], atol=1e-6, rtol=0
        ):
            raise RuntimeError("official mobile-base placement did not reach target XY")
        site_id = robot.eef_site_id["right"]
        site_name = simulation.model.site_id2name(site_id)
        eef_body_id = simulation.model.body_name2id(robot.robot_model.eef_name["right"])
        qpos_indices = np.asarray(robot._ref_arm_joint_pos_indexes, dtype=int)
        qvel_indices = np.asarray(robot._ref_arm_joint_vel_indexes, dtype=int)
        joint_indices = np.asarray(robot._ref_arm_joint_indexes, dtype=int)
        ranges = np.asarray(simulation.model.jnt_range[joint_indices], dtype=np.float64)
        initial = np.asarray(simulation.data.qpos[qpos_indices], dtype=np.float64).copy()
        generator = np.random.default_rng(3074294)
        best: dict[str, object] | None = None
        best_any: dict[str, object] | None = None

        def robot_environment_contacts() -> list[dict[str, object]]:
            contacts: list[dict[str, object]] = []
            for contact in simulation.data.contact:
                first = simulation.model.geom_id2name(contact.geom1) or ""
                second = simulation.model.geom_id2name(contact.geom2) or ""
                first_robot = "robot0" in first or "gripper0" in first
                second_robot = "robot0" in second or "gripper0" in second
                if first_robot != second_robot and float(contact.dist) < 0.0:
                    contacts.append(
                        {
                            "first": first,
                            "second": second,
                            "distance_m": float(contact.dist),
                        }
                    )
            return contacts

        for restart in range(33):
            qpos = initial.copy()
            if restart:
                qpos += generator.normal(0.0, 0.35, size=len(qpos))
                qpos = np.clip(qpos, ranges[:, 0] + 1e-4, ranges[:, 1] - 1e-4)
            simulation.data.qpos[qpos_indices] = qpos
            simulation.forward()
            for iteration in range(600):
                position_error = target_world_m - np.asarray(
                    simulation.data.site_xpos[site_id], dtype=np.float64
                )
                current_orientation = np.asarray(
                    simulation.data.site_xmat[site_id], dtype=np.float64
                ).reshape(3, 3)
                orientation_error = (
                    np.zeros(0)
                    if target_orientation is None
                    else _rotation_error_world(target_orientation, current_orientation)
                )
                error = np.concatenate((position_error, orientation_error))
                if np.linalg.norm(position_error) <= 1e-5 and (
                    target_orientation is None
                    or np.linalg.norm(orientation_error) <= 1e-4
                ):
                    break
                position_jacobian = np.asarray(
                    simulation.data.get_site_jacp(site_name), dtype=np.float64
                ).reshape(3, -1)[:, qvel_indices]
                jacobian = position_jacobian
                if target_orientation is not None:
                    rotation_jacobian = np.asarray(
                        simulation.data.get_site_jacr(site_name), dtype=np.float64
                    ).reshape(3, -1)[:, qvel_indices]
                    jacobian = np.vstack((position_jacobian, rotation_jacobian))
                delta = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + 1e-4 * np.eye(len(error)), error
                )
                norm = float(np.linalg.norm(delta))
                if norm > 0.05:
                    delta *= 0.05 / norm
                qpos = np.clip(
                    qpos + delta, ranges[:, 0] + 1e-4, ranges[:, 1] - 1e-4
                )
                simulation.data.qpos[qpos_indices] = qpos
                simulation.forward()
            position = np.asarray(simulation.data.site_xpos[site_id], dtype=np.float64)
            error_m = float(np.linalg.norm(target_world_m - position))
            current_orientation = np.asarray(
                simulation.data.site_xmat[site_id], dtype=np.float64
            ).reshape(3, 3)
            orientation_error_rad = (
                0.0
                if target_orientation is None
                else float(
                    np.linalg.norm(
                        _rotation_error_world(target_orientation, current_orientation)
                    )
                )
            )
            candidate = {
                "restart": restart,
                "iterations": iteration + 1,
                "error_m": error_m,
                "orientation_error_rad": orientation_error_rad,
                "joint_qpos": qpos.tolist(),
                "eef_position_world_m": position.tolist(),
                "eef_xmat_world": np.asarray(
                    simulation.data.site_xmat[site_id], dtype=np.float64
                )
                .reshape(3, 3)
                .tolist(),
                "robot_environment_contacts": robot_environment_contacts(),
            }
            objective = error_m + orientation_error_rad
            best_any_objective = (
                float("inf")
                if best_any is None
                else float(best_any["error_m"])
                + float(best_any["orientation_error_rad"])
            )
            if best_any is None or objective < best_any_objective:
                best_any = candidate
            if (
                not candidate["robot_environment_contacts"]
                and error_m <= 1e-4
                and orientation_error_rad <= 1e-3
                and (
                    best is None
                    or objective
                    < float(best["error_m"])
                    + float(best["orientation_error_rad"])
                )
            ):
                best = candidate
        if best_any is None:
            raise AssertionError("IK search produced no candidate")
        cartesian_path: list[dict[str, object]] = []
        path_collision: dict[str, object] | None = None
        planned_joint_path: list[np.ndarray] | None = None
        if best is not None:
            target_qpos = np.asarray(best["joint_qpos"], dtype=np.float64)

            def edge_is_clear(first: np.ndarray, second: np.ndarray) -> bool:
                count = max(
                    1,
                    int(
                        np.ceil(float(np.max(np.abs(second - first))) / 0.025)
                    ),
                )
                for alpha in np.linspace(0.0, 1.0, count + 1)[1:]:
                    simulation.data.qpos[qpos_indices] = (
                        (1.0 - alpha) * first + alpha * second
                    )
                    simulation.forward()
                    if robot_environment_contacts():
                        return False
                return True

            if edge_is_clear(initial, target_qpos):
                planned_joint_path = [initial.copy(), target_qpos.copy()]
            else:
                path_collision = {
                    "kind": "direct_joint_path_prohibited_collision"
                }
                planned_joint_path = _rrt_joint_path(
                    initial,
                    target_qpos,
                    ranges,
                    edge_is_clear=edge_is_clear,
                )
            if planned_joint_path is not None:
                path_collision = None
                dense_path: list[np.ndarray] = [planned_joint_path[0]]
                for first, second in pairwise(planned_joint_path):
                    count = max(
                        1,
                        int(
                            np.ceil(
                                float(np.max(np.abs(second - first))) / 0.05
                            )
                        ),
                    )
                    dense_path.extend(
                        (1.0 - alpha) * first + alpha * second
                        for alpha in np.linspace(0.0, 1.0, count + 1)[1:]
                    )
                for path_index, path_qpos in enumerate(dense_path):
                    simulation.data.qpos[qpos_indices] = path_qpos
                    simulation.forward()
                    cartesian_path.append(
                        {
                            "path_index": path_index,
                            "eef_position_world_m": np.asarray(
                                simulation.data.site_xpos[site_id], dtype=np.float64
                            ).tolist(),
                            "eef_orientation_world_xyzw": _matrix_to_quaternion_xyzw(
                                np.asarray(
                                    simulation.data.body_xmat[eef_body_id],
                                    dtype=np.float64,
                                ).reshape(3, 3)
                            ).tolist(),
                        }
                    )
        return {
            "schema": "robocasa-inspect-robot-ik-diagnostic/v1",
            "task": task,
            "seed": seed,
            "target_world_m": target_world_m.tolist(),
            "requested_orientation_world_xyzw": (
                None
                if orientation_world_xyzw is None
                else np.asarray(orientation_world_xyzw, dtype=np.float64).tolist()
            ),
            "initial_base_world_m": initial_base_world_m.tolist(),
            "requested_base_world_m": (
                None if base_world_m is None else base_world_m.tolist()
            ),
            "realized_base_world_m": realized_base_world_m.tolist(),
            "best": best if best is not None else best_any,
            "collision_free": best is not None,
            "planned_joint_path_clear": planned_joint_path is not None,
            "planned_joint_path_nodes": (
                0 if planned_joint_path is None else len(planned_joint_path)
            ),
            "joint_linear_path_collision": path_collision,
            "cartesian_path": cartesian_path,
            "gripper_axis_support_m": _gripper_axis_support_m(
                simulation, site_id
            ),
            "success_was_queried": False,
            "collision_geometry_used_for_safety": True,
            "task_state_was_read": False,
        }
    finally:
        environment.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="CloseDrawer")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target", type=float, nargs=3, required=True)
    parser.add_argument("--base", type=float, nargs=3)
    parser.add_argument("--orientation", type=float, nargs=4)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    run.chmod(0o700)
    episode = run / "episode-tmp"
    episode.mkdir(mode=0o700)
    os.environ["ROBOCASA_RUN_DIR"] = str(run)
    os.environ["ROBOCASA_EPISODE_TMPDIR"] = str(episode)
    result = solve(
        args.task,
        args.seed,
        np.asarray(args.target, dtype=np.float64),
        base_world_m=(
            None if args.base is None else np.asarray(args.base, dtype=np.float64)
        ),
        orientation_world_xyzw=(
            None
            if args.orientation is None
            else np.asarray(args.orientation, dtype=np.float64)
        ),
    )
    target = run / "result.json"
    target.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    target.chmod(0o600)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
