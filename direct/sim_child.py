"""Network-isolated RoboCasa child executing fixed-length numerical control slots.

Runs inside the existing sudo/unshare/setpriv sandbox. The driver writes
``mailbox/command-NNNNNN.json``; the child answers with
``mailbox/observation-NNNNNN.json`` carrying public images, public state,
camera calibration and the execution receipt of the previous command.

Commands (all carry ``schema``, ``sequence``, ``kind``; motion commands also
carry ``observation_id`` of the observation they answer):
  slot      track a joint waypoint path for exactly ``steps`` simulator steps
            (hold at the final waypoint when reached early) with gripper ``g``
  base      hold the arm, drive one base axis for ``motion_steps`` then brake,
            ``steps`` total
  snapshot  save the full simulator state (H8 recovery bank)
  finish    evaluate the official task predicate and seal the outcome
  close     exit without evaluation (infrastructure)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import socket
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from .chassis_hold import ChassisHold
from .kinematics import JOINT_LIMITS, MAX_JOINT_STEP, panda_fk, matrix_to_quat_xyzw

DEFAULT_ACTION_BUDGET = 900
DEFAULT_WALL_BUDGET_S = 1200.0
MAX_ACTION_BUDGET = 30000   # agent-as-policy sessions run thousands of slot steps
TRACKING_LAG_PAUSE = 0.10
VIDEO_EVERY = 4
BASE_VELOCITY_LIMIT = 0.5
SCHEMA = "qwen-direct-observation/v1"


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), default=_default) + "\n")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _default(value: object) -> object:
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"unserializable {type(value)!r}")


def _network_denied() -> str:
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=0.2):
            pass
    except OSError as error:
        return type(error).__name__
    raise RuntimeError("simulator network namespace is not isolated")


# --- controller installation (verbatim behaviour from the released harness) ---

def joint_controller_config(config: Mapping[str, object]) -> dict[str, object]:
    """Use absolute arm joints and hold the kitchen's zero torso extension."""
    output = copy.deepcopy(dict(config))
    body_parts = output["body_parts"]
    old_right = body_parts["right"]
    gripper = copy.deepcopy(old_right["gripper"])
    lower = [limits[0] for limits in JOINT_LIMITS]
    upper = [limits[1] for limits in JOINT_LIMITS]
    body_parts["right"] = {
        "type": "JOINT_POSITION", "input_min": lower, "input_max": upper,
        "output_min": lower, "output_max": upper, "kp": 150, "damping_ratio": 1,
        "impedance_mode": "fixed", "kp_limits": [0, 300], "damping_ratio_limits": [0, 10],
        "qpos_limits": [lower, upper], "interpolation": None, "ramp_ratio": 0.2,
        "input_type": "absolute", "gripper": gripper,
    }
    body_parts["torso"]["input_type"] = "absolute"
    return output


def joint_unmap_action(action: Mapping[str, object]) -> dict[str, object]:
    # RoboSuite's PandaGripper declares drive +1 as closed and -1 as open.
    gripper_drive = 1.0 - 2.0 * float(action["gripper_open"])
    base_motion = list(action["base_motion"])
    base_mode = 1.0 if any(abs(item) > 1e-12 for item in base_motion) else -1.0
    return {
        "robot0_right": list(action["joint_position"]),
        "robot0_right_gripper": gripper_drive,
        "robot0_base": base_motion,
        "robot0_torso": [0.0],
        "robot0_base_mode": base_mode,
    }


def _install_joint_controller() -> None:
    import numpy as np
    from robocasa.utils import env_utils
    from robocasa.wrappers.gym_wrapper import PandaOmronKeyConverter

    original_loader = env_utils.load_composite_controller_config
    joint_config = joint_controller_config(original_loader(controller=None, robot="PandaOmron"))

    def load_config(controller: object = None, robot: object = None) -> object:
        if controller is None and robot == "PandaOmron":
            return copy.deepcopy(joint_config)
        return original_loader(controller=controller, robot=robot)

    def unmap(_cls: object, input_action: Mapping[str, object]) -> dict[str, object]:
        candidate = {
            "joint_position": np.asarray(input_action["action.joint_position"], dtype=np.float64).reshape(-1).tolist(),
            "gripper_open": float(np.asarray(input_action["action.gripper_open"]).reshape(-1)[0]),
            "base_motion": np.asarray(input_action["action.base_motion"], dtype=np.float64).reshape(-1).tolist(),
            "torso": 0.0,
        }
        return {key: np.asarray(value, dtype=np.float64) for key, value in joint_unmap_action(candidate).items()}

    env_utils.load_composite_controller_config = load_config
    PandaOmronKeyConverter.unmap_action = classmethod(unmap)


# --- scene pinning ---

def reset_pinned_scene(environment: object, task: str, seed: int, scenes: Path, run: Path) -> Mapping[str, object]:
    """Reset to the saved initial scene for (task, seed), creating it on first use."""
    import numpy as np
    env = environment.unwrapped
    scene = scenes / f"{task}-seed{seed}.json"
    scenes.mkdir(parents=True, exist_ok=True)
    environment.reset()
    if not scene.exists():
        snapshot = {"model": env.sim.model.get_xml(),
                    "states": env.sim.get_state().flatten().tolist(),
                    "ep_meta": env.get_ep_meta()}
        scene.write_text(json.dumps(snapshot, default=lambda x: x.tolist()))
    snapshot = json.loads(scene.read_text())
    env.set_ep_meta(snapshot["ep_meta"])
    env.reset()
    env.reset_from_xml_string(env.edit_model_xml(snapshot["model"]))
    env.sim.reset()
    env.sim.set_state_from_flattened(np.asarray(snapshot["states"]))
    env.sim.forward()
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()
    actual = env.sim.get_state().flatten()
    if not np.array_equal(actual, np.asarray(snapshot["states"])):
        raise RuntimeError("pinned initial simulator state did not restore exactly")
    meta = env.get_ep_meta()
    _atomic_json(run / "scene.json", {"scene_path": str(scene), "task": task, "seed": seed,
                                      "state_restored_exactly": True, "ep_meta": meta})
    return env.get_observation(env._get_observations(force_update=True))


# --- state readers ---

def _arm_qpos(environment: object) -> list[float]:
    robot = environment.unwrapped.robots[0]
    all_q = list(robot.get_robot_joint_positions())
    names = list(robot.robot_joints)
    return [float(all_q[names.index(name)]) for name in robot.robot_arm_joints]


def _arm_qvel(environment: object) -> list[float]:
    env = environment.unwrapped
    robot = env.robots[0]
    obs = env._get_observations(force_update=False)
    all_v = list(obs["robot0_joint_vel"])
    names = list(robot.robot_joints)
    return [float(all_v[names.index(name)]) for name in robot.robot_arm_joints]


def _wrench(environment: object) -> dict[str, list[float]]:
    robot = environment.unwrapped.robots[0]
    return {"force_n": [float(v) for v in robot.ee_force["right"]],
            "torque_nm": [float(v) for v in robot.ee_torque["right"]]}


def _unloaded_wrist_weight(environment: object) -> float:
    """Gravity load of the robot bodies distal to its wrist force sensor."""
    env = environment.unwrapped
    model = env.sim.model
    sensor = model.sensor_name2id(env.robots[0].gripper["right"].important_sensors["force_ee"])
    body = model.site_bodyid[model.sensor_objid[sensor]]
    return float(model.body_subtreemass[body]) * math.sqrt(sum(float(g) ** 2 for g in model.opt.gravity))


def _wrist_load(force: Sequence[float], unloaded_weight: float) -> float:
    return max(0.0, math.sqrt(sum(float(v) ** 2 for v in force)) - unloaded_weight)


def _public_state(raw: Mapping[str, object], environment: object) -> dict[str, object]:
    import numpy as np
    from robocasa_inspect.camera_geometry import official_camera_calibration, project_world_point
    from .kinematics import base_to_world, quat_xyzw_to_matrix

    q = _arm_qpos(environment)
    base_p = [float(v) for v in np.asarray(raw["state.base_position"]).reshape(-1)]
    base_o = [float(v) for v in np.asarray(raw["state.base_rotation"]).reshape(-1)]
    rel_p = [float(v) for v in np.asarray(raw["state.end_effector_position_relative"]).reshape(-1)]
    rel_o = [float(v) for v in np.asarray(raw["state.end_effector_rotation_relative"]).reshape(-1)]
    finger = [float(v) for v in np.asarray(raw["state.gripper_qpos"]).reshape(-1)]
    fk_p, fk_r = panda_fk(q)
    # Policy-facing orientation is the FK grip_site frame (+z approach, fingers
    # close along local x). The public eef quaternion is that frame rotated 90
    # degrees about z; it is kept only as a reference field.
    tcp_w, tcp_r = base_to_world(rel_p, fk_r, base_p, base_o)
    calibration = official_camera_calibration(environment)
    pixels: dict[str, object] = {}
    for label in ("left", "right", "wrist"):
        try:
            u, v, depth = project_world_point(calibration[label], tcp_w)
            pixels[label] = {"u": round(u, 1), "v": round(v, 1), "visible": bool(0 <= u < 256 and 0 <= v < 256)}
        except ValueError:
            pixels[label] = {"u": None, "v": None, "visible": False}
    x, y, z, w = base_o
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return {
        "arm_q_rad": q,
        "arm_qvel_rad_s": _arm_qvel(environment),
        "tcp_world_position_m": [float(v) for v in tcp_w],
        "tcp_world_quat_xyzw": matrix_to_quat_xyzw(tcp_r),
        "tcp_base_position_m": rel_p,
        "tcp_base_quat_xyzw": matrix_to_quat_xyzw(fk_r),
        "tcp_base_quat_xyzw_public_eef": rel_o,
        "fk_base_position_m": [float(v) for v in fk_p],
        "fk_position_error_m": float(np.linalg.norm(fk_p - np.asarray(rel_p))),
        "base_world_position_m": base_p,
        "base_world_quat_xyzw": base_o,
        "base_world_yaw_rad": yaw,
        "gripper_finger_qpos": finger,
        "gripper_width_m": float(finger[0] - finger[1]) if len(finger) == 2 else None,
        "wrench": _wrench(environment),
        "tcp_pixels": pixels,
        "camera_calibration": calibration,
    }


def _save_images(raw: Mapping[str, object], frame_dir: Path) -> dict[str, dict[str, str]]:
    import numpy as np
    from PIL import Image
    from robocasa_inspect.contracts import CAMERAS
    frame_dir.mkdir(parents=True, mode=0o700)
    images = {}
    for label, key in zip(("left", "right", "wrist"), CAMERAS, strict=True):
        target = frame_dir / f"{label}.png"
        Image.fromarray(np.asarray(raw[key], dtype=np.uint8)).save(target, format="PNG")
        target.chmod(0o600)
        images[label] = {"path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    return images


def _save_video_frame(raw: Mapping[str, object], video_dir: Path, index: int, caption: str) -> None:
    import numpy as np
    from PIL import Image, ImageDraw
    from robocasa_inspect.contracts import CAMERAS
    mosaic = Image.new("RGB", (768, 256))
    for camera_index, key in enumerate(CAMERAS):
        mosaic.paste(Image.fromarray(np.asarray(raw[key], dtype=np.uint8)), (camera_index * 256, 0))
    ImageDraw.Draw(mosaic).text((4, 4), caption, fill=(255, 255, 0))
    mosaic.save(video_dir / f"{index:06d}.png", format="PNG")


def _evaluator_view(environment: object) -> dict[str, object]:
    """Ground truth the agent never sees: target object pose, gripper-object distance, contact, success."""
    import numpy as np
    env = environment.unwrapped
    out: dict[str, object] = {}
    try:
        out["official_success"] = bool(env._check_success())
    except Exception as error:  # noqa: BLE001
        out["official_success_error"] = repr(error)
    try:
        obj_id = env.obj_body_id["obj"]
        obj = np.asarray(env.sim.data.body_xpos[obj_id], dtype=float)
        site = np.asarray(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]], dtype=float)
        out.update({"obj_world_m": obj.tolist(), "gripper_obj_distance_m": float(np.linalg.norm(site - obj)),
                    "gripper_touching_obj": bool(env.check_contact(env.robots[0].gripper["right"], env.objects["obj"]))})
    except Exception as error:  # noqa: BLE001
        out["object_error"] = repr(error)
    return out


class Child:
    def __init__(self, task: str, seed: int, run: Path, scenes: Path, *, action_budget: int, wall_budget_s: float) -> None:
        self.task, self.seed, self.run, self.scenes = task, seed, run, scenes
        self.action_budget, self.wall_budget_s = action_budget, wall_budget_s
        self.total_steps = 0
        self.gripper_open = 1.0
        self.started = time.monotonic()
        self.video_index = 0
        self.decision = 0
        self.trace: list[dict[str, object]] = []
        self.unloaded_wrist_weight: float | None = None

    # -- stepping --
    def step(self, environment: object, joint_position: Sequence[float], gripper_open: float,
             base_motion: Sequence[float]) -> Mapping[str, object]:
        import numpy as np
        payload = {
            "action.joint_position": np.asarray(joint_position, dtype=np.float64),
            "action.gripper_open": np.asarray([gripper_open], dtype=np.float64),
            "action.base_motion": np.asarray(base_motion, dtype=np.float64),
            "action.torso": np.asarray([0.0], dtype=np.float64),
        }
        raw, _, _, _, _ = environment.step(payload)
        self.total_steps += 1
        if self.total_steps % VIDEO_EVERY == 0:
            _save_video_frame(raw, self.run / "video-frames", self.video_index,
                              f"step {self.total_steps} decision {self.decision}")
            self.video_index += 1
        return raw

    def budget_left(self) -> bool:
        return self.total_steps < self.action_budget and (time.monotonic() - self.started) < self.wall_budget_s

    def execute_slot(self, environment: object, command: Mapping[str, object], chassis: ChassisHold,
                     public_raw: Mapping[str, object]) -> tuple[Mapping[str, object], dict[str, object]]:
        import numpy as np
        path = [[float(v) for v in wp] for wp in command["waypoints"]]
        if not path or any(len(wp) != 7 for wp in path):
            raise ValueError("slot needs a non-empty seven-joint waypoint path")
        steps = int(command["steps"])
        if not 1 <= steps <= 64:
            raise ValueError("slot steps must be within 1..64")
        gripper_open = float(command["g"])
        if gripper_open not in (0.0, 1.0):
            raise ValueError("gripper must be 0 or 1")
        initial = _arm_qpos(environment)
        before = _public_state(public_raw, environment)
        commanded: list[float] | None = None
        path_index = 0
        pauses = 0
        peak_force = 0.0
        executed = 0
        raw = public_raw
        trace: list[dict[str, object]] = []
        for _ in range(steps):
            if not self.budget_left():
                break
            actual = _arm_qpos(environment)
            if commanded is None:
                commanded = list(actual)
            if max(abs(c - a) for c, a in zip(commanded, actual, strict=True)) >= TRACKING_LAG_PAUSE:
                waypoint = list(commanded)      # controller lags: hold the commanded target
                pauses += 1
            else:
                target = path[path_index]
                waypoint = [c + max(-MAX_JOINT_STEP, min(MAX_JOINT_STEP, d - c)) for c, d in zip(commanded, target, strict=True)]
                if path_index < len(path) - 1 and max(abs(w - t) for w, t in zip(waypoint, target, strict=True)) < 1e-9:
                    path_index += 1
            raw = self.step(environment, waypoint, gripper_open, chassis.correction(public_raw))
            public_raw = raw
            executed += 1
            commanded = waypoint
            force = _wrist_load(_wrench(environment)["force_n"], self.unloaded_wrist_weight)
            peak_force = max(peak_force, force)
            q_now = _arm_qpos(environment)
            trace.append({"commanded_q": [round(v, 4) for v in waypoint], "actual_q": [round(v, 4) for v in q_now],
                          "tcp_base_m": [round(float(v), 4) for v in panda_fk(q_now)[0]], "force_n": round(force, 2),
                          "path_index": path_index})
        self.gripper_open = gripper_open
        after = _public_state(raw, environment)
        realized = after["arm_q_rad"]
        final_target = path[-1]
        receipt = {
            "kind": "slot", "accepted": True, "steps_requested": steps, "steps_executed": executed,
            "budget_truncated": executed < steps,
            "gripper_command": gripper_open,
            "waypoints_total": len(path), "waypoints_commanded": path_index + 1,
            "trace": trace,
            "final_joint_error_rad": max(abs(a - b) for a, b in zip(realized, final_target, strict=True)),
            "tracking_pauses": pauses,
            "peak_force_n": peak_force,
            "arm_q_before": initial, "arm_q_after": realized,
            "tcp_world_before_m": before["tcp_world_position_m"], "tcp_world_after_m": after["tcp_world_position_m"],
            "tcp_world_quat_after_xyzw": after["tcp_world_quat_xyzw"],
            "gripper_width_before_m": before["gripper_width_m"], "gripper_width_after_m": after["gripper_width_m"],
            "base_world_before_m": before["base_world_position_m"], "base_world_after_m": after["base_world_position_m"],
            "chassis_hold_error_xy_m": chassis.errors(raw)[0],
        }
        return raw, receipt

    def execute_base(self, environment: object, command: Mapping[str, object], chassis: ChassisHold,
                     public_raw: Mapping[str, object]) -> tuple[Mapping[str, object], dict[str, object]]:
        import numpy as np
        from .chassis_hold import base_velocity_input, public_chassis_pose
        chassis_yaw = lambda state: public_chassis_pose(state)[2]
        axis = command["a"]
        velocity = float(command["v"])
        if axis not in ("x", "y", "yaw") or not 0.0 < abs(velocity) <= BASE_VELOCITY_LIMIT:
            raise ValueError("invalid base command")
        steps = int(command["steps"])
        motion_steps = int(command["motion_steps"])
        if not 1 <= motion_steps <= steps <= 64:
            raise ValueError("base motion steps must satisfy 1 <= motion <= steps <= 64")
        gripper_open = float(command["g"])
        hold_q = _arm_qpos(environment)
        before = _public_state(public_raw, environment)
        executed = 0
        raw = public_raw
        trace = []
        for index in range(steps):
            if not self.budget_left():
                break
            base = base_velocity_input(axis, velocity, chassis_yaw(raw), chassis.reset_yaw_rad) if index < motion_steps else [0.0, 0.0, 0.0]
            raw = self.step(environment, hold_q, gripper_open, base)
            executed += 1
            trace.append([round(float(v), 4) for v in np.asarray(raw["state.base_position"]).reshape(-1)] + [round(chassis_yaw(raw), 4)])
        self.gripper_open = gripper_open
        chassis.reset_after_base_action(raw)
        after = _public_state(raw, environment)
        receipt = {
            "kind": "base", "accepted": True, "axis": axis, "velocity": velocity,
            "steps_requested": steps, "steps_executed": executed, "motion_steps": motion_steps,
            "budget_truncated": executed < steps, "gripper_command": gripper_open, "trace": trace,
            "arm_q_before": hold_q, "arm_q_after": after["arm_q_rad"],
            "final_joint_error_rad": max(abs(a - b) for a, b in zip(after["arm_q_rad"], hold_q, strict=True)),
            "base_world_before_m": before["base_world_position_m"], "base_world_after_m": after["base_world_position_m"],
            "base_yaw_before_rad": before["base_world_yaw_rad"], "base_yaw_after_rad": after["base_world_yaw_rad"],
            "tcp_world_before_m": before["tcp_world_position_m"], "tcp_world_after_m": after["tcp_world_position_m"],
            "tcp_world_quat_after_xyzw": after["tcp_world_quat_xyzw"],
            "gripper_width_before_m": before["gripper_width_m"], "gripper_width_after_m": after["gripper_width_m"],
        }
        return raw, receipt

    def publish(self, raw: Mapping[str, object], environment: object, sequence: int,
                receipt: Mapping[str, object] | None) -> dict[str, object]:
        state = _public_state(raw, environment)
        if self.unloaded_wrist_weight is None:
            self.unloaded_wrist_weight = _unloaded_wrist_weight(environment)
        # Reset/restore sensor values can be transient. After physics advances,
        # compare against robot gravity load, never against that transient.
        state["contact_force_delta_n"] = (round(_wrist_load(state["wrench"]["force_n"], self.unloaded_wrist_weight), 2)
                                           if self.total_steps > 0 else None)
        state["unloaded_wrist_weight_n"] = self.unloaded_wrist_weight
        calibration = state.pop("camera_calibration")
        images = _save_images(raw, self.run / "frames" / f"{sequence:06d}")
        evaluator = _evaluator_view(environment)
        instruction = raw.get("annotation.human.task_description")
        record = {
            "schema": SCHEMA, "task": self.task, "seed": self.seed, "sequence": sequence,
            "observation_id": hashlib.sha256(json.dumps({"seq": sequence, "images": images, "state": state},
                                                        sort_keys=True, default=_default).encode()).hexdigest()[:24],
            "instruction": instruction, "images": images, "public_state": state,
            "camera_calibration": calibration,
            "evaluator": evaluator,
            "steps_used": self.total_steps, "steps_budget": self.action_budget,
            "wall_used_s": time.monotonic() - self.started, "wall_budget_s": self.wall_budget_s,
            "gripper_command": self.gripper_open,
        }
        if receipt is not None:
            record["execution"] = dict(receipt)
        _atomic_json(self.run / "mailbox" / f"observation-{sequence:06d}.json", record)
        self.snapshot(environment, self.run / "snapshots" / f"{sequence:06d}.json")
        return record

    def render(self, environment: object, command: Mapping[str, object]) -> dict[str, object]:
        """RGB + metric depth for the requested cameras at the requested size, plus pinhole
        calibration in the world frame (MuJoCo camera convention: x right, y up, -z forward)."""
        import numpy as np
        from PIL import Image
        from robosuite.utils.camera_utils import get_real_depth_map
        env = environment.unwrapped
        sim = env.sim
        names = {"left": "robot0_agentview_left", "right": "robot0_agentview_right", "wrist": "robot0_eye_in_hand"}
        width = int(command.get("width", 512))
        height = int(command.get("height", 512))
        # the offscreen framebuffer is sized by the model; never ask for more than it holds
        width = min(width, int(sim.model.vis.global_.offwidth))
        height = min(height, int(sim.model.vis.global_.offheight))
        target = Path(command["dir"])
        target.mkdir(parents=True, exist_ok=True)
        want_depth = bool(command.get("depth", True))
        out: dict[str, object] = {}
        for label in command["cams"]:
            name = names[label]
            if want_depth:
                rgb, depth = sim.render(width=width, height=height, camera_name=name, depth=True)
                depth = get_real_depth_map(sim, np.asarray(depth)[::-1]).astype(np.float32)
            else:
                rgb, depth = sim.render(width=width, height=height, camera_name=name, depth=False), None
            rgb = np.asarray(rgb, dtype=np.uint8)[::-1]
            Image.fromarray(np.ascontiguousarray(rgb[..., :3])).save(target / f"{label}.png", format="PNG")
            entry: dict[str, object] = {"rgb": str(target / f"{label}.png")}
            if depth is not None:
                np.save(target / f"{label}_depth.npy", depth)
                entry["depth_npy"] = str(target / f"{label}_depth.npy")
            cam_id = sim.model.camera_name2id(name)
            fovy = float(sim.model.cam_fovy[cam_id])
            focal = 0.5 * height / np.tan(np.deg2rad(fovy) / 2.0)
            entry.update({
                "width": width, "height": height, "fx": float(focal), "fy": float(focal),
                "cx": (width - 1.0) / 2.0, "cy": (height - 1.0) / 2.0,
                "camera_position_world_m": np.asarray(sim.data.cam_xpos[cam_id], dtype=float).tolist(),
                "camera_xmat_world": np.asarray(sim.data.cam_xmat[cam_id], dtype=float).reshape(3, 3).tolist(),
            })
            out[label] = entry
        return {"kind": "render", "accepted": True, "cameras": out}

    def snapshot(self, environment: object, path: Path) -> dict[str, object]:
        env = environment.unwrapped
        state = env.sim.get_state().flatten().tolist()
        payload = {"task": self.task, "seed": self.seed, "sim_state": state, "gripper_open": self.gripper_open,
                   "total_steps": self.total_steps, "ep_meta": env.get_ep_meta()}
        _atomic_json(path, payload)
        return {"kind": "snapshot", "accepted": True, "path": str(path), "total_steps": self.total_steps}

    def inspect(self, environment: object, path: Path) -> dict[str, object]:
        """Evaluator-side view of a snapshot: object pose, gripper-object distance, official success."""
        import numpy as np
        self.restore(environment, path)
        env = environment.unwrapped
        out: dict[str, object] = {"kind": "inspect", "accepted": True, "path": str(path)}
        try:
            obj_id = env.obj_body_id["obj"]
            obj = np.asarray(env.sim.data.body_xpos[obj_id])
            site = np.asarray(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]])
            out.update({"obj_world_m": obj.tolist(), "gripper_obj_distance_m": float(np.linalg.norm(site - obj)),
                        "obj_velocity_norm": float(np.linalg.norm(env.sim.data.cfrc_ext[obj_id])) if False else None,
                        "gripper_touching_obj": bool(env.check_contact(env.robots[0].gripper["right"], env.objects["obj"])),
                        "official_success": bool(env._check_success()),
                        "base_world_m": np.asarray(env.sim.data.body_xpos[env.sim.model.body_name2id("base0_base")]).tolist() if "base0_base" in env.sim.model.body_names else None})
        except Exception as error:  # task without a single target object
            out["error"] = repr(error)
        return out

    def restore(self, environment: object, path: Path) -> None:
        import numpy as np
        env = environment.unwrapped
        payload = json.loads(path.read_text())
        env.sim.set_state_from_flattened(np.asarray(payload["sim_state"]))
        env.sim.forward()
        if hasattr(env, "update_state"):
            env.update_state()
        self.gripper_open = float(payload["gripper_open"])


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


def run(task: str, seed: int, run: Path, scenes: Path, *, action_budget: int, wall_budget_s: float,
        restore_from: Path | None = None) -> None:
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa_inspect.active_skills import TerminalOutcomePersister

    if not 1 <= action_budget <= MAX_ACTION_BUDGET:
        raise ValueError("action budget is invalid")
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name in ("mailbox", "frames", "video-frames", "snapshots"):
        (run / name).mkdir(mode=0o700)
    episode_tmp = run / "episode-tmp"
    episode_tmp.mkdir(mode=0o700)
    os.environ["ROBOCASA_EPISODE_TMPDIR"] = str(episode_tmp)
    _atomic_json(run / "simulator.json", {
        "schema": "qwen-direct-simulator/v1", "task": task, "seed": seed,
        "network_probe": _network_denied(), "action_budget": action_budget, "wall_budget_s": wall_budget_s,
        "slot_tracking": {"max_joint_step_rad": MAX_JOINT_STEP, "lag_pause_rad": TRACKING_LAG_PAUSE},
    })
    _install_joint_controller()
    environment = gym.make(f"robocasa/{task}", split="pretrain", seed=seed, renderer="mujoco")
    child = Child(task, seed, run, scenes, action_budget=action_budget, wall_budget_s=wall_budget_s)
    terminal: dict[str, object] = {"status": "incomplete"}
    try:
        raw = reset_pinned_scene(environment, task, seed, scenes, run)
        chassis = ChassisHold.from_public_state(raw)
        if restore_from is not None:
            child.restore(environment, restore_from)
            raw = environment.unwrapped.get_observation(environment.unwrapped._get_observations(force_update=True))
            chassis.reset_after_base_action(raw)
        child.started = time.monotonic()
        sequence = 0
        observation = child.publish(raw, environment, sequence, None)
        while True:
            command = _read_command(run, sequence, timeout_s=wall_budget_s + 300)
            kind = command.get("kind")
            if kind == "close":
                terminal = {"status": "closed", "sequence": sequence}
                break
            if kind == "finish":
                success = bool(environment.unwrapped._check_success())
                outcome = TerminalOutcomePersister(run).seal(
                    trace=child.trace, receipts=[{"sequence": sequence}], evaluate=lambda: success)
                terminal = {"status": "success" if outcome["success"] else "finished_false", "sequence": sequence,
                            "official_success": outcome["success"]}
                break
            if command.get("observation_id") != observation["observation_id"]:
                raise RuntimeError("child received a command for a stale observation")
            child.decision += 1
            if kind == "slot":
                raw, receipt = child.execute_slot(environment, command, chassis, raw)
            elif kind == "base":
                raw, receipt = child.execute_base(environment, command, chassis, raw)
            elif kind == "snapshot":
                receipt = child.snapshot(environment, Path(command["path"]))
            elif kind == "render":
                receipt = child.render(environment, command)
            elif kind == "inspect":
                receipt = child.inspect(environment, Path(command["path"]))
                raw = environment.unwrapped.get_observation(environment.unwrapped._get_observations(force_update=True))
            else:
                raise RuntimeError(f"invalid mailbox command kind {kind!r}")
            child.trace.append({"sequence": sequence, "kind": kind, "steps": receipt.get("steps_executed", 0)})
            sequence += 1
            observation = child.publish(raw, environment, sequence, receipt)
    except Exception as error:
        terminal = {"status": "infrastructure_blocked", "error": repr(error)}
        raise
    finally:
        environment.close()
        terminal["simulator_steps"] = child.total_steps
        terminal["wall_s"] = time.monotonic() - child.started
        terminal["video_frames"] = child.video_index
        _atomic_json(run / "simulator-terminal.json", terminal)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--scenes-dir", required=True, type=Path)
    parser.add_argument("--action-budget", type=int, default=DEFAULT_ACTION_BUDGET)
    parser.add_argument("--wall-budget-s", type=float, default=DEFAULT_WALL_BUDGET_S)
    parser.add_argument("--restore-from", type=Path, default=None)
    args = parser.parse_args()
    run(args.task, args.seed, args.run_dir.resolve(), args.scenes_dir.resolve(),
        action_budget=args.action_budget, wall_budget_s=args.wall_budget_s, restore_from=args.restore_from)


if __name__ == "__main__":
    main()
