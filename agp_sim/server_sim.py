#!/usr/bin/env python3
"""agent-as-policy session server on RoboCasa (PandaOmron).

The same file bridge and the same command set as agp/server_real.py of
agent-as-policy-2026/agent-as-policy: robot_client.py writes <session>/bridge/req_NNNNNN.json,
this server executes them ONE AT A TIME against the RoboCasa simulator child
(direct.executor.Simulator, the mailbox protocol of the Qwen direct-control campaign)
and writes resp_NNNNNN.json. Responses carry only images, calibration, proprioception
and command outcomes; the evaluator-side ground truth the child publishes is written
to <session>/../evaluator.jsonl and never enters a response.

Frames: every pose the agent sees or sends is in the ROBOT BASE frame (the Panda
mount on the mobile base, z up, metres). Tool frame = agent-as-policy convention
(+z out of the fingers, fingers close along tool y); it is the FK grip_site frame
rotated -90 deg about z.

Physics guards (all CLI args): reach radius from the base origin, z range, max
straight-line step. Reachability / joint limits are the IK's job (IK_FAILED), contact
stalls are reported as SETTLE_MISS with the achieved pose.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from direct.executor import READY_Q, Simulator  # noqa: E402
from direct.kinematics import (JOINT_LIMITS, panda_fk, plan_pose_segment,  # noqa: E402
                               quat_xyzw_to_matrix, solve_pose_multistart)

FREE_CMDS = {"status", "help", "state", "deproject"}
MOTION_CMDS = {"move_ee", "move_delta", "move_joints", "home", "gripper", "move_base"}
CAMS = ("left", "right", "wrist")
SLOT_STEPS = 64
MAX_SLOTS = 8
SETTLE_TOL_RAD = 0.03
CONTACT_FORCE_N = 15.0
MEASURE_LIMIT = 24            # free deproject calls allowed between two counted commands
HELD_MIN_WIDTH_M = 0.012      # an empty Panda close ends at ~1-7 mm in this sim; objects here are >= 2 cm
BASE_VELOCITY = 0.5
BASE_STEP_M = 0.048          # measured displacement of one 20-step base slot at v=0.5 (smoke_child)
BASE_STEP_RAD = 0.355
# agent tool frame = FK grip_site frame @ M  (M = Rz(-90 deg)): +z approach, fingers close along tool y
M_AGENT = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
CV_FLIP = np.diag([1.0, -1.0, -1.0])   # MuJoCo camera (x right, y up, -z forward) -> CV (x right, y down, z forward)


# ---------- quaternion helpers (scalar-first) ----------
def _wxyz_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _mat_to_wxyz(R):
    m = np.asarray(R, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / (np.linalg.norm(q) or 1.0)


def _rot_delta_mat(deg_xyz):
    rx, ry, rz = [math.radians(d) for d in deg_xyz]
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _r4(x):
    return round(float(x), 4)


def _pose(p, R):
    q = _mat_to_wxyz(R)
    return {"position": {"x": _r4(p[0]), "y": _r4(p[1]), "z": _r4(p[2])},
            "rotation": {"w": _r4(q[0]), "x": _r4(q[1]), "y": _r4(q[2]), "z": _r4(q[3])}}


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class BootError(RuntimeError):
    pass


class Server:
    def __init__(self, a):
        self.session = Path(a.session).resolve()
        self.bridge = self.session / "bridge"
        self.frames = self.session / "frames"
        for d in (self.bridge, self.frames, self.session / "scratch"):
            d.mkdir(parents=True, exist_ok=True)
        self.log_path = self.session / "server.log"
        self.evaluator_path = self.session.parent / "evaluator.jsonl"
        self.budget = int(a.budget)
        self.counted = 0
        self.free_streak = 0
        self.r_max, self.z_min, self.z_max, self.max_step = float(a.r_max), float(a.z_min), float(a.z_max), float(a.max_step_m)
        self.capture_seq = itertools.count(1)
        self.img_w, self.img_h = int(a.image_size), int(a.image_size)
        self.max_open = None
        self.sim = Simulator(task=a.task, seed=int(a.seed), run=Path(a.run).resolve(), scenes=Path(a.scenes).resolve(),
                             action_budget=int(a.sim_steps), wall_budget_s=float(a.sim_wall_s))
        t0 = time.time()
        obs = self.sim.launch()
        self.instruction = obs.get("instruction")
        self._log(f"BOOT sim launched in {time.time() - t0:.1f}s task={a.task} seed={a.seed} instruction={self.instruction!r}")
        self._record_evaluator("boot")
        # leave the straight-arm singularity (RoboCurve replication: ready_pose=true)
        res = self._run_path([list(READY_Q)], 1.0, label="ready")
        self._log(f"BOOT ready pose: {res.get('ok')} err={res.get('final_joint_error_rad')}")
        st = self._state_raw()
        self.max_open = max(0.07, float(st["gripper_width_m"]))
        self.ready_pose = self._ee_agent(st)
        self._record_evaluator("ready")
        probe = self.sim.send({"kind": "render", "cams": ["wrist"], "width": self.img_w, "height": self.img_h, "depth": False,
                               "dir": str(self.sim.run / "render" / "probe")}, timeout_s=120)["execution"]["cameras"]["wrist"]
        self.img_w, self.img_h = int(probe["width"]), int(probe["height"])   # the offscreen buffer may cap the height
        boot = {"instruction": self.instruction, "ready_pose": self.ready_pose, "max_opening_m": self.max_open,
                "task": a.task, "seed": int(a.seed), "image_size": [self.img_w, self.img_h]}
        (self.session / "server_boot.json").write_text(json.dumps(boot, indent=1, default=_jsonable))

    # ---------- infrastructure ----------
    def _log(self, line):
        with open(self.log_path, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {line}\n")

    def _record_evaluator(self, tag):
        ev = dict(self.sim.observation.get("evaluator") or {})
        ev.update({"t": time.time(), "tag": tag, "counted": self.counted, "steps": self.sim.observation.get("steps_used")})
        with open(self.evaluator_path, "a") as f:
            f.write(json.dumps(ev, default=_jsonable) + "\n")

    def _state_raw(self):
        return self.sim.observation["public_state"]

    def _q(self):
        return [float(v) for v in self._state_raw()["arm_q_rad"]]

    def _ee_agent(self, st=None):
        """Current TCP pose in the base frame, agent tool convention."""
        st = st or self._state_raw()
        p = np.asarray(st["tcp_base_position_m"], dtype=float)
        R_fk = quat_xyzw_to_matrix(st["tcp_base_quat_xyzw"])
        return _pose(p, R_fk @ M_AGENT)

    def _base_of(self, st):
        return {"position": [_r4(v) for v in st["base_world_position_m"]], "yaw_rad": _r4(st["base_world_yaw_rad"])}

    # ---------- motion primitives ----------
    def _run_path(self, waypoints, gripper, label="path"):
        """Track a joint waypoint path with as many 64-step slots as needed. Returns the last receipt
        plus 'ok' (settled within tolerance), total steps and peak contact force."""
        total, peak, err, prev_err, slots = 0, 0.0, None, None, 0
        ex = None
        path = [list(map(float, w)) for w in waypoints]
        while slots < MAX_SLOTS:
            obs = self.sim.send({"kind": "slot", "waypoints": path, "g": float(gripper), "steps": SLOT_STEPS}, timeout_s=600)
            ex = obs.get("execution") or {}
            if ex.get("kind") != "slot":
                raise RuntimeError(f"simulator answered a {ex.get('kind')!r} receipt to a slot command")
            slots += 1
            total += int(ex["steps_executed"])
            peak = max(peak, float(ex.get("peak_force_n", 0.0)))
            err = float(ex["final_joint_error_rad"])
            if err < SETTLE_TOL_RAD or ex.get("budget_truncated"):
                break
            done_wp = int(ex.get("waypoints_commanded", 1))
            path = path[max(0, done_wp - 1):]
            if prev_err is not None and err > prev_err - 0.005:
                break           # no progress: blocked or controller stalled
            prev_err = err
        out = {"ok": err is not None and err < SETTLE_TOL_RAD, "steps": total, "slots": slots,
               "peak_force_n": round(peak, 2), "final_joint_error_rad": _r4(err if err is not None else float("nan")),
               "budget_truncated": bool(ex and ex.get("budget_truncated"))}
        return out

    def _move_to_fk(self, p_base, R_fk, mode, gripper):
        """Plan (straight line or joint-space) to an FK-frame target in the base frame and execute."""
        q = self._q()
        if mode == "plan":
            direct = solve_pose_multistart(q, p_base, R_fk)
            if direct["status"] != "kinematically_reachable":
                return None, {"error": f"IK_FAILED: no joint solution within limits (residual {direct['position_error_m']*1000:.0f} mm)",
                              "error_code": "IK_FAILED", "retryable": False}
            waypoints, path_type = [direct["q"]], "joint_space"
        else:
            plan = plan_pose_segment(q, p_base, R_fk)
            if plan["status"] == "kinematically_reachable":
                waypoints, path_type = plan["waypoints"], "straight_line"
            else:
                direct = solve_pose_multistart(q, p_base, R_fk)
                if direct["status"] == "kinematically_reachable":
                    waypoints, path_type = [direct["q"]], "joint_space_fallback"
                else:
                    failed = plan["failed_result"]
                    why = ("joint-space discontinuity along the straight path (IK branch change)" if plan.get("reason") == "ik_discontinuity"
                           else "no IK solution within joint limits")
                    return None, {"error": f"IK_FAILED: {why}; straight path failed at {plan['failed_fraction']:.0%} "
                                           f"(residual {failed['position_error_m']*1000:.0f} mm); direct IK residual {direct['position_error_m']*1000:.0f} mm",
                                  "error_code": "IK_FAILED", "retryable": False}
        res = self._run_path(waypoints, gripper)
        res["path"] = path_type
        # Cartesian refinement: the joint tolerance (0.03 rad) can leave ~15 mm at the tool; re-solve from
        # where the arm actually is and track once more, unless contact stopped the move.
        for _ in range(2):
            if not res["ok"] or res["peak_force_n"] > CONTACT_FORCE_N:
                break
            q_now = self._q()
            err_m = float(np.linalg.norm(panda_fk(q_now)[0] - np.asarray(p_base, dtype=float)))
            if err_m < 0.004:
                break
            fix = solve_pose_multistart(q_now, p_base, R_fk)
            if fix["status"] != "kinematically_reachable" or fix["max_joint_delta_rad"] > 0.3:
                break
            again = self._run_path([fix["q"]], gripper)
            res["steps"] += again["steps"]; res["slots"] += again["slots"]
            res["peak_force_n"] = max(res["peak_force_n"], again["peak_force_n"])
            res["ok"] = again["ok"]; res["final_joint_error_rad"] = again["final_joint_error_rad"]
            res["refined"] = True
        return res, None

    def _clamp(self, p, cur):
        r = math.hypot(p[0], p[1])
        if r > self.r_max:
            return f"target horizontal distance {r:.3f} m from the base exceeds r_max {self.r_max}"
        if not (self.z_min <= p[2] <= self.z_max):
            return f"target z {p[2]:.3f} m outside [{self.z_min}, {self.z_max}]"
        step = float(np.linalg.norm(np.asarray(p) - np.asarray(cur)))
        if step > self.max_step:
            return f"step {step:.3f} m exceeds max_step_m {self.max_step}; split the move"
        return None

    def _finish_move(self, res, err, target_p, t0, label):
        st = self._state_raw()
        ee = self._ee_agent(st)
        ach = np.asarray(st["tcp_base_position_m"], dtype=float)
        d_mm = round(float(np.linalg.norm(ach - np.asarray(target_p, dtype=float))) * 1000, 1) if target_p is not None else None
        out = {"ok": False, "error": None, "move": label, "ee_pose": ee, "target_error_mm": d_mm,
               "duration_s": round(time.time() - t0, 1), "joints": [_r4(v) for v in st["arm_q_rad"]],
               "gripper_width_m": _r4(st["gripper_width_m"])}
        if err:
            out.update(err)
            return out
        out.update({"steps": res["steps"], "path": res.get("path"), "peak_contact_force_n": res["peak_force_n"]})
        if res["ok"]:
            out["ok"] = True
        elif res["peak_force_n"] > CONTACT_FORCE_N:
            out.update({"error": f"SETTLE_MISS: motion blocked by contact (peak force {res['peak_force_n']:.0f} N); "
                                 f"the arm stopped {d_mm} mm from the target and holds there. Inspect fresh frames, then lift/retreat or retry from here",
                        "error_code": "SETTLE_MISS", "retryable": True})
        elif res.get("budget_truncated"):
            out.update({"error": "simulator step budget exhausted", "error_code": "BUDGET", "retryable": False})
        else:
            out.update({"error": f"SETTLE_MISS: the arm did not settle onto the target (joint error {res['final_joint_error_rad']:.3f} rad, "
                                 f"{d_mm} mm away); it holds where it stopped. Re-observe, then retry with a corrected target or a shorter move",
                        "error_code": "SETTLE_MISS", "retryable": True})
        return out

    # ---------- commands ----------
    def cmd_status(self, a):
        st = self._state_raw()
        return {"ok": True, "robot": "robocasa_panda_omron", "task": self.sim.task, "motion_enabled": True,
                "commands_used": self.counted, "budget_hard": self.budget,
                "clamps": {"r_max": self.r_max, "z_min": self.z_min, "z_max": self.z_max, "max_step_m": self.max_step},
                "max_opening_m": self.max_open, "ready_pose": self.ready_pose, "base_world": self._base_of(st),
                "sim_steps_used": self.sim.observation.get("steps_used"), "time": time.strftime("%H:%M:%S")}

    def cmd_help(self, a):
        return {"ok": True, "commands": {
            "status": "server and budget info (free)",
            "help": "this list (free)",
            "state": "{} -> joints (7, rad), ee_pose (base frame, tool convention), gripper_fraction, gripper_width_m, base_world, contact_force_n (free)",
            "frames": '{"cams": ["left","right","wrist"], "depth": true} (both optional) -> captures the cameras from one instant; saves frames/NNNN_<cam>.png, frames/NNNN_<cam>_depth.npy (float32 metres) and frames/NNNN_calib.json; returns the paths',
            "deproject": '{"capture": N, "cam": "wrist", "u": px, "v": px} -> 3D point (base frame) from the saved depth (5x5 median); add "plane_z": <m> to intersect the pixel ray with the horizontal plane z = plane_z instead. '
                         'Region form: {"capture": N, "cam": "wrist", "region": [u0,v0,u1,v1], "above_z": <m>} -> centroid, min/max, extent and top point of all surface points in that rectangle above the plane (free)',
            "move_ee": '{"position": {"x","y","z"}, "rotation": {"w","x","y","z"}, "mode": "linear"|"plan"} -> straight line holding orientation (default) or joint-space move to the IK solution; returns achieved ee_pose + target_error_mm; ok:false + error_code (CLAMP / IK_FAILED / SETTLE_MISS) if refused or not reached',
            "move_delta": '{"dpos": [dx,dy,dz], "drot_deg": [rx,ry,rz]} (either optional) -> relative straight-line move from the current pose; rotation deltas about the BASE axes',
            "move_joints": '{"joints": [7 floats radians]} -> joint-space move',
            "home": "{} -> the ready/observe posture (arm raised over the workspace, wrist camera looking forward-down)",
            "gripper": '{"action": "open"|"close"} -> returns fraction (0 closed .. 1 open) and width_m; close stops on the object, so a clearly nonzero fraction after close means something is held',
            "move_base": '{"axis": "x"|"y"|"yaw", "distance": <m or rad>} -> drives the mobile base: x forward along its heading, y to its left, yaw counter-clockwise; |distance| <= 0.5 m / 1.0 rad per call; the arm holds its joints; returns base_world before/after',
        }}

    def cmd_state(self, a):
        st = self._state_raw()
        frac = float(st["gripper_width_m"]) / self.max_open
        return {"ok": True, "joints": [_r4(v) for v in st["arm_q_rad"]], "ee_pose": self._ee_agent(st),
                "gripper_fraction": _r4(min(1.0, max(0.0, frac))), "gripper_width_m": _r4(st["gripper_width_m"]),
                "base_world": self._base_of(st), "contact_force_n": st.get("contact_force_delta_n"),
                "sequence": self.sim.observation.get("sequence")}

    def cmd_frames(self, a):
        want = a.get("cams") or list(CAMS)
        for c in want:
            if c not in CAMS:
                return {"ok": False, "error": f"unknown camera {c!r}; valid: left, right, wrist"}
        seq = next(self.capture_seq)
        tmp = self.sim.run / "render" / f"{seq:04d}"
        obs = self.sim.send({"kind": "render", "cams": list(want), "width": self.img_w, "height": self.img_h,
                             "depth": bool(a.get("depth", True)), "dir": str(tmp)}, timeout_s=120)
        cams = obs["execution"]["cameras"]
        st = obs["public_state"]
        R_bw = quat_xyzw_to_matrix(st["base_world_quat_xyzw"])
        base_p = np.asarray(st["base_world_position_m"], dtype=float)
        files, calib = {}, {}
        for label, c in cams.items():
            rgb_src = Path(c["rgb"])
            rgb_dst = self.frames / f"{seq:04d}_{label}.png"
            os.replace(rgb_src, rgb_dst)
            entry = {"rgb": str(rgb_dst)}
            if c.get("depth_npy"):
                d_dst = self.frames / f"{seq:04d}_{label}_depth.npy"
                os.replace(c["depth_npy"], d_dst)
                entry["depth_npy"] = str(d_dst)
            files[label] = entry
            cam_p_w = np.asarray(c["camera_position_world_m"], dtype=float)
            xmat_w = np.asarray(c["camera_xmat_world"], dtype=float)
            cam_p_b = R_bw.T @ (cam_p_w - base_p)
            R_cv_b = R_bw.T @ xmat_w @ CV_FLIP
            K = [[c["fx"], 0.0, c["cx"]], [0.0, c["fy"], c["cy"]], [0.0, 0.0, 1.0]]
            calib[label] = {"intrinsics": K, "pose": _pose(cam_p_b, R_cv_b), "frame": "robot_base",
                            "image_size": [c["width"], c["height"]], "distortion_model": "none",
                            "distortion_coefficients": None, "rectified": True}
        calib["_meta"] = {"joints": [_r4(v) for v in st["arm_q_rad"]], "ee_pose": self._ee_agent(st),
                          "gripper_fraction": _r4(min(1.0, float(st["gripper_width_m"]) / self.max_open)),
                          "base_world": self._base_of(st), "capture_time": time.strftime("%Y-%m-%d %H:%M:%S")}
        cp = self.frames / f"{seq:04d}_calib.json"
        cp.write_text(json.dumps(calib, indent=1, default=_jsonable))
        return {"ok": True, "capture": seq, "files": files, "calibration_file": str(cp),
                "note": "depth npy = float32 metres along the optical axis; calib pose = camera pose in the robot base frame"}

    def cmd_deproject(self, a):
        seq = int(a["capture"]); cam = str(a.get("cam", "wrist"))
        if a.get("region") is not None:
            return self._deproject_region(seq, cam, a)
        u = int(a["u"]); v = int(a["v"])
        cp = self.frames / f"{seq:04d}_calib.json"
        if not cp.exists():
            return {"ok": False, "error": f"no calibration for capture {seq} (take frames first)"}
        allc = json.loads(cp.read_text())
        if cam not in allc:
            return {"ok": False, "error": f"capture {seq} has no camera {cam!r}"}
        calib = allc[cam]
        K = np.asarray(calib["intrinsics"], dtype=float)
        pose = calib["pose"]
        R = _wxyz_to_mat([pose["rotation"][k] for k in "wxyz"])
        t = np.array([pose["position"][k] for k in "xyz"], dtype=float)
        w, h = [int(x) for x in calib["image_size"]]
        if not (0 <= v < h and 0 <= u < w):
            return {"ok": False, "error": f"pixel ({u},{v}) outside image {w}x{h}"}
        ray_c = np.linalg.inv(K) @ np.array([u, v, 1.0])
        if a.get("plane_z") is not None:
            zp = float(a["plane_z"])
            r_w = R @ ray_c
            if abs(r_w[2]) < 1e-6:
                return {"ok": False, "error": "ray is parallel to the horizontal plane"}
            s = (zp - t[2]) / r_w[2]
            if s <= 0:
                return {"ok": False, "error": f"plane z={zp} is behind the camera along that ray"}
            X = t + s * r_w
            return {"ok": True, "mode": "plane", "plane_z": zp, "point_base": {"x": _r4(X[0]), "y": _r4(X[1]), "z": _r4(X[2])},
                    "ray_length_m": _r4(s * float(np.linalg.norm(r_w)))}
        dp = self.frames / f"{seq:04d}_{cam}_depth.npy"
        if not dp.exists():
            return {"ok": False, "error": f"no depth for cam {cam!r} in capture {seq}; use \"plane_z\""}
        depth = np.load(dp)
        win = depth[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
        vals = win[(win > 0) & np.isfinite(win)]
        if vals.size == 0:
            return {"ok": False, "error": "no valid depth in the 5x5 window at that pixel"}
        d = float(np.median(vals))
        X = R @ (d * ray_c) + t
        return {"ok": True, "mode": "depth", "point_base": {"x": _r4(X[0]), "y": _r4(X[1]), "z": _r4(X[2])},
                "depth_m": _r4(d), "valid_in_window": int(vals.size)}

    def _deproject_region(self, seq, cam, a):
        """3D statistics of every valid depth pixel inside a rectangle: centroid, extent and the highest
        point, optionally keeping only points above a horizontal plane (e.g. the counter top). This is how
        an object's centre and size are measured instead of one surface pixel."""
        u0, v0, u1, v1 = [int(x) for x in a["region"]]
        cp = self.frames / f"{seq:04d}_calib.json"
        dp = self.frames / f"{seq:04d}_{cam}_depth.npy"
        if not cp.exists() or not dp.exists():
            return {"ok": False, "error": f"no depth/calibration for capture {seq} camera {cam!r} (take frames first)"}
        calib = json.loads(cp.read_text())[cam]
        K = np.asarray(calib["intrinsics"], dtype=float)
        pose = calib["pose"]
        R = _wxyz_to_mat([pose["rotation"][k] for k in "wxyz"])
        t = np.array([pose["position"][k] for k in "xyz"], dtype=float)
        depth = np.load(dp)
        h, w = depth.shape
        u0, u1 = sorted((max(0, u0), min(w - 1, u1))); v0, v1 = sorted((max(0, v0), min(h - 1, v1)))
        if u1 <= u0 or v1 <= v0:
            return {"ok": False, "error": "region must be [u0, v0, u1, v1] inside the image with u1 > u0 and v1 > v0"}
        vs, us = np.mgrid[v0:v1 + 1, u0:u1 + 1]
        d = depth[v0:v1 + 1, u0:u1 + 1].astype(float)
        ok = (d > 0) & np.isfinite(d)
        rays = np.linalg.inv(K) @ np.stack([us.ravel(), vs.ravel(), np.ones(us.size)])
        X = (R @ (rays * d.ravel())).T + t
        X = X[ok.ravel()]
        if a.get("above_z") is not None:
            X = X[X[:, 2] > float(a["above_z"])]
        if len(X) < 5:
            return {"ok": False, "error": "fewer than 5 valid points in the region (after the above_z filter)"}
        lo, hi = X.min(axis=0), X.max(axis=0)
        c = X.mean(axis=0)
        top = X[np.argmax(X[:, 2])]
        return {"ok": True, "mode": "region", "n_points": int(len(X)),
                "centroid_base": {"x": _r4(c[0]), "y": _r4(c[1]), "z": _r4(c[2])},
                "min_base": {"x": _r4(lo[0]), "y": _r4(lo[1]), "z": _r4(lo[2])},
                "max_base": {"x": _r4(hi[0]), "y": _r4(hi[1]), "z": _r4(hi[2])},
                "extent_m": {"x": _r4(hi[0] - lo[0]), "y": _r4(hi[1] - lo[1]), "z": _r4(hi[2] - lo[2])},
                "top_point_base": {"x": _r4(top[0]), "y": _r4(top[1]), "z": _r4(top[2])},
                "note": "points are the VISIBLE surface: an object's centre is about half its extent below top_point_base "
                        "and in the middle of min/max in x and y"}

    def _gripper_cmd(self):
        return float(self.sim.observation.get("gripper_command", 1.0))

    def cmd_move_ee(self, a):
        t0 = time.time()
        p = [float(a["position"][k]) for k in "xyz"]
        R_agent = _wxyz_to_mat([float(a["rotation"][k]) for k in "wxyz"])
        R_fk = R_agent @ M_AGENT.T
        cur = self._state_raw()["tcp_base_position_m"]
        bad = self._clamp(p, cur)
        if bad:
            return {"ok": False, "error": bad, "error_code": "CLAMP", "ee_pose": self._ee_agent()}
        res, err = self._move_to_fk(np.asarray(p), R_fk, a.get("mode", "linear"), self._gripper_cmd())
        return self._finish_move(res, err, p, t0, "move_ee")

    def cmd_move_delta(self, a):
        t0 = time.time()
        st = self._state_raw()
        cur = np.asarray(st["tcp_base_position_m"], dtype=float)
        dp = a.get("dpos") or [0, 0, 0]
        p = cur + np.asarray([float(v) for v in dp], dtype=float)
        R_fk = quat_xyzw_to_matrix(st["tcp_base_quat_xyzw"])
        R_agent = R_fk @ M_AGENT
        if a.get("drot_deg"):
            R_agent = _rot_delta_mat([float(x) for x in a["drot_deg"]]) @ R_agent
        R_fk = R_agent @ M_AGENT.T
        bad = self._clamp(p.tolist(), cur.tolist())
        if bad:
            return {"ok": False, "error": bad, "error_code": "CLAMP", "ee_pose": self._ee_agent()}
        res, err = self._move_to_fk(p, R_fk, "linear", self._gripper_cmd())
        return self._finish_move(res, err, p.tolist(), t0, "move_delta")

    def cmd_move_joints(self, a):
        t0 = time.time()
        joints = [float(x) for x in a["joints"]]
        if len(joints) != 7:
            return {"ok": False, "error": "joints must have 7 entries (radians)"}
        for i, (v, (lo, hi)) in enumerate(zip(joints, JOINT_LIMITS)):
            if not lo <= v <= hi:
                return {"ok": False, "error": f"JOINT_LIMIT: joint {i} = {v:.3f} outside [{lo:.3f}, {hi:.3f}]", "error_code": "JOINT_LIMIT"}
        res = self._run_path([joints], self._gripper_cmd())
        target_p = panda_fk(joints)[0]
        return self._finish_move(res, None, target_p, t0, "move_joints")

    def cmd_home(self, a):
        t0 = time.time()
        res = self._run_path([list(READY_Q)], self._gripper_cmd(), label="home")
        out = self._finish_move(res, None, panda_fk(READY_Q)[0], t0, "home")
        out.setdefault("note", "home = the ready/observe posture")
        return out

    def cmd_gripper(self, a):
        t0 = time.time()
        act = a.get("action", "close")
        if act == "open":
            g = 1.0
        elif act == "close":
            g = 0.0
        else:
            g = 1.0 if float(act) >= 0.5 else 0.0
        q = self._q()
        obs = self.sim.send({"kind": "slot", "waypoints": [q], "g": g, "steps": 24}, timeout_s=120)
        st = obs["public_state"]
        width = float(st["gripper_width_m"])
        frac = min(1.0, max(0.0, width / self.max_open))
        out = {"ok": True, "action": act, "fraction": _r4(frac), "width_m": _r4(width), "duration_s": round(time.time() - t0, 1)}
        if g == 0.0:
            held = width >= HELD_MIN_WIDTH_M
            out["held"] = held
            out["note"] = (f"closed on something {width * 1000:.0f} mm wide: an object is between the pads" if held else
                           f"the pads closed to {width * 1000:.0f} mm: the grasp is EMPTY (nothing between the fingers). The fingers STAY CLOSED "
                           "until you send gripper open, so open them before the next approach or they will only poke the object; re-observe and re-aim")
        else:
            out["note"] = "gripper open" if frac > 0.9 else f"the gripper opened only to {width * 1000:.0f} mm: something is blocking the fingers"
        return out

    def cmd_move_base(self, a):
        t0 = time.time()
        axis = str(a.get("axis", "x"))
        dist = float(a.get("distance", 0.0))
        if axis not in ("x", "y", "yaw") or dist == 0.0:
            return {"ok": False, "error": 'move_base needs {"axis": "x"|"y"|"yaw", "distance": nonzero}'}
        limit = 1.0 if axis == "yaw" else 0.5
        if abs(dist) > limit:
            return {"ok": False, "error": f"|distance| must be <= {limit} per call", "error_code": "CLAMP"}
        st0 = self._state_raw()
        before = self._base_of(st0)
        v = BASE_VELOCITY if dist > 0 else -BASE_VELOCITY
        steps = 0
        p0 = np.asarray(before["position"][:2], dtype=float)
        yaw0 = float(before["yaw_rad"])
        # closed loop: one 20-step base slot at a time until the measured displacement reaches the request
        # (per-slot displacement differs by axis: ~5 cm in x against furniture drag, ~15 cm in y)
        for _ in range(24):
            obs = self.sim.send({"kind": "base", "a": axis, "v": v, "g": self._gripper_cmd(), "steps": 20, "motion_steps": 16}, timeout_s=120)
            steps += int(obs["execution"]["steps_executed"])
            st = self._state_raw()
            if axis == "yaw":
                moved = abs(((float(st["base_world_yaw_rad"]) - yaw0 + math.pi) % (2 * math.pi)) - math.pi)
            else:
                moved = float(np.linalg.norm(np.asarray(st["base_world_position_m"][:2]) - p0))
            if moved >= abs(dist) - 0.02:
                break
            ex = obs["execution"]
            b0, b1 = ex.get("base_world_before_m"), ex.get("base_world_after_m")
            if axis != "yaw" and b0 and b1 and math.hypot(b1[0] - b0[0], b1[1] - b0[1]) < 0.005:
                break   # blocked: stop pushing
        st = self._state_raw()
        after = self._base_of(st)
        moved = [round(a_ - b_, 4) for a_, b_ in zip(after["position"], before["position"])]
        out = {"ok": True, "axis": axis, "requested": dist, "base_world_before": before, "base_world_after": after,
               "base_moved_world_m": moved, "yaw_moved_rad": _r4(after["yaw_rad"] - before["yaw_rad"]),
               "steps": steps, "duration_s": round(time.time() - t0, 1), "ee_pose": self._ee_agent(st)}
        if axis != "yaw" and math.hypot(*moved[:2]) < min(0.02, abs(dist) * 0.2):
            out["note"] = "the base barely moved: it is probably blocked by furniture in that direction"
        return out

    def cmd_reset(self, a):
        return {"ok": False, "error_code": "NO_RESET",
                "error": "reset is not available: recover by re-observing and dealing with the scene as it now is"}

    # ---------- loop ----------
    def dispatch(self, req):
        cmd = req.get("cmd", "")
        fn = getattr(self, f"cmd_{cmd}", None)
        if fn is None:
            return {"ok": False, "error": f"unknown command {cmd!r}; run help"}
        if cmd not in FREE_CMDS:
            if self.counted >= self.budget:
                return {"ok": False, "error": f"hard command budget ({self.budget}) exhausted", "error_code": "BUDGET"}
            self.counted += 1
            self.free_streak = 0
        elif cmd == "deproject":
            self.free_streak += 1
            if self.free_streak > MEASURE_LIMIT:
                return {"ok": False, "error_code": "MEASUREMENT_LIMIT",
                        "error": f"{self.free_streak - 1} deproject calls since the last motion or capture; measuring again returns "
                                 "the same numbers. Act on them (move_ee / move_delta / gripper / move_base) or take new frames first"}
        try:
            resp = fn(req.get("args") or {})
        except (KeyError, TypeError, ValueError) as e:
            usage = self.cmd_help({})["commands"].get(cmd, "")
            resp = {"ok": False, "error": f"bad arguments for {cmd}: {type(e).__name__} {str(e)[:120]}; usage: {usage[:300]}",
                    "error_code": "INVALID_ARGUMENT"}
        except Exception as e:  # noqa: BLE001
            self._log(traceback.format_exc())
            resp = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}", "error_code": "INTERNAL"}
        if cmd in MOTION_CMDS:
            self._record_evaluator(cmd)
        return resp

    def run(self):
        print(f"READY session={self.session} task={self.sim.task} instruction={self.instruction!r}", flush=True)
        self._log("READY")
        stop = self.session / "SERVER_STOP"
        try:
            while True:
                if stop.exists():
                    self._log("SERVER_STOP seen, finishing")
                    break
                reqs = []
                for f in os.listdir(self.bridge):
                    if f.startswith("req_") and f.endswith(".json"):
                        rid = f[4:-5]
                        if not (self.bridge / f"resp_{rid}.json").exists():
                            try:
                                reqs.append(int(rid))
                            except ValueError:
                                pass
                for rid in sorted(reqs):
                    rp = self.bridge / f"req_{rid:06d}.json"
                    try:
                        req = json.loads(rp.read_text())
                    except Exception:  # noqa: BLE001
                        time.sleep(0.1)
                        try:
                            req = json.loads(rp.read_text())
                        except Exception as e:  # noqa: BLE001
                            req = {"cmd": "_unreadable", "error": str(e)}
                    t0 = time.time()
                    resp = self.dispatch(req)
                    resp["id"] = rid
                    out = self.bridge / f"resp_{rid:06d}.json"
                    tmp = out.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(resp, default=_jsonable))
                    os.replace(tmp, out)
                    self._log(f"#{rid} {req.get('cmd')} ok={resp.get('ok')} {time.time() - t0:.1f}s used={self.counted}"
                              + (f" err={resp.get('error_code')}" if not resp.get('ok') else ""))
                time.sleep(0.2)
        finally:
            self._record_evaluator("final")
            try:
                terminal = self.sim.finish()
            except Exception as e:  # noqa: BLE001
                terminal = {"status": "finish_failed", "error": repr(e)}
            result = {"terminal": terminal, "commands_used": self.counted,
                      "official_success": bool((terminal or {}).get("official_success", False))}
            (self.session.parent / "server_result.json").write_text(json.dumps(result, indent=1, default=_jsonable))
            self._log(f"FINISHED official_success={result['official_success']}")
            try:
                self.sim.close()
            except Exception:  # noqa: BLE001
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--run", required=True, help="simulator run dir (mailbox, snapshots, video frames); NOT inside the session")
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--sim-steps", type=int, default=20000)
    ap.add_argument("--sim-wall-s", type=float, default=4 * 3600)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--r-max", type=float, default=0.95)
    ap.add_argument("--z-min", type=float, default=-0.3)
    ap.add_argument("--z-max", type=float, default=1.5)
    ap.add_argument("--max-step-m", type=float, default=0.40)
    a = ap.parse_args()
    try:
        srv = Server(a)
    except Exception as e:  # noqa: BLE001
        print(f"BOOT_ERROR: {type(e).__name__}: {str(e)[:400]}", flush=True)
        traceback.print_exc()
        sys.exit(2)
    srv.run()


if __name__ == "__main__":
    main()
