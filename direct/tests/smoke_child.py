"""Phase A interface checks against the live simulator (run on h200-4).

Each check names the error it detects:
  fk_error        FK chain vs public relative TCP -> joint order / frame mistakes
  ee_displacement +5 cm world-x EE action -> world/base transform, unit, sign
  slot_steps      exactly 20 steps per slot -> hidden extra steps / early exit
  gripper         close -> width shrinks; open -> width grows -> polarity
  base            +x velocity -> base moves; TCP world pose tracks the base
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from direct.actions import Action
from direct.executor import Simulator, execute
from direct.kinematics import quat_xyzw_to_matrix, matrix_to_quat_xyzw

p = argparse.ArgumentParser(); p.add_argument("--run", type=Path, required=True); p.add_argument("--scenes", type=Path, required=True)
p.add_argument("--task", default="PickPlaceCounterToSink"); p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
t0 = time.monotonic()
sim = Simulator(task=a.task, seed=a.seed, run=a.run, scenes=a.scenes, action_budget=900, wall_budget_s=1200)
obs = sim.launch()
print(f"launch {time.monotonic()-t0:.1f}s instruction={obs['instruction']!r}")
s = obs["public_state"]; results = {}
results["fk_error_m"] = s["fk_position_error_m"]
print("fk_error_m", s["fk_position_error_m"], "tcp_world", np.round(s["tcp_world_position_m"], 4), "width", s["gripper_width_m"])

def ee_move(dx, dy, dz, g=None):
    p0 = np.asarray(s0["tcp_world_position_m"]); o0 = s0["tcp_world_quat_xyzw"]
    act = Action("ee", position_m=tuple(p0 + [dx, dy, dz]), quat_xyzw=tuple(o0), gripper=g)
    t = time.monotonic(); r = execute(sim, act); dt = time.monotonic() - t
    if r.child is None:
        print("REJECTED", r.status, r.detail); sys.exit(2)
    return r, dt

s0 = s
r, dt = ee_move(0.05, 0, 0)
d = np.asarray(r.child["tcp_world_after_m"]) - np.asarray(s0["tcp_world_position_m"])
print(f"ee +5cm x: status={r.status} steps={r.steps} moved={np.round(d,4)} ik_wp={r.detail.get('ik_waypoints')} {dt:.1f}s")
results["ee_x_displacement_m"] = d.tolist(); results["slot_steps"] = r.steps; results["slot_wall_s"] = dt
s0 = sim.observation["public_state"]
r, dt = ee_move(0, 0, -0.05)
d = np.asarray(r.child["tcp_world_after_m"]) - np.asarray(s0["tcp_world_position_m"])
print(f"ee -5cm z: status={r.status} steps={r.steps} moved={np.round(d,4)} {dt:.1f}s")
results["ee_z_displacement_m"] = d.tolist()
s0 = sim.observation["public_state"]
r, dt = ee_move(0, 0, 0, g=0)
print(f"close: status={r.status} steps={r.steps} width {r.child['gripper_width_before_m']:.4f}->{r.child['gripper_width_after_m']:.4f}")
results["width_after_close_m"] = r.child["gripper_width_after_m"]
s0 = sim.observation["public_state"]
r, dt = ee_move(0, 0, 0, g=1)
print(f"open: status={r.status} steps={r.steps} width {r.child['gripper_width_before_m']:.4f}->{r.child['gripper_width_after_m']:.4f}")
results["width_after_open_m"] = r.child["gripper_width_after_m"]
# joint action: +0.2 rad on joint1
s0 = sim.observation["public_state"]; q = list(s0["arm_q_rad"]); q[0] += 0.2
r = execute(sim, Action("joint", q_rad=tuple(q)))
print(f"joint1 +0.2: status={r.status} steps={r.steps} err={r.child['final_joint_error_rad']:.4f} q1 {s0['arm_q_rad'][0]:.3f}->{r.child['arm_q_after'][0]:.3f}")
results["joint_error_rad"] = r.child["final_joint_error_rad"]
# base +x
s0 = sim.observation["public_state"]
r = execute(sim, Action("base", axis="x", velocity=0.5))
db = np.asarray(r.child["base_world_after_m"]) - np.asarray(s0["base_world_position_m"])
dt_ = np.asarray(r.child["tcp_world_after_m"]) - np.asarray(s0["tcp_world_position_m"])
print(f"base x 0.5: status={r.status} steps={r.steps} base moved={np.round(db,4)} tcp moved={np.round(dt_,4)} arm err={r.child['final_joint_error_rad']:.4f}")
results["base_x_displacement_m"] = db.tolist(); results["tcp_after_base_m"] = dt_.tolist()
s0 = sim.observation["public_state"]
r = execute(sim, Action("base", axis="yaw", velocity=0.5))
print(f"base yaw 0.5: yaw {s0['base_world_yaw_rad']:.4f}->{r.child['base_yaw_after_rad']:.4f} steps={r.steps}")
results["base_yaw_delta_rad"] = r.child["base_yaw_after_rad"] - s0["base_world_yaw_rad"]
# unreachable
s0 = sim.observation["public_state"]
r = execute(sim, Action("ee", position_m=(float(s0["tcp_world_position_m"][0]) + 2.0, 0.0, 3.0), quat_xyzw=tuple(s0["tcp_world_quat_xyzw"])))
print(f"unreachable: status={r.status} steps={r.steps} detail={r.detail}")
results["unreachable_status"] = r.status
terminal = sim.finish()
print("terminal", json.dumps(terminal)[:400])
results["terminal"] = terminal; results["total_wall_s"] = time.monotonic() - t0
(a.run / "smoke-results.json").write_text(json.dumps(results, indent=1))
print(f"total {time.monotonic()-t0:.1f}s")
