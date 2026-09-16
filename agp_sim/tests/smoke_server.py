#!/usr/bin/env python3
"""Exercise server_sim.py through robot_client.py exactly as the agent would.

Checks: boot + ready pose, status/state/help, frames (3 cams, depth, calib), the projection
convention (TCP projected with the saved calib must land on the gripper pixel the child reports),
deproject depth vs plane, move_delta (+5 cm x, -5 cm z), move_ee back, gripper close/open,
move_base, home, unknown command, and the sealed official outcome on stop.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
SIM_PY = "/home/jli/work/robocasa-inspect-official/.venv/bin/python"

p = argparse.ArgumentParser()
p.add_argument("--out", required=True); p.add_argument("--task", default="PickPlaceCounterToSink"); p.add_argument("--seed", type=int, default=0)
p.add_argument("--scenes", default="/home/jli/state/agp-sim/scenes")
a = p.parse_args()
out = Path(a.out); session, sim = out / "session", out / "sim"
subprocess.run("pkill -f 'agp_sim/server_sim.py' ; pkill -f 'direct.sim_child'", shell=True); time.sleep(2)
if out.exists():
    import shutil; shutil.rmtree(out)
for d in (session / "bridge", session / "frames", session / "scratch"):
    d.mkdir(parents=True)
shutil_src = HERE / "robot_client.py"
(session / "robot_client.py").write_bytes(shutil_src.read_bytes())
env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT / 'runtime'}")
srv = subprocess.Popen([SIM_PY, str(HERE / "server_sim.py"), "--session", str(session), "--run", str(sim), "--task", a.task,
                        "--seed", str(a.seed), "--scenes", a.scenes], stdout=open(out / "server_boot.log", "w"), stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
t0 = time.time()
while not (session / "server_boot.json").exists():
    if srv.poll() is not None:
        print(open(out / "server_boot.log").read()[-3000:]); sys.exit(2)
    time.sleep(1)
boot = json.loads((session / "server_boot.json").read_text())
print(f"boot {time.time()-t0:.1f}s instruction={boot['instruction']!r} ready={boot['ready_pose']} max_open={boot['max_opening_m']}")


def rc(cmd, args=None):
    t = time.time()
    argv = ["python3", "robot_client.py", ".", cmd] + ([json.dumps(args)] if args is not None else [])
    r = subprocess.run(argv, cwd=session, capture_output=True, text=True)
    try:
        resp = json.loads(r.stdout)
    except json.JSONDecodeError:
        print("BAD RESPONSE", cmd, r.stdout[:300], r.stderr[:300]); sys.exit(3)
    keep = {k: v for k, v in resp.items() if k not in ("files", "calibration_file")}
    print(f"[{time.time()-t:.1f}s] {cmd} {json.dumps(args) if args else ''} -> {json.dumps(keep)[:400]}")
    return resp


def R_of(q):
    w, x, y, z = q["w"], q["x"], q["y"], q["z"]
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


fails = []
st = rc("status"); assert st["ok"]
s0 = rc("state"); assert s0["ok"] and len(s0["joints"]) == 7
rc("help")
fr = rc("frames")
assert fr["ok"] and set(fr["files"]) == {"left", "right", "wrist"}, fr
calib = json.loads(Path(fr["calibration_file"]).read_text())
# projection check: TCP (state ee position, base frame) through each camera's calib vs the child's tcp_pixels
obs = json.loads(sorted(sim.rglob("observation-*.json"))[-1].read_text())
tcp_px = obs["public_state"]["tcp_pixels"]
ee = np.array([s0["ee_pose"]["position"][k] for k in "xyz"])
def child_to_ours(ref, size):   # the child's 256x256 pixel -> our WxH pixel (same fovy, square pixels)
    W, H = size; k = H / 256.0
    return (ref["u"] - 127.5) * k + (W - 1) / 2.0, (ref["v"] - 127.5) * k + (H - 1) / 2.0
for cam in ("left", "right", "wrist"):
    c = calib[cam]; K = np.asarray(c["intrinsics"]); R = R_of(c["pose"]["rotation"]); t = np.array([c["pose"]["position"][k] for k in "xyz"])
    Xc = R.T @ (ee - t)
    if Xc[2] <= 0:
        print(f"  {cam}: TCP behind camera"); continue
    uv = K @ (Xc / Xc[2])
    ref = tcp_px[cam]
    if ref["u"] is None:
        print(f"  {cam}: child says TCP not visible"); continue
    ru, rv = child_to_ours(ref, c["image_size"])
    err = np.hypot(uv[0] - ru, uv[1] - rv)
    print(f"  projection {cam}: ours ({uv[0]:.1f},{uv[1]:.1f}) child ({ru:.1f},{rv:.1f}) err {err:.1f} px")
    if err > 3.0:
        fails.append(f"projection {cam} err {err:.1f}px")
    # deproject that pixel with depth and with the plane at the TCP height: both should be near the TCP
    u, v = int(round(uv[0])), int(round(uv[1]))
    if 0 <= u < c["image_size"][0] and 0 <= v < c["image_size"][1]:
        dp = rc("deproject", {"capture": fr["capture"], "cam": cam, "u": u, "v": v})
        pl = rc("deproject", {"capture": fr["capture"], "cam": cam, "u": u, "v": v, "plane_z": float(ee[2])})
        if pl["ok"]:
            X = np.array([pl["point_base"][k] for k in "xyz"]); e = np.linalg.norm(X - ee)
            print(f"  plane deproject {cam}: err {e*1000:.0f} mm")
            if e > 0.02: fails.append(f"plane deproject {cam} {e*1000:.0f} mm")
        if dp["ok"]:
            X = np.array([dp["point_base"][k] for k in "xyz"]); print(f"  depth deproject {cam}: {X.round(3).tolist()} vs tcp {ee.round(3).tolist()} (depth hits the gripper surface, so a few cm off is fine)")
reg = rc("deproject", {"capture": fr["capture"], "cam": "wrist", "region": [200, 300, 320, 420]})
if not reg["ok"]: fails.append("region deproject")
reg2 = rc("deproject", {"capture": fr["capture"], "cam": "left", "region": [0, 0, 511, 479], "above_z": 0.25})
d = np.load(fr["files"]["wrist"]["depth_npy"]); print(f"  wrist depth: shape {d.shape} min {np.nanmin(d[d>0]):.3f} max {np.nanmax(d[np.isfinite(d)]):.3f}")
cp = rc("check_pose", {"position": {"x": 0.45, "y": 0.0, "z": 0.40}, "rotation": {"w": 0, "x": 1, "y": 0, "z": 0}})
if not cp.get("reachable"): fails.append(f"check_pose reachable: {cp}")
cp = rc("check_pose", {"position": {"x": 0.9, "y": 0.0, "z": 0.60}, "rotation": {"w": 0, "x": 1, "y": 0, "z": 0}})
if cp.get("reachable"): fails.append(f"check_pose far target should be unreachable: {cp}")
# motion
m1 = rc("move_delta", {"dpos": [0.05, 0, 0]})
if not m1["ok"]: fails.append("move_delta +x")
e1 = np.array([m1["ee_pose"]["position"][k] for k in "xyz"]); print(f"  moved {np.round(e1-ee,4).tolist()}")
if abs((e1 - ee)[0] - 0.05) > 0.01 or abs((e1-ee)[1]) > 0.01: fails.append(f"move_delta +x moved {np.round(e1-ee,4).tolist()}")
m2 = rc("move_delta", {"dpos": [0, 0, -0.05]})
e2 = np.array([m2["ee_pose"]["position"][k] for k in "xyz"]); print(f"  moved {np.round(e2-e1,4).tolist()}")
m3 = rc("move_ee", {"position": s0["ee_pose"]["position"], "rotation": s0["ee_pose"]["rotation"], "mode": "linear"})
if not m3["ok"] or m3["target_error_mm"] > 15: fails.append(f"move_ee back {m3.get('target_error_mm')}")
m4 = rc("move_delta", {"drot_deg": [0, 0, 30]})
if not m4["ok"]: fails.append("move_delta yaw 30")
rc("move_ee", {"position": s0["ee_pose"]["position"], "rotation": s0["ee_pose"]["rotation"], "mode": "plan"})
# straight-down canonical pose 20 cm below/forward of ready
p = dict(s0["ee_pose"]["position"]); p["z"] = 0.42; p["x"] += 0.05   # above the counter (top at z~0.22 + fingers)
down = rc("move_ee", {"position": p, "rotation": {"w": 0, "x": 1, "y": 0, "z": 0}})
if not down["ok"]: fails.append(f"straight-down pose: {down.get('error')}")
g = rc("gripper", {"action": "close"}); print(f"  close -> fraction {g['fraction']} width {g['width_m']}")
if g["fraction"] > 0.1: fails.append("close fraction not near 0 when empty")
g = rc("gripper", {"action": "open"}); print(f"  open -> fraction {g['fraction']}")
if g["fraction"] < 0.9: fails.append("open fraction not near 1")
b = rc("move_base", {"axis": "x", "distance": 0.10}); print(f"  base moved {b.get('base_moved_world_m')}")
b = rc("move_base", {"axis": "yaw", "distance": -0.3}); print(f"  yaw moved {b.get('yaw_moved_rad')}")
ab = rc("approach_base", {"target": {"x": 1.2, "y": 0.6}, "standoff": 0.55}); print(f"  approach_base -> {ab.get('target_now')} turned {ab.get('turned_rad')} driven {ab.get('driven_m')} blocked {ab.get('blocked')}")
if not ab.get("ok") or abs(ab["target_now"]["y"]) > 0.25: fails.append(f"approach_base did not face the target: {ab}")
h = rc("home")
if not h["ok"]: fails.append("home")
rc("bogus")
far = rc("move_ee", {"position": {"x": 2.0, "y": 0, "z": 0.5}, "rotation": {"w": 0, "x": 1, "y": 0, "z": 0}})
if far.get("error_code") != "CLAMP": fails.append("far target not clamped")
far = rc("move_ee", {"position": {"x": 0.9, "y": 0, "z": 1.4}, "rotation": {"w": 0, "x": 1, "y": 0, "z": 0}})
print(f"  unreachable -> {far.get('error_code')}")
st = rc("status"); print(f"  commands used {st['commands_used']}")
(session / "SERVER_STOP").touch(); srv.wait(timeout=300)
res = json.loads((out / "server_result.json").read_text()); print("server_result", json.dumps(res)[:300])
ev = [json.loads(l) for l in (out / "evaluator.jsonl").read_text().splitlines()]
print(f"evaluator rows {len(ev)} last {json.dumps(ev[-1])[:200]}")
print("FAILS:", fails if fails else "none", f"total {time.time()-t0:.0f}s")
sys.exit(1 if fails else 0)
