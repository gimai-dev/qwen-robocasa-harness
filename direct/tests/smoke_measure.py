"""Ground-truth check of the ``measure`` primitive against the live simulator.

Errors it detects:
  pixel_convention   the render calibration projects the TCP to a different pixel than
                     the public ``tcp_pixels`` -> u/v flip, intrinsics or frame mistake
  region_geometry    a rectangle around the projected target object deprojects to a
                     centroid farther than 5 cm from the evaluator's object position
  action_path        decode_action + execute produce a "measured" receipt with 0 steps
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from direct.actions import decode_action
from direct.executor import Simulator, execute
from direct.measure import RENDER_SIZE, deproject_region

p = argparse.ArgumentParser(); p.add_argument("--run", type=Path, required=True); p.add_argument("--scenes", type=Path, required=True)
p.add_argument("--task", default="PickPlaceCounterToSink"); p.add_argument("--seed", type=int, default=0)
a = p.parse_args()


def project(calib, X):
    cam = np.asarray(calib["camera_position_world_m"]); xmat = np.asarray(calib["camera_xmat_world"]).reshape(3, 3)
    local = xmat.T @ (np.asarray(X) - cam)
    depth = -local[2]
    return calib["cx"] + calib["fx"] * local[0] / depth, calib["cy"] - calib["fy"] * local[1] / depth, depth


t0 = time.monotonic()
sim = Simulator(task=a.task, seed=a.seed, run=a.run, scenes=a.scenes, action_budget=900, wall_budget_s=1200)
obs = sim.launch()
obj = np.asarray(obs["evaluator"]["obj_world_m"]); tcp = np.asarray(obs["public_state"]["tcp_world_position_m"])
print(f"launch {time.monotonic()-t0:.1f}s obj_world={np.round(obj,4)} tcp_world={np.round(tcp,4)}")
results = {"obj_world_m": obj.tolist(), "cameras": {}}
target = sim.sim / "measure" / "smoke"
obs = sim.send({"kind": "render", "cams": ["left", "right", "wrist"], "width": RENDER_SIZE, "height": RENDER_SIZE, "depth": True, "dir": str(target)})
cams = obs["execution"]["cameras"]
worst = 0.0
for label, calib in cams.items():
    row = {}
    u, v, d = project(calib, tcp)
    public = obs["public_state"]["tcp_pixels"][label]
    if public["visible"]:
        row["tcp_pixel_gap_px"] = float(np.hypot(u - public["u"], v - public["v"]))
        print(f"{label}: tcp projects to ({u:.1f},{v:.1f}) public says ({public['u']},{public['v']}) gap {row['tcp_pixel_gap_px']:.2f}px")
    u, v, d = project(calib, obj)
    row["obj_pixel"] = [round(u, 1), round(v, 1)]; row["obj_depth_m"] = round(float(d), 4)
    if not (0 <= u < RENDER_SIZE and 0 <= v < RENDER_SIZE and d > 0):
        print(f"{label}: object not in view ({u:.0f},{v:.0f})"); results["cameras"][label] = row; continue
    depth = np.load(calib["depth_npy"])
    row["depth_at_obj_px_m"] = float(depth[int(round(v)), int(round(u))])
    region = (int(u) - 6, int(v) - 6, int(u) + 6, int(v) + 6)
    for above in (None, float(obj[2]) - 0.02):
        m = deproject_region(depth, calib, region, above)
        key = "plain" if above is None else "above_z"
        row[key] = m
        if m["ok"]:
            err = float(np.linalg.norm(np.asarray(m["centroid_world_m"]) - obj))
            xy = float(np.linalg.norm(np.asarray(m["centroid_world_m"])[:2] - obj[:2]))
            row[key]["centroid_error_m"] = round(err, 4); row[key]["centroid_xy_error_m"] = round(xy, 4)
            worst = max(worst, xy)
            print(f"{label} {key}: n={m['n_points']} centroid={m['centroid_world_m']} top={m['top_point_world_m']} extent={m['extent_m']} err={err:.3f} xy_err={xy:.3f}")
        else:
            print(f"{label} {key}: {m}")
    results["cameras"][label] = row
# action path through decode + execute on the wrist camera
label = "wrist" if "obj_pixel" in results["cameras"].get("wrist", {}) else "left"
u, v = results["cameras"][label]["obj_pixel"]
raw = {"k": "measure", "cam": label, "region": [u - 8, v - 8, u + 8, v + 8], "above_z": None, "p": None, "o": None, "g": None, "a": None, "v": None, "n": "smoke"}
action = decode_action(raw, interface="ee", measure=True)
before = sim.steps_used()
r = execute(sim, action)
print(f"execute: status={r.status} steps={r.steps} steps_used {before}->{sim.steps_used()} ok={r.detail.get('ok')} centroid={r.detail.get('centroid_world_m')}")
results["execute"] = {"status": r.status, "steps": r.steps, "detail": r.detail}
try:
    decode_action(raw, interface="ee")
    print("ERROR: clean decode accepted measure"); results["clean_rejects_measure"] = False
except ValueError as e:
    print(f"clean decode rejects measure: {e}"); results["clean_rejects_measure"] = True
sim.finish()
results["worst_xy_error_m"] = worst; results["total_wall_s"] = time.monotonic() - t0
(a.run / "smoke-measure.json").write_text(json.dumps(results, indent=1))
print(f"worst centroid xy error {worst:.3f} m; total {time.monotonic()-t0:.1f}s")
