#!/usr/bin/env python3
"""Boot each interface variant and call its extra commands once against the live simulator.

    smoke_variants.py --out DIR        (joint: fk + move_joints hidden move_ee; path: move_path; macro: grasp_at/place_at)
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
SIM_PY = "/home/jli/work/robocasa-inspect-official/.venv/bin/python"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--scenes", default="/home/jli/state/agp-sim/scenes")
ap.add_argument("--only", default="joint,path,macro")
a = ap.parse_args()
fails = []


def boot(iface):
    out = Path(a.out) / iface
    if out.exists():
        shutil.rmtree(out)
    session, sim = out / "session", out / "sim"
    for d in (session / "bridge", session / "frames", session / "scratch"):
        d.mkdir(parents=True)
    shutil.copy(HERE / "robot_client.py", session / "robot_client.py")
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT / 'runtime'}")
    srv = subprocess.Popen([SIM_PY, str(HERE / "server_sim.py"), "--session", str(session), "--run", str(sim), "--task", "PickPlaceCounterToSink",
                            "--seed", "0", "--scenes", a.scenes, "--interface", iface], stdout=open(out / "boot.log", "w"), stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
    while not (session / "server_boot.json").exists():
        if srv.poll() is not None:
            print(open(out / "boot.log").read()[-2000:]); sys.exit(2)
        time.sleep(1)
    return srv, session


def rc(session, cmd, args=None):
    argv = ["python3", "robot_client.py", ".", cmd] + ([json.dumps(args)] if args is not None else [])
    r = subprocess.run(argv, cwd=session, capture_output=True, text=True)
    resp = json.loads(r.stdout)
    print(f"  {cmd} {json.dumps(args)[:90] if args else ''} -> {json.dumps({k: v for k, v in resp.items() if k not in ('files', 'calibration_file', 'commands')})[:300]}")
    return resp


def stop(srv, session):
    (session / "SERVER_STOP").touch(); srv.wait(timeout=300)


for iface in a.only.split(","):
    print(f"=== {iface}")
    srv, session = boot(iface)
    try:
        h = rc(session, "help"); cmds = set(h["commands"])
        st = rc(session, "state"); ee = st["ee_pose"]
        if iface == "joint":
            if "move_ee" in cmds or "fk" not in cmds: fails.append("joint: help exposes wrong commands")
            r = rc(session, "move_ee", {"position": ee["position"], "rotation": ee["rotation"]})
            if r.get("ok") or "unknown" not in r.get("error", ""): fails.append("joint: move_ee not hidden")
            f = rc(session, "fk", {"joints": st["joints"]})
            d = sum((f["ee_pose"]["position"][k] - ee["position"][k]) ** 2 for k in "xyz") ** 0.5
            if not f["ok"] or d > 0.002: fails.append(f"joint: fk mismatch {d}")
            q = list(st["joints"]); q[0] += 0.3
            m = rc(session, "move_joints", {"joints": q})
            if not m["ok"]: fails.append("joint: move_joints")
        elif iface == "path":
            if "move_path" not in cmds: fails.append("path: no move_path")
            p1 = dict(ee["position"]); p1["x"] += 0.05
            p2 = dict(ee["position"]); p2["z"] -= 0.05
            r = rc(session, "move_path", {"waypoints": [{"position": p1, "rotation": ee["rotation"]}, {"position": p2, "rotation": ee["rotation"], "gripper": "close"},
                                                        {"position": ee["position"], "rotation": ee["rotation"], "gripper": "open"}]})
            if not r["ok"] or r["completed"] != 3 or r["results"][1].get("held") is not False: fails.append(f"path: {r}")
            st2 = rc(session, "status")
            if st2["commands_used"] != 1: fails.append(f"path: counted {st2['commands_used']} not 1")
        elif iface == "macro":
            if "grasp_at" not in cmds: fails.append("macro: no grasp_at")
            g = rc(session, "grasp_at", {"position": {"x": 0.45, "y": -0.1, "z": 0.30}, "rotation": {"w": 0, "x": 1, "y": 0, "z": 0}})
            if g.get("held") is not False or g.get("ok"): fails.append(f"macro grasp_at on empty air should be held:false ok:false: {g}")
            if [s["step"] for s in g["steps"]][:3] != ["open", "above", "descend"]: fails.append(f"macro steps {g['steps']}")
            pl = rc(session, "place_at", {"position": {"x": 0.45, "y": -0.1, "z": 0.32}, "rotation": {"w": 0, "x": 1, "y": 0, "z": 0}})
            if not pl["ok"]: fails.append(f"macro place_at {pl}")
    finally:
        stop(srv, session)
print("FAILS:", fails or "none")
sys.exit(1 if fails else 0)
