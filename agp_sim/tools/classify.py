#!/usr/bin/env python3
"""Classify every finished episode under a matrix/runs dir by its deepest milestone and its dominant
failure signal, from result.json + server.log + agent events. Prints a table and writes classify.md.

    python classify.py <dir-with-run-subdirs>...
"""
import json, re, sys
from collections import Counter
from pathlib import Path


def classify(run: Path) -> dict:
    r = json.loads((run / "result.json").read_text())
    m = r.get("milestones", {})
    log = (run / "session" / "server.log").read_text(errors="replace") if (run / "session" / "server.log").exists() else ""
    errs = Counter(re.findall(r"err=(\w+)", log))
    cmds = Counter(re.findall(r"#\d+ (\w+) ok=", log))
    ev = (run / "agent_events.jsonl") if (run / "agent_events.jsonl").exists() else run / "session" / "agent_events.jsonl"
    nudges = sum(1 for l in ev.read_text().splitlines() if '"type": "nudge"' in l) if ev.exists() else 0
    u = r.get("usage", {})
    if r["official_success"]:
        stage = "SUCCESS"
    elif m.get("lifted"):
        stage = "lifted-not-placed"
    elif m.get("touched"):
        stage = "touched-no-lift"
    elif m.get("approached"):
        stage = "approached-no-contact"
    else:
        stage = "never-approached"
    signal = []
    if errs.get("IK_FAILED", 0) >= 3:
        signal.append(f"IK_FAILED x{errs['IK_FAILED']}")
    if errs.get("SETTLE_MISS", 0) >= 2:
        signal.append(f"SETTLE_MISS x{errs['SETTLE_MISS']}")
    if errs.get("CLAMP", 0) >= 2:
        signal.append(f"CLAMP x{errs['CLAMP']}")
    if cmds.get("gripper", 0) + cmds.get("grasp_at", 0) + cmds.get("move_path", 0) == 0:
        signal.append("never closed gripper")
    if nudges:
        signal.append(f"loop nudges x{nudges}")
    if r.get("agent_outcome") in ("wall_clock", "max_turns"):
        signal.append(r["agent_outcome"])
    if cmds.get("move_base", 0) >= 3:
        signal.append(f"move_base x{cmds['move_base']}")
    return {"run": run.name, "task": r["task"], "seed": r["seed"], "stage": stage, "agent": r["agent_verdict"],
            "min_dist": m.get("min_gripper_obj_distance_m"), "cmds": r["cmds_counted"], "moves": cmds.get("move_ee", 0) + cmds.get("move_delta", 0) + cmds.get("move_joints", 0) + cmds.get("move_path", 0) + cmds.get("grasp_at", 0) + cmds.get("place_at", 0),
            "frames": cmds.get("frames", 0), "turns": u.get("turns"), "wall_min": round(r["wall_s"] / 60, 1), "signals": "; ".join(signal)}


rows = []
for d in sys.argv[1:]:
    for rj in sorted(Path(d).glob("*/result.json")):
        rows.append(classify(rj.parent))
    if (Path(d) / "result.json").exists():
        rows.append(classify(Path(d)))
head = "| run | stage | agent | min dist | cmds | moves | frames | turns | min | signals |\n|---|---|---|---|---|---|---|---|---|---|"
lines = [head] + [f"| {r['run']} | {r['stage']} | {r['agent']} | {r['min_dist']} | {r['cmds']} | {r['moves']} | {r['frames']} | {r['turns']} | {r['wall_min']} | {r['signals']} |" for r in rows]
stages = Counter(r["stage"] for r in rows)
text = "\n".join(lines) + "\n\nstages: " + json.dumps(stages) + "\n"
print(text)
if len(sys.argv) == 2:
    (Path(sys.argv[1]) / "classify.md").write_text(text)
