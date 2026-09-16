#!/usr/bin/env python3
"""Compact view of one episode: agent turns (tool calls + truncated outputs), server log tail, evaluator milestones.

    python show_run.py <run_dir> [--last N] [--full]
"""
import argparse, json, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("--last", type=int, default=0); ap.add_argument("--full", action="store_true")
ap.add_argument("--width", type=int, default=220)
a = ap.parse_args()
run = Path(a.run); session = run / "session"
W = a.width if not a.full else 100000
ev = (run / "agent_events.jsonl") if (run / "agent_events.jsonl").exists() else session / "agent_events.jsonl"
rows = [json.loads(l) for l in ev.read_text().splitlines() if l.strip()] if ev.exists() else []
if a.last:
    # keep the last N responses and everything after them
    idx = [i for i, e in enumerate(rows) if e["type"] == "response"]
    rows = rows[idx[-a.last]:] if len(idx) >= a.last else rows
for e in rows:
    t = e["type"]
    if t == "response":
        content = (e.get("content") or "").replace("\n", " ")[:W]
        print(f"R[{e.get('prompt_tokens')}t {e.get('seconds')}s] {content}")
        for tc in e.get("tool_calls", []):
            print(f"   -> {tc['name']} {tc['arguments'][:W]}")
    elif t == "exec":
        print(f"      = {e.get('seconds')}s {e.get('output','').replace(chr(10),' | ')[:W]}")
    elif t == "view_image":
        print(f"      img {e['path']}")
    else:
        print(f"   * {t} {json.dumps({k: v for k, v in e.items() if k not in ('t', 'type')})[:W]}")
print("--- server.log tail")
log = session / "server.log"
if log.exists():
    print("\n".join(log.read_text().splitlines()[-8:]))
evp = run / "evaluator.jsonl"
if evp.exists():
    er = [json.loads(l) for l in evp.read_text().splitlines() if l.strip()]
    d = [r.get("gripper_obj_distance_m") for r in er if r.get("gripper_obj_distance_m") is not None]
    print(f"--- evaluator: rows {len(er)} min dist {min(d):.3f} touched {any(r.get('gripper_touching_obj') for r in er)} "
          f"success {any(r.get('official_success') for r in er)} last {json.dumps(er[-1])[:200]}")
rj = run / "result.json"
if rj.exists():
    r = json.loads(rj.read_text())
    print("--- result:", json.dumps({k: r[k] for k in ("official_success", "agent_verdict", "milestones", "cmds_counted", "wall_s", "agent_outcome")}))
