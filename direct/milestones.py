"""Local-progress milestones per episode from evaluator-side inspection rows.

    python -m direct.milestones --inspection <bank>/inspection.json --out <md>

Milestones (per episode, from simulator snapshots the policy never sees):
  approach   closest TCP-object distance <= 0.10 m
  contact    gripper touched the object
  hold       gripper closed (command 0) with width > 0.005 m while touching the object
  lift       object raised >= 0.03 m above its initial height while held
  displaced  object ended >= 0.10 m from where it started
  success    official predicate
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def per_run(rows):
    seq = sorted(rows, key=lambda r: r["sequence"])
    if not seq or seq[0].get("obj_world_m") is None:
        return None
    start = seq[0]["obj_world_m"]
    out = {"min_dist": min(r["gripper_obj_distance_m"] for r in seq), "contact": False, "hold": False, "lift": False,
           "max_lift": 0.0, "success": any(bool(r.get("official_success")) for r in seq)}
    for r in seq:
        closed = int(round(float(r["gripper_command"]))) == 0
        holding = closed and (r.get("gripper_width_m") or 0) > 0.005 and bool(r.get("gripper_touching_obj"))
        out["contact"] |= bool(r.get("gripper_touching_obj"))
        out["hold"] |= holding
        lift = r["obj_world_m"][2] - start[2]
        if holding:
            out["max_lift"] = max(out["max_lift"], lift)
    out["lift"] = out["max_lift"] >= 0.03
    out["displaced"] = math.dist(seq[-1]["obj_world_m"][:2], start[:2]) >= 0.10
    out["approach"] = out["min_dist"] <= 0.10
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspection", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = json.loads(Path(args.inspection).read_text())
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run"]].append(r)
    table = defaultdict(list)
    detail = []
    for run, rs in by_run.items():
        m = per_run(rs)
        if m is None:
            continue
        m.update({"run": Path(run).name, "task": rs[0]["task"], "seed": rs[0]["seed"], "method": rs[0]["method"]})
        detail.append(m)
        table[m["method"]].append(m)
    lines = ["# Local-progress milestones (evaluator-side, EE-short)", "",
             "| method | n | approach<=0.10m | contact | hold | lift>=3cm | displaced>=10cm | success | median closest dist m |", "|---|---|---|---|---|---|---|---|---|"]
    for method in sorted(table):
        ms = table[method]
        dists = sorted(x["min_dist"] for x in ms)
        med = dists[len(dists) // 2]
        f = lambda k: sum(1 for x in ms if x[k])
        lines.append(f"| {method} | {len(ms)} | {f('approach')} | {f('contact')} | {f('hold')} | {f('lift')} | {f('displaced')} | {f('success')} | {med:.3f} |")
    lines += ["", "## Per run", "", "| method | task | seed | closest m | contact | hold | max lift m | displaced | success |", "|---|---|---|---|---|---|---|---|---|"]
    for m in sorted(detail, key=lambda x: (x["method"], x["task"], x["seed"])):
        lines.append(f"| {m['method']} | {m['task']} | {m['seed']} | {m['min_dist']:.3f} | {m['contact']} | {m['hold']} | {m['max_lift']:.3f} | {m['displaced']} | {m['success']} |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
