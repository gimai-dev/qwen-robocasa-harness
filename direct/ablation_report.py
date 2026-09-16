"""Evaluator-side tables for the measure ablation (clean vs measure on the same starts).

    python -m direct.ablation_report --matrix <matrix dir> --out <md> [--json <path>]

Milestones reuse ``direct.milestones.per_run`` on rows built from the evaluator
field of every published observation (``sim/mailbox/observation-*.json``), so no
re-simulation is needed. For measure runs it also reports how often the model
measured, whether the measurement succeeded, how far the returned centroid was
from the evaluator's object position at that instant, and whether the next
motion target used the measurement (ee target within 5 cm in x, y of the
returned centroid or top point).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from .milestones import per_run

USED_XY_M = 0.05
ON_OBJECT_M = 0.10


def evaluator_rows(run: Path) -> list[dict]:
    rows = []
    for path in sorted((run / "sim" / "mailbox").glob("observation-*.json")):
        o = json.loads(path.read_text())
        e = o.get("evaluator") or {}
        if e.get("obj_world_m") is None:
            continue
        rows.append({"sequence": o["sequence"], "steps_used": o["steps_used"], "obj_world_m": e["obj_world_m"],
                     "gripper_obj_distance_m": e["gripper_obj_distance_m"], "gripper_touching_obj": e.get("gripper_touching_obj"),
                     "official_success": e.get("official_success"), "gripper_command": o["gripper_command"],
                     "gripper_width_m": o["public_state"]["gripper_width_m"]})
    return rows


def measure_stats(run: Path, rows: list[dict]) -> dict:
    obj_by_seq = {r["sequence"]: r["obj_world_m"] for r in rows}
    decisions = [json.loads(l) for l in (run / "decisions.jsonl").read_text().splitlines() if l.strip()]
    out = {"measures": 0, "ok": 0, "errors": [], "cams": defaultdict(int), "with_above_z": 0,
           "centroid_error_m": [], "centroid_xy_error_m": [], "on_object": 0, "used_next": 0, "limit_hits": 0}
    for i, d in enumerate(decisions):
        if d.get("status") == "invalid_action" and "measure limit" in str(d.get("error", "")):
            out["limit_hits"] += 1
        if d.get("status") != "measured":
            continue
        out["measures"] += 1
        r = d["receipt"]
        out["cams"][r["cam"]] += 1
        if r.get("above_z") is not None:
            out["with_above_z"] += 1
        if not r.get("ok"):
            out["errors"].append(r.get("error"))
            continue
        out["ok"] += 1
        obj = obj_by_seq.get(d.get("observation_sequence_after"))
        if obj is not None:
            c = r["centroid_world_m"]
            err = math.dist(c, obj); xy = math.dist(c[:2], obj[:2])
            out["centroid_error_m"].append(round(err, 4)); out["centroid_xy_error_m"].append(round(xy, 4))
            if err <= ON_OBJECT_M:
                out["on_object"] += 1
        nxt = next((x for x in decisions[i + 1:] if x.get("action") and x["action"].get("k") != "measure"), None)
        if nxt and nxt["action"].get("k") == "ee" and nxt["action"].get("p"):
            p = nxt["action"]["p"]
            if min(math.dist(p[:2], r["centroid_world_m"][:2]), math.dist(p[:2], r["top_point_world_m"][:2])) <= USED_XY_M:
                out["used_next"] += 1
    out["cams"] = dict(out["cams"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    matrix = Path(args.matrix)
    runs = []
    for result_path in sorted(matrix.glob("*/result.json")):
        run = result_path.parent
        result = json.loads(result_path.read_text())
        rows = evaluator_rows(run)
        m = per_run(rows) if rows else None
        entry = {"run": run.name, "task": result["task"], "seed": result["seed"], "method": result["method"],
                 "official_success": result.get("official_success"), "termination": result.get("termination"),
                 "decisions": result.get("decisions"), "rejected": result.get("rejected_actions"),
                 "measurements": result.get("measurements", 0), "steps": result.get("simulator_steps"),
                 "wall_s": result.get("wall_s"), "milestones": m}
        if result["method"] == "measure":
            entry["measure"] = measure_stats(run, rows)
        runs.append(entry)
    by_method = defaultdict(list)
    for r in runs:
        by_method[r["method"]].append(r)
    lines = ["# Measure ablation: evaluator-side milestones", "", f"Matrix: `{matrix}`", "",
             "| method | n | approach<=0.10m | contact | hold | lift>=3cm | displaced>=10cm | success | median closest m |",
             "|---|---|---|---|---|---|---|---|---|"]
    for method in sorted(by_method):
        ms = [r["milestones"] for r in by_method[method] if r["milestones"]]
        f = lambda k: sum(1 for x in ms if x[k])
        dists = sorted(x["min_dist"] for x in ms)
        med = dists[len(dists) // 2] if dists else float("nan")
        succ = sum(1 for r in by_method[method] if r["official_success"])
        lines.append(f"| {method} | {len(by_method[method])} | {f('approach')} | {f('contact')} | {f('hold')} | {f('lift')} | {f('displaced')} | {succ} | {med:.3f} |")
    lines += ["", "## Per start", "", "| task | seed | method | success | closest m | contact | hold | max lift m | displaced | decisions | rejected | measures | termination |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(runs, key=lambda x: (x["task"], x["seed"], x["method"])):
        m = r["milestones"] or {}
        lines.append(f"| {r['task']} | {r['seed']} | {r['method']} | {r['official_success']} | {m.get('min_dist', float('nan')):.3f} | {m.get('contact')} | {m.get('hold')} | {m.get('max_lift', 0.0):.3f} | {m.get('displaced')} | {r['decisions']} | {r['rejected']} | {r['measurements']} | {r['termination']} |")
    meas = [r for r in runs if r["method"] == "measure"]
    if meas:
        lines += ["", "## Measurement use (measure runs)", "",
                  "| task | seed | measures | ok | cams | above_z | centroid within 10 cm of object | median centroid err m | median xy err m | next ee target used it | limit hits | errors |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(meas, key=lambda x: (x["task"], x["seed"])):
            s = r["measure"]
            errs = sorted(s["centroid_error_m"]); xys = sorted(s["centroid_xy_error_m"])
            med = lambda v: f"{v[len(v) // 2]:.3f}" if v else "-"
            lines.append(f"| {r['task']} | {r['seed']} | {s['measures']} | {s['ok']} | {s['cams']} | {s['with_above_z']} | {s['on_object']} | {med(errs)} | {med(xys)} | {s['used_next']} | {s['limit_hits']} | {len(s['errors'])} |")
    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text)
    if args.json:
        Path(args.json).write_text(json.dumps(runs, indent=1))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
