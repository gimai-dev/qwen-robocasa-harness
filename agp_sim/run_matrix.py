#!/usr/bin/env python3
"""Run a list of (task, seed) episodes sequentially (resumable) and summarise.

    python run_matrix.py --tasks PickPlaceCounterToSink PickPlaceCounterToDrawer --seeds 0 1 2 \
        --out /home/jli/state/agp-sim/matrix/<name> [--parallel 2] [-- <run_episode args>]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def summarise(out: Path) -> str:
    rows = []
    for rj in sorted(out.glob("*/result.json")):
        r = json.loads(rj.read_text())
        m = r.get("milestones", {})
        u = r.get("usage", {})
        rows.append(r)
    lines = ["| task | seed | success | agent | approach | touched | lifted | min dist m | cmds | turns | tokens in (k) | wall min |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    by_task: dict[str, list[int]] = {}
    for r in rows:
        m = r.get("milestones", {}); u = r.get("usage", {})
        by_task.setdefault(r["task"], []).append(int(r["official_success"]))
        lines.append(f"| {r['task']} | {r['seed']} | {int(r['official_success'])} | {r['agent_verdict']} | {int(bool(m.get('approached')))} | "
                     f"{int(bool(m.get('touched')))} | {int(bool(m.get('lifted')))} | {m.get('min_gripper_obj_distance_m')} | {r['cmds_counted']} | "
                     f"{u.get('turns')} | {round((u.get('prompt_tokens') or 0) / 1000)} | {round(r['wall_s'] / 60, 1)} |")
    total = sum(sum(v) for v in by_task.values()); n = sum(len(v) for v in by_task.values())
    head = [f"# Matrix summary — {out.name}", "", f"Success {total}/{n}" + "".join(f"; {t} {sum(v)}/{len(v)}" for t, v in by_task.items()),
            f"approach {sum(int(bool(r['milestones'].get('approached'))) for r in rows)}/{n}, "
            f"touched {sum(int(bool(r['milestones'].get('touched'))) for r in rows)}/{n}, "
            f"lifted {sum(int(bool(r['milestones'].get('lifted'))) for r in rows)}/{n}", ""]
    text = "\n".join(head + lines) + "\n"
    (out / "summary.md").write_text(text)
    (out / "summary.json").write_text(json.dumps(rows, indent=1, default=str))
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--sim-python", default="/home/jli/work/robocasa-inspect-official/.venv/bin/python")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    extra = [x for x in a.rest if x != "--"]
    jobs = [(t, s) for t in a.tasks for s in a.seeds]

    def run(job):
        t, s = job
        d = out / f"{t}-seed{s}"
        if (d / "result.json").exists():
            return f"skip {d.name}"
        t0 = time.time()
        cmd = [a.sim_python, str(HERE / "run_episode.py"), "--task", t, "--seed", str(s), "--out", str(d), "--sim-python", a.sim_python] + extra
        with open(out / f"{d.name}.log", "w") as log:
            p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        summarise(out)
        return f"{d.name} exit={p.returncode} {round((time.time() - t0) / 60, 1)} min"

    with ThreadPoolExecutor(max_workers=max(1, a.parallel)) as ex:
        for line in ex.map(run, jobs):
            print(line, flush=True)
    print(summarise(out))


if __name__ == "__main__":
    main()
