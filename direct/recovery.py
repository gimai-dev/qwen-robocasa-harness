"""H8 recovery evaluation: inspect clean-run snapshots, select recoverable failure
states, and run clean vs H8 continuations from the same state.

    python -m direct.recovery inspect --runs <matrix dir or run dirs> --out <bank dir>
    python -m direct.recovery select  --bank <bank dir> --max-states 10
    python -m direct.recovery continue --bank <bank dir> --methods clean h8 --out <dir> [--parallel 2]

The evaluated policy never sees the inspection data; it only receives the
restored simulator state, the preceding action/receipt and pre-action images.
Recovery budget: 400 steps, 600 s, 80 decisions (separate from the main score).
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .executor import Simulator

PYTHON = "/home/jli/work/robocasa-inspect-official/.venv/bin/python"
RECOVERY_STEPS, RECOVERY_WALL_S, RECOVERY_DECISIONS = 400, 600.0, 80


def inspect_run(run: Path, out: Path) -> list[dict]:
    result = json.loads((run / "result.json").read_text())
    decisions = [json.loads(l) for l in (run / "decisions.jsonl").read_text().splitlines() if l.strip()]
    snapshots = sorted((run / "sim" / "snapshots").glob("*.json"))
    if not snapshots:
        return []
    work = out / "inspect" / run.name
    sim = Simulator(task=result["task"], seed=result["seed"], run=work, scenes=Path(result["scene"]).parent,
                    action_budget=900, wall_budget_s=1200)
    sim.launch()
    rows = []
    try:
        # observation sequence k follows motion decision k (sequence 0 = initial)
        motion = [d for d in decisions if "slot" not in d and d.get("steps", 0) > 0]
        for index, snapshot in enumerate(snapshots):
            obs = sim.send({"kind": "inspect", "path": str(snapshot)})
            info = obs["execution"]
            preceding = motion[index - 1] if 0 < index <= len(motion) else None
            rows.append({"run": str(run), "task": result["task"], "seed": result["seed"], "method": result["method"],
                         "sequence": index, "snapshot": str(snapshot), "steps_used": obs["steps_used"] if False else None,
                         "preceding_decision": preceding.get("decision") if preceding else None,
                         "preceding_action": preceding.get("action") if preceding else None,
                         "preceding_status": preceding.get("status") if preceding else None,
                         "gripper_width_m": obs["public_state"]["gripper_width_m"], "gripper_command": obs["gripper_command"],
                         "tcp_world_m": obs["public_state"]["tcp_world_position_m"], "base_world_m": obs["public_state"]["base_world_position_m"],
                         **{k: info.get(k) for k in ("obj_world_m", "gripper_obj_distance_m", "gripper_touching_obj", "official_success", "error")}})
    finally:
        sim.close()
    return rows


def select_states(rows: list[dict], max_states: int) -> list[dict]:
    """Recoverable failure: a physical failure event (empty close near the object, or the object lost
    after being held) with the object still within reach and the episode not yet exhausted."""
    by_run: dict[str, list[dict]] = {}
    for row in rows:
        by_run.setdefault(row["run"], []).append(row)
    selected = []
    for run, seq in by_run.items():
        seq.sort(key=lambda r: r["sequence"])
        held_before = False
        for row in seq:
            if row.get("obj_world_m") is None:
                continue
            width = row.get("gripper_width_m") or 0.0
            closed = int(round(float(row["gripper_command"]))) == 0
            holding = closed and width > 0.005 and bool(row.get("gripper_touching_obj"))
            import math
            horizontal = math.dist(row["obj_world_m"][:2], row["base_world_m"][:2])
            failure = None
            if closed and width < 0.005 and row["gripper_obj_distance_m"] < 0.25:
                failure = "empty_close_near_object"
            elif held_before and not holding and row["gripper_obj_distance_m"] > 0.05:
                failure = "object_lost_after_grasp"
            elif row.get("preceding_status") == "unreachable" and False:
                failure = None
            held_before = holding
            if failure and horizontal < 0.75 and row["sequence"] * 20 <= 500 and not row.get("official_success"):
                selected.append({**row, "failure_type": failure})
                break  # one state per source episode; split by source episode
    return selected[:max_states]


def run_continuation(state: dict, method: str, out: Path) -> dict:
    name = f"{Path(state['run']).name}-seq{state['sequence']}-{method}"
    run = out / name
    if (run / "result.json").exists():
        return json.loads((run / "result.json").read_text())
    command = [PYTHON, "-m", "direct.episode", "--task", state["task"], "--seed", str(state["seed"]), "--interface", "ee",
               "--mode", "short", "--method", method, "--out", str(run), "--restore-from", state["snapshot"],
               "--history-from", state["run"], "--history-decision", str(state.get("preceding_decision") or 0),
               "--steps-budget", str(RECOVERY_STEPS), "--wall-budget-s", str(RECOVERY_WALL_S), "--max-decisions", str(RECOVERY_DECISIONS)]
    with (out / f"{name}.log").open("w") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True, cwd=Path(__file__).resolve().parents[1])
    if (run / "result.json").exists():
        result = json.loads((run / "result.json").read_text())
    else:
        result = {"official_success": None, "termination": "launcher_failure"}
    result.update({"state": name, "failure_type": state["failure_type"], "continuation_method": method})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect"); p.add_argument("--runs", nargs="+", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("select"); p.add_argument("--bank", required=True); p.add_argument("--max-states", type=int, default=10)
    p = sub.add_parser("continue"); p.add_argument("--bank", required=True); p.add_argument("--methods", nargs="+", default=["clean", "h8"])
    p.add_argument("--out", required=True); p.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()
    if args.command == "inspect":
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        runs = sorted({p.parent for root in map(Path, args.runs) for p in root.rglob("result.json")})
        rows = []
        for run in runs:
            if json.loads((run / "result.json").read_text()).get("method") != "clean":
                continue
            rows += inspect_run(run, out)
            (out / "inspection.json").write_text(json.dumps(rows, indent=1))
            print(json.dumps({"run": run.name, "snapshots": len([r for r in rows if r["run"] == str(run)])}), flush=True)
        return 0
    bank = Path(args.bank)
    if args.command == "select":
        rows = json.loads((bank / "inspection.json").read_text())
        selected = select_states(rows, args.max_states)
        (bank / "states.json").write_text(json.dumps(selected, indent=1))
        print(json.dumps({"candidates": len(rows), "selected": len(selected), "types": {t: sum(1 for s in selected if s["failure_type"] == t) for t in {s["failure_type"] for s in selected}}}))
        return 0
    states = json.loads((bank / "states.json").read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    jobs = list(itertools.product(states, args.methods))
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for result in pool.map(lambda job: run_continuation(job[0], job[1], out), jobs):
            results.append(result)
            (out / "continuations.json").write_text(json.dumps(results, indent=1))
            print(json.dumps({k: result.get(k) for k in ("state", "continuation_method", "official_success", "termination", "simulator_steps")}), flush=True)
    table = {}
    for r in results:
        table.setdefault(r["continuation_method"], []).append(bool(r.get("official_success")))
    (out / "rsr.json").write_text(json.dumps({m: {"recovered": sum(v), "states": len(v), "rsr": sum(v) / len(v)} for m, v in table.items()}, indent=1))
    print((out / "rsr.json").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
