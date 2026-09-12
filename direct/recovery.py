"""H8 recovery evaluation: inspect clean-run snapshots, select candidate failure
states, and run clean vs H8 continuations from the same state.

    python -m direct.recovery inspect --runs <matrix dir or run dirs> --out <bank dir>
    python -m direct.recovery select  --bank <bank dir> --max-states 10
    python -m direct.recovery select  --bank <bank dir> --qualifications <independent-run/result.json> [...]
    python -m direct.recovery continue --bank <bank dir> --methods clean h8 --out <dir> [--parallel 2]

The evaluated policy never sees the inspection data; it only receives the
restored simulator state, the preceding action/receipt and pre-action images.
Recovery budget: 400 steps, 600 s, 80 decisions (separate from the main score).
Candidates have unknown recoverability until an independent continuation
completes the original task within that budget. Only qualified states enter RSR.
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
from .banks import observation_sequences

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
        preceding_by_sequence = {after: d for d, (before, after) in zip(decisions, observation_sequences(decisions)) if after > before}
        for snapshot in snapshots:
            sequence = int(snapshot.stem)
            obs = sim.send({"kind": "inspect", "path": str(snapshot)})
            info = obs["execution"]
            preceding = preceding_by_sequence.get(sequence)
            rows.append({"run": str(run), "task": result["task"], "seed": result["seed"], "method": result["method"],
                         "sequence": sequence, "snapshot": str(snapshot), "steps_used": json.loads(snapshot.read_text())["total_steps"],
                         "preceding_decision": preceding.get("decision") if preceding else None,
                         "preceding_slot": preceding.get("slot") if preceding else None,
                         "preceding_action": preceding.get("action") if preceding else None,
                         "preceding_status": preceding.get("status") if preceding else None,
                         "gripper_width_m": obs["public_state"]["gripper_width_m"], "gripper_command": obs["gripper_command"],
                         "tcp_world_m": obs["public_state"]["tcp_world_position_m"], "base_world_m": obs["public_state"]["base_world_position_m"],
                         **{k: info.get(k) for k in ("obj_world_m", "gripper_obj_distance_m", "gripper_touching_obj", "official_success", "error")}})
    finally:
        sim.close()
    return rows


def select_states(rows: list[dict], max_states: int) -> list[dict]:
    """Select physical failure candidates; proximity does not prove recoverability."""
    by_run: dict[str, list[dict]] = {}
    for row in rows:
        by_run.setdefault(row["run"], []).append(row)
    selected = []
    for run, seq in by_run.items():
        seq.sort(key=lambda r: r["sequence"])
        held_before = False
        touched_before = False
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
            elif touched_before and not row.get("gripper_touching_obj") and not holding and row["gripper_obj_distance_m"] > 0.08:
                failure = "contact_lost_without_grasp"
            held_before = holding
            touched_before = bool(row.get("gripper_touching_obj")) and not holding
            steps = row.get("steps_used")
            if steps is None:
                steps = row["sequence"] * 20  # old inspection files predate saved step counts
            if failure and horizontal < 0.75 and steps <= 500 and not row.get("official_success"):
                selected.append({**row, "failure_type": failure, "recoverability": "unknown", "qualified_for_rsr": False})
                break  # one state per source episode; split by source episode
    return selected[:max_states]


def qualify_states(states: list[dict], qualification_paths: list[Path]) -> list[dict]:
    """Join separately supplied original-task successes to their exact snapshot.

    These paths designate independent qualification attempts, not the clean/H8
    comparison. Both configured and consumed budgets must meet recovery limits.
    """
    evidence = []
    for path in map(Path, qualification_paths):
        path = path / "result.json" if path.is_dir() else path
        result = json.loads(path.read_text())
        if result.get("official_success") is not True:
            continue
        # No physics advances during teardown/rendering. Timestamp the final
        # state itself; an overrun model call can leave an earlier success intact.
        evidence_wall_s = result.get("final_state_wall_s", result.get("wall_s"))
        if evidence_wall_s is None or not 0 <= evidence_wall_s <= result.get("wall_budget_s", -1) <= RECOVERY_WALL_S:
            continue
        limits = (("steps_budget", "simulator_steps", RECOVERY_STEPS),
                  ("max_decisions", "decisions", RECOVERY_DECISIONS))
        if not all(result.get(configured) is not None and result.get(actual) is not None and
                   0 <= result[actual] <= result[configured] <= limit for configured, actual, limit in limits):
            continue
        evidence.append((path, result))
    qualified = []
    for original in states:
        state = {**original, "recoverability": "unknown", "qualified_for_rsr": False}
        state.pop("qualification", None)
        for path, result in evidence:
            if (result.get("task") == state["task"] and result.get("seed") == state["seed"] and
                    result.get("restore_from") == state["snapshot"]):
                state.update({"recoverability": "verified_task_success_within_budget", "qualified_for_rsr": True,
                              "qualification": {"run": str(path.parent.resolve()), "result": str(path.resolve()),
                                                "simulator_steps": result["simulator_steps"], "wall_s": result["wall_s"],
                                                "decisions": result["decisions"]}})
                break
        qualified.append(state)
    return qualified


def recovery_summary(results: list[dict], methods=()) -> dict:
    table = {method: [] for method in methods}
    for result in results:
        table.setdefault(result["continuation_method"], []).append(result)
    summary = {}
    for method, rows in table.items():
        qualified = [r for r in rows if r.get("qualified_for_rsr")]
        completed = [r for r in qualified if r.get("official_success") is not None]
        candidates = [r for r in rows if not r.get("qualified_for_rsr")]
        recovered = sum(r.get("official_success") is True for r in completed)
        summary[method] = {"recovered": recovered, "states": len(completed),
                           "rsr": recovered / len(completed) if completed else None,
                           "qualified_states": len(qualified), "qualified_incomplete": len(qualified) - len(completed),
                           "candidate_states": len(candidates),
                           "candidate_completions": sum(r.get("official_success") is True for r in candidates),
                           "recoverability": "qualified" if qualified else "unknown"}
    return summary


def run_continuation(state: dict, method: str, out: Path) -> dict:
    name = f"{Path(state['run']).name}-seq{state['sequence']}-{method}"
    run = out / name
    command = [PYTHON, "-m", "direct.episode", "--task", state["task"], "--seed", str(state["seed"]), "--interface", "ee",
               "--mode", "short", "--method", method, "--out", str(run), "--restore-from", state["snapshot"],
               "--history-from", state["run"], "--history-decision", str(state.get("preceding_decision") or 0),
               "--steps-budget", str(RECOVERY_STEPS), "--wall-budget-s", str(RECOVERY_WALL_S), "--max-decisions", str(RECOVERY_DECISIONS)]
    if state.get("preceding_slot") is not None:
        command += ["--history-slot", str(state["preceding_slot"])]
    if not (run / "result.json").exists():
        with (out / f"{name}.log").open("w") as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True, cwd=Path(__file__).resolve().parents[1])
    if (run / "result.json").exists():
        result = json.loads((run / "result.json").read_text())
    else:
        result = {"official_success": None, "termination": "launcher_failure"}
    qualification = state.get("qualification")
    qualified = bool(state.get("qualified_for_rsr") and qualification and Path(qualification["run"]).resolve() != run.resolve())
    result.update({"state": name, "failure_type": state["failure_type"], "continuation_method": method,
                   "qualified_for_rsr": qualified, "qualification": qualification if qualified else None,
                   "recoverability": "verified_task_success_within_budget" if qualified else "unknown"})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect"); p.add_argument("--runs", nargs="+", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("select"); p.add_argument("--bank", required=True); p.add_argument("--max-states", type=int, default=10)
    p.add_argument("--qualifications", nargs="*", default=[], help="independent original-task continuation result files")
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
        selected = qualify_states(selected, [Path(p) for p in args.qualifications])
        (bank / "states.json").write_text(json.dumps(selected, indent=1))
        print(json.dumps({"candidates": len(rows), "selected": len(selected), "qualified": sum(s["qualified_for_rsr"] for s in selected),
                          "types": {t: sum(1 for s in selected if s["failure_type"] == t) for t in {s["failure_type"] for s in selected}}}))
        return 0
    states = json.loads((bank / "states.json").read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    evaluation_runs = {(out / f"{Path(s['run']).name}-seq{s['sequence']}-{m}").resolve() for s in states for m in args.methods}
    for state in states:
        if state.get("qualification") and Path(state["qualification"]["run"]).resolve() in evaluation_runs:
            state.update({"qualified_for_rsr": False, "recoverability": "unknown", "qualification": None})
    jobs = list(itertools.product(states, args.methods))
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for result in pool.map(lambda job: run_continuation(job[0], job[1], out), jobs):
            results.append(result)
            (out / "continuations.json").write_text(json.dumps(results, indent=1))
            print(json.dumps({k: result.get(k) for k in ("state", "continuation_method", "official_success", "termination", "simulator_steps")}), flush=True)
    (out / "rsr.json").write_text(json.dumps(recovery_summary(results, args.methods), indent=1))
    print((out / "rsr.json").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
