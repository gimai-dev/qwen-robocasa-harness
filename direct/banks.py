"""Build H6 (failure experience) and H7 (successful skills) banks from clean development runs.

    python -m direct.banks --runs /home/jli/state/qwen-direct/matrix/phaseB-clean --out /home/jli/state/qwen-direct/banks/dev

Only records from clean development runs are used. Complete-task success is
kept separate from local success (a grasp that lifted the object). Failure
records keep the context, the action, its measured effect and a correction
only when a later action in the same run verifiably fixed the situation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(run: Path):
    result = json.loads((run / "result.json").read_text())
    decisions = [json.loads(l) for l in (run / "decisions.jsonl").read_text().splitlines() if l.strip()]
    return result, decisions


def _phase(decision):
    receipt = decision.get("receipt") or {}
    width = receipt.get("gripper_width_after_m")
    command = (decision.get("action") or {}).get("g")
    return "holding" if command == 0 and width is not None and width > 0.005 else "free"


def failure_records(result, decisions):
    """Failure experience: actions whose measured effect contradicted an obvious intent."""
    records = []
    for index, d in enumerate(decisions):
        if "slot" in d or d.get("action") is None:
            continue
        action, receipt, status = d["action"], d.get("receipt") or {}, d.get("status")
        record = None
        if status == "unreachable":
            record = {"context": f"TCP at {receipt.get('tcp_world_after_m') or d.get('tcp_after')}", "action": action,
                      "effect": "rejected as unreachable by IK; no motion", "reason": receipt.get("reason"),
                      "target_horizontal_distance_from_base_m": receipt.get("target_horizontal_distance_from_base_m")}
        elif action.get("k") == "base" and receipt.get("base_moved_m") is not None and max(abs(v) for v in receipt["base_moved_m"][:2]) < 0.01 and action.get("a") != "yaw":
            record = {"context": "base command against furniture", "action": action, "effect": f"base moved {receipt['base_moved_m']} (blocked)"}
        elif action.get("g") == 0 and receipt.get("gripper_width_after_m") is not None and receipt["gripper_width_after_m"] < 0.005:
            record = {"context": f"close at TCP {receipt.get('tcp_world_after_m')}", "action": action,
                      "effect": f"gripper closed on nothing (width {receipt['gripper_width_after_m']:.4f} m)"}
        if record is None:
            continue
        # verified correction: the next motion decision of a different kind/target that produced motion or a non-empty grasp
        correction = None
        for later in decisions[index + 1:index + 4]:
            if later.get("steps", 0) > 0 and later.get("action") != action:
                lr = later.get("receipt") or {}
                if record["effect"].startswith("gripper closed on nothing"):
                    if later["action"].get("g") == 0 and (lr.get("gripper_width_after_m") or 0) > 0.005:
                        correction = {"action": later["action"], "effect": f"grasp width {lr['gripper_width_after_m']:.4f} m"}
                        break
                elif later.get("status") in ("completed", "partial"):
                    correction = {"action": later["action"], "effect": f"moved to {lr.get('tcp_world_after_m')}"}
                    break
        record["correction"] = correction
        record["correction_verified"] = correction is not None
        records.append({"task": result["task"], "seed": result["seed"], "instruction": None, "phase": _phase(d),
                        "source_run": Path(result.get("video") or "").parent.name, "record": record})
    return records


def skill_records(result, decisions):
    """Successful skills: local successes (a grasp that lifted an object) with geometry and gripper timing."""
    records = []
    motion = [d for d in decisions if "slot" not in d and d.get("steps", 0) > 0]
    for index, d in enumerate(motion):
        action, receipt = d["action"], d.get("receipt") or {}
        if action.get("g") == 0 and (receipt.get("gripper_width_after_m") or 0) > 0.005:
            # verified lift: a later action raises the TCP by >= 3 cm while the width stays > 0.005
            lifted = None
            for later in motion[index + 1:index + 4]:
                lr = later.get("receipt") or {}
                if lr.get("tcp_world_after_m") and receipt.get("tcp_world_after_m") and (lr["tcp_world_after_m"][2] - receipt["tcp_world_after_m"][2]) > 0.03 and (lr.get("gripper_width_after_m") or 0) > 0.005:
                    lifted = later; break
            if lifted is None:
                continue
            approach = [m["action"] for m in motion[max(0, index - 3):index]]
            records.append({"task": result["task"], "seed": result["seed"], "instruction": None, "phase": "free",
                            "source_run": Path(result.get("video") or "").parent.name,
                            "record": {"skill": "grasp_and_lift", "applicability": "object on a flat surface within reach; downward approach",
                                       "approach_actions": approach, "grasp_action": action,
                                       "grasp_tcp_world_m": receipt.get("tcp_world_after_m"), "grasp_width_m": receipt.get("gripper_width_after_m"),
                                       "lift_action": lifted["action"], "gripper_timing": "close in a hold slot after reaching the grasp pose, then lift",
                                       "complete_task_success": bool(result.get("official_success"))}})
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="matrix directories or run directories of CLEAN development runs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    runs = []
    for root in map(Path, args.runs):
        runs += [p.parent for p in root.rglob("result.json")]
    failures, skills, used = [], [], []
    for run in sorted(set(runs)):
        result, decisions = _load(run)
        if result.get("method") != "clean":
            continue
        used.append(str(run))
        for rec in failure_records(result, decisions) + skill_records(result, decisions):
            rec["instruction"] = json.loads((run / "sim" / "scene.json").read_text())["ep_meta"].get("lang", "") if (run / "sim" / "scene.json").exists() else ""
            (failures if "correction" in rec["record"] else skills).append(rec)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "h6-failures.json").write_text(json.dumps({"source_runs": used, "entries": failures}, indent=1))
    (out / "h7-skills.json").write_text(json.dumps({"source_runs": used, "entries": skills}, indent=1))
    print(json.dumps({"runs": len(used), "failure_records": len(failures), "skill_records": len(skills),
                      "skills_with_complete_success": sum(1 for s in skills if s["record"]["complete_task_success"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
