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
import math
from pathlib import Path


def observation_sequences(decisions):
    """Map log rows to pre/post observations, including ready-pose and full slots.

    Historical runs advance one observation per executed slot, never per Qwen
    decision: rejected actions do not advance it. New logs record both ends.
    """
    sequence = 0
    pairs = []
    for decision in decisions:
        before = decision.get("observation_sequence", sequence)
        sequence = decision.get("observation_sequence_after", before + int(decision.get("steps", 0) > 0))
        pairs.append((before, sequence))
    return pairs


def _load(run: Path):
    result = json.loads((run / "result.json").read_text())
    decisions = [json.loads(l) for l in (run / "decisions.jsonl").read_text().splitlines() if l.strip()]
    calls_path = run / "qwen-calls.jsonl"
    contexts = {}
    if calls_path.exists():
        for line in calls_path.read_text().splitlines():
            if not line.strip():
                continue
            call = json.loads(line)
            if call.get("category") == "control":
                contexts[call["decision"]] = json.loads(call["user_text"])
    for d in decisions:
        # A full trajectory's initial call is not the prestate of every slot.
        if "slot" not in d or d["slot"] == 1:
            context = contexts.get(d["decision"], {})
            if "state_before" not in d and context.get("state"):
                d["state_before"] = context["state"]
            if context.get("previous"):
                d["previous"] = context["previous"]
    return result, decisions


def _phase(decision):
    state = decision.get("state_before") or {}
    width = state.get("gripper_width_m")
    command = state.get("gripper_command")
    return "holding" if command == 0 and width is not None and width > 0.005 else "free"


def _round_evidence(value):
    """Use receipt precision for the public numbers copied into H6 context."""
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list):
        return [_round_evidence(item) for item in value]
    if isinstance(value, dict):
        return {key: _round_evidence(item) for key, item in value.items()}
    return value


def failure_records(result, decisions):
    """Failure experience: actions whose measured effect contradicted an obvious intent."""
    records = []
    known_state = {}
    for index, d in enumerate(decisions):
        prestate = d.get("state_before") or known_state.copy()
        receipt = d.get("receipt") or {}
        known_state = dict(prestate)
        for after, before in (("tcp_world_after_m", "tcp_world_position_m"),
                              ("base_world_after_m", "base_world_position_m"),
                              ("gripper_width_after_m", "gripper_width_m")):
            if receipt.get(after) is not None:
                known_state[before] = receipt[after]
        if d.get("steps", 0) > 0 and (d.get("action") or {}).get("g") is not None:
            known_state["gripper_command"] = d["action"]["g"]
        if d.get("action") is None:
            continue
        action, receipt, status = d["action"], d.get("receipt") or {}, d.get("status")
        tcp = prestate.get("tcp_world_position_m", prestate.get("tcp_world_m"))
        context = f"TCP at {tcp}" if tcp is not None else "pre-action TCP unavailable"
        record = None
        if status == "unreachable":
            record = {"context": context, "action": action, "failure_type": "unreachable",
                      "effect": "rejected as unreachable by IK; no motion", "reason": receipt.get("reason"),
                      "target_horizontal_distance_from_base_m": receipt.get("target_horizontal_distance_from_base_m")}
        elif action.get("k") == "base" and receipt.get("base_moved_m") is not None and max(abs(v) for v in receipt["base_moved_m"][:2]) < 0.01 and action.get("a") != "yaw":
            record = {"context": context, "failure_type": "base_blocked", "action": action, "effect": f"base moved {receipt['base_moved_m']} (blocked)"}
        elif action.get("g") == 0 and receipt.get("gripper_width_after_m") is not None and receipt["gripper_width_after_m"] < 0.005:
            record = {"context": context, "failure_type": "empty_close", "action": action,
                      "effect": f"gripper closed on nothing (width {receipt['gripper_width_after_m']:.4f} m)"}
        if record is None:
            continue
        # Keep retrieved context within the existing H6 token budget; detailed
        # robot diagnostics remain in the episode log.
        context_fields = ("tcp_world_position_m", "tcp_world_m", "tcp_world_quat_xyzw", "tcp_quat_xyzw", "arm_q_rad",
                          "base_world_position_m", "base_world_m", "base_world_yaw_rad", "base_yaw_rad",
                          "gripper_command", "gripper_width_m")
        previous = d.get("previous")
        if previous:
            previous = {"action": previous.get("action"), "result": {
                k: v for k, v in (previous.get("result") or {}).items()
                if k in ("status", "reason", "tcp_world_after_m", "gripper_width_after_m")}}
        record.update({"state_before": {k: v for k, v in prestate.items() if k in context_fields},
                       "observation_sequence": d.get("observation_sequence"), "previous": previous, "decision": d["decision"]})
        # verified correction: only an outcome that fixes THIS failure type counts; anything else stays a hypothesis
        correction = None
        executed = []
        for later in decisions[index + 1:index + 4]:
            if later.get("steps", 0) <= 0 or not later.get("action"):
                continue
            lr = later.get("receipt") or {}
            la = later["action"]
            if status == "unreachable":
                executed.append({"action": {k: v for k, v in la.items() if k != "n"},
                                 "effect": {"status": later.get("status"), "steps": later["steps"],
                                            **{k: v for k, v in lr.items() if k in (
                                                "tcp_world_after_m", "arm_q_after", "base_moved_m", "base_world_after_m",
                                                "base_yaw_after_rad", "gripper_width_after_m")}}})
                if la.get("k") == "ee" and later.get("status") == "completed" and all(la.get(k) == action.get(k) for k in ("p", "o")):
                    # The retry succeeded from the state reached by this entire
                    # executed sequence, not necessarily from the failure state.
                    correction = {"sequence": executed}
                    break
            elif record["failure_type"] == "base_blocked":
                moved = lr.get("base_moved_m")
                if la.get("k") == "base" and la.get("a") != action.get("a") and moved and max(abs(v) for v in moved[:2]) >= 0.05:
                    correction = {"action": la, "effect": f"base moved {moved}"}
                    break
            # A wider finger gap alone does not verify a repaired grasp.
        record["correction"] = correction
        record["correction_verified"] = correction is not None
        if status == "unreachable" and correction is not None:
            # Preserve the enabling motions under H6's existing retrieval cap.
            # The failure prestate and action already identify the failed target;
            # previous-turn prose, action notes and derived distance duplicate it.
            record["action"] = {k: v for k, v in action.items() if k != "n"}
            for key in ("previous", "reason", "target_horizontal_distance_from_base_m"):
                record.pop(key, None)
            record = _round_evidence(record)
        records.append({"task": result["task"], "seed": result["seed"], "instruction": None, "phase": _phase({"state_before": prestate}),
                        "source_run": Path(result.get("video") or "").parent.name, "record": record})
    return records


def skill_records(result, decisions, inspection=None):
    """Label retained object-following lifts using offline evaluator rows only.

    Inspection must contain this run's rows; join by actual observation sequence,
    not its historical ``preceding_decision`` annotation. Truth qualifies the
    label but is never copied into the policy-visible skill record.
    """
    if not inspection:
        return []
    records = []
    rows = {r["sequence"]: r for r in inspection}
    motion = [(d, rows.get(after)) for d, (_, after) in zip(decisions, observation_sequences(decisions))
              if d.get("action") and d.get("steps", 0) > 0]

    def holding(row):
        return (row is not None and row.get("gripper_command") == 0 and
                (row.get("gripper_width_m") or 0) > 0.005 and bool(row.get("gripper_touching_obj")) and
                row.get("obj_world_m") is not None and row.get("tcp_world_m") is not None)

    for index, (d, grasp) in enumerate(motion):
        action, receipt = d["action"], d.get("receipt") or {}
        if action.get("g") == 0 and holding(grasp):
            lifted = None
            for later, observed in motion[index + 1:index + 4]:
                if later["action"].get("g") == 1 or not holding(observed):
                    break
                obj_delta = [b - a for a, b in zip(grasp["obj_world_m"], observed["obj_world_m"])]
                tcp_delta = [b - a for a, b in zip(grasp["tcp_world_m"], observed["tcp_world_m"])]
                if obj_delta[2] >= 0.03 and tcp_delta[2] >= 0.03 and math.dist(obj_delta, tcp_delta) <= 0.03:
                    lifted = later; break
            if lifted is None:
                continue
            approach = [m["action"] for m, _ in motion[max(0, index - 3):index]]
            records.append({"task": result["task"], "seed": result["seed"], "instruction": None, "phase": "free",
                            "source_run": Path(result.get("video") or "").parent.name,
                            "record": {"skill": "grasp_and_lift", "applicability": "a close followed by a lift with retained object contact",
                                       "approach_actions": approach, "grasp_action": action,
                                       "grasp_tcp_world_m": receipt.get("tcp_world_after_m"), "grasp_width_m": receipt.get("gripper_width_after_m"),
                                       "lift_action": lifted["action"], "gripper_timing": "close at the grasp pose and retain the closed command through the lift",
                                       "complete_task_success": bool(result.get("official_success"))}})
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="matrix directories or run directories of CLEAN development runs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--inspection", help="offline direct.recovery inspection.json; required to qualify H7 skills")
    args = parser.parse_args()
    runs = []
    for root in map(Path, args.runs):
        runs += [p.parent for p in root.rglob("result.json")]
    failures, skills, used = [], [], []
    inspection_by_run = {}
    if args.inspection:
        for row in json.loads(Path(args.inspection).read_text()):
            inspection_by_run.setdefault(str(Path(row["run"]).resolve()), []).append(row)
    for run in sorted(set(runs)):
        result, decisions = _load(run)
        if result.get("method") != "clean":
            continue
        used.append(str(run))
        for rec in failure_records(result, decisions) + skill_records(result, decisions, inspection_by_run.get(str(run.resolve()))):
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
