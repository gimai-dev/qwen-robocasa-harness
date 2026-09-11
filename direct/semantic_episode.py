"""Show-Harness-style episode: one token per decision, interpreter supplies magnitudes.

    python -m direct.semantic_episode --task PickPlaceCounterToSink --seed 0 --method sem-full --out DIR
Methods: sem (tokens + proprioception + recent moves), sem+plan (+ subtask planner with
completion checks), sem+plan+rec (+ grasp recovery), sem-full (+ agent-view selection).
Perception is RGB only: SAM regions are never shown to the model in this family.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
import traceback
from pathlib import Path

from .actions import SLOT_STEPS
from .episode import DEFAULT_SCENES, MAX_CONSECUTIVE_NO_MOTION, git_revision, render_video
from .executor import Simulator, execute, move_to_ready
from .observation import read_images
from .policy import MalformedOutput, QwenDirectClient
from .semantic import Interpreter, proprio_text
from .semantic_plugins import CHECK_SCHEMA, RecentMoves, Recovery, SubgoalPlanner
from .semantic_policy import SemanticPolicy, build_controller_prompt, image_pair

PROMPTS = Path(__file__).resolve().with_name("prompts")
MOVE_SLOT_STEPS = 6          # 2-4 cm at 1 cm/step plus settle; gripper/base keep the full 20-step slot
SYSTEM_PROMPT = "You are a careful robot controller. Follow the interface exactly."


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--method", default="sem-full", choices=["sem", "sem+plan", "sem+plan+rec", "sem-full"])
    p.add_argument("--out", required=True)
    p.add_argument("--scenes", default=str(DEFAULT_SCENES))
    p.add_argument("--agent-view", default="left", choices=["left", "right"])
    p.add_argument("--steps-budget", type=int, default=900)
    p.add_argument("--wall-budget-s", type=float, default=1200.0)
    p.add_argument("--max-decisions", type=int, default=180)
    a = p.parse_args(argv)
    run = Path(a.out).resolve()
    run.mkdir(parents=True, exist_ok=False)
    use_plan = "plan" in a.method or a.method == "sem-full"
    use_rec = "rec" in a.method or a.method == "sem-full"
    view_select = a.method == "sem-full"
    template = (PROMPTS / "semantic_controller.txt").read_text()
    config = {"task": a.task, "seed": a.seed, "interface": "semantic", "mode": "short", "method": a.method,
              "steps_budget": a.steps_budget, "wall_budget_s": a.wall_budget_s, "max_decisions": a.max_decisions,
              "move_slot_steps": MOVE_SLOT_STEPS, "code_revision": git_revision(), "ready_pose": True, "agent_view": a.agent_view}
    (run / "config.json").write_text(json.dumps(config, indent=1))
    (run / "system-prompt.txt").write_text(SYSTEM_PROMPT + "\n\n" + template)
    decisions: list[dict] = []
    termination, error_text, outcome = "incomplete", None, {}
    tokens: collections.Counter = collections.Counter()
    previous_receipt = None
    started = time.monotonic()
    interp, recent = Interpreter(), RecentMoves()
    recovery = Recovery() if use_rec else None
    planner = None
    client = None
    sim = None

    def log(record: dict) -> None:
        decisions.append(record)
        with (run / "decisions.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    try:
        client = QwenDirectClient(log_path=run / "qwen-calls.jsonl")
        policy = SemanticPolicy(client)
        sim = Simulator(task=a.task, seed=a.seed, run=run, scenes=Path(a.scenes), action_budget=a.steps_budget, wall_budget_s=a.wall_budget_s)
        sim.launch()
        started = time.monotonic()
        ready = move_to_ready(sim)
        log({"decision": 0, "status": "ready_pose", "steps": sim.steps_used(), "receipt": ready.summary()})
        images = read_images(sim.observation)
        if use_plan:
            planner = SubgoalPlanner(client, run)
            planner.plan(sim.observation["instruction"], image_pair(images, a.agent_view), SYSTEM_PROMPT)
        decision, no_motion, forced = 0, 0, None
        while True:
            if decision >= a.max_decisions:
                termination = "decision_budget"; break
            if sim.steps_left() <= 0:
                termination = "step_budget"; break
            if time.monotonic() - started > a.wall_budget_s:
                termination = "wall_budget"; break
            decision += 1
            obs = sim.observation
            state = obs["public_state"]
            images = read_images(obs)
            agent_view = a.agent_view
            if view_select and not state["tcp_pixels"]["left"]["visible"]:
                agent_view = "right"
            proprio = proprio_text(obs, previous_receipt)
            fine = planner.fine() if planner else state["tcp_world_position_m"][2] < 1.15
            subtask = planner.current() if planner else None
            subtask_text = f"{subtask['name']} (done when: {subtask['done_when']})" if subtask else ""
            text = build_controller_prompt(template, task=obs["instruction"], subtask=subtask_text, proprio=proprio,
                                           recent_moves=recent.text(), fine=fine)
            record: dict = {"decision": decision, "steps_used": obs["steps_used"], "subtask": subtask["name"] if subtask else None,
                            "fine": fine, "agent_view": agent_view}
            if forced:
                token = forced
                record["forced"] = True
                forced = None
            else:
                try:
                    if planner:
                        out = policy.choose(system_prompt=SYSTEM_PROMPT, images=image_pair(images, agent_view), decision=decision,
                                            user_text=text + "\nAlso report whether the current subtask's done_when criterion is satisfied (subtask_done).",
                                            schema=CHECK_SCHEMA)
                        if out.get("subtask_done"):
                            planner.advance()
                            record["subtask_done"] = True
                    else:
                        out = policy.choose(system_prompt=SYSTEM_PROMPT, user_text=text, images=image_pair(images, agent_view), decision=decision)
                    token = out["token"]
                except MalformedOutput as error:
                    record.update({"status": "malformed_output", "error": str(error)})
                    log(record)
                    no_motion += 1
                    if no_motion >= MAX_CONSECUTIVE_NO_MOTION:
                        termination = "no_progress"; break
                    continue
            tokens[token] += 1
            record["token"] = token
            action = interp.to_action(token, state, fine=fine)
            if action is None:
                record["status"] = "done"
                log(record)
                termination = "stop"; break
            steps = MOVE_SLOT_STEPS if action.kind == "ee" else SLOT_STEPS
            receipt = execute(sim, action, slot_steps=steps)
            record.update({"status": receipt.status, "steps": receipt.steps, "receipt": receipt.summary()})
            log(record)
            recent.push(token, receipt.status)
            previous_receipt = receipt.summary()
            no_motion = 0 if receipt.steps > 0 else no_motion + 1
            if recovery is not None and receipt.child is not None:
                forced, rollback = recovery.check(token, receipt.child, proprio_text(sim.observation, None))
                if rollback and planner:
                    planner.rollback_to_grasp()
            if no_motion >= MAX_CONSECUTIVE_NO_MOTION:
                termination = "no_progress"; break
        outcome = sim.finish()
    except Exception:
        error_text = traceback.format_exc()
        termination = "infrastructure_error"
        try:
            outcome = sim.close() if sim else {}
        except Exception:
            pass
    finally:
        try:
            if client:
                client.close()
        except Exception:
            pass
    frames = 0
    try:
        frames = render_video(run / "sim", run / "episode.mp4")
    except Exception as error:
        error_text = (error_text or "") + f"\nvideo: {error!r}"
    motion = [d for d in decisions if d.get("steps", 0) > 0 and d["decision"] > 0]
    result = {**config, "official_success": outcome.get("official_success"), "termination": termination,
              "simulator_steps": outcome.get("simulator_steps"), "decisions": len([d for d in decisions if d["decision"] > 0]),
              "slots_executed": len(motion),
              "rejected_actions": sum(1 for d in decisions if d["decision"] > 0 and d.get("steps") == 0 and d.get("status") not in ("done",)),
              "wall_s": round(time.monotonic() - started, 1), **(client.totals() if client else {}),
              "sam_calls": 0, "sam_elapsed_s": 0.0,
              "video": str(run / "episode.mp4") if frames else None, "video_frames": frames,
              "scene": str(Path(a.scenes) / f"{a.task}-seed{a.seed}.json"), "simulator_terminal": outcome,
              "semantic": {"tokens": dict(tokens), "recoveries": recovery.events if recovery else 0,
                           "subtasks": planner.subtasks if planner else None, "subtask_index": planner.index if planner else None},
              "method_summary": {}, "error": error_text}
    (run / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("task", "seed", "method", "official_success", "termination", "simulator_steps", "decisions", "wall_s", "qwen_calls")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
