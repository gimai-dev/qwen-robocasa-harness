#!/usr/bin/env python3
"""One agent-as-policy episode on RoboCasa: session dir + server + Qwen agent loop + result.

    python run_episode.py --task PickPlaceCounterToSink --seed 0 --out /home/jli/state/agp-sim/runs/<name>

Layout of <out>/:
  session/   what the agent sees: PROMPT.md, README_interface.md, robot_client.py, goal/, scratch/, frames/, bridge/, server.log
  agent_events.jsonl, agent_usage.json, agent_messages.json, transcript.md   (harness files, outside the session)
  sim/       the simulator child's run dir (mailbox, snapshots, video-frames, render) — never shown to the agent
  evaluator.jsonl   ground truth after every motion command (object pose, distance, contact, success)
  server_result.json, result.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SIM_PY = "/home/jli/work/robocasa-inspect-official/.venv/bin/python"

sys.path.insert(0, str(HERE))
from agent_loop import Agent  # noqa: E402


def agent_verdict(result_md: Path) -> str:
    if not result_md.exists():
        return "no RESULT.md"
    txt = result_md.read_text(errors="replace")
    judge = (r"(?i)\b(?:(?:i|we)\s+(?:judge|assess|conclude|consider|deem)"
             r"|(?:my|our)\s+(?:honest\s+)?(?:judge?ment|assessment|conclusion|verdict))\b[^.\n]{0,160}")
    if re.search(judge + r"\b(?:unsuccessful|not\s+success|fail(?:ed|ure)?|incomplete)", txt):
        return "fail (agent)"
    if re.search(judge + r"\bsuccess", txt):
        return "success (agent)"
    if re.search(r"(?im)^\W*(result\W*)?(unsuccessful|incomplete|not (fully )?(complete|done)|fail(ed|ure))", txt):
        return "fail (agent)"
    if re.search(r"(?im)^\W*(result\W*)?(success|.*\b(completed|done)\b)", txt):
        return "success (agent)"
    return "unclear (agent)"


def milestones(evaluator_path: Path) -> dict:
    rows = []
    if evaluator_path.exists():
        for line in evaluator_path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    out = {"n_rows": len(rows), "min_gripper_obj_distance_m": None, "touched": False, "lifted": False,
           "success_seen": False, "obj_displaced_m": None}
    z0, p0 = None, None
    for r in rows:
        d = r.get("gripper_obj_distance_m")
        if d is not None:
            out["min_gripper_obj_distance_m"] = d if out["min_gripper_obj_distance_m"] is None else min(out["min_gripper_obj_distance_m"], d)
        if r.get("gripper_touching_obj"):
            out["touched"] = True
        obj = r.get("obj_world_m")
        if obj is not None:
            if p0 is None:
                p0, z0 = obj, obj[2]
            if obj[2] - z0 > 0.03 and (r.get("gripper_touching_obj") or (d is not None and d < 0.06)):
                out["lifted"] = True
            out["obj_displaced_m"] = round(sum((a - b) ** 2 for a, b in zip(obj, p0)) ** 0.5, 3)
        if r.get("official_success"):
            out["success_seen"] = True
    if out["min_gripper_obj_distance_m"] is not None:
        out["min_gripper_obj_distance_m"] = round(out["min_gripper_obj_distance_m"], 3)
        out["approached"] = out["min_gripper_obj_distance_m"] < 0.10
    return out


def transcript(session: Path) -> None:
    ev = session.parent / "agent_events.jsonl"
    if not ev.exists():
        return
    lines = ["# Agent transcript", ""]
    for line in ev.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = e.get("type")
        if t == "response":
            if e.get("content"):
                lines.append(f"**assistant** ({e.get('prompt_tokens')} in / {e.get('completion_tokens')} out, {e.get('seconds')}s):\n\n{e['content']}\n")
            for tc in e.get("tool_calls") or []:
                lines.append(f"→ `{tc['name']}` {tc['arguments']}\n")
        elif t == "exec":
            lines.append(f"```\n$ {e.get('command')}\n{e.get('output')}\n```\n")
        elif t == "view_image":
            lines.append(f"*viewed {e.get('path')}*\n")
        elif t in ("compact", "http_error", "request_error", "done"):
            lines.append(f"_{t}: {json.dumps({k: v for k, v in e.items() if k not in ('t', 'type')})}_\n")
    (session.parent / "transcript.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default="/home/jli/state/agp-sim/scenes")
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--wall-min", type=float, default=45)
    ap.add_argument("--max-turns", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-images", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--prompt", default=str(HERE / "PROMPT_robocasa.md"))
    ap.add_argument("--readme", default=str(HERE / "README_interface_sim.md"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--token-file", default="/home/jli/state/panda-qwen38/api-token")
    ap.add_argument("--sim-python", default=SIM_PY)
    ap.add_argument("--interface", default="full", help="server interface variant: full | joint | path | macro")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--reset-after", type=int, default=3, help="loop events before a context reset (v12); 0 disables")
    a = ap.parse_args()

    out = Path(a.out).resolve()
    if (out / "result.json").exists():
        print(json.dumps({"skipped": True, "result": json.loads((out / "result.json").read_text())}))
        return
    if out.exists():
        shutil.rmtree(out)
    session, sim = out / "session", out / "sim"
    for d in (session / "bridge", session / "frames", session / "scratch", session / "goal"):
        d.mkdir(parents=True)
    shutil.copy(HERE / "robot_client.py", session / "robot_client.py")
    t_start = time.time()

    # 1. server (launches the simulator child, moves to ready, writes server_boot.json)
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT / 'runtime'}", PYTHONDONTWRITEBYTECODE="1")
    boot_log = open(out / "server_boot.log", "w")
    server = subprocess.Popen([a.sim_python, str(HERE / "server_sim.py"), "--session", str(session), "--run", str(sim),
                               "--task", a.task, "--seed", str(a.seed), "--scenes", a.scenes, "--budget", str(a.budget),
                               "--image-size", str(a.image_size), "--interface", a.interface],
                              stdout=boot_log, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
    boot = session / "server_boot.json"
    deadline = time.time() + 900
    while not boot.exists():
        if server.poll() is not None:
            raise SystemExit(f"server died during boot; see {out / 'server_boot.log'}")
        if time.time() > deadline:
            server.kill()
            raise SystemExit("server boot timed out")
        time.sleep(1)
    info = json.loads(boot.read_text())
    instruction = info.get("instruction") or a.task
    rp = info["ready_pose"]
    ready_txt = (f'position ({rp["position"]["x"]:.3f}, {rp["position"]["y"]:.3f}, {rp["position"]["z"]:.3f}), '
                 f'rotation {{"w":{rp["rotation"]["w"]:.3f},"x":{rp["rotation"]["x"]:.3f},"y":{rp["rotation"]["y"]:.3f},"z":{rp["rotation"]["z"]:.3f}}}')

    # 2. what the agent sees
    readme = Path(a.readme).read_text().replace("<READY_POSE>", ready_txt).replace("<MAX_OPEN>", f"{info['max_opening_m']:.4f}") \
        .replace("<IMGSIZE>", f"{info['image_size'][0]}×{info['image_size'][1]}") \
        .replace("<READY_JOINTS>", json.dumps(info.get("ready_joints", [])))
    (session / "README_interface.md").write_text(readme)
    prompt = Path(a.prompt).read_text().replace("<INSTRUCTION>", instruction).replace("<BUDGET>", str(a.budget)) \
        .replace("<MINUTES>", str(int(a.wall_min)))
    (session / "PROMPT.md").write_text(prompt)
    (session / "goal" / "INSTRUCTION.md").write_text(instruction + "\n")
    (out / "episode.json").write_text(json.dumps({"task": a.task, "seed": a.seed, "instruction": instruction, "budget": a.budget,
                                                  "wall_min": a.wall_min, "max_turns": a.max_turns, "temperature": a.temperature,
                                                  "image_size": a.image_size, "max_images": a.max_images, "interface": a.interface, "thinking": a.thinking, "reset_after": a.reset_after,
                                                  "prompt_file": a.prompt, "readme_file": a.readme, "started": t_start}, indent=1))

    # 3. the agent
    token = Path(a.token_file).read_text().strip()
    agent = Agent(session, a.base_url, token, None, max_turns=a.max_turns, wall_s=a.wall_min * 60,
                  temperature=a.temperature, max_images=a.max_images, python_bin=a.sim_python, log_dir=out, thinking=a.thinking)
    agent.reset_after = a.reset_after
    import urllib.request
    req = urllib.request.Request(a.base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        agent.model = json.loads(r.read())["data"][0]["id"]
    t_agent = time.time()
    try:
        outcome = agent.run(prompt)
    except Exception as e:  # noqa: BLE001
        outcome = f"agent_error: {e!r}"
    agent_s = time.time() - t_agent
    transcript(session)

    # 4. stop the server -> sealed official outcome
    (session / "SERVER_STOP").touch()
    try:
        server.wait(timeout=300)
    except subprocess.TimeoutExpired:
        server.kill()
    sr = json.loads((out / "server_result.json").read_text()) if (out / "server_result.json").exists() else {}
    log = (session / "server.log").read_text(errors="replace") if (session / "server.log").exists() else ""
    used = re.findall(r"used=(\d+)", log)
    result = {
        "task": a.task, "seed": a.seed, "instruction": instruction,
        "official_success": bool(sr.get("official_success", False)),
        "agent_outcome": outcome, "agent_verdict": agent_verdict(session / "scratch" / "RESULT.md"),
        "milestones": milestones(out / "evaluator.jsonl"),
        "cmds_counted": int(used[-1]) if used else 0,
        "moves_ok": len(re.findall(r"#\d+ (move_ee|move_delta|move_joints|home|move_base) ok=True", log)),
        "failed_cmds": len(re.findall(r"ok=False", log)),
        "frames": len(re.findall(r"#\d+ frames ok=True", log)),
        "gripper_cmds": len(re.findall(r"#\d+ gripper ok=True", log)),
        "usage": agent.usage, "agent_s": round(agent_s, 1), "wall_s": round(time.time() - t_start, 1),
        "terminal": sr.get("terminal"),
    }
    (out / "result.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({k: result[k] for k in ("task", "seed", "official_success", "agent_verdict", "milestones", "cmds_counted", "wall_s")}, default=str))


if __name__ == "__main__":
    main()
