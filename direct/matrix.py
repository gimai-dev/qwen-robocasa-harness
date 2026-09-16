"""Run an experiment matrix of episodes as subprocesses and summarize.

    python -m direct.matrix --tasks PickPlaceCounterToSink --seeds 0 1 2 \
        --interfaces ee --modes short --methods clean --out /home/jli/state/qwen-direct/matrix/<name> --parallel 2
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PYTHON = "/home/jli/work/robocasa-inspect-official/.venv/bin/python"


def run_one(spec: dict, out: Path) -> dict:
    name = f"{spec['task']}-s{spec['seed']}-{spec['interface']}-{spec['mode']}-{spec['method']}"
    run = out / name
    if (run / "result.json").exists():
        return json.loads((run / "result.json").read_text())
    module = "direct.semantic_episode" if spec["method"].startswith("sem") else "direct.episode"
    command = [PYTHON, "-m", module, "--task", spec["task"], "--seed", str(spec["seed"]), "--method", spec["method"], "--out", str(run)]
    if module == "direct.episode":
        command += ["--interface", spec["interface"], "--mode", spec["mode"], "--method-config", json.dumps(spec.get("method_config", {}))]
        if spec.get("ready_pose"):
            command.append("--ready-pose")
        if spec.get("scenes"):
            command += ["--scenes", str(spec["scenes"])]
    for key in ("steps_budget", "wall_budget_s", "max_decisions"):
        if key in spec:
            command += [f"--{key.replace('_', '-')}", str(spec[key])]
    started = time.monotonic()
    log = out / f"{name}.log"
    with log.open("w") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True,
                                   cwd=Path(__file__).resolve().parents[1])
    if (run / "result.json").exists():
        return json.loads((run / "result.json").read_text())
    return {**spec, "official_success": None, "termination": "launcher_failure", "returncode": completed.returncode,
            "wall_s": round(time.monotonic() - started, 1), "log": str(log)}


def summarize(results: list[dict], out: Path) -> str:
    rows = []
    for r in results:
        rows.append({k: r.get(k) for k in ("task", "seed", "interface", "mode", "method", "official_success", "termination",
                                          "simulator_steps", "decisions", "rejected_actions", "qwen_calls", "prompt_tokens",
                                          "completion_tokens", "wall_s")})
    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["method"], r["interface"], r["mode"], r["task"]), []).append(r)
    lines = ["| method | interface | mode | task | success/attempts | mean steps | mean calls | mean wall s | terminations |", "|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(groups):
        rs = groups[key]
        succ = sum(1 for r in rs if r["official_success"])
        mean = lambda field: round(sum((r[field] or 0) for r in rs) / len(rs), 1)
        terms = {}
        for r in rs:
            terms[r["termination"]] = terms.get(r["termination"], 0) + 1
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {succ}/{len(rs)} | {mean('simulator_steps')} | {mean('qwen_calls')} | {mean('wall_s')} | {terms} |")
    text = "\n".join(lines)
    (out / "summary.md").write_text(text + "\n")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--interfaces", nargs="+", default=["ee"])
    parser.add_argument("--modes", nargs="+", default=["short"])
    parser.add_argument("--methods", nargs="+", default=["clean"])
    parser.add_argument("--method-config", default="{}")
    parser.add_argument("--out", required=True)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--steps-budget", type=int, default=900)
    parser.add_argument("--wall-budget-s", type=float, default=1200.0)
    parser.add_argument("--max-decisions", type=int, default=180)
    parser.add_argument("--scenes", default=None, help="pinned-scene directory passed to every episode")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    specs = [{"task": t, "seed": s, "interface": i, "mode": m, "method": h, "method_config": json.loads(args.method_config),
              "steps_budget": args.steps_budget, "wall_budget_s": args.wall_budget_s, "max_decisions": args.max_decisions,
              **({"scenes": args.scenes} if args.scenes else {})}
             for h, m, i, t, s in itertools.product(args.methods, args.modes, args.interfaces, args.tasks, args.seeds)]
    (out / "matrix.json").write_text(json.dumps(specs, indent=1))
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for result in pool.map(lambda spec: run_one(spec, out), specs):
            results.append(result)
            print(json.dumps({k: result.get(k) for k in ("task", "seed", "interface", "mode", "method", "official_success", "termination", "simulator_steps", "wall_s")}), flush=True)
            summarize(results, out)
    print(summarize(results, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
