"""The approved September repair campaign, in separately reviewed batches.

Run on h200-4: python -m direct.revision_campaign --batch stage2 --parallel 3
Each batch records its fixed manifest, task results, and evaluator-only milestones.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import statistics

from .matrix import run_one
from .milestones import per_run

ROOT = Path("/home/jli/state/qwen-direct/revision-2026-09")
TASKS = ("PickPlaceCounterToSink", "PickPlaceCounterToDrawer", "PickPlaceStoveToCounter")
BATCHES = {
    "stage2": (("ee-clean", "clean", "ee", "short", False),
               ("ee-h2", "h2", "ee", "short", False),
               ("ee-clean-ready", "clean", "ee", "short", True)),
    "stage3": (("joint-clean", "clean", "joint", "short", False),
               ("ee-full", "clean", "ee", "full", False),
               ("joint-full", "clean", "joint", "full", False)),
    "stage4": (("ee-h3", "h3", "ee", "short", False),
               ("ee-h4", "h4", "ee", "short", False)),
    "screen": (("ee-h1", "h1", "ee", "short", False),
               ("ee-h5", "h5", "ee", "short", False),
               ("ee-h6", "h6", "ee", "short", False),
               ("ee-h8", "h8", "ee", "short", False)),
}


def specifications(batch, root):
    specs = []
    for seed in (0, 1, 2):
        for task in TASKS:
            for label, method, interface, mode, ready in BATCHES[batch]:
                spec = {"condition": label, "task": task, "seed": seed, "method": method,
                        "interface": interface, "mode": mode, "ready_pose": ready,
                        "steps_budget": 900, "wall_budget_s": 1200., "max_decisions": 180,
                        "method_config": {}}
                if method == "h6":
                    spec["method_config"] = {"bank": str(root / "banks/h6-failures.json")}
                out = root / batch / label
                spec["out"] = str(out)
                spec["run"] = str(out / f"{task}-s{seed}-{interface}-{mode}-{method}")
                specs.append(spec)
    return specs


def scored_result(result):
    return (isinstance(result.get("official_success"), bool) and
            result.get("termination") not in ("infrastructure_error", "launcher_failure"))


def summarize(specs, out):
    rows = []
    for spec in specs:
        path = Path(spec["run"])
        result = (json.loads((path / "result.json").read_text()) if (path / "result.json").exists()
                  else {"official_success": None, "termination": "launcher_failure"})
        inspection = out / "inspection" / spec["condition"] / f"{path.name}.json"
        milestones = per_run(json.loads(inspection.read_text())) if inspection.exists() else None
        rows.append({"condition": spec["condition"], "run": str(path), "result": result,
                     "milestones": milestones})
    (out / "results.json").write_text(json.dumps(rows, indent=2))
    text = ["# Revised campaign: " + out.name, "",
            "Milestones use offline simulator inspection; the policy does not receive object truth.", "",
            "| condition | completed | infrastructure attempts | success | approach | contact | hold | lift | mean steps | mean calls | mean wall s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for condition in dict.fromkeys(s["condition"] for s in specs):
        attempts = [r for r in rows if r["condition"] == condition]
        group = [r for r in attempts if scored_result(r["result"])]
        success = sum(bool(r["result"].get("official_success")) for r in group)
        progress = [str(sum(bool((r["milestones"] or {}).get(k)) for r in group))
                    if all(r["milestones"] is not None for r in group) else "pending"
                    for k in ("approach", "contact", "hold", "lift")]
        means = [f"{statistics.mean(r['result'].get(k) or 0 for r in group):.1f}" if group else "N/A"
                 for k in ("simulator_steps", "qwen_calls", "wall_s")]
        text.append("| " + " | ".join([condition, str(len(group)), str(len(attempts) - len(group)), str(success), *progress, *means]) + " |")
    (out / "summary.md").write_text("\n".join(text) + "\n")
    return "\n".join(text)


def inspect_spec(spec, out):
    from .recovery import inspect_run
    run = Path(spec["run"])
    directory = out / "inspection" / spec["condition"]
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{run.name}.json"
    if not destination.exists():
        work = directory / "inspect" / run.name
        if work.exists():
            retained = directory / "interrupted"
            retained.mkdir(exist_ok=True)
            attempt = 1
            while (retained / f"{run.name}-{attempt}").exists():
                attempt += 1
            work.rename(retained / f"{run.name}-{attempt}")
        rows = inspect_run(run, directory)
        destination.write_text(json.dumps(rows, indent=1))
    return spec["condition"], run.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", choices=BATCHES, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--parallel", type=int, default=3)
    args = parser.parse_args()
    out = args.root / args.batch
    out.mkdir(parents=True, exist_ok=True)
    specs = specifications(args.batch, args.root)
    if args.batch == "screen" and not (args.root / "banks/h6-failures.json").exists():
        raise SystemExit("Build the clean-development failure bank before this screen.")
    manifest = out / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != specs:
            raise SystemExit("Existing batch manifest differs; use a separate campaign root.")
    else:
        manifest.write_text(json.dumps(specs, indent=2))
    for spec in specs:
        Path(spec["out"]).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        jobs = {pool.submit(run_one, spec, Path(spec["out"])): spec for spec in specs}
        for completed, job in enumerate(as_completed(jobs), 1):
            result = job.result()
            spec = jobs[job]
            print(json.dumps({"phase": "episodes", "completed": completed, "total": len(specs),
                              "condition": spec["condition"], "task": spec["task"], "seed": spec["seed"],
                              "success": result.get("official_success"), "termination": result.get("termination")}), flush=True)
    summarize(specs, out)
    failed = [s for s in specs if not (Path(s["run"]) / "result.json").exists()
              or not scored_result(json.loads((Path(s["run"]) / "result.json").read_text()))]
    if failed:
        raise SystemExit(f"{len(failed)} infrastructure failures need separate retained attempts before inspection.")
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        jobs = [pool.submit(inspect_spec, spec, out) for spec in specs]
        for completed, job in enumerate(as_completed(jobs), 1):
            condition, run = job.result()
            print(json.dumps({"phase": "inspection", "completed": completed, "total": len(specs),
                              "condition": condition, "run": run}), flush=True)
    print(summarize(specs, out), flush=True)


if __name__ == "__main__":
    main()
