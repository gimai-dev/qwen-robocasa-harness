"""Summarize matrices: per-condition success tables, paired differences vs clean, costs.

    python -m direct.analyze --matrices /home/jli/state/qwen-direct/matrix/phaseB-clean [...] --out REPORT-phaseB.md
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def load(matrices):
    rows = []
    for root in map(Path, matrices):
        for path in root.rglob("result.json"):
            r = json.loads(path.read_text())
            r["run_dir"] = str(path.parent)
            rows.append(r)
    return rows


def key(r):
    return (r["method"], r["interface"], r["mode"])


def start(r):
    return (r["task"], r["seed"])


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def paired_diff(a: dict, b: dict):
    """a, b: {start: success}. Returns (n, mean diff, bootstrap 95% interval, wins, losses)."""
    common = sorted(set(a) & set(b))
    if not common:
        return None
    diffs = [int(bool(a[s])) - int(bool(b[s])) for s in common]
    rng = random.Random(0)
    boots = []
    for _ in range(2000):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    return {"n": len(common), "mean_diff": sum(diffs) / len(diffs), "ci95": [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]],
            "wins": sum(1 for d in diffs if d > 0), "losses": sum(1 for d in diffs if d < 0)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline", default="clean")
    args = parser.parse_args()
    rows = load(args.matrices)
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    lines = ["# Results", "", f"Runs: {len(rows)} from {', '.join(args.matrices)}", ""]
    lines += ["## Success per condition and task", "", "| method | interface | mode | task | success/attempts | terminations |", "|---|---|---|---|---|---|"]
    for k in sorted(groups):
        by_task = defaultdict(list)
        for r in groups[k]:
            by_task[r["task"]].append(r)
        for task in sorted(by_task):
            rs = by_task[task]
            terms = defaultdict(int)
            for r in rs:
                terms[r["termination"]] += 1
            lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {task} | {sum(1 for r in rs if r['official_success'])}/{len(rs)} | {dict(terms)} |")
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | **all** | {sum(1 for r in groups[k] if r['official_success'])}/{len(groups[k])} | |")
    lines += ["", "## Paired differences vs baseline (same task+seed starts)", "", "| method | interface | mode | n | success | baseline | mean diff | bootstrap 95% CI | wins/losses |", "|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(groups):
        if k[0] == args.baseline:
            continue
        base_key = (args.baseline, k[1], k[2])
        if base_key not in groups:
            continue
        a = {start(r): r["official_success"] for r in groups[k]}
        b = {start(r): r["official_success"] for r in groups[base_key]}
        d = paired_diff(a, b)
        if d is None:
            continue
        common = set(a) & set(b)
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {d['n']} | {sum(1 for s in common if a[s])}/{d['n']} | {sum(1 for s in common if b[s])}/{d['n']} | {d['mean_diff']:+.3f} | [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | {d['wins']}/{d['losses']} |")
    lines += ["", "## Costs (means per episode)", "", "| method | interface | mode | n | sim steps | decisions | rejected | Qwen calls | prompt tok | completion tok | Qwen s | SAM s | wall s |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(groups):
        rs = groups[k]
        f = lambda field: mean([r.get(field) for r in rs])
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {len(rs)} | {f('simulator_steps'):.0f} | {f('decisions'):.1f} | {f('rejected_actions'):.1f} | {f('qwen_calls'):.1f} | {f('prompt_tokens'):.0f} | {f('completion_tokens'):.0f} | {f('qwen_latency_s'):.0f} | {f('sam_elapsed_s'):.0f} | {f('wall_s'):.0f} |")
    lines += ["", "## Per-run outcomes", "", "| method | interface | mode | task | seed | success | termination | steps | decisions | wall s | run |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (key(r), start(r))):
        lines.append(f"| {r['method']} | {r['interface']} | {r['mode']} | {r['task']} | {r['seed']} | {r['official_success']} | {r['termination']} | {r['simulator_steps']} | {r['decisions']} | {r['wall_s']:.0f} | {Path(r['run_dir']).name} |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
