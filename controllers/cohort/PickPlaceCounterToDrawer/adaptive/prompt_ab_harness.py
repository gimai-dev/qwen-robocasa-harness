"""Prompt A/B harness: run smoke episodes per prompt variant and score outcomes.

The harness never touches controller mechanics. It drives ``candidate.py`` once
per (variant, repeat), pulls the sealed episode artifacts from the remote box,
and reduces every ``wrapper-result.json`` to one fixed metric row judged by
physical evidence: first valid handle contact, deepest milestone, final
success, and a concrete failure class.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

ARTICULATED_CONTACT_MIN_SEPARATION = 0.0032
EPISODE_ACTION_BUDGET = 450
REMOTE_HOST = "h200-2"
ARTIFACT_FILES = ("wrapper-result.json", "critic-evidence.json")
MILESTONE_ORDER = {
    "articulated": ("observe", "approach", "engage", "actuate", "verify_goal"),
    "control": ("observe", "approach", "engage", "actuate", "verify_goal"),
    "grasp_place": (
        "observe",
        "approach",
        "pregrasp",
        "grasp",
        "transport",
        "release",
        "verify_goal",
    ),
}
ROW_FIELDS = (
    "variant",
    "run_dir",
    "task",
    "seed",
    "served_model_id",
    "snapshot_digest",
    "system_prompt_sha256",
    "system_prompt_parts_sha256",
    "critic_prompt_sha256",
    "release_digest",
    "status",
    "success",
    "failure_class",
    "decisions",
    "controller_calls",
    "critic_attempts",
    "rejected_proposals",
    "simulator_steps",
    "action_budget",
    "wall_s",
    "deepest_milestone",
    "first_handle_contact_decision",
    "max_close_separation_m",
    "empty_close_count",
    "wrist_roll_count",
    "video",
)


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _separation(receipt: Mapping[str, object]) -> float | None:
    residual = receipt.get("gripper_residual")
    if not isinstance(residual, Mapping):
        return None
    return _finite(residual.get("measured_end_finger_separation"))


def _close_requested(receipt: Mapping[str, object]) -> bool:
    if receipt.get("requested_gripper") in {"close", "hold"}:
        return True
    if receipt.get("kind") != "move_joints":
        return False
    intent = _finite(receipt.get("gripper_intent"))
    if intent is not None:
        return intent <= 0.0
    targets = receipt.get("requested_targets")
    return isinstance(targets, Mapping) and _finite(targets.get("gripper")) == 0.0


def _is_wrist_roll(receipt: Mapping[str, object]) -> bool:
    if receipt.get("kind") != "move_joints":
        return False
    targets = receipt.get("requested_targets")
    return (
        isinstance(targets, Mapping)
        and "joint7" in targets
        and _finite(targets.get("gripper")) == 1.0
    )


def deepest_milestone(episode: Mapping[str, object]) -> str | None:
    family = str(episode.get("family") or "articulated")
    order = MILESTONE_ORDER.get(family, MILESTONE_ORDER["articulated"])
    history = episode.get("milestone_history")
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        return None
    ranks = [order.index(str(item)) for item in history if str(item) in order]
    return order[max(ranks)] if ranks else None


def classify_failure(
    episode: Mapping[str, object], *, action_budget: int = EPISODE_ACTION_BUDGET
) -> str:
    if episode.get("success") is True:
        return "success"
    status = str(episode.get("status") or "unknown")
    if "safety" in status:
        return "safety_abort"
    if status == "policy_failed_wall_budget":
        return "wall_budget"
    if status == "policy_gave_up":
        return "gave_up"
    if status == "policy_finished_false":
        return "finished_false"
    steps = _finite(episode.get("simulator_steps"))
    if steps is not None and steps >= action_budget:
        return "action_budget_exhausted"
    if status == "policy_failed_proposal_audit":
        records = episode.get("proposal_audit_records")
        contradiction = "unknown"
        if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
            for record in reversed(records):
                if (
                    isinstance(record, Mapping)
                    and record.get("status") != "approved_for_execution"
                ):
                    contradiction = str(record.get("contradiction") or "unknown")
                    break
        return f"revisions_exhausted:{contradiction}"
    return status


def summarize_episode(
    episode: Mapping[str, object],
    *,
    variant: str,
    run_dir: str,
    release_digest: str | None = None,
    action_budget: int = EPISODE_ACTION_BUDGET,
) -> dict[str, object]:
    receipts = episode.get("receipts")
    receipt_list = (
        [item for item in receipts if isinstance(item, Mapping)]
        if isinstance(receipts, Sequence) and not isinstance(receipts, (str, bytes))
        else []
    )
    first_contact: int | None = None
    max_close: float | None = None
    empty_closes = 0
    wrist_rolls = 0
    for index, receipt in enumerate(receipt_list):
        if _is_wrist_roll(receipt):
            wrist_rolls += 1
        if not _close_requested(receipt):
            continue
        separation = _separation(receipt)
        if separation is None:
            continue
        max_close = separation if max_close is None else max(max_close, separation)
        if separation > ARTICULATED_CONTACT_MIN_SEPARATION:
            if first_contact is None:
                first_contact = index
        else:
            empty_closes += 1
    closure = episode.get("proposal_audit_closure")
    rejected = (
        closure.get("rejected_count") if isinstance(closure, Mapping) else None
    )
    if "decisions" not in episode:
        episode = {**episode, "decisions": len(receipt_list)}
    row: dict[str, object] = {
        "variant": variant,
        "run_dir": run_dir,
        "task": episode.get("task"),
        "seed": episode.get("seed"),
        "served_model_id": episode.get("served_model_id"),
        "snapshot_digest": episode.get("snapshot_digest"),
        "system_prompt_sha256": episode.get("system_prompt_sha256"),
        "system_prompt_parts_sha256": episode.get("system_prompt_parts_sha256"),
        "critic_prompt_sha256": episode.get("critic_prompt_sha256"),
        "release_digest": release_digest,
        "status": episode.get("status"),
        "success": episode.get("success"),
        "failure_class": classify_failure(episode, action_budget=action_budget),
        "decisions": episode.get("decisions"),
        "controller_calls": episode.get("qwen_controller_calls"),
        "critic_attempts": episode.get("qwen_critic_attempts"),
        "rejected_proposals": rejected,
        "simulator_steps": episode.get("simulator_steps"),
        "action_budget": action_budget,
        "wall_s": episode.get("wall_s"),
        "deepest_milestone": deepest_milestone(episode),
        "first_handle_contact_decision": first_contact,
        "max_close_separation_m": max_close,
        "empty_close_count": empty_closes,
        "wrist_roll_count": wrist_rolls,
        "video": episode.get("video"),
    }
    assert tuple(row) == ROW_FIELDS
    return row


def build_ab_report(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_variant: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_variant.setdefault(str(row.get("variant")), []).append(row)
    summary: dict[str, object] = {}
    for variant, items in by_variant.items():
        contacts = [
            item["first_handle_contact_decision"]
            for item in items
            if item.get("first_handle_contact_decision") is not None
        ]
        steps = [s for item in items if (s := _finite(item.get("simulator_steps")))]
        walls = [w for item in items if (w := _finite(item.get("wall_s")))]
        summary[variant] = {
            "runs": len(items),
            "successes": sum(1 for item in items if item.get("success") is True),
            "runs_with_handle_contact": len(contacts),
            "earliest_handle_contact_decision": min(contacts) if contacts else None,
            "deepest_milestones": sorted(
                {str(item.get("deepest_milestone")) for item in items}
            ),
            "failure_classes": sorted(
                {str(item.get("failure_class")) for item in items}
            ),
            "mean_simulator_steps": (sum(steps) / len(steps)) if steps else None,
            "mean_wall_s": (sum(walls) / len(walls)) if walls else None,
            "system_prompt_sha256": sorted(
                {str(item.get("system_prompt_sha256")) for item in items}
            ),
            "critic_prompt_sha256": sorted(
                {str(item.get("critic_prompt_sha256")) for item in items}
            ),
        }
    return {
        "schema": "robocasa-qwen-prompt-ab-report/v1",
        "rows": [dict(row) for row in rows],
        "summary": summary,
    }


def render_markdown(report: Mapping[str, object]) -> str:
    columns = (
        "variant",
        "run_dir",
        "failure_class",
        "deepest_milestone",
        "first_handle_contact_decision",
        "max_close_separation_m",
        "empty_close_count",
        "wrist_roll_count",
        "decisions",
        "rejected_proposals",
        "simulator_steps",
        "action_budget",
        "wall_s",
        "system_prompt_sha256",
    )
    lines = ["# Prompt A/B report", "", "| " + " | ".join(columns) + " |"]
    lines.append("|" + "---|" * len(columns))
    rows = report.get("rows")
    for row in rows if isinstance(rows, Sequence) else []:
        cells = []
        for column in columns:
            value = row.get(column) if isinstance(row, Mapping) else None
            if isinstance(value, float):
                value = f"{value:.4g}"
            elif column == "system_prompt_sha256" and isinstance(value, str):
                value = value[:12]
            cells.append("" if value is None else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    summary = report.get("summary")
    for variant, item in (summary.items() if isinstance(summary, Mapping) else []):
        if not isinstance(item, Mapping):
            continue
        lines.append(
            f"- **{variant}**: {item.get('runs')} run(s), "
            f"{item.get('successes')} success(es), "
            f"{item.get('runs_with_handle_contact')} with handle contact "
            f"(earliest at receipt {item.get('earliest_handle_contact_decision')}), "
            f"deepest {item.get('deepest_milestones')}, "
            f"failures {item.get('failure_classes')}, "
            f"mean steps {item.get('mean_simulator_steps')}, "
            f"mean wall {item.get('mean_wall_s')} s"
        )
    lines.append("")
    return "\n".join(lines)


def _load_row(run_dir: Path, variant: str) -> dict[str, object]:
    episode = json.loads((run_dir / "wrapper-result.json").read_text())
    meta_path = run_dir / "candidate-result.json"
    release = None
    action_budget = EPISODE_ACTION_BUDGET
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        release = metadata.get("release_digest")
        configured_budget = metadata.get("smoke_action_budget")
        if isinstance(configured_budget, int) and not isinstance(configured_budget, bool):
            action_budget = configured_budget
    return summarize_episode(
        episode,
        variant=variant,
        run_dir=run_dir.name,
        release_digest=release,
        action_budget=action_budget,
    )


def _write_report(out: Path, rows: Sequence[Mapping[str, object]]) -> None:
    report = build_ab_report(rows)
    (out / "ab-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (out / "ab-report.md").write_text(render_markdown(report))


def _run_variant(
    *,
    root: Path,
    variant: str,
    task: str,
    decisions: int,
    action_budget: int,
    run_dir: Path,
) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(root / "candidate.py"),
        "--smoke-task",
        task,
        "--smoke-protocol",
        "proposal",
        "--smoke-decisions",
        str(decisions),
        "--smoke-action-budget",
        str(action_budget),
        "--prompt-variant",
        variant,
    ]
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(root)},
    )
    (run_dir / "candidate-stderr.txt").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"candidate.py failed for variant {variant}: {completed.stderr[-2000:]}"
        )
    result = json.loads(completed.stdout)
    result["harness_wall_s"] = time.monotonic() - started
    (run_dir / "candidate-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    artifact_root = str(result["artifact_root"])
    for name in ARTIFACT_FILES:
        subprocess.run(
            ["scp", "-q", f"{REMOTE_HOST}:{artifact_root}/smoke/{name}", str(run_dir / name)],
            check=True,
        )
    video = json.loads((run_dir / "wrapper-result.json").read_text()).get("video")
    if isinstance(video, str) and video:
        subprocess.run(
            ["scp", "-q", f"{REMOTE_HOST}:{video}", str(run_dir / "run.mp4")],
            check=False,
        )
    return _load_row(run_dir, variant)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("run", help="run smoke episodes per variant and report")
    run.add_argument("--task", default="OpenToasterOvenDoor")
    run.add_argument("--decisions", type=int, default=24)
    run.add_argument("--smoke-action-budget", type=int, choices=(450, 900), default=450)
    run.add_argument("--variants", default="baseline,rig")
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--out", type=Path, required=True)
    report = sub.add_parser("report", help="rebuild the report from pulled runs")
    report.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    if args.mode == "run":
        variants = [item.strip() for item in str(args.variants).split(",") if item]
        for repeat in range(int(args.repeats)):
            for variant in variants:
                run_dir = out / f"{variant}-{repeat + 1}"
                rows.append(
                    _run_variant(
                        root=root,
                        variant=variant,
                        task=str(args.task),
                        decisions=int(args.decisions),
                        action_budget=int(args.smoke_action_budget),
                        run_dir=run_dir,
                    )
                )
                _write_report(out, rows)
    else:
        for run_dir in sorted(path for path in out.iterdir() if path.is_dir()):
            if not (run_dir / "wrapper-result.json").exists():
                continue
            variant = run_dir.name.rsplit("-", 1)[0]
            rows.append(_load_row(run_dir, variant))
        _write_report(out, rows)
    print(render_markdown(build_ab_report(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
