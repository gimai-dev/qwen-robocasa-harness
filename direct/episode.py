"""One complete episode: reset, observe, decide, execute, score, record.

    python -m direct.episode --task PickPlaceCounterToSink --seed 0 \
        --interface ee --mode short --method clean --out /home/jli/state/qwen-direct/runs/<name>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .actions import (MAX_FULL_SLOTS, SLOT_STEPS, decode_action, full_response_schema, short_response_schema)
from .executor import Receipt, Simulator, execute
from .methods import make_method
from .observation import build_user_message, image_list, read_images
from .perception import SamClient, perceive
from .policy import MAX_TOKENS_FULL, MAX_TOKENS_SHORT, MalformedOutput, QwenDirectClient

PROMPTS = Path(__file__).resolve().with_name("prompts")
DEFAULT_SCENES = Path("/home/jli/state/qwen-direct/scenes")
MAX_CONSECUTIVE_NO_MOTION = 8


def load_system_prompt(interface: str, mode: str, suffix: str) -> str:
    text = (PROMPTS / "system_common.txt").read_text() + "\n" + (PROMPTS / f"system_{interface}_{mode}.txt").read_text()
    return text + ("\n" + suffix if suffix else "")


def render_video(sim_dir: Path, target: Path) -> int:
    frames = sorted((sim_dir / "video-frames").glob("*.png"))
    if not frames:
        return 0
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "10",
                    "-i", str(sim_dir / "video-frames" / "%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(target)], check=True)
    return len(frames)


def git_revision() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


class Episode:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run = Path(args.out).resolve()
        self.run.mkdir(parents=True, exist_ok=False)
        self.config = {
            "task": args.task, "seed": args.seed, "interface": args.interface, "mode": args.mode, "method": args.method,
            "steps_budget": args.steps_budget, "wall_budget_s": args.wall_budget_s, "max_decisions": args.max_decisions,
            "slot_steps": SLOT_STEPS, "max_full_slots": MAX_FULL_SLOTS, "max_tokens_short": MAX_TOKENS_SHORT,
            "max_tokens_full": MAX_TOKENS_FULL, "code_revision": git_revision(), "method_config": json.loads(args.method_config),
            "restore_from": args.restore_from,
        }
        self.method = make_method(args.method, run=self.run, config=self.config["method_config"])
        self.system_prompt = load_system_prompt(args.interface, args.mode, self.method.prompt_suffix())
        (self.run / "system-prompt.txt").write_text(self.system_prompt)
        (self.run / "config.json").write_text(json.dumps(self.config, indent=1))
        self.decisions: list[dict] = []
        self.started = time.monotonic()
        self.termination = "incomplete"
        self.previous_images: dict[str, bytes] | None = None
        self.previous: dict | None = None
        self.no_motion_streak = 0
        self.rejected = 0

    # ---- helpers ----
    def log_decision(self, record: dict) -> None:
        self.decisions.append(record)
        with (self.run / "decisions.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    def observe(self, decision: int) -> tuple[dict, dict, dict[str, bytes]]:
        observation = self.sim.observation
        regions = perceive(self.sam, observation, self.run / "perception" / f"{decision:03d}")
        images = read_images(observation)
        return observation, regions, images

    def receipt_for_prompt(self, action, receipt: Receipt) -> dict:
        return {"action": action.summary() if action is not None else None, "result": receipt.summary()}

    def act(self, action, decision: int) -> Receipt:
        receipt = execute(self.sim, action, slot_steps=self.method.slot_steps(action))
        return receipt

    # ---- short mode ----
    def run_short(self) -> None:
        interface = self.args.interface
        schema = self.method.short_schema(interface) if hasattr(self.method, "short_schema") else short_response_schema(interface=interface)
        decision = 0
        while True:
            if decision >= self.args.max_decisions:
                self.termination = "decision_budget"; break
            if self.sim.steps_left() <= 0:
                self.termination = "step_budget"; break
            if time.monotonic() - self.started > self.args.wall_budget_s:
                self.termination = "wall_budget"; break
            decision += 1
            observation, regions, images = self.observe(decision)
            ctx = {"decision": decision, "observation": observation, "regions": regions, "images": images,
                   "previous_images": self.previous_images, "previous": self.previous, "sim": self.sim,
                   "client": self.client, "system_prompt": self.system_prompt}
            extra, extra_images = self.method.observe(ctx)
            user_text = build_user_message(observation=observation, regions=regions, decision=decision,
                                           max_decisions=self.args.max_decisions, previous=self.previous, extra=extra)
            image_window = self.method.images(ctx, images, self.previous_images)
            if image_window is None:
                image_window = image_list(images, self.previous_images)
            image_window = image_window + extra_images
            record = {"decision": decision, "steps_used": observation["steps_used"], "regions": len(regions["regions"])}
            try:
                call = self.client.complete(system_prompt=self.system_prompt, user_text=user_text, images=image_window,
                                            response_schema=schema, max_tokens=MAX_TOKENS_SHORT, category="control", decision=decision)
            except MalformedOutput as error:
                record.update({"status": "malformed_output", "error": str(error)})
                self.log_decision(record)
                self.previous = {"action": None, "result": {"status": "invalid", "reason": "your previous output was not valid JSON for the schema; emit exactly one action"}}
                self.no_motion_streak += 1
                if self.no_motion_streak >= MAX_CONSECUTIVE_NO_MOTION:
                    self.termination = "no_progress"; break
                continue
            record["reasoning"] = call["parsed"].get("reasoning")
            ctx["call"] = call
            if hasattr(self.method, "choose"):
                try:
                    raw_action = self.method.choose(ctx, call)
                except MalformedOutput as error:
                    record.update({"status": "malformed_output", "error": str(error), "stage": "preview_select"})
                    self.log_decision(record)
                    self.no_motion_streak += 1
                    if self.no_motion_streak >= MAX_CONSECUTIVE_NO_MOTION:
                        self.termination = "no_progress"; break
                    continue
            else:
                raw_action = call["parsed"].get("action")
            state = observation["public_state"]
            try:
                action = decode_action(raw_action, interface=interface, representation=self.method.representation,
                                       current_tcp_world=state["tcp_world_position_m"], current_q=state["arm_q_rad"])
            except ValueError as error:
                record.update({"status": "invalid_action", "raw_action": raw_action, "error": str(error)})
                self.log_decision(record)
                self.previous = {"action": raw_action, "result": {"status": "invalid", "reason": str(error)}}
                self.previous_images = None
                self.no_motion_streak += 1
                if self.no_motion_streak >= MAX_CONSECUTIVE_NO_MOTION:
                    self.termination = "no_progress"; break
                continue
            record["action"] = action.summary()
            if action.kind == "stop":
                record["status"] = "stop"
                self.log_decision(record)
                self.termination = "stop"; break
            receipt = self.act(action, decision)
            record.update({"status": receipt.status, "steps": receipt.steps, "receipt": receipt.summary()})
            if receipt.child is not None:
                record["tcp_after"] = receipt.child["tcp_world_after_m"]
            self.log_decision(record)
            self.method.after_receipt(ctx, action, receipt)
            if receipt.steps > 0:
                self.previous_images = images
                self.no_motion_streak = 0
            else:
                self.rejected += 1
                self.no_motion_streak += 1
            self.previous = self.receipt_for_prompt(action, receipt)
            if receipt.status == "budget_exhausted" and receipt.steps == 0:
                self.termination = "step_budget"; break
            if self.no_motion_streak >= MAX_CONSECUTIVE_NO_MOTION:
                self.termination = "no_progress"; break

    # ---- full mode ----
    def run_full(self) -> None:
        interface = self.args.interface
        schema = full_response_schema(interface=interface)
        observation, regions, images = self.observe(1)
        ctx = {"decision": 1, "observation": observation, "regions": regions, "images": images, "previous_images": None,
               "previous": None, "sim": self.sim, "client": self.client, "system_prompt": self.system_prompt}
        extra, extra_images = self.method.observe(ctx)
        user_text = build_user_message(observation=observation, regions=regions, decision=1,
                                       max_decisions=1, previous=None, extra=extra)
        image_window = (self.method.images(ctx, images, None) or image_list(images, None)) + extra_images
        record = {"decision": 1, "steps_used": 0, "regions": len(regions["regions"])}
        try:
            call = self.client.complete(system_prompt=self.system_prompt, user_text=user_text, images=image_window,
                                        response_schema=schema, max_tokens=MAX_TOKENS_FULL, category="control", decision=1)
        except MalformedOutput as error:
            record.update({"status": "truncated_output" if error.record.get("truncated") else "malformed_output", "error": str(error)})
            self.log_decision(record)
            self.termination = record["status"]; return
        if hasattr(self.method, "revise_sequence"):
            call = self.method.revise_sequence(ctx, call)
        sequence = call["parsed"].get("sequence") or []
        record.update({"reasoning": call["parsed"].get("reasoning"), "sequence_length": len(sequence), "truncated": call.get("truncated")})
        self.log_decision(record)
        state = observation["public_state"]
        tcp = list(state["tcp_world_position_m"]); q = list(state["arm_q_rad"])
        for index, raw_action in enumerate(sequence, start=1):
            entry = {"decision": 1, "slot": index, "raw_action": raw_action}
            try:
                action = decode_action(raw_action, interface=interface, representation=self.method.representation,
                                       current_tcp_world=tcp, current_q=q)
            except ValueError as error:
                entry.update({"status": "invalid_action", "error": str(error)})
                self.log_decision(entry)
                self.termination = f"invalid_action_at_slot_{index}"; return
            entry["action"] = action.summary()
            receipt = self.act(action, 1)
            entry.update({"status": receipt.status, "steps": receipt.steps, "receipt": receipt.summary()})
            self.log_decision(entry)
            if receipt.child is not None:
                tcp = list(receipt.child["tcp_world_after_m"]); q = list(receipt.child["arm_q_after"])
            if receipt.steps == 0:
                self.termination = f"{receipt.status}_at_slot_{index}"; return
            if receipt.status == "budget_exhausted":
                self.termination = "step_budget"; return
        self.termination = "sequence_complete"

    # ---- main ----
    def run_episode(self) -> dict:
        outcome: dict = {}
        error_text = None
        try:
            self.client = QwenDirectClient(log_path=self.run / "qwen-calls.jsonl")
            self.sam = SamClient(self.run / "sam.log")
            self.sim = Simulator(task=self.args.task, seed=self.args.seed, run=self.run, scenes=Path(self.args.scenes),
                                 action_budget=self.args.steps_budget, wall_budget_s=self.args.wall_budget_s,
                                 restore_from=Path(self.args.restore_from) if self.args.restore_from else None)
            self.sim.launch()
            self.started = time.monotonic()
            if self.args.mode == "short":
                self.run_short()
            else:
                self.run_full()
            outcome = self.sim.finish()
        except Exception:
            error_text = traceback.format_exc()
            self.termination = "infrastructure_error"
            try:
                outcome = self.sim.close()
            except Exception:
                pass
        finally:
            for name in ("sam", "client"):
                try:
                    getattr(self, name).close()
                except Exception:
                    pass
        video_frames = 0
        try:
            video_frames = render_video(self.run / "sim", self.run / "episode.mp4")
        except Exception as error:
            error_text = (error_text or "") + f"\nvideo: {error!r}"
        official = outcome.get("official_success")
        motion_decisions = [d for d in self.decisions if d.get("steps", 0) > 0]
        result = {
            **self.config,
            "official_success": bool(official) if official is not None else None,
            "termination": self.termination,
            "simulator_steps": outcome.get("simulator_steps"),
            "decisions": len([d for d in self.decisions if "slot" not in d]),
            "slots_executed": len(motion_decisions),
            "rejected_actions": len([d for d in self.decisions if d.get("steps") == 0 and d.get("status") not in ("stop",)]),
            "wall_s": round(time.monotonic() - self.started, 1),
            **(self.client.totals() if hasattr(self, "client") else {}),
            "sam_calls": getattr(getattr(self, "sam", None), "counter", 0),
            "sam_elapsed_s": round(getattr(getattr(self, "sam", None), "total_elapsed_s", 0.0), 1),
            "video": str(self.run / "episode.mp4") if video_frames else None,
            "video_frames": video_frames,
            "scene": str(Path(self.args.scenes) / f"{self.args.task}-seed{self.args.seed}.json"),
            "simulator_terminal": outcome,
            "method_summary": self.method.finalize(),
            "error": error_text,
        }
        (self.run / "result.json").write_text(json.dumps(result, indent=1))
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--interface", choices=("ee", "joint"), default="ee")
    parser.add_argument("--mode", choices=("short", "full"), default="short")
    parser.add_argument("--method", default="clean")
    parser.add_argument("--method-config", default="{}", help="JSON dict of condition parameters")
    parser.add_argument("--out", required=True)
    parser.add_argument("--scenes", default=str(DEFAULT_SCENES))
    parser.add_argument("--steps-budget", type=int, default=900)
    parser.add_argument("--wall-budget-s", type=float, default=1200.0)
    parser.add_argument("--max-decisions", type=int, default=180)
    parser.add_argument("--restore-from", default=None, help="H8: simulator snapshot to continue from")
    args = parser.parse_args(argv)
    result = Episode(args).run_episode()
    print(json.dumps({k: result[k] for k in ("task", "seed", "interface", "mode", "method", "official_success", "termination",
                                             "simulator_steps", "decisions", "wall_s", "qwen_calls", "completion_tokens") if k in result}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
