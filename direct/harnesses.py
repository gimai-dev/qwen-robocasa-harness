"""H1-H8 conditions. Each changes what Qwen sees or how its numbers are read;
Qwen remains the author of every numerical target."""
from __future__ import annotations

import io
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .actions import SLOT_STEPS, decode_action, short_response_schema
from .methods import Method


# ---------------------------------------------------------------- H1 -------
class H1VisualMarkers(Method):
    """Draw region ids, TCP marker and a coordinate grid on the current images;
    add a gallery of region crops. No memory, no recovery, no extra planning."""
    name = "h1"

    def prompt_suffix(self) -> str:
        return ("VISUAL MARKERS (this condition): the current images carry a 32-pixel grid, each `regions` entry is drawn as a box "
                "labelled with its id (e.g. r3) at its pixel location, and the TCP is drawn as a magenta cross. A fourth image "
                "`region_gallery` shows a crop of every region labelled with its id, colour and world position so you can match ids "
                "to objects. The previous images (if present) are unannotated.")

    def images(self, ctx, current, previous):
        regions = ctx["regions"]
        annotated = []
        for label in ("left", "right", "wrist"):
            image = Image.open(io.BytesIO(current[label])).convert("RGB")
            draw = ImageDraw.Draw(image)
            for x in range(0, 256, 32):
                draw.line([(x, 0), (x, 255)], fill=(255, 255, 0), width=1)
                draw.line([(0, x), (255, x)], fill=(0, 255, 255), width=1)
            for region in regions["regions"]:
                view = region["views"].get(label)
                if view is None:
                    continue
                x, y, w, h = view["bbox_xywh"]
                draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=1)
                draw.text((min(x, 236), max(0, y - 10)), region["id"], fill=(255, 0, 0))
            pixel = ctx["observation"]["public_state"]["tcp_pixels"][label]
            if pixel["visible"]:
                u, v = pixel["u"], pixel["v"]
                draw.line([(u - 6, v), (u + 6, v)], fill=(255, 0, 255), width=2)
                draw.line([(u, v - 6), (u, v + 6)], fill=(255, 0, 255), width=2)
            buffer = io.BytesIO(); image.save(buffer, format="PNG")
            annotated.append((f"current_{label}_annotated", buffer.getvalue()))
        gallery = self._gallery(regions, current)
        if previous is not None:
            annotated += [(f"before_previous_action_{label}", previous[label]) for label in ("left", "right", "wrist")]
        if gallery is not None:
            annotated.append(("region_gallery", gallery))
        return annotated

    def _gallery(self, regions, current):
        rows = regions["regions"]
        if not rows:
            return None
        columns = 4
        tile = 128
        board = Image.new("RGB", (columns * tile, ((len(rows) + columns - 1) // columns) * (tile + 28)), "white")
        draw = ImageDraw.Draw(board)
        for index, region in enumerate(rows):
            label = "wrist" if "wrist" in region["views"] and region["views"]["wrist"]["area_px"] > 100 else next(iter(region["views"]))
            view = region["views"][label]
            x, y, w, h = view["bbox_xywh"]
            image = Image.open(io.BytesIO(current[label])).convert("RGB")
            crop = image.crop((max(0, x - 6), max(0, y - 6), min(256, x + w + 6), min(256, y + h + 6)))
            crop.thumbnail((tile - 8, tile - 8))
            cx, cy = (index % columns) * tile, (index // columns) * (tile + 28)
            board.paste(crop, (cx + 4, cy + 28))
            draw.text((cx + 4, cy + 2), f"{region['id']} rgb{tuple(region['mean_rgb'])}", fill="black")
            draw.text((cx + 4, cy + 14), f"{region['world_m']}", fill="black")
        buffer = io.BytesIO(); board.save(buffer, format="PNG")
        return buffer.getvalue()


# ---------------------------------------------------------------- H2 -------
class H2RelativeActions(Method):
    """EE position as displacement from the current TCP (world axes); joint targets as delta-q."""
    name = "h2"
    representation = "relative"

    def prompt_suffix(self) -> str:
        return ("ACTION REPRESENTATION (this condition): for \"ee\" actions, `p` is a DISPLACEMENT in metres to add to the current "
                "TCP world position (world axes: +x, +y, +z up), not an absolute position; `o` stays an absolute world quaternion. "
                "For \"joint\" actions, `q` is a DELTA in radians added to the current joints. Receipts still report absolute measured values.")


# ---------------------------------------------------------------- H4 -------
class H4ExecutionTiming(Method):
    """Qwen chooses arm/hold duration; base motion and gripper changes use a full slot."""
    name = "h4"

    def short_schema(self, interface):
        return short_response_schema(interface=interface, action_extra={"s": {"type": "integer", "enum": [5, 10, 20]}})

    def prompt_suffix(self) -> str:
        return ("EXECUTION TIMING (this condition): every action carries an integer field `s` in {5, 10, 20}: the number of simulator "
                "steps executed before you are observed again. Motion speed is unchanged (0.01 m per step along the line), so s=5 "
                "moves at most 0.05 m and lets you re-observe sooner; s=20 is the full slot. Short 5/10-step slots apply only to arm/hold "
                "actions without a gripper-command change. Base actions always take 20 steps (16 drive + 4 brake); gripper changes also take 20 steps. "
                "Short slots cost decisions from the same 180-decision budget.")

    def slot_steps(self, action, current_gripper=None) -> int:
        steps = action.raw.get("s", SLOT_STEPS)
        if action.kind == "base" or (action.gripper is not None and action.gripper != current_gripper):
            return SLOT_STEPS
        return int(steps) if steps in (5, 10, 20) else SLOT_STEPS


class H4Fixed5(Method):
    """Control for H4: 5-step arm/hold slots; base motion and gripper changes use 20."""
    name = "h4c"

    def prompt_suffix(self) -> str:
        return ("EXECUTION TIMING (this condition): arm/hold actions that keep the gripper command execute only 5 simulator steps (at most 0.05 m of motion) "
                "before you are observed again. Base actions always take 20 steps (16 drive + 4 brake); gripper changes also take 20 steps. "
                "Plan targets accordingly and expect frequent partial receipts.")

    def slot_steps(self, action, current_gripper=None) -> int:
        if action.kind == "base" or (action.gripper is not None and action.gripper != current_gripper):
            return SLOT_STEPS
        return 5


# ---------------------------------------------------------------- H5 -------
class H5WorkingMemory(Method):
    """Qwen updates a bounded working memory inside the same control call."""
    name = "h5"

    def __init__(self, *, run, config):
        super().__init__(run=run, config=config)
        self.memory = {"current_goal": "", "confirmed_facts": [], "uncertain_beliefs": [], "recent_attempts": []}

    def short_schema(self, interface):
        memory = {"type": "object", "additionalProperties": False, "properties": {
            "current_goal": {"type": "string"},
            "confirmed_facts": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            "uncertain_beliefs": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
            "recent_attempts": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
            "required": ["current_goal", "confirmed_facts", "uncertain_beliefs", "recent_attempts"]}
        return short_response_schema(interface=interface, top_extra={"memory": memory})

    def prompt_suffix(self) -> str:
        return ("WORKING MEMORY (this condition): the prompt carries `working_memory` written by you on the previous call. In every "
                "response also return an updated `memory` object: `current_goal` (the sub-goal you are pursuing now), `confirmed_facts` "
                "(at most 8 short facts verified by receipts or images, e.g. object positions, what the gripper holds), "
                "`uncertain_beliefs` (at most 6 hypotheses not yet verified) and `recent_attempts` (the last 3 relevant actions with "
                "their measured outcome). Keep it factual; it is your only memory beyond the previous receipt.")

    def observe(self, ctx):
        return {"working_memory": self.memory}, []

    def after_receipt(self, ctx, action, receipt):
        call = ctx.get("call")
        if call and isinstance(call["parsed"].get("memory"), dict):
            self.memory = call["parsed"]["memory"]
            with (self.run / "memory-trace.jsonl").open("a") as handle:
                handle.write(json.dumps({"decision": ctx["decision"], "memory": self.memory}) + "\n")

    def finalize(self):
        return {"final_memory": self.memory}


REGISTRY = {"h1": H1VisualMarkers, "h2": H2RelativeActions, "h4": H4ExecutionTiming, "h4c": H4Fixed5, "h5": H5WorkingMemory}


def make(name: str, *, run: Path, config: Mapping[str, object]) -> Method:
    if name in REGISTRY:
        return REGISTRY[name](run=run, config=config)
    from . import harnesses_extra
    return harnesses_extra.make(name, run=run, config=config)
