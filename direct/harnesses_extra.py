"""H3 (propose and preview), H8 (explicit recovery), and the bank-based H6/H7."""
from __future__ import annotations

import io
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .actions import BASE_MOTION_STEPS, action_schema, decode_action, ee_rotation_matrix
from .kinematics import (MAX_JOINT_STEP, panda_fk, plan_pose_segment, solve_pose_multistart,
                         world_to_base, base_to_world, quat_xyzw_to_matrix, matrix_to_quat_xyzw)
from .methods import Method
from .policy import MAX_TOKENS_SHORT

LABELS = ("A", "B", "C")
COLORS = ((255, 0, 0), (0, 200, 0), (0, 128, 255))


def _project(calibration, point):
    from robocasa_inspect.camera_geometry import project_world_point
    try:
        u, v, _ = project_world_point(calibration, point)
        return (u, v) if 0 <= u < 256 and 0 <= v < 256 else None
    except ValueError:
        return None


def _preview_canvas(ctx, view):
    """Overlay the current view as sent, including any H1 annotations."""
    labels = (f"current_{view}", f"current_{view}_annotated")
    data = next((data for label, data in ctx.get("image_window", []) if label in labels), ctx["images"][view])
    return Image.open(io.BytesIO(data)).convert("RGB")


def _selection_images(ctx, previews):
    """Replace current views in place so history/crops keep their original slots."""
    replacements = {}
    for label, data in previews:
        view = label.removeprefix("preview_")
        for original in (f"current_{view}", f"current_{view}_annotated"):
            replacements[original] = (label, data)
    return [replacements.get(label, (label, data)) for label, data in ctx["image_window"]]


# ---------------------------------------------------------------- H3 -------
class H3ProposePreview(Method):
    """Qwen proposes 2-3 candidates; the executor previews them kinematically
    (IK reachability, predicted TCP, projected markers); Qwen selects or revises once."""
    name = "h3"

    def short_schema(self, interface):
        self.interface = interface
        return {"type": "object", "additionalProperties": False,
                "properties": {"reasoning": {"type": "string"},
                               "candidates": {"type": "array", "minItems": 2, "maxItems": 3, "items": action_schema(interface=interface)}},
                "required": ["reasoning", "candidates"]}

    def prompt_suffix(self) -> str:
        return ("PROPOSE AND PREVIEW (this condition): first output 2 to 3 candidate actions (`candidates`, labelled A, B, C in order). "
                "The executor previews each candidate with forward/inverse kinematics only (reachability, predicted TCP after the slot, "
                "and the final target). Circles mark slot endpoints; crosses mark final targets. You then receive the preview and choose one "
                "(`choice` A/B/C) or output a revised action (`choice` \"revise\" with `action`). No physical outcome is simulated. "
                "The second call is logged as an extra model call within the same control decision. In full mode the candidates are 2 to 3 complete sequences and the "
                "preview draws each predicted TCP path. Arm prefixes assume ideal tracking with the executor's joint-rate cap; "
                "contacts and tracking pauses are not predicted. Base motion is approximate: 16 powered steps at |v|=0.5 measured "
                "0.1593 m translation / 0.4637 rad yaw. Translation has a dead zone at |v|<=0.25; yaw uses interpolation of measured calibration, "
                "including small nonzero motion at |v|=0.1. "
                "Shorter durations are scaled estimates.")

    def _preview_action(self, state, raw, method, current_gripper):
        """Advance the public pose estimate by one slot, keeping the final target separate."""
        row = {"candidate": raw}
        try:
            action = decode_action(raw, interface=self.interface, representation=self.representation,
                                   current_tcp_world=state["tcp_world_position_m"], current_q=state["arm_q_rad"])
        except ValueError as error:
            return {**row, "valid": False, "error": str(error)}, state, current_gripper
        steps = method.slot_steps(action, current_gripper=current_gripper)
        row.update({"valid": True, "slot_steps": steps, "reachable": True})
        predicted_state = dict(state)
        q = np.asarray(state["arm_q_rad"], float).copy()
        base = np.asarray(state["base_world_position_m"], float)
        base_quat = state["base_world_quat_xyzw"]
        tcp = np.asarray(state["tcp_world_position_m"], float)
        waypoints = None
        if action.kind == "ee":
            target = list(action.position_m)
            row["final_target_tcp_world_m"] = [round(v, 3) for v in target]
            p_base, r_base = world_to_base(target, ee_rotation_matrix(action), base, base_quat)
            plan = plan_pose_segment(q, p_base, r_base)
            row["distance_m"] = round(plan["segment_length_m"], 3)
            if plan["status"] == "kinematically_reachable":
                waypoints = plan["waypoints"]
                row["path"] = "straight_line"
            else:
                direct = solve_pose_multistart(q, p_base, r_base)
                row.update({"reachable": direct["status"] == "kinematically_reachable",
                            "ik_residual_m": round(direct["position_error_m"], 4)})
                if not row["reachable"]:
                    row["path"] = None
                    return row, state, current_gripper
                waypoints = [direct["q"]]
                row["path"] = "joint_space_interpolation"
        elif action.kind == "joint":
            waypoints = [action.q_rad]
            target, _ = base_to_world(*panda_fk(action.q_rad), base, base_quat)
            row.update({"final_target_tcp_world_m": [round(float(v), 3) for v in target],
                        "max_joint_delta_rad": round(float(np.max(np.abs(np.asarray(action.q_rad) - q))), 3)})
        elif action.kind == "base":
            motion_steps = min(BASE_MOTION_STEPS, steps)
            scale = motion_steps / 16.0
            yaw = state["base_world_yaw_rad"]
            row["motion_steps"] = motion_steps
            if action.axis == "yaw":
                delta = float(np.interp(abs(action.velocity), [0, 0.1, 0.15, 0.3, 0.4, 0.5],
                                        [0, 0.015591, 0.046706, 0.226537, 0.34556, 0.4637])
                              * np.sign(action.velocity) * scale)
                turn = quat_xyzw_to_matrix([0, 0, np.sin(delta / 2), np.cos(delta / 2)])
                # Omron joint_mobile_yaw is 0.21 m behind the public base origin.
                pivot_offset = -0.21 * np.array([np.cos(yaw), np.sin(yaw), 0])
                pivot = base + pivot_offset
                base = pivot - turn @ pivot_offset
                tcp = pivot + turn @ (tcp - pivot)
                base_quat = matrix_to_quat_xyzw(turn @ quat_xyzw_to_matrix(base_quat))
                yaw = float(np.arctan2(np.sin(yaw + delta), np.cos(yaw + delta)))
                row.update({"predicted_base_move_m": None, "predicted_yaw_delta_rad": round(delta, 4)})
                predicted_state["tcp_world_quat_xyzw"] = matrix_to_quat_xyzw(
                    turn @ quat_xyzw_to_matrix(state["tcp_world_quat_xyzw"]))
            else:
                distance = float(0.66 * max(0.0, abs(action.velocity) - 0.25) * np.sign(action.velocity) * scale)
                direction = ([np.cos(yaw), np.sin(yaw), 0] if action.axis == "x" else [-np.sin(yaw), np.cos(yaw), 0])
                displacement = distance * np.asarray(direction)
                base = base + displacement
                tcp = tcp + displacement
                row.update({"predicted_base_move_m": round(distance, 4), "predicted_yaw_delta_rad": None})
            predicted_state.update({"base_world_position_m": base.tolist(), "base_world_quat_xyzw": base_quat,
                                    "base_world_yaw_rad": yaw})
            row.update({"predicted_base_world_m": [round(float(v), 3) for v in base],
                        "predicted_base_yaw_rad": round(yaw, 4)})
        if waypoints is not None:
            # Match execute_slot's per-joint command cap and waypoint advancement,
            # assuming tracking succeeds. This predicts a commanded prefix, not a contact outcome.
            path_index = 0
            for _ in range(steps):
                target_q = np.asarray(waypoints[path_index], float)
                q += np.clip(target_q - q, -MAX_JOINT_STEP, MAX_JOINT_STEP)
                if path_index < len(waypoints) - 1 and np.max(np.abs(q - target_q)) < 1e-9:
                    path_index += 1
            tcp, rotation = base_to_world(*panda_fk(q), base, base_quat)
            predicted_state.update({"arm_q_rad": q.tolist(), "tcp_world_quat_xyzw": matrix_to_quat_xyzw(rotation)})
            row["target_reached_in_slot"] = bool(np.max(np.abs(q - np.asarray(waypoints[-1]))) < 1e-9)
        predicted_state["tcp_world_position_m"] = tcp.tolist()
        row["predicted_tcp_world_m"] = [round(float(v), 3) for v in tcp]
        next_gripper = action.gripper if action.gripper is not None else current_gripper
        return row, predicted_state, next_gripper

    def _preview(self, ctx, candidates):
        obs = ctx["observation"]; state = obs["public_state"]; cal = obs["camera_calibration"]
        rows = []; markers = {label: [] for label in ("left", "right", "wrist")}
        method = ctx.get("method", self)
        for index, raw in enumerate(candidates[:3]):
            label = LABELS[index]
            row, _, _ = self._preview_action(state, raw, method, obs.get("gripper_command"))
            row["label"] = label
            for kind, field in (("prefix", "predicted_tcp_world_m"), ("target", "final_target_tcp_world_m")):
                if field not in row:
                    continue
                pixels = {}
                for view in markers:
                    pixel = _project(cal[view], row[field])
                    if pixel is not None:
                        markers[view].append((label, pixel, index, kind))
                        pixels[view] = [round(pixel[0]), round(pixel[1])]
                row["predicted_tcp_pixels" if kind == "prefix" else "final_target_tcp_pixels"] = pixels
            rows.append(row)
        images = []
        for view in ("left", "right", "wrist"):
            image = _preview_canvas(ctx, view)
            draw = ImageDraw.Draw(image)
            for label, (u, v), index, kind in markers[view]:
                color = COLORS[index]
                if kind == "prefix":
                    draw.ellipse([u - 5, v - 5, u + 5, v + 5], outline=color, width=2)
                else:
                    draw.line([(u - 4, v - 4), (u + 4, v + 4)], fill=color, width=2)
                    draw.line([(u - 4, v + 4), (u + 4, v - 4)], fill=color, width=2)
                draw.text((u + 6, v - 6 if kind == "prefix" else v + 5), f"{label} {kind}", fill=color)
            buffer = io.BytesIO(); image.save(buffer, format="PNG")
            images.append((f"preview_{view}", buffer.getvalue()))
        return rows, images

    def full_schema(self, interface):
        from .actions import MAX_FULL_SLOTS
        self.interface = interface
        seq = {"type": "array", "minItems": 1, "maxItems": MAX_FULL_SLOTS, "items": action_schema(interface=interface, allow_stop=False)}
        return {"type": "object", "additionalProperties": False,
                "properties": {"reasoning": {"type": "string"}, "candidates": {"type": "array", "minItems": 2, "maxItems": 3, "items": seq}},
                "required": ["reasoning", "candidates"]}

    def revise_sequence(self, ctx, call):
        """Full-mode transfer: preview 2-3 candidate sequences as predicted TCP polylines, then select/revise once."""
        obs = ctx["observation"]; state = obs["public_state"]; cal = obs["camera_calibration"]
        candidates = call["parsed"].get("candidates") or []
        rows, paths = [], []
        method = ctx.get("method", self)
        for index, sequence in enumerate(candidates[:3]):
            predicted_state = dict(state)
            current_gripper = obs.get("gripper_command")
            points, first_problem = [tuple(state["tcp_world_position_m"])], None
            slots = []
            for slot, raw in enumerate(sequence, start=1):
                preview, predicted_state, current_gripper = self._preview_action(predicted_state, raw, method, current_gripper)
                slots.append({"slot": slot, **preview})
                if not preview["valid"] or not preview["reachable"]:
                    first_problem = {"slot": slot, "error": preview.get("error", "unreachable")}; break
                points.append(tuple(predicted_state["tcp_world_position_m"]))
            rows.append({"label": LABELS[index], "length": len(sequence), "kinematic_check": "all reachable" if first_problem is None else first_problem,
                         "slots": slots, "predicted_tcp_path_m": [[round(v, 3) for v in p] for p in points]})
            paths.append(points)
        images = []
        for view in ("left", "right", "wrist"):
            image = _preview_canvas(ctx, view); draw = ImageDraw.Draw(image)
            for index, points in enumerate(paths):
                pixels = [_project(cal[view], p) for p in points]
                for a, b in zip(pixels, pixels[1:]):
                    if a and b:
                        draw.line([a, b], fill=COLORS[index], width=2)
                if pixels and pixels[-1]:
                    draw.text(pixels[-1], LABELS[index], fill=COLORS[index])
                for slot in rows[index]["slots"]:
                    target = slot.get("final_target_tcp_world_m")
                    pixel = _project(cal[view], target) if target is not None else None
                    if pixel is not None:
                        u, v = pixel
                        draw.line([(u - 4, v - 4), (u + 4, v + 4)], fill=COLORS[index], width=2)
                        draw.line([(u - 4, v + 4), (u + 4, v - 4)], fill=COLORS[index], width=2)
            buffer = io.BytesIO(); image.save(buffer, format="PNG"); images.append((f"preview_{view}", buffer.getvalue()))
        (self.run / "previews").mkdir(exist_ok=True)
        (self.run / "previews" / "full.json").write_text(json.dumps(rows, indent=1))
        from .actions import MAX_FULL_SLOTS
        seq_schema = {"type": "array", "minItems": 1, "maxItems": MAX_FULL_SLOTS, "items": action_schema(interface=self.interface, allow_stop=False)}
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {"reasoning": {"type": "string"}, "choice": {"type": "string", "enum": [*LABELS[:len(rows)], "revise"]},
                                 "sequence": {"anyOf": [seq_schema, {"type": "null"}]}},
                  "required": ["reasoning", "choice", "sequence"]}
        text = ctx["user_text"] + "\n\n" + json.dumps({"original_proposal": call["parsed"], "preview_of_your_candidate_sequences": rows,
                           "instruction": "Kinematic preview only (no contacts simulated). Paths connect predicted slot endpoints; crosses mark final targets. "
                                          "Choose A/B/C, or choice=\"revise\" with a revised full sequence."}, separators=(",", ":"))
        from .policy import MAX_TOKENS_FULL
        second = ctx["client"].complete(system_prompt=ctx["system_prompt"], user_text=text, images=_selection_images(ctx, images), response_schema=schema,
                                        max_tokens=MAX_TOKENS_FULL, category="preview_select", decision=ctx["decision"])
        choice = second["parsed"].get("choice")
        chosen = candidates[LABELS.index(choice)] if choice in LABELS and LABELS.index(choice) < len(candidates) else second["parsed"].get("sequence")
        with (self.run / "preview-choices.jsonl").open("a") as handle:
            handle.write(json.dumps({"decision": ctx["decision"], "candidates": rows, "choice": choice}) + "\n")
        return {**second, "parsed": {"reasoning": second["parsed"].get("reasoning"), "sequence": chosen or []}, "truncated": call.get("truncated") or second.get("truncated")}

    def choose(self, ctx, call):
        """Second counted call: preview -> chosen or revised action (raw dict)."""
        candidates = call["parsed"].get("candidates") or []
        rows, images = self._preview(ctx, candidates)
        (self.run / "previews").mkdir(exist_ok=True)
        (self.run / "previews" / f"{ctx['decision']:03d}.json").write_text(json.dumps(rows, indent=1))
        candidate_schema = ctx.get("method", self).short_schema(self.interface)["properties"]["candidates"]["items"]
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {"reasoning": {"type": "string"}, "choice": {"type": "string", "enum": [*LABELS[:len(rows)], "revise"]},
                                 "action": {"anyOf": [candidate_schema, {"type": "null"}]}},
                  "required": ["reasoning", "choice", "action"]}
        text = ctx["user_text"] + "\n\n" + json.dumps({"original_proposal": call["parsed"], "preview_of_your_candidates": rows,
                           "instruction": "Choose A/B/C, or choice=\"revise\" with a revised action. The preview images mark each "
                                          "candidate's predicted slot endpoint with a circle and final target with a cross (A red, B green, C blue)."}, separators=(",", ":"))
        second = ctx["client"].complete(system_prompt=ctx["system_prompt"], user_text=text, images=_selection_images(ctx, images), response_schema=schema,
                                        max_tokens=MAX_TOKENS_SHORT, category="preview_select", decision=ctx["decision"])
        choice = second["parsed"].get("choice")
        record = {"candidates": rows, "choice": choice, "revised": second["parsed"].get("action")}
        with (self.run / "preview-choices.jsonl").open("a") as handle:
            handle.write(json.dumps({"decision": ctx["decision"], **record}) + "\n")
        if choice in LABELS and LABELS.index(choice) < len(candidates):
            return candidates[LABELS.index(choice)]
        return second["parsed"].get("action")


# ---------------------------------------------------------------- H8 -------
class H8ExplicitRecovery(Method):
    """Every action carries an observable expected effect; the next turn reports
    the deterministic comparison and asks for a recovery action on mismatch.
    (LIBERO-RECOVER inspires only this branch.)"""
    name = "h8"
    POSITION_TOLERANCE_M = 0.03
    WIDTH_TOLERANCE_M = 0.01

    def __init__(self, *, run, config):
        super().__init__(run=run, config=config)
        self.last_check = None
        self.mismatches = 0
        self.checks = 0

    def short_schema(self, interface):
        from .actions import short_response_schema
        expect = {"type": "object", "additionalProperties": False, "properties": {
            "tcp_world_m": {"anyOf": [{"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}, {"type": "null"}]},
            "gripper_width_m": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "holding_object": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
            "note": {"type": "string"}},
            "required": ["tcp_world_m", "gripper_width_m", "holding_object", "note"]}
        return short_response_schema(interface=interface, top_extra={"expect": expect})

    def prompt_suffix(self) -> str:
        return ("EXPLICIT RECOVERY (this condition): with every action also return `expect`, the observable effect you predict after it: "
                "`tcp_world_m` (expected measured TCP), `gripper_width_m` (expected finger gap, e.g. 0.06 after closing on a 6 cm object, "
                "0.002 if closing on nothing), `holding_object` (true/false) and a short `note` naming the evidence you will check. "
                "The next prompt carries `expectation_check` comparing your prediction with the measurement (tolerance 0.03 m / 0.01 m). "
                "Its matched/mismatch status applies only to the listed `checked_effects` (TCP and finger gap). `holding_check` is unknown "
                "when no public evidence establishes object retention; agreement in TCP and finger gap alone does not verify holding. "
                "On a mismatch, first explain in `reasoning` what physically happened (missed grasp, collision, object slipped, wrong "
                "target) and then output a RECOVERY action that repairs the situation (re-open and re-approach, lift and re-grasp, back off "
                "from contact) instead of continuing the original plan blindly.")

    def observe(self, ctx):
        return ({"expectation_check": self.last_check} if self.last_check else {}), []

    def after_receipt(self, ctx, action, receipt):
        call = ctx.get("call")
        expect = call["parsed"].get("expect") if call else None
        if not expect or receipt.child is None:
            self.last_check = None
            return
        measured_tcp = receipt.child["tcp_world_after_m"]
        measured_width = receipt.child["gripper_width_after_m"]
        deviations = {}
        checked_effects = []
        if expect.get("tcp_world_m") is not None:
            deviations["tcp_error_m"] = round(float(np.linalg.norm(np.asarray(expect["tcp_world_m"]) - np.asarray(measured_tcp))), 3)
            checked_effects.append("tcp_world_m")
        if expect.get("gripper_width_m") is not None and measured_width is not None:
            deviations["width_error_m"] = round(abs(float(expect["gripper_width_m"]) - float(measured_width)), 3)
            checked_effects.append("gripper_width_m")
        mismatch = (deviations.get("tcp_error_m", 0) > self.POSITION_TOLERANCE_M or deviations.get("width_error_m", 0) > self.WIDTH_TOLERANCE_M)
        self.checks += 1
        self.mismatches += int(mismatch)
        self.last_check = {"status": "mismatch" if mismatch else "matched" if checked_effects else "unknown", "expected": expect,
                           "checked_effects": checked_effects,
                           "holding_check": {"status": "unknown", "expected": expect.get("holding_object"), "measured": None,
                                             "reason": "TCP and finger gap do not establish whether an object is retained."},
                           "measured": {"tcp_world_m": [round(v, 3) for v in measured_tcp], "gripper_width_m": round(measured_width, 4) if measured_width is not None else None,
                                        "execution_status": receipt.status}, "deviation": deviations}
        with (self.run / "expectation-checks.jsonl").open("a") as handle:
            handle.write(json.dumps({"decision": ctx["decision"], **self.last_check}) + "\n")

    def finalize(self):
        return {"expectation_checks": self.checks, "mismatches": self.mismatches}


# ---------------------------------------------------------------- H6/H7 ----
class BankMethod(Method):
    """Shared retrieval for H6 (failure experience) and H7 (successful skills).
    Banks are JSON files built from clean development runs (see direct/banks.py)."""
    token_cap = 700

    def __init__(self, *, run, config):
        super().__init__(run=run, config=config)
        bank_path = config.get("bank")
        if not bank_path or not Path(bank_path).exists():
            raise RuntimeError(f"{self.name} requires a bank file built from clean development runs (config 'bank'); prerequisite missing")
        self.bank = json.loads(Path(bank_path).read_text())
        self.retrieved: list[dict] = []

    def _retrieve(self, ctx, max_items):
        task = ctx["observation"]["instruction"]
        state = ctx["observation"]["public_state"]
        phase_hint = "holding" if (state["gripper_width_m"] or 0) > 0.005 and int(round(float(ctx["observation"]["gripper_command"]))) == 0 else "free"
        scored = []
        for entry in self.bank.get("entries", []):
            score = 0.0
            if entry.get("task") == ctx["observation"]["task"] if "task" in ctx["observation"] else False:
                score += 2.0
            if entry.get("phase") == phase_hint:
                score += 1.0
            score += 0.5 * len(set(task.lower().split()) & set(entry.get("instruction", "").lower().split())) / max(1, len(task.split()))
            scored.append((score, entry))
        scored.sort(key=lambda item: -item[0])
        chosen, budget = [], self.token_cap
        for _, entry in scored:
            text = json.dumps(entry["record"], separators=(",", ":"))
            cost = len(text) // 4
            if cost > budget:
                continue
            chosen.append(entry["record"]); budget -= cost
            if len(chosen) >= max_items:
                break
        self.retrieved.append({"decision": ctx["decision"], "count": len(chosen)})
        return chosen


class H6FailureExperience(BankMethod):
    name = "h6"

    def prompt_suffix(self) -> str:
        return ("FAILURE EXPERIENCE (this condition): `failure_experience` lists at most two records from earlier clean attempts in similar "
                "situations: the context, the action taken, its actual measured effect and, when one was verified, the correction that "
                "worked. Records marked `correction_verified: false` are hypotheses. Use them to avoid repeating the same mistake.")

    def observe(self, ctx):
        return {"failure_experience": self._retrieve(ctx, 2)}, []

    def finalize(self):
        return {"retrievals": self.retrieved}


class H7SuccessfulSkills(BankMethod):
    name = "h7"

    def prompt_suffix(self) -> str:
        return ("SUCCESSFUL SKILLS (this condition): `successful_skills` lists interaction fragments that worked in earlier clean runs, "
                "with their applicability conditions, source-scene absolute world poses, and gripper timing. These source coordinates "
                "are examples from a different observation; local grasp-and-lift success is distinct from complete_task_success. "
                "Use the timing and interaction pattern as context and generate fresh numerical targets from the current observation.")

    def observe(self, ctx):
        return {"successful_skills": self._retrieve(ctx, 2)}, []

    def finalize(self):
        return {"retrievals": self.retrieved}


REGISTRY = {"h3": H3ProposePreview, "h8": H8ExplicitRecovery, "h6": H6FailureExperience, "h7": H7SuccessfulSkills}


def make(name: str, *, run: Path, config: Mapping[str, object]) -> Method:
    if name not in REGISTRY:
        raise ValueError(f"unknown method {name!r}")
    return REGISTRY[name](run=run, config=config)
