"""H3 (propose and preview), H8 (explicit recovery), and the bank-based H6/H7."""
from __future__ import annotations

import io
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .actions import Action, action_schema, decode_action, ee_rotation_matrix
from .kinematics import panda_fk, plan_pose_segment, solve_pose_multistart, world_to_base, base_to_world
from .methods import Method
from .policy import MAX_TOKENS_SHORT, MalformedOutput

LABELS = ("A", "B", "C")
COLORS = ((255, 0, 0), (0, 200, 0), (0, 128, 255))


def _project(calibration, point):
    from robocasa_inspect.camera_geometry import project_world_point
    try:
        u, v, _ = project_world_point(calibration, point)
        return (u, v) if 0 <= u < 256 and 0 <= v < 256 else None
    except ValueError:
        return None


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
                "The executor previews each candidate with forward/inverse kinematics only (reachability, predicted TCP position, "
                "the target drawn as a labelled marker in the current images). You then receive the preview and choose one "
                "(`choice` A/B/C) or output a revised action (`choice` \"revise\" with `action`). No physical outcome is simulated. "
                "The second call is counted against the decision budget. In full mode the candidates are 2 to 3 complete sequences and the "
                "preview draws each predicted TCP path.")

    def _preview(self, ctx, candidates):
        obs = ctx["observation"]; state = obs["public_state"]; cal = obs["camera_calibration"]
        q = state["arm_q_rad"]; rows = []; markers = {label: [] for label in ("left", "right", "wrist")}
        for index, raw in enumerate(candidates[:3]):
            label = LABELS[index]
            row = {"label": label, "candidate": raw}
            try:
                action = decode_action(raw, interface=self.interface, representation=self.representation,
                                       current_tcp_world=state["tcp_world_position_m"], current_q=q)
            except ValueError as error:
                row.update({"valid": False, "error": str(error)}); rows.append(row); continue
            row["valid"] = True
            predicted = None
            if action.kind == "ee":
                p_base, r_base = world_to_base(action.position_m, ee_rotation_matrix(action), state["base_world_position_m"], state["base_world_quat_xyzw"])
                plan = plan_pose_segment(q, p_base, r_base)
                if plan["status"] == "kinematically_reachable":
                    row.update({"reachable": True, "path": "straight_line", "distance_m": round(plan["segment_length_m"], 3)})
                    predicted = list(action.position_m)
                else:
                    direct = solve_pose_multistart(q, p_base, r_base)
                    ok = direct["status"] == "kinematically_reachable" and direct["max_joint_delta_rad"] <= 1.6
                    row.update({"reachable": ok, "path": "joint_space_interpolation" if ok else None,
                                "distance_m": round(plan["segment_length_m"], 3),
                                "ik_residual_m": round(direct["position_error_m"], 4)})
                    predicted = list(action.position_m) if ok else None
            elif action.kind == "joint":
                p, r = panda_fk(action.q_rad)
                w, _ = base_to_world(p, r, state["base_world_position_m"], state["base_world_quat_xyzw"])
                predicted = [float(v) for v in w]
                row.update({"reachable": True, "max_joint_delta_rad": round(max(abs(a - b) for a, b in zip(action.q_rad, q)), 3)})
            elif action.kind == "base":
                yaw = state["base_world_yaw_rad"]; d = 0.047 * action.velocity / 0.5
                tcp = np.asarray(state["tcp_world_position_m"], float)
                if action.axis == "x":
                    predicted = (tcp + d * np.array([np.cos(yaw), np.sin(yaw), 0])).tolist()
                elif action.axis == "y":
                    predicted = (tcp + d * np.array([-np.sin(yaw), np.cos(yaw), 0])).tolist()
                else:
                    predicted = tcp.tolist()
                row.update({"reachable": True, "predicted_base_move_m": round(d, 3) if action.axis != "yaw" else None,
                            "predicted_yaw_delta_rad": round(0.216 * action.velocity / 0.5, 3) if action.axis == "yaw" else None})
            else:
                predicted = list(state["tcp_world_position_m"]); row["reachable"] = True
            if predicted is not None:
                row["predicted_tcp_world_m"] = [round(float(v), 3) for v in predicted]
                for view in markers:
                    pixel = _project(cal[view], predicted)
                    if pixel is not None:
                        markers[view].append((label, pixel, index))
                row["predicted_tcp_pixels"] = {view: [round(p[1][0]), round(p[1][1])] for view in markers for p in markers[view] if p[0] == label}
            rows.append(row)
        images = []
        for view in ("left", "right", "wrist"):
            image = Image.open(io.BytesIO(ctx["images"][view])).convert("RGB")
            draw = ImageDraw.Draw(image)
            for label, (u, v), index in markers[view]:
                color = COLORS[index]
                draw.ellipse([u - 5, v - 5, u + 5, v + 5], outline=color, width=2)
                draw.text((u + 6, v - 6), label, fill=color)
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
        for index, sequence in enumerate(candidates[:3]):
            tcp = list(state["tcp_world_position_m"]); q = list(state["arm_q_rad"]); yaw = state["base_world_yaw_rad"]
            points, first_problem = [tuple(tcp)], None
            for slot, raw in enumerate(sequence, start=1):
                try:
                    action = decode_action(raw, interface=self.interface, current_tcp_world=tcp, current_q=q)
                except ValueError as error:
                    first_problem = {"slot": slot, "error": str(error)}; break
                if action.kind == "ee":
                    p_base, r_base = world_to_base(action.position_m, ee_rotation_matrix(action), state["base_world_position_m"], state["base_world_quat_xyzw"])
                    plan = plan_pose_segment(q, p_base, r_base)
                    if plan["status"] != "kinematically_reachable":
                        direct = solve_pose_multistart(q, p_base, r_base)
                        if not (direct["status"] == "kinematically_reachable" and direct["max_joint_delta_rad"] <= 1.6):
                            first_problem = {"slot": slot, "error": "unreachable"}; break
                        q = direct["q"]
                    else:
                        q = plan["waypoints"][-1]
                    tcp = list(action.position_m)
                elif action.kind == "joint":
                    q = list(action.q_rad); p, r = panda_fk(q)
                    tcp = [float(v) for v in base_to_world(p, r, state["base_world_position_m"], state["base_world_quat_xyzw"])[0]]
                elif action.kind == "base":
                    d = 0.66 * max(0.0, abs(action.velocity) - 0.25) * np.sign(action.velocity)
                    if action.axis == "x":
                        tcp = (np.asarray(tcp) + d * np.array([np.cos(yaw), np.sin(yaw), 0])).tolist()
                    elif action.axis == "y":
                        tcp = (np.asarray(tcp) + d * np.array([-np.sin(yaw), np.cos(yaw), 0])).tolist()
                points.append(tuple(tcp))
            rows.append({"label": LABELS[index], "length": len(sequence), "kinematic_check": "all reachable" if first_problem is None else first_problem,
                         "predicted_tcp_path_m": [[round(v, 3) for v in p] for p in points]})
            paths.append(points)
        images = []
        for view in ("left", "right", "wrist"):
            image = Image.open(io.BytesIO(ctx["images"][view])).convert("RGB"); draw = ImageDraw.Draw(image)
            for index, points in enumerate(paths):
                pixels = [_project(cal[view], p) for p in points]
                for a, b in zip(pixels, pixels[1:]):
                    if a and b:
                        draw.line([a, b], fill=COLORS[index], width=2)
                if pixels and pixels[-1]:
                    draw.text(pixels[-1], LABELS[index], fill=COLORS[index])
            buffer = io.BytesIO(); image.save(buffer, format="PNG"); images.append((f"preview_{view}", buffer.getvalue()))
        (self.run / "previews").mkdir(exist_ok=True)
        (self.run / "previews" / "full.json").write_text(json.dumps(rows, indent=1))
        from .actions import MAX_FULL_SLOTS
        seq_schema = {"type": "array", "minItems": 1, "maxItems": MAX_FULL_SLOTS, "items": action_schema(interface=self.interface, allow_stop=False)}
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {"reasoning": {"type": "string"}, "choice": {"type": "string", "enum": [*LABELS[:len(rows)], "revise"]},
                                 "sequence": {"anyOf": [seq_schema, {"type": "null"}]}},
                  "required": ["reasoning", "choice", "sequence"]}
        text = json.dumps({"preview_of_your_candidate_sequences": rows,
                           "instruction": "Kinematic preview only (no contacts simulated). Choose A/B/C, or choice=\"revise\" with a revised full sequence."}, separators=(",", ":"))
        from .policy import MAX_TOKENS_FULL
        second = ctx["client"].complete(system_prompt=ctx["system_prompt"], user_text=text, images=images, response_schema=schema,
                                        max_tokens=MAX_TOKENS_FULL, category="preview_select", decision=1)
        choice = second["parsed"].get("choice")
        chosen = candidates[LABELS.index(choice)] if choice in LABELS and LABELS.index(choice) < len(candidates) else second["parsed"].get("sequence")
        with (self.run / "preview-choices.jsonl").open("a") as handle:
            handle.write(json.dumps({"decision": 1, "candidates": rows, "choice": choice}) + "\n")
        return {**second, "parsed": {"reasoning": second["parsed"].get("reasoning"), "sequence": chosen or []}, "truncated": call.get("truncated") or second.get("truncated")}

    def choose(self, ctx, call):
        """Second counted call: preview -> chosen or revised action (raw dict)."""
        candidates = call["parsed"].get("candidates") or []
        rows, images = self._preview(ctx, candidates)
        (self.run / "previews").mkdir(exist_ok=True)
        (self.run / "previews" / f"{ctx['decision']:03d}.json").write_text(json.dumps(rows, indent=1))
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {"reasoning": {"type": "string"}, "choice": {"type": "string", "enum": [*LABELS[:len(rows)], "revise"]},
                                 "action": {"anyOf": [action_schema(interface=self.interface), {"type": "null"}]}},
                  "required": ["reasoning", "choice", "action"]}
        text = json.dumps({"preview_of_your_candidates": rows,
                           "instruction": "Choose A/B/C, or choice=\"revise\" with a revised action. The preview images mark each "
                                          "candidate's predicted TCP position (A red, B green, C blue)."}, separators=(",", ":"))
        second = ctx["client"].complete(system_prompt=ctx["system_prompt"], user_text=text, images=images, response_schema=schema,
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
        if expect.get("tcp_world_m") is not None:
            deviations["tcp_error_m"] = round(float(np.linalg.norm(np.asarray(expect["tcp_world_m"]) - np.asarray(measured_tcp))), 3)
        if expect.get("gripper_width_m") is not None and measured_width is not None:
            deviations["width_error_m"] = round(abs(float(expect["gripper_width_m"]) - float(measured_width)), 3)
        mismatch = (deviations.get("tcp_error_m", 0) > self.POSITION_TOLERANCE_M or deviations.get("width_error_m", 0) > self.WIDTH_TOLERANCE_M)
        self.checks += 1
        self.mismatches += int(mismatch)
        self.last_check = {"status": "mismatch" if mismatch else "matched", "expected": expect,
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
                "with their applicability conditions, the relative geometry between TCP and object (world-frame offsets), and the "
                "gripper timing. They are references, not commands: generate fresh numerical targets from the current observation.")

    def observe(self, ctx):
        return {"successful_skills": self._retrieve(ctx, 2)}, []

    def finalize(self):
        return {"retrievals": self.retrieved}


REGISTRY = {"h3": H3ProposePreview, "h8": H8ExplicitRecovery, "h6": H6FailureExperience, "h7": H7SuccessfulSkills}


def make(name: str, *, run: Path, config: Mapping[str, object]) -> Method:
    if name not in REGISTRY:
        raise ValueError(f"unknown method {name!r}")
    return REGISTRY[name](run=run, config=config)
