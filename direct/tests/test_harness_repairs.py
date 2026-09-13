"""Focused harness regressions; no model or simulator services are used."""
import copy
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from direct.actions import Action, decode_action
from direct.harnesses import H1VisualMarkers, H4ExecutionTiming, H4Fixed5
from direct.harnesses_extra import H3ProposePreview, H8ExplicitRecovery
from direct.kinematics import base_to_world, matrix_to_quat_xyzw, panda_fk
from direct.methods import make_method


VIEWS = ("left", "right", "wrist")
Q = [0.0, -0.35, 0.0, -2.0, 0.0, 1.65, 0.785]


def yaw_rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class CaptureClient:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, **request):
        self.requests.append(copy.deepcopy(request))
        return next(self.responses)


class HarnessRepairs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="harness-repairs-")
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.method = H3ProposePreview(run=self.run, config={})
        self.method.short_schema("ee")
        base = [1.0, -2.0, 0.3]
        yaw = math.pi / 2
        quat = matrix_to_quat_xyzw(yaw_rotation(yaw))
        tcp, orientation = base_to_world(*panda_fk(Q), base, quat)
        state = {"arm_q_rad": list(Q), "base_world_position_m": base,
                 "base_world_quat_xyzw": quat, "base_world_yaw_rad": yaw,
                 "tcp_world_position_m": tcp.tolist(),
                 "tcp_world_quat_xyzw": matrix_to_quat_xyzw(orientation),
                 "gripper_width_m": 0.078}
        calibration = {"camera_position_world_m": [1, -2, 5], "camera_xmat_world": np.eye(3).tolist(),
                       "cx_px": 127.5, "cy_px": 127.5, "fx_px": 100, "fy_px": 100}
        image = io.BytesIO()
        Image.new("RGB", (256, 256), "white").save(image, format="PNG")
        images = {view: image.getvalue() for view in VIEWS}
        # Preserve labels and bytes from the actual first request, including H1/extras/history.
        window = [(f"current_{view}_annotated", images[view]) for view in VIEWS]
        window += [(f"before_previous_action_{view}", b"previous-" + view.encode()) for view in VIEWS]
        window += [("region_gallery", b"gallery")]
        self.ctx = {"decision": 4, "observation": {"public_state": state, "gripper_command": 1,
                    "camera_calibration": {v: calibration for v in VIEWS}}, "images": images,
                    "system_prompt": "Frozen numerical control contract", "image_window": window,
                    "user_text": json.dumps({"task": "Move the object to the receptacle", "state": state,
                                             "previous": {"action": {"k": "hold", "g": 0},
                                                          "result": {"status": "partial"}},
                                             "working_memory": {"current_goal": "inspect the previous result"}})}

    def test_h1_gallery_keeps_the_last_of_fourteen_region_crops(self):
        source = Image.new("RGB", (256, 256), "black")
        source.paste((255, 0, 0), (200, 200, 220, 220))
        data = io.BytesIO()
        source.save(data, format="PNG")
        regions = [{"id": f"r{i + 1}", "world_m": [0, 0, 0], "mean_rgb": [0, 0, 0],
                    "views": {"left": {"bbox_xywh": [0, 0, 8, 8], "area_px": 64}}}
                   for i in range(14)]
        regions[-1]["views"]["left"] = {"bbox_xywh": [200, 200, 20, 20], "area_px": 400}
        regions[-1]["mean_rgb"] = [255, 0, 0]
        method = H1VisualMarkers(run=self.run, config={})
        gallery = np.asarray(Image.open(io.BytesIO(method._gallery({"regions": regions}, {"left": data.getvalue()}))))
        red = (gallery[:, :, 0] == 255) & (gallery[:, :, 1] == 0) & (gallery[:, :, 2] == 0)
        self.assertGreaterEqual(int(red.sum()), 400, "the final proposal's crop must be visible")

    def proposal_then_select(self, method, candidates, *, full=False, choice="A", revised=None):
        proposal = {"parsed": {"reasoning": "Keep the candidate gripper ordering intact.", "candidates": candidates},
                    "truncated": False}
        key = "sequence" if full else "action"
        selected = {"parsed": {"reasoning": "Select after reviewing the preview.", "choice": choice, key: revised},
                    "truncated": False}
        client = CaptureClient(proposal, selected)
        self.ctx["client"] = client
        schema = method.full_schema("ee") if full else method.short_schema("ee")
        call = client.complete(system_prompt=self.ctx["system_prompt"], user_text=self.ctx["user_text"],
                               images=self.ctx["image_window"], response_schema=schema,
                               max_tokens=4096 if full else 1024, category="control", decision=self.ctx["decision"])
        result = method.revise_sequence(self.ctx, call) if full else method.choose(self.ctx, call)
        return result, proposal, client.requests

    def assert_retained_request(self, proposal, requests):
        self.assertEqual(len(requests), 2)
        first, second = requests
        self.assertTrue(second["user_text"].startswith(first["user_text"] + "\n\n"))
        feedback = json.loads(second["user_text"][len(first["user_text"]) + 2:])
        self.assertEqual(feedback["original_proposal"], proposal["parsed"])
        self.assertEqual(len(second["images"]), len(first["images"]))
        self.assertLessEqual(len(second["images"]), 8)
        for original, selected in zip(first["images"], second["images"]):
            label, _ = original
            if label.startswith("current_"):
                self.assertEqual(selected[0], label.replace("current_", "preview_", 1).removesuffix("_annotated"))
            else:
                self.assertEqual(selected, original)
        self.assertEqual(second["system_prompt"], first["system_prompt"])
        self.assertEqual(second["decision"], first["decision"])
        self.assertEqual(second["category"], "preview_select")

    def test_short_selection_retains_exact_first_payload_images_and_reasoning(self):
        candidates = [{"k": "hold", "g": 0}, {"k": "hold", "g": 1}]
        result, proposal, requests = self.proposal_then_select(self.method, candidates, choice="B")
        self.assert_retained_request(proposal, requests)
        self.assertEqual(result, candidates[1])

    def test_full_selection_retains_complete_sequences_and_first_context(self):
        candidates = [[{"k": "hold", "g": 0, "n": "close"}, {"k": "hold", "g": 1, "n": "release"}],
                      [{"k": "hold", "g": 1}]]
        result, proposal, requests = self.proposal_then_select(self.method, candidates, full=True)
        self.assert_retained_request(proposal, requests)
        self.assertEqual(result["parsed"]["sequence"], candidates[0])

    def test_selection_replaces_current_views_and_preserves_annotations_and_history(self):
        annotated = Image.new("RGB", (256, 256), "white")
        annotated.putpixel((8, 8), (255, 0, 255))
        data = io.BytesIO()
        annotated.save(data, format="PNG")
        self.ctx["image_window"][0] = ("current_left_annotated", data.getvalue())
        _, proposal, requests = self.proposal_then_select(self.method, [{"k": "hold"}, {"k": "hold", "g": 0}])
        self.assert_retained_request(proposal, requests)
        images = dict(requests[1]["images"])
        self.assertEqual(Image.open(io.BytesIO(images["preview_left"])).getpixel((8, 8)), (255, 0, 255))
        for view in VIEWS:
            self.assertIn(f"preview_{view}", images)
            self.assertEqual(images[f"before_previous_action_{view}"], b"previous-" + view.encode())
        self.assertEqual(images["region_gallery"], b"gallery")

    def test_h2_h3_h4_revision_keeps_timing_schema_and_relative_representation(self):
        combo = make_method("h2+h3+h4", run=self.run, config={})
        self.ctx["method"] = combo
        candidates = [{"k": "hold", "g": 1, "s": 5}, {"k": "hold", "g": 0, "s": 10}]
        revised = {"k": "hold", "g": 1, "s": 10}
        result, proposal, requests = self.proposal_then_select(combo, candidates, choice="revise", revised=revised)
        self.assertEqual(requests[1]["response_schema"]["properties"]["action"]["anyOf"][0],
                         requests[0]["response_schema"]["properties"]["candidates"]["items"])
        self.assertEqual(result, revised)
        self.assertEqual(combo.h3.representation, "relative")
        self.assert_retained_request(proposal, requests)

    def test_h4_forces_twenty_only_for_changed_or_unknown_gripper(self):
        method = H4ExecutionTiming(run=self.run, config={})
        for current in (0, 1, None):
            for requested in (0, 1, None):
                for steps in (5, 10, 20):
                    with self.subTest(current=current, requested=requested, steps=steps):
                        action = Action("hold", gripper=requested, raw={"s": steps})
                        expected = 20 if requested is not None and requested != current else steps
                        self.assertEqual(method.slot_steps(action, current_gripper=current), expected)
        # Old callers without a known current command retain the conservative default.
        self.assertEqual(method.slot_steps(Action("hold", gripper=1, raw={"s": 5})), 20)
        self.assertEqual(method.slot_steps(Action("hold", raw={"s": 7}), current_gripper=1), 20)

    def test_h4c_unchanged_explicit_gripper_keeps_five_steps(self):
        method = H4Fixed5(run=self.run, config={})
        for current in (0, 1, None):
            for requested in (0, 1, None):
                with self.subTest(current=current, requested=requested):
                    action = Action("hold", gripper=requested)
                    expected = 20 if requested is not None and requested != current else 5
                    self.assertEqual(method.slot_steps(action, current_gripper=current), expected)
        self.assertEqual(method.slot_steps(Action("hold", gripper=0)), 20)

    def test_short_base_preview_matches_current_calibration_and_heading(self):
        tcp = np.asarray(self.ctx["observation"]["public_state"]["tcp_world_position_m"])
        for axis, direction in (("x", np.array([0, 1, 0])), ("y", np.array([-1, 0, 0]))):
            for velocity in (-0.5, -0.25, 0.1, 0.25, 0.5):
                with self.subTest(axis=axis, velocity=velocity):
                    rows, _ = self.method._preview(self.ctx, [{"k": "base", "a": axis, "v": velocity}])
                    distance = 0.66 * max(0, abs(velocity) - 0.25) * np.sign(velocity)
                    self.assertAlmostEqual(rows[0]["predicted_base_move_m"], distance, places=3)
                    np.testing.assert_allclose(rows[0]["predicted_tcp_world_m"], tcp + distance * direction, atol=0.0006)
                    if abs(velocity) == 0.5:
                        self.assertLess(abs(abs(rows[0]["predicted_base_move_m"]) - 0.1593), 0.007)
        for velocity in (-0.5, 0.5):
            rows, _ = self.method._preview(self.ctx, [{"k": "base", "a": "yaw", "v": velocity}])
            self.assertAlmostEqual(rows[0]["predicted_yaw_delta_rad"], 0.4637 * velocity / 0.5, places=3)

    def test_yaw_preview_interpolates_signed_measured_curve(self):
        # Measured 16-step yaw, plus a midpoint to check interpolation between samples.
        for speed, delta in ((0.1, 0.015591), (0.15, 0.046706), (0.3, 0.226537),
                             (0.35, (0.226537 + 0.34556) / 2), (0.4, 0.34556), (0.5, 0.4637)):
            for sign in (-1, 1):
                with self.subTest(velocity=sign * speed):
                    rows, _ = self.method._preview(self.ctx, [{"k": "base", "a": "yaw", "v": sign * speed}])
                    self.assertAlmostEqual(rows[0]["predicted_yaw_delta_rad"], sign * delta, places=4)
                    if speed == 0.1:
                        state = self.ctx["observation"]["public_state"]
                        self.assertGreater(np.linalg.norm(np.asarray(rows[0]["predicted_base_world_m"]) - state["base_world_position_m"]), 0)

    def test_h4_and_h4c_base_actions_keep_full_slot(self):
        for name in ("h4", "h4c", "h3+h4", "h3+h4c"):
            method = make_method(name, run=self.run, config={})
            for steps in (5, 10, 20):
                for gripper in (None, 0, 1):
                    action = Action("base", axis="x", velocity=0.5, gripper=gripper, raw={"s": steps})
                    self.assertEqual(method.slot_steps(action, current_gripper=1), 20,
                                     (name, steps, gripper))

    def test_base_preview_uses_full_h4_and_h4c_powered_duration(self):
        for name in ("h4", "h4c"):
            self.ctx["method"] = make_method(name, run=self.run, config={})
            for gripper in (None, 0, 1):
                rows, _ = self.method._preview(self.ctx, [{"k": "base", "a": "x", "v": 0.5, "g": gripper, "s": 5}])
                self.assertAlmostEqual(rows[0]["predicted_base_move_m"], 0.165, places=3)
                self.assertEqual(rows[0]["slot_steps"], 20)
                self.assertEqual(rows[0]["motion_steps"], 16)

    def test_full_preview_advances_base_position_yaw_and_ee_transform(self):
        state = self.ctx["observation"]["public_state"]
        original = copy.deepcopy(state)
        delta = 0.226537  # Measured yaw at v=.3.
        turned_yaw = state["base_world_yaw_rad"] + delta
        rotated = yaw_rotation(turned_yaw)
        base = np.asarray(state["base_world_position_m"]) + [0, 0.165, 0]
        tcp_after_translation = np.asarray(state["tcp_world_position_m"]) + [0, 0.165, 0]
        pivot_offset = np.array([-0.21, 0, 0])
        pivot = base + yaw_rotation(state["base_world_yaw_rad"]) @ pivot_offset
        tcp_after_yaw = pivot + yaw_rotation(delta) @ (tcp_after_translation - pivot)
        base = pivot - rotated @ pivot_offset + 0.165 * rotated[:, 1]
        target, orientation = base_to_world(*panda_fk(Q), base, matrix_to_quat_xyzw(rotated))
        sequence = [{"k": "base", "a": "x", "v": 0.5}, {"k": "base", "a": "yaw", "v": 0.3},
                    {"k": "base", "a": "y", "v": 0.5},
                    {"k": "ee", "p": target.tolist(), "o": matrix_to_quat_xyzw(orientation)}]
        plan = {"status": "kinematically_reachable", "waypoints": [Q], "segment_length_m": 0.0}
        with patch("direct.harnesses_extra.plan_pose_segment", return_value=plan) as solve:
            self.proposal_then_select(self.method, [sequence, [{"k": "hold"}]], full=True)
        np.testing.assert_allclose(solve.call_args.args[1], panda_fk(Q)[0], atol=1e-10)
        np.testing.assert_allclose(solve.call_args.args[2], panda_fk(Q)[1], atol=1e-10)
        preview = json.loads((self.run / "previews" / "full.json").read_text())[0]
        np.testing.assert_allclose(preview["predicted_tcp_path_m"][2], tcp_after_yaw, atol=0.0006)
        np.testing.assert_allclose(preview["predicted_tcp_path_m"][-1], target, atol=0.0006)
        self.assertEqual(state, original)

    def test_full_yaw_then_translation_uses_offset_pivot(self):
        state = self.ctx["observation"]["public_state"]
        base = np.asarray(state["base_world_position_m"])
        tcp = base + panda_fk(Q)[0]
        state.update(base_world_yaw_rad=0.0, base_world_quat_xyzw=[0, 0, 0, 1],
                     tcp_world_position_m=tcp.tolist(), tcp_world_quat_xyzw=matrix_to_quat_xyzw(panda_fk(Q)[1]))
        original = copy.deepcopy(state)
        # Three full yaw pulses plus a smaller valid pulse make +90 degrees.
        remainder_delta = math.pi / 2 - 3 * 0.4637
        remainder_velocity = 0.15 + (remainder_delta - 0.046706) / (0.226537 - 0.046706) * (0.3 - 0.15)
        sequence = [{"k": "base", "a": "yaw", "v": 0.5} for _ in range(3)]
        sequence += [{"k": "base", "a": "yaw", "v": remainder_velocity},
                     {"k": "base", "a": "x", "v": 0.5}]
        self.proposal_then_select(self.method, [sequence, [{"k": "hold"}]], full=True)
        preview = json.loads((self.run / "previews" / "full.json").read_text())[0]
        # At yaw=0, pivot=base+[-.21,0,0]. After +90deg the offset is [0,-.21,0],
        # so the base displacement is [-.21,+.21,0]; forward then moves along +y.
        yaw_base = base + [-0.21, 0.21, 0]
        pivot = base + [-0.21, 0, 0]
        yaw_tcp = pivot + yaw_rotation(math.pi / 2) @ (tcp - pivot)
        np.testing.assert_allclose(preview["slots"][3]["predicted_base_world_m"], yaw_base, atol=0.0006)
        np.testing.assert_allclose(preview["predicted_tcp_path_m"][4], yaw_tcp, atol=0.0006)
        np.testing.assert_allclose(preview["slots"][4]["predicted_base_world_m"], yaw_base + [0, 0.165, 0], atol=0.0006)
        np.testing.assert_allclose(preview["predicted_tcp_path_m"][-1], yaw_tcp + [0, 0.165, 0], atol=0.0006)
        self.assertAlmostEqual(preview["slots"][4]["predicted_base_yaw_rad"], math.pi / 2, places=4)
        self.assertEqual(state, original)

    def test_joint_preview_separates_commanded_slot_prefix_from_final_target(self):
        self.method.short_schema("joint")
        target_q = [2.0, *Q[1:]]
        row = self.method._preview(self.ctx, [{"k": "joint", "q": target_q}])[0][0]
        state = self.ctx["observation"]["public_state"]
        prefix, _ = base_to_world(*panda_fk([1.6, *Q[1:]]), state["base_world_position_m"], state["base_world_quat_xyzw"])
        target, _ = base_to_world(*panda_fk(target_q), state["base_world_position_m"], state["base_world_quat_xyzw"])
        np.testing.assert_allclose(row["predicted_tcp_world_m"], prefix, atol=0.0006)
        np.testing.assert_allclose(row["final_target_tcp_world_m"], target, atol=0.0006)
        self.assertFalse(row["target_reached_in_slot"])

    def test_full_relative_sequence_starts_next_slot_from_prefix(self):
        self.method.representation = "relative"
        self.method.full_schema("joint")
        sequence = [{"k": "joint", "q": [2.0, 0, 0, 0, 0, 0, 0]},
                    {"k": "joint", "q": [0.1, 0, 0, 0, 0, 0, 0]}]
        self.ctx["client"] = CaptureClient({"parsed": {"choice": "A", "reasoning": "Use A", "sequence": None}})
        self.method.revise_sequence(self.ctx, {"parsed": {"reasoning": "Two deltas", "candidates": [sequence, [{"k": "hold"}]]}})
        preview = json.loads((self.run / "previews" / "full.json").read_text())[0]
        state = self.ctx["observation"]["public_state"]
        expected, _ = base_to_world(*panda_fk([1.7, *Q[1:]]), state["base_world_position_m"], state["base_world_quat_xyzw"])
        self.assertEqual(preview["kinematic_check"], "all reachable")
        np.testing.assert_allclose(preview["predicted_tcp_path_m"][-1], expected, atol=0.0006)

    def test_fallback_above_old_cutoff_is_reachable_with_partial_slot(self):
        state = self.ctx["observation"]["public_state"]
        target_q = [2.0, *Q[1:]]
        target, rotation = base_to_world(*panda_fk(target_q), state["base_world_position_m"], state["base_world_quat_xyzw"])
        action = {"k": "ee", "p": target.tolist(), "o": matrix_to_quat_xyzw(rotation)}
        with patch("direct.harnesses_extra.plan_pose_segment", return_value={"status": "kinematically_unresolved", "segment_length_m": 0.5}), \
             patch("direct.harnesses_extra.solve_pose_multistart", return_value={"status": "kinematically_reachable", "q": target_q,
                                                                               "max_joint_delta_rad": 2.0, "position_error_m": 0.0}):
            row = self.method._preview(self.ctx, [action])[0][0]
            self.assertTrue(row["reachable"])
            self.assertFalse(row["target_reached_in_slot"])
            self.proposal_then_select(self.method, [[action], [{"k": "hold"}]], full=True)
        preview = json.loads((self.run / "previews" / "full.json").read_text())[0]
        self.assertEqual(preview["kinematic_check"], "all reachable")
        self.assertEqual(preview["predicted_tcp_path_m"][-1], row["predicted_tcp_world_m"])

    def test_h8_numeric_match_does_not_verify_holding_and_keeps_mismatch_counts(self):
        method = H8ExplicitRecovery(run=self.run, config={})
        expect = {"tcp_world_m": [1, 2, 3], "gripper_width_m": 0.06, "holding_object": True, "note": "expect a hold"}
        self.ctx["call"] = {"parsed": {"expect": expect}}
        receipt = SimpleNamespace(child={"tcp_world_after_m": [1, 2, 3], "gripper_width_after_m": 0.06}, status="completed")
        method.after_receipt(self.ctx, Action("hold", gripper=0), receipt)
        self.assertEqual(method.last_check["status"], "matched")
        self.assertEqual(method.last_check["checked_effects"], ["tcp_world_m", "gripper_width_m"])
        self.assertEqual(method.last_check["holding_check"]["status"], "unknown")
        self.assertIsNone(method.last_check["holding_check"]["measured"])
        self.assertTrue(method.last_check["holding_check"]["expected"])
        receipt.child["tcp_world_after_m"] = [1.1, 2, 3]
        method.after_receipt(self.ctx, Action("hold", gripper=0), receipt)
        self.assertEqual(method.last_check["status"], "mismatch")
        self.assertEqual(method.finalize(), {"expectation_checks": 2, "mismatches": 1})
        self.assertEqual(method.observe(self.ctx)[0]["expectation_check"], method.last_check)
        # A holding-only expectation supplies no measurable numeric agreement.
        expect.update(tcp_world_m=None, gripper_width_m=None)
        method.after_receipt(self.ctx, Action("hold"), receipt)
        self.assertEqual(method.last_check["status"], "unknown")
        self.assertEqual(method.finalize(), {"expectation_checks": 3, "mismatches": 1})


if __name__ == "__main__":
    unittest.main()
