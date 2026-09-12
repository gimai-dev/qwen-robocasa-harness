"""Evidence regressions; temporary files and a mock simulator, no services."""
import copy
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from direct import banks, recovery
from direct.actions import Action
from direct.episode import Episode
from direct.executor import Receipt


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def observation(sequence=0, steps=0):
    return {"sequence": sequence, "steps_used": steps, "steps_budget": 900,
            "wall_budget_s": 1200, "wall_used_s": 0, "gripper_command": 1,
            "instruction": "Move the object to the sink.", "public_state": {
                "tcp_world_position_m": [1., 2., 1.], "tcp_world_quat_xyzw": [1., 0., 0., 0.],
                "arm_q_rad": [0., -.1, 0., -1., 0., 1., 0.], "gripper_width_m": .08,
                "base_world_position_m": [1., 1.5, .7], "base_world_yaw_rad": 0., "tcp_pixels": {}}}


class MockSim:
    def __init__(self, obs=None):
        self.observation = obs or observation()

    def steps_left(self):
        return self.observation["steps_budget"] - self.observation["steps_used"]


class CaptureMethod:
    representation = "absolute"

    def observe(self, ctx):
        return {"context_marker": "original"}, [("extra", b"extra")]

    def images(self, ctx, images, previous):
        return [("annotated", b"annotated")]

    def slot_steps(self, action, current_gripper=None):
        return 5 if current_gripper == action.gripper else 20

    def choose(self, ctx, call):
        self.ctx = ctx
        return call["parsed"]["action"]

    def revise_sequence(self, ctx, call):
        self.ctx = ctx
        return call

    def after_receipt(self, ctx, action, receipt):
        pass


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def episode(self):
        ep = Episode.__new__(Episode)
        ep.run = self.root / "continuation"
        ep.run.mkdir(exist_ok=True)
        ep.decisions = []
        ep.previous = None
        ep.previous_images = None
        ep.no_motion_streak = ep.rejected = 0
        ep.started = time.monotonic()
        ep.args = SimpleNamespace(interface="ee", max_decisions=1, wall_budget_s=1200)
        ep.system_prompt = "fixed system prompt"
        ep.method = CaptureMethod()
        ep.sim = MockSim(observation(3, 20))
        ep.observe = lambda _: (ep.sim.observation, {"regions": [], "unpaired": {}},
                                {label: b"current" for label in ("left", "right", "wrist")})
        return ep

    def test_history_uses_selected_preaction_sequence_in_all_five_audited_shapes(self):
        for decision, before, final in [(14, 12, 45), (16, 15, 28), (37, 22, 44), (18, 10, 12), (8, 7, 45)]:
            with self.subTest(decision=decision, before=before):
                source = self.root / f"source-{decision}-{before}"
                rows = [{"decision": i, "steps": 20 if i <= before else 0,
                         "action": {"k": "hold"}, "status": "completed"} for i in range(1, decision)]
                rows.append({"decision": decision, "steps": 20, "action": {"k": "hold", "g": 0},
                             "receipt": {"status": "completed"}})
                write_rows(source / "decisions.jsonl", rows)
                for seq in range(final + 1):
                    for label in ("left", "right", "wrist"):
                        p = source / "sim" / "frames" / f"{seq:06d}" / f"{label}.png"
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_bytes(f"sequence-{seq}".encode())
                ep = self.episode()
                ep.load_history(source, decision)
                self.assertEqual(ep.previous_images["left"], f"sequence-{before}".encode())
                self.assertEqual(ep.previous["action"]["g"], 0)

    def test_history_prefers_logged_sequence_and_never_substitutes_future_frame(self):
        source = self.root / "source"
        write_rows(source / "decisions.jsonl", [{"decision": 4, "steps": 0, "observation_sequence": 8,
                   "action": {"k": "ee"}, "receipt": {"status": "unreachable"}}])
        for seq in (0, 8, 99):
            for label in ("left", "right", "wrist"):
                p = source / "sim" / "frames" / f"{seq:06d}" / f"{label}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(str(seq).encode())
        ep = self.episode()
        ep.load_history(source, 4)
        self.assertEqual(ep.previous_images["left"], b"8")
        (source / "sim/frames/000008/left.png").unlink()
        ep.load_history(source, 4)
        self.assertIsNone(ep.previous_images)

    def test_full_source_history_uses_selected_slot(self):
        source = self.root / "full-source"
        write_rows(source / "decisions.jsonl", [
            {"decision": 1, "sequence_length": 2, "observation_sequence": 0},
            {"decision": 1, "slot": 1, "steps": 20, "observation_sequence": 0, "action": {"k": "hold", "g": 0}},
            {"decision": 1, "slot": 2, "steps": 20, "observation_sequence": 1, "action": {"k": "hold", "g": 1}}])
        for seq in (0, 1, 2):
            for label in ("left", "right", "wrist"):
                p = source / "sim/frames" / f"{seq:06d}" / f"{label}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(str(seq).encode())
        ep = self.episode()
        ep.load_history(source, 1, slot=2)
        self.assertEqual(ep.previous["action"], {"k": "hold", "g": 1})
        self.assertEqual(ep.previous_images["left"], b"1")

    def test_short_and_full_preserve_original_h3_context(self):
        for mode in ("short", "full"):
            with self.subTest(mode=mode):
                ep = self.episode()
                requests = []
                raw = {"k": "hold", "g": 1, "s": 5}
                def complete(**kwargs):
                    requests.append(kwargs)
                    return {"parsed": {"action": raw, "sequence": [raw]}}
                ep.client = SimpleNamespace(complete=complete)
                with patch("direct.episode.execute", return_value=Receipt("unreachable", None, 0, {})):
                    getattr(ep, f"run_{mode}")()
                self.assertEqual(ep.method.ctx.get("user_text"), requests[0]["user_text"])
                self.assertEqual(ep.method.ctx.get("image_window"), [("annotated", b"annotated"), ("extra", b"extra")])
                self.assertIs(ep.method.ctx.get("method"), ep.method)

    def test_short_and_full_log_prestate_and_durations_even_when_zero_steps(self):
        for mode in ("short", "full"):
            with self.subTest(mode=mode):
                ep = self.episode()
                raw = {"k": "hold", "g": 1, "s": 5}
                ep.client = SimpleNamespace(complete=lambda **_: {"parsed": {"action": raw, "sequence": [raw]}})
                received = []
                def reject(sim, action, slot_steps):
                    received.append(slot_steps)
                    return Receipt("unreachable", action, 0, {"reason": "test rejection"})
                with patch("direct.episode.execute", side_effect=reject):
                    getattr(ep, f"run_{mode}")()
                row = ep.decisions[-1]
                self.assertEqual(received, [5])
                self.assertEqual(row.get("observation_sequence"), 3)
                self.assertEqual(row.get("state_before", {}).get("tcp_world_position_m"), [1., 2., 1.])
                self.assertEqual((row.get("requested_steps"), row.get("slot_steps"), row.get("executed_steps")), (5, 5, 0))

    def test_full_logs_each_slots_actual_prestate(self):
        ep = self.episode()
        ep.client = SimpleNamespace(complete=lambda **_: {"parsed": {"sequence": [{"k": "hold", "g": 0}, {"k": "hold", "g": 0, "s": 5}]}})
        def move(sim, action, slot_steps):
            obs = copy.deepcopy(sim.observation)
            obs["sequence"] += 1
            obs["steps_used"] += slot_steps
            obs["gripper_command"] = action.gripper
            obs["public_state"]["tcp_world_position_m"][2] += .1
            sim.observation = obs
            return Receipt("completed", action, slot_steps, {}, {
                "tcp_world_after_m": obs["public_state"]["tcp_world_position_m"], "arm_q_after": obs["public_state"]["arm_q_rad"]})
        with patch("direct.episode.execute", side_effect=move):
            ep.run_full()
        first, second = ep.decisions[-2:]
        self.assertEqual(first.get("observation_sequence"), 3)
        self.assertEqual(second.get("observation_sequence"), 4)
        self.assertEqual(second.get("state_before", {}).get("tcp_world_position_m"), [1., 2., 1.1])
        self.assertEqual(second.get("executed_steps"), 5)

    def test_failure_retains_prestate_and_unrelated_move_is_not_verified_correction(self):
        bad = {"decision": 1, "action": {"k": "ee", "p": [9, 9, 9]}, "status": "unreachable", "steps": 0,
               "state_before": {"tcp_world_position_m": [1., 2., 1.], "gripper_command": 1}, "receipt": {}}
        later = {"decision": 2, "action": {"k": "ee", "p": [1, 2, 1]}, "status": "completed", "steps": 20,
                 "receipt": {"tcp_world_after_m": [1, 2, 1]}}
        record = banks.failure_records({"task": "task", "seed": 0}, [bad, later])[0]["record"]
        self.assertIn("[1.0, 2.0, 1.0]", record["context"])
        self.assertFalse(record["correction_verified"])

    def test_h6_retrieves_enabling_motion_and_retry_within_existing_token_cap(self):
        from direct.harnesses_extra import H6FailureExperience

        target = {"k": "ee", "p": [3.994, -.545, 1.291], "o": [0., 0., 0., 1.], "g": 1,
                  "n": "Move TCP above the object with downward orientation to prepare for grasping"}
        prestate = {"tcp_world_position_m": [4.057820442231495, -.6964177685844125, 1.5527610778808594],
                    "tcp_world_quat_xyzw": [-.2736309455170107, .9595473931961647, .06365308921867113, .018525390652606027],
                    "arm_q_rad": [-.00029368587650741674, -.03920439550917757, -.001385436623367855,
                                  -.15198270298045902, -.0015499009995556604, .24546372406162265, .22579195138150798],
                    "base_world_position_m": [4.057149887084961, -.8418753743171692, .6999998688697815],
                    "base_world_yaw_rad": 1.5694942101035054, "gripper_command": 1., "gripper_width_m": .07845143601298332}
        failure = {"decision": 2, "steps": 0, "status": "unreachable", "action": target,
                   "state_before": prestate, "observation_sequence": 1,
                   "receipt": {"reason": "target pose is not reachable by IK from the current configuration",
                               "target_horizontal_distance_from_base_m": .304},
                   "previous": {"action": target, "result": {"status": "partial", "reason": "slot ended before the target was reached",
                                "tcp_world_after_m": prestate["tcp_world_position_m"], "gripper_width_after_m": .0785}}}
        base = {"decision": 3, "steps": 20, "status": "completed",
                "action": {"k": "base", "a": "x", "v": .5, "n": "Move the base closer so the rejected end-effector target becomes reachable"},
                "receipt": {"base_moved_m": [.0001, .1495, 0.], "base_world_after_m": [4.0572, -.6924, .7],
                            "base_yaw_after_rad": 1.5695, "tcp_world_after_m": [4.0579, -.5469, 1.5528],
                            "gripper_width_after_m": .0785, "final_joint_error_rad": .0041}}
        turn = {"decision": 4, "steps": 20, "status": "completed",
                "action": {"k": "base", "a": "yaw", "v": -.5, "n": "Turn slightly to bring the target into the arm workspace before retrying"},
                "receipt": {"base_moved_m": [0., 0., 0.], "base_world_after_m": [4.0572, -.6924, .7],
                            "base_yaw_after_rad": 1.3, "tcp_world_after_m": [4.0961, -.5522, 1.5528], "gripper_width_after_m": .0785}}
        retry = {"decision": 5, "steps": 20, "status": "completed", "action": target,
                 "receipt": {"tcp_world_after_m": [3.994, -.545, 1.291], "base_world_after_m": [4.0572, -.6924, .7],
                             "base_yaw_after_rad": 1.3, "gripper_width_after_m": .0785}}
        rejected = {"decision": 4, "steps": 0, "status": "invalid_action", "action": {"k": "ee"}}
        variants = [("base_retry", [base, {**retry, "decision": 4}], ["x", None]),
                    ("base_turn_retry", [base, turn, retry], ["x", "yaw", None]),
                    ("zero_step_between", [base, rejected, retry], ["x", None])]
        for name, followup, axes in variants:
            with self.subTest(flow=name):
                run = self.root / name
                write_json(run / "result.json", {"task": "task", "seed": 0, "method": "clean"})
                write_rows(run / "decisions.jsonl", [failure, *followup])
                bank = self.root / f"{name}-bank"
                with patch.object(sys, "argv", ["banks", "--runs", str(run), "--out", str(bank)]):
                    banks.main()
                method = H6FailureExperience(run=self.root, config={"bank": str(bank / "h6-failures.json")})
                ctx = {"decision": 1, "observation": {**observation(), "task": "task"}}
                extra, _ = method.observe(ctx)
                retrieved = extra["failure_experience"]
                self.assertEqual(len(retrieved), 1, "the complete correction must fit H6's existing retrieval budget")
                record = retrieved[0]
                self.assertTrue(record["correction_verified"])
                sequence = record["correction"].get("sequence", [])
                self.assertEqual([step["action"].get("a") for step in sequence], axes)
                self.assertEqual(sequence[0]["effect"]["base_moved_m"], [.0001, .1495, 0.])
                self.assertEqual(sequence[0]["effect"]["steps"], 20)
                self.assertEqual(sequence[-1]["action"]["p"], [3.994, -.545, 1.291])
                self.assertEqual(sequence[-1]["effect"]["tcp_world_after_m"], [3.994, -.545, 1.291])
                self.assertEqual(sequence[-1]["effect"]["status"], "completed")
                self.assertAlmostEqual(record["state_before"]["base_world_position_m"][1], -.8418753743171692, places=4)
                if name == "base_turn_retry":
                    self.assertEqual(sequence[1]["effect"]["base_yaw_after_rad"], 1.3)
                self.assertLessEqual(sum(len(json.dumps(r, separators=(",", ":"))) // 4 for r in retrieved), 700)

        late = [failure, base, rejected, {"decision": 5, "steps": 0, "status": "malformed_output"}, {**retry, "decision": 6}]
        record = banks.failure_records({"task": "task", "seed": 0}, late)[0]["record"]
        self.assertFalse(record["correction_verified"], "a retry after the next three decisions is outside the correction window")

    def skill_fixture(self, open_lift=False):
        motion = [{"decision": 0, "status": "ready_pose", "steps": 20},
                  {"decision": 1, "steps": 0, "status": "unreachable", "action": {"k": "ee"}},
                  {"decision": 2, "steps": 20, "action": {"k": "hold", "g": 0},
                   "receipt": {"tcp_world_after_m": [1, 2, 1], "gripper_width_after_m": .02}},
                  {"decision": 3, "steps": 20, "action": {"k": "hold", "g": 1 if open_lift else 0,
                   "n": "Open gripper to re-grasp spoon" if open_lift else "lift"},
                   "receipt": {"tcp_world_after_m": [1, 2, 1.05], "gripper_width_after_m": .02}}]
        inspection = [{"run": "run", "sequence": seq, "obj_world_m": [1, 2, z - .02],
                       "tcp_world_m": [1, 2, z], "gripper_command": 0, "gripper_width_m": .02,
                       "gripper_touching_obj": True, "preceding_decision": 999} for seq, z in [(2, 1.), (3, 1.05)]]
        return {"task": "task", "seed": 0, "official_success": False}, motion, inspection

    def test_no_inspection_produces_no_skills(self):
        result, decisions, _ = self.skill_fixture()
        self.assertEqual(banks.skill_records(result, decisions[2:]), [])

    def test_false_open_to_regrasp_is_rejected_even_with_lifted_tcp(self):
        result, decisions, inspection = self.skill_fixture(open_lift=True)
        self.assertEqual(banks.skill_records(result, decisions, inspection), [])

    def test_skill_joins_sequence_including_ready_pose_and_requires_object_to_follow(self):
        result, decisions, inspection = self.skill_fixture()
        self.assertEqual(len(banks.skill_records(result, decisions, inspection)), 1)
        for field, value in [("obj_world_m", [1, 2, .98]), ("gripper_command", 1), ("gripper_touching_obj", False)]:
            with self.subTest(field=field):
                invalid = copy.deepcopy(inspection)
                invalid[1][field] = value
                self.assertEqual(banks.skill_records(result, decisions, invalid), [])

    def test_banks_cli_recovers_historical_zero_step_prestate_from_control_call(self):
        run = self.root / "source"
        write_json(run / "result.json", {"task": "task", "seed": 0, "method": "clean"})
        write_rows(run / "decisions.jsonl", [{"decision": 2, "steps": 0, "status": "unreachable",
                   "action": {"k": "ee", "p": [9, 9, 9]}, "receipt": {}}])
        write_rows(run / "qwen-calls.jsonl", [
            {"decision": 2, "category": "control", "user_text": json.dumps({"state": {"tcp_world_m": [4, 5, 6], "gripper_command": 1}, "previous": {"action": {"k": "hold"}}})},
            {"decision": 2, "category": "preview_select", "user_text": "selection text"}])
        out = self.root / "bank"
        with patch.object(sys, "argv", ["banks", "--runs", str(run), "--out", str(out)]):
            banks.main()
        rec = json.loads((out / "h6-failures.json").read_text())["entries"][0]["record"]
        self.assertIn("[4, 5, 6]", rec["context"])
        self.assertEqual(json.loads((out / "h7-skills.json").read_text())["entries"], [])

    def test_banks_cli_joins_inspection_by_full_run_path(self):
        result, decisions, inspection = self.skill_fixture()
        run = self.root / "source"
        write_json(run / "result.json", {**result, "method": "clean"})
        write_rows(run / "decisions.jsonl", decisions)
        inspection_path = self.root / "inspection.json"
        own = [{**row, "run": str(run)} for row in inspection]
        wrong = [{**row, "run": str(self.root / "other/source"), "gripper_command": 1} for row in inspection]
        write_json(inspection_path, own + wrong)
        out = self.root / "bank"
        with patch.object(sys, "argv", ["banks", "--runs", str(run), "--out", str(out), "--inspection", str(inspection_path)]):
            banks.main()
        skills = json.loads((out / "h7-skills.json").read_text())["entries"]
        self.assertEqual(len(skills), 1)
        self.assertNotIn("obj_world_m", json.dumps(skills))

    def test_recovery_cli_rejects_self_qualification_for_both_comparison_arms(self):
        out = self.root / "continuations"
        state = {**self.candidate(), "failure_type": "empty_close_near_object", "qualified_for_rsr": True,
                 "qualification": {"run": str(out / "source-seq8-clean")}}
        bank = self.root / "bank"
        write_json(bank / "states.json", [state])
        for method in ("clean", "h8"):
            write_json(out / f"source-seq8-{method}/result.json", {"official_success": True})
        with patch.object(sys, "argv", ["recovery", "continue", "--bank", str(bank), "--out", str(out)]):
            recovery.main()
        summary = json.loads((out / "rsr.json").read_text())
        for method in ("clean", "h8"):
            self.assertIsNone(summary[method]["rsr"])
            self.assertEqual(summary[method]["candidate_completions"], 1)

    def candidate(self):
        return {"run": "source", "sequence": 8, "snapshot": "/source/sim/snapshots/000008.json",
                "task": "task", "seed": 0, "preceding_decision": 9,
                "obj_world_m": [.1, 0, 1], "base_world_m": [0, 0, .7], "gripper_command": 0,
                "gripper_width_m": .001, "gripper_obj_distance_m": .1, "official_success": False}

    def test_selected_candidates_are_unknown_and_empty_qualified_rsr_is_null(self):
        states = recovery.select_states([self.candidate()], 5)
        self.assertEqual(states[0].get("recoverability"), "unknown")
        self.assertFalse(states[0].get("qualified_for_rsr", True))
        summary = recovery.recovery_summary([{"continuation_method": "clean", "official_success": False,
                                             "qualified_for_rsr": False}])
        self.assertEqual(summary["clean"]["states"], 0)
        self.assertIsNone(summary["clean"]["rsr"])
        self.assertEqual(summary["clean"]["candidate_states"], 1)

    def test_qualification_needs_matching_success_within_all_recovery_budgets(self):
        state = self.candidate()
        result = {"task": "task", "seed": 0, "restore_from": state["snapshot"], "official_success": True,
                  "steps_budget": 400, "wall_budget_s": 600, "max_decisions": 80,
                  "simulator_steps": 399, "wall_s": 599, "decisions": 79}
        path = self.root / "independent/result.json"
        write_json(path, result)
        self.assertTrue(recovery.qualify_states([state], [path])[0]["qualified_for_rsr"])
        for field, value in [("official_success", False), ("simulator_steps", 401), ("wall_s", 601),
                             ("decisions", 81), ("restore_from", "/other/snapshot.json"), ("task", "other")]:
            with self.subTest(field=field):
                write_json(path, {**result, field: value})
                self.assertFalse(recovery.qualify_states([state], [path])[0]["qualified_for_rsr"])

    def test_cached_candidate_continuation_keeps_status_metadata(self):
        state = {**self.candidate(), "failure_type": "empty_close_near_object", "qualified_for_rsr": False}
        out = self.root / "continuations"
        write_json(out / "source-seq8-clean/result.json", {"official_success": False})
        result = recovery.run_continuation(state, "clean", out)
        self.assertEqual(result.get("continuation_method"), "clean")
        self.assertFalse(result.get("qualified_for_rsr", True))

    def test_inspection_joins_snapshot_names_to_full_slots_and_keeps_source_step_count(self):
        run = self.root / "source"
        write_json(run / "result.json", {"task": "task", "seed": 0, "method": "clean", "scene": "/scenes/task.json"})
        write_rows(run / "decisions.jsonl", [
            {"decision": 0, "steps": 20, "status": "ready_pose"},
            {"decision": 1, "sequence_length": 2},
            {"decision": 1, "slot": 1, "steps": 20, "status": "completed", "action": {"k": "hold", "g": 0}},
            {"decision": 1, "slot": 2, "steps": 20, "status": "completed", "action": {"k": "hold", "g": 1}}])
        write_json(run / "sim/snapshots/000000.json", {"total_steps": 0})
        write_json(run / "sim/snapshots/000002.json", {"total_steps": 40})
        obs = observation()
        obs["execution"] = {"obj_world_m": [1, 2, 1], "gripper_touching_obj": True}
        fake = SimpleNamespace(launch=lambda: None, send=lambda _: obs, close=lambda: None)
        with patch("direct.recovery.Simulator", return_value=fake):
            rows = recovery.inspect_run(run, self.root / "inspection")
        self.assertEqual(rows[-1]["sequence"], 2)
        self.assertEqual(rows[-1]["preceding_action"], {"k": "hold", "g": 0})
        self.assertEqual(rows[-1]["steps_used"], 40)


if __name__ == "__main__":
    unittest.main()
