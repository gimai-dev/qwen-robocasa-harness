"""Regressions from the September numerical-control audit; no model calls."""
import json
import math
from pathlib import Path
import unittest

import numpy as np

from direct.actions import decode_action, ee_rotation_matrix
from direct import chassis_hold
from direct.executor import execute
from direct.kinematics import inside_safe_limits, solve_pose_multistart, world_to_base


CASES = json.loads((Path(__file__).parent / "fixtures/ik_rejections.json").read_text())


class RecordingSim:
    def __init__(self, state):
        self.observation = {"public_state": state, "gripper_command": 1}
        self.command = None

    def steps_left(self):
        return 900

    def send(self, command):
        self.command = command
        return {"execution": {"steps_executed": command["steps"], "final_joint_error_rad": 1.0}}


class MotionRepairs(unittest.TestCase):
    def test_recorded_long_fallback_emits_a_partial_slot(self):
        case = CASES[0]
        sim = RecordingSim(case["public_state"])
        action = decode_action(case["action"], interface="ee")
        receipt = execute(sim, action)
        self.assertEqual(receipt.status, "partial")
        self.assertEqual(receipt.steps, 20)
        self.assertIsNotNone(sim.command)
        self.assertTrue(all(inside_safe_limits(q) for q in sim.command["waypoints"]))

    def test_regularization_does_not_hide_recorded_reachable_pose(self):
        case = CASES[1]
        state = case["public_state"]
        action = decode_action(case["action"], interface="ee")
        p, r = world_to_base(action.position_m, ee_rotation_matrix(action),
                             state["base_world_position_m"], state["base_world_quat_xyzw"])
        solved = solve_pose_multistart(state["arm_q_rad"], p, r)
        self.assertEqual(solved["status"], "kinematically_reachable")
        self.assertLessEqual(solved["position_error_m"], .003)
        self.assertTrue(inside_safe_limits(solved["q"]))

    def test_current_heading_translation_through_installed_controller(self):
        # RoboCasa selects LegacyMobileBaseJointVelocityController, which
        # rotates its input by +delta yaw, while
        # translation joints retain the initial heading. Their friction/gain
        # is equivalent to a .25 normalized velocity dead zone on each axis.
        for reset in (0., 1.1):
            for delta in (0., math.pi/4, math.pi/2, -math.pi/3):
                for axis in ("x", "y"):
                    with self.subTest(reset=reset, delta=delta, axis=axis):
                        command = chassis_hold.base_velocity_input(axis, .5, reset+delta, reset)
                        actuator = chassis_hold._rotate(delta, command[:2])
                        net = [math.copysign(max(abs(v)-.25, 0), v) for v in actuator]
                        world = chassis_hold._rotate(reset, net)
                        angle = reset+delta+(math.pi/2 if axis == "y" else 0)
                        np.testing.assert_allclose(world, [.25*math.cos(angle), .25*math.sin(angle)], atol=1e-12)
                        self.assertLessEqual(max(map(abs, command)), 1.)

    def test_hold_has_authority_above_translation_deadzone(self):
        hold = chassis_hold.ChassisHold((0., 0.), 0., 0.)
        state = {"state.base_position": [-.013, 0., .7], "state.base_rotation": [0., 0., 0., 1.]}
        correction = hold.correction(state)
        self.assertGreater(correction[0], .25)
        self.assertLessEqual(np.linalg.norm(correction[:2]), .5)
        state["state.base_position"] = [0., 0., .7]
        self.assertEqual(hold.correction(state), [0., 0., 0.])


if __name__ == "__main__":
    unittest.main()
