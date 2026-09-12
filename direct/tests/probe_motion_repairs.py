"""Physical checks for the recorded executor failures; no Qwen or SAM calls."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from direct.actions import Action, decode_action
from direct.executor import Simulator, execute, move_to_ready
from direct.kinematics import MAX_JOINT_STEP


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases = json.loads((Path(__file__).parent / "fixtures/ik_rejections.json").read_text())
    report = {"ik": [], "base": []}
    for case, sequence in zip(cases, (9, 12)):
        sim = Simulator(task="PickPlaceCounterToDrawer", seed=100, run=args.out / f"decision-{case['decision']}",
                        scenes=Path("/home/jli/state/qwen-direct/scenes"), action_budget=80, wall_budget_s=300,
                        restore_from=Path(case["source_run"]) / f"sim/snapshots/{sequence:06d}.json")
        sim.launch()
        try:
            action = decode_action(case["action"], interface="ee")
            receipt = execute(sim, action)
            assert receipt.steps == 20 and receipt.child is not None, receipt.summary()
            c = receipt.child
            commands = np.asarray([c["arm_q_before"], *[r["commanded_q"] for r in c["trace"]]])
            max_step = float(np.abs(np.diff(commands, axis=0)).max())
            assert max_step <= MAX_JOINT_STEP + .00011, max_step
            row = {"decision": case["decision"], "status": receipt.status, "steps": receipt.steps,
                   "path": receipt.detail.get("path"), "max_commanded_joint_step_rad": max_step,
                   "tcp_displacement_m": float(np.linalg.norm(np.asarray(c["tcp_world_after_m"])-c["tcp_world_before_m"])),
                   "target_error_after_slot_m": float(np.linalg.norm(np.asarray(c["tcp_world_after_m"])-action.position_m)),
                   "chassis_hold_error_xy_m": c["chassis_hold_error_xy_m"]}
            report["ik"].append(row)
            print(json.dumps(row), flush=True)
        finally:
            sim.finish()

    sim = Simulator(task="PickPlaceCounterToSink", seed=0, run=args.out / "base",
                    scenes=Path("/home/jli/state/qwen-direct/scenes"), action_budget=500, wall_budget_s=600)
    sim.launch()
    try:
        # Back into free space before probing translations at turned headings.
        for _ in range(3):
            execute(sim, Action("base", axis="x", velocity=-.5))
        for turn in range(3):
            if turn:
                execute(sim, Action("base", axis="yaw", velocity=.5))
            for axis in ("x", "y"):
                before = sim.observation["public_state"]
                r = execute(sim, Action("base", axis=axis, velocity=-.5))
                displacement = np.asarray(r.child["base_world_after_m"][:2])-before["base_world_position_m"][:2]
                heading = before["base_world_yaw_rad"] + (math.pi/2 if axis == "y" else 0)
                direction = -np.asarray([math.cos(heading), math.sin(heading)])
                along = float(displacement @ direction)
                across = float(displacement @ np.asarray([-direction[1], direction[0]]))
                row = {"turn": turn, "axis": axis, "heading_rad": heading, "displacement_m": displacement.tolist(),
                       "along_m": along, "across_m": across}
                report["base"].append(row)
                print(json.dumps(row), flush=True)
                assert along > .10 and abs(across) < .02, row
                # Return before the next heading/axis, keeping the probe local.
                execute(sim, Action("base", axis=axis, velocity=.5))
        ready = move_to_ready(sim)
        report["ready_pose"] = ready.summary()
    finally:
        sim.finish()
    (args.out / "probe-results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
