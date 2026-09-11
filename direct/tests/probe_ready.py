import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from direct.executor import Simulator, move_to_ready
from direct.kinematics import quat_xyzw_to_matrix
run = Path(sys.argv[1])
sim = Simulator(task="PickPlaceCounterToSink", seed=0, run=run, scenes=Path("/home/jli/state/qwen-direct/scenes"), action_budget=900, wall_budget_s=1200)
sim.launch()
r = move_to_ready(sim); s = sim.observation["public_state"]
R = quat_xyzw_to_matrix(s["tcp_base_quat_xyzw"])
print(f"ready: status={r.status} err={r.child['final_joint_error_rad']:.3f} tcp_base={[round(v,3) for v in s['tcp_base_position_m']]} tool_z_axis_world_z={R[2][2]:.3f} q={[round(v,2) for v in s['arm_q_rad']]}")
sim.finish()
