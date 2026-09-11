"""Measure free-space base displacement per slot for x (backward), y and yaw."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from direct.actions import Action
from direct.executor import Simulator, execute
run = Path(sys.argv[1])
sim = Simulator(task="PickPlaceCounterToSink", seed=0, run=run, scenes=Path("/home/jli/state/qwen-direct/scenes"), action_budget=900, wall_budget_s=1200)
sim.launch()
for axis, v in (("x", -0.3), ("x", -0.4), ("x", -0.45), ("yaw", 0.3), ("yaw", 0.4)):
    s0 = sim.observation["public_state"]
    r = execute(sim, Action("base", axis=axis, velocity=v))
    c = r.child
    d = np.asarray(c["base_world_after_m"]) - np.asarray(s0["base_world_position_m"])
    dyaw = c["base_yaw_after_rad"] - s0["base_world_yaw_rad"]
    print(f"{axis} v={v}: base moved {np.round(d,4)} |xy|={np.linalg.norm(d[:2]):.4f} yaw {dyaw:+.4f} arm err {c['final_joint_error_rad']:.4f}")
    print("   per-step |xy| from start:", [round(float(np.linalg.norm(np.asarray(t[:2]) - np.asarray(s0['base_world_position_m'][:2]))),3) for t in c["trace"]])
sim.finish()
