"""Measure unloaded and fixture-contact wrist feedback without policy calls."""
import argparse
import json
import math
from pathlib import Path
from direct.actions import Action
from direct.executor import Simulator,execute,move_to_ready


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    sim=Simulator(task='PickPlaceCounterToSink',seed=0,run=args.out/'sim-run',
        scenes=Path('/home/jli/state/qwen-direct/scenes'),action_budget=300,wall_budget_s=600)
    sim.launch();rows=[]
    def record(name,receipt=None):
        s=sim.observation['public_state']
        row={'name':name,'steps':sim.steps_used(),'raw_norm_n':math.sqrt(sum(v*v for v in s['wrench']['force_n'])),
             'unloaded_weight_n':s['unloaded_wrist_weight_n'],'load_n':s['contact_force_delta_n'],
             'tcp':s['tcp_world_position_m'],'peak_n':receipt.child.get('peak_force_n') if receipt and receipt.child else None}
        rows.append(row);print(json.dumps(row),flush=True)
    try:
        record('reset')
        record('initial-hold',execute(sim,Action('hold',gripper=1)))
        move_to_ready(sim);record('ready')
        state=sim.observation['public_state'];q=tuple(state['tcp_world_quat_xyzw'])
        clear=tuple(state['tcp_world_position_m']);down=(clear[0],clear[1],.85)
        for index in range(3):
            record(f'contact-{index}',execute(sim,Action('ee',position_m=down,quat_xyzw=q,gripper=1)))
        for index in range(3):
            record(f'retreat-{index}',execute(sim,Action('ee',position_m=clear,quat_xyzw=q,gripper=1)))
        record('free-after',execute(sim,Action('hold',gripper=1)))
    finally:
        sim.finish()
    (args.out/'probe-results.json').write_text(json.dumps(rows,indent=2))
    assert abs(rows[0]['unloaded_weight_n']-5.1012)<.01 and rows[0]['load_n'] is None,rows
    # The original reset pose need not be unloaded. The ready pose and
    # post-retreat hold provide the free-space comparison for this scene.
    assert rows[2]['load_n']<2 and rows[-1]['load_n']<2,rows
    assert max(r['peak_n'] or 0 for r in rows if r['name'].startswith('contact'))>5,rows


if __name__=='__main__': main()
