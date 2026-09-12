"""Physical regression for restored chassis frame and small yaw velocities."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from direct.actions import Action
from direct.executor import Simulator, execute
from direct.harnesses_extra import H3ProposePreview


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    restored=Simulator(task='PickPlaceCounterToSink', seed=0, run=args.out/'restored',
        scenes=Path('/home/jli/state/qwen-direct/scenes'), action_budget=80, wall_budget_s=300,
        restore_from=Path('/home/jli/state/qwen-direct/revision-2026-09/probes/motion-02/base/sim/snapshots/000013.json'))
    restored.launch()
    try:
        state=restored.observation['public_state']
        r=execute(restored, Action('base', axis='x', velocity=-.5))
        delta=np.asarray(r.child['base_world_after_m'][:2])-state['base_world_position_m'][:2]
        yaw=state['base_world_yaw_rad']
        direction=-np.asarray([math.cos(yaw),math.sin(yaw)])
        result={'restored_translation':{'yaw':yaw,'along_m':float(delta@direction),
            'across_m':float(delta@np.asarray([-direction[1],direction[0]]))}}
        print(json.dumps(result),flush=True)
        hold=execute(restored, Action('joint', q_rad=tuple(state['arm_q_rad'])))
        result['restored_hold_error_xy_m']=hold.child['chassis_hold_error_xy_m']
    finally:
        restored.finish()
    sim=Simulator(task='PickPlaceCounterToSink', seed=0, run=args.out/'yaw',
        scenes=Path('/home/jli/state/qwen-direct/scenes'), action_budget=500, wall_budget_s=600)
    sim.launch()
    result['yaw']=[]
    preview=H3ProposePreview(run=args.out,config={}); preview.interface='ee'
    try:
        for _ in range(3):
            execute(sim,Action('base',axis='x',velocity=-.5))
        for v in (.05,.1,.15,.2,.3,.4):
            for sign in (1,-1):
                state=sim.observation['public_state']
                before=state['base_world_yaw_rad']
                forecast,_,_=preview._preview_action(state,{'k':'base','a':'yaw','v':sign*v},preview,sim.observation['gripper_command'])
                r=execute(sim,Action('base',axis='yaw',velocity=sign*v))
                after=r.child['base_yaw_after_rad']
                delta=math.atan2(math.sin(after-before),math.cos(after-before))
                predicted=forecast['predicted_yaw_delta_rad']
                row={'v':sign*v,'actual_rad':delta,'predicted_rad':predicted,'error_rad':abs(delta-predicted)}
                result['yaw'].append(row)
                print(json.dumps(row),flush=True)
    finally:
        sim.finish()
    (args.out/'probe-results.json').write_text(json.dumps(result,indent=2))
    assert result['restored_translation']['along_m']>.10 and abs(result['restored_translation']['across_m'])<.02,result
    assert max(abs(v) for v in result['restored_hold_error_xy_m'])<.01,result
    assert max(r['error_rad'] for r in result['yaw'])<.01,result


if __name__=='__main__':
    main()
