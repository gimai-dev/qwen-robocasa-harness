import json
from pathlib import Path
from types import SimpleNamespace
from adaptive.joint_runner import run_episode
class Probe:
 def __init__(self,**kwargs):self.positions=[]
 def verify(self):pass
 def close(self):pass
 def complete(self,**kwargs):
  self.positions.append(kwargs['public_state']['state.base_position'])
  run.joinpath('base-positions.json').write_text(json.dumps(self.positions))
  if len(self.positions)>3:return SimpleNamespace(command={'kind':'finish','observation_id':kwargs['observation_id'],'note':'base translation probe complete'},evidence={'origin':'base_probe'})
  return SimpleNamespace(command={'kind':'base_action','observation_id':kwargs['observation_id'],'axis':'x','normalized_velocity':-.5,'gripper':'hold','note':'Measure base displacement from supported 0.5 command'},evidence={'origin':'base_probe'})
run=Path('/home/jli/state/recovery-20260908/base-probe-05')
r=run_episode(task='PickPlaceStoveToCounter',seed=7,run=run,max_decisions=5,protocol='skills',client_class=Probe,action_budget=900,wall_budget_s=120,prompt_variant='rig')
run.joinpath('result.json').write_text(json.dumps(r,indent=2));print(json.dumps({'status':r['status'],'steps':r['simulator_steps'],'error':r.get('error')}))
