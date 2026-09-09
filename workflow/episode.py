"""Run one modern route with an explicitly frozen memory input."""
import argparse,json
from pathlib import Path
from adaptive.skill_driver import make_client
from adaptive.joint_runner import run_episode
p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--memory',type=Path,required=True);p.add_argument('--base-backoff',type=float,default=0);p.add_argument('--grasp-height-offset',type=float);p.add_argument('--grasp-candidate',default='auto');a=p.parse_args()
bank=json.loads(a.memory.read_text())
extra={}
if a.grasp_height_offset is not None:extra['grasp_height_offset']=a.grasp_height_offset
result=run_episode(task=a.task,seed=7,run=a.run,max_decisions=180,protocol='skills',client_class=make_client(a.run,grasp_candidate=a.grasp_candidate,place=True,perception='sam2',skill_memory=bank,base_backoff=a.base_backoff,**extra),action_budget=900,wall_budget_s=1200,prompt_variant='rig')
(a.run/'result.json').write_text(json.dumps(result,indent=2))
