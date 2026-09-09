import argparse,json
from pathlib import Path
from adaptive.skill_driver import make_client
from adaptive.joint_runner import run_episode
p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--name',required=True);p.add_argument('--approach-height',type=float);p.add_argument('--transfer-clearance',type=float);p.add_argument('--base-backoff',type=float,default=0);a=p.parse_args()
root=Path('/home/jli/state/recovery-20260908');run=root/a.name
bank=json.loads(Path('/home/jli/state/recovery-20260908/restored-memory.json').read_text())
result=run_episode(task=a.task,seed=7,run=run,max_decisions=180,protocol='skills',client_class=make_client(run,grasp_candidate='auto',place=True,perception='sam2',skill_memory=bank,approach_height=a.approach_height,transfer_clearance=a.transfer_clearance,base_backoff=a.base_backoff),action_budget=900,wall_budget_s=1200,prompt_variant='rig')
e=json.loads((run/'skill-evidence.json').read_text());(run/'result.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'task':a.task,'official_success':(result.get('terminal_outcome') or {}).get('success'),'error_type':result.get('error_type'),'actions':result.get('simulator_steps'),'stage':e['stage'],'reason':e['abort_reason']}),flush=True)
