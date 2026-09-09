import argparse,copy,json
from pathlib import Path
from adaptive.skill_driver import make_client
from adaptive.joint_runner import run_episode
from adaptive.episode_memory import record_episode
from run import reflect,read,write

p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--task',default='PickPlaceCounterToSink');p.add_argument('--seed',type=int,default=7);p.add_argument('--memory',type=Path,default=Path('/home/jli/state/fullten-current-20260908/frozen-memory.json'));a=p.parse_args()
root=Path('/home/jli/state/fullten-current-20260908');root.mkdir(exist_ok=True)
run=root/a.name
bank=read(a.memory)
result=run_episode(task=a.task,seed=a.seed,run=run,max_decisions=180,protocol='skills',
    client_class=make_client(run,grasp_candidate='auto',place=True,perception='sam2',skill_memory=bank),
    action_budget=900,wall_budget_s=1200,prompt_variant='rig')
evidence=read(run/'skill-evidence.json')
result.update(policy_class=evidence['policy_class'],qwen_model_calls=evidence['qwen_calls'],action_budget=900)
write(run/'result.json',result)
assert type((result.get('terminal_outcome') or {}).get('success')) is bool and not result.get('error_type'),result.get('error')
reflection=reflect(run,result,evidence)
write(run/'memory-after.json',record_episode(copy.deepcopy(bank),result,evidence,reflection,a.name))
print(json.dumps({'run':a.name,'official_success':result['terminal_outcome']['success'],'actions':result['simulator_steps'],
    'wall_s':result['wall_s'],'stage':evidence['stage'],'reason':evidence['abort_reason'],'qwen_calls':evidence['qwen_calls']}),flush=True)
