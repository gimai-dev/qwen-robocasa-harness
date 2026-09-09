import copy,json
from pathlib import Path
from adaptive.skill_driver import make_client
from adaptive.joint_runner import run_episode
from adaptive.episode_memory import record_episode
from run import reflect,read,write

root=Path('/home/jli/state/skill-transfer-20260908')
run=root/'PickPlaceCounterToSink-v2';bank=read(root/'PickPlaceCounterToSink/memory-before.json')
result=run_episode(task='PickPlaceCounterToSink',seed=7,run=run,max_decisions=180,protocol='skills',
    client_class=make_client(run,grasp_candidate='auto',place=True,perception='sam2',skill_memory=bank),
    action_budget=900,wall_budget_s=1200,prompt_variant='rig')
evidence=read(run/'skill-evidence.json')
result.update(policy_class=evidence['policy_class'],qwen_model_calls=evidence['qwen_calls'],action_budget=900)
write(run/'result.json',result)
assert type((result.get('terminal_outcome') or {}).get('success')) is bool and not result.get('error_type'),result.get('error')
reflection=reflect(run,result,evidence)
write(run/'memory-after.json',record_episode(copy.deepcopy(bank),result,evidence,reflection,'PickPlaceCounterToSink-v2'))
print(json.dumps({'official_success':result['terminal_outcome']['success'],'actions':result['simulator_steps'],
    'wall_s':result['wall_s'],'stage':evidence['stage'],'reason':evidence['abort_reason'],'qwen_calls':evidence['qwen_calls']}),flush=True)
