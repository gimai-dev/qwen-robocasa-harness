"""Two native transfer trials, with outcome-grounded memory writeback."""
import argparse,copy,json,os,time
from pathlib import Path
from adaptive.skill_driver import make_client
from adaptive.joint_runner import run_episode
from adaptive.episode_memory import record_episode
from robocasa_inspect.model_client import QwenClient,load_authority

def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

def reflect(run,result,evidence):
    facts={'task':result['task'],'seed':result['seed'],'official_success':result['terminal_outcome']['success'],
           'status':result['status'],'actions':result['simulator_steps'],
           'final_stage':evidence['stage'],'stop_reason':evidence['abort_reason']}
    facts['execution_events']=[{k:e[k] for k in ['kind','stage','skill_id','parameters','reason','result','gripper_qpos','end_effector_lift_m'] if k in e}
        for e in evidence['events'] if e['kind'] in ['skill_selected','closed','qwen_grasp_visual_check','grasp_lost','finish']]
    facts['workflow_adaptations']=[e for e in evidence['events'] if e['kind'] in {'palm_clearance_adjustment','carry_clearance','receiver_reanchored','handle_inset','grasp_contact_arrival'}]
    facts['pose_attempts']=[{'stage':e['stage'],'status':e['plan']['status'],
                           'reason':e['plan'].get('reason')} for e in evidence['events'] if e['kind']=='pose_plan']
    grounds=[e['result'] for e in evidence['events'] if e['kind']=='grounding']
    if grounds:
        g=grounds[-1];facts['source_grounding']={'status':g['status'],'target':g.get('evidence',{}).get('target'),
            'rejected_pairs':g.get('evidence',{}).get('rejected_pairs')}
    schema={'type':'object','properties':{'lesson':{'type':'string'},'applicability':{'type':'string'},
        'evidence':{'type':'array','items':{'type':'string'},'maxItems':3},'next_attempt':{'type':'string'},
        'skill_steps':{'type':'array','items':{'type':'string'},'maxItems':6}},
        'required':['lesson','applicability','evidence','next_attempt','skill_steps'],'additionalProperties':False}
    identity,attestation=load_authority(Path(os.environ['QWEN_IDENTITY_MANIFEST']),Path(os.environ['QWEN_SERVER_ATTESTATION']))
    client=QwenClient(base_url='http://127.0.0.1:8002/v1',api_key=Path(os.environ['QWEN_API_TOKEN_FILE']).read_text().strip(),identity=identity,attestation=attestation)
    frames=sorted((run/'sim/frames').glob('*'))
    images={v:(frames[-1]/(v+'.png')).read_bytes() for v in ['left','right','wrist']}
    write(run/'reflection-input.json',facts)
    try:
        client.verify();answer=client.complete(observation_id='reflection-'+run.name,
            system_prompt='Summarize this robot episode under180 words. The official result and executed events are facts. '
                'Qwen visual checks are model judgments. The images are FINAL views; do not infer initial object locations from them. '
                'Separate a tested successful strategy from a failure hypothesis. Recommend one concrete next change based on the actual stop reason. '
                'Failed attempts supply no successful skill. Return JSON.',
            instruction=json.dumps(facts,separators=(',',':')),public_state={},images=images,response_schema=schema,max_tokens=700)
        write(run/'reflection.json',{'facts':facts,'reflection':answer.command,'model_evidence':answer.evidence})
        return answer.command
    finally:client.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--memory',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);bank=read(a.memory);write(a.output/'initial-memory.json',bank)
    cohort={'status':'running','samples':[],'started_at':time.time()};write(a.output/'cohort.json',cohort)
    for task in ['PickPlaceStoveToCounter','PickPlaceCounterToSink']:
        run=a.output/task;print('START '+task,flush=True)
        result=run_episode(task=task,seed=7,run=run,max_decisions=180,protocol='skills',
            client_class=make_client(run,grasp_candidate='auto',place=True,perception='sam2',skill_memory=bank),
            action_budget=900,wall_budget_s=1200,prompt_variant='rig')
        evidence=read(run/'skill-evidence.json');events=evidence['events']
        result.update(policy_class=evidence['policy_class'],qwen_model_calls=evidence['qwen_calls'],action_budget=900)
        write(run/'result.json',result)
        valid=type((result.get('terminal_outcome') or {}).get('success')) is bool and not result.get('error_type')
        selected=next((e for e in events if e['kind']=='skill_selected'),None)
        checks=[e for e in events if e['kind']=='qwen_grasp_visual_check']
        row={'task':task,'seed':7,'measurement_valid':valid,'official_success':(result.get('terminal_outcome') or {}).get('success'),
             'status':result['status'],'stage':evidence['stage'],'reason':evidence['abort_reason'],
             'actions':result.get('simulator_steps'),'wall_s':result.get('wall_s'),'qwen_calls':evidence['qwen_calls'],
             'selected_skill':selected['skill_id'] if selected else None,
             'executed_close':any(e['kind']=='closed' for e in events),
             'qwen_held_after_lift':checks[-1]['result']['held'] if checks else None}
        cohort['samples'].append(row);write(a.output/'cohort.json',cohort)
        if not valid:raise RuntimeError('Invalid measurement '+task+' '+str(result.get('error')))
        reflection=reflect(run,result,evidence)
        bank=record_episode(copy.deepcopy(bank),result,evidence,reflection,task+'-seed7-skill-transfer')
        write(run/'memory-after.json',bank);write(a.output/'current-memory.json',bank)
        row['reflection_complete']=True;write(a.output/'cohort.json',cohort)
        print('DONE '+json.dumps(row),flush=True)
    cohort.update(status='completed',wall_s=time.time()-cohort['started_at']);write(a.output/'cohort.json',cohort)
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
