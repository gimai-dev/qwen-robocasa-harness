"""Run the fixed ten-task list, keeping the same memory input for every task."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/jli/state/fullten-current-20260908')
CODE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
write=lambda p,v:p.write_text(json.dumps(v,indent=2)+'\n')
manifest=read(ROOT/'manifest.json')
cohort={'status':'running','started_at':time.time(),'samples':[],'planned_tasks':manifest['tasks']}
def save():
 rows=cohort['samples']
 cohort.update(completed_tasks=len(rows),valid_tasks=sum(r['valid'] for r in rows),successes=sum(r['official_success'] is True for r in rows),elapsed_s=time.time()-cohort['started_at'])
 write(ROOT/'cohort.json',cohort)
save()
for task in manifest['tasks']:
 cohort['current_task']=task;save();print('START '+task,flush=True)
 command=[sys.executable,str(CODE/'trial.py'),'--name',task,'--task',task,'--seed','7','--memory',str(ROOT/'frozen-memory.json')]
 write(ROOT/(task+'-launch.json'),{'command':command,'code':str(CODE),'memory':str(ROOT/'frozen-memory.json')})
 with (ROOT/(task+'-runtime.log')).open('w') as log:
  proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=CODE)
 run=ROOT/task
 result=read(run/'result.json') if (run/'result.json').exists() else {}
 evidence=read(run/'skill-evidence.json') if (run/'skill-evidence.json').exists() else {}
 events=evidence.get('events',[])
 terminal=result.get('terminal_outcome') or {}
 valid=type(terminal.get('success')) is bool and not result.get('error_type')
 selected=next((e for e in events if e['kind']=='skill_selected'),None)
 held=[e for e in events if e['kind']=='qwen_grasp_visual_check']
 row={'task':task,'seed':7,'exit_code':proc.returncode,'valid':valid,
  'official_success':terminal.get('success') if valid else None,'status':result.get('status','missing_result'),
  'stage':evidence.get('stage'),'reason':evidence.get('abort_reason'),
  'actions':result.get('simulator_steps'),'wall_s':result.get('wall_s'),'qwen_calls':evidence.get('qwen_calls',0),
  'selected_skill':selected['skill_id'] if selected else None,
  'closed':any(e['kind']=='closed' for e in events),'qwen_held':held[-1]['result']['held'] if held else None,
  'intentional_release':any(e['kind']=='intentional_release' for e in events),
  'release_observed':any(e['kind']=='observation' and e.get('stage')=='release' for e in events),
  'grasp_lost':any(e['kind']=='grasp_lost' for e in events),
  'reflection_saved':(run/'reflection.json').exists(),'memory_saved':(run/'memory-after.json').exists(),
  'error_type':result.get('error_type'),'error':result.get('error')}
 cohort['samples'].append(row);save();print('DONE '+json.dumps(row),flush=True)
cohort.pop('current_task',None)
cohort['status']='completed' if cohort['valid_tasks']==10 else 'completed_with_invalid_measurements'
cohort['success_rate']=cohort['successes']/10 if cohort['valid_tasks']==10 else None
save();print('COHORT '+json.dumps({k:cohort[k] for k in ('status','completed_tasks','valid_tasks','successes','success_rate','elapsed_s')}),flush=True)
