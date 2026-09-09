"""One fresh episode per task, route definitions and memory frozen before launch."""
import argparse,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
manifest=json.loads(a.manifest.read_text());(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
(a.output/'memory-before.json').write_bytes(Path(manifest['memory']).read_bytes())
rows=[];started=time.time();python='/home/jli/work/robocasa-inspect-official/.venv/bin/python'
for route in manifest['routes']:
    task=route['task'];run=a.output/task;code=route['code']
    env=['env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+code+':/home/jli/work/robocasa-inspect-official','QWEN_IDENTITY_MANIFEST=/home/jli/state/panda-qwen38/identity.json','QWEN_SERVER_ATTESTATION=/home/jli/state/panda-qwen38/server-attestation.json','QWEN_API_TOKEN_FILE=/home/jli/state/panda-qwen38/api-token']
    memory_path=a.output/'memory-before.json'
    if route.get('memory'):
        memory_path=a.output/(task+'-memory-before.json');memory_path.write_bytes(Path(route['memory']).read_bytes())
    if route['entry']=='legacy':
        command=env+[python,'-m','adaptive.skill_driver','--task',task,'--run',str(run),'--seed','7','--actions','900']+route['args']
    else:
        command=env+[python,str(Path(__file__).with_name('episode.py')),'--task',task,'--run',str(run),'--memory',str(memory_path)]+route['args']
    (a.output/(task+'-launch.json')).write_text(json.dumps(command,indent=2))
    print('START',task,flush=True)
    with (a.output/(task+'-runtime.log')).open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1500)
    native=json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else {}
    evidence=json.loads((run/'skill-evidence.json').read_text()) if (run/'skill-evidence.json').exists() else {}
    row={'task':task,'route':code,'official_success':(native.get('terminal_outcome') or {}).get('success') is True,'native_result_present':bool(native),'exit_code':result.returncode,'actions':native.get('simulator_steps'),'stage':evidence.get('stage'),'reason':evidence.get('abort_reason'),'qwen_calls':evidence.get('qwen_calls')};rows.append(row)
    summary={'planned_tasks':len(manifest['routes']),'completed_tasks':len(rows),'official_successes':sum(r['official_success'] for r in rows),'complete':len(rows)==len(manifest['routes']),'elapsed_s':time.time()-started,'rows':rows}
    if summary['complete']:summary['success_rate']=summary['official_successes']/summary['planned_tasks']
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print('DONE',json.dumps(row),flush=True)
