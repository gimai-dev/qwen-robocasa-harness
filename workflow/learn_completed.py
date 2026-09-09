"""Reflect completed native episodes and attach reproducible execution recipes."""
import argparse,json
from pathlib import Path
from run import reflect
from adaptive.episode_memory import record_episode
p=argparse.ArgumentParser();p.add_argument('--episodes',type=Path,required=True);p.add_argument('--memory',type=Path,required=True);p.add_argument('--routes',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
bank=json.loads(a.memory.read_text());routes=json.loads(a.routes.read_text())['routes'];seen={x['source_run'] for x in bank['lessons']}
for path in sorted(a.episodes.glob('*/result.json'),key=lambda p:p.stat().st_mtime):
    episode=path.parent;native=json.loads(path.read_text());outcome=(native.get('terminal_outcome') or {}).get('success')
    if type(outcome) is not bool or not native.get('task') or not (episode/'skill-evidence.json').exists():continue
    key=str(episode.resolve())
    if key in seen:
        next(x for x in bank['lessons'] if x['source_run']==key)['completed_at']=path.stat().st_mtime
        continue
    evidence=json.loads((episode/'skill-evidence.json').read_text())
    saved=episode/'reflection.json'
    reflection=json.loads(saved.read_text())['reflection'] if saved.exists() else reflect(episode,native,evidence)
    bank=record_episode(bank,native,evidence,reflection,key)
    bank['lessons'][-1]['completed_at']=path.stat().st_mtime
    for card in bank['skills']:
        route=next((r for r in routes if r['task']==card.get('source_task') and r.get('reproduced_success')),None)
        if route:
            card['execution_recipe']={k:route[k] for k in ('code','entry','args','memory') if k in route}
            card['reproduction_run']=route['reproduced_success']
    (episode/'memory-after.json').write_text(json.dumps(bank,indent=2))
    a.output.write_text(json.dumps(bank,indent=2));seen.add(key)
    print(episode.name,outcome,len(bank['skills']),len(bank['lessons']),flush=True)

a.output.write_text(json.dumps(bank,indent=2))
