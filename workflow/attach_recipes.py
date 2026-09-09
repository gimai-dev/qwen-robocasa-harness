"""Keep the complete reproduced controller recipe with each successful skill."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--memory',type=Path,required=True);p.add_argument('--routes',type=Path,required=True);a=p.parse_args()
bank=json.loads(a.memory.read_text());routes=json.loads(a.routes.read_text())['routes']
for card in bank['skills']:
    route=next((r for r in routes if r['task']==card.get('source_task')),None)
    if route is not None:
        card['execution_recipe']={k:route[k] for k in ('code','entry','args','memory') if k in route}
        card['reproduction_run']=route['reproduced_success']
        card['execution_scope']='Task-specific reproduced controller; fresh public RGB localization and native official evaluation required.'
a.memory.write_text(json.dumps(bank,indent=2))
print('Attached complete recipes to',sum('execution_recipe' in c for c in bank['skills']),'skills')
