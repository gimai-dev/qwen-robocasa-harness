"""Retrieve complete successful task routes; keep experimental fallback explicit."""
import argparse,json
from pathlib import Path
from memory_view import relevant_lessons
p=argparse.ArgumentParser();p.add_argument('--memory',type=Path,required=True);p.add_argument('--fallback-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
bank=json.loads(a.memory.read_text());manifest=json.loads(a.fallback_manifest.read_text());manifest['memory']=str(a.memory)
for route in manifest['routes']:
    cards=[c for c in bank['skills'] if c.get('official_success') is True and c.get('source_task')==route['task'] and c.get('execution_recipe')]
    if cards:
        card=cards[-1];route.update(card['execution_recipe']);route['retrieved_skill']=card['id'];route['reproduced_success']=card['reproduction_run']
    else:
        memory_path=a.output.parent/(route['task']+'-retrieved-memory.json')
        skills=[{k:v for k,v in c.items() if k not in {'execution_recipe','reproduction_run','execution_scope'}} for c in bank['skills']]
        memory_path.write_text(json.dumps({'skills':skills,'lessons':relevant_lessons(bank,route['task'])},indent=2));route['memory']=str(memory_path)
a.output.write_text(json.dumps(manifest,indent=2))
