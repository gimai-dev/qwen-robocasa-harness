"""Freeze recipes, evaluate ten tasks, write lessons, and prepare the next round."""
import argparse,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--memory',type=Path,required=True);p.add_argument('--fallback-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);here=Path(__file__).resolve().parent
python=sys.executable
def call(script,*args,env=None):
    subprocess.run([python,str(here/script),*map(str,args)],check=True,env=env)
call('plan_from_memory.py','--memory',a.memory,'--fallback-manifest',a.fallback_manifest,'--output',a.output/'manifest.json')
call('evaluate.py','--manifest',a.output/'manifest.json','--output',a.output/'episodes')
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'/home/jli/work/recovery-validated-20260908/garlic-reviewed:/home/jli/work/robocasa-inspect-official','QWEN_IDENTITY_MANIFEST':'/home/jli/state/panda-qwen38/identity.json','QWEN_SERVER_ATTESTATION':'/home/jli/state/panda-qwen38/server-attestation.json','QWEN_API_TOKEN_FILE':'/home/jli/state/panda-qwen38/api-token'}
call('learn_completed.py','--episodes',a.output/'episodes','--memory',a.memory,'--routes',here/'validated-routes.json','--output',a.output/'memory-after.json',env=env)
(a.output/'next').mkdir()
call('plan_from_memory.py','--memory',a.output/'memory-after.json','--fallback-manifest',a.fallback_manifest,'--output',a.output/'next/manifest.json')
print('Completed:',a.output/'episodes/summary.json',flush=True)
