import json,os
from pathlib import Path
import cv2,numpy as np
from adaptive.rgb_grounding import _triangulate
from robocasa_inspect.model_client import QwenClient,load_authority

root=Path('/home/jli/state/destination-anchor-20260908');out=root/'drain-mask-probe';out.mkdir(exist_ok=True)
source=Path('/home/jli/state/grasp-retry-20260908/PickPlaceCounterToSink-baseline');data=json.loads((source/'sam2-source/public-input.json').read_text())
identity,attestation=load_authority(Path(os.environ['QWEN_IDENTITY_MANIFEST']),Path(os.environ['QWEN_SERVER_ATTESTATION']))
client=QwenClient(base_url='http://127.0.0.1:8002/v1',api_key=Path(os.environ['QWEN_API_TOKEN_FILE']).read_text().strip(),identity=identity,attestation=attestation);client.verify()
points={};traces=[]
try:
 for v in ['left','right']:
  folder=root/'initial-probe'/v;sam=json.loads((folder/'sam-result.json').read_text());ids=[c['id'] for c in sam['candidates']]
  instruction=f'Task: Pick the boxed drink from the counter and place it in the sink. Locate a fixed landmark on the destination floor: select ONLY the small circular METAL DRAIN at the bottom of the sink. Do not select the orange loose object, whole basin, faucet, rim, robot, or countertop. All three images derive from the SAME {v} camera: original scene, numbered binary mask proposals, corresponding original-color crops. Choose the drain mask ID, or visible=false and null if absent. Describe its appearance.'
  schema={'type':'object','properties':{'visible':{'type':'boolean'},'mask_id':{'anyOf':[{'type':'integer','enum':ids},{'type':'null'}]},'description':{'type':'string'}},'required':['visible','mask_id','description'],'additionalProperties':False}
  ans=client.complete(observation_id='drain-mask-'+v,system_prompt='Identify the specified fixed landmark in the supplied RGB crop gallery. Return JSON.',instruction=instruction,public_state={},images={'left':(folder/'original.png').read_bytes(),'right':(folder/'comparison.png').read_bytes(),'wrist':(folder/'gallery.png').read_bytes()},response_schema=schema,max_tokens=700)
  traces.append({'view':v,'instruction':instruction,'response':ans.command,'evidence':ans.evidence});(out/'choices.json').write_text(json.dumps(traces,indent=2));print(v,ans.command,flush=True)
  if ans.command['visible']:
   c=next(c for c in sam['candidates'] if c['id']==ans.command['mask_id']);mask=cv2.imread(c['mask_path'],0)>0;y,x=np.where(mask);points[v]=[float(x.mean()),float(y.mean())]
 if len(points)==2:
  g=_triangulate(points,data['calibration']);print('geometry',g,flush=True)
  if g is not None:(out/'geometry.json').write_text(json.dumps({'target_world_m':g[0].tolist(),'ray_gap_m':g[1],'one_pixel_sensitivity_m':g[2],'pixels':points},indent=2))
finally:client.close()
