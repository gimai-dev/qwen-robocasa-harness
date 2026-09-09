import json,os
from pathlib import Path
import cv2
from adaptive.sam_destination import segment_floor_candidates
from adaptive.temporal_grounding import temporal_floor_point
from robocasa_inspect.model_client import QwenClient,load_authority

root=Path('/home/jli/state/destination-anchor-20260908');root.mkdir(exist_ok=True)
out=root/'initial-probe';out.mkdir(exist_ok=True)
source=Path('/home/jli/state/grasp-retry-20260908/PickPlaceCounterToSink-baseline')
data=json.loads((source/'sam2-source/public-input.json').read_text());images={v:(source/'sim/frames/000000'/(v+'.png')).read_bytes() for v in ['left','right','wrist']}
identity,attestation=load_authority(Path(os.environ['QWEN_IDENTITY_MANIFEST']),Path(os.environ['QWEN_SERVER_ATTESTATION']))
client=QwenClient(base_url='http://127.0.0.1:8002/v1',api_key=Path(os.environ['QWEN_API_TOKEN_FILE']).read_text().strip(),identity=identity,attestation=attestation);client.verify()
selected={};traces=[]
try:
 for v in ['left','right']:
  p=segment_floor_candidates(images[v],out/v,task='Pick the boxed drink from the counter and place it in the sink.',view=v)
  ans=client.complete(observation_id='destination-probe-'+v,system_prompt='Identify the receiving floor of the task destination in RGB. Return JSON.',instruction=p['instruction'],public_state={},images=p['images'],response_schema=p['response_schema'],max_tokens=700,image_roles=p['image_roles'])
  trace={'view':v,'response':ans.command,'evidence':ans.evidence};traces.append(trace);(out/'choices.json').write_text(json.dumps(traces,indent=2));print(trace['view'],trace['response'],flush=True)
  if ans.command['visible']:
   chosen=next(c for c in p['candidates'] if c['id']==ans.command['mask_id']);selected[v]=chosen
 (out/'selected.json').write_text(json.dumps(selected,indent=2))
 if len(selected)==2:
  mask=cv2.imread(selected['right']['mask_path'],cv2.IMREAD_GRAYSCALE)>0
  geom=temporal_floor_point(images['left'],images['right'],data['calibration']['left'],data['calibration']['right'],mask,data['public_state'])
  (out/'geometry.json').write_text(json.dumps(geom,indent=2));print('geometry',geom['status'],geom['evidence'].get('reason'),geom.get('target_base_m'),flush=True)
finally:client.close()
