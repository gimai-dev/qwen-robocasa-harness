import json,os
from pathlib import Path
from adaptive.skill_driver import object_schema
from adaptive.rgb_grounding import _triangulate
from robocasa_inspect.model_client import QwenClient,load_authority

root=Path('/home/jli/state/destination-anchor-20260908');out=root/'landmark-probe';out.mkdir(exist_ok=True)
source=Path('/home/jli/state/grasp-retry-20260908/PickPlaceCounterToSink-baseline')
data=json.loads((source/'sam2-source/public-input.json').read_text());images={v:(source/'sim/frames/000000'/(v+'.png')).read_bytes() for v in ['left','right','wrist']}
identity,attestation=load_authority(Path(os.environ['QWEN_IDENTITY_MANIFEST']),Path(os.environ['QWEN_SERVER_ATTESTATION']))
client=QwenClient(base_url='http://127.0.0.1:8002/v1',api_key=Path(os.environ['QWEN_API_TOKEN_FILE']).read_text().strip(),identity=identity,attestation=attestation);client.verify()
instruction='Task: Pick the boxed drink from the counter and place it in the sink. Before pickup, localize the destination using a FIXED landmark on its receiving floor: the center of the circular metal drain at the bottom of the sink. Do not select the orange loose object, faucet, rim, or countertop. Images are left, right, wrist camera. Every visible point must refer to the SAME drain center. Give point_2d normalized 0..1000, top-left [0,0], bottom-right [1000,1000]; invisible views return null. Describe evidence per view.'
try:
 ans=client.complete(observation_id='initial-drain-probe',system_prompt='Locate the specified fixed destination landmark in RGB. Return JSON.',instruction=instruction,public_state={},images=images,response_schema=object_schema(),max_tokens=700)
 (out/'response.json').write_text(json.dumps({'instruction':instruction,'response':ans.command,'evidence':ans.evidence},indent=2));print(ans.command,flush=True)
 points={v:[x*256/1000 for x in ans.command[v]['point_2d']] for v in ['left','right'] if ans.command[v]['visible']}
 if len(points)==2:
  g=_triangulate(points,data['calibration']);print('geometry',g,flush=True)
  if g is not None:(out/'geometry.json').write_text(json.dumps({'target_world_m':g[0].tolist(),'ray_gap_m':g[1],'one_pixel_sensitivity_m':g[2],'pixels':points},indent=2))
finally:client.close()
