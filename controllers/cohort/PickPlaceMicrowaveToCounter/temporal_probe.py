import json,os
from pathlib import Path
from adaptive.grasp_visual import temporal_grasp_images,grasp_check_instruction
from robocasa_inspect.model_client import QwenClient,load_authority
root=Path('/home/jli/state/destination-anchor-20260908');p=root/'PickPlaceCounterToSink-anchor'
samples=[]
for seq in [9,10,11]:
 d=json.loads((p/f'sim/mailbox/observation-{seq:06d}.json').read_text());samples.append({'state':d['public_state'],'images':{v:(p/f'sim/frames/{seq:06d}'/(v+'.png')).read_bytes() for v in ['left','right','wrist']}})
images,meta=temporal_grasp_images(samples,root/'temporal-probe')
state=samples[-1]['state'];gap=abs(state['state.gripper_qpos'][0]-state['state.gripper_qpos'][1])
instruction=grasp_check_instruction('Pick the boxed drink from the counter and place it in the sink.','carton',meta['camera'],.07,gap)
identity,attestation=load_authority(Path(os.environ['QWEN_IDENTITY_MANIFEST']),Path(os.environ['QWEN_SERVER_ATTESTATION']))
c=QwenClient(base_url='http://127.0.0.1:8002/v1',api_key=Path(os.environ['QWEN_API_TOKEN_FILE']).read_text().strip(),identity=identity,attestation=attestation);c.verify()
schema={'type':'object','properties':{'held':{'type':'boolean'},'description':{'type':'string'}},'required':['held','description'],'additionalProperties':False}
try:
 ans=c.complete(observation_id='temporal-held-probe',system_prompt='Locate objects in the supplied images. Return the requested JSON.',instruction=instruction,public_state={},images=images,response_schema=schema,max_tokens=700,image_roles={'left':'before_close','right':'after_close','wrist':'after_lift'})
 out={'response':ans.command,'evidence':ans.evidence,'context':meta};(root/'temporal-probe/response.json').write_text(json.dumps(out,indent=2));print(ans.command,flush=True)
finally:c.close()
