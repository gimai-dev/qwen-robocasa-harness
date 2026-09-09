import json,os
from pathlib import Path
import cv2,numpy as np
from adaptive.rgb_grounding import _triangulate
from PIL import Image,ImageDraw,ImageFont
from robocasa_inspect.model_client import QwenClient,load_authority

root=Path('/home/jli/state/destination-anchor-20260908');out=root/'drain-roi-probe';out.mkdir(exist_ok=True)
source=Path('/home/jli/state/grasp-retry-20260908/PickPlaceCounterToSink-baseline');data=json.loads((source/'sam2-source/public-input.json').read_text())
identity,attestation=load_authority(Path(os.environ['QWEN_IDENTITY_MANIFEST']),Path(os.environ['QWEN_SERVER_ATTESTATION']))
client=QwenClient(base_url='http://127.0.0.1:8002/v1',api_key=Path(os.environ['QWEN_API_TOKEN_FILE']).read_text().strip(),identity=identity,attestation=attestation);client.verify()
points={};traces=[]
try:
 for v in ['left','right']:
  folder=root/'initial-probe'/v;sam=json.loads((folder/'sam-result.json').read_text());floor=json.loads((root/'initial-probe/selected.json').read_text())[v]
  floor_mask=cv2.imread(floor['mask_path'],0)>0;y,x=np.where(floor_mask)
  hull=cv2.convexHull(np.column_stack((x,y)).astype(np.int32));region=np.zeros_like(floor_mask,dtype=np.uint8);cv2.fillConvexPoly(region,hull,1)
  candidates=[]
  for c in sam['candidates']:
   px,py=c['max_interior_pixel']
   if c['area']<.1*floor['area'] and region[py,px]:candidates.append(c)
  ids=[c['id'] for c in candidates];print('roi_ids',v,ids,flush=True)
  gallery=Image.new('RGB',(256*len(candidates),280),'white');draw=ImageDraw.Draw(gallery);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
  original=Image.open(folder/'original.png')
  for i,c in enumerate(candidates):
   x,y,w,h=c['bbox'];crop=original.crop((max(0,x-4),max(0,y-4),min(256,x+w+5),min(256,y+h+5)));crop.thumbnail((248,244));crop=crop.resize((max(1,int(crop.width*min(248/crop.width,244/crop.height))),max(1,int(crop.height*min(248/crop.width,244/crop.height)))))
   gallery.paste(crop,(i*256+4,32));draw.text((i*256+4,2),f'id={c["id"]}',font=font,fill='black')
  gallery.save(out/(v+'-gallery.png'))
  instruction=f'Task: Pick the boxed drink from the counter and place it in the sink. Locate a fixed landmark on the destination floor: select ONLY the small circular METAL DRAIN at the bottom of the sink. Do not select the orange loose object, whole basin, faucet, rim, robot, or countertop. All three images derive from the SAME {v} camera: original scene, the same original scene, then a numbered enlarged crop gallery of small components inside the independently selected destination basin. Choose the drain mask ID, or visible=false and null if absent. Describe its appearance.'
  schema={'type':'object','properties':{'visible':{'type':'boolean'},'mask_id':{'anyOf':[{'type':'integer','enum':ids},{'type':'null'}]},'description':{'type':'string'}},'required':['visible','mask_id','description'],'additionalProperties':False}
  ans=client.complete(observation_id='drain-mask-'+v,system_prompt='Identify the specified fixed landmark in the supplied RGB crop gallery. Return JSON.',instruction=instruction,public_state={},images={'left':(folder/'original.png').read_bytes(),'right':(folder/'original.png').read_bytes(),'wrist':(out/(v+'-gallery.png')).read_bytes()},response_schema=schema,max_tokens=700)
  traces.append({'view':v,'instruction':instruction,'response':ans.command,'evidence':ans.evidence});(out/'choices.json').write_text(json.dumps(traces,indent=2));print(v,ans.command,flush=True)
  if ans.command['visible']:
   c=next(c for c in sam['candidates'] if c['id']==ans.command['mask_id']);mask=cv2.imread(c['mask_path'],0)>0;y,x=np.where(mask);points[v]=[float(x.mean()),float(y.mean())]
 if len(points)==2:
  g=_triangulate(points,data['calibration']);print('geometry',g,flush=True)
  if g is not None:(out/'geometry.json').write_text(json.dumps({'target_world_m':g[0].tolist(),'ray_gap_m':g[1],'one_pixel_sensitivity_m':g[2],'pixels':points},indent=2))
finally:client.close()
