"""Pre-pick sink anchor from Qwen-selected RGB landmarks and camera calibration."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .sam_destination import segment_floor_candidates
from .rgb_grounding import _triangulate
from .world_anchor import base_pose

def focused_landmark_gallery(proposals, floor, original_bytes, folder):
    import io
    mask=cv2.imread(floor['mask_path'],0)>0
    y,x=np.where(mask)
    hull=cv2.convexHull(np.column_stack((x,y)).astype(np.int32))
    region=np.zeros_like(mask,dtype=np.uint8);cv2.fillConvexPoly(region,hull,1)
    candidates=[c for c in proposals if c['area']<.1*floor['area']
                and region[c['max_interior_pixel'][1],c['max_interior_pixel'][0]]]
    original=Image.open(io.BytesIO(original_bytes)).convert('RGB')
    gallery=Image.new('RGB',(256*max(1,len(candidates)),280),'white')
    draw=ImageDraw.Draw(gallery)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    for i,c in enumerate(candidates):
        x,y,w,h=c['bbox']
        crop=original.crop((max(0,x-4),max(0,y-4),min(original.width,x+w+5),min(original.height,y+h+5)))
        scale=min(248/crop.width,244/crop.height)
        crop=crop.resize((max(1,int(crop.width*scale)),max(1,int(crop.height*scale))))
        gallery.paste(crop,(i*256+4,32));draw.text((i*256+4,2),f'id={c["id"]}',font=font,fill='black')
    path=folder/'landmark-gallery.png';gallery.save(path)
    return candidates,path

def ground_sink_destination(images,calibration,state,folder,query,task):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    points={};selections=[]
    for view in ('left','right'):
        p=segment_floor_candidates(images[view],folder/view,task=task,view=view)
        if not p['candidates']:
            return {'status':'unresolved','reason':f'no destination proposals in {view}'}
        choice=query(p['instruction'],p['images'],p['response_schema'],p['image_roles'])
        floor=next((c for c in p['candidates'] if c['id']==choice.get('mask_id')),None)
        if not choice['visible'] or floor is None:
            return {'status':'unresolved','reason':f'destination not visible in {view}','selection':choice}
        candidates,gallery=focused_landmark_gallery(p['candidates'],floor,images[view],folder/view)
        if not candidates:
            return {'status':'unresolved','reason':f'no fixed-landmark candidates in {view}'}
        ids=[c['id'] for c in candidates]
        schema={'type':'object','properties':{'visible':{'type':'boolean'},'mask_id':{'anyOf':[{'type':'integer','enum':ids},{'type':'null'}]},'description':{'type':'string'}},'required':['visible','mask_id','description'],'additionalProperties':False}
        instruction=(f'Task: {task} Locate a fixed landmark on the destination floor: select ONLY the small circular METAL DRAIN at the bottom of the sink. '
            'Do not select loose objects, whole basin, faucet, rim, robot, or countertop. '
            f'All three images derive from the SAME {view} camera: original scene, the same original scene, then a numbered enlarged crop gallery '
            'of small components inside the independently selected destination basin. Choose the drain mask ID, or visible=false and null if absent. Describe its appearance.')
        answer=query(instruction,{'left':images[view],'right':images[view],'wrist':gallery.read_bytes()},schema,
            {'left':f'original_{view}','right':f'original_{view}','wrist':f'{view}_destination_landmark_candidates'})
        selected=next((c for c in candidates if c['id']==answer.get('mask_id')),None)
        selections.append({'view':view,'floor':choice,'landmark':answer,'candidate_ids':ids})
        (folder/'selections.json').write_text(json.dumps(selections,indent=2))
        if not answer['visible'] or selected is None:
            return {'status':'unresolved','reason':f'fixed destination landmark missing in {view}','selections':selections}
        y,x=np.where(cv2.imread(selected['mask_path'],0)>0)
        points[view]=[float(x.mean()),float(y.mean())]
    result=_triangulate(points,calibration)
    if result is None:
        return {'status':'unresolved','reason':'destination landmark rays disagree','pixels':points,'selections':selections}
    world,gap,sensitivity=result
    target,_=base_pose(world,np.eye(3),state)
    output={'status':'grounded','target_world_m':world.tolist(),'target_base_m':target.tolist(),
        'ray_gap_m':gap,'one_pixel_sensitivity_m':sensitivity,'pixels':points,'selections':selections,
        'method':'Qwen basin selection; focused fixed-drain mask selection; calibrated stereo mask centroids'}
    (folder/'geometry.json').write_text(json.dumps(output,indent=2))
    return output
