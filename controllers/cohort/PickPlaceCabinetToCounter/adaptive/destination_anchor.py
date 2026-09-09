"""Sink anchor from a coarse drain point and readable local SAM candidates."""
import io,json
from itertools import combinations
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from .sam_destination import segment_floor_candidates
from .rgb_grounding import _triangulate
from .world_anchor import base_pose

def ground_sink_destination(images,calibration,state,folder,query,task):
    from .skill_driver import object_schema
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    coarse=query(f'Task: {task}. Locate ONLY the small circular metal drain on the sink bottom. Images are left camera, right camera, wrist camera. Return its visible center in normalized 0–1000 coordinates in each view. Loose food, faucet and robot are not the drain. Mark invisible views false.',images,object_schema(),{v:'original_'+v for v in images})
    (folder/'coarse-drain.json').write_text(json.dumps(coarse,indent=2))
    points={};selections=[]
    for view in ('left','right','wrist'):
        if not coarse[view]['visible'] or coarse[view]['point_2d'] is None:continue
        proposal=segment_floor_candidates(images[view],folder/view,task=task,view=view)
        point=np.array(coarse[view]['point_2d'])*255/1000
        candidates=[c for c in proposal['candidates'] if 12<c['area']<1300 and np.linalg.norm(np.asarray(c['max_interior_pixel'])-point)<35]
        candidates=sorted(candidates,key=lambda c:np.linalg.norm(np.asarray(c['max_interior_pixel'])-point))[:6]
        if not candidates:continue
        original=Image.open(io.BytesIO(images[view])).convert('RGB');gallery=Image.new('RGB',(256*len(candidates),280),'white');draw=ImageDraw.Draw(gallery);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
        for i,c in enumerate(candidates):
            x,y,w,h=c['bbox'];crop=original.crop((max(0,x-5),max(0,y-5),min(256,x+w+6),min(256,y+h+6)));scale=min(244/crop.width,240/crop.height);crop=crop.resize((int(crop.width*scale),int(crop.height*scale)));gallery.paste(crop,(256*i+5,35));draw.text((256*i+5,3),f'id={c["id"]}',font=font,fill='black')
        path=folder/view/'local-drain-gallery.png';gallery.save(path)
        schema={'type':'object','properties':{'visible':{'type':'boolean'},'mask_id':{'anyOf':[{'type':'integer','enum':[c['id'] for c in candidates]},{'type':'null'}]},'description':{'type':'string'}},'required':['visible','mask_id','description'],'additionalProperties':False}
        choice=query(f'All images come from the SAME {view} camera: original scene, repeated original scene, and enlarged numbered crops near the proposed drain. Select ONLY the circular metal sink drain. Identify its metallic rim/holes, and reject loose food, robot, faucet or basin. If no crop is the drain return false and null.',{'left':images[view],'right':images[view],'wrist':path.read_bytes()},schema,{'left':'original_'+view,'right':'original_'+view,'wrist':'local_drain_candidates'})
        selections.append({'view':view,'choice':choice,'candidate_ids':[c['id'] for c in candidates]})
        selected=next((c for c in candidates if c['id']==choice['mask_id']),None)
        if selected is None or not choice['visible']:
            hsv=cv2.cvtColor(cv2.imdecode(np.frombuffer(images[view],np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2HSV)
            geometric=[]
            for candidate in candidates:
                mask=cv2.imread(candidate['mask_path'],0)>0
                contours=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
                contour=max(contours,key=cv2.contourArea);perimeter=cv2.arcLength(contour,True)
                circularity=4*np.pi*cv2.contourArea(contour)/(perimeter*perimeter) if perimeter else 0.
                saturation=float(np.mean(hsv[:,:,1][mask]))
                if 20<candidate['area']<400 and saturation<50 and circularity>.65:geometric.append(candidate)
            if len(geometric)==1:
                selected=geometric[0]
                selections[-1]['geometry_fallback']={'mask_id':selected['id'],'reason':'unique compact, round, low-saturation component near semantic drain point; requires stereo agreement'}
        if selected is not None:
            y,x=np.where(cv2.imread(selected['mask_path'],0)>0);points[view]=[float(x.mean()),float(y.mean())]
        if len(points)>=2:
            for a,b in combinations(points,2):
                tri=_triangulate({a:points[a],b:points[b]},calibration)
                if tri is None:continue
                world,gap,sensitivity=tri;target,_=base_pose(world,np.eye(3),state)
                result={'status':'grounded','target_world_m':world.tolist(),'target_base_m':target.tolist(),'ray_gap_m':gap,'one_pixel_sensitivity_m':sensitivity,'pixels':points,'selections':selections,'method':'coarse Qwen drain bearing; enlarged local SAM crops; calibrated stereo'}
                (folder/'geometry.json').write_text(json.dumps(result,indent=2));return result
    return {'status':'unresolved','reason':'drain unresolved in two calibrated views','pixels':points,'selections':selections}
