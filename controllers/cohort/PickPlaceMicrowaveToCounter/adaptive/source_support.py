"""Grasp a visually selected bulb/clove using the adjacent sink-floor anchor."""
from pathlib import Path
import cv2,numpy as np
from .destination_anchor import ground_sink_destination
from .image_servo import _camera_geometry,_pixel_ray_world,_rotation_xyzw
from .sam_source_grounding import _planar_pixels
from .world_anchor import base_pose

def ground_on_sink(ground,coarse,images,calibration,state,folder,query,anchor=None):
    masks={}
    for view,row in ground.get('evidence',{}).get('selections',{}).items():
        response=row.get('response',{});sam=ground['evidence']['sam']['views'][view]
        selected=next((c for c in sam['candidates'] if c['id']==response.get('mask_id')),None)
        if response.get('visible') and selected:masks[view]=cv2.imread(selected['mask_path'],0)>0
    if not masks:return ground,anchor
    if anchor is None or anchor['status']!='grounded':
        anchor=ground_sink_destination(images,calibration,state,folder,query,'Locate the source sink floor adjacent to '+coarse['target']+'. If there are multiple basins, identify ONLY the drain in the basin containing this source; do not substitute the neighboring basin drain.')
    if anchor['status']!='grounded':return ground,anchor
    view=max(masks,key=lambda v:int(masks[v].sum()));mask=masks[view];yy,xx=np.where(mask)
    rotation=np.asarray(_rotation_xyzw(state['state.base_rotation']));base=np.asarray(state['state.base_position'])
    floor_z=anchor['target_world_m'][2];height=0.
    for _ in range(2):
        points=_planar_pixels(mask,calibration[view],floor_z+height,rotation,base)
        if points is None:return ground,anchor
        box=cv2.boxPoints(cv2.minAreaRect(points));edges=np.roll(box,-1,axis=0)-box;closing=edges[np.argmin(np.linalg.norm(edges,axis=1))];width=float(np.linalg.norm(closing))
        if not .005<width<.08:return ground,anchor
        height=width/2
    closing/=np.linalg.norm(closing)
    pixel=[float(xx.mean()),float(yy.mean())];origin,direction=_pixel_ray_world(pixel,_camera_geometry(calibration[view]));origin,direction=np.asarray(origin),np.asarray(direction)
    scale=(floor_z+height-origin[2])/direction[2]
    if scale<=0:return ground,anchor
    world=origin+scale*direction;target,_=base_pose(world,np.eye(3),state)
    candidate={'feature':'object_center','grasp_candidate':True,'target_world_m':world.tolist(),'target_base_m':target.tolist(),'projected_width_m':width,'jaw_closing_axis_base':[float(closing[0]),float(closing[1]),0.],'geometry_method':'source-mask ray intersected with locally anchored support plane plus estimated half-width height'}
    result={'status':'grounded',**candidate,'candidates':[candidate],'evidence':{**ground['evidence'],'source_support':{'anchor':anchor,'source_camera':view,'source_pixel':pixel,'estimated_center_height_above_floor_m':height,'shape_assumption':'bulb or clove center approximately half its minor width above the adjacent horizontal sink floor'}}}
    return result,anchor
