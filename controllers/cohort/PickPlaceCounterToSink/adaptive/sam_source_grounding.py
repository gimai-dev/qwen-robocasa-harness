"""General Qwen-selected public RGB source masks with paired grasp geometry."""
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np

from .image_servo import _camera_geometry, _pixel_ray_world, _rotation_xyzw
from .rgb_grounding import _triangulate

SAM_PYTHON = '/home/jli/state/qwen-rgb-sam2/.venv/bin/python'
SAM_WORKER = str(Path(__file__).with_name('sam_source_worker.py'))
MAX_GRIPPER_WIDTH_M = .08


def _planar_pixels(mask, camera, world_z, rotation, base):
    yy, xx = np.where(mask)
    points = []
    for x,y in zip(xx,yy):
        origin,direction = _pixel_ray_world((float(x),float(y)),_camera_geometry(camera))
        origin,direction = np.asarray(origin),np.asarray(direction)
        if abs(direction[2]) < 1e-6:
            return None
        scale = (world_z-origin[2])/direction[2]
        if scale <= 0:
            return None
        points.append((rotation.T@(origin+scale*direction-base))[:2])
    return np.asarray(points,dtype=np.float32)


def ground_source_masks(masks, coarse, calibration, public_state):
    """Estimate paired semantic centers and horizontal grasp widths from masks.

    Width is the silhouette projected onto a horizontal plane through the stereo
    center. It is an RGB grasp approximation; this tool does not establish contact.
    """
    views = {}
    for view in ('left','right','wrist'):
        if view not in masks or view not in calibration:
            continue
        mask = np.asarray(masks[view],dtype=bool)
        yy,xx = np.where(mask)
        if len(xx) < 6:
            continue
        views[view] = {'mask':mask,'pixel':[float(xx.mean()),float(yy.mean())],
                       'mask_area_px':int(mask.sum())}
    evidence = {'source':'sam2_qwen_selected_public_rgb_object',
                'target':coarse.get('target'),
                'width_assumption':'horizontal plane through stereo semantic center',
                'per_view':{v:{k:x for k,x in r.items() if k!='mask'} for v,r in views.items()},
                'rejected_pairs':[]}
    rotation = np.asarray(_rotation_xyzw(public_state['state.base_rotation']))
    base = np.asarray(public_state['state.base_position'])
    candidates = []
    for a,b in combinations(views,2):
        points = {a:views[a]['pixel'],b:views[b]['pixel']}
        result = _triangulate(points,calibration)
        if result is None:
            evidence['rejected_pairs'].append({'views':[a,b],'reason':'inconsistent_or_unstable_stereo_centers'})
            continue
        world,gap,sensitivity = result
        planar = {v:_planar_pixels(views[v]['mask'],calibration[v],world[2],rotation,base)
                  for v in (a,b)}
        if any(p is None for p in planar.values()):
            evidence['rejected_pairs'].append({'views':[a,b],'reason':'invalid_horizontal_projection'})
            continue
        reference = 'wrist' if 'wrist' in planar else a
        box = cv2.boxPoints(cv2.minAreaRect(planar[reference]))
        edges = np.roll(box,-1,axis=0)-box
        lengths = np.linalg.norm(edges,axis=1)
        closing = edges[int(np.argmin(lengths))]
        if np.linalg.norm(closing) < .005:
            evidence['rejected_pairs'].append({'views':[a,b],'reason':'unresolved_mask_width'})
            continue
        closing /= np.linalg.norm(closing)
        widths = {v:float(np.ptp(p@closing)) for v,p in planar.items()}
        width = max(widths.values())
        if width > MAX_GRIPPER_WIDTH_M:
            evidence['rejected_pairs'].append({'views':[a,b],
                'reason':'projected_width_exceeds_gripper','widths_m':widths})
            continue
        if max(widths.values()) > 1.5*min(widths.values()):
            evidence['rejected_pairs'].append({'views':[a,b],
                'reason':'inconsistent_projected_widths','widths_m':widths})
            continue
        yaw = float(np.arctan2(closing[1],closing[0]))
        yaw = (yaw+np.pi/2)%np.pi-np.pi/2
        candidates.append({'feature':'object_center','grasp_candidate':True,
            'paired_points':points,'target_world_m':world.tolist(),
            'target_base_m':(rotation.T@(world-base)).tolist(),
            'ray_gap_m':gap,'one_pixel_sensitivity_m':sensitivity,
            'projected_width_m':width,'projected_widths_m':widths,
            'jaw_closing_axis_base':[float(np.cos(yaw)),float(np.sin(yaw)),0.],
            'jaw_closing_yaw_rad':yaw})
    if not candidates and any(r['reason'] in {'projected_width_exceeds_gripper','inconsistent_projected_widths'} for r in evidence['rejected_pairs']):
        from .source_visual_hull import ground_source_hull
        candidates, hull_rejections = ground_source_hull(views,calibration,public_state)
        evidence['visual_hull_rejected_pairs'] = hull_rejections
        evidence['fallback_assumption'] = 'outer volume of intersecting calibrated RGB silhouettes; grasp centered across hull width, stereo height retained'
    if not candidates:
        return {'status':'unresolved','feature':None,'grasp_candidate':False,
                'paired_points':None,'target_base_m':None,'candidates':[],
                'evidence':evidence}
    selected = min(candidates,key=lambda c:c['ray_gap_m'])
    return {'status':'grounded',**selected,'candidates':candidates,'evidence':evidence}


def source_selection_instruction(task,source_hint,view):
    return (f'Task: {task}\nSource name hint: {source_hint}. '
        'Identify ONLY the task source, respecting its object type, shape, and stated source location. '
        'Objects already in the destination are not substitutes. A similar color does not establish identity. '
        'The source may be outside this view; return visible=false and mask_id=null if none matches. '
        f'All three images derive from the SAME {view} camera: original scene, numbered original-color crops, '
        'then corresponding binary masks. Select the source mask ID and describe its shape and support surface.')


def ground_sam_source(images, coarse, calibration, public_state, work_dir, select_callback, *, task):
    """Propose masks, ask the driver's counted Qwen callback, and pair centers.

    Callback: ``select_callback(instruction, images, response_schema) -> command``.
    Each call uses original scene, original-color crops, and separate binary masks
    of ONE camera, explicitly described in the instruction. Return is serializable.
    """
    folder = Path(work_dir).resolve()
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'public-input.json').write_text(json.dumps({
        'coarse':coarse,'calibration':calibration,'public_state':public_state},indent=2))
    visible = [v for v in ('left','right','wrist')
               if v in images and v in calibration and coarse.get(v,{}).get('visible')]
    cameras = visible if len(visible)>=2 else [v for v in ('left','right','wrist')
                                             if v in images and v in calibration]
    request = {'views':{}}
    for view in cameras:
        image_path = folder/(view+'.png')
        image_path.write_bytes(images[view])
        request['views'][view] = {'image_path':str(image_path),
                                  'coarse_point_1000':coarse.get(view,{}).get('point_2d')}
    request_path,result_path = folder/'sam-request.json',folder/'sam-result.json'
    request_path.write_text(json.dumps(request,indent=2))
    with (folder/'sam-worker.log').open('w') as log:
        subprocess.run([SAM_PYTHON,SAM_WORKER,'--request',str(request_path),
                        '--result',str(result_path)],check=True,timeout=240,
                       env={**os.environ,
                            'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'},
                       stdout=log,stderr=subprocess.STDOUT)
    sam = json.loads(result_path.read_text())
    masks, selections = {},{}
    for view,row in sam['views'].items():
        ids = [c['id'] for c in row['candidates']]
        if not ids:
            selections[view] = {'status':'unresolved','reason':'no_mask_proposals'}
            continue
        instruction = source_selection_instruction(task,coarse.get('target','the task source'),view)
        schema = {'type':'object','properties':{
            'visible':{'type':'boolean'},
            'mask_id':{'anyOf':[{'type':'integer','enum':ids},{'type':'null'}]},
            'description':{'type':'string'}},
            'required':['visible','mask_id','description'],'additionalProperties':False}
        trace = {'status':'started','camera':view,'instruction':instruction,
                 'response_schema':schema,'response':None}
        trace_path = folder/(view+'-selection.json')
        trace_path.write_text(json.dumps(trace,indent=2))
        try:
            response = select_callback(instruction,{
                'left':images[view],'right':Path(row['crop_gallery_path']).read_bytes(),
                'wrist':Path(row['binary_gallery_path']).read_bytes()},schema)
        except Exception as error:
            trace.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
            trace_path.write_text(json.dumps(trace,indent=2))
            raise
        trace.update(status='complete',response=response)
        trace_path.write_text(json.dumps(trace,indent=2))
        selections[view] = trace
        selected = next((c for c in row['candidates'] if c['id']==response.get('mask_id')),None)
        if response.get('visible') and selected is not None:
            masks[view] = cv2.imread(selected['mask_path'],cv2.IMREAD_GRAYSCALE)>0
    if any(word in coarse.get('target','').lower() for word in ('spoon','ladle')):
        from .rgb_grounding import ground_ladle_masks
        from .sam_rgb_grounding import _whole_silhouette_center
        result=ground_ladle_masks(masks,coarse,calibration,public_state)
        _whole_silhouette_center(result,masks,calibration,public_state)
    else:result = ground_source_masks(masks,coarse,calibration,public_state)
    result['evidence'].update(sam=sam,selections=selections,work_dir=str(folder))
    (folder/'grounding.json').write_text(json.dumps(result,indent=2))
    return result
