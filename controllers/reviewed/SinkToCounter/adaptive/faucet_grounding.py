"""Public RGB faucet lever localization for a closer viewing approach only."""
from itertools import combinations, product
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np

from .image_servo import _rotation_xyzw
from .rgb_grounding import _triangulate

SAM_PYTHON = '/home/jli/state/qwen-rgb-sam2/.venv/bin/python'
SAM_WORKER = '/home/jli/state/qwen-rgb-sam2/probe/segment_faucet.py'


def ground_faucet_candidates(proposals, calibration, public_state):
    """Find a unique short/thin stereo part among automatic faucet-box masks.

    Apparent tangent-plane dimensions discriminate the short lever from the long
    spout and counter edges in the saved fixture. They are not full 3D dimensions.
    """
    views = {}
    for view,row in proposals['views'].items():
        if view not in calibration:
            continue
        candidates = []
        for candidate in row['candidates']:
            mask = cv2.imread(candidate['mask_path'],cv2.IMREAD_GRAYSCALE)>0
            yy,xx = np.where(mask)
            if len(xx)<6:
                continue
            rectangle = cv2.minAreaRect(np.column_stack((xx,yy)).astype(np.float32))
            candidates.append({'id':candidate['id'], 'mask_path':candidate['mask_path'],
                'centroid_pixel':[float(xx.mean()),float(yy.mean())],
                'rectangle_sides_px':sorted(float(x) for x in rectangle[1])})
        views[view] = candidates
    rotation = np.asarray(_rotation_xyzw(public_state['state.base_rotation']))
    base = np.asarray(public_state['state.base_position'])
    accepted = []
    for a,b in combinations(views,2):
        for ca,cb in product(views[a],views[b]):
            points = {a:ca['centroid_pixel'],b:cb['centroid_pixel']}
            result = _triangulate(points,calibration)
            if result is None:
                continue
            world,gap,sensitivity = result
            if gap>=.01 or sensitivity>=.02:
                continue
            extents = {}
            for view,candidate in ((a,ca),(b,cb)):
                camera = calibration[view]
                distance = np.linalg.norm(world-np.asarray(camera['camera_position_world_m']))
                extents[view] = [float(side*distance/camera['fx_px'])
                                 for side in candidate['rectangle_sides_px']]
            if not all(.002<short<.03 and .02<long<.12 for short,long in extents.values()):
                continue
            accepted.append({'feature':'lever_shaft_center','paired_points':points,
                'target_world_m':world.tolist(),'target_base_m':(rotation.T@(world-base)).tolist(),
                'ray_gap_m':gap,'one_pixel_sensitivity_m':sensitivity,
                'apparent_tangent_extents_m':extents,
                'mask_ids':{a:ca['id'],b:cb['id']},
                'mask_paths':{a:ca['mask_path'],b:cb['mask_path']}})
    common = {'scope':'view_approach_only','grasp_candidate':False,
              'pivot_status':'unresolved','candidates':accepted,
              'evidence':{'source':'Qwen faucet boxes; SAM masks; public stereo; short/thin part dimensions',
                          'candidate_count':len(accepted),
                          'selection':'unique pair; mask IDs are never prescribed',
                          'part_extent_limits_m':{'width':[.002,.03],'length':[.02,.12]}}}
    if len(accepted)!=1:
        return {'status':'unresolved','target_world_m':None,'target_base_m':None,
                **common}
    return {'status':'grounded',**accepted[0],**common}


def ground_faucet_lever(images, calibration, public_state, work_dir, select_callback):
    """Localize a lever for viewing using a driver-owned Qwen box callback.

    Callback signature: ``select_callback(instruction, images, schema) -> dict``.
    The callback makes the only Qwen call. No hinge center or actuation pose is
    inferred: a grounded result explicitly has scope ``view_approach_only``.
    """
    folder = Path(work_dir).resolve()
    folder.mkdir(parents=True,exist_ok=True)
    for view,data in images.items():
        (folder/(view+'.png')).write_bytes(data)
    (folder/'public-input.json').write_text(json.dumps({
        'calibration':calibration,'public_state':public_state},indent=2))
    box = {'anyOf':[{'type':'array','items':{'type':'integer','minimum':0,'maximum':1000},
                     'minItems':4,'maxItems':4},{'type':'null'}]}
    view_schema = {'type':'object','properties':{
        'visible':{'type':'boolean'},'box_2d':box,'description':{'type':'string'}},
        'required':['visible','box_2d','description'],'additionalProperties':False}
    schema = {'type':'object','properties':{v:view_schema for v in ('left','right','wrist')},
              'required':['left','right','wrist'],'additionalProperties':False}
    instruction = (
        'The task is to adjust the running water temperature while keeping water on. '
        'Locate the faucet assembly so its short movable control lever and lever '
        'hinge can be inspected. For each camera, give a tight bounding box around '
        'the faucet assembly, including its short control lever beside the spout. '
        'Coordinates [left,top,right,bottom] use 0..1000 relative to that supplied '
        'image. Include the whole control lever and its pivot. Exclude the long '
        'blue water stream and most of the sink basin. Return null when hidden.'
    )
    trace = {'status':'started','instruction':instruction,'response_schema':schema,'response':None}
    trace_path = folder/'qwen-boxes.json'
    trace_path.write_text(json.dumps(trace,indent=2))
    try:
        boxes = select_callback(instruction,images,schema)
    except Exception as error:
        trace.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
        trace_path.write_text(json.dumps(trace,indent=2))
        raise
    trace.update(status='complete',response=boxes)
    trace_path.write_text(json.dumps(trace,indent=2))
    request = {'views':{}}
    for view in ('left','right','wrist'):
        row = boxes.get(view,{})
        if view in images and view in calibration and row.get('visible') and row.get('box_2d') is not None:
            request['views'][view] = {'image_path':str(folder/(view+'.png')),
                                      'box_1000':row['box_2d']}
    request_path,result_path = folder/'sam-request.json',folder/'sam-result.json'
    request_path.write_text(json.dumps(request,indent=2))
    if len(request['views'])<2:
        proposals = {'views':{}}
    else:
        with (folder/'sam-worker.log').open('w') as log:
            subprocess.run([SAM_PYTHON,SAM_WORKER,'--request',str(request_path),
                            '--result',str(result_path)],check=True,timeout=180,
                           env={**os.environ,'CUDA_VISIBLE_DEVICES':'',
                                'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'},
                           stdout=log,stderr=subprocess.STDOUT)
        proposals = json.loads(result_path.read_text())
    result = ground_faucet_candidates(proposals,calibration,public_state)
    result['evidence'].update(qwen_boxes=trace,sam=proposals,work_dir=str(folder))
    (folder/'grounding.json').write_text(json.dumps(result,indent=2))
    return result
