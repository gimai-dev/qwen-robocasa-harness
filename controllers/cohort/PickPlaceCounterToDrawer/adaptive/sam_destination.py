"""Automatic RGB drawer-floor mask proposals selected by the driver's Qwen call."""
import json
import os
from pathlib import Path
import subprocess

SAM_PYTHON = '/home/jli/state/qwen-rgb-sam2/.venv/bin/python'
SAM_WORKER = '/home/jli/state/qwen-rgb-sam2/probe/segment_amg_v2.py'


def segment_floor_candidates(image_bytes, work_dir):
    """Return mask IDs/paths plus three labelled gallery image slots for Qwen."""
    folder = Path(work_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    image_path = folder/'original.png'
    image_path.write_bytes(image_bytes)
    result_path = folder/'sam-result.json'
    with (folder/'sam-worker.log').open('w') as log:
        subprocess.run([SAM_PYTHON,SAM_WORKER,'--image',str(image_path),
                        '--result',str(result_path)], check=True, timeout=180,
                       env={**os.environ,'CUDA_VISIBLE_DEVICES':'',
                            'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'},
                       stdout=log,stderr=subprocess.STDOUT)
    evidence = json.loads(result_path.read_text())
    candidates = evidence['candidates']
    instruction = (
        'Task: place the held object inside the open drawer. All three images show '
        'the SAME wrist camera observation. The left image slot is the original '
        'scene. The right image slot shows numbered binary masks, with white pixels '
        'marking each candidate in original image coordinates. The wrist slot shows original-color crops from '
        'the same mask bounding boxes, with the same IDs. Select the mask covering '
        'the interior floor of the open drawer, the receiving surface on which the '
        'object can be placed. Distinguish that floor from the countertop, sink, '
        'drawer front, and robot gripper. Return null if no candidate covers the '
        'receiving floor. Describe the selected region.'
    )
    schema = {
        'type':'object', 'properties':{
            'visible':{'type':'boolean'},
            'mask_id':{'anyOf':[{'type':'integer','enum':[c['id'] for c in candidates]},
                               {'type':'null'}]},
            'description':{'type':'string'}},
        'required':['visible','mask_id','description'], 'additionalProperties':False,
    }
    return {'candidates':candidates, 'images':{
                'left':image_bytes,'right':(folder/'comparison.png').read_bytes(),
                'wrist':(folder/'gallery.png').read_bytes()},
            'image_roles':{'left':'original_wrist','right':'same_wrist_candidate_masks',
                           'wrist':'same_wrist_candidate_crops'},
            'instruction':instruction,'response_schema':schema,'evidence':evidence}


def ground_destination_mask(image_bytes, work_dir, select_callback):
    """Select the drawer floor with a driver-owned, counted Qwen callback.

    ``select_callback(instruction, images, response_schema)`` returns the model's
    command dictionary with visible/mask_id/description. The selected mask remains
    in the original image pixel frame for temporal RGB correspondence. The result
    includes mask PNG bytes and its persistent path; it does not estimate depth.
    """
    proposals = segment_floor_candidates(image_bytes, work_dir)
    folder = Path(work_dir).resolve()
    trace = {'status':'started','instruction':proposals['instruction'],
             'image_roles':proposals['image_roles'],
             'response_schema':proposals['response_schema'],'response':None}
    trace_path = folder/'selection.json'
    if not proposals['candidates']:
        trace.update(status='unresolved',reason='no_mask_proposals')
        trace_path.write_text(json.dumps(trace,indent=2))
        return {'status':'unresolved','mask_path':None,'mask_bytes':None,
                'evidence':{'sam':proposals['evidence'],'selection':trace}}
    trace_path.write_text(json.dumps(trace,indent=2))
    try:
        response = select_callback(proposals['instruction'],proposals['images'],
                                   proposals['response_schema'])
    except Exception as error:
        trace.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
        trace_path.write_text(json.dumps(trace,indent=2))
        raise
    trace.update(status='complete',response=response)
    trace_path.write_text(json.dumps(trace,indent=2))
    selected = next((c for c in proposals['candidates']
                     if c['id']==response.get('mask_id')),None)
    evidence = {'sam':proposals['evidence'],'selection':trace}
    if not response.get('visible') or selected is None:
        return {'status':'unresolved','mask_path':None,'mask_bytes':None,'evidence':evidence}
    mask_path = selected['mask_path']
    return {'status':'selected','mask_id':selected['id'],'mask_path':mask_path,
            'mask_bytes':Path(mask_path).read_bytes(),
            'description':response['description'],'candidate':selected,'evidence':evidence}
