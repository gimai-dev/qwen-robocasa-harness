"""Qwen coarse point -> classical bowl seed -> isolated CPU SAM -> stereo handle.

Run on h200-4. SAM's dependencies live outside the simulator environment in
/home/jli/state/qwen-rgb-sam2. This tool makes no model-planner or simulator calls.
"""
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np

from .rgb_grounding import ground_dark_object, ground_ladle_masks
from .image_servo import _camera_geometry, _pixel_ray_world, _rotation_xyzw

SAM_PYTHON = '/home/jli/state/qwen-rgb-sam2/.venv/bin/python'
SAM_WORKER = '/home/jli/state/qwen-rgb-sam2/probe/segment_points.py'


def _whole_silhouette_center(result, masks, calibration, public_state):
    """Estimate footprint center on the selected stereo handle's height plane."""
    result['object_center_base_m'] = None
    if result['status'] != 'grounded' or result['feature'] != 'handle_center':
        return
    plane_z = float(result['target_world_m'][2])
    rotation = np.asarray(_rotation_xyzw(public_state['state.base_rotation']))
    base = np.asarray(public_state['state.base_position'])
    per_view = {}
    centers = []
    # Use exactly the views supporting this handle depth, not other SAM masks.
    for view in result['paired_points']:
        yy, xx = np.where(masks[view])
        camera = _camera_geometry(calibration[view])
        projected = []
        for u, v in zip(xx, yy):
            origin, direction = _pixel_ray_world((float(u), float(v)), camera)
            origin, direction = np.asarray(origin), np.asarray(direction)
            if abs(direction[2]) < 1e-8:
                return
            distance = (plane_z - origin[2]) / direction[2]
            if distance <= 0:
                return
            projected.append((rotation.T @ (origin + distance * direction - base))[:2])
        rectangle = cv2.minAreaRect(np.asarray(projected, dtype=np.float32))
        cx, cy = rectangle[0]
        cz = (plane_z - base[2] - rotation[2, 0] * cx
              - rotation[2, 1] * cy) / rotation[2, 2]
        center = [float(cx), float(cy), float(cz)]
        centers.append(center)
        per_view[view] = {
            'center_base_m': center,
            'rectangle_dimensions_m': list(rectangle[1]),
            'rectangle_base_xy_m': cv2.boxPoints(rectangle).tolist(),
            'mask_area_px': int(len(xx)),
        }
    result['object_center_base_m'] = np.mean(centers, axis=0).tolist()
    result['evidence']['object_center'] = {
        'method': 'mean of full SAM silhouette minimum-area rectangle centers on horizontal handle plane',
        'plane_world_z_m': plane_z,
        'per_view': per_view,
    }


def ground_sam_ladle(images, coarse, calibration, public_state, work_dir):
    """Return serializable geometry, preferring the mutually supported handle.

    Persist public input images, Qwen coarse text, automatic seed evidence, SAM
    masks, and final geometry in ``work_dir``. Bowl-only geometry explicitly has
    ``grasp_candidate=False``. Qwen remains responsible for planning and actions.
    """
    folder = Path(work_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    classical = ground_dark_object(images, coarse, calibration, public_state)
    (folder/'classical-seeds.json').write_text(json.dumps(classical, indent=2))
    (folder/'public-input.json').write_text(json.dumps({
        'coarse':coarse, 'calibration':calibration, 'public_state':public_state,
    }, indent=2))
    views = {}
    for view, evidence in classical['evidence'].get('per_view', {}).items():
        image_path = folder/(view+'.png')
        image_path.write_bytes(images[view])
        views[view] = {'image_path':str(image_path),
                       'point_px':evidence['features']['bowl_center']}
    if len(views) < 2:
        result = {**classical, 'evidence':{**classical['evidence'],
                  'sam_reason':'fewer_than_two_automatic_bowl_seeds'}}
    else:
        request_path = folder/'sam-request.json'
        response_path = folder/'sam-result.json'
        request_path.write_text(json.dumps({'views':views}, indent=2))
        with (folder/'sam-worker.log').open('w') as log:
            subprocess.run([SAM_PYTHON, SAM_WORKER, '--request',str(request_path),
                            '--result',str(response_path)], check=True, timeout=120,
                           env={**os.environ, 'CUDA_VISIBLE_DEVICES':'',
                                'OMP_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4'},
                           stdout=log, stderr=subprocess.STDOUT)
        response = json.loads(response_path.read_text())
        masks = {view:cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)>0
                 for view,row in response['views'].items()}
        result = ground_ladle_masks(masks, coarse, calibration, public_state)
        _whole_silhouette_center(result, masks, calibration, public_state)
        result['evidence']['sam'] = response
        result['evidence']['seed_source'] = 'Qwen coarse point -> classical dark component bowl center'
        result['evidence']['work_dir'] = str(folder)
    (folder/'grounding.json').write_text(json.dumps(result, indent=2))
    return result
