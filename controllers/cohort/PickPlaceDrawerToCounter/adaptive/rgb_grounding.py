"""Public RGB geometry for a Qwen-selected black ladle.

This is a narrow classical perception tool. It estimates visible bowl/handle
features; it neither chooses a robot action nor claims contact or task success.
"""
from itertools import combinations
import re

import cv2
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, distance_transform_edt, label

from .image_servo import (
    _camera_geometry, _closest_ray_points, _pixel_ray_world, _rotation_xyzw,
)


def _features(image_bytes, point):
    rgb = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w = rgb.shape[:2]
    seed = np.asarray(point, dtype=float) * [w / 1000, h / 1000]
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    labels, count = label(gray < 65, np.ones((3, 3)))
    components = []
    for index in range(1, count + 1):
        yy, xx = np.where(labels == index)
        if len(xx) < 6:
            continue
        if not np.any((abs(xx - seed[0]) <= 32) & (abs(yy - seed[1]) <= 32)):
            continue
        components.append((float(np.min((xx-seed[0])**2 + (yy-seed[1])**2)), index))
    if not components:
        return None
    _, selected = min(components)
    # Use the neighborhood for selection, retaining the entire silhouette.
    # Closing repairs sparse grid pixels and small gaps in the visible dark rim.
    mask = binary_fill_holes(binary_closing(labels == selected, np.ones((3, 3))))
    return _features_from_mask(mask, seed.tolist())


def _features_from_mask(mask, coarse_pixel=None):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    clearance = distance_transform_edt(mask)
    by, bx = np.unravel_index(np.argmax(clearance), clearance.shape)
    bowl = np.array([bx, by], dtype=float)
    radius = float(clearance[by, bx])
    yy, xx = np.where(mask)
    pixels = np.column_stack((xx, yy)).astype(float)
    tip = pixels[np.argmax(np.sum((pixels-bowl)**2, axis=1))]
    length = float(np.linalg.norm(tip-bowl))
    if radius < 2:
        return None
    axis = (tip-bowl)/max(length,1.)
    yy_grid, xx_grid = np.indices(mask.shape)
    along = (xx_grid-bx)*axis[0] + (yy_grid-by)*axis[1]
    across = abs((xx_grid-bx)*axis[1] - (yy_grid-by)*axis[0])
    thin = (mask & (along > 1.5*radius) & (along < .9*length)
            & (clearance <= .55*radius) & (across < radius))
    features = {'bowl_center': {'pixel':bowl, 'support':clearance >= .65*radius}}
    hy, hx = np.where(thin)
    if length >= 3*radius and len(hx) >= 4:
        expected = bowl + .6*length*axis
        candidates = np.column_stack((hx, hy))
        index = np.argmin(np.sum((candidates-expected)**2, axis=1))
        features['handle_center'] = {'pixel':candidates[index].astype(float), 'support':thin}
    return {'features':features, 'clearance':clearance, 'mask':mask,
            'evidence':{'coarse_pixel':coarse_pixel, 'mask_area_px':int(mask.sum()),
                        'mask_box_px':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
                        'bowl_radius_px':radius, 'tip_pixel':tip.tolist(),
                        'features':{name:value['pixel'].tolist() for name,value in features.items()}}}


def _epipolar_line(pixel, camera_a, camera_b):
    origin, direction = _pixel_ray_world(tuple(pixel), _camera_geometry(camera_a))
    rotation = np.asarray(camera_b['camera_xmat_world'])
    center = np.asarray(camera_b['camera_position_world_m'])
    projections = []
    for distance in (.5, 2.):
        local = rotation.T @ (np.asarray(origin)+distance*np.asarray(direction)-center)
        z = -local[2]
        projections.append([camera_b['fx_px']*local[0]+camera_b['cx_px']*z,
                            -camera_b['fy_px']*local[1]+camera_b['cy_px']*z, z])
    line = np.cross(*projections)
    norm = np.linalg.norm(line[:2])
    return line/norm if norm > 1e-8 else None


def _triangulate(points, calibration):
    a, b = list(points)
    oa, da = _pixel_ray_world(tuple(points[a]), _camera_geometry(calibration[a]))
    ob, db = _pixel_ray_world(tuple(points[b]), _camera_geometry(calibration[b]))
    try:
        pa, pb, gap, sa, sb = _closest_ray_points(oa, da, ob, db)
    except ValueError:
        return None
    if sa <= 0 or sb <= 0 or gap > .02:
        return None
    midpoint = (np.asarray(pa)+np.asarray(pb))/2
    # At this 256px resolution, near-parallel rays can fit while depth is unstable.
    max_shift = 0.
    for camera in (a, b):
        for axis in (0, 1):
            for offset in (-1., 1.):
                shifted = {name:list(pixel) for name,pixel in points.items()}
                shifted[camera][axis] += offset
                sa_origin, sa_dir = _pixel_ray_world(tuple(shifted[a]), _camera_geometry(calibration[a]))
                sb_origin, sb_dir = _pixel_ray_world(tuple(shifted[b]), _camera_geometry(calibration[b]))
                try:
                    qa, qb, _, ta, tb = _closest_ray_points(sa_origin, sa_dir, sb_origin, sb_dir)
                except ValueError:
                    return None
                if ta <= 0 or tb <= 0:
                    return None
                max_shift = max(max_shift, float(np.linalg.norm((np.asarray(qa)+np.asarray(qb))/2-midpoint)))
    if max_shift > .04:
        return None
    return midpoint, float(gap), max_shift


def _paired_feature(name, a, b, views, calibration):
    fa, fb = views[a]['features'].get(name), views[b]['features'].get(name)
    if fa is None or fb is None:
        return None
    line = _epipolar_line(fa['pixel'], calibration[a], calibration[b])
    if line is None:
        return None
    yy, xx = np.where(fb['support'])
    band = abs(line[0]*xx+line[1]*yy+line[2]) <= 1.5
    xx, yy = xx[band], yy[band]
    if len(xx) == 0:
        return None
    possible = np.column_stack((xx, yy))
    # Same named feature first, with the second image constrained by calibration.
    index = np.argmin(np.sum((possible-fb['pixel'])**2, axis=1))
    matched = possible[index].astype(float)
    # A match elsewhere on the long handle is not the same semantic midpoint.
    if np.linalg.norm(matched-fb['pixel']) > max(3., views[b]['evidence']['bowl_radius_px']):
        return None
    points = {a:fa['pixel'].tolist(), b:matched.tolist()}
    geometry = _triangulate(points, calibration)
    if geometry is None:
        return None
    world, gap, sensitivity = geometry
    return {'feature':name, 'paired_points':points, 'target_world_m':world.tolist(),
            'ray_gap_m':gap, 'one_pixel_sensitivity_m':sensitivity}


def ground_dark_object(images, coarse, calibration, public_state):
    """Return visible black-ladle geometry from RGB, Qwen text, and public cameras.

    `coarse` has target plus per-camera visible/description/point_2d (0..1000).
    Grounded results prefer a paired handle center. A bowl-only result remains a
    localization landmark and explicitly sets `grasp_candidate` false.
    """
    unresolved = _unresolved('classical_public_rgb_black_ladle')
    if not _supported(coarse):
        return unresolved
    views = {}
    for camera in ('left','right','wrist'):
        observation = coarse.get(camera,{})
        if camera not in images or camera not in calibration or not observation.get('visible'):
            continue
        point = observation.get('point_2d')
        if point is None:
            continue
        result = _features(images[camera], point)
        if result is not None:
            views[camera] = result
    return _ground_views(views, calibration, public_state, 'classical_public_rgb_black_ladle')


def _supported(coarse):
    target = str(coarse.get('target', '')).lower()
    descriptions = ' '.join(str(coarse.get(name, {}).get('description', ''))
                            for name in ('left', 'right', 'wrist')).lower()
    return (re.search(r'\bladle\b', target) is not None and
            re.search(r'\b(black|dark)\b', target+' '+descriptions) is not None)


def _unresolved(source):
    return {'status':'unresolved', 'paired_points':None, 'target_base_m':None,
            'feature':None, 'grasp_candidate':False, 'candidates':[],
            'evidence':{'source':source, 'reason':'unsupported_target'}}


def ground_ladle_masks(masks, coarse, calibration, public_state):
    """Pair bowl/handle features of explicit public-RGB black-ladle masks.

    Each mask is a full-resolution boolean array in its camera's pixel frame.
    Prefer a paired handle; bowl-only results are localization landmarks.
    """
    source = 'sam2_public_rgb_elongated_utensil'
    if not any(word in coarse.get('target','').lower() for word in ('spoon','ladle','whisk')):
        return _unresolved(source)
    views = {}
    for camera in ('left', 'right', 'wrist'):
        if camera not in masks or camera not in calibration or not coarse.get(camera, {}).get('visible'):
            continue
        result = _features_from_mask(masks[camera])
        if result is not None:
            views[camera] = result
    return _ground_views(views, calibration, public_state, source)


def _ground_views(views, calibration, public_state, source):
    unresolved = _unresolved(source)
    evidence = {'source':source,
                'reason':'no_corresponding_features',
                'per_view':{camera:{**data['evidence'],'pixel':data['evidence']['features'].get('handle_center',data['evidence']['features']['bowl_center'])} for camera,data in views.items()},
                'method':'Qwen appearance and coarse point; RGB silhouette; bowl/handle features; epipolar correspondence',
                'observations':'RGB and public camera calibration only'}
    rotation = np.asarray(_rotation_xyzw(public_state['state.base_rotation']))
    base = np.asarray(public_state['state.base_position'])
    candidates = []
    # Both external cameras are legal control cameras; pair either with wrist.
    for a,b in combinations(views,2):
        for feature in ('bowl_center','handle_center'):
            candidate = _paired_feature(feature,a,b,views,calibration)
            if candidate is not None:
                world = np.asarray(candidate['target_world_m'])
                candidate['target_base_m'] = (rotation.T @ (world-base)).tolist()
                candidate['grasp_candidate'] = feature == 'handle_center'
                candidates.append(candidate)
    if not candidates:
        return {**unresolved,'evidence':evidence}
    for handle in [c for c in candidates if c['feature']=='handle_center']:
        bowl = next((c for c in candidates if c['feature']=='bowl_center' and c['paired_points'].keys()==handle['paired_points'].keys()),None)
        if bowl is None:
            handle['grasp_candidate'] = False
            continue
        axis = np.asarray(handle['target_base_m'])-np.asarray(bowl['target_base_m'])
        length = np.linalg.norm(axis)
        if length < .015:
            handle['grasp_candidate'] = False
            continue
        axis /= length
        # Reference closing direction in the base XY plane, perpendicular to the
        # reconstructed handle. Qwen/IK chooses the full feasible gripper pose.
        closing = np.cross([0.,0.,1.],axis)
        if np.linalg.norm(closing) < .1:
            handle['grasp_candidate'] = False
            continue
        closing /= np.linalg.norm(closing)
        handle['handle_axis_base'] = axis.tolist()
        handle['jaw_closing_axis_base'] = closing.tolist()
    preferred = next((c for c in candidates if c['grasp_candidate']), candidates[0])
    evidence['reason'] = 'paired_visible_feature'
    return {'status':'grounded', **preferred, 'candidates':candidates, 'evidence':evidence}
