"""Calibrated temporal RGB geometry for a visible, gray circular bowl floor."""
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, label
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .image_servo import _camera_geometry, _closest_ray_points, _pixel_ray_world, _rotation_xyzw
from .sam_destination import segment_floor_candidates


def _select_mask(image_bytes, folder, view, select_callback):
    proposals = segment_floor_candidates(image_bytes, folder)
    roles = {'left': view + '_wrist_original',
             'right': view + '_wrist_binary_mask_overview',
             'wrist': view + '_wrist_original_color_crops'}
    instruction = (
        'Task: place the food in the stand mixer bowl. These three images '
        f'come from the SAME {view} wrist camera observation. The left slot is '
        'the original scene. The right slot shows numbered binary masks in '
        'original image coordinates; white pixels belong to each candidate. '
        'The wrist slot shows original-color crops with those same mask IDs. '
        'Select the circular gray interior floor of the stand mixer bowl. '
        'If only a complete bowl opening including its rim is proposed, select '
        'that opening. Distinguish the receiving bowl from the food, outer mixer '
        'housing, countertop and robot. Return null if no candidate covers the '
        'receiving bowl. Describe the selected original region.'
    )
    trace = {'image_roles': roles, 'instruction': instruction, 'response': None}
    if not proposals['candidates']:
        trace['reason'] = 'no_mask_proposals'
        return None, trace
    response = select_callback(instruction, proposals['images'],
                               proposals['response_schema'], roles)
    trace['response'] = response
    selected = next((c for c in proposals['candidates']
                     if c['id'] == response.get('mask_id')), None)
    Path(folder, 'selection.json').write_text(json.dumps(trace, indent=2))
    if not response.get('visible') or selected is None:
        return None, trace
    trace['selected_mask'] = selected
    return cv2.imread(selected['mask_path'], cv2.IMREAD_GRAYSCALE) > 0, trace


def _gray_floor(image_bytes, mask):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(distance.argmax(), distance.shape)
    # The observed smooth gray floor is separated from the dark curved wall.
    # The seed comes from the semantic mask, never from a supplied pixel.
    support = mask & (abs(gray.astype(float) - float(gray[y, x])) <= 20)
    components, _ = label(support, np.ones((3, 3)))
    floor = binary_fill_holes(components == components[y, x])
    return floor, {'interior_seed_px': [int(x), int(y)],
                   'seed_gray': int(gray[y, x]), 'floor_area_px': int(floor.sum())}


def _fit_circle(masks, calibrations, public_state):
    rays, centers, contours = [], [], []
    for mask, calibration in zip(masks, calibrations):
        found = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_NONE)[0]
        if not found:
            raise ValueError('no_floor_boundary')
        contour = max(found, key=cv2.contourArea).reshape(-1, 2)
        if len(contour) < 5:
            raise ValueError('insufficient_floor_boundary')
        geometry = _camera_geometry(calibration)
        ellipse = cv2.fitEllipse(contour.astype(np.float32))
        centers.append(_pixel_ray_world(ellipse[0], geometry))
        origin, direction = zip(*[_pixel_ray_world(tuple(map(float, p)), geometry)
                                  for p in contour])
        rays.append((np.asarray(origin), np.asarray(direction), calibration['fx_px']))
        contours.append(contour)
    pa, pb, gap, sa, sb = _closest_ray_points(*centers[0], *centers[1])
    if min(sa, sb) <= 0:
        raise ValueError('floor_center_behind_camera')
    center = (np.asarray(pa) + pb) / 2
    origin, direction, _ = rays[0]
    projected = origin + direction * ((center[2] - origin[:, 2]) / direction[:, 2])[:, None]
    radius = np.median(np.linalg.norm(projected[:, :2] - center[:2], axis=1))

    def residual(parameters):
        rows = []
        for origin, direction, fx in rays:
            distance = (parameters[2] - origin[:, 2]) / direction[:, 2]
            points = origin + direction * distance[:, None]
            rows.append((np.linalg.norm(points[:, :2] - parameters[:2], axis=1)
                         - parameters[3]) * fx / distance)
        return np.concatenate(rows)

    fitted = least_squares(residual, [*center, radius], loss='soft_l1',
                           f_scale=1., max_nfev=200)
    center, radius = fitted.x[:3], float(fitted.x[3])
    if not fitted.success or radius <= 0:
        raise ValueError('circle_fit_failed')
    angles = np.linspace(0, 2 * np.pi, 8192)
    circle = center + radius * np.column_stack((np.cos(angles), np.sin(angles), angles * 0))
    errors = {}
    for name, contour, calibration in zip(('previous', 'current'), contours, calibrations):
        local = (np.asarray(calibration['camera_xmat_world']).T @
                 (circle - np.asarray(calibration['camera_position_world_m'])).T).T
        if np.any(local[:, 2] >= 0):
            raise ValueError('fitted_circle_behind_camera')
        pixels = np.column_stack((calibration['fx_px'] * local[:, 0] / -local[:, 2]
                                  + calibration['cx_px'],
                                  -calibration['fy_px'] * local[:, 1] / -local[:, 2]
                                  + calibration['cy_px']))
        distance = cKDTree(pixels).query(contour)[0]
        errors[name] = {'median_px': float(np.median(distance)),
                        'p95_px': float(np.percentile(distance, 95)),
                        'max_px': float(distance.max()), 'boundary_count': len(contour)}
    if max(row['p95_px'] for row in errors.values()) > 2.5:
        raise ValueError('boundaries_do_not_fit_one_horizontal_circle')
    rotation = np.asarray(_rotation_xyzw(public_state['state.base_rotation']))
    base = np.asarray(public_state['state.base_position'])
    return {'status': 'grounded', 'target_world_m': center.tolist(),
            'target_base_m': (rotation.T @ (center - base)).tolist(), 'radius_m': radius,
            'evidence': {'boundary_errors': errors, 'center_ray_gap_m': float(gap),
                         'normal_world': [0., 0., 1.]}}


def ground_circle_destination(previous_image_bytes, current_image_bytes,
                              previous_calibration, current_calibration,
                              public_state, work_dir, select_callback):
    """Return a circular floor center from two wrist RGB views, without known size.

    ``select_callback(instruction, images, response_schema, image_roles)`` is the
    driver's counted Qwen query. Each view gets automatic SAM proposals and one
    semantic selection. The callback must preserve the supplied image roles.
    Targets denote the visible floor, before a caller adds release clearance.
    The geometry assumes a stationary horizontal circular floor with a smooth
    gray interior; this is not a texture or arbitrary-container depth solver.
    """
    baseline = float(np.linalg.norm(np.asarray(previous_calibration['camera_position_world_m'])
                                    - current_calibration['camera_position_world_m']))
    evidence = {'method': 'Qwen-selected SAM regions; automatic gray interior; calibrated horizontal circle fit',
                'camera_baseline_m': baseline, 'selections': {}}
    unresolved = {'status': 'unresolved', 'target_base_m': None, 'target_world_m': None,
                  'radius_m': None, 'evidence': evidence}
    if baseline < .02:
        evidence['reason'] = 'insufficient_camera_translation'
        return unresolved
    folder = Path(work_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    masks = []
    for view, image in [('previous', previous_image_bytes), ('current', current_image_bytes)]:
        view_folder = folder / view
        view_folder.mkdir(parents=True, exist_ok=True)
        mask, selection = _select_mask(image, view_folder, view, select_callback)
        evidence['selections'][view] = selection
        if mask is None or not mask.any():
            evidence['reason'] = 'receiving_bowl_not_selected'
            (folder / 'grounding.json').write_text(json.dumps(unresolved, indent=2))
            return unresolved
        floor, refinement = _gray_floor(image, mask)
        mask_path = view_folder / 'refined-floor-mask.png'
        cv2.imwrite(str(mask_path), floor.astype(np.uint8) * 255)
        selection['refinement'] = {**refinement, 'mask_path': str(mask_path)}
        masks.append(floor)
    evidence['calibrations'] = {'previous': previous_calibration, 'current': current_calibration}
    try:
        result = _fit_circle(masks, [previous_calibration, current_calibration], public_state)
        result['evidence'] = {**evidence, **result['evidence']}
    except ValueError as error:
        evidence['reason'] = str(error)
        result = unresolved
    (folder / 'grounding.json').write_text(json.dumps(result, indent=2))
    return result


def _circle_sensitivity(masks, calibrations):
    """Measure observed center-ray parallax and one-pixel metric sensitivity."""
    points = []
    for mask in masks:
        contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)[0]
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5:
            raise ValueError('insufficient_floor_boundary')
        points.append(cv2.fitEllipse(contour)[0])
    geometries = [_camera_geometry(c) for c in calibrations]

    def triangulate(pixels):
        rays = [_pixel_ray_world(tuple(pixel), geometry)
                for pixel, geometry in zip(pixels, geometries)]
        pa, pb, _, sa, sb = _closest_ray_points(*rays[0], *rays[1])
        if min(sa, sb) <= 0:
            raise ValueError('floor_center_behind_camera')
        return (np.asarray(pa) + pb) / 2, rays

    center, rays = triangulate(points)
    sensitivity = 0.
    for view in (0, 1):
        for axis in (0, 1):
            for offset in (-1., 1.):
                shifted = np.asarray(points).copy()
                shifted[view, axis] += offset
                perturbed, _ = triangulate(shifted)
                sensitivity = max(sensitivity, float(np.linalg.norm(perturbed - center)))
    angle = float(np.degrees(np.arccos(np.clip(np.dot(rays[0][1], rays[1][1]), -1., 1.))))
    return {'center_ray_parallax_deg': angle, 'one_pixel_sensitivity_m': sensitivity}


def ground_circle_destination_views(views, public_state, work_dir, select_callback, *, selection_cache=None):
    """Try recorded wrist views in order, caching the initial semantic floor.

    Each view supplies ``image`` bytes and ``calibration``. Pass chronological
    history including early side motion, not just the final ready image. The
    counted callback has the same four arguments as the pair API. A weak-parallax
    or clipped pair is skipped; the first supported circle is returned in the
    current public robot base frame. No object radius or recorded point is used.
    """
    evidence = {'method': 'cached Qwen/SAM floor; chronological calibrated circle pairs',
                'history_attempts': []}
    unresolved = {'status': 'unresolved', 'target_base_m': None, 'target_world_m': None,
                  'radius_m': None, 'evidence': evidence}
    if len(views) < 2:
        evidence['reason'] = 'insufficient_recorded_views'
        return unresolved
    initial_camera = np.asarray(views[0]['calibration']['camera_position_world_m'])
    candidates = [(index, float(np.linalg.norm(initial_camera -
                   np.asarray(view['calibration']['camera_position_world_m']))))
                  for index, view in enumerate(views[1:], 1)]
    candidates = [(index, baseline) for index, baseline in candidates if baseline >= .02]
    if not candidates:
        evidence['reason'] = 'insufficient_camera_translation'
        return unresolved
    folder = Path(work_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)

    cache = {} if selection_cache is None else selection_cache

    def select_floor(index):
        if index in cache:
            return cache[index]
        view = views[index]
        view_folder = folder / f'view-{index}'
        view_folder.mkdir(parents=True, exist_ok=True)
        mask, selection = _select_mask(view['image'], view_folder,
                                       f'historical_{index}', select_callback)
        if mask is None or not mask.any():
            cache[index] = (None, selection)
            return cache[index]
        floor, refinement = _gray_floor(view['image'], mask)
        raw_path = view_folder / 'refined-floor-mask.png'
        cv2.imwrite(str(raw_path), floor.astype(np.uint8) * 255)
        contour = max(cv2.findContours(floor.astype(np.uint8), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
        # A circular floor projects to a convex region. Fill the observed inward
        # housing notch, without extrapolating past a clipped straight boundary.
        convex = np.zeros_like(floor, dtype=np.uint8)
        cv2.fillConvexPoly(convex, cv2.convexHull(contour), 255)
        convex_path = view_folder / 'convex-floor-mask.png'
        cv2.imwrite(str(convex_path), convex)
        selection['refinement'] = {
            **refinement, 'mask_path': str(raw_path), 'convex_mask_path': str(convex_path),
            'convex_added_pixels': int((convex > 0).sum() - floor.sum()),
        }
        cache[index] = (convex > 0, selection)
        return cache[index]

    initial_mask, initial_selection = select_floor(0)
    evidence['initial_selection'] = initial_selection
    if initial_mask is None:
        evidence['reason'] = 'initial_receiving_bowl_not_selected'
        (folder / 'grounding.json').write_text(json.dumps(unresolved, indent=2))
        return unresolved
    for index, baseline in candidates:
        mask, selection = select_floor(index)
        attempt = {'view_index': index, 'camera_baseline_m': baseline,
                   'selection': selection, 'status': 'unresolved'}
        evidence['history_attempts'].append(attempt)
        if mask is None:
            attempt['reason'] = 'receiving_bowl_not_selected'
            continue
        calibrations = [views[0]['calibration'], views[index]['calibration']]
        try:
            sensitivity = _circle_sensitivity([initial_mask, mask], calibrations)
            attempt.update(sensitivity)
            # The observed small, occluded ready disk moved depth by 31 mm for
            # one pixel and biased release height by 58 mm. Seek a stronger pair.
            if sensitivity['one_pixel_sensitivity_m'] > .02:
                attempt['reason'] = 'insufficient_circle_parallax'
                continue
            result = _fit_circle([initial_mask, mask], calibrations, public_state)
        except ValueError as error:
            attempt['reason'] = str(error)
            continue
        attempt['status'] = 'grounded'
        result['evidence'] = {**evidence, **result['evidence'],
                              'selected_view_indices': [0, index],
                              'calibrations': {'previous': calibrations[0],
                                               'current': calibrations[1]}}
        (folder / 'grounding.json').write_text(json.dumps(result, indent=2))
        return result
    evidence['reason'] = 'no_supported_circle_pair_in_history'
    (folder / 'grounding.json').write_text(json.dumps(unresolved, indent=2))
    return unresolved
