"""Temporal public-RGB geometry for the visible cylindrical faucet control.

This is a cylinder-model estimate, not an observed articulation center. The
caller owns contact validation. Temperature +axis follows the static fixture
contract (sink long axis toward the lever); negative angles cool the water.
"""
from itertools import combinations
from pathlib import Path
import json
import cv2
import numpy as np
from scipy.ndimage import label
from scipy.spatial.transform import Rotation
from .image_servo import _camera_geometry, _pixel_ray_world, _rotation_xyzw
from .panda_embodiment import project_world_point
from .rgb_grounding import _triangulate
from .pose_skill import plan_pose_segment
from .joint_protocol import MAX_JOINT_STEP


def _components(mask):
    labels, count = label(mask, np.ones((3, 3)))
    result = []
    for i in range(1, count+1):
        y, x = np.where(labels == i)
        if len(x) >= 2:
            result.append({'area':len(x), 'point':[float(x.mean()), float(y.mean())]})
    return result


def _features(image, calibration, lever_world):
    p = project_world_point(lever_world, calibration)
    yy, xx = np.indices(image.shape[:2])
    roi = (abs(xx-p['u_px']) < 48) & (yy > p['v_px']-64) & (yy < p['v_px']+40)
    b, g, r = cv2.split(image.astype(float))
    colors = {'red':_components(roi & (r-g > 18) & (r-b > 28)),
              'blue':_components(roi & (b-r > 20) & (b-g > 8))}
    if any(len(v) != 1 for v in colors.values()):
        return None
    center = np.mean([v[0]['point'] for v in colors.values()], axis=0)
    separation = np.linalg.norm(np.subtract(colors['red'][0]['point'], colors['blue'][0]['point']))
    if not 3 < separation < 12:
        return None
    cx, cy = center
    tip_mask = (abs(xx-cx) < 16) & (yy > cy+18) & (yy < cy+45) & (image.min(axis=2) > 128)
    tips = _components(tip_mask)
    if len(tips) != 1 or not 3 <= tips[0]['area'] <= 30:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    row_y = round(cy)-2
    row = np.median(gray[row_y-1:row_y+2], axis=0)
    # The fixture sits against a locally flat basin. Require an actual plateau
    # on both sides; internal shading may cross the same gray value once.
    patch = gray[round(cy)+18:round(cy)+30, max(0, round(cx)-22):round(cx)+23]
    bg = float(np.bincount(patch.astype(np.uint8).ravel()).argmax())
    left, right = round(cx)-3, round(cx)+3
    while left > max(3, cx-24) and not np.all(abs(row[left-2:left+1]-bg) <= 2):
        left -= 1
    while right < min(len(row)-3, cx+24) and not np.all(abs(row[right:right+3]-bg) <= 2):
        right += 1
    if not 14 < right-left < 40 or left <= cx-24 or right >= cx+24:
        return None
    tip = tips[0]['point']
    shaft_region = (abs(xx-tip[0]) < 8) & (yy >= round(tip[1])-10) & (yy <= round(tip[1])-8)
    shaft = _components(shaft_region & (abs(gray-bg) > 3))
    if not shaft or np.ptp([v['point'][0] for v in shaft]) > 7:
        return None
    shaft_point = np.average([v['point'] for v in shaft], axis=0, weights=[v['area'] for v in shaft]).tolist()
    return {'marker':center.tolist(), 'red':colors['red'][0]['point'],
            'blue':colors['blue'][0]['point'], 'tip':tip, 'grip':shaft_point,
            'hub_edges':[[left+.5, row_y], [right-.5, row_y]], 'background_gray':bg}


def _sink_axis(mask, calibration, rotation, base, direction_hint):
    contour = max(cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    geometry = _camera_geometry(calibration)
    camera = np.asarray(calibration['camera_position_world_m'])
    plane_z = camera[2]-1.
    points = []
    for pixel in contour[:, 0, :]:
        origin, direction = map(np.asarray, _pixel_ray_world(pixel.tolist(), geometry))
        distance = (plane_z-origin[2])/direction[2]
        if distance > 0:
            points.append((rotation.T @ (origin+distance*direction-base))[:2])
    rectangle = cv2.boxPoints(cv2.minAreaRect(np.asarray(points, np.float32)))
    edges = np.roll(rectangle, -1, axis=0)-rectangle
    edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
    axis = np.r_[edge/np.linalg.norm(edge), 0.]
    if axis @ direction_hint < 0:
        axis = -axis
    return axis


def _select_contact_rotation(q_start, grip, pivot, axis, shaft_direction):
    """Choose a shaft-crossing jaw sign over the entire intended motion."""
    tool_z = Rotation.from_rotvec(axis*np.pi/4).apply([0., 0., -1.])
    jaw_x = np.cross(shaft_direction, tool_z)
    jaw_x /= np.linalg.norm(jaw_x)
    initial_rotation = np.column_stack([jaw_x, np.cross(tool_z, jaw_x), tool_z])
    approach = grip-.10*tool_z
    candidates = []
    for flipped in (False, True):
        rotation = initial_rotation @ (np.diag([-1., -1., 1.]) if flipped else np.eye(3))
        q = np.asarray(q_start, float)
        targets = [('approach', approach, rotation), ('descend', grip, rotation)]
        targets += [(f"arc_{point['angle_degrees']}", point['target_base_m'], point['target_rotation'])
                    for point in temperature_arc(pivot, axis, grip, rotation)]
        summary = {'jaw_flipped':flipped, 'status':'kinematically_reachable', 'ideal_steps':0}
        for stage, position, target_rotation in targets:
            plan = plan_pose_segment(q, position, target_rotation)
            if plan['status'] != 'kinematically_reachable':
                summary.update(status=plan['status'], failed_stage=stage,
                               failed_result=plan['failed_result'])
                break
            for waypoint in plan['waypoints']:
                new_q = np.asarray(waypoint['q'])
                summary['ideal_steps'] += int(np.ceil(np.max(np.abs(new_q-q))/MAX_JOINT_STEP))
                q = new_q
        candidates.append(summary)
    reachable = [item for item in candidates if item['status']=='kinematically_reachable']
    if not reachable:
        return {'status':'kinematically_unresolved', 'candidates':candidates}
    chosen = min(reachable, key=lambda item:item['ideal_steps'])
    rotation = initial_rotation @ (np.diag([-1., -1., 1.]) if chosen['jaw_flipped'] else np.eye(3))
    return {'status':'kinematically_reachable', 'rotation':rotation.tolist(),
            'pregrasp_base_m':approach.tolist(), 'jaw_flipped':chosen['jaw_flipped'],
            'ideal_steps':chosen['ideal_steps'], 'initial_tilt_degrees':45.,
            'candidates':candidates}


def ground_faucet_hub(frames, public_state, initial_ground, work_dir,
                       sink_axis_base=None, sink_mask=None, sink_calibration=None):
    """Derive a hub axis line and shaft contact from chronological wrist views.

    Each frame has image_bytes and calibration (one wrist camera packet).
    initial_ground is ground_faucet_lever's result. With no supplied measured
    sink axis, use the largest existing SAM wrist mask (>=15% image area), and
    the initial wrist calibration supplied as sink_calibration. No model call.
    """
    folder = Path(work_dir); folder.mkdir(parents=True, exist_ok=True)
    rot = np.asarray(_rotation_xyzw(public_state['state.base_rotation']))
    base = np.asarray(public_state['state.base_position'])
    data, cal = {}, {}
    for i, frame in enumerate(frames):
        image = cv2.imdecode(np.frombuffer(frame['image_bytes'], np.uint8), cv2.IMREAD_COLOR)
        features = _features(image, frame['calibration'], initial_ground['target_world_m'])
        if features:
            key = str(i); data[key] = features; cal[key] = frame['calibration']
            (folder/(key+'.png')).write_bytes(frame['image_bytes'])
    result = {'status':'unresolved', 'accepted_view_count':len(data), 'features':data,
              'method':'RGB markers, temporal rays, cylindrical hub silhouette',
              'pivot_status':'cylinder_model_estimate', 'qwen_calls':0}
    def finish():
        (folder/'geometry.json').write_text(json.dumps(result, indent=2)); return result
    if len(data) < 2:
        return finish()
    accepted = []
    for a, b in combinations(data, 2):
        marker = _triangulate({k:data[k]['marker'] for k in (a,b)}, cal)
        grip = _triangulate({k:data[k]['grip'] for k in (a,b)}, cal)
        tip = _triangulate({k:data[k]['tip'] for k in (a,b)}, cal)
        if all(v is not None and v[1] < .003 and v[2] < .012 for v in (marker,grip,tip)):
            accepted.append((marker[2], a, b, marker, grip, tip))
    if not accepted:
        return finish()
    _, a, b, marker, grip, tip = min(accepted, key=lambda v:v[0])
    marker_base, grip_base, tip_base = [rot.T @ (g[0]-base) for g in (marker,grip,tip)]
    if sink_axis_base is None:
        if sink_mask is None:
            candidates = initial_ground['evidence']['sam']['views']['wrist']['candidates']
            largest = max(candidates, key=lambda c:c['area'])
            mask = cv2.imread(largest['mask_path'], cv2.IMREAD_GRAYSCALE)
            if mask is None or largest['area'] < .15*mask.size:
                return finish()
            sink_mask = mask > 0
        if sink_calibration is None:
            return finish()
        axis = _sink_axis(sink_mask, sink_calibration, rot, base, tip_base-marker_base)
    else:
        axis = np.asarray(sink_axis_base, float); axis /= np.linalg.norm(axis)
    side = np.cross(axis, [0.,0.,1.]); side /= np.linalg.norm(side)
    basis = np.column_stack([side, [0.,0.,1.]])
    matrix, rhs = [], []
    for key, features in data.items():
        for pixel, sign in zip(features['hub_edges'], [-1,1]):
            ow, dw = map(np.asarray, _pixel_ray_world(pixel, _camera_geometry(cal[key])))
            origin, direction = rot.T @ (ow-base), rot.T @ dw
            o2, d2 = basis.T @ origin, basis.T @ direction
            normal = np.array([-d2[1], d2[0]]); normal /= np.linalg.norm(normal)
            if sign*normal[0] < 0: normal = -normal
            matrix.append([*normal, -1.]); rhs.append(normal @ o2)
    fitted = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    pivot = basis @ fitted[:2] + axis*(axis @ marker_base)
    residual = np.asarray(matrix) @ fitted-rhs
    result.update(paired_frames=[a,b], marker_surface_base_m=marker_base.tolist(),
                  temperature_axis_base=axis.tolist(), hub_radius_m=float(fitted[2]),
                  hub_fit_max_residual_m=float(np.max(abs(residual))),
                  marker_ray_gap_m=marker[1], marker_sensitivity_m=marker[2],
                  grip_ray_gap_m=grip[1], grip_sensitivity_m=grip[2])
    if not .01 < fitted[2] < .035 or np.max(abs(residual)) > .003:
        return finish()
    shaft_direction = tip_base-grip_base
    selection = _select_contact_rotation(public_state['state.arm_joint_position'],
                                         grip_base, pivot, axis, shaft_direction)
    result['contact_pose_selection'] = selection
    if selection['status'] != 'kinematically_reachable':
        return finish()
    contact_rotation = np.asarray(selection['rotation'])
    yaw = float(np.arctan2(contact_rotation[1,0], contact_rotation[0,0]))
    result.update(status='estimated', pivot_base_m=pivot.tolist(),
                  grip_base_m=grip_base.tolist(), tip_base_m=tip_base.tolist(),
                  grip_yaw_rad=yaw, grip_rotation_base=contact_rotation.tolist(),
                  pregrasp_base_m=selection['pregrasp_base_m'],
                  limitations=['Hub cylinder is a geometric model; motion has not confirmed its hinge axis.',
                               'Contact and continued water flow require live public observation.'])
    return finish()


def temperature_arc(pivot_base, axis_base, contact_base, rotation, angle_degrees=-30., step_degrees=5.):
    """Rigid arc: keeps axial contact position/radius fixed and rotates the tool.

    For the static sink joint contract a negative angle cools. Holding axial
    position avoids intentionally changing the independent on/off coordinate.
    """
    pivot, axis, contact = map(lambda x:np.asarray(x,float), (pivot_base,axis_base,contact_base))
    axis /= np.linalg.norm(axis)
    count = max(1,int(np.ceil(abs(angle_degrees)/step_degrees)))
    out = []
    for angle in np.linspace(0, angle_degrees, count+1)[1:]:
        turn = Rotation.from_rotvec(axis*np.deg2rad(angle)).as_matrix()
        out.append({'angle_degrees':float(angle), 'target_base_m':(pivot+turn@(contact-pivot)).tolist(),
                    'target_rotation':(turn@np.asarray(rotation)).tolist()})
    return out
